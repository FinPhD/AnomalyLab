import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import FamaMacBeth
from pandas import DataFrame, Series

from anomalylab.empirical.empirical import Empirical
from anomalylab.preprocess import OutlierHandler
from anomalylab.structure import PanelData
from anomalylab.utils import (
    RegModel,
    RegModels,
    RegResult,
    columns_to_list,
    get_significance_star,
    pp,
    round_to_string,
)


@dataclass
class FamaMacBethRegression(Empirical):
    """Class for performing Fama-MacBeth regression analysis.

    This class inherits from the Empirical class and provides methods to
    winsorize data, calculate industry-weighted returns, fit the Fama-MacBeth
    regression model, and format the output results.

    Attributes:
        panel_data (PanelData): Panel data used for analysis.
    """

    def _winsorize(self, is_winsorize: bool, exog: list[str]):
        """Winsorizes the specified columns in the temporary panel data.

        Args:
            is_winsorize (bool): Indicates whether to apply winsorization.
            exog (list[str]): List of column names to apply winsorization on.

        Raises:
            UserWarning: If outliers have been processed and winsorization may not be necessary.
        """
        if is_winsorize:
            if self.panel_data.outliers != "Unprocessed":
                warnings.warn(
                    "Outliers have been processed, winsorization may not be necessary."
                )
            self.panel_data = (
                OutlierHandler(panel_data=self.panel_data)
                .winsorize(
                    columns=exog,
                    method="winsorize",
                    group_columns=self.panel_data.time,
                )
                .panel_data
            )

    def _industry_weighted(
        self,
        endog: list[str],
        industry: Optional[str],
        industry_weighed_method: str,
        weight: Optional[str] = None,
    ) -> None:
        """Calculates industry-weighted returns for the specified endogenous variables.

        This method adjusts the endogenous variables by industry-specific returns,
        either using equal weights or value weights (based on the 'industry_weighed_method').

        Args:
            endog (list[str]): List of endogenous variable names to adjust.
            industry (Optional[str]): The industry column to group by.
            industry_weighed_method (str): Method for industry weighting ('equal' or 'value').
            weight (Optional[str]): Column name for weights if using value weighting.

        Raises:
            ValueError: If the specified weighting method is not recognized or weight is missing for value weighting.
        """
        self.panel_data.check_columns_existence(
            columns=columns_to_list(columns=industry),
            check_range="all",
        )
        if industry is not None:
            if industry_weighed_method == "equal":
                func = "mean"
            elif industry_weighed_method == "value":
                if weight is None:
                    raise ValueError(
                        "When calculating the value-weighted industry return, the weight column must be specified!"
                    )

                def func(x):
                    return np.average(
                        x, weights=self.panel_data.df.loc[x.index, weight]
                    )
            else:
                raise ValueError(
                    "industry_weighed_method must be one of ['value', 'equal']"
                )
            self.panel_data.df[endog] -= self.panel_data.df.groupby(
                by=[self.time, industry]
            )[endog].transform(func=func)

    def _reg(
        self,
        df: DataFrame,
        reg: RegModel,
        is_normalize: bool,
        return_intermediate: bool = False,
    ) -> RegResult:
        """Performs Fama-MacBeth regression on the provided DataFrame.

        This method runs the Fama-MacBeth two-step regression on a given dataset.
        It can optionally normalize the exogenous variables and return intermediate
        results for each time period.

        Args:
            df (DataFrame): DataFrame containing the data for regression.
            reg (RegModel): Model specification containing endogenous and exogenous variables.
            is_normalize (bool): Indicates whether to normalize the exogenous variables.
            return_intermediate (bool): If True, returns intermediate regression results
                (e.g., coefficients, t-values, and R²) for each time period.

        Returns:
            RegResult: Results of the regression including parameters, t-values, p-values, and adjusted R².
        """
        dependent: str = list(reg.keys())[0]
        exogenous: list[str] = list(reg.values())[0]
        df = df[[self.id, self.time] + exogenous + [dependent]].dropna()
        # Exclude time periods with only one observation
        df = df.groupby(self.time).filter(lambda x: len(x) > 1)
        lag: int = math.ceil(4 * (df[self.time].nunique() / 100) ** (4 / 25))
        if is_normalize:
            df[exogenous] = df.groupby(self.time)[exogenous].transform(
                func=lambda x: (x - x.mean()) / x.std()
            )

        df[self.time] = df[self.time].dt.to_timestamp()
        df = df.set_index([self.id, self.time])

        if return_intermediate:
            coef_df = []
            for time, group in df.groupby(self.time):
                y = group[dependent]
                X = sm.add_constant(group[exogenous])  # Add constant term
                model = sm.OLS(y, X)
                results = model.fit()
                coefs = results.params
                coefs[self.time] = time
                coef_df.append(coefs)
            coef_df = DataFrame(coef_df)
            coef_df = coef_df[
                [self.time] + [col for col in coef_df.columns if col != self.time]
            ]
            return DataFrame(coef_df)

        # Fama-MacBeth regression with Newey-West adjustment
        fmb = FamaMacBeth(
            dependent=df[dependent],
            exog=sm.add_constant(df[exogenous]),
            check_rank=False,
        ).fit(cov_type="kernel", debiased=False, bandwidth=lag)

        return RegResult(
            params=fmb.params,
            tvalues=fmb.tstats,
            pvalues=fmb.pvalues,
            mean_obs=str(round(fmb.time_info["mean"])),
            rsquared=(
                df.reset_index(level=df.index.names[0], drop=True)
                .groupby(self.time)
                .apply(
                    func=partial(
                        self._cal_adjusted_r2,
                        dependent=dependent,
                        exogenous=exogenous,
                    )
                )
                .mean()
            ),
        )

    def _cal_adjusted_r2(self, group, dependent: str, exogenous: list[str]):
        """Calculates the adjusted R² for the given group of data.

        This method fits an OLS model to the provided group of data and computes
        the adjusted R².

        Args:
            group: DataFrame group to fit the OLS model.
            dependent (str): Name of the dependent variable.
            exogenous (list[str]): List of exogenous variable names.

        Returns:
            float: Adjusted R² value from the fitted OLS model.
        """
        return (
            sm.OLS(
                endog=group[dependent],
                exog=sm.add_constant(
                    data=group[exogenous],
                ),
            )
            .fit()
            .rsquared_adj
        )

    def _format(
        self,
        reg_result: RegResult,
        decimal: int,
        exog_order: list[str],
    ) -> Series:
        """Formats the regression results into a Pandas Series.

        This method prepares the output of the regression results, rounding values
        to the specified decimal places and adding significance stars.

        Args:
            reg_result (RegResult): Results of the regression.
            decimal (int): Number of decimal places for rounding.
            exog_order (list[str]): Order of exogenous variables in the output.

        Returns:
            Series: Formatted regression results including parameters, t-values, and statistics.
        """
        result: Series = DataFrame(
            data={
                "params": reg_result["params"].map(
                    arg=lambda x: round_to_string(value=x, decimal=decimal)
                )
                + reg_result["pvalues"].map(arg=get_significance_star),
                "tvalues": reg_result["tvalues"].map(
                    arg=lambda x: f"({round_to_string(value=x, decimal=decimal)})"
                ),
            },
            index=exog_order,
        ).stack()  # type: ignore
        result.loc["No. Obs."] = reg_result["mean_obs"]
        result.loc["Adj. R²"] = (
            round_to_string(value=reg_result["rsquared"] * 100, decimal=2) + "%"
        )
        return result

    def _model_parse(
        self,
        regs: Optional[list[list[str] | dict[str, list[str]]] | list | dict],
        endog: Optional[str],
        exog: Optional[list[str]],
    ) -> RegModels:
        """Parses the model specifications into a RegModels object.

        This method parses the model specifications and converts them into a `RegModels` object.
        It supports input in multiple formats, including lists and dictionaries.

        Args:
            regs (Optional[list[list[str] | dict[str, list[str]]] | list | dict]): Model specifications.
            endog (Optional[str]): Name of the dependent variable.
            exog (Optional[list[str]]): List of exogenous variable names.

        Returns:
            RegModels: Parsed model specifications.

        Raises:
            ValueError: If models are not specified correctly or are missing required variables.
        """
        if regs is None:
            if endog is None:
                raise ValueError("dependent variable must be provided.")
            if exog is None:
                raise ValueError("exogenous variables must be provided.")
            return RegModels(models=[{endog: exog}])
        else:
            if all(isinstance(model, list) for model in regs):
                return RegModels(models=[{model[0]: model[1:]} for model in regs])  # type: ignore
            elif all(isinstance(model, dict) for model in regs):
                return RegModels(models=regs)  # type: ignore
            elif isinstance(regs, list):
                return RegModels(models=[{regs[0]: regs[1:]}])  # type: ignore
            elif isinstance(regs, dict):
                return RegModels(models=[regs])
            else:
                raise ValueError("models must be a list of dictionaries or lists.")

    def fit(
        self,
        endog: Optional[str] = None,
        exog: Optional[list[str] | str] = None,
        regs: Optional[list[list[str] | dict[str, list[str]]] | list | dict] = None,
        exog_order: Optional[list[str]] = None,
        reg_names: Optional[list[str]] = None,
        weight: Optional[str] = None,
        industry: Optional[str] = None,
        industry_weighed_method: Literal["value", "equal"] = "value",
        is_winsorize: bool = False,
        is_normalize: bool = False,
        decimal: Optional[int] = None,
        return_intermediate: bool = False,  # New parameter to control whether intermediate results are returned
    ) -> DataFrame:
        """Fits the Fama-MacBeth regression model and returns the results DataFrame or intermediate results.

        This method fits the Fama-MacBeth regression model, processes data
        (including winsorization, industry weighting, normalization), and
        returns either the final regression results or intermediate results.

        Args:
            endog (Optional[str]): Name of the dependent variable.
            exog (Optional[list[str] | str]): List of exogenous variables or a single string.
            regs (Optional[list[list[str] | dict[str, list[str]]] | list | dict]): Model specifications.
            exog_order (Optional[list[str]]): Order of exogenous variables for output.
            reg_names (Optional[list[str]]): Names for the models in the output.
            weight (Optional[str]): Column name for weight in value-weighted calculations.
            industry (Optional[str]): Industry column for weighted calculations.
            industry_weighed_method (Literal["value", "equal"]): Method for weighting industries.
            is_winsorize (bool): Indicates whether to apply winsorization.
            is_normalize (bool): Indicates whether to normalize exogenous variables.
            decimal (Optional[int]): Number of decimal places for rounding in output.
            return_intermediate (bool): If True, returns the intermediate results (e.g., coefficients for each time period).

        Returns:
            DataFrame: DataFrame containing the regression results or intermediate results based on `return_intermediate`.
        """
        # Preparation
        reg_models: RegModels = self._model_parse(
            regs=regs, endog=endog, exog=columns_to_list(exog)
        )
        self._winsorize(is_winsorize=is_winsorize, exog=reg_models.exogenous)
        self._industry_weighted(
            endog=reg_models.dependent,
            industry=industry,
            industry_weighed_method=industry_weighed_method,
            weight=weight,
        )
        exog_order = (exog_order or reg_models.exogenous) + ["const"]

        if return_intermediate:
            intermediate_results = [
                self._reg(
                    df=self.panel_data.df,
                    reg=model,
                    is_normalize=is_normalize,
                    return_intermediate=True,
                )
                for model in reg_models.models
            ]
            return intermediate_results

        # Regression
        df: DataFrame = (
            pd.concat(
                [
                    self._format(
                        reg_result=self._reg(
                            df=self.panel_data.df,
                            reg=model,
                            is_normalize=is_normalize,
                            return_intermediate=False,
                        ),
                        decimal=decimal or self.decimal,
                        exog_order=exog_order,
                    )
                    for model in reg_models.models
                ],
                axis=1,
            )
            .loc[exog_order + ["No. Obs.", "Adj. R²"]]
            .droplevel(level=1)
            .fillna(value="")
        )
        df.index = df.index.where(cond=~df.index.duplicated(), other="")
        df.columns = reg_names or list(
            map(lambda x: f"({x})", range(1, len(reg_models.models) + 1))
        )
        return df


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    fm = FamaMacBethRegression(panel_data=panel)
    result = fm.fit(
        # endog="return",
        # exog="MktCap",
        # exog_order=["MktCap", "Illiq", "IdioVol"],
        regs=[
            # "return",
            # "MktCap",
            ["return", "Illiq"],
            ["return", "IdioVol"],
            ["return", "MktCap", "Illiq", "IdioVol"],
        ],
        # models=[
        #     {"return": ["MktCap"]},
        #     {"return": ["Illiq"]},
        #     {"return": ["IdioVol"]},
        #     {"return": ["MktCap", "Illiq", "IdioVol"]},
        # ],
        # industry="industry",
        # industry_weighed_method="value",
        # weight="MktCap",
        is_winsorize=True,
        is_normalize=True,
        return_intermediate=False,
        # decimal=2,
    )
    pp(result)
    # pp(result[0])
