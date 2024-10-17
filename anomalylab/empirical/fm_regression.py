from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.preprocess import Winsorize
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *
from anomalylab.utils.utils import RegModels, RegResult


@dataclass
class FamaMacBethRegression(Empirical):
    """Class for performing Fama-MacBeth regression analysis.

    Inherits from the Empirical class and provides methods for
    winsorizing, calculating industry-weighted returns, fitting
    the regression model, and formatting the output results.

    Attributes:
        temp_panel (PanelData): Temporary panel data used for analysis.
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
            if self.temp_panel.outliers != "unprocessed":
                warnings.warn(
                    "Outliers have been processed, winsorization may not be necessary."
                )
            self.temp_panel = (
                Winsorize(panel_data=self.temp_panel)
                .winsorize(
                    columns=exog,
                    process_all_characteristics=False,
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

        Args:
            endog (list[str]): List of endogenous variable names to adjust.
            industry (Optional[str]): The industry column to group by.
            industry_weighed_method (str): Method for industry weighting ('equal' or 'value').
            weight (Optional[str]): Column name for weights if using value weighting.

        Raises:
            ValueError: If the specified weighting method is not recognized or weight is missing for value weighting.
        """
        self.temp_panel.check_columns_existence(
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
                func = lambda x: np.average(
                    x, weights=self.temp_panel.df.loc[x.index, weight]
                )
            else:
                raise ValueError(
                    f"industry_weighed_method must be one of ['value', 'equal']"
                )
            self.temp_panel.df[endog] -= self.temp_panel.df.groupby(
                by=["time", industry]
            )[endog].transform(func=func)

    def _reg(
        self,
        df: DataFrame,
        reg: RegModel,
        is_normalize: bool,
    ) -> RegResult:
        """Performs Fama-MacBeth regression on the provided DataFrame.

        Args:
            df (DataFrame): DataFrame containing the data for regression.
            reg (RegModel): Model specification containing endogenous and exogenous variables.
            is_normalize (bool): Indicates whether to normalize the exogenous variables.

        Returns:
            RegResult: Results of the regression including parameters, t-values, p-values, and adjusted R².
        """
        dependent: str = list(reg.keys())[0]
        exogenous: list[str] = list(reg.values())[0]
        df = df[["id", "time"] + exogenous + [dependent]].dropna()
        # Exclude time periods with only one observation
        df = df.groupby("time").filter(lambda x: len(x) > 1)
        lag: int = math.ceil(4 * (df["time"].nunique() / 100) ** (4 / 25))
        if is_normalize:
            df[exogenous] = df.groupby("time")[exogenous].transform(
                func=lambda x: (x - x.mean()) / x.std()
            )
        df["time"] = df["time"].dt.to_timestamp()
        df = df.set_index(["id", "time"])
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
            mean_obs=str(int(fmb.time_info["mean"])),
            rsquared=(
                df.reset_index(level=df.index.names[0], drop=True)
                .groupby("time")
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

        Args:
            reg_result (RegResult): Results of the regression.
            decimal (int): Number of decimal places for rounding.
            exog_order (list[str]): Order of exogenous variables in the output.

        Returns:
            Series: Formatted regression results including parameters, t-values, and statistics.
        """
        result: Series = pd.DataFrame(
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
    ) -> DataFrame:
        """Fits the Fama-MacBeth regression model and returns the results DataFrame.

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

        Returns:
            DataFrame: DataFrame containing the regression results including parameters, t-values, and other statistics.
        """
        # Preparation
        self.temp_panel: PanelData = self.panel_data.copy()
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
        # Regression
        df: DataFrame = (
            pd.concat(
                [
                    self._format(
                        reg_result=self._reg(
                            df=self.temp_panel.df,
                            reg=model,
                            is_normalize=is_normalize,
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

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")

    fm = FamaMacBethRegression(panel_data=panel)
    result = fm.fit(
        # endog="ret",
        # exog="MktCap",
        exog_order=["MktCap"],
        regs=[
            "ret",
            "MktCap",
            # ["ret", "Illiq"],
            # ["ret", "IdioVol"],
            # ["ret", "MktCap", "Illiq", "IdioVol"],
        ],
        # models=[
        #     {"ret": ["MktCap"]},
        #     {"ret": ["Illiq"]},
        #     {"ret": ["IdioVol"]},
        #     {"ret": ["MktCap", "Illiq", "IdioVol"]},
        # ],
        industry="industry",
        industry_weighed_method="value",
        weight="MktCap",
        # is_winsorize=True,
        is_normalize=True,
        # decimal=2,
    )
    pp(object=result)
