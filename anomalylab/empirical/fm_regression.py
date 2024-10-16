from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.preprocess import Winsorize
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *
from anomalylab.utils.utils import RegModels, RegResult


@dataclass
class FamaMacBethRegression(Empirical):
    """
    A class for performing Fama-MacBeth regression analysis on empirical panel data.

    This class extends the Empirical class and offers methods to conduct Fama-MacBeth
    regressions, including options for winsorization, industry weighting, and model parsing.
    The results are formatted and returned in a structured DataFrame.

    Attributes:
        panel_data (PanelData): The panel data object containing the data for regression analysis.
    """

    def _winsorize(self, is_winsorize: bool, exogenous: list[str]):
        """
        Applies winsorization to specified columns if indicated by the user.

        This method checks if winsorization is needed and applies it to the exogenous
        variables if the panel data has not been processed for outliers.

        Args:
            is_winsorize (bool): A flag indicating whether to apply winsorization.
            exogenous (list[str]): The list of exogenous variables to winsorize.
        """
        if is_winsorize:
            if self.temp_panel.outliers != "unprocessed":
                warnings.warn(
                    "Outliers have been processed, winsorization may not be necessary."
                )
            self.temp_panel = (
                Winsorize(panel_data=self.temp_panel)
                .winsorize(
                    columns=exogenous,
                    process_all_characteristics=False,
                )
                .panel_data
            )

    def _industry_weighted(
        self,
        endog: list[str],
        industry_column: Optional[str],
        industry_weighed_method: str,
        weight_column: Optional[str] = None,
    ) -> None:
        """
        Adjusts the endogenous variables based on industry weighting.

        This method checks the existence of the specified industry column and applies
        the chosen weighting method (either 'equal' or 'value') to the endogenous variables
        based on the industry grouping.

        Args:
            endog (list[str]): The list of endogenous variables to adjust.
            industry_column (Optional[str]): The column name representing the industry.
            industry_weighed_method (str): The method of weighting ('equal' or 'value').
            weight_column (Optional[str]): The column to be used as weights if 'value' method is selected.
        """
        self.temp_panel.check_columns_existence(
            columns=columns_to_list(columns=industry_column),
            check_range="all",
        )
        if industry_column is not None:
            if industry_weighed_method == "equal":
                func = "mean"
            elif industry_weighed_method == "value":
                func = lambda x: np.average(
                    x, weights=self.temp_panel.df.loc[x.index, weight_column]
                )
            else:
                raise ValueError(
                    f"industry_weighed_method must be one of ['value', 'equal']"
                )
            self.temp_panel.df[endog] -= self.temp_panel.df.groupby(
                by=["time", industry_column]
            )[endog].transform(func=func)

    def _reg(
        self,
        df: DataFrame,
        model: RegModel,
        is_normalize: bool,
    ) -> RegResult:
        """
        Performs the Fama-MacBeth regression analysis on the specified DataFrame.

        This method prepares the data, runs the regression using the specified model,
        and applies Newey-West adjustment for standard errors.

        Args:
            df (DataFrame): The DataFrame containing the data to be analyzed.
            model (RegModel): The regression model specifying dependent and independent variables.
            is_normalize (bool): A flag indicating whether to normalize the exogenous variables.

        Returns:
            RegResult: An object containing regression results including parameters, t-values,
            p-values, number of observations, and adjusted R-squared.
        """
        dependent: str = list(model.keys())[0]
        exogenous: list[str] = list(model.values())[0]
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
        """
        Calculates the adjusted R-squared for a given group of data.

        This method performs an OLS regression on the specified group and returns
        the adjusted R-squared statistic.

        Args:
            group: The group of data for which to calculate adjusted R-squared.
            dependent (str): The dependent variable name.
            exogenous (list[str]): The list of exogenous variable names.

        Returns:
            float: The adjusted R-squared value from the regression.
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
        exogenous_order: list[str],
    ) -> Series:
        """
        Formats the regression results into a Series for output.

        This method compiles the regression parameters, t-values, and additional
        statistics into a structured Series, rounded to the specified decimal places.

        Args:
            reg_result (RegResult): The regression result object containing various statistics.
            decimal (int): The number of decimal places for rounding.
            exogenous_order (list[str]): The order of exogenous variables for formatting.

        Returns:
            Series: A Series containing formatted regression results.
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
            index=exogenous_order,
        ).stack()  # type: ignore
        result.loc["No. Obs."] = reg_result["mean_obs"]
        result.loc["Adj. R²"] = (
            round_to_string(value=reg_result["rsquared"] * 100, decimal=2) + "%"
        )
        return result

    def _model_parse(
        self,
        models: Optional[list[list[str] | dict[str, list[str]]]],
        endog: Optional[str],
        exog: Optional[list[str]],
    ) -> RegModels:
        """
        Parses the models specified by the user into a structured format.

        This method checks the format of the provided models and ensures that
        they are either lists of exogenous variables or dictionaries mapping
        dependent variables to lists of exogenous variables.

        Args:
            models (Optional[list[list[str] | dict[str, list[str]]]]): The models provided by the user.
            endog (Optional[str]): The dependent variable name.
            exog (Optional[list[str]]): The list of exogenous variable names.

        Returns:
            RegModels: An object containing the parsed regression models.

        Raises:
            ValueError: If the models are not in the expected format.
        """
        if models is None:
            if endog is None:
                raise ValueError("dependent variable must be provided.")
            if exog is None:
                raise ValueError("exogenous variables must be provided.")
            return RegModels(models=[{endog: exog}])
        else:
            if all(isinstance(model, list) for model in models):
                return RegModels(models=[{model[0]: model[1:]} for model in models])  # type: ignore
            elif all(isinstance(model, dict) for model in models):
                return RegModels(models=models)  # type: ignore
            else:
                raise ValueError("models must be a list of dictionaries or lists.")

    def fit(
        self,
        endog: Optional[str] = None,
        exog: Optional[list[str]] = None,
        models: Optional[list[list[str] | dict[str, list[str]]]] = None,
        exog_order: Optional[list[str]] = None,
        model_names: Optional[list[str]] = None,
        weight_column: Optional[str] = None,
        industry_column: Optional[str] = None,
        industry_weighed_method: Literal["value", "equal"] = "value",
        is_winsorize: bool = False,
        is_normalize: bool = False,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """
        Fits the Fama-MacBeth regression model to the panel data.

        This method prepares the data, applies any specified preprocessing steps
        (such as winsorization and industry weighting), and conducts regression
        analysis using the specified models. The results are formatted and returned
        in a DataFrame.

        Args:
            endog (Optional[str]): The dependent variable name.
            exog (Optional[list[str]]): The list of exogenous variable names.
            models (Optional[list[list[str] | dict[str, list[str]]]]): The regression models to fit.
            exog_order (Optional[list[str]]): The order of exogenous variables.
            model_names (Optional[list[str]]): The names for the models in the output.
            weight_column (Optional[str]): The column to be used as weights for industry weighting.
            industry_column (Optional[str]): The column representing the industry.
            industry_weighed_method (Literal["value", "equal"], optional): The method for industry weighting.
                Defaults to "value".
            is_winsorize (bool): A flag indicating whether to apply winsorization. Defaults to False.
            is_normalize (bool): A flag indicating whether to normalize exogenous variables. Defaults to False.
            decimal (Optional[int]): The number of decimal places for rounding results. Defaults to None.

        Returns:
            DataFrame: A DataFrame containing the formatted regression results.

        Raises:
            ValueError: If the dependent or exogenous variables are not provided.
        """
        # Preparation
        self.temp_panel: PanelData = self.panel_data.copy()
        reg_models: RegModels = self._model_parse(models=models, endog=endog, exog=exog)
        self._winsorize(is_winsorize=is_winsorize, exogenous=reg_models.exogenous)
        self._industry_weighted(
            endog=reg_models.dependent,
            industry_column=industry_column,
            industry_weighed_method=industry_weighed_method,
            weight_column=weight_column,
        )
        exog_order = (exog_order or reg_models.exogenous) + ["const"]
        # Regression
        df: DataFrame = (
            pd.concat(
                [
                    self._format(
                        reg_result=self._reg(
                            df=self.temp_panel.df,
                            model=model,
                            is_normalize=is_normalize,
                        ),
                        decimal=decimal or self.decimal,
                        exogenous_order=exog_order,
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
        df.columns = model_names or list(
            map(lambda x: f"({x})", range(1, len(reg_models.models) + 1))
        )
        return df


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")

    fm = FamaMacBethRegression(panel_data=panel)
    result = fm.fit(
        # dependent="ret",
        # exogenous=["MktCap", "Illiq", "IdioVol"],
        exog_order=["MktCap", "Illiq", "IdioVol"],
        models=[
            ["ret", "MktCap"],
            ["ret", "Illiq"],
            ["ret", "IdioVol"],
            ["ret", "MktCap", "Illiq", "IdioVol"],
        ],
        # models=[
        #     {"ret": ["MktCap"]},
        #     {"ret": ["Illiq"]},
        #     {"ret": ["IdioVol"]},
        #     {"ret": ["MktCap", "Illiq", "IdioVol"]},
        # ],
        industry_column="industry",
        industry_weighed_method="value",
        weight_column="MktCap",
        # is_winsorize=True,
        is_normalize=True,
        # decimal=2,
    )
    pp(object=result)
