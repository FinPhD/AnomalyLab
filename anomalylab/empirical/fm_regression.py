from logging import warn, warning

from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.preprocess import Normalize, Winsorize
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *
from anomalylab.utils.utils import RegModels, RegResult


@dataclass
class FamaMacBethRegression(Empirical):

    def _winsorize(self, is_winsorize: bool, exogenous: list[str]):
        if is_winsorize:
            if self.temp_panel.outliers != "unprocessed":
                warning.warn(
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
        dependent: list[str],
        industry_column: Optional[str],
        industry_weighed_method: str,
        weight_column: Optional[str] = None,
    ) -> None:
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
            self.temp_panel.df[dependent] -= self.temp_panel.df.groupby(
                by=["time", industry_column]
            )[dependent].transform(func=func)

    def _reg(
        self,
        df: DataFrame,
        model: RegModel,
        is_normalize: bool,
    ) -> RegResult:
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
        dependent: Optional[str],
        exogenous: Optional[list[str]],
    ) -> RegModels:
        if models is None:
            if dependent is None:
                raise ValueError("dependent variable must be provided.")
            if exogenous is None:
                raise ValueError("exogenous variables must be provided.")
            return RegModels(models=[{dependent: exogenous}])
        else:
            if all(isinstance(model, list) for model in models):
                return RegModels(models=[{model[0]: model[1:]} for model in models])  # type: ignore
            elif all(isinstance(model, dict) for model in models):
                return RegModels(models=models)  # type: ignore
            else:
                raise ValueError("models must be a list of dictionaries or lists.")

    def fit(
        self,
        dependent: Optional[str] = None,
        exogenous: Optional[list[str]] = None,
        models: Optional[list[list[str] | dict[str, list[str]]]] = None,
        exogenous_order: Optional[list[str]] = None,
        model_names: Optional[list[str]] = None,
        weight_column: Optional[str] = None,
        industry_column: Optional[str] = None,
        industry_weighed_method: Literal["value", "equal"] = "value",
        is_winsorize: bool = False,
        is_normalize: bool = False,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        # Preparation
        self.temp_panel: PanelData = self.panel_data.copy()
        reg_models: RegModels = self._model_parse(
            models=models, dependent=dependent, exogenous=exogenous
        )
        self._winsorize(is_winsorize=is_winsorize, exogenous=reg_models.exogenous)
        self._industry_weighted(
            dependent=reg_models.dependent,
            industry_column=industry_column,
            industry_weighed_method=industry_weighed_method,
            weight_column=weight_column,
        )
        exogenous_order = (exogenous_order or reg_models.exogenous) + ["const"]
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
                        exogenous_order=exogenous_order,
                    )
                    for model in reg_models.models
                ],
                axis=1,
            )
            .loc[exogenous_order + ["No. Obs.", "Adj. R²"]]
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
    from anomalylab.preprocess.fillna import FillNa

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="panel", classifications="industry")
    # panel = (
    #     FillNa(panel_data=panel)
    #     .fill_group_column(
    #         group_column="industry",
    #         value="Other",
    #     )
    #     .fillna(
    #         method="mean",
    #         group_columns="time",
    #     )
    #     .panel_data
    # )
    # normalize: Normalize = Normalize(panel_data=panel)
    # normalize.normalize(
    #     group_columns="time",
    # )
    fm = FamaMacBethRegression(panel_data=panel)
    result = fm.fit(
        # dependent="ret",
        # exogenous=["size", "illiquidity"],
        # exogenous_order=["illiquidity", "size", "idiosyncratic_volatility"],
        models=[
            ["ret", "size", "illiquidity"],
            # ["ret", "size"],
            ["ret", "idiosyncratic_volatility"],
        ],
        # models=[{"ret": ["size", "illiquidity"]}],
        # weight_column="size",
        # industry_column="industry",
        # industry_weighed_method="value",
        # is_winsorize=True,
        is_normalize=True,
        decimal=2,
    )
    pp(object=result)
