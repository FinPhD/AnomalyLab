from __future__ import annotations

from importlib import resources

from anomalylab.config import *
from anomalylab.empirical import (
    Correlation,
    FamaMacBethRegression,
    Persistence,
    PortfolioAnalysis,
    Summary,
)
from anomalylab.preprocess import FillNa, Normalize, OutlierHandler, Shift
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils import *
from anomalylab.utils.imports import *
from anomalylab.visualization import FormatExcel


@dataclass
class Panel:
    _df: DataFrame = field(repr=False)
    name: Optional[str] = None
    id: str = "permno"
    time: str = "date"
    frequency: Literal["D", "M", "Y"] = "M"
    ret: str = "return"
    classifications: Optional[list[str] | str] = None
    drop_all_chars_missing: bool = False
    is_copy: bool = False

    def __post_init__(self) -> None:
        self.panel_data: PanelData = PanelData(
            df=self._df,
            name=self.name,
            id=self.id,
            time=self.time,
            frequency=self.frequency,
            ret=self.ret,
            classifications=self.classifications,
            drop_all_chars_missing=self.drop_all_chars_missing,
            is_copy=self.is_copy,
        )
        self._normalize_processor = None
        self._fillna_processor = None
        self._winsorize_processor = None
        self._shift_processor = None
        self._summary_processor = None
        self._correlation_processor = None
        self._persistence_processor = None
        self._portfolio_analysis_processor = None
        self._fm_preprocessor = None
        self._format_preprocessor = None

    def __repr__(self) -> str:
        return repr(self.panel_data)

    def get_original_data(self) -> DataFrame:
        return self._df

    def get_processed_data(self) -> DataFrame:
        return self.panel_data.df

    @property
    def normalize_processor(self) -> Normalize:
        if self._normalize_processor is None:
            self._normalize_processor = Normalize(panel_data=self.panel_data)
        return self._normalize_processor

    @property
    def fillna_processor(self) -> FillNa:
        if self._fillna_processor is None:
            self._fillna_processor = FillNa(panel_data=self.panel_data)
        return self._fillna_processor

    @property
    def winsorize_processor(self) -> OutlierHandler:
        if self._winsorize_processor is None:
            self._winsorize_processor = OutlierHandler(panel_data=self.panel_data)
        return self._winsorize_processor

    @property
    def shift_processor(self) -> Shift:
        if self._shift_processor is None:
            self._shift_processor = Shift(panel_data=self.panel_data)
        return self._shift_processor

    @property
    def summary_processor(self) -> Summary:
        if self._summary_processor is None:
            self._summary_processor = Summary(panel_data=self.panel_data)
        return self._summary_processor

    @property
    def correlation_processor(self) -> Correlation:
        if self._correlation_processor is None:
            self._correlation_processor = Correlation(panel_data=self.panel_data)
        return self._correlation_processor

    @property
    def persistence_processor(self) -> Persistence:
        if self._persistence_processor is None:
            self._persistence_processor = Persistence(panel_data=self.panel_data)
        return self._persistence_processor

    def portfolio_analysis_processor(
        self,
        endog: Optional[str] = None,
        weight: Optional[str] = None,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
    ) -> PortfolioAnalysis:
        self._portfolio_analysis_processor = PortfolioAnalysis(
            panel_data=self.panel_data,
            endog=endog,
            weight=weight,
            models=models,
            factors_series=factors_series,
        )
        return self._portfolio_analysis_processor

    @property
    def fm_preprocessor(self) -> FamaMacBethRegression:
        self._fm_preprocessor = FamaMacBethRegression(panel_data=self.panel_data)
        return self._fm_preprocessor

    def format_preprocessor(self, path: str) -> FormatExcel:
        if self._format_preprocessor is None:
            self._format_preprocessor = FormatExcel(path=path)
        return self._format_preprocessor

    def normalize(
        self,
        columns: Columns = None,
        method: Literal["zscore", "rank"] = "zscore",
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Panel:
        self.panel_data = self.normalize_processor.normalize(
            columns=columns,
            method=method,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        ).panel_data
        return self

    def fillna(
        self,
        columns: Columns = None,
        method: Literal["mean", "median", "constant"] = "mean",
        value: Optional[Union[float, int]] = None,
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Panel:
        self.panel_data = self.fillna_processor.fillna(
            columns=columns,
            method=method,
            value=value,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        ).panel_data
        return self

    def fill_group_column(self, group_column: str, value: Scalar) -> Panel:
        self.panel_data = self.fillna_processor.fill_group_column(
            group_column=group_column, value=value
        ).panel_data
        return self

    def winsorize(
        self,
        columns: Columns = None,
        method: Literal["winsorize", "truncate"] = "winsorize",
        limits: tuple[float, float] = (0.01, 0.01),
        group_columns: Optional[list[str] | str] = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Panel:
        self.panel_data = self.winsorize_processor.winsorize(
            columns=columns,
            method=method,
            limits=limits,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        ).panel_data
        return self

    def shift(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        drop_original: bool = False,
        dropna: bool = False,
    ) -> Panel:
        self.panel_data = self.shift_processor.shift(
            columns=columns,
            periods=periods,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            drop_original=drop_original,
            dropna=dropna,
        ).panel_data
        return self

    def summary(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        return self.summary_processor.average_statistics(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

    def correlation(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        return self.correlation_processor.average_correlation(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

    def persistence(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        return self.persistence_processor.average_persistence(
            columns=columns,
            periods=periods,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

    def transition_matrix(
        self,
        var: str,
        group: int,
        lag: int,
        draw: bool = False,
        path: Optional[str] = None,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        return self.persistence_processor.transition_matrix(
            var=var,
            group=group,
            lag=lag,
            draw=draw,
            path=path,
            decimal=decimal,
        )

    def univariate_analysis(
        self,
        endog: str,
        weight: str,
        core_var: str,
        core_g: int,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
        format: bool = False,
        decimal: Optional[int] = None,
        factor_return: bool = False,
    ) -> tuple:
        return self.portfolio_analysis_processor(
            endog=endog, weight=weight, models=models, factors_series=factors_series
        ).univariate_analysis(
            core_var=core_var,
            core_g=core_g,
            format=format,
            decimal=decimal,
            factor_return=factor_return,
        )

    def bivariate_analysis(
        self,
        endog: str,
        weight: str,
        sort_var: str,
        core_var: str,
        sort_g: int,
        core_g: int,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
        pivot: bool = True,
        format: bool = False,
        sort_type: str = "dependent",
        decimal: Optional[int] = None,
        factor_return: bool = False,
    ) -> tuple:
        return self.portfolio_analysis_processor(
            endog=endog, weight=weight, models=models, factors_series=factors_series
        ).bivariate_analysis(
            sort_var=sort_var,
            core_var=core_var,
            sort_g=sort_g,
            core_g=core_g,
            pivot=pivot,
            format=format,
            sort_type=sort_type,
            decimal=decimal,
            factor_return=factor_return,
        )

    def fm_reg(
        self,
        endog: Optional[str] = None,
        exog: Optional[list[str]] = None,
        regs: Optional[list[list[str] | dict[str, list[str]]]] = None,
        exog_order: Optional[list[str]] = None,
        reg_names: Optional[list[str]] = None,
        weight: Optional[str] = None,
        industry: Optional[str] = None,
        industry_weighed_method: Literal["value", "equal"] = "value",
        is_winsorize: bool = False,
        is_normalize: bool = False,
        decimal: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> DataFrame:
        return self.fm_preprocessor.fit(
            endog=endog,
            exog=exog,
            regs=regs,
            exog_order=exog_order,
            reg_names=reg_names,
            weight=weight,
            industry=industry,
            industry_weighed_method=industry_weighed_method,
            is_winsorize=is_winsorize,
            is_normalize=is_normalize,
            decimal=decimal,
            return_intermediate=return_intermediate,
        )

    def format_excel(
        self,
        path: str,
        align=True,
        line=True,
        convert_brackets=False,
        adjust_col_widths=False,
    ) -> None:
        self.format_preprocessor(path=path).process(
            align=align,
            line=line,
            convert_brackets=convert_brackets,
            adjust_col_widths=adjust_col_widths,
        )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    ts: DataFrame = DataSet.get_time_series_data()
    Models: dict[str, list[str]] = {
        "CAPM": ["MKT(3F)"],
        "FF3": ["MKT(3F)", "SMB(3F)", "HML(3F)"],
        "FF5": ["MKT(5F)", "SMB(5F)", "HML(5F)", "RMW(5F)", "CMA(5F)"],
    }

    panel = Panel(
        df,
        name="Stocks",
        classifications="industry",
        drop_all_chars_missing=True,
        is_copy=False,
    )
    time_series: TimeSeries = TimeSeries(df=ts, name="Factor Series")
    pp(panel)

    # panel.fill_group_column(group_column="industry", value="Other")
    # panel.fillna(
    #     # columns="MktCap",
    #     # method="mean",
    #     group_columns="date",
    #     # no_process_columns="MktCap",
    #     # process_all_characteristics=True,
    # )
    # panel.normalize(
    #     # columns="MktCap",
    #     # method="zscore",
    #     # group_columns="date",
    #     # no_process_columns="MktCap",
    #     # process_all_characteristics=False,
    # )
    # panel.shift()

    panel.winsorize(method="winsorize")
    pp(panel)

    pp(panel.summary())
    pp(panel.correlation())
    pp(panel.persistence(periods=[1, 3, 6, 12, 36, 60]))
    pp(
        panel.transition_matrix(
            "MktCap",
            10,
            12,
            False,
            str(resources.files("anomalylab.datasets")) + "/transition_matrix.png",
        )
    )
    uni_ew, uni_vw = panel.univariate_analysis(
        "return", "MktCap", "Illiq", 10, Models, time_series, factor_return=True
    )
    pp(uni_ew)
    pp(uni_vw)

    bi_ew, bi_vw = panel.bivariate_analysis(
        "return",
        "MktCap",
        "Illiq",
        "IdioVol",
        5,
        5,
        Models,
        time_series,
        True,
        False,
        "dependent",
        factor_return=True,
    )
    pp(bi_ew)
    pp(bi_vw)

    pp(
        panel.fm_reg(
            regs=[
                ["return", "MktCap"],
                ["return", "Illiq"],
                ["return", "IdioVol"],
                ["return", "MktCap", "Illiq", "IdioVol"],
            ],
            exog_order=["MktCap", "Illiq", "IdioVol"],
            weight="MktCap",
            industry="industry",
            industry_weighed_method="value",
            is_winsorize=False,
            is_normalize=True,
        )
    )

    # panel.format_excel("...")
