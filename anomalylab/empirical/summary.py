from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


class Statistics:

    @staticmethod
    def mean(series: Series) -> float:
        return series.mean()

    @staticmethod
    def median(series: Series) -> float:
        return series.median()

    @staticmethod
    def std(series: Series) -> float:
        return series.std()

    @staticmethod
    def skew(series: Series):
        return series.skew()

    @staticmethod
    def kurtosis(series: Series):
        return series.kurtosis()

    @staticmethod
    def min(series: Series):
        return series.min()

    @staticmethod
    def max(series: Series):
        return series.max()

    @staticmethod
    def count(series: Series) -> int:
        return series.count()

    @staticmethod
    def var(series: Series):
        return series.var()


@dataclass
class Summary(Empirical):

    def average_statistics(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )
        df: DataFrame = (
            self.panel_data.df.groupby("time")[columns]
            # Calculate the statistics for each column
            .agg(
                func=[
                    Statistics.mean,
                    Statistics.std,
                    Statistics.skew,
                    Statistics.kurtosis,
                    Statistics.min,
                    Statistics.median,
                    Statistics.max,
                    Statistics.count,
                ]
            )
            # Calculate the average statistics for each time period
            .mean()
            .unstack(level=1)
            .map(func=round_to_string, decimal=decimal or self.decimal)
        )
        df["count"] = df["count"].map(lambda x: f"{float(x):.0f}")
        return df


if __name__ == "__main__":
    from anomalylab.datasets import DataSet
    from anomalylab.preprocess.fillna import FillNa
    from anomalylab.preprocess.normalize import Normalize

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="panel", classifications="industry")
    fill_nan: FillNa = FillNa(panel_data=panel)
    fill_nan.fill_group_column(
        group_column="industry",
        value="Other",
    )
    fill_nan.fillna(
        method="mean",
        group_columns="time",
    )
    normalize: Normalize = Normalize(panel_data=panel)
    normalize.normalize(
        group_columns="time",
    )
    summary = Summary(panel_data=panel)
    pp(
        summary.average_statistics(
            # columns="size",
            # decimal=3,
            # process_all_characteristics=False,
        )
    )
