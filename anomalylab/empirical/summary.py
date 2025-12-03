from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame, Series

from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, columns_to_list, pp, round_to_string


class Statistics:
    @staticmethod
    def mean(series: Series) -> float:
        return series.mean() if not series.isna().all() else None

    @staticmethod
    def median(series: Series) -> float:
        return series.median() if not series.isna().all() else None

    @staticmethod
    def std(series: Series) -> float:
        return series.std() if not series.isna().all() else None

    @staticmethod
    def skew(series: Series):
        return series.skew() if not series.isna().all() else None

    @staticmethod
    def kurtosis(series: Series):
        return series.kurtosis() if not series.isna().all() else None

    @staticmethod
    def min(series: Series):
        return series.min() if not series.isna().all() else None

    @staticmethod
    def max(series: Series):
        return series.max() if not series.isna().all() else None

    @staticmethod
    def count(series: Series) -> int:
        return series.count() if not series.isna().all() else None

    @staticmethod
    def var(series: Series):
        return series.var() if not series.isna().all() else None


@dataclass
class Summary(Empirical):
    """
    A class for generating summary statistics from empirical panel data.

    This class extends the Empirical class and provides functionality to compute
    various statistical measures, such as mean, standard deviation, skewness,
    and kurtosis, for specified columns in the panel data. The results are aggregated
    by time periods, allowing for a concise summary of the data.

    Attributes:
        panel_data (PanelData): The panel data object containing the data to be summarized.
    """

    def average_statistics(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """
        Computes average statistics for specified columns over time periods.

        This method constructs the list of columns to process and calculates various
        statistics for each column, including mean, standard deviation, skewness,
        kurtosis, minimum, median, maximum, and count. The results are averaged
        across time periods and returned as a DataFrame.

        Args:
            columns (Columns, optional): The columns to calculate statistics for. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from processing. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics or not.
                Defaults to True.
            decimal (Optional[int], optional): The number of decimal places to round the results to.
                Defaults to None.

        Returns:
            DataFrame: A DataFrame containing the average statistics for the specified columns.

        Note:
            The DataFrame includes the statistics rounded to the specified number of decimal places,
            and the count of non-null values is formatted as an integer string.
        """
        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )

        df: DataFrame = (
            self.panel_data.df.groupby(self.time)[columns]
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

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    summary = Summary(panel_data=panel)
    pp(
        summary.average_statistics(
            # columns=["MktCap", "Illiq", "IdioVol"],
            # decimal=3,
            # process_all_characteristics=False,
        )
    )
