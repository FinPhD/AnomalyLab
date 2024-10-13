from anomalylab.structure.data import Data
from anomalylab.utils import *
from anomalylab.utils.imports import *


@dataclass
class TimeSeries(Data):
    """
    `TimeSeries` class for handling time series data structure.

    Attributes:
        df (DataFrame):
            The `DataFrame` object that contains the data.
        name (str):
            The name of the object.
        time (str):
            The column name for the time identifier. Defaults to "time".
        time_format (str):
            The format of the time column. Defaults to "%Y%m".
    """

    time: str = "time"
    time_format: str = "%Y%m"
    factors: list[str] = field(init=False)

    def __repr__(self) -> str:
        return f"TimeSeriesData({self.name})"

    def _preprocess(self) -> None:
        """
        Preprocess the `DataFrame` by renaming the time column and identifying factor columns.

        This method renames the time column to a standardized name and identifies remaining columns as factors.
        """
        self.df = self.df.rename(columns={self.time: "time"})
        self.time = "time"
        self.factors = list(self.df.columns)
        self.factors.remove("time")

    def _check_columns(self) -> None:
        """
        Check if the required column is present in the DataFrame and ensure there are additional columns.

        Raises:
            ValueError: If the time column is missing from the DataFrame.
            ValueError: If there are no additional columns for factor returns.
        """
        if self.time not in self.df.columns:
            raise ValueError(f"Missing column in the DataFrame: {self.time}")

        if len(self.df.columns) <= 1:
            raise ValueError("The number of factor returns must be at least 1.")


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_time_series_data()
    time_series: TimeSeries = TimeSeries(df=df, name="FF3 and FF5", time="date")
    pp(time_series)
    pp(time_series.factors)
