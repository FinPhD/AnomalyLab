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
            The column name for the time identifier. Defaults to "date".
        frequency (Literal["D", "M", "Y"]):
            The frequency of the data. Defaults to "M".
        factors (list[str]):
            The list of factor column names. Initialized as an empty list.
    """

    time: str = "date"
    frequency: Literal["D", "M", "Y"] = "M"
    factors: list[str] = field(init=False)

    def __repr__(self) -> str:
        return f"TimeSeriesData({self.name})"  # todo: add frequency

    def _preprocess(self) -> None:
        """
        Preprocess the `DataFrame` by renaming the time column and identifying factor columns.

        This method renames the time column to a standardized name and identifies remaining columns as factors.
        """
        self.df = self.df.rename(columns={self.time: "time"})
        self.time = "time"
        # todo: add support for daily and yearly frequency
        if self.frequency != "M":
            raise NotImplementedError("Only monthly frequency is supported.")
        self.df["time"] = pd.to_datetime(self.df["time"], format="ISO8601")
        self.df["time"] = self.df["time"].dt.to_period(freq=self.frequency)
        self.df = self.df.sort_values(by="time")
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
    time_series: TimeSeries = TimeSeries(df=df, name="CAPM, FF3 and FF5")
    pp(time_series)
    pp(time_series.factors)
