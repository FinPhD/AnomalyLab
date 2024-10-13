from importlib import resources

from anomalylab.utils import *
from anomalylab.utils.imports import *


class DataSet:
    """This class is used to load the datasets for the examples.

    Usages:
        ```python
        >>> df_panel = DataSet.get_panel_data()
        >>> df_time_series = DataSet.get_time_series_data()
        ```
    """

    @classmethod
    def get_panel_data(cls) -> DataFrame:
        """Return the panel data example."""
        return pd.read_csv(
            filepath_or_buffer=str(resources.files("anomalylab.datasets"))
            + "/panel_data.csv"
        )

    @classmethod
    def get_time_series_data(cls) -> DataFrame:
        """Return the time series data example."""
        return pd.read_csv(
            filepath_or_buffer=str(resources.files("anomalylab.datasets"))
            + "/time_series_data.csv"
        )


if __name__ == "__main__":
    ...
    pp(DataSet.get_panel_data())
    pp(DataSet.get_time_series_data())
