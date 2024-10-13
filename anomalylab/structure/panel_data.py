from anomalylab.structure.data import Data
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class PanelData(Data):
    """
    `PanelData` class for handling panel data structure.

    Attributes:
        df (DataFrame):
            The `DataFrame` object that contains the data.
        name (str):
            The name of the object.
        id (str):
            The column name for the firm identifier. Defaults to "id".
        time (str):
            The column name for the time identifier. Defaults to "date".
        ret (str):
            The column name for the return. Defaults to "return".
        frequency (Literal["D", "M", "Y"]):
            The frequency of the data. Defaults to "M".
        classifications (list[str]):
            The list of classification columns. Defaults to ["industry"].
    """

    id: str = "id"
    time: str = "date"
    frequency: Literal["D", "M", "Y"] = "M"
    ret: str = "return"
    classifications: Optional[list[str] | str] = None

    def set_flag(self) -> None:
        """Set default flags for the `PanelData` object."""
        self.fillna = False
        self.normalize = False
        self.shift = False
        self.outliers: Literal["unprocessed", "winsorize", "truncate"] = "unprocessed"

    def __repr__(self) -> str:
        return (
            f"PanelData({self.name})({self.frequency}), "
            f"classifications={self.classifications}, "
            f"fillna={self.fillna}, "
            f"normalize={self.normalize}, "
            f"shift={self.shift}, "
            f"outliers={self.outliers}"
        )

    def _preprocess(self) -> None:
        """
        Preprocess the `DataFrame` by renaming columns and identifying firm characteristics.

        This method renames the `id`, `time`, and `return` columns to standardized names
        and identifies remaining columns as firm characteristics, excluding classifications.
        """
        self.df = self.df.rename(
            columns={
                self.id: "id",
                self.time: "time",
                self.ret: "ret",
            }
        )
        self.id, self.time, self.ret = "id", "time", "ret"
        self.df["id"] = self.df["id"].astype(int)
        # todo: add support for daily and yearly frequency
        if self.frequency != "M":
            raise NotImplementedError("Only monthly frequency is supported.")
        self.df["time"] = pd.to_datetime(self.df["time"], format="ISO8601")
        self.df["time"] = self.df["time"].dt.to_period(freq=self.frequency)
        self.df = self.df.sort_values(by=["time", "id"])
        # Identify remaining columns and set them as firm characteristics, excluding classifications
        self.firm_characteristics: set[str] = set(
            filter(
                lambda x: (
                    x
                    not in ["id", "time", "ret"]
                    + (
                        self.classifications
                        if isinstance(self.classifications, list)
                        else []
                    )
                ),
                self.df.columns,
            )
        )

    def _check_columns(self) -> None:
        """Check if the required columns are present in the DataFrame.

        Raises:
            ValueError: If any required columns are missing from the DataFrame.
            ValueError: If there are no firm characteristics remaining after checking.
        """
        if isinstance(self.classifications, str):
            self.classifications = [self.classifications]
        # Check if the required columns are present in the DataFrame
        required_columns: set[str] = set(
            [self.id, self.time, self.ret] + (self.classifications or [])
        )
        missing_columns: set[str] = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")
        # Check if there are firm characteristics remaining
        if len(self.df.columns) - len(required_columns) < 1:
            raise ValueError("The number of firm characteristics must be at least 1.")
        # Check if there are missing values in the 'id' or 'time' columns
        if self.is_nan(columns=[self.id, self.time]):
            warnings.warn(
                message=f"Missing values found in {self.id} or {self.time} column, rows with missing values have been dropped."
            )
            self.df = self.df.dropna(subset=[self.id, self.time])

    def is_nan(self, columns: list[str]) -> bool:
        """Check if there are missing values in the specified columns."""
        return self.df[columns].isnull().any().any()

    def missing_values_warning(self, columns: list[str], warn: bool = False) -> None:
        """Check for missing values in the specified columns."""
        if self.is_nan(columns=columns):
            message: str = f"Missing values found in {columns}."
            if warn:
                warnings.warn(message=message)
            else:
                raise ValueError(message)

    def transform(
        self,
        columns: list[str] | str,
        func: Callable,
        group_columns: Columns = None,
    ) -> None:
        """Transform the DataFrame using the specified method."""
        columns = columns_to_list(columns=columns)
        group_columns = columns_to_list(columns=group_columns)
        self.check_columns_existence(
            columns=columns + group_columns if group_columns else [],
            check_range="all",
        )
        if group_columns:
            self.missing_values_warning(columns=group_columns)
            self.df[columns] = self.df.groupby(by=group_columns)[columns].transform(
                func=func
            )
        else:
            self.df[columns] = self.df[columns].transform(func=func)

    def check_columns_existence(
        self,
        columns: list[str] | str,
        check_range: Literal["all", "classifications", "characteristics"] = "all",
        warning: bool = False,
    ) -> None:
        columns = columns_to_list(columns=columns)
        if check_range == "all":
            check_columns = set(self.df.columns)
        elif check_range == "classifications":
            if self.classifications is None:
                raise ValueError("No classifications found.")
            check_columns = set(self.classifications)
        elif check_range == "characteristics":
            check_columns = set(self.firm_characteristics)
        else:
            raise ValueError("Invalid check_range value.")
        # Check if the required columns are present in the DataFrame
        missing_columns: set[str] = set(columns) - check_columns
        if missing_columns:
            message: str = (
                f"Missing columns in PanelData({self.name}): {missing_columns}"
            )
            if warning:
                warnings.warn(message=message)
            else:
                raise ValueError(message)


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    pp(df)
    panel_data: PanelData = PanelData(
        df=df,
        name="panel",
        # id="ids",
        # time="dates",
        # classifications="industry",
    )
    pp(panel_data)
    pp(panel_data.firm_characteristics)
