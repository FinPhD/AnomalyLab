import copy
import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import pandas as pd
from pandas import DataFrame

from anomalylab.structure.data import Data
from anomalylab.utils import Columns, columns_to_list, pp


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
            The column name for the firm identifier. Defaults to "permno".
        time (str):
            The column name for the time identifier. Defaults to "date".
        frequency (Literal["D", "M", "Y"]):
            The frequency of the data. Defaults to "M".
        ret (str):
            The column name for the excess return. Defaults to None.
        classifications (list[str]):
            The list of classification columns.
        drop_all_chars_missing (bool):
            Whether to drop all rows where all firm_characteristics are missing. Defaults to False.
    """

    id: str = "permno"
    time: str = "date"
    frequency: Literal["D", "M", "Y"] = "M"
    ret: Optional[str] = None
    classifications: Optional[list[str] | str] = None
    drop_all_chars_missing: bool = False
    is_copy: bool = False

    def set_flag(self) -> None:
        """Set default flags for the `PanelData` object."""
        self.fillna = False
        self.normalize = False
        self.shift = False
        self.outliers: Literal["Unprocessed", "Winsorized", "Truncated"] = "Unprocessed"

    def __repr__(self) -> str:
        return (
            f"PanelData({self.name}), "  # todo: add frequency
            f"Classifications={self.classifications}, "
            f"Fillna={self.fillna}, "
            f"Normalize={self.normalize}, "
            f"Shift={self.shift}, "
            f"Outliers={self.outliers}"
        )

    def _preprocess(self) -> None:
        """
        Preprocess the `DataFrame` by identifying firm characteristics.

        This method identifies remaining columns as firm characteristics, excluding classifications.
        """
        self.df[self.id] = self.df[self.id].astype(int)
        if not isinstance(self.df[self.time].dtype, pd.PeriodDtype):
            self.df[self.time] = pd.to_datetime(self.df[self.time], format="ISO8601")
            self.df[self.time] = self.df[self.time].dt.to_period(freq=self.frequency)
        self.df.sort_values(by=[self.time, self.id], inplace=True)
        basic_column = (
            [self.id, self.time] if self.ret is None else [self.id, self.time, self.ret]
        )
        # Identify remaining columns and set them as firm characteristics, excluding classifications
        self.firm_characteristics: set[str] = set(
            filter(
                lambda x: (
                    x
                    not in basic_column
                    + (
                        self.classifications
                        if isinstance(self.classifications, list)
                        else []
                    )
                ),
                self.df.columns,
            )
        )
        self.clean_data()

    def clean_data(self) -> None:
        """Function to clean data"""
        if self.drop_all_chars_missing:
            # Count the number of rows before cleaning
            initial_row_count: int = len(self.df)

            # Drop rows where all firm_characteristics are missing
            self.df.dropna(
                subset=list(self.firm_characteristics), how="all", inplace=True
            )

            # Count the number of rows after cleaning
            final_row_count: int = len(self.df)

            # Print corresponding message based on the number of rows removed
            rows_removed: int = initial_row_count - final_row_count
            if rows_removed > 0:
                print(
                    f"{rows_removed} rows with all missing firm characteristics have been removed."
                )
            else:
                print("No rows with all missing firm characteristics were found.")

    def _check_columns(self) -> None:
        """Check if the required columns are present in the DataFrame.

        Raises:
            ValueError: If any duplicate column names are found in the DataFrame.
            ValueError: If any required columns are missing from the DataFrame.
            ValueError: If there are no firm characteristics remaining after checking.
        """
        if self.is_copy:
            self.df = copy.deepcopy(self.df)

        # Check for duplicate column names
        duplicated_columns = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicated_columns:
            raise ValueError(
                f"Duplicate column names found in the DataFrame: {duplicated_columns}"
            )

        if isinstance(self.classifications, str):
            self.classifications = [self.classifications]
        # Check if the required columns are present in the DataFrame
        basic_column = (
            [self.id, self.time] if self.ret is None else [self.id, self.time, self.ret]
        )
        required_columns: set[str] = set(basic_column + (self.classifications or []))
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
        use_warning: bool = False,
    ) -> None:
        columns = columns_to_list(columns=columns)
        if check_range == "all":
            check_columns = set(self.df.columns)
        elif check_range == "classifications":
            if self.classifications is None:
                raise ValueError("No classifications found.")
            check_columns = set(self.classifications)
        elif check_range == "characteristics":
            if self.firm_characteristics is None:
                raise ValueError("No firm characteristics found.")
            check_columns = set(self.firm_characteristics)
        else:
            raise ValueError("Invalid check_range value.")
        # Check if the required columns are present in the DataFrame
        missing_columns: set[str] = set(columns) - check_columns
        if missing_columns:
            message: str = (
                f"Missing columns in PanelData({self.name}): {missing_columns}"
            )
            if use_warning:
                warnings.warn(message=message)
            else:
                raise ValueError(message)


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    pp(df)

    panel_data: PanelData = PanelData(
        df=df,
        name="Stocks",
        id="permno",
        time="date",
        ret="return",
        classifications="industry",
        drop_all_chars_missing=True,
        is_copy=False,
    )
    pp(panel_data)
    pp(panel_data.df)
    pp(panel_data.firm_characteristics)
