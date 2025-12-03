from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Union

from pandas import DataFrame, Series

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, Scalar, columns_to_list, pp


@dataclass
class FillMethod:
    """
    A class to handle filling missing values in a Pandas Series.

    This class provides different methods to fill missing values (NaNs) in a
    given Series. The available methods include filling with the mean, median,
    or a specified constant value.

    Attributes:
        series (Series): The Pandas Series containing missing values to be filled.
        value (Optional[Scalar]): The constant value to fill missing entries, if
            using the constant fill method.
    """

    series: Series
    value: Optional[Scalar] = None

    def mean(self) -> Series:
        """
        Fills missing values in the Series with the mean of the Series.

        This method computes the mean of the Series and replaces all NaN
        values with this mean.

        Returns:
            Series: A new Series with NaN values replaced by the mean.
        """
        return self.series.fillna(value=self.series.mean())

    def median(self) -> Series:
        """
        Fills missing values in the Series with the median of the Series.

        This method computes the median of the Series and replaces all NaN
        values with this median.

        Returns:
            Series: A new Series with NaN values replaced by the median.
        """
        return self.series.fillna(value=self.series.median())

    def constant(self) -> Series:
        """
        Fills missing values in the Series with a specified constant value.

        This method replaces NaN values with a user-defined constant value.
        If no value is provided, it raises a ValueError.

        Returns:
            Series: A new Series with NaN values replaced by the constant value.

        Raises:
            ValueError: If the constant fill value is not specified.
        """
        if self.value is None:
            raise ValueError("Fill value is required for constant method.")
        return self.series.fillna(value=self.value)


@dataclass
class FillNa(Preprocessor):
    def fill(
        self,
        series: Series,
        value: Optional[Scalar] = None,
        method: Literal["mean", "median", "constant"] = "mean",
        call_internal: bool = False,
    ) -> Series:
        """Fill missing values in a `Series` using the specified method.

        Args:
            series (Series): The `Series` to fill.
            value (Optional[Scalar]): The value to use for filling (for 'constant' method).
            method (Literal["mean", "median", "constant"]): The method to use for filling missing values.
            call_internal (bool): A flag to call the method internally. Defaults to False.

        Returns:
            Series: The Series with missing values filled.

        Raises:
            ValueError: If all values are missing and no fill_value is provided.
            ValueError: If an invalid method is specified.
        """
        # If the method is called internally, do not check for missing values
        if not call_internal:
            # Check if there are missing values in the series
            if not series.isna().any():
                warnings.warn(message="Missing values not found in the series.")
        # If all values are missing, fill with the fill_value
        if series.isna().all():
            # If fill_value is None, raise an error
            if value is None:
                raise ValueError(
                    "All values are missing and no fill_value is provided."
                )
            return series.fillna(value)
        # Methods for filling missing values
        fill_method = FillMethod(series=series, value=value)
        if hasattr(fill_method, method):
            return getattr(fill_method, method)()
        else:
            raise AttributeError(f"Invalid method specified: {method}")

    def fill_group_column(
        self,
        group_column: str,
        value: Scalar,
    ) -> FillNa:
        """Fills missing values in the specified group column with a constant value.

        Args:
            group_column (str): The name of the group column to fill.
            value (Scalar): The value used for filling.

        Returns:
            `FillNaN`: Returns the instance of the `FillNaN` class for method chaining.
        """
        self.panel_data.df[group_column] = self.fill(
            series=self.panel_data.df[group_column],
            value=value,
            method="constant",
        )
        return self

    def fillna(
        self,
        columns: Columns = None,
        method: Literal["mean", "median", "constant"] = "mean",
        value: Optional[Union[float, int]] = None,
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> FillNa:
        """
        Fills missing values in specified columns of the DataFrame using the chosen method.

        This method allows users to specify which columns to fill missing values in,
        the method of filling (mean, median, or a constant value), and any grouping
        columns to apply the filling operation. It also offers the option to exclude
        certain columns from processing. The method will check for warnings regarding
        missing values and normalization status.

        Args:
            columns (Columns, optional): The columns to fill missing values in. Defaults to None.
            method (Literal["mean", "median", "constant"], optional): The method to use for
                filling missing values. Defaults to "mean".
            value (Optional[Union[float, int]], optional): The constant value to use if the
                method is 'constant'. Defaults to None.
            group_columns (Columns, optional): The columns to group by during filling. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from filling. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics.
                Defaults to True.

        Returns:
            FillNa: The instance of the FillNa class with updated state.

        Note:
            This method modifies the `panel_data` attribute to indicate that missing
            values have been filled.
        """
        # Convert columns to list
        columns = columns_to_list(columns)
        group_columns = columns_to_list(group_columns)
        no_process_columns = columns_to_list(no_process_columns)

        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        )

        # Warn if missing values are not found in the fill_columns
        # or if the data has already been normalized
        self._warning(fill_columns=columns)

        # Fillna the selected columns
        self.panel_data.transform(
            columns=columns,
            func=lambda series: self.fill(
                series=series,
                value=value,
                method=method,
                call_internal=True,
            ),
            group_columns=group_columns,
        )

        self.panel_data.fillna = True
        return self

    def _warning(self, fill_columns: list[str]) -> None:
        """
        Emit warnings regarding missing values and normalization status.

        This method performs the following checks:
        1. It verifies if there are any missing values in the specified `fill_columns`.
        - If no missing values are found, a warning is issued indicating this.
        2. It checks whether the data has already been normalized.
        - If normalized, a warning is issued that missing values were filled with zeros during normalization.
        3. It checks whether missing values were filled previously.
        - If so, a warning is issued to indicate that missing values have already been handled earlier.

        Args:
            fill_columns (list[str]): The list of columns to check for missing values.
        """
        if not self.panel_data.df[fill_columns].any().any():
            warnings.warn(message=f"Missing values not found in {fill_columns}.")
        if self.panel_data.normalize:
            warnings.warn(
                message="The data has already been normalized, and missing values have been filled with 0."
            )
        if self.panel_data.fillna:
            warnings.warn(
                message="The missing values have already been handled earlier."
            )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )

    fill_nan: FillNa = FillNa(panel_data=panel)
    fill_nan.fill_group_column(
        group_column="industry",
        value="Other",
    ).fillna(
        # columns="MktCap",
        method="mean",
        group_columns=["date", "industry"],
        # no_process_columns="MktCap",
        # process_all_characteristics=True,
    )

    panel = fill_nan.panel_data
    pp(panel)
    pp(panel.df.head())
