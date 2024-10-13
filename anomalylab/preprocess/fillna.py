from __future__ import annotations

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class FillMethod:
    series: Series
    value: Optional[Scalar] = None

    def mean(self) -> Series:
        return self.series.fillna(value=self.series.mean())

    def median(self) -> Series:
        return self.series.fillna(value=self.series.median())

    def constant(self) -> Series:
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
        """Fills missing values in a `Series` using the specified method.

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
        """Warn if missing values are not found in the fill_columns or if the data has already been normalized."""
        if not self.panel_data.df[fill_columns].any().any():
            warnings.warn(message=f"Missing values not found in {fill_columns}.")
        if self.panel_data.normalize:
            warnings.warn(
                message=f"The data has already been normalized and the missing values have been filled 0."
            )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="panel", classifications="industry")

    fill_nan: FillNa = FillNa(panel_data=panel)
    fill_nan.fill_group_column(
        group_column="industry",
        value="Other",
    ).fillna(
        # columns="size",
        method="mean",
        group_columns="time",
        # no_process_columns="size",
        process_all_characteristics=True,
    )
