from __future__ import annotations

import warnings
from dataclasses import dataclass

from pandas import DataFrame

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, columns_to_list, pp


@dataclass
class Shift(Preprocessor):
    """
    A class for shifting specified columns in panel data.

    This class extends the Preprocessor class and provides functionality to
    shift specified columns of the panel data by a given number of periods.
    It allows for flexible processing options such as dropping original columns
    and handling missing values.

    Attributes:
        panel_data (PanelData): The panel data object containing the data to be shifted.
    """

    def shift(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        drop_original: bool = False,
        dropna: bool = False,
    ) -> Shift:
        """
        Shifts specified columns of the panel data by a given number of periods.

        This method shifts the data in the specified columns either forward or
        backward in time, depending on the value of `periods`. It constructs
        the list of columns to be processed and handles the merging of the
        shifted data back into the original DataFrame. The user can specify
        whether to drop the original columns and whether to keep only rows
        with matching time and id when merging.

        Args:
            columns (Columns, optional): The columns to shift. Defaults to None.
            periods (int | list[int], optional): The number of periods to shift.
                Can be a single integer or a list of integers. Defaults to 1.
            no_process_columns (Columns, optional): The columns to exclude
                from shifting. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process
                all characteristics or not. Defaults to True.
            drop_original (bool, optional): Whether to drop the original columns
                after shifting. Defaults to False.
            dropna (bool, optional): Whether to only keep rows with matching
                time and id after merging. Defaults to False.

        Returns:
            Shift: The instance of the Shift class with updated state.

        Raises:
            NotImplementedError: Only monthly frequency is supported.
            warnings: The data has already been shifted.

        """

        # todo: add support for daily and yearly frequency
        if self.panel_data.frequency != "M":
            raise NotImplementedError("Only monthly frequency is supported.")

        # Check if the data has already been shifted
        if self.panel_data.shift:
            # raise ValueError("The data has already been shifted.")
            warnings.warn("The data has already been shifted. Proceed with caution.")

        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )

        # Shift the selected columns
        for period in set(
            filter(lambda x: x != 0, [periods] if isinstance(periods, int) else periods)
        ):
            # Copy the columns to shift
            df_shift: DataFrame = self.panel_data.df[
                [self.time, self.id] + columns
            ].copy()
            # Shift the columns
            df_shift[self.time] += period
            # Merge the shifted columns
            self.panel_data.df = self.panel_data.df.merge(
                right=df_shift,
                on=[self.time, self.id],
                # If dropna is True, only keep the rows with matching 'time' and 'id'
                how="inner" if dropna else "left",
                suffixes=("", f"({period})"),
            )

        # Drop the original columns
        if drop_original:
            self.panel_data.df = self.panel_data.df.drop(columns=columns)

        # Set the shift flag to True
        self.panel_data.shift = True
        return self


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    shift = Shift(panel_data=panel)
    shift.shift(
        # columns="MktCap",
        periods=[-1, 1],
        # no_process_columns="MktCap",
        # dropna=True,
    )

    panel = shift.panel_data
    pp(panel)
    pp(panel.df)
