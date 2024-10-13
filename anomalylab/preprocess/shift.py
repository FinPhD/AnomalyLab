from __future__ import annotations

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class Shift(Preprocessor):
    def shift(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        drop_original: bool = False,
        dropna: bool = False,
    ) -> Shift:
        # Check if the data has already been shifted
        if self.panel_data.shift:
            raise ValueError("The data has already been shifted.")
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
            df_shift: DataFrame = self.panel_data.df[["time", "id"] + columns].copy()
            # Shift the columns
            df_shift["time"] += period
            # Merge the shifted columns
            self.panel_data.df = self.panel_data.df.merge(
                right=df_shift,
                on=["time", "id"],
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
    from anomalylab.preprocess.fillna import FillNa
    from anomalylab.preprocess.normalize import Normalize
    from anomalylab.preprocess.winsorize import Winsorize

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
    shift = Shift(panel_data=panel)
    shift.shift(
        # columns="size",
        # periods=[-1, 0, 1, 1],
        # no_process_columns="size",
        # dropna=True,
    )
    pp(panel.df)
