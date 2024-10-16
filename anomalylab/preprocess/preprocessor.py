from __future__ import annotations

from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class Preprocessor(ABC):
    panel_data: PanelData

    def __post_init__(self) -> None:
        self.panel_data = self.panel_data.copy()
        self.id = self.panel_data.id
        self.time = self.panel_data.time

    def construct_process_columns(
        self,
        columns: list[str],
        no_process_columns: list[str],
        process_all_characteristics: bool = True,
    ) -> list[str]:
        if process_all_characteristics and columns == []:
            # Check if the columns exist
            self.panel_data.check_columns_existence(
                columns=no_process_columns,
                check_range="characteristics",
                use_warning=True,
            )
            # Set columns to the characteristics columns, excluding any no_process_columns
            columns = sorted(
                list(
                    set(self.panel_data.firm_characteristics)
                    - set(no_process_columns or [])
                )
            )
        elif (not process_all_characteristics) and (columns == []):
            raise ValueError(
                "If 'process_all_characteristics' is set to False, you must provide columns to process."
            )
        return columns
