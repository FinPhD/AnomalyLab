from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from anomalylab.structure import PanelData


@dataclass
class Preprocessor(ABC):
    """
    Abstract base class for preprocessing panel data.

    This class serves as a foundation for data preprocessing tasks on panel data,
    encapsulating shared functionality and attributes. The preprocessing methods
    will typically operate on specified columns, allowing for flexibility in
    data handling.

    Attributes:
        panel_data (PanelData): The panel data object that contains the data to be processed.
    """

    panel_data: PanelData

    def __post_init__(self) -> None:
        """
        Initializes the Preprocessor instance by creating a copy of the panel_data.

        This method is automatically called after the instance is created to ensure
        that any modifications to the panel_data do not affect the original data.
        It also initializes the id and time attributes from the panel_data.
        """
        self.panel_data = self.panel_data.copy()
        self.id = self.panel_data.id
        self.time = self.panel_data.time

    def construct_process_columns(
        self,
        columns: list[str],
        no_process_columns: list[str],
        process_all_characteristics: bool = True,
    ) -> list[str]:
        """
        Constructs a list of columns to be processed based on provided criteria.

        This method determines which columns to process based on the input
        parameters. If `process_all_characteristics` is True and no specific
        columns are provided, it will set the columns to all characteristics
        excluding those in `no_process_columns`. If `process_all_characteristics`
        is False and no columns are provided, a ValueError is raised.

        Args:
            columns (list[str]): The list of columns specified for processing.
            no_process_columns (list[str]): The list of columns that should not be processed.
            process_all_characteristics (bool, optional): Flag indicating whether to process
                all characteristics. Defaults to True.

        Returns:
            list[str]: The final list of columns to be processed.

        Raises:
            ValueError: If `process_all_characteristics` is False and no columns are provided.
        """
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
