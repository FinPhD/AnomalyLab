from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from pandas import DataFrame, Series
from scipy.stats.mstats import winsorize as winsorization

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.preprocess.truncate import truncate as truncation
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, columns_to_list, pp


class OutlierMethod:
    """
    A class for applying winsorization and truncation methods to a Pandas Series.

    This class provides methods to handle outliers in data by either winsorizing
    (capping extreme values) or truncating (removing values outside specified limits)
    the values in a Series.

    Attributes:
        None
    """

    @staticmethod
    def winsorize(
        series: Series,
        limits: tuple[float, float] = (0.01, 0.01),
    ) -> Series:
        """
        Winsorizes the values in the Series based on the specified limits.

        This method replaces the extreme values of the Series with the values at
        the specified quantiles defined by the limits. Values below the lower limit
        are set to the lower limit value, and values above the upper limit are set
        to the upper limit value. Missing values (NaNs) are preserved.

        Args:
            series (Series): The input Pandas Series to be winsorized.
            limits (tuple[float, float], optional): The limits for winsorization,
                defined as quantiles. Defaults to (0.01, 0.01).

        Returns:
            Series: A new Series with winsorized values.
        """
        return Series(
            data=np.where(
                series.isnull(),
                np.nan,
                winsorization(np.ma.masked_invalid(series), limits=limits),
            ),
            index=series.index,
        )

    @staticmethod
    def truncate(
        series: Series,
        limits: tuple[float, float] = (0.01, 0.01),
    ) -> Series:
        """
        Truncates the values in the Series based on the specified limits.

        This method replaces values in the Series that are outside the specified
        quantile limits with NaN. Values below the lower limit are set to NaN,
        as well as values above the upper limit.

        Args:
            series (Series): The input Pandas Series to be truncated.
            limits (tuple[float, float], optional): The limits for truncation,
                defined as quantiles. Defaults to (0.01, 0.01).

        Returns:
            Series: A new Series with truncated values.
        """
        return Series(
            data=np.where(
                series.isnull(),
                np.nan,
                truncation(np.ma.masked_invalid(series), limits=limits),
            ),
            index=series.index,
        )

    @classmethod
    def call_method(
        cls,
        method: str,
        series: Series,
        limits: tuple[float, float],
    ) -> Series:
        """
        Calls a specified method (winsorize or truncate) on the given Series.

        This class method checks if the specified method exists within the class.
        If it does, it calls that method with the provided Series and limits.
        If the method does not exist, it raises an AttributeError.

        Args:
            cls: The class that is calling this method (WinsorizeMethod).
            method (str): The name of the method to call ('winsorize' or 'truncate').
            series (Series): The input Pandas Series to be processed.
            limits (tuple[float, float]): The limits for the method being called.

        Returns:
            Series: The modified Series after applying the specified method.

        Raises:
            AttributeError: If the specified method does not exist.
        """
        if hasattr(cls, method):
            return getattr(cls, method)(series=series, limits=limits)
        else:
            raise AttributeError(
                f"Method '{method}' not found, use 'winsorize' or 'truncate'."
            )


@dataclass
class OutlierHandler(Preprocessor):
    """
    A class for applying winsorization and truncation methods to panel data.

    This class extends the Preprocessor class and provides functionality to
    winsorize or truncate specified columns of the panel data, helping to handle
    outliers effectively. Users can specify the method, limits, and which columns
    to process.

    Attributes:
        panel_data (PanelData): The panel data object containing the data to be processed.
    """

    def winsorize(
        self,
        columns: Columns = None,
        method: Literal["winsorize", "truncate"] = "winsorize",
        limits: tuple[float, float] = (0.01, 0.01),
        group_columns: Optional[list[str] | str] = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> OutlierHandler:
        """
        Applies winsorization or truncation to specified columns of the panel data.

        This method allows users to specify which columns to process, the method
        to apply (winsorize or truncate), and the limits for the operation. The
        transformed data is then applied back to the panel data.

        Args:
            columns (Columns, optional): The columns to apply winsorization or truncation. Defaults to None.
            method (Literal["winsorize", "truncate"], optional): The method to use for handling outliers.
                Defaults to "winsorize".
            limits (tuple[float, float], optional): The limits for winsorization or truncation, defined as
                quantiles. Defaults to (0.01, 0.01).
            group_columns (list[str] | str, optional): The columns to group by during the transformation.
                Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from processing. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics or not.
                Defaults to True.

        Returns:
            Winsorize: The instance of the Winsorize class with updated state.

        Note:
            This method modifies the `panel_data` attribute to indicate which method was applied to handle outliers.
        """
        # Convert inputs to list
        columns = columns_to_list(columns=columns)
        group_columns = columns_to_list(columns=group_columns)
        no_process_columns = columns_to_list(columns=no_process_columns)

        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        )

        # Perform winsorization or truncation
        self.panel_data.transform(
            columns=columns,
            func=lambda series: OutlierMethod.call_method(
                method=method,
                series=series,
                limits=limits,
            ),
            group_columns=group_columns,
        )
        self.panel_data.outliers = (
            "Winsorized" if method == "winsorize" else "Truncated"
        )
        return self


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    winsorize = OutlierHandler(panel_data=panel)
    winsorize.winsorize(
        # columns="MktCap",
        method="winsorize",
        limits=(0.01, 0.01),
        group_columns="date",
        # no_process_columns=["MktCap"],
    )

    panel = winsorize.panel_data
    pp(panel)
    pp(panel.df.head())
