from __future__ import annotations

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


class WinsorizeMethod:
    @staticmethod
    def winsorize(
        series: Series,
        limits: tuple[float, float] = (0.01, 0.01),
    ) -> Series:
        return pd.Series(
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
        return series.where(
            cond=(series >= series.quantile(limits[0]))
            & (series <= series.quantile(1 - limits[1])),
            other=np.nan,
        )

    @classmethod
    def call_method(
        cls,
        method: str,
        series: Series,
        limits: tuple[float, float],
    ) -> Series:
        if hasattr(cls, method):
            return getattr(cls, method)(series=series, limits=limits)
        else:
            raise AttributeError(
                f"Method '{method}' not found, use 'winsorize' or 'truncate'."
            )


@dataclass
class Winsorize(Preprocessor):
    panel_data: PanelData

    def winsorize(
        self,
        columns: Columns = None,
        method: Literal["winsorize", "truncate"] = "winsorize",
        limits: tuple[float, float] = (0.01, 0.01),
        group_columns: list[str] | str = "time",
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Winsorize:
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
            func=lambda series: WinsorizeMethod.call_method(
                method=method,
                series=series,
                limits=limits,
            ),
            group_columns=group_columns,
        )
        self.panel_data.outliers = method
        return self


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")

    winsorize = Winsorize(panel_data=panel)
    winsorize.winsorize(
        # columns="MktCap",
        method="winsorize",
        limits=(0.01, 0.01),
        group_columns="time",
        # no_process_columns=["MktCap"],
    )

    panel = winsorize.panel_data
    pp(panel)
    pp(panel.df.head())
