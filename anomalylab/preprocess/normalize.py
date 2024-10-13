from __future__ import annotations

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


class NormalizeMethod:
    @staticmethod
    def zscore(df: DataFrame) -> DataFrame:
        return (df - np.mean(df, axis=0)) / np.std(df, axis=0, ddof=1)

    @staticmethod
    def rank(df: DataFrame) -> DataFrame:
        return 2 * (df.rank(method="average") - 1) / (len(df) - 1) - 1

    @classmethod
    def call_method(cls, method: str, df: DataFrame) -> DataFrame:
        if hasattr(cls, method):
            return getattr(cls, method)(df).fillna(value=0)
        else:
            raise AttributeError(
                f"Method '{method}' not found, use 'zscore' or 'rank'."
            )



@dataclass
class Normalize(Preprocessor):

    def normalize(
        self,
        columns: Columns = None,
        method: Literal["zscore", "rank"] = "zscore",
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Normalize:
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
        # Normalize the selected columns
        self.panel_data.transform(
            columns=columns,
            func=lambda df: NormalizeMethod.call_method(method=method, df=df),
            group_columns=group_columns,
        )
        self.panel_data.normalize = True
        return self


if __name__ == "__main__":
    from anomalylab.datasets import DataSet
    from anomalylab.preprocess.fillna import FillNa

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="panel", classifications="industry")
    # fill_nan: FillNaN = FillNaN(panel_data=panel)
    # fill_nan.fill_group_column(
    #     group_column="industry",
    #     value="Other",
    # )
    norm: Normalize = Normalize(panel_data=panel)
    norm.normalize(
        # columns="size",
        method="zscore",
        group_columns="time",
        no_process_columns="size",
    )
    panel = norm.panel_data
    pp(panel)
