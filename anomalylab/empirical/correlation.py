from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class Correlation(Empirical):
    def average_correlation(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )
        columns_number: int = len(columns)
        # Calculate the average correlation
        df_group = self.panel_data.df.groupby("time")[columns]
        merged_corr = np.ones((columns_number, columns_number))
        is_upper = True
        for method in ["pearson", "spearman"]:
            rows, cols = (
                np.triu_indices(columns_number, k=1)
                if is_upper
                else np.tril_indices(columns_number, k=-1)
            )
            merged_corr[rows, cols] = (
                df_group.corr(method=method)
                .groupby(level=1)
                .mean()
                .reindex(index=columns)
                .values[rows, cols]
            )
            is_upper = False
        return pd.DataFrame(data=merged_corr, index=columns, columns=columns).map(
            func=round_to_string, decimal=decimal or self.decimal
        )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet
    from anomalylab.preprocess.fillna import FillNa
    from anomalylab.preprocess.normalize import Normalize

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
    normalize: Normalize = Normalize(panel_data=panel)
    normalize.normalize(
        group_columns="time",
    )
    correlation: Correlation = Correlation(panel_data=panel)
    pp(correlation.average_correlation(decimal=3))
