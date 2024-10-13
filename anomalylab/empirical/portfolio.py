from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class PortfolioAnalysis(Empirical):

    def univariate_analysis(self):
        raise NotImplementedError("This class is not implemented yet.")


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
