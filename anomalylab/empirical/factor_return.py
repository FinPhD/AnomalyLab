from pandas.core.frame import DataFrame

from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.empirical.portfolio import PortfolioAnalysis
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *

if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    ts: DataFrame = DataSet.get_time_series_data()
    Models: dict[str, list[str]] = {
        "CAPM": ["MKT(3F)"],
        "FF3": ["MKT(3F)", "SMB(3F)", "HML(3F)"],
        "FF5": ["MKT(5F)", "SMB(5F)", "HML(5F)", "RMW(5F)", "CMA(5F)"],
    }

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")
    time_series: TimeSeries = TimeSeries(df=ts, name="Factor Series")

    portfolio = PortfolioAnalysis(
        panel,
        endog="return",
        weight="MktCap",
        # models=Models,
        # factors_series=time_series,
    )

    group = portfolio.GroupN(["MktCap", "Illiq", "IdioVol"], [3, 3, 3])
    pp(group)

    # uni_ew, uni_vw = portfolio.univariate_analysis("Illiq", 10)
    # pp(uni_ew)
    # pp(uni_vw)

    # bi_ew, bi_vw = portfolio.bivariate_analysis(
    #     "Illiq", "IdioVol", 10, 10, True, False, "dependent"
    # )
    # pp(bi_ew)
    # pp(bi_vw)
