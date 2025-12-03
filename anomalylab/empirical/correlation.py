from dataclasses import dataclass
from typing import Optional

import numpy as np
from pandas import DataFrame

from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, columns_to_list, pp, round_to_string


@dataclass
class Correlation(Empirical):
    """
    A class for calculating average correlation statistics from empirical panel data.

    This class extends the Empirical class and provides functionality to compute
    average correlation coefficients among specified columns in the panel data over
    time periods using different correlation methods.

    Attributes:
        panel_data (PanelData): The panel data object containing the data for correlation analysis.
    """

    def average_correlation(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """
        Computes average correlation coefficients for specified columns over time periods.

        This method constructs the list of columns to process and calculates the
        average correlation using both Pearson and Spearman methods. The results
        are returned in a DataFrame format, with correlations rounded to the specified
        number of decimal places.

        Args:
            columns (Columns, optional): The columns to calculate correlation for. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from processing. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics or not.
                Defaults to True.
            decimal (Optional[int], optional): The number of decimal places to round the results to.
                Defaults to None.

        Returns:
            DataFrame: A DataFrame containing the average correlation coefficients for the specified columns.
        """
        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )

        columns_number: int = len(columns)
        # Calculate the average correlation
        df_group = self.panel_data.df.groupby(self.time)[columns]
        merged_corr = np.ones(
            (columns_number, columns_number)
        )  # Initialize a matrix for correlations
        is_upper = True

        for method in ["spearman", "pearson"]:
            rows, cols = (
                np.triu_indices(columns_number, k=1)  # Get upper triangle indices
                if is_upper
                else np.tril_indices(columns_number, k=-1)  # Get lower triangle indices
            )
            merged_corr[rows, cols] = (
                df_group.corr(method=method)  # Calculate correlation
                .groupby(level=1)
                .mean()  # Average over time
                .reindex(index=columns)  # Reindex to ensure the order of columns
                .values[rows, cols]  # Fill the corresponding positions
            )
            is_upper = False  # Switch to lower triangle for the next method

        return DataFrame(data=merged_corr, index=columns, columns=columns).map(
            func=round_to_string,
            decimal=decimal or self.decimal,  # Round results to specified decimals
        )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    correlation: Correlation = Correlation(panel_data=panel)
    pp(correlation.average_correlation())
