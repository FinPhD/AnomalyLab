from importlib import resources  # noqa: F401
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from anomalylab.empirical.empirical import Empirical
from anomalylab.preprocess.shift import Shift
from anomalylab.structure import PanelData
from anomalylab.utils import Columns, columns_to_list, pp, round_to_string


@dataclass
class Persistence(Empirical):
    """
    A class for calculating average persistence statistics from empirical panel data
    and calculating the transition matrix for a specified variable and lag.

    This class extends the Empirical class and provides functionality to compute
    average autocorrelations (persistence) and transition matrix of specified variables over given time
    periods. The results can be formatted to a specified number of decimal places.

    Attributes:
        panel_data (PanelData): The panel data object containing the data for persistence analysis.
    """

    def average_persistence(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """
        Computes average persistence (autocorrelation) for specified columns over defined time periods.

        This method constructs the list of columns to process, sorts the DataFrame by date,
        creates lagged variables, and calculates the average correlations for each variable
        and lag period. The results are returned as a DataFrame.

        Args:
            columns (Columns, optional): The columns for which to calculate persistence. Defaults to None.
            periods (int | list[int], optional): The time periods (lags) to consider for autocorrelation.
                Defaults to 1.
            no_process_columns (Columns, optional): The columns to exclude from processing. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics or not.
                Defaults to True.
            decimal (Optional[int], optional): The number of decimal places to round the results to.
                Defaults to None.

        Returns:
            DataFrame: A DataFrame containing the average persistence for specified columns.

        Note:
            The resulting DataFrame contains the average correlations for each lag, formatted to the
            specified number of decimal places.
        """
        # Construct the columns to process
        columns = self.construct_process_columns(
            columns=columns_to_list(columns=columns),
            no_process_columns=columns_to_list(columns=no_process_columns),
            process_all_characteristics=process_all_characteristics,
        )

        # Sort the DataFrame by date
        periods = periods if isinstance(periods, list) else [periods]
        self.panel_data.df = self.panel_data.df.sort_values(by=self.time)

        # Create lagged variable
        panel_data = Shift(self.panel_data).shift(columns, periods).panel_data

        all_persistence = []
        for var in columns:
            all_monthly_corrs = []
            for lag in periods:
                # Store monthly correlations
                monthly_corrs = []

                # Iterate over each month
                for period, group in panel_data.df.groupby(self.time):
                    # Drop NaN values
                    group = group.dropna(subset=[var, f"{var}({lag})"])
                    if not group.empty:
                        corr = group[var].corr(group[f"{var}({lag})"])
                        monthly_corrs.append(
                            {f"{self.time}": period, "lag": lag, "corr": corr}
                        )

                # Append to the list of all correlations
                all_monthly_corrs.extend(monthly_corrs)

            # Convert to DataFrame
            all_monthly_corrs_df = DataFrame(all_monthly_corrs)

            # Calculate average monthly correlations
            mean_corrs_df = (
                all_monthly_corrs_df.groupby("lag")["corr"].mean().reset_index()
            )
            mean_corrs_df.rename(columns={"corr": var}, inplace=True)

            # Transpose the DataFrame
            transposed_mean_corrs_df = mean_corrs_df.set_index("lag").T
            all_persistence.append(transposed_mean_corrs_df)

        # Concatenate all DataFrames
        persistences = pd.concat(all_persistence, axis=0)

        # Format the DataFrame values to the specified number of decimal places
        return persistences.map(func=round_to_string, decimal=decimal or self.decimal)

    def transition_matrix(
        self,
        var: str,
        group: int,
        lag: int,
        draw: bool = False,
        path: Optional[str] = None,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """Calculate the transition matrix for a specified variable and lag.

        This method computes the transition matrix that shows how groups change over time based on
        the specified variable and lag. The groups are determined by portfolios of the variable.

        Args:
            var (str): The variable to compute the transition matrix for.
            group (int): The number of groups to create based on portfolios.
            lag (int): The lag period for the transition.
            draw (bool, optional): Whether to plot the transition matrix heatmap. Default is False.
            path (str, optional): The file path to save the heatmap if draw is True.
            decimal (Optional[int], optional): The number of decimal places to round the results to.
                Defaults to None.

        Returns:
            DataFrame: A DataFrame representing the transition matrix.
        """
        df = self.panel_data.df[[self.time, self.id, var]].copy()
        df = df.dropna(subset=[var])

        # Create groups based on portfolios of the variable
        group_adj = [0, 0.3, 0.7, 1] if group == 3 else group
        df["group"] = df.groupby(self.time)[var].transform(
            lambda x: pd.qcut(x, group_adj, labels=False)
        )

        df = df.sort_values(by=self.time)

        transition_matrix = np.zeros((group, group))

        unique_dates = np.sort(df[self.time].unique())

        for i in range(len(unique_dates) - lag):
            current_date = unique_dates[i]
            next_date = unique_dates[i + lag]

            current_groups = df[df[self.time] == current_date][[self.id, "group"]]
            next_groups = df[df[self.time] == next_date][[self.id, "group"]]

            merged = pd.merge(
                current_groups,
                next_groups,
                on=self.id,
                suffixes=("_current", "_next"),
            )

            merged["group_current"] = merged["group_current"].astype(int)
            merged["group_next"] = merged["group_next"].astype(int)

            for _, row in merged.iterrows():
                transition_matrix[row["group_current"], row["group_next"]] += 1

        # Calculate proportions for each row
        transition_matrix = transition_matrix / transition_matrix.sum(
            axis=1, keepdims=True
        )

        # Create DataFrame for the transition matrix
        transition_matrix_df = DataFrame(
            transition_matrix, columns=range(1, group + 1), index=range(1, group + 1)
        )

        if draw or path is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_matrix_df, annot=False, cmap="YlGnBu", cbar=True)
            plt.xlabel("Next Portfolio")
            plt.ylabel("Current Portfolio")

            if path is not None:
                plt.savefig(path)

            if draw:
                plt.show()

            plt.close()

        # Format the DataFrame values to the specified number of decimal places
        transition_matrix_df = transition_matrix_df.map(
            func=round_to_string, decimal=decimal or self.decimal
        )

        return transition_matrix_df


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    persistence = Persistence(panel)
    pp(persistence.average_persistence(periods=[1, 3, 6, 12, 36, 60]))
    pp(
        persistence.transition_matrix(
            "MktCap",
            10,
            12,
            draw=True,
            # path=str(resources.files("anomalylab.datasets")) + "/transition_matrix.png",
            # decimal=3,
        )
    )
