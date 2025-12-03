import math
import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame, Series

from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils import pp, round_to_string

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class PortfolioAnalysis(Empirical):
    """
    A class for performing portfolio analysis on empirical financial data.

    This class extends the Empirical class and is designed to analyze portfolios
    based on specified endogenous variables, weights, and models. It also allows
    for the inclusion of factor series for more comprehensive analysis.

    Attributes:
        endog (Optional[str]): The name of the endogenous variable for analysis.
        weight (Optional[str]): The name of the variable representing portfolio weights.
        models (Optional[dict[str, list[str]]]): A dictionary mapping model names to lists of exogenous variables.
        factors_series (Optional[TimeSeries]): A TimeSeries object representing factor models.
    """

    endog: Optional[str] = None
    weight: Optional[str] = None
    models: Optional[dict[str, list[str]]] = None
    factors_series: Optional[TimeSeries] = None

    def __post_init__(self):
        """Post-initialization method to validate input parameters and prepare data.

        This method checks if the models are provided without corresponding factor series
        and initializes relevant attributes from the panel data.
        """
        super().__post_init__()
        for field in ["endog", "weight"]:
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be provided")
        if self.models is not None and self.factors_series is None:
            raise ValueError(
                "If 'models' is provided, 'factors_series' must also be provided!"
            )
        self.ft_series = getattr(self.factors_series, "df", None)

    def add_star(self, mean_val: float, p_val: float) -> str:
        """Add significance stars to mean values based on p-value thresholds.

        Args:
            mean_val (float): The mean value to which stars are to be added.
            p_val (float): The p-value corresponding to the mean value.

        Returns:
            str: The mean value formatted with appropriate significance stars.
        """
        if p_val <= 0.01:
            return f"{mean_val}***"
        elif 0.01 < p_val <= 0.05:
            return f"{mean_val}**"
        elif 0.05 < p_val <= 0.1:
            return f"{mean_val}*"
        else:
            return f"{mean_val}"

    def GroupN(
        self,
        vars: Union[str, list[str]],
        groups: Union[int, list[int]],
        sort_type: Literal["independent", "dependent"] = "independent",
        inplace: bool = False,
    ) -> Optional[DataFrame]:
        """Group variables into portfolios based on specified groups.

        This method creates portfolios for the specified variables in the panel data.

        Args:
            vars (Union[str, list[str]]): List of variables to group.
            groups (Union[int, list[int]]): List of integers defining the number of groups for each variable.
            sort_type (Literal["independent", "dependent"]): Type of sorting, either 'independent'
                (group within time period) or 'dependent' (group based on previous variable).
                Defaults to "independent".
            inplace (bool): If True, modifies the original dataset and returns None. Defaults to False.

        Returns:
            DataFrame: If inplace=False (default), returns a new DataFrame with grouped variables.
            None: If inplace=True, modifies the original dataset and returns None.
        """
        # Convert single input to list
        if isinstance(vars, str):
            vars = [vars]

        if isinstance(groups, int):
            groups = [groups]

        groups_adj = [[0, 0.3, 0.7, 1] if g == 3 else g for g in groups]

        # Track original data length for missing value reporting
        original_length = len(self.panel_data.df)

        # Create working copy with NA removal
        out_df = self.panel_data.df.dropna(
            subset=[
                self.time,
                self.id,
                self.endog,
                self.weight,
                *vars,
            ]
        ).copy()

        n_dropped = original_length - len(out_df)

        group_col = [self.time]
        for i, var in enumerate(vars):
            if sort_type == "dependent" and i > 0:
                group_col.append(f"{vars[i - 1]}_g{groups[i - 1]}")
                out_df[f"{var}_g{groups[i]}"] = (
                    out_df.groupby(group_col, observed=False)[var]
                    .apply(
                        lambda x: pd.qcut(
                            x,
                            q=groups_adj[i],
                            labels=[j for j in range(1, groups[i] + 1)],
                        )
                    )
                    .reset_index()
                    .set_index(f"level_{i + 1}")
                    .drop(group_col, axis=1)
                )
            else:
                # Independent sorting
                out_df[f"{var}_g{groups[i]}"] = (
                    out_df.groupby(self.time)[var]
                    .apply(
                        lambda x: pd.qcut(
                            x,
                            q=groups_adj[i],
                            labels=[j for j in range(1, groups[i] + 1)],
                        )
                    )
                    .reset_index()
                    .set_index("level_1")
                    .drop(self.time, axis=1)
                )

            # Ensure integer type for groups
            out_df[f"{var}_g{groups[i]}"] = out_df[f"{var}_g{groups[i]}"].astype(int)

        # Handle inplace operation
        if inplace:
            self.panel_data.df = out_df.copy()
            if n_dropped > 0:
                print(f"Removed {n_dropped} rows with missing values during grouping.")
        else:
            return out_df

    def _claculate_value(
        self, df: DataFrame, decimal: Optional[int] = None, is_endog_return: bool = True
    ) -> dict:
        """Calculate various portfolio performance metrics.

        This method computes mean returns, t-values, Sharpe ratios, and model-adjusted alpha and t values.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.
            decimal (Optional[int]): The number of decimal places for formatting. Defaults to None.
            is_endog_return (bool): Whether the dependent variable is a return. Defaults to True.

        Returns:
            dict: A dictionary containing computed metrics.
        """
        stat_dict = self._calculate_mean_and_t_value(df, is_endog_return)

        if is_endog_return:
            factors_dict = self._calculate_alpha_and_t_value(df)
            sharpe_dict = self._calculate_sharpe(df, decimal)
            return {**stat_dict, **factors_dict, **sharpe_dict}

        return stat_dict

    def _calculate_mean_and_t_value(
        self, df: DataFrame, is_endog_return: bool = True
    ) -> dict:
        """Calculate mean and t-value for the dependent variable.

        This method computes the mean return and its t-value assuming the null hypothesis
        that the mean is zero.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.
            is_endog_return (bool): Whether the dependent variable is a return. Defaults to True.

        Returns:
            dict: A dictionary with mean, t-value, and p-value.
        """
        stat_dict = {}
        T = df[self.time].nunique()
        lag = math.ceil(4 * (T / 100) ** (4 / 25))

        Y = df[self.endog].values
        X = DataFrame({"constant": [1] * len(df[self.endog])}).values
        reg = sm.OLS(Y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
        )

        mean_value = reg.params[0]
        t_value = reg.tvalues[0]
        p_value = reg.pvalues[0]

        key_name = "Return" if is_endog_return else self.endog
        stat_dict[key_name] = mean_value
        stat_dict["t"] = t_value
        stat_dict["p"] = p_value

        return stat_dict

    def _calculate_alpha_and_t_value(self, df: DataFrame) -> dict:
        """Calculate alpha and t-value for specified models.

        This method computes alpha values and their t-statistics for various regression models
        based on the provided factors.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.

        Returns:
            dict: A dictionary containing alpha values, t-values, and p-values for each model.
        """
        if self.ft_series is not None:
            if self.models is not None:
                factors_dict = {}
                df = pd.merge(
                    df,
                    self.ft_series,
                    left_on=self.time,
                    right_on=self.factors_series.time,
                    how="left",
                )
                for model, factors in self.models.items():
                    sub = df.dropna(subset=[self.time, self.endog] + factors)
                    T = sub[self.time].nunique()
                    lag = math.ceil(4 * (T / 100) ** (4 / 25))

                    Y = df[self.endog].values
                    X = df[factors].values
                    X = sm.add_constant(X)
                    reg = sm.OLS(Y, X).fit(
                        cov_type="HAC",
                        cov_kwds={"maxlags": lag, "use_correction": False},
                    )

                    alpha_value = reg.params[0]
                    t_value = reg.tvalues[0]
                    p_value = reg.pvalues[0]

                    factors_dict[f"{model}-α"] = alpha_value
                    factors_dict[f"{model}-t"] = t_value
                    factors_dict[f"{model}-p"] = p_value

                return factors_dict
            else:
                return {}
        else:
            return {}

    def _calculate_sharpe(self, df: DataFrame, decimal: Optional[int] = 0) -> dict:
        """Calculate the Sharpe ratio for the dependent variable.

        This method computes the annualized Sharpe ratio based on the mean and standard deviation
        of the returns.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.

        Returns:
            dict: A dictionary containing the Sharpe ratio.
        """
        sharpe_dict = {}
        series = df[self.endog]
        ret_mean = series.mean() * np.sqrt(12)
        ret_std = series.std()
        sharpe = ret_mean / ret_std
        sharpe = f"{sharpe:.{decimal or self.decimal}f}"
        sharpe_dict["Sharpe"] = sharpe

        return sharpe_dict

    def univariate_analysis(
        self,
        core_var: str,
        core_g: int,
        format: bool = False,
        decimal: Optional[int] = None,
        factor_return: bool = False,
        already_grouped: bool = False,
        is_endog_return: bool = True,
    ) -> tuple:
        """Perform univariate analysis on the specified core variable.

        This method calculates equal-weighted (EW) and value-weighted (VW) returns for a
        given core variable, grouping the data into portfolios. It also computes the difference
        between the highest portfolio and the lowest portfolio.

        Args:
            core_var (str): The core variable for which the analysis is to be performed.
            core_g (int): The group number for portfolio grouping of the core variable.
            format (bool): Whether to format the output for display. Defaults to False.
            decimal (Optional[int]): The number of decimal places for formatting. Defaults to None.
            factor_return (bool): Whether to output factor returns in the analysis. Defaults to False.
            already_grouped (bool): If True, skips the grouping step assuming data has been pre-grouped.
                Defaults to False.
            is_endog_return (bool): Whether the dependent variable is a return. Defaults to True.

        Returns:
            tuple: A tuple containing the equal-weighted and value-weighted results DataFrames.
        """

        if not already_grouped:
            data_d = self.GroupN(core_var, core_g)
        else:
            expected_col = f"{core_var}_g{core_g}"
            if expected_col not in self.panel_data.df.columns:
                raise ValueError(f"Pre-grouped column {expected_col} not found in data")
            data_d = self.panel_data.df.copy()

        ew_ret_d: Series = data_d.groupby([self.time, f"{core_var}_g{core_g}"])[
            self.endog
        ].mean()
        ew_ret_d.index.names = [self.time, core_var]
        vw_ret_d: Series = data_d.groupby([self.time, f"{core_var}_g{core_g}"]).apply(
            lambda x: np.average(x[self.endog], weights=x[self.weight]),  # type: ignore
            include_groups=False,
        )  # type: ignore
        vw_ret_d.index.names = [self.time, core_var]

        def process_group(group: DataFrame) -> Series:
            """Process each group to calculate differences and prepare the output.

            This function computes the difference between the highest portfolio and the lowest
            portfolio, as well as prepares the DataFrame for further analysis.

            Args:
                group (DataFrame): The grouped DataFrame for which to process data.

            Returns:
                Series: The processed Series with differences and averages.
            """
            group = group.sort_index(axis=0, level=[0, 1])

            core_diff = group.iloc[core_g - 1] - group.iloc[0]
            new_index = pd.MultiIndex.from_tuples(
                [(group.index.get_level_values(0)[0], "Diff")],
                names=[self.time, core_var],
            )
            core_diff = Series(core_diff, index=new_index)

            return pd.concat([group, core_diff])

        ew_ret_d = (
            ew_ret_d.groupby(level=0)
            .apply(process_group)
            .reset_index(level=0, drop=True)
        )
        vw_ret_d = (
            vw_ret_d.groupby(level=0)
            .apply(process_group)
            .reset_index(level=0, drop=True)
        )

        if factor_return:
            return ew_ret_d, vw_ret_d

        def generate_time_series_dict(series: Series) -> dict:
            """Generate a dictionary of time series data from the DataFrame.

            This function extracts time series for each unique group and stores them in a dictionary.

            Args:
                series (Series): The Series containing the time series data.

            Returns:
                dict: A dictionary with time series indexed by core variable.
            """
            time_series_dict = {}

            for core in series.index.get_level_values(core_var).unique():
                time_series_dict[core] = (
                    series.loc[(slice(None), core)].to_frame(self.endog).reset_index()
                )

            return time_series_dict

        def calculate_time_series_metrics(
            series: Series, format: bool = format
        ) -> DataFrame:
            """Calculate metrics for each time series and format results.

            This function computes performance metrics for each time series and formats the results
            based on the specified parameters.

            Args:
                series (Series): The Series containing the time series data.
                format (bool): Whether to format the output for display. Defaults to False.

            Returns:
                DataFrame: A DataFrame containing the calculated metrics.
            """
            time_series_dict = generate_time_series_dict(series)

            results = {}

            for key, sr in time_series_dict.items():
                results[key] = self._claculate_value(
                    sr, decimal=decimal, is_endog_return=is_endog_return
                )

            key_name = "Return" if is_endog_return else self.endog
            data = []
            for key, values in results.items():
                if key_name == core_var:
                    if key_name in values:
                        val = values.pop(key_name)
                        new_values = {f"{key_name}_val": val}
                        new_values.update(values)
                        values = new_values

                values[core_var] = key
                data.append(values)

            combined_results = DataFrame(data)

            combined_results.set_index(core_var, inplace=True)

            columns = combined_results.columns[:-1]

            for i in range(0, len(columns), 3):
                subset = combined_results.iloc[:, i : i + 3]

                subset.iloc[:, 0:2] = subset.iloc[:, 0:2].map(
                    func=round_to_string, decimal=decimal or self.decimal
                )

                if format:
                    subset.iloc[:, 1] = subset.iloc[:, 1].apply(lambda x: f"({x})")
                    subset.iloc[:, 0] = subset.apply(
                        lambda row: self.add_star(row.iloc[0], row.iloc[2]),
                        axis=1,
                    )

                combined_results.iloc[:, i : i + 3] = subset

            combined_results = combined_results.loc[
                :, ~combined_results.columns.str.match(r"(^p$|.*-p$)")
            ]

            return combined_results

        ew_table = calculate_time_series_metrics(ew_ret_d)
        vw_table = calculate_time_series_metrics(vw_ret_d)

        return ew_table, vw_table

    def bivariate_analysis(
        self,
        sort_var: str,
        core_var: str,
        sort_g: int,
        core_g: int,
        pivot: bool = True,
        format: bool = False,
        sort_type: str = "dependent",
        decimal: Optional[int] = None,
        factor_return: bool = False,
        already_grouped: bool = False,
        is_endog_return: bool = True,
    ) -> tuple:
        """Perform bivariate analysis on two specified variables.

        This method calculates both equal-weighted (EW) and value-weighted (VW) returns
        for two core variables, grouping the data into portfolios. It further computes
        the differences and averages for better comparative analysis.

        Args:
            sort_var (str): The sorting variable for grouping.
            core_var (str): The core variable for analysis.
            sort_g (int): The group number for portfolio grouping of the sorting variable.
            core_g (int): The group number for portfolio grouping of the core variable.
            pivot (bool): Whether to pivot the results table. Defaults to True.
            format (bool): Whether to format the output for display. Defaults to False.
            type (str): Type of grouping, can be 'dependent' or 'sort_type'. Defaults to 'dependent'.
            decimal (Optional[int]): The number of decimal places to round to. Defaults to None.
            factor_return (bool): Whether to output factor returns in the analysis. Defaults to False.
            already_grouped (bool): If True, skips the grouping step assuming data has been pre-grouped.
                Defaults to False.
            is_endog_return (bool): Whether the dependent variable is a return. Defaults to True.

        Returns:
            tuple: A tuple containing the equal-weighted and value-weighted results DataFrames.
        """

        if not already_grouped:
            data_d = self.GroupN(
                [sort_var, core_var],
                [sort_g, core_g],
                sort_type=sort_type,
            )
        else:
            # Check existence of both pre-grouped columns when using existing groups
            required_columns = [f"{sort_var}_g{sort_g}", f"{core_var}_g{core_g}"]
            # Validate all required grouping columns exist
            missing_cols = [
                col for col in required_columns if col not in self.panel_data.df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Pre-grouped columns missing: {', '.join(missing_cols)}. "
                    f"Required columns: {required_columns}"
                )
            data_d = self.panel_data.df.copy()

        ew_ret_d = data_d.groupby(
            [self.time, f"{sort_var}_g{sort_g}", f"{core_var}_g{core_g}"]
        )[self.endog].mean()
        ew_ret_d.index.names = [self.time, sort_var, core_var]
        vw_ret_d = data_d.groupby(
            [self.time, f"{sort_var}_g{sort_g}", f"{core_var}_g{core_g}"]
        ).apply(
            lambda x: np.average(x[self.endog], weights=x[self.weight]),  # type: ignore
            include_groups=False,
        )
        vw_ret_d.index.names = [self.time, sort_var, core_var]

        def process_group(group: DataFrame) -> DataFrame:
            """Process each group to calculate differences and averages.

            This function computes the difference between the highest portfolio and lowest portfolio,
            and averages among all portfolios.

            Args:
                group (DataFrame): The grouped DataFrame for processing.

            Returns:
                DataFrame: The processed DataFrame with differences and averages.
            """
            group.index = group.index.set_levels(  # type: ignore
                [group.index.levels[0], group.index.levels[1].astype(int)]  # type: ignore
            )
            group.columns = group.columns.astype(int)
            group = group.sort_index(axis=0, level=[0, 1])
            group = group.sort_index(axis=1)

            group["Diff"] = group.iloc[:, core_g - 1] - group.iloc[:, 0]
            group["Avg"] = group.iloc[:, :core_g].mean(axis=1)

            sort_diff = group.iloc[sort_g - 1] - group.iloc[0]
            sort_diff = sort_diff.to_frame().T
            sort_diff.index = pd.MultiIndex.from_tuples(
                [(group.index.get_level_values(0)[0], "Diff")],
                names=[self.time, sort_var],
            )

            sort_avg = group.iloc[:sort_g].mean().to_frame().T
            sort_avg.index = pd.MultiIndex.from_tuples(
                [(group.index.get_level_values(0)[0], "Avg")],
                names=[self.time, sort_var],
            )

            return pd.concat([group, sort_diff, sort_avg])

        # Handle potential name collision if endog is same as sort_var or core_var
        value_col = self.endog
        if value_col in [sort_var, core_var]:
            value_col = f"{self.endog}_val"

        ew_ret_d.name = value_col
        ew_ret_d = ew_ret_d.reset_index()
        ew_ret_d = ew_ret_d.pivot(
            index=[self.time, sort_var], columns=core_var, values=value_col
        )

        vw_ret_d.name = value_col
        vw_ret_d = vw_ret_d.reset_index()
        vw_ret_d = vw_ret_d.pivot(
            index=[self.time, sort_var], columns=core_var, values=value_col
        )

        ew_ret_d = (
            ew_ret_d.groupby(level=0)
            .apply(process_group)
            .reset_index(level=0, drop=True)
        )
        vw_ret_d = (
            vw_ret_d.groupby(level=0)
            .apply(process_group)
            .reset_index(level=0, drop=True)
        )

        if factor_return:
            return ew_ret_d, vw_ret_d

        def generate_time_series_dict(df: DataFrame) -> dict:
            """Generate a dictionary of time series data from the DataFrame.

            This function extracts time series for each unique combination of sorting and core variables.

            Args:
                df (DataFrame): The DataFrame containing the time series data.

            Returns:
                dict: A dictionary with time series indexed by sorting and core variables.
            """
            time_series_dict = {}

            for sort in df.index.get_level_values(sort_var).unique():
                for column in df.columns:
                    time_series = df.loc[(slice(None), sort), column]
                    time_series = (
                        time_series.to_frame(self.endog)
                        .reset_index(level=0)
                        .reset_index(drop=True)
                    )

                    time_series_dict[(sort, column)] = time_series

            return time_series_dict

        def calculate_time_series_metrics(
            df: DataFrame, pivot: bool = pivot, format: bool = format
        ) -> DataFrame:
            """Calculate metrics for each time series and format results.

            This function computes performance metrics for each time series and formats the results
            based on the specified parameters.

            Args:
                df (DataFrame): The DataFrame containing the time series data.
                pivot (bool): Whether to pivot the results table. Defaults to True.
                format (bool): Whether to format the output for display. Defaults to False.

            Returns:
                DataFrame: A DataFrame containing the calculated metrics.
            """
            time_series_dict = generate_time_series_dict(df)

            results = {}

            for key, series in time_series_dict.items():
                value_dict = self._claculate_value(
                    series, decimal=decimal, is_endog_return=is_endog_return
                )
                results[key] = value_dict

            key_name = "Return" if is_endog_return else self.endog
            data = []
            for key, values in results.items():
                if key_name in [sort_var, core_var]:
                    if key_name in values:
                        val = values.pop(key_name)
                        new_values = {f"{key_name}_val": val}
                        new_values.update(values)
                        values = new_values

                values[sort_var] = key[0]
                values[core_var] = key[1]
                data.append(values)

            combined_results = DataFrame(data)

            combined_results.set_index([sort_var, core_var], inplace=True)

            columns = combined_results.columns[:-1]

            for i in range(0, len(columns), 3):
                subset = combined_results.iloc[:, i : i + 3]

                subset.iloc[:, 0:2] = subset.iloc[:, 0:2].map(
                    func=round_to_string, decimal=decimal or self.decimal
                )

                if format:
                    subset.iloc[:, 1] = subset.iloc[:, 1].apply(lambda x: f"({x})")
                    subset.iloc[:, 0] = subset.apply(
                        lambda row: self.add_star(row.iloc[0], row.iloc[2]),
                        axis=1,
                    )

                combined_results.iloc[:, i : i + 3] = subset

            combined_results = combined_results.loc[
                :, ~combined_results.columns.str.match(r"(^p$|.*-p$)")
            ]

            def reorder_diff_avg(df: DataFrame) -> DataFrame:
                """Reorder the rows and columns of a DataFrame to place 'Diff' before 'Avg'.

                This function rearranges the DataFrame to improve readability.

                Args:
                    df (DataFrame): The DataFrame to reorder.

                Returns:
                    DataFrame: The reordered DataFrame.
                """
                columns_order = [
                    col for col in df.columns if col not in ["Diff", "Avg"]
                ] + ["Diff", "Avg"]
                df = df[columns_order]

                index_order = [
                    idx for idx in df.index if idx not in ["Diff", "Avg"]
                ] + ["Diff", "Avg"]
                df = df.loc[index_order]

                return df

            if pivot:
                cols = combined_results.columns[:-1]
                table = []
                for i in range(0, len(cols), 2):
                    subset = combined_results.iloc[:, i : i + 2]
                    subset.reset_index(inplace=True)
                    pivot_mean_table = subset.pivot(
                        index=sort_var,
                        columns=core_var,
                        values=list(subset.columns)[2],
                    )
                    pivot_t_table = subset.pivot(
                        index=sort_var,
                        columns=core_var,
                        values=list(subset.columns)[3],
                    )
                    pivot_mean_table = reorder_diff_avg(pivot_mean_table)
                    pivot_t_table = reorder_diff_avg(pivot_t_table)

                    temp_table = pd.concat(
                        [pivot_mean_table, pivot_t_table],
                        axis=0,
                        keys=[list(subset.columns)[2], list(subset.columns)[3]],
                    )
                    table.append(temp_table)
                return pd.concat(table)
            else:
                return combined_results

        ew_table = calculate_time_series_metrics(ew_ret_d)
        vw_table = calculate_time_series_metrics(vw_ret_d)

        return ew_table, vw_table


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    ts: DataFrame = DataSet.get_time_series_data()
    Models: dict[str, list[str]] = {
        "CAPM": ["MKT(3F)"],
        "FF3": ["MKT(3F)", "SMB(3F)", "HML(3F)"],
        "FF5": ["MKT(5F)", "SMB(5F)", "HML(5F)", "RMW(5F)", "CMA(5F)"],
    }

    panel: PanelData = PanelData(
        df=df, name="Stocks", ret="return", classifications="industry"
    )
    time_series: TimeSeries = TimeSeries(df=ts, name="Factor Series")

    portfolio = PortfolioAnalysis(
        panel,
        endog="IdioVol",
        weight="MktCap",
        # models=Models,
        # factors_series=time_series,
    )

    # portfolio.GroupN("Illiq", 10, inplace=True)
    portfolio.GroupN(["MktCap", "Illiq"], [5, 5], sort_type="dependent", inplace=True)

    uni_ew, uni_vw = portfolio.univariate_analysis(
        "Illiq",
        5,
        format=True,
        # factor_return=False,
        already_grouped=True,
        is_endog_return=False,
    )
    pp(uni_ew)
    pp(uni_vw)

    bi_ew, bi_vw = portfolio.bivariate_analysis(
        "MktCap",
        "Illiq",
        5,
        5,
        False,
        True,
        "dependent",
        # factor_return=False,
        already_grouped=True,
        is_endog_return=False,
    )
    pp(bi_ew)
    pp(bi_vw)
