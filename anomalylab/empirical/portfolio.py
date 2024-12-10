from pandas.core.frame import DataFrame

from anomalylab.config import *
from anomalylab.empirical.empirical import Empirical
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *

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
        type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Group variables into portfolios based on specified groups.

        This method creates portfolios for the specified variables in the panel data.

        Args:
            vars (list of str): List of variables to group.
            groups (list of int): List of integers defining the number of groups for each variable.
            type (str, optional): Type of grouping, can be 'dependent' to adjust based on the previous variable.

        Returns:
            DataFrame: A DataFrame with new columns for grouped variables.
        """
        # Ensure vars is a list
        if isinstance(vars, str):
            vars = [vars]

        if isinstance(groups, int):
            groups = [groups]

        groups_adj = [[0, 0.3, 0.7, 1] if g == 3 else g for g in groups]

        out_df = self.panel_data.df.dropna(
            subset=[
                self.time,
                self.id,
                self.endog,
                self.weight,
                *vars,
            ]
        ).copy()

        # Adjust group definitions
        group_col = [self.time]
        for i, var in enumerate(vars):
            if type == "dependent" and i > 0:
                group_col.append(f"{vars[i-1]}_g{groups[i-1]}")
                # Grouping dependent on the previous variable
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
                    .set_index(f"level_{i+1}")
                    .drop(group_col, axis=1)
                )
            else:
                # Grouping independently
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
                    .set_index(f"level_{1}")
                    .drop(self.time, axis=1)
                )

            # Convert to integer
            out_df[f"{var}_g{groups[i]}"] = out_df[f"{var}_g{groups[i]}"].astype(int)

        return out_df

    def _claculate_value(self, df: pd.DataFrame, decimal: Optional[int] = None) -> dict:
        """Calculate various portfolio performance metrics.

        This method computes mean returns, t-values, Sharpe ratios, and model-adjusted alpha and t values.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.

        Returns:
            dict: A dictionary containing computed metrics.
        """
        stat_dict = self._calculate_mean_and_t_value(df)
        factors_dict = self._calculate_alpha_and_t_value(df)
        sharpe_dict = self._calculate_sharpe(df, decimal)

        return {**stat_dict, **factors_dict, **sharpe_dict}

    def _calculate_mean_and_t_value(self, df: pd.DataFrame) -> dict:
        """Calculate mean and t-value for the dependent variable.

        This method computes the mean return and its t-value assuming the null hypothesis
        that the mean is zero.

        Args:
            df (DataFrame): The DataFrame containing the relevant data for calculations.

        Returns:
            dict: A dictionary with mean, t-value, and p-value.
        """
        stat_dict = {}
        T = df[self.time].nunique()
        lag = math.ceil(4 * (T / 100) ** (4 / 25))

        Y = df[self.endog].values
        X = pd.DataFrame({"constant": [1] * len(df[self.endog])}).values
        reg = sm.OLS(Y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
        )

        mean_value = reg.params[0]
        t_value = reg.tvalues[0]
        p_value = reg.pvalues[0]
        stat_dict["Return"] = mean_value
        stat_dict["t"] = t_value
        stat_dict["p"] = p_value

        return stat_dict

    def _calculate_alpha_and_t_value(self, df: pd.DataFrame) -> dict:
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

    def _calculate_sharpe(self, df: pd.DataFrame, decimal: Optional[int] = 0) -> dict:
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

        Returns:
            tuple: A tuple containing the equal-weighted and value-weighted results DataFrames.
        """
        data_d: DataFrame = self.GroupN(
            core_var,
            core_g,
        )

        ew_ret_d: Series = data_d.groupby([self.time, f"{core_var}_g{core_g}"])[
            self.endog
        ].mean()
        ew_ret_d.index.names = [self.time, core_var]
        vw_ret_d: Series = data_d.groupby([self.time, f"{core_var}_g{core_g}"]).apply(
            lambda x: np.average(x[self.endog], weights=x[self.weight]),  # type: ignore
            include_groups=False,
        )  # type: ignore
        vw_ret_d.index.names = [self.time, core_var]

        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            """Process each group to calculate differences and prepare the output.

            This function computes the difference between the highest portfolio and the lowest
            portfolio, as well as prepares the DataFrame for further analysis.

            Args:
                group (DataFrame): The grouped DataFrame for which to process data.

            Returns:
                DataFrame: The processed DataFrame with differences and averages.
            """
            group = group.sort_index(axis=0, level=[0, 1])

            core_diff = group.iloc[core_g - 1] - group.iloc[0]
            new_index = pd.MultiIndex.from_tuples(
                [(group.index.get_level_values(0)[0], "Diff")],
                names=[self.time, core_var],
            )
            core_diff = pd.Series(core_diff, index=new_index)

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
        ) -> pd.DataFrame:
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
                results[key] = self._claculate_value(sr, decimal=decimal)

            data = []
            for key, values in results.items():
                values[core_var] = key
                data.append(values)

            combined_results = pd.DataFrame(data)

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
                :, ~combined_results.columns.str.endswith("p")
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
        type: str = "dependent",
        decimal: Optional[int] = None,
        factor_return: bool = False,
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
            type (str): Type of grouping, can be 'dependent' or 'independent'. Defaults to 'dependent'.
            decimal (Optional[int]): The number of decimal places to round to. Defaults to None.

        Returns:
            tuple: A tuple containing the equal-weighted and value-weighted results DataFrames.
        """
        data_d = self.GroupN(
            [sort_var, core_var],
            [sort_g, core_g],
            type=type,
        )

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

        def process_group(group: pd.DataFrame) -> pd.DataFrame:
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

        ew_ret_d = ew_ret_d.reset_index()
        ew_ret_d = ew_ret_d.pivot(
            index=[self.time, sort_var], columns=core_var, values=self.endog
        )

        vw_ret_d.name = self.endog
        vw_ret_d = vw_ret_d.reset_index()
        vw_ret_d = vw_ret_d.pivot(
            index=[self.time, sort_var], columns=core_var, values=self.endog
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

        def generate_time_series_dict(df: pd.DataFrame) -> dict:
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
            df: pd.DataFrame, pivot: bool = pivot, format: bool = format
        ) -> pd.DataFrame:
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
                value_dict = self._claculate_value(series, decimal=decimal)
                results[key] = value_dict

            data = []
            for key, values in results.items():
                values[sort_var] = key[0]
                values[core_var] = key[1]
                data.append(values)

            combined_results = pd.DataFrame(data)

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
                :, ~combined_results.columns.str.endswith("p")
            ]

            def reorder_diff_avg(df: pd.DataFrame) -> pd.DataFrame:
                """Reorder the rows and columns of a DataFrame to place 'Diff' before 'Avg'.

                This function rearranges the DataFrame to improve readability.

                Args:
                    df (pd.DataFrame): The DataFrame to reorder.

                Returns:
                    pd.DataFrame: The reordered DataFrame.
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

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")
    time_series: TimeSeries = TimeSeries(df=ts, name="Factor Series")

    portfolio = PortfolioAnalysis(
        panel,
        endog="return",
        weight="MktCap",
        models=Models,
        factors_series=time_series,
    )

    group = portfolio.GroupN("Illiq", 10)
    pp(group)

    uni_ew, uni_vw = portfolio.univariate_analysis("Illiq", 10, factor_return=False)
    pp(uni_ew)
    pp(uni_vw)

    bi_ew, bi_vw = portfolio.bivariate_analysis(
        "Illiq", "IdioVol", 10, 10, False, False, "dependent", factor_return=False
    )
    pp(bi_ew)
    pp(bi_vw)
