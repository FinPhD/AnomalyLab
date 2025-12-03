from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources  # noqa: F401
from typing import Literal, Optional, Union

import pandas as pd
from pandas import DataFrame

from anomalylab.empirical import (
    Correlation,
    FamaMacBethRegression,
    Persistence,
    PortfolioAnalysis,
    Summary,
)
from anomalylab.preprocess import FillNa, Normalize, OutlierHandler, Shift
from anomalylab.structure import PanelData, TimeSeries
from anomalylab.utils import Columns, Scalar, pp
from anomalylab.visualization import FormatExcel


@dataclass
class Panel:
    _df: DataFrame = field(repr=False)
    name: Optional[str] = None
    id: str = "permno"
    time: str = "date"
    frequency: Literal["D", "M", "Y"] = "M"
    ret: Optional[str] = None
    classifications: Optional[list[str] | str] = None
    drop_all_chars_missing: bool = False
    is_copy: bool = False

    def __post_init__(self) -> None:
        self.panel_data: PanelData = PanelData(
            df=self._df,
            name=self.name,
            id=self.id,
            time=self.time,
            frequency=self.frequency,
            ret=self.ret,
            classifications=self.classifications,
            drop_all_chars_missing=self.drop_all_chars_missing,
            is_copy=self.is_copy,
        )
        self.firm_characteristics = self.panel_data.firm_characteristics
        self._normalize_processor = None
        self._fillna_processor = None
        self._winsorize_processor = None
        self._shift_processor = None
        self._summary_processor = None
        self._correlation_processor = None
        self._persistence_processor = None
        self._portfolio_analysis_processor = None
        self._fm_preprocessor = None
        self._format_preprocessor = None

    def __repr__(self) -> str:
        return repr(self.panel_data)

    def get_original_data(self) -> DataFrame:
        return self._df

    def get_processed_data(self) -> DataFrame:
        return self.panel_data.df

    @property
    def normalize_processor(self) -> Normalize:
        if self._normalize_processor is None:
            self._normalize_processor = Normalize(panel_data=self.panel_data)
        return self._normalize_processor

    @property
    def fillna_processor(self) -> FillNa:
        if self._fillna_processor is None:
            self._fillna_processor = FillNa(panel_data=self.panel_data)
        return self._fillna_processor

    @property
    def winsorize_processor(self) -> OutlierHandler:
        if self._winsorize_processor is None:
            self._winsorize_processor = OutlierHandler(panel_data=self.panel_data)
        return self._winsorize_processor

    @property
    def shift_processor(self) -> Shift:
        if self._shift_processor is None:
            self._shift_processor = Shift(panel_data=self.panel_data)
        return self._shift_processor

    @property
    def summary_processor(self) -> Summary:
        if self._summary_processor is None:
            self._summary_processor = Summary(panel_data=self.panel_data)
        return self._summary_processor

    @property
    def correlation_processor(self) -> Correlation:
        if self._correlation_processor is None:
            self._correlation_processor = Correlation(panel_data=self.panel_data)
        return self._correlation_processor

    @property
    def persistence_processor(self) -> Persistence:
        if self._persistence_processor is None:
            self._persistence_processor = Persistence(panel_data=self.panel_data)
        return self._persistence_processor

    def portfolio_analysis_processor(
        self,
        endog: Optional[str] = None,
        weight: Optional[str] = None,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
    ) -> PortfolioAnalysis:
        self._portfolio_analysis_processor = PortfolioAnalysis(
            panel_data=self.panel_data,
            endog=endog,
            weight=weight,
            models=models,
            factors_series=factors_series,
        )
        return self._portfolio_analysis_processor

    @property
    def fm_preprocessor(self) -> FamaMacBethRegression:
        self._fm_preprocessor = FamaMacBethRegression(panel_data=self.panel_data)
        return self._fm_preprocessor

    def format_preprocessor(self, path: str) -> FormatExcel:
        self._format_preprocessor = FormatExcel(path=path)
        return self._format_preprocessor

    def normalize(
        self,
        columns: Columns = None,
        method: Literal["zscore", "rank"] = "zscore",
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        fillna_zero_after_norm: bool = False,
    ) -> Panel:
        """
        Normalizes specified columns of the DataFrame using the chosen method.

        This method converts the provided column arguments into lists, constructs
        the final set of columns to be processed, and applies the normalization
        method to those columns. Grouping can be applied based on specified
        group columns. The normalization can also exclude certain columns
        as specified.

        Args:
            columns (Columns, optional): The columns to normalize. Defaults to None.
            method (Literal["zscore", "rank"], optional): The normalization method
                to use. Defaults to "zscore".
            group_columns (Columns, optional): The columns to group by before
                normalization. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from
                normalization. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all
                characteristics or not. Defaults to True.
            fillna_zero_after_norm (bool): If True, fills NaN values with zero after normalization.
                Defaults to False.

        Returns:
            Normalize: The instance of the Normalize class with updated state.

        Note:
            This method modifies the `panel_data` attribute to indicate that
            normalization has been performed.
        """
        self.panel_data = self.normalize_processor.normalize(
            columns=columns,
            method=method,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            fillna_zero_after_norm=fillna_zero_after_norm,
        ).panel_data
        return self

    def fillna(
        self,
        columns: Columns = None,
        method: Literal["mean", "median", "constant"] = "mean",
        value: Optional[Union[float, int]] = None,
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Panel:
        """
        Fills missing values in specified columns of the DataFrame using the chosen method.

        This method allows users to specify which columns to fill missing values in,
        the method of filling (mean, median, or a constant value), and any grouping
        columns to apply the filling operation. It also offers the option to exclude
        certain columns from processing. The method will check for warnings regarding
        missing values and normalization status.

        Args:
            columns (Columns, optional): The columns to fill missing values in. Defaults to None.
            method (Literal["mean", "median", "constant"], optional): The method to use for
                filling missing values. Defaults to "mean".
            value (Optional[Union[float, int]], optional): The constant value to use if the
                method is 'constant'. Defaults to None.
            group_columns (Columns, optional): The columns to group by during filling. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from filling. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics.
                Defaults to True.

        Returns:
            FillNa: The instance of the FillNa class with updated state.

        Note:
            This method modifies the `panel_data` attribute to indicate that missing
            values have been filled.
        """
        self.panel_data = self.fillna_processor.fillna(
            columns=columns,
            method=method,
            value=value,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        ).panel_data
        return self

    def fill_group_column(self, group_column: str, value: Scalar) -> Panel:
        """Fills missing values in the specified group column with a constant value.

        Args:
            group_column (str): The name of the group column to fill.
            value (Scalar): The value used for filling.

        Returns:
            `FillNaN`: Returns the instance of the `FillNaN` class for method chaining.
        """
        self.panel_data = self.fillna_processor.fill_group_column(
            group_column=group_column, value=value
        ).panel_data
        return self

    def winsorize(
        self,
        columns: Columns = None,
        method: Literal["winsorize", "truncate"] = "winsorize",
        limits: tuple[float, float] = (0.01, 0.01),
        group_columns: Optional[list[str] | str] = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Panel:
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
        self.panel_data = self.winsorize_processor.winsorize(
            columns=columns,
            method=method,
            limits=limits,
            group_columns=group_columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
        ).panel_data
        return self

    def shift(
        self,
        columns: Columns = None,
        periods: int | list[int] = 1,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        drop_original: bool = False,
        dropna: bool = False,
    ) -> Panel:
        """
        Shifts specified columns of the panel data by a given number of periods.

        This method shifts the data in the specified columns either forward or
        backward in time, depending on the value of `periods`. It constructs
        the list of columns to be processed and handles the merging of the
        shifted data back into the original DataFrame. The user can specify
        whether to drop the original columns and whether to keep only rows
        with matching time and id when merging.

        Args:
            columns (Columns, optional): The columns to shift. Defaults to None.
            periods (int | list[int], optional): The number of periods to shift.
                Can be a single integer or a list of integers. Defaults to 1.
            no_process_columns (Columns, optional): The columns to exclude
                from shifting. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process
                all characteristics or not. Defaults to True.
            drop_original (bool, optional): Whether to drop the original columns
                after shifting. Defaults to False.
            dropna (bool, optional): Whether to only keep rows with matching
                time and id after merging. Defaults to False.

        Returns:
            Shift: The instance of the Shift class with updated state.

        Raises:
            NotImplementedError: Only monthly frequency is supported.
            warnings: The data has already been shifted.

        """
        self.panel_data = self.shift_processor.shift(
            columns=columns,
            periods=periods,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            drop_original=drop_original,
            dropna=dropna,
        ).panel_data
        return self

    def summary(
        self,
        columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
        decimal: Optional[int] = None,
    ) -> DataFrame:
        """
        Computes average statistics for specified columns over time periods.

        This method constructs the list of columns to process and calculates various
        statistics for each column, including mean, standard deviation, skewness,
        kurtosis, minimum, median, maximum, and count. The results are averaged
        across time periods and returned as a DataFrame.

        Args:
            columns (Columns, optional): The columns to calculate statistics for. Defaults to None.
            no_process_columns (Columns, optional): The columns to exclude from processing. Defaults to None.
            process_all_characteristics (bool, optional): Whether to process all characteristics or not.
                Defaults to True.
            decimal (Optional[int], optional): The number of decimal places to round the results to.
                Defaults to None.

        Returns:
            DataFrame: A DataFrame containing the average statistics for the specified columns.

        Note:
            The DataFrame includes the statistics rounded to the specified number of decimal places,
            and the count of non-null values is formatted as an integer string.
        """
        return self.summary_processor.average_statistics(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

    def correlation(
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
        return self.correlation_processor.average_correlation(
            columns=columns,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

    def persistence(
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
        return self.persistence_processor.average_persistence(
            columns=columns,
            periods=periods,
            no_process_columns=no_process_columns,
            process_all_characteristics=process_all_characteristics,
            decimal=decimal,
        )

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
        return self.persistence_processor.transition_matrix(
            var=var,
            group=group,
            lag=lag,
            draw=draw,
            path=path,
            decimal=decimal,
        )

    def group(
        self,
        endog: str,
        weight: str,
        vars: Union[str, list[str]],
        groups: Union[int, list[int]],
        sort_type: Literal["independent", "dependent"] = "independent",
        inplace: bool = False,
    ) -> Optional[pd.DataFrame]:
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
        if inplace:
            self.panel_data.df = self.portfolio_analysis_processor(
                endog=endog, weight=weight
            ).GroupN(
                vars=vars,
                groups=groups,
                sort_type=sort_type,
            )
        else:
            return self.portfolio_analysis_processor(endog=endog, weight=weight).GroupN(
                vars=vars,
                groups=groups,
                sort_type=sort_type,
            )

    def univariate_analysis(
        self,
        endog: str,
        weight: str,
        core_var: str,
        core_g: int,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
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
            endog (Optional[str]): The name of the endogenous variable for analysis.
            weight (Optional[str]): The name of the variable representing portfolio weights.
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
        return self.portfolio_analysis_processor(
            endog=endog, weight=weight, models=models, factors_series=factors_series
        ).univariate_analysis(
            core_var=core_var,
            core_g=core_g,
            format=format,
            decimal=decimal,
            factor_return=factor_return,
            already_grouped=already_grouped,
            is_endog_return=is_endog_return,
        )

    def bivariate_analysis(
        self,
        endog: str,
        weight: str,
        sort_var: str,
        core_var: str,
        sort_g: int,
        core_g: int,
        models: Optional[dict[str, list[str]]] = None,
        factors_series: Optional[TimeSeries] = None,
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
            endog (Optional[str]): The name of the endogenous variable for analysis.
            weight (Optional[str]): The name of the variable representing portfolio weights.
            sort_var (str): The sorting variable for grouping.
            core_var (str): The core variable for analysis.
            sort_g (int): The group number for portfolio grouping of the sorting variable.
            core_g (int): The group number for portfolio grouping of the core variable.
            pivot (bool): Whether to pivot the results table. Defaults to True.
            format (bool): Whether to format the output for display. Defaults to False.
            type (str): Type of grouping, can be 'dependent' or 'independent'. Defaults to 'dependent'.
            decimal (Optional[int]): The number of decimal places to round to. Defaults to None.
            factor_return (bool): Whether to output factor returns in the analysis. Defaults to False.
            already_grouped (bool): If True, skips the grouping step assuming data has been pre-grouped.
                Defaults to False.
            is_endog_return (bool): Whether the dependent variable is a return. Defaults to True.

        Returns:
            tuple: A tuple containing the equal-weighted and value-weighted results DataFrames.
        """
        return self.portfolio_analysis_processor(
            endog=endog, weight=weight, models=models, factors_series=factors_series
        ).bivariate_analysis(
            sort_var=sort_var,
            core_var=core_var,
            sort_g=sort_g,
            core_g=core_g,
            pivot=pivot,
            format=format,
            sort_type=sort_type,
            decimal=decimal,
            factor_return=factor_return,
            already_grouped=already_grouped,
            is_endog_return=is_endog_return,
        )

    def fm_reg(
        self,
        endog: Optional[str] = None,
        exog: Optional[list[str]] = None,
        regs: Optional[list[list[str] | dict[str, list[str]]]] = None,
        exog_order: Optional[list[str]] = None,
        reg_names: Optional[list[str]] = None,
        weight: Optional[str] = None,
        industry: Optional[str] = None,
        industry_weighed_method: Literal["value", "equal"] = "value",
        is_winsorize: bool = False,
        is_normalize: bool = False,
        decimal: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> DataFrame:
        """Fits the Fama-MacBeth regression model and returns the results DataFrame or intermediate results.

        This method fits the Fama-MacBeth regression model, processes data
        (including winsorization, industry weighting, normalization), and
        returns either the final regression results or intermediate results.

        Args:
            endog (Optional[str]): Name of the dependent variable.
            exog (Optional[list[str] | str]): List of exogenous variables or a single string.
            regs (Optional[list[list[str] | dict[str, list[str]]] | list | dict]): Model specifications.
            exog_order (Optional[list[str]]): Order of exogenous variables for output.
            reg_names (Optional[list[str]]): Names for the models in the output.
            weight (Optional[str]): Column name for weight in value-weighted calculations.
            industry (Optional[str]): Industry column for weighted calculations.
            industry_weighed_method (Literal["value", "equal"]): Method for weighting industries.
            is_winsorize (bool): Indicates whether to apply winsorization.
            is_normalize (bool): Indicates whether to normalize exogenous variables.
            decimal (Optional[int]): Number of decimal places for rounding in output.
            return_intermediate (bool): If True, returns the intermediate results (e.g., coefficients for each time period).

        Returns:
            DataFrame: DataFrame containing the regression results or intermediate results based on `return_intermediate`.
        """
        return self.fm_preprocessor.fit(
            endog=endog,
            exog=exog,
            regs=regs,
            exog_order=exog_order,
            reg_names=reg_names,
            weight=weight,
            industry=industry,
            industry_weighed_method=industry_weighed_method,
            is_winsorize=is_winsorize,
            is_normalize=is_normalize,
            decimal=decimal,
            return_intermediate=return_intermediate,
        )

    def format_excel(
        self,
        path: str,
        align=True,
        line=True,
        convert_brackets=False,
        adjust_col_widths=False,
    ) -> None:
        """Processes and formats Excel files.

        - If the provided path is a directory, it formats all Excel files in that directory.
        - If the provided path is a file, it formats that specific Excel file.

        Args:
            path (str): The directory or file path of the Excel file(s) to format.
            align (bool): Whether to apply text alignment. Default is True.
            line (bool): Whether to apply borders. Default is True.
            convert_brackets (bool): Whether to convert brackets. Default is False.
            auto_adjust (bool): Whether to adjust column widths. Default is False.
        """
        self.format_preprocessor(path=path).process(
            align=align,
            line=line,
            convert_brackets=convert_brackets,
            adjust_col_widths=adjust_col_widths,
        )


if __name__ == "__main__":
    from anomalylab.datasets import DataSet

    df: DataFrame = DataSet.get_panel_data()
    ts: DataFrame = DataSet.get_time_series_data()
    Models: dict[str, list[str]] = {
        "CAPM": ["MKT(3F)"],
        "FF3": ["MKT(3F)", "SMB(3F)", "HML(3F)"],
        "FF5": ["MKT(5F)", "SMB(5F)", "HML(5F)", "RMW(5F)", "CMA(5F)"],
    }

    panel = Panel(
        df,
        name="Stocks",
        id="permno",
        time="date",
        frequency="M",
        ret="return",
        classifications="industry",
        drop_all_chars_missing=True,
        is_copy=False,
    )
    time_series: TimeSeries = TimeSeries(
        df=ts, name="Factor Series", time="date", frequency="M", is_copy=False
    )
    pp(panel)

    panel.fill_group_column(group_column="industry", value="Other")
    panel.fillna(
        # columns="MktCap",
        method="mean",
        value=0,
        group_columns="date",
        # no_process_columns="MktCap",
        # process_all_characteristics=True,
    )

    panel.winsorize(method="winsorize", group_columns="date")
    pp(panel)

    # panel.normalize(
    #     # columns="MktCap",
    #     method="zscore",
    #     group_columns="date",
    #     # no_process_columns="MktCap",
    #     # process_all_characteristics=False,
    # )
    # panel.shift(periods=1, drop_original=False)

    # summary = panel.summary()
    # pp(summary)

    # correlation = panel.correlation()
    # pp(correlation)

    # persistence = panel.persistence(periods=[1, 3, 6, 12, 36, 60])
    # pp(persistence)
    # pp(
    #     panel.transition_matrix(
    #         var="MktCap",
    #         group=10,
    #         lag=12,
    #         draw=False,
    #         # path=str(resources.files("anomalylab.datasets")) + "/transition_matrix.png",
    #         path="...",
    #         decimal=2,
    #     )
    # )

    panel.group("return", "MktCap", "Illiq", 10, inplace=True)

    uni_ew, uni_vw = panel.univariate_analysis(
        "return",
        "MktCap",
        "Illiq",
        10,
        Models,
        time_series,
        factor_return=False,
        already_grouped=True,
    )
    pp(uni_ew)
    pp(uni_vw)

    bi_ew, bi_vw = panel.bivariate_analysis(
        "return",
        "MktCap",
        "Illiq",
        "IdioVol",
        5,
        5,
        Models,
        time_series,
        True,
        False,
        "dependent",
        factor_return=False,
    )
    pp(bi_ew)
    pp(bi_vw)

    fm_result = panel.fm_reg(
        regs=[
            ["return", "MktCap"],
            ["return", "Illiq"],
            ["return", "IdioVol"],
            ["return", "MktCap", "Illiq", "IdioVol"],
        ],
        exog_order=["MktCap", "Illiq", "IdioVol"],
        weight="MktCap",
        industry="industry",
        industry_weighed_method="value",
        is_winsorize=False,
        is_normalize=True,
    )
    pp(fm_result)

    # output_file_path = "..."
    # with pd.ExcelWriter(output_file_path) as writer:
    #     summary.to_excel(writer, sheet_name="summary")
    #     correlation.to_excel(writer, sheet_name="correlation")
    #     persistence.to_excel(writer, sheet_name="persistence")
    #     uni_ew.to_excel(writer, sheet_name="uni_ew")
    #     uni_vw.to_excel(writer, sheet_name="uni_vw")
    #     bi_ew.to_excel(writer, sheet_name="bi_ew")
    #     bi_vw.to_excel(writer, sheet_name="bi_vw")
    #     fm_result.to_excel(writer, sheet_name="fm_result")

    # panel.format_excel(
    #     output_file_path,
    #     align=True,
    #     line=True,
    #     convert_brackets=False,
    #     adjust_col_widths=True,
    # )
