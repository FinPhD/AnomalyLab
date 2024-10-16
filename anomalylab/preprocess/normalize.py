from __future__ import annotations

from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.structure import PanelData
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


class NormalizeMethod:
    @staticmethod
    def zscore(df: DataFrame) -> DataFrame:
        """
        Standardizes the DataFrame using Z-score normalization.

        This method computes the Z-score for each value in the DataFrame,
        which represents the number of standard deviations the value is from
        the mean. The result is a DataFrame with the same shape as the input
        where each element is transformed to its Z-score.

        Args:
            df (DataFrame): The input DataFrame to be normalized.

        Returns:
            DataFrame: A new DataFrame containing the Z-score normalized values.
        """
        return (df - np.mean(df, axis=0)) / np.std(df, axis=0, ddof=1)

    @staticmethod
    def rank(df: DataFrame) -> DataFrame:
        """
        Ranks the values in the DataFrame and rescales them to be in the range
        of -1 to 1.

        This method assigns ranks to the values in each column of the DataFrame
        using the average ranking method. The ranks are then rescaled to a
        range of -1 to 1, where -1 represents the lowest rank and 1
        represents the highest rank.

        Args:
            df (DataFrame): The input DataFrame to be ranked.

        Returns:
            DataFrame: A new DataFrame containing the rescaled ranks.
        """
        return 2 * (df.rank(method="average") - 1) / (len(df) - 1) - 1

    @classmethod
    def call_method(cls, method: str, df: DataFrame) -> DataFrame:
        """
        Calls a specified normalization method on the input DataFrame.

        This class method checks if the specified method exists within the
        class. If it does, it calls that method with the provided DataFrame
        and fills any resulting NaN values with zero. If the method does not
        exist, it raises an AttributeError.

        Args:
            cls: The class that is calling this method (NormalizeMethod).
            method (str): The name of the method to call ('zscore' or 'rank').
            df (DataFrame): The input DataFrame to be normalized.

        Returns:
            DataFrame: The normalized DataFrame after applying the specified method.

        Raises:
            AttributeError: If the specified method does not exist.
        """
        if hasattr(cls, method):
            return getattr(cls, method)(df).fillna(value=0)
        else:
            raise AttributeError(
                f"Method '{method}' not found, use 'zscore' or 'rank'."
            )


@dataclass
class Normalize(Preprocessor):
    """
    A data normalization class that extends the Preprocessor class.

    This class is responsible for normalizing specified columns of a DataFrame
    using various methods like Z-score or rank normalization. It also provides
    options to group the data and exclude certain columns from normalization.

    Attributes:
        panel_data: The DataFrame or panel data that will undergo normalization.
    """

    def normalize(
        self,
        columns: Columns = None,
        method: Literal["zscore", "rank"] = "zscore",
        group_columns: Columns = None,
        no_process_columns: Columns = None,
        process_all_characteristics: bool = True,
    ) -> Normalize:
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

        Returns:
            Normalize: The instance of the Normalize class with updated state.

        Note:
            This method modifies the `panel_data` attribute to indicate that
            normalization has been performed.
        """
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

    df: DataFrame = DataSet.get_panel_data()

    panel: PanelData = PanelData(df=df, name="Stocks", classifications="industry")
    norm: Normalize = Normalize(panel_data=panel)
    norm.normalize(
        # columns="MktCap",
        method="zscore",
        group_columns="time",
        # no_process_columns="MktCap",
    )

    panel = norm.panel_data
    pp(panel)
    pp(panel.df.head())
