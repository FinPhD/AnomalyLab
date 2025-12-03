import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame
from typing_extensions import Self


@dataclass
class Data(ABC):
    """Abstract class for `Panel` and `TimeSeries` class.

    Attributes:
        df (DataFrame): The DataFrame object that contains the data.
        name (str): The name of the object.
    """

    df: DataFrame
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """
        This method is called after the __init__ method.

        1. Check if the columns are valid.
        2. Preprocess the data.
        3. Set the flag if needed.
        """

        self._check_columns()
        self._preprocess()
        self.set_flag()

    def set_flag(self) -> None:
        """This method is meant to be overridden by subclasses to set flags."""
        pass

    def copy(self) -> Self:
        """Return a deep copy of the object."""
        return copy.deepcopy(self)

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the object."""

    @abstractmethod
    def _check_columns(self) -> None:
        """Check if the columns are valid."""

    @abstractmethod
    def _preprocess(self) -> None:
        """Preprocess the data."""


if __name__ == "__main__":
    pass
