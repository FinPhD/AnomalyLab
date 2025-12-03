from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional, TypedDict, Union

from pandas import Series, Timedelta, Timestamp
from rich import print
from rich.panel import Panel as rich_Panel
from rich.pretty import Pretty

Scalar = Union[str, int, float, bool, Timestamp, Timedelta]
Columns = Optional[list[str] | str]
Info = Union[dict[str, Any], None]
RegModel = dict[str, list[str]]


@dataclass
class RegModels:
    models: list[RegModel]

    def __post_init__(self):
        self.dependent = sorted(
            list({key for item in self.models for key in item.keys()})
        )
        self.exogenous: list[str] = sorted(
            list(
                {
                    value
                    for item in self.models
                    for value in chain.from_iterable(item.values())
                }
            )
        )


class RegResult(TypedDict):
    params: Series
    tvalues: Series
    pvalues: Series
    mean_obs: str
    rsquared: float


def columns_to_list(columns: Columns) -> list[str]:
    if isinstance(columns, str):
        return [columns]
    elif columns is None:
        return []
    return columns


def round_to_string(value: int | float, decimal: int = 2) -> str:
    return f"{value:.{decimal}f}"


def get_significance_star(pvalue):
    return next(
        (
            stars
            for threshold, stars in [(0.01, "***"), (0.05, "**"), (0.1, "*")]
            if pvalue <= threshold
        ),
        "",
    )


def pp(object: Any) -> None:
    """Pretty print an object in a panel"""
    print(
        rich_Panel(
            renderable=Pretty(_object=object, expand_all=True),
            expand=False,
            subtitle_align="center",
        )
    )


if __name__ == "__main__":
    # Test data
    test_models: list[dict[str, list[str]]] = [
        {"y1": ["x1", "x2"]},
        {"y3": ["x2", "x3"]},
        {"y2": ["x3", "x4", "x5"]},
    ]

    # Create an instance of RegModels with test data
    reg_models = RegModels(test_models)

    # Print outputs to verify the behavior
    print("Dependent:", reg_models.dependent)
    print("Exogenous:", reg_models.exogenous)
