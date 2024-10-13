from anomalylab.utils.imports import *

Scalar = Union[str, int, float, bool, Timestamp, Timedelta]
Columns = Optional[list[str] | str]
Info = Union[dict[str, Any], None]
RegModel = dict[str, list[str]]


@dataclass
class RegModels:
    models: list[RegModel]

    def __post_init__(self):
        self.dependent = list({key for item in self.models for key in item.keys()})
        self.exogenous: list[str] = sorted(list(
            {
                value
                for item in self.models
                for value in chain.from_iterable(item.values())
            }
        ))


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


def singleton(cls) -> Callable:
    """
    A singleton pattern decorator for a class.
    ------
    This decorator function takes a class as input and returns a function \n
    that ensures only one instance of the class is created and returned.

    Usages:
    ------
        ```python
        @singleton
        class MyClass:
            def __init__(self, arg1, arg2):
                self.arg1 = arg1
                self.arg2 = arg2

        instance1 = MyClass("value1", "value2")
        instance2 = MyClass("value3", "value4")

        print(instance1 is instance2)
        ```
    """

    instances: dict = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
