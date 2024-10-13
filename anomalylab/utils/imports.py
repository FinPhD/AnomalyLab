import copy
import functools
import math
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from functools import partial, wraps
from itertools import chain
from types import SimpleNamespace
from typing import (Any, Callable, ClassVar, Generic, Iterable, Literal,
                    Optional, Sequence, TypedDict, TypeVar, Union,
                    get_type_hints)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from deprecated import deprecated
from linearmodels import FamaMacBeth
from numpy import float32, float64
from numpy.typing import NDArray
from pandas import (DataFrame, DatetimeIndex, Index, Interval, Period,
                    PeriodIndex, Series, Timedelta, Timestamp)
from pandas.arrays import PeriodArray
from rich import print
from rich.panel import Panel as rich_Panel
from rich.pretty import Pretty, pprint
from scipy.stats import kurtosis, skew
from scipy.stats.mstats import winsorize as winsorization
from tqdm import tqdm
from typing_extensions import NotRequired, Required, Self