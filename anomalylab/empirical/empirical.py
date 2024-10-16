from anomalylab.config import *
from anomalylab.preprocess.preprocessor import Preprocessor
from anomalylab.utils.imports import *
from anomalylab.utils.utils import *


@dataclass
class Empirical(Preprocessor):
    decimal: int = 2


if __name__ == "__main__":
    ...
