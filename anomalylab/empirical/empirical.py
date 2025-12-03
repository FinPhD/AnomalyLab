from dataclasses import dataclass

from anomalylab.preprocess.preprocessor import Preprocessor


@dataclass
class Empirical(Preprocessor):
    decimal: int = 2


if __name__ == "__main__":
    ...
