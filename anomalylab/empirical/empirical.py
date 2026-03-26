import math
from dataclasses import dataclass
from typing import Optional

from anomalylab.preprocess.preprocessor import Preprocessor


@dataclass
class Empirical(Preprocessor):
    decimal: int = 2

    @staticmethod
    def default_hac_lag(T: int, lag: Optional[int] = None) -> int:
        """Return the HAC lag length used for time-series inference.

        Notes:
            This package uses ``floor(4 * (T / 100) ** (2 / 9))`` as the default
            Bartlett-style automatic lag rule when ``lag`` is not supplied. We
            use the same rule for both portfolio analysis and Fama-MacBeth
            regression so that the default lag choice is transparent and
            consistent with a Bartlett-style HAC convention.

            References:
            - statsmodels cov_hac: if ``nlags=None``, then
              ``floor[4(T/100)^(2/9)]`` is used.
              https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html
            - linearmodels FamaMacBeth.fit
              https://bashtage.github.io/linearmodels/devel/panel/panel/linearmodels.panel.model.FamaMacBeth.fit.html
            - linearmodels kernel documentation
              https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.covariance.kernel_optimal_bandwidth.html

        """
        if lag is not None:
            if lag < 0:
                raise ValueError("lag must be non-negative")
            return lag
        return math.floor(4 * (T / 100) ** (2 / 9))


if __name__ == "__main__":
    ...
