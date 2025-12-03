from __future__ import annotations

import numpy as np
import numpy.ma as ma
from scipy._lib._util import _contains_nan


def truncate(
    a,
    limits=None,
    inclusive=(True, True),
    inplace=False,
    axis=None,
    nan_policy="propagate",
):
    """Truncate the input array `a` by replacing values outside specified limits with NaN.

    Args:
        a (array-like): The input array to be truncated.
        limits (tuple or float, optional): The lower and upper limits for truncation.
            If a single float is provided, it is treated as both the lower and upper limit.
        inclusive (tuple of bool, optional): A tuple indicating whether to include the
            lower and upper limits in the truncation. Defaults to (True, True).
        inplace (bool, optional): If True, modify the input array in place.
            Otherwise, return a new array. Defaults to False.
        axis (int, optional): The axis along which to truncate the array.
            If None, the array will be flattened before truncation.
        nan_policy (str, optional): Policy for handling NaN values.
            Options are 'propagate' (default) or 'omit'.

    Returns:
        ndarray: The truncated array with values outside the specified limits replaced with NaN.
    """

    def _truncate1D(
        a, low_limit, up_limit, low_include, up_include, contains_nan, nan_policy
    ):
        """Truncate a 1D array based on specified limits and inclusion criteria.

        Args:
            a (ndarray): The 1D input array to be truncated.
            low_limit (float): The proportion of values to truncate from the lower end.
            up_limit (float): The proportion of values to truncate from the upper end.
            low_include (bool): Whether to include the lower limit in truncation.
            up_include (bool): Whether to include the upper limit in truncation.
            contains_nan (bool): Indicates if the array contains NaN values.
            nan_policy (str): Policy for handling NaN values.

        Returns:
            ndarray: The truncated 1D array with specified values replaced with NaN.
        """
        n = a.count()  # Count non-NaN values in the array
        idx = a.argsort()  # Get sorted indices of the array
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))  # Count the NaN values

        if low_limit:
            if low_include:
                lowidx = int(low_limit * n)  # Calculate index for lower limit
            else:
                lowidx = np.round(low_limit * n).astype(
                    int
                )  # Round for exclusive limit
            if contains_nan and nan_policy == "omit":
                lowidx = min(
                    lowidx, n - nan_count - 1
                )  # Adjust for NaNs if omitting them
            a[idx[:lowidx]] = np.nan  # Replace low values with np.nan

        if up_limit is not None:
            if up_include:
                upidx = n - int(n * up_limit)  # Calculate index for upper limit
            else:
                upidx = n - np.round(n * up_limit).astype(
                    int
                )  # Round for exclusive limit
            if contains_nan and nan_policy == "omit":
                a[idx[upidx:-nan_count]] = (
                    np.nan
                )  # Replace high values with np.nan if omitting NaNs
            else:
                a[idx[upidx:]] = np.nan  # Replace high values with np.nan

        return a

    a = a.astype(float)  # Ensure the input array is of type float
    contains_nan, nan_policy = _contains_nan(a, nan_policy)  # Check for NaN values
    a = ma.array(
        a, copy=np.logical_not(inplace)
    )  # Create a copy of the array if not in-place modification

    if limits is None:
        return a  # Return the array as is if no limits are specified

    if (not isinstance(limits, tuple)) and isinstance(limits, float):
        limits = (limits, limits)  # Convert single float to a tuple

    # Check the limits to ensure they are valid
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1.0 or lolim < 0:
            raise ValueError(errmsg % "beginning" + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1.0 or uplim < 0:
            raise ValueError(errmsg % "end" + "(got %s)" % uplim)

    (loinc, upinc) = inclusive  # Unpack inclusive flags

    if axis is None:
        shp = a.shape  # Store the shape of the array
        return _truncate1D(
            a.ravel(), lolim, uplim, loinc, upinc, contains_nan, nan_policy
        ).reshape(shp)  # Truncate and reshape the array back to its original shape
    else:
        return ma.apply_along_axis(
            _truncate1D, axis, a, lolim, uplim, loinc, upinc, contains_nan, nan_policy
        )  # Apply truncation along the specified axis


if __name__ == "__main__":
    data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # normal data
    data2 = np.array([1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10])  # data with nan
    data3 = np.array([1, 2, 3, 4, 5, 100, 200, 300, 400, 500])  # extreme values
    data4 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])  # data with all identical values

    print("Testing normal data:")
    result1 = truncate(data1, limits=(0.1, 0.1), inclusive=(False, False))
    print("Result:", result1)

    print("\nTesting data with nan:")
    result2 = truncate(data2, limits=(0.1, 0.2), inclusive=(False, False))
    print("Result:", result2)

    print("\nTesting extreme values:")
    result3 = truncate(data3, limits=(0.1, 0.2), inclusive=(False, False))
    print("Result:", result3)

    print("\nTesting data with all identical values:")
    result4 = truncate(data4, limits=(0.1, 0.1), inclusive=(True, True))
    print("Result:", result4)
