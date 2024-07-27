"""Circular statistics utility functions."""

import numpy as np
from floris.utilities import wrap_360
from scipy.stats import circmean


def calc_wd_mean_radial(angles_array_deg, axis=0, nan_policy="omit"):
    """Compute the mean wind direction over a given axis.

    Assumes that the
    input angles are specified in degrees, and returns the mean wind
    direction in degrees. Wrapper for scipy.stats.circmean

    Args:
        angles_array_deg (numpy array): Array of angles in degrees
        axis (int): Axis along which to calculate the mean
            Default is 0
        nan_policy (str): How to handle NaN values.  Default is 'omit'

    Returns:
         np.array: Mean wind direction in degrees
    """
    return circmean(angles_array_deg, high=360.0, axis=axis, nan_policy=nan_policy)


def calculate_wd_statistics(angles_array_deg, axis=0, calc_median_min_max_std=True):
    """Determine statistical properties of an array of wind directions.

    This includes the mean of the array, the median, the standard deviation,
    the minimum value and the maximum value.

    Args:
        angles_array_deg (numpy array): Array of wind directions in degrees
        axis (int): Axis along which to calculate the statistics
            Default is 0
        calc_median_min_max_std (bool): Whether to calculate the median, minimum,
            maximum, and standard deviation of the wind directions
            Default is True

    Returns:
        A tuple containing the following values:
            mean_wd (float): Mean wind direction in [0, 360] deg
            median_wd (float): Median wind direction in [0, 360] deg
            std_wd (float): Standard deviation in deg
            min_wd (float): Minimum wind direction in [0, 360] deg
            max_wd (float): Maximum wind direction in [0, 360] deg
    """
    # Preprocessing
    angles_array_deg = np.array(angles_array_deg, dtype=float)
    angles_array_deg = wrap_360(angles_array_deg)

    # Check for unique cases
    if angles_array_deg.shape[0] <= 0:
        if calc_median_min_max_std:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan

    if np.unique(angles_array_deg).shape[0] == 1:
        mean_wd = angles_array_deg[0]
        if not calc_median_min_max_std:
            return mean_wd

        median_wd = angles_array_deg[0]
        std_wd = 0.0
        min_wd = angles_array_deg[0]
        max_wd = angles_array_deg[0]
        return mean_wd, median_wd, std_wd, min_wd, max_wd

    # Calculate the mean
    mean_wd = calc_wd_mean_radial(angles_array_deg, axis=axis)

    # Return if we dont need to calculate statistical properties
    if not calc_median_min_max_std:
        return mean_wd

    # Upsample mean_wd for next calculations
    new_shape = list(mean_wd.shape)
    new_shape.insert(axis, 1)  # Add dimension at axis
    new_shape = tuple(new_shape)
    mean_wd_full = mean_wd.reshape(new_shape).repeat(angles_array_deg.shape[axis], axis=axis)

    # Copy angles_array_deg and wrap values around its mean value
    angles_wrp = angles_array_deg
    angles_wrp[angles_wrp > (mean_wd_full + 180.0)] += -360.0
    angles_wrp[angles_wrp < (mean_wd_full - 180.0)] += 360.0

    median_wd = wrap_360(np.nanmedian(angles_wrp, axis=axis))
    std_wd = np.nanstd(angles_wrp, axis=axis)
    min_wd = wrap_360(np.nanmin(angles_wrp, axis=axis))
    max_wd = wrap_360(np.nanmax(angles_wrp, axis=axis))

    return mean_wd, median_wd, std_wd, min_wd, max_wd
