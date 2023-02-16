# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
from scipy.stats import circmean

from floris.utilities import wrap_360


def calc_wd_mean_radial(angles_array_deg, axis=0, nan_policy="omit"):
    """
    Compute the mean wind direction over a given axis. Assumes that the 
    input angles are specified in degrees, and returns the mean wind 
    direction in degrees. Wrapper for scipy.stats.circmean

    Inputs:
        angles_array_deg - numpy array or pandas dataframe of input 
            wind directions.
        axis - axis of array/dataframe to take average over
        nan_policy - option to pass to scipy.stats.circmean; defaults to
           'omit'. (Options: 'propagate', 'raise', 'omit')

    Outputs:
        mean_wd - numpy array of mean wind directions over the provided
            axis
    """

    return circmean(angles_array_deg, high=360., axis=axis, 
        nan_policy=nan_policy)


# def calc_wd_mean_radial_list(angles_array_list):
#     if isinstance(angles_array_list, (pd.DataFrame, pd.Series)):
#         array = np.array(angles_array_list)
#     elif isinstance(angles_array_list, list):
#         array = np.vstack(angles_array_list).T
#     else:
#         array = np.array(angles_array_list)

#     # Use unit vectors to calculate the mean
#     dir_x = np.cos(array * np.pi / 180.).sum(axis=1)
#     dir_y = np.sin(array * np.pi / 180.).sum(axis=1)

#     mean_dirs = np.arctan2(dir_y, dir_x)
#     mean_out = wrap_360(mean_dirs * 180. / np.pi)

#     return mean_out


def calculate_wd_statistics(angles_array_deg, axis=0,
                            calc_median_min_max_std=True):
    """Determine statistical properties of an array of wind directions.
    This includes the mean of the array, the median, the standard deviation,
    the minimum value and the maximum value.

    Args:
        angles_array_deg ([float/int]): Array of angles in degrees

    Returns:
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
    mean_wd_full = mean_wd.reshape(new_shape).repeat(
        angles_array_deg.shape[axis], axis=axis)

    # Copy angles_array_deg and wrap values around its mean value
    angles_wrp = angles_array_deg
    angles_wrp[angles_wrp > (mean_wd_full + 180.)] += -360.
    angles_wrp[angles_wrp < (mean_wd_full - 180.)] += 360.

    median_wd = wrap_360(np.nanmedian(angles_wrp, axis=axis))
    std_wd = np.nanstd(angles_wrp, axis=axis)
    min_wd = wrap_360(np.nanmin(angles_wrp, axis=axis))
    max_wd = wrap_360(np.nanmax(angles_wrp, axis=axis))

    return mean_wd, median_wd, std_wd, min_wd, max_wd
