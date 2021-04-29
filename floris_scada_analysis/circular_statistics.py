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
import pandas as pd

from floris_scada_analysis.floris_tools import get_turbs_in_radius


def wrap_360_deg(angles_in):
    if isinstance(angles_in, float) or isinstance(angles_in, int):
        angles_in = [angles_in]

    angles_in = np.array([float(i) for i in angles_in])
    while any(angles_in >= 360.):
        angles_in[angles_in >= 360.] += - 360.
    while any(angles_in < 0):
        angles_in[angles_in < 0.] += 360.
    return angles_in


def calc_wd_mean_radial(angles_array_deg):
    # Use unit vectors to calculate the mean
    wd_x = np.cos(angles_array_deg * np.pi / 180.)
    wd_y = np.sin(angles_array_deg * np.pi / 180.)

    mean_wd = np.arctan2(np.sum(wd_y), np.sum(wd_x))
    mean_wd = wrap_360_deg(mean_wd * 180. / np.pi)

    return mean_wd


def calc_wd_mean_radial_list(angles_array_list):
    if isinstance(angles_array_list, (pd.DataFrame, pd.Series)):
        array = np.array(angles_array_list)
    elif isinstance(angles_array_list, list):
        array = np.vstack(angles_array_list).T
    else:
        array = np.array(angles_array_list)

    # Use unit vectors to calculate the mean
    dir_x = np.cos(array * np.pi / 180.).sum(axis=1)
    dir_y = np.sin(array * np.pi / 180.).sum(axis=1)

    mean_dirs = np.arctan2(dir_y, dir_x)
    mean_out = wrap_360_deg(mean_dirs * 180. / np.pi)

    return mean_out



def calculate_wd_statistics(angles_array_deg):
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
    angles_array_deg = wrap_360_deg(angles_array_deg)
    angles_array_deg = angles_array_deg[~np.isnan(angles_array_deg)]

    # Check for unique cases
    if angles_array_deg.shape[0] <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if np.unique(angles_array_deg).shape[0] == 1:
        mean_wd = angles_array_deg[0]
        median_wd = angles_array_deg[0]
        std_wd = 0.0
        min_wd = angles_array_deg[0]
        max_wd = angles_array_deg[0]
        return mean_wd, median_wd, std_wd, min_wd, max_wd

    # Calculate the mean
    mean_wd = calc_wd_mean_radial(angles_array_deg)

    # Calculate the standard deviation
    var_wd_lin = np.abs(angles_array_deg-mean_wd)
    var_wd_wrapped = np.abs(360-angles_array_deg-mean_wd)
    var_wd = np.min([var_wd_lin, var_wd_wrapped], axis=0)
    std_wd = np.sqrt((1./len(angles_array_deg)) * np.sum(var_wd**2.0))

    # Calculate the min and max values
    if mean_wd == 180.:
        min_wd = np.min(angles_array_deg)
        max_wd = np.max(angles_array_deg)
    elif mean_wd < 180.:
        min_range = np.append(angles_array_deg[angles_array_deg < mean_wd], -360. + angles_array_deg[angles_array_deg > (mean_wd+180.)])
        min_wd = np.min(min_range)
        max_range = angles_array_deg[angles_array_deg < (mean_wd+180.)]
        max_wd = np.max(max_range)
    elif mean_wd > 180.:
        min_range = angles_array_deg[angles_array_deg > (mean_wd - 180.)]
        min_wd = np.min(min_range)
        max_range = np.append(angles_array_deg[angles_array_deg > mean_wd], 360. + angles_array_deg[angles_array_deg < mean_wd - 180.])
        max_wd = np.max(max_range)

    # Wrap to [0, 360] deg
    min_wd = wrap_360_deg(min_wd)
    max_wd = wrap_360_deg(max_wd)

    # Calculate median
    if min_wd <= mean_wd and max_wd >= mean_wd:
        median_wd = np.median(angles_array_deg)
    elif min_wd > mean_wd and max_wd > mean_wd:
        ids_wrapped = (angles_array_deg >= (min_wd - 1e-10))
        angles_array_sorted = angles_array_deg[ids_wrapped] - 360.
        angles_array_sorted = np.append(angles_array_sorted, angles_array_deg[~ids_wrapped])
        median_wd = np.median(angles_array_sorted)
    elif min_wd < mean_wd and max_wd < mean_wd:
        ids_wrapped = (angles_array_deg <= (max_wd + 1e-10))
        angles_array_sorted = angles_array_deg[ids_wrapped] + 360.
        angles_array_sorted = np.append(angles_array_sorted, angles_array_deg[~ids_wrapped])
        median_wd = np.median(angles_array_sorted)
    else:
        raise ValueError('Just a safety check. I think we accounted for all possible situations.')

    # Wrap median to [0, 360]
    median_wd = wrap_360_deg(median_wd)

    return mean_wd, median_wd, std_wd, min_wd, max_wd
