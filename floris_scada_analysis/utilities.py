# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import datetime
import numpy as np
import scipy.interpolate as interp
from sklearn.metrics import pairwise_distances_argmin_min as pwdist
from floris.utilities import wrap_360


def printnow(text):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s: %s' % (now_time, text))


def estimate_dt(time_array):
    """Automatically estimate timestep in a time_array

    Args:
        time_array ([list]): List or dataframe with time entries

    Returns:
        dt ([datetime.timedelta]): Timestep in dt.timedelta format
    """

    if len(time_array) < 2:
        # Assume arbitrary value
        return datetime.timedelta(seconds=0)

    dt = np.median(np.diff(time_array))
    if not isinstance(dt, datetime.timedelta):
        dt = datetime.timedelta(seconds=dt.astype(float)/1e9)

    # Check if data is all ascending
    if dt <= datetime.timedelta(0):
        raise DataError('Please only insert time ascending data.')

    return dt


def get_num_turbines(df):
    # Let's assume that the format of variables is ws_%03d, wd_%03d, and so on
    num_turbines = len([c for c in df.columns if 'pow_' in c and len(c) == 7])

    if num_turbines == 0:
        # Try with wind speed
        num_turbines = len([c for c in df.columns if 'ws_' in c and len(c) == 6])

    if num_turbines == 0:
        # Try with wind direction
        num_turbines = len([c for c in df.columns if 'wd_' in c and len(c) == 6])

    return num_turbines


def interp_within_margin(x, xp, yp, x_margin, kind, wrap_around_360=False):
    # Make sure everything is sorted
    x = np.sort(x)
    id_sort = np.argsort(xp)
    xp = np.array(xp)[id_sort]
    yp = np.array(yp)[id_sort]

    # Use regular interpolation for all values first
    f_reg = interp.interp1d(xp, yp, kind=kind,
                            bounds_error=False,
                            assume_sorted=True)
    y_full = f_reg(x)

    if wrap_around_360 and not (kind == 'nearest'):
        # Shift by 180 deg and interpolate again
        f_180 = interp.interp1d(xp, wrap_360(yp+180.),
                                kind=kind, bounds_error=False,
                                assume_sorted=True)
        y_180 = wrap_360(f_180(x) - 180.)

        # Figure out where we jump by more than 180 deg
        Nx = len(x)
        dymax = np.full(Nx, np.nan)
        for i in range(Nx):
            lrgv_id = np.where((xp - x[i]) > 0)[0]
            lwrv_id = np.where((xp - x[i]) < 0)[0]
            if len(lrgv_id) > 0:
                dy_u = np.abs(yp[lrgv_id[0]] - y_full[i])
                dymax[i] = np.nanmax([dy_u, dymax[i]])
            if len(lwrv_id) > 0:
                dy_l = np.abs(yp[lwrv_id[0]] - y_full[i])
                dymax[i] = np.nanmax([dy_l, dymax[i]])

        # Replace those points with y_180 values
        y_full[dymax > 180.] = y_180[dymax > 180.]

    # Find any values that exist in both arrays: no need to check
    _, comm1, _ = np.intersect1d(x, xp, return_indices=True)
    ids_to_check = np.delete(np.array(range(len(x))), comm1)

    # Find entries outside of range: no need to check
    xp_min = np.min(xp) - x_margin
    xp_max = np.max(xp) + x_margin
    oob = (x[ids_to_check] < xp_min) | (x[ids_to_check] > xp_max)
    y_full[ids_to_check[oob]] = np.nan
    ids_to_check = ids_to_check[~oob]

    # Set any values outside of x_margin to np.nan
    _, dx = pwdist(
        x[ids_to_check].reshape(-1, 1),
        xp.reshape(-1, 1),
        axis=1,
        metric='euclidean'
        )
    dx = np.array([np.min(np.abs(xi-np.array(xp))) for xi in x])
    y_full[dx > x_margin] = np.nan

    return y_full
