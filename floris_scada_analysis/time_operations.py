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
import pandas as pd
from pandas.core.base import DataError

from floris_scada_analysis.circular_statistics import calculate_wd_statistics
from floris_scada_analysis import optimization as fopt


def df_downsample(df, resample_cols_angular, target_dt=600.0, verbose=True):
    """Downsampling of a predefined dataframe with data to a higher
    timestep. This is useful for downsampling 1s to 60s, 600s or hourly data

    Args:
        df ([pd.DataFrame]): Source dataframe with either a column called
        'time' or with the index being time.
        resample_cols_angular ([list of strings]): Names of columns that
        require angular/circular averaging, dealing with 360 degrees wrapping.
        target_dt ([datetime.timedelta], optional): Desired sampling time,
        which should be at least twice the sampling time of the original
        dataframe. Input can also be a float or an integer, which will
        automatically be converted into a datetime.timedelta object where
        the value is assumed to be seconds. Defaults to 600 s.
        verbose (bool, optional): Print progress. Defaults to True.

    Raises:
        UserWarning: Will raise if target_dt is not at least twice the
        sampling time of the source dataframe.

    Returns:
        df_res ([pd.DataFrame]): Resampled dataframe
    """

    # Convert input into datetime.timedelta if necessary
    if isinstance(target_dt, float) or isinstance(target_dt, int):
        target_dt = datetime.timedelta(seconds=target_dt)

    # Estimate sampling time of source dataset
    time_array_src = df.reset_index().time
    dt_src = estimate_dt(time_array_src)

    if target_dt < 2 * dt_src:
        raise UserWarning(
            "Not recommended to downsample with < 2x the original dt."
        )

    # Set up new dataframes: values are based on past [target_dt] time
    time_array_target = np.arange(
        np.min(time_array_src) + target_dt,
        np.max(time_array_src) + target_dt,
        target_dt,
    )
    time_array_target = pd.to_datetime(time_array_target)

    print("Calculating mapping between raw and downsampled data")
    time_target_start_array = [t - target_dt for t in time_array_target]
    time_windows = [[time_target_start_array[i], time_array_target[i]] 
                    for i in range(len(time_target_start_array))]
    tids_array = find_window_in_time_array(time_array_src, time_windows)

    print("Downsampling data and calculating _mean, _median, _min, _max, and _std.")
    df_res = pd.DataFrame({"time": time_array_target})

    resample_cols_regular = [
        c for c in df.columns if c not in resample_cols_angular
        and "index" not in c and "time" not in c
    ]

    # Extract values and append row of np.nans
    ncols = len(resample_cols_regular)
    nws = len(tids_array)
    array_df_reg = df[resample_cols_regular].values
    array_df_reg = np.vstack([array_df_reg, np.array([np.nan] * ncols)])
    id_row_allnans = array_df_reg.shape[0] - 1

    # Update tids_array so that they all have same length
    max_tlen = np.max([len(tid) for tid in tids_array])
    tids_array_full = id_row_allnans * np.ones(max_tlen*nws, dtype='int')
    for ii, ti in enumerate(tids_array):
        tids_array_full[ii*max_tlen:ii*max_tlen+len(ti)] = ti

    # Calculate statistical properties of regular columns
    new_shape = (nws, max_tlen, ncols)
    array_df_reg_tws = array_df_reg[tids_array_full, :].reshape(new_shape)

    print('Downsampling non-angular columns: calculating mean, median, std, min, max...')
    df_res[[c + '_mean' for c in resample_cols_regular]] = (
        np.nanmean(array_df_reg_tws, axis=1)
    )
    df_res[[c + '_median' for c in resample_cols_regular]] = (
        np.nanmedian(array_df_reg_tws, axis=1)
    )
    df_res[[c + '_std' for c in resample_cols_regular]] = (
        np.nanstd(array_df_reg_tws, axis=1)
    )
    df_res[[c + '_min' for c in resample_cols_regular]] = (
        np.nanmin(array_df_reg_tws, axis=1)
    )
    df_res[[c + '_max' for c in resample_cols_regular]] = (
        np.nanmax(array_df_reg_tws, axis=1)
    )

    # Calculate statistical properties of angular columns
    ncols = len(resample_cols_angular)
    new_shape = (nws, max_tlen, ncols)
    array_df_ang = df[resample_cols_angular].values
    array_df_ang = np.vstack([array_df_ang, np.array([np.nan] * ncols)])
    array_df_ang_tws = array_df_ang[tids_array_full, :].reshape(new_shape)

    print('Downsampling angular columns: calculating mean, median, std, min, max...')
    means_ang, medians_ang, stds_ang, mins_ang, maxs_ang = (
        calculate_wd_statistics(array_df_ang_tws, axis=1))

    df_res[[c + '_mean' for c in resample_cols_angular]] = means_ang
    df_res[[c + '_median' for c in resample_cols_angular]] = medians_ang
    df_res[[c + '_std' for c in resample_cols_angular]] = stds_ang
    df_res[[c + '_min' for c in resample_cols_angular]] = mins_ang
    df_res[[c + '_max' for c in resample_cols_angular]] = maxs_ang

    return df_res


def df_resample_to_time_array(df, time_array, circular_cols,
                              interp_method, interp_margin=None):
    df_res = df.head(0).copy()  # Copy with properties but no actual data
    df_res['time'] = time_array

    t0 = time_array[0]
    df_t = np.array(df['time'] - t0, dtype=np.timedelta64)
    xp = df_t/np.timedelta64(1, 's')  # Convert to regular seconds

    # Normalize time variables
    time_array = np.array([t - t0 for t in time_array], dtype=np.timedelta64)
    x = time_array/np.timedelta64(1, 's')

    if interp_margin is None:
        dxx = 0.500001 * np.median(np.diff(x))
    else:
        dxx = np.timedelta64(interp_margin) / np.timedelta64(1, 's')

    cols_to_interp = [c for c in df_res.columns if c not in ['time']]
    for ii, c in enumerate(cols_to_interp):
        if isinstance(circular_cols, bool):
            wrap_around_360 = circular_cols
        elif isinstance(circular_cols[0], bool):
            wrap_around_360 = circular_cols[ii]
        elif isinstance(circular_cols[0], str):
            wrap_around_360 = (c in circular_cols)

        print('  Resampling column %s onto the specified time_array.' % c)
        y = fopt.interp_within_margin(
            x=x, xp=xp, yp=df[c], x_margin=dxx, kind=interp_method,
            wrap_around_360=wrap_around_360)
        df_res[c] = y

    return df_res


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


def find_window_in_time_array(time_array_src, seek_time_windows,
                              predict_jumps=False):
    """This function will look through time_array_src and return the indices
    of each of the entries of time_array_src that are within the region
    seek_time_windows[..][0] < time_array_src[..] <= seek_time_windows[..][1],
    for an array of times to find. This can be used to figure out which
    entries of time_array_src belong to a certain time period, such that
    it can be used for downsampling.

    Args:
        time_array_src ([list]): List of timesteps from the original dataset
        seek_time_windows ([list]): List of lists with each sublist being
        of length 2 and consisting of two timestamps. The two timestamps are
        respectively the minimum and maximum time of that particular time
        window.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    time_array_src = list(time_array_src)
    seek_times_min_remaining = [tw[0] for tw in seek_time_windows]
    seek_times_max_remaining = [tw[1] for tw in seek_time_windows]
    idxs_out_array = []

    # Check time_array_src timestep if predicting jumps
    if predict_jumps:
        dt_src = estimate_dt(time_array_src)

    # Adress behavior at lower end (0)
    i = 0
    idxs_out = []
    # Deal with situations in which entire range is out of array
    while time_array_src[0] > seek_times_max_remaining[0]:
        idxs_out_array.append([])
        seek_times_min_remaining.pop(0)
        seek_times_max_remaining.pop(0)

    # Deal with first entry, partially covered by seek_times[0]
    if time_array_src[0] > seek_times_min_remaining[0]:
        while time_array_src[i] <= seek_times_max_remaining[0]:
            idxs_out.append(i)
            i += 1
            if i == len(time_array_src):
                break
        idxs_out_array.append(idxs_out)
        seek_times_min_remaining.pop(0)
        seek_times_max_remaining.pop(0)

    if len(seek_times_min_remaining) < 1:
        return idxs_out_array

    # Adress behavior at higher end (-1)
    idxs_out_array_end = []
    # Deal with situations in which entire range is out of array
    while time_array_src[-1] < seek_times_min_remaining[-1]:
        idxs_out_array_end.insert(0, [])  # Prepend
        seek_times_min_remaining.pop(-1)
        seek_times_max_remaining.pop(-1)
        if len(seek_times_min_remaining) < 1:
            idxs_out_array = idxs_out_array.extend(idxs_out_array_end)
            return idxs_out_array

    # Deal with situations in which upper part of range is out of array
    while time_array_src[-1] < seek_times_max_remaining[-1]:
        i = len(time_array_src) - 1
        while time_array_src[i] > seek_times_min_remaining[-1]:
            i += -1
        idxs_out = list(range(i + 1, len(time_array_src)))
        idxs_out_array_end.insert(0, idxs_out)  # Prepend
        seek_times_min_remaining.pop(-1)
        seek_times_max_remaining.pop(-1)
        if len(seek_times_min_remaining) < 1:
            idxs_out_array = idxs_out_array.extend(idxs_out_array_end)
            return idxs_out_array

    if len(seek_times_min_remaining) < 1:
        idxs_out_array = idxs_out_array.extend(idxs_out_array_end)
        return idxs_out_array

    # Adress all other situations
    i = 0
    while len(seek_times_min_remaining) > 0:
        seek_tmin = seek_times_min_remaining[0]
        seek_tmax = seek_times_max_remaining[0]

        if predict_jumps:
            i += int((seek_tmin - time_array_src[i]) / dt_src)

        # If necessary, step back until we are at right point
        # iold = i
        while time_array_src[i-1] > seek_tmin:
            i += -1
            if i < 0:
                i = 0
                break
        # print('stepped back from %d to %d...' % (iold, i))

        # If necessary, step forward until we are at the right point
        # iold = i
        while time_array_src[i] <= seek_tmin:
            i += 1
            if i >= len(time_array_src):
                i = len(time_array_src) - 1
                break
        # print('stepped forward from %d to %d...' % (iold, i))

        # Now start appending entries until _max
        idxs_out = []
        while time_array_src[i] <= seek_tmax:
            idxs_out.append(i)
            i += 1
            if i >= len(time_array_src):
                i += -1
                break
        idxs_out_array.append(idxs_out)

        # Remove time window entry
        seek_times_min_remaining.pop(0)
        seek_times_max_remaining.pop(0)

    # Append idxs_out_array_end entries to idxs_out_array
    for ar in idxs_out_array_end:
        idxs_out_array.append(ar)

    return idxs_out_array
