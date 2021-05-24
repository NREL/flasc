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
# from pandas.core.base import DataError
from scipy import interpolate as interp

from floris_scada_analysis.circular_statistics import calculate_wd_statistics
from floris_scada_analysis import utilities as fsut


def df_downsample(df, resample_cols_angular, time_array_target,
                  calc_median_min_max_std=True, verbose=True):
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

    # Estimate sampling time of source dataset
    time_array_src = df.reset_index().time
    dt_src = fsut.estimate_dt(time_array_src)

    target_dts = np.unique(np.diff(time_array_target))
    if len(target_dts) > 1:
        raise UserWarning(
            "Varying timestep found in time_array_target. Exiting..."
        )
    target_dt = target_dts[0]
    if target_dt < 2 * dt_src:
        raise UserWarning(
            "Not recommended to downsample with < 2x the original dt."
        )

    # Set up new dataframes: values are based on past [target_dt] time
    if time_array_target is None:
        start_time = np.min(time_array_src) + target_dt
        end_time = np.max(time_array_src) + 0.99 * target_dt
        time_array_target = np.arange(start_time, end_time, target_dt)

    time_array_target = pd.to_datetime(time_array_target)

    if verbose:
        print("Calculating mapping between raw and downsampled data")
    time_target_start_array = [t - target_dt for t in time_array_target]
    time_windows = [[time_target_start_array[i], time_array_target[i]] 
                    for i in range(len(time_target_start_array))]
    tids_array = find_window_in_time_array(time_array_src, time_windows)

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

    if calc_median_min_max_std:
        if verbose:
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
    else:
        if verbose:
            print('Downsampling non-angular columns: calculating mean...')
        df_res[[c for c in resample_cols_regular]] = (
            np.nanmean(array_df_reg_tws, axis=1)
        )

    # Calculate statistical properties of angular columns
    ncols = len(resample_cols_angular)
    new_shape = (nws, max_tlen, ncols)
    array_df_ang = df[resample_cols_angular].values
    array_df_ang = np.vstack([array_df_ang, np.array([np.nan] * ncols)])
    array_df_ang_tws = array_df_ang[tids_array_full, :].reshape(new_shape)

    if calc_median_min_max_std:
        if verbose:
            print('Downsampling angular columns: calculating mean, median, std, min, max...')
        means_ang, medians_ang, stds_ang, mins_ang, maxs_ang = (
            calculate_wd_statistics(array_df_ang_tws,
                                    axis=1,
                                    calc_median_min_max_std=True)
            )
    else:
        if verbose:
            print('Downsampling angular columns: calculating mean...')
        means_ang = (
            calculate_wd_statistics(array_df_ang_tws,
                                    axis=1,
                                    calc_median_min_max_std=False)
            )

    if calc_median_min_max_std:
        df_res[[c + '_mean' for c in resample_cols_angular]] = means_ang
        df_res[[c + '_median' for c in resample_cols_angular]] = medians_ang
        df_res[[c + '_std' for c in resample_cols_angular]] = stds_ang
        df_res[[c + '_min' for c in resample_cols_angular]] = mins_ang
        df_res[[c + '_max' for c in resample_cols_angular]] = maxs_ang
    else:
        df_res[[c for c in resample_cols_angular]] = means_ang

    return df_res


def df_resample_by_filling_gaps(df, time_array):
    df_res = df.head(0).copy()  # Copy with properties but no actual data
    df_res['time'] = time_array

    if df.shape[0] < 1:
        return df_res

    # Find where datasets overlap
    xt_src = np.array(df['time'], dtype="datetime64[ns]")
    t0 = xt_src[0]
    xt_src = np.array((xt_src - t0) / np.timedelta64(1, 's'), dtype=int)
    xt_trg = np.array(time_array, dtype="datetime64[ns]")
    xt_trg = np.array((xt_trg - t0) / np.timedelta64(1, 's'), dtype=int)
    _, comm1, comm2 = np.intersect1d(xt_src, xt_trg, return_indices=True)

    # Update dataframe with values wherever overlap
    cols = [c for c in df.columns if not 'time' == c]
    df_res.loc[comm2, cols] = df.loc[comm1, cols].values

    return df_res

def df_resample_by_interpolation(df, time_array, circular_cols,
                              interp_method='nearest', interp_margin=None):
    _resample_fill_gaps(df, time_array)
    # Copy with properties but no actual data
    df_res = df.head(0).copy()

    # Fill with np.nan values and the correct time array
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

    # NN interpolation: just find indices and map accordingly for all cols
    if interp_method == 'nearest':
        y_indices = np.array(range(len(x)))
        y_mapping = fsut.interp_within_margin(
            x=x, xp=xp, yp=y_indices, x_margin=dxx, kind='nearest')
        for ii, c in enumerate(cols_to_interp):
            df_res[c] = np.array(df[c])[y_mapping]

    else:
        for ii, c in enumerate(cols_to_interp):
            if isinstance(circular_cols, bool):
                wrap_around_360 = circular_cols
            elif isinstance(circular_cols[0], bool):
                wrap_around_360 = circular_cols[ii]
            elif isinstance(circular_cols[0], str):
                wrap_around_360 = (c in circular_cols)

            print('  Resampling column %s onto the specified time_array.' % c)
            y = fsut.interp_within_margin(
                x=x, xp=xp, yp=df[c], x_margin=dxx, kind=interp_method,
                wrap_around_360=wrap_around_360)
            df_res[c] = y

    return df_res


def find_window_in_time_array(time_array_src, seek_time_windows):
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

    Returns:
        idxs_out_array ([list]): List of range of integers indicating the
        indices of time_array_src that correspond to the respective time
        window defined in seek_time_windows.
    """

    # Get number of windows
    Nw = len(seek_time_windows)

    # Convert source time array to datetime64
    time_array_src = np.array(time_array_src, dtype="datetime64[ns]")

    # Convert time_array_src to integers and set up interpolant
    xq = time_array_src.astype(int)
    yq = np.arange(0, len(xq), 1, dtype='int')

    # Add limit cases and setup interpolant
    xq_ext = np.hstack([0, xq, np.iinfo(int).max])
    yq_ext = np.hstack([yq[0], yq, yq[-1]])
    f = interp.interp1d(xq_ext, yq_ext, kind='nearest')

    # Format the windows appropriately
    windows_x = np.array(seek_time_windows, dtype="datetime64[ns]")
    windows_x = windows_x.astype(int)

    # Interpolate for all windows
    ids_interp = np.array(f(windows_x), dtype='int')

    # Step inside the time window, if currently outside
    ids_min = ids_interp[:, 0]
    ids_max = ids_interp[:, 1]
    ids_min[xq[ids_min] <= windows_x[:, 0]] += 1
    ids_max[xq[ids_max] > windows_x[:, 1]] += -1

    # Convert to a list of ranges
    idxs_out_array = [list(range(ids_min[ii], ids_max[ii] + 1))
                      for ii in range(Nw)]

    return idxs_out_array
