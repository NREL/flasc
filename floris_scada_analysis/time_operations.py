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
from floris_scada_analysis.wd_statistics import calculate_wd_statistics


def df_downsample(df, resample_cols_angular, target_dt=600.0, verbose=True):
    """Downsampling of a predefined dataframe with data to a higher
    timestep. This is useful for downsampling 1s or 60s data to
    600s or even hourly data.

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

    print("Calculating mapping between 1s, 60s and 600s data timestamps")
    time_target_start_array = [t - target_dt for t in time_array_target]
    time_windows = [[time_target_start_array[i], time_array_target[i]] 
                    for i in range(len(time_target_start_array))]
    tids_array = find_window_in_time_array(time_array_src, time_windows)

    print("Creating df_60s and df_600s based on df_1s data")
    df_res = pd.DataFrame({"time": time_array_target})

    resample_cols_regular = [
        c for c in df.columns if c not in resample_cols_angular and "index" not in c
    ]

    for c in resample_cols_regular:
        if verbose:
            print(
                "  Downsampling column "
                + c
                + " to %.0f s using regular statistics."
                % (target_dt / datetime.timedelta(seconds=1))
            )
        df_res[c + "_mean"] = [
            np.nanmean(np.array(df[c])[i]) for i in tids_array
        ]
        df_res[c + "_median"] = [
            np.nanmedian(np.array(df[c])[i]) for i in tids_array
        ]
        df_res[c + "_std"] = [
            np.nanstd(np.array(df[c])[i]) for i in tids_array
        ]
        df_res[c + "_min"] = [
            np.min(np.array(df[c])[tids_array[i]], initial=df_res[c + "_mean"][i])
            for i in range(len(tids_array))
        ]
        df_res[c + "_max"] = [
            np.max(np.array(df[c])[tids_array[i]],initial=df_res[c + "_mean"][i])
            for i in range(len(tids_array))
        ]

    for c in resample_cols_angular:
        if verbose:
            print(
                "  Downsampling column "
                + c
                + " to %.0f s using angular statistics."
                % (target_dt / datetime.timedelta(seconds=1))
            )

        mean_ang_array = np.array([])
        median_ang_array = np.array([])
        std_ang_array = np.array([])
        min_ang_array = np.array([])
        max_ang_array = np.array([])

        for ti in range(len(tids_array)):
            time_ids = tids_array[ti]
            mean_c, median_c, std_c, min_c, max_c = calculate_wd_statistics(
                np.array(df[c][time_ids]), method="radial")

            mean_ang_array = np.append(mean_ang_array, mean_c)
            median_ang_array = np.append(median_ang_array, median_c)
            std_ang_array = np.append(std_ang_array, std_c)
            min_ang_array = np.append(min_ang_array, min_c)
            max_ang_array = np.append(max_ang_array, max_c)

        df_res[c + "_mean"] = mean_ang_array
        df_res[c + "_median"] = median_ang_array
        df_res[c + "_std"] = std_ang_array
        df_res[c + "_min"] = min_ang_array
        df_res[c + "_max"] = max_ang_array

    return df_res


def estimate_dt(time_array):
    """Automatically estimate timestep in a time_array

    Args:
        time_array ([list]): List or dataframe with time entries

    Returns:
        dt ([datetime.timedelta]): Timestep in dt.timedelta format
    """

    # Base estimation on first 5000 entries of the time_array
    dt_array = [
        (time_array[i + 1] - time_array[i]) / datetime.timedelta(seconds=1)
        for i in range(np.min([5000, len(time_array) - 1]))
    ]
    dt = datetime.timedelta(seconds=np.nanmin(dt_array))

    return dt


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

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    dt_src = estimate_dt(time_array_src)

    time_array_src = list(time_array_src)
    seek_times_min_remaining = [tw[0] for tw in seek_time_windows]
    seek_times_max_remaining = [tw[1] for tw in seek_time_windows]
    idxs_out_array = []

    # Adress behavior at lower end (0)
    i = 0
    idxs_out = []
    if time_array_src[0] > seek_times_min_remaining[0]:
        while time_array_src[i] <= seek_times_max_remaining[0]:
            idxs_out.append(i)
            i += 1
        idxs_out_array.append(idxs_out)
        seek_times_min_remaining.pop(0)
        seek_times_max_remaining.pop(0)

    # Adress behavior at higher end (-1)
    idxs_out_array_end = []
    # Deal with situations in which entire range is out of array
    while time_array_src[-1] < seek_times_min_remaining[-1]:
        idxs_out_array_end.insert(0, [])  # Prepend
        seek_times_min_remaining.pop(-1)
        seek_times_max_remaining.pop(-1)
    # Deal with situations in which upper part of range is out of array
    while time_array_src[-1] < seek_times_max_remaining[-1]:
        i = len(time_array_src) - 1
        while time_array_src[i] > seek_times_min_remaining[-1]:
            i += -1
        idxs_out = list(range(i + 1, len(time_array_src)))
        idxs_out_array_end.insert(0, idxs_out)  # Prepend
        seek_times_min_remaining.pop(-1)
        seek_times_max_remaining.pop(-1)

    # Adress all other situations
    i = 1
    di = 0
    while True:  # Continue until oow_time_min is empty
        if time_array_src[i] > seek_times_min_remaining[0]:
            # Step back in steps of 1 until we are exactly at right point
            while time_array_src[i] > seek_times_min_remaining[0]:
                i += -1
            i += 1

            # Marked as bad, now append entries until _max
            idxs_out = []
            while time_array_src[i] <= seek_times_max_remaining[0]:
                idxs_out.append(i)
                i += 1
            idxs_out_array.append(idxs_out)

            # Remove entry in oow_time_min/max
            seek_times_min_remaining.pop(0)
            seek_times_max_remaining.pop(0)
            if len(seek_times_min_remaining) <= 0:
                break

        # Make a guess and step forward exactly this much
        i += -1
        di = int((seek_times_min_remaining[0]-time_array_src[i]) / dt_src + 1)
        if di < 0:
            raise ValueError(
                "Data does not seem to be sorted." +
                "Please only use time-ascending data with this function."
            )

        i += di
        if i > len(time_array_src) - 1:
            i = len(time_array_src) - 1

    # Append idxs_out_array_end entries to idxs_out_array
    for ar in idxs_out_array_end:
        idxs_out_array.append(ar)

    return idxs_out_array
