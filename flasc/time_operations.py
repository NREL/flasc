# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from datetime import timedelta as td
from itertools import product

import numpy as np
import pandas as pd

from floris.utilities import wrap_360

from . import utilities as fsut


def df_movingaverage(
    df_in,
    cols_angular,
    window_width=td(seconds=60),
    min_periods=1,
    center=True,
    calc_median_min_max_std=False,
    return_index_mapping=False,
):

    # Copy and ensure dataframe is indexed by time
    df = df_in.copy()
    if "time" in df.columns:
        df = df.set_index("time")

    # Find non-angular columns
    if isinstance(cols_angular, bool):
        if cols_angular:
            cols_angular = [c for c in df.columns]
        else:
            cols_angular = []
    cols_regular = [c for c in df.columns if c not in cols_angular]

    # Now calculate cos and sin components for angular columns
    sin_cols = ["{:s}_sin".format(c) for c in cols_angular]
    cos_cols = ["{:s}_cos".format(c) for c in cols_angular]
    df[sin_cols] = np.sin(df[cols_angular] * np.pi / 180.0)
    df[cos_cols] = np.cos(df[cols_angular] * np.pi / 180.0)

    # Drop angular columns
    df = df.drop(columns=cols_angular)

    # Now calculate rolling (moving) average
    df_roll = df.rolling(
        window_width,
        center=center,
        axis=0,
        min_periods=min_periods
    )

    # First calculate mean values of non-angular columns
    df_ma = df_roll[cols_regular].mean().copy()

    # Now add mean values of angular columns
    df_ma[cols_angular] = wrap_360(
        np.arctan2(
            df_roll[sin_cols].mean().values,
            df_roll[cos_cols].mean().values
        ) * 180.0 / np.pi
    )

    # Figure out which indices/data points belong to each window
    if (return_index_mapping or calc_median_min_max_std):
        df_tmp = df_ma[[]].copy().reset_index(drop=False)
        df_tmp["tmp"] = 1
        df_tmp = df_tmp.rolling(window_width, center=center, axis=0, on="time")["tmp"]

        # Grab index of first and last time entry for each window
        windows_min = df_tmp.apply(lambda x: x.index[0]).astype(int)
        windows_max = df_tmp.apply(lambda x: x.index[-1]).astype(int)

        # Now create a large array that contains the array of indices, with
        # the values in each row corresponding to the indices upon which that
        # row's moving/rolling average is based. Note that we purposely create
        # a larger matrix than necessary, since some rows/windows rely on more
        # data (indices) than others. This is the case e.g., at the start of
        # the dataset, at the end, and when there are gaps in the data. We fill
        # the remaining matrix entries with "-1".
        dn = int(np.ceil(window_width/fsut.estimate_dt(df_in["time"]))) + 5
        data_indices = -1 * np.ones((df_ma.shape[0], dn), dtype=int)
        for ii in range(len(windows_min)):
            lb = windows_min[ii]
            ub = windows_max[ii]
            ind = np.arange(lb, ub + 1, dtype=int)
            data_indices[ii, ind - lb] = ind

    # Calculate median, min, max, std if necessary
    if calc_median_min_max_std:
        # Append all current columns with "_mean"
        df_ma.columns = ["{:s}_mean".format(c) for c in df_ma.columns]

        # Add statistics for regular columns
        funs = ["median", "min", "max", "std"]
        cols_reg_stats = ["_".join(i) for i in product(cols_regular, funs)]
        df_ma[cols_reg_stats] = df_roll[cols_regular].agg(funs).copy()

        # Add statistics for angular columns
        # Firstly, create matrix with indices for the mean values
        data_indices_mean = np.tile(np.arange(0, df_ma.shape[0]), (dn, 1)).T

        # Grab raw and mean data and format as numpy arrays
        D = df_in[cols_angular].values
        M = df_ma[["{:s}_mean".format(c) for c in cols_angular]].values

        # Add NaN row as last row. This corresponds to the -1 indices
        # that we use as placeholders. This way, those indices do not
        # count towards the final statistics (median, min, max, std).
        D = np.vstack([D, np.nan * np.ones(D.shape[1])])
        M = np.vstack([M, np.nan * np.ones(M.shape[1])])

        # Now create a 3D matrix containing all values. The three dimensions
        # come from:
        #  > [0] one dimension containing the rolling windows,
        #  > [1] one with the raw data underlying each rolling window,
        #  > [2] one for each angular column within the dataset
        values = D[data_indices, :]
        values_mean = M[data_indices_mean, :]

        # Center values around values_mean
        values[values > (values_mean + 180.0)] += -360.0
        values[values < (values_mean - 180.0)] += 360.0

        # Calculate statistical properties and wrap to [0, 360)
        values_median = wrap_360(np.nanmedian(values, axis=1))
        values_min = wrap_360(np.nanmin(values, axis=1))
        values_max = wrap_360(np.nanmax(values, axis=1))
        values_std = wrap_360(np.nanstd(values, axis=1))

        # Save to dataframe
        df_ma[["{:s}_median".format(c) for c in cols_angular]] = values_median
        df_ma[["{:s}_min".format(c) for c in cols_angular]] = values_min
        df_ma[["{:s}_max".format(c) for c in cols_angular]] = values_max
        df_ma[["{:s}_std".format(c) for c in cols_angular]] = values_std

    if return_index_mapping:
        return df_ma, data_indices

    return df_ma


def df_downsample(
    df_in,
    cols_angular,
    window_width=td(seconds=60),
    min_periods=1,
    center=False,
    calc_median_min_max_std=False,
    return_index_mapping=False,
):

    # Copy and ensure dataframe is indexed by time
    df = df_in.copy()
    if "time" in df.columns:
        df = df.set_index("time")

    # Find non-angular columns
    cols_regular = [c for c in df.columns if c not in cols_angular]

    # Now calculate cos and sin components for angular columns
    sin_cols = ["{:s}_sin".format(c) for c in cols_angular]
    cos_cols = ["{:s}_cos".format(c) for c in cols_angular]
    df[sin_cols] = np.sin(df[cols_angular] * np.pi / 180.0)
    df[cos_cols] = np.cos(df[cols_angular] * np.pi / 180.0)

    # Drop angular columns
    df = df.drop(columns=cols_angular)

    # Add _N for each variable to keep track of n.o. data points
    cols_all = df.columns
    cols_N = ["{:s}_N".format(c) for c in cols_all]
    df[cols_N] = 1 - df[cols_all].isna().astype(int)

    # Now calculate downsampled dataframe, automatically
    # mark by label on the right (i.e., "past 10 minutes").
    df_resample = df.resample(window_width, label="right", axis=0)

    # First calculate mean values of non-angular columns
    df_out = df_resample[cols_regular].mean().copy()

    # Now add mean values of angular columns
    df_out[cols_angular] = wrap_360(
        np.arctan2(
            df_resample[sin_cols].mean().values,
            df_resample[cos_cols].mean().values
        ) * 180.0 / np.pi
    )

    # Check if we have enough samples for every measurement
    if min_periods > 1:
        N_counts = df_resample[cols_N].sum()
        df_out[N_counts < min_periods] = None  # Remove data relying on too few samples

    # Figure out which indices/data points belong to each window
    if (return_index_mapping or calc_median_min_max_std):
        df_tmp = df[[]].copy().reset_index(drop=False)
        df_tmp["tmp"] = 1
        df_tmp = df_tmp.resample(window_width, on="time", label="right", axis=0)["tmp"]

        # Grab index of first and last time entry for each window
        def get_first_index(x):
            if len(x) <= 0:
                return -1
            else:
                return x.index[0]
        def get_last_index(x):
            if len(x) <= 0:
                return -1
            else:
                return x.index[-1]

        windows_min = df_tmp.apply(get_first_index).astype(int)
        windows_max = df_tmp.apply(get_last_index).astype(int)

        # Now create a large array that contains the array of indices, with
        # the values in each row corresponding to the indices upon which that
        # row's moving/rolling average is based. Note that we purposely create
        # a larger matrix than necessary, since some rows/windows rely on more
        # data (indices) than others. This is the case e.g., at the start of
        # the dataset, at the end, and when there are gaps in the data. We fill
        # the remaining matrix entries with "-1".
        dn = int(np.ceil(window_width/fsut.estimate_dt(df_in["time"]))) + 5
        data_indices = -1 * np.ones((df_out.shape[0], dn), dtype=int)
        for ii in range(len(windows_min)):
            lb = windows_min[ii]
            ub = windows_max[ii]
            if not ((lb == -1) | (ub == -1)):
                ind = np.arange(lb, ub + 1, dtype=int)
                data_indices[ii, ind - lb] = ind

    # Calculate median, min, max, std if necessary
    if calc_median_min_max_std:
        # Append all current columns with "_mean"
        df_out.columns = ["{:s}_mean".format(c) for c in df_out.columns]

        # Add statistics for regular columns
        funs = ["median", "min", "max", "std"]
        cols_reg_stats = ["_".join(i) for i in product(cols_regular, funs)]
        df_out[cols_reg_stats] = df_resample[cols_regular].agg(funs).copy()

        # Add statistics for angular columns
        # Firstly, create matrix with indices for the mean values
        data_indices_mean = np.tile(np.arange(0, df_out.shape[0]), (dn, 1)).T

        # Grab raw and mean data and format as numpy arrays
        D = df_in[cols_angular].values
        M = df_out[["{:s}_mean".format(c) for c in cols_angular]].values

        # Add NaN row as last row. This corresponds to the -1 indices
        # that we use as placeholders. This way, those indices do not
        # count towards the final statistics (median, min, max, std).
        D = np.vstack([D, np.nan * np.ones(D.shape[1])])
        M = np.vstack([M, np.nan * np.ones(M.shape[1])])

        # Now create a 3D matrix containing all values. The three dimensions
        # come from:
        #  > [0] one dimension containing the rolling windows,
        #  > [1] one with the raw data underlying each rolling window,
        #  > [2] one for each angular column within the dataset
        values = D[data_indices, :]
        values_mean = M[data_indices_mean, :]

        # Center values around values_mean
        values[values > (values_mean + 180.0)] += -360.0
        values[values < (values_mean - 180.0)] += 360.0

        # Calculate statistical properties and wrap to [0, 360)
        values_median = wrap_360(np.nanmedian(values, axis=1))
        values_min = wrap_360(np.nanmin(values, axis=1))
        values_max = wrap_360(np.nanmax(values, axis=1))
        values_std = wrap_360(np.nanstd(values, axis=1))

        # Save to dataframe
        df_out[["{:s}_median".format(c) for c in cols_angular]] = values_median
        df_out[["{:s}_min".format(c) for c in cols_angular]] = values_min
        df_out[["{:s}_max".format(c) for c in cols_angular]] = values_max
        df_out[["{:s}_std".format(c) for c in cols_angular]] = values_std

    if center:
        # Shift time column towards center of the bin
        df_out.index = df_out.index - window_width / 2.0

    if return_index_mapping:
        return df_out, data_indices

    return df_out


def df_resample_by_interpolation(
    df,
    time_array,
    circular_cols,
    interp_method='linear',
    max_gap=None,
    verbose=True
):
    # Copy with properties but no actual data
    df_res = df.head(0).copy()

    # Remove timezones, if any
    df = df.copy()
    time_array = [pd.to_datetime(t).tz_localize(None) for t in time_array]
    time_array = np.array(time_array, dtype='datetime64')
    df["time"] = df["time"].dt.tz_localize(None)

    # Fill with np.nan values and the correct time array (without tz)
    df_res['time'] = time_array

    t0 = time_array[0]
    df_t = np.array(df['time'] - t0, dtype=np.timedelta64)
    xp = df_t/np.timedelta64(1, 's')  # Convert to regular seconds
    xp = np.array(xp, dtype=float)

    # Normalize time variables
    time_array = np.array([t - t0 for t in time_array], dtype=np.timedelta64)
    x = time_array/np.timedelta64(1, 's')

    if max_gap is None:
        max_gap = 1.5 * np.median(np.diff(x))
    else:
        max_gap = np.timedelta64(max_gap) / np.timedelta64(1, 's')

    cols_to_interp = [c for c in df_res.columns if c not in ['time']]

    # NN interpolation: just find indices and map accordingly for all cols
    for ii, c in enumerate(cols_to_interp):
        if isinstance(circular_cols, bool):
            wrap_around_360 = circular_cols
        elif isinstance(circular_cols[0], bool):
            wrap_around_360 = circular_cols[ii]
        elif isinstance(circular_cols[0], str):
            wrap_around_360 = (c in circular_cols)

        dt_raw_median = (df['time'].diff().median() / td(seconds=1))
        if verbose:
            print(
                "  Resampling column '{:s}' with median timestep {:.3f} s onto a prespecified time ".format(c, dt_raw_median) +
                "array with kind={}, max_gap={}".format(interp_method, max_gap) +
                "s, and wrap_around_360={}".format(wrap_around_360)
            )

        fp = np.array(df[c], dtype=float)
        ids = (~np.isnan(xp)) & ~(np.isnan(fp))

        y = fsut.interp_with_max_gap(
            x=x,
            xp=xp[ids],
            fp=fp[ids],
            max_gap=max_gap,
            kind=interp_method,
            wrap_around_360=wrap_around_360
        )
        df_res[c] = y

    return df_res
