# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


# import datetime
import datetime
import numpy as np
import os as os
import pandas as pd
import warnings

from floris.utilities import wrap_360

from ..dataframe_operations import df_reader_writer as fsio
from .. import (
    time_operations as fsato,
    floris_tools as ftools,
    utilities as fsut
)


# Functions related to wind farm analysis for df
def filter_df_by_ws(df, ws_range):
    df = df[df['ws'] >= ws_range[0]]
    df = df[df['ws'] < ws_range[1]]
    return df


def filter_df_by_wd(df, wd_range):
    lb = wd_range[0]
    ub = wd_range[1]

    lb = wrap_360(lb)
    if ub > 360.0:
        ub = wrap_360(ub)

    wd_array = wrap_360(df["wd"])
    if lb > ub:
        df = df[((wd_array >= lb) | (wd_array < ub))]
    else:
        df = df[((wd_array >= lb) & (df['wd'] < ub))]
    return df


def filter_df_by_ti(df, ti_range):
    df = df[df['ti'] >= ti_range[0]]
    df = df[df['ti'] < ti_range[1]]
    return df


def get_num_turbines(df):
    return fsut.get_num_turbines(df)


# Generic functions for column operations
def get_column_mean(df, col_prefix='pow', turbine_list=None,
                    circular_mean=False):
    if turbine_list is None:
        turbine_list = range(get_num_turbines(df))  # Assume all turbines
    elif isinstance(turbine_list, (int, np.integer)):
        turbine_list = [turbine_list]

    col_names = [col_prefix + '_%03d' % ti for ti in turbine_list]
    array = df[col_names].astype(float)

    if circular_mean:
        # Use unit vectors to calculate the mean
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dir_x = np.nanmean(np.cos(array * np.pi / 180.), axis=1)
        dir_y = np.nanmean(np.sin(array * np.pi / 180.), axis=1)

        mean_dirs = np.arctan2(dir_y, dir_x)
        mean_out = wrap_360(mean_dirs * 180. / np.pi)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_out = np.nanmean(array, axis=1)

    return mean_out


def _set_col_by_turbines(col_out, col_prefix, df,
                         turbine_numbers, circular_mean):
    if isinstance(turbine_numbers, str):
        if turbine_numbers.lower() == 'all':
            turbine_numbers = range(get_num_turbines(df=df))

    df[col_out] = get_column_mean(
        df=df,
        col_prefix=col_prefix,
        turbine_list=turbine_numbers,
        circular_mean=circular_mean
        )
    return df


def _set_col_by_n_closest_upstream_turbines(col_out, col_prefix, df, N,
    df_upstream, circular_mean, turb_no, x_turbs, y_turbs, exclude_turbs=[]):
    # Can get df_upstream using floris_tools.get_upstream_turbs_floris()
    df.loc[df.index, col_out] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, 'wd_min']
        wd_max = df_upstream.loc[i, 'wd_max']
        upstr_turbs = df_upstream.loc[i, 'turbines']

        # Calculate distances and get closest N upstream turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]
        x0 = x_turbs[turb_no]
        y0 = y_turbs[turb_no]
        x_upstr = np.array(x_turbs, dtype=float)[upstr_turbs]
        y_upstr = np.array(y_turbs, dtype=float)[upstr_turbs]
        ds = np.sqrt((x_upstr - x0) ** 2.0 + (y_upstr - y0) ** 2.0)
        upstr_turbs_sorted = np.array(upstr_turbs, dtype=int)[np.argsort(ds)]
        upstr_turbs_n_closest = upstr_turbs_sorted[0:N]

        if wd_min > wd_max:  # Wrap around
            ids = (df['wd'] > wd_min) | (df['wd'] <= wd_max)
        else:
            ids = (df['wd'] >= wd_min) & (df['wd'] < wd_max)

        col_mean = get_column_mean(
            df.loc[ids, :],
            col_prefix=col_prefix,
            turbine_list=upstr_turbs_n_closest,
            circular_mean=circular_mean
        )
        df.loc[ids, col_out] = col_mean

    return df


def _set_col_by_upstream_turbines(col_out, col_prefix, df,
                                  df_upstream, circular_mean,
                                  exclude_turbs=[]):
    # Can get df_upstream using floris_tools.get_upstream_turbs_floris()
    df.loc[df.index, col_out] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, 'wd_min']
        wd_max = df_upstream.loc[i, 'wd_max']
        upstr_turbs = df_upstream.loc[i, 'turbines']

        # Exclude particular turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]

        if wd_min > wd_max:  # Wrap around
            ids = (df['wd'] > wd_min) | (df['wd'] <= wd_max)
        else:
            ids = (df['wd'] >= wd_min) & (df['wd'] < wd_max)

        col_mean = get_column_mean(df.loc[ids, :],
                                   col_prefix=col_prefix,
                                   turbine_list=upstr_turbs,
                                   circular_mean=circular_mean)
        df.loc[ids, col_out] = col_mean

    return df


def _set_col_by_radius_from_turbine(col_out, col_prefix, df, turb_no,
                                    x_turbs, y_turbs, max_radius,
                                    circular_mean, include_itself=True):

    turbs_within_radius = ftools.get_turbs_in_radius(
        x_turbs=x_turbs, y_turbs=y_turbs, turb_no=turb_no,
        max_radius=max_radius, include_itself=include_itself)

    if len(turbs_within_radius) < 1:
        print('No turbines within proximity. Try to increase radius.')
        return None

    return _set_col_by_turbines(col_out=col_out, col_prefix=col_prefix, df=df,
                                turbine_numbers=turbs_within_radius,
                                circular_mean=circular_mean)


def _set_col_by_upstream_turbines_in_radius(
    col_out, col_prefix, df, df_upstream, turb_no,
    x_turbs, y_turbs, max_radius, circular_mean,
    include_itself=True):
    """Add a column called [col_out] to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for 
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called [col_ref].
    """

    turbs_in_radius = ftools.get_turbs_in_radius(
        x_turbs=x_turbs, y_turbs=y_turbs, turb_no=turb_no,
        max_radius=max_radius, include_itself=include_itself)

    if len(turbs_in_radius) < 1:
        print('No turbines within proximity. Try to increase radius.')
        return None

    turbs = range(len(x_turbs))
    turbs_outside_radius = [i for i in turbs if i not in turbs_in_radius]
    return _set_col_by_upstream_turbines(
        col_out=col_out,
        col_prefix=col_prefix,
        df=df,
        df_upstream=df_upstream,
        circular_mean=circular_mean,
        exclude_turbs=turbs_outside_radius)


# Helper functions
def set_wd_by_turbines(df, turbine_numbers):
    """Add a column called 'wd' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    the turbines in turbine_numbers.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'wd'.
    """
    return _set_col_by_turbines('wd', 'wd', df, turbine_numbers, True)


def set_wd_by_all_turbines(df):
    """Add a column called 'wd' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    turbines.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'wd'.
    """
    return _set_col_by_turbines('wd', 'wd', df, 'all', True)


def set_wd_by_radius_from_turbine(df, turb_no, x_turbs, y_turbs,
                                  max_radius, include_itself=True):
    return _set_col_by_radius_from_turbine(
        col_out='wd',
        col_prefix='wd',
        df=df,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=True,
        include_itself=include_itself)


def set_ws_by_turbines(df, turbine_numbers):
    """Add a column called 'ws' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    the turbines in turbine_numbers.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_turbines('ws', 'ws', df, turbine_numbers, False)


def set_ws_by_all_turbines(df):
    """Add a column called 'ws' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    turbines.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_turbines('ws', 'ws', df, 'all', False)


def set_ws_by_upstream_turbines(df, df_upstream, exclude_turbs=[]):
    """Add a column called 'ws' in your dataframe with value equal
    to the averaged wind speed measurements of all the turbines
    upstream, excluding the turbines listed in exclude_turbs.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        exclude_turbs ([list, array]): array-like variable containing
        turbine indices that should be excluded in determining the column
        mean quantity.
    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_upstream_turbines(
        col_out='ws',
        col_prefix='ws',
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs)


def set_ws_by_upstream_turbines_in_radius(df, df_upstream, turb_no,
                                          x_turbs, y_turbs,
                                          max_radius,
                                          include_itself=True):
    """Add a column called 'ws' to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out='ws',
        col_prefix='ws',
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself)


def set_ws_by_n_closest_upstream_turbines(df, df_upstream, turb_no,
    x_turbs, y_turbs, exclude_turbs=[], N=5):
    """Add a column called 'pow_ref' to your dataframe, which is the
    mean of the columns pow_%03d for the 5 closest turbines that are
    upstream of the turbine of interest [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for 
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

     Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'pow_ref'.
    """
    return _set_col_by_n_closest_upstream_turbines(
        col_out='ws',
        col_prefix='ws',
        N=N,
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        exclude_turbs=exclude_turbs,
        circular_mean=False)


def set_ti_by_turbines(df, turbine_numbers):
    """Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the
    turbines listed in turbine_numbers.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_turbines('ti', 'ti', df, turbine_numbers, False)


def set_ti_by_all_turbines(df):
    """Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all
    turbines.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_turbines('ti', 'ti', df, 'all', False)


def set_ti_by_upstream_turbines(df, df_upstream, exclude_turbs=[]):
    """Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the turbines
    upstream, excluding the turbines listed in exclude_turbs.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        exclude_turbs ([list, array]): array-like variable containing
        turbine indices that should be excluded in determining the column
        mean quantity.
    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_upstream_turbines(
        col_out='ti',
        col_prefix='ti',
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs)


def set_ti_by_upstream_turbines_in_radius(df, df_upstream, turb_no,
                                          x_turbs, y_turbs,
                                          max_radius,
                                          include_itself=True):
    """Add a column called 'ti' to your dataframe, which is the
    mean of the columns ti_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for 
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out='ti',
        col_prefix='ti',
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself)


def set_pow_ref_by_turbines(df, turbine_numbers):
    """Add a column called 'pow_ref' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the
    turbines listed in turbine_numbers.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_turbines('pow_ref', 'pow', df, turbine_numbers, False)


def set_pow_ref_by_upstream_turbines(df, df_upstream, exclude_turbs=[]):
    """Add a column called 'pow_ref' in your dataframe with value equal
    to the averaged power measurements of all the turbines upstream,
    excluding the turbines listed in exclude_turbs.

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        exclude_turbs ([list, array]): array-like variable containing
        turbine indices that should be excluded in determining the column
        mean quantity.
    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'pow_ref'.
    """
    return _set_col_by_upstream_turbines(
        col_out='pow_ref',
        col_prefix='pow',
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs)


def set_pow_ref_by_upstream_turbines_in_radius(
    df, df_upstream, turb_no, x_turbs,
    y_turbs, max_radius, include_itself=False):
    """Add a column called 'pow_ref' to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for 
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

    Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'pow_ref'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out='pow_ref',
        col_prefix='pow',
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself)


def set_pow_ref_by_n_closest_upstream_turbines(df, df_upstream, turb_no,
    x_turbs, y_turbs, exclude_turbs=[], N=5):
    """Add a column called 'pow_ref' to your dataframe, which is the
    mean of the columns pow_%03d for the 5 closest turbines that are
    upstream of the turbine of interest [turb_no].

    Args:
        df ([pd.DataFrame]): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        df_upstream ([pd.DataFrame]): Dataframe containing rows indicating
        wind direction ranges and the corresponding upstream turbines for 
        that wind direction range. This variable can be generated with
        flasc.floris_tools.get_upstream_turbs_floris(...).
        turb_no ([int]): Turbine number from which the radius should be
        calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius ([float]): Maximum radius for the upstream turbines
        until which they are still considered as relevant/used for the
        calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
        turb_no in the determination of the averaged column quantity. Defaults
        to False.

     Returns:
        df ([pd.DataFrame]): Dataframe which equals the inserted dataframe
        plus the additional column called 'pow_ref'.
    """
    return _set_col_by_n_closest_upstream_turbines(
        col_out='pow_ref',
        col_prefix='pow',
        N=N,
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        exclude_turbs=exclude_turbs,
        circular_mean=False)


def df_reduce_precision(df_in, verbose=False, allow_convert_to_integer=True):
    """Reduce the precision in dataframes from float64 to float32, or possibly
    even further to int32, int16, int8 or even bool. This operation typically
    reduces the size of the dataframe by a factor 2 without any real loss in
    precision. This can make particular operations and data storage much more
    efficient. This can also bring about speed-ups doing calculations with
    these variables.

    Args:
        df_in ([pd.DataFrame]): Dataframe that needs to be reduced.
        verbose (bool, optional): Print progress. Defaults to False.
        allow_convert_to_integer (bool, optional): Allow reduction to integer
           type if possible. Defaults to True.

    Returns:
        df_out ([pd.DataFrame]): Reduced dataframe
    """
    list_out = []
    dtypes = df_in.dtypes
    for ii, c in enumerate(df_in.columns):
        datatype = str(dtypes[c])
        if ((datatype == 'float64') or
            (datatype == 'float32') or
            (datatype == 'float')):
            # Check if can be simplified as integer
            if (not any(np.isnan(df_in[c])) and allow_convert_to_integer and
                all(np.isclose(np.round(df_in[c]),
                               df_in[c], equal_nan=True))):
                unique_values = np.unique(df_in[c])
                if np.array_equal(unique_values, [0, 1]):
                    var_downsampled = df_in[c].astype(bool)
                elif np.max(df_in[c]) < np.iinfo(np.int8).max:
                    var_downsampled = df_in[c].astype(np.int8)
                elif np.max(df_in[c]) < np.iinfo(np.int16).max:
                    var_downsampled = df_in[c].astype(np.int16)
                elif np.max(df_in[c]) < np.iinfo(np.int32).max:
                    var_downsampled = df_in[c].astype(np.int32)
                else:
                    var_downsampled = df_in[c].astype(np.int64)
            else:  # If not, just simplify as float32
                var_downsampled = df_in[c].astype(np.float32)
            max_error = np.max(np.abs(var_downsampled-df_in[c]))
            if verbose:
                print("Column %s ['%s'] was downsampled to %s."
                      % (c, datatype, var_downsampled.dtypes))
                print( "Max error: ", max_error)
        elif ((datatype == 'int64') or
              (datatype == 'int32') or
              (datatype == 'int')):
            if np.array_equal(np.unique(df_in[c]), [0, 1]):
                var_downsampled = df_in[c].astype(bool)
            elif len(np.unique(df_in[c])) < 100:
                var_downsampled = df_in[c].astype(np.int16)
            else:
                var_downsampled = df_in[c].astype(np.int32)
            max_error = np.max(np.abs(var_downsampled-df_in[c]))
            if verbose:
                print("Column %s ['%s'] was downsampled to %s."
                      % (c, datatype, var_downsampled.dtypes))
                print( "Max error: ", max_error)
        else:
            if verbose:
                print("Datatype '%s' not recognized. Not downsampling."
                      % datatype)
            var_downsampled = df_in[c]

        list_out.append(var_downsampled)
    
    df_out = pd.concat(list_out, axis=1, ignore_index=False)
    return df_out


# Functions used for dataframe processing specifically
def df_drop_nan_rows(df, verbose=False):
    """Remove entries in dataframe where all rows (besides 'time')
    have nan values.
    """

    N_init = df.shape[0]
    colnames = [c for c in df.columns if c not in ['time', 'turbid', 'index']]
    df = df.dropna(axis=0, subset=colnames, how='all')

    if verbose:
        print("Reduced dataframe from %d to %d rows." % (N_init, df.shape[0]))

    return df


def df_find_and_fill_data_gaps_with_missing(df, missing_data_buffer=5.):
    """This function takes a pd.DataFrame object and look for large jumps in
       the 'time' column. Rather than simply interpolating these values using
       a ZOH, this rather indicates that measurements are missing. Hence,
       this function finds these time gaps and inserts an additional row
       extra 1 second after the start of the time gap with all 'nan' values.
       This way, the data gap becomes populated with 'nan' values and the data
       will be ignored in any further analysis.

    Args:
        df ([pd.DataFrame]): Merged dataframe for all imported files
        missing_data_buffer (int, optional): If the time gaps are equal or
        larger than this limit [s], then it will consider the data as
        corrupted or missing. Defaults to 10.

    Returns:
        df ([pd.DataFrame]): The postprocessed dataframe where all data
        within large time gaps hold value 'missing'.
    """

    df = df.sort_values(by='time')

    time_values = df['time'].values
    time_delta = np.diff(time_values)

    print('Largest time jump in data is %s s, from %s to %s.'
          % (max(time_delta)/np.timedelta64(1, 's'),
                pd.to_datetime(time_values[np.where(time_delta==max(time_delta))[0][0]]),
                pd.to_datetime(time_values[np.where(time_delta==max(time_delta))[0][0]+1])
                )
            )
    if max(time_delta) >= np.timedelta64(datetime.timedelta(minutes=30)):
        print('Found a gap of > 30 minutes in data.\n' +
              ' Are you missing a data file?')

    dt_buffer = np.timedelta64(missing_data_buffer)
    missing_data_idx = np.where(time_delta >= dt_buffer)[0]
    N_datagaps = len(missing_data_idx)
    td_avg = np.mean(time_delta[missing_data_idx])/np.timedelta64(datetime.timedelta(seconds=1))
    print("  Found %d time jumps in data with an average of %.2f s. Filling datagaps with 'missing'." % (N_datagaps, td_avg))
    times_to_insert = [pd.to_datetime(time_values[i]) + datetime.timedelta(seconds=1) for i in missing_data_idx]

    # Create empty dataframe and insert times_to_insert as nans
    df_entries_missing = df[0:1].reset_index().drop(columns='index').drop(0)
    df_entries_missing['time'] = times_to_insert
    df_entries_missing = df_entries_missing.replace(pd.NaT, 'missing')

    for mi in np.where(time_delta > np.timedelta64(30, 's'))[0]:
        print("  Significant time jump in data of %s s has happened from %s to %s."
              % (time_delta[mi]/np.timedelta64(1, 's'),
                 pd.to_datetime(time_values[mi]),
                 pd.to_datetime(time_values[mi+1])
                 )
              )

    df = df.append(df_entries_missing)  # Add new row with 'missing' entries
    df = df.sort_values(by='time')  # Sort by time
    df = df.reset_index().drop(columns='index')  # Reset index

    return df


def df_sort_and_find_duplicates(df):
    """This function sorts the dataframe and finds rows with equal time index.

    Args:
        df ([pd.DataFrame]): An (unsorted) dataframe

    Returns:
        df ([pd.DataFrame]): Dataframe sorted by time
        duplicate_entries_idx ([list of int]): list with indices of the former
        of two duplicate rows. The indices correspond to the time-sorted df.
    """

    df = df.sort_values(axis=0, by='time', ignore_index=True)
    time_delta = np.diff(df['time'].values)
    duplicate_entries_idx = np.where(np.abs(np.float64(time_delta)) < 1e-3)[0]

    # Clean up
    if 'index' in df.columns:
        df = df.drop(columns='index')

    return df, duplicate_entries_idx


def make_df_wide(df):
    df["turbid"] = df['turbid'].astype(int)
    df = df.reset_index(drop=False)
    if 'index' in df.columns:
        df = df.drop(columns='index')
    df = df.set_index(["time", "turbid"], drop=True)
    df = df.unstack()
    df.columns = ["%s_%s" % c for c in df.columns]
    df = df.reset_index(drop=False)
    return df


def df_sort_and_fix_duplicates(df):
    """This function sorts the dataframe and addresses duplicate rows (i.e.,
    rows in which the time index is equal). It does this by merging the two
    rows, replacing the 'nan' entries of one row with the non-'nan' entries
    of the other row. If someone both rows have different values for the same
    column, then an exception is thrown.

    Args:
        df ([pd.DataFrame]): An (unsorted) dataframe

    Returns:
        df ([pd.DataFrame]): A time-sorted Dataframe in which its duplicate
        rows have been merged.
    """
    # Check and merge any duplicate entries in the dataset
    df, duplicate_time_entries = df_sort_and_find_duplicates(df)
    while len(duplicate_time_entries) > 0:
        di = duplicate_time_entries[0]
        # df_subset = df[di:di+2].copy()

        # Check if any conflicting entries exist within duplicate rows
        column_list = [c for c in df.columns if (c != 'time' and c != 'index')]
        df_merged = df[di:di+1].copy().reset_index(drop=True)  # Start with first row
        for c in column_list:
            x1 = df.loc[di, c]
            x2 = df.loc[di+1, c]

            # Check if either is NaN
            x1_isnan = not (x1 == x1)
            x2_isnan = not (x2 == x2)

            # Check if values conflict
            if x1_isnan:
                is_faulty = False
                df_merged.loc[0, c] = x2
            elif x2_isnan:
                is_faulty = False
                # Do nothing, keep x1
            elif x1 == x2:
                is_faulty = False
                # Do nothing, keep x1
            else:
                is_faulty = True
                df_merged.loc[0, c] = np.nan

            if is_faulty:
                import warnings
                warnings.warn('Found conflicting data entries for timestamp: '
                              + str(df.loc[di, 'time']) + '.')
                print(df.loc[di:di+1, c])
                print('Setting value to np.nan as a safety measure...')

        print('Merged two rows with identical timestamp:',
              df.loc[di, 'time'], '.')
        print('Before merging:')
        print(df[di:di+2])
        print(' ')
        print('After merging:')
        print(df_merged)
        print(' ')

        # Now merge data
        df = df.reset_index().drop([di, di+1])  # Remove dupl. rows
        df = df.append(df_merged)  # Add merged row

        # Sort df by 'time' and recalculate duplicate entries
        df, duplicate_time_entries = df_sort_and_find_duplicates(df)

    return df
