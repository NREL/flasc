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
import numpy as np
# import os as os
import pandas as pd
import warnings

from floris_scada_analysis.circular_statistics import wrap_360_deg
from floris_scada_analysis import time_operations as fsato
from floris_scada_analysis import floris_tools as ftools


# Functions related to wind farm analysis for df
def filter_df_by_ws(df, ws_range):
    df = df[df['ws'] > ws_range[0]]
    df = df[df['ws'] <= ws_range[1]]
    return df


def filter_df_by_wd(df, wd_range):
    df = df[df['wd'] > wd_range[0]]
    df = df[df['wd'] <= wd_range[1]]
    return df


def filter_df_by_ti(df, ti_range):
    df = df[df['ti'] > ti_range[0]]
    df = df[df['ti'] <= ti_range[1]]
    return df


def get_num_turbines(df):
    # Let's assume that the format of variables is ws_%03d, wd_%03d, and so on
    num_turbines = len([c for c in df.columns if 'ws_' in c and len(c) == 6])
    return num_turbines


# Generic functions for column operations
def get_column_mean(df, col_prefix='pow', turbine_list=None,
                    circular_mean=False):
    if turbine_list is None:
        turbine_list = range(get_num_turbines(df))  # Assume all turbines

    col_names = [col_prefix + '_%03d' % ti for ti in turbine_list]
    array = df[col_names]

    if circular_mean:
        # Use unit vectors to calculate the mean
        dir_x = np.cos(array * np.pi / 180.).sum(axis=1)
        dir_y = np.sin(array * np.pi / 180.).sum(axis=1)

        mean_dirs = np.arctan2(dir_y, dir_x)
        mean_out = wrap_360_deg(mean_dirs * 180. / np.pi)
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


def _set_col_by_upstream_turbines(col_out, col_prefix, df,
                                  df_upstream, circular_mean,
                                  exclude_turbs=[]):
    # Can get df_upstream using floris_tools.get_upstream_turbs_floris()
    df[col_out] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, 'wd_min']
        wd_max = df_upstream.loc[i, 'wd_max']
        upstr_turbs = df_upstream.loc[i, 'turbines']

        # Exclude particular turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]

        if wd_min > wd_max:  # Wrap around
            ids = (df['wd'] > wd_min) | (df['wd'] <= wd_max)
        else:
            ids = (df['wd'] > wd_min) & (df['wd'] <= wd_max)

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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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
        floris_scada_analysis.floris_tools.get_upstream_turbs_floris(...).
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


# Other dataframe manipulations
def filter_df_by_status(df, exclude_columns=[]):
    """This function overwrites measurement values with np.nan wherever
    the related status flag for that particular turbine reports a value
    of 0 (status_000 = 0, status_001 = 0, ....). You can exclude particular
    columns from being overwritten by inserting them into the
    exclude_columns list.

    Args:
        df ([pd.DataFrame]): Dataframe with SCADA data with measurements
        formatted according to wd_000, wd_001, wd_002, pow_000, pow_001,
        pow_002, and so on.
        exclude_fields (list, optional): Columns that should not be over-
        written by a np.nan value. Defaults to [], and will only prevent
        overwriting of columns containing the substring "status".

    Returns:
        df([pd.DataFrame]): The dataframe with measurements overwritten
        with a np.nan value wherever that turbine's status flag reports
        a value of 0.
    """

    turbine_list = range(get_num_turbines(df))
    status_cols = ["status_%03d" % ti for ti in turbine_list]
    status_cols = [c for c in status_cols if c in df.columns]
    if len(status_cols) < len(turbine_list):
        print('Found fewer status columns (%d) than turbines (%d).'
              % (len(status_cols), len(turbine_list)) +
              ' Ignoring missing entries.')

    exclude_columns.extend([c for c in df.columns if 'status' in c])
    for c in status_cols:
        ti_string = c[-4::] # Last 4 digits of string: underscore and turb. no
        ti_columns = [s for s in df.columns if s[-4::] == ti_string and
                      not s in exclude_columns]
        df.loc[df[c] == 0, ti_columns] = np.nan

    return df


def df_reduce_precision(df_in, verbose=False):
    """Reduce the precision in dataframes from float64 to float32, or possibly
    even further to int32, int16, int8 or even bool. This operation typically
    reduces the size of the dataframe by a factor 2 without any real loss in
    precision. This can make particular operations and data storage much more
    efficient. This can also bring about speed-ups doing calculations with
    these variables.

    Args:
        df_in ([pd.DataFrame]): Dataframe that needs to be reduced.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        df_out ([pd.DataFrame]): Reduced dataframe
    """
    df_out = pd.DataFrame()
    dtypes = df_in.dtypes
    for ii, c in enumerate(df_in.columns):
        datatype = str(dtypes[c])
        if ((datatype == 'float64') or
            (datatype == 'float32') or
            (datatype == 'float')):
            # Check if can be simplified as integer
            if (not any(np.isnan(df_in[c])) and
                all(np.isclose(np.round(df_in[c]),
                               df_in[c], equal_nan=True))):
                unique_values = np.unique(df_in[c])
                if np.array_equal(unique_values, [0, 1]):
                    df_out[c] = df_in[c].astype(bool)
                elif np.max(df_in[c]) < np.iinfo(np.int8).max:
                    df_out[c] = df_in[c].astype(np.int8)
                elif np.max(df_in[c]) < np.iinfo(np.int16).max:
                    df_out[c] = df_in[c].astype(np.int16)
                elif np.max(df_in[c]) < np.iinfo(np.int32).max:
                    df_out[c] = df_in[c].astype(np.int32)
                else:
                    df_out[c] = df_in[c].astype(np.int64)
            else:  # If not, just simplify as float32
                df_out[c] = df_in[c].astype(np.float32)
            max_error = np.max(np.abs(df_out[c]-df_in[c]))
            if verbose:
                print("Column %s ['%s'] was downsampled to %s."
                      % (c, datatype, df_out.dtypes[ii]))
                print( "Max error: ", max_error)
        elif ((datatype == 'int64') or
              (datatype == 'int32') or
              (datatype == 'int')):
            if all(np.unique(df_in[c]) == [0, 1]):
                df_out[c] = df_in[c].astype(bool)
            elif len(np.unique(df_in[c])) < 100:
                df_out[c] = df_in[c].astype(np.int16)
            else:
                df_out[c] = df_in[c].astype(np.int32)
            max_error = np.max(np.abs(df_out[c]-df_in[c]))
            if verbose:
                print("Column %s ['%s'] was downsampled to %s."
                      % (c, datatype, df_out.dtypes[ii]))
                print( "Max error: ", max_error)
        else:
            if verbose:
                print("Datatype '%s' not recognized. Not downsampling."
                      % datatype)
            df_out[c] = df_in[c]

    return df_out


def batch_load_and_concat_dfs(df_filelist):
    """Function to batch load and concatenate dataframe files. Data
    in floris_scada_analysis is typically split up in monthly data
    files to accommodate very large data files and easy debugging
    and batch processing. A common method for loading data is:
    
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, 'data')
    df_filelist = sqldbm.browse_datafiles(data_path=data_path,
                                          scada_table='scada_data')
    df = dfm.batch_load_and_concat_dfs(df_filelist=df_filelist)

    Args:
        df_filelist ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df_array = []
    for dfn in df_filelist:
        df_array.append(pd.read_feather(dfn))

    df_out = pd.concat(df_array, ignore_index=True)
    df_out = df_out.reset_index(drop=('time' in df_out.columns))
    df_out = df_out.sort_values(by='time')
    return df_out


def batch_split_and_save_dfs(df, save_path, table_name='scada_data'):
    df = df.copy()
    if 'time' not in df.columns:
        df = df.reset_index(drop=False)
    else:
        df = df.reset_index(drop=True)

    time_array = pd.to_datetime(df['time'])
    dt = fsato.estimate_dt(time_array)

    # Check if dataframe is continually ascending
    if (any([float(i) for i in np.diff(time_array)]) <= 0):
        raise KeyError("Time column in dataframe is not ascending.")

    df_array = []
    # time_start = list(time_array)[0]
    # time_end = list(time_array)[-1]

    df_time_windows = []
    years = np.unique([t.year for t in time_array])
    for yr in years:
        months = np.unique([t.month for t in time_array
                            if t.year == yr])
        for mo in months:
            tw0 = pd.to_datetime('%04d-%02d-01 00:00:00' % (yr, mo)) + dt
            if mo == 12:
                tw1 = pd.to_datetime('%04d-%02d-01 00:00:00' % (yr+1, 1))
            else:
                tw1 = pd.to_datetime('%04d-%02d-01 00:00:00' % (yr, mo+1))
            df_time_windows.append([tw0, tw1])

    # Extract time indices
    print('Splitting the data into %d separate months.' % len(df_time_windows))
    id_map = fsato.find_window_in_time_array(time_array, df_time_windows)
    for ii in range(len(id_map)):
        df_sub = df.copy().loc[id_map[ii]].reset_index(drop=True)
        year = list(pd.to_datetime(df_sub.time))[0].year
        month = list(pd.to_datetime(df_sub.time))[0].month
        fn = '%04d-%02d' % (year, month) + '_' + table_name + '.ftr'
        df_sub.to_feather(os.path.join(save_path, fn))
        df_array.append(df_sub)
    print('Saved the output files to %s.' % save_path)

    return df_array


# Functions used for dataframe processing specifically
def df_drop_nan_rows(df, verbose=False):
    """Remove entries in dataframe where all rows (besides 'time')
    have nan values.
    """

    N_init = df.shape[0]
    colnames = [c for c in df.columns if 'time' not in c]
    df = df.dropna(axis=0, subset=colnames, how='all')

    if verbose:
        print("Reduced dataframe from %d to %d rows." % (N_init, df.shape[0]))

    return df


def df_find_and_fill_data_gaps_with_missing(df, missing_data_buffer_s=10.):
    """This function takes a pd.DataFrame object and look for large jumps in
       the 'time' column. Rather than simply interpolating these values using
       a ZOH, this rather indicates that measurements are missing. Hence,
       this function finds these time gaps and inserts an additional row
       extra 1 second after the start of the time gap with all 'nan' values.
       This way, the data gap becomes populated with 'nan' values and the data
       will be ignored in any further analysis.

    Args:
        df ([pd.DataFrame]): Merged dataframe for all imported files
        missing_data_buffer_s (int, optional): If the time gaps are equal or
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

    dt_buffer = np.timedelta64(datetime.timedelta(seconds=missing_data_buffer_s))
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


def restructure_df_files(df_table_name, column_mapping_dict,
                         data_path, target_path):
    print('Loading, processing and saving dataframes for '+df_table_name+'.')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    files_result = browse_datafiles(data_path, scada_table=df_table_name)
    files_result = np.sort(files_result)
    for fi in files_result:
        print('  Reading ' + fi + '.')
        df_in = pd.read_feather(fi)  # Read
        df_out = _restructure_single_df(df_in, column_mapping_dict)
        if df_out is not None:
            if df_out.shape[0] > 0:
                fout = os.path.join(target_path, os.path.basename(fi))
                df_out.reset_index(drop=True).to_feather(fout)


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
        df_subset = df[di:di+2].copy()

        # Check if any conflicting entries exist within duplicate rows
        column_list = [c for c in df.columns if (c != 'time' and c != 'index')]
        for c in column_list:
            is_faulty = False
            x1 = df_subset.iloc[0, :][c]
            x2 = df_subset.iloc[1, :][c]
            if not type(x1) == type(x2):
                is_faulty = True
            elif isinstance(x1, str):
                is_faulty = (not (x1 == x2))
            elif isinstance(x1, float) | isinstance(x1, int):
                is_faulty = not np.array_equal(x1, x2, equal_nan=True)
            else:
                is_faulty = (not (x1 == x2))
            if is_faulty:
                import warnings
                warnings.warn('Found conflicting data entries for timestamp:' +
                              str(df_subset.iloc[0]['time']))
                print(df_subset[c])
                print('Setting value to np.nan as a safety measure...')
                c_indx = np.where([c == df_subset.columns])[1][0]
                df_subset.iloc[0, c_indx] = np.nan
                df_subset.iloc[1, c_indx] = np.nan

        # Now merge data
        df_subset = df_subset.reset_index(drop=('time' in df_subset.columns))
        df_merged = df_subset.head(1).fillna(df_subset.tail(1))
        df = df.reset_index().drop([di, di+1])  # Remove dupl. rows
        df = df.append(df_merged)  # Add merged row

        # Sort df by 'time' and recalculate duplicate entries
        df, duplicate_time_entries = df_sort_and_find_duplicates(df)

    return df
