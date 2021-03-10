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
import warnings

from floris_scada_analysis.circular_statistics import wrap_360_deg

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


def set_wd_by_turbines(df, turbine_numbers):
    # wd_turbines = df[['wd_%03d' % ti for ti in turbine_numbers]]

    # # Use unit vectors to calculate the mean
    # wd_x = np.cos(wd_turbines * np.pi / 180.).sum(axis=1)
    # wd_y = np.sin(wd_turbines * np.pi / 180.).sum(axis=1)

    # mean_wds = np.arctan2(wd_y, wd_x)
    # mean_wds = wrap_360_deg(mean_wds * 180. / np.pi)

    df['wd'] = get_column_mean(df, col_prefix='wd', circular_mean=True)
    return df


def set_wd_by_all_turbines(df):
    num_turbines = get_num_turbines(df)
    return set_wd_by_turbines(df, range(num_turbines))


def set_wd_by_radius_from_turbine(df, turb_no, x_turbs, y_turbs,
                                  max_radius, include_itself=True):

    turbs_within_radius = get_turbs_in_radius(
        x_turbs=x_turbs, y_turbs=y_turbs, turb_no=turb_no,
        max_radius=max_radius, include_itself=include_itself)

    if len(turbs_within_radius) < 1:
        print('No turbines within proximity. Try to increase radius.')
        return None

    return set_wd_by_turbines(df, turbs_within_radius)


def set_ws_by_turbines(df, turbine_numbers):
    ws_turbines = df[['ws_%03d' % ti for ti in turbine_numbers]]
    df['ws'] = np.nanmean(ws_turbines, axis=1)
    return df


def set_ws_by_all_turbines(df):
    num_turbines = get_num_turbines(df)
    return set_ws_by_turbines(range(num_turbines))


def set_ws_by_upstream_turbines(df, df_upstream, exclude_turbs=[]):
    # Can get df_upsteam using floris_tools.get_upstream_turbs_floris()
    df['ws'] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, 'wd_min']
        wd_max = df_upstream.loc[i, 'wd_max']
        upstr_turbs = df_upstream.loc[i, 'turbines']

        # Exclude particular turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]

        if wd_min > wd_max:  # Wrap around
            ids = [a or b for a, b in zip(df['wd'] > wd_min, df['wd'] <= wd_max)]
        else:
            ids = [a and b for a, b in zip(df['wd'] > wd_min, df['wd'] <= wd_max)]

        ws_mean = get_column_mean(df.loc[ids, :], col_prefix='ws',
                                  turbine_list=upstr_turbs)
        df.loc[ids, 'ws'] = ws_mean

    return df


def set_ti_by_turbines(df, turbine_numbers):
    ti_turbines = df[['ti_%03d' % ti for ti in turbine_numbers]]
    df['ti'] = np.nanmean(ti_turbines, axis=1)
    return df


def set_ti_by_all_turbines(df):
    num_turbines = get_num_turbines(df)
    return set_ti_by_turbines(range(num_turbines))


def set_ti_by_upstream_turbines(df, df_upstream, exclude_turbs=[]):
    # Can get df_upsteam using floris_tools.get_upstream_turbs_floris()
    df['ti'] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, 'wd_min']
        wd_max = df_upstream.loc[i, 'wd_max']
        upstr_turbs = df_upstream.loc[i, 'turbines']

        # Exclude particular turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]

        if wd_min > wd_max:  # Wrap around
            ids = [a or b for a, b in zip(df['wd'] > wd_min, df['wd'] <= wd_max)]
        else:
            ids = [a and b for a, b in zip(df['wd'] > wd_min, df['wd'] <= wd_max)]

        ti_mean = get_column_mean(df.loc[ids, :], col_prefix='ti',
                                  turbine_list=upstr_turbs)
        df.loc[ids, 'ti'] = ti_mean

    return df


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
