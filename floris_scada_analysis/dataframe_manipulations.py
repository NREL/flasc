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


def get_num_turbines(df):
    # Let's assume that the format of variables is ws_%03d, wd_%03d, and so on
    num_turbines = len([c for c in df.columns if 'ws_' in c and len(c) == 6])
    return num_turbines


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
            if (
                not np.isnan(df_subset.iloc[0, :][c]) and
                not np.isnan(df_subset.iloc[1, :][c])
               ):
                import warnings
                warnings.warn('Found conflicting data entries for timestamp:' + str(df_subset.iloc[0,0]))
                print(df_subset[c])
                print('Setting value to np.nan as a safety measure...')
                c_indx = np.where([c == df_subset.columns])[1][0]
                df_subset.iloc[0, c_indx] = np.nan
                df_subset.iloc[1, c_indx] = np.nan

        # If no conflicts found, merge data
        df_subset = df_subset.reset_index(drop=('time' in df_subset.columns))
        df_merged = df_subset.head(1).fillna(df_subset.tail(1))
        df = df.reset_index().drop([di, di+1])  # Remove dupl. rows
        df = df.append(df_merged)  # Add merged row

        # Sort df by 'time' and recalculate duplicate entries
        df, duplicate_time_entries = df_sort_and_find_duplicates(df)

    return df
