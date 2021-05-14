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
import os
import pandas as pd

from floris_scada_analysis import time_operations as fsato
from floris_scada_analysis import utilities as fsut


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

    if len(df_array) == 0:
        df_out = pd.DataFrame()
    else:
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
    dt = fsut.estimate_dt(time_array)

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

    # Create output folder
    os.makedirs(save_path, exist_ok=True)

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
