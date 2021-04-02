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
import os
import pandas as pd
import re
# import warnings

from floris_scada_analysis import dataframe_manipulations as dfm


def generate_models_py(table_names, table_upsampling, fn_channel_defs,
                       wide_format, turbine_range, repository_name,
                       settingsfile_name, fout):
    """A function to automatically generate settings.py for an SQL
    wind farm data repository.

    Args:
        table_names ([list]): List of SQL table names
        table_upsampling ([list]): List of bools defining whether
        table will contain upsampled data or not. If yes, then
        additional table entries will be made to include _mean,
        _median, _std, _min and _max for all numerical variables.
        fn_channel_defs ([str]): Path to xlsx with channel definitions.
        wide_format ([bool]): Definition whether table is WIDE or TALL.
        turbine_range ([list]): List of turbine names.
        repository_name ([str]): Name of Python repository
        settingsfile_name ([str]): Name of settings file for Python/SQL
        fout ([str]): Path defining desired output directory for models.py

    Raises:
        Exception: Will raise an error when models.py already exists,
        and when additionally models.py.old is already taken.
    """

    # Initialize default template
    modelspy_str = \
    '''
    # AUTOMATICALLY GENERATED FUNCTION BY 'GENERATE_MODELSPY.PY'
    #      DATE: ''' + datetime.datetime.now().strftime('%H:%M:%S on %A, %B the %dth, %Y') + ''''

    from sqlalchemy import create_engine
    from sqlalchemy import Table, Column, String, Integer, DateTime, UniqueConstraint, REAL
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.engine.url import URL

    from ''' + repository_name + ''' import ''' + settingsfile_name + '''

    DeclarativeBase = declarative_base()


    def db_connect():
        """
        Performs database connection using database settings from settings.py.
        Returns sqlalchemy engine instance
        """
        print(''' + settingsfile_name + '''.DATABASE)
        return create_engine(URL(**''' + settingsfile_name + '''.DATABASE), pool_timeout=60*60*24)


    def create_tables(engine):
        """ Initialize all tables """
        DeclarativeBase.metadata.create_all(engine)


    def recreate_all(engine):
        """ Delete all tables and data and recreate base tables """
        print("DROPPING AND RECREATING ALL TABLES")
        DeclarativeBase.metadata.drop_all(engine)
        DeclarativeBase.metadata.create_all(engine)


    def recreate_by_type(engine, table_name):
        """ Delete tables by type and recreate"""

        table_name = table_name.lower()
    '''

    # Populate recreate_by_type(...) function
    for table_name in table_names:
        modelspy_str = modelspy_str + '''
        if table_name == "''' + table_name + '''":
            DeclarativeBase.metadata.drop_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
            DeclarativeBase.metadata.create_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
        '''


    # Add individual classes for table_name_filenames and table_names
    for iii in range(len(table_names)):
        table_name = table_names[iii]

        modelspy_str = modelspy_str + '''
    class ''' + table_name + '''_filenames_class(DeclarativeBase):
        """Sqlalchemy filenames model"""
        __tablename__ = "''' + table_name + '''_filenames"
        id = Column(Integer, primary_key=True, autoincrement=True)
        filename = Column('filename', String, nullable=False, primary_key=True, unique=True)

    class ''' + table_name + '''_class(DeclarativeBase):
        """Sqlalchemy data model"""
        __tablename__ = "''' + table_name + '''"

        index = Column('index', Integer, primary_key=True, autoincrement=True)'''

        # Add individual variable definitions
        df_defs = pd.read_excel(fn_channel_defs, sheet_name='scada_data')

        # Extract channel mappings and determine to-be-removed channels
        original_name = df_defs[df_defs.columns[0]]  # Raw dataset names
        database_name = df_defs[df_defs.columns[1]]  # Desired database names
        database_omit = df_defs['remove_id']  # To-be-omitted fields
        variable_type = df_defs['var_type']  # Variable type
        remove_fields = [
            database_name[i]
            for i in range(len(original_name))
            if database_omit[i].lower() == "yes"
        ]

        # Create index/unique variables first
        if 'turbid' in database_name.values and not wide_format:
            remove_fields.append('turbid')
            modelspy_str = modelspy_str + ''' 

        # Time and turbine are the indexed columns (OVERWRITE)
        time = Column('time', DateTime, index=True)
        turbid = Column('turbid', Integer,nullable=True, index=True)
        UniqueConstraint('time', 'turbid', name='timeturbid')
        '''
        else:
            modelspy_str = modelspy_str + ''' 

        # Time is the indexed column (OVERWRITE)
        time = Column('time', DateTime, primary_key=True, unique=True, index=True)
        '''

        # Create all necessary db variables in models.py
        remove_fields.append('time')  # Do not repeat time variable
        if wide_format:
            remove_fields.append('turbid')
            db_name_new = []
            vb_type_new = []
            for di in range(len(database_name)):
                if database_name[di] not in remove_fields:
                    for ti in turbine_range:
                        if table_upsampling[iii]:
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_mean")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_median")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_std")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_min")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_max")
                            vb_type_new.append(variable_type[di])
                        else:
                            db_name_new.append(database_name[di] + "_" + str(ti))
                            vb_type_new.append(variable_type[di])
            database_name = db_name_new
            variable_type = vb_type_new

        for i in range(len(database_name)):
            if database_name[i] not in remove_fields:
                modelspy_str = modelspy_str + '''
        ''' + database_name[i] + ''' = Column("''' + database_name[i] + '''", ''' + variable_type[i] + ''', nullable=True)'''
        modelspy_str = modelspy_str + '''
        '''


    # WRITE TO FILE, but make sure no models.py file gets overwritten
    if os.path.exists(fout):
        if os.path.exists(fout + '.old'):
            raise Exception('Both a models.py and models.py.old file exist in fout. Remove one or both.')
        else:
            os.rename(fout, fout+'.old')
            print("Found existing file. Renamed 'models.py' to 'models.py.old'.")

    # Write a new models.py
    text_file = open(fout, "w")
    n = text_file.write(modelspy_str)
    text_file.close()
    print("Finished writing 'models.py' to " + fout + ".")

    return None


def get_latest_time(filelist):
    time_latest = None
    for fi in filelist:
        df = pd.read_feather(fi)
        time_array = df['time']
        if not all(np.isnan(time_array)):
            tmp_time_max = np.max(df['time'])
            if time_latest is None:
                time_latest = tmp_time_max
            else:
                if tmp_time_max > time_latest:
                    time_latest = tmp_time_max

    return time_latest


def browse_datafiles(data_path, scada_table=''):
    if scada_table == '':
        fn_pattern = re.compile('\d\d\d\d-\d\d_.*.ftr')  # Import all
    else:
        fn_pattern = re.compile('\d\d\d\d-\d\d_' + scada_table + '.ftr')

    files_list = []

    for root, _, files in os.walk(data_path):
        for name in files:
            fn_item = fn_pattern.findall(name)
            if len(fn_item) > 0:
                fn_item = fn_item[0]
                fn_path = os.path.join(root, fn_item)
                files_list.append(fn_path)

    # Sort alphabetically/numerically
    files_list = list(np.sort(files_list))
    files_list = [str(f) for f in files_list]

    if len(files_list) == 0:
        print('No data files found in ' + data_path)

    return files_list


def sqldb_get_min_time(dbc, table_name):
    query_string = 'SELECT time FROM ' + table_name + ' ORDER BY time asc LIMIT 1'
    df_time = pd.read_sql_query(query_string, dbc.engine)
    start_time = df_time['time'][0]

    return start_time


def sqldb_get_max_time(dbc, table_name):
    query_string = 'SELECT time FROM ' + table_name + ' ORDER BY time desc LIMIT 1'
    df_time = pd.read_sql_query(query_string, dbc.engine)
    end_time = df_time['time'][0]

    return end_time


# Formerly a_00_initial_download.py
def batch_download_data_from_sql(dbc, destination_path,
                                 df_table_name='scada_data_60s'):
    print("Batch downloading data from table '"
          + df_table_name + "'...")

    # Check current start and end time of database
    db_end_time = sqldb_get_max_time(dbc, df_table_name)
    db_end_time = db_end_time + datetime.timedelta(minutes=10)

    # Check for past files and continue download or start a fresh download
    files_result = browse_datafiles(destination_path,
                                    scada_table=df_table_name)
    print('A total of %d existing files found.' % len(files_result))

    latest_timestamp = get_latest_time(files_result)
    # Next timestamp is going to be next first of the month
    if latest_timestamp is None:
        db_start_time = sqldb_get_min_time(dbc, df_table_name)
        db_start_time = db_start_time - datetime.timedelta(minutes=10)
        current_timestamp = db_start_time
    elif latest_timestamp.month == 12:
        current_timestamp = pd.to_datetime(
            str(latest_timestamp.year+1)+'-01-01')
    else:
        current_timestamp = pd.to_datetime(
            str(latest_timestamp.year)+ '-' + 
            str(latest_timestamp.month+1) + '-01')

    print('Continuing import from timestep: ', current_timestamp)
    while current_timestamp <= db_end_time:
        print('Importing data for ' + 
              str(current_timestamp.strftime("%B")) +
              ', ' + str(current_timestamp.year) + '.')
        if current_timestamp.month == 12:
            next_timestamp = current_timestamp.replace(
                year=current_timestamp.year+1, month=1,
                day=1, hour=0, minute=0, second=0)
        else:
            next_timestamp = current_timestamp.replace(
                month=current_timestamp.month+1,
                day=1, hour=0, minute=0, second=0)

        df_table = dbc.get_table_data_from_db_wide(
            table_name=df_table_name,
            start_time=current_timestamp,
            end_time=next_timestamp
            )

        # Drop NaN rows
        df_table = dfm.df_drop_nan_rows(df_table)

        # Save dataset as a .p file
        fout = os.path.join(destination_path, 
                            current_timestamp.strftime("%Y-%m")
                            + "_" + df_table_name + ".ftr")
        df_table = df_table.reset_index(drop=True)
        df_table.to_feather(fout)
        print('Data for ' + df_table_name +
              ' saved to .ftr files for ' +
              str(current_timestamp.strftime("%B")) +
              ', ' + str(current_timestamp.year) + '.')

        # Update start_time
        current_timestamp = next_timestamp
        print(' ')  # Blank line for log clarity


# Formerly a_01_structure_data.py
def _restructure_single_df(df, column_mapping_dict):
    print('  Processing dataset...')

    if df.shape[0] < 1:
        return None

    # Drop NaN rows and get basic df info
    df = dfm.df_drop_nan_rows(df)
    # no_rows = df.shape[0]

    # Build up the new data frame
    df_structured = pd.DataFrame({'time': df.time})
    col_names_target = list(column_mapping_dict.keys())
    col_names_original = list(column_mapping_dict.values())

    # Map columns of interest
    for i in range(len(col_names_original)):
        cn_original = col_names_original[i]
        cn_target = col_names_target[i]
        df_structured[cn_target] = df[cn_original]
    print('    Copied the columns to the new dataframe with the appropriate naming.')

    return df_structured
