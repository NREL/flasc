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
import glob
import numpy as np
import os
import pandas as pd
import re
from time import perf_counter as timerpc

from floris_scada_analysis import dataframe_manipulations as dfm


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # REPOSITORY CREATION FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # #
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
        df_defs = pd.read_excel(fn_channel_defs, sheet_name=table_name)

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


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # BASIC SQL DATABASE PROPERTY READING
# # # # # # # # # # # # # # # # # # # # # # # # #
def get_first_time_entry_sqldb(sql_engine, table_name):
    # Get the list of scada filenames
    query_string = 'SELECT time FROM ' + table_name + ' ORDER BY time asc LIMIT 1'
    df_time = pd.read_sql_query(query_string, sql_engine)
    start_time = df_time['time'][0]

    return start_time


def get_last_time_entry_sqldb(sql_engine, table_name):
    # Get the list of scada filenames
    query_string = 'SELECT time FROM ' + table_name + ' ORDER BY time desc LIMIT 1'
    df_time = pd.read_sql_query(query_string, sql_engine)
    end_time = df_time['time'][0]

    return end_time


def get_column_names_sqldb(sql_engine, table_name):
    # Find all columns
    df = pd.read_sql_query(
        "SELECT * FROM " + table_name + " WHERE false;", sql_engine)
    column_names = list(df.columns)
    return column_names


def get_timestamp_of_last_downloaded_datafile(filelist):
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


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # RAW DATA READING FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # #
def find_files_to_add_to_sqldb(sqldb_engine, files_paths, filenames_table):
    """This function is used to figure out which files are already
    uploaded to the SQL database, and which files still need to be
    uploaded.

    Args:
        sqldb_engine ([SQL engine]): SQL Engine from sqlalchemy.create_engine
        used to access the SQL database of interest. This is used to call
        which files have previously been uploaded.
        files_paths ([list, str]): List of strings or a single string containing
        the path to the raw data files. One example is:
            files_paths = ['/home/user/data/windfarm1/year1/*.csv',
                           '/home/user/data/windfarm1/year2/*.csv',
                           '/home/user/data/windfarm1/year3/*.csv',]
        filenames_table ([str]): SQL table name containing the filenames of
        the previously uploaded data files.

    Returns:
        files ([list]): List of files that are not yet in the SQL database
    """
    # Convert files_paths to a list
    if isinstance(files_paths, str):
        files_paths = [files_paths]

    # Figure out which files exists on local system
    files = []
    for fpath in files_paths:
        files.extend(glob.glob(fpath))

    # Figure out which files have already been uploaded to sql db
    query_string = "select * from " + filenames_table + ";"
    df = pd.read_sql_query(query_string, sqldb_engine)

    # # Check for the files not in the database
    files = [f for f in files
             if os.path.basename(f) not in df['filename'].values]

    # Sort the file list according to ascending name
    files = sorted(files, reverse=False)

    return files


def browse_downloaded_datafiles(data_path, scada_table=''):
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


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # DATA UPLOAD FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # #
def upload_df_to_sqldb(df, files, sql_engine, table_name,
                       df_chunk_size=10000, sql_chunk_size=10000):
    if df.shape[0] > 0:
        print("Inserting  %d rows into table '%s'."
              % (df.shape[0], table_name))
        df_chunks_id = np.arange(0, df.shape[0], df_chunk_size)
        df_chunks_id = np.append(df_chunks_id, df.shape[0])
        df_chunks_id = np.unique(df_chunks_id)

        for i in range(len(df_chunks_id)-1):
            print("Inserting rows %d to %d" % (df_chunks_id[i],  df_chunks_id[i+1]))
            time_start = timerpc()
            df_sub = df[df_chunks_id[i]:df_chunks_id[i+1]]
            df_sub.to_sql(
                table_name,
                sql_engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=sql_chunk_size,
            )
            time_end = timerpc()
            print("Data insertion took %.1f s." % (time_end - time_start))

    # Add all but last entry file names to data_fn table. Namely, the last
    # filename will be necessary for interpolation with the first new file
    # that will come in during the next upload.
    if isinstance(files, str):
        files = [files]

    for fi in files[0:-1]:
        fi = os.path.basename(fi)  # Only filename.csv
        df_names = pd.DataFrame({"filename": [fi]})
        df_names.to_sql(
            table_name + "_filenames",
            sql_engine,
            if_exists="append",
            index=False,
            chunksize=50,
        )


def omit_last_rows_by_buffer(df, omit_buffer):
    num_rows = df.shape[0]
    df = df[df['time'] < max(df['time']) - omit_buffer]
    print('Omitting last %d rows (%s s) as a buffer for future files.'
          % (num_rows-df.shape[0], omit_buffer))
    return df


def remove_duplicates_with_sqldb(df, sql_engine, table_name):
    min_time = df.time.min()
    max_time = df.time.max()

    time_query = (
        "select time from "+ table_name +
        " where time BETWEEN '%s' and '%s';" % (min_time, max_time)
    )
    
    df_time = pd.read_sql_query(time_query, sql_engine)
    df_time["time"] = pd.to_datetime(df_time.time)
    print("......Before duplicate removal there are %d rows" % df.shape[0])
    df = df[~df.time.isin(df_time.time)]
    print("......After duplicate removal there are %d rows" % df.shape[0])

    # Check for self duplicates in to-be-uploaded dataset
    print("......Before SELF duplicate removal there are %d rows" % df.shape[0])
    if "turbid" in df.columns:
        df = df.drop_duplicates(subset=["time", "turbid"], keep="first")
    else:
        df = df.drop_duplicates(subset=["time"], keep="first")
    print("......After SELF duplicate removal there are %d rows" % df.shape[0])

    # Drop null times in to-be-uploaded dataset
    print(
        "......Before null time/turbid duplicate removal there are %d rows"
        % df.shape[0]
    )
    df = df.dropna(subset=["time"])
    print("......After null time duplicate removal there are %d rows" % df.shape[0])

    return df


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # DATA DOWNLOAD FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # #
def download_table_data_sqldb(sql_engine, table_name, start_time=None, end_time=None):
    # Use join to get the data from tables
    if start_time is None and end_time is None:
        query_string = "select * from " + table_name + " ORDER BY time;"
    elif start_time is None:
        query_string = (
            "select * from %s " % table_name +
            "WHERE time < '%s' ORDER BY time;" % (end_time)
        )
    elif end_time is None:
        query_string = (
            "select * from %s " % table_name +
            "WHERE time >= '%s' ORDER BY time;" % (start_time)
        )
    else:
        query_string = (
            "select * from %s WHERE time >= " % table_name +
            "'%s' AND time < '%s' ORDER BY time;" % (start_time, end_time)
        )

    df = pd.read_sql_query(query_string, sql_engine)

    # Drop a column called index
    if "index" in df.columns:
        df = df.drop(["index"], axis=1)

    # Now set the time to datetime format
    df["time"] = pd.to_datetime(df.time)

    return df


def download_data_for_specific_turbine_sqldb(
    sql_engine, table_name, turbid=1, start_time=None, end_time=None):
    column_names = get_column_names_sqldb(sql_engine, table_name)
    columns_turb = [s for s in column_names if '_'+str(turbid) in s]

    if len(columns_turb) < 1:
        raise IndexError('No turbine found with turbid = ' + str(turbid) + '.')
    columns_turb = ['time', *columns_turb]  # Add time column

    # Get the data from tables
    if start_time is None and end_time is None:
        query_string = (
            "select " + ','.join(['"'+c+'"' for c in columns_turb]) +
            " from " + table_name + " ORDER BY time;"
        )
    elif start_time is None:
        query_string = (
            "select " + ','.join(['"'+c+'"' for c in columns_turb]) +
            " from " + table_name + " WHERE time < '%s' ORDER BY time;"
            % (end_time)
        )
    elif end_time is None:
        query_string = (
            "select " + ','.join(['"'+c+'"' for c in columns_turb]) +
            " from " + table_name + " WHERE time >= '%s' ORDER BY time;"
            % (start_time)
        )
    else:
        query_string = (
            "select " + ','.join(['"'+c+'"' for c in columns_turb]) +
            " from " + table_name + " WHERE time >= '%s' " +
            "AND time < '%s' ORDER BY time;" % (start_time, end_time)
        )

    df = pd.read_sql_query(query_string, sql_engine)

    # Drop a column called index
    if "index" in df.columns:
        df = df.drop(["index"], axis=1)

    # Make sure time column is in datetime format
    df["time"] = pd.to_datetime(df.time)

    return df


def batch_download_data_from_sql(dbc, destination_path, table_name):
    print("Batch downloading data from table %s." % table_name)

    # Check if output directory exists, if not, create
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Check current start and end time of database
    db_end_time = get_first_time_entry_sqldb(dbc.engine, table_name)
    db_end_time = db_end_time + datetime.timedelta(minutes=10)

    # Check for past files and continue download or start a fresh download
    files_result = browse_downloaded_datafiles(destination_path,
                                               scada_table=table_name)
    print('A total of %d existing files found.' % len(files_result))

    # Next timestamp is going to be next first of the month
    latest_timestamp = get_timestamp_of_last_downloaded_datafile(files_result)
    if latest_timestamp is None:
        db_start_time = get_first_time_entry_sqldb(dbc.engine, table_name)
        db_start_time = db_start_time - datetime.timedelta(minutes=10)
        current_timestamp = db_start_time
    elif latest_timestamp.month == 12:
        current_timestamp = pd.to_datetime('%s-01-01' %
            str(latest_timestamp.year+1))
    else:
        current_timestamp = pd.to_datetime("%s-%s-01" %
            str(latest_timestamp.year), str(latest_timestamp.month+1))

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

        df = dbc.get_table_data_from_db_wide(
            table_name=table_name,
            start_time=current_timestamp,
            end_time=next_timestamp
            )

        # Drop NaN rows
        df = dfm.df_drop_nan_rows(df)

        # Save dataset as a .p file
        fout = os.path.join(destination_path, "%s_%s.ftr" %
            (current_timestamp.strftime("%Y-%m"), table_name))

        df = df.reset_index(drop=('time' in df.columns))
        df.to_feather(fout)

        print('Data for ' + table_name +
              ' saved to .ftr files for ' +
              str(current_timestamp.strftime("%B")) +
              ', ' + str(current_timestamp.year) + '.')

        current_timestamp = next_timestamp


def gui_sql_data_explorer(dbc):
    from datetime import timedelta as td
    import os
    import pickle

    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.backends.backend_tkagg as tkagg
    from matplotlib import dates

    class App:
        def __init__(self, master, dbc):

            # Create the options container
            frame_1 = tk.Frame(master, width=200, height=100)

            # Get basic database properties
            table_names = [c for c in dbc.sql_tables if '_filenames' not in c]
            min_time = [dbc.get_first_time_entry(table_name=t) for t in table_names]
            min_time = pd.to_datetime(np.min(min_time))
            max_time = [dbc.get_last_time_entry(table_name=t) for t in table_names]
            max_time = pd.to_datetime(np.max(max_time))

            # Add data table list box
            self.table_choices = table_names
            table_label = tk.Label(frame_1, text="Data Table")
            table_label.pack()
            self.table_listbox = tk.Listbox(
                frame_1, selectmode=tk.SINGLE, exportselection=False, height=4
            )
            self.table_listbox.pack()
            for tci in self.table_choices:
                self.table_listbox.insert(tk.END, tci)
            self.table_listbox.select_set(0)

            # Create a year widget
            year_label = tk.Label(frame_1, text="year")
            year_label.pack()
            var_year = tk.StringVar(root)
            years = tuple(range(min_time.year, max_time.year + 1))
            self.year_select = tk.Spinbox(frame_1, values=years,
                                        textvariable=var_year)
            var_year.set(years[0])
            self.year_select.pack()

            # Create a month widget
            month_label = tk.Label(frame_1, text="month")
            month_label.pack()
            var_month = tk.StringVar(root)
            self.month_select = tk.Spinbox(
                frame_1,
                values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                textvariable=var_month,
            )
            var_month.set(min_time.month)
            self.month_select.pack()

            # Create a day widget
            day_label = tk.Label(frame_1, text="day")
            day_label.pack()
            var_day = tk.StringVar(root)
            self.day_select = tk.Spinbox(frame_1, from_=0, to=32, textvariable=var_day)
            var_day.set(min_time.day)
            self.day_select.pack()

            # Add a connect gap
            self.connect_gap_var = tk.IntVar()
            self.connect_gap = tk.Checkbutton(
                frame_1, text="Connect gap between data", variable=self.connect_gap_var
            )
            self.connect_gap.pack()

            # Add a convert to eastern checkbox
            self.check_eastern_var = tk.IntVar()
            self.eastern_check = tk.Checkbutton(
                frame_1, text="Eastern", variable=self.check_eastern_var
            )
            self.eastern_check.pack()

            # Add a through-to-end checkbox
            self.tte_var = tk.IntVar()
            self.tte_check = tk.Checkbutton(
                frame_1, text="Import data through to end", variable=self.tte_var
            )
            self.tte_check.pack()

            # Add a range checkbox
            self.range_var = tk.IntVar()
            self.range_check = tk.Checkbutton(
                frame_1, text="Import range of days", variable=self.range_var
            )
            self.range_check.pack()

            # Create a num days widget
            num_day_label = tk.Label(frame_1, text="num day")
            num_day_label.pack()
            var_num_day = tk.StringVar(root)
            self.num_day_select = tk.Spinbox(
                frame_1, from_=0, to=90, textvariable=var_num_day
            )
            var_num_day.set("8")
            self.num_day_select.pack()

            # Add a load data button
            self.button_load = tk.Button(frame_1, text="Load Data", command=self.load_data)
            self.button_load.pack()  # side="left")

            # # Add a load pickle button
            # self.button_load_pickle = tk.Button(
            #     frame_1, text="Load Pickle", command=self.load_pickle
            # )
            # self.button_load_pickle.pack()  # side="left")

            # # Add a save data button
            # self.button_save = tk.Button(frame_1, text="Save Data", command=self.save_data)
            # self.button_save.pack()  # side="left")

            # Add a average data button
            self.button_average = tk.Button(
                frame_1, text="Average Data", command=self.average_data
            )
            self.button_average.pack()  # side="left")

            # Add channel 1 list box
            p1_label = tk.Label(frame_1, text="plot 1")
            p1_label.pack()
            self.c1_listbox = tk.Listbox(
                frame_1, selectmode=tk.EXTENDED, exportselection=False, height=5
            )
            self.c1_listbox.pack()
            self.c1_listbox.bind("<<ListboxSelect>>", self.c1_select)

            # Add channel 2 list box
            p2_label = tk.Label(frame_1, text="plot 2")
            p2_label.pack()
            self.c2_listbox = tk.Listbox(
                frame_1, selectmode=tk.EXTENDED, exportselection=False, height=5
            )
            self.c2_listbox.pack()
            self.c2_listbox.bind("<<ListboxSelect>>", self.c2_select)

            # Add channel 3 list box
            p3_label = tk.Label(frame_1, text="plot 3")
            p3_label.pack()
            self.c3_listbox = tk.Listbox(
                frame_1, selectmode=tk.EXTENDED, exportselection=False, height=5
            )
            self.c3_listbox.pack()
            self.c3_listbox.bind("<<ListboxSelect>>", self.c3_select)

            # Add channel 4 list box
            p4_label = tk.Label(frame_1, text="plot 4")
            p4_label.pack()
            self.c4_listbox = tk.Listbox(
                frame_1, selectmode=tk.EXTENDED, exportselection=False, height=5
            )
            self.c4_listbox.pack()
            self.c4_listbox.bind("<<ListboxSelect>>", self.c4_select)

            # Pack the first frame
            frame_1.pack(fill=None, expand=0, side="left")

            # Create the plotting frame
            frame_2 = tk.Frame(master, width=1000, height=500)

            # Init the plotting area
            self.fig = Figure()
            self.ax_1 = self.fig.add_subplot(411)
            self.ax_2 = self.fig.add_subplot(412, sharex=self.ax_1)
            self.ax_3 = self.fig.add_subplot(413, sharex=self.ax_1)
            self.ax_4 = self.fig.add_subplot(414, sharex=self.ax_1)

            self.canvas = FigureCanvasTkAgg(self.fig, master=frame_2)
            tkagg.NavigationToolbar2Tk(self.canvas, master)
            # tkagg.NavigationToolbar2TkAgg(self.canvas, master)
            # self.canvas.show()
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            frame_2.pack(fill="both", expand=1, side="right")
            self.master = master

            # Set up the database connection
            self.dbc = dbc

        # def save_data(self):

        #     # Get x_range
        #     ax_dates = self.ax_1.get_xlim()
        #     ax_dates = [pd.to_datetime(dates.num2date(d), utc=True) for d in ax_dates]

        #     # Get data to save
        #     time = pd.to_datetime(self.df.time, utc=True)
        #     # df_save = self.df[(self.df.time >= ax_dates[0]) & (self.df.time <= ax_dates[1]) ]
        #     df_save = self.df[(time >= ax_dates[0]) & (time <= ax_dates[1])]

        #     # Get a filename
        #     home = os.path.expanduser("~")
        #     save_folder = os.path.join(home, "Desktop/save_folder")

        #     # Make if doesn't exist
        #     if not os.path.exists(save_folder):
        #         os.makedirs(save_folder)

        #     # Create filename
        #     d1 = ax_dates[0]
        #     filename = os.path.join(
        #         save_folder,
        #         "data_%d_%02d_%02d_%02d.p" % (d1.year, d1.month, d1.day, d1.hour),
        #     )

        #     print("Saving...%s" % filename)
        #     df_save.to_pickle(os.path.join(save_folder, filename))
        #     print("...Done")

        def average_data(self):

            # Get x_range
            ax_dates = self.ax_1.get_xlim()
            ax_dates = [pd.to_datetime(dates.num2date(d)) for d in ax_dates]

            # Get data to save
            df_save = self.df[(self.df.time >= ax_dates[0]) & (self.df.time <= ax_dates[1])]

            # C1 particulars
            for listbox in [
                self.c1_listbox,
                self.c2_listbox,
                self.c3_listbox,
                self.c4_listbox,
            ]:
                indices = listbox.curselection()
                channels = [self.df.columns[idx] for idx in indices]

                # Show the averages
                for c in channels:
                    print(c, df_save[c].mean())

        # def load_pickle(self):
        #     pickle_filename = tk.filedialog.askopenfilename(
        #         initialdir="/", title="Select file"
        #     )  # ,filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        #     print(pickle_filename)
        #     self.df = pickle.load(open(pickle_filename, "rb"))

        #     # If requested, reconfig data so-as not to connect gap
        #     if self.connect_gap_var.get() == 0:
        #         print("Putting in nans to avoid gap connect")
        #         # self.df = ut.convert_to_eastern(self.df)
        #         self.df["next_times"] = self.df["time"] + td(seconds=60)
        #         next_times = self.df.next_times[~self.df.next_times.isin(self.df["time"])]
        #         current_times = self.df["time"]
        #         full_times = current_times.append(next_times)

        #         self.df = self.df.set_index("time").reindex(full_times)
        #         self.df.index.name = "time"
        #         self.df = self.df.reset_index().sort_values(by="time")

        #     # Update the channel list box with the available channels
        #     self.c1_listbox.delete(0, tk.END)
        #     for c in self.df.columns:
        #         self.c1_listbox.insert(tk.END, c)

        #     self.c2_listbox.delete(0, tk.END)
        #     for c in self.df.columns:
        #         self.c2_listbox.insert(tk.END, c)

        #     self.c3_listbox.delete(0, tk.END)
        #     for c in self.df.columns:
        #         self.c3_listbox.insert(tk.END, c)

        #     self.c4_listbox.delete(0, tk.END)
        #     for c in self.df.columns:
        #         self.c4_listbox.insert(tk.END, c)

        #     # Clear all axes
        #     for ax in [self.ax_1, self.ax_2, self.ax_3, self.ax_4]:
        #         ax.clear()

        def load_data(self):

            year = int(self.year_select.get())
            month = int(self.month_select.get())
            day = int(self.day_select.get())
            start_date = "%04d-%02d-%02d" % (year, month, day)
            num_days = int(self.num_day_select.get())

            # Determine which table to use
            table_select = self.table_choices[self.table_listbox.curselection()[0]]

            # Determine start and end time
            start_time = pd.to_datetime(start_date)
            if self.tte_var.get() == 1:  # Through the end
                end_time = self.dbc.get_last_time_entry()
                # end_date = '%d-%02d-%02d' % (end_time.year,end_time.month,end_time.day)
                # end_time = None  # Set to None for read functions
            elif self.range_var.get() == 1:  # Plot a range
                end_time = start_time + td(days=num_days)
                # end_date = '%d-%02d-%02d' % (end_time.year,end_time.month,end_time.day)
            else:
                end_time = start_time + td(days=1)
                # end_date = '%d-%02d-%02d' % (end_time.year,end_time.month,end_time.day)

            print("Importing %s from %s to %s" % (table_select, start_time, end_time))
            self.df = self.dbc.get_table_data_from_db_wide(
                table_select, start_time, end_time
            )
            print("...Imported data successfully.")

            # If requested, reconfig data so-as not to connect gap
            second_gap = 600  # 10 minutes between data
            if self.connect_gap_var.get() == 0:
                print("Putting in nans to avoid gap connect")
                # self.df = ut.convert_to_eastern(self.df)
                self.df["next_times"] = self.df["time"] + td(
                    seconds=second_gap
                )
                next_times = self.df.next_times[~self.df.next_times.isin(self.df["time"])]
                current_times = self.df["time"]
                full_times = current_times.append(next_times)

                self.df = self.df.set_index("time").reindex(full_times)
                self.df.index.name = "time"
                self.df = self.df.reset_index().sort_values(by="time")

            # If requested, convert to eastern
            if self.check_eastern_var.get() == 1:
                print("Converting to eastern time")
                self.df = ut.convert_to_eastern(self.df)

            # Update the channel list box with the available channels
            self.c1_listbox.delete(0, tk.END)
            for c in self.df.columns:
                self.c1_listbox.insert(tk.END, c)

            self.c2_listbox.delete(0, tk.END)
            for c in self.df.columns:
                self.c2_listbox.insert(tk.END, c)

            self.c3_listbox.delete(0, tk.END)
            for c in self.df.columns:
                self.c3_listbox.insert(tk.END, c)

            self.c4_listbox.delete(0, tk.END)
            for c in self.df.columns:
                self.c4_listbox.insert(tk.END, c)

            # Clear all axes
            for ax in [self.ax_1, self.ax_2, self.ax_3, self.ax_4]:
                ax.clear()

        def c1_select(self, evt):

            # C1 particulars
            indices = self.c1_listbox.curselection()
            ax = self.ax_1

            # Generic
            channels = [self.df.columns[idx] for idx in indices]

            # Update the tool bar
            self.canvas.toolbar.update()

            # Update axes plot
            ax.clear()
            for c in channels:
                ax.plot(self.df.time, np.array(self.df[c].values), label=c)
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
            # self.canvas.show()

        def c2_select(self, evt):

            # C2 particulars
            indices = self.c2_listbox.curselection()
            ax = self.ax_2

            # Generic
            channels = [self.df.columns[idx] for idx in indices]

            # Update the tool bar
            self.canvas.toolbar.update()

            # Update axes plot
            ax.clear()
            for c in channels:
                ax.plot(self.df.time, self.df[c], label=c)
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
            # self.canvas.show()

        def c3_select(self, evt):

            # C3 particulars
            indices = self.c3_listbox.curselection()
            ax = self.ax_3

            # Generic
            channels = [self.df.columns[idx] for idx in indices]

            # Update the tool bar
            self.canvas.toolbar.update()

            # Update axes plot
            ax.clear()
            for c in channels:
                ax.plot(self.df.time, self.df[c], label=c)
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
            # self.canvas.show()

        def c4_select(self, evt):

            # C4 particulars
            indices = self.c4_listbox.curselection()
            ax = self.ax_4

            # Generic
            channels = [self.df.columns[idx] for idx in indices]

            # Update the tool bar
            self.canvas.toolbar.update()

            # Update axes plot
            ax.clear()
            for c in channels:
                ax.plot(self.df.time, self.df[c], label=c)
            ax.legend()
            ax.grid(True)

            self.canvas.draw()
            # self.canvas.show()

    root = tk.Tk()
    app = App(master=root, dbc=dbc)
    root.mainloop()