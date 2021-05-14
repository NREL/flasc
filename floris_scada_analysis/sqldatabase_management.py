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
        wide_format ([list of bools]): Definition whether table is WIDE.
        turbine_range ([list]): List of turbine names.
        repository_name ([str]): Name of Python repository
        settingsfile_name ([str]): Name of settings file for Python/SQL
        fout ([str]): Path defining desired output directory for models.py

    Raises:
        Exception: Will raise an error when models.py already exists,
        and when additionally models.py.old is already taken.
    """

    # Initialize default template
    modelspy_str = (
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
    )

    # Populate recreate_by_type(...) function
    for table_name in table_names:
        modelspy_str = modelspy_str + (
'''
    if table_name == "''' + table_name + '''":
        DeclarativeBase.metadata.drop_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
        DeclarativeBase.metadata.create_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
'''
        )


    # Add individual classes for table_name_filenames and table_names
    for iii in range(len(table_names)):
        table_name = table_names[iii]

        modelspy_str = modelspy_str + ('''
class ''' + table_name + '''_filenames_class(DeclarativeBase):
    """Sqlalchemy filenames model"""
    __tablename__ = "''' + table_name + '''_filenames"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column('filename', String, nullable=False, primary_key=True, unique=True)

class ''' + table_name + '''_class(DeclarativeBase):
    """Sqlalchemy data model"""
    __tablename__ = "''' + table_name + '''"

    index = Column('index', Integer, primary_key=True, autoincrement=True)'''
        )

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
        if 'turbid' in database_name.values and not wide_format[iii]:
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
        if wide_format[iii]:
            remove_fields.append('turbid')
            db_name_new = []
            vb_type_new = []
            for di in range(len(database_name)):
                if database_name[di] not in remove_fields:
                    for ti in turbine_range:
                        if (table_upsampling[iii] and (variable_type[di] == 'REAL' or variable_type[di] == 'Integer')):
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_mean")
                            if variable_type[di] == 'Integer':
                                vb_type_new.append('REAL')
                            else:
                                vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_median")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_std")
                            if variable_type[di] == 'Integer':
                                vb_type_new.append('REAL')
                            else:
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
    text_file.write(modelspy_str)
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
    if df_time.shape[0] <= 0:
        return None
    else:
        return df_time['time'][0]

def get_last_time_entry_sqldb(sql_engine, table_name):
    # Get the list of scada filenames
    query_string = 'SELECT time FROM ' + table_name + ' ORDER BY time desc LIMIT 1'
    df_time = pd.read_sql_query(query_string, sql_engine)
    if df_time.shape[0] <= 0:
        return None
    else:
        return df_time['time'][0]

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
        fl = glob.glob(fpath)
        if len(fl) <= 0:
            print('No files found in directory %s.' % fpath)
        else:
            files.extend(fl)

    # Figure out which files have already been uploaded to sql db
    query_string = "select * from " + filenames_table + ";"
    df = pd.read_sql_query(query_string, sqldb_engine)

    # # Check for the files not in the database
    files = [f for f in files
             if os.path.basename(f) not in df['filename'].values]

    # Sort the file list according to ascending name
    files = sorted(files, reverse=False)

    return files


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # DATA UPLOAD FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # #
def upload_df_to_sqldb(df, files, sql_engine, table_name,
                       df_chunk_size=10000, sql_chunk_size=10000,
                       leave_out_last_filename=False):
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

    # Leave out last filename if overlap necessary
    if leave_out_last_filename:
        files = files[0:-1]

    for fi in files:
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
    if not 'time' in df.columns:
        df = df.reset_index(drop=False)

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
    db_end_time = get_last_time_entry_sqldb(dbc.engine, table_name)
    db_end_time = db_end_time + datetime.timedelta(minutes=10)

    # Check for past files and continue download or start a fresh download
    files_result = browse_downloaded_datafiles(destination_path,
                                               table_name=table_name)
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
        current_timestamp = pd.to_datetime("%s-%s-01" % (
            str(latest_timestamp.year), str(latest_timestamp.month+1)))

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

    import tkinter as tk
    import tkcalendar as tkcal
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.backends.backend_tkagg as tkagg

    class App:
        def __init__(self, master, dbc):

            # Create the options container
            frame_1 = tk.Frame(master)
            self.master = master

            # Get basic database properties
            self.df = pd.DataFrame()
            table_names = [c for c in dbc.sql_tables if '_filenames' not in c]
            min_table_dates = [dbc.get_first_time_entry(table_name=t)
                               for t in table_names]
            max_table_dates = [dbc.get_last_time_entry(table_name=t)
                               for t in table_names]
            max_nochars_tbname = 4 + int(np.max([len(c) for c in table_names]))

            # Add data table list box
            self.table_choices = table_names
            table_label = tk.Label(frame_1, text="Data Table")
            table_label.pack()
            self.table_listbox = tk.Listbox(
                frame_1, selectmode=tk.EXTENDED,
                exportselection=False, height=4,
                width=max_nochars_tbname
            )
            self.table_listbox.pack()
            for ii, tci in enumerate(self.table_choices):
                id_letter = '[' + chr(97 + ii).upper() + ']'
                self.table_listbox.insert(tk.END, id_letter + ' ' + tci)
            # self.table_listbox.select_set(0)

            # Create a start_date widget
            start_date_label = tk.Label(frame_1, text='Data import: start date')
            start_date_label.pack()
            self.cal_start_date = tkcal.DateEntry(
                frame_1, date_pattern='MM/dd/yyyy',state='disabled')
            self.cal_start_date.pack()

            end_date_label = tk.Label(frame_1, text='Data import: end date')
            end_date_label.pack()
            self.cal_end_date = tkcal.DateEntry(
                frame_1, date_pattern='MM/dd/yyyy', state='disabled')
            self.cal_end_date.pack()


            # Change min and max time depending on table(s) selected
            def update_table_selection(event):
                # Get selected tables
                tids = [i for i in self.table_listbox.curselection()]

                # Determine and update min/max dates
                if len(tids) <= 0:
                    self.cal_start_date.config({'state': 'disabled'})
                    self.cal_end_date.config({'state': 'disabled'})
                else:
                    min_time = [min_table_dates[i] for i in tids]
                    max_time = [max_table_dates[i] for i in tids]
                    min_time = pd.to_datetime(np.min(min_time))
                    max_time = pd.to_datetime(np.max(max_time))
                    mean_time = min_time + (max_time-min_time)/2.
                    cal_dict = {'state': 'normal',
                                'mindate': min_time,
                                'maxdate': max_time}
                    self.cal_start_date.config(cal_dict)
                    self.cal_end_date.config(cal_dict)
                    self.cal_start_date.set_date(mean_time)
                    self.cal_end_date.set_date(mean_time)

            self.table_listbox.bind('<<ListboxSelect>>',
                                    update_table_selection)

            # Add commands to change end_date if start_date > end_date
            def callback_change_enddate(event):
                start_date = self.cal_start_date.get_date()
                end_date = self.cal_end_date.get_date()
                if end_date <= start_date:
                    self.cal_end_date.set_date(
                        date=start_date + td(days=1)
                    )
            def callback_change_startdate(event):
                start_date = self.cal_start_date.get_date()
                end_date = self.cal_end_date.get_date()
                if end_date <= start_date:
                    self.cal_start_date.set_date(
                        date=end_date - td(days=1)
                    )

            self.cal_start_date.bind("<<DateEntrySelected>>",
                                     callback_change_enddate)
            self.cal_end_date.bind("<<DateEntrySelected>>",
                                   callback_change_startdate)

            # Add a load data button
            self.button_load = tk.Button(frame_1, text="Download data",
                                         command=self.load_data)
            self.button_load.pack(pady=10)  # side="left")

            # Add button to remove/add plots
            self.channel_add_button = tk.Button(frame_1,
                                                text='Add plotting channel',
                                                command=self.channel_add)
            self.channel_add_button.pack()
            self.channel_rem_button = tk.Button(frame_1,
                                                text='Remove plotting channel',
                                                command=self.channel_rem)
            self.channel_rem_button.pack()

            # Add (placeholder) channels
            N_channels_max = 10
            self.N_channels = 1
            self.N_channels_max = N_channels_max
            self.channel_label = [[] for _ in range(N_channels_max)]
            self.channel_listbox = [[] for _ in range(N_channels_max)]
            self.channel_selection = [[] for _ in range(N_channels_max)]
            for i in range(N_channels_max):
                self.channel_label[i] = tk.Label(frame_1, text="plot %d" % i)
                self.channel_listbox[i] = tk.Listbox(
                    frame_1, selectmode=tk.EXTENDED,
                    exportselection=False, height=5,
                    width=max_nochars_tbname,
                    state='normal'
                )
                def mapper_func(evt):
                    ci = int(str(evt.widget).replace('.!frame.!listbox', ''))-2
                    self.ci_select(channel_no=ci)
                self.channel_listbox[i].bind("<<ListboxSelect>>", mapper_func)

                if i == 0:
                    self.channel_label[i].pack()
                    self.channel_listbox[i].pack(fill=tk.BOTH, expand=True)

            # Create the plotting frame
            self.frame_2 = tk.Frame(master, width=20, height=500)

            # Pack the first frame
            frame_1.pack(fill=tk.BOTH, expand=False, side="left", padx=5)

            self.create_figures()
            self.master = master

            # Set up the database connection
            self.dbc = dbc

        def channel_add(self):
            if self.N_channels < self.N_channels_max:
                ci = self.N_channels  # New channel
                self.channel_listbox[ci].config({'state': 'normal'})
                self.channel_label[ci].pack()
                self.channel_listbox[ci].pack(fill=tk.BOTH, expand=True)
                self.N_channels = self.N_channels + 1
                self.create_figures()

        def channel_rem(self):
            if self.N_channels > 1:
                ci = self.N_channels - 1  # Last existing channel
                self.channel_listbox[ci].config({'state': 'disabled'})
                self.channel_listbox[ci].pack_forget()
                self.channel_label[ci].pack_forget()
                self.N_channels = self.N_channels - 1
                self.create_figures()

        def load_data(self):
            start_time = self.cal_start_date.get_date()
            end_time = self.cal_end_date.get_date() + datetime.timedelta(days=1)

            # Load specified table(s)
            df_array = []
            table_choices = self.table_choices
            tables_selected = self.table_listbox.curselection()
            for ii in range(len(tables_selected)):
                table_select = table_choices[tables_selected[ii]]

                print("Importing %s from %s to %s"
                      % (table_select, start_time, end_time))
                df = self.dbc.get_table_data_from_db_wide(
                    table_select, start_time, end_time)
                df = df.set_index('time', drop=True)

                if df.shape[0] <= 0:
                    print('No data found in this timerange for table %s'
                          % table_select)
                else:
                    print("...Imported data successfully.")

                    old_col_names = list(df.columns)
                    new_col_names = [chr(97 + tables_selected[ii]).upper()
                                     + '_%s'  % c for c in df.columns]
                    col_mapping = dict(zip(old_col_names,new_col_names))
                    df = df.rename(columns=col_mapping)
                    df_array.append(df)

            # Merge dataframes
            self.df = pd.concat(df_array, axis=1).reset_index(drop=False)

            self.update_channel_cols()
            self.create_figures()
            # # Clear all axes
            # for ax in self.axes:
            #     ax.clear()

            # Update frame width
            nochars_cols = [len(c) for c in self.df.columns]
            max_col_width = np.max(nochars_cols)
            max_tbn_width = 4 + np.max([len(c) for c in table_choices])
            frame_width = int(np.max([max_col_width, max_tbn_width]))
            self.channel_listbox[0].config({'width': frame_width})

        def update_channel_cols(self):
            cols = self.df.columns

            # Update the channel list box with the available channels
            for i in range(self.N_channels_max):
                self.channel_listbox[i].delete(0, tk.END)
                if len(self.table_listbox.curselection()) > 0:
                    for c in cols:
                        self.channel_listbox[i].insert(tk.END, c)

            # Remove any no-longer-existent channels to plot
            for i in range(self.N_channels_max):
                for ii, cn in enumerate(self.channel_selection[i]):
                    if cn not in cols:
                        self.channel_selection[i].pop(ii)
            
            for i in range(self.N_channels):
                for cn in self.channel_selection[i]:
                    id = [i for i in range(len(cols)) if cn==cols[i]][0]
                    self.channel_listbox[i].selection_set(id)


        def update_plot(self, channel_no):
            # Only update if we have anything to plot...
            if ((self.df.shape[0] > 1) &
                any([len(i) > 0 for i in self.channel_selection])):
                # Update the tool bar
                # self.canvas.toolbar.update()

                # Update axis plot
                ax = self.axes[channel_no]
                ax.clear()
                for c in self.channel_selection[channel_no]:
                    ax.plot(self.df.time, np.array(self.df[c].values), label=c)
                ax.legend()
                ax.grid(True)

                self.canvas.draw()
            return None

        def create_figures(self):
            try:
                self.toolbar.destroy()
                self.frame_2.destroy()
            except:
                print('No preexisting figures found.')

            self.frame_2 = tk.Frame(self.master, width=20, height=500)
            self.fig = Figure()
            self.axes = [[] for _ in range(self.N_channels)]
            self.axes[0] = self.fig.add_subplot(self.N_channels, 1, 1)
            self.update_plot(channel_no=0)
            for ii in range(1, self.N_channels):
                self.axes[ii] = self.fig.add_subplot(
                    self.N_channels, 1, ii+1, sharex=self.axes[0])
                self.update_plot(channel_no=ii)

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_2)
            self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas, self.master)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
            self.frame_2.pack(fill="both", expand=True, side="right")
            # self.update_channel_cols()  # Reset column selection

        def ci_select(self, channel_no, evt=None):
            indices = self.channel_listbox[channel_no].curselection()
            channels = [self.df.columns[idx] for idx in indices]
            self.channel_selection[channel_no] = channels
            self.update_plot(channel_no=channel_no)


    root = tk.Tk()
    app = App(master=root, dbc=dbc)
    root.mainloop()