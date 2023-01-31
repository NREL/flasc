# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import numpy as np
import pandas as pd
from time import perf_counter as timerpc

import datetime
from datetime import timedelta as td
import tkinter as tk
import tkcalendar as tkcal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.backends.backend_tkagg as tkagg

import sqlalchemy as sqlalch


class sql_database_manager:

    # Private methods

    def __init__(self, db_name, host, port, username, password):
        self.db_name = db_name
        self.host = host
        self.port = port
        self.username = username
        self._create_sql_engine(password)

    def _create_sql_engine(self, password):
        print("Initializing SQL connection (engine)")
        dr = "postgresql"
        name = self.db_name
        usn = self.username
        address = "%s:%d" % (self.host, self.port)
        self.engine = sqlalch.create_engine(
            url="%s://%s:%s@%s/%s" % (dr, usn, password, address, name)
        )
        self.print_properties()

    def _get_table_names(self):
        return self.engine.table_names()

    def _get_column_names(self, table_name):
        df = pd.read_sql_query(
            "SELECT * FROM " + table_name + " WHERE false;", self.engine
        )
        return list(df.columns)

    def _get_first_time_entry(self, table_name):
        tn = table_name
        column_names = self._get_column_names(tn)
        if 'time' in column_names:
            df_time = pd.read_sql_query(
                sql="SELECT time FROM %s ORDER BY time asc LIMIT 1" % tn,
                con=self.engine
            )
            if df_time.shape[0] > 0:
                return df_time["time"][0]

        return None

    def _get_last_time_entry(self, table_name):
        tn = table_name
        column_names = self._get_column_names(tn)
        if 'time' in column_names:
            df_time = pd.read_sql_query(
                sql="SELECT time FROM %s ORDER BY time desc LIMIT 1" % tn,
                con=self.engine
            )
            if df_time.shape[0] > 0:
                return df_time["time"][0]

        return None

    # General info functions from data
    def print_properties(self):
        table_names = self._get_table_names()
        print("")
        print("Connected to: %s." % str(self.engine.url.host))
        print("Existent tables: ", table_names)
        for tn in table_names:
            t0 = self._get_first_time_entry(tn)
            t1 = self._get_last_time_entry(tn)
            cols = self._get_column_names(tn)
            print("  Table name: %s." % tn)
            print("    Earliest data entry: %s." % str(t0))
            print("    Latest data entry: %s." % str(t1))
            print("    N.o. columns: %d." % len(cols))
        print("")

    def launch_gui(self, turbine_names=None, sort_columns=False):
        root = tk.Tk()

        sql_db_explorer_gui(master=root, dbc=self, turbine_names=turbine_names, sort_columns=sort_columns)
        root.mainloop()

    def get_column_names(self, table_name):
        return self._get_column_names(table_name)

    def batch_get_data(self, table_name, columns=None, start_time=None,
                       end_time=None, fn_out=None, no_rows_per_file=10000):
        if fn_out is None:
            fn_out = table_name + ".ftr"
        if not (fn_out[-4::] == ".ftr"):
            fn_out = fn_out + ".ftr"

        # Ensure 'time' in database
        column_names = self._get_column_names(table_name=table_name)
        if 'time' not in column_names:
            raise KeyError("Cannot find 'time' column in database table.")

        # Get time column from database
        time_in_db = self.get_data(table_name=table_name, columns=['time'],
                                   start_time=start_time, end_time=end_time)
        time_in_db = list(time_in_db['time'])

        splits = np.arange(0, len(time_in_db) - 1, no_rows_per_file, dtype=int)
        splits = np.append(splits, len(time_in_db) - 1)
        splits = np.unique(splits)

        for ii in range(len(splits) - 1):
            print("Downloading subset %d out of %d." % (ii, len(splits) - 1))
            df = self.get_data(
                table_name=table_name,
                columns=columns,
                start_time=time_in_db[splits[ii]],
                end_time=time_in_db[splits[ii+1]]
            )
            fn_out_ii = fn_out + '.%d' % ii
            print("Saving file to %s." % fn_out_ii)
            df.to_feather(fn_out_ii)

    def get_data(
        self, table_name, columns=None, start_time=None, end_time=None
    ):
        # Get the data from tables
        if columns is None:
            query_string = "select * from " + table_name
        else:
            columns_string = ",".join(['"' + c + '"' for c in columns])
            query_string = "select " + columns_string + " from " + table_name

        if start_time is not None:
            query_string += " WHERE time >= '" + str(start_time) + "'"

        if (start_time is not None) and (end_time is not None):
            query_string += " AND time < '" + str(end_time) + "'"
        elif (start_time is None) and (end_time is not None):
            query_string += " WHERE time < '" + str(end_time) + "'"

        query_string += " ORDER BY time;"
        df = pd.read_sql_query(query_string, self.engine)

        # Drop a column called index
        if "index" in df.columns:
            df = df.drop(["index"], axis=1)

        # Make sure time column is in datetime format
        df["time"] = pd.to_datetime(df.time)

        return df

    def send_data(
        self,
        table_name,
        df,
        if_exists="append_new",
        unique_cols=["time"],
        df_chunk_size=2000,
        sql_chunk_size=50
    ):
        table_name = table_name.lower()
        table_names = [t.lower() for t in self._get_table_names()]

        if (if_exists == "append"):
            print("Warning: risk of adding duplicate rows using 'append'.")
            print("You are suggested to use 'append_new' instead.")

        if (if_exists == "append_new") and (table_name in table_names):
            if len(unique_cols) > 1:
                raise NotImplementedError("Not yet implemented.")

            col = unique_cols[0]
            idx_in_db = self.get_data(table_name=table_name, columns=[col])[
                col
            ]

            # Check if values in SQL database are unique
            if not idx_in_db.is_unique:
                raise IndexError(
                    "Column '%s' is not unique in the SQL database." % col
                )

            idx_in_df = set(df[col])
            idx_in_db = set(idx_in_db)
            idx_to_add = np.sort(list(idx_in_df - idx_in_db))
            print(
                "{:d} entries already exist in SQL database.".format(
                    len(idx_in_df) - len(idx_to_add)
                )
            )

            print("Adding {:d} new entries...".format(len(idx_to_add)))
            df_subset = df.set_index('time').loc[idx_to_add].reset_index(
                drop=False)

        else:
            df_subset = df

        if (if_exists == "append_new"):
            if_exists = "append"

        # Upload data
        N = df_subset.shape[0]
        if N < 1:
            print("Skipping data upload. Dataframe is empty.")
        else:
            print("Attempting to insert %d rows into table '%s'."
                % (df_subset.shape[0], table_name))
            df_chunks_id = np.arange(0, df_subset.shape[0], df_chunk_size)
            df_chunks_id = np.append(df_chunks_id, df_subset.shape[0])
            df_chunks_id = np.unique(df_chunks_id)

            time_start_total = timerpc()
            for i in range(len(df_chunks_id)-1):
                Nl = df_chunks_id[i]
                Nu = df_chunks_id[i+1]
                print("Inserting rows %d to %d." % (Nl, Nu))
                time_start_i = timerpc()
                df_sub = df_subset[Nl:Nu]
                df_sub.to_sql(
                    table_name,
                    self.engine,
                    if_exists=if_exists,
                    index=False,
                    method="multi",
                    chunksize=sql_chunk_size,
                )
                time_i = timerpc() - time_start_i
                total_time = timerpc() - time_start_total
                est_time_left = (total_time / Nu) * (N - Nu)
                eta = datetime.datetime.now() + td(seconds=est_time_left)
                eta = eta.strftime("%a, %d %b %Y %H:%M:%S")
                print("Data insertion took %.1f s. ETA: %s." % (time_i, eta))


class sql_db_explorer_gui:
    def __init__(self, master, dbc, turbine_names = None, sort_columns=False):

        # Create the options container
        frame_1 = tk.Frame(master)
        self.master = master

        # Get basic database properties
        self.df = pd.DataFrame()
        table_names = dbc._get_table_names()
        min_table_dates = [
            dbc._get_first_time_entry(table_name=t) for t in table_names
        ]
        max_table_dates = [
            dbc._get_last_time_entry(table_name=t) for t in table_names
        ]
        max_nochars_tbname = 4 + int(np.max([len(c) for c in table_names]))

        # Add data table list box
        self.table_choices = table_names
        table_label = tk.Label(frame_1, text="Data Table")
        table_label.pack()
        self.table_listbox = tk.Listbox(
            frame_1,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=4,
            width=max_nochars_tbname,
        )
        self.table_listbox.pack()
        for ii, tci in enumerate(self.table_choices):
            id_letter = "[" + chr(97 + ii).upper() + "]"
            self.table_listbox.insert(tk.END, id_letter + " " + tci)
        # self.table_listbox.select_set(0)

        # Create a start_date widget
        start_date_label = tk.Label(frame_1, text="Data import: start date")
        start_date_label.pack()
        self.cal_start_date = tkcal.DateEntry(
            frame_1, date_pattern="MM/dd/yyyy", state="disabled"
        )
        self.cal_start_date.pack()

        end_date_label = tk.Label(frame_1, text="Data import: end date")
        end_date_label.pack()
        self.cal_end_date = tkcal.DateEntry(
            frame_1, date_pattern="MM/dd/yyyy", state="disabled"
        )
        self.cal_end_date.pack()

        # Change min and max time depending on table(s) selected
        def update_table_selection(event):
            # Get selected tables
            tids = [i for i in self.table_listbox.curselection()]

            # Determine and update min/max dates
            if len(tids) <= 0:
                self.cal_start_date.config({"state": "disabled"})
                self.cal_end_date.config({"state": "disabled"})
            else:
                min_time = [min_table_dates[i] for i in tids]
                max_time = [max_table_dates[i] for i in tids]
                min_time = pd.to_datetime(np.min(min_time))
                max_time = pd.to_datetime(np.max(max_time))
                mean_time = min_time + (max_time - min_time) / 2.0
                cal_dict = {
                    "state": "normal",
                    "mindate": min_time,
                    "maxdate": max_time,
                }
                self.cal_start_date.config(cal_dict)
                self.cal_end_date.config(cal_dict)
                self.cal_start_date.set_date(mean_time)
                self.cal_end_date.set_date(mean_time)

        self.table_listbox.bind("<<ListboxSelect>>", update_table_selection)

        # Add commands to change end_date if start_date > end_date
        def callback_change_enddate(event):
            start_date = self.cal_start_date.get_date()
            end_date = self.cal_end_date.get_date()
            if end_date <= start_date:
                self.cal_end_date.set_date(date=start_date + td(days=1))

        def callback_change_startdate(event):
            start_date = self.cal_start_date.get_date()
            end_date = self.cal_end_date.get_date()
            if end_date <= start_date:
                self.cal_start_date.set_date(date=end_date - td(days=1))

        self.cal_start_date.bind(
            "<<DateEntrySelected>>", callback_change_enddate
        )
        self.cal_end_date.bind(
            "<<DateEntrySelected>>", callback_change_startdate
        )

        # Add a load data button
        self.button_load = tk.Button(
            frame_1, text="Download data", command=self.load_data
        )
        self.button_load.pack(pady=10)  # side="left")

        # Add button to remove/add plots
        self.channel_add_button = tk.Button(
            frame_1, text="Add plotting channel", command=self.channel_add
        )
        self.channel_add_button.pack()
        self.channel_rem_button = tk.Button(
            frame_1, text="Remove plotting channel", command=self.channel_rem
        )
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
                frame_1,
                selectmode=tk.EXTENDED,
                exportselection=False,
                height=5,
                width=max_nochars_tbname,
                state="normal",
            )

            def mapper_func(evt):
                ci = int(str(evt.widget).replace(".!frame.!listbox", "")) - 2
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

        # Save the turbine names
        self.turbine_names = turbine_names

        # Save the sort columns
        self.sort_columns = sort_columns

    def channel_add(self):
        if self.N_channels < self.N_channels_max:
            ci = self.N_channels  # New channel
            self.channel_listbox[ci].config({"state": "normal"})
            self.channel_label[ci].pack()
            self.channel_listbox[ci].pack(fill=tk.BOTH, expand=True)
            self.N_channels = self.N_channels + 1
            self.create_figures()

    def channel_rem(self):
        if self.N_channels > 1:
            ci = self.N_channels - 1  # Last existing channel
            self.channel_listbox[ci].config({"state": "disabled"})
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

            print(
                "Importing %s from %s to %s"
                % (table_select, start_time, end_time)
            )
            df = self.dbc.get_data(
                table_name=table_select,
                start_time=start_time,
                end_time=end_time,
            )
            df = df.set_index("time", drop=True)

            if df.shape[0] <= 0:
                print(
                    "No data found in this timerange for table %s"
                    % table_select
                )
            else:
                print("...Imported data successfully.")

                old_col_names = list(df.columns)
                new_col_names = [
                    chr(97 + tables_selected[ii]).upper() + "_%s" % c
                    for c in df.columns
                ]
                col_mapping = dict(zip(old_col_names, new_col_names))
                df = df.rename(columns=col_mapping)

                # If specific turbine names are supplied apply them here
                if self.turbine_names is not None:
                    columns = df.columns
                    for t in range(len(self.turbine_names)):
                        columns = [c.replace('%03d' % t,self.turbine_names[t]) for c in columns]
                    df.columns = columns

                df_array.append(df)

        # Merge dataframes
        self.df = pd.concat(df_array, axis=1).reset_index(drop=False)

        # If sorting the columns do it now
        if self.sort_columns:
            self.df = self.df[sorted(self.df.columns)]

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
        self.channel_listbox[0].config({"width": frame_width})

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
                id = [i for i in range(len(cols)) if cn == cols[i]][0]
                self.channel_listbox[i].selection_set(id)

    def update_plot(self, channel_no):
        # Only update if we have anything to plot...
        if (self.df.shape[0] > 1) & any(
            [len(i) > 0 for i in self.channel_selection]
        ):
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
            print("No preexisting figures found.")

        self.frame_2 = tk.Frame(self.master, width=20, height=500)
        self.fig = Figure()
        self.axes = [[] for _ in range(self.N_channels)]
        self.axes[0] = self.fig.add_subplot(self.N_channels, 1, 1)
        self.update_plot(channel_no=0)
        for ii in range(1, self.N_channels):
            self.axes[ii] = self.fig.add_subplot(
                self.N_channels, 1, ii + 1, sharex=self.axes[0]
            )
            self.update_plot(channel_no=ii)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_2)
        self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side="bottom", fill="both", expand=True
        )
        self.frame_2.pack(fill="both", expand=True, side="right")
        # self.update_channel_cols()  # Reset column selection

    def ci_select(self, channel_no, evt=None):
        indices = self.channel_listbox[channel_no].curselection()
        channels = [self.df.columns[idx] for idx in indices]
        self.channel_selection[channel_no] = channels
        self.update_plot(channel_no=channel_no)


# def get_timestamp_of_last_downloaded_datafile(filelist):
#     time_latest = None
#     for fi in filelist:
#         df = pd.read_feather(fi)
#         time_array = df['time']
#         if not all(np.isnan(time_array)):
#             tmp_time_max = np.max(df['time'])
#             if time_latest is None:
#                 time_latest = tmp_time_max
#             else:
#                 if tmp_time_max > time_latest:
#                     time_latest = tmp_time_max

#     return time_latest


# # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # RAW DATA READING FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # #
# def find_files_to_add_to_sqldb(sqldb_engine, files_paths, filenames_table):
#     """This function is used to figure out which files are already
#     uploaded to the SQL database, and which files still need to be
#     uploaded.

#     Args:
#         sqldb_engine ([SQL engine]): SQL Engine from sqlalchemy.create_engine
#         used to access the SQL database of interest. This is used to call
#         which files have previously been uploaded.
#         files_paths ([list, str]): List of strings or a single string containing
#         the path to the raw data files. One example is:
#             files_paths = ['/home/user/data/windfarm1/year1/*.csv',
#                            '/home/user/data/windfarm1/year2/*.csv',
#                            '/home/user/data/windfarm1/year3/*.csv',]
#         filenames_table ([str]): SQL table name containing the filenames of
#         the previously uploaded data files.

#     Returns:
#         files ([list]): List of files that are not yet in the SQL database
#     """
#     # Convert files_paths to a list
#     if isinstance(files_paths, str):
#         files_paths = [files_paths]

#     # Figure out which files exists on local system
#     files = []
#     for fpath in files_paths:
#         fl = glob.glob(fpath)
#         if len(fl) <= 0:
#             print('No files found in directory %s.' % fpath)
#         else:
#             files.extend(fl)

#     # Figure out which files have already been uploaded to sql db
#     query_string = "select * from " + filenames_table + ";"
#     df = pd.read_sql_query(query_string, sqldb_engine)

#     # # Check for the files not in the database
#     files = [f for f in files
#              if os.path.basename(f) not in df['filename'].values]

#     # Sort the file list according to ascending name
#     files = sorted(files, reverse=False)

#     return files


# # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # DATA UPLOAD FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # #


# def omit_last_rows_by_buffer(df, omit_buffer):
#     if not 'time' in df.columns:
#         df = df.reset_index(drop=False)

#     num_rows = df.shape[0]
#     df = df[df['time'] < max(df['time']) - omit_buffer]
#     print('Omitting last %d rows (%s s) as a buffer for future files.'
#           % (num_rows-df.shape[0], omit_buffer))
#     return df


# def remove_duplicates_with_sqldb(df, sql_engine, table_name):
#     min_time = df.time.min()
#     max_time = df.time.max()

#     time_query = (
#         "select time from "+ table_name +
#         " where time BETWEEN '%s' and '%s';" % (min_time, max_time)
#     )

#     df_time = pd.read_sql_query(time_query, sql_engine)
#     df_time["time"] = pd.to_datetime(df_time.time)
#     print("......Before duplicate removal there are %d rows" % df.shape[0])
#     df = df[~df.time.isin(df_time.time)]
#     print("......After duplicate removal there are %d rows" % df.shape[0])

#     # Check for self duplicates in to-be-uploaded dataset
#     print("......Before SELF duplicate removal there are %d rows" % df.shape[0])
#     if "turbid" in df.columns:
#         df = df.drop_duplicates(subset=["time", "turbid"], keep="first")
#     else:
#         df = df.drop_duplicates(subset=["time"], keep="first")
#     print("......After SELF duplicate removal there are %d rows" % df.shape[0])

#     # Drop null times in to-be-uploaded dataset
#     print(
#         "......Before null time/turbid duplicate removal there are %d rows"
#         % df.shape[0]
#     )
#     df = df.dropna(subset=["time"])
#     print("......After null time duplicate removal there are %d rows" % df.shape[0])

#     return df


# def batch_download_data_from_sql(dbc, destination_path, table_name):
#     print("Batch downloading data from table %s." % table_name)

#     # Check if output directory exists, if not, create
#     if not os.path.exists(destination_path):
#         os.makedirs(destination_path)

#     # Check current start and end time of database
#     db_end_time = get_last_time_entry_sqldb(dbc.engine, table_name)
#     db_end_time = db_end_time + datetime.timedelta(minutes=10)

#     # Check for past files and continue download or start a fresh download
#     files_result = fsio.browse_downloaded_datafiles(destination_path,
#                                                     table_name=table_name)
#     print('A total of %d existing files found.' % len(files_result))

#     # Next timestamp is going to be next first of the month
#     latest_timestamp = get_timestamp_of_last_downloaded_datafile(files_result)
#     if latest_timestamp is None:
#         db_start_time = get_first_time_entry_sqldb(dbc.engine, table_name)
#         db_start_time = db_start_time - datetime.timedelta(minutes=10)
#         current_timestamp = db_start_time
#     elif latest_timestamp.month == 12:
#         current_timestamp = pd.to_datetime('%s-01-01' %
#             str(latest_timestamp.year+1))
#     else:
#         current_timestamp = pd.to_datetime("%s-%s-01" % (
#             str(latest_timestamp.year), str(latest_timestamp.month+1)))

#     print('Continuing import from timestep: ', current_timestamp)
#     while current_timestamp <= db_end_time:
#         print('Importing data for ' +
#               str(current_timestamp.strftime("%B")) +
#               ', ' + str(current_timestamp.year) + '.')
#         if current_timestamp.month == 12:
#             next_timestamp = current_timestamp.replace(
#                 year=current_timestamp.year+1, month=1,
#                 day=1, hour=0, minute=0, second=0)
#         else:
#             next_timestamp = current_timestamp.replace(
#                 month=current_timestamp.month+1,
#                 day=1, hour=0, minute=0, second=0)

#         df = dbc.get_table_data_from_db_wide(
#             table_name=table_name,
#             start_time=current_timestamp,
#             end_time=next_timestamp
#             )

#         # Drop NaN rows
#         df = dfm.df_drop_nan_rows(df)

#         # Save dataset as a .ftr file
#         fout = os.path.join(destination_path, "%s_%s.ftr" %
#             (current_timestamp.strftime("%Y-%m"), table_name))

#         df = df.reset_index(drop=('time' in df.columns))
#         df.to_feather(fout)

#         print('Data for ' + table_name +
#               ' saved to .ftr files for ' +
#               str(current_timestamp.strftime("%B")) +
#               ', ' + str(current_timestamp.year) + '.')

#         current_timestamp = next_timestamp
