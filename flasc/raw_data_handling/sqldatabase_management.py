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
import pandas as pd
import polars as pl
from pathlib import Path
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
        self.url = "%s://%s:%s@%s/%s" % (dr, usn, password, address, name)
        self.engine = sqlalch.create_engine(
            url= self.url
        )
        self.inspector = sqlalch.inspect(self.engine)
        self.print_properties()

    def _get_table_names(self):
        return self.inspector.get_table_names()
    
    def _does_table_exist(self, table_name):
        return table_name in self._get_table_names()

    def _get_column_names(self, table_name):
        columns =  self.inspector.get_columns(table_name)
        return [c['name'] for c in columns]
    
    def _create_table_from_df(self, table_name, df):

        print(f'Creating Table: {table_name} with {df.shape[1]} columns')

        # Convert to pandas for upload
        df_pandas = df.to_pandas()
        df_pandas = df_pandas.iloc[:10]

        df_pandas.to_sql(
                    table_name,
                    self.engine,
                    index=False,
                    method="multi"
                )
        
        # Make time unique and an index to speed queries
        query = 'CREATE UNIQUE INDEX idx_time_%s ON %s (time);' % (table_name, table_name)
        print('Setting time to unique index')
        with self.engine.connect() as con:
            rs = con.execute(sqlalch.text(query))
            print(f'...RESULT: {rs}')
            con.commit()  # commit the transaction

    def _remove_duplicated_time(self, table_name, df):

        start_time = df.select(pl.min("time"))[0, 0]
        end_time = df.select(pl.max("time"))[0, 0]
        original_size = df.shape[0]

        print(f'Checking for time entries already in {table_name} between {start_time} and {end_time}')
        time_in_db = self.get_data(table_name,
                                   ['time'],
                                   start_time=start_time,
                                   end_time=end_time,
                                   end_inclusive=True
        )

        df = df.join(time_in_db, on='time',how="anti")
        new_size = df.shape[0]
        if new_size < original_size:
            print(f'...Dataframe size reduced from {original_size} to {new_size} by time values already in {table_name}')
        return df


    def _get_first_time_entry(self, table_name):

        # Get the table corresponding to the table name
        table = sqlalch.Table(table_name, sqlalch.MetaData(), autoload_with=self.engine)

        stmt = sqlalch.select(table.c.time).order_by(table.c.time.asc()).limit(1)
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            for row in result:
                return row[0]

    def _get_last_time_entry(self, table_name):
        # Get the table corresponding to the table name
        table = sqlalch.Table(table_name, sqlalch.MetaData(), autoload_with=self.engine)

        stmt = sqlalch.select(table.c.time).order_by(table.c.time.desc()).limit(1)
        print(stmt)
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            for row in result:
                return row[0]

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

        sql_db_explorer_gui(master=root,
                            dbc=self,
                            turbine_names=turbine_names,
                            sort_columns=sort_columns
                            )
        root.mainloop()

    def get_column_names(self, table_name):
        return self._get_column_names(table_name)

    def batch_get_data(self, table_name, columns=None, start_time=None,
                       end_time=None, fn_out=None, no_rows_per_file=10000):
        if fn_out is None:
            fn_out = table_name + ".ftr"
        if not (fn_out.suffix == ".ftr"):
            fn_out = fn_out.with_suffix(".ftr")

        # Ensure 'time' in database
        column_names = self._get_column_names(table_name=table_name)
        if 'time' not in column_names:
            raise KeyError("Cannot find 'time' column in database table.")

        # Get time column from database
        print("Getting time column from database...")
        time_in_db = self.get_data(table_name=table_name, columns=['time'],
                                   start_time=start_time, end_time=end_time)
        time_in_db = list(time_in_db.select("time").to_numpy().flatten())
        print("...finished,  N.o. entries: %d." % len(time_in_db))

        splits = np.arange(0, len(time_in_db) - 1, no_rows_per_file, dtype=int)
        splits = np.append(splits, len(time_in_db) - 1)
        splits = np.unique(splits)
        print(f"Splitting {len(time_in_db)} entries data into {len(splits)} subsets of {no_rows_per_file}.")

        for ii in range(len(splits) - 1):
            print("Downloading subset %d out of %d." % (ii, len(splits) - 1))
            df = self.get_data(
                table_name=table_name,
                columns=columns,
                start_time=time_in_db[splits[ii]],
                end_time=time_in_db[splits[ii+1]]
            )
            fn_out_ii = fn_out.with_suffix(".ftr.%03d" % ii)
            print("Saving file to %s" % fn_out_ii)
            df.write_ipc(fn_out_ii)

    def get_data(
        self, 
        table_name, 
        columns=None, 
        start_time=None, 
        end_time=None,
        end_inclusive=False,
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
            if end_inclusive:
                query_string += " AND time <= '" + str(end_time) + "'"
            else:
                query_string += " AND time < '" + str(end_time) + "'"
        elif (start_time is None) and (end_time is not None):
            if end_inclusive:
                query_string += " WHERE time <= '" + str(end_time) + "'"
            else:
                query_string += " WHERE time < '" + str(end_time) + "'"

        query_string += " ORDER BY time"

        df = pl.read_database(query_string,self.url)

        # Drop a column called index
        if "index" in df.columns:
            df = df.drop("index")

        # Confirm that the time column is in datetime format
        if "time" in df.columns:
            if not (df.schema["time"] == pl.Datetime):
                df = df.with_columns(pl.col("time").cast(pl.Datetime))

        return df
    

    #TODO: This is a fresh redo check it works
    def send_data(
        self,
        table_name,
        df,
        if_exists="append_new",
        unique_cols=["time"],
        df_chunk_size=2000,
        sql_chunk_size=50
    ):
        
        # Make a local copy
        df_ = df.clone()
        
        # Check if table exists
        if not self._does_table_exist(table_name):

            print(f'{table_name} does not yet exist')

            # Create the table
            self._create_table_from_df(table_name, df_)

        # Check for times already in database
        df_ = self._remove_duplicated_time(table_name, df_)

        # Check if df_ is now
        if df_.shape[0] == 0:
            print('Dataframe is empty')
            return

        # Write to database
        print(f'Inserting {df_.shape[0]} rows into {table_name} in chunks of {df_chunk_size}')
        time_start_total = timerpc()

        # Parition into chunks
        df_list = (df_.with_row_count('id')
            .with_columns(pl.col('id').apply(lambda i: int(i/df_chunk_size)))
            .partition_by('id')
        )

        num_par = len(df_list)
        for df_par_idx, df_par in enumerate(df_list):
            print(f'...inserting chunk {df_par_idx} of {num_par}')
        
            df_par.drop('id').write_database(
                table_name,
                self.url,
                if_exists='append'
            )
        total_time = timerpc() - time_start_total
        print(f'...Finished in {total_time}')


    # #TODO: UPDATE TO POLARS
    # #TODO: Paul note (may 31 2023), POLARS API not up to PANDAS so using PANDAS here
    # def send_data(
    #     self,
    #     table_name,
    #     df,
    #     if_exists="append_new",
    #     unique_cols=["time"],
    #     df_chunk_size=2000,
    #     sql_chunk_size=50
    # ):
    #     table_name = table_name.lower()
    #     table_names = [t.lower() for t in self._get_table_names()]

    #     if (if_exists == "append"):
    #         print("Warning: risk of adding duplicate rows using 'append'.")
    #         print("You are suggested to use 'append_new' instead.")

    #     if (if_exists == "append_new") and (table_name in table_names):
    #         if len(unique_cols) > 1:
    #             raise NotImplementedError("Not yet implemented.")

    #         col = unique_cols[0]
    #         idx_in_db = self.get_data(table_name=table_name, columns=[col])[
    #             col
    #         ]

    #         # Check if values in SQL database are unique
    #         if not idx_in_db.is_unique:
    #             raise IndexError(
    #                 "Column '%s' is not unique in the SQL database." % col
    #             )

    #         idx_in_df = set(df[col])
    #         idx_in_db = set(idx_in_db)
    #         idx_to_add = np.sort(list(idx_in_df - idx_in_db))
    #         print(
    #             "{:d} entries already exist in SQL database.".format(
    #                 len(idx_in_df) - len(idx_to_add)
    #             )
    #         )

    #         print("Adding {:d} new entries...".format(len(idx_to_add)))
    #         df_subset = df.set_index('time').loc[idx_to_add].reset_index(
    #             drop=False)

    #     else:
    #         df_subset = df

    #     if (if_exists == "append_new"):
    #         if_exists = "append"

    #     # Upload data
    #     N = df_subset.shape[0]
    #     if N < 1:
    #         print("Skipping data upload. Dataframe is empty.")
    #     else:
    #         print("Attempting to insert %d rows into table '%s'."
    #             % (df_subset.shape[0], table_name))
    #         df_chunks_id = np.arange(0, df_subset.shape[0], df_chunk_size)
    #         df_chunks_id = np.append(df_chunks_id, df_subset.shape[0])
    #         df_chunks_id = np.unique(df_chunks_id)

    #         time_start_total = timerpc()
    #         for i in range(len(df_chunks_id)-1):
    #             Nl = df_chunks_id[i]
    #             Nu = df_chunks_id[i+1]
    #             print("Inserting rows %d to %d." % (Nl, Nu))
    #             time_start_i = timerpc()
    #             df_sub = df_subset[Nl:Nu]
    #             df_sub.to_sql(
    #                 table_name,
    #                 self.engine,
    #                 if_exists=if_exists,
    #                 index=False,
    #                 method="multi",
    #                 chunksize=sql_chunk_size,
    #             )
    #             time_i = timerpc() - time_start_i
    #             total_time = timerpc() - time_start_total
    #             est_time_left = (total_time / Nu) * (N - Nu)
    #             eta = datetime.datetime.now() + td(seconds=est_time_left)
    #             eta = eta.strftime("%a, %d %b %Y %H:%M:%S")
    #             print("Data insertion took %.1f s. ETA: %s." % (time_i, eta))

#TODO: UPDATE TO POLARS
class sql_db_explorer_gui:
    def __init__(self, master, dbc, turbine_names = None, sort_columns=False):

        # Create the options container
        frame_1 = tk.Frame(master)
        self.master = master

        # Get basic database properties
        self.df = pl.DataFrame()
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
            # df = df.set_index("time", drop=True)

            if df.shape[0] <= 0:
                print(
                    "No data found in this timerange for table %s"
                    % table_select
                )
            else:
                print("...Imported data successfully.")

                old_col_names = [c for c in list(df.columns) if not c=='time']
                new_col_names = [
                    chr(97 + tables_selected[ii]).upper() + "_%s" % c
                    for c in old_col_names
                ]
                col_mapping = dict(zip(old_col_names, new_col_names))
                df = df.rename(col_mapping)

                # If specific turbine names are supplied apply them here
                if self.turbine_names is not None:
                    columns = df.columns
                    for t in range(len(self.turbine_names)):
                        columns = [c.replace('%03d' % t,self.turbine_names[t]) for c in columns]
                    # df.columns = columns
                    df = df.rename(dict(zip(df.columns,columns)))

                df_array.append(df)

        # Merge dataframes
        # self.df = pl.concat(df_array, axis=1)# .reset_index(drop=False)
        df_merge = df_array[0]

        if len(df_array) > 1:
            for df_ in df_array[1:]:
                df_merge = df_merge.join(df_, on='time',how='outer')

        #Save it now
        self.df = df_merge

        # If sorting the columns do it now
        if self.sort_columns:
            # self.df = self.df[sorted(self.df.columns)]
            self.df = self.df.select(sorted(self.df.columns))


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
                ax.plot(self.df['time'], np.array(self.df[c]), label=c)
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
