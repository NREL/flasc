# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .. import utilities as fsut
from ..dataframe_operations import dataframe_manipulations as dfm


def filter_df_by_status(df, exclude_columns=[], drop_all_bad_status=True,
                        drop_all_status_cols=True):
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

    print('WARNING: THIS IS A LEGACY FUNCTION AND FUTURE DATABASES AND ' +
          'DATAFRAMES WILL NOT CONTAIN ANY STATUS_ COLUMNS. INSTEAD, ' +
          'FAULTY DATA IS DIRECTLY OVERWRITTEN BY NP.NAN TO AVOID ITS ' +
          '(ACCIDENTAL) USAGE.')

    turbine_list = range(fsut.get_num_turbines(df))
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

    if drop_all_bad_status:
        Ninit = df.shape[0]
        df = df.dropna(subset=status_cols)
        if Ninit > df.shape[0]:
            print('Dropped %d rows due to all status flags being 0.'
                  % (df.shape[0] - Ninit))

    if drop_all_status_cols:
        self_status_cols = ["status_%03d" % ti for ti in turbine_list]
        self_status_cols = [c for c in self_status_cols if c in df.columns]
        all_status_cols = [*self_status_cols, *status_cols]
        print('Dropping columns: ', all_status_cols)
        df = df.drop(columns=self_status_cols)
    return df


def df_get_no_faulty_measurements(df, turbine):
    if isinstance(turbine, str):
        turbine = int(turbine)
    entryisnan = np.isnan(df['pow_%03d' % turbine].astype(float))
    # cols = [s for s in df.columns if s[-4::] == ('_%03d' % turbine)]
    # entryisnan = (np.sum(np.isnan(df[cols]),axis=1) > 0)
    N_isnan = np.sum(entryisnan)
    return N_isnan


def plot_highlight_data_by_conds(df, conds, ti):
    if not isinstance(conds[0], (list, np.ndarray, pd.Series)):
        conds = [conds]

    # Convert time arrays to a string with 'year+week'
    tfull = [int('%04d%02d' % (i.isocalendar().year, i.isocalendar().week)) for i in df.time]
    time_array = np.unique(tfull)

    # Get number of non-NaN entries before filtering
    if 'wd_000' in df.columns:
        N_df = [
            sum(~np.isnan(df.loc[tfull==t, 'wd_%03d' % ti]))
            for t in time_array
        ]
    else:
        N_df = [
            sum(~np.isnan(df.loc[tfull==t, 'ws_%03d' % ti]))
            for t in time_array
        ]

    N_hl = [np.zeros(len(time_array), 'int') for _ in range(len(conds))]
    conds_combined = np.array([False] * len(conds[0]), dtype=bool)
    for ii in range(len(conds)):
        # Convert time array of occurrences to year+week no.
        conds_new = conds[ii] & (~conds_combined)
        conds_combined = (conds_combined | np.array(conds[ii], dtype=bool))
        subset_time_array = [int('%04d%02d' % (i.isocalendar().year, i.isocalendar().week)) for i in df.loc[conds_new, 'time']]

        for iii in range(len(time_array)):
            # Count occurrences for condition
            N_hl[ii][iii] = (
                np.sum(np.array(subset_time_array) == time_array[iii])
            )

    # Now plot occurrences
    xlbl = ['%s-%s' % (str(t)[0:4], str(t)[4:6]) for t in time_array]
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.bar(xlbl, N_df, width=1, label='Full dataset')
    y_bottom = np.array([0 for _ in N_hl[0]], dtype=int)
    for ii in range(0, len(conds)):
        ax.bar(xlbl, N_hl[ii],
               width=1, bottom=y_bottom,
               label='Data subset (conditions 0 to %d)' % ii,
               edgecolor='gray', linewidth=0.2)
        y_bottom = y_bottom + N_hl[ii]

    # ax.set_xticks(ax.get_xticks()[::4])
    xlbl_short = ['' for _ in range(len(xlbl))]
    stp = int(np.round(len(xlbl)/50))  # About 50 labels is fine
    stp = int(np.max([stp, 1]))
    xlbl_short[::stp] = xlbl[::stp]
    ax.set_xticks(list(ax.get_xticks()))
    ax.set_xticklabels(xlbl_short)
    ax.set_xlabel('Time (year-week no.)')
    ax.set_title('Turbine %03d' % ti)
    ax.legend()
    ax.set_autoscalex_on(True)
    plt.xticks(rotation='vertical')
    fig.tight_layout()

    return fig, ax


def df_mark_turbdata_as_faulty(df, cond, turbine_list, exclude_columns=[], verbose=False):
    if isinstance(turbine_list, (np.integer, int)):
        turbine_list = [turbine_list]

    for ti in turbine_list:
        N_init = df_get_no_faulty_measurements(df, ti)
        cols = [s for s in df.columns if s[-4::] == ('_%03d' % ti)
                and s not in exclude_columns]
        df.loc[cond, cols] = None  # Delete measurements
        N_post = df_get_no_faulty_measurements(df, ti)
        if verbose:
            print('Faulty measurements for turbine %03d increased from %.1f %% to %.1f %%.'
                  % (ti, 100*N_init/df.shape[0], 100*N_post/df.shape[0]))

    # df = dfm.df_drop_nan_rows(df, verbose=verbose)  # Drop nan rows
    return df
