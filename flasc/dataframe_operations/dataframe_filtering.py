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
from scipy.interpolate import interp1d

from .. import utilities as fsut
from ..dataframe_operations import dataframe_manipulations as dfm


# def filter_df_by_status(df, exclude_columns=[], drop_all_bad_status=True,
#                         drop_all_status_cols=True):
#     """This function overwrites measurement values with np.nan wherever
#     the related status flag for that particular turbine reports a value
#     of 0 (status_000 = 0, status_001 = 0, ....). You can exclude particular
#     columns from being overwritten by inserting them into the
#     exclude_columns list.

#     Args:
#         df ([pd.DataFrame]): Dataframe with SCADA data with measurements
#         formatted according to wd_000, wd_001, wd_002, pow_000, pow_001,
#         pow_002, and so on.
#         exclude_fields (list, optional): Columns that should not be over-
#         written by a np.nan value. Defaults to [], and will only prevent
#         overwriting of columns containing the substring "status".

#     Returns:
#         df([pd.DataFrame]): The dataframe with measurements overwritten
#         with a np.nan value wherever that turbine's status flag reports
#         a value of 0.
#     """

#     print('WARNING: THIS IS A LEGACY FUNCTION AND FUTURE DATABASES AND ' +
#           'DATAFRAMES WILL NOT CONTAIN ANY STATUS_ COLUMNS. INSTEAD, ' +
#           'FAULTY DATA IS DIRECTLY OVERWRITTEN BY NP.NAN TO AVOID ITS ' +
#           '(ACCIDENTAL) USAGE.')

#     turbine_list = range(fsut.get_num_turbines(df))
#     status_cols = ["status_%03d" % ti for ti in turbine_list]
#     status_cols = [c for c in status_cols if c in df.columns]
#     if len(status_cols) < len(turbine_list):
#         print('Found fewer status columns (%d) than turbines (%d).'
#               % (len(status_cols), len(turbine_list)) +
#               ' Ignoring missing entries.')

#     exclude_columns.extend([c for c in df.columns if 'status' in c])
#     for c in status_cols:
#         ti_string = c[-4::] # Last 4 digits of string: underscore and turb. no
#         ti_columns = [s for s in df.columns if s[-4::] == ti_string and
#                       not s in exclude_columns]
#         df.loc[df[c] == 0, ti_columns] = np.nan

#     if drop_all_bad_status:
#         Ninit = df.shape[0]
#         df = df.dropna(subset=status_cols)
#         if Ninit > df.shape[0]:
#             print('Dropped %d rows due to all status flags being 0.'
#                   % (df.shape[0] - Ninit))

#     if drop_all_status_cols:
#         self_status_cols = ["status_%03d" % ti for ti in turbine_list]
#         self_status_cols = [c for c in self_status_cols if c in df.columns]
#         all_status_cols = [*self_status_cols, *status_cols]
#         print('Dropping columns: ', all_status_cols)
#         df = df.drop(columns=self_status_cols)
#     return df


def filter_df_by_faulty_impacting_turbines(df, ti, df_impacting_turbines, verbose=True):
    """Assigns a turbine's measurement to NaN for each timestamp for which any of the turbines
      that are shedding a wake on this turbine is reporting NaN measurements.

    Args:
        df (pd.DataFrame): Dataframe with SCADA data with measurements
        formatted according to wd_000, wd_001, wd_002, pow_000, pow_001,
        pow_002, and so on.
        ti (integer): Turbine number for which we are filtering the data.
        Basically, each turbine that impacts that power production of
        turbine 'ti' by more than 0.1% is required to be reporting a
        non-faulty measurement. If not, we classify the measurement of
        turbine 'ti' as faulty because we cannot sufficiently know the
        inflow conditions of this turbine.
        df_impacting_turbines (pd.DataFrame): A Pandas DataFrame in the
        format of:

                               0       1          2   3   4   5   6
                wd                                                 
                0.0       [6, 5]     [5]     [3, 5]  []  []  []  []
                3.0          [6]     [5]     [3, 5]  []  []  []  []
                ...          ...     ...        ...  ..  ..  ..  ..
                354.0  [6, 5, 3]  [5, 0]     [3, 5]  []  []  []  []
                357.0     [6, 5]     [5]  [3, 5, 4]  []  []  []  []

        The columns indicate the turbine of interest, i.e., the turbine that
        is waked, and each row shows which turbines are waking that turbine
        for that particular wind direction ('wd'). Typically calculated using:

            import flasc.floris_tools as ftools
            df_impacting_turbines = ftools.get_all_impacting_turbines(fi)

        verbose (bool, optional): Print information to the console. Defaults
        to True.

    Returns:
        pd.DataFrame: The postprocessed dataframe for 'df', filtered for
        inter-turbine issues like curtailment and turbine downtime.
    """

    # Get number of turbines
    n_turbines = dfm.get_num_turbines(df)

    # Drop all measurements where an upstream turbine is affecting this turbine but also has a NaN measurement itself
    ws_cols = ["ws_{:03d}".format(ti) for ti in range(n_turbines)]
    pow_cols = ["pow_{:03d}".format(ti) for ti in range(n_turbines)]

    # Get array of which turbines affect our test turbine
    wd_array = df["wd"]

    # Create interpolant returning impacting turbines
    xp = np.array(df_impacting_turbines[ti].index, dtype=float)
    fp = np.arange(len(xp), dtype=int)

    # Copy values over from 0 to 360 deg
    if (np.abs(xp[0]) < 0.001) & (np.max(xp) < 360.0):
        xp = np.hstack([xp, 360.0])
        fp = np.hstack([fp, fp[0]])

    # Get nearest neighbor indices
    f = interp1d(
        x=xp,
        y=fp,
        kind="nearest"
    )

    ids = np.array(f(wd_array), dtype=int)
    turbines_impacted = df_impacting_turbines[ti].values[ids]

    # Organize as matrix for easy manipulations
    impacting_turbines_matrix = np.zeros((len(wd_array), n_turbines), dtype=bool)
    for ii, turbines_impacted_onewd in enumerate(turbines_impacted):
        impacting_turbines_matrix[ii, turbines_impacted_onewd] = True

    # Calculate True/False statement whether any of the turbines shedding a wake on our test_turbine has a NaN ws or pow measurement
    test_turbine_impacted_by_nan_ws  = np.any(np.isnan(np.array(df[ws_cols], dtype=float))  & impacting_turbines_matrix, axis=1)
    test_turbine_impacted_by_nan_pow = np.any(np.isnan(np.array(df[pow_cols], dtype=float)) & impacting_turbines_matrix, axis=1)
    test_turbine_impacted = test_turbine_impacted_by_nan_ws | test_turbine_impacted_by_nan_pow

    # Assign test turbine's measurements to NaN if any turbine that is waking this turbine is reporting NaN measurements
    df_out = df_mark_turbdata_as_faulty(
        df=df,
        cond=test_turbine_impacted,
        turbine_list=[ti],
        verbose=verbose
    )
    return df_out


def df_get_no_faulty_measurements(df, turbine):
    if isinstance(turbine, str):
        turbine = int(turbine)
    entryisnan = np.isnan(df['pow_%03d' % turbine].astype(float))
    # cols = [s for s in df.columns if s[-4::] == ('_%03d' % turbine)]
    # entryisnan = (np.sum(np.isnan(df[cols]),axis=1) > 0)
    N_isnan = np.sum(entryisnan)
    return N_isnan


# def plot_highlight_data_by_conds(df, conds, ti, ax=None):
#     if not isinstance(conds[0], (list, np.ndarray, pd.Series)):
#         conds = [conds]

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(15, 8))

#     # Convert time arrays to a string with 'year+week'
#     tfull = [int('%04d%02d' % (i.isocalendar().year, i.isocalendar().week)) for i in df.time]
#     time_array = np.unique(tfull)

#     # Get number of non-NaN entries before filtering
#     if 'wd_000' in df.columns:
#         N_df = [
#             sum(~np.isnan(df.loc[tfull==t, 'wd_%03d' % ti]))
#             for t in time_array
#         ]
#     else:
#         N_df = [
#             sum(~np.isnan(df.loc[tfull==t, 'ws_%03d' % ti]))
#             for t in time_array
#         ]

#     N_hl = [np.zeros(len(time_array), 'int') for _ in range(len(conds))]
#     conds_combined = np.array([False] * len(conds[0]), dtype=bool)
#     for ii in range(len(conds)):
#         # Convert time array of occurrences to year+week no.
#         conds_new = conds[ii] & (~conds_combined)
#         conds_combined = (conds_combined | np.array(conds[ii], dtype=bool))
#         subset_time_array = [int('%04d%02d' % (i.isocalendar().year, i.isocalendar().week)) for i in df.loc[conds_new, 'time']]

#         for iii in range(len(time_array)):
#             # Count occurrences for condition
#             N_hl[ii][iii] = (
#                 np.sum(np.array(subset_time_array) == time_array[iii])
#             )

#     # Now plot occurrences
#     xlbl = ['%s-%s' % (str(t)[0:4], str(t)[4:6]) for t in time_array]

#     ax.bar(xlbl, N_df, width=1, label='Full dataset')
#     y_bottom = np.array([0 for _ in N_hl[0]], dtype=int)
#     for ii in range(0, len(conds)):
#         ax.bar(xlbl, N_hl[ii],
#                width=1, bottom=y_bottom,
#                label='Data subset (conditions 0 to %d)' % ii,
#                edgecolor='gray', linewidth=0.2)
#         y_bottom = y_bottom + N_hl[ii]

#     # ax.set_xticks(ax.get_xticks()[::4])
#     xlbl_short = ['' for _ in range(len(xlbl))]
#     stp = int(np.round(len(xlbl)/50))  # About 50 labels is fine
#     stp = int(np.max([stp, 1]))
#     xlbl_short[::stp] = xlbl[::stp]
#     ax.set_xticks(list(ax.get_xticks()))
#     ax.set_xticklabels(xlbl_short)
#     ax.set_xlabel('Time (year-week no.)')
#     ax.set_title('Turbine %03d' % ti)
#     ax.legend()
#     ax.set_autoscalex_on(True)
#     plt.xticks(rotation='vertical')
#     fig.tight_layout()

#     return fig, ax


def df_mark_turbdata_as_faulty(df, cond, turbine_list, exclude_columns=[]):
    if isinstance(turbine_list, (np.integer, int)):
        turbine_list = [turbine_list]

    for ti in turbine_list:
        N_init = df_get_no_faulty_measurements(df, ti)
        cols = [s for s in df.columns if s[-4::] == ('_%03d' % ti)
                and s not in exclude_columns]
        df.loc[cond, cols] = None  # Delete measurements

    return df
