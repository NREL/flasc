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
    N_pre = df_get_no_faulty_measurements(df, ti)
    df_out = df_mark_turbdata_as_faulty(
        df=df,
        cond=test_turbine_impacted,
        turbine_list=[ti],
    )
    N_post = df_get_no_faulty_measurements(df_out, ti)

    if verbose:
        print(
            "Faulty measurements for WTG {:02d} increased from {:.3f} % to {:.3f} %. Reason: 'Turbine is impacted by faulty upstream turbine'.".format(
                ti, 100.0 * N_pre / df.shape[0], 100.0 * N_post / df.shape[0]
            )
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


def df_mark_turbdata_as_faulty(df, cond, turbine_list, exclude_columns=[]):
    if isinstance(turbine_list, (np.integer, int)):
        turbine_list = [turbine_list]

    for ti in turbine_list:
        cols = [s for s in df.columns if s[-4::] == ('_%03d' % ti)
                and s not in exclude_columns]
        df.loc[cond, cols] = None  # Delete measurements

    return df
