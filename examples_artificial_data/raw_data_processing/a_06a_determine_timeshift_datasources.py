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

from datetime import timedelta as td
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from flasc import optimization as fopt
from flasc import time_operations as fto


def load_data():
    # Load the scada data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "04_wspowcurve_filtered")
    df_scada = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))

    # Load met mast data
    data_path = os.path.join(root_path, "data", "00_raw_data")
    df_metmast = pd.read_feather(os.path.join(data_path, "df_metmast_60s.ftr"))

    return df_scada, df_metmast


if __name__ == "__main__":
    # In this script, we determine the timeshift between different data
    # sources. This is necessary since very often, data is not logged to the
    # same timezone. Often, data from a met mast can be shifted by one
    # or multiple hours compared to the data logged by the turbines. In this
    # code, we calculate the timeshift that best aligns two data sources.
    # In our example case, we will estimate the timeshift between the met
    # mast data and the turbine SCADA data. When generating the data, we
    # applied an artificial timeshift of 2 hours to the met mast data.
    # We hope/expect to reconstruct this timeshift by looking at the data.

    # Load raw (60s data)
    df_scada, df_metmast = load_data()

    # use wind direction signal of turbine 003 for time synchronization
    # since wind directions are less sensitive to waked scenarios than
    # wind speed measurements are.
    scada_cols = ["wd_003"]
    metmast_cols = ["WindDirection_80m"]
    circular_stats = True

    # Now, only keep the data we are interested in
    df_scada = df_scada[["time", "wd_003"]]
    df_metmast = df_metmast[["time", "WindDirection_80m"]]

    # We assume SCADA and met mast are working in same frame of reference,
    # albeit with a shift in their northing calibration. But positive
    # is positive in both (clockwise vs. counterclockwise). If that is
    # not the case, we need to flip either of the two by using
    # 'wind_direction = 360.0 - wind_direction'.

    # Now downsample to 5 min steps to speed up things.
    df_scada = fto.df_downsample(
        df_in=df_scada,
        cols_angular=["wd_003"],
        window_width=td(seconds=300),
        calc_median_min_max_std=False,
        return_index_mapping=False,
    ).reset_index(drop=False)

    df_metmast = fto.df_downsample(
        df_in=df_metmast,
        cols_angular=["WindDirection_80m"],
        window_width=td(seconds=300),
        calc_median_min_max_std=False,
        return_index_mapping=False,
    ).reset_index(drop=False)

    # Explore a 3 hour time shift in either direction. Note that we also
    # need to add 'correct_y_shift=True' because the wind directions
    # between the met mast and the SCADA data may be offset by any number
    # in [-180.0, +180.0] deg, since their northing calibration can be
    # completely different.
    opt_out = fopt.find_timeshift_between_dfs(
        df1=df_scada,
        df2=df_metmast,
        cols_df1=scada_cols,
        cols_df2=metmast_cols,
        use_circular_statistics=circular_stats,
        t_step=td(days=90),
        correct_y_shift=True,
        y_shift_range=np.arange(-180.0, 180.0, 2.0),
        opt_bounds=[(-td(hours=3), td(hours=3))],
        verbose=True,
    )

    # Map outputs to a dataframe and write that as a .csv
    df_out = pd.DataFrame()
    df_out["t0"] = [o["t0"] for o in opt_out]
    df_out["t1"] = [o["t1"] for o in opt_out]
    df_out["opt_time_shift_s"] = [o["x_opt"] / td(seconds=1) for o in opt_out]
    df_out["opt_pearson"] = [-1.0 * o["J_opt"] for o in opt_out]

    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "06_time_synchronization")

    os.makedirs(out_path, exist_ok=True)
    df_out.to_csv(os.path.join(out_path, "opt_timeshift_df_scada.csv"))

    # Extract a single value for optimal shift
    opt_shift = td(seconds=df_out["opt_time_shift_s"].median())

    # Generate plots
    print("Producing plot")
    _, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    for c in scada_cols:
        ax[0].plot(df_scada.time, df_scada[c], "-o", label=c)
        ax[1].plot(df_scada.time, df_scada[c], "-o", label=c)
    for c in metmast_cols:
        ax[0].plot(
            df_metmast.time + opt_shift,
            df_metmast[c],
            linestyle="--",
            label="%s (shift: %s)" % (c, str(opt_shift)),
        )
        ax[2].plot(
            df_metmast.time + opt_shift,
            df_metmast[c],
            linestyle="--",
            label="%s (shift: %s)" % (c, str(opt_shift)),
        )

    plt.legend()
    plt.show()
