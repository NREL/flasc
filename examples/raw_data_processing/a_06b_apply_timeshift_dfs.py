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
import matplotlib.pyplot as plt
import pandas as pd


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
    # In this code, we apply the timeshift found in the script
    # 'a_06a_determine_timeshift_datasources.py' to the dataframes to generate
    # time-corrected dataframes.

    # Manually insert the optimal timeshift found in code a_06a_...py
    timeshift = td(minutes=-120)

    # Load data and calculate timeshifted dataframe
    df_scada, df_metmast = load_data()
    df_met_shifted = df_metmast.copy()
    df_met_shifted.time = df_met_shifted.time + timeshift

    # Plot timeseries before time shift
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    ax[0].plot(df_scada.time, df_scada.wd_003, label="SCADA")
    ax[1].plot(df_scada.time, df_scada.wd_003, label="SCADA")
    ax[0].plot(df_metmast.time, df_metmast.WindDirection_80m, label="Met mast")
    ax[1].plot(df_met_shifted.time, df_met_shifted.WindDirection_80m, label="Met mast")

    ax[0].set_title("Before timeshift")
    ax[1].set_title("After timeshift")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_ylabel("Wind direction (deg)")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    # Save time-shifted data
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "06_time_synchronization")
    df_met_shifted.to_feather(
        os.path.join(out_path, "metmast_data_60s.ftr")
    )
    df_scada.to_feather(os.path.join(out_path, "scada_data_60s.ftr"))
    print("Timeshifted dataframe(s) saved to {:s}.".format(out_path))

    plt.show()
