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
import pandas as pd

from floris.utilities import wrap_360
from flasc.dataframe_operations import dataframe_manipulations as dfm


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "06_time_synchronization")
    df_scada = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    return df_scada


if __name__ == "__main__":
    # This script now removes the estimated wind direction bias for every
    # wind direction signal. We then save this bias-corrected dataframe
    # to our local path folder for then to use in further data analysis,
    # e.g., model validation, wake loss estimation, turbine monitoring.

    # Load the SCADA data
    df = load_data()
    num_turbines = dfm.get_num_turbines(df)

    # Specify paths
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "07_wdbias_filtered_data")

    # Load pre-calculated estimated bias terms
    fn = os.path.join(data_path, "df_bias.csv")
    df_bias = pd.read_csv(fn)

    # Set turbine-individual bias corrections
    for ti in range(num_turbines):
        dx = float(df_bias.loc[df_bias["ref_turbine"] == ti, "wd_bias"])
        print("Removing {:.3f} deg bias for ti = {:03d}.".format(dx, ti))
        df["wd_{:03d}".format(ti)] = wrap_360(df["wd_{:03d}".format(ti)] - dx)

    # Save the dataframe with corrected wind directions
    print("Saving dataframes as .ftr files")
    df.to_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    print("Finished processing. Saved the df dataframes as .ftr files.")
