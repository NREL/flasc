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

from flasc.utilities import printnow as print
from flasc.dataframe_operations import dataframe_manipulations as dfm


def load_data():
    # Load the filtered data
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "00_raw_data")
    df = pd.read_feather(os.path.join(data_path, "df_scada_60s.ftr"))
    return df


if __name__ == "__main__":
    # In this script, we rename the arbitrarily named variables from the
    # SCADA data to our common format: "wd_000", "wd_001", ..., "ws_000",
    # "ws_001", and so on. This helps to further automate and align
    # the next steps in data processing.

    # Set up folders
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data")
    out_path = os.path.join(data_path, "01_in_common_df_format")
    os.makedirs(out_path, exist_ok=True)

    # Load raw data
    df = load_data()

    # In FLORIS, turbines are numbered from 0 to nturbs - 1. In SCADA data,
    # turbines often have a different name. We save the mapping between
    # the turbine indices in FLORIS and the turbine names to a separate .csv
    # file.
    turbine_names = ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
    pd.DataFrame({"turbine_names": turbine_names}).to_csv(
        os.path.join(data_path, "turbine_names.csv")
    )

    # Now map columns to conventional format
    scada_dict = {}
    for ii, tn in enumerate(turbine_names):
        scada_dict.update(
            {
                "ActivePower_{:s}".format(tn): "pow_{:03d}".format(ii),
                "NacWSpeed_{:s}".format(tn): "ws_{:03d}".format(ii),
                "NacTI_{:s}".format(tn): "ti_{:03d}".format(ii),
                "NacWDir_{:s}".format(tn): "wd_{:03d}".format(ii),
                "is_operation_normal_{:s}".format(tn): "is_operation_normal_{:03d}".format(ii),
            }
        )

    df_list = []
    print("formatting Dataframe...")
    df = df.rename(columns=scada_dict)

    # Sort dataframe and save
    df = df.sort_values(axis=0, by="time")
    df = df.reset_index(drop=True)

    fout = os.path.join(out_path, "scada_data_60s.ftr")
    df.to_feather(fout)
    print("Saved processed dataframe to: {:s}.".format(fout))
