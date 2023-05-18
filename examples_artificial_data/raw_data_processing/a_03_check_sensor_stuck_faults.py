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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.turbine_analysis import find_sensor_faults as fsf


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "02_basic_filtered")
    return pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))


if __name__ == "__main__":
    # In this script, we check various variables (here: "wd_00x" and "ws_00x")
    # for sensor-stuck type of faults. This is the situation where the
    # varabiable reports the exact same value for several measurements in a
    # row, which is unrealistic and likely represents an issue with the data.
    #
    df = load_data()
    df = df.reset_index(drop=True)

    df_scada = fsf.filter_sensor_faults(
        df=df,
        columns=["wd", "ws"],
        plot_figures=True,
        figure_save_path=None, #figure_path
    )

    # Save as a single file and as batch files
    # fout = os.path.join(out_path, "scada_data_60s.ftr")
    # print("Processed dataset saved to {:s}.".format(fout))
    # df = df.reset_index(drop=("time" in df.columns))
    # df.to_feather(fout)

    plt.show()
