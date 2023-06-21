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

from floris import tools as wfct

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.yaml")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "04_wspowcurve_filtered")
    df = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    return df


if __name__ == "__main__":
    # This script does not manipulate or filter the data in any way. It is
    # only meant for the visualization of which turbines most often encounter
    # faulty measurements, and which turbines are best used in further post-
    # processing and analysis steps.

    # Load data and load FLORIS model
    df = load_data()
    fi = load_floris()

    layout_x = fi.layout_x
    layout_y = fi.layout_y

    num_turbines = dfm.get_num_turbines(df)
    fault_ratio = np.zeros(num_turbines)
    for ti in range(num_turbines):
        fault_ratio[ti] = (
            dff.df_get_no_faulty_measurements(df, ti) / df.shape[0]
        )

    # Plot layout and colormap
    fig, ax = plt.subplots(figsize=(14, 5))
    for ti in range(num_turbines):
        clr = [fault_ratio[ti], 1.0 - fault_ratio[ti], 0.0]
        ax.plot(
            layout_x[ti],
            layout_y[ti],
            "o",
            markersize=15,
            markerfacecolor=clr,
            markeredgewidth=0.0,
        )
        ax.text(
            layout_x[ti] + 100,
            layout_y[ti],
            "T%03d (%.1f%%)" % (ti, (1.0 - fault_ratio[ti]) * 100.0),
            color="black",
        )
    fig.tight_layout()

    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(
        root_path,
        "data",
        "05_preliminary_fault_analysis",
        "show_filtered_faults_by_layout",
    )
    fig_out = os.path.join(out_path, "faults_by_layout.png")
    print("Saving figure to {:s}.".format(fig_out))
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(fig_out, dpi=300)

    plt.show()
