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

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)
from flasc.turbine_analysis import find_sensor_faults as fsf


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "02_basic_filtered")
    return pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))


def plot_top_sensor_faults(
    df,
    c,
    index_faults,
    N_eval_max=5,
    save_path=None,
    fig_format="png",
    dpi=300,
):

    # Extract largest fault set and plot
    diff_index_faults = np.diff(index_faults)
    diffjumps = np.where(diff_index_faults > 1)[0]
    fault_sets = []
    imin = 0
    for imax in list(diffjumps):
        if (imax - imin) > 1:
            fault_sets.append(index_faults[imin + 1 : imax])
        imin = imax
    if len(index_faults) - imin > 1:
        fault_sets.append(index_faults[imin + 1 : :])
    fault_sets_idx_sorted = np.argsort([len(i) for i in fault_sets])[::-1]
    N_eval = np.min([N_eval_max, len(fault_sets)])
    fig, ax_array = plt.subplots(
        nrows=N_eval, ncols=1, figsize=(5.0, 2.5 * N_eval)
    )

    if N_eval == 1:
        ax_array = [ax_array]

    for i in range(N_eval):
        ax = ax_array[i]
        fault_set_eval = fault_sets[fault_sets_idx_sorted[i]]

        indices_to_plot = range(
            fault_set_eval[0] - 4 * len(fault_set_eval),
            fault_set_eval[-1] + 4 * len(fault_set_eval),
        )
        indices_to_plot = [v for v in indices_to_plot if v in df.index]

        ax.plot(
            df.loc[indices_to_plot, "time"], df.loc[indices_to_plot, c], "o"
        )
        ax.plot(
            df.loc[index_faults, "time"],
            df.loc[index_faults, c],
            "o",
            color="red",
        )
        ax.set_xlim(
            (
                df.loc[indices_to_plot[0], "time"],
                df.loc[indices_to_plot[-1], "time"],
            )
        )
        plt.xticks(rotation="vertical")
        ax.legend(["Good data", "Faulty data"])
        ax.set_ylabel(c)
        ax.set_xlabel("Time")
        ax.set_title("Column '%s', sensor stuck fault %d" % (c, i))

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, "%s_faults.%s" % (c, fig_format))
        fig.savefig(fig_path, dpi=dpi)

    return fig, ax_array


if __name__ == "__main__":
    # In this script, we check various variables (here: "wd_00x" and "ws_00x")
    # for sensor-stuck type of faults. This is the situation where the
    # varabiable reports the exact same value for several measurements in a
    # row, which is unrealistic and likely represents an issue with the data.
    #
    df = load_data()
    df = df.reset_index(drop=True)
    time_array = np.array(df["time"])

    # Define which variables to check for. Here, that is the wind direction
    # and the wind speed according to the turbines.
    num_turbines = dfm.get_num_turbines(df)
    columns_list = []
    columns_list.extend(["ws_%03d" % ti for ti in range(num_turbines)])
    columns_list.extend(["wd_%03d" % ti for ti in range(num_turbines)])

    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "03_sensor_faults_filtered")
    figs_path = os.path.join(out_path, "figures")
    os.makedirs(out_path, exist_ok=True)

    # Settings which indicate a sensor-stuck type of fault: the standard
    # deviation between the [no_consecutive_measurements] number of
    # consecutive measurements is less than [stddev_threshold].
    stddev_threshold = 0.001
    no_consecutive_measurements = 10

    for c in columns_list:
        print("Processing column %s" % c)
        measurement_array = np.array(df[c])
        turb_str = c[-3::]

        index_faults = fsf.find_sensor_stuck(
            measurement_array=measurement_array,
            no_consecutive_measurements=no_consecutive_measurements,
            stddev_threshold=stddev_threshold,
            index_array=df.index,
        )

        if len(index_faults) > 0:
            plot_top_sensor_faults(df, c, index_faults, save_path=figs_path)
            df = dff.df_mark_turbdata_as_faulty(
                df, index_faults, int(turb_str), verbose=True
            )

    # Save as a single file and as batch files
    fout = os.path.join(out_path, "scada_data_60s.ftr")
    print("Processed dataset saved to {:s}.".format(fout))
    df = df.reset_index(drop=("time" in df.columns))
    df.to_feather(fout)

    plt.show()
