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

from matplotlib import pyplot as plt
import pandas as pd

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "01_in_common_df_format")
    return pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))


if __name__ == "__main__":
    # In this script, we do some very basic filtering steps, such as filtering
    # for negative wind speeds and power productions. We also filter the data
    # by one or multiple variables that inherently already tells us if data
    # is good or bad according to the data logger/turbine itself. In our case,
    # this self-flagged variable is "is_operational_normal_00x".

    # Load data and get properties
    df = load_data()
    num_turbines = dfm.get_num_turbines(df)

    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "02_basic_filtered")
    figs_path = os.path.join(out_path, "figures")
    os.makedirs(figs_path, exist_ok=True)

    # Basic filters: address self flags and obviously wrong points
    for ti in range(num_turbines):
        # Specify filtering conditions
        conds = [
            ~df["is_operation_normal_{:03d}".format(ti)],  # Self-status
            df["ws_{:03d}".format(ti)] <= 0.0,  # Non-negative wind speeds
            df["pow_{:03d}".format(ti)] <= 0.0,
        ]  # Non-negative powers

        # Retrieve a single, combined condition array
        conds_combined = conds[0]
        for cond in conds:
            conds_combined = conds_combined | cond

        # Plot time vs filtered data
        fig, ax = dff.plot_highlight_data_by_conds(df, conds, ti)
        ax.legend(
            ["All data", "Bad self-status", "Negative WS", "Negative power"]
        )
        fp = os.path.join(figs_path, "basic_filtering_%03d.png" % ti)
        print("Saving figure to {:s} for turbine {:03d}.".format(fp, ti))
        fig.savefig(fp, dpi=200)
        plt.close(fig)

        # Apply filtering to dataframe
        df = dff.df_mark_turbdata_as_faulty(
            df, conds_combined, ti, verbose=True
        )

    # Remove unnecessary columns after filtering
    self_status_cols = [
        "is_operation_normal_%03d" % ti for ti in range(num_turbines)
    ]
    df = df.drop(columns=self_status_cols)  # Remove self status columns

    # Save as a single file and as batch files
    fout = os.path.join(out_path, "scada_data_60s.ftr")
    print("Savig filtered data to {:s}.".format(fout))
    os.makedirs(out_path, exist_ok=True)
    df = df.reset_index(drop=("time" in df.columns))
    df.to_feather(fout)
