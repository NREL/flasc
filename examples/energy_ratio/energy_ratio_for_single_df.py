# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import os
import pandas as pd

from floris import tools as wfct
from floris_scada_analysis.energy_ratio import energy_ratio
from floris_scada_analysis.dataframe_operations import dataframe_filtering as dff
from floris_scada_analysis.dataframe_operations import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as fsatools


def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(root_dir, '../demo_dataset/demo_dataset_60s.ftr')
    if not os.path.exists(ftr_path):
        raise FileNotFoundError('Please run ./examples/demo_dataset/' +
                                'generate_demo_dataset.py before try' +
                                'ing any of the other examples.')
    df = pd.read_feather(ftr_path)
    return df


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    # fi.vis_layout()
    return fi


if __name__ == '__main__':
    # Load data and floris object
    df = load_data()
    fi = load_floris()

    # Preprocess dataframes using floris
    df = dff.filter_df_by_status(df)
    df = dfm.set_wd_by_all_turbines(df)
    df_upstream = fsatools.get_upstream_turbs_floris(fi, wd_step=5.0)
    df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
    df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

    # limit df to narrow wind direction region
    df = dfm.filter_df_by_wd(df=df, wd_range=[20., 90.])
    df = df.reset_index(drop=True)

    # Initialize energy ratio object for the dataframe
    era = energy_ratio.energy_ratio(
        df_in=df,
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        verbose=True
        )

    # Without bootstrapping
    result = era.get_energy_ratio()
    era.plot_energy_ratio()

    # With bootstrapping
    result = era.get_energy_ratio(N=10, percentiles=[10., 90.])
    era.plot_energy_ratio()
    plt.show()
