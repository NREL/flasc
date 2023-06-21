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
from flasc.energy_ratio import energy_ratio
from flasc.dataframe_operations import \
    dataframe_manipulations as dfm
from flasc import floris_tools as fsatools


def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(root_dir, '..', 'demo_dataset',
                            'demo_dataset_scada_60s.ftr')
    if not os.path.exists(ftr_path):
        raise FileNotFoundError('Please run ./examples_artificial_data/demo_dataset/' +
                                'generate_demo_dataset.py before try' +
                                'ing any of the other examples.')
    df = pd.read_feather(ftr_path)
    return df


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.yaml")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


if __name__ == '__main__':
    # Load data and floris object
    df = load_data()
    fi = load_floris()

    # Visualize layout
    fig, ax = plt.subplots()
    ax.plot(fi.layout_x, fi.layout_y, 'ko')
    for ti in range(len(fi.layout_x)):
        ax.text(fi.layout_x[ti], fi.layout_y[ti], "T{:02d}".format(ti))
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x-direction (m)")
    ax.set_ylabel("y-direction (m)")

    # We first need to define a wd against which we plot the energy ratios
    # In this example, we set the wind direction to be equal to the mean
    # wind direction between all turbines
    df = dfm.set_wd_by_all_turbines(df)

    # We reduce the dataframe to only data where the wind direction
    # is between 20 and 90 degrees.
    df = dfm.filter_df_by_wd(df=df, wd_range=[20., 90.])
    df = df.reset_index(drop=True)

    # We also need to define a reference wind speed and a reference power
    # production against to normalize the energy ratios with. In this
    # example, we set the wind speed equal to the mean wind speed
    # of all upstream turbines. The upstream turbines are automatically
    # derived from the turbine layout and the wind direction signal in
    # the dataframe, df['wd']. The reference power production is set
    # as the average power production of turbines 0 and 6, which are
    # always upstream for wind directions between 20 and 90 deg.
    df_upstream = fsatools.get_upstream_turbs_floris(fi)
    df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
    df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

    # # Initialize energy ratio object for the dataframe
    era = energy_ratio.energy_ratio(df_in=df, verbose=True)

    # Get energy ratio without uncertainty quantification
    era.get_energy_ratio(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=2.0,
    )
    fig, ax = era.plot_energy_ratio()
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()

    fig, ax = era.plot_energy_ratio(polar_plot=True)  # Plot in polar format too
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=50 bootstrap samples and 5-95 percent conf. bounds.
    era.get_energy_ratio(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=2.0,
        N=20,
        percentiles=[5.0, 95.0]
    )
    fig, ax = era.plot_energy_ratio()
    ax[0].set_title("Energy ratios for turbine 001 with UQ "
                    + "(N=20, 90% confidence interval)")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=10 bootstrap samples and block bootstrapping
    era.get_energy_ratio(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=2.0,
        N=20,
        percentiles=[5.0, 95.0],
        num_blocks=20 # Resample over 20 blocks
    )
    fig, ax = era.plot_energy_ratio()
    ax[0].set_title("Energy ratios for turbine 001 with UQ "
                    + "(N=20, Block Bootstrapping)")
    plt.tight_layout()

    plt.show()
