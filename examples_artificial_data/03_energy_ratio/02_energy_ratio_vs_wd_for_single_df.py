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
from flasc.energy_ratio import energy_ratio as er
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput
from flasc.dataframe_operations import \
    dataframe_manipulations as dfm
from flasc import floris_tools as fsatools
from flasc.utilities_examples import load_floris_artificial as load_floris


def load_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir, '..', '01_raw_data_processing', 'postprocessed',
        'df_scada_data_600s_filtered_and_northing_calibrated.ftr'
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            'Please run the scripts in /raw_data_processing/' +
            'before trying any of the other examples.'
        )
    df = pd.read_feather(ftr_path)
    return df


if __name__ == '__main__':
    # Load data and floris object
    df = load_data()
    fi, _ = load_floris()

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
    er_in = EnergyRatioInput([df], ["baseline"])

    # Get energy ratio without uncertainty quantification
    er_out = er.compute_energy_ratio(
        er_in, 
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()
    
    # Also show polar plot
    ax = er_out.plot_energy_ratios(polar_plot=True, show_wind_speed_distribution=False)
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=20 bootstrap samples and 5-95 percent conf. bounds.
    er_out = er.compute_energy_ratio(
        er_in, 
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
        N=20,
        percentiles=[5.0, 95.0]
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title("Energy ratios for turbine 001 with UQ "
                    + "(N=20, 90% confidence interval)")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=20 bootstrap samples and without block bootstrapping.
    er_in_noblocks = EnergyRatioInput([df], ["baseline"], num_blocks=len(df))
    er_out = er.compute_energy_ratio(
        er_in_noblocks, 
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
        N=20,
        percentiles=[5.0, 95.0]
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title("Energy ratios for turbine 001 with UQ "
                    + "(N=20, Normal (not Block) Bootstrapping)")
    plt.tight_layout()

    plt.show()
