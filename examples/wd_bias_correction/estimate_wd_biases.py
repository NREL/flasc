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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from floris import tools as wfct

from floris_scada_analysis.energy_ratio import \
    energy_ratio_wd_bias_estimation as best
from floris_scada_analysis.dataframe_operations import \
    dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as ftools


def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(root_dir, '..', 'demo_dataset',
                            'demo_dataset_60s.ftr')
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
    return fi


def estimate_bias_for_turbine(ti):
    """Estimate the wind direction bias in degrees for a specific turbine.

    Args:
        ti ([int]): Turbine number.

    Returns:
        wd_bias ([float]): Estimated wind direction bias in degrees.
    """
    # Load data and floris object
    df = load_data()
    fi = load_floris()

    # Figure out which turbines are freestream per wind direction
    df_upstream = ftools.get_upstream_turbs_floris(fi)

    # Assign the reference wind direction to be equal to turbine ti's wind
    # direction. This is the signal against which we plot the energy
    # ratios and therefore also the signal we will estimate the offset for.
    df = dfm.set_wd_by_turbines(df, [ti])

    # Define a function that takes in a dataframe and returns a dataframe
    # with an additional column called 'ws' representing the reference
    # wind speeds.
    def df_ws_func(df):
        return dfm.set_ws_by_upstream_turbines_in_radius(
            df=df,
            df_upstream=df_upstream,
            turb_no=ti,
            x_turbs=fi.layout_x,
            y_turbs=fi.layout_y,
            max_radius=5000.0,
            include_itself=True
        )

    # Define a function that takes in a dataframe and returns a dataframe
    # with an additional column called 'pow_ref' representing the reference
    # power productions.
    def df_powref_func(df):
        return dfm.set_pow_ref_by_upstream_turbines_in_radius(
            df=df,
            df_upstream=df_upstream,
            turb_no=ti,
            x_turbs=fi.layout_x,
            y_turbs=fi.layout_y,
            max_radius=5000.0,
            include_itself=True
        )

    # Also, calculate a predefined set of solutions for the FLORIS model of
    # our wind farm. The bias estimation class will leverage this
    # precalculated set of solutions to quickly determine the predicted energy
    # ratios from FLORIS for the corresponding SCADA data energy ratios.
    root_path = os.path.dirname(os.path.abspath(__file__))
    fout_df_fi_approx = os.path.join(root_path, "df_fi_approx.ftr")
    if os.path.exists(fout_df_fi_approx):
        df_fi_approx = pd.read_feather(fout_df_fi_approx)
    else:
        print("Calculating predefined set of FLORIS solutions. "
              + "This may take a couple minutes...")
        df_fi_approx = ftools.calc_floris_approx_table(
            fi=fi,
            wd_array=np.arange(0.0, 360.0, 3.0),
            ws_array=np.arange(6.0, 11.0, 1.0),
            ti_array=None,
            num_workers=2,
            num_threads=10,
        )
        df_fi_approx.to_feather(fout_df_fi_approx)

    # We need to define the test turbines. We may not want to use the
    # turbine we use the wind direction from to also be the test turbine.
    # Namely, there may be some undesired correlations if we do so.
    # Hence, instead, here we take the 2 closest turbines to turbine 'ti'
    # as being the test turbines.
    test_turbines = ftools.get_turbs_in_radius(
        x_turbs=fi.layout_x,
        y_turbs=fi.layout_y,
        turb_no=ti,
        max_radius=9.0e9,
        include_itself=False,
        sort_by_distance=True
    )
    test_turbines = test_turbines[0:2]

    # Now, we have all variables defined to be able to initialize the bias
    # estimation class
    b = best.bias_estimation(
        df=df,
        df_fi_approx=df_fi_approx,
        test_turbines_subset=test_turbines,
        df_ws_mapping_func=df_ws_func,
        df_pow_ref_mapping_func=df_powref_func,
    )

    # Calculate and plot the baseline condition
    b.calculate_baseline(
        time_mask=None,  # Do not limit to a specific time range
        ws_mask=(6.0, 10.0),  # Limit to region II turbine operation
        wd_mask=None,  # Do not limit to specific wind direction range
        ti_mask=None,  # Do not limit to specific turbulence intensity range
        er_wd_step=3.0,  # Energy ratios in steps of 3.0 deg
        er_ws_step=5.0,  # Energy ratios in steps of 5.0 m/s
        er_wd_bin_width=3.0,  # Energy ratio bin width of 3.0 deg
        er_N_btstrp=1,  # No uncertainty quantification
    )
    fig_list, ax_list = b.plot_energy_ratios()
    for ii, ax in enumerate(ax_list):
        ax[0].set_title('Turbine %03d: Baseline wind direction signal'
                        % (test_turbines[ii]))

    # And finally call the optimization solver that estimates the optimal
    # offset to the wind direction in the SCADA dataframe that makes the
    # SCADA data best align with the FLORIS predictions, with the Pearson
    # correlation coefficient being the quantity of interest.
    # Note that we start the solver close to the optimal solution to reduce
    # computation time in this simple example case. For real-world
    # applications, one should search the full space of [-180, 180] deg.
    x_opt, J_opt = b.estimate_wd_bias(
        time_mask=None,  # Do not limit to a specific time range
        ws_mask=(6.0, 10.0),  # Limit to region II turbine operation
        wd_mask=None,  # Do not limit to specific wind direction range
        ti_mask=None,  # Do not limit to specific turbulence intensity range
        opt_search_range=(-25.0, 25.0),  # Search within [-25, +25] deg
        opt_search_brute_dx=5.0,  # Step size for offset calculation
        er_wd_step=3.0,  # Energy ratios in steps of 3.0 deg
        er_ws_step=5.0,  # Energy ratios in steps of 5.0 m/s
        er_wd_bin_width=3.0,  # Energy ratio bin width of 3.0 deg
        er_N_btstrp=1,  # No uncertainty quantification
        plot_iter_path=None  # Do not plot iterations (slow)
    )

    # Plot the energy ratios for the bias-corrected dataframe
    fig_list, ax_list = b.plot_energy_ratios()
    for ii, ax in enumerate(ax_list):
        ax[0].set_title('Turbine %03d: Corrected wind direction signal'
                        % (test_turbines[ii]))

    return x_opt


if __name__ == "__main__":
    # Load FLORIS and get number of turbines
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Visualize layout
    fi.vis_layout()

    # Estimate bias for every turbine's wd signal separately
    wd_bias_array = np.zeros(num_turbs)
    for ti in [0]:  # range(num_turbs):
        wd_bias_array[ti] = estimate_bias_for_turbine(ti)

    print("Estimated wind direction biases:", wd_bias_array)
    plt.show()
