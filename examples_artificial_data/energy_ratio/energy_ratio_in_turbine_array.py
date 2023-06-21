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

from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc.energy_ratio import energy_ratio_suite
from flasc.visualization import plot_floris_layout

from floris.tools.visualization import visualize_cut_plane
from floris.utilities import wrap_360


def load_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir, '..', 'demo_dataset', 'demo_dataset_scada_60s.ftr'
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError('Please run ./examples_artifical/demo_dataset/' +
                                'generate_demo_dataset.py before try' +
                                'ing any of the other examples.')
    df = pd.read_feather(ftr_path)
    return df


def load_floris():
    # Load the FLORIS model for the artificial wind farm
    from floris import tools as wfct
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.yaml")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


def _get_angle(fi, turbine_array):
    # Determine the geometrical angle between the upmost and downmost turbine
    # in an array. This is equal to the wind direction that maximally overlaps
    # the wake from the most upstream turbine on the most downstream turbine.
    # That wind direction is where we want to calculate the energy ratio for
    # each of the three turbines, such that we can see how the wake affects
    # the power production of each turbine in the array.
    t0 = turbine_array[0]
    t1 = turbine_array[-1]
    dy = fi.layout_y[t1] - fi.layout_y[t0]
    dx = fi.layout_x[t1] - fi.layout_x[t0]
    wd = wrap_360(270.0 - np.arctan2(dy, dx) * 180.0 / np.pi)
    return wd


def _calculate_energy_ratios(df, test_turbines, wd_bins, N=1):
    # This function calculates the energy ratio, one value, for each turbine
    # in the 'test_turbines' list. The energy ratio one value for each turbine,
    # corresponding to a single wind direction bin and wind speed bin. The
    # wind direction bin covers the wind direction region that causes maximum
    # wake overlap, thus close to the value returned by _get_angle(). Then,
    # N defines the bootstrapping sample size, defaulting to 1.

    # Load an energy ratio suite from FLASC
    s = energy_ratio_suite.energy_ratio_suite(verbose=False)

    # Designate a reference wind turbine, being the most upstream in the array
    # in our case. Thus, the energy ratio of the most upstream turbine will
    # always be 1.0, and the energy ratios of the other turbines are normalized
    # to the first turbine. Also, the first turbine is used for its wind
    # direction measurement.
    ti_ref = [test_turbines[0]]
    df = dfm.set_wd_by_turbines(df, ti_ref)
    df = dfm.set_ws_by_turbines(df, ti_ref)
    df = dfm.set_pow_ref_by_turbines(df, ti_ref)

    # We filter the data to a subset of wind speeds, from 6 to 10 m/s
    df = dfm.filter_df_by_ws(df, [6, 10])

    # Finally, we add the dataframe to the energy ratio suite.
    s.add_df(df, 'data')

    # Now, we calculate the energy ratio for each turbine for the one wind
    # direction and wind speed bin. We save those values to
    # 'results_energy_ratio'.
    results_energy_ratio = []
    for ti in test_turbines:
        # Get energy ratios
        er = s.get_energy_ratios(
            test_turbines=ti,
            ws_bins=[[6.0, 10.0]],
            wd_bins=wd_bins,
            N=N,
            percentiles=[5.0, 95.0],
            verbose=False
        )
        results_energy_ratio.append(er[0]["er_results"].loc[0])

    # Finally, combine all results into a single dataframe
    results_energy_ratio = pd.concat(results_energy_ratio, axis=1).T
    return results_energy_ratio


def plot_energy_ratios(turbine_array, results_energy_ratio, ax=None, label=None):
    # Here, we plot the energy ratios for a turbine array, where each turbine in 
    # the turbine_array has a single energy ratio value, plus a lower and upper bound
    # if the number of bootstrapping samples was larger than 1, i.e., N > 1.
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.0))

    x = range(len(results_energy_ratio))
    color = next(ax._get_lines.prop_cycler)['color']
    ax.fill_between(
        x,
        results_energy_ratio['baseline_lb'],
        results_energy_ratio['baseline_ub'],
        color=color,
        alpha=0.25
    )
    ax.plot(x, results_energy_ratio['baseline'], '-o', markersize=7, color=color, label=label)
    ax.grid(True)
    ax.set_ylabel('Energy ratio (-)')
    ax.set_xlabel("Turbine ID (-)")
    ax.set_xticks(x)
    ax.set_xticklabels(["{:02d}".format(ti) for ti in turbine_array])
    ax.legend(bbox_to_anchor=(0, 1.08, 1, 0), loc='center', ncol=3, edgecolor='w')

    plt.tight_layout()
    return ax


if __name__ == "__main__":
    # User settings: define which turbine array we want to consider. In this
    # example, turbines 2, 1 and 6 form a straight line in the wind farm
    # and therefore make a good candidate for this analysis. Note that if
    # we flip the order, then we look at the situation that is 180 deg rotated,
    # and thus with a different turbine upstream and with a different set of
    # SCADA data. Both are valid choices and both lead to different figures.
    turbine_array = [2, 1, 6]

    # We also specify the binning width. Note that we are looking at a single
    # wind direction bin, and thus setting wd_bin_width=15 means we look at all
    # measurements that fall -7.5 deg and +7.5 deg near the wind direction that
    # perfectly aligns the wake of turbine 2 with the rotor center of turbine 6.
    wd_bin_width = 15.0

    # We can apply bootstrapping for uncertainty quantification by setting
    # N_bootstrapping to a value larger than 1.
    N_bootstrapping = 50

    # Load FLORIS and load SCADA data
    fi = load_floris()
    df = load_data()

    # Note that we normalize everything in our results to the first turbine in the array
    t0 = turbine_array[0]
    df = dfm.set_wd_by_turbines(df, t0)

    # Define wind direction that perfectly aligns turbine array
    wd = _get_angle(fi, turbine_array)

    # Calculate energy ratio for narrow bin near 'wd'
    results_energy_ratio = _calculate_energy_ratios(
        df=df,
        test_turbines=turbine_array,
        wd_bins=[[wd - wd_bin_width/2.0, wd + wd_bin_width/2.0]],
        N=N_bootstrapping,
    )

    # Plot energy ratios
    ax = plot_energy_ratios(turbine_array, results_energy_ratio)

    # Also plot wake situation according to FLORIS
    plot_floris_layout(fi, plot_terrain=False)

    fig, ax = plt.subplots()
    fi.reinitialize(wind_directions=[wd], wind_speeds=[10.0])
    horizontal_plane = fi.calculate_horizontal_plane(height=90.0)
    visualize_cut_plane(horizontal_plane, ax=ax, title="Horizontal plane")
    plt.show()
