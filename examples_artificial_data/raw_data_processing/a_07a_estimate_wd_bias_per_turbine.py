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
from datetime import timedelta as td
import warnings as wn

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd

from floris.utilities import wrap_360
from floris import tools as wfct

from flasc.energy_ratio import energy_ratio_wd_bias_estimation as best
from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import time_operations as fto
from flasc import optimization as flopt
from flasc import floris_tools as ftools


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
    data_path = os.path.join(root_path, "data", "06_time_synchronization")
    df_scada = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))

    # Downsample to 10 minute averages to speed up things
    cols_angular = [c for c in df_scada if (("wd_" in c) or ("yaw_" in c))]
    df_scada = fto.df_downsample(
        df_scada,
        cols_angular=cols_angular,
        window_width=td(seconds=600),
    )
    return df_scada


def get_bias_for_single_turbine(ti, opt_search_range=[-180.0, 180.0]):
    print("Initializing wd bias estimator object for turbine %03d..." % ti)

    # Load the (downsampled) SCADA data
    df = load_data()

    # Load the FLORIS model
    fi = load_floris()

    # Calculate which turbines are upstream for every wind direction
    df_upstream = ftools.get_upstream_turbs_floris(fi, wd_step=2.0)

    # Specify output directory
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "07_wdbias_filtered_data")
    os.makedirs(out_path, exist_ok=True)

    # We assign the total datasets "true" wind direction as equal to the wind
    # direction of the turbine which we want to perform northing calibration
    # on. In this case, turbine 'ti'.
    df = dfm.set_wd_by_turbines(df, [ti])

    # We define a function that calculates the freestream wind speed based
    # on a dataframe that is inserted. It does this based on knowing which
    # turbines are upstream for what wind directions, and then knowledge
    # of what the wind direction is for every row in the dataframe. However,
    # since the shift the "true" wind direction many times to estimate the
    # northing bias, we cannot precalculate this. It changes with every
    # northing bias guess. Hence, we must insert a function.
    def _set_ws_fun(df):
        return dfm.set_ws_by_upstream_turbines_in_radius(
            df=df,
            df_upstream=df_upstream,
            turb_no=ti,
            x_turbs=fi.layout_x,
            y_turbs=fi.layout_y,
            max_radius=5000.0,
            include_itself=True,
        )

    # We similarly define a function that calculates the reference power. This
    # is typically the power production of one or multiple upstream turbines.
    # Here, we assume it is the average power production of all upstream
    # turbines. Which turbines are upstream depends on the wind direction.
    def _set_pow_ref_fun(df):
        return dfm.set_pow_ref_by_upstream_turbines_in_radius(
            df=df,
            df_upstream=df_upstream,
            turb_no=ti,
            x_turbs=fi.layout_x,
            y_turbs=fi.layout_y,
            max_radius=5000.0,
            include_itself=True,
        )

    # Now we calculate a grid of FLORIS solutions. Since our estimated SCADA
    # data changes as we shift its wind direction, the predicted solutions
    # according to FLORIS will also change. Therefore, we precalculate a grid
    # of FLORIS solutions and insert that into the bias estimation class.
    fout_df_fi_approx = os.path.join(root_path, "df_fi_approx.ftr")
    if os.path.exists(fout_df_fi_approx):
        df_approx = pd.read_feather(fout_df_fi_approx)
    else:
        df_approx = ftools.calc_floris_approx_table(
            fi=fi,
            wd_array=np.arange(0., 360., 3.0),
            ws_array=np.arange(6.0, 10.01, 1.0),
            ti_array=None,
        )
        df_approx.to_feather(fout_df_fi_approx)

    # We now have the reference power productions specified, being equal to
    # the mean power production of all turbines upstream. We also need to
    # define a test power production, which should be waked at least part of
    # the time so that we can match it with our FLORIS predictions. Here, we
    # calculate the energy ratios for the 3 turbines closest to the turbine
    # from which we take the wind direction measurement ('ti').
    turbines_sorted_by_distance = ftools.get_turbs_in_radius(
        x_turbs=fi.layout_x,
        y_turbs=fi.layout_y,
        turb_no=ti,
        max_radius=1.0e9,
        include_itself=False,
        sort_by_distance=True,
    )
    test_turbines = turbines_sorted_by_distance[0:3]

    # Now, we have all information set up and we can initialize the northing
    # bias estimation class.
    fsc = best.bias_estimation(
        df=df,
        df_fi_approx=df_approx,
        test_turbines_subset=test_turbines,
        df_ws_mapping_func=_set_ws_fun,
        df_pow_ref_mapping_func=_set_pow_ref_fun,
    )

    # We create an empty dataframe to save our estimated northing bias
    # corrections to.
    df_bias = pd.DataFrame()

    # We can save the energy ratio curves for every iteration in the
    # optimization process. This is useful for debugging. However, it also
    # significantly slows down the estimation process. We disable it by
    # default by assigning it 'None'.
    plot_iter_path = None  # Disabled, useful for debugging but slow
    # plot_iter_path = os.path.join(out_path, "opt_iters_ti%03d" % ti)

    # Now estimate the wind direction bias while catching warning messages
    # that do not really inform but do pollute the console.
    with wn.catch_warnings():
        wn.filterwarnings(action="ignore", message="All-NaN slice encountered")

        # Estimate bias for the entire time range, from start to end of
        # dataframe, for wind speeds in region II of turbine operation, with
        # in steps of 3.0 deg (wd) and 5.0 m/s (ws). We search over the entire
        # range from -180.0 deg to +180.0 deg, in steps of 5.0 deg. This has
        # appeared to be a good stepsize empirically.
        wd_bias, _ = fsc.estimate_wd_bias(
            time_mask=None,  # For entire dataset
            ws_mask=(6.0, 10.0),
            er_wd_step=3.0,
            er_ws_step=5.0,
            er_wd_bin_width=3.0,
            er_N_btstrp=1,
            opt_search_brute_dx=5.0,
            opt_search_range=opt_search_range,
            plot_iter_path=plot_iter_path
        )
        wd_bias = float(wd_bias[0])  # Convert to float

    # Save the estimated wind direction bias to a local dataframe
    df_bias = df_bias.append(
        {
            "ref_turbine": int(ti),
            "test_turbines": test_turbines,
            "wd_bias": wd_bias,
        },
        ignore_index=True,
    )
    df_bias["ref_turbine"] = df_bias["ref_turbine"].astype(int)

    # Print progress to console
    print("Turbine {}. estimated bias = {} deg.".format(ti, wd_bias))

    # Produce and save calibrated/corrected energy ratio figures
    fn = os.path.join(out_path, "ti{:03d}_energyratios".format(ti))
    fsc.plot_energy_ratios(save_path=fn)
    print("Calibrated energy ratio figures saved to {:s}.".format(fn))

    # Finally, return the estimated wind direction bias
    return df_bias


if __name__ == "__main__":
    # Set up output directory
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "07_wdbias_filtered_data")
    os.makedirs(out_path, exist_ok=True)

    # Load the FLORIS model and extract the number of turbines
    fi = load_floris()
    df = load_data()
    num_turbs = len(fi.layout_x)
    turbine_list = range(num_turbs)

    # We can now calculate the wind direction bias for every turbine
    # separately. That is the most reliable but also the slowest way to
    # calibrate every turbine to true north. A smarter way is to calibrate
    # northing for a single turbine. Then, we can calculate the shift
    # in wind directions between every turbine and our one calibrated turbine.
    # Correcting for that drift should bring us very close to the true north
    # for those turbines. This faster approach is what we do here.

    # Do a detailed estimation for our first turbine
    ti = turbine_list[0]
    df_out = get_bias_for_single_turbine(ti, (-180.0, 180.0))
    wd_ref = wrap_360(df["wd_{:03d}".format(ti)] - float(df_out["wd_bias"]))

    # Now use this knowledge to estimate bias for every other turbine
    df_bias_list = [df_out]
    for ti in turbine_list[1::]:
        # Calculate the offset between this turbine's wind direction and that
        # of the calibrated wind direction of our first turbine. This offset
        # is very likely to be the bias or close to the bias in this turbine's
        # northing.
        wd_test = df["wd_{:03d}".format(ti)]
        x0, _ = flopt.match_y_curves_by_offset(
            wd_ref,
            wd_test,
            dy_eval=np.arange(-180.0, 180.0, 2.0),
            angle_wrapping=True
        )

        # Then, we refine this first guess by evaluating the cost function
        # at [-5.0, 0.0, 5.0] deg around x0, and let the optimizer
        # converge.
        x_search_bounds = np.round(x0) + np.array([-5.0, 5.0])

        # Save the results to a dataframe
        df_out = get_bias_for_single_turbine(ti, x_search_bounds)
        df_bias_list.append(df_out)
        print(" ")

    df_concat = pd.concat(df_bias_list, axis=0)
    df_concat.to_csv(os.path.join(out_path, "df_bias.csv"))
