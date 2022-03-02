# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from datetime import timedelta as td
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from floris import tools as wfct
from floris.utilities import wrap_360

from flasc import (
    floris_tools as ftools,
    optimization as opt,
    utilities as fsut,
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
    # We now cross-compare the wind direction measurements between the wind
    # turbines. The offset between each two timeseries should be a constant
    # value between -360.0 and +360.0 deg. Small differences are allowable,
    # but if we estimate a significantly different offset between the two
    # timeseries for different subsets of the data, it is likely that one
    # or both turbines have had their northing calibration reset at some
    # point in the data. This complicates analysis and therefore we mark
    # data with inconsistent calibrations as faulty.

    # We split the dataset up into time lengths of [bias_timestep] and
    # estimate the offset between neighboring turbines for each data subset.
    # If the offset is inconsistent over the various subsets, we mark the
    # data as faulty.
    #
    # User settings
    bias_timestep = td(days=120)
    nan_thrshld = 0.50  # Max. ratio of nan values to estimate offset

    # Create an output directory
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(
        root_path,
        "data",
        "05_preliminary_fault_analysis",
        "compare_wd_calibration_between_neighboring_turbines",
    )
    os.makedirs(out_path, exist_ok=True)

    # Load data and extract info
    df = load_data()
    fi = load_floris()
    num_turbines = len(fi.layout_x)
    turbine_list = range(num_turbines)

    # Set up time_array and split into chunks
    time_array = np.array(df["time"])
    dt_est = fsut.estimate_dt(time_array)
    t = pd.to_datetime(time_array[0])
    idx_chunks = list()
    while t < pd.to_datetime(time_array[-1]):
        t_next = t + bias_timestep
        idx_chunks.append((time_array >= t) & (time_array < t_next))
        t = t_next
    N_chnks = len(idx_chunks)

    # Get reference turbines and create placeholder dataframes
    N_rt = 5  # Number of reference turbines
    turbs_ref_list = [[] for _ in turbine_list]
    bias_output_list = [[] for _ in turbine_list]
    for ti in turbine_list:
        turbs_in_radius = ftools.get_turbs_in_radius(
            fi.layout_x,
            fi.layout_y,
            ti,
            max_radius=1.0e9,
            include_itself=False,
            sort_by_distance=True,
        )
        turbs_ref_list[ti] = turbs_in_radius[0:N_rt]
        bias_output_list[ti] = pd.DataFrame(
            data=np.reshape(np.full(N_chnks * N_rt, np.nan), (N_chnks, N_rt)),
            columns=["T%03d" % ti_ref for ti_ref in turbs_ref_list[ti]],
        )

    for ti in turbine_list:
        print("Matching curves for turbine %03d..." % ti)
        ref_turb_subset = turbs_ref_list[ti]
        ref_turb_subset = [
            r
            for r in ref_turb_subset
            if all(np.isnan(bias_output_list[ti]["T%03d" % r]))
        ]

        for ii, idx_chunk in enumerate(idx_chunks):
            df_subset = df.loc[idx_chunk]

            for ti_ref in ref_turb_subset:
                wd_ref = np.array(df_subset["wd_%03d" % (ti_ref)])
                wd_turb = np.array(df_subset["wd_%03d" % ti])

                wd_ref = wrap_360(wd_ref)
                wd_turb = wrap_360(wd_turb)

                if sum(np.isnan(wd_turb)) / len(wd_turb) < nan_thrshld:
                    dx_opt, J_opt = opt.match_y_curves_by_offset(
                        yref=wd_ref, 
                        ytest=wd_turb,
                        angle_wrapping=True,
                    )
                    # fig, ax = plt.subplots()
                    # ax.plot(wrap_360(wd_turb_sub - dx_opt), 'o')
                    # ax.plot(wd_ref_sub, 'o')
                    # plt.show()
                else:
                    dx_opt = np.nan
                bias_output_list[ti].loc[ii, "T%03d" % ti_ref] = dx_opt
                if ti in turbs_ref_list[ti_ref]:
                    bias_output_list[ti_ref].loc[ii, "T%03d" % ti] = -dx_opt
                # print('Estimated dx_opt = %.3f, J_opt = %.3f' % (dx_opt, J_opt))
                # plt.show()

        print(bias_output_list[ti])
        # Define path for output figures
        # root_path = os.path.dirname(os.path.abspath(__file__))
        # out_path = os.path.join(root_path, 'data', '05_preliminary_fault_analysis')
        # os.makedirs(out_path, exist_ok=True)

    # Find turbines where dx barely changes (low variance)
    turb_is_clean = ["bad" for _ in turbine_list]
    for ti in turbine_list:
        df_out = bias_output_list[ti]
        for ti_ref in turbs_ref_list[ti]:
            # If synced by less than 5 degrees std, then both OK
            if np.nanstd(df_out["T%03d" % ti_ref]) < 5.0:
                turb_is_clean[ti] = "clean"
                turb_is_clean[ti_ref] = "clean"
            else:
                if (turb_is_clean[ti] == "clean") & (
                    turb_is_clean[ti_ref] == "clean"
                ):
                    turb_is_clean[ti] = "disputed"
                    turb_is_clean[ti_ref] = "disputed"

    for ti in turbine_list:
        bias_output_list[ti].to_csv(
            os.path.join(out_path, "estimated_offsets_%03d.csv" % ti)
        )
        if turb_is_clean[ti] == "disputed":
            print(
                "Turbine %03d may or may not have jumps in WD measurement calibration. [DISPUTED]"
                % ti
            )
        elif turb_is_clean[ti] == "clean":
            print(
                "Turbine %03d seems to have no jumps in its WD measurement calibration. [CLEAN]"
                % ti
            )
        elif turb_is_clean[ti] == "bad":
            print(
                "Turbine %03d seems to have one or multiple jumps in its WD measurement calibration. [BAD]"
                % ti
            )

    # Plot layout and colormap
    fig, ax = plt.subplots(figsize=(14, 5))
    for ti in turbine_list:
        if turb_is_clean[ti] == "clean":
            clr = "green"
        elif turb_is_clean[ti] == "bad":
            clr = "red"
        elif turb_is_clean[ti] == "disputed":
            clr = "orange"

        ax.plot(
            fi.layout_x[ti],
            fi.layout_y[ti],
            "o",
            markersize=15,
            markerfacecolor=clr,
            markeredgewidth=0.0,
        )
        ax.text(
            fi.layout_x[ti] + 100,
            fi.layout_y[ti],
            "T%03d (%s)" % (ti, turb_is_clean[ti]),
            color="black",
        )
    fig.tight_layout()

    fig_out = os.path.join(out_path, "calibration_quality_by_layout.png")
    plt.savefig(fig_out, dpi=300)
    print("Calibration figure saved to {:s}.".format(fig_out))

    plt.show()
