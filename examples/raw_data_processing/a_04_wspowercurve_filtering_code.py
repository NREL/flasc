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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris.tools import floris_interface as wfct

from flasc.dataframe_operations import dataframe_filtering as dff
from flasc.turbine_analysis import ws_pow_filtering as wspcf
from flasc import time_operations as top
   

def load_floris():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.FlorisInterface(
        os.path.join(root_path, "..", "demo_dataset", "demo_floris_input.json")
    )
    return fi


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "03_sensor_faults_filtered")
    df = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    return df


if __name__ == "__main__":
    # In this script, we filter the data by looking at the wind-speed vs.
    # power curves. We detect points that are too far away from the median
    # power curve and identify them as faulty. Furthermore, we detect
    # curtailment periods in this manner, which are often present in
    # historical data.
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, "data", "04_wspowcurve_filtered")
    fig_path = os.path.join(out_path, "figures")

    # Load data
    df = load_data()

    # Load the FLORIS model for the wind farm. This is not used for anything
    # besides plotting the floris-predicted wind speed-power curve on top
    # of the actual data.
    fi = load_floris()

    # Downsample data. Not necessary here, but can be useful if we have 1 Hz
    # data available. Namely, it's hard to detect outliers on such a high
    # resolution. Instead, we are better off downsampling the data to 60s or
    # even 600s and filter the data based on decisions there. The following
    # downsampled dataframe should then be inserted into the wind speed power
    # curve filtering class. Mapping the filtering back to the high-resolution
    # data is done by a couple lines of code as found at the end of this
    # script.
    #
    # df_movavg, data_indices_mapping = top.df_movingaverage(
    #     df_in=df_1s,
    #     cols_angular=[
    #         c for c in df_1s.columns if (
    #             ("vane_" in c) or
    #             ("yaw_" in c) or
    #             ("wd_" in c) or
    #             ("direction" in c)
    #         )
    #     ],
    #     window_width=td(seconds=600),
    #     calc_median_min_max_std=False,
    #     return_index_mapping=True,
    # )

    # Create output directories
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    # Initialize the wind-speed power curve filtering class
    turbine_list = "all"
    # turbine_list = [5]  # Can also look at specific turbines
    ws_pow_filtering = wspcf.ws_pw_curve_filtering(
        df=df, turbine_list=turbine_list, rated_powers=5000.0
    )

    # Add a window: all data to the left or right of this window is bad
    # This is an easy way to remove curtailment if the default filtering
    # methods do not or insufficiently pick up these outliers.
    ws_pow_filtering.window_add(
        ws_range=[0.0, 10.2],
        pow_range=[3100.0, 3200.0],
        axis=0,
        turbines="all",
    )
    ws_pow_filtering.filter_by_windows()

    # Now filter by deviations from the median power curve
    ws_pow_filtering.filter_by_power_curve()

    # Extract and save turbine power curves estimated from the data
    df_pow_curve = ws_pow_filtering.pw_curve_df
    df_pow_curve.to_csv(os.path.join(out_path, "power_curves.csv"))

    # Plot and save data for current dataframe
    ws_pow_filtering.plot_outliers_vs_time(save_path=fig_path)
    ws_pow_filtering.plot(fi=fi, save_path=fig_path)

    # # Map filtering back to 1s data [disabled here]
    # print("Mapping filtering from moving-average data back to 1s data.")
    # df_1s_filt = df_1s.copy()
    # df_filters = ws_pow_filtering.df_filters
    # for ti in ws_pow_filtering.turbine_list:
    #     bad_id_res = (df_filters[ti]["status"] == False)  # Resampled data
    #     bad_id_raw = data_indices_mapping[bad_id_res, :] # Raw data
    #     bad_id_raw = np.unique(bad_id_raw.flatten())
    #     bad_id_raw = bad_id_raw[bad_id_raw > -0.001]  # Remove -1 placeholders

    #     df_1s_filt = dff.df_mark_turbdata_as_faulty(
    #         df=df_1s_filt,
    #         cond=bad_id_raw,
    #         turbine_list=[ti],
    #     )
    #
    # fn = os.path.join(out_path, "scada_data_1s.ftr")
    # print("Saving processed dataframe to {:s}.".format(fn))
    # df_1s_filt.to_feather(fn)

    fn = os.path.join(out_path, "scada_data_60s.ftr")
    ws_pow_filtering.save_df(fn)
    print("Saved processed dataframe to {:s}.".format(fn))

    plt.show()
