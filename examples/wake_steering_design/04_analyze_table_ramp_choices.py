# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from flasc.wake_steering.lookup_table_tools import get_yaw_angles_interpolant
from _local_helper_functions import load_floris, optimize_yaw_angles, evaluate_optimal_yaw_angles


if __name__ == "__main__":
    # Define std_wd range
    ramp_up_list = [[3.0, 4.0], [4.0, 5.0], [4.0, 6.0], [4.0, 8.0]]
    ramp_down_list = [[9.0, 11.0], [10.0, 11.0], [10.0, 12.0], [12.0, 14.0]]
    result_list = []

    # Optimize yaw angles nominally
    df_opt = optimize_yaw_angles()

    for ramp_up_ws in ramp_up_list:
        for ramp_down_ws in ramp_down_list:
            # Make an interpolant
            yaw_angle_interpolant = get_yaw_angles_interpolant(
                df_opt,
                ramp_up_ws=ramp_up_ws,
                ramp_down_ws=ramp_down_ws
                )

            # Calculate AEP uplift
            AEP_baseline, AEP_opt, _ = evaluate_optimal_yaw_angles(
                yaw_angle_interpolant=yaw_angle_interpolant,
            )

            # Calculate AEP uplift
            uplift = 100.0 * (AEP_opt - AEP_baseline) / AEP_baseline
            result_list.append(
                pd.DataFrame(
                    {
                        "ramp_up_ws": [str(ramp_up_ws)],
                        "ramp_down_ws": [str(ramp_down_ws)],
                        "AEP uplift (%)": [uplift]
                    },
                )
            )

    # Print all results to console
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    with pd.option_context('display.max_rows', None):
        print(df_result)

    # Plot as a table/colormap
    df_result = df_result.set_index(["ramp_up_ws", "ramp_down_ws"]).unstack()
    df_result.columns = ["ramp_down_ws={}".format(p) for p in ramp_down_list]
    ax = sns.heatmap(df_result, linecolor="black", linewidths=1, annot=True, fmt=".2f")
    ax.set_title("AEP uplift (%)")
    plt.tight_layout()
    plt.show()
