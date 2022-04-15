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
    # Define pP range
    pP_list = [1.0, 1.88, 2.0, 3.0]
    result_list = []

    # Compare optimizing over different pPs and evaluating over different pPs
    for pP_opt in pP_list:
        # Optimize yaw angles
        df_opt = optimize_yaw_angles(
            fi=load_floris(pP=pP_opt),
        )

        # Make an interpolant
        yaw_angle_interpolant = get_yaw_angles_interpolant(df_opt)  # Create yaw angle interpolant

        # Calculate AEP uplift
        for pP_eval in pP_list:
            AEP_baseline, AEP_opt, _ = evaluate_optimal_yaw_angles(
                fi=load_floris(pP=pP_eval),
                yaw_angle_interpolant=yaw_angle_interpolant,
            )

            # Calculate AEP uplift
            uplift = 100.0 * (AEP_opt - AEP_baseline) / AEP_baseline
            result_list.append(
                pd.DataFrame(
                    {"pP_opt": [pP_opt], "pP_eval": [pP_eval], "AEP uplift (%)": [uplift]},
                )
            )

    # Print all results to console
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    with pd.option_context('display.max_rows', None):
        print(df_result)

    # Plot as a table/colormap
    df_result = df_result.set_index(["pP_opt", "pP_eval"]).unstack()
    df_result.columns = ["pP_eval={:.2f}".format(p) for p in pP_list]
    ax = sns.heatmap(df_result, linecolor="black", linewidths=1, annot=True, fmt=".2f")
    ax.set_title("AEP uplift (%)")
    plt.tight_layout()
    plt.show()