import pandas as pd
import seaborn as sns
from _local_helper_functions import evaluate_optimal_yaw_angles, optimize_yaw_angles
from matplotlib import pyplot as plt

from flasc.utilities.lookup_table_tools import get_yaw_angles_interpolant

if __name__ == "__main__":
    # Define turbulence intensity range
    ti_list = [0.06, 0.095, 0.130]
    result_list = []

    # Compare optimizing and evaluating over different turbulence intensities
    for ti_opt in ti_list:  # TODO: with Floris v4, could optimize all at once
        print("Optimizing yaw angles with turbulence_intensity={:.2f}".format(ti_opt))
        # Optimize yaw angles
        df_opt = optimize_yaw_angles(opt_turbulence_intensities=ti_opt)

        # Make an interpolant
        yaw_angle_interpolant = get_yaw_angles_interpolant(df_opt)  # Create yaw angle interpolant

        # Calculate AEP uplift
        for ti_eval in ti_list:
            AEP_baseline, AEP_opt, _ = evaluate_optimal_yaw_angles(
                yaw_angle_interpolant=yaw_angle_interpolant,
                eval_ti=ti_eval,
            )

            # Calculate AEP uplift
            uplift = 100.0 * (AEP_opt - AEP_baseline) / AEP_baseline
            result_list.append(
                pd.DataFrame(
                    {"ti_opt": [ti_opt], "ti_eval": [ti_eval], "AEP uplift (%)": [uplift]},
                )
            )

    # Print all results to console
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    with pd.option_context("display.max_rows", None):
        print(df_result)

    # Plot as a table/colormap
    df_result = df_result.set_index(["ti_opt", "ti_eval"]).unstack()
    df_result.columns = ["ti_eval={:.2f}".format(p) for p in ti_list]
    ax = sns.heatmap(df_result, linecolor="black", linewidths=1, annot=True, fmt=".2f")
    ax.set_title("AEP uplift (%)")
    plt.tight_layout()
    plt.show()
