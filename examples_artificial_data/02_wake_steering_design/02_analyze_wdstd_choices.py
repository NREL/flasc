import pandas as pd
import seaborn as sns
from _local_helper_functions import evaluate_optimal_yaw_angles, optimize_yaw_angles
from floris.uncertain_floris_model import UncertainFlorisModel
from matplotlib import pyplot as plt

from flasc.utilities.lookup_table_tools import get_yaw_angles_interpolant
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris


def load_floris_with_uncertainty(std_wd=0.0):
    fm, _ = load_floris()  # Load nominal floris object
    if std_wd > 0.001:
        fm = UncertainFlorisModel(fm.core.as_dict(), wd_std=std_wd)  # Load uncertainty object
    return fm


if __name__ == "__main__":
    # Define std_wd range
    std_wd_list = [0.0, 3.0, 5.0]
    result_list = []

    # Compare optimizing over different std_wd and evaluating over different std_wd values
    for std_wd_opt in std_wd_list:
        print("Optimizing yaw angles with std_wd={:.2f}".format(std_wd_opt))
        # Optimize yaw angles
        df_opt = optimize_yaw_angles(
            fm=load_floris_with_uncertainty(std_wd=std_wd_opt),
        )

        # Make an interpolant
        yaw_angle_interpolant = get_yaw_angles_interpolant(df_opt)  # Create yaw angle interpolant

        # Calculate AEP uplift
        for std_wd_eval in std_wd_list:
            print("Evalating AEP uplift with std_wd={:.2f}".format(std_wd_eval))
            AEP_baseline, AEP_opt, _ = evaluate_optimal_yaw_angles(
                fm=load_floris_with_uncertainty(std_wd=std_wd_eval),
                yaw_angle_interpolant=yaw_angle_interpolant,
            )

            # Calculate AEP uplift
            uplift = 100.0 * (AEP_opt - AEP_baseline) / AEP_baseline
            result_list.append(
                pd.DataFrame(
                    {
                        "std_wd_opt": [std_wd_opt],
                        "std_wd_eval": [std_wd_eval],
                        "AEP uplift (%)": [uplift],
                    },
                )
            )

    # Print all results to console
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    with pd.option_context("display.max_rows", None):
        print(df_result)

    # Plot as a table/colormap
    df_result = df_result.set_index(["std_wd_opt", "std_wd_eval"]).unstack()
    df_result.columns = ["std_wd_eval={:.2f}".format(p) for p in std_wd_list]
    ax = sns.heatmap(df_result, linecolor="black", linewidths=1, annot=True, fmt=".2f")
    ax.set_title("AEP uplift (%)")
    plt.tight_layout()
    plt.show()
