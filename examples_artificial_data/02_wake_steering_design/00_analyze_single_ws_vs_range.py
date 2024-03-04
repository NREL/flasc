import numpy as np
from _local_helper_functions import evaluate_optimal_yaw_angles, optimize_yaw_angles
from matplotlib import pyplot as plt

from flasc.utilities.lookup_table_tools import get_yaw_angles_interpolant
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris
from flasc.visualization import plot_floris_layout, plot_layout_with_waking_directions
from flasc.yaw_optimizer_visualization import (
    plot_offsets_wd,
    plot_offsets_wswd_heatmap,
    plot_uplifts_by_atmospheric_conditions,
)

if __name__ == "__main__":
    # Load FLORIS model and plot layout (and additional information)
    fi, _ = load_floris()
    plot_floris_layout(fi)
    plot_layout_with_waking_directions(fi, limit_dist_D=5, limit_num=3)

    # Compare optimizing over all wind speeds vs. optimizing over a single wind speed
    AEP_baseline_array = []
    AEP_opt_array = []
    df_out_array = []
    for opt_ws_array in [[8.0], np.arange(5.0, 12.0, 1.0)]:
        # Optimize yaw angles
        df_opt = optimize_yaw_angles(opt_wind_speeds=opt_ws_array)

        # Make an interpolant
        yaw_angle_interpolant = get_yaw_angles_interpolant(df_opt)  # Create yaw angle interpolant

        # Calculate AEP uplift
        AEP_baseline, AEP_opt, df_out = evaluate_optimal_yaw_angles(
            yaw_angle_interpolant=yaw_angle_interpolant,
        )

        # Collect solutions, save to a list
        AEP_baseline_array.append(AEP_baseline)
        AEP_opt_array.append(AEP_opt)
        df_out_array.append(df_out.copy())

    # Vizualize optimal wake steering schedule for all wind speeds
    # for a single turbine (index 2)
    ax, cbar = plot_offsets_wswd_heatmap(df_offsets=df_opt, turb_id=2)
    ax.set_title("T02 offset schedule")

    ax = plot_offsets_wd(df_offsets=df_opt, turb_id=2, ws_plot=[5, 12], alpha=0.5)
    ax = plot_offsets_wd(
        df_offsets=df_opt, turb_id=2, ws_plot=7.0, color="C0", label="7.0 m/s", ax=ax
    )
    ax.set_title("T02 offset schedule")
    ax.legend()

    # Calculate AEP uplifts
    uplift_one_ws = 100.0 * (AEP_opt_array[0] - AEP_baseline_array[0]) / AEP_baseline_array[0]
    uplift_multi_ws = 100.0 * (AEP_opt_array[1] - AEP_baseline_array[1]) / AEP_baseline_array[1]
    print("\n\n =====================================================")
    print("AEP uplift with one wind speed optimization: {:.3f} %".format(uplift_one_ws))
    print("AEP uplift with all wind speeds optimization: {:.3f} %".format(uplift_multi_ws))

    # Plot uplifts for either solution across wind directions and speeds
    plot_uplifts_by_atmospheric_conditions(
        df_out_array, labels=["Opt. over one WS", "Opt. over all WS"]
    )

    plt.show()
