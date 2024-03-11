import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.analysis import total_uplift as tup
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris
from flasc.visualization import plot_binned_mean_and_ci, plot_layout_with_waking_directions

if __name__ == "__main__":
    # Generate the data as in example 05_wake_steering_example.py

    # Construct a simple 3-turbine wind farm with a
    # Reference turbine (0)
    # Controlled turbine (1)
    # Downstream turbine (2)
    np.random.seed(0)

    fi, _ = load_floris()
    fi.set(layout_x=[0, 0, 5 * 126], layout_y=[5 * 126, 0, 0])

    # Show the wind farm
    plot_layout_with_waking_directions(fi)

    # Create a time history of points where the wind speed and wind
    # direction step different combinations
    ws_points = np.arange(5.0, 10.0, 1.0)
    wd_points = np.arange(
        250.0,
        290.0,
        1.0,
    )
    num_points_per_combination = 5  # 5 # How many "seconds" per combination

    ws_array = []
    wd_array = []
    for ws in ws_points:
        for wd in wd_points:
            for i in range(num_points_per_combination):
                ws_array.append(ws)
                wd_array.append(wd)
    t = np.arange(len(ws_array))

    fig, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].plot(t, ws_array, label="Wind Speed")
    axarr[0].set_ylabel("m/s")
    axarr[0].legend()
    axarr[0].grid(True)
    axarr[1].plot(t, wd_array, label="Wind Direction")
    axarr[1].set_ylabel("deg")
    axarr[1].legend()
    axarr[1].grid(True)

    # Compute the power of the second turbine for two cases
    # Baseline: The front turbine is aligned to the wind
    # WakeSteering: The front turbine is yawed 25 deg
    fi.set(wind_speeds=ws_array, wind_directions=wd_array)
    fi.run()
    power_baseline_ref = fi.get_turbine_powers().squeeze()[:, 0].flatten()
    power_baseline_control = fi.get_turbine_powers().squeeze()[:, 1].flatten()
    power_baseline_downstream = fi.get_turbine_powers().squeeze()[:, 2].flatten()

    yaw_angles = np.zeros([len(t), 3]) * 25
    yaw_angles[:, 1] = 25  # Set control turbine yaw angles to 25 deg
    fi.set(yaw_angles=yaw_angles)
    fi.run()
    power_wakesteering_ref = fi.get_turbine_powers().squeeze()[:, 0].flatten()
    power_wakesteering_control = fi.get_turbine_powers().squeeze()[:, 1].flatten()
    power_wakesteering_downstream = fi.get_turbine_powers().squeeze()[:, 2].flatten()

    # Build up the data frames needed for energy ratio suite
    df_baseline = pd.DataFrame(
        {
            "wd": wd_array,
            "ws": ws_array,
            "pow_ref": power_baseline_ref,
            "pow_000": power_baseline_ref,
            "pow_001": power_baseline_control,
            "pow_002": power_baseline_downstream,
        }
    )

    df_wakesteering = pd.DataFrame(
        {
            "wd": wd_array,
            "ws": ws_array,
            "pow_ref": power_wakesteering_ref,
            "pow_000": power_wakesteering_ref,
            "pow_001": power_wakesteering_control,
            "pow_002": power_wakesteering_downstream,
        }
    )

    # Create alternative versions of each of the above dataframes
    # where the wd/ws are perturbed by noise
    df_baseline_noisy = pd.DataFrame(
        {
            "wd": wd_array + np.random.randn(len(wd_array)) * 2,
            "ws": ws_array,  #  + np.random.randn(len(ws_array)),
            "pow_ref": power_baseline_ref,
            "pow_000": power_baseline_ref,
            "pow_001": power_baseline_control,
            "pow_002": power_baseline_downstream,
        }
    )

    df_wakesteering_noisy = pd.DataFrame(
        {
            "wd": wd_array + np.random.randn(len(wd_array)) * 2,
            "ws": ws_array,  # + np.random.randn(len(ws_array)),
            "pow_ref": power_wakesteering_ref,
            "pow_000": power_wakesteering_ref,
            "pow_001": power_wakesteering_control,
            "pow_002": power_wakesteering_downstream,
        }
    )

    # Use the function plot_binned_mean_and_ci to show the noise in wind speed
    fig, ax = plt.subplots(1, 1, sharex=True)
    ws_edges = np.append(ws_points - 0.5, ws_points[-1] + 0.5)
    plot_binned_mean_and_ci(df_baseline.ws, df_baseline_noisy.ws, ax=ax, x_edges=ws_edges)
    ax.set_xlabel("Wind Speed (m/s) [Baseline]")
    ax.set_ylabel("Wind Speed (m/s) [Baseline (Noisy)]")
    ax.grid(True)

    # Calculate the energy uplift in the downstream turbine
    # first directly from the data
    p_change_data = (
        100
        * (df_wakesteering.pow_002.sum() - df_baseline.pow_002.sum())
        / df_baseline.pow_002.sum()
    )

    p_change_data_noisy = (
        100
        * (df_wakesteering_noisy.pow_002.sum() - df_baseline_noisy.pow_002.sum())
        / df_baseline_noisy.pow_002.sum()
    )

    print(" ")
    print("=======Direct Calculation======")
    print(
        f"The power increase in the turbine is {p_change_data:.3}% in the"
        f" non-noisy data and {p_change_data_noisy:.3}% in the noisy data"
    )

    # Calculate the uplift on the non-noisy data
    er_in = EnergyRatioInput(
        [df_baseline, df_wakesteering], ["baseline", "wake_steering"], num_blocks=1
    )

    total_uplift_result = tup.compute_total_uplift(
        er_in,
        ref_turbines=[0],
        test_turbines=[2],
        use_predefined_wd=True,
        use_predefined_ws=True,
        weight_by="min",
        uplift_pairs=["baseline", "wake_steering"],
        uplift_names=["uplift"],
    )

    uplift_non_noisy = total_uplift_result["uplift"]["energy_uplift_ctr_pc"]

    # Calculate the uplift on the noisy data
    er_in = EnergyRatioInput(
        [df_baseline_noisy, df_wakesteering_noisy], ["baseline", "wake_steering"], num_blocks=1
    )

    total_uplift_result_noisy = tup.compute_total_uplift(
        er_in,
        ref_turbines=[0],
        test_turbines=[2],
        use_predefined_wd=True,
        use_predefined_ws=True,
        weight_by="min",
        uplift_pairs=["baseline", "wake_steering"],
        uplift_names=["uplift"],
    )

    uplift_noisy = total_uplift_result_noisy["uplift"]["energy_uplift_ctr_pc"]
    print("=======Total Uplift======")
    print(
        f"The uplift calculated using the compute_total_uplift module "
        f" is {uplift_non_noisy:.3}% in the"
        f" non-noisy data and {uplift_noisy:.3}% in the noisy data"
    )

    # Recompute using bootstrapping to understand uncertainty bounds
    # Calculate the uplift on the non-noisy data
    er_in = EnergyRatioInput(
        [df_baseline, df_wakesteering],
        ["baseline", "wake_steering"],
        num_blocks=df_baseline.shape[0],  # Use N blocks to do non-block boostrapping
    )

    total_uplift_result = tup.compute_total_uplift(
        er_in,
        ref_turbines=[0],
        test_turbines=[2],
        use_predefined_wd=True,
        use_predefined_ws=True,
        weight_by="min",
        uplift_pairs=["baseline", "wake_steering"],
        uplift_names=["uplift"],
        N=100,
    )

    uplift_non_noisy_lb = total_uplift_result["uplift"]["energy_uplift_lb_pc"]
    uplift_non_noisy_ub = total_uplift_result["uplift"]["energy_uplift_ub_pc"]

    # Calculate the uplift on the noisy data
    er_in = EnergyRatioInput(
        [df_baseline_noisy, df_wakesteering_noisy],
        ["baseline", "wake_steering"],
        num_blocks=df_baseline.shape[0],  # Use N blocks to do non-block boostrapping
    )

    total_uplift_result_noisy = tup.compute_total_uplift(
        er_in,
        ref_turbines=[0],
        test_turbines=[2],
        use_predefined_wd=True,
        use_predefined_ws=True,
        weight_by="min",
        uplift_pairs=["baseline", "wake_steering"],
        uplift_names=["uplift"],
        N=100,
    )

    uplift_noisy_lb = total_uplift_result_noisy["uplift"]["energy_uplift_lb_pc"]
    uplift_noisy_ub = total_uplift_result_noisy["uplift"]["energy_uplift_ub_pc"]

    print("=======Bootstrap Confidence Invervals======")
    print(
        f"The 90% confidence interval for the non-noisy data is: "
        f"({uplift_non_noisy_lb:.2f},{uplift_non_noisy_ub:.2f})"
    )

    print(
        f"The 90% confidence interval for the noisy data is: "
        f"({uplift_noisy_lb:.2f},{uplift_noisy_ub:.2f})"
    )
