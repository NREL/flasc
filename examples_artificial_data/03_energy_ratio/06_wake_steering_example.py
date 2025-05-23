import floris.layout_visualization as layoutviz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from flasc import FlascDataFrame
from flasc.analysis import energy_ratio as er
from flasc.analysis.analysis_input import AnalysisInput
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris
from flasc.visualization import plot_binned_mean_and_ci

if __name__ == "__main__":
    # Construct a simple 3-turbine wind farm with a
    # Reference turbine (0)
    # Controlled turbine (1)
    # Downstream turbine (2)
    np.random.seed(0)

    fm, _ = load_floris()
    fm.set(layout_x=[0, 0, 5 * 126], layout_y=[5 * 126, 0, 0])

    # Show the wind farm
    ax = layoutviz.plot_turbine_points(fm)
    layoutviz.plot_turbine_labels(fm, ax=ax)
    layoutviz.plot_waking_directions(fm, ax=ax)
    ax.grid()
    ax.set_xlabel("x coordinate [m]")
    ax.set_ylabel("y coordinate [m]")

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
    fm.set(
        wind_speeds=ws_array,
        wind_directions=wd_array,
        turbulence_intensities=0.06 * np.ones_like(ws_array),
    )
    fm.run()
    power_baseline_ref = fm.get_turbine_powers().squeeze()[:, 0].flatten()
    power_baseline_control = fm.get_turbine_powers().squeeze()[:, 1].flatten()
    power_baseline_downstream = fm.get_turbine_powers().squeeze()[:, 2].flatten()

    yaw_angles = np.zeros([len(t), 3]) * 25
    yaw_angles[:, 1] = 25  # Set control turbine yaw angles to 25 deg
    fm.set(yaw_angles=yaw_angles)
    fm.run()
    power_wakesteering_ref = fm.get_turbine_powers().squeeze()[:, 0].flatten()
    power_wakesteering_control = fm.get_turbine_powers().squeeze()[:, 1].flatten()
    power_wakesteering_downstream = fm.get_turbine_powers().squeeze()[:, 2].flatten()

    # Build up the data frames needed for energy ratio suite
    df_baseline = FlascDataFrame(
        {
            "wd": wd_array,
            "ws": ws_array,
            "pow_ref": power_baseline_ref,
            "pow_000": power_baseline_ref,
            "pow_001": power_baseline_control,
            "pow_002": power_baseline_downstream,
        }
    )

    df_wakesteering = FlascDataFrame(
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
    df_baseline_noisy = FlascDataFrame(
        {
            "wd": wd_array + np.random.randn(len(wd_array)) * 2,
            "ws": ws_array + np.random.randn(len(ws_array)),
            "pow_ref": power_baseline_ref,
            "pow_000": power_baseline_ref,
            "pow_001": power_baseline_control,
            "pow_002": power_baseline_downstream,
        }
    )

    df_wakesteering_noisy = FlascDataFrame(
        {
            "wd": wd_array + np.random.randn(len(wd_array)) * 2,
            "ws": ws_array + np.random.randn(len(ws_array)),
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

    # Make a color palette that visually links the nominal and noisy data sets together
    color_palette = sns.color_palette("Paired", 4)[::-1]

    # Initialize the energy ratio input object
    a_in = AnalysisInput(
        [df_baseline, df_wakesteering, df_baseline_noisy, df_wakesteering_noisy],
        ["Baseline", "WakeSteering", "Baseline (Noisy)", "WakeSteering (Noisy)"],
    )

    # Calculate and plot the energy ratio for the downstream turbine [2]
    # With respect to reference turbine [0]
    # datasets with uncertainty quantification using 50 bootstrap samples
    er_out = er.compute_energy_ratio(
        a_in,
        test_turbines=[2],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
        N=10,
        percentiles=[5.0, 95.0],
        uplift_pairs=[("Baseline", "WakeSteering"), ("Baseline (Noisy)", "WakeSteering (Noisy)")],
        uplift_names=["Clean", "Noisy"],
        weight_by="min",
    )

    er_out.plot_energy_ratios(
        color_dict={
            "Baseline": "blue",
            "WakeSteering": "green",
            "Baseline (Noisy)": "C9",
            "WakeSteering (Noisy)": "C8",
        }
    )

    er_out.plot_uplift(
        color_dict={
            "Clean": "green",
            "Noisy": "C8",
        }
    )

    plt.show()
