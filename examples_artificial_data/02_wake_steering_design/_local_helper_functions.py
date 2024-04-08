import numpy as np
import pandas as pd
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.uncertain_floris_model import UncertainFlorisModel
from floris.wind_data import WindRose

from flasc.utilities.utilities_examples import load_floris_artificial as load_floris


def load_wind_climate_interpolant():
    # Assume every wind condition is equally likely. Takes (wd, ws) inputs
    def wind_climate(wd, ws):
        freq = np.zeros_like(ws)
        freq[(ws > 4.0) & (ws < 11.0)] = 1.0
        return freq

    return wind_climate


# Define default optimization settings
def optimize_yaw_angles(
    fi=None,
    opt_wind_directions=np.arange(0.0, 360.0, 3.0),
    opt_wind_speeds=[8.0],
    opt_turbulence_intensities=0.06,
    opt_minimum_yaw=0.0,
    opt_maximum_yaw=20.0,
    opt_turbine_weights=None,
    opt_Ny_passes=[5, 4],
    opt_std_wd=0.0,
    opt_verify_convergence=False,
):
    if fi is None:
        fi, _ = load_floris()

    # Handle turbulence intensity input
    if not hasattr(opt_turbulence_intensities, "__len__"):
        opt_turbulence_intensities = opt_turbulence_intensities * np.ones(
            (len(opt_wind_directions), len(opt_wind_speeds))
        )

    # Update FLORIS model with atmospheric conditions
    fm = fm.copy()
    fm.set(
        wind_data=WindRose(
            wind_directions=np.array(opt_wind_directions),
            wind_speeds=np.array(opt_wind_speeds),
            ti_table=np.array(opt_turbulence_intensities)
        )
    )

    # Add uncertainty, if applicable
    if opt_std_wd > 0.001:
        fm = UncertainFlorisModel(fm.copy())

    # Do optimization
    yaw_opt = YawOptimizationSR(
        fi,
        minimum_yaw_angle=opt_minimum_yaw,
        maximum_yaw_angle=opt_maximum_yaw,
        Ny_passes=opt_Ny_passes,
        turbine_weights=opt_turbine_weights,
        exclude_downstream_turbines=True,
        verify_convergence=opt_verify_convergence,
    )
    return yaw_opt.optimize()


# Define default evaluation settings
def evaluate_optimal_yaw_angles(
    yaw_angle_interpolant,
    wind_climate_interpolant=load_wind_climate_interpolant(),
    fi=None,
    eval_wd_array=np.arange(0.0, 360.0, 3.0),
    eval_ws_array=np.arange(4.0, 16.0, 1.0),
    eval_ti=0.06,
    eval_std_wd=0.0,
):
    # Sort inputs
    eval_wd_array = np.sort(eval_wd_array)
    eval_ws_array = np.sort(eval_ws_array)

    if fi is None:
        fi, _ = load_floris()

    # Handle turbulence intensity input
    if not hasattr(eval_ti, "__len__"):
        eval_ti = eval_ti * np.ones(
            (len(eval_wd_array), len(eval_ws_array))
        )

    # Update FLORIS model with atmospheric conditions
    fm = fm.copy()
    fm.set(
        wind_data=WindRose(
            wind_directions=np.array(eval_wd_array),
            wind_speeds=np.array(eval_ws_array),
            ti_table=np.array(eval_ti)
        )
    )

    # Include uncertainty in the FLORIS model, if applicable
    if eval_std_wd > 0.001:
        opt_unc_options = dict({"std_wd": eval_std_wd, "pmf_res": 1.0, "pdf_cutoff": 0.995})
        fm = UncertainFlorisModel(fm.copy(), unc_options=opt_unc_options)

    # Get wind rose frequency
    wd_mesh = fm.floris.flow_field.wind_directions
    ws_mesh = fm.floris.flow_field.wind_speeds
    freq = wind_climate_interpolant(wd_mesh, ws_mesh)
    freq = freq / np.sum(freq)

    # Interpolate yaw angles
    ti = fm.floris.flow_field.turbulence_intensities
    yaw_angles_opt = yaw_angle_interpolant(wd_mesh, ws_mesh, ti)

    # Evaluate solutions in FLORIS
    fm.set(yaw_angles=np.zeros_like(yaw_angles_opt))
    fm.run()
    baseline_powers = fm.get_farm_power()
    baseline_powers = np.nan_to_num(baseline_powers, nan=0.0)

    fm = fm.copy()
    fm.set(yaw_angles=yaw_angles_opt)
    fm.run()
    optimized_powers = fm.get_farm_power()
    optimized_powers = np.nan_to_num(optimized_powers, nan=0.0)

    # Prepare results: collect in dataframe and calculate AEPs
    df_out = pd.DataFrame(
        {
            "wind_direction": wd_mesh.flatten(),
            "wind_speed": ws_mesh.flatten(),
            "turbulence_intensity": eval_ti.flatten(),
            "frequency": freq.flatten(),
            "farm_power_baseline": baseline_powers.flatten(),
            "farm_power_opt": optimized_powers.flatten(),
        }
    )

    AEP_baseline = np.sum(np.multiply(baseline_powers, freq))
    AEP_opt = np.sum(np.multiply(optimized_powers, freq))

    return AEP_baseline, AEP_opt, df_out
