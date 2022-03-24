# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import copy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from floris.tools.floris_interface import FlorisInterface
from floris.tools.uncertainty_interface import UncertaintyInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


def load_floris(pP=2.0):
    # Import example floris model
    try:
        root_path = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(root_path, "..", "demo_dataset", "demo_floris_input.yaml")
        fi = FlorisInterface(fn)
    except:
        fn = os.path.join("..", "demo_dataset", "demo_floris_input.yaml")
        fi = FlorisInterface(fn)

    # Now assign a new pP value
    tdefs = [copy.deepcopy(t) for t in fi.floris.farm.turbine_definitions]
    for ii in range(len(tdefs)):
        tdefs[ii]["pP"] = pP

    fi.reinitialize(turbine_type=tdefs)

    return fi


def load_wind_climate_interpolant():
    # Assume every wind condition is equally likely. Takes (wd, ws) inputs
    def wind_climate(wd, ws):
        freq = np.zeros_like(ws)
        freq[(ws > 4.0) & (ws < 11.0)] = 1.0
        return freq

    return wind_climate


# Define default optimization settings
def optimize_yaw_angles(
    fi=load_floris(),
    opt_wind_directions=np.arange(0.0, 360.0, 3.0),
    opt_wind_speeds=[8.0],
    opt_turbulence_intensity=0.06,
    opt_minimum_yaw=0.0,
    opt_maximum_yaw=20.0,
    opt_turbine_weights=None,
    opt_Ny_passes=[5, 4],
    opt_std_wd=0.0,
    opt_verify_convergence=False,
):
    # Update FLORIS model with atmospheric conditions
    fi = fi.copy()
    fi.reinitialize(
        wind_directions=opt_wind_directions,
        wind_speeds=opt_wind_speeds,
        turbulence_intensity=opt_turbulence_intensity,
    )

    # Add uncertainty, if applicable
    if opt_std_wd > 0.001:
        fi = UncertaintyInterface(fi.copy())

    # Do optimization
    yaw_opt = YawOptimizationSR(
        fi,
        minimum_yaw_angle=opt_minimum_yaw,
        maximum_yaw_angle=opt_maximum_yaw,
        Ny_passes=opt_Ny_passes,
        turbine_weights=opt_turbine_weights,
        exclude_downstream_turbines=True,
        verify_convergence=opt_verify_convergence
    )
    return yaw_opt.optimize()


# Define default evaluation settings
def evaluate_optimal_yaw_angles(
    yaw_angle_interpolant,
    wind_climate_interpolant=load_wind_climate_interpolant(),
    fi=load_floris(),
    eval_wd_array=np.arange(0.0, 360.0, 3.0),
    eval_ws_array=np.arange(4.0, 16.0, 1.0),
    eval_ti=0.06,
    eval_std_wd=0.0,
):
    # Sort inputs
    eval_wd_array = np.sort(eval_wd_array)
    eval_ws_array = np.sort(eval_ws_array)

    # Update floris object
    fi = fi.copy()
    fi.reinitialize(
        wind_directions=eval_wd_array,
        wind_speeds=eval_ws_array,
        turbulence_intensity=eval_ti,
    )

    # Include uncertainty in the FLORIS model, if applicable
    if (eval_std_wd > 0.001):
        opt_unc_options = dict(
            {'std_wd': eval_std_wd, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}
        )
        fi = UncertaintyInterface(fi.copy(), unc_options=opt_unc_options)

    # Get wind rose frequency
    wd_mesh, ws_mesh = np.meshgrid(
        fi.floris.flow_field.wind_directions,
        fi.floris.flow_field.wind_speeds, 
        indexing='ij'
    )
    freq = wind_climate_interpolant(wd_mesh, ws_mesh)
    freq = freq / np.sum(freq)

    # Interpolate yaw angles
    wd_mesh, ws_mesh = np.meshgrid(
        fi.floris.flow_field.wind_directions,
        fi.floris.flow_field.wind_speeds, 
        indexing='ij'
    )
    ti = fi.floris.flow_field.turbulence_intensity * np.ones_like(wd_mesh)
    yaw_angles_opt = yaw_angle_interpolant(wd_mesh, ws_mesh, ti)

    # Evaluate solutions in FLORIS
    fi.calculate_wake(np.zeros_like(yaw_angles_opt))
    baseline_powers = fi.get_farm_power()
    baseline_powers = np.nan_to_num(baseline_powers, nan=0.0)

    fi = fi.copy()
    fi.calculate_wake(yaw_angles_opt)
    optimized_powers = fi.get_farm_power()
    optimized_powers = np.nan_to_num(optimized_powers, nan=0.0)

    # Prepare results: collect in dataframe and calculate AEPs
    df_out = pd.DataFrame({
        "wind_direction": wd_mesh.flatten(),
        "wind_speed": ws_mesh.flatten(),
        "turbulence_intensity": eval_ti * np.ones_like(ws_mesh).flatten(),
        "frequency": freq.flatten(),
        "farm_power_baseline": baseline_powers.flatten(),
        "farm_power_opt": optimized_powers.flatten(),
    })

    AEP_baseline = np.sum(np.multiply(baseline_powers, freq))
    AEP_opt = np.sum(np.multiply(optimized_powers, freq))

    return AEP_baseline, AEP_opt, df_out
