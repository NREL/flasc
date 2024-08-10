"""Utility functions for examples."""

import copy
from pathlib import Path
from time import perf_counter as timerpc

import floris.layout_visualization as layoutviz
import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel, UncertainFlorisModel


def load_floris_smarteole(wake_model="gch", wd_std=0.0):
    """Load a FlorisModel object for the wind farm at hand.

    Args:
        wake_model (str, optional): The wake model that FLORIS should use. Common
          options are 'cc', 'gch', 'jensen', 'turbopark' and 'emgauss'
           . Defaults to "gch".
        operation_modes (array, optional): Array or list of integers denoting each
          turbine's operation mode. When None is specified, will assume each turbine
          is in its first operation mode (0). Defaults to None.
        wd_std (float, optional): Uncertainty; standard deviation in the inflow
          wind direction in degrees. Defaults to 0.0 deg meaning no uncertainty.

    Returns:
        FlorisModel: Floris object.
    """
    # Use the local FLORIS GCH/CC model for the wake model settings
    root_path = (
        Path(__file__).resolve().parents[2] / "examples_smarteole" / "floris_input_smarteole"
    )
    fn = root_path / "{:s}.yaml".format(wake_model)

    # Initialize FLORIS model and format appropriately
    fm = FlorisModel(fn)

    # Add uncertainty
    if wd_std > 0.01:
        unc_options = {
            "std_wd": wd_std,  # Standard deviation for inflow wind direction (deg)
            "pmf_res": 1.0,  # Resolution over which to calculate angles (deg)
            "pdf_cutoff": 0.995,  # Probability density function cut-off (-)
        }
        fm = UncertainFlorisModel(fm, unc_options=unc_options)

    # Add turbine weighing terms. These are typically used to distinguish
    # between turbines of interest and neighboring farms. This is particularly
    # helpful when you have information about surrounding wind farms.
    turbine_weights = np.ones(len(fm.layout_x), dtype=float)

    return (fm, turbine_weights)


def load_floris_artificial(wake_model="gch", wd_std=0.0, cosine_exponent=None):
    """Load a FlorisModel object for the wind farm at hand.

    Args:
        wake_model (str, optional): The wake model that FLORIS should use. Common
          options are 'cc', 'gch', 'jensen',  'turbopark' and 'emgauss'
          . Defaults to "gch".
        wd_std (float, optional): Uncertainty; standard deviation in the inflow
          wind direction in degrees. Defaults to 0.0 deg meaning no uncertainty.
        cosine_exponent (float, optional): The cosine exponent for the power-thrust
            table. Defaults to None.

    Returns:
        FlorisModel: Floris object.
    """
    # Use the local FLORIS GCH/CC model for the wake model settings
    root_path = (
        Path(__file__).resolve().parents[2] / "examples_artificial_data" / "floris_input_artificial"
    )
    fn = root_path / "{:s}.yaml".format(wake_model)

    # Now assign the turbine locations and information
    layout_x = [1630.222, 1176.733, 816.389, 755.938, 0.0, 1142.24, 1553.102]
    layout_y = [0.0, 297.357, 123.431, 575.544, 647.779, 772.262, 504.711]

    # Initialize FLORIS model and format appropriately
    fm = FlorisModel(fn)
    fm.set(
        layout_x=layout_x,
        layout_y=layout_y,
    )

    # Update Pp if specified
    if cosine_exponent is not None:
        tdefs = [copy.deepcopy(t) for t in fm.core.farm.turbine_definitions]
        for ii in range(len(tdefs)):
            tdefs[ii]["power_thrust_table"]["cosine_loss_exponent_yaw"] = cosine_exponent

        fm.set(turbine_type=tdefs)

    # Add uncertainty
    if wd_std > 0.01:
        unc_options = {
            "std_wd": wd_std,  # Standard deviation for inflow wind direction (deg)
            "pmf_res": 1.0,  # Resolution over which to calculate angles (deg)
            "pdf_cutoff": 0.995,  # Probability density function cut-off (-)
        }
        fm = UncertainFlorisModel(fm, unc_options=unc_options)

    # Add turbine weighing terms. These are typically used to distinguish
    # between turbines of interest and neighboring farms. This is particularly
    # helpful when you have information about surrounding wind farms.
    turbine_weights = np.ones(len(layout_x), dtype=float)

    return (fm, turbine_weights)


if __name__ == "__main__":
    # Load and time the artificial FLORIS model
    t0 = timerpc()
    fm, turbine_weights = load_floris_artificial()
    print("Time spent to load the FLORIS model (artificial): {:.2f} s.".format(timerpc() - t0))
    ax = layoutviz.plot_turbine_points(fm)
    layoutviz.plot_turbine_labels(fm, ax=ax)
    ax.grid()

    # Load and time the Smarteole FLORIS model
    t0 = timerpc()
    fm, turbine_weights = load_floris_smarteole()
    print("Time spent to load the FLORIS model (smarteole): {:.2f} s.".format(timerpc() - t0))
    ax = layoutviz.plot_turbine_points(fm)
    layoutviz.plot_turbine_labels(fm, ax=ax)
    ax.grid()

    plt.show()
