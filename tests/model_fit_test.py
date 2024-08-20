import numpy as np
import pandas as pd
import pytest

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.utilities_examples import load_floris_artificial


def get_simple_inputs_gch():
    n_findex = 5

    # Create a simple dataframe
    df = pd.DataFrame(
        {
            "time": np.arange(0, n_findex),
            "pow_000": np.random.uniform(0.0, 1000.0, n_findex),
            "ws_000": np.random.uniform(0.0, 10.0, n_findex),
            "wd_000": np.random.uniform(0.0, 360.0, n_findex),
        }
    )

    # Load floris and set to single turbine layout
    fm, _ = load_floris_artificial(wake_model="gch")
    fm.set(layout_x=[0.0], layout_y=[0.0])

    # Define cost_function_handle as a simple function
    def cost_function_handle():
        return None

    # Define the optimization algorithm as a simple function
    def optimization_algorithm():
        return None

    # Define the parameters to tune the kA parameter of GCH
    parameter_list = [("wake", "wake_velocity_parameters", "gauss", "ka")]
    parameter_name_list = ["kA"]
    parameter_range_list = [(0.1, 0.5)]
    parameter_index_list = None

    return (
        df,
        fm,
        cost_function_handle,
        optimization_algorithm,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    )


def test_instantiate_model_fit():
    # Get simple inputs
    (
        df,
        fm,
        cost_function_handle,
        optimization_algorithm,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object
    model_fit = ModelFit(
        df,
        fm,
        cost_function_handle,
        optimization_algorithm,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    )

    # Check if the ModelFit object is correctly instantiated
    assert model_fit.n_turbines == 1
    assert model_fit.parameter_list == parameter_list
    assert model_fit.parameter_name_list == parameter_name_list
    assert model_fit.parameter_range_list == parameter_range_list

    assert len(model_fit.parameter_index_list) == len(parameter_list)

    # Check the initialization of the initial parameter values
    assert len(model_fit.initial_parameter_values) == len(parameter_list)
    assert model_fit.initial_parameter_values[0] == 0.38


def test_turbine_number():
    # Get simple inputs
    (
        df,
        fm,
        cost_function_handle,
        optimization_algorithm,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    ) = get_simple_inputs_gch()

    # Change the number of turbines in the FlorisModel
    fm.set(layout_x=[0.0, 1000.0], layout_y=[0.0, 0.0])

    with pytest.raises(ValueError):
        # Instantiate the ModelFit object
        ModelFit(
            df,
            fm,
            cost_function_handle,
            optimization_algorithm,
            parameter_list,
            parameter_name_list,
            parameter_range_list,
            parameter_index_list,
        )
