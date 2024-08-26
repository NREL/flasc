import numpy as np
import pandas as pd
import pytest

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.utilities_examples import load_floris_artificial


def get_simple_inputs_gch():
    # Create a simple dataframe
    df = pd.DataFrame(
        {
            "time": np.array([0, 1, 2]),
            "pow_000": np.array([1000.0, np.nan, 1200.0]),
            "ws_000": np.array([8.0, 9.0, 10.0]),
            "wd_000": np.array([270.0, 270.0, 270.0]),
        }
    )

    # Assign ws_000 to ws and wd_000 to wd using the assign function
    df = df.assign(ws=df["ws_000"], wd=df["wd_000"])

    # Load floris and set to single turbine layout
    fm, _ = load_floris_artificial(wake_model="gch")
    fm.set(layout_x=[0.0], layout_y=[0.0])

    # Define cost_function as a simple function
    def cost_function():
        return None

    # Define the parameters to tune the kA parameter of GCH
    parameter_list = [("wake", "wake_velocity_parameters", "gauss", "ka")]
    parameter_name_list = ["kA"]
    parameter_range_list = [(0.1, 0.5)]
    parameter_index_list = None

    # Define the optimization algorithm as a simple function
    def optimization_algorithm():
        return None

    return (
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    )


def test_instantiate_model():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object without parameters or optimization
    ModelFit(
        df,
        fm,
        cost_function,
    )

    # Instantiate the ModelFit object with parameters and optimization
    ModelFit(
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    )


def test_df():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        _,
        _,
        _,
        _,
        _,
    ) = get_simple_inputs_gch()

    # Remove the wd column from the dataframe
    df = df.drop(columns=["wd"])

    # Instantiate the ModelFit object without parameters or optimization
    with pytest.raises(ValueError):
        ModelFit(
            df,
            fm,
            cost_function,
        )


def test_turbine_number():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        _,
        _,
        _,
        _,
        _,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object without parameters or optimization
    model_fit = ModelFit(
        df,
        fm,
        cost_function,
    )

    # Check the number of turbines
    assert model_fit.n_turbines == 1

    # Change the number of turbines in the FlorisModel
    fm.set(layout_x=[0.0, 1000.0], layout_y=[0.0, 0.0])

    with pytest.raises(ValueError):
        # Instantiate the ModelFit object
        ModelFit(
            df,
            fm,
            cost_function,
        )


def test_get_set_param_no_params():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        _,
        _,
        _,
        _,
        _,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object without parameters or optimization
    model_fit = ModelFit(
        df,
        fm,
        cost_function,
    )

    # Assert that initial_parameter_values is a numpy array with length 0
    np.testing.assert_array_equal(model_fit.initial_parameter_values, np.array([]))

    # Get that get_parameter_values returns an empty numpy array
    np.testing.assert_array_equal(model_fit.get_parameter_values(), np.array([]))


def test_get_set_param_with_params():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object with parameters and optimization
    model_fit = ModelFit(
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    )

    # Check the initialization of the initial parameter values
    np.testing.assert_array_equal(model_fit.initial_parameter_values, np.array([0.38]))
    np.testing.assert_array_equal(model_fit.get_parameter_values(), np.array([0.38]))

    # Change the model parameter values
    model_fit.set_parameter_values(np.array([10.0]))

    # Check the set value
    np.testing.assert_array_equal(model_fit.get_parameter_values(), np.array([10.0]))


def test_run_floris():
    # Get simple inputs
    (
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    ) = get_simple_inputs_gch()

    # Instantiate the ModelFit object with parameters and optimization
    model_fit = ModelFit(
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
        optimization_algorithm,
    )

    df_floris = model_fit.run_floris_model()

    # df_floris is a FlaskDataFrame
    assert isinstance(df_floris, pd.DataFrame)

    # df and df_floris have the same number of rows
    assert df.shape[0] == df_floris.shape[0]

    # df['ws] == df_floris['ws']
    np.testing.assert_array_equal(df["ws"].values, df_floris["ws"].values)

    # The second element of df_floris['pow_000'] is a NaN
    assert np.isnan(df_floris.loc[1, "pow_000"])

    # Check that the first element in power corresponds to power of 8 m/s
    # for default NREL 5MW turbine
    assert np.isclose(df_floris.loc[0, "pow_000"], 1753.9, atol=10)
