"""This module contains the optimization algorithms for the model fitting."""

import numpy as np

from flasc.model_fit.model_fit import ModelFit


# Define an algorithm with works by sweeping through the parameter space
def sweep_opt_sequential(mf: ModelFit, n_points=10) -> dict:
    """Optimize the model parameters by sweeping through the parameter space.

    Args:
        mf: ModelFit object
        n_points: Number of points to evaluate in the parameter space

    Returns:
        Dictionary containing the optimal parameter values, the parameter values tested,
            and the cost values
    """
    # Start from the initial parameter values
    parameter_values = mf.get_parameter_values()

    # Set up records of parameters tested
    parameter_values_sweep_record = {}
    cost_values_record = {}

    for i, (parameter_name, parameter_range) in enumerate(
        zip(
            mf.parameter_name_list,
            mf.parameter_range_list,
        )
    ):
        print(f"Optimizing parameter '{parameter_name}' ({i+1}/{len(mf.parameter_list)})")
        print(f".Testing range {parameter_range} in {n_points} steps")

        parameter_values_sweep = np.linspace(parameter_range[0], parameter_range[1], n_points)
        cost_values = np.zeros(n_points)

        for j, parameter_value in enumerate(parameter_values_sweep):
            print(f"..Testing parameter value {parameter_value} ({j+1}/{n_points})")
            parameter_values[i] = parameter_value
            cost_values[j] = mf.set_parameter_and_evaluate(parameter_values)

        optimal_index = np.argmin(cost_values)

        # Save the optimal value
        parameter_values[i] = parameter_values_sweep[optimal_index]
        print(f".Found optimal value for parameter '{parameter_name}': {parameter_values[i]}")

        # Record the values tests
        parameter_values_sweep_record[parameter_name] = parameter_values_sweep
        cost_values_record[parameter_name] = cost_values

    # Print the final results in table
    print("Optimization results:")
    print("Parameter name\tOptimal value")
    for parameter_name, parameter_value in zip(mf.parameter_name_list, parameter_values):
        print(f"{parameter_name}\t{parameter_value}")

    # Return results as dictionary
    return {
        "parameter_values": parameter_values,
        "parameter_values_sweep_record": parameter_values_sweep_record,
        "cost_values_record": cost_values_record,
    }
