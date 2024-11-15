"""This module contains the optimization algorithms for the model fitting."""

import numpy as np
import optuna

from flasc.model_fit.model_fit import ModelFit


def atomic_opt_optuna(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False
) -> dict:
    """Optimize the model parameters using Optuna.

    Args:
        mf: ModelFit object
        n_trials: Number of trials to run. Defaults to None (100).
        timeout: Timeout for the optimization. Defaults to None.
        turbine_groupings (Dict[str, Tuple], optional): Dictionary of turbine groupings.
            Defaults to None.
        verbose: Whether to print out the optimization process. Defaults to False.

    Returns:
        Dictionary containing the optimal parameter values
    """

    # Set up the objective function for optuna
    def objective(trial):
        parameter_values = []
        for p_idx in range(mf.n_parameters):
            parameter_name = mf.parameter_name_list[p_idx]
            parameter_range = mf.parameter_range_list[p_idx]
            parameter_values.append(
                trial.suggest_float(parameter_name, parameter_range[0], parameter_range[1])
            )

        return mf.set_parameter_and_evaluate(parameter_values, turbine_groupings)

    # Run the optimization
    study = optuna.create_study()

    # Seed the initial value
    init_dict = {}
    for pname, pval in zip(mf.parameter_name_list, mf.get_parameter_values()):
        init_dict[pname] = pval
    study.enqueue_trial(init_dict)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Make a list of the best parameter values
    best_params = []
    for parameter_name in mf.parameter_name_list:
        best_params.append(study.best_params[parameter_name])

    # Return results as dictionary
    result_dic = {
        "parameter_values": best_params,
        "best_cost": study.best_value,
    }

    # Returns results and the study object
    return result_dic, study


def opt_optuna_with_unc(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False
) -> dict:
    """Optimize the model parameters using Optuna.

    Args:
        mf: ModelFit object
        n_trials: Number of trials to run. Defaults to None (100).
        timeout: Timeout for the optimization. Defaults to None.
        turbine_groupings (Dict[str, Tuple], optional): Dictionary of turbine groupings.
            Defaults to None.
        verbose: Whether to print out the optimization process. Defaults to False.

    Returns:
        Dictionary containing the optimal parameter values
    """

    # Set up the objective function for optuna
    def objective(trial):
        # Set wd_std
        mf.set_wd_std(wd_std=trial.suggest_float("wd_std", 0.1, 6.0))

        parameter_values = []
        for p_idx in range(mf.n_parameters):
            parameter_name = mf.parameter_name_list[p_idx]
            parameter_range = mf.parameter_range_list[p_idx]
            parameter_values.append(
                trial.suggest_float(parameter_name, parameter_range[0], parameter_range[1])
            )

        return mf.set_parameter_and_evaluate(parameter_values, turbine_groupings)

    # Run the optimization
    study = optuna.create_study()

    # Seed the initial value
    init_dict = {"wd_std": 3.0}
    for pname, pval in zip(mf.parameter_name_list, mf.get_parameter_values()):
        init_dict[pname] = pval
    study.enqueue_trial(init_dict)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Make a list of the best parameter values
    best_params = []
    for parameter_name in mf.parameter_name_list:
        best_params.append(study.best_params[parameter_name])

    # Return results as dictionary
    result_dic = {
        "wd_std": study.best_params["wd_std"],
        "parameter_values": best_params,
        "best_cost": study.best_value,
    }

    # Returns results and the study object
    return result_dic, study


def atomic_opt_sweep_sequential(
    mf: ModelFit, n_points=None, turbine_groupings=None, verbose=False
) -> dict:
    """Optimize the model parameters by sweeping through the parameter space.

    Args:
        mf: ModelFit object
        n_points: Number of points to evaluate in the parameter space.  Defaults to None.  If None,
            will use the default value of 10.
        turbine_groupings (Dict[str, Tuple], optional): Dictionary of turbine groupings.
            Defaults to None.
        verbose (bool, optional): Whether to print out the optimization process. Defaults to False.

    Returns:
        Dictionary containing the optimal parameter values, the parameter values tested,
            and the cost values
    """
    if n_points is None:
        n_points = 10

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
        if verbose:
            print(f"Optimizing parameter '{parameter_name}' ({i+1}/{len(mf.parameter_list)})")
            print(f".Testing range {parameter_range} in {n_points} steps")

        parameter_values_sweep = np.linspace(parameter_range[0], parameter_range[1], n_points)
        cost_values = np.zeros(n_points)

        for j, parameter_value in enumerate(parameter_values_sweep):
            if verbose:
                print(f"..Testing parameter value {parameter_value:g} ({j+1}/{n_points})")
            parameter_values[i] = parameter_value
            cost_values[j] = mf.set_parameter_and_evaluate(parameter_values, turbine_groupings)

        optimal_index = np.argmin(cost_values)

        # Save the optimal value
        parameter_values[i] = parameter_values_sweep[optimal_index]
        if verbose:
            print(f".Found optimal value for parameter '{parameter_name}': {parameter_values[i]}")

        # Record the optimized cost
        best_cost = cost_values[optimal_index]
        if verbose:
            print(f".best cost: {best_cost}")

        # Record the values tests
        parameter_values_sweep_record[parameter_name] = parameter_values_sweep
        cost_values_record[parameter_name] = cost_values

    # Print the final results in table
    if verbose:
        print("Optimization results:")
        print("Parameter name\tOptimal value")
    for parameter_name, parameter_value in zip(mf.parameter_name_list, parameter_values):
        if verbose:
            print(f"{parameter_name}\t{parameter_value}")

    # Return results as dictionary
    return {
        "parameter_values": parameter_values,
        "best_cost": best_cost,
        "parameter_values_sweep_record": parameter_values_sweep_record,
        "cost_values_record": cost_values_record,
    }
