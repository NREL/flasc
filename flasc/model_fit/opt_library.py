"""This module contains the optimization algorithms for the model fitting."""

import optuna

from flasc.model_fit.model_fit import ModelFit


def opt_optuna(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False, seed=None
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
    if seed is not None:
        study = optuna.create_study()
    else:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))

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


def opt_optuna_with_wd_std(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False
) -> dict:
    """Optimize the model parameters using Optuna including wd_std.

    This version includes the wind direction standard deviation of the UncertainFlorisModel
      as a parameter to optimize.

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


# Temporarily alias the old function names for continutity with older codes, will eventually remove
def atomic_opt_optuna(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False, seed=None
) -> dict:
    """Alias to opt_optuna."""
    return opt_optuna(mf, n_trials, timeout, turbine_groupings, verbose, seed)


def opt_optuna_with_unc(
    mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False
) -> dict:
    """Alias to opt_optuna_with_wd_std."""
    return opt_optuna_with_wd_std(mf, n_trials, timeout, turbine_groupings, verbose)
