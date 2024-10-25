"""This module contains the optimization algorithms for the model fitting."""

import numpy as np
import optuna
from floris.utilities import wrap_180

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.floris_tools import get_upstream_turbs_floris

## AGGREGATE FUNCTIONS
## AGGREGATE FUNCTIONS CALL CONDITIONING FUNCTIONS WITHIN A LOOP STRUCTURE


def agg_opt_sectors(
    mf: ModelFit,
    wd_sectors: np.array = np.arange(0, 360, 15.0),
    wind_radius: float = 15.0,
    atomic_opt: str = "atomic_opt_optuna",
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Loop over wind direction sectors and optimize the model parameters for each.

    Args:
        mf (ModelFit): ModelFit object
        wd_sectors (np.array, optional): Array of wind direction sectors to optimize over.
            Defaults to np.arange(0,360,15.).
        wind_radius (float, optional): Radius around wind direction to consider. Defaults to 15.
        atomic_opt (str, optional): Atomic optimization function to use.
            Defaults to "atomic_opt_optuna".
        verbose (bool, optional): Whether to print out the optimization process. Defaults to False.
        kwargs: Additional keyword arguments to pass to the atomic optimization function

    Returns:
        dict: Dictionary containing the mean, median, max, min, and std of the parameter values
            across the sectors, the matrix of parameter values, the wind direction sectors
    """
    num_sectors = len(wd_sectors)
    n_parameters = len(mf.parameter_list)
    res_mat = np.zeros((num_sectors, n_parameters))

    print(f"Starting optimization across {num_sectors} sectors")

    for wd_idx, wd in enumerate(wd_sectors):
        print(f"Optimizing for sector {wd_idx+1}/{num_sectors} ({wd} degrees)")
        res = opt_sector(
            mf, wd, wind_radius=wind_radius, atomic_opt=atomic_opt, verbose=verbose, **kwargs
        )
        res_mat[wd_idx, :] = res["parameter_values"]

    # Define some results
    mean_parameters = np.mean(res_mat, axis=0)
    median_parameters = np.median(res_mat, axis=0)
    max_parameters = np.max(res_mat, axis=0)
    min_parameters = np.min(res_mat, axis=0)
    std_parameters = np.std(res_mat, axis=0)

    # Return results as dictionary
    return {
        "mean_parameters": mean_parameters,
        "median_parameters": median_parameters,
        "max_parameters": max_parameters,
        "min_parameters": min_parameters,
        "std_parameters": std_parameters,
        "res_mat": res_mat,
        "wd_sectors": wd_sectors,
    }


## CONDITIONING FUNCTIONS
## CONDITIONING FUNCTIONS SET UP AN OPTIMIZATION PROBLEM AND CALL AN ATOMIC FUNCTION


def opt_sector(
    mf: ModelFit,
    sector: float,
    wind_radius: float = 15.0,
    atomic_opt: str = "atomic_opt_optuna",
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Optimize the model parameters for a given wind direction sector.

    Args:
        mf (ModelFit): ModelFit object
        sector (float): Wind direction sector to optimize over
        wind_radius (float, optional): Radius around wind direction to consider. Defaults to 15.
        atomic_opt (str, optional): Atomic optimization function to use.
            Defaults to "atomic_opt_optuna".
        verbose (bool, optional): Whether to print out the optimization process. Defaults to False.
        kwargs: Additional keyword arguments to pass to the atomic optimization function

    Returns:
        dict: Dictionary containing the optimal parameter values, the parameter values tested,
            and the cost values
    """
    # Get df_scada from within mf
    df_ = mf.df.copy()

    # Get fm from within mf
    fmodel = mf.fmodel
    n_turbines = fmodel.n_turbines

    # Limit df_scada to rows where distance from 'wd' column to wind_direction is
    # less than wind_radius
    df_ = df_[np.abs(wrap_180(df_["wd"] - sector)) < wind_radius]

    # print some info about shape
    if verbose:
        print(f"Shape of df_ after filtering: {df_.shape}")
        print(f"Min/Max Wind Direction (0 -- 360): {df_['wd'].min()}, {df_['wd'].max()}")
        print(
            f"Min/Max Wind Direction (-180 -- +180): "
            f"{wrap_180(df_['wd']).min()}, {wrap_180(df_['wd']).max()}"
        )

    # Get the table of upstream turbines
    upstream_turbs = get_upstream_turbs_floris(fmodel, wd_step=1.0)

    # Select the upstream turbines for middle of sector
    # TODO: This could be perhaps more precise
    upstream_turbs = upstream_turbs[
        (upstream_turbs["wd_min"] <= sector) & (upstream_turbs["wd_max"] >= sector)
    ].iloc[0]["turbines"]
    downstream_turbs = [i for i in range(n_turbines) if i not in upstream_turbs]

    if verbose:
        print(f"Upstream turbines for sector {sector}: {upstream_turbs}")
        print(f"Downstream turbines for sector {sector}: {downstream_turbs}")

    # Set the turbine groupings
    turbine_groupings = {"pow_ref": upstream_turbs, "pow_test": downstream_turbs}

    # Make a new model fit for this sector
    mf_ = ModelFit(
        df_,
        mf.fmodel,
        mf.cost_function,
        mf.parameter_list,
        mf.parameter_name_list,
        mf.parameter_range_list,
        mf.parameter_index_list,
    )

    # Now call the atomic optimization function
    if atomic_opt == "atomic_opt_optuna":
        # Check if n_trials is in kwargs
        if "n_trials" in kwargs:
            n_trials = kwargs["n_trials"]
        else:
            n_trials = None

        return atomic_opt_optuna(
            mf_, n_trials=n_trials, turbine_groupings=turbine_groupings, verbose=verbose
        )

    elif atomic_opt == "atomic_opt_sweep_sequential":
        # Check if n_points is in kwargs
        if "n_points" in kwargs:
            n_points = kwargs["n_points"]
        else:
            n_points = None

        # Pass this through to atomic_opt_sweep_sequential
        return atomic_opt_sweep_sequential(
            mf_, n_points=n_points, turbine_groupings=turbine_groupings, verbose=verbose
        )


def opt_pair(
    mf: ModelFit,
    ref_idx: int,
    test_idx: int,
    wind_radius: float = 15.0,
    atomic_opt: str = "atomic_opt_optuna",
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Optimize the model parameters for a pair of turbines.

    Args:
        mf (ModelFit): ModelFit object
        ref_idx (int): Reference turbine index
        test_idx (int): Test turbine index
        wind_radius (float, optional): Radius around wind direction to consider.
            Defaults to 15.
        atomic_opt (str, optional): Atomic optimization function to use.
            Defaults to "atomic_opt_optuna".
        verbose (bool, optional): Whether to print out the optimization process.
            Defaults to False.
        kwargs: Additional keyword arguments to pass to the atomic optimization function

    Returns:
        dict: Dictionary containing the optimal parameter values, the parameter values tested,
            and the cost values
    """
    # Get df_scada from within mf
    df_ = mf.df.copy()

    # Get fm from within mf
    fmodel = mf.fmodel

    # Find the direction from ref_idx to test_idx
    x_ref = fmodel.layout_x[ref_idx]
    y_ref = fmodel.layout_y[ref_idx]
    x_test = fmodel.layout_x[test_idx]
    y_test = fmodel.layout_y[test_idx]

    dx = x_test - x_ref
    dy = y_test - y_ref
    angle_rad = np.arctan2(dy, dx)
    angle_deg = 270 - np.rad2deg(angle_rad)
    wind_direction = angle_deg % 360

    # Limit df_scada to rows where distance from 'wd' column to wind_direction
    #  is less than wind_radius
    df_ = df_[np.abs(wrap_180(df_["wd"] - wind_direction)) < wind_radius]

    # Make a new model fit from this
    mf_ = ModelFit(
        df_,
        mf.fmodel,
        mf.cost_function,
        mf.parameter_list,
        mf.parameter_name_list,
        mf.parameter_range_list,
        mf.parameter_index_list,
    )

    # Set up turbine groupings
    turbine_groupings = {"pow_ref": [ref_idx], "pow_test": [test_idx]}

    # Now call the atomic optimization function
    if atomic_opt == "atomic_opt_optuna":
        # Check if n_trials is in kwargs
        if "n_trials" in kwargs:
            n_trials = kwargs["n_trials"]
        else:
            n_trials = None

        return atomic_opt_optuna(
            mf_, n_trials=n_trials, turbine_groupings=turbine_groupings, verbose=verbose
        )

    elif atomic_opt == "atomic_opt_sweep_sequential":
        # Check if n_points is in kwargs
        if "n_points" in kwargs:
            n_points = kwargs["n_points"]
        else:
            n_points = None

        # Pass this through to atomic_opt_sweep_sequential
        return atomic_opt_sweep_sequential(
            mf_, n_points=n_points, turbine_groupings=turbine_groupings, verbose=verbose
        )


## ATOMIC FUNCTIONS
## ATOMIC FUNCTIONS SHOULD PERFORM AN OPTIMIZATION AND RETURN A SINGLE RESULT


def atomic_opt_optuna(mf: ModelFit, n_trials=100, timeout=None, turbine_groupings=None, verbose=False) -> dict:
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
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Make a list of the best parameter values
    best_params = []
    for parameter_name in mf.parameter_name_list:
        best_params.append(study.best_params[parameter_name])

    # Return results as dictionary
    return {
        "parameter_values": best_params,
        "best_cost": study.best_value,
    }


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
