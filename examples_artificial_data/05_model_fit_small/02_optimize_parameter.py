import pickle

import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
)

from flasc.model_fit.cost_library import turbine_power_error_abs
from flasc.model_fit.model_fit import ModelFit
from flasc.model_fit.opt_library import opt_optuna

""" Use ModelFit optimization to find the optimal wake expansion value that best fits the data """

# Since ModelFit is always parallel this is important to include
if __name__ == "__main__":
    # Parameters
    time_out = 3

    # Load the data from previous example
    with open("two_turbine_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack
    df = data["df"]
    fm_default = data["fm_default"]
    parameter = data["parameter"]
    we_value_original = data["we_value_original"]
    we_value_set = data["we_value_set"]

    # Now pass the above cost function to the ModelFit class
    mf = ModelFit(
        df,
        fm_default,
        turbine_power_error_abs,
        parameter_list=[parameter],
        parameter_name_list=["wake expansion"],
        parameter_range_list=[(0.01, 0.07)],
        parameter_index_list=[],
    )

    # Compute the baseline cost
    print("Evaluating baseline cost")
    baseline_cost = mf.evaluate_floris()

    # Optimize
    opt_result, study = opt_optuna(mf, timeout=time_out, n_trials=None)

    # Print results
    print("----------------------------")
    print(f"Default parameter: {we_value_original}")
    print(f"Set parameter: {we_value_set}")
    print()
    print(f"Calibrated parameter value:  {opt_result['parameter_values'][0]:.2f}")
    print("----------------------------")

    # Show an optuna progress plot
    plot_optimization_history(study)
    plt.show()
