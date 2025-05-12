import pickle

import matplotlib.pyplot as plt
import numpy as np

from flasc.model_fit.cost_library import turbine_power_error_abs
from flasc.model_fit.model_fit import ModelFit

# Since ModelFit is always parallel this is important to include
if __name__ == "__main__":
    # Load the data from previous example
    with open("two_turbine_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack
    df_u = data["df_u"]  # Get data from uncertain model
    fm_default = data["fm_default"]
    ufm_default = data["ufm_default"]
    parameter = data["parameter"]
    we_value_original = data["we_value_original"]
    we_value_set = data["we_value_set"]

    # Now loop over a range of values for the parameter and evaluate the cost function
    n_steps = 10
    param_result = []
    cost_result = []
    cost_result_u = []
    for i, param_value in enumerate(np.arange(0.01, 0.07, 0.01)):
        print(f"Evaluating cost function for parameter value {param_value} ({i + 1}/{n_steps})")
        fm_default.set_param(parameter, param_value)
        mf = ModelFit(
            df_u,
            fm_default,
            turbine_power_error_abs,
        )
        cost_value = mf.evaluate_floris()

        param_result.append(param_value)
        cost_result.append(cost_value)
        print(f"... cost value: {cost_value}")

        print("--repeat for uncertain model--")
        ufm_default.set_param(parameter, param_value)
        mf = ModelFit(
            df_u,
            ufm_default,
            turbine_power_error_abs,
        )
        cost_value = mf.evaluate_floris()

        cost_result_u.append(cost_value)
        print(f"~~~ cost value: {cost_value}")

    # Show the results
    fix, ax = plt.subplots()
    ax.plot(param_result, cost_result, label="Certain model")
    ax.plot(param_result, cost_result_u, label="Uncertain model")
    ax.axvline(we_value_original, color="k", linestyle="--", label="Original value")
    ax.axvline(we_value_set, color="r", linestyle="--", label="Set value")
    ax.set_xlabel("Wake expansion value")
    ax.set_ylabel("Cost value")
    ax.legend()
    plt.show()
