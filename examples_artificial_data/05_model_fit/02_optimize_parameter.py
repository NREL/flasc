import pickle

import numpy as np
from scipy.optimize import minimize

from flasc.model_fit.model_fit import ModelFit

# Since ModelFit is always parallel this is important to include
if __name__ == "__main__":
    # Load the data from previous example
    with open("two_turbine_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack
    df = data["df"]
    fm_default = data["fm_default"]
    parameter = data["parameter"]
    we_value_original = data["we_value_original"]
    we_value_set = data["we_value_set"]

    # Define a cost function that checks the RMSE between the power of the second turbine
    def cost_function(df_scada, df_floris):
        return np.sqrt(np.mean((df_scada["pow_001"].values - df_floris["pow_001"].values) ** 2))

    # Define an optimization function based on scipy minimize
    def optimization_algorithm(optimization_step, x0, bounds, **kwargs):
        result = minimize(optimization_step, x0, bounds=bounds, method="slsqp")
        return result

    # Now pass the above cost function to the ModelFit class
    mf = ModelFit(
        df,
        fm_default,
        cost_function,
        parameter_list=[parameter],
        parameter_name_list=["wake expansion"],
        parameter_range_list=[(0.01, 0.07)],
        parameter_index_list=[],
        optimization_algorithm=optimization_algorithm,
    )

    # Call the optimization
    mf.optimize_parameters()
