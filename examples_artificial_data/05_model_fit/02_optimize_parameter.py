import pickle

import numpy as np

from flasc.model_fit.model_fit import ModelFit
from flasc.model_fit.opt_library import sweep_opt_sequential

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
    def cost_function(df_scada, df_floris, fmodel):
        return np.sqrt(np.mean((df_scada["pow_001"].values - df_floris["pow_001"].values) ** 2))

    # Now pass the above cost function to the ModelFit class
    mf = ModelFit(
        df,
        fm_default,
        cost_function,
        parameter_list=[parameter],
        parameter_name_list=["wake expansion"],
        parameter_range_list=[(0.01, 0.07)],
        parameter_index_list=[],
    )

    # Optimize the parameter
    sweep_opt_sequential(mf)
