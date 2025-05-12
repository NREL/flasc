import pickle

import matplotlib.pyplot as plt
import numpy as np

from flasc.model_fit.cost_library import turbine_power_error_abs
from flasc.model_fit.model_fit import ModelFit

# Load the data from previous example
with open("two_turbine_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack
df = data["df"]
fm_default = data["fm_default"]
parameter = data["parameter"]
we_value_original = data["we_value_original"]
we_value_set = data["we_value_set"]


# Set up the ModelFit object with the data and FLORIS model
# Use the absolute turbine power error as cost function
mf = ModelFit(
    df,
    fm_default,
    turbine_power_error_abs,
)

# Evaluate the model
cost_value = mf.evaluate_floris()
print(f"Cost value: {cost_value} for model with original parameter value {we_value_original}")

# Now loop over a range of values for the parameter and evaluate the cost function
n_steps = 10
param_result = []
cost_result = []
for i, param_value in enumerate(np.arange(0.01, 0.07, 0.01)):
    print(f"Evaluating cost function for parameter value {param_value} ({i + 1}/{n_steps})")
    fm_default.set_param(parameter, param_value)
    mf = ModelFit(
        df,
        fm_default,
        turbine_power_error_abs,
    )
    cost_value = mf.evaluate_floris()

    param_result.append(param_value)
    cost_result.append(cost_value)
    print(f"... cost value: {cost_value}")

# Show the results
fix, ax = plt.subplots()
ax.plot(param_result, cost_result)
ax.axvline(we_value_original, color="k", linestyle="--", label="Original value")
ax.axvline(we_value_set, color="r", linestyle="--", label="Set value")
ax.set_xlabel("Wake expansion value")
ax.set_ylabel("Cost value")
ax.legend()
plt.show()
