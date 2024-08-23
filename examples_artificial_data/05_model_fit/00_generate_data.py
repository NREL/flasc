import pickle

import numpy as np
from floris import TimeSeries

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.utilities_examples import load_floris_artificial

""" This example sets up a simple two turbine data set with the Jensen wake model.  Data is saved
with both uncertain and certain wind speed data."""

# Parameters
N = 365 * 24  # * 6  # Assume 1 year of 10 minute data
wd_std = 3.0  # Standard deviation of wind direction in for uncertain model
we_value_set = 0.03  # Wake expansion value that will be the used to generate the test data

# Get default FLORIS model
fm_default, _ = load_floris_artificial(wake_model="jensen")

# Wait for param functions
# ufm, _ = load_floris_artificial(wake_model="jensen", wd_std=wd_std)

# Set a simple two turbine layout
layout_x = [0.0, 126.0 * 6.0]
layout_y = [0.0, 0.0]

# Generate a random series of wind speeds and directions with wind directions
# focused on turbine 0 waking turbine 1
np.random.seed(0)
wind_directions = np.random.uniform(230.0, 310.0, N)
wind_speeds = np.random.uniform(4.0, 15.0, N)
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=0.06
)

# Set layout and inflow
fm_default.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
# ufm.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)

# Get a new model with a different wake expansion value
fm_param = fm_default.copy()

# Set the FLORIS model parameter
parameter = ("wake", "wake_velocity_parameters", "jensen", "we")
we_value_original = fm_param.get_param(parameter)
fm_param.set_param(parameter, we_value_set)

# ufm.set_param(parameter, we_value)

# Run
fm_param.run()
# ufm.run()

# Get the turbine powers in kW
powers = fm_param.get_turbine_powers()

# Build the dataframe
df = ModelFit.form_flasc_dataframe(
    wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers
)

# Save the dataframe and default model and target parameter to a pickle file
with open("two_turbine_data.pkl", "wb") as f:
    pickle.dump(
        {
            "df": df,
            "fm_default": fm_default,
            "parameter": parameter,
            "we_value_original": we_value_original,
            "we_value_set": we_value_set,
        },
        f,
    )
