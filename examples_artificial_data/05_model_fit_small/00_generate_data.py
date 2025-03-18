import pickle

import numpy as np
from floris import TimeSeries, UncertainFlorisModel

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.utilities_examples import load_floris_artificial

""" This example sets up a simple two turbine data set with the Jensen wake model.  Data is saved
with both uncertain and certain wind speed data."""

# Parameters
N = 200  # Number of data points
wd_std = 3.0  # Standard deviation of wind direction in for uncertain model
we_value_set = 0.03  # Wake expansion value that will be the used to generate the test data

# Resolution parameters
# These are used in the UncertainFlorisModel
ws_resolution = 0.25
wd_resolution = 2.0

# Get default FLORIS model
fm_default, _ = load_floris_artificial(wake_model="jensen")
ufm_default = UncertainFlorisModel(
    fm_default.copy(), wd_std=wd_std, ws_resolution=ws_resolution, wd_resolution=wd_resolution
)

# Set a simple two turbine layout
layout_x = [0.0, 126.0 * 6.0]
layout_y = [0.0, 0.0]

# Generate a random series of wind speeds and directions with wind directions
# focused on turbine 0 waking turbine 1
np.random.seed(0)
wind_directions = np.random.uniform(230.0, 310.0, N)
wind_speeds = np.random.uniform(4.0, 15.0, N)

# wind_directions = np.array([270.0])
# wind_speeds = np.array([8.0])

time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=0.06
)

# Set layout and inflow
fm_default.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
ufm_default.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)

# Get a new model with a different wake expansion value
fm_param = fm_default.copy()
ufm_param = ufm_default.copy()

# Set the FLORIS model parameter
parameter = ("wake", "wake_velocity_parameters", "jensen", "we")
we_value_original = fm_param.get_param(parameter)
fm_param.set_param(parameter, we_value_set)
ufm_param.set_param(parameter, we_value_set)

# Run
fm_param.run()
ufm_param.run()

# Get the turbine powers in kW
powers = fm_param.get_turbine_powers() / 1000
powers_u = ufm_param.get_turbine_powers() / 1000

# Make a time column for the flasc_dataframe convention
time = np.arange(N)

# Build the dataframe
df = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers
)

df_u = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers_u
)

# Save the dataframe and default model and target parameter to a pickle file
with open("two_turbine_data.pkl", "wb") as f:
    pickle.dump(
        {
            "df": df,
            "df_u": df_u,
            "fm_default": fm_default,
            "ufm_default": ufm_default,
            "parameter": parameter,
            "we_value_original": we_value_original,
            "we_value_set": we_value_set,
        },
        f,
    )
