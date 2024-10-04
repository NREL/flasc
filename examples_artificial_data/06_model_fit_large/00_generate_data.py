import pickle

import numpy as np
from floris import TimeSeries

from flasc.model_fit.model_fit import ModelFit
from flasc.utilities.utilities_examples import load_floris_artificial

""" Generate data from a farm and specify the model to generate the data and the model to tune.
Use wd_std = 3 in every case. """

# Parameters
N = 1000  # Number of data points
wd_std = 3.0  # Standard deviation of wind direction in for uncertain model
D = 126.0  # Rotor diameter of 5MW

# Get FLORIS models for different wake models, use default parameters
fm_jensen, _ = load_floris_artificial(wake_model="jensen", wd_std=wd_std)
fm_gch, _ = load_floris_artificial(wake_model="gch", wd_std=wd_std)
fm_emg, _ = load_floris_artificial(wake_model="emgauss", wd_std=wd_std)
fm_turbo, _ = load_floris_artificial(wake_model="turbopark", wd_std=wd_std)

# Use a farm with 14 turbines in a non-gridded layout
layout_x = [
    0,
    5 * D,
    10 * D,
    15 * D,
    20 * D,
    0,
    5 * D,
    10 * D,
    20 * D,
    0,
    5 * D,
    10 * D,
    15 * D,
    20 * D,
]
layout_y = [
    0,
    0.25 * D,
    -D,
    D,
    0,
    7 * D,
    5.5 * D,
    6 * D,
    7 * D,
    16 * D,
    15 * D,
    14 * D,
    12 * D,
    11 * D,
]

# Assign this layout to each of the models
fm_jensen.set(layout_x=layout_x, layout_y=layout_y)
fm_gch.set(layout_x=layout_x, layout_y=layout_y)
fm_emg.set(layout_x=layout_x, layout_y=layout_y)
fm_turbo.set(layout_x=layout_x, layout_y=layout_y)

# # Show the layout (Just while setting up)
# fig, ax = plt.subplots()
# plot_turbine_points(fm_jensen, ax)
# plot_turbine_labels(fm_jensen, ax)
# plt.show()


# Generate a random series of wind speeds and directions with wind directions
# focused on wake loss inducing wind speeds
np.random.seed(0)
wind_directions = np.round(np.random.uniform(0.0, 360.0, N))
wind_speeds = np.round(np.random.uniform(4.0, 15.0, N))

# Generate a time series assuming 6% TI
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=0.06
)

# # Set inflow
fm_jensen.set(wind_data=time_series)
fm_gch.set(wind_data=time_series)
fm_emg.set(wind_data=time_series)
fm_turbo.set(wind_data=time_series)

# Run
print("Running Jensen...")
fm_jensen.run()
print("Running GCH...")
fm_gch.run()
print("Running EMG...")
fm_emg.run()
print("Running Turbo...")
fm_turbo.run()

# Get the turbine powers in kW
powers_jensen = fm_jensen.get_turbine_powers() / 1000
powers_gch = fm_gch.get_turbine_powers() / 1000
powers_emg = fm_emg.get_turbine_powers() / 1000
powers_turbo = fm_turbo.get_turbine_powers() / 1000

# Make a time column for the flasc_dataframe convention
time = np.arange(N)

# Build the dataframes
df_jensen = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers_jensen
)
df_gch = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers_gch
)
df_emg = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers_emg
)
df_turbo = ModelFit.form_flasc_dataframe(
    time=time, wind_directions=wind_directions, wind_speeds=wind_speeds, powers=powers_turbo
)

# Finally get the models that will be used in tuning, in this case with and without uncertainty
fmodel_emg, _ = load_floris_artificial(wake_model="emgauss")
fmodel_emg_unc, _ = load_floris_artificial(wake_model="emgauss", wd_std=wd_std)

# Assign this layout to each of the models
fmodel_emg.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
fmodel_emg_unc.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)


# Save the dataframe and default model and target parameter to a pickle file
with open("farm_data.pkl", "wb") as f:
    pickle.dump(
        {
            "df_jensen": df_jensen,
            "df_gch": df_gch,
            "df_emg": df_emg,
            "df_turbo": df_turbo,
            "fmodel_emg": fmodel_emg,
            "fmodel_emg_unc": fmodel_emg_unc,
        },
        f,
    )
