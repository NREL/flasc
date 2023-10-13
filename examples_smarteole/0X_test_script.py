import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from floris.tools import FlorisInterface

from flasc.model_tuning.floris_tuner import FlorisTuner

# Suppress warnings
import warnings

from flasc.utilities_examples import load_floris_smarteole

warnings.filterwarnings('ignore')


# Specify offsets
start_of_offset = 200 # deg
end_of_offset = 240 # deg

# Load FLORIS
fi, _ = load_floris_smarteole(wake_model="emgauss")

# Define D
D = fi.floris.farm.rotor_diameters[0]

# Specify SCADA file path and load the dataframe
scada_path = os.path.join(Path.cwd(), "postprocessed", "df_scada_data_60s_filtered_and_northing_calibrated.ftr")
df_scada = pd.read_feather(scada_path)



# Limit SCADA to this region
df_scada = df_scada[(df_scada.wd_smarteole > (start_of_offset - 20)) &
                    (df_scada.wd_smarteole < (end_of_offset + 20))]

# Assign wd, ws and pow ref and subset SCADA based on reference variables used in the SMARTEOLE wake steering experiment (TODO reference the experiment)
df_scada = (df_scada
    .assign(
        wd = lambda df_: df_['wd_smarteole'],
        ws = lambda df_: df_['ws_smarteole'],
        pow_ref = lambda df_: df_['pow_ref_smarteole']
    )
)

# Use only the baseline data for the wake recovery tuning
num_turbines = 7
df_scada_baseline = df_scada[df_scada.control_mode=='baseline']
num_wd_base = df_scada_baseline.shape[0]
yaw_angles_base = np.zeros([num_wd_base,1,num_turbines])
yaw_angles_base[:,0, 5] = df_scada_baseline.wind_vane_005.values # Apply angles to SMV6



# Instantiate a FLORIS model tuner object
floris_tuner_baseline = FlorisTuner(fi=fi,
                                    df_scada=df_scada_baseline,
                                    yaw_angles=yaw_angles_base) 

# Specify a range of wake expansion rates (assuming no breakpoints) values
wake_expansion_rates = np.linspace(start=0.005, 
                                   stop=0.05,
                                   num=20)

# wake_expansion_rates = [0.024]


# Determine the optimal value for wake expansion rates
floris_tuner_baseline.evaluate_parameter_list(param=['wake','wake_velocity_parameters','empirical_gauss','wake_expansion_rates'],
                                          param_values=wake_expansion_rates,
                                          param_idx = 0,
                                          test_turbines=[4],
                                          ref_turbines=[5],
                                          use_predefined_ref=False,
                                          )


print('Computing Error')

floris_tuner_baseline.calculate_param_errors()
floris_tuner_baseline.plot_error()
floris_tuner_baseline.plot_energy_ratios()


## WAKE STEEERING
# Get the FI model with the best parameter applied
fi_2 = floris_tuner_baseline.apply_best_param()

# Now repeat the tuning on wake steering data
df_all = df_scada.copy()
num_wd = df_all.shape[0]
yaw_angles = np.zeros([num_wd,1,num_turbines])
yaw_angles[:,0, 5] = df_all.wind_vane_005.values # Apply angles to SMV6
df_all['df_mode'] = df_all['control_mode'] # df_mode column must exist



# Use all the data this time and identify the modes of operation for computing uplift
print('Tuning against wake steering data')

# # Instantiate a FLORIS model tuner object
floris_tuner_wake_steering = FlorisTuner(fi=fi_2,
                                    df_scada=df_all,
                                    yaw_angles=yaw_angles) 

# Specify a range of wake expansion rates (assuming no breakpoints) values
horizontal_deflection_gains = np.linspace(start=0, 
                                   stop=4,
                                   num=10)

floris_tuner_wake_steering.evaluate_parameter_list(param=['wake','wake_deflection_parameters','empirical_gauss','horizontal_deflection_gain_D'],
                                          param_values=horizontal_deflection_gains,
                                          test_turbines=[4],
                                          ref_turbines=[5],
                                          use_predefined_ref=False,
                                          compare_uplift=True,
                                          df_mode_order = ['baseline','controlled']
                                          )


print('Computing Error')

floris_tuner_wake_steering.calculate_param_errors(compare_uplift=True)
floris_tuner_wake_steering.plot_error()
floris_tuner_wake_steering.plot_energy_ratio_uplifts()

plt.show()

