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

def load_scada(scada_path: str):
    """
    Load SCADA

    Args:
        scada_path (:py:obj:`str`): Path to load SCADA from.

    Returns:
        df_scada (:py:obj:`pd.DataFrame`): SCADA data.
    """
    
    return pd.read_feather(scada_path)

# Specify SCADA file path and load the dataframe
scada_path = os.path.join(Path.cwd(), "postprocessed", "df_scada_data_60s_filtered_and_northing_calibrated.ftr")
df_scada = load_scada(scada_path=scada_path)

# Specify offsets
start_of_offset = 200 # deg
end_of_offset = 240 # deg

# Limit SCADA to this region
df_scada = df_scada[(df_scada.wd_smarteole > (start_of_offset - 20)) &
                    (df_scada.wd_smarteole < (end_of_offset + 20))]

print(df_scada.shape)

# Assign wd, ws and pow ref and subset SCADA based on reference variables used in the SMARTEOLE wake steering experiment (TODO reference the experiment)
df_scada = (df_scada
    .assign(
        wd = lambda df_: df_['wd_smarteole'],
        ws = lambda df_: df_['ws_smarteole'],
        pow_ref = lambda df_: df_['pow_ref_smarteole']
    )
)

# Split SCADA into baseline and wake steeering (controlled)
df_scada_baseline = df_scada[df_scada.control_mode=='baseline']
df_scada_controlled = df_scada[df_scada.control_mode=='controlled']



fi, _ = load_floris_smarteole(wake_model="emgauss")

# Define D
D = fi.floris.farm.rotor_diameters[0]


# Instantiate a FLORIS model tuner object
floris_tuner_baseline = FlorisTuner(fi=fi,
                                    df_scada=df_scada_baseline) 

# Specify a range of wake expansion rates (assuming no breakpoints) values
wake_expansion_rates = np.linspace(start=0.005, 
                                   stop=0.05,
                                   num=20)

# print(wake_expansion_rates)

# Determine the optimal value for wake expansion rates
floris_tuner_baseline.evaluate_parameter_list(param=['wake','wake_velocity_parameters','empirical_gauss','wake_expansion_rates'],
                                          param_values=wake_expansion_rates,
                                          idx = 0,
                                          test_turbines=[4],
                                          ref_turbines=[5],
                                          use_predefined_ref=False,
                                          )

# print(floris_tuner_baseline.result_dict[0.01 ].df_result)

print('Computing Error')

floris_tuner_baseline.calculate_param_errors()

floris_tuner_baseline.plot_error()

floris_tuner_baseline.plot_energy_ratios()

plt.show()