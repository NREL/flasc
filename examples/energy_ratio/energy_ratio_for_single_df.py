# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from floris import tools as wfct
from floris_scada_analysis import energy_ratio as er
from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as fsatools


# Load dataframe with scada data
root_dir = os.path.dirname(os.path.abspath(__file__))
ftr_path = os.path.join(root_dir, '../demo_dataset/demo_dataset_60s.ftr')
if not os.path.exists(ftr_path):
    raise FileNotFoundError('Please run ./examples/demo_dataset/' +
                            'generate_demo_dataset.py before try' +
                            'ing any of the other examples.')
df = pd.read_feather(ftr_path)

# Initialize the FLORIS interface fi
print('Initializing the FLORIS object for our demo wind farm')
file_path = os.path.dirname(os.path.abspath(__file__))
fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
fi = wfct.floris_interface.FlorisInterface(fi_path)
fi.vis_layout()

# Preprocess dataframes using floris
df = dfm.filter_df_by_status(df)
df = dfm.set_wd_by_all_turbines(df)
df_upstream = fsatools.get_upstream_turbs_floris(fi, wd_step=5.0)
df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
df = dfm.set_ti_by_upstream_turbines(df, df_upstream)
df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

# limit df to narrow wind direction region
df = dfm.filter_df_by_wd(df=df, wd_range=[20., 90.])
df = df.reset_index(drop=True)

# Calculate energy ratio for single category df
era = er.energy_ratio(df=df,
                      test_turbines=[1],
                      wd_step=2.0,
                      ws_step=1.0,
                      verbose=True)

era.get_energy_ratio(N=1)  # No bootstrapping
era.plot_energy_ratio()

# Calculate energy ratio for two category df
df['category'] = 'baseline'
# Increase pow of turbine 1 on/off every 30 minutes
ids_opt = np.arange(0, df.shape[0]-60, 60)
ids_opt = [np.arange(i, i+30, 1) for i in ids_opt]
ids_opt = np.array(ids_opt).flatten()
df.loc[ids_opt, 'pow_003'] *= 1.20  # Increased power of T3
df.loc[ids_opt, 'category'] = 'optimized'
df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[6])

era = er.energy_ratio(df=df,
                      test_turbines=[3],
                      wd_step=2.0,
                      ws_step=1.0,
                      verbose=True)

era.get_energy_ratio(N=10)  # With bootstrapping
era.plot_energy_ratio()
plt.show()