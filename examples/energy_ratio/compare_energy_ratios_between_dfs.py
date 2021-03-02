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
from floris.utilities import wrap_360

from floris_scada_analysis import scada_analysis as sca
from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as fsatools


# Load dataframe with scada data
root_dir = os.path.dirname(os.path.abspath(__file__))
ftr_path = os.path.join(root_dir, '../demo_dataset/demo_dataset_60s.ftr')
df = pd.read_feather(ftr_path)

# Initialize the FLORIS interface fi
print('Initializing the FLORIS object for our demo wind farm')
file_path = os.path.dirname(os.path.abspath(__file__))
fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
fi = wfct.floris_interface.FlorisInterface(fi_path)
fi.vis_layout()

# Preprocess dataframes using floris
df = dfm.set_wd_by_all_turbines(df)
df_upstream = fsatools.get_upstream_turbs_floris(fi, wd_step=5.0)
df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
df = dfm.set_ti_by_upstream_turbines(df, df_upstream)

# Generate second dataset with shifted x-axis
df2 = df.copy()
df2['wd'] = wrap_360(df2['wd'] + 7.5)

# Initialize an energy ratio analysis object and add dfs
fsc = sca.scada_analysis()
fsc.add_df(df, 'Original data')
fsc.add_df(df2, 'Data with wd bias of 7.5 degrees')
fsc.print_dfs()
fsc.set_masks(wd_range=[70., 110.], ti_range=[0., 0.25])
fsc.get_energy_ratios(test_turbines=[1], ref_turbines=[0],
                      dep_turbines=[], wd_step=2.0,
                      ws_step=1.0, N=1, verbose=False)
fsc.print_dfs()
fsc.plot_energy_ratios(superimpose=True)