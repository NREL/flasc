# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from datetime import timedelta as td
import matplotlib.pyplot as plt
import os
import pandas as pd

from floris import tools as wfct
from floris.utilities import wrap_360

from floris_scada_analysis import bias_estimation as best
from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as ftools


# Load dataframe with scada data
root_dir = os.path.dirname(os.path.abspath(__file__))
out_path = root_dir
ftr_path = os.path.join(root_dir, '../demo_dataset/demo_dataset_60s.ftr')
if not os.path.exists(ftr_path):
    raise FileNotFoundError('Please run ./examples/demo_dataset/' +
                            'generate_demo_dataset.py before try' +
                            'ing any of the other examples.')
df = pd.read_feather(ftr_path)
df = dfm.filter_df_by_status(df)

# Initialize the FLORIS interface fi
print('Initializing the FLORIS object for our demo wind farm')
file_path = os.path.dirname(os.path.abspath(__file__))
fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
fi = wfct.floris_interface.FlorisInterface(fi_path)
fi.vis_layout()

# Figure out which turbines are freestream per wind direction
df_upstream = ftools.get_upstream_turbs_floris(fi, wd_step=2.0)

# Add an artificial 7.5 deg bias on the turbine WD measurements
col_names = ['wd_%03d' % ti for ti in range(len(fi.layout_x))]
df[col_names] = wrap_360(df[col_names] + 7.5)

# Define 'wd', 'ws' and restrict dataframe to a ws region of interest
df = dfm.filter_df_by_status(df)
df = dfm.set_wd_by_all_turbines(df)
df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
df = dfm.set_ti_by_upstream_turbines(df, df_upstream)
df = dfm.filter_df_by_ws(df, [6.0, 11.0])

# Calculate floris solutions for our wind farm dataset:
# (note that we assume a fixed value for TI in calculating our
#  df_ti approximations. This is because absolute wake depth
#  is not so essential to determine the wd_bias. The main
#  thing is the relative position of the wakes. Ignoring TI
#  significantly reduces the n.o. calculations and speeds up
#  the codecode.)
df_fi = df[['time', 'ws', 'wd']].copy()
fout_df_fi_approx = os.path.join(out_path, "df_fi_approx.ftr")
if os.path.exists(fout_df_fi_approx):
    df_approx = pd.read_feather(fout_df_fi_approx)
    df_fi, _ = ftools.calc_floris_approx(df_fi, fi, df_approx=df_approx)
else:
    df_fi, df_approx = ftools.calc_floris_approx(df_fi, fi, wd_step=2.0,
                                                 num_threads=40)
    df_approx.to_feather(fout_df_fi_approx)

# Add a 'pow_ref' column to dfs specifying the reference power
df = dfm.set_pow_ref_by_upstream_turbines(df, df_upstream)
df_fi = dfm.set_pow_ref_by_upstream_turbines(df_fi, df_upstream)

# Initialize an bias_estimation object
test_turbines = [0, 1]
fsc = best.bias_estimation(df=df,
                           df_fi=df_fi,
                           fi=fi,
                           test_turbines_subset=test_turbines,
                           sliding_window_lb_ub=[-td(days=45), td(days=45)],
                           df_upstream=df_upstream,
                           eo_ws_step=5.0,
                           eo_wd_step=2.0)

df_bias = pd.DataFrame()
current_time = list(fsc.df.time)[0]
end_time = list(fsc.df.time)[-1]
sliding_window_stepsize = td(days=30)
while current_time <= end_time:
    fsc.set_current_time(new_time=current_time)
    wd_bias, success = fsc.estimate_wd_bias()

    # Save estimated bias to a .csv
    df_bias = df_bias.append({'test_turbines': test_turbines,
                              'current_time': current_time,
                              'wd_bias': wd_bias}, ignore_index=True)
    df_bias.to_csv(os.path.join(out_path, 'df_bias.csv'))

    # Print progress to terminal
    print('Estimated wd bias to be %.1f degrees' % wd_bias)
    fsc.plot_energy_ratios()
    plt.show()

    # Move forward in time and repeat the estimation process
    current_time = current_time + sliding_window_stepsize
