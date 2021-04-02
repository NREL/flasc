# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from floris_scada_analysis.dataframe_manipulations import df_drop_nan_rows
import numpy as np
import os
import pandas as pd
import streamlit as st

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import ws_pow_filtering as wspcf


# Use the full page for figures
st.set_page_config(layout="wide")

@st.cache()
def load_data():    
    print("Loading .ftr data.")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(root_dir, '../demo_dataset/demo_dataset_60s.ftr')
    if not os.path.exists(ftr_path):
        raise FileNotFoundError('Please run ./examples/demo_dataset/' +
                                'generate_demo_dataset.py before try' +
                                'ing any of the other examples.')
    df_60s = pd.read_feather(ftr_path)
    return df_60s

@st.cache(allow_output_mutation=True)
def load_class():
    df_60s = load_data()
    ws_pow_filtering = wspcf.ws_pw_curve_filtering(
        df=df_60s, turbine_list=[0])
    return df_60s, ws_pow_filtering


# Load dataset and initialize class
df_full, ws_pow_filtering = load_class()
t0 = min(pd.to_datetime(df_full.time))
t1 = max(pd.to_datetime(df_full.time))

# Time filtering of dataset
st.sidebar.markdown('## Dataset selection')
time_start = st.sidebar.date_input('Start date', value=t0, min_value=t0, max_value=t1)
time_end = st.sidebar.date_input('End date', value=t1, min_value=time_start, max_value=t1)

# Cut data down to time region
df = df_full.copy()
df = df[pd.to_datetime(np.array(df.time)).tz_localize(None) >= pd.to_datetime(time_start)]
df = df[pd.to_datetime(np.array(df.time)).tz_localize(None) <= pd.to_datetime(time_end)]
num_turbines = dfm.get_num_turbines(df)
ws_pow_filtering.set_df(df)

# Generalized settings
st.sidebar.markdown('## General filter settings')
pow_step = st.sidebar.slider('Discretization step in power direction (kW)', 5., 250.,
                             50., step=5.)
wind_dev = st.sidebar.slider('Maximum wind speed deviation within bin (m/s)', .25, 5.0,
                             2.0, step=0.25)
max_bin = st.sidebar.slider('Maximum bin for which to filter out deviations (kW)', 4200., 5000.,
                            4800., step=10.)
turbines_options = ['all']
turbines_options.extend(list(range(num_turbines)))
turbs_analyzed = st.sidebar.multiselect(label="Turbines to analyze",
                                        options=turbines_options,
                                        default=[0])
cfmplots = st.sidebar.selectbox(label='Generate confirmation plots',
                           options=[True, False], index=1)

# Set turbines to analyze
if 'all' in turbs_analyzed:
    turbs_analyzed = 'all'
ws_pow_filtering.set_turbine_mode(turbine_list=turbs_analyzed)
window_list = ws_pow_filtering.window_list

# Add new window
st.sidebar.markdown('## Add or remove a window')
if st.sidebar.button('Add new window'):
    est_rated_pow = ws_pow_filtering.est_rated_pow[0]
    window_list.append({'idx': len(window_list),
                        'ws_range': [0., 50.],
                        'pow_range': [0., 1.1*est_rated_pow],
                        'axis': int(0),
                        'turbines': turbs_analyzed})

# Remove window options
if st.sidebar.button('Delete last window'):
    if len(window_list) > 0:
        window_list.pop(-1)

# Show all windows
windows_st_list = []
for wdw in window_list:
    st.sidebar.markdown('## Window %d settings' % wdw['idx'])
    windows_st_list.append(
    [st.sidebar.slider("[%d] Wind speed range" % wdw['idx'], 0.0, 50.0,
                       wdw['ws_range'], 0.5, format='%.1f'),
    st.sidebar.slider("[%d] Power range" % wdw['idx'], 0.0, 5500.0,
                      wdw['pow_range'], 0.5, format='%.1f'),
    st.sidebar.selectbox(label='[%d] Axis' % wdw['idx'],
                         options=[0, 1], index=wdw['axis']),
    st.sidebar.multiselect(label="[%d] Turbines" % wdw['idx'],
                           options=turbines_options,
                           default=['all'])
    ])

# Update values in WS-power curve filter
ws_pow_filtering.set_binning_properties(pow_step=pow_step,
                                        ws_dev=wind_dev,
                                        max_pow_bin=max_bin)

# Update filter values with streamlit app values
ws_pow_filtering.window_list = window_list
for i in range(len(window_list)):
    wdw = windows_st_list[i]
    turbine_list = wdw[3]
    if 'all' in turbine_list:
        turbine_list = list(range(num_turbines))
    ws_pow_filtering.window_list[i]['ws_range'] = wdw[0]
    ws_pow_filtering.window_list[i]['pow_range']= wdw[1]
    ws_pow_filtering.window_list[i]['axis']= int(wdw[2])
    ws_pow_filtering.window_list[i]['turbines'] = turbine_list

# Filter data using default settings
ws_pow_filtering.apply_filters()

# Saving dataframe
st.sidebar.markdown('## Export filtered dataframe')
savepath = st.sidebar.text_input(
    "Save path", 'examples/ws_pow_curve_filtering/df_filtered.ftr')
if st.sidebar.button('Save'):
    ws_pow_filtering.df.to_feather(savepath)
    st.sidebar.write('Saved dataframe to %s.' % savepath)

# Plot and save data for current dataframe
st.markdown('## Filtering results')
figs = ws_pow_filtering.plot(draw_windows=True,
                             confirm_plot=cfmplots)

if turbs_analyzed == 'all':
    turbs_analyzed = list(range(num_turbines))

for idx, f in enumerate(figs):
    ti = turbs_analyzed[idx]
    st.write('Turbine %03d' % ti)
    st.write(f)

    N = ws_pow_filtering.df.shape[0]
    N_selfflg = sum(ws_pow_filtering.df['self_status_%03d' % ti]==0)
    N_oow = sum(ws_pow_filtering.df_out_of_windows[ti])
    N_oowsdev = sum(ws_pow_filtering.df_out_of_ws_dev[ti])

    # Plot point distribution/filtered
    N_list = [N-N_selfflg-N_oow-N_oowsdev, N_selfflg, N_oow, N_oowsdev]
    fd = wspcf.plot_filtering_distribution(
        N_list=N_list,
        label_list=['good', 'self-flagged',
                    'window-filered', 'ws dev filtered'])
    st.write(fd)
