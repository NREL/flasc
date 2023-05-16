# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
from time import perf_counter as timerpc

import numpy as np
import pandas as pd
import streamlit as st

from flasc.dataframe_operations import (
    dataframe_manipulations as dfm,
)
from flasc.turbine_analysis import ws_pow_filtering as wspcf


# In this script, we filter the data by looking at the wind-speed vs.
# power curves using a GUI, through 'streamlit'. It holds the same
# function as a_04_wspowercurve_filtering_code.py, but is much more
# intuitive since the user can directly see the effect of the applied
# filters. You can run this script by:
#
# streamlit run a_04b_wspowercurve_filtering_gui.py
#
# Though, we recommend you to use the a_04_wspowercurve_filtering_code.py
# as it has gone through more tests and usage. The streamlit app can be
# useful to explore the options within the windspeed-power curve filtering
# class, but its not recommend for widespread usage.
#


# Use the full page for figures
st.set_page_config(layout="wide")


@st.cache()
def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "03_sensor_faults_filtered")
    df = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    return df


@st.cache(allow_output_mutation=True)
def load_class():
    df_60s = load_data()
    ws_pow_filtering = wspcf.ws_pw_curve_filtering(df=df_60s, turbine_list=[0])
    df_full_filtered = ws_pow_filtering.df
    df_time_array = pd.to_datetime(
        np.array(df_full_filtered.time)
    ).tz_localize(None)
    return df_full_filtered, df_time_array, ws_pow_filtering


# Load dataset and initialize class
df_full, df_time_array, ws_pow_filtering = load_class()
t0 = list(df_full.time)[0]
t1 = list(df_full.time)[-1]

# Time filtering of dataset
st.sidebar.markdown("## Dataset selection")
time_start = st.sidebar.date_input(
    "Start date", value=t0, min_value=t0, max_value=t1
)
time_end = st.sidebar.date_input(
    "End date", value=t1, min_value=time_start, max_value=t1
)

# Cut data down to time region
start_time = timerpc()
df = df_full
print("Time passed copying df: ", timerpc() - start_time)
df = df[
    (df_time_array >= pd.to_datetime(time_start))
    & (df_time_array <= pd.to_datetime(time_end))
].reset_index(drop=True)
print("Time passed resetting index: ", timerpc() - start_time)
num_turbines = dfm.get_num_turbines(df)
ws_pow_filtering.df = df
ws_pow_filtering.reset_filters()
print("Time passed writing df: ", timerpc() - start_time)

# Turbine specifier
turbines_options = ["all"]
turbines_options.extend(list(range(num_turbines)))
turbs_analyzed = st.sidebar.multiselect(
    label="Turbines to analyze", options=turbines_options, default=[0]
)

# Filter settings
st.sidebar.markdown("## Filtering methods")
filter_windows = st.sidebar.checkbox("Filter 1: windows", value=False)
filter_meanfromcurve = st.sidebar.checkbox(
    "Filter 2: distance from mean curve", value=True
)
filter_wsdev = st.sidebar.checkbox("Filter 3: by ws std. dev.", value=False)

# Set turbines to analyze
if "all" in turbs_analyzed:
    turbs_analyzed = "all"
ws_pow_filtering._set_turbine_mode(turbine_list=turbs_analyzed)
window_list = ws_pow_filtering.window_list

# Add new window
if filter_windows:
    st.sidebar.markdown("## FILTER 1: Window filter")
    st.sidebar.markdown("# Add or remove a window")
    if st.sidebar.button("Add new window"):
        est_rated_pow = ws_pow_filtering.rated_powers[0]
        window_list.append(
            {
                "idx": len(window_list),
                "ws_range": [0.0, 50.0],
                "pow_range": [0.0, 1.1 * est_rated_pow],
                "axis": int(0),
                "turbines": turbs_analyzed,
            }
        )

    # Remove window options
    if st.sidebar.button("Delete last window"):
        if len(window_list) > 0:
            window_list.pop(-1)

    # Show all windows
    windows_st_list = []
    for wdw in window_list:
        ws_range = (float(wdw["ws_range"][0]), float(wdw["ws_range"][1]))
        pow_range = (float(wdw["pow_range"][0]), float(wdw["pow_range"][1]))
        st.sidebar.markdown("# Window %d settings" % wdw["idx"])
        windows_st_list.append(
            [
                st.sidebar.slider(
                    "[%d] Wind speed range" % wdw["idx"],
                    0.0,
                    50.0,
                    ws_range,
                    0.5,
                    format="%.1f",
                ),
                st.sidebar.slider(
                    "[%d] Power range" % wdw["idx"],
                    0.0,
                    5500.0,
                    pow_range,
                    0.5,
                    format="%.1f",
                ),
                st.sidebar.selectbox(
                    label="[%d] Axis" % wdw["idx"],
                    options=[0, 1],
                    index=wdw["axis"],
                ),
                st.sidebar.multiselect(
                    label="[%d] Turbines" % wdw["idx"],
                    options=turbines_options,
                    default=["all"],
                ),
            ]
        )

    # Update filter values with streamlit app values
    ws_pow_filtering.window_list = window_list
    for i in range(len(window_list)):
        wdw = windows_st_list[i]
        turbine_list = wdw[3]
        if "all" in turbine_list:
            turbine_list = list(range(num_turbines))
        ws_pow_filtering.window_list[i]["ws_range"] = wdw[0]
        ws_pow_filtering.window_list[i]["pow_range"] = wdw[1]
        ws_pow_filtering.window_list[i]["axis"] = int(wdw[2])
        ws_pow_filtering.window_list[i]["turbines"] = turbine_list

if filter_meanfromcurve:
    st.sidebar.markdown("## FILTER 2: distance from mean curve")
    ws_ml = st.sidebar.slider(
        "Left bound multiplier", 0.80, 1.0, 0.95, step=0.01
    )
    ws_mr = st.sidebar.slider(
        "Right bound multiplier", 1.0, 1.20, 1.05, step=0.01
    )

if filter_wsdev:
    st.sidebar.markdown("## FILTER 3: Wind speed st. dev. filter")
    min_bin = st.sidebar.slider(
        "Minimum power for which to filter out deviations (kW)",
        0.0,
        5000.0,
        0.0,
        step=10.0,
    )
    max_bin = st.sidebar.slider(
        "Maximum power for which to filter out deviations (kW)",
        float(min_bin + 10.0),
        5000.0,
        float(np.max([4800.0, min_bin + 10.0])),
        step=10.0,
    )
    pow_step = st.sidebar.slider(
        "Power bin width (kW)", 5.0, 250.0, 50.0, step=5.0
    )
    wind_dev = st.sidebar.slider(
        "Maximum wind speed deviation within bin (m/s)",
        0.25,
        5.0,
        2.0,
        step=0.25,
    )

# Apply filters
if filter_windows:
    ws_pow_filtering.filter_by_windows()
if filter_meanfromcurve:
    ws_pow_filtering.filter_by_power_curve(m_ws_lb=ws_ml, m_ws_rb=ws_mr)
if filter_wsdev:
    ws_pow_filtering.filter_by_wsdev(
        pow_bin_width=pow_step,
        max_ws_dev=wind_dev,
        pow_min=min_bin,
        pow_max=max_bin,
    )

# Saving dataframe
st.sidebar.markdown("## Export filtered dataframe")
root_path = os.path.dirname(os.path.abspath(__file__))
savepath = st.sidebar.text_input(
    "Specify path",
    os.path.join(
        root_path, "data", "04_wspowcurve_filtered", "scada_data_60s.ftr"
    )
)
if st.sidebar.button("Save"):
    ws_pow_filtering.df.to_feather(savepath)
    st.sidebar.write("Saved dataframe to %s." % savepath)

# Plot and save data for current dataframe
st.markdown("## Filtering results")
figs = ws_pow_filtering.plot()

if turbs_analyzed == "all":
    turbs_analyzed = list(range(num_turbines))

for idx, f in enumerate(figs):
    ti = turbs_analyzed[idx]
    st.write("Turbine %03d" % ti)
    st.write(f)
