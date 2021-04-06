import numpy as np
import pandas as pd


from floris_scada_analysis import circular_statistics as cs
from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as ftools


# Create a demo dataframe
N = 100
df_full = pd.DataFrame()
wd_array = np.array([10., 12., 8., 5.])
ws_array = np.array([5., 17., 0., 29.])
ti_array = np.array([0.03, 0.09, 0.25, 0.30])
pow_array = np.array([1500., 1800., 3500., 50.])

for ti in range(len(wd_array)):
    df_full['wd_%03d' % ti] = np.repeat(wd_array[ti], N)
    df_full['ws_%03d' % ti] = np.repeat(ws_array[ti], N)
    df_full['ti_%03d' % ti] = np.repeat(ti_array[ti], N)
    df_full['pow_%03d' % ti] = np.repeat(pow_array[ti], N)

df_upstream = pd.DataFrame({'wd_min': [0., 180.],
                            'wd_max': [180., 360.],
                            'turbines': [[0, 1], [2, 3]]})


def check_function(a, b, description='', rtol=1.0e-5, atol=1.0e-8):
    if isinstance(a, float) or isinstance(a, int):
        a = float(a)
        b = float(a)
        check_bool = all(
            np.isclose(a, b, equal_nan=True, rtol=rtol, atol=atol)
            )
    elif isinstance(a, pd.Series) or isinstance(a, pd.DataFrame):
        a = np.array(a)
        b = np.array(b)
        check_bool = all(
            np.isclose(a, b, equal_nan=True, rtol=rtol, atol=atol)
            )
    elif isinstance(a, str):
        a = str(a)
        b = str(b)
        check_bool = (a == b)
    else:
        check_bool = (a == b)

    if check_bool:
        print('Function %s works correctly.' % description)
    else:
        raise KeyError('Issue found with function %s.' % description)


# Test set_*_by_all_turbines functions
df_test = df_full.copy()
df_test = dfm.set_wd_by_all_turbines(df_test)
wd_mean = cs.calc_wd_mean_radial(angles_array_deg=wd_array)
check_function(df_test['wd'], wd_mean, 'set_wd_by_all_turbines')

df_test = dfm.set_ws_by_all_turbines(df_test)
check_function(df_test['ws'], np.mean(ws_array), 'set_ws_by_all_turbines')

df_test = dfm.set_ti_by_all_turbines(df_test)
check_function(df_test['ti'], np.mean(ti_array), 'set_ti_by_all_turbines')


# Test set_*_by_turbines functions
df_test = df_full.copy()
turbine_list = [0, 2]
df_test = dfm.set_wd_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['wd'], np.mean(wd_array[turbine_list]), 'set_wd_by_turbines')

df_test = dfm.set_ws_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['ws'], np.mean(ws_array[turbine_list]), 'set_ws_by_turbines')

df_test = dfm.set_ti_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['ti'], np.mean(ti_array[turbine_list]), 'set_ti_by_turbines')


# Test set_*_by_turbines functions
df_test = df_full.copy()
turbine_list = [0, 2]
df_test = dfm.set_wd_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['wd'], np.mean(wd_array[turbine_list]), 'set_wd_by_turbines')

df_test = dfm.set_ws_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['ws'], np.mean(ws_array[turbine_list]), 'set_ws_by_turbines')

df_test = dfm.set_ti_by_turbines(df_test, turbine_numbers=turbine_list)
check_function(df_test['ti'], np.mean(ti_array[turbine_list]), 'set_ti_by_turbines')


# Test set_*_by_upstream_turbines functions
df_test = df_full.copy()
df_test = dfm.set_wd_by_all_turbines(df_test)
df_test = dfm.set_ws_by_upstream_turbines(df_test, df_upstream)
check_function(df_test['ws'], np.mean(ws_array[[0, 1]]), 'set_ws_by_upstream_turbines')

df_test = dfm.set_ti_by_upstream_turbines(df_test, df_upstream)
check_function(df_test['ti'], np.mean(ti_array[[0, 1]]), 'set_ti_by_upstream_turbines')

df_test = dfm.set_pow_ref_by_upstream_turbines(df_test, df_upstream)
check_function(df_test['pow_ref'], np.mean(pow_array[[0, 1]]), 'set_pow_ref_by_upstream_turbines')


# Test set_*_by_upstream_turbines_in_radius functions
df_test = df_full.copy()
df_test = dfm.set_wd_by_all_turbines(df_test)
df_test = dfm.set_ws_by_upstream_turbines_in_radius(
    df_test, df_upstream, turb_no=0, x_turbs=np.array([0., 500., 1000., 1500.]),
    y_turbs=np.array([0., 500., 1000., 1500.]), max_radius=1000, include_itself=True)
check_function(df_test['ws'], np.mean(ws_array[[0, 1]]), 'set_ws_by_upstream_turbines_in_radius (1)')

df_test = dfm.set_ws_by_upstream_turbines_in_radius(
    df_test, df_upstream, turb_no=0, x_turbs=np.array([0., 500., 1000., 1500.]),
    y_turbs=np.array([0., 500., 1000., 1500.]), max_radius=500, include_itself=True)
check_function(df_test['ws'], np.mean(ws_array[[0]]), 'set_ws_by_upstream_turbines_in_radius (2)')

df_test = dfm.set_ti_by_upstream_turbines_in_radius(
    df_test, df_upstream, turb_no=0, x_turbs=np.array([0., 500., 1000., 1500.]),
    y_turbs=np.array([0., 500., 1000., 1500.]), max_radius=1000, include_itself=False)
check_function(df_test['ti'], np.mean(ti_array[[1]]), 'set_ti_by_upstream_turbines_in_radius')

df_test = dfm.set_pow_ref_by_upstream_turbines_in_radius(
    df_test, df_upstream, turb_no=0, x_turbs=np.array([0., 500., 1000., 1500.]),
    y_turbs=np.array([0., 500., 1000., 1500.]), max_radius=1000, include_itself=True)
check_function(df_test['pow_ref'], np.mean(pow_array[[0, 1]]), 'set_pow_ref_by_upstream_turbines_in_radius')
