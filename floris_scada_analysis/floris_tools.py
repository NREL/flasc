import numpy as np
import pandas as pd
from scipy import interpolate

def get_turbine_cutin_ws(fCpInterp):
    dx = np.diff(fCpInterp.x)
    dx = np.median(dx)
    ws_cutin_ti = fCpInterp.x[0] - dx
    ii = 0
    while fCpInterp.y[ii] < 1.0e-4:
        ws_cutin_ti = fCpInterp.x[ii]
        ii = ii + 1

    return ws_cutin_ti


def get_turbine_cutout_ws(fCpInterp):
    dx = np.diff(fCpInterp.x)
    dx = np.median(dx)
    ws_cutout_ti = fCpInterp.x[-1] + dx
    ii = len(fCpInterp.x) - 1
    while fCpInterp.y[ii] < 1.0e-4:
        ws_cutout_ti = fCpInterp.x[ii]
        ii = ii - 1

    return ws_cutout_ti


# Define an approximate calc_floris() function
def calc_floris_approx(df, fi, ws_step=0.5, wd_step=1.0, ti_step=None, method='linear'):
    num_turbines = len(fi.layout_x)

    # Start by ensuring simple index for df
    df = df.reset_index(drop=True)

    # Derive bins from wd_array and ws_array
    ws_array = df['ws']
    wd_array = df['wd']
    if 'ti' in df.columns:
        ti_array = df['ti']
    else:
        ti_array = np.repeat(0.06, df.shape[0])

    wd_min = np.max([np.min(wd_array), 0.0])
    wd_max = np.min([np.max(wd_array), 360.0])

    ws_cutin = [get_turbine_cutin_ws(t.fCpInterp) for t in fi.floris.farm.turbines]
    ws_cutin = np.min(ws_cutin)  # Take the minimum of all
    ws_cutout = [get_turbine_cutout_ws(t.fCpInterp) for t in fi.floris.farm.turbines]
    ws_cutout = np.max(ws_cutout)
    ws_min = np.max([np.min(ws_array), ws_cutin]) 
    ws_max = np.min([np.max(ws_array), ws_cutout])

    ti_min = np.max([np.min(ti_array), 0.0])
    ti_max = np.min([np.max(ti_array), 0.30])

    wd_array_approx = np.arange(wd_min, wd_max + wd_step, wd_step)
    ws_array_approx = np.arange(ws_min, ws_max + ws_step, ws_step)
    if ti_step is None:
        ti_array_approx = np.min(fi.floris.farm.turbulence_intensity)
    else:
        ti_array_approx = np.arange(ti_min, ti_max + ti_step, ti_step)

    xyz_grid = np.array(np.meshgrid(
        wd_array_approx, ws_array_approx, ti_array_approx, indexing='ij'))
    df_approx = pd.DataFrame(
        {'wd': np.reshape(xyz_grid[0], [-1, 1]).flatten(),
         'ws': np.reshape(xyz_grid[1], [-1, 1]).flatten(),
         'ti': np.reshape(xyz_grid[2], [-1, 1]).flatten()})

    for colname in ['pow', 'ws', 'wd', 'ti']:
        for ti in range(num_turbines):
            df_approx[colname + '_%03d' % ti] = 0.0

    if df.shape[0] <= df_approx.shape[0]:
        print('Approximation would not reduce number of cases with the current settings (N_raw=' +str(N_raw)+', N_approx='+str(N_approx)+')')
        print('Calculating the exact solutions for this dataset. Avoiding any approximations.')
        breakpoint()
        # return calc_floris()

    print('Reducing calculations from ' + str(df.shape[0]) + ' to ' + str(df_approx.shape[0]) + ' cases.')
    for idx in range(df_approx.shape[0]):
        if (np.remainder(idx, 100) == 0) or idx == df_approx.shape[0]-1:  # Print output every 100 steps:
            print('  Progress: finished %.1f percent (%d/%d cases).'
                  % (100.*idx/df_approx.shape[0], idx, df_approx.shape[0]))

        fi.reinitialize_flow_field(wind_speed=df_approx.loc[idx, 'ws'],
                                   wind_direction=df_approx.loc[idx, 'wd'],
                                   turbulence_intensity=df_approx.loc[idx, 'ti'])
        fi.calculate_wake()

        for ti in range(num_turbines):
            df_approx.loc[idx, 'pow_%03d' % ti] = np.array(fi.get_turbine_power())[ti]/1000.
            df_approx.loc[idx, 'wd_%03d' % ti] = df_approx.loc[idx, 'wd']  # Assume uniform for now
            df_approx.loc[idx, 'ws_%03d' % ti] = fi.floris.farm.turbines[ti].average_velocity
            df_approx.loc[idx, 'ti_%03d' % ti] = fi.floris.farm.turbines[ti]._turbulence_intensity

    # Map individual data entries to full DataFrame
    print('Now mapping the precalculated solutions from FLORIS to the dataframe entries...')
    print("  Creating a gridded interpolant with interpolation method '" + method + "'.")
    
    # Create interpolant
    if len(ti_array_approx) <= 1:
        print('    Performing 2D interpolation')
        shape_y = [len(wd_array_approx), len(ws_array_approx)]
        values = np.reshape(np.array(df_approx['pow_000']), shape_y)
        xyz_tuple = (wd_array_approx, ws_array_approx)
    else:
        print('    Performing 3D interpolation')
        shape_y = [len(wd_array_approx), len(ws_array_approx), len(ti_array_approx)]
        # values = np.reshape(np.array(df_approx['pow_000']), shape_y)
        values = np.reshape(np.array(df_approx['pow_000']), shape_y)
        xyz_tuple = (wd_array_approx, ws_array_approx, ti_array_approx)
    f = interpolate.RegularGridInterpolator(xyz_tuple, values,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=np.nan)

    # Create a new dataframe based on df
    df_out = df[['time', 'wd', 'ws']]
    if 'ti' in df.columns:
        df_out['ti'] = df['ti']
    else:
        df_out['ti'] = np.nan

    # Use interpolant to determine values for all turbines and variables
    for varname in ['pow', 'wd', 'ws', 'ti']:
        print('     Interpolating ' + varname + ' for all turbines...')
        for ti in range(num_turbines):
            colname = varname + '_%03d' % ti
            f.values = np.reshape(np.array(df_approx[colname]), shape_y)
            df_out[colname] = f(df[['wd', 'ws', 'ti']])

    # Overwrite the np.nan values for entries where ws < ws_cutin
    idxs_lowws = (df_out['ws'] < ws_cutin)
    df_out.loc[idxs_lowws, ['wd_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'wd']
    df_out.loc[idxs_lowws, ['ws_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'ws']
    df_out.loc[idxs_lowws, ['ti_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'ti']
    df_out.loc[idxs_lowws, ['pow_%03d' % ti for ti in range(num_turbines)]] = 0.0

    print('Finished calculating the FLORIS solutions for the dataframe.')
    return df_out
