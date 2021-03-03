# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import pandas as pd
from scipy import interpolate


def _get_turbine_cutin_ws(fCpInterp):
    dx = np.diff(fCpInterp.x)
    dx = np.median(dx)
    ws_cutin_ti = fCpInterp.x[0] - dx
    ii = 0
    while fCpInterp.y[ii] < 1.0e-4:
        ws_cutin_ti = fCpInterp.x[ii]
        ii = ii + 1

    return ws_cutin_ti


def _get_turbine_cutout_ws(fCpInterp):
    dx = np.diff(fCpInterp.x)
    dx = np.median(dx)
    ws_cutout_ti = fCpInterp.x[-1] + dx
    ii = len(fCpInterp.x) - 1
    while fCpInterp.y[ii] < 1.0e-4:
        ws_cutout_ti = fCpInterp.x[ii]
        ii = ii - 1

    return ws_cutout_ti


# Define an approximate calc_floris() function
def calc_floris(df, fi):
    """Calculate the FLORIS predictions for a particular wind direction, wind speed
    and turbulence intensity set. This function calculates the exact solutions.

    Args:
        df ([pd.DataFrame]): Dataframe with at least the columns 'time', 'wd' 
        and 'ws'. Can optionally also have the column 'ti' and 'time'. Any
        other column will be ignored.
        fi ([FlorisInterface]): Floris object for the wind farm of interest

    Returns:
        [type]: [description]
    """

    num_turbines = len(fi.layout_x)

    # Start by ensuring simple index for df
    df = df.reset_index(drop=('time' in df.columns))

    # Generate new dataframe called df_out
    if not ('ti' in df.columns):
        df['ti'] = np.min(fi.floris.farm.turbulence_intensity)

    if 'time' in df.columns:
        df_out = df[['time', 'wd', 'ws', 'ti']]
    else:
        df_out = df[['wd', 'ws', 'ti']]

    # Create placeholders for turbine measurements
    for colname in ['pow', 'ws', 'wd', 'ti']:
        for ti in range(num_turbines):
            df_out[colname + '_%03d' % ti] = 0.0

    for idx in range(df_out.shape[0]):
        if (np.remainder(idx, 100) == 0) or idx == df_out.shape[0]-1:  # Print output every 100 steps:
            print('  Progress: finished %.1f percent (%d/%d cases).'
                  % (100.*idx/df_out.shape[0], idx, df_out.shape[0]))

        fi.reinitialize_flow_field(wind_speed=df_out.loc[idx, 'ws'],
                                   wind_direction=df_out.loc[idx, 'wd'],
                                   turbulence_intensity=df_out.loc[idx, 'ti'])
        fi.calculate_wake()

        for ti in range(num_turbines):
            df_out.loc[idx, 'pow_%03d' % ti] = np.array(fi.get_turbine_power())[ti]/1000.
            df_out.loc[idx, 'wd_%03d' % ti] = df_out.loc[idx, 'wd']  # Assume uniform for now
            df_out.loc[idx, 'ws_%03d' % ti] = fi.floris.farm.turbines[ti].average_velocity
            df_out.loc[idx, 'ti_%03d' % ti] = fi.floris.farm.turbines[ti]._turbulence_intensity

    print('Finished calculating the FLORIS solutions for the dataframe.')
    return df_out


def calc_floris_approx(df, fi, ws_step=0.5, wd_step=1.0, ti_step=None, method='linear'):
    """Calculate the FLORIS predictions for a particular wind direction, wind speed
    and turbulence intensity set. This function approximates the exact solutions
    by binning the wd, ws and ti, calculating floris for the mean of those bins,
    and then finally mapping those solutions using linear/nearest-neighbor
    interpolation.

    Args:
        df ([pd.DataFrame]): Dataframe with at least the columns 'time', 'wd' 
        and 'ws'. Can optionally also have the column 'ti' and 'time'. Any
        other column will be ignored.
        fi ([FlorisInterface]): Floris object for the wind farm of interest
        ws_step (float, optional): Wind speed bin width in m/s. Defaults to 0.5.
        wd_step (float, optional): Wind direction bin width in deg. Defaults to 1.0.
        ti_step ([type], optional): Turbulence intensity bin width in [-]. Should be 
        a value between 0 and 1.0. If left empty, will assume one fixed value for TI
        and not bin over various options. This significantly speeds up calculations.
        Defaults to None.
        method (str, optional): Interpolation method. Options are 'linear' and
        'nearest'. Nearest is faster but linear is more accurate. Defaults to
        'linear'.

    Returns:
        [type]: [description]
    """

    num_turbines = len(fi.layout_x)

    # Start by ensuring simple index for df
    df = df.reset_index(drop=('time' in df.columns))

    # Derive bins from wd_array and ws_array
    ws_array = df['ws']
    wd_array = df['wd']
    if 'ti' in df.columns:
        ti_array = df['ti']
    else:
        ti_fi = np.min(fi.floris.farm.turbulence_intensity)
        ti_array = np.repeat(ti_fi, df.shape[0])

    wd_min = np.max([np.min(wd_array), 0.0])
    wd_max = np.min([np.max(wd_array), 360.0])

    ws_cutin = [_get_turbine_cutin_ws(t.fCpInterp) for t in fi.floris.farm.turbines]
    ws_cutin = np.min(ws_cutin)  # Take the minimum of all
    ws_cutout = [_get_turbine_cutout_ws(t.fCpInterp) for t in fi.floris.farm.turbines]
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

    if df.shape[0] <= df_approx.shape[0]:
        print('Approximation would not reduce number of cases with the current settings (N_raw=' +str(N_raw)+', N_approx='+str(N_approx)+')')
        print('Calculating the exact solutions for this dataset. Avoiding any approximations.')
        return calc_floris(df, fi)

    # Calculate approximate solutions
    print('Reducing calculations from ' + str(df.shape[0]) + ' to ' + str(df_approx.shape[0]) + ' cases.')
    df_approx = calc_floris(df_approx, fi)

    # for idx in range(df_approx.shape[0]):
    #     if (np.remainder(idx, 100) == 0) or idx == df_approx.shape[0]-1:  # Print output every 100 steps:
    #         print('  Progress: finished %.1f percent (%d/%d cases).'
    #               % (100.*idx/df_approx.shape[0], idx, df_approx.shape[0]))

    #     fi.reinitialize_flow_field(wind_speed=df_approx.loc[idx, 'ws'],
    #                                wind_direction=df_approx.loc[idx, 'wd'],
    #                                turbulence_intensity=df_approx.loc[idx, 'ti'])
    #     fi.calculate_wake()

    #     for ti in range(num_turbines):
    #         df_approx.loc[idx, 'pow_%03d' % ti] = np.array(fi.get_turbine_power())[ti]/1000.
    #         df_approx.loc[idx, 'wd_%03d' % ti] = df_approx.loc[idx, 'wd']  # Assume uniform for now
    #         df_approx.loc[idx, 'ws_%03d' % ti] = fi.floris.farm.turbines[ti].average_velocity
    #         df_approx.loc[idx, 'ti_%03d' % ti] = fi.floris.farm.turbines[ti]._turbulence_intensity

    # Map individual data entries to full DataFrame
    print('Now mapping the precalculated solutions from FLORIS to the dataframe entries...')
    print("  Creating a gridded interpolant with interpolation method '" + method + "'.")

    # Create interpolant
    if xyz_grid.shape[3]==1:
        print('    Performing 2D interpolation')
        shape_y = [len(wd_array_approx), len(ws_array_approx)]
        values = np.reshape(np.array(df_approx['pow_000']), shape_y)
        xyz_tuple = (wd_array_approx, ws_array_approx)
    else:
        print('    Performing 3D interpolation')
        shape_y = [len(wd_array_approx), len(ws_array_approx), len(ti_array_approx)]
        values = np.reshape(np.array(df_approx['pow_000']), shape_y)
        xyz_tuple = (wd_array_approx, ws_array_approx, ti_array_approx)
    f = interpolate.RegularGridInterpolator(xyz_tuple, values,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=np.nan)

    # Create a new dataframe based on df
    df_out = df[['time', 'wd', 'ws']].copy()
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
            if xyz_grid.shape[3]==1:
                df_out[colname] = f(df[['wd', 'ws']])
            else:
                df_out[colname] = f(df[['wd', 'ws', 'ti']])

    # Overwrite the np.nan values for entries where ws < ws_cutin
    idxs_lowws = (df_out['ws'] < ws_cutin)
    df_out.loc[idxs_lowws, ['wd_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'wd']
    df_out.loc[idxs_lowws, ['ws_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'ws']
    df_out.loc[idxs_lowws, ['ti_%03d' % ti for ti in range(num_turbines)]] = df_out.loc[idxs_lowws, 'ti']
    df_out.loc[idxs_lowws, ['pow_%03d' % ti for ti in range(num_turbines)]] = 0.0

    print('Finished calculating the FLORIS solutions for the dataframe.')
    return df_out


def get_turbs_in_radius(x_turbs, y_turbs, turb_no, max_radius, include_itself):
    dr_turb = np.sqrt((x_turbs - x_turbs[turb_no])**2.0 + (y_turbs - y_turbs[turb_no])**2.0)
    turbs_within_radius = np.where(dr_turb <= max_radius)[0]
    if not include_itself:
        turbs_within_radius = [ti for ti in turbs_within_radius if not ti == turb_no]

    return turbs_within_radius


def get_upstream_turbs_floris(fi, wd_step=1.0, verbose=False):
    if verbose:
        print('Determining upstream turbines using FLORIS for wd_step = %.1f deg.' %(wd_step))
    upstream_turbs_ids = []  # turbine numbers that are freestream
    upstream_turbs_wds = []  # lower bound of bin
    for wd in np.arange(0., 360., wd_step):
        fi.reinitialize_flow_field(wind_direction=wd, wind_speed=8.0)
        fi.calculate_wake()
        power_out = np.array(fi.get_turbine_power())
        power_wake_loss = np.max(power_out) - power_out
        turbs_freestream = list(np.where(power_wake_loss < 0.01)[0])
        if len(upstream_turbs_wds) == 0:
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)
        elif not(turbs_freestream == upstream_turbs_ids[-1]):
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)

    # Connect at 360 degrees
    if upstream_turbs_ids[0] == upstream_turbs_ids[-1]:
        upstream_turbs_wds.pop(0)
        upstream_turbs_ids.pop(0)

    # Go from list to bins for upstream_turbs_wds
    upstream_turbs_wds = [[upstream_turbs_wds[i], upstream_turbs_wds[i+1]] for i in range(len(upstream_turbs_wds)-1)]
    upstream_turbs_wds.append([upstream_turbs_wds[-1][-1], upstream_turbs_wds[0][0]])

    df_upstream = pd.DataFrame({'wd_min': [wd[0] for wd in upstream_turbs_wds],
                                'wd_max': [wd[1] for wd in upstream_turbs_wds],
                                'turbines': upstream_turbs_ids})

    return df_upstream


if __name__ == '__main__':
    import os
    import floris.tools as wfct

    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../examples/demo_dataset/demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)

    df_upstream = get_upstream_turbs_floris(fi)
    print(df_upstream)