# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from copy import deepcopy as dcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import DataError
from scipy import interpolate
from time import perf_counter as timerpc

from flasc import utilities as fsut

from floris.utilities import wrap_360, wrap_180


def _run_fi_serial(df_subset, fi, include_unc=False,
                   unc_pmfs=None, unc_options=None, verbose=False):
    """Evaluate the FLORIS solutions for a set of wind directions,
    wind speeds and turbulence intensities in serial (non-
    parallelized) mode.

    Args:
        df_subset ([pd.DataFrame]): Dataframe containing the columns
        'wd', 'ws' and 'ti'. The FLORIS power predictions will be
        calculated for each row/set of ambient conditions.
        fi ([floris]): FLORIS object for the farm of interest.
        verbose (bool, optional): Print information to terminal, used
        for debugging. Defaults to False.

    Returns:
        df_out ([pd.DataFrame]): Identical to the inserted dataframe,
        df_subset, but now with additional columns containing the
        predicted power production for each turbine, as pow_000, ...
        pow_00N.
    """
    nturbs = len(fi.layout_x)
    df_out = df_subset.sort_values(by=["wd", "ws"])

    use_model_params = ('model_params_dict' in df_subset.columns)
    use_yaw = ('yaw_000' in df_subset.columns)

    if (use_model_params | include_unc):
        raise NotImplementedError("Functionality not yet implemented since moving to floris v3.0.")

    # Specify dataframe columns
    pow_cols = ["pow_{:03d}".format(ti) for ti in range(nturbs)]
    ws_cols = ["ws_{:03d}".format(ti) for ti in range(nturbs)]
    wd_cols = ["wd_{:03d}".format(ti) for ti in range(nturbs)]
    ti_cols = ["ti_{:03d}".format(ti) for ti in range(nturbs)]
    
    yaw_rel = np.zeros((df_out.shape[0], nturbs))
    if use_yaw:
        yaw_cols = ['yaw_%03d' % ti for ti in range(nturbs)]
        wd = np.array(df_out['wd'], dtype=float)
        yaw_rel = wrap_180(
            (
                np.array(df_out[yaw_cols], dtype=float) -
                np.stack((wd,) * nturbs, axis=0).T
            )
        )

        if np.any(np.abs(yaw_rel) > 30.0):
            raise DataError('Yaw should be defined in domain [0, 360) deg.')

    if 'ti' not in df_out.columns:
        df_out['ti'] = np.min(fi.floris.farm.turbulence_intensity)

    # Perform grid-style calculation, if possible
    n_unq = (
        df_out["ws"].nunique() *
        df_out["wd"].nunique() *
        df_out["ti"].nunique()
    )
    if n_unq == df_out.shape[0]:
        # Reformat things to grid style calculation
        wd_array = np.sort(df_out["wd"].unique())
        ws_array = np.sort(df_out["ws"].unique())
        ti = df_out["ti"].unique()[0]

        # Specify interpolant to map data appropriately
        X, Y = np.meshgrid(wd_array, ws_array, indexing='ij')
        if use_yaw:
            # Map the yaw angles in the appropriate format
            F = interpolate.NearestNDInterpolator(
                df_out[["wd", "ws"]],
                yaw_rel
            )
            yaw_angles = F(X, Y)
        else:
            yaw_angles = np.zeros((len(wd_array), len(ws_array), nturbs))

        # Calculate the FLORIS solutions in grid-style
        fi.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array,
            turbulence_intensity=ti,
        )
        fi.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = fi.get_turbine_powers(
            # include_unc=include_unc,
            # unc_pmfs=unc_pmfs,
            # unc_options=unc_options
        )

        # Format the found solutions back to the dataframe format
        Fp = interpolate.NearestNDInterpolator(
            np.vstack([X.flatten(), Y.flatten()]).T,
            np.reshape(turbine_powers, (-1, nturbs))
        )
        Fws = interpolate.NearestNDInterpolator(
            np.vstack([X.flatten(), Y.flatten()]).T,
            np.reshape(
                np.mean(fi.floris.flow_field.u, axis=(3, 4)),
                (-1, nturbs)
            )
        )
        Fti = interpolate.NearestNDInterpolator(
            np.vstack([X.flatten(), Y.flatten()]).T,
            np.reshape(
                fi.floris.flow_field.turbulence_intensity_field[:, :, :, 0, 0],
                (-1, nturbs)
            )
        )

        # Finally save solutions to the dataframe
        df_out.loc[df_out.index, pow_cols] = Fp(df_out[["wd", "ws"]]) / 1000.0
        df_out.loc[df_out.index, wd_cols] = np.tile(df_out["wd"], (nturbs, 1)).T
        df_out.loc[df_out.index, ws_cols] = Fws(df_out[["wd", "ws"]])
        df_out.loc[df_out.index, ti_cols] = Fti(df_out[["wd", "ws"]])

    else:
        # If cannot process in grid-style format, process one by one (SLOW)
        for iii, idx in enumerate(df_out.index):
            if (
                verbose and
                ((np.remainder(idx, 100) == 0) or idx == df_out.shape[0]-1)
            ): 
                print('  Progress: finished %.1f percent (%d/%d cases).'
                    % (100.*idx/df_out.shape[0], idx, df_out.shape[0]))

            # # Update model parameters, if present in dataframe
            # if use_model_params:
            #     params = df_out.loc[idx, 'model_params_dict']
            #     fi.set_model_parameters(params=params, verbose=False)

            fi.reinitialize(
                wind_speeds=[df_out.loc[idx, 'ws']],
                wind_directions=[df_out.loc[idx, 'wd']],
                turbulence_intensity=df_out.loc[idx, 'ti']
            )

            fi.calculate_wake(np.expand_dims(yaw_rel[iii, :], axis=[0, 1]))
            turbine_powers = np.squeeze(
                fi.get_turbine_powers(
                # include_unc=include_unc,
                # unc_pmfs=unc_pmfs,
                # unc_options=unc_options
                )
            )
            df_out.loc[idx, pow_cols] = turbine_powers / 1000.
            df_out.loc[idx, wd_cols] = np.repeat(
                df_out.loc[idx, 'wd'],
                nturbs   # Assumed to be uniform
            ) 
            df_out.loc[idx, ws_cols] = np.squeeze(
                np.mean(fi.floris.flow_field.u, axis=(3, 4))
            )
            df_out.loc[idx, ti_cols] = np.squeeze(
                fi.floris.flow_field.turbulence_intensity_field
            )

    return df_out


def calc_floris(df, fi, num_workers, job_worker_ratio=5, include_unc=False,
                unc_pmfs=None, unc_options=None, use_mpi=False):
    """Calculate the FLORIS predictions for a particular wind direction, wind speed
    and turbulence intensity set. This function calculates the exact solutions.

    Args:
        df ([pd.DataFrame]): Dataframe with at least the columns 'time', 'wd'
        and 'ws'. Can optionally also have the column 'ti' and 'time'.

        If the dataframe has columns 'yaw_000' through 'yaw_<nturbs>', then it
        will calculate the floris solutions for those yaw angles too.

        If the dataframe has column 'model_params_dict', then it will change
        the floris model parameters for every run with the values therein.

        fi ([FlorisInterface]): Floris object for the wind farm of interest

    Returns:
        [type]: [description]
    """

    nturbs = len(fi.layout_x)

    # Create placeholders
    df[['pow_%03d' % ti for ti in range(nturbs)]] = np.nan

    # Copy yaw angles, if possible
    yaw_cols = ['yaw_%03d' % ti for ti in range(nturbs)]
    yaw_cols = [c for c in yaw_cols if c in df.columns]
    if len(yaw_cols) > 0:
        if np.any(df[yaw_cols] < 0.):
            raise DataError('Yaw should be defined in domain [0, 360) deg.')

    # Split dataframe into subset dataframes for parallelization, if necessary
    if num_workers > 1:
        df_list = []

        # See if we can simply split the problem up into a grid of conditions
        num_jobs = num_workers * job_worker_ratio
        n_unq = df["ws"].nunique() * df["wd"].nunique() * df["ti"].nunique()
        if n_unq == df.shape[0]:
            # Data is a grid of atmospheric conditions. Can divide and exploit
            # the benefit of grid processing in floris v3.0.
            Nconds_per_ti = df["ws"].nunique() * df["wd"].nunique()
            Njobs_per_ti = int(np.floor(num_jobs / df["ti"].nunique()))
            dN = int(np.ceil(Nconds_per_ti / Njobs_per_ti))

            for ti in df["ti"].unique():
                df_subset = df[df["ti"] == ti]
                for ij in range(Njobs_per_ti):
                    df_list.append(df_subset.iloc[(ij*dN):((ij+1)*dN)])
                
        else:
            # If cannot be formatted to grid style, split blindly
            dN = int(np.ceil(df.shape[0] / num_jobs))
            for ij in range(num_jobs):
                df_list.append(df.iloc[(ij*dN):((ij+1)*dN)])


    # Calculate solutions
    start_time = timerpc()
    if num_workers <= 1:
        print("Calculating floris solutions (non-parallelized)")
        df_out = _run_fi_serial(
            df_subset=df,
            fi=fi,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options,
            verbose=True
        )
    else:
        print('Calculating with num_workers = %d and job_worker_ratio = %d'
            % (num_workers, job_worker_ratio))
        print('Each thread contains about %d FLORIS evaluations.' % dN)

        # Define a tuple of arguments
        multiargs = []
        for df_mp in df_list:
            df_mp = df_mp.reset_index(drop=True)
            multiargs.append(
                (df_mp, dcopy(fi), include_unc, unc_pmfs, unc_options, False)
            )

        if use_mpi:
            # Use an MPI implementation, useful for HPC
            from mpi4py.futures import MPIPoolExecutor as pool_executor
        else:
            # Use Pythons internal multiprocessing functionality
            from multiprocessing import Pool as pool_executor

        with pool_executor(num_workers) as pool:
            df_list = pool.starmap(_run_fi_serial, multiargs)

        df_out = pd.concat(df_list).reset_index(drop=True)
        if 'index' in df_out.columns:
            df_out = df_out.drop(columns='index')

    t = timerpc() - start_time
    print('Finished calculating the FLORIS solutions for the dataframe.')
    print('Total wall time: %.3f s.' % t)
    print('Mean wall time / function evaluation: %.3f s.' % (t/df.shape[0]))
    return df_out


def interpolate_floris_from_df_approx(
    df,
    df_approx,
    method='linear',
    verbose=True
):
    # Format dataframe and get number of turbines
    df = df.reset_index(drop=('time' in df.columns))
    nturbs = fsut.get_num_turbines(df_approx)

    # Check if turbulence intensity is provided in the dataframe 'df'
    if 'ti' not in df.columns:
        if df_approx["ti"].nunique() > 3:
            raise ValueError("You must include a 'ti' column in your df.")
        ti_ref = np.median(df_approx["ti"])
        print("No 'ti' column found in dataframe. Assuming {}".format(ti_ref))
        df["ti"] = ti_ref

    # Define which variables we must map from df_approx to df
    varnames = ['pow']
    if 'ws_000' in df_approx.columns:
        varnames.append('ws')
    if 'wd_000' in df_approx.columns:
        varnames.append('wd')
    if 'ti_000' in df_approx.columns:
        varnames.append('ti')

    # Map individual data entries to full DataFrame
    if verbose:
        print("Mapping the precalculated solutions " +
              "from FLORIS to the dataframe...")
        print("  Creating a gridded interpolant with " +
              "interpolation method '%s'." % method)

    # Make a copy from wd=0.0 deg to wd=360.0 deg for wrapping
    if not (df_approx["wd"] == 360.0).any():
        df_subset = df_approx[df_approx["wd"] == 0.0].copy()
        df_subset["wd"] = 360.0
        df_approx = pd.concat([df_approx, df_subset], axis=0).reset_index(drop=True)

    # Copy TI to lower and upper bound
    df_ti_lb = df_approx.loc[df_approx["ti"] == df_approx['ti'].min()].copy()
    df_ti_ub = df_approx.loc[df_approx["ti"] == df_approx['ti'].max()].copy()
    df_ti_lb["ti"] = 0.0
    df_ti_ub["ti"] = 1.0
    df_approx = pd.concat(
        [df_approx, df_ti_lb, df_ti_ub],
        axis=0
    ).reset_index(drop=True)

    # Copy WS to lower and upper bound
    df_ws_lb = df_approx.loc[df_approx["ws"] == df_approx['ws'].min()].copy()
    df_ws_ub = df_approx.loc[df_approx["ws"] == df_approx['ws'].max()].copy()
    df_ws_lb["ws"] = 0.0
    df_ws_ub["ws"] = 99.0
    df_approx = pd.concat(
        [df_approx, df_ws_lb, df_ws_ub],
        axis=0
    ).reset_index(drop=True)

    # Convert df_approx dataframe into a regular grid
    wd_array_approx = np.sort(df_approx["wd"].unique())
    ws_array_approx = np.sort(df_approx["ws"].unique())
    ti_array_approx = np.sort(df_approx["ti"].unique())
    xg, yg, zg = np.meshgrid(
        wd_array_approx,
        ws_array_approx,
        ti_array_approx,
        indexing='ij',
    )

    grid_dict = dict()
    for varname in varnames:
        colnames = ['{:s}_{:03d}'.format(varname, ti) for ti in range(nturbs)]
        f = interpolate.NearestNDInterpolator(
            df_approx[["wd", "ws", "ti"]],
            df_approx[colnames]
        )
        grid_dict["{:s}".format(varname)] = f(xg, yg, zg)

    # Prepare an minimal output dataframe
    cols_to_copy = ["wd", "ws", "ti"]
    if "time" in df.columns:
        cols_to_copy.append("time")
    df_out = df[cols_to_copy].copy()

    # Use interpolant to determine values for all turbines and variables
    for varname in varnames:
        if verbose:
            print('     Interpolating ' + varname + ' for all turbines...')
        colnames = ['{:s}_{:03d}'.format(varname, ti) for ti in range(nturbs)]
        f = interpolate.RegularGridInterpolator(
            points=(wd_array_approx, ws_array_approx, ti_array_approx),
            values=grid_dict[varname],
            method=method,
            bounds_error=False,
        )
        df_out.loc[df_out.index, colnames] = f(df[['wd', 'ws', 'ti']])

    return df_out


def calc_floris_approx_table(
    fi,
    wd_array=np.arange(0.0, 360.0, 1.0),
    ws_array=np.arange(0.001, 26.001, 1.0),
    ti_array=None,
    ):

    # if ti_array is None, use the current value in the FLORIS object
    if ti_array is None:
        ti = fi.floris.flow_field.turbulence_intensity
        ti_array = np.array([ti], dtype=float)

    fi = fi.copy()  # Create independent copy that we can manipulate
    num_turbines = len(fi.layout_x)

    # Format input arrays
    wd_array = np.sort(wd_array)
    ws_array = np.sort(ws_array)
    ti_array = np.sort(ti_array)
    wd_mesh, ws_mesh = np.meshgrid(wd_array, ws_array, indexing='ij')
    N_approx = len(wd_array) * len(ws_array) * len(ti_array)
    print(
        'Generating a df_approx table of FLORIS solutions ' +
        'covering a total of {:d} cases.'.format(N_approx)
    )

    # Create solutions, one set per turbulence intensity
    df_list = []
    for turb_intensity in ti_array:
        # Calculate solutions
        fi.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array,
            turbulence_intensity=turb_intensity,
        )
        fi.calculate_wake()
        turbine_powers = fi.get_turbine_powers()

        # Create a dictionary to save solutions in
        solutions_dict = {"wd": wd_mesh.flatten(), "ws": ws_mesh.flatten()}
        solutions_dict["ti"] = turb_intensity * np.ones(len(wd_array) * len(ws_array))
        for turbi in range(num_turbines):
            solutions_dict["pow_{:03d}".format(turbi)] = \
                turbine_powers[:, :, turbi].flatten()
        df_list.append(pd.DataFrame(solutions_dict))

    print('Finished calculating the FLORIS solutions for the dataframe.')
    df_approx = pd.concat(df_list, axis=0).sort_values(by=["ti", "ws", "wd"])
    df_approx = df_approx.reset_index(drop=True)

    return df_approx


def get_turbs_in_radius(x_turbs, y_turbs, turb_no, max_radius,
                        include_itself, sort_by_distance=False):
    """Determine which turbines are within a certain radius of other
    wind turbines.

    Args:
        x_turbs ([list, array]): Long. locations of turbines
        y_turbs ([list, array]): Lat. locations of turbines
        turb_no ([int]): Turbine number for which the distance is
        calculated w.r.t. the other turbines.
        max_radius ([float]): Maximum distance between turbines to be
        considered within the radius of turbine [turb_no].
        include_itself ([type]): Include itself in the list of turbines
        within the radius.
        sort_by_distance (bool, optional): Sort the output list of turbines
        according to distance to the turbine, from closest to furthest (but
        still within radius). Defaults to False.

    Returns:
        turbs_within_radius ([list]): List of turbines that are within the
        prespecified distance from turbine [turb_no].
    """
    dr_turb = np.sqrt((x_turbs - x_turbs[turb_no])**2.0 +
                      (y_turbs - y_turbs[turb_no])**2.0)
    turbine_list = np.array(range(len(x_turbs)))

    if sort_by_distance:
        indices_sorted = np.argsort(dr_turb)
        dr_turb = dr_turb[indices_sorted]
        turbine_list = turbine_list[indices_sorted]

    turbs_within_radius = turbine_list[dr_turb <= max_radius]
    if not include_itself:
        turbs_within_radius = [ti for ti in turbs_within_radius
                               if not ti == turb_no]

    return turbs_within_radius


def get_upstream_turbs_floris(fi, wd_step=0.1, wake_slope=0.10,
                              plot_lines=False):
    """Determine which turbines are operating in freestream (unwaked)
    flow, for the entire wind rose. This function will return a data-
    frame where each row will present a wind direction range and a set
    of wind turbine numbers for which those turbines are operating
    upstream. This is useful in determining the freestream conditions.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wd_step (float, optional): Wind direction discretization step.
        It will test what the upstream turbines are every [wd_step]
        degrees. A lower number means more accurate results, but
        typically there's no real benefit below 2.0 deg or so.
        Defaults to 0.1.
        wake_slope (float, optional): linear slope of the wake (dy/dx)
        plot_lines (bool, optional): Enable plotting wakes/turbines.
        Defaults to False.

    Returns:
        df_upstream ([pd.DataFrame]): A Pandas Dataframe in which each row
        contains a wind direction range and a list of turbine numbers. For
        that particular wind direction range, the turbines numbered are
        all upstream according to the FLORIS predictions. Depending on
        the FLORIS model parameters and ambient conditions, these results
        may vary slightly. Though, having minimal wake losses should not
        noticably affect your outcomes. Empirically, this approach has
        yielded good results with real SCADA data for determining what
        turbines are waked/unwaked and has served useful for determining
        what turbines to use as reference.
    """

    # Get farm layout
    x = fi.layout_x
    y = fi.layout_y
    n_turbs = len(x)
    D = [t["rotor_diameter"] for t in fi.floris.farm.turbine_definitions]
    D = np.array(D, dtype=float)

    # Setup output list
    upstream_turbs_ids = []  # turbine numbers that are freestream
    upstream_turbs_wds = []  # lower bound of bin

    # Rotate farm and determine freestream/waked turbines
    for wd in np.arange(0., 360., wd_step):
        is_freestream = [True for _ in range(n_turbs)]
        x_rot = (np.cos((wd-270.) * np.pi / 180.) * x -
                 np.sin((wd-270.) * np.pi / 180.) * y)
        y_rot = (np.sin((wd-270.) * np.pi / 180.) * x +
                 np.cos((wd-270.) * np.pi / 180.) * y)

        if plot_lines:
            fig, ax = plt.subplots()
            for ii in range(n_turbs):
                ax.plot(x_rot[ii] * np.ones(2), [y_rot[ii] - D[ii] / 2, y_rot[ii] + D[ii] / 2], 'k')
            for ii in range(n_turbs):
                ax.text(x_rot[ii], y_rot[ii], 'T%03d' % ii)
            ax.axis('equal')

        srt = np.argsort(x_rot)
        x_rot_srt = x_rot[srt]
        y_rot_srt = y_rot[srt]
        for ii in range(n_turbs):
            x0 = x_rot_srt[ii]
            y0 = y_rot_srt[ii]

            def yw_upper(x):
                y = (y0 + D[ii]) + (x-x0) * wake_slope
                if isinstance(y, (float, np.float64, np.float32)):
                    if x < (x0 + 0.01):
                        y = -np.Inf
                else:
                    y[x < x0 + 0.01] = -np.Inf
                return y

            def yw_lower(x):
                y = (y0 - D[ii]) - (x-x0) * wake_slope
                if isinstance(y, (float, np.float64, np.float32)):
                    if x < (x0 + 0.01):
                        y = -np.Inf
                else:
                    y[x < x0 + 0.01] = -np.Inf
                return y

            is_in_wake = lambda xt, yt: ((yt < yw_upper(xt)) &
                                         (yt > yw_lower(xt)))

            is_freestream = (is_freestream &
                             ~is_in_wake(x_rot_srt, y_rot_srt + D/2.) &
                             ~is_in_wake(x_rot_srt, y_rot_srt - D/2.))

            if plot_lines:
                x1 = np.max(x_rot_srt) + 500.
                ax.fill_between([x0, x1, x1, x0],
                                [yw_upper(x0+0.02), yw_upper(x1),
                                 yw_lower(x1), yw_lower(x0+0.02)],
                                alpha=0.1, color='k', edgecolor=None)

        usrt = np.argsort(srt)
        is_freestream = is_freestream[usrt]
        turbs_freestream = list(np.where(is_freestream)[0])

        if len(upstream_turbs_wds) == 0:
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)
        elif not(turbs_freestream == upstream_turbs_ids[-1]):
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)

        if plot_lines:
            ax.set_title('wd = %03d' % wd)
            ax.set_xlim([np.min(x_rot)-500., x1])
            ax.set_ylim([np.min(y_rot)-500., np.max(y_rot)+500.])
            ax.plot(x_rot[turbs_freestream],
                    y_rot[turbs_freestream],
                    'o', color='green')

    # # Connect at 360 degrees
    # if upstream_turbs_ids[0] == upstream_turbs_ids[-1]:
    #     upstream_turbs_wds.pop(0)
    #     upstream_turbs_ids.pop(0)

    # Go from list to bins for upstream_turbs_wds
    upstream_turbs_wds = [[upstream_turbs_wds[i], upstream_turbs_wds[i+1]]
                          for i in range(len(upstream_turbs_wds)-1)]
    upstream_turbs_wds.append([upstream_turbs_wds[-1][-1], 360.])

    df_upstream = pd.DataFrame({'wd_min': [wd[0] for wd in upstream_turbs_wds],
                                'wd_max': [wd[1] for wd in upstream_turbs_wds],
                                'turbines': upstream_turbs_ids})

    return df_upstream


# Wrapper function to easily set new TI values
def _fi_set_ws_wd_ti(fi, wd=None, ws=None, ti=None):
    nturbs = len(fi.layout_x)

    # Convert scalar values to lists
    if not isinstance(wd, list):
        if isinstance(wd, np.ndarray):
            wd = list(wd)
        elif wd is not None:
            wd = list(np.repeat(wd, nturbs))
    if not isinstance(ws, list):
        if isinstance(ws, np.ndarray):
            ws = list(ws)
        elif ws is not None:
            ws = list(np.repeat(ws, nturbs))
    if not isinstance(ti, list):
        if isinstance(ti, np.ndarray):
            ti = list(ti)
        elif ti is not None:
            ti = list(np.repeat(ti, nturbs))

    wind_layout = (np.array(fi.layout_x), np.array(fi.layout_y))

    fi.reinitialize_flow_field(
        wind_layout=wind_layout,
        wind_direction=wd,
        wind_speed=ws,
        turbulence_intensity=ti
    )
    return fi


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np

#     import floris.tools as wfct

#     wd = 240.
#     D = 120.
#     x = np.array([0., 5., 10., 15., 0., 5., 10., 15., 0., 5., 10., 15.]) * D
#     y = np.array([0., 0., 0., 0., 5., 5., 5., 5., 10., 10., 10., 10.]) * D
#     nTurbs = len(x)

#     input_json = '/home/bdoekeme/python_scripts/flasc/examples/demo_dataset/demo_floris_input.json'
#     fi = wfct.floris_interface.FlorisInterface(input_json)

#     wd_step = 60.0
#     for wd in np.arange(0., 360., wd_step):
#         fi.reinitialize_flow_field(layout_array=(x, y), wind_direction=wd, turbulence_intensity=0.25)
#         fi.calculate_wake()
#         hor_plane = fi.get_hor_plane()
#         fig, ax = plt.subplots()
#         wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
#         fi.vis_layout(ax=ax)
#         ax.set_title('wind direction = %03d' % wd)

#     get_upstream_turbs_floris(fi, wd_step=wd_step, plot_lines=True)
#     plt.show()
