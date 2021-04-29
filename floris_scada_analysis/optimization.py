# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import copy
from datetime import timedelta as td
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
import scipy.stats as spst

from floris.utilities import wrap_360

from floris_scada_analysis import circular_statistics as css
from floris_scada_analysis import floris_tools as ftools
from floris_scada_analysis import time_operations as fsato


def interp_within_margin(x, xp, yp, x_margin, kind, wrap_around_360=False):
    # Make sure everything is sorted
    x = np.sort(x)
    id_sort = np.argsort(xp)
    xp = np.array(xp)[id_sort]
    yp = np.array(yp)[id_sort]

    # Use regular interpolation for all values first
    f_reg = interp.interp1d(xp, yp, kind=kind,
                            bounds_error=False,
                            assume_sorted=True)
    y_full = f_reg(x)

    if wrap_around_360 and not (kind == 'nearest'):
        # Shift by 180 deg and interpolate again
        f_180 = interp.interp1d(xp, wrap_360(yp+180.),
                                kind=kind, bounds_error=False,
                                assume_sorted=True)
        y_180 = wrap_360(f_180(x) - 180.)

        # Figure out where we jump by more than 180 deg
        Nx = len(x)
        dymax = np.full(Nx, np.nan)
        for i in range(Nx):
            lrgv = xp[(xp - x[i]) > 0]
            lwrv = xp[(xp - x[i]) < 0]
            if len(lrgv) > 0:
                dy_u = np.abs(yp[int(np.min(lrgv))] - y_full[i])
                dymax[i] = np.nanmax([dy_u, dymax[i]])
            if len(lwrv) > 0:
                dy_l = np.abs(yp[int(np.max(lwrv))] - y_full[i])
                dymax[i] = np.nanmax([dy_l, dymax[i]])

        # Replace those points with y_180 values
        y_full[dymax > 180.] = y_180[dymax > 180.]

    # Set any values outside of x_margin to np.nan
    dx = np.array([np.min(np.abs(xi-np.array(xp))) for xi in x])
    y_full[dx > x_margin] = np.nan

    return y_full


def find_timeshift_between_dfs(df1, df2, cols_df1, cols_df2,
                               use_circular_statistics=False,
                               t_step=td(days=30), opt_bounds=None,
                               opt_Ns=None, verbose=True):

    # Get min and max time for dataframes
    min_time = np.max([np.min(df1.time), np.min(df2.time)])
    max_time = np.min([np.max(df1.time), np.max(df2.time)])

    # Convert to arrays of a single mean quantity
    print('Determining one column-average per dataframe to sync.')
    if use_circular_statistics:
        df1['y1'] = css.calc_wd_mean_radial_list(df1[cols_df1])
        df2['y2'] = css.calc_wd_mean_radial_list(df2[cols_df2])
    else:
        df1['y1'] = np.nanmean(df1[cols_df1], axis=1)
        df2['y2'] = np.nanmean(df2[cols_df2], axis=1)

    # Cut down df1 and df2 to a minimal dataframe
    df1 = df1[['time', 'y1']]
    df2 = df2[['time', 'y2']]

    # Create a shared time array and sync both dfs on it
    dt = np.min([fsato.estimate_dt(df1['time']),
                 fsato.estimate_dt(df2['time'])])
    ddx = dt / td(seconds=1)
    print('Resampling dataframes to a common time vector. ' +
          'This may take a while...')
    time_array = [min_time + i * dt for i in
                  range(int(np.ceil((max_time-min_time)/dt)))]
    df1 = fsato.df_resample_to_time_array(
        df1, time_array=time_array,
        circular_cols=use_circular_statistics,
        interp_method='linear', interp_margin=dt
        )
    df2 = fsato.df_resample_to_time_array(
        df2, time_array=time_array,
        circular_cols=use_circular_statistics,
        interp_method='linear', interp_margin=dt
        )

    # Look at comparison per t_step
    current_time = min_time
    output_list = []
    print('Estimating required timeshift for df1.')
    while current_time < max_time:
        t0 = current_time
        t1 = current_time + t_step
        id_sub = (df1.time >= t0) & (df1.time < t1)
        df1_sub = df1[id_sub]
        df2_sub = df2[id_sub]

        # Create x, y1 and y2 vectors
        x = np.array(df1_sub['time'])
        x = np.array((x-x[0])/np.timedelta64(1, 's'))
        y1 = np.array(df1_sub['y1'])
        y2 = np.array(df2_sub['y2'])

        if verbose:
            print('   Calculating timeshift for t0: %s, t1: %s'
                  % (str(t0), str(t1)))

        def cost_fun(x_shift):
            # Shift data along x-axis and then fit along y-axis
            y1_cor = interp_within_margin(x, x - x_shift, y1, ddx, 'linear')
            y_bias, J = match_wd_curves_by_offset(y1_cor, y2, dwd=1.0)
            y1_cor = wrap_360(y1_cor - y_bias)

            # Remove NaNs
            ids = (~np.isnan(y1_cor)) & (~np.isnan(y2))
            y2_cor = y2[ids]
            y1_cor = y1_cor[ids]

            # Calculate score
            if ((len(y1_cor) < 10) | (len(y2_cor) < 10)):
                cost = np.nan
            else:
                cost = -1. * spst.pearsonr(y1_cor, y2_cor)[0]
            return cost

        # Optimize using scipy.brute()
        if opt_bounds is None:
            opt_bounds = [(-td(hours=10), td(hours=10))]
        elif isinstance(opt_bounds[0], (tuple, list)):
            opt_bounds = opt_bounds  # Fine
        else:
            opt_bounds = [opt_bounds]

        if isinstance(opt_bounds[0][0], np.timedelta64):
            opt_bounds[0] = (opt_bounds[0][0] / np.timedelta64(1, 's'),
                             opt_bounds[0][1] / np.timedelta64(1, 's'))
        if isinstance(opt_bounds[0][0], td):
            opt_bounds[0] = (opt_bounds[0][0] / td(seconds=1),
                             opt_bounds[0][1] / td(seconds=1))

        if opt_Ns is None:
            opt_Ns = int((opt_bounds[0][1]-opt_bounds[0][0]) / 600.)

        finish = opt.fmin  # Can also be None
        verbose = True
        x_opt, J_opt, x_all, J_all = opt.brute(
            cost_fun,
            ranges=opt_bounds,
            Ns=opt_Ns,
            finish=finish,
            disp=verbose,
            full_output=True)
        print('     Optimal time shift for df_1: %d s (%.2f hours).'
              % (x_opt[0], x_opt[0]/3600.))

        output_list.append({
            't0': t0,
            't1': t1,
            'x_opt': td(seconds=x_opt[0]),
            'J_opt': J_opt,
            'x': x_all,
            'J': J_all
        })

        current_time = current_time + t_step

    return output_list


def find_bias_x(x_1, y_1, x_2, y_2, search_range, search_dx):
    x_1 = np.array(x_1)
    y_2 = np.array(y_2)

    def errfunc(dx, x_1, x_2, y_1, y_2):
        y_1_cor = np.interp(x_2, x_1 - dx, y_1)

        # Clean up data
        clean_data = (~np.isnan(y_1_cor)) & (~np.isnan(y_2))
        y_1_cor = y_1_cor[clean_data]
        y_2 = y_2[clean_data]

        if all(np.isnan(y_1_cor)) and all(np.isnan(y_2)):
            cost = np.nan
        else:
            cost = -1.0 * spst.pearsonr(y_1_cor, y_2)[0]
        return cost

    cost_min = 1.0e15
    success = False
    p1 = [0.0]
    for dx_opt in np.arange(search_range[0], search_range[1], search_dx):
        cost_eval = errfunc(dx_opt, x_1, x_2, y_1, y_2)
        if cost_eval <= cost_min:
            p1 = [dx_opt]
            cost_min = cost_eval
            success = True
    dx_opt = p1[0]

    return dx_opt, success


def match_wd_curves_by_offset(wd_ref, wd_turb, dwd=2.0):
    J_opt = np.nan
    dwd_opt = np.nan
    for dx in np.arange(0., 360., dwd):
        wd_turb_cor = wrap_360(wd_turb - dx)
        wd_error = np.abs(wd_ref-wd_turb_cor)
        wd_error[wd_error > 180.] = 360. - wd_error[wd_error > 180.]
        J = np.nanmean(wd_error**2.0)
        if np.isnan(J_opt):
            if not np.isnan(J):
                J_opt = J
                dwd_opt = dx
        elif J < J_opt:
            J_opt = J
            dwd_opt = dx
    return dwd_opt, J_opt


def find_wd_bias_by_energy_ratios(er_wd_list, er_scada_list,
                                  er_floris_list, search_range,
                                  search_dx):

    def errfunc(dx, er_wd, er_scada, er_floris):
        x_total = []
        y_scada = []
        y_floris = []
        for ii in range(len(er_wd)):
            x = np.array(er_wd_list[ii])
            x_cor = wrap_360(x - dx)  # Removing bias from wd measurement
            y_cor = np.array(er_scada[ii])  # And corresponding en. ratios
            y_cor = y_cor[np.argsort(x_cor)]  # Sort ascending
            x_cor = np.sort(x_cor)  # Sort ascending
            y_scada_interp = np.interp(x, x_cor, y_cor,
                                       left=np.nan, right=np.nan)

            x_total.extend(ii*400. + x)
            y_floris.extend(er_floris[ii])
            y_scada.extend(y_scada_interp)

        # Make sure SCADA data is ascending
        x_total = np.array(x_total)
        y_floris = np.array(y_floris)
        y_scada = np.array(y_scada)

        # Clean up data
        clean_data = (~np.isnan(y_floris)) & (~np.isnan(y_scada))
        x_total = x_total[clean_data]
        y_scada = y_scada[clean_data]
        y_floris = y_floris[clean_data]

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for ii in range(int(np.ceil(np.max(x_total)/400))):
        #     ids = (x_total >= (ii*400)) & (x_total < (ii+1)*400)
        #     ax.plot(x_total[ids], y_floris[ids], '--', color='black')
        #     ax.plot(x_total[ids], y_scada[ids], '--', color='red')
        # plt.show()

        if all(np.isnan(y_scada)) and all(np.isnan(y_floris)):
            cost = np.nan
        else:
            cost = -1.0 * spst.pearsonr(y_scada, y_floris)[0]
        return cost

    cost_min = 1.0e15
    success = False
    dx_opt = 0.0
    for dx in np.arange(search_range[0], search_range[1], search_dx):
        cost_eval = errfunc(dx=dx,
                            er_wd=er_wd_list,
                            er_scada=er_scada_list,
                            er_floris=er_floris_list)
        if cost_eval <= cost_min:
            dx_opt = dx
            cost_min = cost_eval
            success = True

    wd_bias = dx_opt
    return wd_bias, success


def estimate_ti(fi, P_measured, Ns, bounds, turbine_upstream,
                turbines_downstream, refine_with_fmin=False,
                verbose=False):
    # Make copy so that existing object is not changed
    fi = copy.deepcopy(fi)
    num_turbines = len(fi.layout_x)
    ti_0 = np.mean(fi.floris.farm.turbulence_intensity)

    # Define a cost function
    def cost_fun(ti):
        ti_array = np.repeat(ti_0, num_turbines)
        ti_array[turbine_upstream] = ti
        ftools._fi_set_ws_wd_ti(fi, ti=ti_array)
        fi.calculate_wake()
        Pturbs = np.array(fi.get_turbine_power())
        Pturbs = Pturbs[turbines_downstream]
        se = (P_measured-Pturbs)**2.0
        mse = np.mean(se)
        return mse

    if refine_with_fmin:
        finish = opt.fmin
    else:
        finish = None

    # Ensure appropriate format
    if not (isinstance(bounds[0], tuple) | isinstance(bounds[0], list)):
        bounds = [bounds]

    # Optimize using grid search approach
    x_opt, J_opt, x, J = opt.brute(cost_fun,
                                   ranges=bounds,
                                   Ns=Ns,
                                   finish=finish,
                                   disp=verbose,
                                   full_output=True)

    opt_result = {'x_opt': x_opt, 'J_opt': J_opt,  'x': x, 'J': J}
    return opt_result
