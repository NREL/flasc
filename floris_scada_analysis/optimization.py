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
import numpy as np
import scipy.optimize as opt
import scipy.stats as spst

from floris.utilities import wrap_360

from floris_scada_analysis import floris_tools as ftools


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
