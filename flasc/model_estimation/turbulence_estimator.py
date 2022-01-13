# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import floris.tools as wfct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import floris_tools as ftools, optimization as opt


class ti_estimator():
    def __init__(self, fi):
        self.fi = fi
        self.num_turbs = len(fi.layout_x)

        self._reset_outputs()

    def _reset_outputs(self):
        self.opt_farm = None
        self.opt_turbines = None
        self.turbine_list_ordered = None
        self.turbine_pairs = None
        self.P_measured = None

    def set_measurements(self, P_measured):
        if isinstance(P_measured, int) | isinstance(P_measured, float):
            P_measured = [P_measured]
        if isinstance(P_measured, list):
            P_measured = np.array(P_measured)

        self._reset_outputs()
        self.P_measured = P_measured

    def get_turbine_order(self):
        wd = (180 - self.fi.floris.farm.wind_direction[0]) * np.pi / 180.
        rotz = np.matrix([[np.cos(wd), -np.sin(wd), 0],
                          [np.sin(wd), np.cos(wd), 0],
                          [0, 0, 1]])
        x0 = np.mean(self.fi.layout_x)
        y0 = np.mean(self.fi.layout_y)

        xyz_init = np.matrix([np.array(self.fi.layout_x) - x0,
                              np.array(self.fi.layout_y) - y0,
                              [0. for _ in range(self.num_turbs)]])

        xyz_rot = rotz * xyz_init
        x_rot = np.array(xyz_rot[0, :])[0]
        turbine_list_ordered = np.argsort(x_rot)

        self.turbine_list_ordered = turbine_list_ordered
        return turbine_list_ordered

    def get_turbine_pairs(self, wake_loss_thrs=0.20):
        fi = self.fi
        fi.calculate_wake()
        power_baseline = np.array(fi.get_turbine_power())
        disabled_turb_cp_ct = {'wind_speed': [0., 50.],
                               'power': [0., 0.],
                               'thrust': [0.0001, 0.0001]}
        regular_turb_cp_ct = fi.floris.farm.turbines[0].power_thrust_table
        df_pairs = pd.DataFrame(
            {'turbine': pd.Series([], dtype='int'),
             'affected_turbines': pd.Series([], dtype='int')})
        for ti in range(self.num_turbs):
            fi.change_turbine(
                [ti], {"power_thrust_table": disabled_turb_cp_ct})
            fi.calculate_wake()
            power_excl = np.array(fi.get_turbine_power())
            power_excl[ti] = power_baseline[ti]  # Placeholder
            wake_losses = 1 - power_baseline / power_excl
            affectedturbs = np.where(wake_losses >= wake_loss_thrs)[0]
            df_pairs = df_pairs.append(
                {'turbine': int(ti), 'affected_turbines': affectedturbs},
                ignore_index=True)
            fi.change_turbine([ti], {"power_thrust_table": regular_turb_cp_ct})

        # Save to self
        df_pairs = df_pairs.set_index('turbine', drop=True)
        self.turbine_pairs = df_pairs
        return df_pairs

    def plot_flowfield(self):
        self.fi.calculate_wake()
        fig, ax = plt.subplots()
        hor_plane = self.fi.get_hor_plane()
        wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        return fig, ax, hor_plane

    def floris_set_ws_wd_ti(self, wd=None, ws=None, ti=None):
        self.fi = ftools._fi_set_ws_wd_ti(self.fi, wd=wd, ws=ws, ti=ti)

    def _check_measurements(self):
        if self.P_measured is None:
            raise ValueError('Please specify measurements using .set_measurements(P_measured) before attempting to estimate the turbulence intensity.')

    def estimate_farmaveraged_ti(self,
                                 Ns=50,
                                 bounds=(0.01, 0.50),
                                 refine_with_fmin=False,
                                 verbose=False):

        self._check_measurements()
        out = opt.estimate_ti(fi=self.fi,
                              P_measured=self.P_measured,
                              Ns=Ns,
                              bounds=bounds,
                              turbine_upstream=range(self.num_turbs),
                              turbines_downstream=range(self.num_turbs),
                              refine_with_fmin=refine_with_fmin,
                              verbose=verbose)

        self.opt_farm = out
        ti_opt = out['x_opt']
        self.floris_set_ws_wd_ti(ti=ti_opt)
        print('Optimal farm-averaged ti: %.3f' % ti_opt)

        return ti_opt

    def estimate_local_tis(self,
                           Ns=50,
                           bounds=(0.01, 0.50),
                           refine_with_fmin=False,
                           verbose=False):

        self._check_measurements()
        turbines_sorted = self.turbine_list_ordered
        df_turbine_pairs = self.turbine_pairs

        out_array = [[] for _ in range(self.num_turbs)]
        ti_array = np.repeat(self.opt_farm['x_opt'], self.num_turbs)
        for ti in turbines_sorted:
            turbs_aff = df_turbine_pairs.loc[ti, 'affected_turbines']
            if len(turbs_aff) > 0:
                out = opt.estimate_ti(fi=self.fi,
                                      P_measured=self.P_measured[turbs_aff],
                                      Ns=Ns,
                                      bounds=bounds,
                                      turbine_upstream=ti,
                                      turbines_downstream=turbs_aff,
                                      refine_with_fmin=refine_with_fmin,
                                      verbose=verbose)
                ti_array[ti] = out['x_opt']
                self.floris_set_ws_wd_ti(ti=ti_array)
            else:
                out = {'x_opt': self.opt_farm['x_opt'],
                       'J_opt': np.nan, 'x': [], 'J': []}

            out_array[ti] = out

        self.opt_turbines = out_array
        for ti in range(self.num_turbs):
            print('Optimal ti for turbine %03d: %.3f' % (ti, ti_array[ti]))

        return out_array

    def plot_cost_function_farm(self):
        fig, ax = plt.subplots()
        ax.plot(self.opt_farm['x'], self.opt_farm['J'])
        ax.plot(self.opt_farm['x_opt'], self.opt_farm['J_opt'], 'ro')
        ax.set_ylabel('Cost function')
        ax.set_xlabel('Turbulence intensity (-)')
        ax.grid(True)
        ax.set_title('Farm-wide turbulence intensity estimation: cost function J')

    def plot_cost_functions_turbines(self):
        for ti in range(self.num_turbs):
            fig, ax = plt.subplots()
            ax.plot(self.opt_turbines[ti]['x'], self.opt_turbines[ti]['J'])
            ax.plot(self.opt_turbines[ti]['x_opt'], self.opt_turbines[ti]['J_opt'], 'ro')
            ax.set_ylabel('Cost function')
            ax.set_xlabel('Turbulence intensity (-)')
            ax.grid(True)
            ax.set_title('Turbulence intensity estimation for turbine %03d: cost function J' % ti)

    def plot_power_bars(self):
        fi = self.fi
        fi.calculate_wake()
        fig, ax = plt.subplots()
        ax.bar(x=np.array(range(self.num_turbs))-0.15,
               height=fi.get_turbine_power(), width=.3)
        ax.bar(x=np.array(range(self.num_turbs))+0.15,
               height=self.P_measured, width=.3)
        ax.set_title('Measurement and FLORIS comparison')
        ax.set_ylabel('Power')
        ax.set_xlabel('Turbine number')
        ax.legend(['FLORIS', 'SCADA'])
        return fig, ax
