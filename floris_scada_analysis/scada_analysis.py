# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.base import DataError

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import energy_ratio as er
from floris_scada_analysis import time_operations as fsato


class scada_analysis():
    """
    """

    def __init__(self, verbose=False):
        self.df_list = []
        self.num_turbines = []
        self.turbine_names = []

        # Placeholders for masks
        self.wd_range = None
        self.wd_range_ids = []
        self.ws_range = None
        self.ws_range_ids = []
        self.ti_range = None
        self.ti_range_ids = []
        self.time_range = None
        self.time_range_ids = []
        self.verbose = verbose

        if verbose:
            print('Initialized energy_ratio_analysis() object.')

    def add_df(self, df, name):
        if not ('wd' in df.columns and 'ws' in df.columns):
            raise ImportError("Could not find all columns. Ensure that" +
                              "you have columns 'wd' and 'ws' in your df.")

        num_turbines = dfm.get_num_turbines(df)
        if len(self.df_list) < 1:
            self.num_turbines = num_turbines
            self.turbine_names = [str(i) for i in range(num_turbines)]

        if self.num_turbines != num_turbines:
            print('Added dataframe seems to have a different number of ' +
                  'turbines than the existing dataframe(s). Skipping ' +
                  'addition.')
            return None

        if 'category' in df:
            if len(np.unique(df['category'])) > 2:
                raise KeyError('More than 2 different category values ' +
                               'are found. Please limit yourself to 2.')
        else:
            print("No 'category' column found in df. Adding column with " +
                  "all values 'baseline'.")
            df = df.copy()
            df.loc[:, 'category'] = 'baseline'

        categories = np.unique(df['category'])
        new_entry = dict({'df': df, 'name': name, 'categories': categories})
        self.df_list.append(new_entry)

        default_ids = np.array([True for _ in range(df.shape[0])])
        self.wd_range_ids.append(default_ids)
        self.ws_range_ids.append(default_ids)
        self.ti_range_ids.append(default_ids)
        self.time_range_ids.append(default_ids)

        # Force update mask for new dataframe
        idx = len(self.df_list) - 1
        wd_range_tmp = self.wd_range
        ws_range_tmp = self.ws_range
        ti_range_tmp = self.ti_range
        time_range_tmp = self.time_range
        self.wd_range = None
        self.ws_range = None
        self.ti_range = None
        self.time_range = None
        self.set_masks(ws_range=ws_range_tmp,
                       wd_range=wd_range_tmp,
                       ti_range=ti_range_tmp,
                       time_range=time_range_tmp,
                       df_ids=[idx])

    def remove_df(self, index):
        if self.verbose:
            print("Removing dataframe with name '" +
                  self.df_list[index]['name'] + "'.")

        # Remove dataframe
        self.df_list.pop(index)

        # Remove mask indices for this dataframe
        self.wd_range_ids.pop(index)
        self.ws_range_ids.pop(index)
        self.ti_range_ids.pop(index)
        self.time_range_ids.pop(index)

        if len(self.df_list) < 1:
            # Reset variables
            self.num_turbines = []

    def print_dfs(self):
        for ii in range(len(self.df_list)):
            print('___ DATAFRAME %d ___' % ii)
            keys = [c for c in self.df_list[ii].keys()]
            for c in keys:
                var = self.df_list[ii][c]
                if isinstance(var, pd.DataFrame):
                    print(
                        '  [' + str(ii) + '] ' + c + ': ' +
                        'pd.Dataframe() with shape ', var.shape
                        )
                else:
                    print('  [' + str(ii) + '] ' + c + ': ', var)
            print(' ')

    def set_masks(self, ws_range=None, wd_range=None, ti_range=None,
                  time_range=None, df_ids=None):
        if self.verbose:
            print("Extracting a mask over the df: 'df_subset'.")

        if df_ids is None:
            df_ids = range(len(self.df_list))
        elif isinstance(df_ids, (int, np.integer, float)):
            df_ids = [int(df_ids)]

        if (ws_range is not None) and not (ws_range == self.ws_range):
            self.ws_range = ws_range
            for ii in df_ids:
                df = self.df_list[ii]['df']
                ids = (df['ws'] > ws_range[0]) & (df['ws'] <= ws_range[1])
                self.ws_range_ids[ii] = np.array(ids)

        if (wd_range is not None) and not (wd_range == self.wd_range):
            self.wd_range = wd_range
            for ii in df_ids:
                df = self.df_list[ii]['df']
                ids = (df['wd'] > wd_range[0]) & (df['wd'] <= wd_range[1])
                self.wd_range_ids[ii] = np.array(ids)

        if (ti_range is not None) and not (ti_range == self.ti_range):
            self.ti_range = ti_range
            for ii in df_ids:
                df = self.df_list[ii]['df']
                ids = (df['ti'] > ti_range[0]) & (df['ti'] <= ti_range[1])
                self.ti_range_ids[ii] = np.array(ids)

        if (time_range is not None) and not (time_range == self.time_range):
            self.time_range = time_range
            for ii in df_ids:
                df = self.df_list[ii]['df']
                ids = np.array([False for _ in range(df.shape[0])])
                indices_out = fsato.find_window_in_time_array(
                    df['time'], seek_time_windows=[list(time_range)])
                ids[indices_out[0]] = True
                self.time_range_ids[ii] = ids

        # Update masked dataframe(s)
        for ii in df_ids:
            mask = (
                (self.wd_range_ids[ii]) &
                (self.ws_range_ids[ii]) &
                (self.ti_range_ids[ii]) &
                (self.time_range_ids[ii])
            )
            df_full = self.df_list[ii]['df']
            self.df_list[ii]['df_subset'] = df_full[mask]

    def set_turbine_names(self, turbine_names):
        if not len(turbine_names) == self.num_turbines:
            raise DataError(
                'The length of turbine_names is incorrect.'
                'Length should  be %d (specified: %d).'
                % (self.num_turbines, len(turbine_names)))
        self.turbine_names = turbine_names

    def clear_energy_ratio_results(self, ii):
        self.self.df_list[ii].pop('er_results')
        self.self.df_list[ii].pop('er_test_turbines')
        self.self.df_list[ii].pop('er_ref_turbines')
        self.self.df_list[ii].pop('er_dep_turbines')
        self.self.df_list[ii].pop('er_wd_step')
        self.self.df_list[ii].pop('er_ws_step')
        self.self.df_list[ii].pop('er_bootstrap_N')

    def clear_all_energy_ratio_results(self):
        for ii in range(len(self.df_list)):
            self.clear_energy_ratio_results(ii)

    def get_energy_ratios(self, test_turbines, wd_step, ws_step,
                          N=1, verbose=False):

        for ii in range(len(self.df_list)):
            df_subset = self.df_list[ii]['df_subset']
            era = er.energy_ratio(df_in=df_subset,
                                  test_turbines=test_turbines,
                                  wd_step=wd_step,
                                  ws_step=ws_step,
                                  verbose=verbose)
            er_result = era.get_energy_ratio(N=N)

            self.df_list[ii]['er_results'] = er_result
            self.df_list[ii]['er_test_turbines'] = test_turbines
            self.df_list[ii]['er_wd_step'] = wd_step
            self.df_list[ii]['er_ws_step'] = ws_step
            self.df_list[ii]['er_bootstrap_N'] = N

    def plot_energy_ratios(self, superimpose=True):
        if superimpose:
            _, ax = plt.subplots()

        for ii in range(len(self.df_list)):
            if not superimpose:
                _, ax = plt.subplots()

            result = self.df_list[ii]['er_results']
            data_name = self.df_list[ii]['name']
            categories = self.df_list[0]['categories']
            test_turbine_names = [self.turbine_names[i] for i in
                                  self.df_list[ii]['er_test_turbines']]

            ax.plot(result.wd_bin, result.baseline,
                    label=data_name+': '+categories[0])
            ax.fill_between(result.wd_bin, result.baseline_l,
                            result.baseline_u, alpha=0.15)
            if 'controlled' in result.columns:
                ax.plot(result.wd_bin, result.controlled,
                        label=data_name+': '+categories[1])
                ax.fill_between(result.wd_bin, result.controlled_l,
                                result.controlled_u, alpha=0.15)
            ax.legend()
            plt.title(str(['Test turbines:', test_turbine_names]))
            plt.xlabel('Wind direction (degrees)')
            plt.ylabel('Energy ratio (-)')
            plt.grid(True)
