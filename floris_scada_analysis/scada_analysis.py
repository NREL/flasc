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


class scada_analysis():
    """
    """

    def __init__(self,
                 ws_range_filter=None,
                 wd_range_filter=None,
                 ti_range_filter=None,
                 verbose=False):
        self.df_list = []
        self.num_turbines = []
        self.turbine_names = []

        self.ws_range = ws_range_filter
        self.wd_range = wd_range_filter
        self.ti_range = ti_range_filter

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
        self._apply_df_mask(-1)  # Last entry

    def remove_df(self, index):
        if self.verbose:
            print("Removing dataframe with name '" +
                  self.df_list[index]['name'] + "'.")
        self.df_list.pop(index)

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
                    print('  [' + str(ii) + '] '+ c + ': ' +
                          'pd.Dataframe() with shape ', var.shape)
                else:
                    print('  [' + str(ii) + '] '+ c + ': ', var)
            print(' ')

    def _apply_df_mask(self, ii):
        df = self.df_list[ii]['df'].copy()
        if self.ws_range is not None:
            df = dfm.filter_df_by_ws(df, self.ws_range)
        if self.wd_range is not None:
            df = dfm.filter_df_by_wd(df, self.wd_range)
        if self.ti_range is not None:
            if 'ti' in df.columns:
                df = dfm.filter_df_by_ti(df, self.ti_range)
        self.df_list[ii]['df_subset'] = df

    def set_masks(self, ws_range=None, wd_range=None, ti_range=None):
        if self.verbose:
            print("Applying a mask over the dataframes and adding 'df_subset")
        if ws_range is not None:
            self.ws_range = ws_range
        if wd_range is not None:
            self.wd_range = wd_range
        if ti_range is not None:
            self.ti_range = ti_range

        for ii in range(len(self.df_list)):
            self._apply_df_mask(ii)
            if self.verbose:
                print('df_subset[%d] is %d rows (df is %d rows).'
                      %(ii, self.df_list[ii]['df_subset'].shape[0],
                          self.df_list[ii]['df'].shape[0]))

    def set_turbine_names(self, turbine_names):
        if not len(turbine_names) == self.num_turbines:
            raise DataError(
                'The length of turbine_names is incorrect.'
                'Length should  be %d (specified: %d).' % 
                (self.num_turbines, len(turbine_names)))
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
            df = self.df_list[ii]['df_subset'].copy()
            era = er.energy_ratio(df=df,
                                  test_turbines=test_turbines,
                                  wd_step=wd_step,
                                  ws_step=ws_step,
                                  verbose=self.verbose)
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
