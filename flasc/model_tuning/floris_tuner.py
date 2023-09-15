# Copyright 2023 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import copy
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from floris.tools import FlorisInterface
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)
from flasc.energy_ratio import energy_ratio as er
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput
from flasc.model_tuning.tuner_utils import replicate_nan_values, set_fi_param


class FlorisTuner():
    """
    Given a FLORIS model, FlorisTuner provides a suite of functions for tuning said model,
    extracting the updated parameters and writing those parameters to a YAML file. 

    Args:
        fi (:py:obj:`FlorisInterface`): FLORIS model to be tuned. 

        df_scada (:py:obj:`pd.DataFrame`): SCADA data.



    """

    def __init__(self, 
                 fi: FlorisInterface,
                 df_scada: pd.DataFrame,
                 yaw_angles: np.array = None,
                 ):

        # Confirm the df_scada has columns 'ws', 'wd'
        if not all([col in df_scada.columns for col in ['ws', 'wd']]):
            raise ValueError('df_scada must have columns "ws" and "wd", assign these values using the functions in dataframe_manipulations.py')
        
        # Save the SCADA dataframe and get the number of turbines and number of rows
        self.df_scada = df_scada.copy()
        self.num_turbines = dfm.get_num_turbines(df_scada)
        self.num_rows = df_scada.shape[0]

        # Confirm FLORIS and df_scada have the same number of turbines
        if self.num_turbines != len(fi.layout_x):
            raise ValueError('df_scada and fi must have the same number of turbines')

        # Save the initial FLORIS object and initialize the final FLORIS object to None
        self.fi_init = fi.copy()
        self.fi_final = None # initialize for later tuning
        
        # Since running in time_series mode, the yaw angles matrix
        # should have dimensions now_rows x 1 x num_turbines
        if yaw_angles is not None:
            if yaw_angles.shape != (self.num_rows, 1, self.num_turbines):
                raise ValueError('yaw_angles must have dimensions num_rows x 1 x num_turbines')
        self.yaw_angles = yaw_angles

        # Initialize the result dict
        self.result_dict = {}

    # def copy(self):
    #     pass
        
    # Build df_floris from fi
    def _resim_floris(self, fi):

        # Get wind speeds and directions
        wind_speeds = self.df_scada['ws'].values
        wind_directions = self.df_scada['wd'].values

        # Set up the FLORIS model
        # fi = self.fi_init.copy()
        fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions, time_series=True)
        fi.calculate_wake(yaw_angles=self.yaw_angles)

        # Get the turbines in kW
        turbine_powers = fi.get_turbine_powers().squeeze()/1000

        # Generate FLORIS dataframe
        df_floris = pd.DataFrame(data=turbine_powers,
                                    columns=[f'pow_{i:>03}' for i in range(self.num_turbines)])

        # Assign the FLORIS results to a dataframe
        df_floris = df_floris.assign(ws=wind_speeds,
                                        wd=wind_directions)#,
                                        # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

        # Make sure the NaN values in the SCADA data appear in the same locations in the
        # FLORIS data
        df_floris = replicate_nan_values(self.df_scada, df_floris)

        # If df_scada includes a df_mode column copy it over to floris
        if 'df_mode' in self.df_scada.columns:
            df_floris['df_mode'] = self.df_scada['df_mode'].values

        return df_floris
    

    # Set floris parameter
    def _set_floris_param_and_resim(self,         
                    param: List[str], 
                    value: Any, 
                    param_idx: Optional[int] = None) -> FlorisInterface:
        """_summary_

        Args:
            param (List[str]): _description_
            value (Any): _description_
            param_idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            FlorisInterface: _description_
        """        
        # Get an fi model with the parameter set
        fi = set_fi_param(self.fi_init, param, value, param_idx)

        # Get the floris dataframe
        df_floris = self._resim_floris(fi)

        return df_floris

    
    def _get_energy_ratios(self, 
                          df_floris: pd.DataFrame,
                          ref_turbines: list[int] = None,
                          test_turbines: list[int] = None,
                          use_predefined_ref = False,
                          wd_step: float = 2.0,
                          wd_min = 0.0,
                          wd_max = 360.0,
                          ws_step: float = 1.0,
                          ws_min = 0.0,
                          ws_max = 50.0,
                          wd_bin_overlap_radius = 0.,
                          compute_uplift = False,
                          df_mode_order = [],
                          verbose: bool = True):
        """
        Generates SCADA and FLORIS energy ratios.

        Args:
            case (:py:obj:`str`): Case being analyzed. Either "baseline" or "controlled".
            df_floris (:py:obj:`pd.DataFrame`): FLORIS predictions to be compared with SCADA.
            wd_step (:py:obj:`float`): Wind direction discretization step size. Specifies which wind directions the energy ratio will be calculated for. Note that that not each bin has a width of this value. This variable is ignored if 'wd_bins' is specified. Defaults to 3.0.
            ws_step (:py:obj:`float`): Wind speed discretization step size. Specifies the resolution and width of the wind speed bins. This variable is ignored if 'ws_bins' is specified. Defaults to 5.0.
            wd_bin_width (:py:obj:`float`): Wind direction bin width. Should be equal or larger than 'wd_step'. Note that in the literature, it is not uncommon to specify a bin width that is larger than the step size to cover for variability in the wind direction measurements. By setting a large value for 'wd_bin_width', this provides a perspective of the larger-scale wake losses in the wind farm. When no value is provided, 'wd_bin_width' = 'wd_step'. Defaults to None.
            wd_bins (:py:obj:`np.array[float]`): Wind direction bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind direction bin. Overlap between bins is supported. This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size and bin width. Defaults to None.
            ws_bins (:py:obj:`np.array[float]`): Wind speed bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind speed bin. Overlap between bins is not currently supported.This variable overwrites the settings for 'ws_step', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size. Defaults to None.
            N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). If N = 1, no UQ will be performed. Defaults to 1.
            percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. This value is only relevant if N > 1 is specified. Defaults to [5.0, 95.0].
            balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each wind direction and wind speed bin in the collective of dataframes. The frequency of a certain bin is equal to the minimum number of occurrences among all the dataframes. This ensures an "apples to apples" comparison. Recommended to set to 'True'. Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data). Defaults to True.
            return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides more information and displays the energy ratio for every wind direction and wind speed bin, among others. This is particularly helpful in determining if the bins are well balanced. Defaults to False.
            num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. If num_blocks = -1, then do not use block bootstrapping and follow the normal approach of randomly sampling 'num_samples' with replacement. Defaults to -1.
            compute_uplift: 
            df_mode_order: 
            verbose (:py:obj:`bool`): Specify printing to console. Defaults to True.

        Returns:


        """

        # If compute_uplift is true, self.df_scada must have a column 'df_mode'
        if compute_uplift:
            if not 'df_mode' in self.df_scada.columns:
                raise ValueError("df_scada must have column 'df_mode' if computing uplift")
            if not len(df_mode_order) == 2:
                raise ValueError("if computing uplift, df_mode_order must have length 2")
            
            # Split df_scada and df_floris by df_mode
            df_scada_0 = self.df_scada[self.df_scada['df_mode'] == df_mode_order[0]].copy()
            df_scada_1 = self.df_scada[self.df_scada['df_mode'] == df_mode_order[1]].copy()

            df_floris_0 = df_floris[df_floris['df_mode'] == df_mode_order[0]].copy()
            df_floris_1 = df_floris[df_floris['df_mode'] == df_mode_order[1]].copy()

            er_in = EnergyRatioInput(
                [df_scada_0, df_scada_1, df_floris_0, df_floris_1],
                ["SCADA0","SCADA1","FLORIS0","FLORIS1"]
            )

            er_out = er.compute_energy_ratio(
                er_in,
                ref_turbines=ref_turbines,
                test_turbines=test_turbines,
                use_predefined_ref=use_predefined_ref,
                use_predefined_wd=True,
                use_predefined_ws=True,
                wd_step=wd_step,
                wd_min=wd_min,
                wd_max=wd_max,
                ws_step=ws_step,
                ws_min=ws_min,
                ws_max=ws_max,
                wd_bin_overlap_radius=wd_bin_overlap_radius,
                N=1,
                uplift_pairs=[("SCADA0","SCADA1"),("FLORIS0","FLORIS1")],
                uplift_names=["Uplift_SCADA","Uplift_FLORIS"]
            )
            
        else:
            er_in = EnergyRatioInput(
                [self.df_scada, df_floris], 
                ["SCADA", "FLORIS"]
            )

            er_out = er.compute_energy_ratio(
                er_in,
                ref_turbines=ref_turbines,
                test_turbines=test_turbines,
                use_predefined_ref=use_predefined_ref,
                use_predefined_wd=True,
                use_predefined_ws=True,
                wd_step=wd_step,
                wd_min=wd_min,
                wd_max=wd_max,
                ws_step=ws_step,
                ws_min=ws_min,
                ws_max=ws_max,
                wd_bin_overlap_radius=wd_bin_overlap_radius,
                N=1,
            )

        return er_out
    
    def evaluate_parameter_list(self,
                            param: List[str], 
                            param_values: List[Any], 
                            param_idx: Optional[int] = None,
                            ref_turbines: list[int] = None,
                            test_turbines: list[int] = None,
                            use_predefined_ref = False,
                            wd_step: float = 2.0,
                            wd_min = 0.0,
                            wd_max = 360.0,
                            ws_step: float = 1.0,
                            ws_min = 0.0,
                            ws_max = 50.0,
                            wd_bin_overlap_radius = 0.,
                            compare_uplift = False,
                            df_mode_order = [],
                            verbose: bool = True ):
        
        # Save the param name as a string
        self.param_name = param[-1]

        # Save the param also as the complete list
        self.param_full = copy.deepcopy(param)

        # Save the param index
        self.param_idx = param_idx

        # Save compare uplift and df_mode_order
        self.compare_uplift = compare_uplift
        self.df_mode_order = df_mode_order

        # Append the index
        if param_idx is not None:
            self.param_name = f'{self.param_name}[{param_idx}]'

        # Save the list of values
        self.param_values = np.array(param_values)
        num_param_values = len(param_values)

        for i, value in enumerate(param_values):

            print(f'Parameter {i+1} of {num_param_values}...')

            # Set the parameter and resim
            df_floris = self._set_floris_param_and_resim(param, value, param_idx)

            # Get the energy ratios
            er_out = self._get_energy_ratios(df_floris,
                                             ref_turbines=ref_turbines,
                                             test_turbines=test_turbines,
                                             use_predefined_ref=use_predefined_ref,
                                             wd_step=wd_step,
                                             wd_min=wd_min,
                                             wd_max=wd_max,
                                             ws_step=ws_step,
                                             ws_min=ws_min,
                                             ws_max=ws_max,
                                             wd_bin_overlap_radius=wd_bin_overlap_radius,
                                             compute_uplift=compare_uplift,
                                             df_mode_order=df_mode_order,
                                             verbose=verbose)

            # Save the result
            self.result_dict[value] = er_out

    def _compute_er_error(self, 
                          er_out,
                          max_floris_value = 0.95):

        # Remove rows where the FLORIS value is above max_floris_value
        df_ = (er_out.df_result
               .copy()
               [er_out.df_result['FLORIS'] <=  max_floris_value]
        )
        
        # Grab the columns
        scada_vals = df_['SCADA'].values
        floris_vals = df_['FLORIS'].values
        count_vals = df_['count_SCADA'].values


        # Return the mean squared error
        return mean_squared_error(y_true=scada_vals, 
                                 y_pred=floris_vals, 
                                 sample_weight=count_vals)

    def _compute_uplift_error(self, 
                          er_out,
                          min_abs_floris_value = 1.0):

        # Remove rows where the FLORIS is near 0
        df_ = (er_out.df_result
               .copy()
               [er_out.df_result['Uplift_FLORIS'].abs() >=  min_abs_floris_value]
        )
       
        # Grab the columns
        scada_vals = df_['Uplift_SCADA'].values
        floris_vals = df_['Uplift_FLORIS'].values
        count_vals = df_['count_SCADA0'].values


        # Return the mean squared error
        return mean_squared_error(y_true=scada_vals, 
                                 y_pred=floris_vals, 
                                 sample_weight=count_vals)

    def calculate_param_errors(self,
                               compare_uplift = False):

        self.error_values = np.zeros_like(self.param_values)

        for idx, param in enumerate(self.param_values):

            er_out = self.result_dict[param]

            if not compare_uplift:
                self.error_values[idx] = self._compute_er_error(er_out)
            else:
                self.error_values[idx] = self._compute_uplift_error(er_out)

        self.best_param = self.param_values[np.argmin(self.error_values)]
        self.best_error = np.min(self.error_values)

    def apply_best_param(self):

        return set_fi_param(self.fi_init, self.param_full, self.best_param, self.param_idx)

    
    def plot_error(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.param_values, self.error_values, 'k.-')
        ax.scatter(self.best_param, self.best_error,color='r',marker='o')


        ax.set_xlabel(self.param_name)
        ax.set_ylabel("Error")
        ax.grid(True)

    def plot_energy_ratios(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        # Plot all the FLORIS results
        for idx, param in enumerate(self.param_values):

            df_ = self.result_dict[param].df_result

            # If this is the best result set aside to plot last
            if param == self.best_param:
                df_best = df_.copy(deep=True)
                continue

            if param < self.best_param:
                # Color the less than as blue
                ax.plot(df_['wd_bin'], df_['FLORIS'], 'b-',alpha=0.3)
            else:
                # Color the greater than as red
                ax.plot(df_['wd_bin'], df_['FLORIS'], 'r-',alpha=0.3)

        # Now plot SCADA
        ax.plot(df_best['wd_bin'], df_best['SCADA'], 'k.-',label='SCADA')

        # Plot best FLORIS results
        ax.plot(df_best['wd_bin'], df_best['FLORIS'], 'g.-',label=f'FLORIS ({self.best_param})')
            
        ax.grid(True)
        ax.legend()

    def plot_energy_ratio_uplifts(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        # Plot all the FLORIS results
        for idx, param in enumerate(self.param_values):

            df_ = self.result_dict[param].df_result

            # If this is the best result set aside to plot last
            if param == self.best_param:
                df_best = df_.copy(deep=True)
                continue

            if param < self.best_param:
                # Color the less than as blue
                ax.plot(df_['wd_bin'], df_['Uplift_FLORIS'], 'b-',alpha=0.3)
            else:
                # Color the greater than as red
                ax.plot(df_['wd_bin'], df_['Uplift_FLORIS'], 'r-',alpha=0.3)

        # Now plot SCADA
        ax.plot(df_best['wd_bin'], df_best['Uplift_SCADA'], 'k.-',label='SCADA')

        # Plot best FLORIS results
        ax.plot(df_best['wd_bin'], df_best['Uplift_FLORIS'], 'g.-',label=f'FLORIS ({self.best_param})')
            
        ax.grid(True)
        ax.legend()




    # def write_yaml(self, filepath: str):
    #     """
    #     Write tuned FLORIS parameters to a YAML file.

    #     Args:
    #             filepath (:py:obj:`str`): Path that YAML file will be written to. 
         
    #     """

    #     # Check if file already exists
    #     if os.path.isfile(filepath):
    #          print(f'FLORIS YAML file {filepath} exists. Skipping...')

    #     # If it does not exist, write a new YAML file for tuned FLORIS parameters
    #     else:         
    #          fi_dict = self.fi_tuned.floris.as_dict()

    #          # Save the file path for future reference
    #          self.yaml = filepath
             
    #          print(f'Writing new FLORIS YAML file to `{filepath}`...')
             
    #          # Wrtie the YAML file
    #          with open(filepath, 'w') as f:
    #             yaml.dump(fi_dict, f)
                
    #          print('Finished writing FLORIS YAML file.')

    # def get_untuned_floris(self) -> FlorisInterface:
    #     """
    #     Return untuned FLORIS model.

    #     Returns:
    #             fi_untuned (:py:obj:`FlorisInterface`): Untuned FLORIS model.
        
    #     """

    #     return self.fi_untuned

    # def get_tuned_floris(self) -> FlorisInterface:
    #     """
    #     Return tuned FLORIS model.

    #     Returns:
    #             fi_tuned (:py:obj:`FlorisInterface`): Tuned FLORIS model.

    #     """

    #     return self.fi_tuned

    # def get_yaml(self) -> str:
    #     """
    #     Return directory of the YAML file containing the tuned FLORIS parameters.

    #     Returns:
    #             yaml (:py:obj:`str`): Directory of YAML file.

    #     """

    #     return self.yaml
    
    