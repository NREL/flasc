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
# import copy
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
from flasc.model_tuning.error_result import ErrorResult
from flasc.model_tuning.tuner_utils import set_fi_param


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
                 yaw_angles: np.array = None,):

        # Confirm the df_scada has columns 'ws', 'wd'
        if not all([col in df_scada.columns for col in ['ws', 'wd']]):
            raise ValueError('df_scada must have columns "ws" and "wd", assign these values using the functions in dataframe_manipulations.py')
        
        # Save the SCADA dataframe and get the number of turbines and number of rows
        self.df_scada = df_scada.copy()
        self.num_turbines = dfm.get_num_turbines(df_scada)
        self.num_rows = df_scada.shape[0]

        # # Save the wind speeds and directions
        # self.wind_speeds = df_scada['ws'].values
        # self.wind_directions = df_scada['wd'].values

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

    def copy(self):
        pass

        
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

        df_floris = df_floris.assign(ws=wind_speeds,
                                        wd=wind_directions)#,
                                        # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

        return df_floris
    

    # Set floris parameter
    def _set_floris_param_and_resim(self,         
                    param: List[str], 
                    value: Any, 
                    idx: Optional[int] = None) -> FlorisInterface:
        """_summary_

        Args:
            param (List[str]): _description_
            value (Any): _description_
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            FlorisInterface: _description_
        """        
        # Get an fi model with the parameter set
        fi = set_fi_param(self.fi_init, param, value, idx)

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

            verbose (:py:obj:`bool`): Specify printing to console. Defaults to True.

        Returns:


        """

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
    
    def evaluate_parameters(self,
                            param: List[str], 
                            param_values: List[Any], 
                            idx: Optional[int] = None,
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
                            verbose: bool = True ):
        
        # Save the list of values
        self.param_values = param_values
        num_param_values = len(param_values)

        for i, value in enumerate(param_values):

            print(f'Parameter {i+1} of {num_param_values}...')

            # Set the parameter and resim
            df_floris = self._set_floris_param_and_resim(param, value, idx)

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
                                             verbose=verbose)

            # Save the result
            self.result_dict[value] = er_out



            
    # def evaluate_error(self, 
    #                    case: str,
    #                    df_floris: pd.DataFrame, 
    #                    wd_step: float = 3.0,
    #                    ws_step: float = 5.0,
    #                    wd_bin_width: float = None,
    #                    wd_bins: npt.NDArray[np.float64] = None,
    #                    ws_bins: npt.NDArray[np.float64] = None,
    #                    N: int = 1,
    #                    percentiles: list[float] = [5.0, 95.0],
    #                    balance_bins_between_dfs: bool = True,
    #                    return_detailed_output: bool = False,
    #                    num_blocks: int = -1,
    #                    verbose: bool = True) -> float:
    #     """
    #     Compares SCADA and FLORIS energy ratios and evaluates their error (mean squared error).

    #     Args:
    #         case (:py:obj:`str`): Case being analyzed. Either "baseline" or "controlled".

    #         df_floris (:py:obj:`pd.DataFrame`): FLORIS predictions to be compared with SCADA.

    #         wd_step (:py:obj:`float`): Wind direction discretization step size. Specifies which wind directions the energy ratio will be calculated for. Note that that not each bin has a width of this value. This variable is ignored if 'wd_bins' is specified. Defaults to 3.0.

    #         ws_step (:py:obj:`float`): Wind speed discretization step size. Specifies the resolution and width of the wind speed bins. This variable is ignored if 'ws_bins' is specified. Defaults to 5.0.

    #         wd_bin_width (:py:obj:`float`): Wind direction bin width. Should be equal or larger than 'wd_step'. Note that in the literature, it is not uncommon to specify a bin width that is larger than the step size to cover for variability in the wind direction measurements. By setting a large value for 'wd_bin_width', this provides a perspective of the larger-scale wake losses in the wind farm. When no value is provided, 'wd_bin_width' = 'wd_step'. Defaults to None.

    #         wd_bins (:py:obj:`np.array[float]`): Wind direction bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind direction bin. Overlap between bins is supported. This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size and bin width. Defaults to None.

    #         ws_bins (:py:obj:`np.array[float]`): Wind speed bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind speed bin. Overlap between bins is not currently supported. This variable overwrites the settings for 'ws_step', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size. Defaults to None.
            
    #         N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). If N = 1, no UQ will be performed. Defaults to 1.

    #         percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. This value is only relevant if N > 1 is specified. Defaults to [5.0, 95.0].

    #         balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each wind direction and wind speed bin in the collective of dataframes. The frequency of a certain bin is equal to the minimum number of occurrences among all the dataframes. This ensures an "apples to apples" comparison. Recommended to set to 'True'. Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data). Defaults to True.

    #         return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides more information and displays the energy ratio for every wind direction and wind speed bin, among others. This is particularly helpful in determining if the bins are well balanced. Defaults to False.

    #         num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. If num_blocks = -1, then do not use block bootstrapping and follow the normal approach of randomly sampling 'num_samples' with replacement. Defaults to -1.

    #         verbose (:py:obj:`bool`): Specify printing to console. Defaults to True.

    #     Returns:
    #         err (:py:obj:`float`): Mean squared error between SCADA and FLORIS energy ratios. 

    #     """

    #     # Generate energy ratios for SCADA and FLORIS
    #     energy_ratios, _ = self.get_energy_ratios(case=case,
    #                                            df_floris=df_floris,
    #                                            wd_step=wd_step,
    #                                            ws_step=ws_step,
    #                                            wd_bin_width=wd_bin_width,
    #                                            wd_bins=wd_bins,
    #                                            ws_bins=ws_bins,
    #                                            N=N,
    #                                            percentiles=percentiles,
    #                                            balance_bins_between_dfs=balance_bins_between_dfs,
    #                                            return_detailed_output=return_detailed_output,
    #                                            num_blocks=num_blocks,
    #                                            verbose=verbose)
        

    #     # Take mean squared error between SCADA and FLORIS energy ratios
    #     # TODO: The 'baseline' key for energy ratios is confusing for the 'controlled' case and needs to be updated
    #     scada_energy_ratios = energy_ratios[0]['er_results']['baseline']   
    #     floris_energy_ratios = energy_ratios[1]['er_results']['baseline']

    #     datapoints_per_wd_bin = energy_ratios[0]['er_results']['bin_count']

    #     err = mean_squared_error(y_true=scada_energy_ratios, 
    #                              y_pred=floris_energy_ratios, 
    #                              sample_weight=datapoints_per_wd_bin)

    #     return err

    # def tune_floris(self, 
    #                 case: str,
    #                 pow_ref_columns: list[int],
    #                 param_name: str,
    #                 param_values: npt.NDArray[np.float64],
    #                 time_series: bool = True,
    #                 wd_step: float = 3.0,
    #                 ws_step: float = 5.0,
    #                 wd_bin_width: float = None,
    #                 wd_bins: npt.NDArray[np.float64] = None,
    #                 ws_bins: npt.NDArray[np.float64] = None,
    #                 N: int = 1,
    #                 percentiles: list[float] = [5.0, 95.0],
    #                 balance_bins_between_dfs: bool = True,
    #                 return_detailed_output: bool = False,
    #                 num_blocks: int = -1,
    #                 verbose: bool = True,
    #                 plot_err: bool = False,
    #                 plot_energy_ratios: bool = False) -> (FlorisInterface, float, float, interp1d, list[float]):
    #     """
    #     Tune FLORIS model. 

    #     Args:
    #         case (:py:obj:`str`): Case being analyzed. Either "baseline" or "controlled".

    #         pow_ref_columns (:py:obj:`list[int]`): Reference power columns that should be used for generating FLORIS predictions. 

    #         param_name (:py:obj:`str`): Name of parameter being tuned.

    #         param_values (:py:obj:`np.array[float]`): Range of parameter values being evaluated for optimality. Optimality is defined as the value that yields the minimum error between SCADA and FLORIS energy ratios.      
            
    #         time_series (:py:obj:`bool`): Specify if time series should be used for generating FLORIS predictions. Defaults to True.

    #         wd_step (:py:obj:`float`): Wind direction discretization step size. Specifies which wind directions the energy ratio will be calculated for. Note that that not each bin has a width of this value. This variable is ignored if 'wd_bins' is specified. Defaults to 3.0.

    #         ws_step (:py:obj:`float`): Wind speed discretization step size. Specifies the resolution and width of the wind speed bins. This variable is ignored if 'ws_bins' is specified. Defaults to 5.0.

    #         wd_bin_width (:py:obj:`float`): Wind direction bin width. Should be equal or larger than 'wd_step'. Note that in the literature, it is not uncommon to specify a bin width that is larger than the step size to cover for variability in the wind direction measurements. By setting a large value for 'wd_bin_width', this provides a perspective of the larger-scale wake losses in the wind farm. When no value is provided, 'wd_bin_width' = 'wd_step'. Defaults to None.

    #         wd_bins (:py:obj:`np.array[float]`): Wind direction bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind direction bin. Overlap between bins is supported. This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size and bin width. Defaults to None.

    #         ws_bins (:py:obj:`np.array[float]`): Wind speed bins over which the energy ratio will be calculated. Each entry of this array must contain exactly two float values, the lower and upper bound for that wind speed bin. Overlap between bins is not currently supported. This variable overwrites the settings for 'ws_step', and instead allows the user to directly specify the binning properties, rather than deriving them from the data and an assigned step size. Defaults to None.
            
    #         N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). If N = 1, no UQ will be performed. Defaults to 1.

    #         percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. This value is only relevant if N > 1 is specified. Defaults to [5.0, 95.0].

    #         balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each wind direction and wind speed bin in the collective of dataframes. The frequency of a certain bin is equal to the minimum number of occurrences among all the dataframes. This ensures an "apples to apples" comparison. Recommended to set to 'True'. Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data). Defaults to True.

    #         return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides more information and displays the energy ratio for every wind direction and wind speed bin, among others. This is particularly helpful in determining if the bins are well balanced. Defaults to False.

    #         num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. If num_blocks = -1, then do not use block bootstrapping and follow the normal approach of randomly sampling 'num_samples' with replacement. Defaults to -1.

    #         verbose (:py:obj:`bool`): Specify printing to console. Defaults to True.

    #         plot_err (:py:obj:`bool`): Specify displaying a visualization of the mean squared error between the SCADA and FLORIS energy ratios. Defaults to False.

    #         plot_energy_ratios (:py:obj:`bool`): Specify displaying a visualization of the SCADA and FLORIS energy ratios. Defaults to False. 

    #     Returns:
    #         fi_tuned (:py:obj:`FlorisInterface`): Tuned FLORIS model.

    #         optimal_param (:py:obj:`float`): Optimal parameter value found. 

    #         optimal_err (:py:obj:`float`): Minimum error associated with the optimal parameter value.

    #         err_curve (:py:obj:): Interpolated function representing the error curve.

    #         true_errs (:py:obj:`list[float]`): List of the mean squared errors calculated between SCADA and FLORIS energy ratios for each value in the range of parameter values being evaluated for optimality.

    #     """
        
    #     # Extract dictionary of parameters from the untuned FLORIS model object for parameter tuning
    #     fi_dict_mod = self.fi_untuned.floris.as_dict()

    #     # Calculate the mean squared error between SCADA and FLORIS energy ratios for each value in the range of parameters
    #     true_errs = []

    #     # If baseline case, tune wake expansion rate(s) 
    #     if case == 'baseline':
    #         for p in param_values:
    #             # Update 1st wake expansion rate parameter
    #             fi_dict_mod['wake']['wake_velocity_parameters']['empirical_gauss']\
    #             ['wake_expansion_rates'][0] = p # TODO: Need to address breakpoints case

    #             # Instantiate FLORIS model object with updated wake expansion rate parameter
    #             self.fi_tuned = FlorisInterface(fi_dict_mod)

    #             # Generate FLORIS dataframe for SCADA comparison
    #             df_floris = self.get_floris_df(fi=self.fi_tuned,
    #                                            pow_ref_columns=pow_ref_columns,
    #                                            time_series=time_series)
                
    #             # Calculate error
    #             err = self.evaluate_error(case=case,
    #                                       df_floris=df_floris, 
    #                                       wd_step=wd_step,
    #                                       ws_step=ws_step,
    #                                       wd_bin_width=wd_bin_width,
    #                                       wd_bins=wd_bins,
    #                                       ws_bins=ws_bins,
    #                                       N=N,
    #                                       percentiles=percentiles,
    #                                       balance_bins_between_dfs=balance_bins_between_dfs,
    #                                       return_detailed_output=return_detailed_output,
    #                                       num_blocks=num_blocks,
    #                                       verbose=verbose)
                
    #             # Track error
    #             true_errs.append(err)
                
    #     # If controlled case, tune horizontal deflection gain
    #     elif case == 'controlled':
    #         for p in param_values:
    #             # Update horizontal deflection gain parameter
    #             fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
    #             ['horizontal_deflection_gain_D'] = p

    #             # Instantiate FLORIS model object with updated horizontal deflection gain parameter
    #             self.fi_tuned = FlorisInterface(fi_dict_mod)

    #             # Generate FLORIS dataframe for SCADA comparison
    #             df_floris = self.get_floris_df(fi=self.fi_tuned,
    #                                            pow_ref_columns=pow_ref_columns,
    #                                            time_series=time_series)
                
    #             # Calculate error
    #             err = self.evaluate_error(case=case,
    #                                       df_floris=df_floris, 
    #                                       wd_step=wd_step,
    #                                       ws_step=ws_step,
    #                                       wd_bin_width=wd_bin_width,
    #                                       wd_bins=wd_bins,
    #                                       ws_bins=ws_bins,
    #                                       N=N,
    #                                       percentiles=percentiles,
    #                                       balance_bins_between_dfs=balance_bins_between_dfs,
    #                                       return_detailed_output=return_detailed_output,
    #                                       num_blocks=num_blocks,
    #                                       verbose=verbose)
                
    #             # Track error
    #             true_errs.append(err) # TODO: Consider options for supplying a custom error function, weighting/filtering error points used in the error curve interpolation, options for using raw power vs. energy ratios

    #     # Raise an error if an invalid case is provided
    #     else:
    #         raise ValueError("Can only evaluate the 'baseline' or 'controlled' case.")
        
    #     # Interpolate a function to represent the error curve using the calculated errors and their respective parameter values
    #     err_curve = interp1d(x=param_values,
    #                          y=true_errs,
    #                          fill_value='extrapolate')

    #     # Minimize the curve to determine the parameter value for which the minimum error occured
    #     min_of_err_curve = minimize_scalar(err_curve)

    #     # Extract the optimal parameter value and its associated error
    #     optimal_param = float(min_of_err_curve['x'])
    #     optimal_err = float(min_of_err_curve['fun'])

    #     # Update FLORIS model object with the optimal parameter that was found 

    #     # If baseline case, set wake expansion rate(s) to the optimal parameter value
    #     if case == 'baseline':
    #         fi_dict_mod['wake']['wake_velocity_parameters']['empirical_gauss']\
    #         ['wake_expansion_rates'][0] = optimal_param 

    #         self.fi_tuned = FlorisInterface(fi_dict_mod)      

    #     # If controlled case, set horizontal deflection gain to the optimal parameter value
    #     elif case == 'controlled':
    #         fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
    #         ['horizontal_deflection_gain_D'] = optimal_param

    #         self.fi_tuned = FlorisInterface(fi_dict_mod)

    #     # If 'plot_err' is True, plot the error for SCADA and FLORIS energy ratios
    #     if plot_err:
    #         # Specify title and xlabel
    #         err_plot_title = f'Error for {case.capitalize()} Case ({param_name} = Range({param_values[0]}, {param_values[-1]}))'
    #         err_plot_xlabel = param_name
    #         err_plot_ylabel = 'Error'
        
    #         predicted_errs = err_curve(param_values)

    #         self.plot_errs(title=err_plot_title,
    #                        xlabel=err_plot_xlabel,
    #                        ylabel=err_plot_ylabel,
    #                        true_errs=true_errs,
    #                        param_values=param_values,
    #                        predicted_errs=predicted_errs,
    #                        optimal_err=optimal_err,
    #                        optimal_param=optimal_param)

    #     # If 'plot_energy_ratios' is True, plot the energy ratios for SCADA and FLORIS
    #     if plot_energy_ratios:
    #         # Specify title
    #         energy_ratios_plot_title =f'Turbine ' + ', '.join([f'{t:>03}' for t in self.test_turbines]) + ' Energy Ratios'

    #         # Generate dataframe for tuned FLORIS model
    #         df_floris = self.get_floris_df(fi=self.fi_tuned,
    #                                        pow_ref_columns=pow_ref_columns,
    #                                        time_series=time_series)

    #         # Get energy ratios suite for SCADA and tuned FLORIS model
    #         _, s = self.get_energy_ratios(case=case,
    #                                       df_floris=df_floris,
    #                                       wd_step=wd_step,
    #                                       ws_step=ws_step,
    #                                       wd_bin_width=wd_bin_width,
    #                                       wd_bins=wd_bins,
    #                                       N=N,
    #                                       percentiles=percentiles,
    #                                       balance_bins_between_dfs=balance_bins_between_dfs,
    #                                       return_detailed_output=return_detailed_output,
    #                                       num_blocks=num_blocks,
    #                                       verbose=verbose)

    #         self.plot_energy_ratios(title=energy_ratios_plot_title,
    #                                 s=s) 
            
    #     return self.fi_tuned, optimal_param, optimal_err, err_curve, true_errs
    
    # def plot_errs(self, 
    #               title: str, 
    #               xlabel: str,
    #               ylabel: str,
    #               param_values: npt.NDArray[np.float64],
    #               true_errs: list[float], 
    #               predicted_errs: list[float], 
    #               optimal_err: float, 
    #               optimal_param: float):
    #     """
    #     Plot the mean squared error between the SCADA and FLORIS energy ratios.

    #     Args:
    #             title (:py:obj:`str`): Title of plot.

    #             xlabel (:py:obj:`str`): x-axis label of plot.

    #             ylabel (:py:obj:`str`): y-axis label of plot.

    #             param_values (:py:obj:`np.array[float]`): Range of parameter values being evaluated for optimality. Optimality is defined as the value that yields the minimum error between SCADA and FLORIS energy ratios.  

    #             true_errs (:py:obj:`list[float]`): List of the mean squared errors calculated between SCADA and FLORIS energy ratios for each value in the range of parameter values being evaluated for optimality.

    #             predicted_ers (:py:obj:`str`): List of the mean squared errors between SCADA and FLORIS energy ratios as predicted by the interpolated function representing the error curve.

    #             optimal_param (:py:obj:`float`): Optimal parameter value found. 

    #             optimal_err (:py:obj:`float`): Minimum error associated with the optimal parameter value.

    #     """

    #     # Plot true error
    #     plt.plot(param_values,
    #              true_errs,
    #              'b.', 
    #              label='err (true)')

    #     # Plot predicted error (error curve)
    #     plt.plot(param_values, 
    #              predicted_errs,
    #              'g-', 
    #              label='err (interpolated)')

    #     # Indicate the local minimum of the true error
    #     min_err_idx = np.argmin(true_errs)
    #     min_param = param_values[min_err_idx]
    #     min_err = true_errs[min_err_idx]
    #     plt.plot(min_param, 
    #              min_err, 
    #              'ro', 
    #              label='minima (local)')

    #     # Indicate the minimum determined by the optimization solution 
    #     plt.plot(optimal_param, 
    #              optimal_err, 
    #              'y*', 
    #              label='minima (optimizer)')

    #     # Label the plot
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.legend()
    
    # def plot_energy_ratios(self, title: str, s: energy_ratio_suite):
    #     """
    #     Plot the SCADA and FLORIS energy ratios.

    #     Args:
    #             title (:py:obj:`str`): Title of plot.

    #             s (:py:obj:`energy_ratio_suite`): Energy ratios suite object.
        
    #     """

    #     # Plot SCADA and FLORIS energy ratios
    #     ax = s.plot_energy_ratios(show_barplot_legend=False)

    #     # Label the plot
    #     ax[0].set_title(title)

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
    
    