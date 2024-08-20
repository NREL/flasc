"""Modular Class for Calibrating FLORIS models to SCADA data."""

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from floris import FlorisModel, ParallelFlorisModel

from flasc.data_processing import dataframe_manipulations as dfm
from flasc.flasc_dataframe import FlascDataFrame


class ModelFit:
    """Class Sumamry.

    Description
    """

    def __init__(
        self,
        df: pd.DataFrame | FlascDataFrame,
        fmodel: FlorisModel | ParallelFlorisModel,
        cost_function_handle: Callable[[FlascDataFrame, FlascDataFrame], float],
        optimization_algorithm: Callable,
        parameter_list: List[List] | List[Tuple],
        parameter_name_list: List[str],
        parameter_range_list: List[List] | List[Tuple],
        parameter_index_list: List[int] | None = None,
    ):
        """Initialize the ModelFit class.

        Args:
            df (pd.DataFrame | FlascDataFrame): DataFrame containing SCADA data.
            fmodel (FlorisModel | ParallelFlorisModel): FLORIS model to calibrate.
            cost_function_handle (Callable): Handle to the cost function.
            optimization_algorithm (Callable): Handle to the optimization algorithm.
            parameter_list (List[List] | List[Tuple]): List of FLORIS parameters to calibrate.
            parameter_name_list (List[str]): List of names for the parameters.
            parameter_range_list (List[List] | List[Tuple]): List of parameter ranges.
            parameter_index_list (List[int], optional): List of parameter indices. Defaults to None.
        """
        # Save the dataframe as a FlascDataFrame
        self.df = FlascDataFrame(df)

        # Save the FlorisModel
        self.fmodel = fmodel

        # Check if fmodel if FlorisModel or ParallelFlorisModel
        if not isinstance(fmodel, (FlorisModel, ParallelFlorisModel)):
            raise ValueError("fmodel must be a FlorisModel or ParallelFlorisModel.")
        if isinstance(fmodel, ParallelFlorisModel):
            self.is_parallel = True
        else:
            self.is_parallel = False

        # Get the number of turbines and confirm that
        # the dataframe and floris model have the same number of turbines
        self.n_turbines = dfm.get_num_turbines(self.df)

        if self.n_turbines != fmodel.n_turbines:
            raise ValueError(
                "The number of turbines in the dataframe and the Floris model do not match."
            )

        # Save the FLORIS mode
        self.fmodel = fmodel

        # Save the cost function handle
        self.cost_function_handle = cost_function_handle

        # Save the optimization algorithm
        self.optimization_algorithm = optimization_algorithm

        # Confirm that parameter_list, parameter_name_list,
        # and parameter_range_list are the same length
        if len(parameter_list) != len(parameter_name_list) or len(parameter_list) != len(
            parameter_range_list
        ):
            raise ValueError(
                "parameter_list, parameter_name_list, and parameter_range_list"
                " must be the same length."
            )

        # Save the parameter list, name list, and range list
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.parameter_range_list = parameter_range_list

        # If parameter list is provided, ensure it is the same length as parameter_list
        if parameter_index_list is not None:
            if len(parameter_index_list) != len(parameter_list):
                raise ValueError("parameter_index_list must be the same length as parameter_list.")
            self.parameter_index_list = parameter_index_list

        # If not provided, set as list of None
        else:
            self.parameter_index_list = [None] * len(parameter_list)

        # Save the number of parameters
        self.n_parameters = len(parameter_list)

        # Initialize the initial parameter values
        self.initial_parameter_values = self._get_parameter_values()

    def _get_parameter_values(self):
        """Get the current parameter values from the FLORIS model.

        Returns:
            np.ndarray: Array of parameter values.
        """
        parameter_values = np.zeros(self.n_parameters)

        for i, (parameter, parameter_index) in enumerate(
            zip(self.parameter_list, self.parameter_index_list)
        ):
            parameter_values[i] = self.fmodel.get_param(parameter, parameter_index)

        return parameter_values

    def _set_parameter_values(self, parameter_values):
        """Set the parameter values in the FLORIS model.

        Args:
            parameter_values (np.ndarray): Array of parameter values.
        """
        # Check that parameters values is self.n_parameters long
        if len(parameter_values) != self.n_parameters:
            raise ValueError("parameter_values must have length equal to the number of parameters.")

        for i, (parameter, parameter_index) in enumerate(
            zip(self.parameter_list, self.parameter_index_list)
        ):
            self.fmodel.set_param(parameter, parameter_values[i], parameter_index)
