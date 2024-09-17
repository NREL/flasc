"""Modular Class for computing a fitness evaluation of a FLORIS model to SCADA data."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from floris import FlorisModel, UncertainFlorisModel
from floris.parallel_floris_model_2 import ParallelFlorisModel

from flasc.data_processing import dataframe_manipulations as dfm
from flasc.flasc_dataframe import FlascDataFrame
from flasc.utilities.tuner_utilities import replicate_nan_values


class ModelFit:
    """Fit a FlorisModel to SCADA data.

    Description
    """

    def __init__(
        self,
        df: pd.DataFrame | FlascDataFrame,
        fmodel: FlorisModel | ParallelFlorisModel | UncertainFlorisModel,
        cost_function: Callable[
            [
                FlascDataFrame,
                FlascDataFrame,
                FlorisModel | ParallelFlorisModel | UncertainFlorisModel | None,
            ],
            float,
        ],
        parameter_list: List[List] | List[Tuple] = [],
        parameter_name_list: List[str] = [],
        parameter_range_list: List[List] | List[Tuple] = [],
        parameter_index_list: List[int] = [],
    ):
        """Initialize the ModelFit class.

        Args:
            df (pd.DataFrame | FlascDataFrame): DataFrame containing SCADA data.
            fmodel (FlorisModel | ParallelFlorisModel | UncertainFlorisModel):
                FLORIS model to calibrate.
            cost_function (Callable): Handle to the cost function.
            parameter_list (List[List] | List[Tuple]): List of FLORIS parameters to calibrate.  If
                None, no parameters are calibrated.  Defaults to None.
            parameter_name_list (List[str]): List of names for the parameters.  If None, no names
                are provided.  Defaults to None.
            parameter_range_list (List[List] | List[Tuple]): List of parameter ranges.  If None, no
                ranges are provided.  Defaults to None.
            parameter_index_list (List[int], optional): List of parameter indices. Defaults to None.
        """
        # Save the dataframe as a FlascDataFrame
        self.df = FlascDataFrame(df)

        # Check the dataframe
        self._check_flasc_dataframe(self.df)

        # Check if fmodel if FlorisModel or ParallelFlorisModel
        if not isinstance(fmodel, (FlorisModel, ParallelFlorisModel, UncertainFlorisModel)):
            raise ValueError(
                "fmodel must be a FlorisModel, ParallelFlorisModel or UncertainFlorisModel."
            )

        # # If FlorisModel, set to ParallelFlorisModel
        # if isinstance(fmodel, FlorisModel)
        #     self.pfmodel = ParallelFlorisModel(
        #         fmodel,
        #     )
        # elif  isinstance(fmodel, UncertainFlorisModel):

        # else:
        #     self.pfmodel = fmodel.copy()

        # For now, simply save a copy of the model
        # TODO: Don't copy?
        self.fmodel = fmodel.copy()

        # Get the number of turbines and confirm that
        # the dataframe and floris model have the same number of turbines
        self.n_turbines = dfm.get_num_turbines(self.df)

        if self.n_turbines != self.fmodel.n_turbines:
            raise ValueError(
                "The number of turbines in the dataframe and the Floris model do not match."
            )

        # Check that the cost function has 3 inputs, the SCADA dataframe, the FLORIS dataframe,
        # and the FLORIS model
        if not callable(cost_function):
            raise ValueError("cost_function must be a callable function.")
        if len(cost_function.__code__.co_varnames) != 3:
            raise ValueError("cost_function must have 3 inputs: df_scada, df_floris, fmodel.")

        # Save the cost function handle
        self.cost_function = cost_function

        # Confirm that parameter_list, parameter_name_list, and parameter_range_list and
        # parameter_index_list are lists
        if not isinstance(parameter_list, list):
            raise ValueError("parameter_list must be a list.")
        if not isinstance(parameter_name_list, list):
            raise ValueError("parameter_name_list must be a list.")
        if not isinstance(parameter_range_list, list):
            raise ValueError("parameter_range_list must be a list.")
        if not isinstance(parameter_index_list, list):
            raise ValueError("parameter_index_list must be a list.")

        # Confirm that parameter_list, parameter_name_list,
        # and parameter_range_list are the same length
        if len(parameter_list) != len(parameter_name_list) or len(parameter_list) != len(
            parameter_range_list
        ):
            raise ValueError(
                "parameter_list, parameter_name_list, and parameter_range_list"
                " must be the same length."
            )

        # If any of parameter_list, parameter_name_list, or parameter_range_list are provided,
        # (in that they have lengths greater than 0) then all must be provided
        if len(parameter_list) > 0 or len(parameter_name_list) > 0 or len(parameter_range_list) > 0:
            if (
                len(parameter_list) == 0
                or len(parameter_name_list) == 0
                or len(parameter_range_list) == 0
            ):
                raise ValueError(
                    "If any of parameter_list, parameter_name_list, or parameter_range_list"
                    " are provided, all must be provided."
                )

        # Save the parameter list, name list, and range list
        self.parameter_list = parameter_list
        self.parameter_name_list = parameter_name_list
        self.parameter_range_list = parameter_range_list

        # Save the number of parameters
        self.n_parameters = len(parameter_list)

        # If parameter_index_list is empty, set as a list of None equal to the number of parameters
        if len(parameter_index_list) == 0:
            self.parameter_index_list = [None] * self.n_parameters

        # Else ensure it is the same length as parameter_list
        else:
            if len(parameter_index_list) != self.n_parameters:
                raise ValueError("parameter_index_list must be the same length as parameter_list.")
            self.parameter_index_list = parameter_index_list

        # Initialize the initial parameter values
        self.initial_parameter_values = self.get_parameter_values()

    def _check_flasc_dataframe(self, df: FlascDataFrame) -> None:
        """Check that the provided FlascDataFrame is valid.

        Args:
            df (FlascDataFrame): DataFrame to check.
        """
        # Data frame must contain a 'ws' and 'wd' column
        if "ws" not in df.columns or "wd" not in df.columns:
            raise ValueError("DataFrame must contain 'ws' and 'wd' columns.")

    @staticmethod
    def form_flasc_dataframe(
        time: np.ndarray, wind_directions: np.ndarray, wind_speeds: np.ndarray, powers: np.ndarray
    ) -> FlascDataFrame:
        """Form a FlascDataFrame from wind directions, wind speeds, and powers.

        Args:
            time (np.ndarray): Array of time values.
            wind_directions (np.ndarray): Array of wind directions.
            wind_speeds (np.ndarray): Array of wind speeds.
            powers (np.ndarray): Array of powers.  Must be (n_findex, n_turbines).

        Returns:
            FlascDataFrame: FlascDataFrame containing the wind directions, wind speeds, and powers.
        """
        # Check that lengths of time, wind directions
        if time.shape[0] != wind_directions.shape[0]:
            raise ValueError("time and wind_directions must have the same length.")

        # Check that the shapes of the arrays are correct
        if wind_directions.shape[0] != wind_speeds.shape[0]:
            raise ValueError("wind_directions and wind_speeds must have the same length.")

        if wind_directions.shape[0] != powers.shape[0]:
            raise ValueError("wind_directions and powers (0th axis) must have the same length.")

        if powers.ndim != 2:
            raise ValueError("powers must be a 2D array.")

        # Name the power columns
        pow_cols = [f"pow_{i:>03}" for i in range(powers.shape[1])]

        # Assign the powers
        _df = pd.DataFrame(data=powers, columns=pow_cols)
        # Assign the wind directions and wind speeds
        _df = _df.assign(time=time, wd=wind_directions, ws=wind_speeds)

        # Re-order the columns
        _df = _df[["time", "wd", "ws"] + pow_cols]

        return FlascDataFrame(_df)

    def run_floris_model(self, **kwargs) -> FlascDataFrame:
        """Run the FLORIS model with the current parameter values.

        Given the provided FLORIS model and SCADA data, run the FLORIS model
        and generate a FlascDataFrame of FLORIS values.  Note **kwargs are
        provided to allow additional settings to be passed to the
        ParallelFlorisModel.set method.

        Args:
            **kwargs: Additional keyword arguments to pass to the
                ParallelFlorisModel.set method.

        Returns:
            FlascDataFrame: _description_
        """
        # Get the wind speeds, wind directions and turbulence intensities
        time = self.df["time"].values
        wind_speeds = self.df["ws"].values
        wind_directions = self.df["wd"].values

        # TODO: Possible code for handling TI, but we might not want to force
        # TI inclusion
        # if "ti" in self.df.columns:
        #     turbulence_intensities = self.df["ti"].values
        # else:
        #     turbulence_intensities = None

        # For now just set to first value of current model
        turbulence_intensities = np.ones_like(wind_speeds) * self.fmodel.turbulence_intensities[0]

        # Set the ParallelFlorisModel model
        self.fmodel.set(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            turbulence_intensities=turbulence_intensities,
            **kwargs,
        )

        # Run the model
        self.fmodel.run()

        # Get the turbines in kW
        turbine_powers = self.fmodel.get_turbine_powers() / 1000

        # Generate FLORIS dataframe
        df_floris = self.form_flasc_dataframe(time, wind_directions, wind_speeds, turbine_powers)

        # Make sure the NaN values in the SCADA data appear in the same locations in the
        # FLORIS data
        df_floris = replicate_nan_values(self.df, df_floris)

        # Return df_floris
        return df_floris

    def evaluate_floris(self, **kwargs) -> float:
        """Evaluate the FLORIS model.

        Given the current parameter values, run the FLORIS model and evaluate the cost function.

        Returns:
            float: cost value.
        """
        # Run the FLORIS model
        df_floris = self.run_floris_model(**kwargs)

        print(df_floris)

        # Evaluate the cost function passing the FlorisModel within the ParallelFlorisModel
        return self.cost_function(self.df, df_floris, self.fmodel)

    def set_parameter_and_evaluate(self, parameter_values: np.ndarray, **kwargs) -> float:
        """Internal function to evaluate the cost function with a given set of parameters.

        Args:
            parameter_values (np.ndarray): Array of parameter values.
            **kwargs: Additional keyword arguments to pass to the optimization algorithm.

        Returns:
            float: Cost value.
        """
        # Set the parameter values
        self.set_parameter_values(parameter_values)

        # Evaluate the cost function
        return self.evaluate_floris(**kwargs)

    def get_parameter_values(
        self,
    ) -> np.ndarray:
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

    def set_parameter_values(
        self,
        parameter_values: np.ndarray,
    ) -> None:
        """Set the parameter values in the FLORIS model.

        Args:
            parameter_values (np.ndarray): Array of parameter values.
        """
        # Check that parameters values is len(parameter_list) long
        if len(parameter_values) != self.n_parameters:
            raise ValueError("parameter_values must have length equal to the number of parameters.")

        for i, (parameter, parameter_index) in enumerate(
            zip(self.parameter_list, self.parameter_index_list)
        ):
            self.fmodel.set_param(parameter, parameter_values[i], parameter_index)
