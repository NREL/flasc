"""Modular Class for computing a fitness evaluation of a FLORIS model to SCADA data."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from floris import FlorisModel, ParallelFlorisModel

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
        fmodel: FlorisModel | ParallelFlorisModel,
        cost_function: Callable[[FlascDataFrame, FlascDataFrame], float],
        parameter_list: List[List] | List[Tuple] | None = None,
        parameter_name_list: List[str] | None = None,
        parameter_range_list: List[List] | List[Tuple] | None = None,
        parameter_index_list: List[int] | None = None,
        optimization_algorithm: Callable | None = None,
    ):
        """Initialize the ModelFit class.

        Args:
            df (pd.DataFrame | FlascDataFrame): DataFrame containing SCADA data.
            fmodel (FlorisModel | ParallelFlorisModel): FLORIS model to calibrate.
            cost_function (Callable): Handle to the cost function.
            parameter_list (List[List] | List[Tuple]): List of FLORIS parameters to calibrate.  If
                None, no parameters are calibrated.  Defaults to None.
            parameter_name_list (List[str]): List of names for the parameters.  If None, no names
                are provided.  Defaults to None.
            parameter_range_list (List[List] | List[Tuple]): List of parameter ranges.  If None, no
                ranges are provided.  Defaults to None.
            parameter_index_list (List[int], optional): List of parameter indices. Defaults to None.
            optimization_algorithm (Callable): Handle to the optimization algorithm.  If None,
                no optimization can be performed but fitness can still be evaluated.
                Defaults to None.
        """
        # Save the dataframe as a FlascDataFrame
        self.df = FlascDataFrame(df)

        # Check the dataframe
        self._check_flasc_dataframe(self.df)

        # Check if fmodel if FlorisModel or ParallelFlorisModel
        if not isinstance(fmodel, (FlorisModel, ParallelFlorisModel)):
            raise ValueError("fmodel must be a FlorisModel or ParallelFlorisModel.")

        # If FlorisModel, set to ParallelFlorisModel with 16 workers
        if isinstance(fmodel, FlorisModel):
            max_workers = 16
            self.pfmodel = ParallelFlorisModel(
                fmodel,
                max_workers=max_workers,
                n_wind_condition_splits=max_workers,
            )
        else:
            self.pfmodel = fmodel.copy()

        # Get the number of turbines and confirm that
        # the dataframe and floris model have the same number of turbines
        self.n_turbines = dfm.get_num_turbines(self.df)

        if self.n_turbines != self.pfmodel.n_turbines:
            raise ValueError(
                "The number of turbines in the dataframe and the Floris model do not match."
            )

        # Save the cost function handle
        self.cost_function = cost_function

        # If any of parameter_list, parameter_name_list, or parameter_range_list are provided,
        # ensure all are provided
        if (
            (parameter_list is None)
            or (parameter_name_list is None)
            or (parameter_range_list is None)
        ):
            if (
                parameter_list is not None
                or parameter_name_list is not None
                or parameter_range_list is not None
            ):
                raise ValueError(
                    "If any of parameter_list, parameter_name_list, or parameter_range_list "
                    " are provided, all must be provided."
                )

        # If parameter list is provided get the number of get the number of parameters and
        # set up the parameter index list if not provided
        if parameter_list is not None:
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

            # Save the number of parameters
            self.n_parameters = len(parameter_list)

            # If parameter list is provided, ensure it is the same length as parameter_list
            if parameter_index_list is not None:
                if len(parameter_index_list) != self.n_parameters:
                    raise ValueError(
                        "parameter_index_list must be the same length as parameter_list."
                    )
                self.parameter_index_list = parameter_index_list

            # If not provided, set as list of None
            else:
                self.parameter_index_list = [None] * self.n_parameters

            # Initialize the initial parameter values
            self.initial_parameter_values = self.get_parameter_values()

        # Else initialize parameters as None
        else:
            self.parameter_list = []
            self.parameter_name_list = []
            self.parameter_range_list = []
            self.parameter_index_list = []
            self.n_parameters = 0
            self.initial_parameter_values = np.array([])

        # Save the optimization algorithm
        self.optimization_algorithm = optimization_algorithm

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
        wind_directions: np.ndarray, wind_speeds: np.ndarray, powers: np.ndarray
    ) -> FlascDataFrame:
        """Form a FlascDataFrame from wind directions, wind speeds, and powers.

        Args:
            wind_directions (np.ndarray): Array of wind directions.
            wind_speeds (np.ndarray): Array of wind speeds.
            powers (np.ndarray): Array of powers.  Must be (n_findex, n_turbines).

        Returns:
            FlascDataFrame: FlascDataFrame containing the wind directions, wind speeds, and powers.
        """
        # Check that the shapes of the arrays are correct
        if wind_directions.shape[0] != wind_speeds.shape[0]:
            raise ValueError("wind_directions and wind_speeds must have the same length.")

        if wind_directions.shape[0] != powers.shape[0]:
            raise ValueError("wind_directions and powers must have the same length.")

        if powers.ndim != 2:
            raise ValueError("powers must be a 2D array.")

        # Assign the powers
        _df = FlascDataFrame(data=powers, columns=[f"pow_{i:>03}" for i in range(powers.shape[1])])
        # Assign the wind directions and wind speeds
        _df = _df.assign(wd=wind_directions, ws=wind_speeds)

        return _df

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
        wind_speeds = self.df["ws"].values
        wind_directions = self.df["wd"].values

        # TODO: Possible code for handling TI, but we might not want to force
        # TI inclusion
        # if "ti" in self.df.columns:
        #     turbulence_intensities = self.df["ti"].values
        # else:
        #     turbulence_intensities = None

        # For now just set to first value of current model
        turbulence_intensities = np.ones_like(wind_speeds) * self.pfmodel.turbulence_intensities[0]

        # Set the ParallelFlorisModel model
        self.pfmodel.set(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            turbulence_intensities=turbulence_intensities,
            **kwargs,
        )

        # Get the turbines in kW
        turbine_powers = self.pfmodel.get_turbine_powers() / 1000

        # Generate FLORIS dataframe
        df_floris = self.form_flasc_dataframe(wind_directions, wind_speeds, turbine_powers)

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

        # Evaluate the cost function
        return self.cost_function(self.df, df_floris)

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
            parameter_values[i] = self.pfmodel.get_param(parameter, parameter_index)

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
            self.pfmodel.set_param(parameter, parameter_values[i], parameter_index)
