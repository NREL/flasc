"""Analyze the results of model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from great_tables import GT

from flasc.flasc_dataframe import FlascDataFrame


class ResultsAnalysis:
    """Analyze results.

    Description
    """

    def __init__(
        self,
        df_scada: pd.DataFrame | FlascDataFrame,
        floris_powers: list,
        floris_models: list,
        floris_calibrations: list,
    ):
        """Initialize the ResultsAnalysis object.

        Args:
            df_scada: DataFrame containing SCADA data.
            floris_powers: List of power results from FLORIS
            floris_models: List of wake models
            floris_calibrations: List of calibration settings for each model
        """
        # Check that floris_powers, floris_models, and floris_calibrations have the same length
        if not (len(floris_powers) == len(floris_models) == len(floris_calibrations)):
            raise ValueError(
                "floris_powers, floris_models, and floris_calibrations must have the same length"
            )

        # Get the number of floris cases
        self.n_floris_cases = len(floris_powers)

        # Confirm that each 2D table in floris_powers has the same shape
        if not all(
            floris_powers[0].shape == floris_powers[i].shape for i in range(1, self.n_floris_cases)
        ):
            raise ValueError("All power tables must have the same shape")

        # Confirm that df_scada has the number of rows as the power tables
        if df_scada.shape[0] != floris_powers[0].shape[0]:
            raise ValueError(
                "The number of rows in df_scada must match the number of rows in the power tables"
            )

        # If df_scada is not already a FlascDataFrame, convert it
        if not isinstance(df_scada, FlascDataFrame):
            self.df_scada = FlascDataFrame(df_scada)
        else:
            # If it is already a FlascDataFrame, just assign it
            self.df_scada = df_scada

        # Check that self.df_scada.n_turbines == floris_powers[0].shape[1]
        if self.df_scada.n_turbines != floris_powers[0].shape[1]:
            raise ValueError(
                "The number of turbines in df_scada must match the number of columns in the power "
                "tables."
            )

        # Save the number of turbines
        self.n_turbines = self.df_scada.n_turbines
        self.n_findex = self.df_scada.shape[0]

        # Save the FLORIS inputs
        self.floris_powers = floris_powers
        self.floris_models = floris_models
        self.floris_calibrations = floris_calibrations

        # Generate the names of the power columns
        self.power_column_names = [f"pow_{i:03d}" for i in range(self.n_turbines)]

        # Generate the large table for computing grouped statistics
        df_list = []
        for floris_case in range(self.n_floris_cases):
            for t in range(self.n_turbines):
                scada_data = self.df_scada[self.power_column_names[t]].values
                floris_data = self.floris_powers[floris_case][:, t]

                # should already by true but as triple check
                # assign a nan to floris_data where scada_data is nan
                floris_data[np.isnan(scada_data)] = np.nan

                model = self.floris_models[floris_case]
                calibration = self.floris_calibrations[floris_case]

                df_temp = pd.DataFrame(
                    {
                        "scada": scada_data,
                        "floris": floris_data,
                        "turbine": t,
                        "model": model,
                        "calibration": calibration,
                    }
                )
                df_list.append(df_temp)
        self.df_full_table = pd.concat(df_list, ignore_index=True)
        self.df_full_table = self.df_full_table.reset_index(drop=True)

        # Add the error column
        self.df_full_table["error"] = self.df_full_table["scada"] - self.df_full_table["floris"]

        # Add the absolute error column
        self.df_full_table["error_abs"] = self.df_full_table["error"].abs()

    # Plot the error distributions
    def plot_error(self, ax=None):
        """Plot the error distributions.

        Args:
            ax: Axes to plot on. If None, create a new figure.
        """
        sns.catplot(
            data=self.df_full_table,
            x="model",
            hue="calibration",
            y="error",
            kind="box",
            showfliers=False,
        )

    # Plot the error distributions in absolute
    def plot_error_abs(self, ax=None):
        """Plot the error distributions in absolute.

        Args:
            ax: Axes to plot on. If None, create a new figure.
        """
        sns.catplot(
            data=self.df_full_table.assign(error_abs=self.df_full_table["error"].abs()),
            x="model",
            hue="calibration",
            y="error_abs",
            kind="box",
            showfliers=False,
        )

    # Generate the table of statistics of turbine error
    def table_turbine_error(
        self,
        dict_model_names={},
        dict_calibration_names={},
        decimals=1,
        calculation="median",
        use_abs=False,
    ):
        """Generate a summary table of the errors.

        Args:
            dict_model_names: Dictionary of model names to replace in the table.
            dict_calibration_names: Dictionary of calibration names to replace in the table.
            decimals: Number of decimals to display in the table.
            calculation: Type of calculation to perform. Options are 'median' or 'mean' or 'std'
            use_abs: If True, use absolute error instead of raw error.

        Returns:
            GT table of median errors
        """
        if use_abs:
            error_term = "error_abs"
            title_suffix = " (absolute)"
        else:
            error_term = "error"
            title_suffix = ""

        if calculation == "mean":
            result_table = (
                self.df_full_table.groupby(["model", "calibration"])[error_term]
                .mean()
                .reset_index()
            )
        elif calculation == "std":
            result_table = (
                self.df_full_table.groupby(["model", "calibration"])[error_term].std().reset_index()
            )
        elif calculation == "median":
            result_table = (
                self.df_full_table.groupby(["model", "calibration"])[error_term]
                .median()
                .reset_index()
            )
        else:
            raise ValueError("calculation must be 'median', 'mean', or 'std'")

        # If dict_model_names is not empty, replace the model names
        if dict_model_names:
            result_table["model"] = result_table["model"].map(dict_model_names)
        # If dict_calibration_names is not empty, replace the calibration names
        if dict_calibration_names:
            result_table["calibration"] = result_table["calibration"].map(dict_calibration_names)

        cal_levels = list(result_table.calibration.unique())

        # Pivot to table
        result_table = result_table.pivot(
            index="model", columns="calibration", values=error_term
        ).reset_index()

        gt_table = (
            GT(result_table)
            .tab_header(title=f"{calculation.capitalize()} Turbine Error (kW)" + title_suffix)
            .cols_move_to_start(columns=["model"] + cal_levels)
            .fmt_number(columns=cal_levels, decimals=decimals)
        )
        return gt_table

    # Generate the table of median errors
    def table_farm_energy(
        self,
        dict_model_names={},
        dict_calibration_names={},
        decimals=1,
        use_pchange=False,
    ):
        """Generate a summary table of the errors.

        Args:
            dict_model_names: Dictionary of model names to replace in the table.
            dict_calibration_names: Dictionary of calibration names to replace in the table.
            decimals: Number of decimals to display in the table.
            use_pchange: If True, express results in percent change

        Returns:
            GT table of median errors
        """
        # First compute the SCADA total energy
        # First see what the SCADA result is (same in every case)
        scada_table = self.df_full_table.groupby(["model", "calibration"])["scada"].sum() / 6e6  #
        scada_table = scada_table.reset_index()

        # Make sure there is only one unique value of the 'scada' column
        if scada_table["scada"].nunique() != 1:
            raise ValueError("There should be only one unique value of the 'scada' column")

        scada_result = scada_table["scada"].values[0]
        print(f"SCADA RESULT: {scada_result:.2f} GWh")

        # Now compute the result table
        result_table = self.df_full_table.groupby(["model", "calibration"]).floris.sum() / 6e6

        # If using p_change
        if use_pchange:
            result_table = result_table - scada_result
            result_table = result_table / scada_result * 100

        result_table = result_table.reset_index()

        # If dict_model_names is not empty, replace the model names
        if dict_model_names:
            result_table["model"] = result_table["model"].map(dict_model_names)
        # If dict_calibration_names is not empty, replace the calibration names
        if dict_calibration_names:
            result_table["calibration"] = result_table["calibration"].map(dict_calibration_names)

        cal_levels = list(result_table.calibration.unique())

        # Pivot to table
        result_table = result_table.pivot(
            index="model", columns="calibration", values="floris"
        ).reset_index()

        # Get ready the table
        if use_pchange:
            table_sub_title = "Percent Change From SCADA (%)"
        else:
            table_sub_title = f"GWh (SCADA: {scada_result:.2f} GWh)"

        gt_table = (
            GT(result_table)
            .tab_header(title="Total Energy Production", subtitle=table_sub_title)
            .cols_move_to_start(columns=["model"] + cal_levels)
            .fmt_number(columns=cal_levels, decimals=decimals)
        )
        return gt_table
