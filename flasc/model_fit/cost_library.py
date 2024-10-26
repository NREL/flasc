"""Library of cost functions for the optimization."""

from typing import List

import pandas as pd
from floris import FlorisModel

from flasc.data_processing.dataframe_manipulations import (
    _set_col_by_turbines,
    set_pow_ref_by_turbines,
)
from flasc.flasc_dataframe import FlascDataFrame


def total_wake_loss_error(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
    fm_: FlorisModel,
    turbine_groupings: List = None,
):
    """Evaluate the overall wake loss from pow_ref to pow_test as percent reductions.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data
        fm_ (FlorisModel): FLORIS model (Unused but required for compatibility)
        turbine_groupings (List): List of turbine groupings.  Defaults to None.
            In None case, assumes pow_ref and pow_test are already identified (note
            this can be challenging to effect within FLORIS resimulation results).

    Returns:
        float: Overall wake losses squared error

    """
    if turbine_groupings is not None:
        # Set the reference turbines in both frames
        df_scada = set_pow_ref_by_turbines(df_scada, turbine_groupings["pow_ref"])
        df_floris = set_pow_ref_by_turbines(df_floris, turbine_groupings["pow_ref"])

        # Set the test turbines in both frames
        df_scada = _set_col_by_turbines(
            "pow_test", "pow", df_scada, turbine_groupings["pow_test"], False
        )
        df_floris = _set_col_by_turbines(
            "pow_test", "pow", df_floris, turbine_groupings["pow_test"], False
        )

    scada_wake_loss = df_scada["pow_ref"].values - df_scada["pow_test"].values
    floris_wake_loss = df_floris["pow_ref"].values - df_floris["pow_test"].values

    return ((scada_wake_loss - floris_wake_loss) ** 2).sum()

def simple_floris_error(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
    fm_: FlorisModel,
    turbine_groupings: List = None,
):
    """Evaluate the overall wake loss from pow_ref to pow_test as percent reductions.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data
        fm_ (FlorisModel): FLORIS model (Unused but required for compatibility)
        turbine_groupings (List): List of turbine groupings.  Defaults to None.
            In None case, assumes pow_ref and pow_test are already identified (note
            this can be challenging to effect within FLORIS resimulation results).

    Returns:
        float: Overall wake losses squared error

    """
    df_scada = set_pow_ref_by_turbines(df_scada, list(range(df_scada.n_turbines)))
    df_floris = set_pow_ref_by_turbines(df_floris, list(range(df_scada.n_turbines)))


    error = df_scada["pow_ref"].values - df_floris["pow_ref"].values
    error = error ** 2
    return error.sum()


def turbine_by_turbine(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
    fm_: FlorisModel,
    turbine_groupings: List = None,
):
    """Evaluate the overall wake loss from pow_ref to pow_test as percent reductions.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data
        fm_ (FlorisModel): FLORIS model (Unused but required for compatibility)
        turbine_groupings (List): List of turbine groupings.  Defaults to None.
            In None case, assumes pow_ref and pow_test are already identified (note
            this can be challenging to effect within FLORIS resimulation results).

    Returns:
        float: Overall wake losses squared error

    """
    turbine_columns = [f"pow_{i:03d}" for i in range(df_scada.n_turbines)]

    df_error = (df_scada[turbine_columns] - df_floris[turbine_columns])**2

    return df_error.sum().sum()


