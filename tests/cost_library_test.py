import numpy as np
import pandas as pd
from floris import FlorisModel

from flasc import FlascDataFrame
from flasc.model_fit.cost_library import turbine_power_error, turbine_power_error_abs


def setup_data():
    # Create a simple dataframe for SCADA data
    df_scada = FlascDataFrame(
        pd.DataFrame(
            {
                "time": np.array([0, 1, 2]),
                "pow_000": np.array([1000.0, 1100.0, 1200.0]),
                "pow_001": np.array([900.0, 950.0, 1000.0]),
            }
        )
    )

    # Create a simple dataframe for FLORIS data
    df_floris = FlascDataFrame(
        pd.DataFrame(
            {
                "time": np.array([0, 1, 2]),
                "pow_000": np.array([1050.0, 1150.0, 1250.0]),
                "pow_001": np.array([950.0, 1000.0, 1050.0]),
            }
        )
    )

    # Create a dummy FlorisModel object
    fm = FlorisModel(configuration="defaults")

    return df_scada, df_floris, fm


def test_turbine_power_error():
    df_scada, df_floris, fm = setup_data()

    error = turbine_power_error(df_scada, df_floris, fm)
    expected_error = ((df_scada["pow_000"] - df_floris["pow_000"]) ** 2).sum() + (
        (df_scada["pow_001"] - df_floris["pow_001"]) ** 2
    ).sum()

    assert error == expected_error


def test_turbine_power_error_abs():
    df_scada, df_floris, fm = setup_data()

    error = turbine_power_error_abs(df_scada, df_floris, fm)
    expected_error = (df_scada["pow_000"] - df_floris["pow_000"]).abs().sum() + (
        df_scada["pow_001"] - df_floris["pow_001"]
    ).abs().sum()

    assert error == expected_error
