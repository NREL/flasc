

import pandas as pd


from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.expected_power_analysis import total_uplift_expected_power, _bin_and_group_dataframe_expected_power

import numpy as np

def load_data():

    # Test the returned energy ratio assuming alternative weightings of the wind speed bins
    df_base = pd.DataFrame(
        {
            "wd": [270, 270.0,270.0, 270.0, 280.0, 280.0, 280.0, 290.0],
            "ws": [8.0, 8.0,8.0, 9.0, 8.0, 8.0, 9.0, 8.0],
            "pow_000": [10.0, 20.0,np.nan, 10.0, np.nan, 10.0, np.nan,10.0],
            "pow_001": [10.0, 20.0, np.nan,10.0,10.0, 20.0, np.nan, 10.0]
        }
    )

    df_wake_steering = pd.DataFrame(
        {
            "wd": [270, 270.0,  280.0, 280.0, 280.0, 290.0],
            "ws": [8.0, 8.0, 8.0, 8.0, 9.0, 8.0],
            "pow_000": [10.0, 10.0, 10.0, 10.,10.,10.0],
            "pow_001": [20., 30., np.nan,np.nan, 10.0,10.0]
        }
    )

    a_in = AnalysisInput(
        [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
    )

    return a_in

def test_bin_and_group_dataframe_expected_power():

    a_in = load_data()

    df_ = _bin_and_group_dataframe_expected_power(df_=a_in.get_df(),
                                      test_cols=["pow_000", "pow_001"],
                                      wd_cols=["wd"],
                                      ws_cols=["ws"],
                                      wd_step=1.0,
                                      wd_min=0.5,
                                      ws_min=0.5)
    
    # Sort df_ by wd_bin and ws_bin and df_name
    df_ = df_.sort(["wd_bin", "ws_bin", "df_name"])

    # Test the values
    np.testing.assert_array_equal(df_["wd_bin"].to_numpy(), np.array([270.0, 270.0, 280.0, 280.0, 290.0, 290.0]))
    np.testing.assert_array_equal(df_["ws_bin"].to_numpy(), np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0]))
    np.testing.assert_array_equal(df_["df_name"].to_numpy(), np.array(["baseline", "wake_steering", "baseline", "wake_steering","baseline", "wake_steering", ]))
    np.testing.assert_array_equal(df_["pow_000_mean"].to_numpy(), np.array([15.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
    np.testing.assert_array_equal(df_["pow_001_mean"].to_numpy(), np.array([15.0, 25.0, 15.0, np.nan,10.0,10.0]))
    np.testing.assert_array_equal(df_["pow_000_var"].to_numpy(), np.array([50.0, 0.0, np.nan, 0.0, np.nan,np.nan]))
    np.testing.assert_array_equal(df_["pow_000_count"].to_numpy(), np.array([2,2,1,2,1,1]))
    np.testing.assert_array_equal(df_["pow_001_count"].to_numpy(), np.array([2,2,2,0,1,1]))

# def test_expected_power_analysis():



#     total_uplift_expected_power(a_in=a_in)