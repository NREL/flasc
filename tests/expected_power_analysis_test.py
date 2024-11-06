import unittest

import pandas as pd


from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.expected_power_analysis import total_uplift_expected_power


class TestExpectedPowerAnalysis(unittest.TestCase):
    def test_expected_power_analysis(self):


        # Test the returned energy ratio assuming alternative weightings of the wind speed bins
        df_base = pd.DataFrame(
            {
                "wd": [270, 270.0, 270.0, 270.0, 272.0],
                "ws": [7.0, 8.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0, 10.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0, 10.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [270, 270.0, 270.0, 270.0, 272.0],
                "ws": [7.0, 7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0, 20.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0, 30.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        total_uplift_expected_power(a_in=a_in)