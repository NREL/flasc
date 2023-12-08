import unittest

import numpy as np
import pandas as pd

from flasc.turbine_analysis import yaw_pow_fitting as ywpf


def get_df_upstream():
    df_upstream = pd.DataFrame({"wd_min": [0.0], "wd_max": [360.0], "turbines": [[0, 1]]})
    return df_upstream


def load_data():
    N = 10000
    ws = np.ones(N) * 8.0
    cp = 0.45

    np.random.seed(0)  # Fixed seed
    vane = 5.0 * np.random.randn(N)
    pow = 0.5 * 1.225 * (0.25 * np.pi * 120.0**2) * ws**3 * cp * np.cos(vane * np.pi / 180.0) ** 2.0

    vane_ref = 5.0 * np.random.randn(N)
    pow_ref = (
        0.5
        * 1.225
        * (0.25 * np.pi * 120.0**2)
        * ws**3
        * cp
        * np.cos(vane_ref * np.pi / 180.0) ** 2.0
    )

    df = pd.DataFrame(
        {
            "wd": np.ones(N),
            "ws": ws,
            "vane": vane,
            "pow": pow,
            "pow_ref": pow_ref,
        }
    )

    return df


class TestYawCurveEstimation(unittest.TestCase):
    def test_cos_pp_fitting(self):
        # Initialize yaw-power curve filtering
        df = load_data()
        df_upstream = get_df_upstream()

        yaw_pow_filtering = ywpf.yaw_pow_fitting(df, df_upstream, ti=0)
        yaw_pow_filtering.calculate_curves()
        x_opt = yaw_pow_filtering.estimate_cos_pp_fit()
        self.assertAlmostEqual(x_opt[2], 1.9783977939193778)
