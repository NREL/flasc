import numpy as np
import pandas as pd

import unittest

from flasc.optimization import (
    find_timeshift_between_dfs,
    match_y_curves_by_offset
)


class TestOptimization(unittest.TestCase):
    def test_estimation_dy_offset(self):
        # Define a reference signal
        t = np.linspace(0, 4*np.pi, 1000)
        yref = 180 + 180 * np.sin(0.01 * t)

        # Define a shifted signal with 360 deg angle wrapping
        ytest = 44.0 + yref
        ytest = ytest + 5 * np.random.randn(1000)
        ytest[ytest >= 360.0] += -360.0

        # Estimate shift in curves between the two
        dy, _ = match_y_curves_by_offset(
            yref,
            ytest,
            dy_eval=np.arange(-180.0, 180.0, 1.0),
            angle_wrapping=True,
        )

        self.assertAlmostEqual(dy, 44.0)

    def test_estimation_df_timeshift(self):
        # Define a reference signal
        t = pd.date_range(
            "2019-01-10 12:15:01",
            "2019-01-10 16:15:01",
            freq='1s'
        )
        t1 = pd.to_datetime("2000-01-01 00:00:00")
        y1 = 180 + 180 * np.sin(0.001 * (t - t1) / np.timedelta64(1, 's'))
        df1 = pd.DataFrame({"time": t, "wd_000": y1})

        # Now define similar signal but shifted by 5 minutes
        t2 = pd.to_datetime("2000-01-01 00:5:00")
        y2 = 180 + 180 * np.sin(0.001 * (t - t2) / np.timedelta64(1, 's'))
        df2 = pd.DataFrame({"time": t, "wd_000": y2})

        out = find_timeshift_between_dfs(
            df1=df1,
            df2=df2,
            cols_df1=["wd_000"],
            cols_df2=["wd_000"],
            use_circular_statistics=True,
            opt_bounds=[np.timedelta64(-60, 'm'), np.timedelta64(60, 'm')],
            opt_Ns=13,
        )
        x = np.timedelta64(out[0]["x_opt"]) / np.timedelta64(1, 's')
        self.assertAlmostEqual(x, -300.0)  # Should be equal to 5 minute shift
