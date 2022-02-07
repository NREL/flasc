from datetime import timedelta as td
import numpy as np
import pandas as pd

import unittest

from flasc.time_operations import df_resample_by_interpolation


class TestDataFrameResamplingInterpolation(unittest.TestCase):
    def test_resample_by_interpolation(self):
        # Define a 'raw' data signal with gaps in time stamps
        time = [
            "2019-01-01 12:50:00",
            "2019-01-01 12:50:03",
            "2019-01-01 12:50:04",
            "2019-01-01 12:50:05",
            "2019-01-01 12:51:03",
            "2019-01-01 12:51:04",
            "2019-01-01 12:51:07",
            "2019-01-01 12:51:10",
        ]
        time = [pd.to_datetime(t) for t in time]

        df = pd.DataFrame(
            {
                "time": time,
                "wd_000": [355, 356, 1, 2, 359, 1, 0, 359.9],
                "ws_000": [7.0, 7.1, 7.2, 7.0, 6.9, 7.0, 7.0, 7.0],
                "vane_000": [-3, -3, -4, -3, -5, 0, 2, 3]
            }
        )

        # Now resample that by filling in the gaps
        df_res = df_resample_by_interpolation(
            df=df,
            time_array=pd.date_range(time[0], time[-1], freq=td(seconds=1)),
            circular_cols=["wd_000"],
            interp_method='linear',
            max_gap=td(seconds=5),  # Maximum gap of 5 seconds
            verbose=False,
        )

        # Make sure values are NaN at the start of the gap
        self.assertTrue(np.isnan(df_res.loc[11, "wd_000"]))
        self.assertTrue(np.isnan(df_res.loc[11, "ws_000"]))
        self.assertTrue(np.isnan(df_res.loc[11, "vane_000"]))

        # Make sure values are NaN at the end of the gap
        self.assertTrue(np.isnan(df_res.loc[57, "wd_000"]))
        self.assertTrue(np.isnan(df_res.loc[57, "ws_000"]))
        self.assertTrue(np.isnan(df_res.loc[57, "vane_000"]))

        # Make sure linear interpolation works correctly over gaps
        self.assertAlmostEqual(df_res.loc[68, "wd_000"], 359.9667, places=3)
        self.assertAlmostEqual(df_res.loc[68, "vane_000"], 2.3333, places=3)
