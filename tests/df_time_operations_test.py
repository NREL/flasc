from datetime import timedelta as td
import numpy as np
import pandas as pd

import unittest

from flasc.time_operations import (
    df_resample_by_interpolation,
    df_movingaverage,
    df_downsample,
)


def load_data():
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
    df = pd.DataFrame(
        {
            "time": [pd.to_datetime(t) for t in time],
            "wd_000": [355, 356, 1, 2, 359, 1, 0, 359.9],
            "ws_000": [7.0, 7.1, 7.2, 7.0, 6.9, 7.0, 7.0, 7.0],
            "vane_000": [-3, -3, -4, -3, -5, 0, 2, 3]
        }
    )
    return df


class TestDataFrameResampling(unittest.TestCase):
# if __name__ == "__main__":
#     if True:
    def test_downsampling(self):
        df = load_data()
        df_ds, data_indices = df_downsample(
            df_in=df,
            cols_angular=["wd_000"],
            window_width=td(seconds=5),
            min_periods=1,
            center=True,
            calc_median_min_max_std=True,
            return_index_mapping=True,
        )

        # Check solutions: for first row
        self.assertAlmostEqual(df_ds.iloc[0]["ws_000_mean"], 7.10)
        self.assertAlmostEqual(df_ds.iloc[0]["ws_000_std"], 0.10)
        self.assertAlmostEqual(df_ds.iloc[0]["wd_000_std"], 2.624669, places=4)
        self.assertTrue(np.all(np.unique(data_indices[0, :]) == [-1, 0, 1, 2]))

        # Check solutions: for big chunk of data in middle of dataframe (Nones)
        self.assertTrue(df_ds.iloc[4:11].isna().all().all())
        self.assertTrue(np.all(np.unique(data_indices[4:11, :]) == [-1]))

        # Check solutions: for one but last row
        self.assertAlmostEqual(df_ds.iloc[-2]["ws_000_mean"], 7.0)
        self.assertTrue(np.all(np.unique(data_indices[-2, :]) == [-1, 6]))
        self.assertTrue(np.isnan(df_ds.iloc[-2]["vane_000_std"]))

    def test_moving_average(self):
        df = load_data()
        df_ma, data_indices = df_movingaverage(
            df_in=df,
            cols_angular=["wd_000"],
            window_width=td(seconds=5),
            min_periods=1,
            center=True,
            calc_median_min_max_std=True,
            return_index_mapping=True,
        )

        # Check solutions: for first row which just used one value for mov avg
        self.assertAlmostEqual(df_ma.iloc[0]["ws_000_mean"], 7.0)
        self.assertTrue(np.isnan(df_ma.iloc[0]["ws_000_std"]))
        self.assertTrue(np.all(np.unique(data_indices[0, :]) == [-1, 0]))

        # Check solutions: second row with multiple values
        self.assertTrue(np.all(np.unique(data_indices[1, :]) == [-1, 1, 2, 3]))
        self.assertAlmostEqual(df_ma.iloc[1]["wd_000_mean"], 359.667246, places=4)  # confirm circular averaging
        self.assertAlmostEqual(df_ma.iloc[1]["wd_000_std"], 2.624669, places=4)  # confirm circular std

        # Check solutions: sixth row, for good measure
        self.assertTrue(np.all(np.unique(data_indices[6, :]) == [-1, 6]))
        self.assertAlmostEqual(df_ma.iloc[6]["wd_000_mean"], 0.0)  # confirm circular averaging
        self.assertTrue(np.isnan(df_ma.iloc[6]["ws_000_std"]))
        self.assertTrue(np.isnan(df_ma.iloc[6]["vane_000_std"]))

    def test_resample_by_interpolation(self):
        # Now resample that by filling in the gaps
        df = load_data()
        df_res = df_resample_by_interpolation(
            df=df,
            time_array=pd.date_range(
                df.iloc[0]["time"],
                df.iloc[-1]["time"],
                freq=td(seconds=1)
            ),
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
