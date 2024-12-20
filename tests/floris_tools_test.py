import unittest

import numpy as np
import pandas as pd
from floris import FlorisModel

from flasc.utilities.floris_tools import (
    add_gaussian_blending_to_floris_approx_table,
    calc_floris_approx_table,
    estimate_ws_with_floris,
    get_dependent_turbines_by_wd,
    interpolate_floris_from_df_approx,
)
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris


class TestFlorisTools(unittest.TestCase):
    def test_floris_merge(self):
        fm_1, _ = load_floris()
        fm_2 = fm_1.copy()
        fm_2.set(layout_x=[-500.0, -500.0], layout_y=[0.0, 500.0])

        # Check if layouts are merged appropriately
        fm_merged = FlorisModel.merge_floris_models([fm_1, fm_2])
        self.assertTrue(np.all(fm_merged.layout_x == np.hstack([fm_1.layout_x, fm_2.layout_x])))
        self.assertTrue(np.all(fm_merged.layout_y == np.hstack([fm_1.layout_y, fm_2.layout_y])))
        #
        # Check if layouts are merged appropriately
        fm_merged = FlorisModel.merge_floris_models([fm_1, fm_2], reference_wind_height=200.0)
        self.assertTrue(fm_merged.core.flow_field.reference_wind_height == 200.0)

        # Also test that we raise a ValueError if we have two different reference wind heights and
        # don't specify a reference_wind_height for the merged model
        with self.assertRaises(ValueError):
            fm_1.set(reference_wind_height=90.0)
            fm_2.set(reference_wind_height=91.0)
            fm_merged = FlorisModel.merge_floris_models([fm_1, fm_2])

    def test_floris_approx_table(self):
        # Load FLORIS object
        fm, _ = load_floris()

        # Single core calculation
        df_fm_approx = calc_floris_approx_table(
            fm,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
        )

        # Multi core calculation
        df_fm_approx_multi = calc_floris_approx_table(
            fm,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
        )

        # Make sure singlecore and multicore solutions are equal
        self.assertTrue((df_fm_approx == df_fm_approx_multi).all().all())

        # Ensure there are no NaN entries
        self.assertTrue(~df_fm_approx.isna().any().any())

        # Ensure dataframe shape and columns
        # self.assertTrue(("wd_000" in df_fi_approx.columns))
        # self.assertTrue(("ws_001" in df_fi_approx.columns))
        # self.assertTrue(("ti_002" in df_fi_approx.columns))
        self.assertTrue(("pow_003" in df_fm_approx.columns))
        self.assertAlmostEqual(df_fm_approx.shape[0], 10)

        # Now interpolate from table
        df = pd.DataFrame(
            {
                "wd": [2.2, 5.8, 6.9],
                "ws": [8.1, 8.3, 8.8],
                "ti": [0.06, 0.06, 0.06],
                "ws_000": [8, 8, 8],
                "ws_001": [8, 8, 8],
                "ws_002": [8, 8, np.nan],
                "ws_003": [8, 8, 8],
                "ws_004": [8, 8, 8],
                "ws_005": [np.nan, 8, 8],
                "ws_006": [8, 8, 8],
                "pow_000": [1.0e6, 1.0e6, np.nan],
                "pow_001": [1.1e6, 1.1e6, np.nan],
                "pow_002": [1.2e6, 1.2e6, 1.2e6],
                "pow_003": [1.3e6, 1.3e6, 1.3e6],
                "pow_004": [1.4e6, 1.4e6, 1.4e6],
                "pow_005": [1.5e6, 1.5e6, 1.5e6],
                "pow_006": [np.nan, 1.6e6, 1.6e6],
            }
        )
        df["time"] = 0.0  # Empty array
        df = interpolate_floris_from_df_approx(df, df_fm_approx)

        # Ensure that NaNs are mimicked appropriately
        self.assertTrue(~df[["pow_003", "pow_004"]].isna().any().any())
        self.assertTrue(np.isnan(df.loc[2, "pow_000"]))
        self.assertTrue(np.isnan(df.loc[2, "pow_001"]))
        self.assertTrue(np.isnan(df.loc[0, "pow_006"]))

        # Ensure dataframe shape and columns
        # self.assertTrue(("wd_000" in df.columns))
        # self.assertTrue(("ws_001" in df.columns))
        # self.assertTrue(("ti_002" in df.columns))
        self.assertTrue(("pow_003" in df.columns))
        self.assertAlmostEqual(df.shape[0], 3)

    def test_gauss_blur(self):
        # Load FLORIS object
        fi, _ = load_floris()

        # Get FLORIS approx. table
        df_fi_approx = calc_floris_approx_table(
            fi,
            wd_array=np.arange(0.0, 360.0, 3.0),
            ws_array=[8.0],
            ti_array=[0.08],
        )

        # Apply Gaussian blending
        df_fi_approx_gauss = add_gaussian_blending_to_floris_approx_table(df_fi_approx)

        # Make sure that table dimensions are identical
        self.assertTrue(
            np.all(df_fi_approx_gauss[["wd", "ws", "ti"]] == df_fi_approx[["wd", "ws", "ti"]])
        )

        # Results should be smoothed, so highest point is lower and lowest point is higher
        self.assertTrue(
            df_fi_approx_gauss[[f"pow_{ti:03d}" for ti in range(7)]].sum(axis=1).max()
            < df_fi_approx[[f"pow_{ti:03d}" for ti in range(7)]].sum(axis=1).max()
        )
        self.assertTrue(
            df_fi_approx_gauss[[f"pow_{ti:03d}" for ti in range(7)]].sum(axis=1).min()
            > df_fi_approx[[f"pow_{ti:03d}" for ti in range(7)]].sum(axis=1).min()
        )

    def test_get_dependent_turbines_by_wd(self):
        # Load FLORIS object
        fi, _ = load_floris()

        # compute the dependency on turbine 2 at 226 degrees
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]))
        self.assertEqual(dep[0], [1, 6])

        # Test the change_threshold
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]), change_threshold=0.01)
        self.assertEqual(dep[0], [1])

        # Test the limit_number
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]), limit_number=1)
        self.assertEqual(dep[0], [1])

    def test_estimate_ws_with_floris(self):
        # Load FLORIS object
        fm, _ = load_floris()

        # Set as two turbine layout
        fm.set(layout_x=[0.0, 0.0], layout_y=[0.0, 500.0])

        # Create a sample SCADA dataframe
        df_scada = pd.DataFrame(
            {
                "ws_000": [0.5, 8.5, 20.0],
                "ws_001": [8.0, 8.5, 9.0],
                "pow_000": [100.0, 100.0, 100.0],
                "pow_001": [1000.0, 1000.0, 1000.0],
            }
        )

        # Estimate wind speed using FLORIS
        df_estimated = estimate_ws_with_floris(df_scada, fm)

        # Check if the estimated wind speed columns are added
        self.assertTrue("ws_est_000" in df_estimated.columns)
        self.assertTrue("ws_est_001" in df_estimated.columns)

        # Check if the estimated wind speed gain columns are added
        self.assertTrue("ws_est_gain_000" in df_estimated.columns)
        self.assertTrue("ws_est_gain_001" in df_estimated.columns)

        # Check that the third element of ws_est_000 are
        # unchanged from ws_000
        self.assertTrue(
            np.all(df_estimated["ws_est_000"].values[[2]] == df_scada["ws_000"].values[[2]])
        )

        # Check the the middle element of ws_est_000 is less than from ws_000
        self.assertTrue(df_estimated["ws_est_000"].values[1] < df_scada["ws_000"].values[1])

        # Check that estimated middle value for turbine 1 is greater that that of turbine 0
        self.assertTrue(df_estimated["ws_est_001"].values[1] > df_estimated["ws_est_000"].values[1])
