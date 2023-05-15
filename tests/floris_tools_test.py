import os
import numpy as np
import pandas as pd

import unittest
from flasc.floris_tools import (
    calc_floris_approx_table,
    merge_floris_objects,
    interpolate_floris_from_df_approx,
    get_dependent_turbines_by_wd
)

from floris import tools as wfct


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../examples_artificial_data/demo_dataset/demo_floris_input.yaml")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


class TestFlorisTools(unittest.TestCase):
    def test_floris_merge(self):
        fi_1 = load_floris()
        fi_2 = fi_1.copy()
        fi_2.reinitialize(layout_x=[-500.0, -500.0], layout_y=[0.0, 500.0])

        # Check if layouts are merged appropriately
        fi_merged = merge_floris_objects([fi_1, fi_2])
        self.assertTrue(np.all(fi_merged.layout_x == np.hstack([fi_1.layout_x, fi_2.layout_x])))
        self.assertTrue(np.all(fi_merged.layout_y == np.hstack([fi_1.layout_y, fi_2.layout_y])))
# 
        # Check if layouts are merged appropriately
        fi_merged = merge_floris_objects([fi_1, fi_2], reference_wind_height=200.0)
        self.assertTrue(fi_merged.floris.flow_field.reference_wind_height == 200.0)

        # Also test that we raise a UserWarning if we have two different reference wind heights and
        # don't specify a reference_wind_height for the merged object
        with self.assertRaises(UserWarning):
            fi_1.reinitialize(reference_wind_height=90.0)
            fi_2.reinitialize(reference_wind_height=91.0)
            fi_merged = merge_floris_objects([fi_1, fi_2])

    def test_floris_approx_table(self):
        # Load FLORIS object
        fi = load_floris()

        # Single core calculation
        df_fi_approx = calc_floris_approx_table(
            fi,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
        )

        # Multi core calculation
        df_fi_approx_multi = calc_floris_approx_table(
            fi,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
        )

        # Make sure singlecore and multicore solutions are equal
        self.assertTrue((df_fi_approx == df_fi_approx_multi).all().all())

        # Ensure there are no NaN entries
        self.assertTrue(~df_fi_approx.isna().any().any())
        
        # Ensure dataframe shape and columns
        # self.assertTrue(("wd_000" in df_fi_approx.columns))
        # self.assertTrue(("ws_001" in df_fi_approx.columns))
        # self.assertTrue(("ti_002" in df_fi_approx.columns))
        self.assertTrue(("pow_003" in df_fi_approx.columns))
        self.assertAlmostEqual(df_fi_approx.shape[0], 10)

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
        df = interpolate_floris_from_df_approx(df, df_fi_approx)

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

    def test_get_dependent_turbines_by_wd(self):
        # Load FLORIS object
        fi = load_floris()

        # compute the dependency on turbine 2 at 226 degrees
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]))
        self.assertEqual(dep[0], [1, 6])

        # Test the change_threshold
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]), 
            change_threshold=0.01)
        self.assertEqual(dep[0], [1])

        # Test the limit_number
        dep = get_dependent_turbines_by_wd(fi, 2, np.array([226]), 
            limit_number=1)
        self.assertEqual(dep[0], [1])