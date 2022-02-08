import os
import numpy as np
import pandas as pd

import unittest
from flasc.floris_tools import (
    calc_floris_approx_table,
    interpolate_floris_from_df_approx
)

from floris import tools as wfct


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "..", "examples", "demo_dataset", "demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


class TestFlorisTools(unittest.TestCase):
    def test_floris_approx_table(self):
# if __name__ == "__main__":
#     if True:
        fi = load_floris()

        # Single core calculation
        df_fi_approx = calc_floris_approx_table(
            fi,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
            num_workers=1,
            num_threads=1,
        )

        # Multi core calculation
        df_fi_approx_multi = calc_floris_approx_table(
            fi,
            wd_array=np.arange(0.0, 10.0, 2.0),
            ws_array=[8.0, 9.0],
            ti_array=[0.08],
            num_workers=2,
            num_threads=2,
        )

        # Make sure singlecore and multicore solutions are equal
        self.assertTrue((df_fi_approx == df_fi_approx_multi).all().all())

        # Ensure there are no NaN entries
        self.assertTrue(~df_fi_approx.isna().any().any())
        
        # Ensure dataframe shape and columns
        self.assertTrue(("wd_000" in df_fi_approx.columns))
        self.assertTrue(("ws_001" in df_fi_approx.columns))
        self.assertTrue(("ti_002" in df_fi_approx.columns))
        self.assertTrue(("pow_003" in df_fi_approx.columns))
        self.assertAlmostEqual(df_fi_approx.shape[0], 10)

        # Now interpolate from table
        df = pd.DataFrame({"wd": [2.2, 5.8, 6.9], "ws": [8.1, 8.3, 8.8]})
        df["time"] = 0.0  # Empty array
        df = interpolate_floris_from_df_approx(df, df_fi_approx)

        # Ensure there are no NaN entries
        self.assertTrue(~df.isna().any().any())

        # Ensure dataframe shape and columns
        self.assertTrue(("wd_000" in df.columns))
        self.assertTrue(("ws_001" in df.columns))
        self.assertTrue(("ti_002" in df.columns))
        self.assertTrue(("pow_003" in df.columns))
        self.assertAlmostEqual(df.shape[0], 3)
