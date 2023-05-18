import numpy as np
import pandas as pd
import os

import unittest

from floris.tools import FlorisInterface

from flasc.dataframe_operations import dataframe_filtering as dff
from flasc import floris_tools as ftools
from flasc.turbine_analysis.ws_pow_filtering import ws_pw_curve_filtering


def load_floris():
    # Initialize the FLORIS interface fi
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../examples_artificial_data/demo_dataset/demo_floris_input.yaml")
    fi = FlorisInterface(fi_path)
    return fi


def load_data():
    # Create tiny subset
    return pd.DataFrame({
        "wd": [255.0],
        "ws": [8.0],
        "ti": [0.07],
        "ws_000": [8.0],
        "ws_001": [8.0],
        "ws_002": [8.0],
        "ws_003": [8.0],
        "ws_004": [8.0],
        "ws_005": [8.0],
        "ws_006": [8.0],
        "pow_000": [2.3e6],
        "pow_001": [2.3e6],
        "pow_002": [2.3e6],
        "pow_003": [2.3e6],
        "pow_004": [2.3e6],
        "pow_005": [2.3e6],
        "pow_006": [2.3e6],
    })


class TestDataFrameFiltering(unittest.TestCase):
    def test_ws_pow_filtering(self):
# if __name__ == "__main__":
        # Test basic filtering operations
        df = load_data()
        df["ws_001"] = np.nan
        w = ws_pw_curve_filtering(df)
        for ti in range(7):
            # Filter for NaN wind speeds
            w.filter_by_condition(
                condition=w.df["ws_{:03d}".format(ti)].isna(),
                label="Wind speed is NaN",
                ti=ti,
            )

        # Verify that ws_001 and pow_001 are only NaNs
        self.assertTrue(w.get_df()[["ws_001", "pow_001"]].isna().all().all())
        self.assertTrue(not w.get_df()[[c for c in df.columns if not "001" in c]].isna().any().any())
        self.assertTrue((w.df_filters["WTG_001"] =="Wind speed is NaN").all())

        # Test basic filtering operations: sensor-stuck filtering
        df = load_data()
        df = pd.concat([df] * 10, axis=0).reset_index(drop=True)
        df = df * (1 + 0.05 * np.random.randn(*np.shape(df.to_numpy())))
        df.loc[[4, 5, 6, 7], "ws_004"] = 8.801  # Assign 4 measurements to be stuck

        w = ws_pw_curve_filtering(df)
        w.filter_by_sensor_stuck_faults(columns=["wd", "ws_004"], ti=4)
        df_filtered = w.get_df()
        array_is_stuck = np.array(df_filtered["ws_004"].isna(), dtype=bool)
        self.assertTrue(np.all(array_is_stuck[[4, 5, 6, 7]]))
        self.assertTrue(not np.any(array_is_stuck[[0, 1, 2, 3, 8, 9]]))

        # Test interactive filtering. Not a real unit test but make sure it runs.
        # Now filter iteratively by deviations from the median power curve
        w.filter_by_power_curve(
            ti=ti,
            ws_deadband=1.5,
            pow_deadband=70.0,
            cutoff_ws=20.0,
            m_pow_rb=0.97,
        )

    def test_impacting_filtering(self):
        # Read file and load FLORIS
        fi = load_floris()
        num_turbs = len(fi.layout_x)

        # Determine which turbines impact which other turbines through their wakes
        df_impacting_turbines = ftools.get_all_impacting_turbines(fi_in=fi, change_threshold=0.01)

        # Create tiny subset
        df_base = load_data()

        # Try a couple scenarios. 1st scenario: T04 waking T03 and T04 is NaN
        # print("\n\n Creating scenario where T4 wakes T3 and T6 and T4 is faulty:")
        df = df_base.copy()
        df["wd"] = 275.0
        df["pow_004"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_003", "pow_004", "pow_006"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_001", "pow_002", "pow_005"]].isna().any().any())  # Non-NaN
                    
        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T1 is faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_001"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001"]].isna().all().all())
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_005", "pow_006"]].isna().any().any())
    
        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T1 and T5 are faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_001"] = np.nan
        df["pow_005"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001", "pow_005"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_006"]].isna().any().any())  # Non-NaN

        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T5 is faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_005"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001", "pow_005"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_006"]].isna().any().any())  # Non-NaN

        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1, T06 wakes T0, and T5 and T6 are faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_005"] = np.nan
        df["pow_006"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_000", "pow_001", "pow_005", "pow_006"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_002", "pow_003", "pow_004"]].isna().any().any())  # Non-NaN

        # plot_layout_only(fi)
        # plt.show()
