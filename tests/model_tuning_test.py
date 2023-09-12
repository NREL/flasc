from flasc.model_tuning.tuner_utils import replicate_nan_values
import unittest

import pandas as pd
import numpy as np

class TestModelTuning(unittest.TestCase):

    def test_replicate_nan_values(self):
        # Sample dataframes
        data_1 = {'A': [1, 2, np.nan, 4],
                'B': [5, np.nan, 7, 8],
                'C': [np.nan, 1, 1, 1]}
        data_2 = {'A': [10, 20, 30, 40],
                'B': [50, 60, 70, 80]}
        df_1 = pd.DataFrame(data_1)
        df_2 = pd.DataFrame(data_2)

        # Call the function to replicate NaN values
        result_df = replicate_nan_values(df_1, df_2)

        # Expected output
        expected_df_1 = pd.DataFrame({'A': [1, 2, np.nan, 4],
                                    'B': [5, np.nan, 7, 8],
                                    'C': [np.nan, 1, 1, 1]})
        expected_df_2 = pd.DataFrame({'A': [10, 20, np.nan, 40],
                                    'B': [50, np.nan, 70, 80]})

        # Check if the result matches the expected output
        assert result_df.equals(expected_df_2)
        assert df_1.equals(expected_df_1)
        