import numpy as np
from scipy.stats import circmean

import unittest

from flasc import circular_statistics as cs


class TestDataframeManipulations(unittest.TestCase):
    def test_circular_statistics(self):
        angles_array = np.array(
            [
                [355.0, 341.2, 2.1],
                [13.2, 344.9, 356.1],
                [2.2, 334.0, 356.5]
            ]
        )
        mean_array, median_array, std_array, min_array, max_array = (
            cs.calculate_wd_statistics(
                angles_array,
                axis=0,
                calc_median_min_max_std=True,
            )
        )

        # Compare to SciPy solutions
        means_scipy = circmean(angles_array, axis=0, high=360.0)
        self.assertAlmostEqual(mean_array[0], means_scipy[0], places=4)
        self.assertAlmostEqual(mean_array[1], means_scipy[1], places=4)
        self.assertAlmostEqual(mean_array[2], means_scipy[2], places=4)

        # Compare to precalculated solutions
        self.assertAlmostEqual(mean_array[0], 3.461333, places=4)
        self.assertAlmostEqual(mean_array[1], 340.03507657, places=4)
        self.assertAlmostEqual(mean_array[2], 358.2326068, places=4)

        self.assertAlmostEqual(median_array[0], 2.2, places=4)
        self.assertAlmostEqual(median_array[1], 341.2, places=4)
        self.assertAlmostEqual(median_array[2], 356.5, places=4)

        self.assertAlmostEqual(std_array[0], 7.48390866, places=4)
        self.assertAlmostEqual(std_array[1], 4.52572892, places=4)
        self.assertAlmostEqual(std_array[2], 2.73901848, places=4)

        self.assertAlmostEqual(min_array[0], 355.0, places=4)
        self.assertAlmostEqual(min_array[1], 334.0, places=4)
        self.assertAlmostEqual(min_array[2], 356.1, places=4)

        self.assertAlmostEqual(max_array[0], 13.2, places=4)
        self.assertAlmostEqual(max_array[1], 344.9, places=4)
        self.assertAlmostEqual(max_array[2], 2.1, places=4)
