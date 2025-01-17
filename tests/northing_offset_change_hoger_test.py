import numpy as np
import pandas as pd

from flasc import FlascDataFrame
from flasc.data_processing.northing_offset_change_hoger import (
    _discretize,
    _shorth_mode,
    homogenize_hoger,
)


def test_discretize():
    """Test discretize function."""

    x = pd.Series([0, 5, 1000, 2, 75])
    expected_result = pd.Series([1, 1, 2, 1, 1])
    threshold = 100
    result = _discretize(x, threshold)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(x)
    np.testing.assert_array_equal(result, expected_result)


def test_shorth_mode():
    """Test shorth function."""
    x = pd.Series([1.0, 1.0, 1.0, 1.5, 2.0])
    expected_result = 1.0
    result = _shorth_mode(x)
    assert isinstance(result, np.float64)
    assert result == expected_result


def test_homogenize_hoger():
    """Test homogenize_hoger function."""
    N = 100

    df = FlascDataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=N, freq="600s"),
            "wd": np.random.randint(0, 360, N),
            "ws": np.random.randint(0, 20, N),
            "pow_000": np.random.randint(0, 100, N),
            "pow_001": np.random.randint(0, 100, N),
            "pow_002": np.random.randint(0, 100, N),
            "pow_003": np.random.randint(0, 100, N),
            "pow_004": np.random.randint(0, 100, N),
            "wd_000": np.zeros(N),
            "wd_001": np.zeros(N),
            "wd_002": np.zeros(N),
            "wd_003": np.zeros(N),
            "wd_004": np.zeros(N),
        }
    )

    # Add a step change at N/2 in wd_004
    df.loc[N // 2 :, "wd_004"] = 20

    # If threshold is larger than number of points, df_hom should match df
    df_hom, d2 = homogenize_hoger(df.copy(), threshold=N * 2)
    assert df.equals(df_hom)

    # If threshold is smaller than number of points, df_hom should homogenize wd_004
    df_hom, d2 = homogenize_hoger(df.copy(), threshold=10)
    assert not df.equals(df_hom)
    assert df_hom["wd_004"].nunique() == 1  # Test homogenize_hoger column

    # All columns besides wd_004 are unchanged
    assert df["wd_000"].equals(df_hom["wd_000"])
    assert df["wd_001"].equals(df_hom["wd_001"])
    assert df["wd_002"].equals(df_hom["wd_002"])
    assert df["wd_003"].equals(df_hom["wd_003"])

    # If threshold == N should homogenize all columns
    df_hom, d2 = homogenize_hoger(df.copy(), threshold=N)
    assert not df.equals(df_hom)
    assert df_hom["wd_004"].nunique() == 1  # Test homogenized column


def test_homogenize_hoger_double_change():
    """Test homogenize_hoger function with two changes."""
    N = 250

    df = FlascDataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=N, freq="600s"),
            "wd": np.random.randint(0, 360, N),
            "ws": np.random.randint(0, 20, N),
            "pow_000": np.random.randint(0, 100, N),
            "pow_001": np.random.randint(0, 100, N),
            "pow_002": np.random.randint(0, 100, N),
            "pow_003": np.random.randint(0, 100, N),
            "pow_004": np.random.randint(0, 100, N),
            "wd_000": np.zeros(N),
            "wd_001": np.zeros(N),
            "wd_002": np.zeros(N),
            "wd_003": np.zeros(N),
            "wd_004": np.zeros(N),
        }
    )

    # Add a step change at N/2 in wd_004
    df.loc[N // 3 :, "wd_004"] = 20
    df.loc[2 * N // 3 :, "wd_004"] = 40

    # If threshold is smaller than number of points, df_hom should homogenize wd_004
    df_hom, d2 = homogenize_hoger(df.copy(), threshold=N // 5)
    assert not df.equals(df_hom)
    assert df_hom["wd_004"].nunique() == 1  # Test homogenized column

    # All columns besides wd_004 are unchanged
    assert df["wd_000"].equals(df_hom["wd_000"])
    assert df["wd_001"].equals(df_hom["wd_001"])
    assert df["wd_002"].equals(df_hom["wd_002"])
    assert df["wd_003"].equals(df_hom["wd_003"])

    # If threshold is larger than number of points, df_hom should match df
    df_hom, d2 = homogenize_hoger(df.copy(), threshold=N * 2)
    assert df.equals(df_hom)
