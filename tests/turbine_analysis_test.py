import numpy as np
import pandas as pd

from flasc.data_processing.find_sensor_faults import find_sensor_stuck_faults


def test_find_sensor_stuck_faults():
    df_test = pd.DataFrame({"a": [0, 0, 0, 1, 2, 3, 4], "b": [4, 5, 6, 6, 7, 7, 7]})

    # Test default behavior
    results_a = find_sensor_stuck_faults(df_test, columns=["a"], ti=0, plot_figures=False)
    assert (results_a == np.array([0, 1, 2])).all()

    results_b = find_sensor_stuck_faults(df_test, columns=["b"], ti=0, plot_figures=False)
    assert (results_b == np.array([4, 5, 6])).all()

    results_ab = find_sensor_stuck_faults(df_test, columns=["a", "b"], ti=0, plot_figures=False)
    assert (results_ab == np.array([0, 1, 2, 4, 5, 6])).all()

    results_ba = find_sensor_stuck_faults(df_test, columns=["b", "a"], ti=0, plot_figures=False)
    assert (results_ab == results_ba).all()

    results_ab2 = find_sensor_stuck_faults(
        df_test, columns=["a", "b"], ti=0, n_consecutive_measurements=2, plot_figures=False
    )
    assert (results_ab2 == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    # Test returning by column
    results = find_sensor_stuck_faults(
        df_test, columns=["a", "b"], ti=0, plot_figures=False, return_by_column=True
    )
    test_dict = {"a": np.array([0, 1, 2]), "b": np.array([4, 5, 6])}
    assert results["a"].size == 3
    assert results["b"].size == 3
    assert results.keys() == test_dict.keys()
    assert all((results[k] == test_dict[k]).all() for k in test_dict)

    # Test stddev_threshold
    df_test = pd.DataFrame({"a": [0, 0.1, -0.1, 0.05, 1]})
    std_true = np.std(df_test["a"][:-1])

    results = find_sensor_stuck_faults(df_test, columns=["a"], ti=0, plot_figures=False)
    assert results.size == 0  # Empty array, no fault detected

    results = find_sensor_stuck_faults(
        df_test, columns=["a"], ti=0, stddev_threshold=std_true * 2, plot_figures=False
    )
    assert (results == np.array([0, 1, 2, 3])).all()
