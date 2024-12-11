# See https://nrel.github.io/flasc/ for documentation

# The purpose of these tests is to provide a consistent timing test function
# to the flasc_metrics repository.  Even if the FLASC API changes internally
# these functions should perform equivalent tasks and provide a consistent
# timing test.

import os
import time
import warnings

import numpy as np
import pandas as pd

from flasc.analysis import energy_ratio as erp
from flasc.analysis.analysis_input import AnalysisInput

N_ITERATIONS = 5


def load_data_and_prep_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir,
        "..",
        "..",
        "examples_artificial_data",
        "raw_data_processing",
        "postprocessed",
        "df_scada_data_600s_filtered_and_northing_calibrated.ftr",
    )

    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            "Please run the scripts in /raw_data_processing/"
            + "before trying any of the other examples."
        )

    df = pd.read_feather(ftr_path)

    # Let 0 be the reference turbine (pow/ws/wd) and 1 be the test turbine
    df["ws"] = df["ws_000"]
    df["wd"] = df["wd_000"]
    df["pow_ref"] = df["pow_000"]
    df["pow_test"] = df["pow_001"]

    return df


# Time how long it takes to compute the energy ratio for a single turbine
# using N=20 bootstraps
def time_energy_ratio_with_bootstrapping():
    # Number of bootstraps
    N = 20

    # Load the data
    df = load_data_and_prep_data()

    # Build the polars energy table object
    # Speciy num_blocks = num_rows to implement normal boostrapping
    a_in = AnalysisInput([df], ["baseline"], num_blocks=df.shape[0])

    # For forward consistency, define the bins by the edges
    ws_edges = np.arange(5, 25, 1.0)
    wd_edges = np.arange(0, 360, 2.0)

    # Get what new polars needs from this
    ws_max = np.max(ws_edges)
    ws_min = np.min(ws_edges)
    ws_step = ws_edges[1] - ws_edges[0]
    wd_max = np.max(wd_edges)
    wd_min = np.min(wd_edges)
    wd_step = wd_edges[1] - wd_edges[0]

    # # Run this calculation N_ITERATIONS times and take the average time
    time_results = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        start_time = time.time()

        _ = erp.compute_energy_ratio(
            a_in,
            ["baseline"],
            test_turbines=[1],
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            ws_max=ws_max,
            ws_min=ws_min,
            ws_step=ws_step,
            wd_max=wd_max,
            wd_min=wd_min,
            wd_step=wd_step,
            N=N,
        )

        end_time = time.time()
        time_results[i] = end_time - start_time

    # Return the average time
    return np.mean(time_results)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Test loading the data
    df = load_data_and_prep_data()
    print(df.head())
    print(df.shape)

    # Test timing the energy ratio
    print(time_energy_ratio_with_bootstrapping())
