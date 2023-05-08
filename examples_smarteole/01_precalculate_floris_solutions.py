from pathlib import Path
import numpy as np
from time import perf_counter as timerpc

from floris.tools import ParallelComputingInterface
from flasc import floris_tools as ftools

from models import load_smarteole_floris


if __name__ == "__main__":
    # User settings
    max_workers = 16
    wake_models = ["jensen", "turbopark", "gch", "cc"]

    # Precalculate FLORIS solutions
    root_path = Path(__file__).resolve().parent / "precalculated_floris_solutions"
    root_path.mkdir(exist_ok=True)

    for wake_model in wake_models:
        fn = root_path / "df_fi_approx_{:s}.ftr".format(wake_model)
        if fn.is_file():
            print("FLORIS table for '{:s}' model exists. Skipping...".format(wake_model))
            continue

        start_time = timerpc()
        print("Precalculating FLORIS table for '{:s}' model...".format(wake_model))
        fi_pci = ParallelComputingInterface(
            fi=load_smarteole_floris(wake_model=wake_model),
            max_workers=max_workers,
            n_wind_direction_splits=max_workers,
            print_timings=True,
        )
        df_fi_approx = ftools.calc_floris_approx_table(
            fi=fi_pci,
            wd_array=np.arange(0.0, 360.01, 3.0),
            ws_array=np.arange(1.0, 30.01, 1.0),
            ti_array=[0.03, 0.06, 0.09, 0.12, 0.15],
        )
        end_time = timerpc()
        print("Computation time: {:.2f} s".format(end_time - start_time))
        df_fi_approx.to_feather(fn)
