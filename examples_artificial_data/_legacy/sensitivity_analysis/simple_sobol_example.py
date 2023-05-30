import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import floris.tools as wfct
from floris.logging_manager import configure_console_log
from pandas.errors import DataError

from flasc import floris_sensitivity_analysis as fsasa


def load_floris():
    # initialize FLORIS model
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_json = os.path.join(
        root_dir, '../demo_dataset/demo_floris_input.json')
    fi = wfct.floris_interface.FlorisInterface(input_json)
    configure_console_log(False)  # Disable INFO statements

    # Set default scenario
    fi.reinitialize_flow_field(wind_direction=62.)

    return fi


def plot_hor_flowfield(fi):
    fi.calculate_wake()
    hor_plane = fi.get_hor_plane()
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    fi.vis_layout(ax=ax)
    return fig, ax


if __name__ == '__main__':
    # Load FLORIS
    fi = load_floris()
    plot_hor_flowfield(fi)
    plt.show()

    # Define SA problem and initialize class
    problem = {'num_vars': 3,
               'names': ['alpha', 'ti_constant', 'ti_ai'],
               'bounds': [[0.058, 5.8], [0.05, 5.], [0.08, 8.]]}
    N = int(2000)

    fsba = fsasa.floris_sobol_analysis(fi=fi, problem=problem,
                                       calc_second_order=False)
    fsba.generate_samples(N)

    # Check if precalculated solutions exist
    root_path = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(root_path,
                           'prec_solutions_calc2ndorder%s_N%04d.ftr'
                           % (str(fsba.calc_second_order), N))
    if os.path.exists(df_path):
        # Load from existing file/solutions
        df = pd.read_feather(df_path)
        if not all(df['samples_x'] == fsba.samples_x):
            raise DataError('Somehow generated samples are different.')
        fsba.samples_y = np.array(df['samples_y'])
    else:
        # Calculate Sobol sensitivity for single wake situation
        samples_y = fsba.calculate_wfpower_for_samples(print_progress=True)
        df = pd.DataFrame({'samples_x': fsba.samples_x,
                           'samples_y': samples_y})
        df.to_feather(df_path)

    fsba.get_sobol_sensitivity_indices(verbose=True)
    fsba.plot_convergence()
    fsba.plot_sobol_results()
    plt.show()
