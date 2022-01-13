import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

import floris.tools as wfct

from flasc.model_estimation.floris_sensitivity_analysis import (
    floris_sobol_analysis,
)

def _save_pickle(dict_in, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(dict_in, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(fn):
    with open(fn, 'rb') as handle:
        dict_out = pickle.load(handle)
    return dict_out


def load_floris(nrows=1, ncols=1, row_spacing_D=5.0, col_spacing_D=3.0,
                wd=None, ws=None, ti=None):

    # initialize FLORIS model
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_json = os.path.join(
        root_dir, '../demo_dataset/demo_floris_input.json')
    json_dict = json.load(open(input_json))
    json_dict['logging']['console']['enable'] = False
    json_dict['logging']['console']['level'] = 'WARNING'
    fi = wfct.floris_interface.FlorisInterface(input_dict=json_dict)

    D = fi.floris.farm.turbines[0].rotor_diameter
    x_row = np.arange(0, nrows) * row_spacing_D * D
    y_col = np.arange(0, ncols) * col_spacing_D * D
    x, y = np.meshgrid(x_row, y_col)

    fi.reinitialize_flow_field(layout_array=(x[0], y[0]),
                               wind_direction=wd,
                               wind_speed=ws,
                               turbulence_intensity=ti)

    return fi


def calculate_sensitivity(fi, N, calc_second_order, num_threads=8):
    # Define SA problem and initialize class

    problem = {
        'num_vars': 11,
        'names': [
            'ad',
            'alpha',
            'bd',
            'beta',
            'eps_gain',
            'ka',
            'kb',
            'ti_ai',
            'ti_constant',
            'ti_downstream',
            'ti_initial'
            ],
        'bounds': [
            [-0.10, 0.10],  # ad
            [0.4, 0.65],  # alpha, value between 0 and 1 [theoretically] based on empirical data
            [-0.04, 0.0],  # bd
            [0.06, 0.09],  # beta: value derived from analogy of jet flows
            [0.10, 0.30],  # eps_gain
            [0.18, 0.58],  # ka
            [0.002, 0.006],  # kb
            [0.55, 0.95],  # ti_ai: Crespo had 0.8325, Bay et al. have 0.8
            [0.25, 0.9],  # ti_constant: Crespo had 0.73, Bay et al. have 0.5
            [-1.0, -0.1],  # ti_downstream: exponent
            [0.01, 0.15],  # ti_initial: Crespo had 0.0325, Bay has 0.10
            ]
        }

    fsba = floris_sobol_analysis(fi=fi, problem=problem,
                                       calc_second_order=calc_second_order)
    fsba.generate_samples(N)
    print('Generated %d samples.' % fsba.samples_x.shape[0])

    # Calculate Sobol sensitivity for single wake situation
    samples_y = fsba.calculate_wfpower_for_samples(num_threads=num_threads)

    # Save outputs to a dataframe
    df = pd.DataFrame({'samples_y': samples_y})
    df[fsba.problem['names']] = fsba.samples_x

    fsba.get_sobol_sensitivity_indices(verbose=True)
    dict_out = {
        'problem': fsba.problem,
        'calc_second_order': fsba.calc_second_order,
        'N': fsba.N,
        'samples_x': fsba.samples_x,
        'samples_y': fsba.samples_y,
        'Si': fsba.Si
    }
    return dict_out


def plot_hor_flowfield(fi):
    fi.calculate_wake()
    hor_plane = fi.get_hor_plane()
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    fi.vis_layout(ax=ax)
    return fig, ax


def _case_wrapper(nrows, ncols, row_spacing, N, calc_second_order, wd=270.):
    fi = load_floris(nrows=nrows, ncols=ncols, row_spacing_D=row_spacing,
                     wd=wd)

    # Filename
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, 'case_archive')
    fn = os.path.join(
        data_path,
        'nrows%02d_ncols%02d_spacing%02d_wd_%.1f_N%05d.p'
        % (nrows, ncols, row_spacing, wd, N)
        )

    if os.path.exists(fn):
        print('Loading precalculated solution...')
        return _load_pickle(fn), fi

    dict_si = calculate_sensitivity(fi, N, calc_second_order)
    _save_pickle(dict_si, fn)

    return dict_si, fi


def plot_results(si_dict, fi):
    plot_hor_flowfield(fi)
    fsba = floris_sobol_analysis(
        fi=fi,
        problem=si_dict['problem'],
        calc_second_order=si_dict['calc_second_order']
        )
    fsba.samples_x = si_dict['samples_x']
    fsba.samples_y = si_dict['samples_y']
    fsba.N = si_dict['N']
    fsba.Si = si_dict['Si']

    fsba.plot_sobol_results()
    fsba.plot_convergence()


if __name__ == '__main__':
    N = 1000
    calc_second_order = False

    # # One turbine situation
    # print('One turbine case')
    # si_dict, fi = _case_wrapper(
    #     nrows=1, ncols=1, row_spacing=5.0, N=N,
    #     calc_second_order=calc_second_order, wd=270.
    #     )
    # plot_results(si_dict, fi)

    # # Two turbine situation, aligned
    # print('Two turbine case')
    # si_dict, fi = _case_wrapper(
    #     nrows=2, ncols=1, row_spacing=5.0, N=N,
    #     calc_second_order=calc_second_order, wd=270.
    #     )
    # plot_results(si_dict, fi)

    # # Two turbine situation, partial overlap
    # print('Two turbine case')
    # si_dict, fi = _case_wrapper(
    #     nrows=2, ncols=1, row_spacing=5.0, N=N,
    #     calc_second_order=calc_second_order, wd=267.
    #     )
    # plot_results(si_dict, fi)

    # print('Three turbine case')
    # si_dict, fi = _case_wrapper(
    #     nrows=3, ncols=1, row_spacing=5.0, N=N,
    #     calc_second_order=calc_second_order
    #     )
    # plot_results(si_dict, fi)

    print('Four turbine case')
    si_dict, fi = _case_wrapper(
        nrows=4, ncols=1, row_spacing=5.0, N=N,
        calc_second_order=calc_second_order
        )
    plot_results(si_dict, fi)

    # print('Four turbine case (partial overlap)')
    # si_dict, fi = _case_wrapper(
    #     nrows=4, ncols=1, row_spacing=5.0, N=N,
    #     calc_second_order=calc_second_order, wd=267.
    #     )
    # plot_results(si_dict, fi)

    print('Four turbine case (3D spacing)')
    si_dict, fi = _case_wrapper(
        nrows=4, ncols=1, row_spacing=3.0, N=N,
        calc_second_order=calc_second_order
        )
    plot_results(si_dict, fi)

    print('Four turbine case (9D spacing)')
    si_dict, fi = _case_wrapper(
        nrows=4, ncols=1, row_spacing=9.0, N=N,
        calc_second_order=calc_second_order
        )
    plot_results(si_dict, fi)

    plt.show()
