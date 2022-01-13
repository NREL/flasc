import copy
import os

import floris.tools as wfct
import matplotlib.pyplot as plt
import numpy as np


from flasc import floris_tools as ftools
from flasc import turbulence_estimator as fsatiestimator


if __name__ == "__main__":
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    fi.vis_layout()

    # Generate measurement
    fi_scada = copy.deepcopy(fi)
    fi_scada = ftools._fi_set_ws_wd_ti(fi_scada, ws=8.95, wd=235.5,
                                       ti=[0.08, 0.12, 0.12, 0.18, 0.18, 0.18, 0.12])
    fi_scada.calculate_wake()
    P_measured = np.array(fi_scada.get_turbine_power())
    P_measured = P_measured * (1+.05*np.random.rand(len(P_measured)))

    # Initialize estimation class
    tb = fsatiestimator.ti_estimator(fi=fi)
    tb.floris_set_ws_wd_ti(wd=236., ws=9.0, ti=0.05)
    tb.set_measurements(P_measured=P_measured)
    fig, ax = tb.plot_power_bars()
    ax.set_title('Power bars before optimization')

    # Plot flow field error (m/s)
    hor_plane = fi_scada.get_hor_plane()
    hor_plane_est = tb.fi.get_hor_plane()
    hor_plane.df[['u', 'v', 'w']] -= hor_plane_est.df[['u', 'v', 'w']]
    fig, ax = plt.subplots()
    im = wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    fi.vis_layout(ax=ax)
    plt.colorbar(im, label='Wind speed error (m/s)')
    ax.set_title('Wind speed error before estimation')

    # Estimation on a farm-wide level
    tb.estimate_farmaveraged_ti(Ns=50, bounds=(0.01, 0.50), verbose=True)
    # tb.plot_cost_function_farm()
    fig, ax = tb.plot_power_bars()
    ax.set_title('Power bars after farm-wide optimization')

    # Plot flow field error (m/s)
    hor_plane = fi_scada.get_hor_plane()
    hor_plane_est = tb.fi.get_hor_plane()
    hor_plane.df[['u', 'v', 'w']] -= hor_plane_est.df[['u', 'v', 'w']]
    fig, ax = plt.subplots()
    im = wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    fi.vis_layout(ax=ax)
    plt.colorbar(im, label='Wind speed error (m/s)')
    ax.set_title('Wind speed error after farm-wide estimation')

    # Estimation on a turbine level
    tb.get_turbine_pairs()
    tb.get_turbine_order()
    print('Turbine order:', tb.turbine_list_ordered)
    tb.estimate_local_tis(Ns=50, bounds=(0.01, 0.50),
                          refine_with_fmin=True, verbose=True)

    # tb.plot_cost_functions_turbines()
    fig, ax = tb.plot_power_bars()
    ax.set_title('Power bars after turbine-individual optimization')

    # Plot flow field error (m/s)
    hor_plane = fi_scada.get_hor_plane()
    hor_plane_est = tb.fi.get_hor_plane()
    hor_plane.df[['u', 'v', 'w']] -= hor_plane_est.df[['u', 'v', 'w']]
    fig, ax = plt.subplots()
    im = wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    fi.vis_layout(ax=ax)
    plt.colorbar(im, label='Wind speed error (m/s)')
    ax.set_title('Wind speed error after turbine-individual estimation')

    # Plot estimated flow field (m/s)
    fig, ax = plt.subplots()
    im = wfct.visualization.visualize_cut_plane(hor_plane_est, ax=ax)
    plt.colorbar(im, label='Estimated wind field (m/s)')
    ax.set_title('Estimated wind field from turbine-individual estimation')
    wfct.visualization.plot_turbines_with_fi(ax=ax, fi=tb.fi)
    fi.vis_layout(ax=ax)
    plt.show()
