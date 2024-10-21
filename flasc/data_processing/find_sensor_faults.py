"""Module for finding sensor-stuck faults in a dataframe."""

import os

import matplotlib.pyplot as plt
import numpy as np

from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def find_sensor_stuck_faults(
    df,
    columns,
    ti,
    stddev_threshold=0.001,
    n_consecutive_measurements=3,
    plot_figures=True,
    verbose=False,
    return_by_column=False,
):
    """Find sensor-stuck faults in a dataframe.

    Args:
        df (pd.DataFrame | FlascDataFrame): The dataframe containing the data.
        columns (list): The columns to check for sensor-stuck faults.
        ti (Any): unused
        stddev_threshold (float, optional): The threshold for the standard deviation of the
            consecutive measurements. Defaults to 0.001.
        n_consecutive_measurements (int, optional): The number of consecutive measurements to
            compare. Defaults to 3.
        plot_figures (bool, optional): Whether to plot figures for the sensor-stuck faults.
            Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        return_by_column (bool, optional): Whether to return the faults by column.
            Defaults to False.

    Returns:
        np.array: The indices of the sensor-stuck faults
    """
    # Settings which indicate a sensor-stuck type of fault: the standard
    # deviation between the [no_consecutive_measurements] number of
    # consecutive measurements is less than [stddev_threshold].

    # TODO: remove unused argument 'ti'

    index_faults = {c: np.array([]) for c in columns}
    for c in columns:
        if verbose:
            logger.info("Processing column %s" % c)
        measurement_array = np.array(df[c])

        column_index_faults = _find_sensor_stuck_single_timearray(
            measurement_array=measurement_array,
            no_consecutive_measurements=n_consecutive_measurements,
            stddev_threshold=stddev_threshold,
            index_array=df.index,
        )

        if (plot_figures) & (len(column_index_faults) > 0):
            _plot_top_sensor_faults(df, c, column_index_faults)

        index_faults[c] = column_index_faults

    if return_by_column:
        return index_faults
    else:
        return np.unique(np.concatenate([v for v in index_faults.values()]))


def _plot_top_sensor_faults(
    df,
    c,
    index_faults,
    N_eval_max=5,
    save_path=None,
    fig_format="png",
    dpi=300,
):
    # Extract largest fault set and plot
    diff_index_faults = np.diff(index_faults)
    diffjumps = np.where(diff_index_faults > 1)[0]
    fault_sets = []
    imin = 0
    for imax in list(diffjumps):
        if (imax - imin) > 1:
            fault_sets.append(index_faults[imin + 1 : imax])
        imin = imax
    if len(index_faults) - imin > 1:
        fault_sets.append(index_faults[imin + 1 : :])
    fault_sets_idx_sorted = np.argsort([len(i) for i in fault_sets])[::-1]
    N_eval = np.min([N_eval_max, len(fault_sets)])
    fig, ax_array = plt.subplots(nrows=N_eval, ncols=1, figsize=(5.0, 2.5 * N_eval))

    if N_eval == 1:
        ax_array = [ax_array]

    for i in range(N_eval):
        ax = ax_array[i]
        fault_set_eval = fault_sets[fault_sets_idx_sorted[i]]

        indices_to_plot = range(
            fault_set_eval[0] - 4 * len(fault_set_eval),
            fault_set_eval[-1] + 4 * len(fault_set_eval),
        )
        indices_to_plot = [v for v in indices_to_plot if v in df.index]

        ax.plot(df.loc[indices_to_plot, "time"], df.loc[indices_to_plot, c], "o")
        ax.plot(
            df.loc[index_faults, "time"],
            df.loc[index_faults, c],
            "o",
            color="red",
        )
        ax.set_xlim(
            (
                df.loc[indices_to_plot[0], "time"],
                df.loc[indices_to_plot[-1], "time"],
            )
        )
        plt.xticks(rotation="vertical")
        ax.legend(["Good data", "Faulty data"])
        ax.set_ylabel(c)
        ax.set_xlabel("Time")
        ax.set_title("Column '%s', sensor stuck fault %d" % (c, i))

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, "%s_faults.%s" % (c, fig_format))
        fig.savefig(fig_path, dpi=dpi)

    return fig, ax_array


def _find_sensor_stuck_single_timearray(
    measurement_array, no_consecutive_measurements=6, stddev_threshold=0.05, index_array=None
):
    # Create index array, if unspecified
    N = len(measurement_array)
    if index_array is None:
        index_array = np.array(range(N))

    # Ensure variable types
    index_array = np.array(index_array)
    measurement_array = np.array(measurement_array)

    # Remove nans from measurement array
    index_array = index_array[~np.isnan(measurement_array)]
    measurement_array = measurement_array[~np.isnan(measurement_array)]

    def format_array(array_in, row_length):
        array_in = np.array(array_in)
        Nm = row_length - 1
        C = array_in[0:-Nm]
        for ii in range(1, Nm):
            C = np.vstack([C, array_in[ii : -Nm + ii]])
        C = np.vstack([C, array_in[Nm::]]).T
        return C

    Cindex = format_array(index_array, row_length=no_consecutive_measurements)
    Cmeas = format_array(measurement_array, row_length=no_consecutive_measurements)

    # Get standard deviations and determine faults
    std_array = np.std(Cmeas, axis=1)
    indices_faulty = np.unique(Cindex[std_array < stddev_threshold, :])

    return indices_faulty
