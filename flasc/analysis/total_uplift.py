"""Module for computing the total uplift in energy production."""

# This is a work in progress as we try to synthesize ideas from the
# table based methods and energy ratios back into one thing,
# some ideas we're incorporating:

# Conversion from polars to pandas
# Constructing tables (but now including tables of ratios)
# Keeping track of frequencies is matching sized tables

import warnings

import numpy as np
import polars as pl

import flasc.utilities.energy_ratio_utilities as util
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.data_processing.dataframe_manipulations import df_reduce_precision
from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# Internal version, returns a polars dataframe
def _compute_total_uplift_single(
    df_,
    df_names,
    ref_cols,
    test_cols,
    wd_cols,
    ws_cols,
    wd_step=2.0,
    wd_min=0.0,
    wd_max=360.0,
    ws_step=1.0,
    ws_min=0.0,
    ws_max=50.0,
    bin_cols_in=["wd_bin", "ws_bin"],
    weight_by="min",  # min, sum
    df_freq_pl=None,
    wd_bin_overlap_radius=0.0,
    uplift_pairs=[],
    uplift_names=[],
    remove_all_nulls=False,
):
    """Compute the total change in energy production between two sets of turbines.

    Args:
        df_ (pl.DataFrame): A dataframe containing the data to use in the calculation.
        df_names (list): A list of names to give to the dataframes.
        ref_cols (list[str]): A list of columns to use as the reference turbines
        test_cols (list[str]): A list of columns to use as the test turbines
        wd_cols (list[str]): A list of columns to derive the wind directions from
        ws_cols (list[str]): A list of columns to derive the wind speeds from
        wd_step (float): The width of the wind direction bins.
        wd_min (float): The minimum wind direction to use.
        wd_max (float): The maximum wind direction to use.
        ws_step (float): The width of the wind speed bins.
        ws_min (float): The minimum wind speed to use.
        ws_max (float): The maximum wind speed to use.
        bin_cols_in (list[str]): A list of column names
            to use for the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', or 'sum'.  'min' means
            the minimum count across the dataframes
            is used to weight the energy ratio.   'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.   Defaults to 'min'.
        df_freq_pl (pl.Dataframe): Polars dataframe of pre-provided per bin weights
        wd_bin_overlap_radius (float): The distance
            in degrees one wd bin overlaps into the next, must be
            less or equal to half the value of wd_step
        uplift_pairs: (list[tuple]): List of pairs of df_names to compute uplifts for. Each element
            of the list should be a tuple (or list) of length 2, where the first element will be the
            base case in the uplift calculation and the second element will be the test case in the
            uplift calculation. If None, no uplifts are computed.
        uplift_names: (list[str]): Names for the uplift columns, following the order of the
            pairs specified in uplift_pairs. If None, will default to "uplift_df_name1_df_name2",
        remove_all_nulls: (bool): Construct reference and test by strictly requiring all data to be
            available. If False, a minimum one data point from
            ref_cols, test_cols, wd_cols, and ws_cols
            must be available to compute the bin. Defaults to False.

    Returns:
        A tuple (dict, pl.DataFrame): containing the results of the computation
            and the frequency table.  The dictionary contains the uplift results indexed by the
            uplift_names.  The dataframe contains the weights for each wind direction
                and wind speed bin.
    """
    # Get the number of dataframes
    num_df = len(df_names)

    bin_cols_without_df_name = [c for c in bin_cols_in if c != "df_name"]

    # Filter df_ to remove null values
    null_filter = util.filter_all_nulls if remove_all_nulls else util.filter_any_nulls
    df_ = null_filter(df_, ref_cols, test_cols, ws_cols, wd_cols)
    if len(df_) == 0:
        raise RuntimeError("After removing nulls, no data remains for computation.")

    # Apply binning to dataframe and aggregate bins
    df_ = util.bin_and_group_dataframe(
        df_,
        ref_cols,
        test_cols,
        wd_cols,
        ws_cols,
        wd_step,
        wd_min,
        wd_max,
        ws_step,
        ws_min,
        ws_max,
        wd_bin_overlap_radius,
        remove_all_nulls,
        bin_cols_without_df_name,
        num_df,
    )

    # Determine the weighting of the ws/wd bins
    df_, df_freq_pl = util.add_bin_weights(df_, df_freq_pl, bin_cols_without_df_name, weight_by)

    # If total uplift requested, compute at this point
    total_uplift_result = {}
    for uplift_pair, uplift_name in zip(uplift_pairs, uplift_names):
        df_total = (
            df_.filter(pl.col("df_name").is_in(uplift_pair))
            .with_columns(
                power_ratio=pl.col("pow_test") / pl.col("pow_ref"),
                weighted_pow_ref=pl.col("pow_ref") * pl.col("count"),
            )
            .with_columns(total_count_per_bin=pl.col("count").sum().over(bin_cols_without_df_name))
            .with_columns(
                weighted_pow_ref=pl.col("weighted_pow_ref") / pl.col("total_count_per_bin")
            )
            .with_columns(
                weighted_pow_ref=pl.col("weighted_pow_ref").sum().over(bin_cols_without_df_name)
            )
            .pivot(
                values=["power_ratio"],
                columns="df_name",
                index=bin_cols_without_df_name + ["weight", "weighted_pow_ref"],
                aggregate_function="first",
            )
            # Renorm the weight
            .with_columns(pl.col("weight") / pl.col("weight").sum())
            .with_columns(delta_power_ratio=pl.col(uplift_pair[1]) - pl.col(uplift_pair[0]))
            .with_columns(
                delta_aep=pl.col("weight")
                * pl.col("delta_power_ratio")
                * pl.col("weighted_pow_ref"),
                base_aep=pl.col("weight") * pl.col(uplift_pair[0]) * pl.col("weighted_pow_ref"),
            )
            .sum()
        )

        delta_aep = 8760 * df_total.select("delta_aep").item()
        percent_delta_aep = 100 * (
            df_total.select("delta_aep").item() / df_total.select("base_aep").item()
        )

        if np.isnan(delta_aep):
            if ws_min < 5.0:
                warnings.warn(
                    "NaNs detected in power ratios. This can result from "
                    + "the reference power being 0, which can occur when wind speed is "
                    "very low. Try setting ws_min keyword argument to remove 0 power "
                    "wind speeds."
                )
            else:
                warnings.warn("NaNs detected in power ratios.")

        total_uplift_result[uplift_name] = {
            "energy_uplift_ctr": delta_aep,
            "energy_uplift_lb": None,
            "energy_uplift_ub": None,
            "energy_uplift_ctr_pc": percent_delta_aep,
            "energy_uplift_lb_pc": None,
            "energy_uplift_ub_pc": None,
        }

    return total_uplift_result, df_freq_pl


# Bootstrap function wraps the _compute_energy_ratio function
def _compute_total_uplift_bootstrap(
    er_in,
    ref_cols,
    test_cols,
    wd_cols,
    ws_cols,
    wd_step=2.0,
    wd_min=0.0,
    wd_max=360.0,
    ws_step=1.0,
    ws_min=0.0,
    ws_max=50.0,
    bin_cols_in=["wd_bin", "ws_bin"],
    weight_by="min",  # min, sum
    df_freq_pl=None,
    wd_bin_overlap_radius=0.0,
    uplift_pairs=[],
    uplift_names=[],
    N=1,
    percentiles=[5.0, 95.0],
    remove_all_nulls=False,
):
    """Compute the total change in energy between two sets of turbines with bootstrapping.

    Args:
        er_in (EnergyRatioInput): An EnergyRatioInput object
            containing the data to use in the calculation.
        ref_cols (list[str]): A list of columns to use as the reference turbines
        test_cols (list[str]): A list of columns to use as the test turbines
        wd_cols (list[str]): A list of columns to derive the wind directions from
        ws_cols (list[str]): A list of columns to derive the wind speeds from
        wd_step (float): The width of the wind direction bins.
        wd_min (float): The minimum wind direction to use.
        wd_max (float): The maximum wind direction to use.
        ws_step (float): The width of the wind speed bins.
        ws_min (float): The minimum wind speed to use.
        ws_max (float): The maximum wind speed to use.
        bin_cols_in (list[str]): A list of column names to use for
            the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', or 'sum'.  'min' means
            the minimum count across the dataframes
            is used to weight the energy ratio. 'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.
        df_freq_pl (pl.Dataframe): Polars dataframe of pre-provided per bin weights
        wd_bin_overlap_radius (float): The distance
            in degrees one wd bin overlaps into the next, must be
            less or equal to half the value of wd_step
        uplift_pairs: (list[tuple]): List of pairs of df_names to compute uplifts for. Each element
            of the list should be a tuple (or list) of length 2, where the first element will be the
            base case in the uplift calculation and the second element will be the test case in the
            uplift calculation. If None, no uplifts are computed.
        uplift_names: (list[str]): Names for the uplift columns, following the order of the
            pairs specified in uplift_pairs. If None, will default to "uplift_df_name1_df_name2"
        N (int): The number of bootstrap samples to use.
        percentiles: (list or None): percentiles to use when returning energy ratio bounds.
            If specified as None with N > 1 (bootstrapping), defaults to [5, 95].
        remove_all_nulls: (bool): Construct reference and test by strictly requiring all data to be
                available. If False, a minimum one data point
                from ref_cols, test_cols, wd_cols, and ws_cols
                must be available to compute the bin. Defaults to False.


    Returns:
        pl.DataFrame: A dataframe containing the energy ratio between the two sets of turbines.

    """
    # Otherwise run the function N times and concatenate the results to compute statistics
    uplift_single_outs = [
        _compute_total_uplift_single(
            er_in.resample_energy_table(perform_resample=(i != 0)),
            er_in.df_names,
            ref_cols,
            test_cols,
            wd_cols,
            ws_cols,
            wd_step,
            wd_min,
            wd_max,
            ws_step,
            ws_min,
            ws_max,
            bin_cols_in,
            weight_by,
            df_freq_pl,
            wd_bin_overlap_radius,
            uplift_pairs,
            uplift_names,
            remove_all_nulls,
        )
        for i in range(N)
    ]
    # df_concat = pl.concat([uplift_single_out[0] for uplift_single_out in uplift_single_outs])
    # First output contains the original table; use that df_freq_pl
    df_freq_pl = uplift_single_outs[0][1]

    # Add in the statistics
    total_uplift_result = {}

    for uplift_name in uplift_names:
        delta_aeps = np.zeros(N)
        percent_delta_aeps = np.zeros(N)

        for i in range(N):
            delta_aeps[i] = uplift_single_outs[i][0][uplift_name]["energy_uplift_ctr"]
            percent_delta_aeps[i] = uplift_single_outs[i][0][uplift_name]["energy_uplift_ctr_pc"]

        delta_aep_central = delta_aeps[0]
        delta_aep_lb = np.quantile(delta_aeps, percentiles[0] / 100)
        delta_aep_ub = np.quantile(delta_aeps, percentiles[1] / 100)

        percent_delta_aep_central = percent_delta_aeps[0]
        percent_delta_aep_lb = np.quantile(percent_delta_aeps, percentiles[0] / 100)
        percent_delta_aep_ub = np.quantile(percent_delta_aeps, percentiles[1] / 100)

        total_uplift_result[uplift_name] = {
            "energy_uplift_ctr": delta_aep_central,
            "energy_uplift_lb": delta_aep_lb,
            "energy_uplift_ub": delta_aep_ub,
            "energy_uplift_ctr_pc": percent_delta_aep_central,
            "energy_uplift_lb_pc": percent_delta_aep_lb,
            "energy_uplift_ub_pc": percent_delta_aep_ub,
        }

    return total_uplift_result, df_freq_pl


def compute_total_uplift(
    er_in: EnergyRatioInput,
    ref_turbines=None,
    test_turbines=None,
    wd_turbines=None,
    ws_turbines=None,
    use_predefined_ref=False,
    use_predefined_wd=False,
    use_predefined_ws=False,
    wd_step=2.0,
    wd_min=0.0,
    wd_max=360.0,
    ws_step=1.0,
    ws_min=0.0,
    ws_max=50.0,
    bin_cols_in=["wd_bin", "ws_bin"],
    weight_by="min",  # min or sum
    df_freq=None,
    wd_bin_overlap_radius=0.0,
    uplift_pairs=None,
    uplift_names=None,
    N=1,
    percentiles=None,
    remove_all_nulls=False,
) -> dict:  # dict output for now, may change later
    """Compute the energy ratio between two sets of turbines with bootstrapping.

    Args:
        er_in (EnergyRatioInput): An EnergyRatioInput object
            containing the data to use in the calculation.
        ref_turbines (list[int]): A list of turbine numbers to use as the reference.
        test_turbines (list[int]): A list of turbine numbers to use as the test.
        ws_turbines (list[int]): A list of turbine numbers to use for the wind speeds
        wd_turbines (list[int]): A list of turbine numbers to use for the wind directions
        use_predefined_ref (bool): If True, use the pow_ref column of df_ as the reference power.
        use_predefined_ws (bool): If True, use the ws column of df_ as the wind speed.
        use_predefined_wd (bool): If True, use the wd column of df_ as the wind direction.
        wd_step (float): The width of the wind direction bins.
        wd_min (float): The minimum wind direction to use.
        wd_max (float): The maximum wind direction to use.
        ws_step (float): The width of the wind speed bins.
        ws_min (float): The minimum wind speed to use.
        ws_max (float): The maximum wind speed to use.
        bin_cols_in (list[str]): A list of column names to
            use for the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', , or 'sum'.  'min' means
            the minimum count across the dataframes is used to weight the energy ratio.
            'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.
        df_freq (pd.Dataframe): A dataframe which specifies the
            frequency of the ws/wd bin combinations.  Provides
            a method to use an explicit or long-term weigthing of bins.  Dataframe should include
            columns ws, wd and freq_val.  ws and wd should
            orrespond to the bin centers resulting from
            the choices of the ws/wd_min / _max / _step.
            In the case that df_freq has extra bins that aren't included
            in those given by ws/wd min, max, step, they will
            be ignored in the energy ratio calculation.
            Any bins given by ws/wd min, max, step not present in
            df_freq will be assigned a frequency of zero.
            Defaults to None.
        wd_bin_overlap_radius (float): The distance in degrees
            one wd bin overlaps into the next, must be
            less or equal to half the value of wd_step
        uplift_pairs: (list[tuple]): List of pairs of df_names to compute uplifts for. Each element
            of the list should be a tuple (or list) of length 2, where the first element will be the
            base case in the uplift calculation and the second element will be the test case in the
            uplift calculation. If None, no uplifts are computed.
        uplift_names: (list[str]): Names for the uplift columns, following the order of the
            pairs specified in uplift_pairs. If None, will default to "uplift_df_name1_df_name2"
        N (int): The number of bootstrap samples to use.
        percentiles: (list or None): percentiles to use when returning energy ratio bounds.
            If specified as None with N > 1 (bootstrapping), defaults to [5, 95].
        remove_all_nulls: (bool): Construct reference and test by strictly requiring all data to be
                available. If False, a minimum one data point from ref_cols, test_cols, wd_cols,
                and ws_cols
                must be available to compute the bin. Defaults to False.

    Returns:
        EnergyRatioOutput: An EnergyRatioOutput object containing the energy ratio between the
        two sets of turbines.

    """
    # Get the polars dataframe from within the er_in
    df_ = er_in.get_df()

    # Check that inputs are valid
    util.check_compute_energy_ratio_inputs(
        df_,
        ref_turbines,
        test_turbines,
        wd_turbines,
        ws_turbines,
        use_predefined_ref,
        use_predefined_wd,
        use_predefined_ws,
        wd_step,
        wd_min,
        wd_max,
        ws_step,
        ws_min,
        ws_max,
        bin_cols_in,
        weight_by,
        df_freq,
        wd_bin_overlap_radius,
        uplift_pairs,
        uplift_names,
        False,
        N,
        percentiles,
        remove_all_nulls,
    )

    # Set up the column names for the reference and test power
    if not use_predefined_ref:
        ref_cols = [f"pow_{i:03d}" for i in ref_turbines]
    else:
        ref_cols = ["pow_ref"]

    if not use_predefined_ws:
        ws_cols = [f"ws_{i:03d}" for i in ws_turbines]
    else:
        ws_cols = ["ws"]

    if not use_predefined_wd:
        wd_cols = [f"wd_{i:03d}" for i in wd_turbines]
    else:
        wd_cols = ["wd"]

    # Confirm uplift pairs provided correctly
    if uplift_pairs is None:
        uplift_pairs = []
    elif isinstance(uplift_pairs[0], str) and len(uplift_pairs) == 2:
        # Single pair provided, not in list of lists
        uplift_pairs = [uplift_pairs]
    else:
        for up in uplift_pairs:
            if len(up) != 2:
                raise ValueError("uplift_pairs should be a list of tuples of length 2.")
    if uplift_names is not None:
        if len(uplift_names) != len(uplift_pairs):
            raise ValueError("Length of uplift_names should match length of uplift_pairs")
    else:
        uplift_names = ["uplift_" + up[1] + "/" + up[0] for up in uplift_pairs]

    # Convert the numbered arrays to appropriate column names
    test_cols = [f"pow_{i:03d}" for i in test_turbines]

    # If df_freq is provided, confirm is consistent with ws/wd min max and
    # prepare a polars table of weights
    if df_freq is not None:
        # Maybe not test, not sure yet
        # ws_edges = np.arange(ws_min, ws_max+ws_step,ws_step)
        # ws_labels = ws_edges[:-1] + np.diff(ws_edges)/2.0
        # wd_edges = np.arange(wd_min, wd_max+wd_step,wd_step)
        # wd_labels = wd_edges[:-1] + np.diff(wd_edges)/2.0

        # Conver to polars dataframe
        df_freq_pl = pl.from_pandas(df_reduce_precision(df_freq, allow_convert_to_integer=False))

        # Rename the columns
        df_freq_pl = df_freq_pl.rename({"ws": "ws_bin", "wd": "wd_bin", "freq_val": "weight"})

    else:
        df_freq_pl = None

    # If N=1, don't use bootstrapping
    if N == 1:
        if percentiles is not None:
            logger.warn("percentiles can only be used with bootstrapping (N > 1).")
        # Compute the energy ratio
        total_uplift_result, df_freq_pl = _compute_total_uplift_single(
            df_,
            er_in.df_names,
            ref_cols,
            test_cols,
            wd_cols,
            ws_cols,
            wd_step,
            wd_min,
            wd_max,
            ws_step,
            ws_min,
            ws_max,
            bin_cols_in,
            weight_by,
            df_freq_pl,
            wd_bin_overlap_radius,
            uplift_pairs,
            uplift_names,
            remove_all_nulls,
        )
    else:
        if percentiles is None:
            percentiles = [5, 95]
        elif not hasattr(percentiles, "__len__") or len(percentiles) != 2:
            raise ValueError(
                "percentiles should be a two element list of the "
                + "upper and lower desired percentiles."
            )

        total_uplift_result, df_freq_pl = _compute_total_uplift_bootstrap(
            er_in,
            ref_cols,
            test_cols,
            wd_cols,
            ws_cols,
            wd_step,
            wd_min,
            wd_max,
            ws_step,
            ws_min,
            ws_max,
            bin_cols_in,
            weight_by,
            df_freq_pl,
            wd_bin_overlap_radius,
            uplift_pairs,
            uplift_names,
            N,
            percentiles,
        )

    # Do we want some kind of more complex return object? Or are we OK
    # returning just the total_uplift_result dictionary?
    return total_uplift_result
