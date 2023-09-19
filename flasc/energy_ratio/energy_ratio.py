# This is a work in progress as we try to synthesize ideas from the 
# table based methods and energy ratios back into one thing, 
# some ideas we're incorporating:

# Conversion from polars to pandas
# Constructing tables (but now including tables of ratios)
# Keeping track of frequencies is matching sized tables

import warnings
import numpy as np
import polars as pl

from flasc.energy_ratio.energy_ratio_output import EnergyRatioOutput
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput
from flasc.energy_ratio.energy_ratio_utilities import (
    add_ws_bin,
    add_wd,
    add_wd_bin,
    add_power_ref,
    add_power_test,
    add_reflected_rows,
    check_compute_energy_ratio_inputs,
    filter_all_nulls,
    filter_any_nulls
)
from flasc.dataframe_operations.dataframe_manipulations import df_reduce_precision


# Internal version, returns a polars dataframe
def _compute_energy_ratio_single(df_,
                         df_names,
                         ref_cols,
                         test_cols,
                         wd_cols,
                         ws_cols,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         weight_by = 'min', #min, sum
                         df_freq_pl = None,
                         wd_bin_overlap_radius = 0.,
                         uplift_pairs = [],
                         uplift_names = [],
                         remove_all_nulls = False
                         ):

    """
    Compute the energy ratio between two sets of turbines.

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
        bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', or 'sum'.  'min' means
            the minimum count across the dataframes is used to weight the energy ratio.   'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.   Defaults to 'min'.
        df_freq_pl (pl.Dataframe) Polars dataframe of pre-provided per bin weights
        wd_bin_overlap_radius (float): The distance in degrees one wd bin overlaps into the next, must be 
            less or equal to half the value of wd_step
        uplift_pairs: (list[tuple]): List of pairs of df_names to compute uplifts for. Each element 
            of the list should be a tuple (or list) of length 2, where the first element will be the 
            base case in the uplift calculation and the second element will be the test case in the 
            uplift calculation. If None, no uplifts are computed.
        uplift_names: (list[str]): Names for the uplift columns, following the order of the 
            pairs specified in uplift_pairs. If None, will default to "uplift_df_name1_df_name2",
        remove_all_nulls: (bool): Construct reference and test by strictly requiring all data to be 
            available. If False, a minimum one data point from ref_cols, test_cols, wd_cols, and ws_cols
            must be available to compute the bin. Defaults to False.

    Returns:
        pl.DataFrame: A dataframe containing the energy ratio for each wind direction bin
    """

    # Get the number of dataframes
    num_df = len(df_names)

    # Filter df_ to remove null values
    null_filter = filter_all_nulls if remove_all_nulls else filter_any_nulls
    df_ = null_filter(df_, ref_cols, test_cols, ws_cols, wd_cols)
    if len(df_) == 0:
        raise RuntimeError("After removing nulls, no data remains for computation.")

    # If wd_bin_overlap_radius is not zero, add reflected rows
    if wd_bin_overlap_radius > 0.:

        # Need to obtain the wd column now rather than during binning
        df_ = add_wd(df_, wd_cols, remove_all_nulls)

        # Add reflected rows
        edges = np.arange(wd_min, wd_max + wd_step, wd_step)
        df_ = add_reflected_rows(df_, edges, wd_bin_overlap_radius)

    # Assign the wd/ws bins
    df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max, remove_all_nulls=remove_all_nulls)
    df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max, remove_all_nulls=remove_all_nulls)

    # Assign the reference and test power columns
    df_ = add_power_ref(df_, ref_cols)
    df_ = add_power_test(df_, test_cols)

    bin_cols_without_df_name = [c for c in bin_cols_in if c != 'df_name']
    bin_cols_with_df_name = bin_cols_without_df_name + ['df_name']

    # Group df_
    df_ = (df_
        .filter(pl.all_horizontal(pl.col(bin_cols_with_df_name).is_not_null())) # Select for all bin cols present
        .group_by(bin_cols_with_df_name, maintain_order=True)
        .agg([pl.mean("pow_ref"), pl.mean("pow_test"),pl.count()])

        # Enforce that each ws/wd bin combination has to appear in all dataframes
        .filter(pl.count().over(bin_cols_without_df_name) == num_df)

    )
    # Determine the weighting of the ws/wd bins

    if df_freq_pl is None:
        # Determine the weights per bin as either the min or sum count
        df_ = (df_
            .with_columns(
                [
                    # Get the weighting by counts
                    pl.col('count').min().over(bin_cols_without_df_name).alias('weight')  if weight_by == 'min' else
                    pl.col('count').sum().over(bin_cols_without_df_name).alias('weight')
                ]
            )
        )
    
    else:
        # Use the weights in df_freq_pl directly
        df_ = (df_.join(df_freq_pl, on=['wd_bin','ws_bin'], how='left')
              .with_columns(pl.col('weight'))# .fill_null(strategy="zero"))
        )

        # Check if all the values in the weight column are null
        if df_['weight'].is_null().all():
            raise RuntimeError("None of the ws/wd bins in data appear in df_freq")
        
        # Check if any of the values in the weight column are null
        if df_['weight'].is_null().any():
            warnings.warn('Some bins in data are not in df_freq and will get 0 weight')

        # Fill the null values with zeros
        df_= df_.with_columns(pl.col('weight').fill_null(strategy="zero"))


    # Calculate energy ratios
    df_ = (df_
        .with_columns(
            [
                pl.col('pow_ref').mul(pl.col('weight')).alias('ref_energy'), # Compute the reference energy
                pl.col('pow_test').mul(pl.col('weight')).alias('test_energy'), # Compute the test energy
            ]
        )
        .group_by(['wd_bin','df_name'], maintain_order=True)
        .agg([pl.sum("ref_energy"), pl.sum("test_energy"),pl.sum("count")])
        .with_columns(
            energy_ratio = pl.col('test_energy') / pl.col('ref_energy')
        )
        .pivot(values=['energy_ratio','count'], columns='df_name', index='wd_bin',aggregate_function='first')
        .rename({f'energy_ratio_df_name_{n}' : n for n in df_names})
        .rename({f'count_df_name_{n}' : f'count_{n}'  for n in df_names})
        .sort('wd_bin')
    )

    # In the case of two turbines, compute an uplift column
    for upp, upn in zip(uplift_pairs, uplift_names):
        df_ = df_.with_columns(
            (100 * (pl.col(upp[1]) - pl.col(upp[0])) / pl.col(upp[0])).alias(upn)
        )

    # Enforce a column order
    df_ = df_.select(['wd_bin'] + df_names + uplift_names + [f'count_{n}' for n in df_names])

    return(df_)

# Bootstrap function wraps the _compute_energy_ratio function
def _compute_energy_ratio_bootstrap(er_in,
                         ref_cols,
                         test_cols,
                         wd_cols,
                         ws_cols,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         weight_by = 'min', #min, sum
                         df_freq_pl = None,
                         wd_bin_overlap_radius = 0.,
                         uplift_pairs = [],
                         uplift_names = [],
                         N = 1,
                         percentiles=[5., 95.],
                         remove_all_nulls=False,
                         ):
    
    """
    Compute the energy ratio between two sets of turbines with bootstrapping

    Args:
        er_in (EnergyRatioInput): An EnergyRatioInput object containing the data to use in the calculation.
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
        bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', or 'sum'.  'min' means
            the minimum count across the dataframes is used to weight the energy ratio. 'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.
        df_freq_pl (pl.Dataframe) Polars dataframe of pre-provided per bin weights
        wd_bin_overlap_radius (float): The distance in degrees one wd bin overlaps into the next, must be 
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
                available. If False, a minimum one data point from ref_cols, test_cols, wd_cols, and ws_cols
                must be available to compute the bin. Defaults to False.


    Returns:
        pl.DataFrame: A dataframe containing the energy ratio between the two sets of turbines.

    """

    # Otherwise run the function N times and concatenate the results to compute statistics

    df_concat = pl.concat([_compute_energy_ratio_single(er_in.resample_energy_table(i),
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
                        remove_all_nulls
                        ) for i in range(N)])

    bound_names = er_in.df_names + uplift_names

    return (df_concat
            .group_by(['wd_bin'], maintain_order=True)
            .agg([pl.first(n) for n in bound_names] + 
                    [pl.quantile(n, percentiles[0]/100).alias(n + "_ub") for n in bound_names] +
                    [pl.quantile(n, percentiles[1]/100).alias(n + "_lb") for n in bound_names] + 
                    [pl.first(f'count_{n}') for n in er_in.df_names]
                )
            .sort('wd_bin')
            )

def compute_energy_ratio(er_in: EnergyRatioInput,
                         ref_turbines = None,
                         test_turbines = None,
                         wd_turbines = None,
                         ws_turbines = None,
                         use_predefined_ref = False,
                         use_predefined_wd = False,
                         use_predefined_ws = False,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         weight_by = 'min', #min or sum
                         df_freq = None,
                         wd_bin_overlap_radius = 0.,
                         uplift_pairs = None,
                         uplift_names = None,
                         N = 1,
                         percentiles = None,
                         remove_all_nulls = False
                         )-> EnergyRatioOutput:
    
    """
    Compute the energy ratio between two sets of turbines with bootstrapping

    Args:
        er_in (EnergyRatioInput): An EnergyRatioInput object containing the data to use in the calculation.
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
        bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.
        weight_by (str): How to weight the energy ratio, options are 'min', , or 'sum'.  'min' means
            the minimum count across the dataframes is used to weight the energy ratio.   'sum' means the sum of the counts
            across the dataframes is used to weight the energy ratio.
        df_freq (pd.Dataframe): A dataframe which specifies the frequency of the ws/wd bin combinations.  Provides
            a method to use an explicit or long-term weigthing of bins.  Dataframe should include
            columns ws, wd and freq_val.  ws and wd should correspond to the bin centers resulting from
            the choices of the ws/wd_min / _max / _step.  In the case that df_freq has extra bins that aren't included 
            in those given by ws/wd min, max, step, they will be ignored in the energy ratio calculation. 
            Any bins given by ws/wd min, max, step not present in df_freq will be assigned a frequency of zero. 
            Defaults to None.
        wd_bin_overlap_radius (float): The distance in degrees one wd bin overlaps into the next, must be 
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
                available. If False, a minimum one data point from ref_cols, test_cols, wd_cols, and ws_cols
                must be available to compute the bin. Defaults to False.

    Returns:
        EnergyRatioOutput: An EnergyRatioOutput object containing the energy ratio between the two sets of turbines.

    """

    # Get the polars dataframe from within the er_in
    df_ = er_in.get_df()

    # Check that inputs are valid
    check_compute_energy_ratio_inputs(
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
        N,
        percentiles,
        remove_all_nulls
    )
    
     # Set up the column names for the reference and test power
    if not use_predefined_ref:
        ref_cols = [f'pow_{i:03d}' for i in ref_turbines]
    else:
        ref_cols = ['pow_ref']

    if not use_predefined_ws:
        ws_cols = [f'ws_{i:03d}' for i in ws_turbines]
    else:
        ws_cols = ['ws']

    if not use_predefined_wd:
        wd_cols = [f'wd_{i:03d}' for i in wd_turbines]
    else:
        wd_cols = ['wd']

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
        uplift_names = ["uplift_"+up[1]+"/"+up[0] for up in uplift_pairs]

    # Convert the numbered arrays to appropriate column names
    test_cols = [f'pow_{i:03d}' for i in test_turbines]

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
        df_freq_pl = df_freq_pl.rename({
            'ws':'ws_bin',
            'wd':'wd_bin',
            'freq_val':'weight'
        })

    else:
        df_freq_pl = None

    # If N=1, don't use bootstrapping
    if N == 1:
        if percentiles is not None:
            print("percentiles can only be used with bootstrapping (N > 1).")
        # Compute the energy ratio
        df_res = _compute_energy_ratio_single(df_,
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
                        remove_all_nulls
                    )
    else:
        if percentiles is None:
            percentiles = [5, 95]
        elif not hasattr(percentiles, "__len__") or len(percentiles) != 2:
            raise ValueError("percentiles should be a two element list of the "+\
                "upper and lower desired percentiles.")

        df_res = _compute_energy_ratio_bootstrap(er_in,
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
                            percentiles
                        )
    
    # Sort df_res by df_names, ws, wd

    # Return the results as an EnergyRatioOutput object
    return EnergyRatioOutput(df_res.to_pandas(), 
                                er_in,
                                ref_cols, 
                                test_cols, 
                                wd_cols,
                                ws_cols,
                                uplift_names,
                                wd_step,
                                wd_min,
                                wd_max,
                                ws_step,
                                ws_min,
                                ws_max,
                                bin_cols_in,
                                weight_by,
                                wd_bin_overlap_radius,
                                N)





# # Use method of Eric Simley's slide 2
# def _compute_uplift_in_region_single(df_,
#                          df_names,
#                          ref_cols,
#                          test_cols,
#                          wd_cols,
#                          ws_cols,
#                          wd_step = 2.0,
#                          wd_min = 0.0,
#                          wd_max = 360.0,
#                          ws_step = 1.0,
#                          ws_min = 0.0,
#                          ws_max = 50.0,
#                          bin_cols_in = ['wd_bin','ws_bin']
#                          ):
    
#     """
#     Compute the energy  uplift between two dataframes using method of Eric Simley's slide 2
#     Args:
#         df_ (pl.DataFrame): A dataframe containing the data to use in the calculation.
#         df_names (list): A list of names to give to the dataframes. 
#         ref_cols (list[str]): A list of columns to use as the reference turbines
#         test_cols (list[str]): A list of columns to use as the test turbines
#         wd_cols (list[str]): A list of columns to derive the wind directions from
#         ws_cols (list[str]): A list of columns to derive the wind speeds from
#         wd_step (float): The width of the wind direction bins.
#         wd_min (float): The minimum wind direction to use.
#         wd_max (float): The maximum wind direction to use.
#         ws_step (float): The width of the wind speed bins.
#         ws_min (float): The minimum wind speed to use.
#         ws_max (float): The maximum wind speed to use.
#         bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.

#     Returns:
#         pl.DataFrame: A dataframe containing the energy uplift
#     """

#     # Filter df_ that all the columns are not null
#     df_ = df_.filter(pl.all_horizontal(pl.col(ref_cols + test_cols + ws_cols + wd_cols).is_not_null()))

#     # Assign the wd/ws bins
#     df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max)
#     df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max)

#     # Assign the reference and test power columns
#     df_ = add_power_ref(df_, ref_cols)
#     df_ = add_power_test(df_, test_cols)

#     bin_cols_without_df_name = [c for c in bin_cols_in if c != 'df_name']
#     bin_cols_with_df_name = bin_cols_without_df_name + ['df_name']
    
#     df_ = (df_.with_columns(
#             power_ratio = pl.col('pow_test') / pl.col('pow_ref'))
#         .filter(pl.all_horizontal(pl.col(bin_cols_with_df_name).is_not_null())) # Select for all bin cols present
#         .group_by(bin_cols_with_df_name, maintain_order=True)
#         .agg([pl.mean("pow_ref"), pl.mean("power_ratio"),pl.count()]) 
#         .with_columns(
#             [
#                 pl.col('count').min().over(bin_cols_without_df_name).alias('count_min'), # Find the min across df_name
#                 pl.col('pow_ref').mul(pl.col('power_ratio')).alias('pow_test'), # Compute the test power
#             ]
#         )

#         .pivot(values=['power_ratio','pow_test','pow_ref','count_min'], columns='df_name', index=['wd_bin','ws_bin'],aggregate_function='first')
#         .drop_nulls()
#         .with_columns(
#             f_norm = pl.col(f'count_min_df_name_{df_names[0]}') / pl.col(f'count_min_df_name_{df_names[0]}').sum()
#         )
#         .with_columns(
#             delta_power_ratio = pl.col(f'power_ratio_df_name_{df_names[1]}') - pl.col(f'power_ratio_df_name_{df_names[0]}'),
#             pow_ref_both_cases = pl.concat_list([f'pow_ref_df_name_{n}' for n in df_names]).list.mean() 
#         )
#         .with_columns(
#             delta_energy = pl.col('delta_power_ratio') * pl.col('f_norm') * pl.col('pow_ref_both_cases'), # pl.col(f'pow_ref_df_name_{df_names[0]}'),
#             base_test_energy = pl.col(f'pow_test_df_name_{df_names[0]}') * pl.col('f_norm')
#         )

#     )

#     return pl.DataFrame({'delta_energy':8760 * df_['delta_energy'].sum(),
#                             'base_test_energy':8760 * df_['base_test_energy'].sum(),
#                             'uplift':100 * df_['delta_energy'].sum() / df_['base_test_energy'].sum()})
                            

# def _compute_uplift_in_region_bootstrap(df_,
#                          df_names,
#                          ref_cols,
#                          test_cols,
#                          wd_cols,
#                          ws_cols,
#                          wd_step = 2.0,
#                          wd_min = 0.0,
#                          wd_max = 360.0,
#                          ws_step = 1.0,
#                          ws_min = 0.0,
#                          ws_max = 50.0,
#                          bin_cols_in = ['wd_bin','ws_bin'],
#                          N = 20,
#                          ):
    
#     """
#     Compute the uplift in a region using bootstrap resampling

#     Args:
#         df_ (pl.DataFrame): A dataframe containing the data to use in the calculation.
#         df_names (list): A list of names to give to the dataframes. 
#         ref_cols (list[str]): A list of columns to use as the reference turbines
#         test_cols (list[str]): A list of columns to use as the test turbines
#         wd_cols (list[str]): A list of columns to derive the wind directions from
#         ws_cols (list[str]): A list of columns to derive the wind speeds from
#         ws_step (float): The width of the wind speed bins.
#         ws_min (float): The minimum wind speed to use.
#         ws_max (float): The maximum wind speed to use.
#         wd_step (float): The width of the wind direction bins.
#         wd_min (float): The minimum wind direction to use.
#         wd_max (float): The maximum wind direction to use.
#         bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.
#         N (int): The number of bootstrap samples to use.

#     Returns:
#         pl.DataFrame: A dataframe containing the energy uplift
#     """
    
#     df_concat = pl.concat([_compute_uplift_in_region_single(resample_energy_table(df_, i),
#                           df_names,
#                          ref_cols,
#                          test_cols,
#                          wd_cols,
#                          ws_cols,
#                          wd_step,
#                          wd_min,
#                          wd_max,
#                          ws_step,
#                          ws_min,
#                          ws_max,
#                          bin_cols_in,
#                          ) for i in range(N)])
    
#     return pl.DataFrame({
#         'delta_energy_exp':df_concat['delta_energy'][0],
#         'delta_energy_ub':df_concat['delta_energy'].quantile(0.95),
#         'delta_energy_lb':df_concat['delta_energy'].quantile(0.05),
#         'base_test_energy_exp':df_concat['base_test_energy'][0],
#         'base_test_energy_ub':df_concat['base_test_energy'].quantile(0.95),
#         'base_test_energy_lb':df_concat['base_test_energy'].quantile(0.05),
#         'uplift_exp':df_concat['uplift'][0],
#         'uplift_ub':df_concat['uplift'].quantile(0.95),
#         'uplift_lb':df_concat['uplift'].quantile(0.05),
#     })


# def compute_uplift_in_region(df_,
#                          df_names,
#                          ref_turbines=None,
#                          test_turbines= None,
#                          wd_turbines=None,
#                          ws_turbines=None,
#                          use_predefined_ref = False,
#                          use_predefined_wd = False,
#                          use_predefined_ws = False,
#                          wd_step = 2.0,
#                          wd_min = 0.0,
#                          wd_max = 360.0,
#                          ws_step = 1.0,
#                          ws_min = 0.0,
#                          ws_max = 50.0,
#                          bin_cols_in = ['wd_bin','ws_bin'],
#                          N = 1,
#                          ):
    
#     """
#     Compute the energy ratio between two sets of turbines with bootstrapping

#     Args:
#         df_ (pl.DataFrame): A dataframe containing the data to use in the calculation.
#         df_names (list): A list of names to give to the dataframes. 
#         ref_turbines (list[int]): A list of turbine numbers to use as the reference.
#         test_turbines (list[int]): A list of turbine numbers to use as the test.
#         ws_turbines (list[int]): A list of turbine numbers to use for the wind speeds
#         wd_turbines (list[int]): A list of turbine numbers to use for the wind directions
#         use_predefined_ref (bool): If True, use the pow_ref column of df_ as the reference power.
#         use_predefined_ws (bool): If True, use the ws column of df_ as the wind speed.
#         use_predefined_wd (bool): If True, use the wd column of df_ as the wind direction.
#         wd_step (float): The width of the wind direction bins.
#         wd_min (float): The minimum wind direction to use.
#         wd_max (float): The maximum wind direction to use.
#         ws_step (float): The width of the wind speed bins.
#         ws_min (float): The minimum wind speed to use.
#         ws_max (float): The maximum wind speed to use.
#         bin_cols_in (list[str]): A list of column names to use for the wind speed and wind direction bins.
#         N (int): The number of bootstrap samples to use.

#     Returns:
#         pl.DataFrame: A dataframe containing the energy ratio between the two sets of turbines.

#     """

#     # Check if inputs are valid
#     # If use_predefined_ref is True, df_ must have a column named 'pow_ref'
#     if use_predefined_ref:
#         if 'pow_ref' not in df_.columns:
#             raise ValueError('df_ must have a column named pow_ref when use_predefined_ref is True')
#         # If ref_turbines supplied, warn user that it will be ignored
#         if ref_turbines is not None:
#             warnings.warn('ref_turbines will be ignored when use_predefined_ref is True')
#     else:
#         # ref_turbine must be supplied
#         if ref_turbines is None:
#             raise ValueError('ref_turbines must be supplied when use_predefined_ref is False')
        
#     # If use_predefined_ws is True, df_ must have a column named 'ws'
#     if use_predefined_ws:
#         if 'ws' not in df_.columns:
#             raise ValueError('df_ must have a column named ws when use_predefined_ws is True')
#         # If ws_turbines supplied, warn user that it will be ignored
#         if ws_turbines is not None:
#             warnings.warn('ws_turbines will be ignored when use_predefined_ws is True')
#     else:
#         # ws_turbine must be supplied
#         if ws_turbines is None:
#             raise ValueError('ws_turbines must be supplied when use_predefined_ws is False')

#     # If use_predefined_wd is True, df_ must have a column named 'wd'
#     if use_predefined_wd:
#         if 'wd' not in df_.columns:
#             raise ValueError('df_ must have a column named wd when use_predefined_wd is True')
#         # If wd_turbines supplied, warn user that it will be ignored
#         if wd_turbines is not None:
#             warnings.warn('wd_turbines will be ignored when use_predefined_wd is True')
#     else:
#         # wd_turbine must be supplied
#         if wd_turbines is None:
#             raise ValueError('wd_turbines must be supplied when use_predefined_wd is False')
        
#     # Confirm that test_turbines is a list of ints or a numpy array of ints
#     if not isinstance(test_turbines, list) and not isinstance(test_turbines, np.ndarray):
#         raise ValueError('test_turbines must be a list or numpy array of ints')

#     # Confirm that test_turbines is not empty  
#     if len(test_turbines) == 0:
#         raise ValueError('test_turbines cannot be empty')

#     num_df = len(df_names)

#     # Confirm num_df == 2
#     if num_df != 2:
#         raise ValueError('Number of dataframes must be 2')

#     # Set up the column names for the reference and test power
#     if not use_predefined_ref:
#         ref_cols = [f'pow_{i:03d}' for i in ref_turbines]
#     else:
#         ref_cols = ['pow_ref']

#     if not use_predefined_ws:
#         ws_cols = [f'ws_{i:03d}' for i in ws_turbines]
#     else:
#         ws_cols = ['ws']

#     if not use_predefined_wd:
#         wd_cols = [f'wd_{i:03d}' for i in wd_turbines]
#     else:
#         wd_cols = ['wd']

#     # Convert the numbered arrays to appropriate column names
#     test_cols = [f'pow_{i:03d}' for i in test_turbines]

#     # If N=1, don't use bootstrapping
#     if N == 1:
#         # Compute the energy ratio
#         df_res = _compute_uplift_in_region_single(df_,
#                         df_names,
#                         ref_cols,
#                         test_cols,
#                         wd_cols,
#                         ws_cols,
#                         wd_step,
#                         wd_min,
#                         wd_max,
#                         ws_step,
#                         ws_min,
#                         ws_max,
#                         bin_cols_in)
#     else:
#         df_res = _compute_uplift_in_region_bootstrap(df_,
#                             df_names,
#                             ref_cols,
#                             test_cols,
#                             wd_cols,
#                             ws_cols,
#                             wd_step,
#                             wd_min,
#                             wd_max,
#                             ws_step,
#                             ws_min,
#                             ws_max,
#                             bin_cols_in,
#                             N)

#     # Return the results as an EnergyRatioResult object
#     return EnergyRatioResult(df_res, 
#                                 df_names,
#                                 df_,
#                                 ref_cols, 
#                                 test_cols, 
#                                 wd_cols,
#                                 ws_cols,
#                                 wd_step,
#                                 wd_min,
#                                 wd_max,
#                                 ws_step,
#                                 ws_min,
#                                 ws_max,
#                                 bin_cols_in,
#                                 N)


