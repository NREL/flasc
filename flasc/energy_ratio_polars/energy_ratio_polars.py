# This is a work in progress as we try to synthesize ideas from the 
# table based methods and energy ratios back into one thing, 
# some ideas we're incorporating:

# Conversion from polars to pandas
# Constructing tables (but now including tables of ratios)
# Keeping track of frequencies is matching sized tables

import numpy as np
import polars as pl

# def get_mid_bins(bin_edges):
#     """_summary_

#     Args:
#         bin_edges (NDArray): a set of bin edges
#     """

#     print(bin_edges[:-1] + np.diff(bin_edges)/2.0)

def cut(col_name, edges):
    """
    Bins the values in the specified column according to the given edges.

    Parameters:
    col_name (str): The name of the column to bin.
    edges (array-like): The edges of the bins. Values will be placed into the bin
                        whose left edge is the largest edge less than or equal to
                        the value, and whose right edge is the smallest edge
                        greater than the value.

    Returns:
    expression: An expression object that can be used to bin the column.
    """
    c = pl.col(col_name)
    labels = edges[:-1] + np.diff(edges)/2.0
    expr = pl.when(c < edges[0]).then(None)
    for edge, label in zip(edges[1:], labels):
        expr = expr.when(c < edge).then(label)
    expr = expr.otherwise(None)

    return expr


def bin_column(df_, col_name, bin_col_name, edges):
    """
    Bins the values in the specified column of a Polars DataFrame according to the given edges.

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    col_name (str): The name of the column to bin.
    bin_col_name (str): The name to give the new column containing the bin labels.
    edges (array-like): The edges of the bins. Values will be placed into the bin
                        whose left edge is the largest edge less than or equal to
                        the value, and whose right edge is the smallest edge
                        greater than the value.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional column containing the bin labels.
    """
    return df_.with_columns(
        cut(
            col_name=col_name,
            edges = edges
        ).alias(bin_col_name)
    )
    
def add_ws_bin(df_, ws_cols, ws_step=1.0, ws_min=-0.5, ws_max=50.0):
    """
    Add the ws_bin column to a dataframe, given which columns to average over
    and the step sizes to use

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    ws_cols (str): The name of the columns to average across.
    ws_step (float): Step size for binning
    ws_min (float): Minimum wind speed
    ws_max (float): Maximum wind speed

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """

    edges = np.arange(ws_min, ws_max+ws_step,ws_step)

    df_with_mean_ws =  (
        # df_.select(pl.exclude('ws_bin')) # In case ws_bin already exists
        df_.with_columns(
            df_.select(ws_cols).mean(axis=1).alias('ws_bin')
            # ws_bin = pl.concat_list(ws_cols).arr.mean() # Initially ws_bin is just the mean
        )
        .filter(
            pl.all(pl.col(ws_cols).is_not_null()) # Select for all bin cols present
        ) 

        .filter(
            (pl.col('ws_bin') > ws_min) &  # Filter the mean wind speed
            (pl.col('ws_bin') < ws_max) &
            (pl.col('ws_bin').is_not_null())
        )
    )

    return bin_column(df_with_mean_ws, 'ws_bin', 'ws_bin', edges)

def add_wd_bin(df_, wd_cols, wd_step=2.0, wd_min=0.0, wd_max=360.0):
    """
    Add the wd_bin column to a dataframe, given which columns to average over
    and the step sizes to use

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    wd_cols (str): The name of the columns to average across.
    wd_step (float): Step size for binning
    wd_min (float): Minimum wind direction
    wd_max (float): Maximum wind direction

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """

    edges = np.arange(wd_min, wd_max + wd_step, wd_step)
    
    # Gather up intermediate column names and final column names
    wd_cols_cos = [c + '_cos' for c in wd_cols]
    wd_cols_sin = [c + '_sin' for c in wd_cols]
    cols_to_return = df_.columns
    if 'wd_bin' not in cols_to_return:
        cols_to_return = cols_to_return + ['wd_bin']
    

    df_with_mean_wd =  (
        # df_.select(pl.exclude('wd_bin')) # In case wd_bin already exists
        df_.filter(
            pl.all(pl.col(wd_cols).is_not_null()) # Select for all bin cols present
        ) 
        # Add the cosine columns
        .with_columns(
        [
            pl.col(wd_cols).mul(np.pi/180).cos().suffix('_cos'),
            pl.col(wd_cols).mul(np.pi/180).sin().suffix('_sin'),
        ]
        )
    )
    df_with_mean_wd = (
        df_with_mean_wd
        .with_columns(
        [
            df_with_mean_wd.select(wd_cols_cos).mean(axis=1).alias('cos_mean'),
            df_with_mean_wd.select(wd_cols_sin).mean(axis=1).alias('sin_mean'),
            # pl.concat_list(wd_cols_cos).arr.mean().alias('cos_mean'),
            # pl.concat_list(wd_cols_sin).arr.mean().alias('sin_mean'),
        ]
        )
        .with_columns(
            wd_bin = np.mod(pl.reduce(np.arctan2, [pl.col('sin_mean'), pl.col('cos_mean')])
                            .mul(180/np.pi), 360.0)
        )
        .filter(
            (pl.col('wd_bin') > wd_min) &  # Filter the mean wind speed
            (pl.col('wd_bin') < wd_max) &
            (pl.col('wd_bin').is_not_null())
        )
        .select(cols_to_return) # Select for just the columns we want to return
    )

    return bin_column(df_with_mean_wd, 'wd_bin', 'wd_bin', edges)

def generate_block_list(N, num_blocks=10):
    """Generate an np.array of length N where each element is an integer between 0 and num_blocks-1
    with each value repeating N/num_blocks times.

    Args:
        N (int): Length of the array to generate
        num_blocks (int): Number of blocks to generate
        
    """

    # Test than N and num_blocks are integers greater than 0
    if not isinstance(N, int) or not isinstance(num_blocks, int):
        raise ValueError('N and num_blocks must be integers')
    if N <= 0 or num_blocks <= 0:
        raise ValueError('N and num_blocks must be greater than 0')
    
    # Num blocks must be less than or equal to N
    if num_blocks > N:
        raise ValueError('num_blocks must be less than or equal to N')


    block_list = np.zeros(N)
    for i in range(num_blocks):
        block_list[i*N//num_blocks:(i+1)*N//num_blocks] = i
    return block_list.astype(int)

def get_energy_table(
        df_list, 
        df_names=None,
        num_blocks=10,):
    """
    Given a list of dataframes, return a single dataframe with a column
    indicating which dataframe the row came from as well as a block
    list to use in bootstrapping.

    Parameters:
    df_list (list): A list of dataframes to combine.
    df_names (list): A list of names to give to the dataframes. If None,
                        the dataframes will be named df_0, df_1, etc.
    n_blocks (int): The number of blocks to add to the block column for later bootstrapping.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional column containing the df_names
    """

    
    if df_names is None:
        df_names = ['df_'+str(i) for i in range(len(df_list))]

    # Add a name column to each dataframe
    for i in range(len(df_list)):
        df_list[i] = df_list[i].with_columns([
            pl.lit(df_names[i]).alias('df_name')
        ])

    # Add a block column to each dataframe
    for i in range(len(df_list)):
        df_list[i] = df_list[i].with_columns([
            pl.Series(generate_block_list(df_list[i].shape[0], num_blocks=num_blocks)).alias('block')
        ])

    return pl.concat(df_list)

def resample_energy_table(df_e_, i):
    """Use the block column of an energy table to resample the data.

    Args:
        df_e_ (pl.DataFrame): An energy table with a block column

    Returns:
        pl.DataFrame: A new energy table with (approximately)
            the same number of rows as the original
    """

    if i == 0: #code to return as is
        return df_e_
    
    else:

        num_blocks = df_e_['block'].max() + 1
        
        # Generate a random np.array, num_blocks long, where each element is
        #  an integer between 0 and num_blocks-1
        block_list = np.random.randint(0, num_blocks, num_blocks)
        
        return pl.DataFrame(
            {
                'block':block_list
            }
            ).join(df_e_, how='inner', on='block')


def add_test_power(df_, test_cols):

    return df_.with_columns(
        df_.select(test_cols).mean(axis=1).alias('test_pow')
    )


def add_ref_power(df_, ref_cols):

    return df_.with_columns(
        df_.select(ref_cols).mean(axis=1).alias('ref_pow')
    )



def compute_energy_ratio(df_,
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines=None,
                         wd_turbines=None,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         ):


    num_df = len(df_names)

    # if ws_turbines is not supplied, assume equivalent to ref_turbines
    if ws_turbines is None:
        ws_turbines = ref_turbines

    # if wd_turbines is not supplied, assume equivalent to ref_turbines
    if wd_turbines is None:
        wd_turbines = ref_turbines

    # Convert the numbered arrays to appropriate column names
    ref_turbines = [f'pow_{i:03d}' for i in ref_turbines]
    test_turbines = [f'pow_{i:03d}' for i in test_turbines]
    ws_turbines = [f'ws_{i:03d}' for i in ws_turbines]
    wd_turbines = [f'wd_{i:03d}' for i in wd_turbines]

    # Filter df_ that all the columns are not null
    df_ = df_.filter(pl.all(pl.col(ref_turbines + test_turbines + ws_turbines + wd_turbines).is_not_null()))

    # Assign the wd/ws bins
    df_ = add_ws_bin(df_, ws_turbines, ws_step, ws_min, ws_max)
    df_ = add_wd_bin(df_, wd_turbines, wd_step, wd_min, wd_max)

    # Assign the reference and test power columns
    df_ = add_ref_power(df_, ref_turbines)
    df_ = add_test_power(df_, test_turbines)

    # # Compute the power ratio
    # df_ = df_.with_columns(
    #     power_ratio = pl.col('test_pow') / pl.col('ref_pow')
    # )

    bin_cols_without_df_name = [c for c in bin_cols_in if c != 'df_name']
    bin_cols_with_df_name = bin_cols_without_df_name + ['df_name']
    
    df_ = (df_
           #.with_columns(
            #    power_ratio = pl.col('test_pow') / pl.col(f'ref_pow'))
        .filter(pl.all(pl.col(bin_cols_with_df_name).is_not_null())) # Select for all bin cols present
        .groupby(bin_cols_with_df_name, maintain_order=True)
        .agg([pl.mean("ref_pow"), pl.mean("test_pow"),pl.count()]) 
        .with_columns(
            [
                pl.col('count').min().over(bin_cols_without_df_name).alias('count_min')#, # Find the min across df_name
                #pl.col('ref_pow').mul(pl.col('power_ratio')).alias('test_pow'), # Compute the test power
            ]
        )
        .with_columns(
            [
                pl.col('ref_pow').mul(pl.col('count_min')).alias('ref_energy'), # Compute the reference energy
                pl.col('test_pow').mul(pl.col('count_min')).alias('test_energy'), # Compute the test energy
            ]
        )
        .groupby(['wd_bin','df_name'], maintain_order=True)
        .agg([pl.sum("ref_energy"), pl.sum("test_energy"),pl.sum("count")])
        .with_columns(
            energy_ratio = pl.col('test_energy') / pl.col('ref_energy')
        )
        .pivot(values=['energy_ratio','count'], columns='df_name', index='wd_bin',aggregate_function='first')
        .rename({f'energy_ratio_df_name_{n}' : n for n in df_names})
        .rename({f'count_df_name_{n}' : f'count_{n}'  for n in df_names})
        .sort('wd_bin')
    )

    # This probably doesn't belong in here but for now
    if num_df == 2:
        df_ = df_.with_columns(
            uplift = 100 * (pl.col(df_names[1]) - pl.col(df_names[0])) / pl.col(df_names[0])
        )

        # Enforce a column order
        df_ = df_.select(['wd_bin'] + df_names + ['uplift'] + [f'count_{n}' for n in df_names])

    else:
        # Enforce a column order
        df_ = df_.select(['wd_bin'] + df_names +  [f'count_{n}' for n in df_names])

    

    return(df_)
        
def compute_energy_ratio_bootstrap(df_,
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines=None,
                         wd_turbines=None,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         N = 20,
                         ):
    
    df_concat = pl.concat([compute_energy_ratio(resample_energy_table(df_, i),
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines,
                         wd_turbines,
                         bin_cols_in,
                         ws_step,
                         ws_min,
                         ws_max,
                         wd_step,
                         wd_min,
                         wd_max,
                         ) for i in range(N)])
    
    if 'uplift' in df_concat.columns:
        df_names_with_uplift = df_names + ['uplift']
    else:
        df_names_with_uplift = df_names


    return(df_concat
           .groupby(['wd_bin'], maintain_order=True)
           .agg([pl.first(n) for n in df_names_with_uplift] + 
                [pl.quantile(n, 0.95).alias(n + "_ub") for n in df_names_with_uplift] +
                [pl.quantile(n, 0.05).alias(n + "_lb") for n in df_names_with_uplift] + 
                [pl.first(f'count_{n}') for n in df_names]
           )
           .sort('wd_bin')
    )

# Use method of Eric Simley's slide 2
def compute_uplift_in_region(df_,
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines=None,
                         wd_turbines=None,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         ):


    num_df = len(df_names)

    # Confirm num_df == 2
    if num_df != 2:
        raise ValueError('Number of dataframes must be 2')

    # if ws_turbines is not supplied, assume equivalent to ref_turbines
    if ws_turbines is None:
        ws_turbines = ref_turbines

    # if wd_turbines is not supplied, assume equivalent to ref_turbines
    if wd_turbines is None:
        wd_turbines = ref_turbines

    # Convert the numbered arrays to appropriate column names
    ref_turbines = [f'pow_{i:03d}' for i in ref_turbines]
    test_turbines = [f'pow_{i:03d}' for i in test_turbines]
    ws_turbines = [f'ws_{i:03d}' for i in ws_turbines]
    wd_turbines = [f'wd_{i:03d}' for i in wd_turbines]

    # Filter df_ that all the columns are not null
    df_ = df_.filter(pl.all(pl.col(ref_turbines + test_turbines + ws_turbines + wd_turbines).is_not_null()))

    # Assign the wd/ws bins
    df_ = add_ws_bin(df_, ws_turbines, ws_step, ws_min, ws_max)
    df_ = add_wd_bin(df_, wd_turbines, wd_step, wd_min, wd_max)

    # Assign the reference and test power columns
    df_ = add_ref_power(df_, ref_turbines)
    df_ = add_test_power(df_, test_turbines)

    bin_cols_without_df_name = [c for c in bin_cols_in if c != 'df_name']
    bin_cols_with_df_name = bin_cols_without_df_name + ['df_name']
    
    df_ = (df_.with_columns(
            power_ratio = pl.col('test_pow') / pl.col('ref_pow'))
        .filter(pl.all(pl.col(bin_cols_with_df_name).is_not_null())) # Select for all bin cols present
        .groupby(bin_cols_with_df_name, maintain_order=True)
        .agg([pl.mean("ref_pow"), pl.mean("power_ratio"),pl.count()]) 
        .with_columns(
            [
                pl.col('count').min().over(bin_cols_without_df_name).alias('count_min'), # Find the min across df_name
                pl.col('ref_pow').mul(pl.col('power_ratio')).alias('test_pow'), # Compute the test power
            ]
        )

        .pivot(values=['power_ratio','test_pow','ref_pow','count_min'], columns='df_name', index=['wd_bin','ws_bin'],aggregate_function='first')
        .drop_nulls()
        .with_columns(
            f_norm = pl.col(f'count_min_df_name_{df_names[0]}') / pl.col(f'count_min_df_name_{df_names[0]}').sum()
        )
        .with_columns(
            delta_power_ratio = pl.col(f'power_ratio_df_name_{df_names[1]}') - pl.col(f'power_ratio_df_name_{df_names[0]}'),
            ref_pow_both_cases = pl.concat_list([f'ref_pow_df_name_{n}' for n in df_names]).arr.mean() 
        )
        .with_columns(
            delta_energy = pl.col('delta_power_ratio') * pl.col('f_norm') * pl.col('ref_pow_both_cases'), # pl.col(f'ref_pow_df_name_{df_names[0]}'),
            base_test_energy = pl.col(f'test_pow_df_name_{df_names[0]}') * pl.col('f_norm')
        )

    )

    return pl.DataFrame({'delta_energy':8760 * df_['delta_energy'].sum(),
                            'base_test_energy':8760 * df_['base_test_energy'].sum(),
                            'uplift':100 * df_['delta_energy'].sum() / df_['base_test_energy'].sum()})
                            

def compute_uplift_in_region_bootstrap(df_,
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines=None,
                         wd_turbines=None,
                         bin_cols_in = ['wd_bin','ws_bin'],
                         ws_step = 1.0,
                         ws_min = 0.0,
                         ws_max = 50.0,
                         wd_step = 2.0,
                         wd_min = 0.0,
                         wd_max = 360.0,
                         N = 20,
                         ):
    
    df_concat = pl.concat([compute_uplift_in_region(resample_energy_table(df_, i),
                         ref_turbines,
                         test_turbines,
                         df_names,
                         ws_turbines,
                         wd_turbines,
                         bin_cols_in,
                         ws_step,
                         ws_min,
                         ws_max,
                         wd_step,
                         wd_min,
                         wd_max,
                         ) for i in range(N)])
    
    return pl.DataFrame({
        'delta_energy_exp':df_concat['delta_energy'][0],
        'delta_energy_ub':df_concat['delta_energy'].quantile(0.95),
        'delta_energy_lb':df_concat['delta_energy'].quantile(0.05),
        'base_test_energy_exp':df_concat['base_test_energy'][0],
        'base_test_energy_ub':df_concat['base_test_energy'].quantile(0.95),
        'base_test_energy_lb':df_concat['base_test_energy'].quantile(0.05),
        'uplift_exp':df_concat['uplift'][0],
        'uplift_ub':df_concat['uplift'].quantile(0.95),
        'uplift_lb':df_concat['uplift'].quantile(0.05),
    })