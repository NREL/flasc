import polars as pl
import numpy as np

from typing import Union, List, Optional



def cut(col_name: str,
        edges: Union[np.ndarray, list],
    ) -> pl.Expr:
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


def bin_column(df_: pl.DataFrame,
               col_name: str,
               bin_col_name: str, 
               edges: Union[np.ndarray, list],
    ) -> pl.DataFrame:
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

def add_ws(df_: pl.DataFrame,
           ws_cols: List[str],
    ) -> pl.DataFrame:
    """
    Add the ws column to a dataframe, given which columns to average over
    

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    ws_cols (list(str)): The name of the columns to average across.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional ws column
    """


    df_with_mean_ws =  (
        # df_.select(pl.exclude('ws_bin')) # In case ws_bin already exists
        df_.with_columns(
            # df_.select(ws_cols).mean(axis=1).alias('ws_bin')
            ws = pl.concat_list(ws_cols).list.mean() # Initially ws_bin is just the mean
        )
        .filter(
            pl.all_horizontal(pl.col(ws_cols).is_not_null()) # Select for all bin cols present
        ) 

        .filter(
            (pl.col('ws').is_not_null())
        )
    )

    return df_with_mean_ws
    
def add_ws_bin(df_: pl.DataFrame,
               ws_cols: List[str],
               ws_step: float=1.0, 
               ws_min: float=-0.5,
               ws_max: float=50.0, 
               edges: Optional[Union[np.ndarray, list]]=None,
    ) -> pl.DataFrame:
    """
    Add the ws_bin column to a dataframe, given which columns to average over
    and the step sizes to use

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    ws_cols (list(str)): The name of the columns to average across.
    ws_step (float): Step size for binning
    ws_min (float): Minimum wind speed
    ws_max (float): Maximum wind speed
    edges (array-like): The edges of the bins. Values will be placed into the bin
                        whose left edge is the largest edge less than or equal to
                        the value, and whose right edge is the smallest edge
                        greater than the value.  Defaults to None, in which case
                        the edges are generated using ws_step, ws_min, and ws_max.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """

    if edges is None:
        edges = np.arange(ws_min, ws_max+ws_step,ws_step)
    
    # Check if edges is a list or numpy array or similar
    elif len(edges) < 2:
        raise ValueError("edges must have length of at least 2")

    df_with_mean_ws = add_ws(df_, ws_cols)

    # Filter to min and max
    df_with_mean_ws = df_with_mean_ws.filter(
        (pl.col('ws') >= ws_min) &  # Filter the mean wind speed
        (pl.col('ws') < ws_max)
    )

    return bin_column(df_with_mean_ws, 'ws', 'ws_bin', edges)

def add_wd(df_:pl.DataFrame,
          wd_cols: List[str],
    ) -> pl.DataFrame:
    """
    Add the wd column to a dataframe, given which columns to average over
    

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    wd_cols (list(str)): The name of the columns to average across.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional wd column
    """

    
    # Gather up intermediate column names and final column names
    wd_cols_cos = [c + '_cos' for c in wd_cols]
    wd_cols_sin = [c + '_sin' for c in wd_cols]
    cols_to_return = df_.columns
    if 'wd' not in cols_to_return:
        cols_to_return = cols_to_return + ['wd']
    

    df_with_mean_wd =  (
        # df_.select(pl.exclude('wd_bin')) # In case wd_bin already exists
        df_.filter(
            pl.all_horizontal(pl.col(wd_cols).is_not_null()) # Select for all bin cols present
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
            # df_with_mean_wd.select(wd_cols_cos).mean(axis=1).alias('cos_mean'),
            # df_with_mean_wd.select(wd_cols_sin).mean(axis=1).alias('sin_mean'),
            pl.concat_list(wd_cols_cos).list.mean().alias('cos_mean'),
            pl.concat_list(wd_cols_sin).list.mean().alias('sin_mean'),
        ]
        )
        .with_columns(
            wd = np.mod(pl.reduce(np.arctan2, [pl.col('sin_mean'), pl.col('cos_mean')])
                            .mul(180/np.pi), 360.0)
        )
        .filter(
            (pl.col('wd').is_not_null())
        )
        .select(cols_to_return) # Select for just the columns we want to return
    )

    return df_with_mean_wd

# (df_, wd_cols, wd_step=2.0, wd_min=0.0, wd_max=360.0, edges=None):@# 
def add_wd_bin(df_: pl.DataFrame,
                wd_cols: List[str],
                wd_step: float=2.0,
                wd_min: float=0.0,
                wd_max: float=360.0,
                edges: Optional[Union[np.ndarray, list]]=None,
    ):
    """
    Add the wd_bin column to a dataframe, given which columns to average over
    and the step sizes to use

    Parameters:
    df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
    wd_cols (list(str)): The name of the columns to average across.
    wd_step (float): Step size for binning
    wd_min (float): Minimum wind direction
    wd_max (float): Maximum wind direction
    edges (array-like): The edges of the bins. Values will be placed into the bin
                    whose left edge is the largest edge less than or equal to
                    the value, and whose right edge is the smallest edge
                    greater than the value.  Defaults to None, in which case
                    the edges are generated using ws_step, ws_min, and ws_max.

    Returns:
    pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """

    if edges is None:
        edges = np.arange(wd_min, wd_max + wd_step, wd_step)
    
    # If not none, edges must have lenght of at least 2
    elif len(edges) < 2:
        raise ValueError("edges must have length of at least 2")
    
    

    # Add in the mean wd column
    df_with_mean_wd = add_wd(df_, wd_cols)

    # Filter to min and max
    df_with_mean_wd = df_with_mean_wd.filter(
        (pl.col('wd') >= wd_min) &  # Filter the mean wind speed
        (pl.col('wd') < wd_max)
    )

    return bin_column(df_with_mean_wd, 'wd', 'wd_bin', edges)


def add_power_test(df_: pl.DataFrame,
                     test_cols: List[str],
    ) -> pl.DataFrame:

    return df_.with_columns(
        pow_test = pl.concat_list(test_cols).list.mean()
    )


def add_power_ref(df_: pl.DataFrame,
                   ref_cols: List[str]):

    return df_.with_columns(
        pow_ref = pl.concat_list(ref_cols).list.mean()
    )


def add_reflected_rows(df_: pl.DataFrame,
                       edges: Union[np.ndarray, list], 
                       overlap_distance: float):
    """
    Adds rows to a datrame with where the wind direction is reflected around the neearest edge if within overlap_distance

    Given a wind direction DataFrame `df_`, this function adds reflected rows to the DataFrame such that each wind direction
    in the original DataFrame has a corresponding reflected wind direction. The reflected wind direction is calculated by
    subtracting the wind direction from the nearest edge in `edges` and then subtracting that difference again from the
    original wind direction. The resulting wind direction is then wrapped around to the range [0, 360) degrees. The function
    returns a new DataFrame with the original rows and the added reflected rows.

    This function enables overlapping bins in the energy ratio functions

    Parameters
    ----------
    df_ : polars.DataFrame
        The DataFrame to add reflected rows to.
    edges : numpy.ndarray
        An array of wind direction edges to use for reflection.  (Should be same as used in energy ratio)
    overlap_distance : float
        The maximum distance between a wind direction and an edge for the wind direction to be considered overlapping.

    Returns
    -------
    polars.DataFrame
        A new DataFrame with the original rows and the added reflected rows.
    """

    df_add = df_.clone()
    wd = df_add['wd'].to_numpy()
    diff_matrix = wd[:,None] - edges
    abs_diff_matrix = np.abs(diff_matrix)
    idx = np.argmin(abs_diff_matrix, axis=1)
    signed_mins = diff_matrix[np.arange(len(diff_matrix)), idx]
    df_add = (df_add.with_columns(pl.Series(name='distances',values=signed_mins,dtype=pl.Float32))
        .filter(pl.col('distances').abs() < overlap_distance)
        .with_columns(np.mod((pl.col('wd') - pl.col('distances') * 2),360.0))
        .drop('distances')
        )
    

    return pl.concat([df_, df_add])