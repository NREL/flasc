"""Energy ratio input module."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import polars as pl

from flasc import FlascDataFrame
from flasc.data_processing.dataframe_manipulations import df_reduce_precision


def generate_block_list(N: int, num_blocks: int = 10):
    """Generate an np.array of length N where each element is an integer between 0 and num_blocks-1.

    Generate an np.array of length N where each element is an integer between 0 and num_blocks-1
    with each value repeating N/num_blocks times.

    Args:
        N (int): Length of the array to generate
        num_blocks (int): Number of blocks to generate. Defaults to 10.

    Returns:
        np.array: An array of length N with values between 0 and num_blocks-1

    """
    # Test than N and num_blocks are integers greater than 0
    if not isinstance(N, int) or not isinstance(num_blocks, int):
        raise ValueError("N and num_blocks must be integers")
    if N <= 0 or num_blocks <= 0:
        raise ValueError("N and num_blocks must be greater than 0")

    # Num blocks must be less than or equal to N
    if num_blocks > N:
        raise ValueError("num_blocks must be less than or equal to N")

    block_list = np.zeros(N)
    for i in range(num_blocks):
        block_list[i * N // num_blocks : (i + 1) * N // num_blocks] = i
    return block_list.astype(int)


class AnalysisInput:
    """AnalysisInput class.

    This class holds the structured inputs for calculating energy ratios
    """

    def __init__(
        self,
        df_list_in: List[pd.DataFrame | FlascDataFrame],
        df_names: List[str],
        num_blocks: int = 10,
        schema_overrides: dict = None,
    ) -> None:
        """Initialize the AnalysisInput class.

        Args:
            df_list_in (List[pd.DataFrame | FlascDataFrame]): A list of pandas dataframes
                or FlascDataFrames to be used in analysis
            df_names (List[str]): A list of names for the dataframes
            num_blocks (int): The number of blocks to use for the energy ratio calculation.
                Defaults to 10.
            schema_overrides (dict): A dictionary of schema overrides to use when converting
                the dataframes to polars. Defaults to None.
        """
        # Reduce precision if needed and convert to polars
        df_list = [
            pl.from_pandas(
                df_reduce_precision(df, allow_convert_to_integer=False),
                schema_overrides=schema_overrides,
            )
            for df in df_list_in
        ]

        # Get minimal set of columns for the dataframes; drop the rest
        keep_columns = df_list[0].columns
        for df in df_list:
            keep_columns = [c for c in df.columns if c in keep_columns]
        df_list = [df.select(keep_columns) for df in df_list]

        # If df_names not provided, give simple numbered names
        if df_names is None:
            df_names = ["df_" + str(i) for i in range(len(df_list))]

        # Add a name column to each dataframe
        for i in range(len(df_list)):
            df_list[i] = df_list[i].with_columns([pl.lit(df_names[i]).alias("df_name")])

        # Add a block column to each dataframe
        for i in range(len(df_list)):
            df_list[i] = df_list[i].with_columns(
                [
                    pl.Series(
                        generate_block_list(df_list[i].shape[0], num_blocks=num_blocks)
                    ).alias("block")
                ]
            )

        # Store the results
        self.df_pl = pl.concat(df_list, rechunk=True)
        self.df_names = df_names
        self.num_blocks = num_blocks

    def get_df(self) -> pl.DataFrame:
        """Get the concatenated dataframe.

        Returns:
            pl.DataFrame: The concatenated dataframe
        """
        return self.df_pl.clone()

    def resample_energy_table(self, perform_resample: bool = True) -> pl.DataFrame:
        """Use the block column of an energy table to resample the data.

        Args:
            perform_resample: Boolean, if False returns original energy table. Defaults to True.

        Returns:
            pl.DataFrame: A new energy table with (approximately)
                the same number of rows as the original
        """
        if perform_resample:
            # Generate a random np.array, num_blocks long, where each element is
            #  an integer between 0 and num_blocks-1
            block_list = np.random.randint(0, self.num_blocks, self.num_blocks)

            return pl.DataFrame({"block": block_list}).join(self.df_pl, how="inner", on="block")
        else:
            return self.get_df()
