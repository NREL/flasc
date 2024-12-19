"""Expected power analysis by wind direction or wind speed."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.expected_power_analysis import total_uplift_expected_power
from flasc.analysis.expected_power_analysis_utilities import (
    _add_wd_ws_bins,
)


class _total_uplift_expected_power_by_:
    """Compute total uplift expected power by wind direction or wind speed."""

    def __init__(
        self,
        wd_or_ws: str,
        a_in: AnalysisInput,
        uplift_pairs: List[Tuple[str, str]],
        uplift_names: List[str],
        test_turbines: List[int],
        wd_turbines: List[int] = None,
        ws_turbines: List[int] = None,
        use_predefined_wd: bool = False,
        use_predefined_ws: bool = False,
        wd_step: float = 2.0,
        wd_min: float = 0.0,
        wd_max: float = 360.0,
        ws_step: float = 1.0,
        ws_min: float = 0.0,
        ws_max: float = 50.0,
        bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
        weight_by: str = "min",  # min or sum
        df_freq: pd.DataFrame = None,
        use_standard_error: bool = True,
        N: int = 1,
        percentiles: List[float] = [2.5, 97.5],
        remove_any_null_turbine_bins: bool = False,
        cov_terms: str = "zero",
    ):
        """Calculates total uplift expected power by wind direction.

        Args:
        wd_or_ws (str): The column name to bin the dataframes by.  Can be "wd" or "ws".
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        test_turbines (List[int]): A list of turbine indices to test
        wd_turbines (List[int]): A list of turbine indices for wind direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for wind speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin. Defaults to None.
        use_standard_error (bool): Use standard error for the uplift calculation. Defaults to True.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The percentiles to calculate for the bootstrap samples.
            Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        cov_terms (str): Use directly computed covariance terms, or fill with zeros or variances.
            Can be "zero", "var" or "cov".  If "zero" all covariance terms are set to zero.  if
            "var" all covariance terms are set to the product of the variances.  If "cov" the
            covariance terms are used as is with missing terms set to product of the
            variances.  Defaults to "zero".
        """
        # wd_or_ws must be either "wd" or "ws"
        if wd_or_ws not in ["wd", "ws"]:
            raise ValueError("wd_or_ws must be either 'wd' or 'ws'")

        # Get the polars dataframe from within the a_in, as well as the df_names and num_blocks
        df_ = a_in.get_df()
        df_names = a_in.df_names
        num_blocks = a_in.num_blocks

        # Set up the column names for the wind speed and wind direction cols
        if not use_predefined_ws:
            ws_cols = [f"ws_{i:03d}" for i in ws_turbines]
        else:
            ws_cols = ["ws"]

        if not use_predefined_wd:
            wd_cols = [f"wd_{i:03d}" for i in wd_turbines]
        else:
            wd_cols = ["wd"]

        # Add the wd and ws bins
        df_ = _add_wd_ws_bins(
            df_,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
        )

        # Save the schema to be sure exact datatypes reused when polars dataframe reconstructed
        schema = df_.schema.copy()

        # Make a single pandas version
        df_pandas = df_.to_pandas()

        # Convert df_ back into a list of pandas dataframes for subsetting
        df_list = []
        for df_name in df_names:
            df_list.append(df_pandas[df_pandas["df_name"] == df_name])

        # Get a sorted list of unique values of wd_bin within df_
        w_bins = df_[f"{wd_or_ws}_bin"].unique().sort().to_numpy()

        # Initialize the results arrays
        energy_uplift_ctr_pc = np.zeros((len(uplift_pairs), len(w_bins))) * np.nan
        energy_uplift_lb_pc = np.zeros((len(uplift_pairs), len(w_bins))) * np.nan
        energy_uplift_ub_pc = np.zeros((len(uplift_pairs), len(w_bins))) * np.nan

        # Loop over the wind (dir/speed) bins
        for w_idx, w_bin in enumerate(w_bins):
            # Make a subset a_in
            df_list_sub = [d[d[f"{wd_or_ws}_bin"] == w_bin] for d in df_list]

            # If the subset is empty, or has less than num_blocks rows, skip
            if any([len(d) < num_blocks for d in df_list_sub]):
                continue

            a_in_sub = AnalysisInput(df_list_sub, df_names, num_blocks, schema_overrides=schema)

            # Run the analysis
            try:
                epao = total_uplift_expected_power(
                    a_in=a_in_sub,
                    uplift_pairs=uplift_pairs,
                    uplift_names=uplift_names,
                    test_turbines=test_turbines,
                    wd_turbines=wd_turbines,
                    ws_turbines=ws_turbines,
                    use_predefined_wd=use_predefined_wd,
                    use_predefined_ws=use_predefined_ws,
                    wd_step=wd_step,
                    wd_min=wd_min,
                    wd_max=wd_max,
                    ws_step=ws_step,
                    ws_min=ws_min,
                    ws_max=ws_max,
                    bin_cols_in=bin_cols_in,
                    weight_by=weight_by,
                    df_freq=df_freq,
                    use_standard_error=use_standard_error,
                    N=N,
                    percentiles=percentiles,
                    remove_any_null_turbine_bins=remove_any_null_turbine_bins,
                    cov_terms=cov_terms,
                )

            # If a value error is raised simply continue
            except ValueError:
                continue

            # Loop over the uplift pairs
            for up_idx, uplift_name in enumerate(uplift_names):
                energy_uplift_ctr_pc[up_idx, w_idx] = epao.uplift_results[uplift_name][
                    "energy_uplift_ctr_pc"
                ]
                energy_uplift_lb_pc[up_idx, w_idx] = epao.uplift_results[uplift_name][
                    "energy_uplift_lb_pc"
                ]
                energy_uplift_ub_pc[up_idx, w_idx] = epao.uplift_results[uplift_name][
                    "energy_uplift_ub_pc"
                ]

        # Store the inputs
        self.wd_or_ws = wd_or_ws
        self.a_in = a_in
        self.uplift_pairs = uplift_pairs
        self.uplift_names = uplift_names
        self.test_turbines = test_turbines
        self.wd_turbines = wd_turbines
        self.ws_turbines = ws_turbines
        self.use_predefined_wd = use_predefined_wd
        self.use_predefined_ws = use_predefined_ws
        self.wd_step = wd_step
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.ws_step = ws_step
        self.ws_min = ws_min
        self.ws_max = ws_max
        self.bin_cols_in = bin_cols_in
        self.weight_by = weight_by
        self.df_freq = df_freq
        self.use_standard_error = use_standard_error
        self.N = N
        self.percentiles = percentiles
        self.remove_any_null_turbine_bins = remove_any_null_turbine_bins
        self.cov_terms = cov_terms

        # Store the results
        self.w_bins = w_bins
        self.energy_uplift_ctr_pc = energy_uplift_ctr_pc
        self.energy_uplift_lb_pc = energy_uplift_lb_pc
        self.energy_uplift_ub_pc = energy_uplift_ub_pc

    def plot(self, ax=None, color_dict={}) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the results.

        Args:
            ax (plt.Axes): An existing matplotlib Axes object to plot on.  Defaults to None.
            color_dict (Dict): A dictionary of uplift names and colors.  Defaults to {}.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.

        """
        # If ax is None, create a new figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.0, 3.0))

        else:
            # Get the figure from the ax
            fig = ax.get_figure()

        # If the color dict is empty fill using a matplotlib palette
        if len(color_dict) == 0:
            num_uplift_names = len(self.uplift_names)
            c_pal = plt.cm.get_cmap("tab10", num_uplift_names)
            color_dict = {
                uplift_name: c_pal(up_idx) for up_idx, uplift_name in enumerate(self.uplift_names)
            }

        # Loop over the uplift names
        for up_idx, uplift_name in enumerate(self.uplift_names):
            # Plot the results
            ax.plot(
                self.w_bins,
                self.energy_uplift_ctr_pc[up_idx],
                label=uplift_name,
                color=color_dict[uplift_name],
            )
            ax.fill_between(
                self.w_bins,
                self.energy_uplift_lb_pc[up_idx],
                self.energy_uplift_ub_pc[up_idx],
                color=color_dict[uplift_name],
                alpha=0.2,
            )

        ax.grid(True)
        # ax.set_xlabel(f"{self.wd_or_ws}_bin")
        ax.set_ylabel("Energy Uplift (%)")
        ax.axhline(0, color="black", linestyle="--")
        ax.legend()
        return fig, ax


class total_uplift_expected_power_by_wd(_total_uplift_expected_power_by_):
    """Compute total uplift expected power by wind direction."""

    def __init__(
        self,
        a_in: AnalysisInput,
        uplift_pairs: List[Tuple[str, str]],
        uplift_names: List[str],
        test_turbines: List[int],
        wd_turbines: List[int] = None,
        ws_turbines: List[int] = None,
        use_predefined_wd: bool = False,
        use_predefined_ws: bool = False,
        wd_step: float = 2.0,
        wd_min: float = 0.0,
        wd_max: float = 360.0,
        ws_step: float = 1.0,
        ws_min: float = 0.0,
        ws_max: float = 50.0,
        bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
        weight_by: str = "min",  # min or sum
        df_freq: pd.DataFrame = None,
        use_standard_error: bool = True,
        N: int = 1,
        percentiles: List[float] = [2.5, 97.5],
        remove_any_null_turbine_bins: bool = False,
        cov_terms: str = "zero",
    ) -> None:
        """Calculates total uplift expected power by wind direction.

        Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        test_turbines (List[int]): A list of turbine indices to test
        wd_turbines (List[int]): A list of turbine indices for wind direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for wind speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin. Defaults to None.
        use_standard_error (bool): Use standard error for the uplift calculation. Defaults to True.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The percentiles to calculate for the bootstrap samples.
            Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        cov_terms (str): Use directly computed covariance terms, or fill with zeros or variances.
            Can be "zero", "var" or "cov".  If "zero" all covariance terms are set to zero.  if
            "var" all covariance terms are set to the product of the variances.  If "cov" the
            covariance terms are used as is with missing terms set to product of the
            variances.  Defaults to "zero".
        """
        super().__init__(
            wd_or_ws="wd",
            a_in=a_in,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            test_turbines=test_turbines,
            wd_turbines=wd_turbines,
            ws_turbines=ws_turbines,
            use_predefined_wd=use_predefined_wd,
            use_predefined_ws=use_predefined_ws,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq=df_freq,
            use_standard_error=use_standard_error,
            N=N,
            percentiles=percentiles,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
            cov_terms=cov_terms,
        )

    def plot(self, ax=None, color_dict={}) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the results.

        Args:
            ax (plt.Axes): An existing matplotlib Axes object to plot on.  Defaults to None.
            color_dict (Dict): A dictionary of uplift names and colors.  Defaults to {}.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.

        """
        fig, ax = super().plot(ax=ax, color_dict=color_dict)
        ax.set_xlabel("Wind Direction (Deg)")


class total_uplift_expected_power_by_ws(_total_uplift_expected_power_by_):
    """Compute total uplift expected power by wind speed."""

    def __init__(
        self,
        a_in: AnalysisInput,
        uplift_pairs: List[Tuple[str, str]],
        uplift_names: List[str],
        test_turbines: List[int],
        wd_turbines: List[int] = None,
        ws_turbines: List[int] = None,
        use_predefined_wd: bool = False,
        use_predefined_ws: bool = False,
        wd_step: float = 2.0,
        wd_min: float = 0.0,
        wd_max: float = 360.0,
        ws_step: float = 1.0,
        ws_min: float = 0.0,
        ws_max: float = 50.0,
        bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
        weight_by: str = "min",  # min or sum
        df_freq: pd.DataFrame = None,
        use_standard_error: bool = True,
        N: int = 1,
        percentiles: List[float] = [2.5, 97.5],
        remove_any_null_turbine_bins: bool = False,
        cov_terms: str = "zero",
    ) -> None:
        """Calculates total uplift expected power by wind direction.

        Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        test_turbines (List[int]): A list of turbine indices to test
        wd_turbines (List[int]): A list of turbine indices for wind direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for wind speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin. Defaults to None.
        use_standard_error (bool): Use standard error for the uplift calculation. Defaults to True.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The percentiles to calculate for the bootstrap samples.
            Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        cov_terms (str): Use directly computed covariance terms, or fill with zeros or variances.
            Can be "zero", "var" or "cov".  If "zero" all covariance terms are set to zero.  if
            "var" all covariance terms are set to the product of the variances.  If "cov" the
            covariance terms are used as is with missing terms set to product of the
            variances.  Defaults to "zero".
        """
        super().__init__(
            wd_or_ws="ws",
            a_in=a_in,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            test_turbines=test_turbines,
            wd_turbines=wd_turbines,
            ws_turbines=ws_turbines,
            use_predefined_wd=use_predefined_wd,
            use_predefined_ws=use_predefined_ws,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq=df_freq,
            use_standard_error=use_standard_error,
            N=N,
            percentiles=percentiles,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
            cov_terms=cov_terms,
        )

    def plot(self, ax=None, color_dict={}) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the results.

        Args:
            ax (plt.Axes): An existing matplotlib Axes object to plot on.  Defaults to None.
            color_dict (Dict): A dictionary of uplift names and colors.  Defaults to {}.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.

        """
        fig, ax = super().plot(ax=ax, color_dict=color_dict)
        ax.set_xlabel("Wind Speed (m/s)")
