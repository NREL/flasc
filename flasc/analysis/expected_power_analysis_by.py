"""Module for performing expected power analyses by wind direction or wind speed."""

# This module contains functions for computing the total wind farm energy uplift between two modes
# of control (e.g., wind farm control and baseline control) binned by wind direction or wind speed
# using methods described in the report:
#
# Kanev, S., "AWC Validation Methodology,"" TNO 2020 R11300, Tech. rep., TNO, Petten,
# The Netherlands, 2020. https://resolver.tno.nl/uuid:fdae4c94-fbcc-4337-b49f-5a39c93ef2cf.
#
# Specifically, this module computes the total uplift along with the confidence interval of the
# total uplift estimate by implementing Equations 4.11 - 4.29 in the abovementioned report. To
# determine total energy for the two control modes being compared, the expected wind farm power
# is computed for each wind direction/wind speed bin. This is done by summing the expected power
# of each individual turbine for the bin. Note that by computing expected power at the turbine
# level before summing, the method does not require that all test turbines are operating normally
# at each timestamp. Therefore, fewer timestamps need to be discarded than for total uplift methods
# that require all test turbines to be operating normally at each timestamp, such as the energy
# ratio-based approach. Total wind farm energy is then computed by summing the expected farm power
# values weighted by their frequencies of occurrence over all wind condition bins. However, because
# the test turbine power values in this method are not normalized by the power of reference
# turbines, the method may be more sensitive to wind speed variations within bins and other
# atmospheric conditions aside from wind speed or direction that are not controlled for.
#
# The module provides two approaches for quantifying uncertainty in the total uplift. First,
# bootstrapping can be used to estimate uncertainty by randomly resampling the input data with
# replacement and computing the resulting uplift for many iterations. The second option computes
# uncertainty in the total uplift following the approach in the abovementioned TNO report by
# propagating the standard errors of the expected wind farm power in each wind condition bin for
# the two control modes. Benefits of the method of propagating standard errors include an analytic
# expression for calculating uncertainty, rather than replying on the empirical bootstrapping
# approach, and higher computational efficiency compared to bootstrapping, which relies on
# computing the uplift for many different iterations. Some drawbacks of the method include the need
# to linearize the formula for total uplift to permit the standard error to be calculated,
# resulting in an approximation of the uncertainty, as well as challenges with computing the
# required variances and covariances of wind turbine power in bins with very little data.

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        """Calculates total uplift using expected power methods by wind direction or wind speed.

        Args:
        wd_or_ws (str): The column name to bin the dataframes and compute total uplift by.
            Can be "wd" or "ws".
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin.
            Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
            Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".
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

        # Save this full version
        self.df_pandas = df_pandas

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

    def plot(
        self, ax=None, color_dict={}, overwrite_label=None, plot_unc=True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the total uplift results by wind direction or wind speed.

        Args:
            ax (plt.Axes): An existing matplotlib Axes object to plot on.  Defaults to None.
            color_dict (Dict): A dictionary of uplift names and colors.  Defaults to {}.
            overwrite_label (str): If not None, overwrite the uplift names with this label.
                Defaults to None.
            plot_unc (bool): If True, plot the uncertainty bounds.  Defaults to True.


        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.

        """
        # If ax is None, create a new figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.0, 3.0))

        else:
            # Get the figure from the ax
            fig = ax.get_figure()

        # If overwrite_labels is not None
        if overwrite_label is not None:
            # color_dict can't be empty
            if len(color_dict) == 0:
                raise ValueError("color_dict must be provided if overwrite_labels is not None")

            # The entry in color_dict must include overwrite_label
            if overwrite_label not in color_dict:
                raise ValueError(f"overwrite_label '{overwrite_label}' not in color")

            # Set uplift_names to be the overwrite_label
            uplift_names = [overwrite_label]
        else:
            uplift_names = self.uplift_names

        # If the color dict is empty fill using a matplotlib palette
        if len(color_dict) == 0:
            num_uplift_names = len(self.uplift_names)
            c_pal = plt.cm.get_cmap("tab10", num_uplift_names)
            color_dict = {
                uplift_name: c_pal(up_idx) for up_idx, uplift_name in enumerate(self.uplift_names)
            }

        # Loop over the uplift names
        for up_idx, uplift_name in enumerate(uplift_names):
            # Plot the results
            ax.plot(
                self.w_bins,
                self.energy_uplift_ctr_pc[up_idx],
                label=uplift_name,
                color=color_dict[uplift_name],
            )
            if plot_unc:
                ax.fill_between(
                    self.w_bins,
                    self.energy_uplift_lb_pc[up_idx],
                    self.energy_uplift_ub_pc[up_idx],
                    color=color_dict[uplift_name],
                    alpha=0.2,
                )

        ax.grid(True)

        if self.wd_or_ws == "wd":
            ax.set_xlabel("Wind Direction (Deg)")
        else:
            ax.set_xlabel("Wind Speed (m/s)")

        ax.set_ylabel("Energy Uplift (%)")
        ax.axhline(0, color="black", linestyle="--")
        ax.legend()
        return fig, ax

    def plot_with_distributions(
        self, axarr=None, color_dict={}
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot results by wind direction or wind speed with histograms and distributions.

        Plot the total uplift results by wind direction or wind speed with accompanying histograms
        and distributions of wind speed or wind direction.

        Args:
            axarr (List[plt.Axes]): An existing list of matplotlib Axes objects to plot on.
                Defaults to None.
            color_dict (Dict): A dictionary of uplift names and colors.  Defaults to {}.

        Returns:
            Tuple[plt.Figure, List[plt.Axes]]: A tuple containing the matplotlib Figure
                and list of Axes objects.

        """
        # If axarr is None, create a new figure
        if axarr is None:
            fig, axarr = plt.subplots(3, 1, figsize=(7.0, 10.0))
        else:
            # Get the figure from the ax
            fig = axarr[0].get_figure()

        ## AXIS 0 is the UPLIFT
        # Plot the normal results on the top
        self.plot(ax=axarr[0], color_dict=color_dict)

        # Remove the x-label from axarr[0]
        axarr[0].set_xlabel("")

        if self.wd_or_ws == "wd":
            x_var = "wd_bin"
            y_var = "ws_bin"
            x_label = "Wind Direction (Deg)"
            y_label = "Wind Speed (m/s)"
        else:
            x_var = "ws_bin"
            y_var = "wd_bin"
            x_label = "Wind Speed (m/s)"
            y_label = "Wind Direction (Deg)"

        ## AXIS 1 is histogram
        sns.histplot(
            data=self.df_pandas, x=x_var, ax=axarr[1], bins=sorted(self.df_pandas[x_var].unique())
        )
        axarr[1].set_ylabel("Count")

        ## AXIS 2 is the DISTRIBUTIONS as scatter plot
        # Count the values in wd_bin, ws_bin in self.df_pandas
        df_bin_counts = (
            self.df_pandas.groupby(["wd_bin", "ws_bin"]).size().reset_index(name="count")
        )

        # Plot the distributions on the bottom
        sns.scatterplot(
            data=df_bin_counts,
            x=x_var,
            y=y_var,
            size="count",
            hue="count",
            ax=axarr[2],
            legend=True,
            color="k",
        )
        axarr[2].set_xlabel(x_label)
        axarr[2].set_ylabel(y_label)

        return fig, axarr


class total_uplift_expected_power_by_wd(_total_uplift_expected_power_by_):
    """Compute total uplift using expected power methods by wind direction."""

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
        """Calculates total uplift using expected power methods by wind direction.

        Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin.
            Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
            Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".
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


class total_uplift_expected_power_by_ws(_total_uplift_expected_power_by_):
    """Compute total uplift using expected power methods by wind speed."""

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
        """Calculates total uplift using expected power methods by wind speed.

        Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin.
            Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
                Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".
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


def total_uplift_expected_power_by_wd_shift_ws_min(
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
    n_step: int = 10,
    ax=None,
):
    """Calculate total uplift by wind direction for different minimum wind speed values.

    Calculate total uplift using expected power methods by wind direction for different wind speed
    minimum ws_min values. Plots total uplift as a function of wind direction for each choice of
    ws_min.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin.
            Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
                Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".
        n_step (int): The number of steps to perform the minimum wind speed sweep over.
            Defaults to 10.
        ax (plt.Axes): An existing matplotlib Axes object to plot on.  Defaults to None.
    """
    # Get the min_steps to try
    ws_min_values = np.linspace(ws_min, ws_min + ws_step, n_step)

    # Define the labels for plotting
    labels = [f"ws_min = {ws_min_loop:.1f}" for ws_min_loop in ws_min_values]

    # Define the color dict
    c_pal = plt.cm.get_cmap("viridis", n_step)
    color_dict = {label: c_pal(idx) for idx, label in enumerate(labels)}

    # Declare a figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))

    # Loop over these values
    for ws_min_loop, label in zip(ws_min_values, labels):
        t_ws = total_uplift_expected_power_by_wd(
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
            ws_min=ws_min_loop,
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

        t_ws.plot(ax=ax, color_dict=color_dict, overwrite_label=label, plot_unc=False)
