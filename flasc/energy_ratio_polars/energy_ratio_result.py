import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from flasc.energy_ratio_polars.energy_ratio_utilities import add_ws_bin, add_wd_bin


class EnergyRatioResult:
    """  This class is used to store the results of the energy ratio calculations
    and provide convenient methods for plotting and saving the results.
    """
    def __init__(self,
                    df_result, 
                    df_names,
                    energy_table,
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
                    wd_bin_overlap_radius,
                    N
                  ):

        self.df_result = df_result
        self.df_names = df_names
        self.energy_table = energy_table
        self.num_df = len(df_names)
        self.ref_cols = ref_cols
        self.test_cols = test_cols
        self.wd_cols = wd_cols
        self.ws_cols = ws_cols
        self.wd_step = wd_step
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.ws_step = ws_step
        self.ws_min = ws_min
        self.ws_max = ws_max
        self.bin_cols_in = bin_cols_in
        self.wd_bin_overlap_radius = wd_bin_overlap_radius
        self.N = N

        # self.df_freq = self._compute_df_freq()

    def _compute_df_freq(self):
        """ Compute the of ws/wd as previously computed but not presently
        computed with the energy calculation. """
        
        #TODO: I don't think so, but should this function count overlapping bins?

        # Temporary copy of energy table
        df_ = self.energy_table.clone()

        # Filter df_ that all the columns are not null
        df_ = df_.filter(pl.all(pl.col(self.ref_cols + self.test_cols + self.ws_cols + self.wd_cols).is_not_null()))

        # Assign the wd/ws bins
        df_ = add_ws_bin(df_, self.ws_cols, self.ws_step, self.ws_min, self.ws_max)
        df_ = add_wd_bin(df_, self.wd_cols, self.wd_step, self.wd_min, self.wd_max)

        # Get the bin count by wd, ws and df_name
        df_group = df_.groupby(['wd_bin','ws_bin','df_name']).count()

        return df_.to_pandas(), df_group.to_pandas()

    def plot(self,
        df_names_subset = None,
        labels = None,
        color_dict = None,
        axarr = None,
        polar_plot=False,
        show_wind_speed_distrubution=True,
    ):

        # Only allow showing the wind speed distribution if polar_plot is False
        if polar_plot and show_wind_speed_distrubution:
            raise ValueError('show_wind_speed_distrubution cannot be True if polar_plot is True')

        # If df_names_subset is None, plot all the dataframes
        if df_names_subset is None:
            df_names_subset = self.df_names

        # If df_names_subset is not a list, convert it to a list
        if not isinstance(df_names_subset, list):
            df_names_subset = [df_names_subset]

        # Total number of energy ratios to plot
        N = len(df_names_subset)

        # If labels is None, use the dataframe names
        if labels is None:
            labels = df_names_subset

        # If labels is not a list, convert it to a list
        if not isinstance(labels, list):
            labels = [labels]

        # Confirm that the length of labels is the same as the length of df_names_subset
        if len(labels) != N:
            raise ValueError('Length of labels must be the same as the length of df_names_subset')

        # Generate the default colors using the seaborn color palette
        default_colors = sns.color_palette('colorblind', N)

        # If color_dict is None, use the default colors
        if color_dict is None:
            color_dict = {labels[i]: default_colors[i] for i in range(N)}

        # If color_dict is not a dictionary, raise an error
        if not isinstance(color_dict, dict):
            raise ValueError('color_dict must be a dictionary')

        # Make sure the keys of color_dict are in df_names_subset
        if not all([label in df_names_subset for label in color_dict.keys()]):
            raise ValueError('color_dict keys must be in df_names_subset')

        if axarr is None:
            if polar_plot:
                _, axarr = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), subplot_kw={'projection': 'polar'})
            else:
                if show_wind_speed_distrubution:
                    num_rows = 3 # Add rows to show wind speed distribution
                else:
                    num_rows = 2
                _, axarr = plt.subplots(nrows=num_rows, ncols=1, sharex=True, figsize=(10, 5))

        # Set the bar width using self.wd_step
        bar_width = (0.7 / N) * self.wd_step
        if polar_plot:
            bar_width = bar_width * np.pi / 180.0

        # For plotting, get a pandas dataframe
        df = self.df_result.to_pandas()

        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)

        # Add NaNs to avoid connecting plots over gaps
        dwd = np.min(x[1::] - x[0:-1])
        jumps = np.where(np.diff(x) > dwd * 1.50)[0]
        if len(jumps) > 0:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "wd_bin": x[jumps] + dwd / 2.0,
                            "N_bin": [0] * len(jumps),
                        }
                    )
                ],
                axis=0,
                ignore_index=False,
            )
            df = df.iloc[np.argsort(df["wd_bin"])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = np.linspace(np.min(x) - 4.0, np.max(x) + 4.0, 1000)

        if polar_plot:
            x = (90.0 - x) * np.pi / 180.0  # Convert to radians
            xlims = (90.0 - xlims) * np.pi / 180.0  # Convert to radians

        # Plot the horizontal line at 1
        axarr[0].plot(xlims, np.ones_like(xlims), color="black")

        # Plot the energy ratios
        for df_name, label in zip(df_names_subset, labels):

            axarr[0].plot(x, df[df_name], "-o", markersize=3.0, label=label, color=color_dict[label])

            # If data includes upper and lower bounds plot them
            if df_name + "_ub" in df.columns:

                axarr[0].fill_between(
                    x,
                    df[df_name + "_lb"],
                    df[df_name + "_ub"],
                    alpha=0.25,
                    color=color_dict[label],
                )

        # Format the energy ratio plot
        axarr[0].legend()
        axarr[0].grid(visible=True, which="major", axis="both", color="gray")
        axarr[0].grid(visible=True, which="minor", axis="both", color="lightgray")
        axarr[0].minorticks_on()
        # axarr[0].set_grid(True)

        # Plot the bin counts
        df_unbinned, df_freq = self._compute_df_freq()
        df_freq_sum_all_ws = df_freq.groupby(["wd_bin","df_name"]).sum().reset_index()


        for i, (df_name, label) in enumerate(zip(df_names_subset, labels)):
            df_sub = df_freq_sum_all_ws[df_freq_sum_all_ws["df_name"] == df_name]
            
            x = np.array(df_sub["wd_bin"], dtype=float)
            if polar_plot:
                x = (90.0 - x) * np.pi / 180.0  # Convert to radians
            axarr[1].bar(x - (i - N / 2) * bar_width, df_sub["count"], width=bar_width, label = label, color=color_dict[label])

        axarr[1].legend()

        # Get the bins
        wd_bins = np.array(df_freq["wd_bin"].unique(), dtype=float)
        ws_bins = np.array(df_freq["ws_bin"].unique(), dtype=float)
        num_wd_bins = len(wd_bins)
        num_ws_bins = len(ws_bins)

        if show_wind_speed_distrubution:
            # Plot the wind speed distribution in df_freq as a heat map with wd on the x-axis and ws on the y-axis
            
            ax = axarr[2]
            for df_name, label in zip(df_names_subset, labels):
                df_sub = df_freq[df_freq["df_name"] == df_name]
                ax.scatter(df_unbinned["wd_bin"], df_unbinned["ws_bin"], c=color_dict[label],alpha=0.25, s=1)


    def plot_uplift(self,
        axarr = None,
        polar_plot=False,
        show_wind_speed_distrubution=True,
    ):
        self.plot(
            df_names_subset = 'uplift',
            labels = ['uplift'],
            color_dict = {'uplift':'k'},
            axarr = axarr,
            polar_plot=polar_plot,
            show_wind_speed_distrubution=show_wind_speed_distrubution,
        )
            
