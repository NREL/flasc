"""Module to calculate and visualize the heterogeneity in the inflow wind speed."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial._qhull
from floris.utilities import wrap_360
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from flasc.analysis import energy_ratio as er
from flasc.analysis.analysis_input import AnalysisInput
from flasc.data_processing import dataframe_manipulations as dfm
from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# Standalone function to easily extract energy ratios for narrow wind direction bin
def _get_energy_ratio(df, ti, wd_bins, ws_range):
    # Calculate energy ratios
    a_in = AnalysisInput([df], ["data"])
    return er.compute_energy_ratio(
        a_in,
        test_turbines=[ti],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=15.0,
        wd_bin_overlap_radius=0.0,
        wd_min=wd_bins[0][0],
        wd_max=wd_bins[0][1],
        ws_min=ws_range[0],
        ws_max=ws_range[1],
        N=10,
        percentiles=[5.0, 95.0],
    )


# Heterogeneity mapper class with all internal functions to calculate,
# extract and plot heterogeneity
# derived from upstream turbine's power measurements
class heterogeneity_mapper:
    """Class for calculating and visualizing the heterogeneity in the inflow wind speed.

    This class is useful to calculate the energy ratios of a set
    of upstream turbines to then derive the heterogeneity in the
    inflow wind speed. This can be helpful in characterizing the
    ambient wind speed distribution for operational assets where
    we do not have such information. Note that heterogeneity may
    come from multiple sources, not only neighboring farms but also
    blockage effects of the farm itself.
    """

    # Private functions
    def __init__(self, df_raw, fm):
        """Initialize the heterogeneity_mapper class.

        Args:
            df_raw (pd.DataFrame): The raw SCADA data to use for the analysis.
            fm (FlorisModel): The FLORIS model
                to use for the analysis.
        """
        # Save to self
        self.df_raw = df_raw
        self.fm = fm
        self.df_heterogeneity = None
        self.df_fi_hetmap = None

    def _process_single_wd(self, wd, wd_bin_width, ws_range, df_upstream):
        # In this function, we calculate the energy ratios of all upstream
        # turbines for a single wind direction bin and single wind speed bin.
        # The difference in energy ratios between different upstream turbines
        # gives a strong indication of the heterogeneity in the inflow wind
        # speeds for that mean inflow wind direction.
        logger.info("Processing wind direction = {:.1f} deg.".format(wd))
        wd_bins = [[wd - wd_bin_width / 2.0, wd + wd_bin_width / 2.0]]

        # Determine which turbines are upstream
        if wd > df_upstream.iloc[0]["wd_max"]:
            turbine_array = df_upstream.loc[
                (wd > df_upstream["wd_min"]) & (wd <= df_upstream["wd_max"]), "turbines"
            ].values[0]

        # deal with wd = 0 deg (or close to 0.0)
        else:
            turbine_array = df_upstream.loc[
                (wrap_360(wd + 180) > wrap_360(df_upstream["wd_min"] + 180.0))
                & (wrap_360(wd + 180) <= wrap_360(df_upstream["wd_max"] + 180)),
                "turbines",
            ].values[0]

        # Load data and limit region
        df = self.df_raw.copy()
        pow_cols = ["pow_{:03d}".format(t) for t in turbine_array]
        df = df.dropna(subset=pow_cols)

        # Filter dataframe and set a reference wd and ws
        df = dfm.filter_df_by_wd(df, [wd - wd_bin_width, wd + wd_bin_width])
        df = dfm.set_ws_by_turbines(df, turbine_array)
        df = dfm.filter_df_by_ws(df, ws_range)

        # Set reference power for df and df_fi as the average power
        # of all upstream turbines
        df = dfm.set_pow_ref_by_turbines(df, turbine_array)
        df = df.dropna(subset=["wd", "ws", "pow_ref"])

        results_scada = []
        for ti in turbine_array:
            # Get energy ratios
            er = _get_energy_ratio(df, ti, wd_bins, ws_range)
            results_scada.append(er.df_result.loc[0])

        results_scada = pd.concat(results_scada, axis=1).T
        energy_ratios = np.array(results_scada["data"], dtype=float)
        energy_ratios_lb = np.array(results_scada["data_lb"], dtype=float)
        energy_ratios_ub = np.array(results_scada["data_ub"], dtype=float)

        return pd.DataFrame(
            {
                "wd": [wd],
                "wd_bin_width": [wd_bin_width],
                "upstream_turbines": [turbine_array],
                "energy_ratios": [energy_ratios],
                "energy_ratios_lb": [energy_ratios_lb],
                "energy_ratios_ub": [energy_ratios_ub],
                "ws_ratios": [energy_ratios ** (1 / 3)],
                "bin_count": [np.array(results_scada["count_data"], dtype=int)],
            }
        )

    # Public functions
    def estimate_heterogeneity(
        self,
        df_upstream,
        wd_array=np.arange(0.0, 360.0, 3.0),
        wd_bin_width=6.0,
        ws_range=[6.0, 11.0],
    ):
        """Estimate the heterogeneity in the inflow wind speed.

        Args:
            df_upstream (_type_): _description_
            wd_array (_type_, optional): _description_. Defaults to np.arange(0.0, 360.0, 3.0).
            wd_bin_width (float, optional): _description_. Defaults to 6.0.
            ws_range (list, optional): _description_. Defaults to [6.0, 11.0].

        Returns:
            pd.DataFrame: A dataframe containing the energy ratios for all upstream turbines
                for each wind direction bin.
        """
        df_list = [
            self._process_single_wd(wd, wd_bin_width, ws_range, df_upstream) for wd in wd_array
        ]
        self.df_heterogeneity = pd.concat(df_list).reset_index(drop=True)
        return self.df_heterogeneity

    def plot_graphs(self, ylim=[0.8, 1.2], pdf_save_path=None):
        """Plot the energy ratios for all upstream turbines for each wind direction bin.

        Args:
            ylim (list, optional): The y-axis limits for the plots. Defaults to [0.8, 1.2].
            pdf_save_path (str, optional): The path to save the plots as a PDF. Defaults to None.
        """
        if self.df_heterogeneity is None:
            raise UserWarning("Please call 'estimate_heterogeneity(...)' first.")

        if pdf_save_path is not None:
            pdf = PdfPages(pdf_save_path)

        # Plot the results one by one
        for _, df_row in self.df_heterogeneity.iterrows():
            fig, ax = plt.subplots(figsize=(7, 4))
            turbine_array = df_row["upstream_turbines"]

            wd = df_row["wd"]
            N = df_row["bin_count"][0]

            x = range(len(turbine_array))
            ax.fill_between(
                x, df_row["energy_ratios_lb"], df_row["energy_ratios_ub"], color="k", alpha=0.30
            )
            ax.plot(x, df_row["energy_ratios"], "-o", color="k", label="SCADA")
            ax.grid(True, which="major")
            ax.set_ylabel("Energy ratio of upstream \n turbines w.r.t. the average (-)")
            ax.set_title("Wind direction = {:.1f} deg. Bin count: {:d}.".format(wd, N))
            ax.set_ylim(ylim)
            ax.set_xticks(x)
            ax.set_yticks(np.arange(ylim[0] - 0.05, ylim[1] + 0.05, 0.10))
            plt.minorticks_on()
            ax.set_xticklabels(["T{:03d}".format(t) for t in turbine_array])
            ax.set_yticklabels(["{:.2f}".format(f) for f in ax.get_yticks()])

            ax2 = ax.twinx()
            ax2.plot(x, df_row["energy_ratios"], "-o", color="orange", label="SCADA")
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(ax.get_yticks())
            ax2.set_yticklabels(
                ["{:.2f}".format(np.cbrt(f)) for f in ax.get_yticks()], color="orange"
            )
            ax2.set_ylabel(
                "WS ratio of upstream \n turbines w.r.t. the average (-)", color="orange"
            )

            plt.tight_layout()

            if pdf_save_path is not None:
                pdf.savefig(fig)
                plt.close("all")

        if pdf_save_path is not None:
            logger.info("Plots saved to '{:s}'.".format(pdf_save_path))
            pdf.close()

    def generate_floris_hetmap(self):
        """Generate a dataframe for a FLORIS heterogeneous map.

        Returns:
            pd.DataFrame: A dataframe containing the FLORIS heterogeneous map values.
        """
        if self.df_heterogeneity is None:
            raise UserWarning("Please call 'estimate_heterogeneity(...)' first.")

        # Determine FLORIS heterogeneous map
        fm = self.fm
        ll = 2.0 * np.sqrt(
            (np.max(fm.layout_x) - np.min(fm.layout_x)) ** 2.0
            + (np.max(fm.layout_y) - np.min(fm.layout_y)) ** 2.0
        )
        locations_x = []
        locations_y = []
        speed_ups = []
        for ii in range(self.df_heterogeneity.shape[0]):
            df_row = self.df_heterogeneity.loc[ii]
            turbs = df_row["upstream_turbines"]
            wd = (270.0 - df_row["wd"]) * np.pi / 180.0
            x_turbs = np.array(fm.layout_x, dtype=float)[turbs]
            y_turbs = np.array(fm.layout_y, dtype=float)[turbs]

            xlocs = np.hstack(
                [xt + ll * np.cos(wd) * np.linspace(-1.0, 1.0, 100) for xt in x_turbs]
            )
            ylocs = np.hstack(
                [yt + ll * np.sin(wd) * np.linspace(-1.0, 1.0, 100) for yt in y_turbs]
            )
            speedup_onewd = np.repeat(df_row["ws_ratios"], 100)
            locations_x.append(xlocs)
            locations_y.append(ylocs)
            speed_ups.append(speedup_onewd)

        df_fi_hetmap = pd.DataFrame(
            {
                "wd": self.df_heterogeneity["wd"],
                "x": locations_x,
                "y": locations_y,
                "speed_up": speed_ups,
            }
        )

        self.df_fi_hetmap = df_fi_hetmap
        return df_fi_hetmap

    # # Visualization
    def plot_layout(self, ylim=[0.8, 1.2], plot_background_flow=False, pdf_save_path=None):
        """Plot the layout of the wind farm with the inflow wind speed heterogeneity.

        Args:
            ylim (list, optional): The y-axis limits for the plots. Defaults to [0.8, 1.2].
            plot_background_flow (bool, optional): Whether to plot the background flow.
                Defaults to False.
            pdf_save_path (str, optional): The path to save the plots as a PDF. Defaults to None.

        Returns:
            tuple: The figure and axis objects.
        """
        if self.df_heterogeneity is None:
            raise UserWarning("Please call 'estimate_heterogeneity(...)' first.")

        if plot_background_flow and self.df_fi_hetmap is None:
            raise UserWarning("Please call 'generate_floris_hetmap(...)' first.")

        if pdf_save_path is not None:
            pdf = PdfPages(pdf_save_path)

        # Plot the results one by one
        fm = self.fm
        for _, df_row in self.df_heterogeneity.iterrows():
            non_upstream_turbines = [
                ti for ti in range(len(fm.layout_x)) if ti not in df_row["upstream_turbines"]
            ]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                x=fm.layout_x[non_upstream_turbines],
                y=fm.layout_y[non_upstream_turbines],
                s=300,
                c="lightgray",
            )
            for ti in non_upstream_turbines:
                ax.text(fm.layout_x[ti], fm.layout_y[ti], "T{:02d}".format(ti))

            im = ax.scatter(
                x=fm.layout_x[df_row["upstream_turbines"]],
                y=fm.layout_y[df_row["upstream_turbines"]],
                c=df_row["ws_ratios"],
                s=300,
                cmap="jet",
                vmin=ylim[0],
                vmax=ylim[1],
                edgecolor="black",
            )
            for ii, ti in enumerate(df_row["upstream_turbines"]):
                ax.text(
                    fm.layout_x[ti],
                    fm.layout_y[ti],
                    "T{:02d} ({:+.1f}%)".format(ti, 100.0 * (df_row["ws_ratios"][ii] - 1.0)),
                    weight="bold",
                )

            # Add arrow plot
            xm = np.min(fm.layout_x) - 100.0
            ym = np.max(fm.layout_y) + 500.0
            radius = 120.0
            theta = np.linspace(0.0, 2 * np.pi, 100)
            xcirc = np.cos(theta) * radius + xm
            ycirc = np.sin(theta) * radius + ym
            ax.plot(xcirc, ycirc, color="black", linewidth=2)
            plt.arrow(
                x=xm,
                y=ym,
                dx=np.cos(-(df_row["wd"] - 270.0) * np.pi / 180.0) * radius,
                dy=np.sin(-(df_row["wd"] - 270.0) * np.pi / 180.0) * radius,
                width=0.125 * radius,
                head_width=0.6 * radius,
                head_length=0.75 * radius,
                length_includes_head=True,
                color="black",
            )

            # Add title, colorbar
            ax.set_title("Wind direction: {:.1f} deg".format(df_row["wd"]))
            clb = plt.colorbar(im, ax=ax)
            clb.set_label("Wind speed ratio (-)")

            # Add plot to ensure equal axis does not crop plot too much
            ax.plot(
                np.max(fm.layout_x) + 500.0, np.min(fm.layout_y), ".", color="white", markersize=1
            )
            ax.axis("equal")
            plt.tight_layout()

            if plot_background_flow:
                df_hetmap = self.df_fi_hetmap.copy()
                id_hetmap = np.where(df_hetmap["wd"] == df_row["wd"])[0][0]
                df_hetmap_row = df_hetmap.loc[id_hetmap]

                if len(np.unique(df_hetmap_row["speed_up"])) <= 1:
                    # Add some noise to prevent issues
                    df_hetmap_row["speed_up"] += 0.0001 * np.random.randn(
                        len(df_hetmap_row["speed_up"])
                    )

                xlim_plot = ax.get_xlim()
                ylim_plot = ax.get_ylim()
                x, y = np.meshgrid(
                    np.linspace(xlim_plot[0], xlim_plot[1], 100),
                    np.linspace(ylim_plot[0], ylim_plot[1], 100),
                    indexing="ij",
                )
                x = x.flatten()
                y = y.flatten()

                try:
                    lin_interpolant = LinearNDInterpolator(
                        points=np.vstack([df_hetmap_row["x"], df_hetmap_row["y"]]).T,
                        values=df_hetmap_row["speed_up"],
                        fill_value=np.nan,
                    )
                    lin_values = lin_interpolant(x, y)
                except scipy.spatial._qhull.QhullError:
                    logger.warning("QhullError occurred. Falling back to nearest neighbor. ")
                    lin_values = np.nan * np.ones_like(x)

                nearest_interpolant = NearestNDInterpolator(
                    x=np.vstack([df_hetmap_row["x"], df_hetmap_row["y"]]).T,
                    y=df_hetmap_row["speed_up"],
                )
                nn_values = nearest_interpolant(x, y)
                ids_isnan = np.isnan(lin_values)

                het_map_mesh = np.array(lin_values, copy=True)
                het_map_mesh[ids_isnan] = nn_values[ids_isnan]

                # Produce plot
                im = ax.tricontourf(
                    x, y, het_map_mesh, cmap="jet", vmin=ylim[0], vmax=ylim[1], levels=50, zorder=-1
                )

            if pdf_save_path is not None:
                pdf.savefig(fig)
                plt.close("all")

        if pdf_save_path is not None:
            logger.info("Plots saved to '{:s}'.".format(pdf_save_path))
            pdf.close()

        return fig, ax
