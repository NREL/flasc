"""Module for fitting yaw loss power curve."""
import os

import matplotlib.pyplot as plt
import numpy as np
from floris.utilities import wrap_180
from scipy import optimize as opt

from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# TODO: Is this class still used anywhere?
class yaw_pow_fitting:
    """Class for fitting yaw loss power curve."""

    def __init__(self, df, df_upstream=None, ti=0):  # , turbine_list='all'):
        """Initialize the yaw power curve fitting object.

        Args:
            df (pd.DataFrame | FlascDataFrame): DataFrame containing the relevant data.
            df_upstream (pd.DataFrame): DataFrame containing the upstream conditions.
            ti (int): Index of the turbine to fit the yaw power curve to.
        """
        logger.info("Initializing yaw power curve filtering object.")
        # Assign dataframes to self
        # self.df_upstream = df_upstream
        self.set_df(df, df_upstream, ti)

        # # Set turbines specified by user
        # self.set_turbine_mode(turbine_list)

    def set_df(self, df, df_upstream, ti):
        """Set the dataframe for the yaw power curve fitting object.

        Args:
            df (pd.DataFrame | FlascDataFrame): DataFrame containing the relevant data.
            df_upstream (pd.DataFrame): DataFrame containing the upstream conditions.
            ti (int): Index of the turbine to fit the yaw power curve to.
        """
        # if 'vane_000' not in df.columns:
        #     raise KeyError('vane_000 not found in dataset.')

        # # Get true total number of turbines
        # self.num_turbines_all = fsut.get_num_turbines(df)
        # self.full_turbine_list = range(self.num_turbines_all)

        # Set df using only the relevant columns
        rlvnt_cols = ["pow", "pow_ref", "vane", "wd"]
        # rlvnt_cols.extend(['pow_%03d' % ti for ti in range(self.num_turbines_all)])
        # rlvnt_cols.extend(['vane_%03d' % ti for ti in range(self.num_turbines_all)])
        # rlvnt_cols = [c for c in rlvnt_cols if c in df.columns]
        df = df[rlvnt_cols]

        # Reset output variables
        self.bins_x = []
        self.bins_y = []
        self.bins_N = []

        self.x_opt = [(None, None)]
        self.bins_y_opt = []

        # Filter by upstream conditions
        if df_upstream is not None:
            df_upstr_ti = df_upstream[[ti in tl for tl in df_upstream["turbines"]]]
            df_upstr_ti = df_upstr_ti.reset_index(drop=True)
            in_range = [False for _ in range(df.shape[0])]
            for i in range(df_upstr_ti.shape[0]):
                wd_min = df_upstr_ti.loc[i, "wd_min"]
                wd_max = df_upstr_ti.loc[i, "wd_max"]
                in_range = in_range | ((df["wd"] >= wd_min) & (df["wd"] <= wd_max))
            df = df.loc[in_range]

        self.df = df

    # def set_turbine_mode(self, turbine_list):
    #     if isinstance(turbine_list, str):
    #         if turbine_list == 'all':
    #             num_turbines = fsut.get_num_turbines(self.df)
    #             turbine_list = range(num_turbines)
    #         else:
    #             raise KeyError('Invalid turbine_list specified.')

    #     self.turbine_list = turbine_list
    #     self.num_turbines = len(turbine_list)

    def calculate_curves(self, vane_bounds=(-15.0, 15.0), dv=1.0, Pmin=10.0):
        """Calculate the yaw-power curve.

        Args:
            vane_bounds (tuple): Tuple containing the lower and upper bounds of the vane angle.
                Default is (-15.0, 15.0).
            dv (float): Bin width for the vane angle. Default is 1.0.
            Pmin (float): Minimum power value to consider. Default is 10.0.
        """
        df = self.df
        # df_upstream = self.df_upstream
        # turbine_list = self.turbine_list

        logger.info("Determining yaw-power curve...")

        df = df[df["pow_ref"] > Pmin]

        # Define vane and (normalized) power measurements
        vane = wrap_180(np.array(df["vane"]))

        # Filter for viable conditions
        ids_good = (
            (vane >= vane_bounds[0])
            & (vane <= vane_bounds[1])
            & (df["pow"] > Pmin)
            & (df["pow_ref"] > Pmin)
        )
        vane = vane[ids_good]
        Pnorm = df.loc[ids_good, "pow"] / df.loc[ids_good, "pow_ref"]
        logger.info("  Number of useful datapoints: %d." % len(vane))

        # Bin data
        logger.info("  Binning data...")
        bins_x = np.arange(vane_bounds[0], vane_bounds[1], dv)
        bins_y = np.zeros_like(bins_x)
        bins_N = np.zeros_like(bins_x)

        for ii, edge_x_l in enumerate(bins_x):
            edge_x_r = edge_x_l + dv
            yi = Pnorm[(vane >= edge_x_l) & (vane < edge_x_r)]
            bins_N[ii] = yi.shape[0]
            bins_y[ii] = np.nanmean(yi)

        # if np.any(bins_N > 0):
        #     bins_y = np.array(bins_y) / np.nanmax(bins_y[bins_N/np.max(bins_N) > 0.10])

        self.bins_x = bins_x
        self.bins_y = bins_y
        self.bins_N = bins_N

    def estimate_cos_pp_fit(
        self,
        opt_yshift_range=None,
        opt_bias_range=(-15.0, 15.0),
        opt_pp_range=(1.0, 10.0),
        opt_Ns=41,
    ):
        """Estimate the best fit for a cos(x-x0)^pp curve.

        Args:
            opt_yshift_range (tuple): Tuple containing the lower and upper bounds for the y shift.
                Default is None.
            opt_bias_range (tuple): Tuple containing the lower and upper bounds for the bias.
                Default is (-15.0, 15.0).
            opt_pp_range (tuple): Tuple containing the lower and upper bounds for the power.
                Default is (1.0, 10.0).
            opt_Ns (int): Number of samples to use for the optimization. Default is 41.

        Returns:
            x_opt (np.array): Optimal parameters for the cos(x-x0)^pp curve.
                Where x[0] is the y shift, x[1] is the bias, and x[2] is the exponent pp.
        """
        # for ti in self.turbine_list:
        bins_x = self.bins_x
        bins_y = self.bins_y
        bins_N = self.bins_N

        if len(bins_x) <= 0:
            raise ValueError(
                "Please calculate curves using "
                + ".calculate_curves() before "
                + "estimating a fit."
            )

        # Define an approximating function
        def approx_func(x):
            y = x[0] * np.cos((bins_x - x[1]) * np.pi / 180.0) ** x[2]
            return y

        # Define a cost function
        def cost(x):
            # x[0] is the x offset, x[1] is the cos coefficient
            y_fit = approx_func(x)
            J_fit = np.multiply(bins_N, (y_fit - bins_y) ** 2.0)
            J_sum = np.nanmean(J_fit)
            return J_sum

        if opt_yshift_range is None:
            opt_yshift_range = (np.nanmin(bins_y), np.nanmax(bins_y))

        logger.info("Fitting a cos(x-x0)^pp curve to the data...")
        x_opt, J_opt, x, J = opt.brute(
            func=cost,
            ranges=(opt_yshift_range, opt_bias_range, opt_pp_range),
            Ns=opt_Ns,
            finish=opt.fmin,
            full_output=True,
            disp=True,
        )
        logger.info(f"x_opt: {x_opt}")
        y_opt = approx_func(x_opt)

        self.x_opt = x_opt
        self.bins_y_opt = y_opt

        return x_opt

    def plot(self, save_path=None, fig_dpi=250):
        """Plot the yaw-power curve.

        Args:
            save_path (str): Path to save the figure. Default is None.
            fig_dpi (int): DPI of the figure. Default is 250.

        Returns:
            A tuple (matplotlib.figure.Figure, matplotlib.axes.Axes) containing the figure and axes.
        """
        # for ti in self.turbine_list:
        bins_x = self.bins_x
        bins_y = self.bins_y
        bins_N = self.bins_N

        x_opt = self.x_opt
        y_opt = self.bins_y_opt

        if len(bins_x) <= 0:
            raise ValueError(
                "Please calculate curves using " + ".calculate_curves() before " + "plotting."
            )

        # Plot bins and averaged curve
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(bins_x, bins_y, "o-", label="Data")
        if len(y_opt) > 0:
            ax[0].plot(
                bins_x,
                y_opt,
                "--",
                label=("Fit (x0=%.3f, x1=%.3f, x2=%.3f)" % (x_opt[0], x_opt[1], x_opt[2])),
            )
        ax[0].set_xlabel("Vane measurement (deg)")
        ax[0].set_ylabel("Relative power production (-)")
        ax[0].grid("minor")
        ax[0].legend()

        ax[1].bar(bins_x, bins_N)
        ax[1].set_xlabel("Vane measurement (deg)")
        ax[1].set_ylabel("Number of data points (-)")

        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=fig_dpi)

        return fig, ax
