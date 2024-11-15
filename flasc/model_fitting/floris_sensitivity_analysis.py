"""_summary_."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import DataError
from SALib.analyze import sobol
from SALib.sample import saltelli

from flasc.logging_manager import LoggingManager
from flasc.utilities import floris_tools as ftools

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


class floris_sobol_analysis:
    """_summary_."""

    def __init__(self, fi, problem, calc_second_order=False):
        """_summary_.

        Args:
            fi (_type_): _description_
            problem (_type_): _description_
            calc_second_order (bool, optional): _description_. Defaults to False.
        """
        self.fm = fi

        # Default parameters
        self.param_dict = {
            "ad": 0.0,
            "alpha": 0.58,
            "bd": 0.0,
            "beta": 0.077,
            "eps_gain": 0.20,
            "ka": 0.38,
            "kb": 0.004,
            "ti_ai": 0.80,
            "ti_constant": 0.50,
            "ti_downstream": -0.32,
            "ti_initial": 0.10,
        }

        self.problem = problem
        self.Si = None
        self.calc_second_order = calc_second_order
        self.samples_x = None
        self.samples_y = None
        self.N = None

    def _get_model_params_dict(self, id):
        new_params_dict = {}
        n_params = len(self.problem["names"])
        for i in range(n_params):
            var_name = self.problem["names"][i]
            value = self.samples_x[id, i]
            new_params_dict[var_name] = value

        keys = list(new_params_dict.keys())
        values = list(new_params_dict.values())
        for i in range(len(keys)):
            self.param_dict[keys[i]] = values[i]

        params = {
            "Wake Deflection Parameters": {
                "ad": self.param_dict["ad"],
                "alpha": self.param_dict["alpha"],
                "bd": self.param_dict["bd"],
                "beta": self.param_dict["beta"],
                "eps_gain": self.param_dict["eps_gain"],
                "ka": self.param_dict["ka"],
                "kb": self.param_dict["kb"],
                "use_secondary_steering": True,
            },
            "Wake Turbulence Parameters": {
                "ti_ai": self.param_dict["ti_ai"],
                "ti_constant": self.param_dict["ti_constant"],
                "ti_downstream": self.param_dict["ti_downstream"],
                "ti_initial": self.param_dict["ti_initial"],
            },
            "Wake Velocity Parameters": {
                "alpha": self.param_dict["alpha"],
                "beta": self.param_dict["beta"],
                "eps_gain": self.param_dict["eps_gain"],
                "ka": self.param_dict["ka"],
                "kb": self.param_dict["kb"],
                "use_yaw_added_recovery": True,
            },
        }

        return params
        # self.fm.set_model_parameters(params=params, verbose=False)

    def _create_evals_dataframe(self):
        Nt = self.samples_x.shape[0]
        params_array = [[] for _ in range(Nt)]
        for id in range(Nt):
            params_array[id] = self._get_model_params_dict(id)
        df = pd.DataFrame({"model_params_dict": params_array})
        self.df_eval = df

    # Step 1: generating samples for a particular problem
    def generate_samples(self, N, problem=None, calc_second_order=None):
        """_summary_.

        Args:
            N (_type_): _description_
            problem (_type_, optional): _description_. Defaults to None.
            calc_second_order (_type_, optional): _description_. Defaults to None.
        """
        if problem is None:
            problem = self.problem

        if calc_second_order is None:
            calc_second_order = self.calc_second_order
        self.problem = problem
        self.calc_second_order = calc_second_order

        Ns = N
        self.samples_x = saltelli.sample(problem, Ns, calc_second_order=calc_second_order)

        self.N = self.samples_x.shape[0]
        self.samples_y = np.zeros(self.N)
        self._create_evals_dataframe()

    def calculate_wfpower_for_samples(self, num_threads=1):
        """_summary_.

        Args:
            num_threads (int, optional): _description_. Defaults to 1.

        Raises:
            DataError: _description_

        Returns:
            _type_: _description_
        """
        if self.samples_x is None:
            raise DataError("Please run generate_samples first.")

        # Copy and write wd and ws to dataframe
        # Nt = self.df_eval.shape[0]
        df = self.df_eval
        df["wd"] = self.fm.core.farm.wind_direction[0]
        df["ws"] = self.fm.core.farm.wind_speed[0]

        # Calculate floris predictions
        df_out = ftools.calc_floris(df, self.fi, num_threads=10, num_workers=2)
        pow_cols = ["pow_%03d" % ti for ti in range(len(self.fm.layout_x))]
        self.samples_y = np.array(df_out[pow_cols].sum(axis=1), dtype=float)

        return self.samples_y

    def get_sobol_sensitivity_indices(self, verbose=False):
        """_summary_.

        Args:
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        self.Si = sobol.analyze(
            self.problem,
            self.samples_y,
            print_to_console=verbose,
            calc_second_order=self.calc_second_order,
        )
        return self.Si

    def plot_sobol_results(self, save_path=None, fig_format="png", fig_dpi=200):
        """_summary_.

        Args:
            save_path (_type_, optional): _description_. Defaults to None.
            fig_format (str, optional): _description_. Defaults to "png".
            fig_dpi (int, optional): _description_. Defaults to 200.

        Raises:
            DataError: _description_

        Returns:
            _type_: _description_
        """
        if self.Si is None:
            raise DataError(
                "No Sobol results to show. " + "Have you run get_sobol_sensitivity_indices()?"
            )
        problem = self.problem
        if self.calc_second_order:
            fig, ax = plt.subplots(2)
        else:
            fig, ax = plt.subplots()
            ax = [ax]
        width = 0.30
        Nv = problem["num_vars"]

        # Plot first order sensitivity plus uncertainties
        _ = ax[0].bar(
            np.arange(Nv) - width / 2,
            self.Si["S1"],
            width=0.3,
            align="center",
            label="First order sensitivity",
            color="deepskyblue",
            edgecolor="black",
            hatch="//",
        )
        # ax[0].bar_label(bar1, padding=7, fmt='%.1e')

        for i in range(Nv):
            ymean = self.Si["S1"][i]
            ystd = self.Si["S1_conf"][i]
            ax[0].plot(
                np.repeat(i - width / 2, 3), ymean + np.array([-ystd, 0.0, ystd]), color="black"
            )
            ax[0].plot(i - width / 2, ymean, "o", color="black")
            ax[0].plot(
                [i - 0.75 * width, i - 0.25 * width], [ymean - ystd, ymean - ystd], color="black"
            )
            ax[0].plot(
                [i - 0.75 * width, i - 0.25 * width], [ymean + ystd, ymean + ystd], color="black"
            )

        # Plot total sensitivity including uncertainties
        _ = ax[0].bar(
            np.arange(Nv) + width / 2,
            self.Si["ST"],
            width=0.3,
            align="center",
            label="Total sensitivity",
            color="orangered",
            edgecolor="black",
            hatch=".",
        )
        # ax[0].bar_label(bar2, padding=7, fmt='%.1e')

        for i in range(Nv):
            ymean = self.Si["ST"][i]
            ystd = self.Si["ST_conf"][i]
            ax[0].plot(
                np.repeat(i + width / 2, 3), ymean + np.array([-ystd, 0.0, ystd]), color="black"
            )
            ax[0].plot(i + width / 2, ymean, "o", color="black")
            ax[0].plot(
                [i + 0.25 * width, i + 0.75 * width], [ymean - ystd, ymean - ystd], color="black"
            )
            ax[0].plot(
                [i + 0.25 * width, i + 0.75 * width], [ymean + ystd, ymean + ystd], color="black"
            )

        # Plot settings
        ax[0].set_xticks(range(Nv))
        ax[0].set_xticklabels(problem["names"])
        ax[0].set_title("First order effects")
        ax[0].set_ylabel("Sensitivity (-)")
        ax[0].legend()
        ax[0].grid("minor")
        ax[0].set_ylim([0, 1.1])

        # Second order tabular/imshow plot, if calculated
        if self.calc_second_order:
            im = ax[1].imshow(self.Si["S2"][:-1, :])
            ax[1].set_xticks(range(Nv))
            ax[1].set_yticks(range(Nv - 1))
            ax[1].set_xticklabels(self.problem["names"])
            ax[1].set_yticklabels(self.problem["names"][:-1])
            ax[1].set_title("Second order effects")
            for ii in range(Nv - 1):
                for jj in range(Nv):
                    ax[1].text(
                        y=ii, x=jj, s="%.3e" % self.Si["S2"][ii, jj], ha="center", color="white"
                    )
            plt.copper()
            plt.colorbar(im, ax=ax[1])
            fig.tight_layout()

            if save_path is not None:
                plt.savefig(save_path + "/Sobol_sensitivity_indices.%s" % fig_format, dpi=fig_dpi)

        return fig, ax

    def plot_convergence(self, save_path=None, fig_format="png", fig_dpi=200):
        """_summary_.

        Args:
            save_path (_type_, optional): _description_. Defaults to None.
            fig_format (str, optional): _description_. Defaults to "png".
            fig_dpi (int, optional): _description_. Defaults to 200.

        Returns:
            _type_: _description_
        """
        logger.info("Analyzing convergence...")

        # Create copies of original results
        samples_x_full = self.samples_x
        samples_y_full = self.samples_y
        Si_full = self.Si
        N = self.N

        # Create Sobol outputs for data subsets
        Si_list = []
        self.generate_samples(N=10)
        dN = self.samples_x.shape[0]
        N_array = np.arange(dN, N, dN, dtype="int")
        if N_array[-1] < N:
            N_array = np.append(N_array, N)

        for n in N_array:
            self.samples_y = samples_y_full[0:n]
            self.get_sobol_sensitivity_indices(verbose=False)
            Si_list.append(self.Si)

        # Restore original results
        self.samples_x = samples_x_full
        self.samples_y = samples_y_full
        self.Si = Si_full
        self.N = N

        # Plot convergence for S1
        Nv = self.problem["num_vars"]
        fig, ax = plt.subplots(nrows=Nv, sharex=True)
        for i in range(Nv):
            ax[i].errorbar(
                x=N_array,
                y=[S["S1"][i] for S in Si_list],
                yerr=[S["S1_conf"][i] for S in Si_list],
                label="First order sensitivity (S1)",
            )
            ax[i].errorbar(
                x=N_array,
                y=[S["ST"][i] for S in Si_list],
                yerr=[S["ST_conf"][i] for S in Si_list],
                label="Total sensitivity (ST)",
            )
            ax[i].plot(
                [0, self.N], [self.Si["S1"][i], self.Si["S1"][i]], "--", color="gray", label=None
            )
            ax[i].plot(
                [0, self.N], [self.Si["ST"][i], self.Si["ST"][i]], "--", color="gray", label=None
            )
            ax[i].legend()
            ax[i].grid("minor")
            ax[i].set_ylabel(self.problem["names"][i])
            ax[i].set_xlabel("Number of iterations")

        if save_path is not None:
            plt.savefig(save_path + "/Sobol_convergence_order1.%s" % fig_format, dpi=fig_dpi)

        if self.calc_second_order:
            for i in range(Nv - 1):
                for j in range(i + 1, Nv):
                    fig, ax = plt.subplots()
                    ax.errorbar(
                        x=N_array,
                        y=[S["S2"][i, j] for S in Si_list],
                        yerr=[S["S2_conf"][i, j] for S in Si_list],
                        label=("Second order sensitivity with %s" % self.problem["names"][j]),
                    )
                    ax.plot(
                        [0, self.N],
                        [self.Si["S2"][i, j], self.Si["S2"][i, j]],
                        "--",
                        color="black",
                        label=None,
                    )
                    ax.legend()
                    ax.set_ylabel(self.problem["names"][i])
                    ax.set_xlabel("Number of iterations")
                    ax.grid("minor")
                    if save_path is not None:
                        plt.savefig(
                            save_path
                            + "/Sobol_convergence_order2_%s.%s"
                            % (self.problem["names"][j], fig_format),
                            dpi=fig_dpi,
                        )

        return fig, ax
