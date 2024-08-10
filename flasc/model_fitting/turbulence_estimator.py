"""_summary_."""

import floris as wfct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.logging_manager import LoggingManager
from flasc.utilities import floris_tools as ftools, optimization as opt

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# TODO IS THIS USED ANYWHERE?
class ti_estimator:
    """_summary_."""

    def __init__(self, fm):
        """_summary_.

        Args:
            fm (_type_): _description_
        """
        self.fm = fm
        self.num_turbs = len(fm.layout_x)

        self._reset_outputs()

    def _reset_outputs(self):
        self.opt_farm = None
        self.opt_turbines = None
        self.turbine_list_ordered = None
        self.turbine_pairs = None
        self.P_measured = None

    def set_measurements(self, P_measured):
        """_summary_.

        Args:
            P_measured (_type_): _description_
        """
        if isinstance(P_measured, int) | isinstance(P_measured, float):
            P_measured = [P_measured]
        if isinstance(P_measured, list):
            P_measured = np.array(P_measured)

        self._reset_outputs()
        self.P_measured = P_measured

    def get_turbine_order(self):
        """_summary_.

        Returns:
            _type_: _description_
        """
        wd = (180 - self.fm.core.farm.wind_direction[0]) * np.pi / 180.0
        rotz = np.matrix([[np.cos(wd), -np.sin(wd), 0], [np.sin(wd), np.cos(wd), 0], [0, 0, 1]])
        x0 = np.mean(self.fm.layout_x)
        y0 = np.mean(self.fm.layout_y)

        xyz_init = np.matrix(
            [
                np.array(self.fm.layout_x) - x0,
                np.array(self.fm.layout_y) - y0,
                [0.0 for _ in range(self.num_turbs)],
            ]
        )

        xyz_rot = rotz * xyz_init
        x_rot = np.array(xyz_rot[0, :])[0]
        turbine_list_ordered = np.argsort(x_rot)

        self.turbine_list_ordered = turbine_list_ordered
        return turbine_list_ordered

    def get_turbine_pairs(self, wake_loss_thrs=0.20):
        """_summary_.

        Args:
            wake_loss_thrs (float, optional): _description_. Defaults to 0.20.

        Returns:
            _type_: _description_
        """
        fm = self.fi
        fm.run()
        power_baseline = np.array(fm.get_turbine_power())
        disabled_turb_cp_ct = {
            "wind_speed": [0.0, 50.0],
            "power": [0.0, 0.0],
            "thrust_coefficient": [0.0001, 0.0001],
        }
        regular_turb_cp_ct = fm.core.farm.turbines[0].power_thrust_table
        df_pairs = pd.DataFrame(
            {"turbine": pd.Series([], dtype="int"), "affected_turbines": pd.Series([], dtype="int")}
        )
        for ti in range(self.num_turbs):
            fm.change_turbine([ti], {"power_thrust_table": disabled_turb_cp_ct})
            fm.run()
            power_excl = np.array(fm.get_turbine_power())
            power_excl[ti] = power_baseline[ti]  # Placeholder
            wake_losses = 1 - power_baseline / power_excl
            affectedturbs = np.where(wake_losses >= wake_loss_thrs)[0]
            df_pairs = df_pairs.append(
                {"turbine": int(ti), "affected_turbines": affectedturbs}, ignore_index=True
            )
            fm.change_turbine([ti], {"power_thrust_table": regular_turb_cp_ct})

        # Save to self
        df_pairs = df_pairs.set_index("turbine", drop=True)
        self.turbine_pairs = df_pairs
        return df_pairs

    def plot_flowfield(self):
        """_summary_.

        Returns:
            _type_: _description_
        """
        self.fm.run()
        fig, ax = plt.subplots()
        hor_plane = self.fm.get_hor_plane()
        wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        return fig, ax, hor_plane

    def floris_set_ws_wd_ti(self, wd=None, ws=None, ti=None):
        """_summary_.

        Args:
            wd (_type_, optional): _description_. Defaults to None.
            ws (_type_, optional): _description_. Defaults to None.
            ti (_type_, optional): _description_. Defaults to None.
        """
        self.fm = ftools._fi_set_ws_wd_ti(self.fi, wd=wd, ws=ws, ti=ti)

    def _check_measurements(self):
        if self.P_measured is None:
            raise ValueError(
                "Please specify measurements using .set_measurements(P_measured) "
                "before attempting to estimate the turbulence intensity."
            )

    def estimate_farmaveraged_ti(
        self, Ns=50, bounds=(0.01, 0.50), refine_with_fmin=False, verbose=False
    ):
        """_summary_.

        Args:
            Ns (int, optional): _description_. Defaults to 50.
            bounds (tuple, optional): _description_. Defaults to (0.01, 0.50).
            refine_with_fmin (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        self._check_measurements()
        out = opt.estimate_ti(
            fi=self.fi,
            P_measured=self.P_measured,
            Ns=Ns,
            bounds=bounds,
            turbine_upstream=range(self.num_turbs),
            turbines_downstream=range(self.num_turbs),
            refine_with_fmin=refine_with_fmin,
            verbose=verbose,
        )

        self.opt_farm = out
        ti_opt = out["x_opt"]
        self.floris_set_ws_wd_ti(ti=ti_opt)
        logger.info("Optimal farm-averaged ti: %.3f" % ti_opt)

        return ti_opt

    def estimate_local_tis(self, Ns=50, bounds=(0.01, 0.50), refine_with_fmin=False, verbose=False):
        """_summary_.

        Args:
            Ns (int, optional): _description_. Defaults to 50.
            bounds (tuple, optional): _description_. Defaults to (0.01, 0.50).
            refine_with_fmin (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        self._check_measurements()
        turbines_sorted = self.turbine_list_ordered
        df_turbine_pairs = self.turbine_pairs

        out_array = [[] for _ in range(self.num_turbs)]
        ti_array = np.repeat(self.opt_farm["x_opt"], self.num_turbs)
        for ti in turbines_sorted:
            turbs_aff = df_turbine_pairs.loc[ti, "affected_turbines"]
            if len(turbs_aff) > 0:
                out = opt.estimate_ti(
                    fi=self.fi,
                    P_measured=self.P_measured[turbs_aff],
                    Ns=Ns,
                    bounds=bounds,
                    turbine_upstream=ti,
                    turbines_downstream=turbs_aff,
                    refine_with_fmin=refine_with_fmin,
                    verbose=verbose,
                )
                ti_array[ti] = out["x_opt"]
                self.floris_set_ws_wd_ti(ti=ti_array)
            else:
                out = {"x_opt": self.opt_farm["x_opt"], "J_opt": np.nan, "x": [], "J": []}

            out_array[ti] = out

        self.opt_turbines = out_array
        for ti in range(self.num_turbs):
            logger.info("Optimal ti for turbine %03d: %.3f" % (ti, ti_array[ti]))

        return out_array

    def plot_cost_function_farm(self):
        """_summary_."""
        fig, ax = plt.subplots()
        ax.plot(self.opt_farm["x"], self.opt_farm["J"])
        ax.plot(self.opt_farm["x_opt"], self.opt_farm["J_opt"], "ro")
        ax.set_ylabel("Cost function")
        ax.set_xlabel("Turbulence intensity (-)")
        ax.grid(True)
        ax.set_title("Farm-wide turbulence intensity estimation: cost function J")

    def plot_cost_functions_turbines(self):
        """_summary_."""
        for ti in range(self.num_turbs):
            fig, ax = plt.subplots()
            ax.plot(self.opt_turbines[ti]["x"], self.opt_turbines[ti]["J"])
            ax.plot(self.opt_turbines[ti]["x_opt"], self.opt_turbines[ti]["J_opt"], "ro")
            ax.set_ylabel("Cost function")
            ax.set_xlabel("Turbulence intensity (-)")
            ax.grid(True)
            ax.set_title("Turbulence intensity estimation for turbine %03d: cost function J" % ti)

    def plot_power_bars(self):
        """_summary_.

        Returns:
            _type_: _description_
        """
        fm = self.fi
        fm.run()

        fig, ax = plt.subplots()
        ax.bar(x=np.array(range(self.num_turbs)) - 0.15, height=fm.get_turbine_power(), width=0.3)
        ax.bar(x=np.array(range(self.num_turbs)) + 0.15, height=self.P_measured, width=0.3)
        ax.set_title("Measurement and FLORIS comparison")
        ax.set_ylabel("Power")
        ax.set_xlabel("Turbine number")
        ax.legend(["FLORIS", "SCADA"])
        return fig, ax
