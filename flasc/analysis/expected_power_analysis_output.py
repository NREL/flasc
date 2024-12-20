"""Output module for expected power analysis methods."""

import pandas as pd

from flasc.analysis.analysis_input import AnalysisInput


class ExpectedPowerAnalysisOutput:
    """Store the results of the expected power analysis calculations.

    Additionally provide convenient methods for plotting and saving the results.
    """

    def __init__(
        self,
        uplift_results: dict,
        a_in: AnalysisInput,
        test_turbines: list = None,
        wd_turbines: list = None,
        ws_turbines: list = None,
        use_predefined_wd: bool = False,
        use_predefined_ws: bool = False,
        wd_step: float = 2.0,
        wd_min: float = 0.0,
        wd_max: float = 360.0,
        ws_step: float = 1.0,
        ws_min: float = 0.0,
        ws_max: float = 50.0,
        bin_cols_in: list = ["wd_bin", "ws_bin"],
        weight_by: str = "min",  # min or sum
        df_freq: pd.DataFrame = None,
        uplift_pairs: list = None,
        uplift_names: list = None,
        use_standard_error: bool = True,
        N: int = 1,
        percentiles: list = [2.5, 97.5],
        remove_any_null_turbine_bins: bool = False,
        cov_terms: str = "zero",
    ) -> None:
        """Init the object with the results of the uplift analysis."""
        self.uplift_results = uplift_results
        self.a_in = a_in
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
        self.uplift_pairs = uplift_pairs
        self.uplift_names = uplift_names
        self.use_standard_error = use_standard_error
        self.N = N
        self.percentiles = percentiles
        self.remove_any_null_turbine_bins = remove_any_null_turbine_bins
        self.cov_terms = cov_terms

    def _return_uplift_string(self):
        return (
            f"{self.uplift_results['scada_uplift']['energy_uplift_ctr_pc']:+0.2f}%, ("
            f"{self.uplift_results['scada_uplift']['energy_uplift_lb_pc']:+0.2f}% - "
            f"{self.uplift_results['scada_uplift']['energy_uplift_ub_pc']:+0.2f}%)"
            f" -- N={self.uplift_results['scada_uplift']['count']}"
        )

    def print_uplift(self):
        """Print the uplift results."""
        print(self._return_uplift_string())
