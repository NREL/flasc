"""Analyze SCADA data using expected power methods."""


from flasc.analysis.analysis_input import AnalysisInput
from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def _total_uplift_expected_power_single():
    pass


def _total_uplift_expected_power_with_standard_error():
    pass


def _total_uplift_expected_power_with_bootstrapping():
    pass


def total_uplift_expected_power(
    a_in: AnalysisInput,
):
    """Calculate the total uplift in energy production using expected power methods."""
    pass
