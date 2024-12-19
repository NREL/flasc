"""Deprecated EnergyRatioInput class for backwards compatibility."""

import warnings

from flasc.analysis.analysis_input import AnalysisInput


# Define EnergyRatioInput class as a wrapper for AnalysisInput
# that prints a deprecation warning on init
class EnergyRatioInput(AnalysisInput):
    """Deprecated EnergyRatioInput class for backwards compatibility."""

    def __init__(self, *args, **kwargs):
        """Initialize the EnergyRatioInput class with a deprecation warning."""
        warnings.warn(
            "EnergyRatioInput is deprecated. Use AnalysisInput instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
