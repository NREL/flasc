
from pathlib import Path

from floris.tools import FlorisInterface, UncertaintyInterface



def load_smarteole_floris(wake_model="gch", wd_std=0.0):
    """Load a FlorisInterface object for the wind farm at hand.

    Args:
        wake_model (str, optional): The wake model that FLORIS should use. Common
          options are 'cc', 'gch', 'jensen' and 'turbopark'. Defaults to "cc".
        operation_modes (array, optional): Array or list of integers denoting each
          turbine's operation mode. When None is specified, will assume each turbine
          is in its first operation mode (0). Defaults to None.
        wd_std (float, optional): Uncertainty; standard deviation in the inflow
          wind direction in degrees. Defaults to 0.0 deg meaning no uncertainty.

    Returns:
        FlorisInterface: Floris object.
    """

    # Use the local FLORIS GCH/CC model for the wake model settings
    root_path = Path(__file__).resolve().parent / "floris_input"
    fn = root_path / "{:s}.yaml".format(wake_model)

    # Initialize FLORIS model and format appropriately
    fi = FlorisInterface(fn)

    # Add uncertainty
    if wd_std > 0.01:
        unc_options = {
            "std_wd": wd_std,  # Standard deviation for inflow wind direction (deg)
            "pmf_res": 1.0,  # Resolution over which to calculate angles (deg)
            "pdf_cutoff": 0.995,  # Probability density function cut-off (-)
        }
        fi = UncertaintyInterface(fi, unc_options=unc_options)

    return fi
