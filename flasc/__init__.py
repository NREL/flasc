# -*- coding: utf-8 -*-

"""Raw data handling module for FLORIS SCADA Analysis repository."""

__author__ = """Bart Doekemeijer, Paul Fleming"""
__email__ = "paul.fleming@nrel.gov, michael.sinner@nrel.gov"
__version__ = "0.1.0"

from pathlib import Path

with open(Path(__file__).parent / "version.py") as _version_file:
    __version__ = _version_file.read().strip()

# from . import (
#     optimization,
#     visualization,
#     yaw_optimizer_visualization,
# )
