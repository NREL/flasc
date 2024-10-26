# -*- coding: utf-8 -*-

"""Raw data handling module for FLORIS SCADA Analysis repository."""

__author__ = """Bart Doekemeijer, Paul Fleming"""
__email__ = "paul.fleming@nrel.gov, michael.sinner@nrel.gov"

from importlib.metadata import version

__version__ = version("flasc")

from .flasc_dataframe import FlascDataFrame
