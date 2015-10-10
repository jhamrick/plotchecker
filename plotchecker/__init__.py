"""
A set of utilities for testing matplotlib plots in an object-oriented manner.
"""

from ._version import version_info, __version__

from .base import PlotChecker, InvalidPlotError
from .lineplot import LinePlotChecker
from .scatterplot import ScatterPlotChecker
from .barplot import BarPlotChecker