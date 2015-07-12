import numpy as np
import itertools

from .base import PlotChecker, InvalidPlotError


class LinePlotChecker(PlotChecker):
    """A plot checker for line plots."""

    def __init__(self, axis, try_permutations=False):
        super(LinePlotChecker, self).__init__(axis)
        self.try_permutations = try_permutations
        self.lines = self.axis.get_lines()

        # check that there are only lines or collections, not both
        if len(self.lines) == 0:
            raise InvalidPlotError("No data found")

    def _assert_equal(self, x, y):
        if self.try_permutations:
            for perm in itertools.permutations(np.arange(len(x))):
                if (x[list(perm)] == y).all():
                    return
            raise AssertionError

        else:
            np.testing.assert_equal(x, y)

    @property
    def x_data(self):
        return np.array([x.get_xydata()[:, 0] for x in self.lines]).T

    def assert_x_data_equal(self, x_data):
        self._assert_equal(x_data.T, self.x_data.T)

    @property
    def y_data(self):
        return np.array([x.get_xydata()[:, 1] for x in self.lines]).T

    def assert_y_data_equal(self, y_data):
        self._assert_equal(y_data.T, self.y_data.T)

    @property
    def colors(self):
        return np.array([self._color2rgb(x.get_color()) for x in self.lines])

    def assert_colors_equal(self, colors):
        colors = np.array([self._color2rgb(x) for x in colors])
        if len(colors) == 1:
            colors = self._tile_or_trim(self.x_data, colors)
        self._assert_equal(colors, self.colors)

    @property
    def linewidths(self):
        return np.array([self._color2rgb(x.get_linewidth()) for x in self.lines])

    def assert_linewidths_equal(self, linewidths):
        self._assert_equal(linewidths, self.linewidths)

    @property
    def markerfacecolors(self):
        return np.array([self._color2rgb(x.get_markerfacecolor()) for x in self.lines])

    def assert_markerfacecolors_equal(self, markerfacecolors):
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        if len(markerfacecolors) == 1:
            markerfacecolors = self._tile_or_trim(self.x_data, markerfacecolors)
        self._assert_equal(markerfacecolors, self.markerfacecolors)

    @property
    def markeredgecolors(self):
        return np.array([self._color2rgb(x.get_markeredgecolor()) for x in self.lines])

    def assert_markeredgecolors_equal(self, markeredgecolors):
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        if len(markeredgecolors) == 1:
            markeredgecolors = self._tile_or_trim(self.x_data, markeredgecolors)
        self._assert_equal(markeredgecolors, self.markeredgecolors)

    @property
    def markeredgewidths(self):
        return np.array([x.get_markeredgewidth() for x in self.lines])

    def assert_markeredgewidths_equal(self, markeredgewidths):
        if not hasattr(markeredgewidths, '__iter__'):
            markeredgewidths = np.array([markeredgewidths])
        if len(markeredgewidths) == 1:
            markeredgewidths = self._tile_or_trim(self.x_data, markeredgewidths)
        self._assert_equal(markeredgewidths, self.markeredgewidths)

    @property
    def markersizes(self):
        return np.array([x.get_markersize() for x in self.lines])

    def assert_markersizes_equal(self, markersizes):
        self._assert_equal(markersizes, self.markersizes)

    @property
    def markers(self):
        return np.array([x.get_marker() for x in self.lines])

    def assert_markers_equal(self, markers):
        self._assert_equal(np.array(markers), self.markers)