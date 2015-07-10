import numpy as np

from .base import PlotChecker, InvalidPlotError


class LinePlotChecker(PlotChecker):
    """A plot checker for line plots."""

    def __init__(self, axis):
        super(LinePlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()

        # check that there are only lines or collections, not both
        if len(self.lines) == 0:
            raise InvalidPlotError("No data found")

    @property
    def x_data(self):
        return np.array([x.get_xydata()[:, 0] for x in self.lines]).T

    def assert_x_data_equal(self, x_data):
        np.testing.assert_equal(x_data, self.x_data)

    def assert_x_data_almost_equal(self, x_data):
        np.testing.assert_almost_equal(x_data, self.x_data)

    @property
    def y_data(self):
        return np.array([x.get_xydata()[:, 1] for x in self.lines]).T

    def assert_y_data_equal(self, y_data):
        np.testing.assert_equal(y_data, self.y_data)

    def assert_y_data_almost_equal(self, y_data):
        np.testing.assert_almost_equal(y_data, self.y_data)

    @property
    def colors(self):
        return np.array([self._color2rgb(x.get_color()) for x in self.lines])

    def assert_colors_equal(self, colors):
        colors = np.array([self._color2rgb(x) for x in colors])
        if len(colors) == 1:
            colors = self._tile_or_trim(self.x_data, colors)
        np.testing.assert_equal(colors, self.colors)

    def assert_colors_almost_equal(self, colors):
        colors = np.array([self._color2rgb(x) for x in colors])
        if len(colors) == 1:
            colors = self._tile_or_trim(self.x_data, colors)
        np.testing.assert_almost_equal(colors, self.colors)

    @property
    def linewidths(self):
        return np.array([self._color2rgb(x.get_linewidth()) for x in self.lines])

    def assert_linewidths_equal(self, linewidths):
        np.testing.assert_equal(linewidths, self.linewidths)

    def assert_linewidths_almost_equal(self, linewidths):
        np.testing.assert_almost_equal(linewidths, self.linewidths)

    @property
    def markerfacecolors(self):
        return np.array([self._color2rgb(x.get_markerfacecolor()) for x in self.lines])

    def assert_markerfacecolors_equal(self, markerfacecolors):
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        if len(markerfacecolors) == 1:
            markerfacecolors = self._tile_or_trim(self.x_data, markerfacecolors)
        np.testing.assert_equal(markerfacecolors, self.markerfacecolors)

    def assert_markerfacecolors_almost_equal(self, markerfacecolors):
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        if len(markerfacecolors) == 1:
            markerfacecolors = self._tile_or_trim(self.x_data, markerfacecolors)
        np.testing.assert_almost_equal(markerfacecolors, self.markerfacecolors)

    @property
    def markeredgecolors(self):
        return np.array([self._color2rgb(x.get_markeredgecolor()) for x in self.lines])

    def assert_markeredgecolors_equal(self, markeredgecolors):
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        if len(markeredgecolors) == 1:
            markeredgecolors = self._tile_or_trim(self.x_data, markeredgecolors)
        np.testing.assert_equal(markeredgecolors, self.markeredgecolors)

    def assert_markeredgecolors_almost_equal(self, markeredgecolors):
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        if len(markeredgecolors) == 1:
            markeredgecolors = self._tile_or_trim(self.x_data, markeredgecolors)
        np.testing.assert_almost_equal(markeredgecolors, self.markeredgecolors)

    @property
    def markeredgewidths(self):
        return np.array([x.get_markeredgewidth() for x in self.lines])

    def assert_markeredgewidths_equal(self, markeredgewidths):
        if not hasattr(markeredgewidths, '__iter__'):
            markeredgewidths = np.array([markeredgewidths])
        if len(markeredgewidths) == 1:
            markeredgewidths = self._tile_or_trim(self.x_data, markeredgewidths)
        np.testing.assert_equal(markeredgewidths, self.markeredgewidths)

    def assert_markeredgewidths_almost_equal(self, markeredgewidths):
        if not hasattr(markeredgewidths, '__iter__'):
            markeredgewidths = np.array([markeredgewidths])
        if len(markeredgewidths) == 1:
            markeredgewidths = self._tile_or_trim(self.x_data, markeredgewidths)
        np.testing.assert_almost_equal(markeredgewidths, self.markeredgewidths)

    @property
    def markersizes(self):
        return np.array([x.get_markersize() for x in self.lines])

    def assert_markersizes_equal(self, markersizes):
        np.testing.assert_equal(markersizes, self.markersizes)

    def assert_markersizes_almost_equal(self, markersizes):
        np.testing.assert_almost_equal(markersizes, self.markersizes)

    @property
    def markers(self):
        return np.array([x.get_marker() for x in self.lines])

    def assert_markers_equal(self, markers):
        np.testing.assert_equal(markers, self.markers)
