import numpy as np
import itertools
import matplotlib

from .base import PlotChecker, InvalidPlotError


class LinePlotChecker(PlotChecker):
    """A plot checker for line plots."""

    def __init__(self, axis):
        super(LinePlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.perm = list(range(len(self.lines)))

        # check that there are only lines or collections, not both
        if len(self.lines) == 0:
            raise InvalidPlotError("No data found")

    def _assert_equal(self, x, y):
        np.testing.assert_equal(x[self.perm], y)

    def find_permutation(self, attr_name, attr_val):
        if attr_name in ('colors', 'markerfacecolors', 'markeredgecolors'):
            x = np.array([self._color2rgb(i) for i in attr_val])
        else:
            x = np.array(attr_val)

        y = getattr(self, attr_name)
        for perm in itertools.permutations(np.arange(len(x))):
            if (x[list(perm)] == y).all():
                self.perm = list(perm)
                return

        raise AssertionError("Could not find correct permutation of attr '{}'".format(attr_name))

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
        return np.array([x.get_linewidth() for x in self.lines])

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

    @property
    def labels(self):
        legend = self.axis.get_legend()
        if legend is None:
            return np.array([])
        return np.array([x.get_text() for x in legend.texts])

    def assert_labels_equal(self, labels):
        self._assert_equal(np.array(labels), self.labels)

    @property
    def alphas(self):
        all_alphas = np.empty(len(self.lines))
        for i, x in enumerate(self.lines):
            if x.get_alpha() is None:
                all_alphas[i] = self._color2alpha(x.get_color())
            else:
                all_alphas[i] = x.get_alpha()
        return all_alphas

    def assert_alphas_equal(self, alphas):
        if not hasattr(alphas, '__iter__'):
            alphas = np.array([alphas])
        if len(alphas) == 1:
            alphas = self._tile_or_trim(self.x_data, alphas)
        self._assert_equal(np.array(alphas), self.alphas)
