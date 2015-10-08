import numpy as np
import itertools

from .base import PlotChecker, InvalidPlotError


class LinePlotChecker(PlotChecker):
    """A plot checker for line plots."""

    def __init__(self, axis):
        super(LinePlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.perm = list(range(len(self.lines)))

        # check that there are some lines plotted
        if len(self.lines) == 0:
            raise InvalidPlotError("No data found")

    def _assert_equal(self, attr, expected, actual, perm=None):
        if len(expected) != len(actual):
            raise AssertionError(
                "Invalid length for attribute '{}': {} (expected {})".format(
                    attr, len(actual), len(expected)))

        if perm is None:
            perm = self.perm

        for i in range(len(expected)):
            try:
                np.testing.assert_equal(actual[i], expected[perm[i]])
            except AssertionError:
                raise AssertionError(
                    "Attribute '{}' does not match for line {} (expected: {}, actual: {})".format(
                        attr, i, expected[perm[i]], actual[i]))

    def find_permutation(self, attr_name, attr_vals):
        """Find the order of the lines such that the plotted attribute (given
        by `attr_name`) has values in the order given by `attr_vals`.

        """
        if attr_name in ('colors', 'markerfacecolors', 'markeredgecolors'):
            expected = np.array([self._color2rgb(i) for i in attr_vals])
        else:
            expected = attr_vals

        actual = getattr(self, attr_name)

        if len(expected) != len(actual):
            raise AssertionError(
                "Invalid length for attribute '{}': {} (expected {})".format(
                    attr_name, len(actual), len(expected)))

        for perm in itertools.permutations(np.arange(len(expected))):
            try:
                self._assert_equal(attr_name, expected, actual, perm=perm)
            except AssertionError:
                pass
            else:
                self.perm = perm
                return

        raise AssertionError(
            "Could not match plotted values {} to expected values {} for attr '{}'".format(
                actual, expected, attr_name))

    def assert_num_lines(self, num_lines):
        """Assert that the plot has the given number of lines."""
        if num_lines != len(self.lines):
            raise AssertionError(
                "Plot has incorrect number of lines: {} (expected {})".format(
                    len(self.lines), num_lines))

    @property
    def x_data(self):
        return [x.get_xydata()[:, 0] for x in self.lines]

    def assert_x_data_equal(self, x_data):
        """Assert that the given x_data is equivalent to the plotted x data.
        x_data should be a list of lists/arrays, where the number of
        lists/arrays is equal to the (expected) number of plotted lines.

        """
        self._assert_equal("x_data", x_data, self.x_data)

    @property
    def y_data(self):
        return [x.get_xydata()[:, 1] for x in self.lines]

    def assert_y_data_equal(self, y_data):
        """Assert that the given y_data is equivalent to the plotted y data.
        y_data should be a list of lists/arrays, where the number of
        lists/arrays is equal to the (expected) number of plotted lines.

        """
        self._assert_equal("y_data", y_data, self.y_data)

    @property
    def colors(self):
        return np.array([self._color2rgb(x.get_color()) for x in self.lines])

    def assert_colors_equal(self, colors):
        """Assert that the given colors are equivalent to the plotted line
        colors. colors should be a list of colors with length equal to the
        (expected) number of plotted lines. The colors can be given either as
        RGB arrays, matplotlib colors (e.g. 'b' or 'red'), or hex values.

        """
        colors = np.array([self._color2rgb(x) for x in colors])
        self._assert_equal("colors", colors, self.colors)

    @property
    def linewidths(self):
        return [x.get_linewidth() for x in self.lines]

    def assert_linewidths_equal(self, linewidths):
        """Assert that the given line widths are equivalent to the plotted line
        widths. linewidths should be a list with length equal to the (expected)
        number of plotted lines.

        """
        self._assert_equal("linewidths", linewidths, self.linewidths)

    @property
    def markerfacecolors(self):
        return [self._color2rgb(x.get_markerfacecolor()) for x in self.lines]

    def assert_markerfacecolors_equal(self, markerfacecolors):
        """Assert that the given colors are equivalent to the plotted marker
        face colors. markerfacecolors should be a list of colors with length
        equal to the (expected) number of plotted lines. The colors can be given
        either as RGB arrays, matplotlib colors (e.g. 'b' or 'red'), or hex
        values.

        """
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        self._assert_equal("markerfacecolors", markerfacecolors, self.markerfacecolors)

    @property
    def markeredgecolors(self):
        return [self._color2rgb(x.get_markeredgecolor()) for x in self.lines]

    def assert_markeredgecolors_equal(self, markeredgecolors):
        """Assert that the given colors are equivalent to the plotted marker
        edge colors. markeredgecolors should be a list of colors with length
        equal to the (expected) number of plotted lines. The colors can be given
        either as RGB arrays, matplotlib colors (e.g. 'b' or 'red'), or hex
        values.

        """
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        self._assert_equal("markeredgecolors", markeredgecolors, self.markeredgecolors)

    @property
    def markeredgewidths(self):
        return [x.get_markeredgewidth() for x in self.lines]

    def assert_markeredgewidths_equal(self, markeredgewidths):
        """Assert that the given marker edge widths are equivalent to the
        plotted marker edge widths. markeredgewidths should be a list with
        length equal to the (expected) number of plotted lines.

        """
        self._assert_equal("markeredgewidths", markeredgewidths, self.markeredgewidths)

    @property
    def markersizes(self):
        return [x.get_markersize() for x in self.lines]

    def assert_markersizes_equal(self, markersizes):
        """Assert that the given marker sizes are equivalent to the plotted
        marker sizes. markersizes should be a list with length equal to the
        (expected) number of plotted lines.

        """
        self._assert_equal("markersizes", markersizes, self.markersizes)

    @property
    def markers(self):
        return [self._parse_marker(x.get_marker()) for x in self.lines]

    def assert_markers_equal(self, markers):
        """Assert that the given markers are equivalent to the plotted markers.
        markers should be a list with length equal to the (expected) number of
        plotted lines.

        """
        markers = [self._parse_marker(x) for x in markers]
        self._assert_equal("markers", markers, self.markers)

    @property
    def labels(self):
        legend = self.axis.get_legend()
        if legend is None:
            return []
        return [x.get_text() for x in legend.texts]

    def assert_labels_equal(self, labels):
        """Assert that the given labels are equivalent to the legend labels.
        labels should be a list with length equal to the (expected) number of
        plotted lines.

        """
        self._assert_equal("labels", labels, self.labels)

    @property
    def alphas(self):
        all_alphas = []
        for x in self.lines:
            if x.get_alpha() is None:
                all_alphas.append(self._color2alpha(x.get_color()))
            else:
                all_alphas.append(x.get_alpha())
        return all_alphas

    def assert_alphas_equal(self, alphas):
        """Assert that the given alphas are equivalent to the plotted alphas.
        alphas should be a list of alpha values with length equal to the
        (expected) number of plotted lines.

        """
        self._assert_equal("alphas", alphas, self.alphas)
