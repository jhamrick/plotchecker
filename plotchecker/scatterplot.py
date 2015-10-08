import numpy as np

from .base import PlotChecker, InvalidPlotError

class ScatterPlotChecker(PlotChecker):
    """A plot checker for scatter plots."""

    def __init__(self, axis):
        super(ScatterPlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.collections = self.axis.collections

        # check that there are only lines or collections, not both
        if len(self.lines) == 0 and len(self.collections) == 0:
            raise InvalidPlotError("No data found")

        # check that if there are lines, linestyle is ''
        for x in self.lines:
            if len(x.get_xydata()) > 1 and x.get_linestyle() != 'None':
                raise InvalidPlotError("This is supposed to be a scatter plot, but it has lines!")
            if self._parse_marker(x.get_marker()) == '':
                raise InvalidPlotError("This is supposed to be a scatter plot, but there are no markers!")

    def assert_num_points(self, num_points):
        """Assert that the plot has the given number of points."""
        if num_points != len(self.x_data):
            raise AssertionError(
                "Plot has incorrect number of points: {} (expected {})".format(
                    len(self.x_data), num_points))

    @property
    def x_data(self):
        all_x_data = []
        if len(self.lines) > 0:
            all_x_data.append(np.concatenate([x.get_xydata()[:, 0] for x in self.lines]))
        if len(self.collections) > 0:
            all_x_data.append(np.concatenate([x.get_offsets()[:, 0] for x in self.collections]))
        return np.concatenate(all_x_data, axis=0)

    def assert_x_data_equal(self, x_data):
        """Assert that the given x_data is equivalent to the plotted x data.
        x_data should be a list or array of numbers with length equal to the
        (expected) number of plotted points.

        """
        np.testing.assert_equal(self.x_data, x_data)

    @property
    def y_data(self):
        all_y_data = []
        if len(self.lines) > 0:
            all_y_data.append(np.concatenate([x.get_xydata()[:, 1] for x in self.lines]))
        if len(self.collections) > 0:
            all_y_data.append(np.concatenate([x.get_offsets()[:, 1] for x in self.collections]))
        return np.concatenate(all_y_data, axis=0)

    def assert_y_data_equal(self, y_data):
        """Assert that the given y_data is equivalent to the plotted y data.
        y_data should be a list or array of numbers with length equal to the
        (expected) number of plotted points.

        """
        np.testing.assert_equal(self.y_data, y_data)

    @property
    def colors(self):
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([self._color2rgb(x.get_markerfacecolor())])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array([self._color2rgb(i) for i in x.get_facecolors()])
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_colors_equal(self, colors):
        """Assert that the given colors are equivalent to the plotted point
        colors. colors should either be a list of colors with length equal to
        the (expected) number of plotted points, or a single color (that should
        apply to all the points). The colors can be given either as RGB arrays,
        matplotlib colors (e.g. 'b' or 'red'), or hex values.

        """
        colors = np.array([self._color2rgb(x) for x in colors])
        if len(colors) == 1:
            colors = self._tile_or_trim(self.x_data, colors)
        np.testing.assert_equal(self.colors, colors)

    @property
    def edgecolors(self):
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([self._color2rgb(x.get_markeredgecolor())])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array([self._color2rgb(i) for i in x.get_edgecolors()])
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_edgecolors_equal(self, edgecolors):
        """Assert that the given edge colors are equivalent to the plotted point
        edge colors. edgecolors should either be a list of colors with length
        equal to the (expected) number of plotted points, or a single color
        (that should apply to all the points). The colors can be given either as
        RGB arrays, matplotlib colors (e.g. 'b' or 'red'), or hex values.

        """
        edgecolors = np.array([self._color2rgb(x) for x in edgecolors])
        if len(edgecolors) == 1:
            edgecolors = self._tile_or_trim(self.x_data, edgecolors)
        np.testing.assert_equal(self.edgecolors, edgecolors)

    @property
    def edgewidths(self):
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([x.get_markeredgewidth()])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array(x.get_linewidths())
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_edgewidths_equal(self, edgewidths):
        """Assert that the given edge widths are equivalent to the plotted point
        edge widths. edgewidths should either be a list of numbers with length
        equal to the (expected) number of plotted points, or a single number
        (that should apply to all the points).

        """
        if not hasattr(edgewidths, '__iter__'):
            edgewidths = np.array([edgewidths])
        if len(edgewidths) == 1:
            edgewidths = self._tile_or_trim(self.x_data, edgewidths)
        np.testing.assert_equal(self.edgewidths, edgewidths)

    @property
    def sizes(self):
        all_sizes = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                sizes = np.array([x.get_markersize() ** 2])
                all_sizes.append(self._tile_or_trim(points, sizes))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                sizes = x.get_sizes()
                all_sizes.append(self._tile_or_trim(points, sizes))

        return np.concatenate(all_sizes, axis=0)

    def assert_sizes_equal(self, sizes):
        """Assert that the given sizes are equivalent to the plotted point
        sizes. sizes should either be a list of numbers with length equal to the
        (expected) number of plotted points, or a single number (that should
        apply to all the points).

        Note: sizes is the square of markersizes.

        """
        np.testing.assert_equal(self.sizes, sizes)

    @property
    def markersizes(self):
        return np.sqrt(self.sizes)

    def assert_markersizes_equal(self, markersizes):
        """Assert that the given marker sizes are equivalent to the plotted
        point marker sizes. markersizes should either be a list of numbers with
        length equal to the (expected) number of plotted points, or a single
        number (that should apply to all the points).

        Note: markersizes is the square root of sizes.

        """
        np.testing.assert_equal(self.markersizes, markersizes)

    @property
    def markers(self):
        raise NotImplementedError("markers are unrecoverable for scatter plots")

    def assert_markers_equal(self, markers):
        """Assert that the markers are equivalent to the plotted markers.
        markers should either be a list of marker styles with length equal to
        the (expected) number of plotted points, or a single marker style (that
        should apply to all the points).

        Note: markers aren't actually recoverable from collections (i.e. when
        `plt.scatter` is used, so for now this assumes that the marker is 'o' if
        it's a collection)

        """
        np.testing.assert_equal(self.markers, markers)

    @property
    def alphas(self):
        all_alphas = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                if x.get_alpha() is None:
                    alpha = np.array([self._color2alpha(x.get_markerfacecolor())])
                else:
                    alpha = np.array([x.get_alpha()])
                all_alphas.append(self._tile_or_trim(points, alpha))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                if x.get_alpha() is None:
                    alpha = np.array([self._color2alpha(i) for i in x.get_facecolors()])
                else:
                    alpha = np.array([x.get_alpha()])
                all_alphas.append(self._tile_or_trim(points, alpha))

        return np.concatenate(all_alphas)

    def assert_alphas_equal(self, alphas):
        """Assert that the given alphas are equivalent to the plotted point
        alpha values. alphas should either be a list of numbers with length
        equal to the (expected) number of plotted points, or a single number
        (that should apply to all the points).

        """
        if not hasattr(alphas, '__iter__'):
            alphas = np.array([alphas])
        if len(alphas) == 1:
            alphas = self._tile_or_trim(self.x_data, alphas)
        np.testing.assert_equal(self.alphas, alphas)
