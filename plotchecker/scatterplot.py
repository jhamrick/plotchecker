import numpy as np

from .base import PlotChecker, InvalidPlotError

class ScatterPlotChecker(PlotChecker):
    """A plot checker for scatter plots.

    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)

    """

    def __init__(self, axis):
        """Initialize the scatter plot checker."""

        super(ScatterPlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.collections = self.axis.collections

        # check that there are only lines or collections, not both
        if len(self.lines) == 0 and len(self.collections) == 0:
            raise InvalidPlotError("No data found")

        # check that if there are lines, linestyle is '' and markers are not ''
        for x in self.lines:
            if len(x.get_xydata()) > 1 and x.get_linestyle() != 'None':
                raise InvalidPlotError("This is supposed to be a scatter plot, but it has lines!")
            if self._parse_marker(x.get_marker()) == '':
                raise InvalidPlotError("This is supposed to be a scatter plot, but there are no markers!")

    def _parse_expected_attr(self, attr_name, attr_val):
        """Ensure that the given expected attribute values are in the right shape."""
        if attr_name in ('colors', 'edgecolors'):
            # if it's a color, first check if it's just a single color -- if it's
            # not a single color, this command will throw an error and we can try
            # iterating over the multiple colors that were given
            try:
                attr_val = np.array([self._color2rgb(attr_val)])
            except (ValueError, TypeError):
                attr_val = np.array([self._color2rgb(x) for x in attr_val])

        elif not hasattr(attr_val, '__iter__'):
            # if it's not a color, then just make sure we have an array
            attr_val = np.array([attr_val])

        # tile the given values if we've only been given one, so it's the same
        # shape as the data
        if len(attr_val) == 1:
            attr_val = self._tile_or_trim(self.x_data, attr_val)

        return attr_val

    def assert_num_points(self, num_points):
        """Assert that the plot has the given number of points.

        Parameters
        ----------
        num_points : int

        """
        if num_points != len(self.x_data):
            raise AssertionError(
                "Plot has incorrect number of points: {} (expected {})".format(
                    len(self.x_data), num_points))

    @property
    def x_data(self):
        """The x-values of the plotted data (1-D array)."""
        all_x_data = []
        if len(self.lines) > 0:
            all_x_data.append(np.concatenate([x.get_xydata()[:, 0] for x in self.lines]))
        if len(self.collections) > 0:
            all_x_data.append(np.concatenate([x.get_offsets()[:, 0] for x in self.collections]))
        return np.concatenate(all_x_data, axis=0)

    def assert_x_data_equal(self, x_data):
        """Assert that the given x-data is equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.x_data`.

        Parameters
        ----------
        x_data : 1-D array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted points.

        """
        np.testing.assert_equal(self.x_data, x_data)

    def assert_x_data_allclose(self, x_data, **kwargs):
        """Assert that the given x-data is almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.x_data`.

        Parameters
        ----------
        x_data : 1-D array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(self.x_data, x_data, **kwargs)

    @property
    def y_data(self):
        """The y-values of the plotted data (1-D array)."""
        all_y_data = []
        if len(self.lines) > 0:
            all_y_data.append(np.concatenate([x.get_xydata()[:, 1] for x in self.lines]))
        if len(self.collections) > 0:
            all_y_data.append(np.concatenate([x.get_offsets()[:, 1] for x in self.collections]))
        return np.concatenate(all_y_data, axis=0)

    def assert_y_data_equal(self, y_data):
        """Assert that the given y-data is equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.y_data`.

        Parameters
        ----------
        y_data : 1-D array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted points.

        """
        np.testing.assert_equal(self.y_data, y_data)

    def assert_y_data_allclose(self, y_data, **kwargs):
        """Assert that the given y-data is almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.y_data`.

        Parameters
        ----------
        y_data : 1-D array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(self.y_data, y_data, **kwargs)

    @property
    def colors(self):
        """The colors of the plotted points. Columns correspond to RGB values."""
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
        """Assert that the given colors are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.colors`.

        Parameters
        ----------
        colors : single color, or list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        np.testing.assert_equal(
            self.colors,
            self._parse_expected_attr("colors", colors))

    def assert_colors_allclose(self, colors, **kwargs):
        """Assert that the given colors are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.colors`.

        Parameters
        ----------
        colors : single color, or list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.colors,
            self._parse_expected_attr("colors", colors),
            **kwargs)

    @property
    def alphas(self):
        """The alpha values of the plotted points."""
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
        """Assert that the given alpha values are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.alphas`.

        Parameters
        ----------
        alphas :
            The expected alpha values. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.

        """
        np.testing.assert_equal(
            self.alphas, self._parse_expected_attr("alphas", alphas))

    def assert_alphas_allclose(self, alphas, **kwargs):
        """Assert that the given alpha values are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.alphas`.

        Parameters
        ----------
        alphas :
            The expected alpha values. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.alphas,
            self._parse_expected_attr("alphas", alphas),
            **kwargs)

    @property
    def edgecolors(self):
        """The edge colors of the plotted points. Columns correspond to RGB values."""
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
        """Assert that the given edge colors are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgecolors`.

        Parameters
        ----------
        edgecolors : single color, or list of expected edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        np.testing.assert_equal(
            self.edgecolors,
            self._parse_expected_attr("edgecolors", edgecolors))

    def assert_edgecolors_allclose(self, edgecolors, **kwargs):
        """Assert that the given edge colors are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgecolors`.

        Parameters
        ----------
        edgecolors : single color, or list of expected edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.edgecolors,
            self._parse_expected_attr("edgecolors", edgecolors),
            **kwargs)

    @property
    def edgewidths(self):
        """The edge widths of the plotted points."""
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
        """Assert that the given edge widths are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgewidths`.

        Parameters
        ----------
        edgewidths :
            The expected edge widths. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.

        """
        np.testing.assert_equal(
            self.edgewidths,
            self._parse_expected_attr("edgewidths", edgewidths))

    def assert_edgewidths_allclose(self, edgewidths, **kwargs):
        """Assert that the given edge widths are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgewidths`.

        Parameters
        ----------
        edgewidths :
            The expected edge widths. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.edgewidths,
            self._parse_expected_attr("edgewidths", edgewidths),
            **kwargs)

    @property
    def sizes(self):
        """The size of the plotted points. This is the square of
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.

        """
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
        """Assert that the given point sizes are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.

        Parameters
        ----------
        sizes :
            The expected point sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.

        """
        np.testing.assert_equal(
            self.sizes,
            self._parse_expected_attr("sizes", sizes))

    def assert_sizes_allclose(self, sizes, **kwargs):
        """Assert that the given point sizes are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.

        Parameters
        ----------
        sizes :
            The expected point sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.sizes,
            self._parse_expected_attr("sizes", sizes),
            **kwargs)

    @property
    def markersizes(self):
        """The marker size of the plotted points. This is the square root of
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.

        """
        return np.sqrt(self.sizes)

    def assert_markersizes_equal(self, markersizes):
        """Assert that the given marker sizes are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.

        Parameters
        ----------
        markersizes :
            The expected marker sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.

        """
        np.testing.assert_equal(
            self.markersizes,
            self._parse_expected_attr("markersizes", markersizes))

    def assert_markersizes_allclose(self, markersizes, **kwargs):
        """Assert that the given marker sizes are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.

        Parameters
        ----------
        markersizes :
            The expected marker sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.markersizes,
            self._parse_expected_attr("markersizes", markersizes),
            **kwargs)

    @property
    def markers(self):
        """The marker styles of the plotted points. Unfortunately, this
        information is currently unrecoverable from matplotlib, and so this
        attribute is not actually implemented.

        """
        raise NotImplementedError("markers are unrecoverable for scatter plots")

    def assert_markers_equal(self, markers):
        """Assert that the given marker styles are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markers`.

        Note: information about marker style is currently unrecoverable from
        collections in matplotlib, so this method is not actually implemented.

        Parameters
        ----------
        markers :
            The expected marker styles. This should either be a single style
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.

        """
        np.testing.assert_equal(
            self.markers, self._parse_expected_attr("markers", markers))
