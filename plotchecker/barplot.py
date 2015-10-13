import numpy as np

from .base import PlotChecker, InvalidPlotError


class BarPlotChecker(PlotChecker):
    """A plot checker for bar plots.

    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)

    """

    def __init__(self, axis):
        """Initialize the bar plot checker."""
        super(BarPlotChecker, self).__init__(axis)
        self._patches = np.array(self.axis.patches)
        self._patches = self._patches[np.argsort([p.get_x() for p in self._patches])]

        if len(self._patches) == 0:
            raise InvalidPlotError("no data found")

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
            attr_val = self._tile_or_trim(self.centers, attr_val)

        return attr_val

    def assert_num_bars(self, num_bars):
        """Assert that the plot has the given number of bars.

        Parameters
        ----------
        num_bars : int

        """
        if num_bars != len(self._patches):
            raise AssertionError(
                "Plot has incorrect number of bars: {} (expected {})".format(
                    len(self._patches), num_bars))

    @property
    def centers(self):
        """The centers of the plotted bars."""
        return np.array([p.get_x() + (p.get_width() / 2) for p in self._patches])

    def assert_centers_equal(self, centers):
        """Assert that the given centers are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.centers`.

        Parameters
        ----------
        centers : 1-D array-like
            The expected centers. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.centers,
            self._parse_expected_attr("centers", centers))

    def assert_centers_allclose(self, centers, **kwargs):
        """Assert that the given centers are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.centers`.

        Parameters
        ----------
        centers : 1-D array-like
            The expected centers. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.centers,
            self._parse_expected_attr("centers", centers),
            **kwargs)

    @property
    def heights(self):
        """The heights of the plotted bars."""
        return np.array([p.get_height() for p in self._patches])

    def assert_heights_equal(self, heights):
        """Assert that the given heights are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.heights`.

        Parameters
        ----------
        heights : 1-D array-like
            The expected heights. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.heights,
            self._parse_expected_attr("heights", heights))

    def assert_heights_allclose(self, heights, **kwargs):
        """Assert that the given heights are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.heights`.

        Parameters
        ----------
        heights : 1-D array-like
            The expected heights. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.heights,
            self._parse_expected_attr("heights", heights),
            **kwargs)

    @property
    def widths(self):
        """The widths of the plotted bars."""
        return np.array([p.get_width() for p in self._patches])

    def assert_widths_equal(self, widths):
        """Assert that the given widths are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.widths`.

        Parameters
        ----------
        widths : 1-D array-like
            The expected widths. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.widths,
            self._parse_expected_attr("widths", widths))

    def assert_widths_allclose(self, widths, **kwargs):
        """Assert that the given widths are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.widths`.

        Parameters
        ----------
        widths : 1-D array-like
            The expected widths. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.widths,
            self._parse_expected_attr("widths", widths),
            **kwargs)

    @property
    def bottoms(self):
        """The y-coordinates of the bottoms of the plotted bars."""
        return np.array([p.get_y() for p in self._patches])

    def assert_bottoms_equal(self, bottoms):
        """Assert that the given bottoms are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.bottoms`.

        Parameters
        ----------
        bottoms : 1-D array-like
            The expected bottoms. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.bottoms,
            self._parse_expected_attr("bottoms", bottoms))

    def assert_bottoms_allclose(self, bottoms, **kwargs):
        """Assert that the given bottoms are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.bottoms`.

        Parameters
        ----------
        bottoms : 1-D array-like
            The expected bottoms. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.bottoms,
            self._parse_expected_attr("bottoms", bottoms),
            **kwargs)

    @property
    def colors(self):
        """The colors of the plotted bars."""
        return np.array([self._color2rgb(p.get_facecolor()) for p in self._patches])

    def assert_colors_equal(self, colors):
        """Assert that the given colors are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.colors`.

        Parameters
        ----------
        colors : single color, or list of expected colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        np.testing.assert_equal(
            self.colors,
            self._parse_expected_attr("colors", colors))

    def assert_colors_allclose(self, colors, **kwargs):
        """Assert that the given colors are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.colors`.

        Parameters
        ----------
        colors : single color, or list of expected edge colors
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
    def edgecolors(self):
        """The edge colors of the plotted bars."""
        return np.array([self._color2rgb(p.get_edgecolor()) for p in self._patches])

    def assert_edgecolors_equal(self, edgecolors):
        """Assert that the given edgecolors are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.edgecolors`.

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
        """Assert that the given edgecolors are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.edgecolors`.

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
    def alphas(self):
        """The alpha values of the plotted bars."""
        all_alphas = []

        for p in self._patches:
            if p.get_alpha() is None:
                alpha = self._color2alpha(p.get_facecolor())
            else:
                alpha = p.get_alpha()
            all_alphas.append(alpha)

        return np.array(all_alphas)

    def assert_alphas_equal(self, alphas):
        """Assert that the given alphas are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.alphas`.

        Parameters
        ----------
        alphas : 1-D array-like
            The expected alphas. The number of elements should be equal to
            the (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.alphas,
            self._parse_expected_attr("alphas", alphas))

    def assert_alphas_allclose(self, alphas, **kwargs):
        """Assert that the given alphas are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.alphas`.

        Parameters
        ----------
        alphas : 1-D array-like
            The expected alphas. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.alphas,
            self._parse_expected_attr("alphas", alphas),
            **kwargs)

    @property
    def linewidths(self):
        """The line widths of the plotted bars."""
        return np.array([p.get_linewidth() for p in self._patches])

    def assert_linewidths_equal(self, linewidths):
        """Assert that the given linewidths are equivalent to the plotted
        :attr:`~plotchecker.BarPlotChecker.linewidths`.

        Parameters
        ----------
        linewidths : 1-D array-like
            The expected linewidths. The number of elements should be equal to
            the (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).

        """
        np.testing.assert_equal(
            self.linewidths,
            self._parse_expected_attr("linewidths", linewidths))

    def assert_linewidths_allclose(self, linewidths, **kwargs):
        """Assert that the given linewidths are almost equal to the plotted
        :attr:`~plotchecker.BarPlotChecker.linewidths`.

        Parameters
        ----------
        linewidths : 1-D array-like
            The expected linewidths. The number of elements should be equal to the
            (expected) number of plotted bars, or just a single value (which
            will then be applied to all bars).
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(
            self.linewidths,
            self._parse_expected_attr("linewidths", linewidths),
            **kwargs)

