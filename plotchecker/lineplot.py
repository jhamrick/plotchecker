import numpy as np
import itertools

from .base import PlotChecker, InvalidPlotError


class LinePlotChecker(PlotChecker):
    """A plot checker for line plots.

    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)

    """

    def __init__(self, axis):
        """Initialize the line plot checker."""
        super(LinePlotChecker, self).__init__(axis)
        self._lines = self.axis.get_lines()
        self._perm = list(range(len(self._lines)))

        # check that there are some lines plotted
        if len(self._lines) == 0:
            raise InvalidPlotError("No data found")

    def _assert_equal(self, attr, expected, actual, perm=None, func=None, **kwargs):
        """Helper method for asserting that attributes are equal. This should
        only really be called by other methods in this class.

        Because the plot may have multiple lines, we want to first check whether
        the number of expected attributes is the same as the actual number of
        expected attributes. Then, assuming they are, we go check the equality
        of the attribute for each line separately. That way we can give a useful
        error message to users about which line, specifically, doesn't match.

        Parameters
        ----------
        attr : string
            The name of the attribute
        expected :
            The expected value of the attribute
        actual :
            The actual value of the attribute
        perm : list of integers (default: ``None``)
            The permutation between the expected lines and actual lines. If not
            given, then ``self._perm`` is used.
        func : function (default=``numpy.testing.assert_equal``)
            An assertion function to check for equality.
        kwargs :
            Additional keyword arguments to pass to ``func``

        """
        # first check that the number of attributes matches
        if len(expected) != len(actual):
            raise AssertionError(
                "Invalid length for attribute '{}': {} (expected {})".format(
                    attr, len(actual), len(expected)))

        # figure out the value of the permutation
        if perm is None:
            perm = self._perm

        # set the default value of the function, if necessary
        if func is None:
            func = np.testing.assert_equal

        # check each line separately
        for i in range(len(expected)):
            try:
                func(actual[i], expected[perm[i]], **kwargs)
            except AssertionError:
                raise AssertionError(
                    "Attribute '{}' does not match for line {} (expected: {}, actual: {})".format(
                        attr, i, expected[perm[i]], actual[i]))

    def _assert_allclose(self, attr, expected, actual, perm=None, **kwargs):
        """Wrapper for ``self._assert_equal`` that passes
        ``numpy.testing.assert_allclose`` as the assertion function.

        """
        self._assert_equal(
            attr, expected, actual,
            perm=perm,
            func=np.testing.assert_allclose,
            **kwargs)

    def find_permutation(self, attr_name, attr_vals):
        """Find the order of the lines such that the given attribute (given
        by ``attr_name`` and ``attr_vals``) has values in the same order as those
        that were plotted.

        This function is useful if you are not sure what order the lines were
        plotted in. If, for example, you know there should be one red line, one
        blue line, and one green line---but not the order---you can use this
        method with ``'color'`` as the attribute name and ``['red', 'green',
        'blue']`` as the attribute values. Then, any subsequent assertions will
        use the permutation found by this function.

        Parameters
        ----------
        attr_name : string
            The name of the attribute to use for finding the permutation.
        attr_vals :
            The expected values of the attribute

        Examples
        --------

        Let's say the instructions are to create three lines: one red line with
        data ``xr`` and ``yr``; one blue line with data ``xb`` and ``yb``, and
        one green line with data ``xg`` and ``yg``. Here is one way that plot
        might be created:

        .. code:: python

            fig, ax = plt.subplots()
            ax.plot(xg, yg, 'g-')
            ax.plot(xb, yb, 'b-')
            ax.plot(xr, yr, 'r-')

        However, the lines could have been plotted in any order. So, for
        example, the following would fail, because it assumes the order is
        'red', 'green', and 'blue':

        .. code:: python

            pc = LinePlotChecker(ax)
            pc.assert_x_data_equal([xr, xg, xb])    # fails

        To avoid having to check every permutation, you can use this
        ``find_permutation`` method to do it for you:

        .. code:: python

            pc = LinePlotChecker(ax)
            pc.find_permutation('color', ['r', 'g', 'b'])
            pc.assert_x_data_equal([xr, xg, xb])    # passes

        """
        # if the attribute is one of the color attributes, we need to make sure
        # to properly parse the values into RGB tuples
        if attr_name in ('colors', 'markerfacecolors', 'markeredgecolors'):
            expected = np.array([self._color2rgb(i) for i in attr_vals])
        else:
            expected = attr_vals

        # get the actual value of the attribute
        actual = getattr(self, attr_name)

        # check that the length matches
        if len(expected) != len(actual):
            raise AssertionError(
                "Invalid length for attribute '{}': {} (expected {})".format(
                    attr_name, len(actual), len(expected)))

        # look through all the permutations of the expected values, and try to
        # match them with the actual values. If a permutation is found where
        # the values match, then set that as the permutation for the class. If
        # no permutation is found, then raise an error.
        for perm in itertools.permutations(np.arange(len(expected))):
            try:
                self._assert_equal(attr_name, expected, actual, perm=perm)
            except AssertionError:
                pass
            else:
                self._perm = perm
                return

        raise AssertionError(
            "Could not match plotted values {} to expected values {} for attr '{}'".format(
                actual, expected, attr_name))

    def assert_num_lines(self, num_lines):
        """Assert that the plot has the given number of lines.

        Parameters
        ----------
        num_lines : int

        """
        if num_lines != len(self._lines):
            raise AssertionError(
                "Plot has incorrect number of lines: {} (expected {})".format(
                    len(self._lines), num_lines))

    @property
    def x_data(self):
        """The x-values of the plotted data (list of arrays, one array per line)."""
        return [x.get_xydata()[:, 0] for x in self._lines]

    def assert_x_data_equal(self, x_data):
        """Assert that the given x-data is equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.x_data`.

        Parameters
        ----------
        x_data : list of array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted lines.

        """
        self._assert_equal("x_data", x_data, self.x_data)

    def assert_x_data_allclose(self, x_data, **kwargs):
        """Assert that the given x-data is almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.x_data`.

        Parameters
        ----------
        x_data : list of array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose("x_data", x_data, self.x_data, **kwargs)

    @property
    def y_data(self):
        """The y-values of the plotted data (list of arrays, one array per line)."""
        return [x.get_xydata()[:, 1] for x in self._lines]

    def assert_y_data_equal(self, y_data):
        """Assert that the given y-data is equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.y_data`.

        Parameters
        ----------
        y_data : list of array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted lines.

        """
        self._assert_equal("y_data", y_data, self.y_data)

    def assert_y_data_allclose(self, y_data, **kwargs):
        """Assert that the given y-data is almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.y_data`.

        Parameters
        ----------
        y_data : list of array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose("y_data", y_data, self.y_data, **kwargs)

    @property
    def colors(self):
        """The colors of the plotted lines. Each color is a RGB 3-tuple."""
        return np.array([self._color2rgb(x.get_color()) for x in self._lines])

    def assert_colors_equal(self, colors):
        """Assert that the given colors are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.colors`.

        Parameters
        ----------
        colors : list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        colors = np.array([self._color2rgb(x) for x in colors])
        self._assert_equal("colors", colors, self.colors)

    def assert_colors_allclose(self, colors, **kwargs):
        """Assert that the given colors are almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.colors`.

        Parameters
        ----------
        colors : list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        colors = np.array([self._color2rgb(x) for x in colors])
        self._assert_allclose("colors", colors, self.colors, **kwargs)

    @property
    def alphas(self):
        """The alpha values of the plotted lines."""
        all_alphas = []
        for x in self._lines:
            if x.get_alpha() is None:
                all_alphas.append(self._color2alpha(x.get_color()))
            else:
                all_alphas.append(x.get_alpha())
        return all_alphas

    def assert_alphas_equal(self, alphas):
        """Assert that the given alpha values are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.alphas`.

        Parameters
        ----------
        alphas : list of floats
            The expected alpha values, with length equal to the (expected)
            number of plotted lines.

        """
        self._assert_equal("alphas", alphas, self.alphas)

    def assert_alphas_allclose(self, alphas, **kwargs):
        """Assert that the given alpha values are almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.alphas`.

        Parameters
        ----------
        alphas : list of floats
            The expected alpha values, with length equal to the (expected)
            number of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose("alphas", alphas, self.alphas, **kwargs)

    @property
    def linewidths(self):
        """The line widths of the plotted lines."""
        return [x.get_linewidth() for x in self._lines]

    def assert_linewidths_equal(self, linewidths):
        """Assert that the given line widths are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.linewidths`.

        Parameters
        ----------
        linewidths : list of numbers
            The expected linewidths, with length equal to the (expected) number
            of plotted lines.

        """
        self._assert_equal("linewidths", linewidths, self.linewidths)

    def assert_linewidths_allclose(self, linewidths, **kwargs):
        """Assert that the given line widths are almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.linewidths`.

        Parameters
        ----------
        linewidths : list of numbers
            The expected linewidths, with length equal to the (expected) number
            of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose("linewidths", linewidths, self.linewidths, **kwargs)

    @property
    def markerfacecolors(self):
        """The colors of the marker faces for the plotted lines."""
        return [self._color2rgb(x.get_markerfacecolor()) for x in self._lines]

    def assert_markerfacecolors_equal(self, markerfacecolors):
        """Assert that the given marker face colors are equivalent to the
        plotted :attr:`~plotchecker.LinePlotChecker.markerfacecolors`.

        Parameters
        ----------
        markerfacecolors : list of expected marker face colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        self._assert_equal("markerfacecolors", markerfacecolors, self.markerfacecolors)

    def assert_markerfacecolors_allclose(self, markerfacecolors, **kwargs):
        """Assert that the given marker face colors are almost equal to the
        plotted :attr:`~plotchecker.LinePlotChecker.markerfacecolors`.

        Parameters
        ----------
        markerfacecolors : list of expected marker face colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        markerfacecolors = np.array([self._color2rgb(x) for x in markerfacecolors])
        self._assert_allclose(
            "markerfacecolors", markerfacecolors, self.markerfacecolors, **kwargs)

    @property
    def markeredgecolors(self):
        """The colors of the marker edges for the plotted lines."""
        return [self._color2rgb(x.get_markeredgecolor()) for x in self._lines]

    def assert_markeredgecolors_equal(self, markeredgecolors):
        """Assert that the given marker edge colors are equivalent to the
        plotted :attr:`~plotchecker.LinePlotChecker.markeredgecolors`.

        Parameters
        ----------
        markeredgecolors : list of expected marker edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.

        """
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        self._assert_equal("markeredgecolors", markeredgecolors, self.markeredgecolors)

    def assert_markeredgecolors_allclose(self, markeredgecolors, **kwargs):
        """Assert that the given marker edge colors are almost equal to the
        plotted :attr:`~plotchecker.LinePlotChecker.markeredgecolors`.

        Parameters
        ----------
        markeredgecolors : list of expected marker edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        markeredgecolors = np.array([self._color2rgb(x) for x in markeredgecolors])
        self._assert_allclose(
            "markeredgecolors", markeredgecolors, self.markeredgecolors, **kwargs)

    @property
    def markeredgewidths(self):
        """The widths of the marker edges for the plotted lines."""
        return [x.get_markeredgewidth() for x in self._lines]

    def assert_markeredgewidths_equal(self, markeredgewidths):
        """Assert that the given marker edge widths are equivalent to the
        plotted :attr:`~plotchecker.LinePlotChecker.markeredgewidths`.

        Parameters
        ----------
        markeredgewidths : list of expected marker edge widths
            The expected edge widths, with length equal to the (expected)
            number of plotted lines.

        """
        self._assert_equal("markeredgewidths", markeredgewidths, self.markeredgewidths)

    def assert_markeredgewidths_allclose(self, markeredgewidths, **kwargs):
        """Assert that the given marker edge widths are almost equal to the
        plotted :attr:`~plotchecker.LinePlotChecker.markeredgewidths`.

        Parameters
        ----------
        markeredgewidths : list of expected marker edge widths
            The expected edge widths, with length equal to the (expected)
            number of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose(
            "markeredgewidths", markeredgewidths, self.markeredgewidths, **kwargs)

    @property
    def markersizes(self):
        """The marker sizes for the plotted lines."""
        return [x.get_markersize() for x in self._lines]

    def assert_markersizes_equal(self, markersizes):
        """Assert that the given marker sizes are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.markersizes`.

        Parameters
        ----------
        markersizes : list of expected marker sizes
            The expected marker sizes, with length equal to the (expected)
            number of plotted lines.

        """
        self._assert_equal("markersizes", markersizes, self.markersizes)

    def assert_markersizes_allclose(self, markersizes, **kwargs):
        """Assert that the given marker sizes are almost equal to the plotted
        :attr:`~plotchecker.LinePlotChecker.markersizes`.

        Parameters
        ----------
        markersizes : list of expected marker sizes
            The expected marker sizes, with length equal to the (expected)
            number of plotted lines.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        self._assert_allclose(
            "markersizes", markersizes, self.markersizes, **kwargs)

    @property
    def markers(self):
        """The marker types for the plotted lines."""
        return [self._parse_marker(x.get_marker()) for x in self._lines]

    def assert_markers_equal(self, markers):
        """Assert that the given markers are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.markers`.

        Parameters
        ----------
        markers : list of strings
            The expected markers, with length equal to the (expected)
            number of plotted lines.

        """
        markers = [self._parse_marker(x) for x in markers]
        self._assert_equal("markers", markers, self.markers)

    @property
    def labels(self):
        """The legend labels of the plotted lines."""
        legend = self.axis.get_legend()
        if legend is None:
            return []
        return [x.get_text() for x in legend.texts]

    def assert_labels_equal(self, labels):
        """Assert that the given legend labels are equivalent to the plotted
        :attr:`~plotchecker.LinePlotChecker.labels`.

        Parameters
        ----------
        labels : list of strings
            The expected legend labels, with length equal to the (expected)
            number of plotted lines.

        """
        self._assert_equal("labels", labels, self.labels)
