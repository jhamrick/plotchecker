from __future__ import division

import matplotlib
import matplotlib.colors
import matplotlib.markers
import numpy as np
import six
import warnings


try:
    _named_colors = matplotlib.colors.ColorConverter.colors.copy()
    for colorname, hexcode in matplotlib.colors.cnames.items():
        _named_colors[colorname] = matplotlib.colors.hex2color(hexcode)
except: # pragma: no cover
    warnings.warn("Could not get matplotlib colors, named colors will not be available")
    _named_colors = {}


class InvalidPlotError(Exception):
    pass


class PlotChecker(object):
    """A generic object to test plots.

    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)

    """

    _named_colors = _named_colors

    def __init__(self, axis):
        """Initialize the PlotChecker object."""
        self.axis = axis

    @classmethod
    def _color2rgb(cls, color):
        """Converts the given color to a 3-tuple RGB color.

        Parameters
        ----------
        color :
            Either a matplotlib color name (e.g. ``'r'`` or ``'red'``), a
            hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or a 4-tuple RGBA
            color.

        Returns
        -------
        rgb : 3-tuple RGB color

        """
        if isinstance(color, six.string_types):
            if color in cls._named_colors:
                return tuple(cls._named_colors[color])
            else:
                return tuple(matplotlib.colors.hex2color(color))
        elif hasattr(color, '__iter__') and len(color) == 3:
            return tuple(float(x) for x in color)
        elif hasattr(color, '__iter__') and len(color) == 4:
            return tuple(float(x) for x in color[:3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _color2alpha(cls, color):
        """Converts the given color to an alpha value. For all cases except
        RGBA colors, this value will be 1.0.

        Parameters
        ----------
        color :
            Either a matplotlib color name (e.g. ``'r'`` or ``'red'``), a
            hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or a 4-tuple RGBA
            color.

        Returns
        -------
        alpha : float

        """
        if isinstance(color, six.string_types):
            return 1.0
        elif hasattr(color, '__iter__') and len(color) == 3:
            return 1.0
        elif hasattr(color, '__iter__') and len(color) == 4:
            return float(color[3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _parse_marker(cls, marker):
        """Converts the given marker to a consistent marker type. In practice,
        this is basically just making sure all null markers (``''``, ``'None'``,
        ``None``) get converted to empty strings.

        Parameters
        ----------
        marker : string
            The marker type

        Returns
        -------
        marker : string

        """
        if marker is None or marker == 'None':
            return ''
        return marker

    @classmethod
    def _tile_or_trim(cls, x, y):
        """Tiles or trims the first dimension of ``y`` so that ``x.shape[0]`` ==
        ``y.shape[0]``.

        Parameters
        ----------
        x : array-like
            A numpy array with any number of dimensions.
        y : array-like
            A numpy array with any number of dimensions.

        """
        xn = x.shape[0]
        yn = y.shape[0]
        if xn > yn:
            numrep = int(np.ceil(xn / yn))
            y = np.tile(y, (numrep,) + (1,) * (y.ndim - 1))
            yn = y.shape[0]
        if xn < yn:
            y = y[:xn]
        return y

    @property
    def title(self):
        """The title of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_title().strip()

    def assert_title_equal(self, title):
        """Asserts that the given title is the same as the plotted
        :attr:`~plotchecker.PlotChecker.title`.

        Parameters
        ----------
        title : string
            The expected title

        """
        title = title.strip()
        if self.title != title:
            raise AssertionError(
                "title is incorrect: '{}'' (expected '{}')".format(
                    self.title, title))

    def assert_title_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.title` is
        non-empty.

        """
        if self.title == '':
            raise AssertionError("no title")

    @property
    def xlabel(self):
        """The xlabel of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_xlabel().strip()

    def assert_xlabel_equal(self, xlabel):
        """Asserts that the given xlabel is the same as the plotted
        :attr:`~plotchecker.PlotChecker.xlabel`.

        Parameters
        ----------
        xlabel : string
            The expected xlabel

        """
        xlabel = xlabel.strip()
        if self.xlabel != xlabel:
            raise AssertionError(
                "xlabel is incorrect: '{}'' (expected '{}')".format(
                    self.xlabel, xlabel))

    def assert_xlabel_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.xlabel` is
        non-empty.

        """
        if self.xlabel == '':
            raise AssertionError("no xlabel")

    @property
    def ylabel(self):
        """The ylabel of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_ylabel().strip()

    def assert_ylabel_equal(self, ylabel):
        """Asserts that the given ylabel is the same as the plotted
        :attr:`~plotchecker.PlotChecker.ylabel`.

        Parameters
        ----------
        ylabel : string
            The expected ylabel

        """
        ylabel = ylabel.strip()
        if self.ylabel != ylabel:
            raise AssertionError(
                "ylabel is incorrect: '{}'' (expected '{}')".format(
                    self.ylabel, ylabel))

    def assert_ylabel_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.ylabel` is
        non-empty.

        """
        if self.ylabel == '':
            raise AssertionError("no ylabel")

    @property
    def xlim(self):
        """The x-axis limits of the matplotlib plot."""
        return self.axis.get_xlim()

    def assert_xlim_equal(self, xlim):
        """Asserts that the given xlim is the same as the plot's
        :attr:`~plotchecker.PlotChecker.xlim`.

        Parameters
        ----------
        xlim : 2-tuple
            The expected xlim

        """
        if self.xlim != xlim:
            raise AssertionError(
                "xlim is incorrect: {} (expected {})".format(
                    self.xlim, xlim))

    @property
    def ylim(self):
        """The y-axis limits of the matplotlib plot."""
        return self.axis.get_ylim()

    def assert_ylim_equal(self, ylim):
        """Asserts that the given ylim is the same as the plot's
        :attr:`~plotchecker.PlotChecker.ylim`.

        Parameters
        ----------
        ylim : 2-tuple
            The expected ylim

        """
        if self.ylim != ylim:
            raise AssertionError(
                "ylim is incorrect: {} (expected {})".format(
                    self.ylim, ylim))

    @property
    def xticks(self):
        """The tick locations along the plot's x-axis."""
        return self.axis.get_xticks()

    def assert_xticks_equal(self, xticks):
        """Asserts that the given xticks are the same as the plot's
        :attr:`~plotchecker.PlotChecker.xticks`.

        Parameters
        ----------
        xticks : list
            The expected tick locations on the x-axis

        """
        np.testing.assert_equal(self.xticks, xticks)

    @property
    def yticks(self):
        """The tick locations along the plot's y-axis."""
        return self.axis.get_yticks()

    def assert_yticks_equal(self, yticks):
        """Asserts that the given yticks are the same as the plot's
        :attr:`~plotchecker.PlotChecker.yticks`.

        Parameters
        ----------
        yticks : list
            The expected tick locations on the y-axis

        """
        np.testing.assert_equal(self.yticks, yticks)

    @property
    def xticklabels(self):
        """The tick labels along the plot's x-axis, stripped of whitespace."""
        return [x.get_text().strip() for x in self.axis.get_xticklabels()]

    def assert_xticklabels_equal(self, xticklabels):
        """Asserts that the given xticklabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.xticklabels`.

        Parameters
        ----------
        xticklabels : list
            The expected tick labels on the x-axis

        """
        xticklabels = [x.strip() for x in xticklabels]
        np.testing.assert_equal(self.xticklabels, xticklabels)

    @property
    def yticklabels(self):
        """The tick labels along the plot's y-axis, stripped of whitespace."""
        return [x.get_text().strip() for x in self.axis.get_yticklabels()]

    def assert_yticklabels_equal(self, yticklabels):
        """Asserts that the given yticklabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.yticklabels`.

        Parameters
        ----------
        yticklabels : list
            The expected tick labels on the y-axis

        """
        yticklabels = [y.strip() for y in yticklabels]
        np.testing.assert_equal(self.yticklabels, yticklabels)


    @property
    def _texts(self):
        """All ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        texts = []
        for x in self.axis.get_children():
            if not isinstance(x, matplotlib.text.Text):
                continue
            if x == self.axis.title:
                continue
            if x == getattr(self.axis, '_left_title', None):
                continue
            if x == getattr(self.axis, '_right_title', None):
                continue
            texts.append(x)
        return texts

    @property
    def textlabels(self):
        """The labels of all ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        return [x.get_text().strip() for x in self._texts]

    def assert_textlabels_equal(self, textlabels):
        """Asserts that the given textlabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.textlabels`.

        Parameters
        ----------
        textlabels : list
            The expected text labels on the plot

        """
        textlabels = [x.strip() for x in textlabels]
        np.testing.assert_equal(self.textlabels, textlabels)

    @property
    def textpoints(self):
        """The locations of all ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        return np.vstack([x.get_position() for x in self._texts])

    def assert_textpoints_equal(self, textpoints):
        """Asserts that the given locations of the text objects are the same as
        the plot's :attr:`~plotchecker.PlotChecker.textpoints`.

        Parameters
        ----------
        textpoints : array-like, N-by-2
            The expected text locations on the plot, where the first column
            corresponds to the x-values, and the second column corresponds to
            the y-values.

        """
        np.testing.assert_equal(self.textpoints, textpoints)

    def assert_textpoints_allclose(self, textpoints, **kwargs):
        """Asserts that the given locations of the text objects are almost the
        same as the plot's :attr:`~plotchecker.PlotChecker.textpoints`.

        Parameters
        ----------
        textpoints : array-like, N-by-2
            The expected text locations on the plot, where the first column
            corresponds to the x-values, and the second column corresponds to
            the y-values.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``

        """
        np.testing.assert_allclose(self.textpoints, textpoints, **kwargs)
