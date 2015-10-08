from __future__ import division

import matplotlib
import matplotlib.colors
import matplotlib.markers
import numpy as np
import six


class InvalidPlotError(Exception):
    pass


class PlotChecker(object):

    _named_colors = matplotlib.colors.ColorConverter.colors.copy()
    for colorname, hexcode in matplotlib.colors.cnames.items():
        _named_colors[colorname] = matplotlib.colors.hex2color(hexcode)

    def __init__(self, axis):
        self.axis = axis

    @classmethod
    def _color2rgb(cls, color):
        if isinstance(color, six.string_types):
            if color in cls._named_colors:
                return tuple(cls._named_colors[color])
            else:
                return tuple(matplotlib.colors.hex2color(color))
        elif hasattr(color, '__iter__') and len(color) == 3:
            return tuple(color)
        elif hasattr(color, '__iter__') and len(color) == 4:
            return tuple(color[:3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _color2alpha(cls, color):
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
        if marker is None or marker == 'None':
            return ''
        return marker

    @classmethod
    def _tile_or_trim(cls, x, y):
        """Tiles or trims `y` so that `x.shape[0]` == `y.shape[0]`"""
        xn = x.shape[0]
        yn = y.shape[0]
        if xn > yn:
            numrep = np.ceil(xn / yn)
            y = np.tile(y, (numrep,) + (1,) * (y.ndim - 1))
            yn = y.shape[0]
        if xn < yn:
            y = y[:xn]
        return y

    @property
    def title(self):
        return self.axis.get_title().strip()

    def assert_title_equal(self, title):
        """Asserts that the given title is the same as the plotted title."""
        title = title.strip()
        if self.title != title:
            raise AssertionError(
                "title is incorrect: '{}'' (expected '{}')".format(
                    self.title, title))

    def assert_title_exists(self):
        """Asserts that the plotted title is non-empty."""
        if self.title == '':
            raise AssertionError("no title")

    @property
    def xlabel(self):
        return self.axis.get_xlabel().strip()

    def assert_xlabel_equal(self, xlabel):
        """Asserts that the given xlabel is the same as the plotted xlabel."""
        xlabel = xlabel.strip()
        if self.xlabel != xlabel:
            raise AssertionError(
                "xlabel is incorrect: '{}'' (expected '{}')".format(
                    self.xlabel, xlabel))

    def assert_xlabel_exists(self):
        """Asserts that the plotted xlabel is non-empty."""
        if self.xlabel == '':
            raise AssertionError("no xlabel")

    @property
    def ylabel(self):
        return self.axis.get_ylabel().strip()

    def assert_ylabel_equal(self, ylabel):
        """Asserts that the given ylabel is the same as the plotted ylabel."""
        ylabel = ylabel.strip()
        if self.ylabel != ylabel:
            raise AssertionError(
                "ylabel is incorrect: '{}'' (expected '{}')".format(
                    self.ylabel, ylabel))

    def assert_ylabel_exists(self):
        """Asserts that the plotted ylabel is non-empty."""
        if self.ylabel == '':
            raise AssertionError("no ylabel")

    @property
    def xlim(self):
        return self.axis.get_xlim()

    def assert_xlim_equal(self, xlim):
        if self.xlim != xlim:
            raise AssertionError(
                "xlim is incorrect: {} (expected {})".format(
                    self.xlim, xlim))

    @property
    def ylim(self):
        return self.axis.get_ylim()

    def assert_ylim_equal(self, ylim):
        if self.ylim != ylim:
            raise AssertionError(
                "ylim is incorrect: {} (expected {})".format(
                    self.ylim, ylim))

    @property
    def xticks(self):
        return self.axis.get_xticks()

    def assert_xticks_equal(self, xticks):
        np.testing.assert_equal(xticks, self.xticks)

    @property
    def yticks(self):
        return self.axis.get_yticks()

    def assert_yticks_equal(self, yticks):
        np.testing.assert_equal(yticks, self.yticks)

    @property
    def xticklabels(self):
        return [x.get_text().strip() for x in self.axis.get_xticklabels()]

    def assert_xticklabels_equal(self, xticklabels):
        xticklabels = [x.strip() for x in xticklabels]
        np.testing.assert_equal(xticklabels, self.xticklabels)

    @property
    def yticklabels(self):
        return [x.get_text().strip() for x in self.axis.get_yticklabels()]

    def assert_yticklabels_equal(self, yticklabels):
        yticklabels = [y.strip() for y in yticklabels]
        np.testing.assert_equal(yticklabels, self.yticklabels)


# def get_label_text(ax):
#     text = [x for x in ax.get_children()
#             if isinstance(x, matplotlib.text.Text)]
#     text = [x for x in text if x.get_text() != ax.get_title()]
#     text = [x for x in text if x.get_text().strip() != '']
#     return [x.get_text().strip() for x in text]


# def get_label_pos(ax):
#     text = [x for x in ax.get_children()
#             if isinstance(x, matplotlib.text.Text)]
#     text = [x for x in text if x.get_text() != ax.get_title()]
#     text = [x for x in text if x.get_text().strip() != '']
#     return np.vstack([x.get_position() for x in text])


# def get_imshow_data(ax):
#     image, = ax.get_images()
#     return image._A
