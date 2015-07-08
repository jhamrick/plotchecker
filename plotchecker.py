from __future__ import division

import matplotlib
import matplotlib.colors
import numpy as np


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
        if isinstance(color, str):
            if color in cls._named_colors:
                return np.array(cls._named_colors[color], dtype=float)
            else:
                return np.array(matplotlib.colors.hex2color(color), dtype=float)
        elif len(color) == 3:
            return np.array(color, dtype=float)
        elif len(color) == 4:
            return np.array(color, dtype=float)[:3]
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _color2alpha(cls, color):
        if isinstance(color, str):
            return 1.0
        elif len(color) == 3:
            return 1.0
        elif len(color) == 4:
            return float(color[3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _tile_or_trim(cls, x, y):
        xn = x.shape[0]
        yn = y.shape[0]
        if xn > yn:
            numrep = np.ceil(xn / yn)
            y = np.tile(y, (numrep,) + (1,) * (y.ndim - 1))
        if xn < yn:
            y = y[:xn]
        return y


class ScatterPlotChecker(PlotChecker):
    """A plot checker for scatter plots (i.e., no lines)."""

    def __init__(self, axis):
        super(ScatterPlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.collections = self.axis.collections

        # check that there are only lines or collections, not both
        if len(self.lines) == 0 and len(self.collections) == 0:
            raise InvalidPlotError("No data found. Did you call plt.plot or plt.scatter?")
        elif len(self.lines) > 0 and len(self.collections) > 0:
            raise InvalidPlotError("Both lines and collections found. This probably means you called both plt.plot and plt.scatter!")

        # check that if there are lines, linestyle is ''
        for x in self.lines:
            if len(x.get_xydata()) > 1 and x.get_linestyle() != 'None':
                raise InvalidPlotError("This is supposed to be a scatter plot, but it has lines!")

    @property
    def x_data(self):
        if len(self.lines) > 0:
            return np.concatenate([x.get_xydata()[:, 0] for x in self.lines])
        elif len(self.collections) > 0:
            return np.concatenate([x.get_offsets()[:, 0] for x in self.collections])
        else:
            raise InvalidPlotError("No data found")

    @property
    def y_data(self):
        if len(self.lines) > 0:
            return np.concatenate([x.get_xydata()[:, 1] for x in self.lines])
        elif len(self.collections) > 0:
            return np.concatenate([x.get_offsets()[:, 1] for x in self.collections])
        else:
            raise InvalidPlotError("No data found")

    @property
    def alphas(self):
        all_alphas = []
        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                if x.get_alpha() is not None:
                    alphas = np.array([x.get_alpha()])
                else:
                    alphas = np.array([self._color2alpha(x.get_markerfacecolor())])
                all_alphas.append(self._tile_or_trim(points, alphas))

        elif len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                if x.get_alpha() is not None:
                    alphas = np.array([x.get_alpha()])
                else:
                    alphas = np.array([self._color2alpha(i) for i in x.get_facecolors()])
                all_alphas.append(self._tile_or_trim(points, alphas))

        else:
            raise InvalidPlotError("No data found")

        return np.concatenate(all_alphas, axis=0)

    @property
    def colors(self):
        all_colors = []
        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([self._color2rgb(x.get_markerfacecolor())])
                all_colors.append(self._tile_or_trim(points, colors))

        elif len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array([self._color2rgb(i) for i in x.get_facecolors()])
                all_colors.append(self._tile_or_trim(points, colors))

        else:
            raise InvalidPlotError("No data found")

        return np.concatenate(all_colors, axis=0)

    @property
    def sizes(self):
        all_sizes = []
        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                sizes = np.array([x.get_markersize() ** 2])
                all_sizes.append(self._tile_or_trim(points, sizes))

        elif len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                sizes = x.get_sizes()
                all_sizes.append(self._tile_or_trim(points, sizes))

        else:
            raise InvalidPlotError("No data found")

        return np.concatenate(all_sizes, axis=0)


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