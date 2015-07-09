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
        if isinstance(color, six.string_types):
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

        # check that if there are lines, linestyle is ''
        for x in self.lines:
            if len(x.get_xydata()) > 1 and x.get_linestyle() != 'None':
                raise InvalidPlotError("This is supposed to be a scatter plot, but it has lines!")

    @property
    def x_data(self):
        all_x_data = []

        if len(self.lines) > 0:
            all_x_data.append(np.concatenate([x.get_xydata()[:, 0] for x in self.lines]))

        if len(self.collections) > 0:
            all_x_data.append(np.concatenate([x.get_offsets()[:, 0] for x in self.collections]))

        return np.concatenate(all_x_data, axis=0)

    def assert_x_data_equal(self, x_data):
        np.testing.assert_equal(x_data, self.x_data)

    def assert_x_data_almost_equal(self, x_data):
        np.testing.assert_almost_equal(x_data, self.x_data)

    @property
    def y_data(self):
        all_y_data = []

        if len(self.lines) > 0:
            all_y_data.append(np.concatenate([x.get_xydata()[:, 1] for x in self.lines]))

        if len(self.collections) > 0:
            all_y_data.append(np.concatenate([x.get_offsets()[:, 1] for x in self.collections]))

        return np.concatenate(all_y_data, axis=0)

    def assert_y_data_equal(self, y_data):
        np.testing.assert_equal(y_data, self.y_data)

    def assert_y_data_almost_equal(self, y_data):
        np.testing.assert_almost_equal(y_data, self.y_data)

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

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                if x.get_alpha() is not None:
                    alphas = np.array([x.get_alpha()])
                else:
                    alphas = np.array([self._color2alpha(i) for i in x.get_facecolors()])
                all_alphas.append(self._tile_or_trim(points, alphas))

        return np.concatenate(all_alphas, axis=0)

    def assert_alphas_equal(self, alphas):
        np.testing.assert_equal(alphas, self.alphas)

    def assert_alphas_almost_equal(self, alphas):
        np.testing.assert_almost_equal(alphas, self.alphas)

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
        edgecolors = np.array([self._color2rgb(x) for x in edgecolors])
        if len(edgecolors) == 1:
            edgecolors = self._tile_or_trim(self.x_data, edgecolors)
        np.testing.assert_equal(edgecolors, self.edgecolors)

    def assert_edgecolors_almost_equal(self, edgecolors):
        edgecolors = np.array([self._color2rgb(x) for x in edgecolors])
        if len(edgecolors) == 1:
            edgecolors = self._tile_or_trim(self.x_data, edgecolors)
        np.testing.assert_almost_equal(edgecolors, self.edgecolors)

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
        if not hasattr(edgewidths, '__iter__'):
            edgewidths = np.array([edgewidths])
        if len(edgewidths) == 1:
            edgewidths = self._tile_or_trim(self.x_data, edgewidths)
        np.testing.assert_equal(edgewidths, self.edgewidths)

    def assert_edgewidths_almost_equal(self, edgewidths):
        if not hasattr(edgewidths, '__iter__'):
            edgewidths = np.array([edgewidths])
        if len(edgewidths) == 1:
            edgewidths = self._tile_or_trim(self.x_data, edgewidths)
        np.testing.assert_almost_equal(edgewidths, self.edgewidths)

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
        np.testing.assert_equal(sizes, self.sizes)

    def assert_sizes_almost_equal(self, sizes):
        np.testing.assert_almost_equal(sizes, self.sizes)

    @property
    def markersizes(self):
        return np.sqrt(self.sizes)

    def assert_markersizes_equal(self, markersizes):
        np.testing.assert_equal(markersizes, self.markersizes)

    def assert_markersizes_almost_equal(self, markersizes):
        np.testing.assert_almost_equal(markersizes, self.markersizes)

    @property
    def markers(self):
        all_markers = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                markers = np.array([x.get_marker()])
                all_markers.append(self._tile_or_trim(points, markers))

        if len(self.collections) > 0:
            # Can't get the marker style from a collection :( :( :(
            pass

        return np.concatenate(all_markers, axis=0)

    def assert_markers_equal(self, markers):
        np.testing.assert_equal(markers, self.markers)


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