import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from plotchecker import ScatterPlotChecker

# set matplotlib defaults
matplotlib.rcdefaults()

class TestScatterPlotChecker(object):

    # coordinates
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)

    # colors
    c = np.linspace(0, 1, 10)[:, None] * np.ones((10, 3))
    c0 = c[[0]] * np.ones((10, 3))
    default_scatter_c = np.array([[0, 0, 1]]) * np.ones((10, 3))
    default_plot_c = np.array([
        [0., 0., 1.],
        [0., 0.5, 0.],
        [1., 0., 0.],
        [0., 0.75, 0.75],
        [0.75, 0., 0.75],
        [0.75, 0.75, 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0.5, 0.],
        [1., 0., 0.]])

    # alphas
    a = np.linspace(0, 1, 10)
    a0 = a[[0]] * np.ones(10)
    default_scatter_a = np.ones(10)
    default_plot_a = np.ones(10)

    # color + alpha
    ca = np.concatenate([c, a[:, None]], axis=1)

    # sizes
    s = np.linspace(10, 100, 10)
    s0 = s[[0]] * np.ones(10)
    default_scatter_s = np.ones(10) * 20
    default_plot_s = np.ones(10) * 36

    # markers
    m = np.array(['s'] * 10)
    m0 = np.array(['s'] * 10)
    default_scatter_m = np.array(['o'] * 10)
    default_plot_m = np.array(['None'] * 10)


    def setup(self):
        self.fig, self.axis = plt.subplots()

    def teardown(self):
        plt.close(self.fig)

    def test_forloop_scatter_points(self):
        """Using scatter in a for loop"""
        for i in range(self.x.size):
            self.axis.scatter(self.x[i], self.y[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)

        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.default_scatter_c)
        pc.assert_alphas_equal(self.default_scatter_a)
        pc.assert_sizes_equal(self.default_scatter_s)
        #pc.assert_markers_equal(self.default_scatter_m)

    def test_forloop_scatter_points_single(self):
        """Using scatter in a for loop with single colors and sizes"""
        for i in range(self.x.size):
            self.axis.scatter(self.x[i], self.y[i], c=self.ca[i], s=self.s[i], marker=self.m[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c)
        pc.assert_alphas_equal(self.a)
        pc.assert_sizes_equal(self.s)
        #pc.assert_markers_equal(self.m)

    def test_forloop_scatter_points_all(self):
        """Using scatter in a for loop with all colors and sizes"""
        for i in range(self.x.size):
            self.axis.scatter(self.x[i], self.y[i], c=self.ca, s=self.s, marker=self.m[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c0)
        pc.assert_alphas_equal(self.a0)
        pc.assert_sizes_equal(self.s0)
        #pc.assert_markers_equal(self.m0)

    def test_scatter_points(self):
        """Using scatter on its own"""
        self.axis.scatter(self.x, self.y)

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.default_scatter_c)
        pc.assert_alphas_equal(self.default_scatter_a)
        pc.assert_sizes_equal(self.default_scatter_s)
        #pc.assert_markers_equal(self.default_scatter_m)

    def test_scatter_points_single(self):
        """Using scatter on its own with single colors and sizes"""
        self.axis.scatter(self.x, self.y, c=self.c[0], s=self.s[0], marker=self.m[0])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c0)
        pc.assert_sizes_equal(self.s0)
        #pc.assert_markers_equal(self.m0)

    def test_scatter_points_all(self):
        """Using scatter on its own with all colors and sizes"""
        self.axis.scatter(self.x, self.y, c=self.c, s=self.s, marker=self.m[0])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c)
        pc.assert_sizes_equal(self.s)
        #pc.assert_markers_equal(self.m)

    def test_forloop_plot_points(self):
        """Using plot within a for loop"""
        for i in range(self.x.size):
            self.axis.plot(self.x[i], self.y[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.default_plot_c)
        pc.assert_sizes_equal(self.default_plot_s)
        pc.assert_markers_equal(self.default_plot_m)

    def test_forloop_plot_points_alpha(self):
        """Using plot within a for loop"""
        for i in range(self.x.size):
            self.axis.plot(self.x[i], self.y[i], alpha=self.a[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.default_plot_c)
        pc.assert_alphas_equal(self.a)
        pc.assert_sizes_equal(self.default_plot_s)
        pc.assert_markers_equal(self.default_plot_m)

    def test_forloop_plot_points_single(self):
        """Using plot within a for loop with single colors and sizes"""
        for i in range(self.x.size):
            self.axis.plot(self.x[i], self.y[i], color=self.ca[i], ms=np.sqrt(self.s[i]), marker=self.m[i])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c)
        pc.assert_alphas_equal(self.a)
        assert_almost_equal(pc.sizes, self.s)
        pc.assert_markers_equal(self.m)

    def test_plot_points(self):
        """Using plot on its own"""
        self.axis.plot(self.x, self.y, ls='')

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.default_scatter_c)
        pc.assert_sizes_equal(self.default_plot_s)
        pc.assert_markers_equal(self.default_plot_m)

    def test_plot_points_single(self):
        """Using plot on its own with single colors and sizes"""
        self.axis.plot(self.x, self.y, color=self.c[0], ms=np.sqrt(self.s[0]), ls='', marker=self.m[0])

        pc = ScatterPlotChecker(self.axis)
        pc.assert_x_data_equal(self.x)
        pc.assert_y_data_equal(self.y)
        pc.assert_colors_equal(self.c0)
        assert_almost_equal(pc.sizes, self.s0)
        pc.assert_markers_equal(self.m0)

