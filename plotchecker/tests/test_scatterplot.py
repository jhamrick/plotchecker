import numpy as np
import pytest

from .. import ScatterPlotChecker, InvalidPlotError


def test_empty_plot(axis):
    """Is an error thrown when there is nothing plotted?"""
    with pytest.raises(InvalidPlotError):
        ScatterPlotChecker(axis)


def test_bad_plot(axis):
    """Is an error thrown when there are lines rather than points plotted?"""
    # first just try for a single set of points
    x0 = [1, 2.17, 3.3, 4]
    y0 = [2.5, 3.25, 4.4, 5]
    axis.plot(x0, y0)

    with pytest.raises(InvalidPlotError):
        ScatterPlotChecker(axis)


def test_num_points(axis):
    """Are the number of points correct?"""
    # first just try for a single set of points
    x0 = [1, 2.17, 3.3, 4]
    y0 = [2.5, 3.25, 4.4, 5]
    axis.plot(x0, y0, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_num_points(4)

    # now plot another line
    x1 = [2, 3.17, 4.3, 5, 6]
    y1 = [1.5, 2.25, 3.4, 4, 7]
    axis.plot(x1, y1, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_num_points(9)

    # do it with scatter
    x2 = [4.9, 0, 1]
    y2 = [10, 20, 30]
    axis.scatter(x2, y2)
    pc = ScatterPlotChecker(axis)
    pc.assert_num_points(12)

    # plot some more things with scatter
    x3 = [2, 3.17, 4.3, 12]
    y3 = [1.5, 2.25, 3.4, 23]
    axis.scatter(x3, y3)
    pc = ScatterPlotChecker(axis)
    pc.assert_num_points(16)

    # and now two more lines (six points), plotted at the same time
    x4 = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y4 = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x4.T, y4.T, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_num_points(22)

    with pytest.raises(AssertionError):
        pc.assert_num_points(16)


def test_data(axis):
    """Are the x and y values correct?"""
    # first just try for a single set of points
    x0 = [1, 2.17, 3.3, 4]
    y0 = [2.5, 3.25, 4.4, 5]
    axis.plot(x0, y0, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_x_data_equal(x0)
    pc.assert_y_data_equal(y0)

    # now plot another line
    x1 = [2, 3.17, 4.3, 5, 6]
    y1 = [1.5, 2.25, 3.4, 4, 7]
    axis.plot(x1, y1, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_x_data_equal(np.concatenate([x0, x1]))
    pc.assert_y_data_equal(np.concatenate([y0, y1]))

    # do it with scatter
    x2 = [4.9, 0, 1]
    y2 = [10, 20, 30]
    axis.scatter(x2, y2)
    pc = ScatterPlotChecker(axis)
    pc.assert_x_data_equal(np.concatenate([x0, x1, x2]))
    pc.assert_y_data_equal(np.concatenate([y0, y1, y2]))

    # plot some more things with scatter
    x3 = [2, 3.17, 4.3, 12]
    y3 = [1.5, 2.25, 3.4, 23]
    axis.scatter(x3, y3)
    pc = ScatterPlotChecker(axis)
    pc.assert_x_data_equal(np.concatenate([x0, x1, x2, x3]))
    pc.assert_y_data_equal(np.concatenate([y0, y1, y2, y3]))

    # and now two more lines (six points), plotted at the same time
    x4 = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y4 = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x4.T, y4.T, 'o')
    pc = ScatterPlotChecker(axis)
    pc.assert_x_data_equal(np.concatenate([x0, x1, x4.ravel(), x2, x3]))
    pc.assert_y_data_equal(np.concatenate([y0, y1, y4.ravel(), y2, y3]))


def test_edgecolors(axis):
    x = np.random.rand(10)
    y = np.random.rand(10)
    c = np.random.rand(10, 3)

    for i in range(5):
        axis.plot(x[i], y[i], 'o', markeredgecolor=c[i])
    for i in range(5, 10):
        axis.scatter(x[i], y[i], edgecolor=c[i])

    pc = ScatterPlotChecker(axis)
    pc.assert_edgecolors_equal(c)


def test_edgewidths(axis):
    x = np.random.rand(10)
    y = np.random.rand(10)
    w = np.arange(1, 11)

    for i in range(5):
        axis.plot(x[i], y[i], 'o', markeredgewidth=w[i])
    for i in range(5, 10):
        axis.scatter(x[i], y[i], linewidth=w[i])

    pc = ScatterPlotChecker(axis)
    pc.assert_edgewidths_equal(w)


@pytest.mark.xfail(reason="markers are unrecoverable from scatter plots")
def test_markers(axis):
    x = np.random.rand(10)
    y = np.random.rand(10)
    m = ['o', '.', 's', 'D', 'v', '^', '<', '>', 'H', '+']

    for i in range(5):
        axis.plot(x[i], y[i], marker=m[i])
    for i in range(5, 10):
        axis.scatter(x[i], y[i], marker=m[i])

    pc = ScatterPlotChecker(axis)
    pc.assert_markers_equal(m)


def test_example_1(axes):
    x = np.random.rand(20)
    y = np.random.rand(20)

    # create a scatter plot with plot
    axes[0].plot(x, y, 'o', color='b', ms=5, alpha=0.8)

    # create a scatter plot with scatter
    axes[1].scatter(x, y, s=25, linewidth=0.5, alpha=0.8)

    # create a scatter plot with plot *and* scatter!
    axes[2].plot(x[:10], y[:10], 'o', color='b', ms=5, alpha=0.8)
    axes[2].scatter(x[10:], y[10:], s=25, linewidth=0.5, alpha=0.8)

    for ax in axes:
        pc = ScatterPlotChecker(ax)
        pc.assert_x_data_equal(x)
        pc.assert_y_data_equal(y)
        pc.assert_colors_equal('b')
        pc.assert_edgecolors_equal('k')
        pc.assert_edgewidths_equal(0.5)
        pc.assert_sizes_equal(25)
        pc.assert_markersizes_equal(5)
        pc.assert_alphas_equal(0.8)


def test_example_2(axes):
    x = np.random.rand(20)
    y = np.random.rand(20)

    # choose some random colors and sizes
    colors = np.random.rand(20, 4)
    sizes = np.random.rand(20) * 5

    # create a scatter plot with plot, using a loop
    for i in range(20):
        axes[0].plot(x[i], y[i], 'o', color=colors[i], ms=sizes[i])

    # create a scatter plot with scatter
    axes[1].scatter(x, y, c=colors, s=sizes ** 2, linewidth=0.5)

    # create a scatter plot with scatter, using a loop
    for i in range(20):
        axes[2].scatter(x[i], y[i], c=colors[i], s=sizes[i] ** 2, linewidth=0.5)

    for ax in axes:
        pc = ScatterPlotChecker(ax)
        pc.assert_x_data_equal(x)
        pc.assert_y_data_equal(y)
        pc.assert_colors_equal(colors)
        pc.assert_edgecolors_equal('k')
        pc.assert_edgewidths_equal(0.5)
        pc.assert_sizes_equal(sizes ** 2)
        pc.assert_markersizes_equal(sizes)
        pc.assert_alphas_equal(colors[:, 3])


def test_bad_colors_and_sizes(axis):
    x = np.random.rand(10)
    y = np.random.rand(10)
    c = np.random.rand(10, 3)
    s = np.random.rand(10) * 5

    for i in range(10):
        axis.scatter(x[i], y[i], c=c, s=s)

    pc = ScatterPlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_colors_equal(c)
    with pytest.raises(AssertionError):
        pc.assert_sizes_equal(s)

    pc.assert_colors_equal(c[[0]])
    pc.assert_sizes_equal(s[0])
