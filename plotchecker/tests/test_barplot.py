import pytest
import numpy as np

from .. import BarPlotChecker, InvalidPlotError


def test_empty_plot(axis):
    """Is an error thrown when there is nothing plotted?"""
    with pytest.raises(InvalidPlotError):
        BarPlotChecker(axis)


def test_num_bars(axis):
    """Are the number of bars correct?"""
    x = np.arange(10)
    y = np.linspace(1, 5, 10)
    axis.bar(x, y)

    pc = BarPlotChecker(axis)
    pc.assert_num_bars(10)

    with pytest.raises(AssertionError):
        pc.assert_num_bars(6)


def test_data_left(axis):
    """Are the x and y values correct?"""
    x = np.arange(10)
    y = np.linspace(1, 5, 10)
    b = np.linspace(0, 1, 10)
    axis.bar(x, y, bottom=b, align='edge')

    pc = BarPlotChecker(axis)
    pc.assert_centers_equal(x + 0.4)
    pc.assert_heights_equal(y)
    pc.assert_bottoms_equal(b)


def test_data_left_allclose(axis):
    """Are the x and y values almost correct?"""
    err = 1e-12
    x = np.arange(1, 11)
    y = np.linspace(1, 5, 10)
    b = np.linspace(0.1, 1, 10)
    axis.bar(x + err, y + err, bottom=b + err, align='edge')

    pc = BarPlotChecker(axis)

    with pytest.raises(AssertionError):
        pc.assert_centers_equal(x + 0.4)
    with pytest.raises(AssertionError):
        pc.assert_heights_equal(y)
    with pytest.raises(AssertionError):
        pc.assert_bottoms_equal(b)

    with pytest.raises(AssertionError):
        pc.assert_centers_allclose(x + 0.4, rtol=1e-13)
    with pytest.raises(AssertionError):
        pc.assert_heights_allclose(y, rtol=1e-13)
    with pytest.raises(AssertionError):
        pc.assert_bottoms_allclose(b, rtol=1e-13)

    pc.assert_centers_allclose(x + 0.4)
    pc.assert_heights_allclose(y)
    pc.assert_bottoms_allclose(b)


def test_data_center(axis):
    """Are the x and y values correct with align=center?"""
    x = np.arange(10)
    y = np.linspace(1, 5, 10)
    b = np.linspace(0, 1, 10)
    axis.bar(x, y, bottom=b, align='center')

    pc = BarPlotChecker(axis)
    pc.assert_centers_equal(x)
    pc.assert_heights_equal(y)
    pc.assert_bottoms_equal(b)


def test_data_center_allclose(axis):
    """Are the x and y values almost correct?"""
    err = 1e-12
    x = np.arange(1, 11)
    y = np.linspace(1, 5, 10)
    b = np.linspace(0.1, 1, 10)
    axis.bar(x + err, y + err, bottom=b + err, align='center')

    pc = BarPlotChecker(axis)

    with pytest.raises(AssertionError):
        pc.assert_centers_equal(x)
    with pytest.raises(AssertionError):
        pc.assert_heights_equal(y)
    with pytest.raises(AssertionError):
        pc.assert_bottoms_equal(b)

    with pytest.raises(AssertionError):
        pc.assert_centers_allclose(x, rtol=1e-13)
    with pytest.raises(AssertionError):
        pc.assert_heights_allclose(y, rtol=1e-13)
    with pytest.raises(AssertionError):
        pc.assert_bottoms_allclose(b, rtol=1e-13)

    pc.assert_centers_allclose(x)
    pc.assert_heights_allclose(y)
    pc.assert_bottoms_allclose(b)


def test_widths(axis):
    """Are the widths correct?"""
    x = np.arange(10)
    y = np.linspace(1, 5, 10)
    w = np.linspace(0.5, 1, 10)

    for i in range(len(x)):
        axis.bar(x[i], y[i], width=w[i], align='center')

    pc = BarPlotChecker(axis)
    pc.assert_centers_equal(x)
    pc.assert_heights_equal(y)
    pc.assert_widths_equal(w)


def test_widths_allclose(axis):
    """Are the widths almost correct?"""
    err = 1e-12
    x = np.arange(10)
    y = np.linspace(1, 5, 10)
    w = np.linspace(0.5, 1, 10)

    for i in range(len(x)):
        axis.bar(x[i], y[i], width=w[i] + err, align='center')

    pc = BarPlotChecker(axis)
    pc.assert_centers_equal(x)
    pc.assert_heights_equal(y)

    with pytest.raises(AssertionError):
        pc.assert_widths_equal(w)
    with pytest.raises(AssertionError):
        pc.assert_widths_allclose(w, rtol=1e-13)
    pc.assert_widths_allclose(w)


def test_colors(axis):
    """Are the colors correct?"""
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    colors = ['r', '#FF0000', (0, 0, 0), 'blue', (0, 1, 0.5, 1)]

    for i in range(len(x)):
        axis.bar(x[i], y[i], color=colors[i], align='center')

    pc = BarPlotChecker(axis)
    pc.assert_colors_equal(colors)


def test_colors_allclose(axis):
    """Are the colors almost correct?"""
    err = 1e-12
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    colors = np.array([(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.8)])

    for i in range(len(x)):
        axis.bar(x[i], y[i], color=tuple(colors[i] + err), align='center')

    pc = BarPlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_colors_equal(colors)
    with pytest.raises(AssertionError):
        pc.assert_colors_allclose(colors, rtol=1e-13)
    pc.assert_colors_allclose(colors)


def test_edgecolors(axis):
    """Are the edgecolors correct?"""
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    edgecolors = ['r', '#FF0000', (0, 0, 0), 'blue', (0, 1, 0.5, 1)]

    for i in range(len(x)):
        axis.bar(x[i], y[i], edgecolor=edgecolors[i], align='center')

    pc = BarPlotChecker(axis)
    pc.assert_edgecolors_equal(edgecolors)


def test_edgecolors_allclose(axis):
    """Are the edgecolors almost correct?"""
    err = 1e-12
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    edgecolors = np.array([(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.8)])

    for i in range(len(x)):
        axis.bar(x[i], y[i], edgecolor=tuple(edgecolors[i] + err), align='center')

    pc = BarPlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_edgecolors_equal(edgecolors)
    with pytest.raises(AssertionError):
        pc.assert_edgecolors_allclose(edgecolors, rtol=1e-13)
    pc.assert_edgecolors_allclose(edgecolors)


def test_alphas(axis):
    """Are the alphas correct?"""
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    alphas = np.linspace(0.1, 1, 5)

    for i in range(len(x)):
        axis.bar(x[i], y[i], alpha=alphas[i], align='center')

    pc = BarPlotChecker(axis)
    pc.assert_alphas_equal(alphas)


def test_alphas_allclose(axis):
    """Are the alphas almost correct?"""
    err = 1e-12
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    alphas = np.linspace(0.1, 0.99, 5)

    for i in range(len(x)):
        axis.bar(x[i], y[i], alpha=alphas[i] + err, align='center')

    pc = BarPlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_alphas_equal(alphas)
    with pytest.raises(AssertionError):
        pc.assert_alphas_allclose(alphas, rtol=1e-13)
    pc.assert_alphas_allclose(alphas)


def test_linewidths(axis):
    """Are the linewidths correct?"""
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    linewidths = np.linspace(0.1, 1, 5)

    for i in range(len(x)):
        axis.bar(x[i], y[i], linewidth=linewidths[i], align='center')

    pc = BarPlotChecker(axis)
    pc.assert_linewidths_equal(linewidths)


def test_linewidths_allclose(axis):
    """Are the linewidths almost correct?"""
    err = 1e-12
    x = np.arange(5)
    y = np.linspace(1, 5, 5)
    linewidths = np.linspace(0.1, 1, 5)

    for i in range(len(x)):
        axis.bar(x[i], y[i], linewidth=linewidths[i] + err, align='center')

    pc = BarPlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_linewidths_equal(linewidths)
    with pytest.raises(AssertionError):
        pc.assert_linewidths_allclose(linewidths, rtol=1e-13)
    pc.assert_linewidths_allclose(linewidths)


def test_example(axes):
    x = np.arange(1, 11)
    y = np.random.rand(10)

    colors = np.random.rand(10, 4)
    alphas = colors[:, 3]
    widths = (np.random.rand(10) / 2) + 0.5

    # plot some bars
    for i in range(len(x)):
        axes[0].bar(x[i], y[i], width=widths[i], align='center', color=colors[i])

    # plot them in a different order
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    for i in idx:
        axes[1].bar(x[i] - (widths[i] / 2), y[i], width=widths[i], align='edge', color=colors[i, :3], alpha=alphas[i])

    # plot them in yet another order
    idx = np.arange(len(x))[::-1]
    for i in idx:
        axes[2].bar(x[i], y[i], width=widths[i], align='center', color=colors[i, :3], alpha=alphas[i])

    for ax in axes:
        pc = BarPlotChecker(ax)
        pc.assert_centers_equal(x)
        pc.assert_heights_equal(y)
        pc.assert_widths_equal(widths)
        pc.assert_bottoms_equal(0)
        pc.assert_colors_equal(colors)
        pc.assert_edgecolors_equal('k')
        pc.assert_alphas_equal(alphas)
        pc.assert_linewidths_equal(1)
