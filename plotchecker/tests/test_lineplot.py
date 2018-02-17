import pytest
import numpy as np

from .. import LinePlotChecker, InvalidPlotError


def test_empty_plot(axis):
    """Is an error thrown when there is nothing plotted?"""
    with pytest.raises(InvalidPlotError):
        LinePlotChecker(axis)


def test_num_lines(axis):
    """Are the number of lines correct?"""
    # first just try for a single line
    x0 = [1, 2.17, 3.3, 4]
    y0 = [2.5, 3.25, 4.4, 5]
    axis.plot(x0, y0)
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(1)

    # now plot another line
    x1 = [2, 3.17, 4.3, 5, 6]
    y1 = [1.5, 2.25, 3.4, 4, 7]
    axis.plot(x1, y1)
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(2)

    # do a line without x values
    y2 = [10, 20, 30]
    axis.plot(y2)
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(3)

    # and now two more lines, plotted at the same time
    x3 = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y3 = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x3.T, y3.T)
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(5)

    with pytest.raises(AssertionError):
        pc.assert_num_lines(6)


def test_data(axis):
    """Are the x and y values correct?"""
    # first just try for a single line
    x0 = [1, 2.17, 3.3, 4]
    y0 = [2.5, 3.25, 4.4, 5]
    axis.plot(x0, y0)
    pc = LinePlotChecker(axis)
    pc.assert_x_data_equal([x0])
    pc.assert_y_data_equal([y0])

    # now plot another line
    x1 = [2, 3.17, 4.3, 5, 6]
    y1 = [1.5, 2.25, 3.4, 4, 7]
    axis.plot(x1, y1)
    pc = LinePlotChecker(axis)
    pc.assert_x_data_equal([x0, x1])
    pc.assert_y_data_equal([y0, y1])

    # do a line without x values
    x2 = [0, 1, 2]
    y2 = [10, 20, 30]
    axis.plot(y2)
    pc = LinePlotChecker(axis)
    pc.assert_x_data_equal([x0, x1, x2])
    pc.assert_y_data_equal([y0, y1, y2])

    # and now two more lines, plotted at the same time
    x3 = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y3 = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x3.T, y3.T)
    pc = LinePlotChecker(axis)
    pc.assert_x_data_equal([x0, x1, x2] + list(x3))
    pc.assert_y_data_equal([y0, y1, y2] + list(y3))


def test_data_allclose(axis):
    """Are the x and y values almost correct?"""
    err = 1e-12
    x0 = np.array([1, 2.17, 3.3, 4])
    y0 = np.array([2.5, 3.25, 4.4, 5])
    axis.plot(x0 + err, y0 + err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_x_data_equal([x0])
    with pytest.raises(AssertionError):
        pc.assert_y_data_equal([y0])

    with pytest.raises(AssertionError):
        pc.assert_x_data_allclose([x0], rtol=1e-13)
    with pytest.raises(AssertionError):
        pc.assert_y_data_allclose([y0], rtol=1e-13)

    pc.assert_x_data_allclose([x0])
    pc.assert_y_data_allclose([y0])


def test_colors(axis):
    """Are the colors correct?"""
    # first just try for a single line using rgb
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color=[0, 1, 1])
    pc = LinePlotChecker(axis)
    pc.assert_colors_equal([[0, 1, 1]])

    # add another line, using hex values
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color='#FF0000')
    pc = LinePlotChecker(axis)
    pc.assert_colors_equal([[0, 1, 1], '#FF0000'])

    # add another line, using matplotlib colors
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color='g')
    pc = LinePlotChecker(axis)
    pc.assert_colors_equal([[0, 1, 1], '#FF0000', 'g'])

    # add another line, using full matplotlib color names
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color='magenta')
    pc = LinePlotChecker(axis)
    pc.assert_colors_equal([[0, 1, 1], '#FF0000', 'g', 'magenta'])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, color='k')
    pc = LinePlotChecker(axis)
    pc.assert_colors_equal([[0, 1, 1], '#FF0000', 'g', 'magenta', 'k', 'k'])


def test_colors_allclose(axis):
    """Are the colors almost correct?"""
    err = 1e-12
    color = np.array([0.1, 1, 1])
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color=color - err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_colors_equal([color])
    with pytest.raises(AssertionError):
        pc.assert_colors_allclose([color], rtol=1e-13)
    pc.assert_colors_allclose([color])


def test_linewidths(axis):
    """Are the linewidths correct?"""
    # first just try for a single line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], linewidth=1)
    pc = LinePlotChecker(axis)
    pc.assert_linewidths_equal([1])

    # add another line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], linewidth=2)
    pc = LinePlotChecker(axis)
    pc.assert_linewidths_equal([1, 2])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, linewidth=4)
    pc = LinePlotChecker(axis)
    pc.assert_linewidths_equal([1, 2, 4, 4])


def test_linewidths_allclose(axis):
    """Are the linewidths almost correct?"""
    err = 1e-12
    lw = 1
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], lw=lw + err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_linewidths_equal([lw])
    with pytest.raises(AssertionError):
        pc.assert_linewidths_allclose([lw], rtol=1e-13)
    pc.assert_linewidths_allclose([lw])


def test_markerfacecolors(axis):
    """Are the marker face colors correct?"""
    # inherit the color from the line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', color='c')
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c'])

    # add another line, using rgb
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markerfacecolor=[0, 1, 1])
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c', [0, 1, 1]])

    # add another line, using hex values
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markerfacecolor='#FF0000')
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c', [0, 1, 1], '#FF0000'])

    # add another line, using matplotlib colors
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markerfacecolor='g')
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c', [0, 1, 1], '#FF0000', 'g'])

    # add another line, using full matplotlib color names
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markerfacecolor='magenta')
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c', [0, 1, 1], '#FF0000', 'g', 'magenta'])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, marker='o', markerfacecolor='k')
    pc = LinePlotChecker(axis)
    pc.assert_markerfacecolors_equal(['c', [0, 1, 1], '#FF0000', 'g', 'magenta', 'k', 'k'])


def test_markerfacecolors_allclose(axis):
    """Are the markerfacecolors almost correct?"""
    err = 1e-12
    markerfacecolor = np.array([0.1, 1, 1])
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], markerfacecolor=list(markerfacecolor + err))

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_markerfacecolors_equal([markerfacecolor])
    with pytest.raises(AssertionError):
        pc.assert_markerfacecolors_allclose([markerfacecolor], rtol=1e-13)
    pc.assert_markerfacecolors_allclose([markerfacecolor])


def test_markeredgecolors(axis):
    """Are the marker edge colors correct?"""
    # inherit the color from the line -- this should actually be the default (grey)
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', color='c')
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75]])

    # add another line, using rgb
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgecolor=[0, 1, 1])
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75], [0, 1, 1]])

    # add another line, using hex values
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgecolor='#FF0000')
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75], [0, 1, 1], '#FF0000'])

    # add another line, using matplotlib colors
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgecolor='g')
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75], [0, 1, 1], '#FF0000', 'g'])

    # add another line, using full matplotlib color names
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgecolor='magenta')
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75], [0, 1, 1], '#FF0000', 'g', 'magenta'])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, marker='o', markeredgecolor='k')
    pc = LinePlotChecker(axis)
    pc.assert_markeredgecolors_equal([[0, 0.75, 0.75], [0, 1, 1], '#FF0000', 'g', 'magenta', 'k', 'k'])


def test_markeredgecolors_allclose(axis):
    """Are the markeredgecolors almost correct?"""
    err = 1e-12
    markeredgecolor = np.array([0.1, 1, 1])
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], markeredgecolor=list(markeredgecolor + err))

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_markeredgecolors_equal([markeredgecolor])
    with pytest.raises(AssertionError):
        pc.assert_markeredgecolors_allclose([markeredgecolor], rtol=1e-13)
    pc.assert_markeredgecolors_allclose([markeredgecolor])


def test_markeredgewidths(axis):
    """Are the markeredgewidths correct?"""
    # first just try for a single line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgewidth=1)
    pc = LinePlotChecker(axis)
    pc.assert_markeredgewidths_equal([1])

    # add another line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markeredgewidth=2)
    pc = LinePlotChecker(axis)
    pc.assert_markeredgewidths_equal([1, 2])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, marker='o', markeredgewidth=4)
    pc = LinePlotChecker(axis)
    pc.assert_markeredgewidths_equal([1, 2, 4, 4])


def test_markeredgewidths_allclose(axis):
    """Are the markeredgewidths almost correct?"""
    err = 1e-12
    markeredgewidth = 1
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], markeredgewidth=markeredgewidth + err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_markeredgewidths_equal([markeredgewidth])
    with pytest.raises(AssertionError):
        pc.assert_markeredgewidths_allclose([markeredgewidth], rtol=1e-13)
    pc.assert_markeredgewidths_allclose([markeredgewidth])


def test_markersizes(axis):
    """Are the markersizes correct?"""
    # first just try for a single line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markersize=1)
    pc = LinePlotChecker(axis)
    pc.assert_markersizes_equal([1])

    # add another line
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o', markersize=2)
    pc = LinePlotChecker(axis)
    pc.assert_markersizes_equal([1, 2])

    # and now two more lines, plotted at the same time
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, marker='o', markersize=4)
    pc = LinePlotChecker(axis)
    pc.assert_markersizes_equal([1, 2, 4, 4])


def test_markersizes_allclose(axis):
    """Are the markersizes almost correct?"""
    err = 1e-12
    markersize = 1
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], markersize=markersize + err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_markersizes_equal([markersize])
    with pytest.raises(AssertionError):
        pc.assert_markersizes_allclose([markersize], rtol=1e-13)
    pc.assert_markersizes_allclose([markersize])


def test_markers(axis):
    """Are the markers correct?"""
    # first just try for a single line with no markers
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5])
    pc = LinePlotChecker(axis)
    pc.assert_markers_equal([''])

    # now use an empty marker
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='')
    pc = LinePlotChecker(axis)
    pc.assert_markers_equal(['', ''])

    # now use the o marker
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='o')
    pc = LinePlotChecker(axis)
    pc.assert_markers_equal(['', '', 'o'])

    # add another line with the . marker
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], marker='.')
    pc = LinePlotChecker(axis)
    pc.assert_markers_equal(['', '', 'o', '.'])

    # and now two more lines, plotted at the same time, with the D marker
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, marker='D')
    pc = LinePlotChecker(axis)
    pc.assert_markers_equal(['', '', 'o', '.', 'D', 'D'])


def test_kwarg_labels(axis):
    """Are the legend labels correct when given as kwargs?"""
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], label='foo')
    axis.plot([2.17, 3.3, 4], [3.25, 4.4, 5], label='bar')
    axis.plot([1, 2.17, 3.3], [2.5, 3.25, 4.4], label='baz')

    # make sure it fails before the legend is created
    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_labels_equal(['foo', 'bar', 'baz'])

    axis.legend()
    pc = LinePlotChecker(axis)
    pc.assert_labels_equal(['foo', 'bar', 'baz'])


def test_legend_labels(axis):
    """Are the legend labels correct when they are passed into the legend call?"""
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5])
    axis.plot([2.17, 3.3, 4], [3.25, 4.4, 5])
    axis.plot([1, 2.17, 3.3], [2.5, 3.25, 4.4])

    # make sure it fails before the legend is created
    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_labels_equal(['foo', 'bar', 'baz'])

    axis.legend(['foo', 'bar', 'baz'])
    pc = LinePlotChecker(axis)
    pc.assert_labels_equal(['foo', 'bar', 'baz'])


def test_legend_handles_and_labels(axis):
    """Are the legend labels correct when they are passed into the legend call with the corresponding handle?"""
    l0, = axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5])
    l1, = axis.plot([2.17, 3.3, 4], [3.25, 4.4, 5])
    l2, = axis.plot([1, 2.17, 3.3], [2.5, 3.25, 4.4])

    # make sure it fails before the legend is created
    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_labels_equal(['foo', 'bar', 'baz'])

    axis.legend([l0, l1, l2], ['foo', 'bar', 'baz'])
    pc = LinePlotChecker(axis)
    pc.assert_labels_equal(['foo', 'bar', 'baz'])


def test_alphas(axis):
    """Are the alphas correct?"""
    # first just try for a single line using rgb
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color=[0, 1, 1])
    pc = LinePlotChecker(axis)
    pc.assert_alphas_equal([1])

    # get the alpha value from rgba
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], color=[0, 1, 1, 0.5])
    pc = LinePlotChecker(axis)
    pc.assert_alphas_equal([1, 0.5])

    # specify the alpha value explicitly
    x = np.array([[4.3, 5, 6], [5.3, 6, 7]])
    y = np.array([[3.4, 4, 7], [10.2, 9, 8]])
    axis.plot(x.T, y.T, alpha=0.2)
    pc = LinePlotChecker(axis)
    pc.assert_alphas_equal([1, 0.5, 0.2, 0.2])


def test_alphas_allclose(axis):
    """Are the alphas almost correct?"""
    err = 1e-12
    alpha = 0.5
    axis.plot([1, 2.17, 3.3, 4], [2.5, 3.25, 4.4, 5], alpha=alpha + err)

    pc = LinePlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_alphas_equal([alpha])
    with pytest.raises(AssertionError):
        pc.assert_alphas_allclose([alpha], rtol=1e-13)
    pc.assert_alphas_allclose([alpha])


def test_permutations(axis):
    x = np.linspace(0, 1, 20)[None] * np.ones((3, 20))
    y = x ** np.array([1, 2, 3])[:, None]

    colors = ['r', 'g', 'b']
    markers = ['o', 's', 'D']
    labels = ['Line A', 'Line B', 'Line C']

    # plot lines in a different order from the values
    for i in [2, 0, 1]:
        axis.plot(x[i], y[i], color=colors[i], marker=markers[i], label=labels[i], alpha=0.5)
    axis.legend()

    # do the permutation based off of colors
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(3)
    pc.find_permutation('colors', colors)
    pc.assert_x_data_equal(x)
    pc.assert_y_data_equal(y)
    pc.assert_colors_equal(colors)
    pc.assert_markers_equal(markers)
    pc.assert_labels_equal(labels)
    pc.assert_alphas_equal([0.5, 0.5, 0.5])

    # do the permutation based off of markers
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(3)
    pc.find_permutation('markers', markers)
    pc.assert_x_data_equal(x)
    pc.assert_y_data_equal(y)
    pc.assert_colors_equal(colors)
    pc.assert_markers_equal(markers)
    pc.assert_labels_equal(labels)
    pc.assert_alphas_equal([0.5, 0.5, 0.5])

    # do the permutation based off of labels
    pc = LinePlotChecker(axis)
    pc.assert_num_lines(3)
    pc.find_permutation('labels', labels)
    pc.assert_x_data_equal(x)
    pc.assert_y_data_equal(y)
    pc.assert_colors_equal(colors)
    pc.assert_markers_equal(markers)
    pc.assert_labels_equal(labels)
    pc.assert_alphas_equal([0.5, 0.5, 0.5])

    with pytest.raises(AssertionError):
        pc.find_permutation('labels', labels[:-1])
    with pytest.raises(AssertionError):
        pc.find_permutation('labels', [x + 'a' for x in labels])
