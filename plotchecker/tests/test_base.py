import pytest
import numpy as np

from .. import PlotChecker


def test_color2rgb():
    assert PlotChecker._color2rgb('r') == (1, 0, 0)
    assert PlotChecker._color2rgb('black') == (0, 0, 0)
    assert PlotChecker._color2rgb([0, 1, 0]) == (0, 1, 0)
    assert PlotChecker._color2rgb([0, 1, 0, 1]) == (0, 1, 0)

    with pytest.raises(ValueError):
        PlotChecker._color2rgb('foo')
    with pytest.raises(ValueError):
        PlotChecker._color2rgb(1)


def test_color2alpha():
    assert PlotChecker._color2alpha('r') == 1
    assert PlotChecker._color2alpha('black') == 1
    assert PlotChecker._color2alpha([0, 1, 0]) == 1
    assert PlotChecker._color2alpha([0, 1, 0, 0.5]) == 0.5

    with pytest.raises(ValueError):
        PlotChecker._color2alpha(1)


def test_tile_or_trim():
    x = np.array([1, 2, 3])
    y0 = np.array([4, 5])
    y1 = PlotChecker._tile_or_trim(x, y0)
    np.testing.assert_array_equal(y1, np.array([4, 5, 4]))

    x = np.array([1, 2, 3])
    y0 = np.array([[0, -1, -2]])
    y1 = PlotChecker._tile_or_trim(x, y0)
    np.testing.assert_array_equal(y1, np.array([[0, -1, -2], [0, -1, -2], [0, -1, -2]]))

    x = np.array([[1, 0], [2, 1], [3, 4]])
    y0 = np.array([[0, -1, -2]])
    y1 = PlotChecker._tile_or_trim(x, y0)
    np.testing.assert_array_equal(y1, np.array([[0, -1, -2], [0, -1, -2], [0, -1, -2]]))


def test_title_assertions(axis):
    pc = PlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_title_exists()

    axis.set_title("foo")
    pc.assert_title_exists()
    pc.assert_title_equal("foo")
    with pytest.raises(AssertionError):
        pc.assert_title_equal("bar")


def test_xlabel_assertions(axis):
    pc = PlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_xlabel_exists()

    axis.set_xlabel("foo")
    pc.assert_xlabel_exists()
    pc.assert_xlabel_equal("foo")
    with pytest.raises(AssertionError):
        pc.assert_xlabel_equal("bar")


def test_ylabel_assertions(axis):
    pc = PlotChecker(axis)
    with pytest.raises(AssertionError):
        pc.assert_ylabel_exists()

    axis.set_ylabel("foo")
    pc.assert_ylabel_exists()
    pc.assert_ylabel_equal("foo")
    with pytest.raises(AssertionError):
        pc.assert_ylabel_equal("bar")


def test_assert_xlim_equal(axis):
    pc = PlotChecker(axis)
    axis.set_xlim(0, 1)
    pc.assert_xlim_equal((0, 1))
    with pytest.raises(AssertionError):
        pc.assert_xlim_equal((1, 0))


def test_assert_ylim_equal(axis):
    pc = PlotChecker(axis)
    axis.set_ylim(0, 1)
    pc.assert_ylim_equal((0, 1))
    with pytest.raises(AssertionError):
        pc.assert_ylim_equal((1, 0))


def test_assert_xticks_equal(axis):
    pc = PlotChecker(axis)
    axis.set_xticks([0, 1, 2, 3])
    pc.assert_xticks_equal([0, 1, 2, 3])
    with pytest.raises(AssertionError):
        pc.assert_xticks_equal([0, 1])


def test_assert_yticks_equal(axis):
    pc = PlotChecker(axis)
    axis.set_yticks([0, 1, 2, 3])
    pc.assert_yticks_equal([0, 1, 2, 3])
    with pytest.raises(AssertionError):
        pc.assert_yticks_equal([0, 1])


def test_assert_xticklabels_equal(axis):
    pc = PlotChecker(axis)
    axis.set_xticks([0, 1, 2, 3])
    axis.set_xticklabels(['a', 'b', 'c', 'd'])
    pc.assert_xticklabels_equal(['a', 'b', 'c', 'd'])
    with pytest.raises(AssertionError):
        pc.assert_xticklabels_equal(['a', 'b', 'c'])


def test_assert_yticklabels_equal(axis):
    pc = PlotChecker(axis)
    axis.set_yticks([0, 1, 2, 3])
    axis.set_yticklabels(['a', 'b', 'c', 'd'])
    pc.assert_yticklabels_equal(['a', 'b', 'c', 'd'])
    with pytest.raises(AssertionError):
        pc.assert_yticklabels_equal(['a', 'b', 'c'])

def test_texts(axis):
    x = np.random.rand(10)
    y = np.random.rand(10)
    t = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    for i in range(len(x)):
        axis.text(x[i], y[i], t[i])
    axis.set_title('foo')
    axis.set_xlabel('bar')
    axis.set_ylabel('baz')

    pc = PlotChecker(axis)
    pc.assert_textlabels_equal(t)
    pc.assert_textpoints_equal(np.array([x, y]).T)

def test_texts_allclose(axis):
    err = 1e-12
    x = np.round(np.random.rand(10), decimals=3)
    y = np.round(np.random.rand(10), decimals=3)
    x[x < 1e-3] = 1e-3
    y[y < 1e-3] = 1e-3
    t = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    for i in range(len(x)):
        axis.text(x[i] + err, y[i] + err, t[i])
    axis.set_title('foo')
    axis.set_xlabel('bar')
    axis.set_ylabel('baz')

    pc = PlotChecker(axis)
    pc.assert_textlabels_equal(t)
    with pytest.raises(AssertionError):
        pc.assert_textpoints_equal(np.array([x, y]).T)
    with pytest.raises(AssertionError):
        pc.assert_textpoints_allclose(np.array([x, y]).T, rtol=1e-13)
    pc.assert_textpoints_allclose(np.array([x, y]).T)
