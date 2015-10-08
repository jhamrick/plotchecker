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

