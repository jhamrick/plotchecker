# plotchecker

[![Build Status](https://travis-ci.org/jhamrick/plotchecker.svg?branch=master)](https://travis-ci.org/jhamrick/plotchecker)
[![codecov.io](http://codecov.io/github/jhamrick/plotchecker/coverage.svg?branch=master)](http://codecov.io/github/jhamrick/plotchecker?branch=master)
[![Documentation Status](https://readthedocs.org/projects/plotchecker/badge/?version=latest)](http://plotchecker.readthedocs.org/en/latest/?badge=latest)

A set of utilities for checking and grading matplotlib plots. **Please note that `plotchecker` is only compatible with Python 3, and not legacy Python 2**. Documentation is available on [Read The Docs](https://plotchecker.readthedocs.org/).

## Installation

To install `plotchecker`:

```
pip3 install plotchecker
```

## Background

The inspiration for this library comes from including plotting exercises in programming assignments. Often, there are multiple possible ways to solve a problem; for example, if students are asked to create a "scatter plot", the following are all valid methods of doing so:

```python
# Method 1
plt.plot(x, y, 'o')

# Method 2
plt.scatter(x, y)

# Method 3
for i in range(len(x)):
    plt.plot(x[i], y[i], 'o')

# Method 4
for i in range(len(x)):
    plt.scatter(x[i], y[i])
```

Unfortunately, each of the above approaches also creates a different underlying representation of the data in matplotlib. Method 1 creates a single Line object; Method 2 creates a single Collection; Method 3 creates *n* Line objects, where *n* is the number of points; and Method 4 creates *n* Collection objects. Testing for all of these different edge cases is a huge burden on instructors.

While some of the above options are certainly better than others in terms of simplicity and performance, it doesn't seem quite fair to ask students to create their plots in a very specific way when all we've asked them for is a scatter plot. If they look pretty much identical visually, why isn't it a valid approach?

Enter `plotchecker`, which aims to abstract away from these differences and expose a simple interface for instructors to check students' plots. All that is necessary is access to the `Axes` object, and then you can write a common set of tests for plots independent of how they were created.

```python
from plotchecker import ScatterPlotChecker

axis = plt.gca()
pc = ScatterPlotChecker(axis)
pc.assert_x_data_equal(x)
pc.assert_y_data_equal(y)
...
```

Please see the [Examples.ipynb](Examples.ipynb) notebook for futher examples on how `plotchecker` can be used.

Caveats: there are *many* ways that plots can be created in matplotlib. `plotchecker` almost certainly misses some of the edge cases. If you find any, please submit a bug report (or even better, a PR!).
