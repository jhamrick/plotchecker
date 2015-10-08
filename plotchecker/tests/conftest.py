import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def axis(request):
    fig, ax = plt.subplots()

    def fin():
        plt.close(fig)
    request.addfinalizer(fin)

    return ax


@pytest.fixture
def axes(request):
    fig, axes = plt.subplots(1, 3)

    def fin():
        plt.close(fig)
    request.addfinalizer(fin)

    return axes
