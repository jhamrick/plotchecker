import matplotlib.pyplot as plt
import pytest

@pytest.fixture
def axis(request):
    fig, ax = plt.subplots()

    def fin():
        plt.close(fig)
    request.addfinalizer(fin)

    return ax
