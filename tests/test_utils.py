import numpy as np

from heston_mc.utils import simple_returns_from_prices, standard_error


def test_simple_returns_shape():
    prices = np.array([
        [100.0, 101.0, 103.0],
        [100.0, 99.0, 100.0],
    ])
    rets = simple_returns_from_prices(prices)
    assert rets.shape == (2, 2)


def test_standard_error_nonnegative():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    se = standard_error(x)
    assert se >= 0.0
