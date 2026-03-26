import numpy as np

from heston_mc.utils import discounted_call_payoff, standard_error


def test_discounted_call_payoff_shape():
    terminal_spots = np.array([90.0, 100.0, 110.0])
    payoffs = discounted_call_payoff(
        terminal_spots,
        strike=100.0,
        rate=0.0,
        maturity=1.0,
    )
    assert payoffs.shape == terminal_spots.shape


def test_standard_error_nonnegative():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    se = standard_error(x)
    assert se >= 0.0
