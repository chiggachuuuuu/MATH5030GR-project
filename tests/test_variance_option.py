import numpy as np

from heston_mc.variance_option import variance_option_payoff


def test_variance_option_payoff():
    realized_variance = np.array([0.02, 0.05, 0.08])
    strike = 0.04
    
    expected_payoff = np.array([0.00, 0.01, 0.04])
    actual_payoff = variance_option_payoff(realized_variance, strike)
    
    np.testing.assert_array_almost_equal(actual_payoff, expected_payoff)