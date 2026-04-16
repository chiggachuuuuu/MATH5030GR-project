
from __future__ import annotations

import time

import numpy as np

from .interfaces import PricingResult, SimulationResult
from .realized_variance import realized_variance_from_prices
from .utils import discount, standard_error


def variance_option_payoff(
    realized_variance: np.ndarray,
    strike: float,
) -> np.ndarray:
    """
    Pathwise payoff of a variance call option:
        max(RV - K_var, 0)

    Parameters
    ----------
    realized_variance : np.ndarray
        One realized variance value per simulated path.
    strike : float
        Variance strike.

    Returns
    -------
    np.ndarray
        Payoff for each path.
    """
    if realized_variance.ndim != 1:
        raise ValueError("realized_variance must be 1D")
    if strike < 0:
        raise ValueError("strike must be nonnegative")

    return np.maximum(realized_variance - strike, 0.0)


def price_variance_option(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
) -> PricingResult:
    """
    Price a variance option under plain Monte Carlo.

    Workflow
    --------
    1. Compute realized variance from simulated stock paths.
    2. Compute variance option payoff path by path.
    3. Discount the payoff back to time 0.
    4. Estimate price by sample average.
    5. Report Monte Carlo standard error and runtime.

    Parameters
    ----------
    sim_result : SimulationResult
        Simulation output containing stock paths, variance paths, and dt.
    strike : float
        Variance strike K_var.
    rate : float
        Risk-free rate used for discounting.
    maturity : float
        Time to maturity.

    Returns
    -------
    PricingResult
        Monte Carlo pricing summary.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    start_time = time.time()

    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )

    payoff = variance_option_payoff(realized_variance, strike)
    discounted_payoff = discount(payoff, rate, maturity)

    price = float(np.mean(discounted_payoff))
    std_err = standard_error(discounted_payoff)
    runtime = time.time() - start_time

    return PricingResult(
        price=price,
        std_error=std_err,
        n_paths=sim_result.stock_paths.shape[0],
        n_steps=sim_result.stock_paths.shape[1] - 1,
        runtime_seconds=runtime,
        method_name="plain_mc_variance_option",
    )
