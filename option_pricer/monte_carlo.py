"""Monte Carlo pricing for European options under Black-Scholes dynamics."""

from numbers import Integral

import numpy as np


def mc_price(S0, K, r, sigma, T, option_type="call", n_paths=100_000, seed=None):
    """Price a European option by Monte Carlo simulation.

    Terminal prices are sampled exactly from geometric Brownian motion:
        S_T = S0 * exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z),
    with Z ~ N(0, 1). The discounted average payoff is the price estimate.

    Parameters
    ----------
    S0 : float
        Current price of the underlying asset (> 0).
    K : float
        Strike price (> 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying (> 0).
    T : float
        Time to maturity in years (> 0).
    option_type : str
        "call" or "put".
    n_paths : int
        Number of simulated paths (>= 2).
    seed : int or None
        Seed for the random number generator, for reproducible results.

    Returns
    -------
    tuple of float
        (price estimate, standard error of the estimate).
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive.")
    if not isinstance(n_paths, Integral) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least 2.")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put".')

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    s_T = S0 * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(s_T - K, 0.0)
    else:
        payoffs = np.maximum(K - s_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_error = discount * payoffs.std(ddof=1) / np.sqrt(n_paths)

    return float(price), float(std_error)