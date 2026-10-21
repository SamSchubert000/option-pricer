"""Black-Scholes closed-form pricing for European options."""

from math import erf, exp, log, sqrt


def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S0, K, r, sigma, T, option_type="call"):
    """Price a European option using the Black-Scholes formula.

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

    Returns
    -------
    float
        The option price.
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive.")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put".')

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    discount = exp(-r * T)

    if option_type == "call":
        return S0 * norm_cdf(d1) - K * discount * norm_cdf(d2)
    return K * discount * norm_cdf(-d2) - S0 * norm_cdf(-d1)