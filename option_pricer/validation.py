"""Validation of the Monte Carlo pricer against the Black-Scholes benchmark."""

import numpy as np

from option_pricer.black_scholes import bs_price
from option_pricer.monte_carlo import mc_price

# Parameter sets covering in/at/out of the money, low/high volatility,
# short/long maturities, and both option types.
DEFAULT_CASES = [
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "call"},
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "put"},
    {"S0": 100, "K": 80, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "call"},
    {"S0": 100, "K": 120, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "call"},
    {"S0": 100, "K": 120, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "put"},
    {"S0": 100, "K": 80, "r": 0.05, "sigma": 0.20, "T": 1.0, "option_type": "put"},
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.10, "T": 1.0, "option_type": "call"},
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.50, "T": 1.0, "option_type": "call"},
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.30, "T": 0.25, "option_type": "put"},
    {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.30, "T": 5.0, "option_type": "call"},
    {"S0": 100, "K": 100, "r": 0.00, "sigma": 0.20, "T": 1.0, "option_type": "call"},
    {"S0": 50, "K": 55, "r": 0.03, "sigma": 0.25, "T": 2.0, "option_type": "put"},
]


def _next_seed(rng):
    """Draw an integer seed so each Monte Carlo run is independent."""
    return int(rng.integers(0, 2**32 - 1))


def compare_methods(cases=None, n_paths=500_000, seed=0, z_threshold=3.0):
    """Compare Monte Carlo prices with exact Black-Scholes prices.

    For each parameter set, the Monte Carlo estimate is counted as agreeing
    with Black-Scholes if it lies within ``z_threshold`` standard errors.

    Parameters
    ----------
    cases : list of dict or None
        Each dict holds S0, K, r, sigma, T and option_type. Defaults to
        DEFAULT_CASES.
    n_paths : int
        Number of Monte Carlo paths per case.
    seed : int
        Seed controlling all random draws, so the report is reproducible.
    z_threshold : float
        Maximum allowed |estimate - exact| in units of standard error.

    Returns
    -------
    list of dict
        One dict per case containing the inputs plus ``exact``, ``estimate``,
        ``std_error``, ``z_score`` and ``passed``.
    """
    if cases is None:
        cases = DEFAULT_CASES

    rng = np.random.default_rng(seed)
    results = []

    for case in cases:
        exact = bs_price(**case)
        estimate, std_error = mc_price(**case, n_paths=n_paths, seed=_next_seed(rng))
        difference = estimate - exact

        if std_error > 0:
            z_score = difference / std_error
        else:
            z_score = 0.0 if difference == 0 else float("inf")

        results.append(
            {
                **case,
                "exact": exact,
                "estimate": estimate,
                "std_error": std_error,
                "z_score": z_score,
                "passed": abs(z_score) <= z_threshold,
            }
        )

    return results


def convergence_study(
    S0, K, r, sigma, T, option_type="call", n_values=None, n_repeats=30, seed=0
):
    """Measure how the Monte Carlo error shrinks as the number of paths grows.

    For each value of n_paths, the pricer is run ``n_repeats`` times with
    independent seeds and the root-mean-square error against the exact
    Black-Scholes price is recorded. Theory says this should fall like
    1 / sqrt(n_paths).

    Returns
    -------
    dict
        ``n_values`` (array), ``rmse`` (array), ``mean_std_error`` (array,
        the average reported standard error at each n) and ``exact`` (float).
    """
    if n_values is None:
        n_values = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    exact = bs_price(S0, K, r, sigma, T, option_type)
    rng = np.random.default_rng(seed)

    rmse = []
    mean_std_error = []

    for n in n_values:
        errors = np.empty(n_repeats)
        std_errors = np.empty(n_repeats)
        for i in range(n_repeats):
            estimate, std_error = mc_price(
                S0, K, r, sigma, T, option_type, n_paths=n, seed=_next_seed(rng)
            )
            errors[i] = estimate - exact
            std_errors[i] = std_error
        rmse.append(np.sqrt(np.mean(errors**2)))
        mean_std_error.append(std_errors.mean())

    return {
        "n_values": np.array(n_values),
        "rmse": np.array(rmse),
        "mean_std_error": np.array(mean_std_error),
        "exact": exact,
    }


def estimate_convergence_rate(n_values, errors):
    """Fit log(error) = slope * log(n) + c and return the slope.

    Monte Carlo error should scale as n^(-1/2), so the slope should be
    close to -0.5.
    """
    slope, _ = np.polyfit(np.log(n_values), np.log(errors), 1)
    return float(slope)


def run_validation(seed=0):
    """Run both validation checks and print a report.

    Returns
    -------
    tuple
        (comparison results, convergence results).
    """
    print("=== Monte Carlo vs Black-Scholes ===")
    header = (
        f"{'type':<5} {'S0':>5} {'K':>5} {'sigma':>6} {'T':>5} "
        f"{'exact':>9} {'MC':>9} {'std err':>8} {'z':>6}  result"
    )
    print(header)
    print("-" * len(header))

    comparison = compare_methods(seed=seed)
    for row in comparison:
        print(
            f"{row['option_type']:<5} {row['S0']:>5} {row['K']:>5} "
            f"{row['sigma']:>6.2f} {row['T']:>5.2f} "
            f"{row['exact']:>9.4f} {row['estimate']:>9.4f} "
            f"{row['std_error']:>8.4f} {row['z_score']:>6.2f}  "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )

    n_passed = sum(row["passed"] for row in comparison)
    print(f"\n{n_passed} of {len(comparison)} cases within 3 standard errors.\n")

    print("=== Convergence (ATM call, S0=K=100, r=0.05, sigma=0.2, T=1) ===")
    convergence = convergence_study(100, 100, 0.05, 0.2, 1.0, "call", seed=seed)
    print(f"{'n_paths':>10} {'RMSE':>10} {'mean std err':>13}")
    for n, err, se in zip(
        convergence["n_values"], convergence["rmse"], convergence["mean_std_error"]
    ):
        print(f"{n:>10,} {err:>10.5f} {se:>13.5f}")

    slope = estimate_convergence_rate(convergence["n_values"], convergence["rmse"])
    print(f"\nFitted log-log slope: {slope:.3f} (theory: -0.5)")

    return comparison, convergence


if __name__ == "__main__":
    run_validation()