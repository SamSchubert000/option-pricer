"""Plots comparing the Monte Carlo pricer with the Black-Scholes benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from option_pricer.black_scholes import bs_price
from option_pricer.monte_carlo import mc_price
from option_pricer.validation import convergence_study, estimate_convergence_rate


def _save(fig, save_path):
    """Save a figure to ``save_path`` (creating folders) if a path is given."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_convergence(convergence, save_path=None):
    """Log-log plot of Monte Carlo error against the number of paths.

    Parameters
    ----------
    convergence : dict
        Output of ``validation.convergence_study``.
    save_path : str or Path or None
        If given, the figure is saved there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = np.asarray(convergence["n_values"], dtype=float)
    rmse = np.asarray(convergence["rmse"])
    mean_se = np.asarray(convergence["mean_std_error"])

    # Reference line with slope -1/2, anchored at the first RMSE value.
    reference = rmse[0] * np.sqrt(n[0] / n)
    slope = estimate_convergence_rate(n, rmse)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n, rmse, "o-", label=f"RMSE vs exact (fitted slope {slope:.2f})")
    ax.loglog(n, mean_se, "s--", label="Mean reported standard error")
    ax.loglog(n, reference, ":", color="grey", label=r"$1/\sqrt{N}$ reference")

    ax.set_xlabel("Number of paths, N")
    ax.set_ylabel("Error in option price")
    ax.set_title("Monte Carlo convergence to the Black-Scholes price")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _save(fig, save_path)
    return fig


def plot_price_vs_strike(
    S0=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type="call",
    strikes=None,
    n_paths=200_000,
    seed=0,
    save_path=None,
):
    """Compare Black-Scholes and Monte Carlo prices across strikes.

    The top panel shows the exact price curve with Monte Carlo estimates
    (error bars are 95% intervals, which are small at this scale). The bottom
    panel shows the difference in standard errors, which should stay mostly
    within the +/-2 band if the pricer is correct.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if strikes is None:
        strikes = np.linspace(0.6 * S0, 1.4 * S0, 17)
    strikes = np.asarray(strikes, dtype=float)

    dense_strikes = np.linspace(strikes.min(), strikes.max(), 200)
    bs_curve = [bs_price(S0, k, r, sigma, T, option_type) for k in dense_strikes]

    rng = np.random.default_rng(seed)
    mc_prices = np.empty(len(strikes))
    mc_errors = np.empty(len(strikes))
    exact_at_strikes = np.empty(len(strikes))

    for i, k in enumerate(strikes):
        mc_prices[i], mc_errors[i] = mc_price(
            S0,
            k,
            r,
            sigma,
            T,
            option_type,
            n_paths=n_paths,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        exact_at_strikes[i] = bs_price(S0, k, r, sigma, T, option_type)

    z_scores = (mc_prices - exact_at_strikes) / mc_errors

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax_top.plot(dense_strikes, bs_curve, "-", label="Black-Scholes (exact)")
    ax_top.errorbar(
        strikes,
        mc_prices,
        yerr=1.96 * mc_errors,
        fmt="o",
        capsize=3,
        label=f"Monte Carlo ({n_paths:,} paths, 95% interval)",
    )
    ax_top.set_ylabel("Option price")
    ax_top.set_title(
        f"European {option_type} price against strike "
        f"(S0={S0}, r={r}, sigma={sigma}, T={T})"
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bottom.axhline(0, color="black", linewidth=0.8)
    ax_bottom.axhline(2, color="grey", linestyle="--", linewidth=0.8)
    ax_bottom.axhline(-2, color="grey", linestyle="--", linewidth=0.8)
    ax_bottom.plot(strikes, z_scores, "o")
    ax_bottom.set_xlabel("Strike, K")
    ax_bottom.set_ylabel("(MC - BS) / std error")
    ax_bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def make_all_plots(output_dir="data", seed=0):
    """Create and save every figure into ``output_dir``."""
    output_dir = Path(output_dir)

    convergence = convergence_study(100, 100, 0.05, 0.2, 1.0, "call", seed=seed)
    fig = plot_convergence(convergence, output_dir / "convergence.png")
    plt.close(fig)

    for option_type in ("call", "put"):
        fig = plot_price_vs_strike(
            option_type=option_type,
            seed=seed,
            save_path=output_dir / f"price_vs_strike_{option_type}.png",
        )
        plt.close(fig)

    print(f"Saved figures to {output_dir.resolve()}")


if __name__ == "__main__":
    make_all_plots(Path(__file__).resolve().parent.parent / "data")