"""End-to-end demonstration of the option_pricer package.

Run from the project root with:
    python demo/run_demo.py
"""

from pathlib import Path

from option_pricer.black_scholes import bs_price
from option_pricer.monte_carlo import mc_price
from option_pricer.plots import make_all_plots
from option_pricer.validation import run_validation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def single_option_example():
    """Price one European call and put with both methods."""
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
    print("=== Single option example ===")
    print(f"S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}\n")

    for option_type in ("call", "put"):
        exact = bs_price(S0, K, r, sigma, T, option_type)
        estimate, std_error = mc_price(
            S0, K, r, sigma, T, option_type, n_paths=1_000_000, seed=1
        )
        print(f"European {option_type}:")
        print(f"  Black-Scholes (exact): {exact:.4f}")
        print(f"  Monte Carlo:           {estimate:.4f} +/- {std_error:.4f}")
        print(f"  95% interval:          [{estimate - 1.96 * std_error:.4f}, "
              f"{estimate + 1.96 * std_error:.4f}]")
        print(f"  Exact price inside interval: "
              f"{abs(estimate - exact) <= 1.96 * std_error}\n")


def main():
    single_option_example()
    run_validation()
    print()
    make_all_plots(DATA_DIR)


if __name__ == "__main__":
    main()