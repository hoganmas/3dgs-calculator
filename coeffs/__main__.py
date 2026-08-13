"""CLI: print R matrices, or approximate a LaTeX expression by gaussians."""

from __future__ import annotations

import argparse

import numpy as np

from coeffs.gaussians import compute_R
from coeffs.pipeline import format_gaussian_sum, latex_to_gaussians
from coeffs.plot import save_or_show


def _print_matrix(name: str, M: np.ndarray) -> None:
    print(f"\n{name} ({M.shape[0]}x{M.shape[1]}):")
    if M.size == 0:
        print("  <empty>")
        return
    with np.printoptions(suppress=True, linewidth=np.inf, precision=8):
        for row in M:
            print(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Decompose polynomials / LaTeX expressions into sums of gaussians."
    )
    parser.add_argument(
        "latex",
        nargs="?",
        default=None,
        help=r'LaTeX expression, e.g. "\sin(x)" or "e^{x}"',
    )
    parser.add_argument("-N", type=int, default=4, help="Taylor / polynomial degree")
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.5,
        help="Spacing between gaussian centers",
    )
    parser.add_argument(
        "--x0", type=float, default=0.0, help="Taylor expansion point"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show gaussian-splat visualization",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save plot to this path (implies plotting without requiring a display if --no-show)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --save)",
    )
    parser.add_argument("--xmin", type=float, default=-3.0)
    parser.add_argument("--xmax", type=float, default=3.0)
    args = parser.parse_args(argv)

    if args.latex is None:
        result = compute_R(args.N, args.epsilon)
        print(f"N = {args.N}, epsilon = {args.epsilon}")
        for key in ("H_even", "A_even", "R_even", "H_odd", "A_odd", "R_odd"):
            _print_matrix(key, result[key])
        return

    approx = latex_to_gaussians(
        args.latex, N=args.N, epsilon=args.epsilon, x0=args.x0
    )
    err = approx.error(remove_envelope=True)

    print(f"input:    {approx.latex}")
    print(f"parsed:   {approx.expr}")
    print(f"taylor:   {approx.taylor}")
    print(f"N={approx.N}, epsilon={approx.epsilon}, x0={approx.x0}")
    print(f"coeffs p: {approx.coeffs}")
    print(f"w_even:   {approx.w_even}")
    print(f"w_odd:    {approx.w_odd}")
    print("centers (c, W):")
    for center, weight in approx.centers:
        print(f"  {center:+.6g}: {weight:.8g}")
    print(f"sum ≈ {format_gaussian_sum(approx.centers)}")
    print(
        f"P(x) ≈ e^{{x^2/2}} * sum; "
        f"max_rel error on [-2,2] = {err['max_rel']:.3e}"
    )

    if args.plot or args.save:
        save_or_show(
            approx,
            output=args.save,
            x_min=args.xmin,
            x_max=args.xmax,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
