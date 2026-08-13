"""LaTeX expression → Taylor polynomial → sum of gaussians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex

from coeffs.gaussians import (
    approximation_error,
    expand_gaussian_centers,
    gaussian_weights,
)


@dataclass
class GaussianApproximation:
    latex: str
    expr: sp.Expr
    taylor: sp.Expr
    N: int
    epsilon: float
    x0: float
    coeffs: np.ndarray
    w_even: np.ndarray
    w_odd: np.ndarray
    centers: list[tuple[float, float]]

    def error(self, x: np.ndarray | None = None, *, remove_envelope: bool = True) -> dict:
        return approximation_error(
            self.coeffs,
            self.N,
            self.epsilon,
            x=x,
            remove_envelope=remove_envelope,
        )


def parse_expression(latex: str, variable: str = "x") -> sp.Expr:
    """
    Parse a LaTeX math expression into a SymPy expression.

    Also accepts plain SymPy strings (e.g. 'sin(x)') as a fallback.
    """
    text = latex.strip()
    if not text:
        raise ValueError("Expression is empty.")

    x = sp.Symbol(variable)
    replacements = {sp.Symbol("e"): sp.E, sp.Symbol("E"): sp.E}

    candidates: list[sp.Expr] = []
    for parser in (parse_latex, sp.sympify):
        try:
            candidates.append(parser(text))
        except Exception:
            continue

    rejected: list[str] = []
    for raw in candidates:
        expr = raw.subs(replacements)
        free = {str(s) for s in expr.free_symbols}
        if free <= {variable}:
            return sp.simplify(expr.subs(sp.Symbol(variable), x))
        extras = sorted(free - {variable})
        rejected.append(f"{expr} (extra symbols: {', '.join(extras)})")

    if rejected:
        raise ValueError(
            f"Need a function of '{variable}' only — {rejected[0]}"
        )
    raise ValueError(f"Could not parse expression: {text}")

def taylor_coefficients(
    expr: sp.Expr,
    N: int,
    *,
    variable: str = "x",
    x0: float = 0.0,
) -> tuple[np.ndarray, sp.Expr]:
    """
    Return monomial coefficients p[i] of the degree-N Taylor polynomial of expr
    about x0, together with the SymPy truncated series (without O term).
    """
    x = sp.Symbol(variable)
    series = expr.series(x, x0, N + 1).removeO()
    poly = sp.Poly(sp.expand(series), x)
    coeffs = np.array([float(poly.coeff_monomial(x**i)) for i in range(N + 1)])
    return coeffs, series


def latex_to_gaussians(
    latex: str,
    N: int,
    epsilon: float,
    *,
    x0: float = 0.0,
    variable: str = "x",
) -> GaussianApproximation:
    """
    Full pipeline:
      LaTeX → symbolic expr → degree-N Taylor poly → R=A^{-1} → gaussian weights
    """
    expr = parse_expression(latex, variable=variable)
    coeffs, taylor = taylor_coefficients(expr, N, variable=variable, x0=x0)
    w_even, w_odd, _ = gaussian_weights(coeffs, N, epsilon)
    centers = expand_gaussian_centers(w_even, w_odd, epsilon)
    return GaussianApproximation(
        latex=latex,
        expr=expr,
        taylor=taylor,
        N=N,
        epsilon=epsilon,
        x0=x0,
        coeffs=coeffs,
        w_even=w_even,
        w_odd=w_odd,
        centers=centers,
    )


def format_gaussian_sum(centers: list[tuple[float, float]], precision: int = 6) -> str:
    """Pretty-print sum_j W_j exp(-(x - c_j)^2 / 2)."""
    parts = []
    for center, weight in centers:
        if abs(weight) < 10 ** (-precision):
            continue
        w = f"{weight:.{precision}g}"
        if abs(center) < 1e-15:
            parts.append(f"{w} * exp(-x^2 / 2)")
        else:
            parts.append(f"{w} * exp(-(x - ({center:.{precision}g}))^2 / 2)")
    return " + ".join(parts) if parts else "0"
