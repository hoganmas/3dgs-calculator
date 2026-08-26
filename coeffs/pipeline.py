"""1D / 3D polynomial → sum-of-gaussians via R = A^{-1} (tensor product in 3D)."""

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

ALLOWED_VARS = frozenset({"x", "y", "z"})


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
    # Each entry: {"x", "y", "z", "weight"}
    centers: list[dict[str, float]]
    dims: int = 1
    coeff_tensor: np.ndarray | None = None
    variables: tuple[str, ...] = ("x",)

    def error(self, x: np.ndarray | None = None, *, remove_envelope: bool = True) -> dict:
        if self.dims != 1:
            return {
                "max_abs": float("nan"),
                "max_rel": float("nan"),
                "target": None,
                "approx": None,
                "x": x,
                "w_even": self.w_even,
                "w_odd": self.w_odd,
            }
        return approximation_error(
            self.coeffs,
            self.N,
            self.epsilon,
            x=x,
            remove_envelope=remove_envelope,
        )


def parse_expression(latex: str, variables: frozenset[str] = ALLOWED_VARS) -> sp.Expr:
    """
    Parse a LaTeX / SymPy expression in variables subset of {x,y,z}.
    """
    text = latex.strip()
    if not text:
        raise ValueError("Expression is empty.")

    replacements = {sp.Symbol("e"): sp.E, sp.Symbol("E"): sp.E}
    canon = {name: sp.Symbol(name) for name in sorted(variables)}

    candidates: list[sp.Expr] = []
    for parser in (parse_latex, sp.sympify):
        try:
            candidates.append(parser(text))
        except Exception:
            continue

    rejected: list[str] = []
    for raw in candidates:
        expr = raw.subs(replacements)
        rename = {
            s: canon[s.name]
            for s in expr.free_symbols
            if getattr(s, "name", None) in variables
        }
        expr = expr.subs(rename)

        free = {s.name for s in expr.free_symbols}
        if free <= variables:
            return sp.simplify(expr)
        extras = sorted(free - set(variables))
        rejected.append(f"{expr} (extra symbols: {', '.join(extras)})")

    allowed = ", ".join(sorted(variables))
    if rejected:
        raise ValueError(f"Need a function of {{{allowed}}} only — {rejected[0]}")
    raise ValueError(f"Could not parse expression: {text}")


def _active_variables(expr: sp.Expr) -> tuple[str, ...]:
    names = {s.name for s in expr.free_symbols} & ALLOWED_VARS
    # Preserve x,y,z order; constants → treat as 1D in x
    ordered = tuple(v for v in ("x", "y", "z") if v in names)
    return ordered if ordered else ("x",)


def taylor_coefficients(
    expr: sp.Expr,
    N: int,
    *,
    variable: str = "x",
    x0: float = 0.0,
) -> tuple[np.ndarray, sp.Expr]:
    """1D monomial coefficients p[i] of the degree-N Taylor polynomial."""
    x = sp.Symbol(variable)
    series = expr.series(x, x0, N + 1).removeO()
    poly = sp.Poly(sp.expand(series), x)
    coeffs = np.array([float(poly.coeff_monomial(x**i)) for i in range(N + 1)])
    return coeffs, series


def multivariate_taylor_coefficients(
    expr: sp.Expr,
    N: int,
    *,
    x0: float = 0.0,
) -> tuple[np.ndarray, sp.Expr, tuple[str, ...]]:
    """
    Coefficient tensor C[i,j,k] of x^i y^j z^k for the truncated Taylor polynomial
    of total per-variable degree ≤ N about (x0,x0,x0).
    """
    x, y, z = sp.symbols("x y z")
    out = expr
    for v in (x, y, z):
        try:
            out = out.series(v, x0, N + 1).removeO()
        except (AttributeError, ValueError, TypeError):
            # Already polynomial / independent of v
            pass
    out = sp.expand(out)
    C = np.zeros((N + 1, N + 1, N + 1), dtype=float)
    try:
        poly = sp.Poly(out, x, y, z)
        for exponents, coeff in poly.as_dict().items():
            i, j, k = exponents
            if i <= N and j <= N and k <= N:
                C[i, j, k] = float(coeff)
    except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError):
        C[0, 0, 0] = float(sp.N(out))
    return C, out, _active_variables(expr)


def _one_d_center_grid(N: int, epsilon: float) -> list[float]:
    k_max = max(N // 2, (N + 1) // 2)
    return [k * epsilon for k in range(-k_max, k_max + 1)]


def monomial_basis_weights(N: int, epsilon: float) -> tuple[list[float], np.ndarray]:
    """
    For each power i=0..N, the 1D splat weight vector on the shared center grid.
    Returns (centers_1d, basis) with basis.shape == (N+1, len(centers_1d)).
    """
    centers_1d = _one_d_center_grid(N, epsilon)
    basis = np.zeros((N + 1, len(centers_1d)), dtype=float)
    for power in range(N + 1):
        p = np.zeros(N + 1, dtype=float)
        p[power] = 1.0
        w_even, w_odd, _ = gaussian_weights(p, N, epsilon)
        expanded = dict(expand_gaussian_centers(w_even, w_odd, epsilon))
        basis[power] = np.array([expanded.get(c, 0.0) for c in centers_1d])
    return centers_1d, basis


def tensor_product_centers(
    coeff_tensor: np.ndarray,
    N: int,
    epsilon: float,
    *,
    tol: float = 1e-12,
) -> list[dict[str, float]]:
    """
    P(x,y,z)=Σ C_ijk x^i y^j z^k  →  3D splat weights via
    W = Σ C_ijk (w^{(i)} ⊗ w^{(j)} ⊗ w^{(k)}).
    """
    centers_1d, basis = monomial_basis_weights(N, epsilon)
    n = len(centers_1d)
    W = np.zeros((n, n, n), dtype=float)

    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                c = coeff_tensor[i, j, k]
                if abs(c) < tol:
                    continue
                W += c * np.einsum("a,b,c->abc", basis[i], basis[j], basis[k])

    points: list[dict[str, float]] = []
    for a, cx in enumerate(centers_1d):
        for b, cy in enumerate(centers_1d):
            for c, cz in enumerate(centers_1d):
                w = float(W[a, b, c])
                if abs(w) < tol:
                    continue
                points.append({"x": float(cx), "y": float(cy), "z": float(cz), "weight": w})
    return points


def _centers_1d_as_dicts(pairs: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [
        {"x": float(c), "y": 0.0, "z": 0.0, "weight": float(w), "center": float(c)}
        for c, w in pairs
    ]


def latex_to_gaussians(
    latex: str,
    N: int,
    epsilon: float,
    *,
    x0: float = 0.0,
) -> GaussianApproximation:
    """
    LaTeX → Taylor poly → gaussian splat weights.

    - 1D (only x): original even/odd R path
    - 2D/3D (y and/or z present): multivariate Taylor + tensor-product splats
    """
    expr = parse_expression(latex)
    variables = _active_variables(expr)

    if variables == ("x",):
        coeffs, taylor = taylor_coefficients(expr, N, variable="x", x0=x0)
        w_even, w_odd, _ = gaussian_weights(coeffs, N, epsilon)
        centers = _centers_1d_as_dicts(expand_gaussian_centers(w_even, w_odd, epsilon))
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
            dims=1,
            variables=variables,
        )

    coeff_tensor, taylor, _ = multivariate_taylor_coefficients(expr, N, x0=x0)
    centers = tensor_product_centers(coeff_tensor, N, epsilon)
    coeffs = np.array([float(np.sum(coeff_tensor[i, :, :])) for i in range(N + 1)])

    return GaussianApproximation(
        latex=latex,
        expr=expr,
        taylor=taylor,
        N=N,
        epsilon=epsilon,
        x0=x0,
        coeffs=coeffs,
        w_even=np.zeros(0),
        w_odd=np.zeros(0),
        centers=centers,
        dims=3 if "z" in variables else 2,
        coeff_tensor=coeff_tensor,
        variables=variables,
    )


def format_gaussian_sum(centers: list, precision: int = 6) -> str:
    """Pretty-print sum of gaussians (1D or 3D center dicts)."""
    parts = []
    for entry in centers:
        if isinstance(entry, dict):
            weight = entry["weight"]
            x, y, z = entry.get("x", 0.0), entry.get("y", 0.0), entry.get("z", 0.0)
        else:
            x, weight = entry
            y = z = 0.0
        if abs(weight) < 10 ** (-precision):
            continue
        w = f"{weight:.{precision}g}"
        if abs(y) < 1e-15 and abs(z) < 1e-15:
            if abs(x) < 1e-15:
                parts.append(f"{w} * exp(-r^2 / 2)")
            else:
                parts.append(f"{w} * exp(-(x - ({x:.{precision}g}))^2 / 2)")
        else:
            parts.append(
                f"{w} * exp(-0.5*((x-{x:.{precision}g})^2+"
                f"(y-{y:.{precision}g})^2+(z-{z:.{precision}g})^2))"
            )
    return " + ".join(parts) if parts else "0"
