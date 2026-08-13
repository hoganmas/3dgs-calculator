"""Visualize a gaussian-splat decomposition of a Taylor polynomial."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coeffs.gaussians import (
    eval_gaussian_sum,
    eval_polynomial,
    eval_polynomial_from_gaussians,
)
from coeffs.pipeline import GaussianApproximation


def _single_gaussian(x: np.ndarray, center: float, weight: float) -> np.ndarray:
    return weight * np.exp(-0.5 * (x - center) ** 2)


def sp_latex(approx: GaussianApproximation) -> str:
    """Best-effort title fragment from the input expression."""
    try:
        from sympy import latex

        return latex(approx.expr)
    except Exception:
        return approx.latex.replace("$", "")


def plot_gaussian_splats(
    approx: GaussianApproximation,
    *,
    x_min: float = -3.0,
    x_max: float = 3.0,
    num: int = 800,
    show_original: bool = True,
):
    """
    Draw the decomposition as overlapping gaussian splats.

    Top: individual weighted gaussians (left axis) and their sum vs
         P(x)e^{-x^2/2} (right axis).
    Bottom: Taylor polynomial vs envelope-removed reconstruction.
    """
    x = np.linspace(x_min, x_max, num)
    centers = [(c, w) for c, w in approx.centers if abs(w) > 1e-12]
    weights = np.array([abs(w) for _, w in centers], dtype=float)
    vmax = float(np.max(weights)) if weights.size else 1.0

    fig, (ax_splats, ax_poly) = plt.subplots(
        2,
        1,
        figsize=(10, 7.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    ax_sum = ax_splats.twinx()

    cmap_pos = plt.cm.Blues
    cmap_neg = plt.cm.Reds
    gauss_sum = eval_gaussian_sum(x, approx.w_even, approx.w_odd, approx.epsilon)
    target_env = eval_polynomial(approx.coeffs, x) * np.exp(-0.5 * x**2)

    for center, weight in centers:
        y = _single_gaussian(x, center, weight)
        strength = abs(weight) / vmax
        cmap = cmap_pos if weight >= 0 else cmap_neg
        color = cmap(0.35 + 0.55 * strength)
        ax_splats.fill_between(x, 0.0, y, color=color, alpha=0.25, linewidth=0)
        ax_splats.plot(x, y, color=color, lw=1.1, alpha=0.85)

    for center, weight in centers:
        strength = abs(weight) / vmax
        color = "#2f6db3" if weight >= 0 else "#c23b3b"
        ax_splats.scatter(
            [center],
            [0.0],
            s=35 + 160 * strength,
            c=[color],
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )

    (line_sum,) = ax_sum.plot(
        x,
        gauss_sum,
        color="#1a1a1a",
        lw=2.3,
        label=r"$\sum W_j\,e^{-(x-c_j)^2/2}$",
        zorder=6,
    )
    (line_tgt,) = ax_sum.plot(
        x,
        target_env,
        color="#2a6f4e",
        lw=1.9,
        ls="--",
        label=r"$P(x)\,e^{-x^2/2}$",
        zorder=6,
    )

    ax_splats.axhline(0.0, color="#999999", lw=0.8)
    ax_splats.set_ylabel("individual splats", color="#555555")
    ax_sum.set_ylabel("sum / target", color="#1a1a1a")
    ax_splats.set_title(
        rf"Gaussian splats for ${sp_latex(approx)}$"
        + rf"  (N={approx.N}, $\epsilon$={approx.epsilon:g})"
    )
    ax_sum.legend(handles=[line_sum, line_tgt], loc="upper right", frameon=False)
    ax_splats.grid(True, alpha=0.25)

    # Keep sum axis focused on the reconstructed signal, not the huge cancelling lobes
    sum_ref = np.concatenate([gauss_sum, target_env])
    sum_ref = sum_ref[np.isfinite(sum_ref)]
    if sum_ref.size:
        pad = 0.2 * (np.max(np.abs(sum_ref)) + 1e-9)
        ax_sum.set_ylim(float(np.min(sum_ref) - pad), float(np.max(sum_ref) + pad))

    poly = eval_polynomial(approx.coeffs, x)
    poly_hat = eval_polynomial_from_gaussians(
        x, approx.w_even, approx.w_odd, approx.epsilon
    )
    ax_poly.plot(x, poly, color="#2a6f4e", lw=2.0, label=r"Taylor $P(x)$")
    ax_poly.plot(
        x,
        poly_hat,
        color="#1a1a1a",
        lw=2.0,
        ls="--",
        label=r"$e^{x^2/2}\sum W_j\,e^{-(x-c_j)^2/2}$",
    )

    if show_original:
        try:
            f = np.vectorize(lambda t: float(approx.expr.subs({"x": t})))
            y_orig = f(x)
            if np.all(np.isfinite(y_orig)):
                ax_poly.plot(
                    x,
                    y_orig,
                    color="#8a6d3b",
                    lw=1.4,
                    alpha=0.85,
                    label=r"original $f(x)$",
                )
        except Exception:
            pass

    ax_poly.axhline(0.0, color="#999999", lw=0.8)
    ax_poly.set_xlabel("x")
    ax_poly.set_ylabel("polynomial")
    ax_poly.legend(loc="best", frameon=False)
    ax_poly.grid(True, alpha=0.25)

    y_ref = np.concatenate([poly, poly_hat])
    y_ref = y_ref[np.isfinite(y_ref)]
    if y_ref.size:
        pad = 0.15 * (np.max(np.abs(y_ref)) + 1e-9)
        ax_poly.set_ylim(float(np.min(y_ref) - pad), float(np.max(y_ref) + pad))

    return fig, (ax_splats, ax_poly)


def save_or_show(
    approx: GaussianApproximation,
    *,
    output: str | Path | None = None,
    x_min: float = -3.0,
    x_max: float = 3.0,
    show: bool = True,
):
    fig, _ = plot_gaussian_splats(approx, x_min=x_min, x_max=x_max)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
