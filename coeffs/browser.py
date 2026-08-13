"""Browser entrypoint loaded into Pyodide."""

from __future__ import annotations

import json
import math

from coeffs.pipeline import latex_to_gaussians, parse_expression


def _error_payload(exc: BaseException) -> str:
    return json.dumps({"ok": False, "error": str(exc).strip()})


def parse_check(latex: str) -> str:
    """Validate / preview-parse an expression in {x,y,z}. Always returns JSON."""
    try:
        expr = parse_expression(latex)
        free = sorted(s.name for s in expr.free_symbols)
        return json.dumps(
            {
                "ok": True,
                "parsed": str(expr),
                "variables": free or ["x"],
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return _error_payload(exc)


def decompose(
    latex: str,
    N: int = 4,
    epsilon: float = 0.5,
    x0: float = 0.0,
) -> str:
    """
    Run the full pipeline and return a JSON string for the JS UI.

    On failure returns {"ok": false, "error": "..."} instead of raising.
    """
    try:
        approx = latex_to_gaussians(
            latex, N=int(N), epsilon=float(epsilon), x0=float(x0)
        )
        err = approx.error(remove_envelope=True)
        max_rel = err["max_rel"]
        max_abs = err["max_abs"]
        payload = {
            "ok": True,
            "error": None,
            "latex": approx.latex,
            "parsed": str(approx.expr),
            "taylor": str(approx.taylor),
            "N": approx.N,
            "epsilon": approx.epsilon,
            "x0": approx.x0,
            "dims": approx.dims,
            "variables": list(approx.variables),
            "coeffs": [float(c) for c in approx.coeffs],
            "w_even": [float(w) for w in approx.w_even],
            "w_odd": [float(w) for w in approx.w_odd],
            "centers": [
                {
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                    "z": float(c["z"]),
                    "weight": float(c["weight"]),
                    # backward-compatible alias used by the 1D canvas
                    "center": float(c["x"]),
                }
                for c in approx.centers
            ],
            "max_rel_error": None if (isinstance(max_rel, float) and math.isnan(max_rel)) else float(max_rel),
            "max_abs_error": None if (isinstance(max_abs, float) and math.isnan(max_abs)) else float(max_abs),
            "splat_count": len(approx.centers),
        }
        return json.dumps(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return _error_payload(exc)
