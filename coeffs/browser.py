"""Browser entrypoint loaded into Pyodide."""

from __future__ import annotations

import json

from coeffs.pipeline import latex_to_gaussians, parse_expression


def _error_payload(exc: BaseException) -> str:
    return json.dumps({"ok": False, "error": str(exc).strip()})


def parse_check(latex: str) -> str:
    """Validate / preview-parse an expression. Always returns JSON."""
    try:
        expr = parse_expression(latex)
        return json.dumps({"ok": True, "parsed": str(expr), "error": None})
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

    On failure returns {"ok": false, "error": "..."} instead of raising,
    so the worker never surfaces a Python traceback.
    """
    try:
        approx = latex_to_gaussians(
            latex, N=int(N), epsilon=float(epsilon), x0=float(x0)
        )
        err = approx.error(remove_envelope=True)
        payload = {
            "ok": True,
            "error": None,
            "latex": approx.latex,
            "parsed": str(approx.expr),
            "taylor": str(approx.taylor),
            "N": approx.N,
            "epsilon": approx.epsilon,
            "x0": approx.x0,
            "coeffs": [float(c) for c in approx.coeffs],
            "w_even": [float(w) for w in approx.w_even],
            "w_odd": [float(w) for w in approx.w_odd],
            "centers": [
                {"center": float(c), "weight": float(w)} for c, w in approx.centers
            ],
            "max_rel_error": float(err["max_rel"]),
            "max_abs_error": float(err["max_abs"]),
        }
        return json.dumps(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return _error_payload(exc)
