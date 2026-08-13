import numpy as np
from scipy.special import factorial


def hermite_monomial_coeff(n: int, m: int) -> float:
    """Coefficient of x^m in probabilists' Hermite He_n(x)."""
    if m > n or (n - m) % 2 != 0:
        return 0.0
    k = (n - m) // 2
    return float(
        ((-1) ** k * factorial(n)) / ((2**k) * factorial(m) * factorial(k))
    )


def build_H_even(N: int) -> np.ndarray:
    """H_even: even Hermite coeffs -> even monomial coeffs [x^0, x^2, ..., ]."""
    n_even = N // 2 + 1
    H = np.zeros((n_even, n_even))
    for col, deg in enumerate(range(0, 2 * n_even, 2)):
        for row, power in enumerate(range(0, 2 * n_even, 2)):
            H[row, col] = hermite_monomial_coeff(deg, power)
    return H


def build_H_odd(N: int) -> np.ndarray:
    """H_odd: odd Hermite coeffs -> odd monomial coeffs [x^1, x^3, ..., ]."""
    n_odd = (N + 1) // 2
    H = np.zeros((n_odd, n_odd))
    for col, deg in enumerate(range(1, 2 * n_odd, 2)):
        for row, power in enumerate(range(1, 2 * n_odd, 2)):
            H[row, col] = hermite_monomial_coeff(deg, power)
    return H


def build_A_even(N: int, epsilon: float) -> np.ndarray:
    """
    Even gaussian-expansion matrix.
    A[j, k] = 2 / (2j)! * exp(-k^2 ε^2 / 2) * (k ε)^{2j}
    for power x^{2j} and center offset k = 0..N/2.
    """
    n_even = N // 2 + 1
    A = np.zeros((n_even, n_even))
    for j in range(n_even):
        for k in range(n_even):
            power_term = 1.0 if (k == 0 and j == 0) else (k * epsilon) ** (2 * j)
            A[j, k] = (
                2.0
                / factorial(2 * j)
                * np.exp(-0.5 * (k * epsilon) ** 2)
                * power_term
            )
    return A


def build_A_odd(N: int, epsilon: float) -> np.ndarray:
    """
    Odd gaussian-expansion matrix.
    A[j, k] = 2 / (2j+1)! * exp(-(k+1)^2 ε^2 / 2) * ((k+1) ε)^{2j+1}
    for power x^{2j+1} and center offset i = k+1 = 1..(N+1)/2.
    """
    n_odd = (N + 1) // 2
    A = np.zeros((n_odd, n_odd))
    for j in range(n_odd):
        for k in range(n_odd):
            i = k + 1
            A[j, k] = (
                2.0
                / factorial(2 * j + 1)
                * np.exp(-0.5 * (i * epsilon) ** 2)
                * (i * epsilon) ** (2 * j + 1)
            )
    return A


def compute_R(N: int, epsilon: float):
    """
    Compute R = A^{-1} for even and odd blocks.

    Given monomial coefficient vector p (split into even/odd powers),
    gaussian weights are w = R @ p_block, and

      P(x) e^{-x^2/2} ≈ G_even(x)·w_even + G_odd(x)·w_odd

    where
      G_even = [2 e^{-x^2/2}, ..., e^{-(x-kε)^2/2}+e^{-(x+kε)^2/2}, ...]
      G_odd  = [e^{-(x-kε)^2/2}-e^{-(x+kε)^2/2}, ...]  (k = 1, 2, ...)

    Hermite conversion h = H^{-1} p is available separately; since p = H h,
    w = A^{-1} p = A^{-1} H h.
    """
    H_even = build_H_even(N)
    A_even = build_A_even(N, epsilon)
    R_even = np.linalg.inv(A_even)

    H_odd = build_H_odd(N)
    A_odd = build_A_odd(N, epsilon)
    R_odd = np.linalg.inv(A_odd) if A_odd.size else np.zeros((0, 0))

    return {
        "H_even": H_even,
        "A_even": A_even,
        "R_even": R_even,
        "H_odd": H_odd,
        "A_odd": A_odd,
        "R_odd": R_odd,
    }


def split_even_odd_coeffs(p: np.ndarray, N: int):
    """Split monomial coeffs p[i]=coeff of x^i into even/odd blocks of size for degree N."""
    p = np.asarray(p, dtype=float)
    n_even = N // 2 + 1
    n_odd = (N + 1) // 2
    p_even = np.array(
        [p[2 * j] if 2 * j < len(p) else 0.0 for j in range(n_even)], dtype=float
    )
    p_odd = np.array(
        [p[2 * j + 1] if 2 * j + 1 < len(p) else 0.0 for j in range(n_odd)],
        dtype=float,
    )
    return p_even, p_odd


def gaussian_weights(p: np.ndarray, N: int, epsilon: float):
    """Map full monomial coefficient vector p to (w_even, w_odd) via R = A^{-1}."""
    result = compute_R(N, epsilon)
    p_even, p_odd = split_even_odd_coeffs(p, N)
    w_even = result["R_even"] @ p_even
    w_odd = result["R_odd"] @ p_odd if p_odd.size else np.zeros(0)
    return w_even, w_odd, result


def eval_polynomial(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate P(x) = sum p[i] x^i."""
    y = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(np.asarray(p, dtype=float)):
        y += coeff * x**i
    return y


def eval_gaussian_sum(
    x: np.ndarray, w_even: np.ndarray, w_odd: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Evaluate sum of shifted gaussians for the enveloped polynomial:

      G_even·w_even + G_odd·w_odd
    """
    y = np.zeros_like(x, dtype=float)
    if len(w_even):
        y += w_even[0] * 2.0 * np.exp(-0.5 * x**2)
        for k in range(1, len(w_even)):
            y += w_even[k] * (
                np.exp(-0.5 * (x - k * epsilon) ** 2)
                + np.exp(-0.5 * (x + k * epsilon) ** 2)
            )
    for k in range(1, len(w_odd) + 1):
        y += w_odd[k - 1] * (
            np.exp(-0.5 * (x - k * epsilon) ** 2)
            - np.exp(-0.5 * (x + k * epsilon) ** 2)
        )
    return y


def approximation_error(
    p: np.ndarray,
    N: int,
    epsilon: float,
    x: np.ndarray | None = None,
) -> dict:
    """
    Compare P(x) e^{-x^2/2} against the gaussian sum from R.

    Returns max absolute and relative errors on the sample grid.
    """
    if x is None:
        x = np.linspace(-2.0, 2.0, 401)

    w_even, w_odd, _ = gaussian_weights(p, N, epsilon)
    target = eval_polynomial(p, x) * np.exp(-0.5 * x**2)
    approx = eval_gaussian_sum(x, w_even, w_odd, epsilon)
    abs_err = np.abs(target - approx)
    max_abs = float(np.max(abs_err))
    scale = float(np.max(np.abs(target))) + 1e-15
    return {
        "max_abs": max_abs,
        "max_rel": max_abs / scale,
        "target": target,
        "approx": approx,
        "x": x,
        "w_even": w_even,
        "w_odd": w_odd,
    }


def _print_matrix(name: str, M: np.ndarray) -> None:
    print(f"\n{name} ({M.shape[0]}x{M.shape[1]}):")
    if M.size == 0:
        print("  <empty>")
        return
    with np.printoptions(suppress=True, linewidth=np.inf, precision=8):
        for row in M:
            print(row)


if __name__ == "__main__":
    N = 4
    epsilon = 1.0

    result = compute_R(N, epsilon)

    print(f"N = {N}, epsilon = {epsilon}")
    for key in ("H_even", "A_even", "R_even", "H_odd", "A_odd", "R_odd"):
        _print_matrix(key, result[key])
