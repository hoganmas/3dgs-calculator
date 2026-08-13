"""
Tests that R = A^{-1} yields a sum-of-gaussians approximation of P(x) e^{-x^2/2}.

Run:
  .venv/bin/python -m unittest coeffs.test_approximation -v
"""

from __future__ import annotations

import unittest

import numpy as np

from coeffs.gaussians import (
    approximation_error,
    build_H_even,
    build_H_odd,
    compute_R,
    eval_gaussian_sum,
    eval_polynomial_from_gaussians,
    gaussian_weights,
    hermite_monomial_coeff,
)


class TestHermiteBasis(unittest.TestCase):
    def test_known_hermite_polynomials(self):
        # He_0=1, He_1=x, He_2=x^2-1, He_3=x^3-3x, He_4=x^4-6x^2+3
        self.assertEqual(hermite_monomial_coeff(0, 0), 1.0)
        self.assertEqual(hermite_monomial_coeff(1, 1), 1.0)
        self.assertEqual(hermite_monomial_coeff(2, 0), -1.0)
        self.assertEqual(hermite_monomial_coeff(2, 2), 1.0)
        self.assertEqual(hermite_monomial_coeff(3, 1), -3.0)
        self.assertEqual(hermite_monomial_coeff(3, 3), 1.0)
        self.assertEqual(hermite_monomial_coeff(4, 0), 3.0)
        self.assertEqual(hermite_monomial_coeff(4, 2), -6.0)
        self.assertEqual(hermite_monomial_coeff(4, 4), 1.0)

    def test_H_inverse_matches_notes(self):
        H = build_H_even(4)
        H_inv = np.linalg.inv(H)
        expected = np.array(
            [
                [1.0, 1.0, 3.0],
                [0.0, 1.0, 6.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(H_inv, expected, atol=1e-12)

        H_odd = build_H_odd(4)
        H_odd_inv = np.linalg.inv(H_odd)
        expected_odd = np.array(
            [
                [1.0, 3.0],
                [0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(H_odd_inv, expected_odd, atol=1e-12)


class TestHe2AnalyticalWeights(unittest.TestCase):
    def test_he2_weights_match_derivation(self):
        """
        For He_2(x)=x^2-1 with center convention G0=2 e^{-x^2/2}:
          w0 = -1/2 - 1/ε^2
          w1 = e^{ε^2/2} / ε^2
        """
        eps = 0.75
        p = np.array([-1.0, 0.0, 1.0])
        w_even, w_odd, _ = gaussian_weights(p, N=2, epsilon=eps)

        expected_w0 = -0.5 - 1.0 / eps**2
        expected_w1 = np.exp(0.5 * eps**2) / eps**2
        np.testing.assert_allclose(w_even, [expected_w0, expected_w1], rtol=1e-12)
        np.testing.assert_allclose(w_odd, np.zeros(1), atol=1e-12)


class TestGaussianApproximation(unittest.TestCase):
    def test_constant_polynomial(self):
        err = approximation_error(np.array([1.0]), N=0, epsilon=0.5)
        self.assertLess(err["max_abs"], 1e-12)

    def test_linear_polynomial_improves_with_smaller_epsilon(self):
        p = np.array([0.0, 1.0])
        coarse = approximation_error(p, N=1, epsilon=0.8)
        fine = approximation_error(p, N=1, epsilon=0.2)
        self.assertLess(fine["max_abs"], coarse["max_abs"])
        self.assertLess(fine["max_rel"], 0.05)

    def test_quadratic_he2_improves_with_smaller_epsilon(self):
        p = np.array([-1.0, 0.0, 1.0])  # He_2
        coarse = approximation_error(p, N=2, epsilon=1.0)
        fine = approximation_error(p, N=2, epsilon=0.25)
        self.assertLess(fine["max_abs"], coarse["max_abs"])
        self.assertLess(fine["max_rel"], 0.05)

    def test_mixed_degree4_polynomial(self):
        p = np.array([1.0, -0.5, 0.25, 0.1, -0.05])
        err = approximation_error(p, N=4, epsilon=0.2)
        self.assertLess(err["max_rel"], 0.05)

    def test_odd_cubic_polynomial(self):
        p = np.array([0.0, 1.0, 0.0, -0.3])
        err = approximation_error(p, N=3, epsilon=0.2)
        self.assertLess(err["max_rel"], 0.08)

    def test_A_recovers_monomial_coeffs(self):
        N, eps = 4, 0.4
        p = np.array([2.0, -1.0, 0.5, 0.25, -0.1])
        w_even, w_odd, result = gaussian_weights(p, N, eps)

        p_even = np.array([p[0], p[2], p[4]])
        p_odd = np.array([p[1], p[3]])
        np.testing.assert_allclose(result["A_even"] @ w_even, p_even, atol=1e-10)
        np.testing.assert_allclose(result["A_odd"] @ w_odd, p_odd, atol=1e-10)

    def test_reconstruction_is_even_or_odd_when_poly_is(self):
        x = np.linspace(-2.0, 2.0, 201)
        w_even, w_odd, _ = gaussian_weights(np.array([1.0, 0.0, -0.5, 0.0, 0.1]), 4, 0.3)
        approx = eval_gaussian_sum(x, w_even, w_odd, 0.3)
        np.testing.assert_allclose(approx, approx[::-1], atol=1e-10)

        w_even, w_odd, _ = gaussian_weights(np.array([0.0, 1.0, 0.0, -0.2]), 3, 0.3)
        approx = eval_gaussian_sum(x, w_even, w_odd, 0.3)
        np.testing.assert_allclose(approx, -approx[::-1], atol=1e-10)

    def test_envelope_removal_recovers_polynomial(self):
        p = np.array([1.0, -0.5, 0.25, 0.1, -0.05])
        err = approximation_error(p, N=4, epsilon=0.2, remove_envelope=True)
        self.assertLess(err["max_rel"], 0.05)

        x = np.linspace(-1.5, 1.5, 201)
        w_even, w_odd, _ = gaussian_weights(p, 4, 0.2)
        poly_hat = eval_polynomial_from_gaussians(x, w_even, w_odd, 0.2)
        # Sanity: not identically the enveloped curve
        enveloped = eval_gaussian_sum(x, w_even, w_odd, 0.2)
        self.assertGreater(float(np.max(np.abs(poly_hat - enveloped))), 0.1)


class TestMatrixShapes(unittest.TestCase):
    def test_shapes_even_N(self):
        result = compute_R(4, 1.0)
        self.assertEqual(result["R_even"].shape, (3, 3))
        self.assertEqual(result["R_odd"].shape, (2, 2))
        self.assertEqual(result["A_even"].shape, (3, 3))
        self.assertEqual(result["H_even"].shape, (3, 3))

    def test_shapes_odd_N(self):
        result = compute_R(5, 1.0)
        self.assertEqual(result["R_even"].shape, (3, 3))
        self.assertEqual(result["R_odd"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
