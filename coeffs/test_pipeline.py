"""Tests for LaTeX → Taylor → gaussian pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from coeffs.pipeline import latex_to_gaussians, parse_expression, taylor_coefficients


class TestParseLatex(unittest.TestCase):
    def test_sin(self):
        expr = parse_expression(r"\sin(x)")
        self.assertEqual(str(expr), "sin(x)")

    def test_exp(self):
        expr = parse_expression(r"e^{x}")
        # Should be true exponential, not Symbol('e')**x
        x = list(expr.free_symbols)[0]
        coeffs, _ = taylor_coefficients(expr, 4)
        np.testing.assert_allclose(coeffs, [1.0, 1.0, 0.5, 1 / 6, 1 / 24], rtol=1e-12)

    def test_plain_sympy_fallback(self):
        expr = parse_expression("cos(x)")
        self.assertEqual(str(expr), "cos(x)")


class TestLatexToGaussians(unittest.TestCase):
    def test_sin_pipeline(self):
        approx = latex_to_gaussians(r"\sin(x)", N=5, epsilon=0.25)
        # sin Taylor: x - x^3/6 + x^5/120
        np.testing.assert_allclose(
            approx.coeffs,
            [0.0, 1.0, 0.0, -1 / 6, 0.0, 1 / 120],
            rtol=1e-12,
        )
        self.assertTrue(np.allclose(approx.w_even, 0.0, atol=1e-12))
        err = approx.error(remove_envelope=True)
        self.assertLess(err["max_rel"], 0.08)

    def test_exp_pipeline(self):
        approx = latex_to_gaussians(r"e^{x}", N=4, epsilon=0.2)
        np.testing.assert_allclose(
            approx.coeffs, [1.0, 1.0, 0.5, 1 / 6, 1 / 24], rtol=1e-12
        )
        err = approx.error(remove_envelope=True)
        self.assertLess(err["max_rel"], 0.05)
        # Should have both even and odd centers
        self.assertGreater(len(approx.centers), 2)

    def test_rational_geometric(self):
        approx = latex_to_gaussians(r"\frac{1}{1-x}", N=4, epsilon=0.2)
        np.testing.assert_allclose(approx.coeffs, np.ones(5), rtol=1e-12)
        err = approx.error(x=np.linspace(-1.0, 1.0, 201), remove_envelope=True)
        self.assertLess(err["max_rel"], 0.05)


if __name__ == "__main__":
    unittest.main()
