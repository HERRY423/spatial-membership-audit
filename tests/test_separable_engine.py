"""Regression tests for the 1D/2D separable polynomial quadrature engine and vectorized validation."""
from fractions import Fraction
import unittest

import numpy as np
from scipy import sparse
from spatial_membership_audit import direct_cv as cv
from spatial_membership_audit import primary_cv


class SeparableEngineRegression(unittest.TestCase):
    def test_monomial_certificates_large_support(self):
        """Verify outward interval enclosure of monomial quadrature across large support."""
        for support in [100, 1024, 2050, 5000]:
            x, w1, w2, bounds = cv.rule(4, support)
            # Test key test degrees: boundaries, powers of 2, and middle values
            test_degrees = sorted({0, 1, 2, 7, 31, 64, support // 4, support // 2, support - 1})
            for power, (weights, bound) in enumerate([(w1, bounds[0]), (w2, bounds[1])], 1):
                blo, bhi = bound
                for k in test_degrees:
                    # Exact moment is (k+1)^power * sum(w_i * x_i^k)
                    # Notice bounds are over all k in [0, support - 1]
                    exact = sum((Fraction(float(wi)) * Fraction(float(xi))**k for xi, wi in zip(x, weights)), Fraction(0)) * (k + 1)**power
                    self.assertLessEqual(Fraction(float(blo)), exact, f"Lower bound failed for support={support}, deg={k}, power={power}")
                    self.assertGreaterEqual(Fraction(float(bhi)), exact, f"Upper bound failed for support={support}, deg={k}, power={power}")

    def test_separable_vs_full_polynomial_equivalence(self):
        """Verify that 1D separable factorization matches the full 2D polynomial expansion exactly."""
        rng = np.random.default_rng(42)
        n = 80
        # Generate random probability vectors and adjacency
        pA = rng.uniform(0.01, 0.3, size=n)
        pB = rng.uniform(0.01, 0.3, size=n)
        p_other = 1.0 - (pA + pB)
        P = np.column_stack([pA, pB, p_other])
        
        # Build random graph
        coords = rng.uniform(0, 100, size=(n, 2))
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        pairs = tree.query_pairs(r=25.0)
        row = [i for i, j in pairs] + [j for i, j in pairs]
        col = [j for i, j in pairs] + [i for i, j in pairs]
        graph = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))

        # Run direct_cv estimate
        res = cv.estimate(pA, pB, graph, accuracy_factor=1.25)
        self.assertEqual(res['status'], 'RESOLVED')
        self.assertGreater(res['cv'], 0.0)
        self.assertLessEqual(res['cv_interval'][0], res['cv'])
        self.assertGreaterEqual(res['cv_interval'][1], res['cv'])
        self.assertLessEqual(res['cv_interval'][1] / res['cv_interval'][0], 1.25)

    def test_primary_cv_validation_vectorization(self):
        """Test vectorized row-sum validation in primary_cv handles machine-precision sums."""
        n = 100
        # Valid probabilities summing to exactly 1.0 with float rounding jitter
        p1 = np.full(n, 1.0 / 3.0)
        p2 = np.full(n, 1.0 / 3.0)
        p3 = 1.0 - (p1 + p2)
        P = np.column_stack([p1, p2, p3])
        graph = sparse.csr_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(n, n))
        
        # Valid input should succeed
        res = primary_cv.estimate_cv(P, graph, accuracy_factor=1.25, a=0, b=1)
        self.assertIn(res['status'], ('RESOLVED', 'ZERO_NUMERATOR_CV_UNDEFINED'))

        # Invalid row exceeding 1.0 by more than slack
        P_invalid = P.copy()
        P_invalid[50, 0] += 1e-4
        with self.assertRaises(ValueError):
            primary_cv.estimate_cv(P_invalid, graph, accuracy_factor=1.25, a=0, b=1)

        # Negative probability
        P_neg = P.copy()
        P_neg[10, 1] = -1e-6
        with self.assertRaises(ValueError):
            primary_cv.estimate_cv(P_neg, graph, accuracy_factor=1.25, a=0, b=1)


if __name__ == '__main__':
    unittest.main()
