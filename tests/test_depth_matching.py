"""Finite-support distribution references, boundary and invariance checks."""
import unittest
import numpy as np
from scipy.stats import binom, betabinom
from spatial_membership_audit.depth_matching import bidirectional_retention, thin_histogram_summary


class TestDepthMatching(unittest.TestCase):
    def test_only_deeper_side_and_common_target(self):
        a, b = np.array([8., 2., 4.]), np.array([2., 8., 4.])
        fit = bidirectional_retention(a, b)
        np.testing.assert_array_equal(fit.retention_a, [.25, 1., 1.])
        np.testing.assert_array_equal(fit.retention_b, [1., .25, 1.])
        np.testing.assert_allclose(a*fit.retention_a, b*fit.retention_b)
        np.testing.assert_array_equal(fit.status, ["THIN_A", "THIN_B", "TIED"])

    def test_exchange_symmetry(self):
        a, b = [0., 1., 2., 7.], [1., 0., 3., 7.]
        ab, ba = bidirectional_retention(a, b), bidirectional_retention(b, a)
        np.testing.assert_array_equal(ab.retention_a, ba.retention_b)
        np.testing.assert_array_equal(ab.retention_b, ba.retention_a)
        np.testing.assert_array_equal(ab.common_mean, ba.common_mean)

    def test_zero_states(self):
        fit = bidirectional_retention([0., 2., 0.], [0., 0., 3.])
        np.testing.assert_array_equal(fit.retention_a, [1., 0., 1.])
        np.testing.assert_array_equal(fit.retention_b, [1., 1., 0.])
        np.testing.assert_array_equal(fit.status, ["BOTH_ZERO", "COMMON_TARGET_ZERO", "COMMON_TARGET_ZERO"])

    def test_invalid_means(self):
        for a, b in [([], []), ([1], [1, 2]), ([np.nan], [1]), ([-1], [1]), ([[1]], [[1]]), ([np.inf], [1])]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                bidirectional_retention(a, b)

    def test_rates_cannot_be_mutated(self):
        with self.assertRaises(ValueError):
            bidirectional_retention([2], [1]).retention_a[0] = 1.

    def test_full_probability_mass_reference(self):
        h = np.array([7, 5, 0, 2, 3, 1, 4])
        for p in [.02, .37, .97]:
            for rho in [0., .01, .05, .1, .2, .9]:
                with self.subTest(p=p, rho=rho):
                    pmf = np.zeros(len(h))
                    for n, freq in enumerate(h):
                        k = np.arange(n+1)
                        if rho == 0:
                            conditional = binom.pmf(k, n, p)
                        else:
                            kappa = (1-rho)/rho
                            conditional = betabinom.pmf(k, n, p*kappa, (1-p)*kappa)
                        pmf[:n+1] += freq/h.sum()*conditional
                    x = np.arange(len(h))
                    mean = float(pmf @ x)
                    ref = [mean, float(pmf @ ((x-mean)**2)), pmf[0]]
                    got = thin_histogram_summary(h, p, rho=rho)
                    np.testing.assert_allclose([got['mean'], got['variance'], got['zero_fraction']], ref, rtol=2e-11, atol=2e-13)

    def test_degenerate_retention(self):
        h = [2, 3, 1]
        for rho in [0, .01, .2]:
            zero, one = thin_histogram_summary(h, 0, rho=rho), thin_histogram_summary(h, 1, rho=rho)
            self.assertEqual((zero['mean'], zero['variance'], zero['zero_fraction']), (0., 0., 1.))
            self.assertAlmostEqual(one['mean'], 5/6)
            self.assertAlmostEqual(one['variance'], 7/6-(5/6)**2)
            self.assertEqual(one['zero_fraction'], 1/3)

    def test_all_zero_counts(self):
        for p in [0, .2, 1]:
            s = thin_histogram_summary([9], p, rho=.2)
            self.assertEqual((s['mean'], s['variance'], s['zero_fraction']), (0., 0., 1.))

    def test_overdispersion_preserves_mean_and_increases_variance(self):
        low = thin_histogram_summary([2, 3, 4, 5], .4)
        high = thin_histogram_summary([2, 3, 4, 5], .4, rho=.1)
        self.assertEqual(high['mean'], low['mean'])
        self.assertGreater(high['variance'], low['variance'])
        self.assertGreater(high['zero_fraction'], low['zero_fraction'])

    def test_binomial_limit(self):
        a = thin_histogram_summary([3, 2, 1, 9], .45)
        b = thin_histogram_summary([3, 2, 1, 9], .45, rho=1e-12)
        for field in ['mean', 'variance', 'zero_fraction']:
            self.assertAlmostEqual(a[field], b[field], places=10)

    def test_invalid_histograms_or_parameters(self):
        for h, p, rho in [([], .5, 0), ([0, 0], .5, 0), ([-1, 2], .5, 0), ([1.5, 2], .5, 0), ([1, np.nan], .5, 0), ([1], -1, 0), ([1], 2, 0), ([1], .5, 1), ([1], .5, -.1), ([1], .5, np.nan)]:
            with self.subTest(h=h, p=p, rho=rho), self.assertRaises(ValueError):
                thin_histogram_summary(h, p, rho=rho)


if __name__ == '__main__':
    unittest.main()
