"""Unit tests for within-run Rarefaction-Extrapolation (Poisson-SIMEX) and guardrails."""
import math
import unittest
import numpy as np

from spatial_membership_audit.extrapolation import (
    ExtrapolationResult,
    ThinningLadder,
    build_extrapolation_guardrail,
    extrapolate_ladder,
    noise_coordinate,
    select_mitm_target_depth,
    summarize_thinning_ladder,
    target_noise_coordinate,
)


class TestExtrapolation(unittest.TestCase):
    def test_noise_coordinates_and_mitm_symmetry(self):
        self.assertAlmostEqual(noise_coordinate(0.25, 16.0, "inv_sqrtd"), 0.5)
        self.assertAlmostEqual(noise_coordinate(0.25, 16.0, "inv_d"), 0.25)
        self.assertAlmostEqual(noise_coordinate(0.25, 16.0, "log_inv_d"), math.log1p(0.25))
        np.testing.assert_allclose(
            noise_coordinate([1.0, 0.25], 16.0, "inv_sqrtd"),
            [0.25, 0.5],
        )
        self.assertEqual(target_noise_coordinate(np.inf, "inv_sqrtd"), 0.0)
        self.assertAlmostEqual(target_noise_coordinate(25.0, "inv_sqrtd"), 0.2)

        for mode in ("geometric", "half_deep", "high_frac"):
            d_ab = select_mitm_target_depth(8.0, 32.0, mode=mode)
            d_ba = select_mitm_target_depth(32.0, 8.0, mode=mode)
            self.assertAlmostEqual(d_ab, d_ba, places=12)
        self.assertAlmostEqual(select_mitm_target_depth(8.0, 32.0, mode="geometric"), 16.0)
        self.assertAlmostEqual(select_mitm_target_depth(8.0, 32.0, mode="half_deep"), 16.0)
        self.assertAlmostEqual(select_mitm_target_depth(8.0, 32.0, mode="high_frac", high_frac=0.75), 24.0)
        # Small depth gap where high_frac * deep < shallow returns shallow
        self.assertAlmostEqual(select_mitm_target_depth(28.0, 32.0, mode="half_deep"), 28.0)

    def test_summarize_ladder_valid_zeros_and_availability(self):
        q_vals = [1.0, 0.75, 0.50]
        r0 = np.array([2.0, 0.0, 1.5])
        z0 = np.array([2.0, -1.0, 1.0])
        r1 = np.array([[1.6, 0.0, 1.0], [2.0, 0.0, np.nan], [1.8, 0.0, 1.2], [1.8, 0.0, 1.1]])
        z1 = np.array([[1.5, -0.5, 0.5], [1.8, -0.4, np.nan], [1.6, -0.6, 0.7], [1.7, -0.5, 0.6]])
        r2 = np.array([[1.4, 0.0, 0.9], [1.6, 0.0, 0.8], [1.5, 0.0, 0.7], [1.5, 0.0, 0.6]])
        z2 = np.array([[1.1, -0.2, 0.3], [1.3, -0.1, 0.2], [1.2, -0.2, 0.1], [1.2, -0.1, 0.1]])

        ladder = summarize_thinning_ladder([r0, r1, r2], [z0, z1, z2], q_vals, availability_threshold=0.90)
        self.assertIsInstance(ladder, ThinningLadder)
        self.assertAlmostEqual(ladder.r_mean[1, 0], 1.8)
        self.assertEqual(ladder.r_mean[1, 1], 0.0)
        self.assertTrue(ladder.valid[1, 0])
        self.assertFalse(ladder.valid[1, 1])  # zero R excluded from log-regression
        self.assertFalse(ladder.valid[1, 2])  # p_plus = 0.75 < 0.90
        self.assertGreater(ladder.v_mc[0, 0], 0.0)  # native proxy populated from thinned banks
        with self.assertRaises(ValueError):
            ladder.r_mean[0, 0] = 9.0

    def test_exact_linear_in_inv_sqrtd_recovery_and_guardrail(self):
        # Construct exact noise-attenuated log-R linear in 1/sqrt(q * s_k)
        s_k = 10.0
        d_star = 40.0
        q_vals = np.array([1.0, 0.85, 0.75, 0.65, 0.50])
        xi_vals = noise_coordinate(q_vals, s_k, "inv_sqrtd")
        xi_star = target_noise_coordinate(d_star, "inv_sqrtd")

        n_per_pair = 8
        pair_ids = np.repeat(["A--B", "C--D"], n_per_pair)
        U = len(pair_ids)
        rng = np.random.default_rng(123)
        y_star = rng.uniform(-0.8, 1.2, size=U)
        # Slope shares linear prior on y_star (common attenuation trend per pair)
        slopes = np.where(pair_ids == "A--B", -0.6 - 0.5 * y_star, -0.4 - 0.4 * y_star)
        # Re-parameterize as linear in xi: y(xi) = y_star + b * (xi - xi_star)
        # Notice under exact linear law y0 = y_star + b*(xi0-xi_star), b depends linearly on y0 too.
        true_r_star = np.exp(y_star)

        B = 8
        r_banks = []
        z_banks = []
        for idx_m, qv in enumerate(q_vals):
            y_m = y_star + slopes * (xi_vals[idx_m] - xi_star)
            r_m = np.exp(y_m)
            if idx_m == 0:
                r_banks.append(r_m)
                z_banks.append(np.ones(U))
            else:
                # Duplicate exact draws plus balanced zero-mean jitter
                jitter = np.linspace(-1e-6, 1e-6, B)[:, None]
                r_banks.append(r_m[None, :] * np.exp(jitter))
                z_banks.append(np.ones((B, U)))

        ladder = summarize_thinning_ladder(r_banks, z_banks, q_vals)
        depths = np.full(U, 8.0)
        depths[0] = 1.0  # triggers low-depth guardrail on unit 0
        flips = np.full(U, 0.15)
        flips[8] = 0.60  # triggers flip guardrail on unit 8
        gmask = build_extrapolation_guardrail(depths, flips, min_median_depth=2.0, max_cond_flip_rate=0.45)
        fallback = np.full(U, 1.25)

        for method in ("unit_linear", "hier_eb", "prior_only", "shrunk_step"):
            res = extrapolate_ladder(
                ladder,
                s_k,
                d_star,
                pair_ids,
                method=method,
                scale="inv_sqrtd",
                guardrail_mask=gmask,
                fallback_r=fallback,
            )
            self.assertIsInstance(res, ExtrapolationResult)
            self.assertTrue(res.guardrail_triggered[0])
            self.assertTrue(res.guardrail_triggered[8])
            self.assertEqual(res.r_pred[0], 1.25)
            self.assertEqual(res.r_pred[8], 1.25)
            self.assertEqual(res.extrap_se_logr[0], 0.0)
            ok = ~res.guardrail_triggered
            if method == "shrunk_step":
                np.testing.assert_allclose(res.r_pred[ok], true_r_star[ok], rtol=2e-2)
            else:
                np.testing.assert_allclose(res.r_pred[ok], true_r_star[ok], rtol=5e-3)

    def test_validation_errors(self):
        r0 = np.array([1.0, 2.0])
        z0 = np.array([1.0, 1.0])
        r1 = np.ones((4, 2))
        z1 = np.ones((4, 2))
        with self.assertRaises(ValueError):
            summarize_thinning_ladder([r0, r1], [z0, z1], [0.9, 0.5])  # q[0] != 1.0
        with self.assertRaises(ValueError):
            summarize_thinning_ladder([r0, np.ones((1, 2))], [z0, np.ones((1, 2))], [1.0, 0.5])  # B < 2
        with self.assertRaises(ValueError):
            noise_coordinate(0.5, 0.0)
        with self.assertRaises(ValueError):
            target_noise_coordinate(-1.0)
        with self.assertRaises(ValueError):
            select_mitm_target_depth(10.0, 20.0, mode="invalid")
        with self.assertRaises(ValueError):
            build_extrapolation_guardrail([-1.0, 5.0])


if __name__ == "__main__":
    unittest.main()
