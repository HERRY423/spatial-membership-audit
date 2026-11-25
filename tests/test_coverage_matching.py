"""Unit tests for conditional instability metrics and split-half coverage matching."""
import math
import unittest
import numpy as np

from spatial_membership_audit.coverage_matching import (
    ConditionalInstability,
    CoverageMatchResult,
    SplitHalfCurve,
    build_split_half_curve,
    conditional_instability_metrics,
    match_split_half_coverage,
    split_half_disagreement,
)


class TestCoverageMatching(unittest.TestCase):
    def test_conditional_flip_invariant_to_unresolved_zero_inflation(self):
        # Diagnostic 0 regression check: appending UNRESOLVED / zero-count cells
        # lowers unconditional flip rate but leaves conditional metrics exact.
        base_core = np.array(["Epi", "T", "Epi", "Mye"])
        draws_core = np.array([
            ["Epi", "Mye", "Epi", "Mye"],
            ["T",   "T",   "Epi", "Mye"],
            ["Epi", "T",   "T",   "Mye"],
        ])
        m_pure = conditional_instability_metrics(
            base_core, draws_core, unresolved_label="UNRESOLVED"
        )
        self.assertIsInstance(m_pure, ConditionalInstability)
        self.assertEqual(m_pure.unresolved_frac, 0.0)

        # Pad with 6 UNRESOLVED zero cells that never flip across draws
        base_pad = np.concatenate([base_core, np.repeat("UNRESOLVED", 6)])
        draws_pad = np.hstack([draws_core, np.repeat("UNRESOLVED", 18).reshape(3, 6)])
        m_pad = conditional_instability_metrics(
            base_pad, draws_pad, unresolved_label="UNRESOLVED"
        )
        self.assertAlmostEqual(m_pad.unresolved_frac, 0.6)
        self.assertEqual(m_pad.n_classifiable, 4)
        self.assertAlmostEqual(m_pad.cond_flip_rate, m_pure.cond_flip_rate, places=12)
        self.assertAlmostEqual(m_pad.cond_pair_disagreement, m_pure.cond_pair_disagreement, places=12)
        self.assertLess(m_pad.uncond_flip_rate, m_pure.uncond_flip_rate)

        # Explicit classifiable_mask works identically even with integer labels
        mask = np.array([True, True, True, True] + [False] * 6)
        m_mask = conditional_instability_metrics(base_pad, draws_pad, classifiable_mask=mask)
        self.assertAlmostEqual(m_mask.cond_flip_rate, m_pure.cond_flip_rate, places=12)

    def test_split_half_disagreement_and_all_unresolved(self):
        a = np.array([["A", "B", "UNRESOLVED", "A"], ["A", "A", "UNRESOLVED", "B"]])
        b = np.array([["A", "A", "UNRESOLVED", "A"], ["B", "A", "UNRESOLVED", "B"]])
        psi, ufrac, n_cls = split_half_disagreement(a, b, unresolved_label="UNRESOLVED")
        self.assertEqual(n_cls, 3)
        self.assertAlmostEqual(ufrac, 0.25)
        # Replicate 0: [False, True, False], Replicate 1: [True, False, False] -> 2/6
        self.assertAlmostEqual(psi, 2.0 / 6.0, places=12)

        psi0, ufrac0, n_cls0 = split_half_disagreement(
            np.repeat("UNRESOLVED", 5),
            np.repeat("UNRESOLVED", 5),
            unresolved_label="UNRESOLVED",
        )
        self.assertEqual(n_cls0, 0)
        self.assertEqual(ufrac0, 1.0)
        self.assertTrue(math.isnan(psi0))

    def test_pava_and_exact_depth_ratio_recovery(self):
        # Synthetic Poisson scaling law: instability psi(d) = 0.8 / sqrt(d)
        # Run A has depth 10, Run B has depth 20 -> equal split-half instability occurs at q_B = 0.5 * q_A.
        q_grid = np.array([0.10, 0.20, 0.35, 0.50, 0.75, 0.85, 1.00])
        psi_a = np.clip(0.8 / np.sqrt(10.0 * q_grid), 0.0, 1.0)
        psi_b = np.clip(0.8 / np.sqrt(20.0 * q_grid), 0.0, 1.0)

        # Verify PAVA repairs a noisy bump while preserving sorted monotone curves
        psi_bumpy = psi_a.copy()
        psi_bumpy[4], psi_bumpy[5] = psi_bumpy[5], psi_bumpy[4]
        curve_bumpy = build_split_half_curve(q_grid[::-1], psi_bumpy[::-1], enforce_monotone=True)
        self.assertTrue(np.all(np.diff(curve_bumpy.psi_vals) <= 1e-12))

        ca = build_split_half_curve(q_grid, psi_a, unresolved_frac=0.20)
        cb = build_split_half_curve(q_grid, psi_b, unresolved_frac=0.02)
        self.assertIsInstance(ca, SplitHalfCurve)

        for pairing, ref_q in (("onesided", 1.0), ("both", 0.75)):
            res_ab = match_split_half_coverage(ca, cb, q0=0.75, pairing=pairing)
            res_ba = match_split_half_coverage(cb, ca, q0=0.75, pairing=pairing)
            self.assertIsInstance(res_ab, CoverageMatchResult)
            self.assertEqual(res_ab.noisier_side, "A")
            self.assertEqual(res_ba.noisier_side, "B")
            self.assertAlmostEqual(res_ab.q_a, ref_q, places=12)
            # Log-q linear interpolation approximates 0.5 * ref_q cleanly
            self.assertAlmostEqual(res_ab.q_b, 0.5 * ref_q, delta=0.02)
            self.assertAlmostEqual(res_ab.q_a, res_ba.q_b, places=12)
            self.assertAlmostEqual(res_ab.q_b, res_ba.q_a, places=12)
            self.assertFalse(res_ab.clamped)

    def test_equal_depth_unequal_snr_and_clamping(self):
        q_grid = np.array([0.20, 0.50, 0.75, 1.00])
        ca_high_noise = build_split_half_curve(q_grid, [0.55, 0.45, 0.35, 0.30])
        cb_low_noise = build_split_half_curve(q_grid, [0.25, 0.15, 0.10, 0.08])
        res = match_split_half_coverage(ca_high_noise, cb_low_noise, q0=0.75, pairing="both", min_q=0.10)
        self.assertEqual(res.noisier_side, "A")
        self.assertTrue(res.clamped)
        self.assertAlmostEqual(res.q_b, 0.20, places=12)

        with self.assertRaises(ValueError):
            build_split_half_curve([0.5], [0.2])
        with self.assertRaises(ValueError):
            build_split_half_curve([0.5, 0.5], [0.2, 0.3])
        with self.assertRaises(ValueError):
            match_split_half_coverage(ca_high_noise, cb_low_noise, pairing="invalid")


if __name__ == "__main__":
    unittest.main()
