"""Unit tests for unsupervised auto-q depth selection and DUA ensemble inference."""
import math
import unittest
import numpy as np
from scipy import sparse

from spatial_membership_audit.depth_matching import AutoQSelection, marginal_depth_ratio, select_auto_q
from spatial_membership_audit.dua_inference import (
    DUAResult,
    conformal_prediction_band,
    decompose_dua_uncertainty,
    exact_perm_z_from_draws,
    joint_perm_pairs_from_draws,
    summarize_dua_draws,
)
from spatial_membership_audit.permutation_bridge import observed_permutation_bridge


class TestAutoQSelection(unittest.TestCase):
    def test_balanced_vs_mismatched_routing(self):
        rng = np.random.default_rng(7)
        a_bal = rng.integers(50, 150, size=500)
        b_bal = rng.integers(50, 145, size=500)
        sel_bal = select_auto_q(a_bal, b_bal, balance_threshold=0.85, sym_q=0.75, pairing="both")
        self.assertIsInstance(sel_bal, AutoQSelection)
        self.assertEqual(sel_bal.mode, "BALANCED_SYM")
        self.assertEqual(sel_bal.q_a, 0.75)
        self.assertEqual(sel_bal.q_b, 0.75)

        sel_bal_1s = select_auto_q(a_bal, b_bal, balance_threshold=0.85, pairing="onesided")
        self.assertEqual(sel_bal_1s.mode, "BALANCED_ONESIDED")
        self.assertEqual(max(sel_bal_1s.q_a, sel_bal_1s.q_b), 1.0)

        a_deep = np.array([100.0, 200.0, 150.0, 0.0])
        b_shal = np.array([20.0, 40.0, 30.0, 0.0])
        sel_mis = select_auto_q(a_deep, b_shal, balance_threshold=0.85, sym_q=0.75, pairing="both")
        self.assertEqual(sel_mis.mode, "AUTO_Q_BOTH")
        self.assertEqual(sel_mis.deeper_side, "A")
        self.assertAlmostEqual(sel_mis.ratio_medpos, 0.2, places=12)
        self.assertAlmostEqual(sel_mis.q_a, 0.75 * 0.2, places=12)
        self.assertAlmostEqual(sel_mis.q_b, 0.75, places=12)

        sel_1s = select_auto_q(a_deep, b_shal, balance_threshold=0.85, sym_q=0.75, pairing="onesided")
        self.assertEqual(sel_1s.mode, "AUTO_Q_ONESIDED")
        self.assertAlmostEqual(sel_1s.q_a, 0.2, places=12)
        self.assertAlmostEqual(sel_1s.q_b, 1.0, places=12)

    def test_zero_inflation_medpos_vs_mean_and_all_zero_fallback(self):
        # 80% zeros on A, positive median still matches B, whereas mean differs strongly
        a = np.array([0.0] * 80 + [20.0] * 20)
        b = np.array([20.0] * 100)
        r_med, _, sa, sb = marginal_depth_ratio(a, b, method="medpos")
        r_mn, side_mn, _, _ = marginal_depth_ratio(a, b, method="mean")
        self.assertAlmostEqual(r_med, 1.0, places=12)
        self.assertEqual(sa, 20.0)
        self.assertEqual(sb, 20.0)
        self.assertAlmostEqual(r_mn, 0.2, places=12)
        self.assertEqual(side_mn, "B")

        # All-zero cohort fallback to mean
        z_a = np.zeros(10)
        z_b = np.array([5.0, 5.0])
        r_z, side_z, s0, s1 = marginal_depth_ratio(z_a, z_b, method="medpos")
        self.assertEqual(r_z, 0.0)
        self.assertEqual(side_z, "B")
        self.assertEqual(s0, 0.0)
        self.assertEqual(s1, 5.0)
        r_both0, side_b0, _, _ = marginal_depth_ratio(np.zeros(4), np.zeros(6), method="medpos")
        self.assertEqual(r_both0, 1.0)
        self.assertEqual(side_b0, "TIED")

    def test_exchange_symmetry_and_validation(self):
        a = np.array([0.0, 12.0, 36.0, 48.0])
        b = np.array([4.0, 8.0, 12.0])
        for pairing in ("both", "onesided"):
            for method in ("medpos", "mean"):
                ab = select_auto_q(a, b, pairing=pairing, method=method)
                ba = select_auto_q(b, a, pairing=pairing, method=method)
                self.assertAlmostEqual(ab.q_a, ba.q_b, places=12)
                self.assertAlmostEqual(ab.q_b, ba.q_a, places=12)
                self.assertAlmostEqual(ab.ratio_medpos, ba.ratio_medpos, places=12)

        for bad_call in [
            lambda: marginal_depth_ratio([], [1.0]),
            lambda: marginal_depth_ratio([1.0], [-1.0]),
            lambda: marginal_depth_ratio([np.nan], [1.0]),
            lambda: marginal_depth_ratio([[1.0]], [1.0]),
            lambda: marginal_depth_ratio([1.0], [1.0], method="bad"),
            lambda: select_auto_q([1.0], [1.0], balance_threshold=0.0),
            lambda: select_auto_q([1.0], [1.0], sym_q=1.5),
            lambda: select_auto_q([1.0], [1.0], pairing="bad"),
        ]:
            with self.assertRaises(ValueError):
                bad_call()

    def test_default_threshold_free_common_depth_and_continuity(self):
        # Default balance_threshold=1.0 produces continuous COMMON_DEPTH routing across 0.85
        a_ref = np.array([1000.0, 1000.0])
        b_below = np.array([849.0, 849.0])
        b_above = np.array([851.0, 851.0])
        b_tied = np.array([1000.0, 1000.0])

        sel_below = select_auto_q(a_ref, b_below)
        sel_above = select_auto_q(a_ref, b_above)
        sel_tied = select_auto_q(a_ref, b_tied)
        self.assertEqual(sel_below.balance_threshold, 1.0)
        self.assertEqual(sel_below.mode, "COMMON_DEPTH_BOTH")
        self.assertEqual(sel_above.mode, "COMMON_DEPTH_BOTH")
        self.assertEqual(sel_tied.mode, "COMMON_DEPTH_BOTH")
        self.assertAlmostEqual(sel_below.q_a, 0.75 * 0.849, places=12)
        self.assertAlmostEqual(sel_above.q_a, 0.75 * 0.851, places=12)
        self.assertLess(abs(sel_above.q_a - sel_below.q_a), 0.002)
        self.assertAlmostEqual(sel_tied.q_a, 0.75, places=12)
        self.assertAlmostEqual(sel_tied.q_b, 0.75, places=12)

        sel_1s = select_auto_q(a_ref, b_above, pairing="onesided")
        self.assertEqual(sel_1s.mode, "COMMON_DEPTH_ONESIDED")
        self.assertAlmostEqual(sel_1s.q_a, 0.851, places=12)
        self.assertAlmostEqual(sel_1s.q_b, 1.0, places=12)

    def test_cosmx_pr1_post_qc_exact_ratio(self):
        # Exact post-QC CosMx PR_1 classifier-input medpos: STJ=116, WCM=110 -> r=110/116=0.94827586
        stj = np.array([0.0, 116.0, 116.0])
        wcm = np.array([0.0, 110.0, 110.0])
        sel = select_auto_q(stj, wcm)
        expected_r = 110.0 / 116.0
        self.assertEqual(sel.mode, "COMMON_DEPTH_BOTH")
        self.assertEqual(sel.deeper_side, "A")
        self.assertAlmostEqual(sel.ratio_medpos, expected_r, places=12)
        self.assertAlmostEqual(sel.q_a, 0.75 * expected_r, places=12)
        self.assertAlmostEqual(sel.q_b, 0.75, places=12)


class TestDUAInference(unittest.TestCase):
    @staticmethod
    def graph(n, edges):
        rows, cols = zip(*(edges + [(v, u) for u, v in edges]))
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    def test_zeros_contribute_to_mean_and_mc_variance(self):
        r = np.array([0., 2., 2., np.nan])
        z = np.array([-1., 1., 1., np.nan])
        result = summarize_dua_draws(r, z, availability_threshold=.75)
        self.assertAlmostEqual(result.r_mean, 4./3.)
        self.assertAlmostEqual(result.mc_se_logr, .5)
        self.assertAlmostEqual(result.det_var_logr, .75)
        self.assertEqual(result.n_defined, 3)
        self.assertEqual(result.p_plus, .75)
        self.assertTrue(result.available)
        self.assertTrue(math.isnan(result.z_dua))
        self.assertTrue(math.isnan(result.ci95_width_logr))

    def test_draw_availability_and_zero_mean(self):
        r = np.array([[0., 1., np.nan], [0., 1., np.nan], [0., np.nan, np.nan]])
        z = np.full_like(r,np.nan)
        result = summarize_dua_draws(r,z)
        np.testing.assert_array_equal(result.available, [True,False,False])
        np.testing.assert_array_equal(result.available_z, [False,False,False])
        self.assertEqual(result.r_mean[0], 0.)
        self.assertTrue(math.isnan(result.mc_se_logr[0]))
        self.assertEqual(result.mc_se_logr[1], 0.)
        with self.assertRaises(ValueError):
            result.r_mean[1] = 7.

    def test_exact_graph_against_independent_permutation_bridge(self):
        g = self.graph(5,[(0,1),(1,2),(2,3),(0,4),(1,4)])
        labels = np.array([[0,0,1,1,2],[2,0,1,0,1],[0,2,1,1,0]])
        def stats(permuted):
            pairs = [observed_permutation_bridge(g,row,a=0,b=1) for row in permuted]
            return [x["ratio"] for x in pairs], [x["z"] for x in pairs]
        r,z = stats(labels)
        k = 39
        rng = np.random.default_rng(16)
        bank = np.array([np.mean(stats(labels[:,rng.permutation(5)])[1]) for _ in range(k)])
        res = exact_perm_z_from_draws(g,labels,n_permutations=k,seed=16)
        self.assertIsInstance(res,DUAResult)
        self.assertAlmostEqual(res.r_mean,np.mean(r))
        self.assertAlmostEqual(res.z_unadj,np.mean(z))
        self.assertAlmostEqual(res.null_sd_z,bank.std(ddof=1))
        self.assertAlmostEqual(res.z_dua,np.mean(z)/bank.std(ddof=1))
        self.assertAlmostEqual(res.permutation_p_z,(1+np.sum(abs(bank)>=abs(np.mean(z))-1e-14))/(k+1))
        self.assertAlmostEqual(res.mc_se_logr,np.std(r,ddof=1)/(np.mean(r)*np.sqrt(3)))

    def test_exhaustive_joint_null_and_duplicate_draws(self):
        from itertools import permutations
        g = self.graph(4,[(0,1),(1,2),(2,3)])
        labels = np.array([[0,1,0,2],[0,1,2,0]])
        bank = []
        for perm in permutations(range(4)):
            bank.append(np.mean([observed_permutation_bridge(g,row[list(perm)],a=0,b=1)["z"] for row in labels]))
        # Enumerating the common permutation gives exactly zero expected mean z.
        self.assertAlmostEqual(np.mean(bank),0.)
        self.assertLess(np.var(bank),1.)
        zz = [observed_permutation_bridge(g,row,a=0,b=1)["z"] for row in labels]
        out=summarize_dua_draws([1.,1.],zz,null_mean_z=bank)
        self.assertAlmostEqual(out.z_dua,np.mean(zz)/np.std(bank,ddof=1))
        dup = exact_perm_z_from_draws(g,np.repeat(labels[:1],3,axis=0),seed=5)
        base = exact_perm_z_from_draws(g,np.repeat(labels[:1],2,axis=0),seed=5)
        self.assertAlmostEqual(dup.null_sd_z,base.null_sd_z)
        self.assertAlmostEqual(dup.permutation_p_z,base.permutation_p_z)

    def test_fixed_graph_null_degenerate_does_not_drop_defined_ratio(self):
        g = self.graph(3,[(0,1),(1,2),(0,2)])
        out=exact_perm_z_from_draws(g,[[0,1,2],[2,1,0]])
        self.assertEqual(out.r_mean,1.)
        self.assertEqual(out.p_plus,1.)
        self.assertFalse(out.available_z)
        self.assertTrue(math.isnan(out.permutation_p_z))
        self.assertEqual(out.spatial_var_logr,0.)

    def test_explicit_sparse_zeros_are_not_edges(self):
        g = self.graph(4, [(0,1),(1,2),(2,3)])
        with_zero = g.tolil()
        with_zero.rows[0].append(3)
        with_zero.data[0].append(0.)
        labels = [[0,0,1,1],[0,2,1,1]]
        clean = exact_perm_z_from_draws(g,labels,seed=17)
        result = exact_perm_z_from_draws(with_zero.tocsr(),labels,seed=17)
        self.assertEqual(clean.r_mean,result.r_mean)
        self.assertEqual(clean.permutation_p_z,result.permutation_p_z)

    def test_separate_spatial_proxy_and_independent_repeat_prediction(self):
        r = np.array([[0.,1.], [2.,1.], [1.,1.]])
        z = np.zeros_like(r)
        errors = np.column_stack([np.arange(1,21),np.ones(20)*.25])
        res = summarize_dua_draws(r,z,ratio_null_var=np.ones_like(r)*.2,repeat_log_errors=errors)
        np.testing.assert_allclose(res.spatial_var_logr,[.2,.2])
        np.testing.assert_allclose(res.ci95_logr_hi,[20.,.25])
        np.testing.assert_allclose(res.ci95_width_logr,[40.,.5])
        self.assertTrue(np.isnan(res.se_total_logr).all())
        self.assertTrue(np.isnan(res.z_dua_logr).all())
        self.assertTrue(np.isinf(summarize_dua_draws(r,z,repeat_log_errors=errors[:18]).ci95_width_logr).all())
        logproxy = summarize_dua_draws(r,z,vlog_draws=np.ones_like(r))
        self.assertTrue(math.isnan(logproxy.spatial_var_logr[0]))
        self.assertEqual(logproxy.spatial_var_logr[1],1.)
        with self.assertRaises(ValueError):
            summarize_dua_draws([-.1,1],[0,0])
        with self.assertRaises(ValueError):
            summarize_dua_draws([1,2],[0,1],null_mean_z=[[1],[2]])

    def test_joint_perm_pairs_and_null_corr_and_conformal_band(self):
        g = self.graph(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6), (1, 4)])
        labels = np.array([
            [0, 0, 1, 1, 2, 2, 0],
            [0, 2, 1, 1, 2, 0, 1],
            [1, 0, 1, 2, 2, 0, 0],
            [0, 0, 1, 1, 2, 2, 1],
        ])
        pairs = [(0, 1), (0, 2), (1, 2)]
        multi = joint_perm_pairs_from_draws(g, labels, pairs, n_permutations=49, seed=19)
        self.assertEqual(set(multi.keys()), set(pairs))
        for a, b in pairs:
            ref = exact_perm_z_from_draws(g, labels, a=a, b=b, n_permutations=49, seed=19)
            got = multi[(a, b)]
            self.assertAlmostEqual(got.r_mean, ref.r_mean, places=12)
            self.assertAlmostEqual(got.z_unadj, ref.z_unadj, places=12)
            self.assertAlmostEqual(got.null_sd_z, ref.null_sd_z, places=12)
            self.assertAlmostEqual(got.z_dua, ref.z_dua, places=12)
            self.assertAlmostEqual(got.permutation_p_z, ref.permutation_p_z, places=12)
            self.assertEqual(got.n_defined, 4)
            self.assertEqual(got.n_defined_z, 4)
            self.assertGreaterEqual(got.null_corr_z, -1.0 / 3.0)
            self.assertLessEqual(got.null_corr_z, 1.0)

        # Duplicate draws imply near-unit null_corr_z
        dup_labels = np.repeat(labels[:1], 3, axis=0)
        dup_res = joint_perm_pairs_from_draws(g, dup_labels, [(0, 1)], n_permutations=199, seed=11)[(0, 1)]
        self.assertGreater(dup_res.null_corr_z, 0.70)

        # Standalone conformal prediction band for point estimates (e.g., extrapolated log-R)
        cal_errs = np.linspace(-0.4, 0.4, 20)
        lo_s, hi_s, w_s = conformal_prediction_band(0.5, cal_errs, alpha=0.05)
        self.assertAlmostEqual(lo_s, 0.1, places=12)
        self.assertAlmostEqual(hi_s, 0.9, places=12)
        self.assertAlmostEqual(w_s, 0.8, places=12)

        pts = np.array([0.5, np.nan, -0.2])
        errs_2d = np.column_stack([cal_errs, cal_errs, cal_errs * 0.5])
        lo_v, hi_v, w_v = conformal_prediction_band(pts, errs_2d, alpha=0.05)
        self.assertAlmostEqual(w_v[0], 0.8, places=12)
        self.assertTrue(math.isnan(w_v[1]))
        self.assertAlmostEqual(w_v[2], 0.4, places=12)
        _, _, w_inf = conformal_prediction_band(0.5, cal_errs[:18], alpha=0.05)
        self.assertTrue(math.isinf(w_inf))

        with self.assertRaises(ValueError):
            joint_perm_pairs_from_draws(g, labels, [(0, 0)])
        with self.assertRaises(ValueError):
            joint_perm_pairs_from_draws(g, labels, [(0, 1), (0, 1)])

    def test_pooled_conformal_decompose_and_missing_draw_parity(self):
        import spatial_membership_audit as sma
        self.assertTrue(hasattr(sma, "decompose_dua_uncertainty"))
        self.assertTrue(hasattr(sma, "match_split_half_coverage"))
        self.assertTrue(hasattr(sma, "extrapolate_ladder"))

        # Pooled 1D calibration residuals broadcast cleanly over 1D point vector
        cal_1d = np.linspace(-0.25, 0.25, 20)
        lo_p, hi_p, w_p = conformal_prediction_band(np.array([0.0, 1.0, np.nan]), cal_1d, alpha=0.05)
        np.testing.assert_allclose(w_p[:2], [0.5, 0.5])
        self.assertTrue(math.isnan(w_p[2]))

        # Missing draw (class 1 absent in draw 3): exact vs multi-pair parity without warnings
        g = self.graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
        labels_miss = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2, 0],
            [0, 0, 2, 2, 2, 0],  # class 1 absent -> R=NaN, z=NaN
            [0, 1, 0, 1, 2, 2],
        ])
        ref = exact_perm_z_from_draws(g, labels_miss, a=0, b=1, availability_threshold=0.75, n_permutations=49, seed=7)
        got = joint_perm_pairs_from_draws(g, labels_miss, [(0, 1)], availability_threshold=0.75, n_permutations=49, seed=7)[(0, 1)]
        self.assertEqual(ref.n_defined, 3)
        self.assertEqual(ref.n_defined_z, 3)
        self.assertAlmostEqual(got.z_dua, ref.z_dua, places=12)
        self.assertAlmostEqual(got.permutation_p_z, ref.permutation_p_z, places=12)

        # Decompose uncertainty separates all 3 statistical regimes cleanly
        dec = decompose_dua_uncertainty(
            np.array([0.0, 2.0, 2.0]),
            np.array([-1.0, 1.0, 1.0]),
            null_mean_z=np.array([-0.5, 0.5, -0.2, 0.2]),
            repeat_log_errors=cal_1d,
            alpha=0.05,
        )
        self.assertAlmostEqual(dec["regime_1_fixed_input_mc"]["r_mean"], 4.0 / 3.0)
        self.assertAlmostEqual(dec["regime_1_fixed_input_mc"]["mc_se_logr"], 0.5)
        self.assertTrue(np.isfinite(dec["regime_2_spatial_permutation_null"]["z_dua"]))
        self.assertAlmostEqual(dec["regime_3_conformal_repeat_prediction"]["ci_width_logr"], 0.5)


if __name__ == "__main__":
    unittest.main()
