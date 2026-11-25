"""Unit tests for stability-weighted spatial colocalization and exact permutation variance."""
import itertools
import math
import unittest
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from spatial_membership_audit.permutation_bridge import observed_permutation_bridge
from spatial_membership_audit.stability_weighting import (
    StabilityWeightedResult,
    cell_stability_weights,
    exact_weighted_perm_pair,
    stability_weighted_colocalization,
)


def _adj_from_edges(n: int, edges: list[tuple[int, int]]):
    if not edges:
        return sparse.csr_matrix((n, n), dtype=float)
    rows, cols = zip(*(edges + [(v, u) for u, v in edges]))
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


class TestStabilityWeighting(unittest.TestCase):
    def test_binary_weights_match_permutation_bridge_exactly(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]
        adj = _adj_from_edges(6, edges)
        labels = np.array([0, 1, 0, 2, 1, 0])
        ref = observed_permutation_bridge(adj, labels, a=0, b=1)
        got = exact_weighted_perm_pair(adj, (labels == 0).astype(float), (labels == 1).astype(float))
        self.assertIsInstance(got, StabilityWeightedResult)
        self.assertAlmostEqual(got.observed_weighted, float(ref["observed"]), places=11)
        self.assertAlmostEqual(got.expected_weighted, ref["mean"], places=11)
        self.assertAlmostEqual(got.var_weighted, ref["variance"], places=11)
        self.assertAlmostEqual(got.r_weighted, ref["ratio"], places=11)
        self.assertAlmostEqual(got.z_weighted, ref["z"], places=11)
        self.assertAlmostEqual(got.effective_n_a, 3.0, places=11)
        self.assertAlmostEqual(got.effective_n_b, 2.0, places=11)
        self.assertAlmostEqual(got.mean_weight_a, 1.0, places=11)

    def test_continuous_weights_match_exact_n_factorial_enumeration(self):
        # 8-node irregular graph, continuous disjoint weights -> enumerate all 8! = 40,320 permutations
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7), (1, 5)]
        n = 8
        adj = _adj_from_edges(n, edges)
        wa = np.array([1.0, 0.75, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
        wb = np.array([0.0, 0.0, 0.0, 0.90, 0.45, 0.20, 0.0, 0.0])
        got = exact_weighted_perm_pair(adj, wa, wb)

        e_arr = np.array(edges, dtype=np.int32)
        e0, e1 = e_arr[:, 0], e_arr[:, 1]
        sum_o = 0.0
        sum_o2 = 0.0
        cnt = 0
        for perm in itertools.permutations(range(n)):
            idx = np.asarray(perm, dtype=np.int32)
            pwa = wa[idx]
            pwb = wb[idx]
            o_val = float((pwa[e0] * pwb[e1] + pwa[e1] * pwb[e0]).sum())
            sum_o += o_val
            sum_o2 += o_val * o_val
            cnt += 1
        exact_mean = sum_o / cnt
        exact_var = sum_o2 / cnt - exact_mean * exact_mean
        self.assertAlmostEqual(got.expected_weighted, exact_mean, places=10)
        self.assertAlmostEqual(got.var_weighted, exact_var, places=10)
        expected_r = got.observed_weighted / exact_mean
        expected_z = (got.observed_weighted - exact_mean) / math.sqrt(exact_var)
        self.assertAlmostEqual(got.r_weighted, expected_r, places=10)
        self.assertAlmostEqual(got.z_weighted, expected_z, places=10)

    def test_monte_carlo_null_calibration_on_geometric_graph(self):
        rng = np.random.default_rng(20261120)
        n = 200
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        edges = cKDTree(coords).query_pairs(r=14.0, output_type="ndarray")
        adj = _adj_from_edges(n, [(int(u), int(v)) for u, v in edges])

        base_labels = rng.choice([0, 1, 2], size=n, p=[0.35, 0.25, 0.40])
        draws = np.tile(base_labels, (20, 1))
        # Flip subset with heterogeneous rates
        flip_probs = np.where(np.arange(n) % 4 == 0, 0.45, 0.05)
        for bi in range(20):
            flips = rng.random(n) < flip_probs
            draws[bi, flips] = (base_labels[flips] + 1) % 3

        weights = cell_stability_weights(base_labels, draws, power=1.0)
        wa = np.where(base_labels == 0, weights, 0.0)
        wb = np.where(base_labels == 1, weights, 0.0)
        obs_res = stability_weighted_colocalization(adj, base_labels, draws, a=0, b=1, power=1.0)

        e0, e1 = edges[:, 0], edges[:, 1]
        z_null = np.empty(1000, dtype=np.float64)
        sd_ref = math.sqrt(obs_res.var_weighted)
        for k in range(1000):
            perm = rng.permutation(n)
            pwa, pwb = wa[perm], wb[perm]
            o_k = float((pwa[e0] * pwb[e1] + pwa[e1] * pwb[e0]).sum())
            z_null[k] = (o_k - obs_res.expected_weighted) / sd_ref

        self.assertLess(abs(float(z_null.mean())), 0.08)
        self.assertGreater(float(z_null.std(ddof=1)), 0.92)
        self.assertLess(float(z_null.std(ddof=1)), 1.08)
        fpr = float((np.abs(z_null) > 1.96).mean())
        self.assertGreater(fpr, 0.03)
        self.assertLess(fpr, 0.07)

    def test_cell_stability_weights_power_edge_cases_and_validation(self):
        base = np.array([0, 1, 2, 0])
        draws = np.array([
            [0, 1, 2, 1],
            [0, 0, 2, 1],
            [0, 1, 2, 0],
            [0, 1, 1, 0],
        ])
        w1 = cell_stability_weights(base, draws, power=1.0)
        w2 = cell_stability_weights(base, draws, power=2.0)
        np.testing.assert_allclose(w1, [1.0, 0.75, 0.75, 0.5], atol=1e-12)
        np.testing.assert_allclose(w2, [1.0, 0.5625, 0.5625, 0.25], atol=1e-12)

        # Degenerate zero-edge / missing-class / n=2 checks
        adj_empty = _adj_from_edges(4, [])
        res_empty = exact_weighted_perm_pair(adj_empty, [1, 0, 1, 0], [0, 1, 0, 0])
        self.assertTrue(math.isnan(res_empty.r_weighted))
        self.assertTrue(math.isnan(res_empty.z_weighted))

        adj_n2 = _adj_from_edges(2, [(0, 1)])
        res_n2 = exact_weighted_perm_pair(adj_n2, [0.8, 0.0], [0.0, 0.5])
        self.assertAlmostEqual(res_n2.r_weighted, 1.0, places=12)
        self.assertTrue(math.isnan(res_n2.z_weighted))

        adj4 = _adj_from_edges(4, [(0, 1), (1, 2), (2, 3)])
        for bad_call in [
            lambda: cell_stability_weights(base, draws, power=-0.5),
            lambda: cell_stability_weights([], draws),
            lambda: cell_stability_weights(base, draws[:, :2]),
            lambda: exact_weighted_perm_pair(adj4, [0.5, 0.0, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0]),  # overlapping
            lambda: exact_weighted_perm_pair(adj4, [-0.1, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
            lambda: exact_weighted_perm_pair(adj4, [np.nan, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
            lambda: stability_weighted_colocalization(adj4, base, draws, a=1, b=1),
        ]:
            with self.assertRaises(ValueError):
                bad_call()


if __name__ == "__main__":
    unittest.main()
