"""Integration tests for the v3.0 scverse (AnnData / Squidpy) standardization API."""
from __future__ import annotations

import inspect
import unittest
import numpy as np

from spatial_membership_audit import (
    DEFAULT_PROTOCOL_V3,
    __version__,
    get_default_protocol_v3,
)
from spatial_membership_audit.standardize import (
    StandardizedComparison,
    standardize_colocalization_anndata,
)

try:
    import anndata as ad
    import squidpy as sq
    HAS_SCVERSE = True
except Exception:
    HAS_SCVERSE = False


def _make_synthetic_adata(seed: int, depth_scale: float, n_cells: int = 120):
    rng = np.random.default_rng(seed)
    genes = ["EPCAM", "KRT8", "CD3E", "CD8A", "CD68", "LYZ", "ACTA2", "COL1A1"]
    # Three spatial bands so Epithelial/T_cells/Myeloid have structured non-trivial colocalization
    coords = np.column_stack([rng.uniform(0, 120, size=n_cells), rng.uniform(0, 120, size=n_cells)])
    true_cls = np.where(coords[:, 0] < 45, 0, np.where(coords[:, 0] < 85, 1, 2))
    lam = np.full((n_cells, len(genes)), 0.4 * depth_scale, dtype=float)
    lam[true_cls == 0, 0:2] = 12.0 * depth_scale
    lam[true_cls == 1, 2:4] = 12.0 * depth_scale
    lam[true_cls == 2, 4:6] = 12.0 * depth_scale
    counts = rng.poisson(lam).astype(np.int64)
    adata = ad.AnnData(X=counts)
    adata.obs_names = [f"c_{seed}_{i}" for i in range(n_cells)]
    adata.var_names = genes
    adata.obsm["spatial"] = coords
    adata.obs["region"] = "core_1"
    return adata


def _simple_lineage_classifier(counts, gene_names):
    x = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    epi = x[:, 0] + x[:, 1]
    tcell = x[:, 2] + x[:, 3]
    mye = x[:, 4] + x[:, 5]
    scores = np.column_stack([epi, tcell, mye])
    names = np.array(["Epithelial", "T_cells", "Myeloid"])
    return names[np.argmax(scores, axis=1)]


@unittest.skipUnless(HAS_SCVERSE, "anndata and squidpy required")
class TestStandardizeScverseAPI(unittest.TestCase):
    def setUp(self):
        self.ad_deep = _make_synthetic_adata(seed=101, depth_scale=2.0, n_cells=120)
        self.ad_shal = _make_synthetic_adata(seed=202, depth_scale=0.6, n_cells=120)

    def test_rarefaction_and_squidpy_format(self):
        res = standardize_colocalization_anndata(
            self.ad_deep,
            self.ad_shal,
            classifier_fn=_simple_lineage_classifier,
            mode="rarefaction",
            region_key="region",
            radius_um=25.0,
            n_draws=6,
            n_permutations=19,
            seed=42,
        )
        self.assertIsInstance(res, StandardizedComparison)
        self.assertEqual(res.mode, "rarefaction")
        self.assertLess(res.rates["q_a"], res.rates["q_b"])
        self.assertEqual(len(res.table), 6)  # 2 runs x 3 pairs
        self.assertTrue(np.all(np.isfinite(res.table["r_est"])))
        self.assertTrue(np.all(np.isfinite(res.table["permutation_p_z"])))

        sq_dict = res.to_squidpy_nhood_enrichment(run="A", region="core_1", stat="z_dua")
        self.assertEqual(sq_dict["zscore"].shape, (3, 3))
        self.assertEqual(sq_dict["count"].shape, (3, 3))
        np.testing.assert_allclose(sq_dict["zscore"], sq_dict["zscore"].T, equal_nan=True)

    def test_coverage_matching_mode(self):
        res = standardize_colocalization_anndata(
            self.ad_deep,
            self.ad_shal,
            classifier_fn=_simple_lineage_classifier,
            mode="coverage",
            region_key="region",
            radius_um=25.0,
            n_draws=6,
            n_permutations=19,
            split_half_grid=(1.0, 0.75, 0.50, 0.25),
            split_half_reps=2,
            seed=7,
        )
        self.assertEqual(res.mode, "coverage")
        self.assertIn("target_psi", res.rates)
        self.assertEqual(len(res.table), 6)

    def test_mitm_mode_and_guardrail(self):
        res = standardize_colocalization_anndata(
            self.ad_deep,
            self.ad_shal,
            classifier_fn=_simple_lineage_classifier,
            mode="mitm",
            region_key="region",
            radius_um=25.0,
            n_draws=6,
            n_permutations=19,
            extrap_method="prior_only",
            seed=11,
        )
        self.assertEqual(res.mode, "mitm")
        self.assertGreater(res.rates["target_depth"], res.rates["medpos_b"])
        self.assertLess(res.rates["target_depth"], res.rates["medpos_a"])
        self.assertEqual(len(res.table), 6)

    def test_stability_weighted_with_precomputed_squidpy_graph(self):
        ad_sq = self.ad_deep.copy()
        if hasattr(sq.gr, "spatial_neighbors_radius"):
            sq.gr.spatial_neighbors_radius(ad_sq, radius=25.0)
        else:
            sq.gr.spatial_neighbors(ad_sq, coord_type="generic", radius=25.0)
        res = standardize_colocalization_anndata(
            ad_sq,
            classifier_fn=_simple_lineage_classifier,
            mode="stability_weighted",
            region_key="region",
            n_draws=6,
            seed=19,
        )
        self.assertEqual(res.mode, "stability_weighted")
        self.assertEqual(len(res.table), 3)
        self.assertTrue(np.all(np.isfinite(res.table["r_est"])))

    def test_v3_frozen_default_protocol_and_stratified_probe(self):
        self.assertEqual(__version__, "3.0.0")
        p = DEFAULT_PROTOCOL_V3
        self.assertEqual(p["protocol_version"], "3.0.0")
        self.assertEqual(p["primary_cross_platform"]["mode"], "split_half")
        self.assertEqual(p["primary_cross_platform"]["pairing"], "both")
        self.assertAlmostEqual(p["primary_cross_platform"]["q0"], 0.75)
        self.assertEqual(p["primary_cross_platform"]["n_draws"], 20)
        self.assertEqual(p["primary_cross_platform"]["split_half_max_cells"], 10000)
        self.assertEqual(p["secondary_moderate_gap"]["mode"], "mitm")
        self.assertEqual(p["single_specimen_map"]["mode"], "stability_weighted")
        self.assertEqual(get_default_protocol_v3()["protocol_version"], "3.0.0")

        sig = inspect.signature(standardize_colocalization_anndata)
        self.assertEqual(sig.parameters["mode"].default, "split_half")
        self.assertEqual(sig.parameters["q0"].default, 0.75)
        self.assertEqual(sig.parameters["pairing"].default, "both")
        self.assertEqual(sig.parameters["n_draws"].default, 20)
        self.assertEqual(sig.parameters["split_half_max_cells"].default, 10000)

        res = standardize_colocalization_anndata(
            self.ad_deep,
            self.ad_shal,
            classifier_fn=_simple_lineage_classifier,
            region_key="region",
            n_draws=6,
            n_permutations=19,
            split_half_max_cells=60,
            split_half_reps=2,
            seed=123,
        )
        self.assertEqual(res.mode, "split_half")
        self.assertEqual(res.metadata["n_probe_cells_a"], 60)
        self.assertEqual(res.metadata["n_probe_cells_b"], 60)
        self.assertEqual(len(res.table), 6)


if __name__ == "__main__":
    unittest.main()
