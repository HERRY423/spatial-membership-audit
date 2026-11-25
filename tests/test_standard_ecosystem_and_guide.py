"""Unit tests for custom_rules, action_guide, and standard_ecosystem modules."""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Point
import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel

from spatial_membership_audit.analytic import marker_probabilities
from spatial_membership_audit.definitions import candidate_labels
from spatial_membership_audit.rules import stromal_myeloid_probabilities
from spatial_membership_audit.custom_rules import (
    COMMON125_11MARKER_SPEC,
    EPCAM_STROMAL_4GENE_SPEC,
    GENERIC_3LINEAGE_6MARKER_SPEC,
    STROMAL_MYELOID_4GENE_SPEC,
    AllOf,
    AnyOf,
    CandidateClassSpec,
    FrozenLinearCentroidClassifier,
    GeneSumThreshold,
    GeneThreshold,
    MarkerRuleSpec,
    evaluate_marker_rule_labels,
    evaluate_marker_rule_probabilities,
    frozen_classifier_probabilities,
    marker_rule_from_dict,
)
from spatial_membership_audit.action_guide import (
    ACTION_REGIME_CATALOG,
    ActionThresholds,
    detect_retention_non_monotonicity,
    summarize_operational_actions,
    triage_audit_record,
    triage_audit_table,
)
from spatial_membership_audit.standard_ecosystem import (
    audit_anndata,
    audit_spatialdata,
    audit_squidpy_nhood,
    extract_anndata_from_spatialdata,
    prepare_giotto_audit_bundle,
)


class TestCustomRules(unittest.TestCase):
    """Verify exact parity with hardcoded rules and arbitrary user rule flexibility."""

    def test_exact_parity_with_hardcoded_11marker_and_4gene_rules(self):
        rng = np.random.default_rng(20261031)
        genes_11 = list(COMMON125_11MARKER_SPEC.required_genes)
        counts_11 = rng.poisson(lam=1.8, size=(120, len(genes_11))).astype(np.int64)
        eligible = rng.random(120) > 0.15

        # 1. Baseline deterministic label parity at p=1.0 across all 4 candidate classes
        ref_labels = candidate_labels(counts_11, genes_11, eligible)
        new_labels = evaluate_marker_rule_labels(
            counts_11, genes_11, eligible, COMMON125_11MARKER_SPEC
        )
        for cname in COMMON125_11MARKER_SPEC.classes:
            np.testing.assert_array_equal(ref_labels == cname, new_labels == cname)
        np.testing.assert_array_equal(
            ref_labels == "LOW_DETECTION_BASELINE",
            new_labels == "LOW_DETECTION_BASELINE",
        )

        # 2. Exact analytical thinning probability parity across p
        for p_val in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
            ref_p = marker_probabilities(counts_11, genes_11, eligible, p_val)
            new_p = evaluate_marker_rule_probabilities(
                counts_11, genes_11, eligible, p_val, COMMON125_11MARKER_SPEC
            )
            np.testing.assert_allclose(ref_p, new_p, atol=1e-15, rtol=1e-15)

        # 3. Exact parity with 4-gene stromal/myeloid rule and epithelial/stromal rule
        genes_4 = ["COL1A1", "FBN1", "CSF1R", "TYROBP", "EPCAM", "KRT7", "PDGFRA", "COL5A2"]
        counts_4 = rng.poisson(lam=1.5, size=(100, len(genes_4))).astype(np.int64)
        elig_4 = rng.random(100) > 0.1
        for p_val in (1.0, 0.75, 0.5, 0.25):
            ref_sm = stromal_myeloid_probabilities(counts_4[:, :4], elig_4, p_val)
            new_sm = evaluate_marker_rule_probabilities(
                counts_4, genes_4, elig_4, p_val, STROMAL_MYELOID_4GENE_SPEC
            )
            np.testing.assert_allclose(ref_sm, new_sm, atol=1e-15, rtol=1e-15)

            ref_es = stromal_myeloid_probabilities(counts_4[:, 4:], elig_4, p_val)
            new_es = evaluate_marker_rule_probabilities(
                counts_4, genes_4, elig_4, p_val, EPCAM_STROMAL_4GENE_SPEC
            )
            np.testing.assert_allclose(ref_es, new_es, atol=1e-15, rtol=1e-15)

    def test_custom_user_dict_rule_and_mutual_exclusivity_check(self):
        custom_dict = {
            "rule_id": "neuro_glia_custom_v1",
            "lineages": {
                "GLUT_NEURON": {
                    "op": "all",
                    "predicates": [
                        {"op": "threshold", "gene": "SLC17A7", "min_count": 2},
                        {"op": "any", "predicates": [
                            {"op": "threshold", "gene": "SNAP25", "min_count": 1},
                            {"op": "threshold", "gene": "SYT1", "min_count": 1},
                        ]},
                    ],
                },
                "GABA_NEURON": {
                    "op": "sum",
                    "genes": ["GAD1", "GAD2"],
                    "min_sum": 2,
                    "min_each": 1,
                },
                "ASTROCYTE": {"op": "threshold", "gene": "AQP4", "min_count": 1},
            },
            "vetoes": {
                "DOUBLET_VETO": {
                    "op": "all",
                    "predicates": [
                        {"op": "threshold", "gene": "PTPRC", "min_count": 1},
                        {"op": "threshold", "gene": "PECAM1", "min_count": 1},
                    ],
                }
            },
        }
        spec = marker_rule_from_dict(custom_dict)
        self.assertEqual(spec.classes, ("GLUT_NEURON", "GABA_NEURON", "ASTROCYTE"))
        genes = ["SLC17A7", "SNAP25", "SYT1", "GAD1", "GAD2", "AQP4", "PTPRC", "PECAM1"]
        counts = np.array([
            [3, 2, 0, 0, 0, 0, 0, 0],  # Pure GLUT_NEURON
            [0, 0, 0, 2, 1, 0, 0, 0],  # Pure GABA_NEURON
            [0, 0, 0, 0, 0, 4, 0, 0],  # Pure ASTROCYTE
            [3, 2, 0, 0, 0, 0, 2, 2],  # Vetoed by DOUBLET_VETO
        ], dtype=np.int64)
        elig = np.ones(4, dtype=bool)
        p1 = evaluate_marker_rule_probabilities(counts, genes, elig, 1.0, spec)
        np.testing.assert_allclose(
            p1,
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]),
        )
        p_half = evaluate_marker_rule_probabilities(counts, genes, elig, 0.5, spec)
        self.assertTrue(np.all((p_half >= 0.0) & (p_half <= 1.0)))
        self.assertTrue(np.all(p_half.sum(axis=1) <= 1.0 + 1e-12))
        # Notice cell 3 has positive GLUT probability at p=0.5 because DOUBLET_VETO can drop out!
        self.assertGreater(p_half[3, 0], 0.0)

        # Verify non-mutually-exclusive candidate definitions raise ValueError
        with self.assertRaises(ValueError):
            MarkerRuleSpec(
                rule_id="bad_overlapping_rule",
                gates={
                    "G1": GeneThreshold("A", 1),
                    "G2": GeneThreshold("B", 1),
                },
                candidates=[
                    CandidateClassSpec("C1", positive_gate="G1"),
                    CandidateClassSpec("C2", positive_gate="G2"),  # Missing forbidden_gates=("G1",)
                ],
            )

    def test_frozen_classifier_probabilities_and_order_invariance(self):
        genes = ["EPCAM", "KRT8", "CD3D", "CD3E", "CD68", "CD163"]
        weights = np.array([
            [1.2, -0.5, -0.5],
            [1.0, -0.4, -0.4],
            [-0.5, 1.3, -0.4],
            [-0.4, 1.1, -0.4],
            [-0.5, -0.4, 1.2],
            [-0.4, -0.4, 1.1],
        ])
        clf = FrozenLinearCentroidClassifier(
            genes=genes,
            classes=("EPITHELIAL", "T_CELL", "MACROPHAGE"),
            weights=weights,
            min_margin=0.2,
        )
        counts = np.array([
            [8, 6, 0, 0, 0, 0],
            [0, 0, 7, 5, 0, 0],
            [0, 0, 0, 0, 9, 4],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int64)
        elig = np.array([True, True, True, True])
        res = frozen_classifier_probabilities(
            counts, genes, elig, 0.5,
            classifier_fn=clf,
            classes=("EPITHELIAL", "T_CELL", "MACROPHAGE"),
            n_draws=256,
            seed=42,
        )
        probs = res["probabilities"]
        self.assertEqual(probs.shape, (4, 3))
        self.assertGreater(probs[0, 0], 0.9)
        self.assertGreater(probs[1, 1], 0.9)
        self.assertGreater(probs[2, 2], 0.9)
        np.testing.assert_allclose(probs[3], 0.0)

        # Verify a classifier that depends on cross-cell batch statistics is rejected
        def bad_cohort_dependent_clf(x_batch, g_list):
            # Uses cohort mean across cells (violates single-cell independence)
            cohort_mean = x_batch[:, 0].mean()
            return np.where(x_batch[:, 0] > cohort_mean, "EPITHELIAL", "T_CELL")

        with self.assertRaises(ValueError):
            frozen_classifier_probabilities(
                counts, genes, elig, 0.5,
                classifier_fn=bad_cohort_dependent_clf,
                classes=("EPITHELIAL", "T_CELL"),
                n_draws=64,
                seed=42,
            )


class TestActionGuide(unittest.TestCase):
    """Verify the 11-regime operational triage engine and real study archetypes."""

    def test_all_regimes_and_real_archetypes(self):
        self.assertEqual(len(ACTION_REGIME_CATALOG), 11)

        # Archetype 1: g27_14 conditioning trap (P+ = 0.0110, CV = 0.4533)
        trap = triage_audit_record({
            "P_plus": 0.0110085686261421,
            "cv": 0.4532724417159052,
            "cv_status": "RESOLVED",
            "baseline_state": "FINITE_POSITIVE",
            "baseline_R": 6.469,
        })
        self.assertEqual(trap["regime"], "TIER_5A_CONDITIONING_TRAP")
        self.assertTrue(trap["conditioning_trap"])
        self.assertTrue(trap["high_ratio_fragility_warning"])

        # Archetype 2: g0_0 severe fragility (P+ = 0.0885, CV = 1.6746, R_0 = 17.84)
        fragile = triage_audit_record({
            "P_plus": 0.0884592964207394,
            "cv": 1.6745874247350082,
            "cv_status": "RESOLVED",
            "baseline_state": "FINITE_POSITIVE",
            "baseline_R": 17.84312459651388,
        })
        self.assertEqual(fragile["regime"], "TIER_5B_SEVERE_FRAGILITY")
        self.assertTrue(fragile["high_ratio_fragility_warning"])

        # Archetype 3: idx90 vs idx204 (same baseline C=1, Na=14, Nb=5, divergent P+ at p=0.5)
        idx90 = triage_audit_record({
            "P_plus": 0.793927,
            "cv": 0.3712,
            "cv_status": "RESOLVED",
            "baseline_state": "FINITE_POSITIVE",
            "baseline_R": 1.15,
        })
        idx204 = triage_audit_record({
            "P_plus": 0.996488,
            "cv": 0.3712,
            "cv_status": "RESOLVED",
            "baseline_state": "FINITE_POSITIVE",
            "baseline_R": 1.15,
        })
        self.assertEqual(idx90["regime"], "TIER_4_AVAILABILITY_WARNING")
        self.assertEqual(idx204["regime"], "TIER_2_MODERATE_DISPERSION")

        # Archetype 4: g13_18 non-monotonic veto entry
        curve_g13_18 = [
            {"region": "g13_18", "pair": "B--CD8", "p": 1.0, "P_plus": 0.0, "cv": None, "cv_status": "SKIPPED_P1", "baseline_state": "STRUCTURAL_NA"},
            {"region": "g13_18", "pair": "B--CD8", "p": 0.9, "P_plus": 0.146028, "cv": 0.5667, "cv_status": "RESOLVED", "baseline_state": "STRUCTURAL_NA"},
            {"region": "g13_18", "pair": "B--CD8", "p": 0.5, "P_plus": 0.335754, "cv": 0.6591, "cv_status": "RESOLVED", "baseline_state": "STRUCTURAL_NA"},
            {"region": "g13_18", "pair": "B--CD8", "p": 0.25, "P_plus": 0.072578, "cv": 0.7521, "cv_status": "RESOLVED", "baseline_state": "STRUCTURAL_NA"},
        ]
        nm_map = detect_retention_non_monotonicity(
            curve_g13_18, group_keys=("region", "pair"), p_key="p"
        )
        nm_info = nm_map[("g13_18", "B--CD8")]
        self.assertTrue(nm_info["is_non_monotonic"])
        self.assertGreater(nm_info["max_adjacent_increase"], 0.14)
        triaged_curve = triage_audit_table(curve_g13_18, group_keys=("region", "pair"), p_key="p")
        for row in triaged_curve:
            if row["p"] < 1.0:
                self.assertEqual(row["regime"], "NON_MONOTONIC_VETO_ENTRY")
            self.assertTrue(row["is_non_monotonic_trajectory"])

    def test_saved_1728_breast_curve_triage(self):
        curve_csv = pathlib.Path(
            r"C:\Reproducution\manuscript\bioinformatics_v64_2026-10-29\evidence\curve\p_curve_records.csv"
        )
        if not curve_csv.exists():
            self.skipTest("Saved p_curve_records.csv not present")
        df = pd.read_csv(curve_csv)
        triaged = triage_audit_table(
            df.to_dict(orient="records"),
            group_keys=("region", "pair"),
            p_key="p",
        )
        self.assertEqual(len(triaged), 1728)
        summary = summarize_operational_actions(triaged, p_key="p")
        self.assertEqual(summary["total_records"], 1728)
        self.assertGreater(summary["conditioning_trap_records"], 0)
        self.assertGreater(summary["non_monotonic_trajectory_records"], 0)


class TestStandardEcosystem(unittest.TestCase):
    """Verify seamless integration with AnnData, SpatialData, Squidpy, and Giotto."""

    def setUp(self):
        rng = np.random.default_rng(1729)
        self.genes = list(GENERIC_3LINEAGE_6MARKER_SPEC.required_genes)
        n_cells = 36
        counts = np.zeros((n_cells, len(self.genes)), dtype=np.int64)
        # First 12 cells: EPITHELIAL (EPCAM, KRT8)
        counts[:12, 0] = rng.integers(2, 6, size=12)
        counts[:12, 1] = rng.integers(1, 5, size=12)
        # Next 12 cells: T_CELL (CD3D, CD3E)
        counts[12:24, 2] = rng.integers(2, 6, size=12)
        counts[12:24, 3] = rng.integers(1, 5, size=12)
        # Last 12 cells: MACROPHAGE (CD68, CD163)
        counts[24:, 4] = rng.integers(2, 6, size=12)
        counts[24:, 5] = rng.integers(2, 6, size=12)

        # Create 6x6 grid coordinates (spacing 10 um)
        xs, ys = np.meshgrid(np.arange(6) * 10.0, np.arange(6) * 10.0)
        xy = np.column_stack([xs.ravel(), ys.ravel()])

        obs = pd.DataFrame(
            {
                "region": ["FOV_A"] * 18 + ["FOV_B"] * 18,
                "eligible": [True] * n_cells,
            },
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=self.genes)
        self.adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
        self.adata.obsm["spatial"] = xy

    def test_audit_anndata_modes_and_immutability(self):
        # 1. Default inplace=False does not mutate adata.uns
        df_direct = audit_anndata(
            self.adata,
            p_grid=[1.0, 0.5],
            rule="generic_3lineage_6marker",
            region_key="region",
            graph_method="radius",
            radius_um=15.0,
            cv_mode="direct",
            inplace=False,
        )
        self.assertNotIn("spatial_membership_audit", self.adata.uns)
        self.assertEqual(len(df_direct), 2 * 2 * 3)  # 2 p values * 2 regions * 3 pairs
        self.assertIn("regime", df_direct.columns)
        self.assertIn("action_zh", df_direct.columns)
        self.assertIn("action_en", df_direct.columns)

        # 2. inplace=True stores table, summary, and provenance in adata.uns
        df_auto = audit_anndata(
            self.adata,
            p=0.5,
            rule="generic_3lineage_6marker",
            region_key="region",
            graph_method="radius",
            radius_um=15.0,
            cv_mode="auto",
            inplace=True,
        )
        self.assertIn("spatial_membership_audit", self.adata.uns)
        self.assertEqual(len(df_auto), 6)

    def test_audit_spatialdata_with_shapes_and_table(self):
        # Build a SpatialData object where coordinates live in sdata.shapes['cell_circles']
        adata_no_obsm = self.adata.copy()
        xy = adata_no_obsm.obsm.pop("spatial")
        adata_no_obsm.obs["region"] = "cell_circles"
        adata_no_obsm.obs["instance_id"] = np.arange(adata_no_obsm.n_obs)

        gdf = gpd.GeoDataFrame(
            {"radius": np.full(adata_no_obsm.n_obs, 4.0)},
            geometry=[Point(float(x), float(y)) for x, y in xy],
            index=np.arange(adata_no_obsm.n_obs),
        )
        shapes_el = ShapesModel.parse(gdf)
        table_el = TableModel.parse(
            adata_no_obsm,
            region="cell_circles",
            region_key="region",
            instance_key="instance_id",
        )
        sdata = sd.SpatialData(shapes={"cell_circles": shapes_el}, tables={"table": table_el})

        extracted = extract_anndata_from_spatialdata(sdata, table_key="table", shapes_key="cell_circles")
        self.assertIn("spatial", extracted.obsm)
        np.testing.assert_allclose(extracted.obsm["spatial"], xy)

        sdata_df = audit_spatialdata(
            sdata,
            table_key="table",
            shapes_key="cell_circles",
            p=0.5,
            rule="generic_3lineage_6marker",
            graph_method="radius",
            radius_um=15.0,
            cv_mode="delta",
            inplace=True,
        )
        self.assertEqual(len(sdata_df), 3)
        self.assertIn("spatial_membership_audit", sdata.tables["table"].uns)

    def test_audit_squidpy_nhood_and_giotto_bundle(self):
        sq_df = audit_squidpy_nhood(
            self.adata,
            p=0.5,
            rule="generic_3lineage_6marker",
            region_key="region",
            graph_method="radius",
            radius_um=15.0,
            n_perms=50,
            cv_mode="direct",
            inplace=True,
        )
        self.assertEqual(len(sq_df), 6)
        self.assertIn("squidpy_zscore", sq_df.columns)
        self.assertIn("squidpy_audit_flag", sq_df.columns)
        self.assertIn("spatial_membership_audit_squidpy", self.adata.uns)

        # Test Giotto bundle preparation and export
        local_tmp_root = pathlib.Path(r"C:\Reproducution\software\spatial_membership_audit\tests")
        with tempfile.TemporaryDirectory(dir=local_tmp_root) as tmpdir:
            out_path = pathlib.Path(tmpdir) / "giotto_export"
            bundle = prepare_giotto_audit_bundle(
                self.adata,
                p=0.5,
                rule="generic_3lineage_6marker",
                region_key="region",
                graph_method="radius",
                radius_um=15.0,
                cv_mode="delta",
                out_dir=out_path,
            )
            self.assertIn("giotto_stub_null_mean", bundle["audit_table"].columns)
            self.assertTrue((out_path / "giotto_audit_triage.csv").exists())
            self.assertTrue((out_path / "giotto_spatial_network.csv").exists())
            self.assertTrue((out_path / "giotto_cell_metadata.csv").exists())


if __name__ == "__main__":
    unittest.main()
