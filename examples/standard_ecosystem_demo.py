"""End-to-end demonstration of standard spatial ecosystem adapters and biologist action guide.

Demonstrates:
  1. Arbitrary user-defined declarative marker rules (MarkerRuleSpec / dict)
     and frozen cell-wise classifiers (FrozenLinearCentroidClassifier) beyond
     the hardcoded 11-marker and 4-gene rules;
  2. Out-of-the-box audit on AnnData (`audit_anndata`), SpatialData (`audit_spatialdata`),
     Squidpy (`audit_squidpy_nhood`), and Giotto (`prepare_giotto_audit_bundle`);
  3. Biologist's Operational Decision Guide (`triage_audit_table` and
     `summarize_operational_actions`) on the saved 1,728-row breast retention
     curve (`p_curve_records.csv`).
"""
from __future__ import annotations

import json
import pathlib
import sys

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Point
import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel

# Ensure local package is importable when run directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spatial_membership_audit.action_guide import (
    summarize_operational_actions,
    triage_audit_record,
    triage_audit_table,
)
from spatial_membership_audit.custom_rules import (
    FrozenLinearCentroidClassifier,
    evaluate_marker_rule_probabilities,
    frozen_classifier_probabilities,
    marker_rule_from_dict,
)
from spatial_membership_audit.standard_ecosystem import (
    audit_anndata,
    audit_spatialdata,
    audit_squidpy_nhood,
    prepare_giotto_audit_bundle,
)


def build_demo_anndata_and_spatialdata() -> tuple[ad.AnnData, sd.SpatialData]:
    """Create a synthetic 8-gene tumor microenvironment AnnData and SpatialData."""
    rng = np.random.default_rng(20261031)
    genes = ["EPCAM", "KRT8", "CD3D", "CD3E", "CD8A", "CD68", "CD163", "MZB1"]
    n_cells = 48
    counts = np.zeros((n_cells, len(genes)), dtype=np.int64)

    # 18 Tumor epithelial cells (EPCAM, KRT8)
    counts[:18, 0] = rng.integers(2, 7, size=18)
    counts[:18, 1] = rng.integers(1, 5, size=18)
    # 15 Cytotoxic T cells (CD3D + CD3E >= 2 and CD8A >= 1) — some near threshold
    counts[18:33, 2] = rng.integers(1, 4, size=15)
    counts[18:33, 3] = rng.integers(1, 3, size=15)
    counts[18:33, 4] = rng.integers(1, 3, size=15)
    # 15 Macrophages (CD68 >= 1 and CD163 >= 1)
    counts[33:, 5] = rng.integers(1, 5, size=15)
    counts[33:, 6] = rng.integers(1, 4, size=15)
    # Add plasma veto (MZB1) to 2 cells so they can enter under thinning
    counts[17, 7] = 1
    counts[32, 7] = 1

    xs, ys = np.meshgrid(np.arange(8) * 12.0, np.arange(6) * 12.0)
    xy = np.column_stack([xs.ravel(), ys.ravel()])

    obs = pd.DataFrame(
        {
            "fov": ["FOV_1"] * 24 + ["FOV_2"] * 24,
            "qc_pass": [True] * n_cells,
            "region": ["cells_shapes"] * n_cells,
            "instance_id": np.arange(n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.obsm["spatial"] = xy

    gdf = gpd.GeoDataFrame(
        {"radius": np.full(n_cells, 5.0)},
        geometry=[Point(float(x), float(y)) for x, y in xy],
        index=np.arange(n_cells),
    )
    sdata = sd.SpatialData(
        shapes={"cells_shapes": ShapesModel.parse(gdf)},
        tables={
            "table": TableModel.parse(
                adata.copy(),
                region="cells_shapes",
                region_key="region",
                instance_key="instance_id",
            )
        },
    )
    return adata, sdata


def main() -> None:
    adata, sdata = build_demo_anndata_and_spatialdata()

    # 1. Custom user-defined 8-gene TME rule (with sum gate and veto)
    custom_tme_rule = marker_rule_from_dict({
        "rule_id": "custom_tme_cd8_mac_v1",
        "lineages": {
            "TUMOR_EPI": {
                "op": "any",
                "predicates": [
                    {"op": "threshold", "gene": "EPCAM", "min_count": 2},
                    {"op": "threshold", "gene": "KRT8", "min_count": 2},
                ],
            },
            "CD8_T": {
                "op": "all",
                "predicates": [
                    {"op": "sum", "genes": ["CD3D", "CD3E"], "min_sum": 2, "min_each": 1},
                    {"op": "threshold", "gene": "CD8A", "min_count": 1},
                ],
            },
            "TAM_MAC": {
                "op": "all",
                "predicates": [
                    {"op": "threshold", "gene": "CD68", "min_count": 1},
                    {"op": "threshold", "gene": "CD163", "min_count": 1},
                ],
            },
        },
        "vetoes": {
            "PLASMA_VETO": {"op": "threshold", "gene": "MZB1", "min_count": 1},
        },
    })

    print("=== 1. AnnData Multi-Point Retention Audit (Custom 8-Gene Rule) ===")
    df_adata = audit_anndata(
        adata,
        p_grid=[1.0, 0.75, 0.5, 0.25],
        rule=custom_tme_rule,
        region_key="fov",
        graph_method="radius",
        radius_um=18.0,
        cv_mode="direct",
    )
    print(
        df_adata[["region", "label_a", "label_b", "p", "p_finite", "cv", "regime", "action_code"]]
        .head(12)
        .to_string(index=False)
    )

    print("\n=== 2. SpatialData Audit (Shapes + Table) ===")
    df_sdata = audit_spatialdata(
        sdata,
        table_key="table",
        shapes_key="cells_shapes",
        p=0.5,
        rule=custom_tme_rule,
        region_key="fov",
        graph_method="radius",
        radius_um=18.0,
        cv_mode="auto",
    )
    print(
        df_sdata[["region", "label_a", "label_b", "p_finite", "cv", "engine", "regime"]]
        .to_string(index=False)
    )

    print("\n=== 3. Squidpy Side-by-Side Neighborhood Enrichment + Audit ===")
    df_sq = audit_squidpy_nhood(
        adata,
        p=0.5,
        rule=custom_tme_rule,
        region_key="fov",
        graph_method="radius",
        radius_um=18.0,
        n_perms=100,
        cv_mode="direct",
    )
    print(
        df_sq[["region", "label_a", "label_b", "squidpy_zscore", "p_finite", "cv", "squidpy_audit_flag"]]
        .to_string(index=False)
    )

    print("\n=== 4. Giotto Bundle Preparation ===")
    giotto_bundle = prepare_giotto_audit_bundle(
        adata,
        p=0.5,
        rule=custom_tme_rule,
        region_key="fov",
        graph_method="radius",
        radius_um=18.0,
        cv_mode="delta",
    )
    print(
        giotto_bundle["audit_table"][
            ["region", "label_a", "label_b", "p_finite", "cv", "giotto_stub_null_mean", "regime"]
        ].to_string(index=False)
    )

    # 5. Empirical summary on saved 1,728-row breast retention curve
    curve_csv = (
        ROOT.parents[1]
        / "manuscript"
        / "bioinformatics_v64_2026-10-29"
        / "evidence"
        / "curve"
        / "p_curve_records.csv"
    )
    if curve_csv.exists():
        print("\n=== 5. Empirical Triage Breakdown on Saved 1,728-Row Breast Retention Curve ===")
        curve_df = pd.read_csv(curve_csv)
        triaged = pd.DataFrame(
            triage_audit_table(
                curve_df.to_dict(orient="records"),
                group_keys=("region", "pair"),
                p_key="p",
            )
        )
        summary = summarize_operational_actions(triaged, p_key="p")
        print(json.dumps(summary["regime_counts_by_p"], indent=2))
        print(
            f"Total records: {summary['total_records']}, "
            f"Conditioning traps flagged: {summary['conditioning_trap_records']}, "
            f"Non-monotonic records flagged: {summary['non_monotonic_trajectory_records']}"
        )


if __name__ == "__main__":
    main()
