"""Self-contained runnable example of spatial-membership-audit v2.0.0 scverse API."""
from __future__ import annotations

import anndata as ad
import numpy as np
from spatial_membership_audit import standardize_colocalization_anndata


def make_demo_adata(seed: int, depth_scale: float, n_cells: int = 180) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    genes = ["EPCAM", "KRT8", "CD3E", "CD8A", "CD68", "LYZ"]
    coords = np.column_stack([rng.uniform(0, 140, n_cells), rng.uniform(0, 140, n_cells)])
    zone = np.where(coords[:, 0] < 50, 0, np.where(coords[:, 0] < 95, 1, 2))
    lam = np.full((n_cells, len(genes)), 0.35 * depth_scale)
    lam[zone == 0, 0:2] = 10.0 * depth_scale
    lam[zone == 1, 2:4] = 10.0 * depth_scale
    lam[zone == 2, 4:6] = 10.0 * depth_scale
    adata = ad.AnnData(X=rng.poisson(lam).astype(np.int64))
    adata.obs_names = [f"cell_{seed}_{i}" for i in range(n_cells)]
    adata.var_names = genes
    adata.obsm["spatial"] = coords
    adata.obs["core"] = "demo_core_1"
    return adata


def frozen_marker_classifier(counts, gene_names):
    x = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    scores = np.column_stack([x[:, 0] + x[:, 1], x[:, 2] + x[:, 3], x[:, 4] + x[:, 5]])
    labels = np.array(["Epithelial", "T_cells", "Myeloid"])
    return labels[np.argmax(scores, axis=1)]


def main() -> None:
    ad_deep = make_demo_adata(seed=1, depth_scale=2.5)
    ad_shal = make_demo_adata(seed=2, depth_scale=0.7)

    for mode in ("rarefaction", "coverage", "mitm"):
        res = standardize_colocalization_anndata(
            ad_deep,
            ad_shal,
            classifier_fn=frozen_marker_classifier,
            mode=mode,
            region_key="core",
            radius_um=25.0,
            n_draws=10,
            n_permutations=49,
            seed=2026,
        )
        print(f"\n=== mode={mode} | rates={res.rates} ===")
        print(res.table[["run", "label_a", "label_b", "r_est", "z_dua", "permutation_p_z", "guardrail_triggered"]])

    sw = standardize_colocalization_anndata(
        ad_deep,
        classifier_fn=frozen_marker_classifier,
        mode="stability_weighted",
        region_key="core",
        radius_um=25.0,
        n_draws=10,
        seed=2026,
    )
    print("\n=== mode=stability_weighted (native deep run) ===")
    print(sw.table[["label_a", "label_b", "r_est", "z_unadj", "effective_n_a", "effective_n_b"]])


if __name__ == "__main__":
    main()
