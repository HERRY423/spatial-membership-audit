# Spatial Colocalization Standardization with AnnData & Squidpy (`v2.0.0`)

This quickstart shows how to standardize spatial cell-type colocalization across
two `AnnData` objects (different imaging runs, centers, panels, or platforms)
using any frozen cell-wise classifier callable, and export results directly into
Squidpy's `nhood_enrichment` matrix format.

## Operational Regime Selection

| Scenario | Recommended `mode` | Primary Purpose |
|---|---|---|
| Single run at native depth | `"stability_weighted"` | Preserve full transcript depth while downweighting unstable boundary cells ($w_i = 1 - f_i$) |
| Cross-run / cross-center with moderate shallow depth (median $\ge 15$ tx) | `"mitm"` | Jointly improve comparability and preserve deep-side contrast/protein fidelity via $1/\sqrt{d}$ hierarchical EB extrapolation + automatic guardrail |
| Cross-panel or cross-platform unequal per-transcript SNR | `"coverage"` | Align conditional split-half classifier disagreement $\psi(q)$ without ground-truth labels |
| Ultra-sparse shallow run or strict zero-effect FDR control | `"rarefaction"` | Threshold-free post-QC common-depth rarefaction (`auto-q`) with exact synchronized-permutation `DUA v2` inference |

## Minimal Example

```python
import squidpy as sq
from spatial_membership_audit import standardize_colocalization_anndata

# 1. Cross-run / cross-platform Meet-in-the-Middle (MITM) standardization
res_mitm = standardize_colocalization_anndata(
    adata_xenium,
    adata_cosmx,
    classifier_fn=my_frozen_classifier,   # (counts_matrix, gene_names) -> 1D labels
    mode="mitm",                          # or "coverage", "rarefaction"
    region_key="core",
    radius_um=25.0,
    n_draws=20,
    n_permutations=199,
    seed=2026,
)
print(res_mitm.rates)
print(res_mitm.table.head())

# 2. Export directly to Squidpy's nhood_enrichment structure
adata_xenium.uns["lineage_nhood_enrichment"] = res_mitm.to_squidpy_nhood_enrichment(
    run="A", region="core_1", stat="z_dua"
)

# 3. Single-run native-depth stability-weighted colocalization (accepts Squidpy graph)
sq.gr.spatial_neighbors(adata_xenium, coord_type="generic", radius=25.0)
res_native = standardize_colocalization_anndata(
    adata_xenium,
    classifier_fn=my_frozen_classifier,
    mode="stability_weighted",
    region_key="core",
    n_draws=20,
    seed=2026,
)
```

See [v2_scverse_standardization.py](../examples/v2_scverse_standardization.py) for an executable self-contained script.
