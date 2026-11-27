# `spatial-membership-audit` (v3.0.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Protocol Freeze](https://img.shields.io/badge/protocol--freeze-v3.0.0--20261125-1E8449.svg)](https://github.com/HERRY423/spatial-membership-audit/releases/tag/v3.0.0-protocol-freeze-20261125)

**Rarefaction–extrapolation standardization of cell-type colocalization across imaging spatial transcriptomics platforms.**

`spatial-membership-audit` implements a publicly frozen two-estimand standardization framework (`DEFAULT_PROTOCOL_V3`) for cell-type colocalization across heterogeneous imaging spatial transcriptomics (iST) instruments (10x Xenium, NanoString CosMx, Vizgen MERSCOPE, MERFISH+), gene panels, storage durations, upstream resegmentation (`Proseg`), downstream spatial tools (`Squidpy`, `spicyR`, `CRAWDAD`, `PANORAMIC`), and classifiers (`SingleR`, `InSituType`, `CellTypist`).

---

## Frozen Two-Estimand Framework (`DEFAULT_PROTOCOL_V3`)

| Operational Regime | Estimand | Method in `spatial-membership-audit` | Key Properties |
|---|---|---|---|
| **Regime 1 (Primary Default)** | Cross-cohort & cross-platform comparability | `mode="split_half"` (`q0=0.75`, `pairing="both"`, `B=20`) + `joint_perm_pairs_from_draws` | Unsupervised conditional Poisson split-half coverage matching (Props. 1–3); zero paired training samples; eliminates SNR & depth-driven false discoveries. |
| **Regime 2 (Optional Secondary)** | Moderate depth gaps (`<= 6–7x`) balancing comparability & native fidelity | `mode="mitm"` (`mitm_mode="geometric"`, `1/sqrt(d)` EB + low-depth guardrail) | Preserves 82–90% deep contrast & orthogonal protein fidelity; automatically falls back to rarefaction when starved. |
| **Regime 3 (Single Specimen)** | Single-specimen architectural mapping | `mode="stability_weighted"` (`w_i = 1 - f_i`, unthinned) | Closed-form `O(M)` weighted fixed-margin permutation moments; zero discarded molecules; improves protein IF concordance. |
| **Regime 4 (Sparsity Guardrail)** | Extreme sparsity (`< 20–25` tx/cell, `> 10x` gap, or `boundary_clamped=True`) | Automatic guardrail fallback + explicit diagnostic flags | Prevents upward extrapolation collapse and flags lower-bound clamping transparently. |

Core runtime dependencies are strictly **NumPy** and **SciPy**; optional extras (`[io]`, `[scverse]`, `[validation]`) add `AnnData`/`Squidpy` and cohort analysis tooling.

---

## Installation & Quickstart

```bash
pip install git+https://github.com/HERRY423/spatial-membership-audit.git
# Optional scverse / AnnData integration:
pip install "spatial-membership-audit[scverse] @ git+https://github.com/HERRY423/spatial-membership-audit.git"
```

### Minimal Example

```python
from spatial_membership_audit import (
    get_default_protocol_v3,
    summarize_dua_draws,
    standardize_colocalization_anndata,
)

# Inspect frozen protocol specification
proto = get_default_protocol_v3()
primary = proto["primary_cross_platform"]
assert proto["protocol_version"] == "3.0.0"
assert primary["mode"] == "split_half" and primary["q0"] == 0.75 and primary["n_draws"] == 20

# Zero-inclusive conditional mean ratio & delta-method MC SE (Proposition 1)
result = summarize_dua_draws([0, 2, 2], [-1, 1, 1])
assert abs(result.r_mean - 4 / 3) < 1e-12
assert abs(result.mc_se_logr - 0.5) < 1e-12
```

See [tutorials/v2_scverse_standardization.md](tutorials/v2_scverse_standardization.md) and [protocol/FROZEN_DEFAULT_PROTOCOL_v3.0.md](protocol/FROZEN_DEFAULT_PROTOCOL_v3.0.md) for full `AnnData`/`Squidpy` workflows (`to_squidpy_nhood_enrichment()`).

### Running the Test Suite (`248 passed, 79 subtests passed`)

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
```

---

## Manuscript Reproducibility & Protocol Freeze Audit

To separate pre-specified prospective validation from manuscript reporting:

1. **Immutable Pre-Fetch Protocol Freeze Tag:** [`v3.0.0-protocol-freeze-20261125`](https://github.com/HERRY423/spatial-membership-audit/releases/tag/v3.0.0-protocol-freeze-20261125) (commit `aa2ec5f1a8c8d488ad5bb95df8b2331e1b40141e`, UTC `2026-11-25T14:12:08Z`, `protocol/protocol_v3_default.json` SHA-256 `468de661fb885938a396e9a536ba90ca6f36b5a6ceab2c279a08cd7c682e9459`) was tagged **prior to fetching** the independent `Wang 2025 nTMA` 15-organ normal tissue microarray (`2026-11-26T15:41:02Z–15:44:17Z`).
2. **Manuscript Reproducibility Archive (`reproducibility/`, tag `v3.0.0-manuscript-archive`):** Adds **zero changes** to `src/`, `tests/`, or `protocol/` relative to the freeze tag, and bundles:
   - All derived `U_common` CSV summaries, `2,880`-cohort controlled simulation tables, native Linux `Proseg v3.2.0` / `spicyR 1.22.0` / `CRAWDAD 1.0.1` / `PANORAMIC 0.99.4` benchmarks, `InSituType 2.0` & `CellTypist 1.7.1` replications, prospective `Wang 2025 nTMA` outputs, `35`-core 3-factor ablation tables, and archived `S4` sensitivity sweeps (`reproducibility/data/`);
   - Native Linux environment manifest (`env_versions.json`), protocol freeze receipt, post-freeze nTMA UTC fetch receipt, and SHA-256 verification script (`python reproducibility/verify_archive_checksums.py`);
   - Figure-generation scripts (`reproducibility/figure_scripts/`).

---

## Documentation & License

- Frozen Protocol Spec: [protocol/FROZEN_DEFAULT_PROTOCOL_v3.0.md](protocol/FROZEN_DEFAULT_PROTOCOL_v3.0.md)
- scverse / Squidpy Tutorial: [tutorials/v2_scverse_standardization.md](tutorials/v2_scverse_standardization.md)
- Reproducibility Guide: [reproducibility/README.md](reproducibility/README.md)
- Historical Development Log (`0.2.0`–`2.0.0rc2`): [docs/HISTORICAL_DEVELOPMENT_LOG.md](docs/HISTORICAL_DEVELOPMENT_LOG.md)
- License: [MIT License](LICENSE) | Citation: [CITATION.cff](CITATION.cff)
