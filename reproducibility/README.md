# Manuscript Reproducibility Archive (`v3.0.0-manuscript-archive`)

This directory archives all lightweight derived tables, native Linux execution scripts, environment manifests, protocol-freeze receipts, and figure scripts supporting the *Genome Biology* Methodology manuscript:

> *Rarefaction–extrapolation standardization of cell-type colocalization across imaging spatial transcriptomics platforms*

## Integrity & Chronological Protocol Freeze

- **Immutable pre-fetch freeze tag:** `v3.0.0-protocol-freeze-20261125` (commit `aa2ec5f1a8c8d488ad5bb95df8b2331e1b40141e`, UTC `2026-11-25T14:12:08Z`).
- **Zero code drift:** All package source files (`src/`), tests (`tests/`), and frozen protocol definitions (`protocol/`) in this archive tag are **byte-for-byte identical** to `v3.0.0-protocol-freeze-20261125`.
- **Verify all `74` archive files in <1 second:**
  ```bash
  python reproducibility/verify_archive_checksums.py
  ```
- **Verify all `248` software tests (`+ 79` subtests):**
  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
  ```

## Contents

| Subdirectory | Contents |
|---|---|
| `data/v3_protocol_freeze_2026-11-25/` | Public GitHub tag verification receipt (`protocol_freeze_receipt.json`) |
| `data/v3_ntma_prospective_2026-11-25/` | Post-freeze UTC fetch receipt (`ntma_fetch_receipt.json`), prospective `Wang 2025 nTMA` summaries (`U_common`, nulls, organ reversals, IF) & Linux scripts |
| `data/v3_unified_audit_2026-11-25/` | Unified `U_common` CSVs (Tables 1–3, 6), 3-tier paired null tests, TMA-block CIs, and archived 423-row `S4` sensitivity sweep |
| `data/v3_simulation_benchmark_2026-11-25/` | `2,880`-cohort controlled factorial simulation summary, discard power-loss curves, segregation check (`mu_base = -0.45`), and simulation script |
| `data/v3_linux_competitors_2026-11-25/` | Native Ubuntu Linux (`R 4.6.1`) `Proseg v3.2.0`, `spicyR 1.22.0`, `CRAWDAD 1.0.1`, `PANORAMIC 0.99.4` summaries, `env_versions.json`, and scripts |
| `data/v3_classifier_independence_2026-11-25/` | `InSituType 2.0` & `CellTypist 1.7.1` cross-platform TMA & M242T IMC replications and scripts |
| `data/v3_35core_decomposition_2026-11-25/` | 35-core negative probe reproduction, 6-gene marker-sparsity diagnostics, 60-cell 3-factor ablation, and scripts |
| `data/p7_dua_theory_and_audit_2026-11-23/` | Finite-sample Proposition 1–3, split-conformal, and `1/sqrt(d)` MITM empirical verification tables |
| `manuscript_tables/` | Multi-sheet `Additional_file_2_Supplementary_Tables.xlsx`, `EVIDENCE_CROSSWALK.md`, `ANALYSIS_HISTORY.md`, and `package_verification.json` |
| `figure_scripts/` | Publication figure scripts (`Figures 1–6` & `Supplementary Figures 1–2`) |
