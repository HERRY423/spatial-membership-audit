# Changelog

All notable changes to `spatial-membership-audit` are documented in this file.

## [3.0.0] - 2026-11-25

### Changed
- **Frozen Default Protocol (`DEFAULT_PROTOCOL_V3`)**: Promoted unsupervised two-sided conditional split-half coverage matching (`mode="split_half"`, `pairing="both"`, `q0=0.75`, `n_draws=20`, 9-point grid `1.00..0.10`, `split_half_max_cells=10000`, `split_half_reps=3`, `enforce_monotone=True`, zero-inclusive `mean(O_b)/mean(E_b)`, synchronized joint-permutation DUA) as the single primary cross-platform / cross-cohort default in `standardize_colocalization_anndata` (retaining `"coverage"` as a backward-compatible alias).
- **Explicit Secondary and Single-Specimen Modes**: Retained `mode="mitm"` (`geometric` target, `hier_eb`, `inv_sqrtd`, flip/depth guardrail) as the optional secondary moderate-gap mode and `mode="stability_weighted"` for single-specimen native-depth mapping.
- **Deterministic Stratified Split-Half Probing**: Added `_stratified_probe_indices` (`split_half_max_cells=10000`) to `standardize_colocalization_anndata` to cap probe cost on multi-hundred-thousand-cell cohorts.
- **Frozen Protocol Specification**: Added `protocol/FROZEN_DEFAULT_PROTOCOL_v3.0.md` and `protocol/protocol_v3_default.json` defining the pre-specified protocol, unified 2D evaluation framework, and prospective `nTMA` verification plan prior to data opening.

## [2.0.0] - 2026-11-24

### Added
- **Unified scverse / Squidpy high-level API (`standardize.py`)**: Added `standardize_colocalization_anndata` and `StandardizedComparison.to_squidpy_nhood_enrichment()` supporting all four operational modes (`"rarefaction"`, `"coverage"`, `"mitm"`, `"stability_weighted"`) directly on raw-count `AnnData` objects with any frozen classifier callable and Squidpy spatial graphs.
- **Within-run Rarefaction–Extrapolation (`extrapolation.py`)**: Added $1/\sqrt{d}$ shot-noise coordinate extrapolation (`summarize_thinning_ladder`, `select_mitm_target_depth`, `build_extrapolation_guardrail`, `extrapolate_ladder`) with hierarchical Empirical Bayes shrinkage (`prior_only`, `hier_eb`, `shrunk_step`) and low-depth / high-flip guardrails falling back to rarefaction.
- **Conditional split-half coverage matching (`coverage_matching.py`)**: Added zero-inflation-invariant split-half disagreement curves (`conditional_instability_metrics`, `split_half_disagreement`, `build_split_half_curve`, `match_split_half_coverage`) with antitonic PAVA smoothing for cross-panel and cross-platform SNR alignment.
- **Vectorized multi-pair Joint-Permutation DUA & three-regime decomposition (`dua_inference.py`)**: Added `joint_perm_pairs_from_draws`, `conformal_prediction_band`, and `decompose_dua_uncertainty`, separating fixed-input MC SE, exact finite-sample synchronized spatial permutation nulls, and out-of-sample split-conformal repeat bands.
- **Executable tutorial (`tutorials/v2_scverse_standardization.md`, `examples/v2_scverse_standardization.py`)** and scverse integration tests (`tests/test_standardize_scverse.py`).

### Fixed
- Cleaned corrupted escape characters in `README.md` and synchronized `CITATION.cff` / `pyproject.toml` metadata for release `2.0.0`.

## 2.0.0rc1 (local review candidate, 2026-11-22)

* The ratio denominator now includes defined zero-edge draws. The MC precision of log(mean R) uses their ratio-scale variance and the count of defined draws.
* Spatial testing of the ensemble mean uses a synchronized permutation across draws. The graph entry point constructs its own null bank. The array entry point needs the null ensemble means for calibration. A missing null or zero null SD yields an undefined standardized score.
* Repeat prediction requires independently calibrated repeat errors. Unit variance inflation, v1 DUA-logR tests and automatic v1 total-variance intervals have been retired. Use the archived 1.0.0 distribution for historical analyses.
* The validation extra provides the Python libraries used for the accompanying cohort analyses. The numerical API retains its two core dependencies.

## [1.0.0] - 2026-11-21

### Added
- **Threshold-free post-QC common-depth rarefaction (`depth_matching.py`)**: Replaced the legacy discontinuous `0.85` depth-ratio branch in `select_auto_q` with continuous post-QC common-depth rarefaction (`balance_threshold=1.0` default), $q_k = q_0 \cdot \min(s_A, s_B)/s_k$ for $q_0 \in (0, 1]$ (`q0=1.0` parameter-free one-sided rarefaction; `q0=0.75` two-sided stabilization), preserving legacy threshold opt-in via explicit `balance_threshold < 1.0`.
- **Three-regime DUA variance decomposition (`dua_inference.py`)**: Extended `summarize_dua_draws` with explicit Monte Carlo ensemble precision (`mc_se_logr`), within-run spatial permutation null documentation, cross-repeat predictive total variance (`DUA-z`, `DUA-logR`), availability gating ($P_+ \ge 0.90$), and 95% predictive interval width reporting (`ci95_width_logr`).
- **Thinning-stability weighting (`stability_weighting.py`)**: Native-depth unthinned estimator (`w_i = 1 - f_i`) with exact $O(M)$ closed-form fixed-size weighted permutation moments (`cell_stability_weights`, `exact_weighted_perm_pair`, `stability_weighted_colocalization`).
- **Regression suite (`tests/test_auto_q_and_dua.py`)**: Full coverage of threshold-free `select_auto_q`, multi-cohort zero-mismatch depth alignment, DUA variance regimes, and weighted permutation moments (230+ unit and regression tests).
- **Release packaging**: Added `CHANGELOG.md`, `CITATION.cff`, and verified sdist/wheel distributions (`dist/`).

### Changed
- Bumped package version from `0.2.0` to `1.0.0` across `pyproject.toml` and `spatial_membership_audit.__version__`.
- Documented single-thread BLAS pinning (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) for deterministic parallel `direct_cv.estimate` execution.

## [0.2.0] - 2026-10-13

- Initial prototype release with observed-window O/E and directional $G(r)$ CLI, read-only Xenium/CosMx count inspection, exact conditional CV engine (`direct_cv`), and scverse/Squidpy/Giotto endpoint bridges.
