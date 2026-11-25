# Frozen Default Protocol v3.0.0 (`spatial-membership-audit`)

**Protocol Version:** `3.0.0`  
**Status:** FROZEN PRIOR TO OPENING `Wang et al. (2025) nTMA` EXPRESSION DATA  
**Package:** `spatial-membership-audit` (`DEFAULT_PROTOCOL_V3`)

---

## 1. Single Frozen Default & Secondary Operational Protocol

All new independent datasets (`Wang et al. (2025) nTMA`, never opened prior to tagging this commit) MUST be evaluated using ONLY the single frozen default configuration below (plus the optional secondary moderate-gap configuration and fixed reference baselines). No alternative `q_0` grids, one-sided variants, or extrapolation variants may be tuned or cherry-picked on `nTMA`. All parameter sweeps and alternative variants belong strictly in Supplementary sensitivity analyses on development/retrospective cohorts.

### 1.1 Primary Default — Cross-Platform / Cross-Cohort Standardization
- **Estimator:** Two-sided Unsupervised Conditional Poisson Split-Half Coverage Matching (`mode="split_half"`, `pairing="both"`, `q0=0.75`)
- **Split-half probe:** Grid `q ∈ {1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10}`, `split_half_max_cells = 10000` (region-stratified), `split_half_reps = 3`, conditional on classifiable cells (`T_i > 0`), PAVA monotone non-increasing projection (`enforce_monotone=True`), log-linear interpolation clamped to `[0.05, 1.00]`.
- **Monte Carlo ensemble:** `n_draws = 20` independent conditional binomial thinning draws (`X^(b) ~ Binomial(X_obs, q)`), zero-inclusive ratio estimator `R_hat = mean(O_b) / mean(E_b)` on `E_b > 0`, availability gate `P_+ >= 0.90`.
- **Spatial neighborhood graph:** Undirected `r = 25 µm` radius graph (primary; `r = 50 µm` pre-specified sensitivity).
- **Spatial inference:** Synchronized joint-permutation DUA (`n_permutations = 199` cohort sweeps / `999` single-unit diagnostics, Phipson–Smyth plus-one p-value).

### 1.2 Optional Secondary — Moderate-Gap Fidelity-Preserving Standardization
- **Estimator:** Geometric-mean Meet-in-the-Middle Rarefaction–Extrapolation (`mode="mitm"`, `mitm_mode="geometric"`, `d* = sqrt(s_A * s_B)`), `1/sqrt(d)` empirical-Bayes slope pooling (`extrap_method="hier_eb"`, `ladder_qs=[1.00, 0.85, 0.75, 0.65, 0.50]`), with automatic low-depth / high-flip guardrail (`min_median_depth=2.0`, `max_cond_flip_rate=0.45`; recommended operating floor `>= 25` transcripts/cell) falling back to rarefaction.

### 1.3 Single-Specimen Architectural Mapping (No Cross-Cohort Contrast)
- **Estimator:** Unthinned Stability Weighting (`mode="stability_weighted"`, `w_i = 1 - f_i`, `stability_q = 0.75`, `n_draws = 20`, exact weighted fixed-margin permutation moments).

---

## 2. Unified Evaluation Framework (Pre-Specified)

1. **Primary 2D Endpoint:** Joint improvement on **(a) Cross-platform reproducibility (`ICC(1,1)`) & Paired Zero-Effect False Discovery Rate (`BH-FDR < 0.05`)** AND **(b) Biological Signal Retention (`η²` variance explained across tissue/organ types and Deep/Shallow `SD` ratio)**.
2. **Strict Common Unit Set (`U_common`):** Every comparative table evaluates ALL methods on the **exact same common set of evaluable `(unit, pair)` entries (`U_common`)** defined across all compared methods, while reporting each method's raw availability `%` separately.
3. **Dual Metric Reporting (Correlation + Error):** Every benchmark table reports BOTH correlation/rank metrics (`ICC`, `Pearson r` / `Spearman ρ`) AND error metrics (`MAE`, median `|Δ log R|`) alongside `η²` and `SD` ratio. Note upfront: soft plug-in shrinkage toward zero lowers raw `MAE` against noisy targets solely via variance compression (`SD << 1`) while degrading correlation and structure; correlation + `MAE` + `SD` + `η²` must therefore be interpreted jointly.
4. **Statistical Testing & Clustered Uncertainty:**
   - Zero-effect null tests (`Δ_bio = 0` on paired cores) use **paired t-test and paired Wilcoxon signed-rank test** (BH-FDR `< 0.05` across 15 pairs; additionally reporting within-tissue paired tests and random 50/50 disjoint two-center splits).
   - Confidence intervals use **tissue-stratified core-clustered bootstrap** AND **spatial TMA-block / leave-one-organ-out clustered bootstrap** (`B_boot = 2000`).

---

## 3. Pre-Specified Prospective Independent Cohort: `Wang et al. (2025) nTMA`

**Cohort:** Normal Multi-Tissue Microarray (`nTMA`, distinct TMA block spanning 17 normal organs from Wang et al. 2025, Zenodo `10.5281/zenodo.16848917`; expression `.h5ad` matrices **unopened** at protocol freeze time).  
**Tracks:**
- **`Track N-A`**: `2023_xenium_breast_nTMA` vs `2023_cosmx_multitissue_nTMA`
- **`Track N-B`**: `2023_xenium_breast_nTMA` vs `2023_merscope_breast_nTMA`
- **`Track N-C`**: `2023_xenium_breast_nTMA` vs `2023_xenium_panhuman_nTMA`

**Core Qualification Rules (identical to `tTMA1`):** Core IDs present in both runs with a unique concordant `tissue_type` label, excluding `MARKER` cores, positive shared-HPCA counts, and verified µm coordinates.  
**Fixed Comparison Arms on `nTMA`:**
1. `Naive` (unstandardized native labels)
2. `MNN rescaleBatches` (expression batch correction)
3. `Soft plug-in (τ = 0.05)`
4. `Depth-only rarefaction (2s, q_0 = 0.75)`
5. **`Split-half coverage match (2s, q_0 = 0.75) + Joint-Perm` [PRIMARY DEFAULT]**
6. **`MITM (geom, EB) + Guardrail` [OPTIONAL SECONDARY]**
7. Linux spatial baselines (`CRAWDAD`, `spicyR`, `PANORAMIC` on Track N-A).
