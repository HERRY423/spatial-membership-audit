#!/usr/bin/env python3
"""
P1-5 & P1-4 Analytical Synthesis:
  1. Complete Balanced vs Asymmetric Soft Plug-in Decomposition across ALL 11 benchmarks
     and BOTH SingleR softmax (tau=0.05/0.01) AND calibrated native posteriors
     (InSituType 2.0 generative probabilities & CellTypist 1.7.1 logistic probabilities).
  2. Complete 1s vs 2s and Split-half vs Depth-only cross-cohort & simulation synthesis.
Outputs:
  - p1_soft_variance_decomposition.csv
  - p1_1s_vs_2s_and_depth_comparison.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"c:\Reproducution")
REV = ROOT / "review"
OUT = REV / "v3_p1_hardening_2026-11-27"
OUT.mkdir(parents=True, exist_ok=True)

aud = REV / "v3_unified_audit_2026-11-25"
ntma = REV / "v3_ntma_prospective_2026-11-25"
clf = REV / "v3_classifier_independence_2026-11-25"
sim = REV / "v3_simulation_benchmark_2026-11-25"

t1 = pd.read_csv(aud / "audit_table1_ucommon.csv")
t1_null = pd.read_csv(aud / "audit_table1_paired_null_tests.csv")
t2 = pd.read_csv(aud / "audit_table2_ucommon.csv")
ttma = pd.read_csv(aud / "audit_ttma1_ucommon.csv")
ttma_null = pd.read_csv(aud / "audit_ttma1_paired_null_tests.csv")
ttma_rev = pd.read_csv(aud / "audit_ttma1_reversals_clustered_cis.csv")
nt_u = pd.read_csv(ntma / "ntma_ucommon_summary.csv")
nt_n = pd.read_csv(ntma / "ntma_paired_null_tests.csv")
nt_r = pd.read_csv(ntma / "ntma_organ_reversals.csv")
cl_u = pd.read_csv(clf / "classifiers_tma_ucommon.csv")
cl_n = pd.read_csv(clf / "classifiers_tma_paired_null.csv")
cl_r = pd.read_csv(clf / "classifiers_tma_reversals.csv")

# =============================================================================
# 1. Soft Plug-in Balanced vs Asymmetric + Native Posterior Synthesis
# =============================================================================
soft_rows = []

def add_row(cohort, regime, clf_type, soft_variant, u_n,
            naive_icc, soft_icc, sh_icc,
            naive_mae, soft_mae, sh_mae,
            soft_sd_a, soft_sd_b, sh_sd_a, sh_sd_b,
            soft_red_pct, sh_red_pct,
            naive_fdr_pan=np.nan, soft_fdr_pan=np.nan, sh_fdr_pan=np.nan,
            naive_fdr_disj=np.nan, soft_fdr_disj=np.nan, sh_fdr_disj=np.nan,
            naive_rev=np.nan, soft_rev=np.nan, sh_rev=np.nan,
            truth_note=""):
    soft_rows.append(dict(
        cohort=cohort,
        asymmetry_regime=regime,
        classifier_family=clf_type,
        soft_variant=soft_variant,
        u_common_n=int(u_n),
        naive_icc=round(float(naive_icc), 4),
        soft_icc=round(float(soft_icc), 4),
        primary_icc=round(float(sh_icc), 4),
        delta_icc_soft_vs_primary=round(float(soft_icc - sh_icc), 4),
        naive_mae=round(float(naive_mae), 4),
        soft_mae=round(float(soft_mae), 4),
        primary_mae=round(float(sh_mae), 4),
        soft_sd_a=round(float(soft_sd_a), 3),
        soft_sd_b=round(float(soft_sd_b), 3),
        soft_sd_asymmetry=round(float(abs(soft_sd_a - soft_sd_b)), 3),
        primary_sd_a=round(float(sh_sd_a), 3),
        primary_sd_b=round(float(sh_sd_b), 3),
        soft_red_pct=round(float(soft_red_pct), 2),
        primary_red_pct=round(float(sh_red_pct), 2),
        naive_fdr_pan_pct=naive_fdr_pan,
        soft_fdr_pan_pct=soft_fdr_pan,
        primary_fdr_pan_pct=sh_fdr_pan,
        naive_fdr_disj_pct=naive_fdr_disj,
        soft_fdr_disj_pct=soft_fdr_disj,
        primary_fdr_disj_pct=sh_fdr_disj,
        naive_rev_pct=naive_rev,
        soft_rev_pct=soft_rev,
        primary_rev_pct=sh_rev,
        truth_and_mechanism_note=truth_note,
    ))

# 4 Balanced / Near-Balanced cohorts (SingleR softmax tau=0.05)
def get1(df, b, arm):
    return df[(df["benchmark"] == b) & (df["arm"] == arm)].iloc[0]

r_nv, r_sf, r_sh = get1(t1, "MERFISH_Heart_G3", "Naive"), get1(t1, "MERFISH_Heart_G3", "Soft_plugin_tau005"), get1(t1, "MERFISH_Heart_G3", "Split_half_2s_P2")
add_row("MERFISH+ Heart G3", "Balanced_same_platform", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae"], r_sf["mae"], r_sh["mae"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        truth_note="Symmetric entropy: equal shrinkage (SD 0.76/0.69) suppresses random error faster than signal -> higher ICC (+0.127 vs SH) at ~27% contrast compression")

r_nv, r_sf, r_sh = get1(t1, "Touchstone_Xenium_PR1", "Naive"), get1(t1, "Touchstone_Xenium_PR1", "Soft_plugin_tau005"), get1(t1, "Touchstone_Xenium_PR1", "Depth_matched_MC200")
add_row("Touchstone Xenium PR1", "Balanced_same_platform", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae"], r_sf["mae"], r_sh["mae"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        truth_note="Balanced 98 vs 102 tx/cell: symmetric shrinkage (SD 0.54/0.56) lifts ICC (+0.264) by compressing ~45% of spatial SD")

def get_tt(tr, st, m):
    return ttma[(ttma["track"] == tr) & (ttma["stratum"] == st) & (ttma["method"] == m)].iloc[0]

def get_ttn(tr, st, m):
    return ttma_null[(ttma_null["track"] == tr) & (ttma_null["stratum"] == st) & (ttma_null["method"] == m)].iloc[0]

r_nv, r_sf, r_sh = get_tt("trackC", "all_qualified", "Naive"), get_tt("trackC", "all_qualified", "Soft_plugin_tau005"), get_tt("trackC", "all_qualified", "Split_half_2s_default")
n_nv, n_sf, n_sh = get_ttn("trackC", "all_qualified", "Naive"), get_ttn("trackC", "all_qualified", "Soft_plugin_tau005"), get_ttn("trackC", "all_qualified", "Split_half_2s_default")
add_row("tTMA1 Track C (Xenium Breast vs Panhuman)", "Balanced_cross_panel", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae_cross"], r_sf["mae_cross"], r_sh["mae_cross"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        round(100*n_nv["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sf["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sh["paired_t_pancancer_fdr05_rate"], 1),
        round(100*n_nv["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sf["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sh["disjoint_50_50_mean_fdr05_rate"], 1),
        truth_note="Equal depth (37 vs 38 tx): symmetric shrinkage (SD 0.62/0.59) boosts ICC (0.887 vs 0.833) & Re-ID, yet STILL raises pan-cancer paired-t FDR (13.3% -> 33.3% vs 6.7% SH)")

def get_nt(tr, arm):
    return nt_u[(nt_u["track"] == tr) & (nt_u["radius_um"] == 25) & (nt_u["arm"] == arm)].iloc[0]

def get_ntn(tr, arm):
    return nt_n[(nt_n["track"] == tr) & (nt_n["arm"] == arm)].iloc[0]

def get_ntr(tr, arm):
    return nt_r[(nt_r["track"] == tr) & (nt_r["arm"] == arm)].iloc[0]

r_nv, r_sf, r_sh = get_nt("Track_NC_Xenium_Breast_vs_Panhuman", "Naive"), get_nt("Track_NC_Xenium_Breast_vs_Panhuman", "Soft_plugin_tau005"), get_nt("Track_NC_Xenium_Breast_vs_Panhuman", "Split_half_2s_default")
n_nv, n_sf, n_sh = get_ntn("Track_NC_Xenium_Breast_vs_Panhuman", "Naive"), get_ntn("Track_NC_Xenium_Breast_vs_Panhuman", "Soft_plugin_tau005"), get_ntn("Track_NC_Xenium_Breast_vs_Panhuman", "Split_half_2s_default")
add_row("nTMA Track N-C (Xenium Breast vs Panhuman)", "Balanced_cross_panel", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae"], r_sf["mae"], r_sh["mae"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        round(n_nv["pan_paired_t_fdr_pct"], 1), round(n_sf["pan_paired_t_fdr_pct"], 1), round(n_sh["pan_paired_t_fdr_pct"], 1),
        round(n_nv["disjoint_50_50_welch_fdr_pct"], 1), round(n_sf["disjoint_50_50_welch_fdr_pct"], 1), round(n_sh["disjoint_50_50_welch_fdr_pct"], 1),
        0.0, 0.0, 0.0,
        truth_note="Balanced 21 vs 19 tx: symmetric shrinkage (SD 0.60/0.58) boosts ICC (0.864 vs 0.819) at 40% SD compression & lower Re-ID (87.5% vs 92.5%)")

# Asymmetric cohorts (SingleR softmax tau=0.05)
r_nv, r_sf, r_sh = get1(t1, "Wang_HTMA_r1_vs_r2", "Naive"), get1(t1, "Wang_HTMA_r1_vs_r2", "Soft_plugin_tau005"), get1(t1, "Wang_HTMA_r1_vs_r2", "Depth_only_2s_q075")
wn_nv = t1_null[(t1_null["benchmark"] == "Wang_HTMA_r1_vs_r2") & (t1_null["arm"] == "Naive")].iloc[0]
wn_sf = t1_null[(t1_null["benchmark"] == "Wang_HTMA_r1_vs_r2") & (t1_null["arm"] == "Soft_plugin_tau005")].iloc[0]
wn_sh = t1_null[(t1_null["benchmark"] == "Wang_HTMA_r1_vs_r2") & (t1_null["arm"] == "Depth_only_2s_q075")].iloc[0]
add_row("Wang HTMA r1 vs r2", "Asymmetric_depth_same_plat", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae"], r_sf["mae"], r_sh["mae"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        round(wn_nv["pan_paired_t_fdr_pct"], 1), round(wn_sf["pan_paired_t_fdr_pct"], 1), round(wn_sh["pan_paired_t_fdr_pct"], 1),
        round(wn_nv["disjoint_50_50_welch_fdr_pct"], 1), round(wn_sf["disjoint_50_50_welch_fdr_pct"], 1), round(wn_sh["disjoint_50_50_welch_fdr_pct"], 1),
        truth_note="Asymmetric depth: unequal shrinkage (SD 0.62 vs 0.43) halves r1 bio eta2 (0.128->0.069), stalls ICC (0.639 vs 0.855), inflates paired-t FDR (40%->60% vs 0%)")

r_nv, r_sf, r_sh = get_tt("trackA", "all_qualified", "Naive"), get_tt("trackA", "all_qualified", "Soft_plugin_tau005"), get_tt("trackA", "all_qualified", "Split_half_2s_default")
n_nv, n_sf, n_sh = get_ttn("trackA", "all_qualified", "Naive"), get_ttn("trackA", "all_qualified", "Soft_plugin_tau005"), get_ttn("trackA", "all_qualified", "Split_half_2s_default")
rv_nv = 100 * ttma_rev[(ttma_rev["track"] == "trackA") & (ttma_rev["method"] == "Naive")].iloc[0]["sign_rev_rate"]
rv_sf = 100 * ttma_rev[(ttma_rev["track"] == "trackA") & (ttma_rev["method"] == "Soft_plugin_tau005")].iloc[0]["sign_rev_rate"]
rv_sh = 100 * ttma_rev[(ttma_rev["track"] == "trackA") & (ttma_rev["method"] == "Split_half_2s_default")].iloc[0]["sign_rev_rate"]
add_row("tTMA1 Track A (Xenium vs CosMx)", "Asymmetric_cross_platform", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae_cross"], r_sf["mae_cross"], r_sh["mae_cross"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        round(100*n_nv["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sf["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sh["paired_t_pancancer_fdr05_rate"], 1),
        round(100*n_nv["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sf["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sh["disjoint_50_50_mean_fdr05_rate"], 1),
        round(rv_nv, 1), round(rv_sf, 1), round(rv_sh, 1),
        truth_note="Asymmetric SNR+depth: Soft worsens disjoint FDR (36.3%->54.5% vs 3.0% SH), spikes reversals (12%->22% vs 2% SH), degrades CosMx IF MAE (0.484->0.580)")

r_nv, r_sf, r_sh = get_tt("trackB", "all_qualified", "Naive"), get_tt("trackB", "all_qualified", "Soft_plugin_tau005"), get_tt("trackB", "all_qualified", "Split_half_2s_default")
n_nv, n_sf, n_sh = get_ttn("trackB", "all_qualified", "Naive"), get_ttn("trackB", "all_qualified", "Soft_plugin_tau005"), get_ttn("trackB", "all_qualified", "Split_half_2s_default")
rv_nv = 100 * ttma_rev[(ttma_rev["track"] == "trackB") & (ttma_rev["method"] == "Naive")].iloc[0]["sign_rev_rate"]
rv_sf = 100 * ttma_rev[(ttma_rev["track"] == "trackB") & (ttma_rev["method"] == "Soft_plugin_tau005")].iloc[0]["sign_rev_rate"]
rv_sh = 100 * ttma_rev[(ttma_rev["track"] == "trackB") & (ttma_rev["method"] == "Split_half_2s_default")].iloc[0]["sign_rev_rate"]
add_row("tTMA1 Track B (Xenium vs MERSCOPE)", "Asymmetric_cross_platform", "SingleR_corr", "Softmax_tau005", r_nv["u_common_n"],
        r_nv["icc"], r_sf["icc"], r_sh["icc"], r_nv["mae_cross"], r_sf["mae_cross"], r_sh["mae_cross"],
        r_sf["sd_ratio_a"], r_sf["sd_ratio_b"], r_sh["sd_ratio_a"], r_sh["sd_ratio_b"],
        r_sf["red_pct"], r_sh["red_pct"],
        round(100*n_nv["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sf["paired_t_pancancer_fdr05_rate"], 1), round(100*n_sh["paired_t_pancancer_fdr05_rate"], 1),
        round(100*n_nv["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sf["disjoint_50_50_mean_fdr05_rate"], 1), round(100*n_sh["disjoint_50_50_mean_fdr05_rate"], 1),
        round(rv_nv, 1), round(rv_sf, 1), round(rv_sh, 1),
        truth_note="3.7x depth gap: Soft drops ICC BELOW Naive (0.693->0.571 vs 0.773 SH), doubles disjoint FDR (23.2%->46.2%), spikes reversals (11.1%->27.8%)")

# Non-strawman test: Calibrated Native Posterior Classifiers (InSituType 2.0 & CellTypist 1.7.1)
for ds, clname in [
    ("tTMA1_TrackA_Xenium_CosMx_IF", "InSituType"),
    ("tTMA1_TrackB_Xenium_MERSCOPE", "InSituType"),
    ("nTMA_TrackNA_Xenium_CosMx_IF", "InSituType"),
    ("tTMA1_TrackA_Xenium_CosMx_IF", "CellTypist"),
    ("tTMA1_TrackB_Xenium_MERSCOPE", "CellTypist"),
    ("nTMA_TrackNA_Xenium_CosMx_IF", "CellTypist"),
]:
    c_nv = cl_u[(cl_u["dataset"] == ds) & (cl_u["classifier"] == clname) & (cl_u["arm"] == "Naive")].iloc[0]
    c_sf = cl_u[(cl_u["dataset"] == ds) & (cl_u["classifier"] == clname) & (cl_u["arm"] == "Soft_plugin_native_prob")].iloc[0]
    c_sh = cl_u[(cl_u["dataset"] == ds) & (cl_u["classifier"] == clname) & (cl_u["arm"] == "Split_half_2s_default")].iloc[0]
    cn_nv = cl_n[(cl_n["dataset"] == ds) & (cl_n["classifier"] == clname) & (cl_n["arm"] == "Naive")].iloc[0]
    cn_sf = cl_n[(cl_n["dataset"] == ds) & (cl_n["classifier"] == clname) & (cl_n["arm"] == "Soft_plugin_native_prob")].iloc[0]
    cn_sh = cl_n[(cl_n["dataset"] == ds) & (cl_n["classifier"] == clname) & (cl_n["arm"] == "Split_half_2s_default")].iloc[0]
    cr_nv = cl_r[(cl_r["dataset"] == ds) & (cl_r["classifier"] == clname) & (cl_r["arm"] == "Naive")].iloc[0]
    cr_sf = cl_r[(cl_r["dataset"] == ds) & (cl_r["classifier"] == clname) & (cl_r["arm"] == "Soft_plugin_native_prob")].iloc[0]
    cr_sh = cl_r[(cl_r["dataset"] == ds) & (cl_r["classifier"] == clname) & (cl_r["arm"] == "Split_half_2s_default")].iloc[0]
    fam = "Generative_Poisson_NB" if clname == "InSituType" else "Discriminative_Logistic"
    note = (
        f"Calibrated native posterior (NO SD collapse: {c_sf['sd_ratio_a']:.2f}/{c_sf['sd_ratio_b']:.2f}), "
        f"yet fails to correct cross-platform shift (+{c_sf['red_pct']:.1f}% vs +{c_sh['red_pct']:.1f}% SH) "
        f"and worsens/leaves high disjoint FDR ({cn_sf['disjoint_50_50_welch_fdr_pct']:.1f}% vs {cn_sh['disjoint_50_50_welch_fdr_pct']:.1f}%)"
    )
    add_row(f"{ds} [{clname}]", "Asymmetric_cross_platform", fam, "Native_posterior_prob", c_nv["u_common_n"],
            c_nv["icc"], c_sf["icc"], c_sh["icc"], c_nv["mae"], c_sf["mae"], c_sh["mae"],
            c_sf["sd_ratio_a"], c_sf["sd_ratio_b"], c_sh["sd_ratio_a"], c_sh["sd_ratio_b"],
            c_sf["red_pct"], c_sh["red_pct"],
            round(cn_nv["pan_paired_t_fdr_pct"], 1), round(cn_sf["pan_paired_t_fdr_pct"], 1), round(cn_sh["pan_paired_t_fdr_pct"], 1),
            round(cn_nv["disjoint_50_50_welch_fdr_pct"], 1), round(cn_sf["disjoint_50_50_welch_fdr_pct"], 1), round(cn_sh["disjoint_50_50_welch_fdr_pct"], 1),
            round(cr_nv["top_k_reversal_pct"], 1), round(cr_sf["top_k_reversal_pct"], 1), round(cr_sh["top_k_reversal_pct"], 1),
            truth_note=note)

df_soft = pd.DataFrame(soft_rows)
df_soft.to_csv(OUT / "p1_soft_variance_decomposition.csv", index=False)
print("=== P1-5 Soft Plug-in Balanced vs Asymmetric & Native Posterior Summary ===")
print(df_soft[["cohort", "asymmetry_regime", "soft_variant", "naive_icc", "soft_icc", "primary_icc", "soft_sd_a", "soft_sd_b", "soft_red_pct", "primary_red_pct", "soft_fdr_disj_pct", "primary_fdr_disj_pct"]].to_string(index=False))

# =============================================================================
# 2. Complete 1s vs 2s and Split-half vs Depth-only Simulation Synthesis
# =============================================================================
sf = pd.read_csv(sim / "simulation_factorial_summary.csv")
piv_rows = []
for (dr, snr, d), grp in sf.groupby(["depth_ratio", "snr_mismatch", "delta_true_target"]):
    if d not in (0.0, 0.30):
        continue
    m = grp.set_index("method")
    piv_rows.append(dict(
        depth_ratio=dr,
        snr_mismatch=snr,
        delta_true=d,
        mean_qa_sh2s=round(float(m.loc["SplitHalf_2s_Default", "mean_qa_sh2s"]), 4),
        sh2s_rej_pct=round(100 * float(m.loc["SplitHalf_2s_Default", "reject_rate_p05"]), 1),
        sh1s_rej_pct=round(100 * float(m.loc["SplitHalf_1s", "reject_rate_p05"]), 1),
        depth2s_rej_pct=round(100 * float(m.loc["Depth_Raref_2s", "reject_rate_p05"]), 1),
        sh2s_mae=round(float(m.loc["SplitHalf_2s_Default", "mae"]), 4),
        sh1s_mae=round(float(m.loc["SplitHalf_1s", "mae"]), 4),
        depth2s_mae=round(float(m.loc["Depth_Raref_2s", "mae"]), 4),
        sh2s_sd=round(float(m.loc["SplitHalf_2s_Default", "sd_ratio_deep"]), 3),
        sh1s_sd=round(float(m.loc["SplitHalf_1s", "sd_ratio_deep"]), 3),
        clamped_regime=(float(m.loc["SplitHalf_2s_Default", "mean_qa_sh2s"]) <= 0.105),
    ))
df_cmp = pd.DataFrame(piv_rows)
df_cmp.to_csv(OUT / "p1_1s_vs_2s_and_depth_comparison.csv", index=False)
print("\n=== P1-4 Simulation: SplitHalf_2s vs SplitHalf_1s vs Depth_Raref_2s ===")
print(df_cmp.to_string(index=False))
