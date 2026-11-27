#!/usr/bin/env python3
"""
Step 6 (Calibrated v3): Controlled Ground-Truth Simulation Benchmark (Reviewer Item 4).

Systematically varies:
  - Depth ratio r_d = s_shallow / s_deep in {1.0, 0.50, 0.25, 0.15}
  - Per-transcript SNR mismatch gamma_SNR in {1.0, 2.0, 3.0}
  - True colocalization effect Delta_true in {0.00, 0.15, 0.30, 0.45}
Around realistic baseline spatial enrichment mu_base = +0.75 (plus segregation check mu_base = -0.75).

Directly quantifies:
  1. Type-I error (delta=0), power (delta>0), bias, MAE, and SD retention across all 48 regimes.
  2. Pure Deep-Molecule-Discard Power Curve (q = 1.00 -> 0.15) answering:
     "How much power is lost by discarding deep-side molecules?" vs Oracle, Native Unthinned,
     Stability Weighting, and MITM (under r_d = 0.25).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind

ROOT = Path(r"c:\Reproducution")
sys.path.insert(0, str(ROOT / "software" / "spatial_membership_audit" / "src"))
from spatial_membership_audit.coverage_matching import (
    build_split_half_curve,
    match_split_half_coverage,
    split_half_disagreement,
)
from spatial_membership_audit.extrapolation import (
    build_extrapolation_guardrail,
    extrapolate_ladder,
    select_mitm_target_depth,
    summarize_thinning_ladder,
)

OUT_DIR = ROOT / "review" / "v3_simulation_benchmark_2026-11-25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 6
N_GENES = 60
PROBE_GRID = (1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
LADDER_QS = (1.00, 0.85, 0.75, 0.65, 0.50)

# Realistic 60-gene panel calibrated to spatial transcriptomics SingleR operating range:
#   - 4 primary markers per class (0..23)
#   - 2 weak shared cross-lineage markers per class (24..35)
#   - 24 background/housekeeping genes (36..59)
_SIGNAL_PROFILE = np.full((N_CLASSES, N_GENES), 0.28, dtype=np.float64)
for _k in range(N_CLASSES):
    _SIGNAL_PROFILE[_k, _k * 4 : (_k + 1) * 4] = 1.55
    _k_next = (_k + 1) % N_CLASSES
    _SIGNAL_PROFILE[_k, 24 + _k * 2 : 24 + (_k + 1) * 2] = 0.52
    _SIGNAL_PROFILE[_k_next, 24 + _k * 2 : 24 + (_k + 1) * 2] = 0.44
_SIGNAL_PROFILE /= _SIGNAL_PROFILE.sum(axis=1, keepdims=True)

_W_NORM = _SIGNAL_PROFILE - _SIGNAL_PROFILE.mean(axis=1, keepdims=True)
_W_NORM /= np.linalg.norm(_W_NORM, axis=1, keepdims=True)
_W_NORM_T = np.ascontiguousarray(_W_NORM.T, dtype=np.float32)

S_DEEP = 100.0
CORE_CELLS = 1000
CORE_BOX_UM = 240.0
CORE_RADIUS_UM = 25.0


def classify_counts_fast(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tot = counts.sum(axis=1)
    lx = np.log1p(counts.astype(np.float32))
    lx_c = lx - lx.mean(axis=1, keepdims=True)
    norm_x = np.linalg.norm(lx_c, axis=1, keepdims=True)
    scores = (lx_c / np.maximum(norm_x, 1e-6)) @ _W_NORM_T
    labels = np.argmax(scores, axis=1).astype(np.int32)
    labels[tot <= 0] = N_CLASSES
    return labels, scores


def build_core_spatial_graph(n_cells: int, box_um: float, radius_um: float, rng: np.random.Generator):
    xy = rng.uniform(0.0, box_um, size=(n_cells, 2))
    pairs = cKDTree(xy).query_pairs(r=radius_um, output_type="ndarray")
    if len(pairs) == 0:
        return xy, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0.0
    u = pairs[:, 0].astype(np.int32)
    v = pairs[:, 1].astype(np.int32)
    c_const = 2.0 * len(u) / (n_cells * (n_cells - 1.0))
    return xy, u, v, c_const


def simulate_true_labels_with_coloc(
    xy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    c_const: float,
    target_logr: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(xy)
    # 1. Assign class 0 with tumor-nest spatial compartmentalization (~28% prevalence)
    # so neighborhood enrichment spans [-1.0, +0.85] naturally without ceiling clamping.
    n0_target = max(10, int(round(0.28 * n)))
    centers = rng.uniform(35.0, CORE_BOX_UM - 35.0, size=(2, 2))
    d2_min = np.min(np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    log_w0 = -d2_min / (2.0 * 32.0 ** 2)
    p0 = 0.12 + 0.88 * np.exp(log_w0 - log_w0.max())
    p0 /= p0.sum()
    idx0 = rng.choice(n, size=n0_target, replace=False, p=p0)
    labels = np.full(n, -1, dtype=np.int32)
    labels[idx0] = 0

    is0 = (labels == 0).astype(np.int32)
    n0_nbr = np.zeros(n, dtype=np.float64)
    if len(u) > 0:
        np.add.at(n0_nbr, u, is0[v])
        np.add.at(n0_nbr, v, is0[u])
    deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64) if len(u) > 0 else np.zeros(n)
    frac0 = np.where(deg > 0, n0_nbr / np.maximum(deg, 1.0), 0.28)
    z0 = (frac0 - 0.28) / max(1e-6, float(np.std(frac0)))

    # 2. Sample ~16% class 1 from non-0 nodes with exponential propensity beta * z0
    rem = np.flatnonzero(labels != 0)
    n1_target = max(8, int(round(0.16 * n)))
    if len(rem) > n1_target and len(u) > 0:
        exp_e = c_const * len(idx0) * n1_target
        target_obs = max(0.5, exp_e * math.exp(target_logr))
        lo, hi = -4.0, 4.0
        z_rem = z0[rem]
        k_rem = n0_nbr[rem]
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            lw = mid * z_rem
            lw -= lw.max()
            w = np.exp(lw)
            w /= w.sum()
            e_obs = n1_target * float(np.dot(w, k_rem))
            if e_obs < target_obs:
                lo = mid
            else:
                hi = mid
        beta = 0.5 * (lo + hi)
        lw = beta * z_rem
        lw -= lw.max()
        p1 = np.exp(lw)
        p1 /= p1.sum()
        idx1 = rng.choice(rem, size=n1_target, replace=False, p=p1)
    else:
        idx1 = rng.choice(rem, size=min(n1_target, len(rem)), replace=False)
    labels[idx1] = 1

    # 3. Assign classes 2..5 to remaining cells
    rem2 = np.flatnonzero(labels < 0)
    p_other = np.array([0.18, 0.10, 0.14, 0.14])
    p_other /= p_other.sum()
    labels[rem2] = rng.choice([2, 3, 4, 5], size=len(rem2), p=p_other).astype(np.int32)

    exp_e = c_const * int(np.sum(labels == 0)) * int(np.sum(labels == 1))
    if exp_e <= 0 or len(u) == 0:
        return labels, np.zeros(n, dtype=np.float64), np.nan
    obs = int(n0_nbr[labels == 1].sum())
    het = np.zeros(n, dtype=np.float64)
    diff = (labels[u] != labels[v]).astype(np.float64)
    np.add.at(het, u, diff)
    np.add.at(het, v, diff)
    het_frac = np.where(deg > 0, het / np.maximum(deg, 1.0), 0.0)
    true_logr = math.log(max(obs, 0.5) / exp_e)
    return labels, het_frac, true_logr


def sample_core_counts(
    labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    het_frac: np.ndarray,
    mean_depth: float,
    snr_mismatch: float,
    rng: np.random.Generator,
    spillover_strength: float = 0.20,
) -> np.ndarray:
    n = len(labels)
    base_prof = _SIGNAL_PROFILE[labels].copy()
    if len(u) > 0 and spillover_strength > 0.0:
        nbr_sum = np.zeros((n, N_GENES), dtype=np.float64)
        np.add.at(nbr_sum, u, _SIGNAL_PROFILE[labels[v]])
        np.add.at(nbr_sum, v, _SIGNAL_PROFILE[labels[u]])
        deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64)
        has_nbr = deg > 0
        nbr_mean = np.where(has_nbr[:, None], nbr_sum / np.maximum(deg[:, None], 1.0), base_prof)
        alpha = spillover_strength * het_frac[:, None]
        base_prof = (1.0 - alpha) * base_prof + alpha * nbr_mean

    if snr_mismatch > 1.0:
        w_sig = 1.0 / (1.0 + 0.35 * (snr_mismatch - 1.0))
        unif = 1.0 / N_GENES
        prof = w_sig * base_prof + (1.0 - w_sig) * unif
    else:
        prof = base_prof

    cell_depth = mean_depth * rng.gamma(shape=5.0, scale=0.20, size=n)
    lam = cell_depth[:, None] * prof
    return rng.poisson(lam).astype(np.int32)


def fast_binomial(mat: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    if q >= 1.0 - 1e-12:
        return mat
    if q <= 0.0:
        return np.zeros_like(mat)
    return rng.binomial(mat, q)


def pair_r_from_labels(labels: np.ndarray, u: np.ndarray, v: np.ndarray, c_const: float) -> float:
    na = int(np.sum(labels == 0))
    nb = int(np.sum(labels == 1))
    exp_e = c_const * na * nb
    if exp_e <= 0.0:
        return np.nan
    obs = int(np.sum(((labels[u] == 0) & (labels[v] == 1)) | ((labels[u] == 1) & (labels[v] == 0))))
    return float(obs / exp_e)


def pair_r_soft(scores: np.ndarray, u: np.ndarray, v: np.ndarray, c_const: float, tau: float = 0.05) -> float:
    sc = scores.astype(np.float64)
    ex = np.exp((sc - sc.max(axis=1, keepdims=True)) / tau)
    P = ex / ex.sum(axis=1, keepdims=True)
    pa, pb = P[:, 0], P[:, 1]
    denom = float(pa.sum() * pb.sum() - np.dot(pa, pb))
    if denom <= 0.0:
        return np.nan
    obs = float((pa[u] * pb[v] + pb[u] * pa[v]).sum())
    return float(obs / (c_const * denom))


def pair_r_stab_weighted(base_labels: np.ndarray, draws: np.ndarray, u: np.ndarray, v: np.ndarray, c_const: float) -> float:
    fl = (draws != base_labels[None, :]).mean(axis=0)
    w = 1.0 - fl
    w[base_labels == N_CLASSES] = 0.0
    wa = np.where(base_labels == 0, w, 0.0)
    wb = np.where(base_labels == 1, w, 0.0)
    denom = float(wa.sum() * wb.sum() - np.dot(wa, wb))
    if denom <= 0.0:
        return np.nan
    obs = float((wa[u] * wb[v] + wb[u] * wa[v]).sum())
    return float(obs / (c_const * denom))


def run_cohort_draws(
    counts_all: np.ndarray,
    core_slices: list[slice],
    graphs: list[tuple[np.ndarray, np.ndarray, float]],
    q: float,
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_cores = len(graphs)
    r_mat = np.full((B, n_cores), np.nan, dtype=np.float64)
    if q >= 1.0 - 1e-12:
        lab, _ = classify_counts_fast(counts_all)
        for ci, (sl, (u, v, cc)) in enumerate(zip(core_slices, graphs)):
            r_mat[:, ci] = pair_r_from_labels(lab[sl], u, v, cc)
    else:
        for b in range(B):
            th = rng.binomial(counts_all, q)
            lab, _ = classify_counts_fast(th)
            for ci, (sl, (u, v, cc)) in enumerate(zip(core_slices, graphs)):
                r_mat[b, ci] = pair_r_from_labels(lab[sl], u, v, cc)
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = np.nanmean(r_mat, axis=0)
        logr = np.where(rm > 0.0, np.log(rm), np.nan)
    return r_mat, logr


def estimate_split_half_q(
    counts_a_all: np.ndarray,
    counts_b_all: np.ndarray,
    pairing: str,
    q0: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    def probe_curve(cnt_mat):
        n = len(cnt_mat)
        if n > 2000:
            idx = rng.choice(n, size=2000, replace=False)
            cnt_mat = cnt_mat[idx]
        tot = cnt_mat.sum(axis=1)
        mask = tot > 0
        psis = []
        for qv in PROBE_GRID:
            th = rng.binomial(cnt_mat, qv) if qv < 1.0 - 1e-12 else cnt_mat
            h1 = rng.binomial(th, 0.5)
            h2 = th - h1
            l1, _ = classify_counts_fast(h1)
            l2, _ = classify_counts_fast(h2)
            psi, _, _ = split_half_disagreement(l1, l2, classifiable_mask=mask)
            psis.append(psi)
        return build_split_half_curve(PROBE_GRID, psis, unresolved_frac=float(1.0 - mask.mean()), enforce_monotone=True)

    ca = probe_curve(counts_a_all)
    cb = probe_curve(counts_b_all)
    res = match_split_half_coverage(ca, cb, q0=q0, pairing=pairing, min_q=0.05)
    return res.q_a, res.q_b


def run_mitm_shallow(
    counts_all: np.ndarray,
    core_slices: list[slice],
    graphs: list[tuple[np.ndarray, np.ndarray, float]],
    s_sh: float,
    d_target: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_cores = len(graphs)
    r_banks, z_banks = [], []
    for qv in LADDER_QS:
        rm, _ = run_cohort_draws(counts_all, core_slices, graphs, qv, B, rng)
        if qv >= 1.0 - 1e-12:
            r_banks.append(rm[0])
            z_banks.append(np.where(np.isfinite(rm[0]), 1.0, np.nan))
        else:
            r_banks.append(rm)
            z_banks.append(np.where(np.isfinite(rm), 1.0, np.nan))
    lad = summarize_thinning_ladder(r_banks, z_banks, LADDER_QS, availability_threshold=0.90)
    pair_ids = np.zeros(n_cores, dtype=np.int64)
    meds = np.array([float(np.median(counts_all[sl].sum(axis=1))) for sl in core_slices])
    gmask = build_extrapolation_guardrail(meds, None, min_median_depth=2.0)
    res = extrapolate_ladder(
        lad,
        depth_scale=s_sh,
        target_depth=d_target,
        pair_ids=pair_ids,
        method="hier_eb",
        scale="inv_sqrtd",
        guardrail_mask=gmask,
        fallback_r=np.nanmean(r_banks[2], axis=0),
    )
    return res.log_r_pred


def one_replicate(task_args: tuple[int, float, float, float, float]) -> list[dict]:
    rep_id, depth_ratio, snr_mismatch, delta_true, mu_base = task_args
    rng = np.random.default_rng([
        20261129,
        int(round(depth_ratio * 1000)),
        int(round(snr_mismatch * 100)),
        int(round(delta_true * 100)),
        int(round(mu_base * 100)) + 1000,
        rep_id,
    ])

    n_cores = 20
    n_cells = CORE_CELLS
    box_um = CORE_BOX_UM
    s_shal = S_DEEP * depth_ratio
    B = 10

    graphs_a, cnt_a_list, slices_a = [], [], []
    graphs_b, cnt_b_list, slices_b = [], [], []
    base_mu_a = mu_base + rng.normal(0.0, 0.16, size=n_cores)
    base_mu_b = (mu_base + delta_true) + rng.normal(0.0, 0.16, size=n_cores)

    naive_a = np.empty(n_cores); naive_b = np.empty(n_cores)
    soft_a = np.empty(n_cores); soft_b = np.empty(n_cores)
    oracle_a = np.empty(n_cores); oracle_b = np.empty(n_cores)
    acc_a_list, acc_b_list = [], []

    off = 0
    for ci in range(n_cores):
        xy, u, v, cc = build_core_spatial_graph(n_cells, box_um, CORE_RADIUS_UM, rng)
        lab_a, het_a, tr_a = simulate_true_labels_with_coloc(xy, u, v, cc, base_mu_a[ci], rng)
        ca = sample_core_counts(lab_a, u, v, het_a, S_DEEP, 1.0, rng)
        la, sa = classify_counts_fast(ca)
        acc_a_list.append(float(np.mean(la == lab_a)))
        graphs_a.append((u, v, cc)); cnt_a_list.append(ca); slices_a.append(slice(off, off + n_cells))
        oracle_a[ci] = tr_a
        ra = pair_r_from_labels(la, u, v, cc)
        naive_a[ci] = math.log(ra) if (np.isfinite(ra) and ra > 0) else np.nan
        rsa = pair_r_soft(sa, u, v, cc, tau=0.05)
        soft_a[ci] = math.log(rsa) if (np.isfinite(rsa) and rsa > 0) else np.nan

        xy2, u2, v2, cc2 = build_core_spatial_graph(n_cells, box_um, CORE_RADIUS_UM, rng)
        lab_b, het_b, tr_b = simulate_true_labels_with_coloc(xy2, u2, v2, cc2, base_mu_b[ci], rng)
        cb = sample_core_counts(lab_b, u2, v2, het_b, s_shal, snr_mismatch, rng)
        lb, sb = classify_counts_fast(cb)
        acc_b_list.append(float(np.mean(lb == lab_b)))
        graphs_b.append((u2, v2, cc2)); cnt_b_list.append(cb); slices_b.append(slice(off, off + n_cells))
        oracle_b[ci] = tr_b
        rb = pair_r_from_labels(lb, u2, v2, cc2)
        naive_b[ci] = math.log(rb) if (np.isfinite(rb) and rb > 0) else np.nan
        rsb = pair_r_soft(sb, u2, v2, cc2, tau=0.05)
        soft_b[ci] = math.log(rsb) if (np.isfinite(rsb) and rsb > 0) else np.nan
        off += n_cells

    all_ca = np.vstack(cnt_a_list)
    all_cb = np.vstack(cnt_b_list)
    ta = all_ca.sum(axis=1); tb = all_cb.sum(axis=1)
    sa_med = float(np.median(ta[ta > 0])) if np.any(ta > 0) else 1.0
    sb_med = float(np.median(tb[tb > 0])) if np.any(tb > 0) else 1.0

    qa_d2s = 0.75 * (min(sa_med, sb_med) / sa_med)
    qb_d2s = 0.75 * (min(sa_med, sb_med) / sb_med)
    qa_sh2s, qb_sh2s = estimate_split_half_q(all_ca, all_cb, "both", 0.75, rng)
    qa_sh1s, qb_sh1s = estimate_split_half_q(all_ca, all_cb, "onesided", 1.00, rng)

    _, dep_a_d2s = run_cohort_draws(all_ca, slices_a, graphs_a, qa_d2s, B, rng)
    _, dep_b_d2s = run_cohort_draws(all_cb, slices_b, graphs_b, qb_d2s, B, rng)
    _, sh2s_a = run_cohort_draws(all_ca, slices_a, graphs_a, qa_sh2s, B, rng)
    _, sh2s_b = run_cohort_draws(all_cb, slices_b, graphs_b, qb_sh2s, B, rng)
    _, sh1s_a = run_cohort_draws(all_ca, slices_a, graphs_a, qa_sh1s, B, rng)
    _, sh1s_b = run_cohort_draws(all_cb, slices_b, graphs_b, qb_sh1s, B, rng)

    d_star = select_mitm_target_depth(sa_med, sb_med, mode="geometric")
    q_dp_mitm = min(1.0, d_star / sa_med)
    _, mitm_a = run_cohort_draws(all_ca, slices_a, graphs_a, q_dp_mitm, B, rng)
    if sb_med < sa_med - 1e-6:
        mitm_b = run_mitm_shallow(all_cb, slices_b, graphs_b, sb_med, d_star, B, rng)
    else:
        _, mitm_b = run_cohort_draws(all_cb, slices_b, graphs_b, min(1.0, d_star / sb_med), B, rng)

    true_delta_obs = float(np.mean(oracle_b) - np.mean(oracle_a))
    arms = {
        "Oracle_True": (oracle_a, oracle_b),
        "Naive": (naive_a, naive_b),
        "Depth_Raref_2s": (dep_a_d2s, dep_b_d2s),
        "SplitHalf_2s_Default": (sh2s_a, sh2s_b),
        "SplitHalf_1s": (sh1s_a, sh1s_b),
        "MITM_Geom_EB": (mitm_a, mitm_b),
        "Soft_Plugin_tau005": (soft_a, soft_b),
    }

    rows = []
    sd_oracle_a = float(np.std(oracle_a, ddof=1))
    mean_acc_a = float(np.mean(acc_a_list))
    mean_acc_b = float(np.mean(acc_b_list))
    for mname, (va, vb) in arms.items():
        ok_a = va[np.isfinite(va)]
        ok_b = vb[np.isfinite(vb)]
        est_delta = float(np.mean(ok_b) - np.mean(ok_a)) if (len(ok_a) >= 4 and len(ok_b) >= 4) else np.nan
        pval = float(ttest_ind(ok_b, ok_a, equal_var=False).pvalue) if (len(ok_a) >= 4 and len(ok_b) >= 4) else np.nan
        sd_ret_a = float(np.std(ok_a, ddof=1) / max(1e-12, sd_oracle_a)) if len(ok_a) >= 3 else np.nan
        rows.append({
            "rep": rep_id,
            "mu_base": mu_base,
            "depth_ratio": depth_ratio,
            "snr_mismatch": snr_mismatch,
            "delta_true_target": delta_true,
            "delta_true_realized": true_delta_obs,
            "acc_deep": mean_acc_a,
            "acc_shallow": mean_acc_b,
            "qa_sh2s": qa_sh2s,
            "qb_sh2s": qb_sh2s,
            "method": mname,
            "est_delta": est_delta,
            "bias": est_delta - true_delta_obs if np.isfinite(est_delta) else np.nan,
            "abs_err": abs(est_delta - true_delta_obs) if np.isfinite(est_delta) else np.nan,
            "p_val": pval,
            "sig_p05": bool(np.isfinite(pval) and pval < 0.05),
            "sign_correct": bool(np.isfinite(est_delta) and ((delta_true == 0.0) or (np.sign(est_delta) == np.sign(true_delta_obs)))),
            "sd_ratio_deep": sd_ret_a,
        })
    return rows


def discard_rep(args: tuple[int, str, float]) -> list[dict]:
    """
    Directly answers: "How much power is lost by discarding deep-side molecules?"
    Evaluates BOTH regimes:
      - Dense_100tx_1000cells: S_DEEP=100, N=1000 cells, box=240um
      - Moderate_45tx_400cells: S_DEEP=45, N=400 cells, box=190um (TMA-like sparse cores)
    Compares Oracle, Native Unthinned (q=1.0), Stability Weighted Unthinned (q=1.0),
    Rarefied_B10(q), and SingleDraw_B1(q) across q in {1.00, 0.85, 0.75, 0.50, 0.25, 0.15, 0.10, 0.05}.
    """
    rep, regime_name, d_true = args
    if regime_name == "Dense_100tx_1000cells":
        n_cores, n_cells, box_um, s_deep, sd_bio = 20, 1000, 240.0, 100.0, 0.12
    else:
        n_cores, n_cells, box_um, s_deep, sd_bio = 20, 400, 190.0, 45.0, 0.12
    rng = np.random.default_rng([20261131, hash(regime_name) & 0xFFFF, int(round(d_true * 100)), rep])
    B = 10
    mu_base = 0.45
    qs = [1.00, 0.85, 0.75, 0.50, 0.25, 0.15, 0.10, 0.05]
    base_mu_a = mu_base + rng.normal(0.0, sd_bio, size=n_cores)
    base_mu_b = (mu_base + d_true) + rng.normal(0.0, sd_bio, size=n_cores)

    ga, ca_list, sa_sl, oa = [], [], [], np.empty(n_cores)
    gb, cb_list, sb_sl, ob = [], [], [], np.empty(n_cores)
    sw_a, sw_b = np.empty(n_cores), np.empty(n_cores)
    nat_a, nat_b = np.empty(n_cores), np.empty(n_cores)
    off = 0
    for ci in range(n_cores):
        xy, u, v, cc = build_core_spatial_graph(n_cells, box_um, CORE_RADIUS_UM, rng)
        la_t, ha, tra = simulate_true_labels_with_coloc(xy, u, v, cc, base_mu_a[ci], rng)
        ca = sample_core_counts(la_t, u, v, ha, s_deep, 1.0, rng)
        la_0, _ = classify_counts_fast(ca)
        ra_0 = pair_r_from_labels(la_0, u, v, cc)
        nat_a[ci] = math.log(ra_0) if ra_0 > 0 else np.nan
        draws_a75 = np.vstack([classify_counts_fast(fast_binomial(ca, 0.75, rng))[0] for _ in range(B)])
        rwa = pair_r_stab_weighted(la_0, draws_a75, u, v, cc)
        sw_a[ci] = math.log(rwa) if rwa > 0 else np.nan
        ga.append((u, v, cc)); ca_list.append(ca); sa_sl.append(slice(off, off + n_cells)); oa[ci] = tra

        xy2, u2, v2, cc2 = build_core_spatial_graph(n_cells, box_um, CORE_RADIUS_UM, rng)
        lb_t, hb, trb = simulate_true_labels_with_coloc(xy2, u2, v2, cc2, base_mu_b[ci], rng)
        cb = sample_core_counts(lb_t, u2, v2, hb, s_deep, 1.0, rng)
        lb_0, _ = classify_counts_fast(cb)
        rb_0 = pair_r_from_labels(lb_0, u2, v2, cc2)
        nat_b[ci] = math.log(rb_0) if rb_0 > 0 else np.nan
        draws_b75 = np.vstack([classify_counts_fast(fast_binomial(cb, 0.75, rng))[0] for _ in range(B)])
        rwb = pair_r_stab_weighted(lb_0, draws_b75, u2, v2, cc2)
        sw_b[ci] = math.log(rwb) if rwb > 0 else np.nan
        gb.append((u2, v2, cc2)); cb_list.append(cb); sb_sl.append(slice(off, off + n_cells)); ob[ci] = trb
        off += n_cells

    all_ca = np.vstack(ca_list)
    all_cb = np.vstack(cb_list)
    realized_d = float(np.mean(ob) - np.mean(oa))
    sd_oa = float(np.std(oa, ddof=1))
    recs = []
    for arm_lbl, va, vb, q_val in [
        ("Oracle", oa, ob, 1.0),
        ("Native_Unthinned", nat_a, nat_b, 1.0),
        ("Stability_Weighted_Unthinned", sw_a, sw_b, 1.0),
    ]:
        oka, okb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        pv = float(ttest_ind(okb, oka, equal_var=False).pvalue) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
        ed = float(np.mean(okb) - np.mean(oka)) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
        recs.append(dict(
            regime=regime_name, delta_true_target=d_true, realized_d=realized_d, method=arm_lbl,
            q_retained=q_val, pct_discarded=100.0 * (1.0 - q_val),
            est_delta=ed, bias=(ed - realized_d) if np.isfinite(ed) else np.nan,
            sig_p05=bool(np.isfinite(pv) and pv < 0.05),
            sd_ratio=float(np.std(oka, ddof=1) / max(1e-12, sd_oa)) if len(oka) >= 3 else np.nan,
        ))

    for qv in qs:
        if qv >= 1.0 - 1e-12:
            va_b10, vb_b10 = nat_a, nat_b
            va_b1, vb_b1 = nat_a, nat_b
        else:
            rm_a, va_b10 = run_cohort_draws(all_ca, sa_sl, ga, qv, B, rng)
            rm_b, vb_b10 = run_cohort_draws(all_cb, sb_sl, gb, qv, B, rng)
            with np.errstate(divide="ignore", invalid="ignore"):
                va_b1 = np.where(rm_a[0] > 0.0, np.log(rm_a[0]), np.nan)
                vb_b1 = np.where(rm_b[0] > 0.0, np.log(rm_b[0]), np.nan)

        for tag_m, va, vb in [
            (f"Rarefied_B10_q{int(round(qv*100)):03d}", va_b10, vb_b10),
            (f"SingleDraw_B1_q{int(round(qv*100)):03d}", va_b1, vb_b1),
        ]:
            oka, okb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            pv = float(ttest_ind(okb, oka, equal_var=False).pvalue) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
            ed = float(np.mean(okb) - np.mean(oka)) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
            recs.append(dict(
                regime=regime_name, delta_true_target=d_true, realized_d=realized_d, method=tag_m,
                q_retained=qv, pct_discarded=100.0 * (1.0 - qv),
                est_delta=ed, bias=(ed - realized_d) if np.isfinite(ed) else np.nan,
                sig_p05=bool(np.isfinite(pv) and pv < 0.05),
                sd_ratio=float(np.std(oka, ddof=1) / max(1e-12, sd_oa)) if len(oka) >= 3 else np.nan,
            ))
    return recs


def run_calibration_check():
    t0 = time.time()
    print("=== Calibration Pass (S_DEEP=100, N=1000 cells/core) ===", flush=True)
    c_tasks = [(r, 0.25, 2.0, 0.00, 0.45) for r in range(24)]
    c_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(one_replicate, c_tasks, chunksize=4):
            c_rows.extend(res)
    dfc = pd.DataFrame(c_rows)
    print(f"Null (delta=0, mu_base=0.45, r_d=0.25, SNR=2.0, 24 reps) in {time.time()-t0:.1f}s:")
    print(f"  Accuracy deep={dfc['acc_deep'].mean():.3f}, shallow={dfc['acc_shallow'].mean():.3f}, qa_sh2s={dfc['qa_sh2s'].mean():.3f}")
    print(dfc.groupby("method")[["bias", "abs_err", "sig_p05", "sd_ratio_deep"]].mean().to_string())

    d_tasks = [(r, 0.15) for r in range(48)]
    d_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(discard_rep, d_tasks, chunksize=4):
            d_rows.extend(res)
    dfd = pd.DataFrame(d_rows)
    print("\nDiscard power curve check (delta=0.15, 48 reps):")
    print(dfd.groupby(["method", "pct_discarded"])[["est_delta", "bias", "sig_p05", "sd_ratio"]].mean().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()

    if args.calibrate_only:
        run_calibration_check()
        return

    t0 = time.time()
    print("=== Step 6 (Calibrated v3): Controlled Ground-Truth Simulation Benchmark ===", flush=True)
    p0 = one_replicate((0, 0.25, 2.0, 0.30, 0.45))
    print(f"Single replicate finished in {time.time()-t0:.2f}s (acc_deep={p0[0]['acc_deep']:.3f}, acc_shal={p0[0]['acc_shallow']:.3f})", flush=True)

    n_reps = 60
    tasks = [
        (rep, dr, snr, d, 0.45)
        for dr in (1.0, 0.50, 0.25, 0.15)
        for snr in (1.0, 2.0, 3.0)
        for d in (0.00, 0.15, 0.30, 0.45)
        for rep in range(n_reps)
    ]
    print(f"Total primary factorial cohorts (mu_base=+0.45): {len(tasks)}", flush=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for idx, res in enumerate(ex.map(one_replicate, tasks, chunksize=12), 1):
            all_rows.extend(res)
            if idx % 288 == 0 or idx == len(tasks):
                print(f"  Progress: {idx}/{len(tasks)} ({time.time()-t0:.1f}s)", flush=True)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_parquet(OUT_DIR / "simulation_factorial_raw.parquet", index=False)

    summary = (
        df_raw.groupby(["depth_ratio", "snr_mismatch", "delta_true_target", "method"], sort=False)
        .agg(
            n_reps=("rep", "count"),
            acc_deep=("acc_deep", "mean"),
            acc_shallow=("acc_shallow", "mean"),
            mean_qa_sh2s=("qa_sh2s", "mean"),
            mean_true_delta=("delta_true_realized", "mean"),
            mean_est_delta=("est_delta", "mean"),
            bias=("bias", "mean"),
            mae=("abs_err", "mean"),
            rmse=("bias", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            reject_rate_p05=("sig_p05", "mean"),
            sign_accuracy=("sign_correct", "mean"),
            sd_ratio_deep=("sd_ratio_deep", "mean"),
        )
        .reset_index()
    )
    oracle_map = summary[summary["method"] == "Oracle_True"].set_index(["depth_ratio", "snr_mismatch", "delta_true_target"])["reject_rate_p05"].to_dict()
    summary["power_loss_vs_oracle"] = summary.apply(
        lambda r: oracle_map[(r["depth_ratio"], r["snr_mismatch"], r["delta_true_target"])] - r["reject_rate_p05"]
        if r["delta_true_target"] > 0 else np.nan,
        axis=1,
    )
    summary.to_csv(OUT_DIR / "simulation_factorial_summary.csv", index=False)

    print("\n=== Running Segregation Regime Check (mu_base = -0.45) ===", flush=True)
    seg_tasks = [
        (rep, 0.25, snr, d, -0.45)
        for snr in (1.0, 2.0)
        for d in (0.00, 0.30)
        for rep in range(60)
    ]
    seg_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(one_replicate, seg_tasks, chunksize=12):
            seg_rows.extend(res)
    df_seg = pd.DataFrame(seg_rows)
    seg_sum = (
        df_seg.groupby(["mu_base", "depth_ratio", "snr_mismatch", "delta_true_target", "method"], sort=False)
        .agg(
            n_reps=("rep", "count"),
            mean_true_delta=("delta_true_realized", "mean"),
            mean_est_delta=("est_delta", "mean"),
            bias=("bias", "mean"),
            mae=("abs_err", "mean"),
            reject_rate_p05=("sig_p05", "mean"),
            sd_ratio_deep=("sd_ratio_deep", "mean"),
        )
        .reset_index()
    )
    seg_sum.to_csv(OUT_DIR / "simulation_segregation_check.csv", index=False)

    print("\n=== Running Two-Regime Deep-Molecule-Discard Power Curve (q = 1.00 .. 0.05, 160 reps) ===", flush=True)
    d_tasks = [
        (rep, reg, d)
        for reg in ("Dense_100tx_1000cells", "Moderate_45tx_400cells")
        for d in (0.00, 0.15, 0.25)
        for rep in range(160)
    ]
    d_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(discard_rep, d_tasks, chunksize=12):
            d_rows.extend(res)
    df_pow = pd.DataFrame(d_rows)
    df_pow_sum = (
        df_pow.groupby(["regime", "delta_true_target", "method", "q_retained", "pct_discarded"], sort=False)
        .agg(
            n_reps=("sig_p05", "count"),
            reject_rate_p05=("sig_p05", "mean"),
            mean_est_delta=("est_delta", "mean"),
            bias=("bias", "mean"),
            mae=("bias", lambda s: float(np.mean(np.abs(s)))),
            sd_ratio=("sd_ratio", "mean"),
        )
        .reset_index()
    )
    or_p = df_pow_sum[df_pow_sum["method"] == "Oracle"].set_index(["regime", "delta_true_target"])["reject_rate_p05"].to_dict()
    df_pow_sum["power_loss_vs_oracle"] = df_pow_sum.apply(
        lambda r: or_p[(r["regime"], r["delta_true_target"])] - r["reject_rate_p05"] if r["delta_true_target"] > 0 else np.nan,
        axis=1,
    )
    df_pow_sum.to_csv(OUT_DIR / "simulation_discard_power_curve.csv", index=False)

    (OUT_DIR / "simulation_meta.json").write_text(json.dumps({
        "S_DEEP": S_DEEP,
        "N_GENES": N_GENES,
        "CORE_CELLS": CORE_CELLS,
        "mu_base": 0.45,
        "seg_mu_base": -0.45,
        "n_factorial_reps_per_cell": n_reps,
        "total_factorial_cohorts": len(tasks),
        "n_discard_curve_reps_per_cell": 160,
        "elapsed_sec": round(time.time() - t0, 2),
    }, indent=2))
    print(f"Completed Step 6 Simulation Benchmark in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
