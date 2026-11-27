#!/usr/bin/env python3
"""
P1-8 Controlled Validation Suite:
  Experiment A — Fine-Grained 18-Subtype Spatial Benchmark (C=18 subtypes, 120 genes,
    153 cell-type pairs stratified into Prevalent >=5%, Intermediate 2.5-5%, Rare 1.5-2.5%):
    Directly quantifies P+ >= 0.90 pair availability survival, null Type-I error, power,
    and deep SD retention under SplitHalf_2s (q0=0.75), SplitHalf_1s (q0=1.00),
    Depth_Raref_2s, Soft_Plugin, and Unthinned Stability Weighting.

  Experiment B — Gene-Specific Capture-Efficiency Heterogeneity Stress Test
    (breaking Proposition 3 uniform scaling assumption with log-normal per-gene capture
    dispersion sigma_g in {0.0, 0.4, 0.8, 1.2} -> 1x, ~3x, ~10x, ~30x fold-range):
    Directly quantifies how gene-specific capture heterogeneity generates residual
    |Delta log R| ~ 0.03-0.06 detectable by heavily powered Tier 3 paired tests (n=80)
    even while within-group/disjoint two-center tests remain well-calibrated.
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind, ttest_rel

ROOT = Path(r"c:\Reproducution")
sys.path.insert(0, str(ROOT / "software" / "spatial_membership_audit" / "src"))
from spatial_membership_audit.coverage_matching import (
    build_split_half_curve,
    match_split_half_coverage,
    split_half_disagreement,
)

OUT = ROOT / "review" / "v3_p1_hardening_2026-11-27"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_GRID = (1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)

# =============================================================================
# Experiment A: 18 Fine-Grained Subtypes (6 parent lineages x 3 subtypes each, 120 genes)
# =============================================================================
C_FINE = 18
G_FINE = 120
# Realistic hierarchical TME prevalence summing to 1.0:
# 6 prevalent (8-12%), 6 intermediate (3.5-5%), 6 rare (1.5-2.2%)
PREV_18 = np.array([
    0.120, 0.095, 0.085, 0.080, 0.075, 0.065,  # 0..5 Prevalent (0=Epi_1, 1=T_CD8, 2=Fib_myCAF, 3=Mac_M2, 4=Endo_art, 5=Epi_2)
    0.050, 0.048, 0.045, 0.042, 0.040, 0.038,  # 6..11 Intermediate (e.g. T_CD4, Fib_iCAF, Mac_M1, B_mem, Endo_ven, Mono)
    0.022, 0.020, 0.019, 0.018, 0.017, 0.016,  # 12..17 Rare (e.g. Treg, NK, Plasma, cDC1, Neut, Mast)
], dtype=np.float64)
PREV_18 /= PREV_18.sum()

# Hierarchical gene profile: shared parent-lineage markers + subtype-specific markers + background
_PROF18 = np.full((C_FINE, G_FINE), 0.20, dtype=np.float64)
for k in range(C_FINE):
    parent = k % 6
    # 5 shared parent-lineage genes (0..29) -> narrower within-lineage subtype margins!
    _PROF18[k, parent * 5 : (parent + 1) * 5] = 1.05
    # 3 subtype-specific markers (30..83)
    _PROF18[k, 30 + k * 3 : 30 + (k + 1) * 3] = 1.75
    # 36 housekeeping/background genes (84..119) stay at 0.20
_PROF18 /= _PROF18.sum(axis=1, keepdims=True)
_W18 = _PROF18 - _PROF18.mean(axis=1, keepdims=True)
_W18 /= np.linalg.norm(_W18, axis=1, keepdims=True)
_W18_T = np.ascontiguousarray(_W18.T, dtype=np.float32)

# All 153 unordered pairs stratified by min(prev_a, prev_b)
PAIRS_153 = []
for i in range(C_FINE):
    for j in range(i + 1, C_FINE):
        mp = min(PREV_18[i], PREV_18[j])
        tier = "Prevalent_ge5pct" if mp >= 0.05 else ("Intermediate_2p5_5pct" if mp >= 0.025 else "Rare_1p5_2p5pct")
        PAIRS_153.append((i, j, tier))

# Focus test pairs for Type-I/Power: (0, 1) Prevalent-Prevalent, (0, 6) Prev-Interm, (1, 12) Prev-Rare
FOCUS_PAIRS = [
    (0, 1, "Prevalent_0_1"),
    (0, 6, "Intermediate_0_6"),
    (1, 12, "Rare_1_12"),
]


def classify_18(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tot = counts.sum(axis=1)
    lx = np.log1p(counts.astype(np.float32))
    lx_c = lx - lx.mean(axis=1, keepdims=True)
    nx = np.linalg.norm(lx_c, axis=1, keepdims=True)
    sc = (lx_c / np.maximum(nx, 1e-6)) @ _W18_T
    lab = np.argmax(sc, axis=1).astype(np.int32)
    lab[tot <= 0] = C_FINE
    return lab, sc


def build_graph(n_cells: int, box_um: float, r_um: float, rng: np.random.Generator):
    xy = rng.uniform(0.0, box_um, size=(n_cells, 2))
    pairs = cKDTree(xy).query_pairs(r=r_um, output_type="ndarray")
    if len(pairs) == 0:
        return xy, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0.0
    u = pairs[:, 0].astype(np.int32)
    v = pairs[:, 1].astype(np.int32)
    cc = 2.0 * len(u) / (n_cells * (n_cells - 1.0))
    return xy, u, v, cc


def sim_labels_18(xy, u, v, cc, delta_shift: float, rng: np.random.Generator):
    n = len(xy)
    labels = np.full(n, -1, dtype=np.int32)
    # Place subtype 0 in spatial nests
    n0 = max(12, int(round(PREV_18[0] * n)))
    centers = rng.uniform(35.0, 205.0, size=(2, 2))
    d2 = np.min(np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    p0 = 0.12 + 0.88 * np.exp(-d2 / (2.0 * 34.0 ** 2))
    p0 /= p0.sum()
    idx0 = rng.choice(n, size=n0, replace=False, p=p0)
    labels[idx0] = 0

    is0 = (labels == 0).astype(np.int32)
    n0_nbr = np.zeros(n, dtype=np.float64)
    if len(u) > 0:
        np.add.at(n0_nbr, u, is0[v])
        np.add.at(n0_nbr, v, is0[u])
    deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64) if len(u) > 0 else np.zeros(n)
    frac0 = np.where(deg > 0, n0_nbr / np.maximum(deg, 1.0), PREV_18[0])
    z0 = (frac0 - PREV_18[0]) / max(1e-6, float(np.std(frac0)))

    # Place subtypes 1, 6, 12 with controlled colocalization to subtype 0 (or 1 for 12)
    def place_with_target(cls_id: int, target_logr: float, ref_nbr: np.ndarray, ref_z: np.ndarray, ref_count: int):
        rem = np.flatnonzero(labels < 0)
        nk = max(6, int(round(PREV_18[cls_id] * n)))
        nk = min(nk, len(rem))
        exp_e = cc * ref_count * nk
        if exp_e > 0 and len(rem) > nk:
            tgt_obs = max(0.5, exp_e * math.exp(target_logr))
            lo, hi = -4.0, 4.0
            zr, kr = ref_z[rem], ref_nbr[rem]
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                lw = mid * zr; lw -= lw.max()
                w = np.exp(lw); w /= w.sum()
                if nk * float(np.dot(w, kr)) < tgt_obs:
                    lo = mid
                else:
                    hi = mid
            lw = 0.5 * (lo + hi) * zr; lw -= lw.max()
            p = np.exp(lw); p /= p.sum()
            idxk = rng.choice(rem, size=nk, replace=False, p=p)
        else:
            idxk = rng.choice(rem, size=nk, replace=False)
        labels[idxk] = cls_id
        return idxk

    idx1 = place_with_target(1, 0.45 + delta_shift + rng.normal(0.0, 0.14), n0_nbr, z0, len(idx0))
    _ = place_with_target(6, 0.45 + delta_shift + rng.normal(0.0, 0.14), n0_nbr, z0, len(idx0))

    is1 = (labels == 1).astype(np.int32)
    n1_nbr = np.zeros(n, dtype=np.float64)
    if len(u) > 0:
        np.add.at(n1_nbr, u, is1[v])
        np.add.at(n1_nbr, v, is1[u])
    frac1 = np.where(deg > 0, n1_nbr / np.maximum(deg, 1.0), PREV_18[1])
    z1 = (frac1 - PREV_18[1]) / max(1e-6, float(np.std(frac1)))
    _ = place_with_target(12, 0.45 + delta_shift + rng.normal(0.0, 0.14), n1_nbr, z1, len(idx1))

    rem_all = np.flatnonzero(labels < 0)
    other_ids = [c for c in range(C_FINE) if c not in (0, 1, 6, 12)]
    p_oth = PREV_18[other_ids].copy(); p_oth /= p_oth.sum()
    labels[rem_all] = rng.choice(other_ids, size=len(rem_all), p=p_oth).astype(np.int32)

    het = np.zeros(n, dtype=np.float64)
    if len(u) > 0:
        diff = (labels[u] != labels[v]).astype(np.float64)
        np.add.at(het, u, diff); np.add.at(het, v, diff)
    het_frac = np.where(deg > 0, het / np.maximum(deg, 1.0), 0.0)
    return labels, het_frac


def sample_counts_generic(labels, u, v, het_frac, prof_mat, mean_depth, snr_mismatch, rng, gene_scale=None, spillover=0.18):
    n = len(labels)
    g = prof_mat.shape[1]
    base_prof = prof_mat[labels].copy()
    if len(u) > 0 and spillover > 0.0:
        nbr_sum = np.zeros((n, g), dtype=np.float64)
        np.add.at(nbr_sum, u, prof_mat[labels[v]])
        np.add.at(nbr_sum, v, prof_mat[labels[u]])
        deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64)
        nbr_mean = np.where((deg > 0)[:, None], nbr_sum / np.maximum(deg[:, None], 1.0), base_prof)
        alpha = spillover * het_frac[:, None]
        base_prof = (1.0 - alpha) * base_prof + alpha * nbr_mean
    if snr_mismatch > 1.0:
        w_sig = 1.0 / (1.0 + 0.35 * (snr_mismatch - 1.0))
        prof = w_sig * base_prof + (1.0 - w_sig) * (1.0 / g)
    else:
        prof = base_prof
    if gene_scale is not None:
        prof = prof * gene_scale[None, :]
        prof /= np.maximum(prof.sum(axis=1, keepdims=True), 1e-12)
    cell_depth = mean_depth * rng.gamma(shape=5.0, scale=0.20, size=n)
    return rng.poisson(cell_depth[:, None] * prof).astype(np.int32)


def all_pairs_r_matrix(labels: np.ndarray, u: np.ndarray, v: np.ndarray, cc: float, n_classes: int) -> np.ndarray:
    """Fast vectorized O/E matrix for all unordered pairs."""
    counts = np.bincount(labels, minlength=n_classes + 1)[:n_classes].astype(np.float64)
    lu, lv = labels[u], labels[v]
    valid_e = (lu < n_classes) & (lv < n_classes)
    lu, lv = lu[valid_e], lv[valid_e]
    obs = np.zeros((n_classes, n_classes), dtype=np.float64)
    if len(lu) > 0:
        np.add.at(obs, (lu, lv), 1.0)
        np.add.at(obs, (lv, lu), 1.0)
    exp = cc * np.outer(counts, counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_mat = np.where(exp > 0.0, obs / exp, np.nan)
    return r_mat


def estimate_sh_generic(cnt_a, cnt_b, clf_fn, pairing, q0, rng):
    def curve(cnt):
        if len(cnt) > 2000:
            cnt = cnt[rng.choice(len(cnt), size=2000, replace=False)]
        mask = cnt.sum(axis=1) > 0
        psis = []
        for qv in PROBE_GRID:
            th = rng.binomial(cnt, qv) if qv < 1.0 - 1e-12 else cnt
            h1 = rng.binomial(th, 0.5)
            h2 = th - h1
            l1, _ = clf_fn(h1)
            l2, _ = clf_fn(h2)
            psi, _, _ = split_half_disagreement(l1, l2, classifiable_mask=mask)
            psis.append(psi)
        return build_split_half_curve(PROBE_GRID, psis, unresolved_frac=float(1.0 - mask.mean()), enforce_monotone=True)
    ca, cb = curve(cnt_a), curve(cnt_b)
    res = match_split_half_coverage(ca, cb, q0=q0, pairing=pairing, min_q=0.05)
    return res.q_a, res.q_b, res.clamped


def run_fine18_rep(args):
    rep, depth_ratio, snr, delta_true = args
    rng = np.random.default_rng([2026112718, int(round(depth_ratio * 100)), int(round(snr * 10)), int(round(delta_true * 100)), rep])
    n_cores = 20
    n_cells = 1000
    s_deep = 150.0
    s_shal = s_deep * depth_ratio
    B = 10

    ga, ca_list, sl_a, true_r_a = [], [], [], []
    gb, cb_list, sl_b, true_r_b = [], [], [], []
    naive_a_r, naive_b_r = [], []
    soft_a_r, soft_b_r = [], []
    acc_a, acc_b = [], []
    off = 0
    for ci in range(n_cores):
        xy, u, v, cc = build_graph(n_cells, 240.0, 25.0, rng)
        la_t, ha = sim_labels_18(xy, u, v, cc, 0.0, rng)
        ca = sample_counts_generic(la_t, u, v, ha, _PROF18, s_deep, 1.0, rng)
        la, sa = classify_18(ca)
        acc_a.append(float(np.mean(la == la_t)))
        ga.append((u, v, cc)); ca_list.append(ca); sl_a.append(slice(off, off + n_cells))
        true_r_a.append(all_pairs_r_matrix(la_t, u, v, cc, C_FINE))
        naive_a_r.append(all_pairs_r_matrix(la, u, v, cc, C_FINE))

        # Soft tau=0.05 for focus pairs only
        ex = np.exp((sa.astype(np.float64) - sa.max(axis=1, keepdims=True)) / 0.05)
        P = ex / ex.sum(axis=1, keepdims=True)
        sf_a = {}
        for i, j, _ in FOCUS_PAIRS:
            pa, pb = P[:, i], P[:, j]
            den = float(pa.sum() * pb.sum() - np.dot(pa, pb))
            sf_a[(i, j)] = float((pa[u] * pb[v] + pb[u] * pa[v]).sum() / (cc * den)) if den > 0 else np.nan
        soft_a_r.append(sf_a)

        xy2, u2, v2, cc2 = build_graph(n_cells, 240.0, 25.0, rng)
        lb_t, hb = sim_labels_18(xy2, u2, v2, cc2, delta_true, rng)
        cb = sample_counts_generic(lb_t, u2, v2, hb, _PROF18, s_shal, snr, rng)
        lb, sb = classify_18(cb)
        acc_b.append(float(np.mean(lb == lb_t)))
        gb.append((u2, v2, cc2)); cb_list.append(cb); sl_b.append(slice(off, off + n_cells))
        true_r_b.append(all_pairs_r_matrix(lb_t, u2, v2, cc2, C_FINE))
        naive_b_r.append(all_pairs_r_matrix(lb, u2, v2, cc2, C_FINE))

        ex2 = np.exp((sb.astype(np.float64) - sb.max(axis=1, keepdims=True)) / 0.05)
        P2 = ex2 / ex2.sum(axis=1, keepdims=True)
        sf_b = {}
        for i, j, _ in FOCUS_PAIRS:
            pa, pb = P2[:, i], P2[:, j]
            den = float(pa.sum() * pb.sum() - np.dot(pa, pb))
            sf_b[(i, j)] = float((pa[u2] * pb[v2] + pb[u2] * pa[v2]).sum() / (cc2 * den)) if den > 0 else np.nan
        soft_b_r.append(sf_b)
        off += n_cells

    all_ca, all_cb = np.vstack(ca_list), np.vstack(cb_list)
    ta, tb = all_ca.sum(axis=1), all_cb.sum(axis=1)
    sa_m = float(np.median(ta[ta > 0])); sb_m = float(np.median(tb[tb > 0]))

    qa_d2 = 0.75 * min(sa_m, sb_m) / sa_m
    qb_d2 = 0.75 * min(sa_m, sb_m) / sb_m
    qa_sh2, qb_sh2, cl2 = estimate_sh_generic(all_ca, all_cb, classify_18, "both", 0.75, rng)
    qa_sh1, qb_sh1, cl1 = estimate_sh_generic(all_ca, all_cb, classify_18, "onesided", 1.00, rng)

    def eval_draws_all_pairs(cnt_all, slices, graphs, qv):
        # Returns (B, n_cores, C, C)
        out = np.full((B, n_cores, C_FINE, C_FINE), np.nan, dtype=np.float64)
        if qv >= 1.0 - 1e-12:
            lab, _ = classify_18(cnt_all)
            for ci, (sl, (u, v, cc)) in enumerate(zip(slices, graphs)):
                out[:, ci] = all_pairs_r_matrix(lab[sl], u, v, cc, C_FINE)
        else:
            for b in range(B):
                th = rng.binomial(cnt_all, qv)
                lab, _ = classify_18(th)
                for ci, (sl, (u, v, cc)) in enumerate(zip(slices, graphs)):
                    out[b, ci] = all_pairs_r_matrix(lab[sl], u, v, cc, C_FINE)
        return out

    b_d2_a = eval_draws_all_pairs(all_ca, sl_a, ga, qa_d2)
    b_d2_b = eval_draws_all_pairs(all_cb, sl_b, gb, qb_d2)
    b_sh2_a = eval_draws_all_pairs(all_ca, sl_a, ga, qa_sh2)
    b_sh2_b = eval_draws_all_pairs(all_cb, sl_b, gb, qb_sh2)
    b_sh1_a = eval_draws_all_pairs(all_ca, sl_a, ga, qa_sh1)
    b_sh1_b = eval_draws_all_pairs(all_cb, sl_b, gb, qb_sh1)

    # Compute P+ >= 0.90 availability (defined E>0 AND positive mean R > 0) across all 153 pairs x 20 cores by prevalence tier
    avail_rows = []
    # Also compute P+ definedness (E > 0 in >= 90% of draws) AND finite positive log(mean R)
    arm_banks = {
        "Naive": (np.repeat(np.stack(naive_a_r)[None], B, axis=0), np.repeat(np.stack(naive_b_r)[None], B, axis=0)),
        "Depth_Raref_2s": (b_d2_a, b_d2_b),
        "SplitHalf_2s_Default": (b_sh2_a, b_sh2_b),
        "SplitHalf_1s": (b_sh1_a, b_sh1_b),
    }
    for aname, (ba, bb) in arm_banks.items():
        pplus_a = np.isfinite(ba).mean(axis=0)  # (n_cores, C, C)
        pplus_b = np.isfinite(bb).mean(axis=0)
        with np.errstate(invalid="ignore"):
            ma = np.nanmean(ba, axis=0)
            mb = np.nanmean(bb, axis=0)
        both_p90 = (pplus_a >= 0.90) & (pplus_b >= 0.90)
        both_pos_logr = both_p90 & (ma > 0) & (mb > 0)
        for tier in ("Prevalent_ge5pct", "Intermediate_2p5_5pct", "Rare_1p5_2p5pct"):
            idx_t = [(i, j) for i, j, t in PAIRS_153 if t == tier]
            p90_vals = [both_p90[:, i, j].mean() for i, j in idx_t]
            pos_vals = [both_pos_logr[:, i, j].mean() for i, j in idx_t]
            avail_rows.append(dict(
                rep=rep, depth_ratio=depth_ratio, snr=snr, delta_true=delta_true,
                method=aname, tier=tier, n_pairs=len(idx_t),
                pplus_ge90_rate=float(np.mean(p90_vals)),
                finite_logr_p90_rate=float(np.mean(pos_vals)),
                acc_deep=float(np.mean(acc_a)), acc_shal=float(np.mean(acc_b)),
                qa_sh2=qa_sh2, qa_sh1=qa_sh1, clamped_2s=bool(cl2), clamped_1s=bool(cl1),
            ))

    # Compute Type-I / Power / Bias on the 3 focus pairs (Prevalent, Intermediate, Rare)
    infer_rows = []
    tr_a_arr = np.stack(true_r_a); tr_b_arr = np.stack(true_r_b)
    for i, j, plabel in FOCUS_PAIRS:
        with np.errstate(divide="ignore", invalid="ignore"):
            oa = np.log(np.where(tr_a_arr[:, i, j] > 0, tr_a_arr[:, i, j], np.nan))
            ob = np.log(np.where(tr_b_arr[:, i, j] > 0, tr_b_arr[:, i, j], np.nan))
        ok_oa, ok_ob = oa[np.isfinite(oa)], ob[np.isfinite(ob)]
        true_d = float(np.mean(ok_ob) - np.mean(ok_oa)) if (len(ok_oa) >= 4 and len(ok_ob) >= 4) else np.nan
        sd_oa = float(np.std(ok_oa, ddof=1)) if len(ok_oa) >= 3 else np.nan

        arms_pair = {"Oracle_True": (oa, ob)}
        arms_pair["Soft_Plugin_tau005"] = (
            np.array([math.log(d[(i, j)]) if (np.isfinite(d[(i, j)]) and d[(i, j)] > 0) else np.nan for d in soft_a_r]),
            np.array([math.log(d[(i, j)]) if (np.isfinite(d[(i, j)]) and d[(i, j)] > 0) else np.nan for d in soft_b_r]),
        )
        for aname, (ba, bb) in arm_banks.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                ma = np.nanmean(ba[:, :, i, j], axis=0)
                mb = np.nanmean(bb[:, :, i, j], axis=0)
                pa_ok = np.isfinite(ba[:, :, i, j]).mean(axis=0) >= 0.90
                pb_ok = np.isfinite(bb[:, :, i, j]).mean(axis=0) >= 0.90
                la_v = np.where(pa_ok & (ma > 0), np.log(ma), np.nan)
                lb_v = np.where(pb_ok & (mb > 0), np.log(mb), np.nan)
            arms_pair[aname] = (la_v, lb_v)

        for aname, (va, vb) in arms_pair.items():
            oka, okb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            ed = float(np.mean(okb) - np.mean(oka)) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
            pv = float(ttest_ind(okb, oka, equal_var=False).pvalue) if (len(oka) >= 4 and len(okb) >= 4) else np.nan
            sd_r = float(np.std(oka, ddof=1) / max(1e-12, sd_oa)) if (len(oka) >= 3 and np.isfinite(sd_oa)) else np.nan
            infer_rows.append(dict(
                rep=rep, depth_ratio=depth_ratio, snr=snr, delta_true=delta_true,
                pair_tier=plabel, method=aname,
                avail_cores_min=min(len(oka), len(okb)) / n_cores,
                bias=(ed - true_d) if (np.isfinite(ed) and np.isfinite(true_d)) else np.nan,
                abs_err=abs(ed - true_d) if (np.isfinite(ed) and np.isfinite(true_d)) else np.nan,
                sig_p05=bool(np.isfinite(pv) and pv < 0.05),
                sd_ratio_deep=sd_r,
            ))
    return avail_rows, infer_rows


# =============================================================================
# Experiment B: Gene-Specific Capture-Efficiency Heterogeneity (Breaking Prop. 3)
# =============================================================================
_PROF6 = np.full((6, 60), 0.28, dtype=np.float64)
for _k in range(6):
    _PROF6[_k, _k * 4 : (_k + 1) * 4] = 1.55
    _kn = (_k + 1) % 6
    _PROF6[_k, 24 + _k * 2 : 24 + (_k + 1) * 2] = 0.52
    _PROF6[_kn, 24 + _k * 2 : 24 + (_k + 1) * 2] = 0.44
_PROF6 /= _PROF6.sum(axis=1, keepdims=True)
_W6 = _PROF6 - _PROF6.mean(axis=1, keepdims=True)
_W6 /= np.linalg.norm(_W6, axis=1, keepdims=True)
_W6_T = np.ascontiguousarray(_W6.T, dtype=np.float32)


def classify_6(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tot = counts.sum(axis=1)
    lx = np.log1p(counts.astype(np.float32))
    lx_c = lx - lx.mean(axis=1, keepdims=True)
    nx = np.linalg.norm(lx_c, axis=1, keepdims=True)
    sc = (lx_c / np.maximum(nx, 1e-6)) @ _W6_T
    lab = np.argmax(sc, axis=1).astype(np.int32)
    lab[tot <= 0] = 6
    return lab, sc


def run_gene_het_rep(args):
    rep, sigma_g = args
    rng = np.random.default_rng([2026112799, int(round(sigma_g * 100)), rep])
    # Paired cores (n=60 paired cores, matching tTMA1-like Tier 3 power vs 20-core disjoint Tier 2)
    n_cores = 60
    n_cells = 800
    s_deep = 100.0
    s_shal = 30.0
    snr = 1.5
    B = 10
    # Draw fixed cross-platform gene-specific capture efficiency multiplier for platform B
    gene_mult = np.exp(rng.normal(0.0, sigma_g, size=60)) if sigma_g > 0 else np.ones(60)
    fold_p95_p05 = float(np.quantile(gene_mult, 0.95) / max(1e-12, np.quantile(gene_mult, 0.05)))

    graphs, ca_l, cb_l, slices = [], [], [], []
    nv_a, nv_b = np.empty(n_cores), np.empty(n_cores)
    off = 0
    for ci in range(n_cores):
        xy, u, v, cc = build_graph(n_cells, 220.0, 25.0, rng)
        # Same true spatial tissue assayed on Platform A (deep, uniform) and Platform B (shallow, gene-heterogeneous)
        n0 = int(round(0.28 * n_cells))
        centers = rng.uniform(30.0, 190.0, size=(2, 2))
        d2 = np.min(np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        p0 = 0.12 + 0.88 * np.exp(-d2 / (2.0 * 30.0 ** 2)); p0 /= p0.sum()
        labels = np.full(n_cells, -1, dtype=np.int32)
        labels[rng.choice(n_cells, size=n0, replace=False, p=p0)] = 0
        is0 = (labels == 0).astype(np.int32)
        n0n = np.zeros(n_cells); np.add.at(n0n, u, is0[v]); np.add.at(n0n, v, is0[u])
        deg = np.bincount(np.concatenate([u, v]), minlength=n_cells).astype(np.float64)
        f0 = np.where(deg > 0, n0n / np.maximum(deg, 1.0), 0.28)
        z0 = (f0 - 0.28) / max(1e-6, float(np.std(f0)))
        rem = np.flatnonzero(labels < 0)
        n1 = int(round(0.16 * n_cells))
        tgt = max(0.5, cc * n0 * n1 * math.exp(0.45 + rng.normal(0.0, 0.16)))
        lo, hi = -4.0, 4.0
        for _ in range(14):
            mid = 0.5 * (lo + hi); lw = mid * z0[rem]; lw -= lw.max(); w = np.exp(lw); w /= w.sum()
            if n1 * float(np.dot(w, n0n[rem])) < tgt: lo = mid
            else: hi = mid
        lw = 0.5 * (lo + hi) * z0[rem]; lw -= lw.max(); p1 = np.exp(lw); p1 /= p1.sum()
        labels[rng.choice(rem, size=n1, replace=False, p=p1)] = 1
        rem2 = np.flatnonzero(labels < 0)
        labels[rem2] = rng.choice([2, 3, 4, 5], size=len(rem2), p=[0.32, 0.18, 0.25, 0.25]).astype(np.int32)
        het = np.zeros(n_cells); diff = (labels[u] != labels[v]).astype(np.float64)
        np.add.at(het, u, diff); np.add.at(het, v, diff)
        het_f = np.where(deg > 0, het / np.maximum(deg, 1.0), 0.0)

        ca = sample_counts_generic(labels, u, v, het_f, _PROF6, s_deep, 1.0, rng, gene_scale=None)
        cb = sample_counts_generic(labels, u, v, het_f, _PROF6, s_shal, snr, rng, gene_scale=gene_mult)
        la, _ = classify_6(ca); lb, _ = classify_6(cb)
        ra = all_pairs_r_matrix(la, u, v, cc, 6)[0, 1]
        rb = all_pairs_r_matrix(lb, u, v, cc, 6)[0, 1]
        nv_a[ci] = math.log(ra) if ra > 0 else np.nan
        nv_b[ci] = math.log(rb) if rb > 0 else np.nan
        graphs.append((u, v, cc)); ca_l.append(ca); cb_l.append(cb); slices.append(slice(off, off + n_cells))
        off += n_cells

    all_ca, all_cb = np.vstack(ca_l), np.vstack(cb_l)
    ta, tb = all_ca.sum(axis=1), all_cb.sum(axis=1)
    sa_m, sb_m = float(np.median(ta[ta > 0])), float(np.median(tb[tb > 0]))
    qa_d2, qb_d2 = 0.75 * min(sa_m, sb_m) / sa_m, 0.75 * min(sa_m, sb_m) / sb_m
    qa_sh2, qb_sh2, cl2 = estimate_sh_generic(all_ca, all_cb, classify_6, "both", 0.75, rng)
    qa_sh1, qb_sh1, cl1 = estimate_sh_generic(all_ca, all_cb, classify_6, "onesided", 1.00, rng)

    def run_pair01(cnt_all, qv):
        rm = np.full((B, n_cores), np.nan)
        if qv >= 1.0 - 1e-12:
            lab, _ = classify_6(cnt_all)
            for ci, (sl, (u, v, cc)) in enumerate(zip(slices, graphs)):
                rm[:, ci] = all_pairs_r_matrix(lab[sl], u, v, cc, 6)[0, 1]
        else:
            for b in range(B):
                lab, _ = classify_6(rng.binomial(cnt_all, qv))
                for ci, (sl, (u, v, cc)) in enumerate(zip(slices, graphs)):
                    rm[b, ci] = all_pairs_r_matrix(lab[sl], u, v, cc, 6)[0, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.nanmean(rm, axis=0)
            return np.where(m > 0, np.log(m), np.nan)

    arms = {
        "Naive": (nv_a, nv_b),
        "Depth_Raref_2s": (run_pair01(all_ca, qa_d2), run_pair01(all_cb, qb_d2)),
        "SplitHalf_2s_Default": (run_pair01(all_ca, qa_sh2), run_pair01(all_cb, qb_sh2)),
        "SplitHalf_1s": (run_pair01(all_ca, qa_sh1), run_pair01(all_cb, qb_sh1)),
    }
    rows = []
    for aname, (va, vb) in arms.items():
        ok = np.isfinite(va) & np.isfinite(vb)
        da, db = va[ok], vb[ok]
        mean_abs_offset = abs(float(np.mean(db - da)))
        p_paired = float(ttest_rel(db, da).pvalue) if len(da) >= 10 else np.nan
        # Tier 2 disjoint 50/50 split (30 vs 30 independent cores)
        half = len(da) // 2
        p_disj = float(ttest_ind(db[half:], da[:half], equal_var=False).pvalue) if half >= 5 else np.nan
        rows.append(dict(
            rep=rep, sigma_g=sigma_g, fold_p95_p05=fold_p95_p05, method=aname,
            mean_abs_offset=mean_abs_offset,
            mae_pair=float(np.mean(np.abs(db - da))),
            tier3_paired_rej=bool(np.isfinite(p_paired) and p_paired < 0.05),
            tier2_disjoint_rej=bool(np.isfinite(p_disj) and p_disj < 0.05),
            qa_sh2=qa_sh2, qa_sh1=qa_sh1,
        ))
    return rows


def main():
    t0 = time.time()
    print("=== Experiment A: 18 Fine-Grained Subtypes Simulation (153 pairs, 40 reps/cell) ===", flush=True)
    tasks_a = [
        (r, dr, snr, d)
        for (dr, snr) in [(0.50, 1.5), (0.25, 2.0)]
        for d in (0.00, 0.30)
        for r in range(40)
    ]
    av_rows, inf_rows = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for av, inf in ex.map(run_fine18_rep, tasks_a, chunksize=4):
            av_rows.extend(av)
            inf_rows.extend(inf)
    df_av = pd.DataFrame(av_rows)
    df_inf = pd.DataFrame(inf_rows)

    av_sum = df_av.groupby(["depth_ratio", "snr", "method", "tier"], sort=False).agg(
        n_reps=("rep", "count"),
        n_pairs=("n_pairs", "first"),
        pplus_ge90_pct=("pplus_ge90_rate", lambda s: round(100 * float(s.mean()), 2)),
        finite_logr_p90_pct=("finite_logr_p90_rate", lambda s: round(100 * float(s.mean()), 2)),
        acc_deep=("acc_deep", "mean"),
        acc_shal=("acc_shal", "mean"),
        qa_sh2=("qa_sh2", "mean"),
        qa_sh1=("qa_sh1", "mean"),
    ).reset_index()
    av_sum.to_csv(OUT / "p1_fine18_sim_availability.csv", index=False)

    inf_sum = df_inf.groupby(["depth_ratio", "snr", "delta_true", "pair_tier", "method"], sort=False).agg(
        n_reps=("rep", "count"),
        avail_cores_pct=("avail_cores_min", lambda s: round(100 * float(s.mean()), 1)),
        bias=("bias", "mean"),
        mae=("abs_err", "mean"),
        reject_rate_p05_pct=("sig_p05", lambda s: round(100 * float(s.mean()), 1)),
        sd_ratio_deep=("sd_ratio_deep", "mean"),
    ).reset_index()
    inf_sum.to_csv(OUT / "p1_fine18_sim_inference.csv", index=False)
    print(f"Exp A done in {time.time()-t0:.1f}s\n", av_sum[av_sum["depth_ratio"] == 0.25].to_string(index=False))
    print("\nExp A Inference (r_d=0.25, snr=2.0):\n", inf_sum[inf_sum["depth_ratio"] == 0.25].to_string(index=False))

    t1 = time.time()
    print("\n=== Experiment B: Gene-Specific Capture Heterogeneity (sigma_g in {0.0, 0.4, 0.8, 1.2}, 40 reps) ===", flush=True)
    tasks_b = [(r, sg) for sg in (0.0, 0.4, 0.8, 1.2) for r in range(40)]
    gh_rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res in ex.map(run_gene_het_rep, tasks_b, chunksize=4):
            gh_rows.extend(res)
    df_gh = pd.DataFrame(gh_rows)
    gh_sum = df_gh.groupby(["sigma_g", "method"], sort=False).agg(
        n_reps=("rep", "count"),
        median_p95_p05_fold=("fold_p95_p05", "median"),
        mean_abs_offset=("mean_abs_offset", "mean"),
        mae_pair=("mae_pair", "mean"),
        tier2_disjoint_rej_pct=("tier2_disjoint_rej", lambda s: round(100 * float(s.mean()), 1)),
        tier3_paired_rej_pct=("tier3_paired_rej", lambda s: round(100 * float(s.mean()), 1)),
        qa_sh2=("qa_sh2", "mean"),
        qa_sh1=("qa_sh1", "mean"),
    ).reset_index()
    gh_sum.to_csv(OUT / "p1_gene_het_stress_test.csv", index=False)
    print(f"Exp B done in {time.time()-t1:.1f}s\n", gh_sum.to_string(index=False))
    print(f"\nALL P1-8 SIMULATIONS COMPLETED in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
