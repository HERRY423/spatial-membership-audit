#!/usr/bin/env python3
"""
Step 7 Remote Orchestrator on Colab Linux (v3linux):
Independent Cell-Type Classifiers (Native InSituType + CellTypist) across 4 Benchmarks (Item 5):
  1. tTMA1 Track A (2024 Xenium vs CosMx+IF, 117 cores, 119 HPCA genes) + CosMx IF + paired null + reversals
  2. tTMA1 Track B (2023 Xenium vs MERSCOPE, 78 cores, 246 HPCA genes) + paired null + reversals
  3. M242T 38-plex Dual-Segmentation IMC (H vs M, 16 tiles, r=25um) + dual IMC truth + boundary flip OR
  4. Wang 2025 nTMA Track N-A (Xenium Breast vs CosMx+IF, 47 normal organ cores, 119 HPCA genes) + CosMx IF + paired null

Both classifiers train deterministically on the EXACT SAME 6-lineage + OTHER HPCA reference, solve
their OWN Split-half 2s (q0=0.75) rates via frozen DEFAULT_PROTOCOL_V3, and are evaluated strictly on
ONE identical U_common per (dataset, classifier).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from itertools import combinations
from pathlib import Path

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, ttest_ind, ttest_rel, wilcoxon
from sklearn.metrics import cohen_kappa_score
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from spatial_membership_audit.coverage_matching import (
    build_split_half_curve,
    conditional_instability_metrics,
    match_split_half_coverage,
)
from spatial_membership_audit.dua_inference import summarize_dua_draws
from spatial_membership_audit.extrapolation import (
    build_extrapolation_guardrail,
    extrapolate_ladder,
    select_mitm_target_depth,
    summarize_thinning_ladder,
)

WORK = Path("/content/v3work/classifiers")
OUT_DIR = WORK / "results"
WORK.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ST_R = "/content/v3work/bin/micromamba"
R_ENV = "/content/v3work/mamba/st_r"
IT_R = WORK / "run_insitutype_worker.R"
HPCA_RDA = Path("/content/v3work/hpca.rda")
HPCA_EXP = WORK / "hpca_export"

CLASSES = ["Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid"]
PAIRS = [(i, j, f"{CLASSES[i]}--{CLASSES[j]}") for i in range(6) for j in range(i + 1, 6)]
PAIR_NAMES = [p[2] for p in PAIRS]
ALL_CLASSES = sorted(CLASSES + ["OTHER"])
LABEL_CODES_PY = ["UNRESOLVED"] + ALL_CLASSES
CMAP_PY = {l: i for i, l in enumerate(LABEL_CODES_PY)}
LUT_PY = np.full(256, 6, dtype=np.int32)
for k, cls in enumerate(CLASSES):
    LUT_PY[CMAP_PY[cls]] = k


# =============================================================================
# Shared Statistical Helpers
# =============================================================================
def icc_1_1(l0, l1):
    if len(l0) < 2:
        return math.nan
    w = float(np.mean((l0 - l1) ** 2) / 2.0)
    b = float(np.var(0.5 * (l0 + l1), ddof=1) - 0.5 * w)
    return float(b / (b + w)) if (b + w) > 0.0 else math.nan


def safe_log(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(a) & (a > 0.0)
    out[ok] = np.log(a[ok])
    return out


def weighted_med(vals, weights):
    ok = weights > 0
    if not np.any(ok):
        return math.nan
    v, w = vals[ok], weights[ok]
    order = np.argsort(v, kind="mergesort")
    sv, sw = v[order], w[order]
    tot = int(sw.sum()); cs = np.cumsum(sw)
    if tot % 2 == 1:
        return float(sv[np.searchsorted(cs, tot // 2 + 1, side="left")])
    return float(0.5 * (sv[np.searchsorted(cs, tot // 2, side="left")] + sv[np.searchsorted(cs, tot // 2 + 1, side="left")]))


def cluster_boot_weights(clusters, n_boot=2000, seed=20261125):
    rng = np.random.default_rng(seed)
    uniq = sorted(np.unique(clusters).tolist())
    cl_idx = [np.flatnonzero(clusters == u) for u in uniq]
    k = len(uniq)
    w = np.zeros((n_boot, len(clusters)), dtype=np.int16)
    for b in range(n_boot):
        for ci in rng.integers(0, k, size=k):
            w[b, cl_idx[ci]] += 1
    return w


def compute_bio_eta2(log_vec, pair_arr, tt_arr):
    etas = []
    for pname in PAIR_NAMES:
        m = (pair_arr == pname) & np.isfinite(log_vec)
        if m.sum() < 12:
            continue
        vals, tts = log_vec[m], tt_arr[m]
        groups = [vals[tts == tt] for tt in np.unique(tts) if (tts == tt).sum() >= 2]
        if len(groups) >= 4:
            av = np.concatenate(groups); gr = av.mean()
            ss_t = float(np.sum((av - gr) ** 2))
            ss_b = float(sum(len(g) * (g.mean() - gr) ** 2 for g in groups))
            if ss_t > 0.0:
                etas.append(ss_b / ss_t)
    return float(np.mean(etas)) if etas else math.nan


def build_core_graphs(npz_path: Path):
    z = np.load(npz_path)
    cptr, eptr, u_all, v_all = z["cell_ptr"], z["edge_ptr"], z["u"], z["v"]
    graphs = []
    for ci in range(len(cptr) - 1):
        s0, s1 = int(cptr[ci]), int(cptr[ci + 1])
        e0p, e1p = int(eptr[ci]), int(eptr[ci + 1])
        n = s1 - s0; eu, ev = u_all[e0p:e1p], v_all[e0p:e1p]; m = len(eu)
        if n >= 2 and m > 0:
            deg = np.bincount(np.concatenate([eu, ev]), minlength=n).astype(np.float64)
            n_adj = float(0.5 * np.sum(deg * (deg - 1.0)))
            n_disj = float(m * (m - 1.0) * 0.5 - n_adj)
            cc = 2.0 * m / (n * (n - 1.0))
        else:
            n_adj = n_disj = cc = 0.0
        graphs.append(dict(s0=s0, s1=s1, n=n, m=m, e0=eu, e1=ev, n_adj=n_adj, n_disj=n_disj, c_const=cc))
    return graphs


def exact_perm_15_core(g, loc_codes, lut):
    R = np.full(15, np.nan); z = np.full(15, np.nan)
    n, m = g["n"], g["m"]
    if n < 2 or m == 0:
        return R, z
    mapped = lut[loc_codes]
    cnt = np.bincount(mapped, minlength=7)[:6].astype(np.float64)
    c0, c1 = mapped[g["e0"]], mapped[g["e1"]]
    keep = (c0 < 6) & (c1 < 6) & (c0 != c1)
    o_mat = np.bincount(np.minimum(c0[keep], c1[keep]) * 6 + np.maximum(c0[keep], c1[keep]), minlength=36).astype(np.float64) if keep.any() else np.zeros(36)
    cc, na_, nd_ = g["c_const"], g["n_adj"], g["n_disj"]
    d3 = n * (n - 1.0) * (n - 2.0) if n >= 3 else 0.0
    d4 = d3 * (n - 3.0) if n >= 4 else 0.0
    for pi, (a, b, _) in enumerate(PAIRS):
        na, nb = cnt[a], cnt[b]
        if na <= 0 or nb <= 0:
            continue
        o = o_mat[a * 6 + b]; ep = cc * na * nb
        R[pi] = o / ep
        pa = (na * nb * (na + nb - 2.0)) / d3 if n >= 3 else 0.0
        pd_ = (4.0 * na * max(0.0, na - 1.0) * nb * max(0.0, nb - 1.0)) / d4 if (n >= 4 and na >= 2 and nb >= 2) else 0.0
        vp = max(0.0, ep + 2.0 * na_ * pa + 2.0 * nd_ * pd_ - ep * ep)
        if vp > 0.0:
            z[pi] = (o - ep) / math.sqrt(vp)
    return R, z


def eval_codes(graphs, arr, lut):
    U = len(graphs) * 15
    if arr.ndim == 1:
        R, z = np.full(U, np.nan), np.full(U, np.nan)
        for ci, g in enumerate(graphs):
            R[ci*15:(ci+1)*15], z[ci*15:(ci+1)*15] = exact_perm_15_core(g, arr[g["s0"]:g["s1"]], lut)
        return R, z
    B = len(arr)
    Rb, zb = np.full((B, U), np.nan), np.full((B, U), np.nan)
    for b in range(B):
        for ci, g in enumerate(graphs):
            Rb[b, ci*15:(ci+1)*15], zb[b, ci*15:(ci+1)*15] = exact_perm_15_core(g, arr[b, g["s0"]:g["s1"]], lut)
    return Rb, zb


def eval_soft_probs_15(graphs, P_6: np.ndarray):
    """P_6: (N, 6) posterior probabilities ordered as CLASSES."""
    out = np.full((len(graphs), 15), np.nan)
    valid = np.all(np.isfinite(P_6), axis=1) & (P_6.sum(axis=1) > 1e-6)
    P = np.where(valid[:, None], np.clip(P_6, 0.0, 1.0), 0.0)
    for ci, g in enumerate(graphs):
        if g["n"] < 2 or g["m"] == 0:
            continue
        Ps = P[g["s0"]:g["s1"]]; sums = Ps.sum(axis=0)
        e0, e1, cc = g["e0"], g["e1"], g["c_const"]
        for pi, (a, b, _) in enumerate(PAIRS):
            d = float(sums[a] * sums[b] - np.dot(Ps[:, a], Ps[:, b]))
            if d > 0.0:
                o = float((Ps[e0, a] * Ps[e1, b] + Ps[e0, b] * Ps[e1, a]).sum())
                out[ci, pi] = o / (cc * d)
    return out.ravel()


# =============================================================================
# CellTypist Model Builder (HPCA 6-lineage + OTHER)
# =============================================================================
def load_exported_hpca():
    meta = json.loads((HPCA_EXP / "hpca_export_meta.json").read_text())
    genes = [g for g in (HPCA_EXP / "hpca_genes_all.txt").read_text().splitlines() if g.strip()]
    labels = [l for l in (HPCA_EXP / "hpca_labels_6lin.txt").read_text().splitlines() if l.strip()]
    # R writes column-major (genes x samples), so reshape (n_samples, n_genes) in C-order!
    raw = np.fromfile(HPCA_EXP / "hpca_data_f32.bin", dtype=np.float32).reshape(meta["n_samples"], meta["n_genes"])
    return raw, genes, labels


def train_celltypist_on_panel(hpca_raw: np.ndarray, hpca_genes: list[str], hpca_labels: list[str], panel_genes: list[str]):
    gmap = {g: i for i, g in enumerate(hpca_genes)}
    cidx = [gmap[g] for g in panel_genes]
    X_lin = np.maximum(np.power(2.0, hpca_raw[:, cidx].astype(np.float64)) - 1.0, 0.0)
    # Normalize each reference sample to CP10k then log1p — exact CellTypist convention!
    rs = X_lin.sum(axis=1, keepdims=True)
    X_norm = np.log1p(X_lin / np.maximum(rs, 1e-6) * 1e4).astype(np.float32)
    adata_ref = ad.AnnData(X=X_norm, obs=pd.DataFrame({"lineage": hpca_labels}), var=pd.DataFrame(index=panel_genes))
    model = celltypist.train(
        adata_ref, labels="lineage", C=1.0, max_iter=500, use_SGD=False,
        feature_selection=False, check_expression=False, n_jobs=max(2, os.cpu_count() or 8),
    )
    return model


def predict_celltypist_counts(mat_c_by_g: np.ndarray, panel_genes: list[str], model, return_probs: bool = False):
    n = mat_c_by_g.shape[0]
    codes = np.zeros(n, dtype=np.uint8)  # 0 = UNRESOLVED
    probs_6 = np.full((n, 6), np.nan, dtype=np.float64) if return_probs else None
    tot = mat_c_by_g.sum(axis=1)
    pos = np.flatnonzero(tot > 0)
    if len(pos) > 0:
        for s in range(0, len(pos), 50000):
            idx = pos[s:s + 50000]
            sub = mat_c_by_g[idx].astype(np.float32)
            rs = sub.sum(axis=1, keepdims=True)
            X_norm = np.log1p(sub / np.maximum(rs, 1e-6) * 1e4)
            ad_q = ad.AnnData(X=X_norm, var=pd.DataFrame(index=panel_genes))
            pred = celltypist.annotate(ad_q, model=model, majority_voting=False)
            labs = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
            codes[idx] = np.array([CMAP_PY[l] for l in labs], dtype=np.uint8)
            if return_probs:
                pm = pred.probability_matrix
                for k, cls in enumerate(CLASSES):
                    if cls in pm.columns:
                        probs_6[idx, k] = pm[cls].to_numpy(np.float64)
                    else:
                        probs_6[idx, k] = 0.0
    return (codes, probs_6) if return_probs else codes


def run_celltypist_probe(mat_a: np.ndarray, mat_b: np.ndarray, panel_genes: list[str], model):
    q_grid = [1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10]
    res = {"q_grid": q_grid}
    for s_tag, mat, seed0 in (("a", mat_a, 2026801), ("b", mat_b, 2026802)):
        rng0 = np.random.default_rng(seed0)
        nc = len(mat)
        sidx = np.sort(rng0.choice(nc, size=min(10000, nc), replace=False))
        sub = mat[sidx]
        cls = sub.sum(axis=1) > 0
        ufrac = float((~cls).mean())
        cpsi = []
        for qi, qv in enumerate(q_grid):
            p1 = 0.5 * qv
            p2 = min(1.0, max(0.0, p1 / (1.0 - p1)))
            dc = []
            for rep in range(3):
                rng = np.random.default_rng(880000 + (0 if s_tag == "a" else 5000) + qi * 100 + rep)
                y1 = rng.binomial(sub, p1)
                y2 = rng.binomial(sub - y1, p2)
                c1 = predict_celltypist_counts(y1, panel_genes, model)
                c2 = predict_celltypist_counts(y2, panel_genes, model)
                dc.append(float((c1 != c2)[cls].mean()))
            cpsi.append(float(np.mean(dc)))
        res[s_tag] = {"unresolved_frac": ufrac, "cond_psi": cpsi}
    return res


def run_celltypist_bank(mat: np.ndarray, qv: float, panel_genes: list[str], model, base_codes: np.ndarray, seed_base: int, B: int = 20):
    if qv >= 1.0 - 1e-6:
        return np.tile(base_codes, (B, 1))
    out = np.empty((B, len(mat)), dtype=np.uint8)
    for rep in range(B):
        rng = np.random.default_rng(seed_base + rep)
        thin = rng.binomial(mat, qv)
        out[rep] = predict_celltypist_counts(thin, panel_genes, model)
    return out


# =============================================================================
# Core TMA Benchmark Evaluator (tTMA1 Track A, tTMA1 Track B, nTMA Track N-A)
# =============================================================================
def eval_tma_dataset(
    ds_name: str, tdir: Path, mat_a: np.ndarray, mat_b: np.ndarray, genes: list[str],
    cells_a: pd.DataFrame, cells_b: pd.DataFrame, cores_df: pd.DataFrame,
    ga: list[dict], gb: list[dict], has_if: bool,
    hpca_raw: np.ndarray, hpca_genes: list[str], hpca_labels: list[str],
    comp_rows: list, null_rows: list, rev_rows: list, if_rows: list, rate_log: dict,
):
    print(f"\n================ {ds_name} ({len(cores_df)} cores, {len(genes)} genes) ================", flush=True)
    wdir = WORK / ds_name
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "genes.txt").write_text("\n".join(genes) + "\n")
    np.ascontiguousarray(mat_a, dtype=np.int32).tofile(wdir / "run_a_counts.bin")
    np.ascontiguousarray(mat_b, dtype=np.int32).tofile(wdir / "run_b_counts.bin")

    ta_pos = mat_a.sum(axis=1); tb_pos = mat_b.sum(axis=1)
    sa = float(np.median(ta_pos[ta_pos > 0])); sb = float(np.median(tb_pos[tb_pos > 0]))
    r_depth = min(sa, sb) / max(sa, sb)
    sh, dp = ("a", "b") if sa <= sb else ("b", "a")
    s_sh, s_dp = min(sa, sb), max(sa, sb)
    d_geom = select_mitm_target_depth(sa, sb, mode="geometric")
    q_mitm_dp = min(1.0, d_geom / s_dp)

    cores = cores_df["core"].astype(str).to_numpy()
    tt = cores_df["tissue_type"].astype(str).to_numpy()
    n_cores = len(cores)
    cent = cells_a.groupby(cells_a["core"].astype(str))[["x_um", "y_um"]].median()
    xs, ys = cent.loc[cores, "x_um"].to_numpy(float), cent.loc[cores, "y_um"].to_numpy(float)
    bx = np.clip(np.floor((xs - xs.min()) / max(1e-6, xs.max() - xs.min()) * 4).astype(int), 0, 3)
    by = np.clip(np.floor((ys - ys.min()) / max(1e-6, ys.max() - ys.min()) * 3).astype(int), 0, 2)
    tma_blks = np.array([f"{ix}_{iy}" for ix, iy in zip(bx, by)])
    bw_tma = cluster_boot_weights(tma_blks, 2000, 20261125)

    def score_classifier(clf_name: str, base_a, base_b, probs_a6, probs_b6,
                         draws_a_aq, draws_b_aq, draws_a_p2, draws_b_p2,
                         sh_lad_draws: dict[float, np.ndarray], dp_mitm_draws: np.ndarray,
                         lut: np.ndarray, cmap: dict, p2_info: dict):
        nr_a, nz_a = eval_codes(ga, base_a, lut)
        nr_b, nz_b = eval_codes(gb, base_b, lut)
        rs_a = eval_soft_probs_15(ga, probs_a6)
        rs_b = eval_soft_probs_15(gb, probs_b6)

        ra_aq, za_aq = eval_codes(ga, draws_a_aq, lut)
        rb_aq, zb_aq = eval_codes(gb, draws_b_aq, lut)
        da_aq = summarize_dua_draws(ra_aq, za_aq, availability_threshold=0.90)
        db_aq = summarize_dua_draws(rb_aq, zb_aq, availability_threshold=0.90)
        r_aq_a = np.where(da_aq.available, da_aq.r_mean, np.nan)
        r_aq_b = np.where(db_aq.available, db_aq.r_mean, np.nan)

        ra_p2, za_p2 = eval_codes(ga, draws_a_p2, lut)
        rb_p2, zb_p2 = eval_codes(gb, draws_b_p2, lut)
        da_p2 = summarize_dua_draws(ra_p2, za_p2, availability_threshold=0.90)
        db_p2 = summarize_dua_draws(rb_p2, zb_p2, availability_threshold=0.90)
        r_p2_a = np.where(da_p2.available, da_p2.r_mean, np.nan)
        r_p2_b = np.where(db_p2.available, db_p2.r_mean, np.nan)

        # MITM geom EB + guardrail
        lad_q = [1.00, 0.85, 0.75, 0.65, 0.50]
        sh_g = ga if sh == "a" else gb
        dp_g = gb if sh == "a" else ga
        sh_rb = [nr_a if sh == "a" else nr_b] + [eval_codes(sh_g, sh_lad_draws[q], lut)[0] for q in lad_q[1:]]
        sh_zb = [nz_a if sh == "a" else nz_b] + [eval_codes(sh_g, sh_lad_draws[q], lut)[1] for q in lad_q[1:]]
        lad_sh = summarize_thinning_ladder(sh_rb, sh_zb, lad_q, availability_threshold=0.90)
        sh_mat = mat_a if sh == "a" else mat_b
        sh_base = base_a if sh == "a" else base_b
        tx_sh = sh_mat.sum(axis=1).astype(float)
        c_med = np.array([float(np.median(tx_sh[g["s0"]:g["s1"]])) for g in sh_g])
        c_flp = np.array([conditional_instability_metrics(sh_base[g["s0"]:g["s1"]], sh_lad_draws[0.50][:, g["s0"]:g["s1"]], unresolved_label=0).cond_flip_rate for g in sh_g])
        gmask = build_extrapolation_guardrail(np.repeat(c_med, 15), np.repeat(c_flp, 15), min_median_depth=2.0, max_cond_flip_rate=0.45)
        ex = extrapolate_ladder(lad_sh, depth_scale=s_sh, target_depth=d_geom, pair_ids=np.tile(np.arange(15), n_cores), method="hier_eb", guardrail_mask=None)
        r_sh_m = np.exp(ex.log_r_pred)
        rdm, zdm = eval_codes(dp_g, dp_mitm_draws, lut)
        ddm = summarize_dua_draws(rdm, zdm, availability_threshold=0.90)
        r_dp_m = np.where(ddm.available, ddm.r_mean, np.nan)
        rm_a_raw, rm_b_raw = (r_sh_m, r_dp_m) if sh == "a" else (r_dp_m, r_sh_m)
        r_mitm_a = np.where(gmask, rm_a_raw, r_p2_a)
        r_mitm_b = np.where(gmask, rm_b_raw, r_p2_b)

        arms_log = {
            "Naive": (safe_log(nr_a), safe_log(nr_b)),
            "Soft_plugin_native_prob": (safe_log(rs_a), safe_log(rs_b)),
            "Depth_only_2s_q075": (safe_log(r_aq_a), safe_log(r_aq_b)),
            "Split_half_2s_default": (safe_log(r_p2_a), safe_log(r_p2_b)),
            "MITM_geom_eb_guardrail": (safe_log(r_mitm_a), safe_log(r_mitm_b)),
        }
        d0 = np.isfinite(arms_log["Naive"][0]) & np.isfinite(arms_log["Naive"][1])
        uc = d0.copy()
        for la, lb in arms_log.values():
            uc &= np.isfinite(la) & np.isfinite(lb)
        n_uc = int(uc.sum())
        uc_idx = np.repeat(np.arange(n_cores), 15)[uc]
        pair_u = np.tile(PAIR_NAMES, n_cores)[uc]
        tt_u = np.repeat(tt, 15)[uc]
        na_u, nb_u = arms_log["Naive"][0][uc], arms_log["Naive"][1][uc]
        diff_n = np.abs(na_u - nb_u); med_n = float(np.median(diff_n))
        sd_na, sd_nb = float(np.std(na_u, ddof=1)), float(np.std(nb_u, ddof=1))

        u_mat = uc.reshape(n_cores, 15)
        uniq_tt = sorted(np.unique(tt).tolist())

        for ak, (la, lb) in arms_log.items():
            la_u, lb_u = la[uc], lb[uc]
            diff_a = np.abs(la_u - lb_u)
            med_a, mae_a = float(np.median(diff_a)), float(np.mean(diff_a))
            red = 100.0 * (1.0 - med_a / med_n) if med_n > 0 else math.nan
            bvs = [100.0 * (1.0 - weighted_med(diff_a, bw_tma[b][uc_idx]) / weighted_med(diff_n, bw_tma[b][uc_idx])) for b in range(len(bw_tma))]
            bvs = [v for v in bvs if np.isfinite(v)]
            lo, hi = (float(np.quantile(bvs, 0.025)), float(np.quantile(bvs, 0.975))) if bvs else (math.nan, math.nan)
            comp_rows.append(dict(
                dataset=ds_name, classifier=clf_name, arm=ak, n_cores=n_cores,
                d0_n=int(d0.sum()), u_common_n=n_uc,
                icc=icc_1_1(la_u, lb_u), pearson_r=float(np.corrcoef(la_u, lb_u)[0, 1]),
                spearman_rho=float(spearmanr(la_u, lb_u).statistic),
                mae=mae_a, med_disc=med_a, red_pct=red, red_ci_lo_tma=lo, red_ci_hi_tma=hi,
                sd_ratio_a=float(np.std(la_u, ddof=1) / max(1e-12, sd_na)),
                sd_ratio_b=float(np.std(lb_u, ddof=1) / max(1e-12, sd_nb)),
                bio_eta2_a=compute_bio_eta2(la_u, pair_u, tt_u),
                bio_eta2_b=compute_bio_eta2(lb_u, pair_u, tt_u),
            ))

            # 3-tier paired zero-effect null tests & tissue-vs-tissue reversals
            L0, L1 = np.where(u_mat, la.reshape(n_cores, 15), np.nan), np.where(u_mat, lb.reshape(n_cores, 15), np.nan)
            pt, pw = [], []
            for pi in range(15):
                ok = np.isfinite(L0[:, pi]) & np.isfinite(L1[:, pi])
                if ok.sum() >= 5:
                    d = L0[ok, pi] - L1[ok, pi]
                    pt.append(float(ttest_rel(L0[ok, pi], L1[ok, pi]).pvalue))
                    pw.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
            qt = multipletests(pt, alpha=0.05, method="fdr_bh")[0] if pt else np.array([], bool)
            qw = multipletests(pw, alpha=0.05, method="fdr_bh")[0] if pw else np.array([], bool)
            wpt = []
            for o in uniq_tt:
                om = tt == o
                for pi in range(15):
                    ok = om & np.isfinite(L0[:, pi]) & np.isfinite(L1[:, pi])
                    if ok.sum() >= 3:
                        wpt.append(float(ttest_rel(L0[ok, pi], L1[ok, pi]).pvalue))
            wqt = multipletests(wpt, alpha=0.05, method="fdr_bh")[0] if wpt else np.array([], bool)
            rng_sp = np.random.default_rng(2026112577)
            fdr_d = []
            for _ in range(20):
                hm = np.zeros(n_cores, dtype=bool)
                for o in uniq_tt:
                    idx = np.flatnonzero(tt == o)
                    perm = rng_sp.permutation(idx)
                    hm[perm[:max(1, len(perm) // 2)]] = True
                psd = []
                for pi in range(15):
                    va = L0[hm, pi]; va = va[np.isfinite(va)]
                    vb = L1[~hm, pi]; vb = vb[np.isfinite(vb)]
                    if len(va) >= 4 and len(vb) >= 4:
                        psd.append(float(ttest_ind(va, vb, equal_var=False).pvalue))
                if psd:
                    fdr_d.append(100.0 * float(multipletests(psd, alpha=0.05, method="fdr_bh")[0].mean()))
            null_rows.append(dict(
                dataset=ds_name, classifier=clf_name, arm=ak, u_common_n=n_uc,
                pan_paired_t_fdr_pct=100.0 * float(qt.mean()) if len(qt) else math.nan,
                pan_wilcoxon_fdr_pct=100.0 * float(qw.mean()) if len(qw) else math.nan,
                within_group_paired_t_fdr_pct=100.0 * float(wqt.mean()) if len(wqt) else math.nan,
                disjoint_50_50_welch_fdr_pct=float(np.mean(fdr_d)) if fdr_d else math.nan,
            ))

            # Tissue-vs-tissue sign reversals
            g_ok = [o for o in uniq_tt if (tt == o).sum() >= 2]
            crecs = []
            for o1, o2 in combinations(g_ok, 2):
                m1, m2 = tt == o1, tt == o2
                for pi in range(15):
                    a1, a2 = L0[m1, pi], L0[m2, pi]; b1, b2 = L1[m1, pi], L1[m2, pi]
                    a1, a2, b1, b2 = a1[np.isfinite(a1)], a2[np.isfinite(a2)], b1[np.isfinite(b1)], b2[np.isfinite(b2)]
                    if len(a1) >= 2 and len(a2) >= 2 and len(b1) >= 2 and len(b2) >= 2:
                        crecs.append((float(a1.mean() - a2.mean()), float(b1.mean() - b2.mean())))
            if crecs:
                ca_arr = np.array(crecs)
                tk = min(50, len(ca_arr))
                ti = np.argsort(-np.abs(ca_arr[:, 0]))[:tk]
                rev_rows.append(dict(
                    dataset=ds_name, classifier=clf_name, arm=ak, n_contrasts=len(ca_arr), top_k=tk,
                    contrast_r=float(np.corrcoef(ca_arr[:, 0], ca_arr[:, 1])[0, 1]),
                    top_k_reversal_pct=100.0 * float(np.mean(np.sign(ca_arr[ti, 0]) != np.sign(ca_arr[ti, 1]))),
                ))

        # Same-cell CosMx IF concordance if available (side b is CosMx+IF!)
        if has_if and "Mean.PanCK" in cells_b.columns:
            raw_pan = np.log1p(pd.to_numeric(cells_b["Mean.PanCK"], errors="coerce").fillna(0.0).to_numpy(float))
            rng = np.random.default_rng(42)
            gmm = GaussianMixture(n_components=2, random_state=42).fit(rng.choice(raw_pan, size=min(100000, len(raw_pan)), replace=False).reshape(-1, 1))
            hi = int(np.argmax(gmm.means_.ravel()))
            ppos = gmm.predict_proba(raw_pan.reshape(-1, 1))[:, hi] >= 0.5
            r68 = np.log1p(pd.to_numeric(cells_b["Mean.CD68"], errors="coerce").fillna(0.0).to_numpy(float))
            r45 = np.log1p(pd.to_numeric(cells_b["Mean.CD45"], errors="coerce").fillna(0.0).to_numpy(float))
            z68, z45 = np.zeros(len(cells_b)), np.zeros(len(cells_b))
            for g in gb:
                s0, s1 = g["s0"], g["s1"]
                for arr_in, z_out in ((r68, z68), (r45, z45)):
                    sub = arr_in[s0:s1]; med = float(np.median(sub)); mad = float(1.4826 * np.median(np.abs(sub - med)))
                    z_out[s0:s1] = (sub - med) / max(mad, 1e-6)
            r_if = np.full((len(gb), 3), np.nan)
            for ci, g in enumerate(gb):
                s0, s1 = g["s0"], g["s1"]
                if g["n"] < 2 or g["m"] == 0:
                    continue
                tum = ppos[s0:s1]; mac = (~tum) & (z68[s0:s1] > 1.5); imm = (~tum) & ((z68[s0:s1] > 1.5) | (z45[s0:s1] > 1.5)); strom = (~tum) & (~imm)
                na = float(tum.sum()); e0, e1, cc = g["e0"], g["e1"], g["c_const"]
                for ai, mb in enumerate((mac, imm, strom)):
                    nb = float(mb.sum())
                    if na > 0 and nb > 0:
                        r_if[ci, ai] = float(((tum[e0] & mb[e1]) | (tum[e1] & mb[e0])).sum()) / (cc * na * nb)
            l_if = safe_log(r_if.ravel())

            ce, cm, cs_ = cmap["Epithelial"], cmap["Myeloid"], cmap["Stroma"]
            cimm = np.array([cmap["T_cells"], cmap["B_cell"], cmap["Myeloid"]], dtype=np.uint8)
            def e3_codes(arr):
                def one(cv):
                    o = np.full((len(gb), 3), np.nan)
                    for ci, g in enumerate(gb):
                        if g["n"] < 2 or g["m"] == 0:
                            continue
                        loc = cv[g["s0"]:g["s1"]]; ma = loc == ce; na = float(ma.sum())
                        if na == 0:
                            continue
                        e0, e1, cc = g["e0"], g["e1"], g["c_const"]
                        for ai, mb in enumerate((loc == cm, np.isin(loc, cimm), loc == cs_)):
                            nb = float(mb.sum())
                            if nb > 0:
                                o[ci, ai] = float(((ma[e0] & mb[e1]) | (ma[e1] & mb[e0])).sum()) / (cc * na * nb)
                    return o.ravel()
                if arr.ndim == 1:
                    return safe_log(one(arr))
                st = np.vstack([one(arr[b]) for b in range(len(arr))])
                ok = np.isfinite(st).any(0); mu = np.full(st.shape[1], np.nan); mu[ok] = np.nanmean(st[:, ok], axis=0)
                return safe_log(mu)

            def e3_soft(P6):
                # P6 columns ordered as CLASSES = [Epithelial, Endothelial, Stroma, B_cell, T_cells, Myeloid]
                v = np.all(np.isfinite(P6), axis=1); P = np.where(v[:, None], np.clip(P6, 0.0, 1.0), 0.0)
                o = np.full((len(gb), 3), np.nan)
                for ci, g in enumerate(gb):
                    if g["n"] < 2 or g["m"] == 0:
                        continue
                    Ps = P[g["s0"]:g["s1"]]; wa = Ps[:, 0]; sa_ = float(wa.sum())
                    e0, e1, cc = g["e0"], g["e1"], g["c_const"]
                    for ai, wb in enumerate((Ps[:, 5], Ps[:, 3] + Ps[:, 4] + Ps[:, 5], Ps[:, 2])):
                        sb_ = float(wb.sum()); d = float(sa_ * sb_ - np.dot(wa, wb))
                        if sa_ > 0 and sb_ > 0 and d > 0:
                            o[ci, ai] = float((wa[e0] * wb[e1] + wa[e1] * wb[e0]).sum()) / (cc * d)
                return safe_log(o.ravel())

            if_arms = {
                "Naive": e3_codes(base_b),
                "Soft_plugin_native_prob": e3_soft(probs_b6),
                "Depth_only_2s_q075": e3_codes(draws_b_aq),
                "Split_half_2s_default": e3_codes(draws_b_p2),
            }
            uc_if = np.isfinite(l_if)
            for lv in if_arms.values():
                uc_if &= np.isfinite(lv)
            li_u = l_if[uc_if]; sd_if = float(np.std(li_u, ddof=1))
            for ak, lv in if_arms.items():
                la_u = lv[uc_if]
                if_rows.append(dict(
                    dataset=ds_name, classifier=clf_name, arm=ak, u_common_n=int(uc_if.sum()),
                    icc_vs_if=icc_1_1(la_u, li_u), pearson_r_vs_if=float(np.corrcoef(la_u, li_u)[0, 1]),
                    spearman_rho_vs_if=float(spearmanr(la_u, li_u).statistic),
                    mae_vs_if=float(np.mean(np.abs(la_u - li_u))),
                    sd_ratio_vs_if=float(np.std(la_u, ddof=1) / max(1e-12, sd_if)),
                ))
        rate_log[f"{ds_name}__{clf_name}"] = p2_info
        print(f"  [{ds_name} | {clf_name}] U_common={n_uc}, Split_half_2s ICC={comp_rows[-2]['icc']:.3f} vs Naive {comp_rows[-5]['icc']:.3f}", flush=True)

    # --- Classifier A: Native Linux InSituType ---
    subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(IT_R), "probe", str(wdir), str(HPCA_RDA)], check=True)
    it_pr = json.loads((wdir / "insitutype_probe.json").read_text())
    it_p2 = match_split_half_coverage(
        build_split_half_curve(it_pr["q_grid"], it_pr["a"]["cond_psi"], unresolved_frac=it_pr["a"]["unresolved_frac"]),
        build_split_half_curve(it_pr["q_grid"], it_pr["b"]["cond_psi"], unresolved_frac=it_pr["b"]["unresolved_frac"]),
        q0=0.75, pairing="both", min_q=0.05,
    )
    sched_it = {
        sh: {"q085": 0.85, "q075": 0.75, "q065": 0.65, "q050": 0.50, "p2": float(it_p2.q_a if sh == "a" else it_p2.q_b)},
        dp: {"aq": 0.75 * r_depth, "mitm": q_mitm_dp, "p2": float(it_p2.q_a if dp == "a" else it_p2.q_b)},
    }
    (wdir / "insitutype_schedule.json").write_text(json.dumps(sched_it, indent=2))
    subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(IT_R), "banks", str(wdir), str(HPCA_RDA)], check=True)

    lt_it = pd.read_csv(wdir / "insitutype_label_codes.tsv", sep="\t")
    cmap_it = dict(zip(lt_it["label"], lt_it["code"]))
    lut_it = np.full(256, 6, dtype=np.int32)
    for k, cls in enumerate(CLASSES):
        lut_it[cmap_it[cls]] = k
    it_ba = np.fromfile(wdir / "it_base_a.bin", dtype=np.uint8)
    it_bb = np.fromfile(wdir / "it_base_b.bin", dtype=np.uint8)
    it_pa = pd.read_csv(wdir / "it_base_a_probs.tsv.gz", sep="\t")[CLASSES].to_numpy(float)
    it_pb = pd.read_csv(wdir / "it_base_b_probs.tsv.gz", sep="\t")[CLASSES].to_numpy(float)

    def ld_it(side, tag):
        nc = len(mat_a) if side == "a" else len(mat_b)
        return np.fromfile(wdir / f"it_bank_{side}_{tag}.bin", dtype=np.uint8).reshape(20, nc)

    sh_lad_it = {0.85: ld_it(sh, "q085"), 0.75: ld_it(sh, "q075"), 0.65: ld_it(sh, "q065"), 0.50: ld_it(sh, "q050")}
    score_classifier(
        "InSituType", it_ba, it_bb, it_pa, it_pb,
        ld_it("a", "q075" if sh == "a" else "aq"), ld_it("b", "q075" if sh == "b" else "aq"),
        ld_it("a", "p2"), ld_it("b", "p2"), sh_lad_it, ld_it(dp, "mitm"), lut_it, cmap_it,
        dict(q_a=it_p2.q_a, q_b=it_p2.q_b, target_psi=it_p2.target_psi, noisier=it_p2.noisier_side),
    )

    # --- Classifier B: CellTypist (Python L2-Logistic trained on HPCA 6 lineages) ---
    ct_model = train_celltypist_on_panel(hpca_raw, hpca_genes, hpca_labels, genes)
    ct_pr = run_celltypist_probe(mat_a, mat_b, genes, ct_model)
    ct_p2 = match_split_half_coverage(
        build_split_half_curve(ct_pr["q_grid"], ct_pr["a"]["cond_psi"], unresolved_frac=ct_pr["a"]["unresolved_frac"]),
        build_split_half_curve(ct_pr["q_grid"], ct_pr["b"]["cond_psi"], unresolved_frac=ct_pr["b"]["unresolved_frac"]),
        q0=0.75, pairing="both", min_q=0.05,
    )
    ct_ba, ct_pa = predict_celltypist_counts(mat_a, genes, ct_model, return_probs=True)
    ct_bb, ct_pb = predict_celltypist_counts(mat_b, genes, ct_model, return_probs=True)

    sh_mat, dp_mat = (mat_a, mat_b) if sh == "a" else (mat_b, mat_a)
    sh_base, dp_base = (ct_ba, ct_bb) if sh == "a" else (ct_bb, ct_ba)
    sh_lad_ct = {
        q: run_celltypist_bank(sh_mat, q, genes, ct_model, sh_base, 810000 + int(q * 1000))
        for q in (0.85, 0.75, 0.65, 0.50)
    }
    dp_aq_ct = run_celltypist_bank(dp_mat, 0.75 * r_depth, genes, ct_model, dp_base, 820000)
    dp_mitm_ct = run_celltypist_bank(dp_mat, q_mitm_dp, genes, ct_model, dp_base, 830000)
    p2_a_ct = (sh_lad_ct[0.75] if (sh == "a" and abs(ct_p2.q_a - 0.75) < 1e-4)
               else run_celltypist_bank(mat_a, float(ct_p2.q_a), genes, ct_model, ct_ba, 840000))
    p2_b_ct = (sh_lad_ct[0.75] if (sh == "b" and abs(ct_p2.q_b - 0.75) < 1e-4)
               else run_celltypist_bank(mat_b, float(ct_p2.q_b), genes, ct_model, ct_bb, 850000))

    score_classifier(
        "CellTypist", ct_ba, ct_bb, ct_pa, ct_pb,
        sh_lad_ct[0.75] if sh == "a" else dp_aq_ct, sh_lad_ct[0.75] if sh == "b" else dp_aq_ct,
        p2_a_ct, p2_b_ct, sh_lad_ct, dp_mitm_ct, LUT_PY, CMAP_PY,
        dict(q_a=ct_p2.q_a, q_b=ct_p2.q_b, target_psi=ct_p2.target_psi, noisier=ct_p2.noisier_side),
    )
    # Clean large temp binary count files
    for p in (wdir / "run_a_counts.bin", wdir / "run_b_counts.bin"):
        if p.exists():
            p.unlink()


# =============================================================================
# M242T 38-plex Dual-Segmentation IMC under InSituType + CellTypist
# =============================================================================
def eval_m242t_classifiers(hpca_raw, hpca_genes, hpca_labels, m242t_rows: list, flip_rows: list):
    print("\n================ M242T Dual-Seg IMC (H vs M, r=25um) ================", flush=True)
    mdir = Path("/content/v3work/m242t")
    roi = pd.read_parquet(mdir / "xenium_roi_cells.parquet")
    indep = pd.read_parquet(mdir / "imc_indep_cells.parquet")
    e25_x = np.load(mdir / "xenium_edges_r25.npz")
    e25_i = np.load(mdir / "imc_indep_edges_r25.npz")
    import h5py
    with h5py.File(mdir / "xenium_roi_counts.h5", "r") as hf:
        all_genes = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in hf["genes"][:]]
        mat_g_c = sp.csc_matrix((hf["data"][:], hf["indices"][:], hf["indptr"][:]), shape=tuple(int(v) for v in hf["shape"][:]))
    hpca_set = set(hpca_genes)
    keep_idx = [i for i, g in enumerate(all_genes) if g in hpca_set]
    genes = [all_genes[i] for i in keep_idx]
    mat_H = np.ascontiguousarray(mat_g_c[keep_idx, :].T.toarray(), dtype=np.int32)  # (N, G)
    rng_m = np.random.default_rng(2026112301)
    mat_M = rng_m.binomial(mat_H, 0.15).astype(np.int32)

    tiles = sorted(roi["tile"].unique().tolist())
    tile_x = roi["tile"].to_numpy()
    tile_i = indep["tile"].to_numpy()
    ux, vx = e25_x["u"], e25_x["v"]
    ui, vi = e25_i["u"], e25_i["v"]

    # Build 16-tile graph list in exact evaluate_p5_imc convention (min_count=5)
    IMC_LINEAGES = ["B_cell", "Endothelial", "Epithelial", "Myeloid", "Stroma", "T_cells"]
    IMC_PAIRS = [(a, b) for a in range(6) for b in range(a + 1, 6)]

    def tile_logr_hard(codes, t_arr, u, v, lut6, min_count=5):
        out = np.full((len(tiles), 15), np.nan)
        et = t_arr[u]
        for ti, t in enumerate(tiles):
            idx_t = np.flatnonzero(t_arr == t)
            tc = lut6[codes[idx_t]]
            valid = tc < 7
            nc = int(valid.sum())
            if nc < 2:
                continue
            cnt = np.bincount(tc[valid], minlength=7)
            em = et == t
            eu, ev = lut6[codes[u[em]]], lut6[codes[v[em]]]
            ok_e = (eu < 7) & (ev < 7)
            eu, ev = eu[ok_e], ev[ok_e]
            ne = len(eu); denom = nc * (nc - 1) / 2.0
            if ne == 0 or denom <= 0:
                continue
            m_mat = np.zeros((7, 7), dtype=np.int64)
            np.add.at(m_mat, (eu, ev), 1); m_sym = m_mat + m_mat.T
            for pi, (a, b) in enumerate(IMC_PAIRS):
                if cnt[a] >= min_count and cnt[b] >= min_count:
                    ep = ne * (cnt[a] * cnt[b]) / denom
                    if ep > 0 and m_sym[a, b] > 0:
                        out[ti, pi] = math.log(m_sym[a, b] / ep)
        return out

    def tile_r_bank(bank, t_arr, u, v, lut6, min_count=5):
        out = np.full((len(bank), len(tiles), 15), np.nan)
        et = t_arr[u]
        for b in range(len(bank)):
            codes = bank[b]
            for ti, t in enumerate(tiles):
                idx_t = np.flatnonzero(t_arr == t)
                tc = lut6[codes[idx_t]]; valid = tc < 7; nc = int(valid.sum())
                if nc < 2:
                    continue
                cnt = np.bincount(tc[valid], minlength=7)
                em = et == t; eu, ev = lut6[codes[u[em]]], lut6[codes[v[em]]]
                ok_e = (eu < 7) & (ev < 7); eu, ev = eu[ok_e], ev[ok_e]; ne = len(eu)
                denom = nc * (nc - 1) / 2.0
                if ne == 0 or denom <= 0:
                    continue
                m_mat = np.zeros((7, 7), dtype=np.int64)
                np.add.at(m_mat, (eu, ev), 1); m_sym = m_mat + m_mat.T
                for pi, (a, b) in enumerate(IMC_PAIRS):
                    if cnt[a] >= min_count and cnt[b] >= min_count:
                        ep = ne * (cnt[a] * cnt[b]) / denom
                        if ep > 0:
                            out[b, ti, pi] = m_sym[a, b] / ep
        return out

    def tile_logr_soft(P6_imc_order, t_arr, u, v, min_mass=5.0):
        out = np.full((len(tiles), 15), np.nan)
        et = t_arr[u]
        valid = np.all(np.isfinite(P6_imc_order), axis=1) & (P6_imc_order.sum(axis=1) > 1e-6)
        P = np.where(valid[:, None], np.clip(P6_imc_order, 0.0, 1.0), 0.0)
        for ti, t in enumerate(tiles):
            tm = (t_arr == t) & valid; nc = int(tm.sum())
            if nc < 2:
                continue
            Pt = P[tm]; mass = Pt.sum(0)
            em = (et == t) & valid[u] & valid[v]; uu, vv = u[em], v[em]; ne = len(uu)
            denom = nc * (nc - 1) / 2.0
            if ne == 0 or denom <= 0:
                continue
            C = P[uu].T @ P[vv]; M = C + C.T; sp_ = Pt.T @ Pt
            for pi, (a, b) in enumerate(IMC_PAIRS):
                if mass[a] >= min_mass and mass[b] >= min_mass:
                    ep = ne * max(0.0, float(mass[a] * mass[b] - sp_[a, b])) / denom
                    if ep > 0 and M[a, b] > 0:
                        out[ti, pi] = math.log(float(M[a, b]) / ep)
        return out

    id_lut = np.arange(256, dtype=np.int32)
    gt_poly = tile_logr_hard(roi["imc_hard_code"].to_numpy(np.uint8), tile_x, ux, vx, id_lut)
    gt_ind = tile_logr_hard(indep["imc_hard_code"].to_numpy(np.uint8), tile_i, ui, vi, id_lut)

    # Independent IMC heterotypic boundary indicator mapped to Xenium cells via NN <= 8 um
    xy_x = roi[["x_imc_um", "y_imc_um"]].to_numpy(float)
    xy_i = indep[["x_imc_um", "y_imc_um"]].to_numpy(float)
    imc_i_c = indep["imc_hard_code"].to_numpy(int)
    deg_i = np.zeros(len(indep), dtype=int); het_i = np.zeros(len(indep), dtype=int)
    diff_i = (imc_i_c[ui] != imc_i_c[vi]).astype(int)
    np.add.at(deg_i, ui, 1); np.add.at(deg_i, vi, 1)
    np.add.at(het_i, ui, diff_i); np.add.at(het_i, vi, diff_i)
    hf_i = np.where(deg_i > 0, het_i / np.maximum(deg_i, 1), np.nan)
    dists, nn_idx = cKDTree(xy_i).query(xy_x, k=1)
    hf_indep_on_x = np.where(dists <= 8.0, hf_i[nn_idx], np.nan)
    tx_log = np.log1p(roi["transcript_counts"].to_numpy(float))

    d_H = float(np.median(mat_H.sum(1)[mat_H.sum(1) > 0]))
    d_M = float(np.median(mat_M.sum(1)[mat_M.sum(1) > 0]))
    d_geom = math.sqrt(d_H * d_M); q_mitm_H = min(1.0, d_geom / d_H)

    wdir = WORK / "M242T_H_vs_M"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "genes.txt").write_text("\n".join(genes) + "\n")
    mat_H.tofile(wdir / "run_a_counts.bin")
    mat_M.tofile(wdir / "run_b_counts.bin")

    def eval_one_m242t_clf(clf_name, bH, bM, pH_imc, pM_imc, dH_aq, dM_75, dH_p2, dM_p2, dH_mitm, lad_M_draws, lut_imc):
        lH_nv = tile_logr_hard(bH, tile_x, ux, vx, lut_imc)
        lM_nv = tile_logr_hard(bM, tile_x, ux, vx, lut_imc)
        lH_sf = tile_logr_soft(pH_imc, tile_x, ux, vx)
        lM_sf = tile_logr_soft(pM_imc, tile_x, ux, vx)

        def mean_b(bank):
            rb = tile_r_bank(bank, tile_x, ux, vx, lut_imc)
            ok = np.isfinite(rb).any(0)
            mu = np.full(rb.shape[1:], np.nan)
            mu[ok] = np.nanmean(rb[:, ok], axis=0)
            return safe_log(mu), rb

        lH_aq, _ = mean_b(dH_aq)
        lM_75, rb_M75 = mean_b(dM_75)
        lH_p2, _ = mean_b(dH_p2)
        lM_p2, _ = mean_b(dM_p2)
        lH_mitm, _ = mean_b(dH_mitm)

        # MITM upward extrapolation on M ladder
        lad_q = [1.00, 0.85, 0.75, 0.65, 0.50]
        r_M_nv = np.exp(lM_nv).ravel()
        rb_list = [r_M_nv] + [tile_r_bank(lad_M_draws[q], tile_x, ux, vx, lut_imc).reshape(20, -1) for q in lad_q[1:]]
        zb_list = [np.where(np.isfinite(x), 0.0, np.nan) for x in rb_list]
        lad = summarize_thinning_ladder(rb_list, zb_list, lad_q, availability_threshold=0.90)
        ex = extrapolate_ladder(lad, depth_scale=d_M, target_depth=d_geom, pair_ids=np.tile(np.arange(15), len(tiles)), method="hier_eb")
        lM_mitm = ex.log_r_pred.reshape(len(tiles), 15)

        arm_pairs = {
            "Naive": (lH_nv, lM_nv),
            "Soft_plugin_native_prob": (lH_sf, lM_sf),
            "Depth_only_2s_q075": (lH_aq, lM_75),
            "Split_half_2s_default": (lH_p2, lM_p2),
            "MITM_geom_eb_secondary": (lH_mitm, lM_mitm),
        }
        uc = np.isfinite(gt_poly) & np.isfinite(gt_ind)
        for ld, ls in arm_pairs.values():
            uc &= np.isfinite(ld) & np.isfinite(ls)
        n_uc = int(uc.sum())
        diff_n = np.abs(lH_nv[uc] - lM_nv[uc]); med_n = float(np.median(diff_n))
        sd_dn = float(np.std(lH_nv[uc], ddof=1)); sd_sn = float(np.std(lM_nv[uc], ddof=1))

        def pair_r(est, gt):
            pa = np.nanmean(np.where(uc, est, np.nan), axis=0)
            pb = np.nanmean(np.where(uc, gt, np.nan), axis=0)
            ok = np.isfinite(pa) & np.isfinite(pb)
            return float(np.corrcoef(pa[ok], pb[ok])[0, 1]) if ok.sum() >= 3 else math.nan

        for ak, (ld, ls) in arm_pairs.items():
            da = np.abs(ld[uc] - ls[uc]); lavg = 0.5 * (ld + ls)
            m242t_rows.append(dict(
                dataset="M242T_H_vs_M_r25um", classifier=clf_name, arm=ak, u_common_n=n_uc,
                comp_icc=icc_1_1(ld[uc], ls[uc]), comp_mae=float(np.mean(da)),
                comp_med_disc=float(np.median(da)), comp_red_pct=100.0 * (1.0 - float(np.median(da)) / med_n),
                r_pair_imc_poly=pair_r(lavg, gt_poly),
                mae_imc_poly=0.5 * (float(np.mean(np.abs(ld[uc] - gt_poly[uc]))) + float(np.mean(np.abs(ls[uc] - gt_poly[uc])))),
                r_pair_imc_indep=pair_r(lavg, gt_ind),
                mae_imc_indep=0.5 * (float(np.mean(np.abs(ld[uc] - gt_ind[uc]))) + float(np.mean(np.abs(ls[uc] - gt_ind[uc])))),
                sd_ratio_deep=float(np.std(ld[uc], ddof=1) / max(1e-12, sd_dn)),
                sd_ratio_shal=float(np.std(ls[uc], ddof=1) / max(1e-12, sd_sn)),
            ))

        # Independent-IMC heterotypic interface flip OR (controlling for log1p depth + tile-clustered SE!)
        flipped = ((bH != bM) & (bH > 0) & (bM > 0)).astype(float)
        ok_f = np.isfinite(hf_indep_on_x) & (bH > 0) & (bM > 0)
        is_bound = (hf_indep_on_x[ok_f] > 0.0).astype(float)
        X = sm.add_constant(np.column_stack([is_bound, tx_log[ok_f]]))
        fit = sm.Logit(flipped[ok_f], X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": tile_x[ok_f]})
        or_val = float(math.exp(fit.params[1]))
        ci = fit.conf_int()
        flip_rows.append(dict(
            dataset="M242T_H_vs_M", classifier=clf_name, n_cells=int(ok_f.sum()),
            flip_interior_pct=100.0 * float(flipped[ok_f][is_bound == 0].mean()),
            flip_boundary_pct=100.0 * float(flipped[ok_f][is_bound == 1].mean()),
            boundary_or_depth_adj=or_val, or_ci_lo=float(math.exp(ci[1, 0])), or_ci_hi=float(math.exp(ci[1, 1])),
            p_value=float(fit.pvalues[1]),
        ))
        print(f"  [M242T | {clf_name}] U_common={n_uc}, boundary flip OR={or_val:.3f} (p={fit.pvalues[1]:.2e})", flush=True)

    # InSituType on M242T
    subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(IT_R), "probe", str(wdir), str(HPCA_RDA)], check=True)
    it_pr = json.loads((wdir / "insitutype_probe.json").read_text())
    it_p2 = match_split_half_coverage(
        build_split_half_curve(it_pr["q_grid"], it_pr["a"]["cond_psi"], unresolved_frac=it_pr["a"]["unresolved_frac"]),
        build_split_half_curve(it_pr["q_grid"], it_pr["b"]["cond_psi"], unresolved_frac=it_pr["b"]["unresolved_frac"]),
        q0=0.75, pairing="both", min_q=0.05,
    )
    (wdir / "insitutype_schedule.json").write_text(json.dumps({
        "a": {"aq": 0.75 * (d_M / d_H), "mitm": q_mitm_H, "p2": float(it_p2.q_a)},
        "b": {"q085": 0.85, "q075": 0.75, "q065": 0.65, "q050": 0.50, "p2": float(it_p2.q_b)},
    }, indent=2))
    subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(IT_R), "banks", str(wdir), str(HPCA_RDA)], check=True)
    lt_it = pd.read_csv(wdir / "insitutype_label_codes.tsv", sep="\t")
    cmap_it = dict(zip(lt_it["label"], lt_it["code"]))
    lut_it_imc = np.full(256, 7, dtype=np.int32)
    for k, cls in enumerate(IMC_LINEAGES):
        lut_it_imc[cmap_it[cls]] = k
    lut_it_imc[cmap_it["OTHER"]] = 6
    nc = len(mat_H)
    eval_one_m242t_clf(
        "InSituType",
        np.fromfile(wdir / "it_base_a.bin", dtype=np.uint8),
        np.fromfile(wdir / "it_base_b.bin", dtype=np.uint8),
        pd.read_csv(wdir / "it_base_a_probs.tsv.gz", sep="\t")[IMC_LINEAGES].to_numpy(float),
        pd.read_csv(wdir / "it_base_b_probs.tsv.gz", sep="\t")[IMC_LINEAGES].to_numpy(float),
        np.fromfile(wdir / "it_bank_a_aq.bin", dtype=np.uint8).reshape(20, nc),
        np.fromfile(wdir / "it_bank_b_q075.bin", dtype=np.uint8).reshape(20, nc),
        np.fromfile(wdir / "it_bank_a_p2.bin", dtype=np.uint8).reshape(20, nc),
        np.fromfile(wdir / "it_bank_b_p2.bin", dtype=np.uint8).reshape(20, nc),
        np.fromfile(wdir / "it_bank_a_mitm.bin", dtype=np.uint8).reshape(20, nc),
        {q: np.fromfile(wdir / f"it_bank_b_q{int(q*100):03d}.bin", dtype=np.uint8).reshape(20, nc) for q in (0.85, 0.75, 0.65, 0.50)},
        lut_it_imc,
    )

    # CellTypist on M242T
    ct_m = train_celltypist_on_panel(hpca_raw, hpca_genes, hpca_labels, genes)
    ct_pr = run_celltypist_probe(mat_H, mat_M, genes, ct_m)
    ct_p2 = match_split_half_coverage(
        build_split_half_curve(ct_pr["q_grid"], ct_pr["a"]["cond_psi"], unresolved_frac=ct_pr["a"]["unresolved_frac"]),
        build_split_half_curve(ct_pr["q_grid"], ct_pr["b"]["cond_psi"], unresolved_frac=ct_pr["b"]["unresolved_frac"]),
        q0=0.75, pairing="both", min_q=0.05,
    )
    bH_ct, pH_ct6 = predict_celltypist_counts(mat_H, genes, ct_m, return_probs=True)
    bM_ct, pM_ct6 = predict_celltypist_counts(mat_M, genes, ct_m, return_probs=True)
    # Reorder pH_ct6 from CLASSES to IMC_LINEAGES
    ord_imc = [CLASSES.index(c) for c in IMC_LINEAGES]
    lut_py_imc = np.full(256, 7, dtype=np.int32)
    for k, cls in enumerate(IMC_LINEAGES):
        lut_py_imc[CMAP_PY[cls]] = k
    lut_py_imc[CMAP_PY["OTHER"]] = 6
    lad_M_ct = {q: run_celltypist_bank(mat_M, q, genes, ct_m, bM_ct, 910000 + int(q * 1000)) for q in (0.85, 0.75, 0.65, 0.50)}
    eval_one_m242t_clf(
        "CellTypist", bH_ct, bM_ct, pH_ct6[:, ord_imc], pM_ct6[:, ord_imc],
        run_celltypist_bank(mat_H, 0.75 * (d_M / d_H), genes, ct_m, bH_ct, 920000),
        lad_M_ct[0.75],
        run_celltypist_bank(mat_H, float(ct_p2.q_a), genes, ct_m, bH_ct, 930000),
        lad_M_ct[0.75] if abs(ct_p2.q_b - 0.75) < 1e-4 else run_celltypist_bank(mat_M, float(ct_p2.q_b), genes, ct_m, bM_ct, 940000),
        run_celltypist_bank(mat_H, q_mitm_H, genes, ct_m, bH_ct, 950000),
        lad_M_ct, lut_py_imc,
    )
    for p in (wdir / "run_a_counts.bin", wdir / "run_b_counts.bin"):
        if p.exists():
            p.unlink()


def main():
    t0 = time.time()
    # Export HPCA reference once for CellTypist
    subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(IT_R), "export_hpca", str(HPCA_RDA), str(HPCA_EXP)], check=True)
    hpca_raw, hpca_genes, hpca_labels = load_exported_hpca()
    (OUT_DIR / "celltypist_model_config.json").write_text(json.dumps(dict(
        classifier="CellTypist (L2-regularized multinomial logistic regression)",
        celltypist_version=getattr(celltypist, "__version__", "unknown"),
        reference="HPCA_6lineage_plus_OTHER (identical to SingleR & InSituType)",
        normalization="log1p(CP10k) on shared panel counts",
        hyperparameters=dict(C=1.0, max_iter=500, use_SGD=False, feature_selection=False, majority_voting=False),
    ), indent=2))

    comp_rows, null_rows, rev_rows, if_rows, m242t_rows, flip_rows = [], [], [], [], [], []
    rate_log = {}

    # 1 & 2: tTMA1 Track A & Track B
    for ds_name, tdir, has_if in [
        ("tTMA1_TrackA_Xenium_CosMx_IF", Path("/content/v3work/ttma1/Track_A_2024_Xenium_vs_CosMx_IF"), True),
        ("tTMA1_TrackB_Xenium_MERSCOPE", Path("/content/v3work/ttma1/Track_B_2023_Xenium_vs_MERSCOPE_255g"), False),
    ]:
        genes = [g for g in (tdir / "genes.txt").read_text().splitlines() if g.strip()]
        mat_a = sp.load_npz(tdir / "run_a_counts_csr.npz").toarray().astype(np.int32)
        mat_b = sp.load_npz(tdir / "run_b_counts_csr.npz").toarray().astype(np.int32)
        eval_tma_dataset(
            ds_name, tdir, mat_a, mat_b, genes,
            pd.read_parquet(tdir / "run_a_cells.parquet"), pd.read_parquet(tdir / "run_b_cells.parquet"),
            pd.read_csv(tdir / "cores.csv"),
            build_core_graphs(tdir / "run_a_edges_r25.npz"), build_core_graphs(tdir / "run_b_edges_r25.npz"),
            has_if, hpca_raw, hpca_genes, hpca_labels,
            comp_rows, null_rows, rev_rows, if_rows, rate_log,
        )

    # 3: nTMA Track N-A (already prepared by Step 3!)
    ntma_dir = Path("/content/v3work/ntma/data/Track_NA_Xenium_vs_CosMx_IF")
    if ntma_dir.exists():
        genes_n = [g for g in (ntma_dir / "genes.txt").read_text().splitlines() if g.strip()]
        ng_n = len(genes_n)
        ca_n = pd.read_parquet(ntma_dir / "run_a_cells.parquet")
        cb_n = pd.read_parquet(ntma_dir / "run_b_cells.parquet")
        ma_n = np.fromfile(ntma_dir / "run_a_counts.bin", dtype=np.int32).reshape(len(ca_n), ng_n)
        mb_n = np.fromfile(ntma_dir / "run_b_counts.bin", dtype=np.int32).reshape(len(cb_n), ng_n)
        eval_tma_dataset(
            "nTMA_TrackNA_Xenium_CosMx_IF", ntma_dir, ma_n, mb_n, genes_n, ca_n, cb_n,
            pd.read_csv(ntma_dir / "cores.csv"),
            build_core_graphs(ntma_dir / "run_a_edges_r25.npz"), build_core_graphs(ntma_dir / "run_b_edges_r25.npz"),
            True, hpca_raw, hpca_genes, hpca_labels,
            comp_rows, null_rows, rev_rows, if_rows, rate_log,
        )

    # 4: M242T Dual-Seg IMC
    eval_m242t_classifiers(hpca_raw, hpca_genes, hpca_labels, m242t_rows, flip_rows)

    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "classifiers_tma_ucommon.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT_DIR / "classifiers_tma_paired_null.csv", index=False)
    pd.DataFrame(rev_rows).to_csv(OUT_DIR / "classifiers_tma_reversals.csv", index=False)
    pd.DataFrame(if_rows).to_csv(OUT_DIR / "classifiers_tma_if_concordance.csv", index=False)
    pd.DataFrame(m242t_rows).to_csv(OUT_DIR / "classifiers_m242t_ucommon.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(OUT_DIR / "classifiers_m242t_boundary_flip.csv", index=False)
    (OUT_DIR / "classifiers_rates_summary.json").write_text(json.dumps(rate_log, indent=2))
    it_cfg_src = WORK / "tTMA1_TrackA_Xenium_CosMx_IF" / "insitutype_model_config.json"
    if it_cfg_src.exists():
        (OUT_DIR / "insitutype_model_config.json").write_text(it_cfg_src.read_text())

    print("\n=== STEP 7 CLASSIFIER INDEPENDENCE SUMMARY (InSituType & CellTypist) ===")
    print(pd.DataFrame(comp_rows)[["dataset", "classifier", "arm", "u_common_n", "icc", "mae", "red_pct", "sd_ratio_a", "sd_ratio_b"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- M242T Dual-IMC ---")
    print(pd.DataFrame(m242t_rows)[["classifier", "arm", "u_common_n", "comp_icc", "comp_mae", "r_pair_imc_poly", "r_pair_imc_indep", "sd_ratio_deep"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- M242T Boundary Flip OR ---")
    print(pd.DataFrame(flip_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSTEP 7 COMPLETE IN {(time.time()-t0)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
