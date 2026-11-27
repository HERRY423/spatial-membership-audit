#!/usr/bin/env python3
"""
Step 5A Remote Orchestrator on Colab Linux:
Runs native Linux spicyR, CRAWDAD, and PANORAMIC 0.99.4 on specified track(s) under BOTH:
  - Native (unstandardized) cell-type calls
  - Paired with frozen DEFAULT_PROTOCOL_V3 (Split-half 2s, q0=0.75) matched labels
Evaluates strictly on ONE identical U_common per (track, tool) with:
  ICC, Pearson r, Spearman rho, MAE, median |delta|, red_pct (+ TMA-block 95% CI),
  SD_ratio_a/b, bio_eta2_a/b, 3-tier paired zero-effect BH-FDR, and native mixed-effects FDR.
Usage: python3 run_competitors_5a.py <track_spec...>
  e.g. python3 run_competitors_5a.py ttma1_A ttma1_B
       python3 run_competitors_5a.py ntma_NA
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

WORK = Path("/content/v3work/competitors")
OUT_DIR = WORK / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ST_R = "/content/v3work/bin/micromamba"
R_ENV = "/content/v3work/mamba/st_r"
SPICY_R = WORK / "run_spicyr_crawdad.R"
PAN_R = WORK / "run_panoramic.R"

TRACK_MAP = {
    "ttma1_A": ("tTMA1_TrackA_Xenium_CosMx", Path("/content/v3work/ttma1/Track_A_2024_Xenium_vs_CosMx_IF")),
    "ttma1_B": ("tTMA1_TrackB_Xenium_MERSCOPE", Path("/content/v3work/ttma1/Track_B_2023_Xenium_vs_MERSCOPE_255g")),
    "ntma_NA": ("nTMA_TrackNA_Xenium_CosMx", Path("/content/v3work/ntma/data/Track_NA_Xenium_vs_CosMx_IF")),
}

CLASSES = ["Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid"]
PAIR_NAMES = [f"{CLASSES[i]}--{CLASSES[j]}" for i in range(6) for j in range(i + 1, 6)]


def icc_1_1(l0, l1):
    if len(l0) < 2:
        return math.nan
    w = float(np.mean((l0 - l1) ** 2) / 2.0)
    b = float(np.var(0.5 * (l0 + l1), ddof=1) - 0.5 * w)
    return float(b / (b + w)) if (b + w) > 0.0 else math.nan


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


def compute_bio_eta2(vec, pair_arr, tt_arr):
    etas = []
    for pname in PAIR_NAMES:
        m = (pair_arr == pname) & np.isfinite(vec)
        if m.sum() < 12:
            continue
        vals, tts = vec[m], tt_arr[m]
        groups = [vals[tts == tt] for tt in np.unique(tts) if (tts == tt).sum() >= 2]
        if len(groups) >= 4:
            av = np.concatenate(groups); gr = av.mean()
            ss_t = float(np.sum((av - gr) ** 2))
            ss_b = float(sum(len(g) * (g.mean() - gr) ** 2 for g in groups))
            if ss_t > 0.0:
                etas.append(ss_b / ss_t)
    return float(np.mean(etas)) if etas else math.nan


def load_mat(path: Path, cores: list[str]) -> np.ndarray:
    df = pd.read_csv(path, dtype={"core": str}).set_index("core")
    return df.reindex(cores)[PAIR_NAMES].to_numpy(np.float64)


def eval_tool_ucommon(tlabel: str, tool: str, metric: str, arms: dict[str, tuple[np.ndarray, np.ndarray]],
                      cores: np.ndarray, tt: np.ndarray, bw_tma: np.ndarray, sum_rows: list, null_rows: list):
    n_cores = len(cores)
    d0 = np.isfinite(arms["Naive"][0]) & np.isfinite(arms["Naive"][1])
    uc = d0.copy()
    for ma, mb in arms.values():
        uc &= np.isfinite(ma) & np.isfinite(mb)
    n_uc = int(uc.sum())
    if n_uc < 10:
        print(f"  [warn] {tlabel} | {tool} ({metric}): U_common too small ({n_uc})", flush=True)
        return

    uc_flat = uc.ravel()
    uc_cidx = np.repeat(np.arange(n_cores), 15)[uc_flat]
    pair_u = np.tile(PAIR_NAMES, n_cores)[uc_flat]
    tt_u = np.repeat(tt, 15)[uc_flat]

    na_u = arms["Naive"][0].ravel()[uc_flat]
    nb_u = arms["Naive"][1].ravel()[uc_flat]
    diff_n = np.abs(na_u - nb_u); med_n = float(np.median(diff_n))
    sd_na, sd_nb = float(np.std(na_u, ddof=1)), float(np.std(nb_u, ddof=1))
    uniq_tt = sorted(np.unique(tt).tolist())

    for aname, (ma, mb) in arms.items():
        la_u, lb_u = ma.ravel()[uc_flat], mb.ravel()[uc_flat]
        da = np.abs(la_u - lb_u); med_a = float(np.median(da))
        red = 100.0 * (1.0 - med_a / med_n) if med_n > 0 else math.nan
        bvs = [100.0 * (1.0 - weighted_med(da, bw_tma[b][uc_cidx]) / weighted_med(diff_n, bw_tma[b][uc_cidx])) for b in range(len(bw_tma))]
        bvs = [v for v in bvs if np.isfinite(v)]
        lo, hi = (float(np.quantile(bvs, 0.025)), float(np.quantile(bvs, 0.975))) if bvs else (math.nan, math.nan)
        sum_rows.append(dict(
            track=tlabel, tool=tool, metric=metric, arm=aname, n_cores=n_cores,
            d0_n=int(d0.sum()), u_common_n=n_uc,
            icc=icc_1_1(la_u, lb_u), pearson_r=float(np.corrcoef(la_u, lb_u)[0, 1]),
            spearman_rho=float(spearmanr(la_u, lb_u).statistic),
            mae=float(np.mean(da)), med_disc=med_a, red_pct=red,
            red_ci_lo_tma=lo, red_ci_hi_tma=hi,
            sd_ratio_a=float(np.std(la_u, ddof=1) / max(1e-12, sd_na)),
            sd_ratio_b=float(np.std(lb_u, ddof=1) / max(1e-12, sd_nb)),
            bio_eta2_a=compute_bio_eta2(la_u, pair_u, tt_u),
            bio_eta2_b=compute_bio_eta2(lb_u, pair_u, tt_u),
        ))

        # 3-tier paired zero-effect null tests strictly on U_common
        L0, L1 = np.where(uc, ma, np.nan), np.where(uc, mb, np.nan)
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
            track=tlabel, tool=tool, metric=metric, arm=aname, u_common_n=n_uc,
            pan_paired_t_fdr_pct=100.0 * float(qt.mean()) if len(qt) else math.nan,
            pan_wilcoxon_fdr_pct=100.0 * float(qw.mean()) if len(qw) else math.nan,
            within_group_paired_t_fdr_pct=100.0 * float(wqt.mean()) if len(wqt) else math.nan,
            disjoint_50_50_welch_fdr_pct=float(np.mean(fdr_d)) if fdr_d else math.nan,
        ))


def main():
    t0 = time.time()
    keys = sys.argv[1:] or ["ttma1_A", "ttma1_B"]
    tag = "_".join(keys)
    sum_rows, null_rows, native_test_rows = [], [], []

    for k in keys:
        tlabel, tdir = TRACK_MAP[k]
        odir = WORK / tlabel
        odir.mkdir(parents=True, exist_ok=True)
        print(f"\n================ Step 5A Native Linux Competitors: {tlabel} ================", flush=True)

        # Export lightweight TSVs of (core, x_um, y_um) for system R 4.6.1 PANORAMIC
        ca = pd.read_parquet(tdir / "run_a_cells.parquet", columns=["core", "x_um", "y_um"])
        cb = pd.read_parquet(tdir / "run_b_cells.parquet", columns=["core", "x_um", "y_um"])
        ca.to_csv(odir / "_cells_a.tsv", sep="\t", index=False)
        cb.to_csv(odir / "_cells_b.tsv", sep="\t", index=False)

        # 1. Run spicyR (B=5) + CRAWDAD (B=3, multi-scale 100 & 200 um) in st_r
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(SPICY_R), "run", str(tdir), str(odir), tlabel, "5", "3"], check=True)

        # 2. Run PANORAMIC 0.99.4 (B=3) in system R 4.6.1
        subprocess.run(["/usr/bin/Rscript", str(PAN_R), str(tdir), str(odir), tlabel, "3"], check=True)

        # 3. Evaluate all 3 tools strictly on U_common
        cores_df = pd.read_csv(tdir / "cores.csv")
        cores = cores_df["core"].astype(str).to_numpy()
        cores_list = cores.tolist()
        tt = cores_df["tissue_type"].astype(str).to_numpy()
        cent = ca.groupby(ca["core"].astype(str))[["x_um", "y_um"]].median()
        xs, ys = cent.loc[cores, "x_um"].to_numpy(float), cent.loc[cores, "y_um"].to_numpy(float)
        bx = np.clip(np.floor((xs - xs.min()) / max(1e-6, xs.max() - xs.min()) * 4).astype(int), 0, 3)
        by = np.clip(np.floor((ys - ys.min()) / max(1e-6, ys.max() - ys.min()) * 3).astype(int), 0, 2)
        bw_tma = cluster_boot_weights(np.array([f"{ix}_{iy}" for ix, iy in zip(bx, by)]), 2000, 20261125)

        for tool, metric, pfx in [
            ("spicyR", "L_r25", "spicyr"),
            ("CRAWDAD", "Z_scale100", "crawdad_z100"),
            ("CRAWDAD", "meanZ_multiscale", "crawdad_auc"),
            ("PANORAMIC", "local_comp_enrichment_r25", "panoramic"),
        ]:
            arms = {
                "Naive": (load_mat(odir / f"{pfx}_naive_a.csv", cores_list), load_mat(odir / f"{pfx}_naive_b.csv", cores_list)),
                "Split_half_2s_default": (load_mat(odir / f"{pfx}_p2_mean_a.csv", cores_list), load_mat(odir / f"{pfx}_p2_mean_b.csv", cores_list)),
            }
            b1_a = odir / f"{pfx}_p2_b1_a.csv"
            if b1_a.exists():
                arms["Split_half_2s_B1"] = (load_mat(b1_a, cores_list), load_mat(odir / f"{pfx}_p2_b1_b.csv", cores_list))
            eval_tool_ucommon(tlabel, tool, metric, arms, cores, tt, bw_tma, sum_rows, null_rows)

        # Collect native mixed-effects zero-effect test results (spicyR::spicy & panoramic_meta_mv)
        sp_mx = odir / "spicyr_native_paired_test.csv"
        if sp_mx.exists():
            df_m = pd.read_csv(sp_mx)
            pcol = next((c for c in ("p.value", "pvalue", "p_value", "p") if c in df_m.columns), None)
            fcol = next((c for c in ("adj.p.value", "fdr", "p.adjust", "FDR") if c in df_m.columns), None)
            for arm_l, grp in df_m.groupby("arm"):
                # Keep off-diagonal target pairs only
                if "from" in grp.columns and "to" in grp.columns:
                    grp = grp[grp["from"] != grp["to"]]
                pv = pd.to_numeric(grp[pcol], errors="coerce").dropna().to_numpy() if pcol else np.array([])
                fv = pd.to_numeric(grp[fcol], errors="coerce").dropna().to_numpy() if fcol else (multipletests(pv, method="fdr_bh")[1] if len(pv) else np.array([]))
                native_test_rows.append(dict(
                    track=tlabel, tool="spicyR::spicy_mixed_subject_core", arm=arm_l,
                    n_tests=len(fv), n_fdr05=int((fv < 0.05).sum()), fdr05_pct=100.0 * float((fv < 0.05).mean()) if len(fv) else math.nan,
                ))
        pan_mx = odir / "panoramic_native_paired_meta.csv"
        if pan_mx.exists():
            df_p = pd.read_csv(pan_mx)
            fcol = next((c for c in ("fdr_diff", "fdr", "padj", "p_adj") if c in df_p.columns), None)
            pcol = next((c for c in ("p_diff", "pval", "p_value", "p") if c in df_p.columns), None)
            for arm_l, grp in df_p.groupby("arm"):
                fv = pd.to_numeric(grp[fcol], errors="coerce").dropna().to_numpy() if fcol else np.array([])
                if len(fv) == 0 and pcol:
                    pv = pd.to_numeric(grp[pcol], errors="coerce").dropna().to_numpy()
                    fv = multipletests(pv, method="fdr_bh")[1] if len(pv) else np.array([])
                native_test_rows.append(dict(
                    track=tlabel, tool="PANORAMIC::meta_mv_REML_patient_core", arm=arm_l,
                    n_tests=len(fv), n_fdr05=int((fv < 0.05).sum()), fdr05_pct=100.0 * float((fv < 0.05).mean()) if len(fv) else math.nan,
                ))

        # Clean temp TSVs
        for tmp_f in (odir / "_cells_a.tsv", odir / "_cells_b.tsv"):
            if tmp_f.exists():
                tmp_f.unlink()

    pd.DataFrame(sum_rows).to_csv(OUT_DIR / f"competitors_5a_ucommon_{tag}.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT_DIR / f"competitors_5a_paired_null_{tag}.csv", index=False)
    pd.DataFrame(native_test_rows).to_csv(OUT_DIR / f"competitors_5a_native_mixed_{tag}.csv", index=False)
    print("\n=== STEP 5A SUMMARY ===")
    print(pd.DataFrame(sum_rows)[["track", "tool", "metric", "arm", "u_common_n", "icc", "mae", "red_pct", "sd_ratio_a", "sd_ratio_b"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- Paired Zero-Effect Null Tests ---")
    print(pd.DataFrame(null_rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n--- Native Mixed-Effects Paired Tests ---")
    print(pd.DataFrame(native_test_rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSTEP 5A ({tag}) COMPLETE IN {(time.time()-t0)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
