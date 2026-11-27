#!/usr/bin/env python3
"""
P1-4 & P1-8 Evaluator:
  Part A: Evaluates Split-half 2s (default) & Split-half 1s on the EXACT SAME U_common
          unit sets used in audit_table1_ucommon.csv & audit_table2_ucommon.csv across:
            1. Wang HTMA r1 vs r2 (U=490, 12 TMA blocks + 3-tier paired nulls)
            2. Wang 1-yr r2 vs y24 (U=391, 9 TMA blocks + 3-tier paired nulls)
            3. Touchstone Xenium PR1 (U=113, diagnostic showing hard >=20-gene pruning collapse!)
            4. Touchstone CosMx PR1 (U=415, 24 spatial blocks)
            5. CosMx Breast q=0.25 (U=152 pooled, BR1=81, BR2=71, comp + IF_z15 + IF_cont)
  Part B: Evaluates Real-Data 15-Subtype Fine-Grained TME Audit on Wang HTMA r1 vs r2
          (53 cores, 105 cell-type pairs from hpca$main_types), stratified by subtype
          prevalence tier (Prevalent >=5%, Intermediate 2-5%, Rare <2%, All 105 pairs):
          exact P+ >= 0.90 availability survival, U_common ICC/r/rho/MAE/red_pct/SD/eta2/FDR.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"c:\Reproducution")
OUT_DIR = ROOT / "review" / "v3_p1_hardening_2026-11-27"
DRAW_DIR = OUT_DIR / "draws"
P1_DIR = ROOT / "review" / "top_phase1_2026-11-19"
P2_DIR = ROOT / "review" / "top_phase2_2026-11-19"
AUD_DIR = ROOT / "review" / "v3_unified_audit_2026-11-25"
B1B6_DIR = ROOT / "review" / "baselines_b1_b6_2026-11-20"
CLOSURE_DIR = ROOT / "review" / "closure_release_2026-11-21"
AUDIT20_DIR = ROOT / "review" / "selector_dua_rarefaction_audit_2026-11-20"
PILOT_DIR = ROOT / "review" / "p1_extrapolation_pilot_2026-11-22"

sys.path.insert(0, str(ROOT / "software" / "spatial_membership_audit" / "src"))
sys.path.insert(0, str(AUD_DIR))
sys.path.insert(0, str(P1_DIR / "scripts"))
sys.path.insert(0, str(P2_DIR / "scripts"))
sys.path.insert(0, str(B1B6_DIR))
sys.path.insert(0, str(PILOT_DIR))

import audit_table1_table2_ucommon as aud_mod
import analyze_x2_pr1 as pr1_mod
import analyze_x3_cosmx_if as x3_mod
import analyze_x5_autoq as x5_mod
import run_b4_combat as b4_mod
import p1_models as p1m
from spatial_membership_audit.coverage_matching import build_split_half_curve, match_split_half_coverage

CLASSES_6 = aud_mod.CLASSES_6
PAIR_NAMES_15 = aud_mod.PAIR_NAMES_15


def score_on_fixed_ucommon(
    benchmark: str, scope: str, arm_name: str,
    la: np.ndarray, lb: np.ndarray,
    na_full: np.ndarray, nb_full: np.ndarray,
    u_common: np.ndarray,
    cluster_designs: dict,
    pair_arr=None, tt_arr=None, pair_names=None,
) -> dict:
    d0_mask = np.isfinite(na_full) & np.isfinite(nb_full)
    ok_raw = np.isfinite(la) & np.isfinite(lb)
    raw_avail_n = int((ok_raw & d0_mask).sum())
    raw_avail_pct = 100.0 * raw_avail_n / max(1, int(d0_mask.sum()))

    na_u, nb_u = na_full[u_common], nb_full[u_common]
    diff_n = np.abs(na_u - nb_u)
    med_n = float(np.median(diff_n))
    sd_na = float(np.std(na_u, ddof=1))
    sd_nb = float(np.std(nb_u, ddof=1))

    la_u, lb_u = la[u_common], lb[u_common]
    ok_sub = np.isfinite(la_u) & np.isfinite(lb_u)
    la_v, lb_v = la_u[ok_sub], lb_u[ok_sub]
    diff_a = np.abs(la_v - lb_v)
    med_a = float(np.median(diff_a)) if len(diff_a) else math.nan
    mae_a = float(np.mean(diff_a)) if len(diff_a) else math.nan
    red_pt = 100.0 * (1.0 - med_a / med_n) if (len(diff_a) and med_n > 0) else math.nan
    icc_v = aud_mod.icc_1_1(la_v, lb_v) if len(la_v) >= 2 else math.nan
    pr_v = float(np.corrcoef(la_v, lb_v)[0, 1]) if (len(la_v) >= 2 and np.std(la_v) > 1e-12 and np.std(lb_v) > 1e-12) else math.nan
    sp_v = float(spearmanr(la_v, lb_v).statistic) if len(la_v) > 2 else math.nan
    sd_ra = float(np.std(la_v, ddof=1) / max(1e-12, sd_na)) if len(la_v) >= 2 else math.nan
    sd_rb = float(np.std(lb_v, ddof=1) / max(1e-12, sd_nb)) if len(lb_v) >= 2 else math.nan
    eta_a = aud_mod.compute_bio_eta2(la_v, pair_arr[u_common][ok_sub], tt_arr[u_common][ok_sub], pair_names) if (pair_arr is not None and len(la_v) >= 15) else math.nan
    eta_b = aud_mod.compute_bio_eta2(lb_v, pair_arr[u_common][ok_sub], tt_arr[u_common][ok_sub], pair_names) if (pair_arr is not None and len(lb_v) >= 15) else math.nan

    row = dict(
        benchmark=benchmark, scope=scope, arm=arm_name,
        d0_n=int(d0_mask.sum()), u_common_n=int(u_common.sum()), eval_n=int(ok_sub.sum()),
        raw_avail_n=raw_avail_n, raw_avail_pct=round(raw_avail_pct, 2),
        icc=round(icc_v, 4) if np.isfinite(icc_v) else np.nan,
        pearson_r=round(pr_v, 4) if np.isfinite(pr_v) else np.nan,
        spearman_rho=round(sp_v, 4) if np.isfinite(sp_v) else np.nan,
        mae=round(mae_a, 4) if np.isfinite(mae_a) else np.nan,
        med_disc=round(med_a, 4) if np.isfinite(med_a) else np.nan,
        red_pct=round(red_pt, 2) if np.isfinite(red_pt) else np.nan,
        sd_ratio_a=round(sd_ra, 3) if np.isfinite(sd_ra) else np.nan,
        sd_ratio_b=round(sd_rb, 3) if np.isfinite(sd_rb) else np.nan,
        bio_eta2_a=round(eta_a, 4) if np.isfinite(eta_a) else np.nan,
        bio_eta2_b=round(eta_b, 4) if np.isfinite(eta_b) else np.nan,
    )
    for ci_name, (u2c, bw) in cluster_designs.items():
        if ok_sub.all():
            lo, hi = aud_mod.boot_red_ci(diff_a, diff_n, u2c[u_common], bw)
            row[f"ci_lo_{ci_name}"] = round(lo, 2)
            row[f"ci_hi_{ci_name}"] = round(hi, 2)
        else:
            row[f"ci_lo_{ci_name}"] = np.nan
            row[f"ci_hi_{ci_name}"] = np.nan
    return row


def eval_table12_continuity():
    print("=== PART A: Table 1 & Table 2 Split-half 2s & 1s Continuity on EXACT U_common ===", flush=True)
    rates = json.loads((OUT_DIR / "table12_split_half_rates.json").read_text(encoding="utf-8"))
    t1_rows = []
    null_rows_all = []
    t2_rows = []

    # ------------------------------------------------------------------
    # 1. Wang HTMA r1 vs r2 (U_common = 490)
    # ------------------------------------------------------------------
    DATA_W = P1_DIR / "data" / "wang_htma"
    P1_WANG = P1_DIR / "x2_wang_htma"
    X5_DIR = P2_DIR / "x5_autoq"
    B3_W = CLOSURE_DIR / "b3_wang"
    codes_df = pd.read_csv(P1_WANG / "singler_label_codes.tsv", sep="\t")
    clookup = dict(zip(codes_df["label"], codes_df["code"]))
    unres = int(clookup.get("UNRESOLVED", 255))
    tcodes = [clookup[c] for c in CLASSES_6]
    score_cols = sorted([c for c in codes_df["label"] if c != "UNRESOLVED"])

    split_df = pd.read_csv(P1_WANG / "core_split.csv"); split_df["core"] = split_df["core"].astype(str)
    eval_df = split_df[split_df["split"] == "evaluation"].sort_values("core").reset_index(drop=True)
    cores = eval_df["core"].to_numpy(str); y_tt = eval_df["tt_r1"].to_numpy(str)
    n_cores = len(eval_df); U = n_cores * 15
    unit_core_idx = np.repeat(np.arange(n_cores), 15)
    pair_arr = np.tile(PAIR_NAMES_15, n_cores); tt_arr = np.repeat(y_tt, 15)

    meta1 = pd.read_parquet(DATA_W / "r1_eval_meta.parquet"); meta1["core"] = meta1["core"].astype(str)
    meta2 = pd.read_parquet(DATA_W / "r2_eval_meta.parquet"); meta2["core"] = meta2["core"].astype(str)
    sc1 = pd.read_csv(P1_WANG / "baseline_r1_eval_scores.tsv.gz", sep="\t")
    sc2 = pd.read_csv(P1_WANG / "baseline_r2_eval_scores.tsv.gz", sep="\t")
    b1 = sc1["label"].map(clookup).to_numpy(np.uint8)
    b2 = sc2["label"].map(clookup).to_numpy(np.uint8)

    def load_g(rn, meta, core_list):
        orig = np.full(int(meta["orig_run_idx"].max()) + 1, -1, dtype=np.int32)
        orig[meta["orig_run_idx"].to_numpy()] = np.arange(len(meta), dtype=np.int32)
        return [(cid, orig[np.load(DATA_W / f"graphs_r25/{rn}_core_{cid}.npz")["idx"]],
                 np.load(DATA_W / f"graphs_r25/{rn}_core_{cid}.npz")["edges"]) for cid in core_list]

    g1, g2 = load_g("r1", meta1, cores), load_g("r2", meta2, cores)

    def eval_hard(codes, gl):
        return np.concatenate([x5_mod.hard_ratio_stats(codes[ix], e, len(ix), tcodes)[0] for _, ix, e in gl])

    def eval_bank(bank, gl):
        B = len(bank); out = np.full((B, len(gl) * 15), np.nan)
        for rep in range(B):
            for ci, (_, ix, e) in enumerate(gl):
                out[rep, ci * 15:(ci + 1) * 15] = x5_mod.hard_ratio_stats(bank[rep, ix], e, len(ix), tcodes)[0]
        return out

    # Reconstruct exact U_common=490 mask from audit_wang_r1_r2
    nr0, nr1 = eval_hard(b1, g1), eval_hard(b2, g2)
    na_log, nb_log = aud_mod.safe_log(nr0), aud_mod.safe_log(nr1)
    p1_s = pd.read_csv(B3_W / "preds_r1_singler.tsv", sep="\t")
    p2_s = pd.read_csv(B3_W / "preds_r2_singler.tsv", sep="\t")
    u_w12 = np.isfinite(na_log) & np.isfinite(nb_log)
    for col in ("rescaleBatches", "harmony_cluster", "scvi_expr", "scvi_cluster"):
        c1 = np.array([clookup.get(s, unres) for s in p1_s[col].to_numpy(str)], dtype=np.uint8)
        c2 = np.array([clookup.get(s, unres) for s in p2_s[col].to_numpy(str)], dtype=np.uint8)
        u_w12 &= np.isfinite(aud_mod.safe_log(eval_hard(c1, g1))) & np.isfinite(aud_mod.safe_log(eval_hard(c2, g2)))
    d1_s75 = np.fromfile(P1_WANG / "singler_draws/r1_sym075.bin", dtype=np.uint8).reshape(20, len(meta1))
    d2_aq1 = np.fromfile(X5_DIR / "draws/r2_autoq_medpos.bin", dtype=np.uint8).reshape(20, len(meta2))
    d2_aq2 = np.fromfile(X5_DIR / "draws/r2_autoq_medpos_both.bin", dtype=np.uint8).reshape(20, len(meta2))
    r1_75_m = aud_mod.safe_nanmean_axis0(eval_bank(d1_s75, g1))
    r2_aq1_m = aud_mod.safe_nanmean_axis0(eval_bank(d2_aq1, g2))
    r2_aq2_m = aud_mod.safe_nanmean_axis0(eval_bank(d2_aq2, g2))
    u_w12 &= np.isfinite(aud_mod.safe_log(r1_75_m)) & np.isfinite(aud_mod.safe_log(r2_aq1_m)) & np.isfinite(aud_mod.safe_log(r2_aq2_m))
    # Also MITM mask
    d2_s50 = np.fromfile(P1_WANG / "singler_draws/r2_sym050.bin", dtype=np.uint8).reshape(20, len(meta2))
    u_w12 &= np.isfinite(aud_mod.safe_log(aud_mod.safe_nanmean_axis0(eval_bank(d2_s50, g2))))
    assert int(u_w12.sum()) == 490, f"Expected Wang r1/r2 U_common=490, got {int(u_w12.sum())}"

    d2_sh = np.fromfile(DRAW_DIR / "wang_r2_eval_sh_q0100.bin", dtype=np.uint8).reshape(20, len(meta2))
    r2_sh_m = aud_mod.safe_nanmean_axis0(eval_bank(d2_sh, g2))

    tma_blks = aud_mod.make_wang_tma_blocks(meta1, cores, nx=4, ny=3)
    bw_tma = aud_mod.cluster_boot_weights(tma_blks, n_boot=2000, seed=2026112501)
    sh_arms_w12 = {
        "Split_half_2s_default": (aud_mod.safe_log(r1_75_m), aud_mod.safe_log(r2_sh_m)),
        "Split_half_1s": (na_log, aud_mod.safe_log(r2_sh_m)),
    }
    for aname, (la, lb) in sh_arms_w12.items():
        row = score_on_fixed_ucommon(
            "Wang_HTMA_r1_vs_r2", "eval_53cores_12blocks", aname, la, lb, na_log, nb_log, u_w12,
            {"tma_block": (unit_core_idx, bw_tma)}, pair_arr, tt_arr, PAIR_NAMES_15
        )
        row["clamped"] = bool(rates["Wang_HTMA_r1_vs_r2"]["split_half_2s" if "2s" in aname else "split_half_1s"]["clamped"])
        row["q_a"] = rates["Wang_HTMA_r1_vs_r2"]["split_half_2s" if "2s" in aname else "split_half_1s"]["q_a"]
        row["q_b"] = rates["Wang_HTMA_r1_vs_r2"]["split_half_2s" if "2s" in aname else "split_half_1s"]["q_b"]
        t1_rows.append(row)
    null_rows_all.extend(aud_mod.run_paired_zero_effect_tests("Wang_HTMA_r1_vs_r2", sh_arms_w12, u_w12, n_cores, y_tt, PAIR_NAMES_15))
    print("  [1/5] Wang r1 vs r2 done (U=490)", flush=True)

    # ------------------------------------------------------------------
    # 2. Wang 1-yr r2 vs y24 (U_common = 391)
    # ------------------------------------------------------------------
    my = pd.read_parquet(DATA_W / "y24_eval_meta.parquet"); my["core"] = my["core"].astype(str)
    common_cores = set(meta2["core"]) & set(my["core"])
    eval_y = eval_df[eval_df["core"].isin(common_cores)].sort_values("core").reset_index(drop=True)
    cores_y = eval_y["core"].to_numpy(str); y_tt_y = eval_y["tt_r1"].to_numpy(str)
    ny_cores = len(eval_y)
    g2y, gy = load_g("r2", meta2, cores_y), load_g("y24", my, cores_y)
    scy = pd.read_csv(P1_WANG / "baseline_y24_eval_scores.tsv.gz", sep="\t")
    by = scy["label"].map(clookup).to_numpy(np.uint8)
    nr2y, nry = eval_hard(b2, g2y), eval_hard(by, gy)
    l2y_nv, ly_nv = aud_mod.safe_log(nr2y), aud_mod.safe_log(nry)

    d2_s75 = np.fromfile(P1_WANG / "singler_draws/r2_sym075.bin", dtype=np.uint8).reshape(20, len(meta2))
    dy_s75 = np.fromfile(P1_WANG / "singler_draws/y24_sym075.bin", dtype=np.uint8).reshape(20, len(my))
    d2_aq1y = np.fromfile(X5_DIR / "draws/r2_y24_autoq_medpos.bin", dtype=np.uint8).reshape(20, len(meta2))
    r2_75y = aud_mod.safe_nanmean_axis0(eval_bank(d2_s75, g2y))
    ry_75y = aud_mod.safe_nanmean_axis0(eval_bank(dy_s75, gy))
    r2_aq1y = aud_mod.safe_nanmean_axis0(eval_bank(d2_aq1y, g2y))
    u_wy = (np.isfinite(l2y_nv) & np.isfinite(ly_nv) & np.isfinite(aud_mod.safe_log(r2_75y)) &
            np.isfinite(aud_mod.safe_log(ry_75y)) & np.isfinite(aud_mod.safe_log(r2_aq1y)))
    assert int(u_wy.sum()) == 391, f"Expected Wang r2/y24 U_common=391, got {int(u_wy.sum())}"

    r2_sh_y = aud_mod.safe_nanmean_axis0(eval_bank(d2_sh, g2y))
    tma_blks_y = aud_mod.make_wang_tma_blocks(meta2, cores_y, nx=4, ny=3)
    bw_tma_y = aud_mod.cluster_boot_weights(tma_blks_y, n_boot=2000, seed=2026112501)
    ucy_idx = np.repeat(np.arange(ny_cores), 15)
    sh_arms_wy = {
        "Split_half_2s_default": (aud_mod.safe_log(r2_sh_y), aud_mod.safe_log(ry_75y)),
        "Split_half_1s": (aud_mod.safe_log(r2_sh_y), ly_nv),
    }
    for aname, (la, lb) in sh_arms_wy.items():
        row = score_on_fixed_ucommon(
            "Wang_HTMA_r2_vs_y24", f"eval_{ny_cores}cores_{len(np.unique(tma_blks_y))}blocks", aname,
            la, lb, l2y_nv, ly_nv, u_wy, {"tma_block": (ucy_idx, bw_tma_y)},
            np.tile(PAIR_NAMES_15, ny_cores), np.repeat(y_tt_y, 15), PAIR_NAMES_15
        )
        row["clamped"] = True
        row["q_a"] = rates["Wang_HTMA_r2_vs_y24"]["split_half_2s" if "2s" in aname else "split_half_1s"]["q_a"]
        row["q_b"] = rates["Wang_HTMA_r2_vs_y24"]["split_half_2s" if "2s" in aname else "split_half_1s"]["q_b"]
        t1_rows.append(row)
    null_rows_all.extend(aud_mod.run_paired_zero_effect_tests("Wang_HTMA_r2_vs_y24", sh_arms_wy, u_wy, ny_cores, y_tt_y, PAIR_NAMES_15))
    print("  [2/5] Wang r2 vs y24 done (U=391)", flush=True)

    # ------------------------------------------------------------------
    # 3. Touchstone Xenium PR1 (U_common = 113; hard >=20-gene pruning diagnostic)
    # ------------------------------------------------------------------
    PR1 = ROOT / "review" / "ncs_g2_pr1_2026-11-09"
    G2_STR = ROOT / "review" / "ncs_g2_xenium_structure_2026-11-09"
    OUT_PR1 = P1_DIR / "x2_touchstone_pr1"
    gsms_x = ("GSM8513829", "GSM8513837")
    gpaths_x = sorted(p for p in (PR1 / "grid").glob("GSM8513829_*.npz") if not p.name.endswith("_grid_assignment.npz"))
    gmeta_x, grids_x = [], {0: [], 1: []}
    for p in gpaths_x:
        gx, gy_i = map(int, p.stem.split("_")[-2:])
        gkey = f"{gx},{gy_i}"; blk = f"{gx // 2},{gy_i // 2}"
        for si, gsm in enumerate(gsms_x):
            z = np.load(PR1 / "grid" / f"{gsm}_{gx}_{gy_i}.npz")
            grids_x[si].append((gkey, blk, z["indices"], z["edges"]))
        for _, _, pname in pr1_mod.XEN_PAIRS:
            gmeta_x.append(dict(grid=gkey, block=blk, pair=pname))
    mdf_x = pd.DataFrame(gmeta_x)
    blocks_x = sorted(mdf_x["block"].unique()); bmap_x = {b: i for i, b in enumerate(blocks_x)}
    bi_x = np.array([bmap_x[b] for b in mdf_x["block"]])
    rng_bx = np.random.default_rng(2026111905)
    bw_x = np.array([np.bincount(rng_bx.integers(0, len(blocks_x), size=len(blocks_x)), minlength=len(blocks_x)) for _ in range(2000)], dtype=np.int16)

    tmap = {t: k + 1 for k, t in enumerate(pr1_mod.XEN_TARGETS)}
    sc_x = {si: pd.read_csv(OUT_PR1 / f"{gsm}_baseline_scores.tsv.gz", sep="\t") for si, gsm in enumerate(gsms_x)}
    base_x = {si: sc_x[si]["label"].map(lambda v: tmap.get(v, 0)).to_numpy(np.uint8) for si in (0, 1)}

    def eval_x(si, codes):
        return np.concatenate([pr1_mod.hard_stats(codes[ix], e, len(ix), [1, 2, 3], pr1_mod.XEN_PAIRS)[0] for _, _, ix, e in grids_x[si]])

    xnr0, xnr1 = eval_x(0, base_x[0]), eval_x(1, base_x[1])
    xl0_nv, xl1_nv = aud_mod.safe_log(xnr0), aud_mod.safe_log(xnr1)
    old_cmp = pd.read_csv(G2_STR / "comparisons.csv")
    u_x = np.isfinite(xl0_nv) & np.isfinite(xl1_nv)
    for oarm in ("MC200_total", "scuttle"):
        sub = old_cmp.loc[(old_cmp["arm"] == oarm) & (old_cmp["grid"] != "WHOLE")].set_index(["grid", "pair"]).loc[pd.MultiIndex.from_frame(mdf_x[["grid", "pair"]])]
        u_x &= np.isfinite(aud_mod.safe_log(sub["R_STJ"].to_numpy(float))) & np.isfinite(aud_mod.safe_log(sub["R_WCM"].to_numpy(float)))
    for si, gsm in enumerate(gsms_x):
        arr = np.fromfile(OUT_PR1 / f"{gsm}_sym075.bin", dtype=np.uint8).reshape(20, len(sc_x[si]))
        rm = np.vstack([eval_x(si, arr[r]) for r in range(20)])
        u_x &= np.isfinite(aud_mod.safe_log(aud_mod.safe_nanmean_axis0(rm)))
        if si == 1:
            x_wcm75 = aud_mod.safe_nanmean_axis0(rm)
    assert int(u_x.sum()) == 113, f"Expected Xenium PR1 U_common=113, got {int(u_x.sum())}"

    arr_xsh = np.fromfile(DRAW_DIR / "xenium_pr1_stj_sh_q0100.bin", dtype=np.uint8).reshape(20, len(sc_x[0]))
    x_stj_sh = aud_mod.safe_nanmean_axis0(np.vstack([eval_x(0, arr_xsh[r]) for r in range(20)]))
    row_x2s = score_on_fixed_ucommon(
        "Touchstone_Xenium_PR1", f"85grids_{len(blocks_x)}blocks", "Split_half_2s_unfallback_clamped",
        aud_mod.safe_log(x_stj_sh), aud_mod.safe_log(x_wcm75), xl0_nv, xl1_nv, u_x, {"spatial_block": (bi_x, bw_x)}
    )
    row_x2s["clamped"] = True; row_x2s["q_a"] = 0.10; row_x2s["q_b"] = 0.75
    t1_rows.append(row_x2s)
    print(f"  [3/5] Xenium PR1 pruning-inversion diagnostic done: avail={row_x2s['raw_avail_pct']}% (confirms curve-inversion fallback needed!)", flush=True)

    # ------------------------------------------------------------------
    # 4. Touchstone CosMx PR1 (U_common = 415)
    # ------------------------------------------------------------------
    gsms_c = ("GSM8513808", "GSM8513827")
    codes_c = pd.read_csv(OUT_PR1 / "cosmx_label_codes.tsv", sep="\t")
    clookup_c = dict(zip(codes_c["label"], codes_c["code"]))
    tcodes_c = [clookup_c[c] for c in pr1_mod.COS_CLASSES]
    gpaths_c = sorted(p for p in (PR1 / "grid").glob("GSM8513808_*.npz") if not p.name.endswith("_grid_assignment.npz"))
    gmeta_c, grids_c = [], {0: [], 1: []}
    for p in gpaths_c:
        gx, gy_i = map(int, p.stem.split("_")[-2:])
        gkey = f"{gx},{gy_i}"; blk = f"{gx // 2},{gy_i // 2}"
        for si, gsm in enumerate(gsms_c):
            z = np.load(PR1 / "grid" / f"{gsm}_{gx}_{gy_i}.npz")
            grids_c[si].append((gkey, blk, z["indices"], z["edges"]))
        for _, _, pname in pr1_mod.COS_PAIRS:
            gmeta_c.append(dict(grid=gkey, block=blk, pair=pname))
    mdf_c = pd.DataFrame(gmeta_c)
    blocks_c = sorted(mdf_c["block"].unique()); bmap_c = {b: i for i, b in enumerate(blocks_c)}
    bi_c = np.array([bmap_c[b] for b in mdf_c["block"]])
    rng_bc = np.random.default_rng(2026111906)
    bw_c = np.array([np.bincount(rng_bc.integers(0, len(blocks_c), size=len(blocks_c)), minlength=len(blocks_c)) for _ in range(2000)], dtype=np.int16)

    sc_c = {si: pd.read_csv(OUT_PR1 / f"{gsm}_cosmx_baseline_scores.tsv.gz", sep="\t") for si, gsm in enumerate(gsms_c)}
    base_c = {si: sc_c[si]["label"].map(clookup_c).to_numpy(np.uint8) for si in (0, 1)}

    def eval_c(si, codes):
        return np.concatenate([pr1_mod.hard_stats(codes[ix], e, len(ix), tcodes_c, pr1_mod.COS_PAIRS)[0] for _, _, ix, e in grids_c[si]])

    def eval_c_bank(si, path):
        arr = np.fromfile(path, dtype=np.uint8).reshape(20, len(sc_c[si]))
        return aud_mod.safe_nanmean_axis0(np.vstack([eval_c(si, arr[r]) for r in range(20)]))

    cnr0, cnr1 = eval_c(0, base_c[0]), eval_c(1, base_c[1])
    cl0_nv, cl1_nv = aud_mod.safe_log(cnr0), aud_mod.safe_log(cnr1)
    c_s75_0 = eval_c_bank(0, OUT_PR1 / "GSM8513808_sym075.bin")
    c_s75_1 = eval_c_bank(1, OUT_PR1 / "GSM8513827_sym075.bin")
    c_aq1_0 = eval_c_bank(0, P2_DIR / "x3_cosmx_if/draws/GSM8513808_autoq_medpos_1sided.bin")
    c_aq2_0 = eval_c_bank(0, P2_DIR / "x3_cosmx_if/draws/GSM8513808_autoq_medpos_both.bin")
    u_c = (np.isfinite(cl0_nv) & np.isfinite(cl1_nv) & np.isfinite(aud_mod.safe_log(c_s75_0)) &
           np.isfinite(aud_mod.safe_log(c_s75_1)) & np.isfinite(aud_mod.safe_log(c_aq1_0)) & np.isfinite(aud_mod.safe_log(c_aq2_0)))
    assert int(u_c.sum()) == 415, f"Expected CosMx PR1 U_common=415, got {int(u_c.sum())}"

    c_sh2s_0 = eval_c_bank(0, DRAW_DIR / "cosmx_pr1_stj_sh2s.bin")
    c_sh1s_0 = eval_c_bank(0, DRAW_DIR / "cosmx_pr1_stj_sh1s.bin")
    for aname, la, lb, rkey in [
        ("Split_half_2s_default", aud_mod.safe_log(c_sh2s_0), aud_mod.safe_log(c_s75_1), "split_half_2s"),
        ("Split_half_1s", aud_mod.safe_log(c_sh1s_0), cl1_nv, "split_half_1s"),
    ]:
        row = score_on_fixed_ucommon(
            "Touchstone_CosMx_PR1", f"{len(gpaths_c)}grids_{len(blocks_c)}blocks", aname,
            la, lb, cl0_nv, cl1_nv, u_c, {"spatial_block": (bi_c, bw_c)}
        )
        row["clamped"] = False
        row["q_a"] = rates["Touchstone_CosMx_PR1"][rkey]["q_a"]
        row["q_b"] = rates["Touchstone_CosMx_PR1"][rkey]["q_b"]
        t1_rows.append(row)
    print("  [4/5] CosMx PR1 done (U=415)", flush=True)

    # ------------------------------------------------------------------
    # 5. CosMx Breast q=0.25 (U_common = 152 pooled, BR1=81, BR2=71)
    # ------------------------------------------------------------------
    X3_DIR = P2_DIR / "x3_cosmx_if"
    br_codes = pd.read_csv(X3_DIR / "cosmx_br_label_codes.tsv", sep="\t")
    B3_OUT = B1B6_DIR / "b3_outputs"
    slides = {}
    for ds_name, gsm in [("cosmx_br1", "GSM8513809"), ("cosmx_br2", "GSM8513811")]:
        c = x3_mod.build_breast_run_cache(gsm, br_codes, include_stress=True)
        F = len(c["fovs"])
        def bank3(arr):
            return aud_mod.safe_nanmean_axis0(np.vstack([x3_mod.eval_3axes_hard(c, arr[b])[0].ravel() for b in range(20)]))
        R_dp = x3_mod.eval_3axes_hard(c, c["base_code"])[0].ravel()
        R_sh = x3_mod.eval_3axes_hard(c, c["sh_base_code"])[0].ravel()
        R_sh75 = bank3(c["sh_draws_75"])
        R_aq1 = bank3(c["dp_autoq_1s"])
        R_aq2 = bank3(c["dp_autoq_bt"])
        R_dp50 = bank3(np.fromfile(PILOT_DIR / f"draws/{gsm}_deep_sym050.bin", dtype=np.uint8).reshape(20, len(c["meta"])))
        R_sh2s = bank3(np.fromfile(DRAW_DIR / f"{gsm}_deep_sh2s.bin", dtype=np.uint8).reshape(20, len(c["meta"])))
        R_sh1s = bank3(np.fromfile(DRAW_DIR / f"{gsm}_deep_sh1s.bin", dtype=np.uint8).reshape(20, len(c["meta"])))
        # Reconstruct exact U_common mask (all Table 2 arms + B3)
        s_dp_df = pd.read_csv(B3_OUT / ds_name / "preds_Deep_singler.tsv", sep="\t")
        s_sh_df = pd.read_csv(B3_OUT / ds_name / "preds_Shallow_singler.tsv", sep="\t")
        cmap = c["cmap"]; unres_b = cmap.get("UNRESOLVED", 7)
        b3_ok = np.ones(F * 3, dtype=bool)
        for col in ("resolvi_expr", "scvi_expr", "harmony_cluster", "rescaleBatches"):
            a1 = x3_mod.eval_3axes_hard(c, np.array([cmap.get(s, unres_b) for s in s_sh_df[col].to_numpy(str)], dtype=np.uint8))[0].ravel()
            a2 = x3_mod.eval_3axes_hard(c, np.array([cmap.get(s, unres_b) for s in s_dp_df[col].to_numpy(str)], dtype=np.uint8))[0].ravel()
            b3_ok &= np.isfinite(aud_mod.safe_log(a1)) & np.isfinite(aud_mod.safe_log(a2))
        R_if15 = x3_mod.eval_3axes_if(c, "z15_primary")[0].ravel()
        R_ifct = x3_mod.eval_3axes_if(c, "continuous_w")[0].ravel()
        # MITM finite mask
        js = json.loads((X3_DIR / f"draws/{gsm}_shallow25_autoq.json").read_text())
        s_sh_v, s_dp_v = float(js["shallow25_medpos_ov"]), float(js["deep_medpos_ov"])
        q_sh = [1.00, 0.85, 0.75, 0.65, 0.50]
        def bank3_raw(arr):
            return np.vstack([x3_mod.eval_3axes_hard(c, arr[b])[0].ravel() for b in range(20)])
        sh_r_l = [
            R_sh,
            bank3_raw(np.fromfile(AUDIT20_DIR / f"draws/{gsm}_shallow25_q085.bin", dtype=np.uint8).reshape(20, len(c["meta"]))),
            bank3_raw(c["sh_draws_75"]),
            bank3_raw(np.fromfile(AUDIT20_DIR / f"draws/{gsm}_shallow25_q065.bin", dtype=np.uint8).reshape(20, len(c["meta"]))),
            bank3_raw(np.fromfile(AUDIT20_DIR / f"draws/{gsm}_shallow25_q050.bin", dtype=np.uint8).reshape(20, len(c["meta"]))),
        ]
        lad_sh = p1m.summarize_ladder_points(sh_r_l, [np.where(np.isfinite(x), 0.0, np.nan) for x in sh_r_l], q_sh, p_plus_gate=0.90)
        r_sh_mitm = p1m.fit_predict_logr_models(lad_sh, s_sh_v, 0.50 * s_dp_v, np.tile(np.arange(3), F), scale="S2_inv_sqrtd")["E3_hier_eb"]
        b3_ok &= (np.isfinite(aud_mod.safe_log(R_if15)) & np.isfinite(aud_mod.safe_log(R_ifct)) &
                  np.isfinite(aud_mod.safe_log(R_dp)) & np.isfinite(aud_mod.safe_log(R_sh)) &
                  np.isfinite(aud_mod.safe_log(R_sh75)) & np.isfinite(aud_mod.safe_log(R_aq1)) &
                  np.isfinite(aud_mod.safe_log(R_aq2)) & np.isfinite(aud_mod.safe_log(R_dp50)) &
                  np.isfinite(aud_mod.safe_log(r_sh_mitm)))
        slides[gsm] = dict(
            if15=R_if15, ifct=R_ifct, dp_nv=R_dp, sh_nv=R_sh, sh75=R_sh75,
            dp_sh2s=R_sh2s, dp_sh1s=R_sh1s, u_ok=b3_ok,
        )

    for scope_name, gsms, exp_n in [
        ("POOLED_54FOVs", ["GSM8513809", "GSM8513811"], 152),
        ("Patient_BR1_4066_27FOVs", ["GSM8513809"], 81),
        ("Patient_BR2_4399_27FOVs", ["GSM8513811"], 71),
    ]:
        u_m = np.concatenate([slides[g]["u_ok"] for g in gsms])
        assert int(u_m.sum()) == exp_n, f"{scope_name}: expected {exp_n}, got {int(u_m.sum())}"
        li_u = aud_mod.safe_log(np.concatenate([slides[g]["if15"] for g in gsms]))[u_m]
        lc_u = aud_mod.safe_log(np.concatenate([slides[g]["ifct"] for g in gsms]))[u_m]
        lsh_nv = aud_mod.safe_log(np.concatenate([slides[g]["sh_nv"] for g in gsms]))[u_m]
        ldp_nv = aud_mod.safe_log(np.concatenate([slides[g]["dp_nv"] for g in gsms]))[u_m]
        diff_n = np.abs(lsh_nv - ldp_nv); med_n = float(np.median(diff_n))
        sd_dp_nv, sd_sh_nv = float(np.std(ldp_nv, ddof=1)), float(np.std(lsh_nv, ddof=1))
        n_fov = len(u_m) // 3; fov_ids = np.repeat(np.arange(n_fov), 3)
        bw = aud_mod.cluster_boot_weights(np.arange(n_fov), n_boot=2000, seed=2026112522)

        for aname, sh_key, dp_key in [
            ("Split_half_2s_default", "sh75", "dp_sh2s"),
            ("Split_half_1s", "sh_nv", "dp_sh1s"),
        ]:
            lsh = aud_mod.safe_log(np.concatenate([slides[g][sh_key] for g in gsms]))[u_m]
            ldp = aud_mod.safe_log(np.concatenate([slides[g][dp_key] for g in gsms]))[u_m]
            diff_a = np.abs(lsh - ldp); med_a = float(np.median(diff_a))
            lo, hi = aud_mod.boot_red_ci(diff_a, diff_n, fov_ids[u_m], bw)
            lavg = 0.5 * (lsh + ldp)
            t2_rows.append(dict(
                benchmark="CosMx_Breast_X3c_q025", scope=scope_name, arm=aname, u_common_n=exp_n,
                comp_icc=round(aud_mod.icc_1_1(lsh, ldp), 4),
                comp_pearson_r=round(float(np.corrcoef(lsh, ldp)[0, 1]), 4),
                comp_mae=round(float(np.mean(diff_a)), 4),
                comp_med_disc=round(med_a, 4),
                comp_red_pct=round(100.0 * (1.0 - med_a / med_n), 2),
                comp_red_ci_lo=round(lo, 2), comp_red_ci_hi=round(hi, 2),
                truth1_icc=round(aud_mod.icc_1_1(lavg, li_u), 4),
                truth1_spearman_rho=round(float(spearmanr(lavg, li_u).statistic), 4),
                truth1_mae=round(float(0.5 * (np.mean(np.abs(lsh - li_u)) + np.mean(np.abs(ldp - li_u)))), 4),
                truth2_spearman_rho=round(float(spearmanr(lavg, lc_u).statistic), 4),
                truth2_mae=round(float(0.5 * (np.mean(np.abs(lsh - lc_u)) + np.mean(np.abs(ldp - lc_u)))), 4),
                sd_ratio_deep=round(float(np.std(ldp, ddof=1) / max(1e-12, sd_dp_nv)), 3),
                sd_ratio_shal=round(float(np.std(lsh, ddof=1) / max(1e-12, sd_sh_nv)), 3),
            ))
    print("  [5/5] CosMx Breast q=0.25 done (U=152 / 81 / 71)", flush=True)

    df_t1 = pd.DataFrame(t1_rows)
    df_t1_n = pd.DataFrame(null_rows_all)
    df_t2 = pd.DataFrame(t2_rows)
    df_t1.to_csv(OUT_DIR / "p1_table1_split_half_continuity.csv", index=False)
    df_t1_n.to_csv(OUT_DIR / "p1_table1_split_half_nulls.csv", index=False)
    df_t2.to_csv(OUT_DIR / "p1_table2_cosmx_q025_split_half.csv", index=False)
    print("\n=== Table 1 Split-half Continuity ===")
    print(df_t1.to_string(index=False))
    print("\n=== Table 1 Split-half Paired Nulls ===")
    print(df_t1_n[["benchmark", "arm", "pan_paired_t_fdr_pct", "pan_wilcoxon_fdr_pct", "within_cancer_paired_t_fdr_pct", "disjoint_50_50_welch_fdr_pct"]].to_string(index=False))
    print("\n=== Table 2 CosMx Breast q=0.25 Split-half Continuity ===")
    print(df_t2.to_string(index=False))


def eval_fine15_wang():
    print("\n=== PART B: Real-Data 15-Subtype Fine-Grained TME Audit (Wang HTMA r1 vs r2, 105 pairs) ===", flush=True)
    DATA_W = P1_DIR / "data" / "wang_htma"
    P1_WANG = P1_DIR / "x2_wang_htma"
    codes_df = pd.read_csv(OUT_DIR / "wang_fine15_label_codes.tsv", sep="\t")
    clookup = dict(zip(codes_df["label"], codes_df["code"]))
    fine15 = [
        "Epithelial_cells", "Endothelial_cells", "Fibroblasts", "Smooth_muscle_cells",
        "Tissue_stem_cells", "MSC", "Keratinocytes", "Neurons",
        "T_cells", "NK_cell", "B_cell",
        "Macrophage", "Monocyte", "DC", "Neutrophils",
    ]
    tcodes = [int(clookup[c]) for c in fine15]
    score_cols = sorted([c for c in codes_df["label"] if c != "UNRESOLVED"])
    n_c = len(fine15)
    pairs_105 = [(i, j, f"{fine15[i]}--{fine15[j]}") for i in range(n_c) for j in range(i + 1, n_c)]
    n_p = len(pairs_105)

    split_df = pd.read_csv(P1_WANG / "core_split.csv"); split_df["core"] = split_df["core"].astype(str)
    eval_df = split_df[split_df["split"] == "evaluation"].sort_values("core").reset_index(drop=True)
    cores = eval_df["core"].to_numpy(str); y_tt = eval_df["tt_r1"].to_numpy(str)
    n_cores = len(cores)

    meta1 = pd.read_parquet(DATA_W / "r1_eval_meta.parquet"); meta1["core"] = meta1["core"].astype(str)
    meta2 = pd.read_parquet(DATA_W / "r2_eval_meta.parquet"); meta2["core"] = meta2["core"].astype(str)
    sc1 = pd.read_csv(DRAW_DIR / "wang_fine15_r1_baseline_scores.tsv.gz", sep="\t")
    sc2 = pd.read_csv(DRAW_DIR / "wang_fine15_r2_baseline_scores.tsv.gz", sep="\t")
    b1 = sc1["label"].map(clookup).to_numpy(np.uint8)
    b2 = sc2["label"].map(clookup).to_numpy(np.uint8)

    # Compute baseline prevalence of each subtype on deep r2
    prev2 = {c: float(np.mean(sc2["label"] == c)) for c in fine15}
    pair_tiers = []
    for i, j, pname in pairs_105:
        mp = min(prev2[fine15[i]], prev2[fine15[j]])
        t = "Prevalent_ge5pct" if mp >= 0.05 else ("Intermediate_2_5pct" if mp >= 0.02 else "Rare_lt2pct")
        pair_tiers.append(t)
    pair_tiers = np.array(pair_tiers)
    print("  105 pair tier counts:", {t: int(np.sum(pair_tiers == t)) for t in ("Prevalent_ge5pct", "Intermediate_2_5pct", "Rare_lt2pct")})

    # Solve split-half rate from fine15 curves
    cf = json.loads((OUT_DIR / "wang_fine15_split_half_curves.json").read_text())
    c1 = build_split_half_curve(cf["r1"]["q_vals"], cf["r1"]["psi_vals"], unresolved_frac=cf["r1"]["unresolved_frac"], enforce_monotone=True)
    c2 = build_split_half_curve(cf["r2"]["q_vals"], cf["r2"]["psi_vals"], unresolved_frac=cf["r2"]["unresolved_frac"], enforce_monotone=True)
    r2s = match_split_half_coverage(c1, c2, q0=0.75, pairing="both", min_q=0.05)
    r1s = match_split_half_coverage(c1, c2, q0=1.00, pairing="onesided", min_q=0.05)
    print(f"  Fine-15 Split-half match: 2s=({r2s.q_a:.4f}, {r2s.q_b:.4f}, clamp={r2s.clamped}) | 1s=({r1s.q_a:.4f}, {r1s.q_b:.4f}, clamp={r1s.clamped})")

    def load_g(rn, meta):
        orig = np.full(int(meta["orig_run_idx"].max()) + 1, -1, dtype=np.int32)
        orig[meta["orig_run_idx"].to_numpy()] = np.arange(len(meta), dtype=np.int32)
        return [(cid, orig[np.load(DATA_W / f"graphs_r25/{rn}_core_{cid}.npz")["idx"]],
                 np.load(DATA_W / f"graphs_r25/{rn}_core_{cid}.npz")["edges"]) for cid in cores]
    g1, g2 = load_g("r1", meta1), load_g("r2", meta2)

    # Fast vectorized all-105-pair O/E computation per core
    code_to_slot = np.full(int(max(tcodes)) + 5, -1, dtype=np.int32)
    for slot, code in enumerate(tcodes):
        code_to_slot[code] = slot
    pi_idx = np.array([i for i, j, _ in pairs_105], dtype=np.int32)
    pj_idx = np.array([j for i, j, _ in pairs_105], dtype=np.int32)

    def eval_105_hard(codes, gl):
        out = np.full((n_cores, n_p), np.nan, dtype=np.float64)
        for ci, (_, ix, e) in enumerate(gl):
            n = len(ix); m = len(e)
            if n < 2 or m == 0:
                continue
            sl = code_to_slot[codes[ix]]
            cnt = np.bincount(sl[sl >= 0], minlength=n_c).astype(np.float64)
            cc = 2.0 * m / (n * (n - 1.0))
            su, sv = sl[e[:, 0]], sl[e[:, 1]]
            ok_e = (su >= 0) & (sv >= 0)
            su, sv = su[ok_e], sv[ok_e]
            obs = np.zeros((n_c, n_c), dtype=np.float64)
            if len(su) > 0:
                np.add.at(obs, (su, sv), 1.0)
                np.add.at(obs, (sv, su), 1.0)
            exp = cc * cnt[pi_idx] * cnt[pj_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[ci] = np.where(exp > 0.0, obs[pi_idx, pj_idx] / exp, np.nan)
        return out

    def eval_105_bank(bin_path, n_cells, gl):
        arr = np.fromfile(bin_path, dtype=np.uint8).reshape(20, n_cells)
        res = np.full((20, n_cores, n_p), np.nan, dtype=np.float64)
        for b in range(20):
            res[b] = eval_105_hard(arr[b], gl)
        return res

    def eval_105_soft(sc_df, gl, tau=0.05):
        mat = sc_df[score_cols].to_numpy(np.float64)
        valid = np.isfinite(mat).all(axis=1)
        P_all = np.zeros((len(mat), n_c), dtype=np.float64)
        c_idx = [score_cols.index(c) for c in fine15]
        if valid.any():
            mv = mat[valid]
            ex = np.exp((mv - mv.max(axis=1, keepdims=True)) / tau)
            P_all[valid] = (ex / ex.sum(axis=1, keepdims=True))[:, c_idx]
        out = np.full((n_cores, n_p), np.nan, dtype=np.float64)
        for ci, (_, ix, e) in enumerate(gl):
            n = len(ix); m = len(e)
            if n < 2 or m == 0:
                continue
            P = P_all[ix]
            cc = 2.0 * m / (n * (n - 1.0))
            sP = P.sum(axis=0)
            dotP = P.T @ P
            denom = sP[pi_idx] * sP[pj_idx] - dotP[pi_idx, pj_idx]
            Pu, Pv = P[e[:, 0]], P[e[:, 1]]
            obs = (Pu[:, pi_idx] * Pv[:, pj_idx] + Pu[:, pj_idx] * Pv[:, pi_idx]).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[ci] = np.where(denom > 0.0, obs / (cc * denom), np.nan)
        return out

    r1_nv = eval_105_hard(b1, g1)
    r2_nv = eval_105_hard(b2, g2)
    r1_sf = eval_105_soft(sc1, g1, 0.05)
    r2_sf = eval_105_soft(sc2, g2, 0.05)

    b_r1_75 = eval_105_bank(DRAW_DIR / "wang_fine15_r1_q075.bin", len(meta1), g1)
    b_r2_d1 = eval_105_bank(DRAW_DIR / "wang_fine15_r2_depth1s.bin", len(meta2), g2)
    b_r2_d2 = eval_105_bank(DRAW_DIR / "wang_fine15_r2_depth2s.bin", len(meta2), g2)
    b_r2_sh = eval_105_bank(DRAW_DIR / "wang_fine15_r2_sh_q010.bin", len(meta2), g2)

    def summarize_bank_p90(bank):
        pplus = np.isfinite(bank).mean(axis=0)
        mean_r = aud_mod.safe_nanmean_axis0(bank)
        gated_r = np.where(pplus >= 0.90, mean_r, np.nan)
        return pplus, mean_r, gated_r

    p_r1_75, m_r1_75, g_r1_75 = summarize_bank_p90(b_r1_75)
    p_r2_d1, m_r2_d1, g_r2_d1 = summarize_bank_p90(b_r2_d1)
    p_r2_d2, m_r2_d2, g_r2_d2 = summarize_bank_p90(b_r2_d2)
    p_r2_sh, m_r2_sh, g_r2_sh = summarize_bank_p90(b_r2_sh)

    # Naive D0 finite positive mask (53 cores x 105 pairs)
    d0_mask = np.isfinite(aud_mod.safe_log(r1_nv)) & np.isfinite(aud_mod.safe_log(r2_nv))

    arm_spec = {
        "Naive": (r1_nv, r2_nv, np.isfinite(r1_nv).astype(float), np.isfinite(r2_nv).astype(float)),
        "Soft_plugin_tau005": (r1_sf, r2_sf, np.isfinite(r1_sf).astype(float), np.isfinite(r2_sf).astype(float)),
        "Depth_only_1s": (r1_nv, g_r2_d1, np.isfinite(r1_nv).astype(float), p_r2_d1),
        "Depth_only_2s_q075": (g_r1_75, g_r2_d2, p_r1_75, p_r2_d2),
        "Split_half_1s": (r1_nv, g_r2_sh, np.isfinite(r1_nv).astype(float), p_r2_sh),
        "Split_half_2s_default": (g_r1_75, g_r2_sh, p_r1_75, p_r2_sh),
    }

    # Strict U_common across ALL 6 arms
    u_all = d0_mask.copy()
    for aname, (ra, rb, _, _) in arm_spec.items():
        u_all &= np.isfinite(aud_mod.safe_log(ra)) & np.isfinite(aud_mod.safe_log(rb))

    tma_blks = aud_mod.make_wang_tma_blocks(meta1, cores, nx=4, ny=3)
    bw_tma = aud_mod.cluster_boot_weights(tma_blks, n_boot=2000, seed=2026112501)
    unit_core_idx = np.repeat(np.arange(n_cores), n_p)
    pair_arr = np.tile([p[2] for p in pairs_105], n_cores)
    tt_arr = np.repeat(y_tt, n_p)
    tier_mat = np.broadcast_to(pair_tiers[None, :], (n_cores, n_p))

    l1_nv_flat, l2_nv_flat = aud_mod.safe_log(r1_nv).ravel(), aud_mod.safe_log(r2_nv).ravel()
    fine_rows = []
    for scope_lbl, t_mask in [
        ("All_105_fine_pairs", np.ones((n_cores, n_p), dtype=bool)),
        ("Prevalent_ge5pct", tier_mat == "Prevalent_ge5pct"),
        ("Intermediate_2_5pct", tier_mat == "Intermediate_2_5pct"),
        ("Rare_lt2pct", tier_mat == "Rare_lt2pct"),
    ]:
        d0_s = (d0_mask & t_mask).ravel()
        u_s = (u_all & t_mask).ravel()
        n_d0_s, n_u_s = int(d0_s.sum()), int(u_s.sum())
        if n_u_s < 10:
            continue
        diff_n = np.abs(l1_nv_flat[u_s] - l2_nv_flat[u_s]); med_n = float(np.median(diff_n))
        sd_n1, sd_n2 = float(np.std(l1_nv_flat[u_s], ddof=1)), float(np.std(l2_nv_flat[u_s], ddof=1))

        for aname, (ra, rb, ppa, ppb) in arm_spec.items():
            la_f, lb_f = aud_mod.safe_log(ra).ravel(), aud_mod.safe_log(rb).ravel()
            p90_surv = float(np.mean(((ppa >= 0.90) & (ppb >= 0.90)).ravel()[d0_s])) * 100.0
            fin_surv = float(np.mean((np.isfinite(la_f) & np.isfinite(lb_f))[d0_s])) * 100.0
            la_u, lb_u = la_f[u_s], lb_f[u_s]
            diff_a = np.abs(la_u - lb_u); med_a = float(np.median(diff_a))
            lo, hi = aud_mod.boot_red_ci(diff_a, diff_n, unit_core_idx[u_s], bw_tma)
            fine_rows.append(dict(
                scope=scope_lbl, arm=aname, d0_n=n_d0_s, u_common_n=n_u_s,
                pplus_ge90_of_d0_pct=round(p90_surv, 2),
                finite_logr_p90_of_d0_pct=round(fin_surv, 2),
                icc=round(aud_mod.icc_1_1(la_u, lb_u), 4),
                pearson_r=round(float(np.corrcoef(la_u, lb_u)[0, 1]), 4),
                spearman_rho=round(float(spearmanr(la_u, lb_u).statistic), 4),
                mae=round(float(np.mean(diff_a)), 4),
                med_disc=round(med_a, 4),
                red_pct=round(100.0 * (1.0 - med_a / med_n), 2),
                tma_ci_lo=round(lo, 2), tma_ci_hi=round(hi, 2),
                sd_ratio_a=round(float(np.std(la_u, ddof=1) / max(1e-12, sd_n1)), 3),
                sd_ratio_b=round(float(np.std(lb_u, ddof=1) / max(1e-12, sd_n2)), 3),
            ))

    df_f = pd.DataFrame(fine_rows)
    df_f.to_csv(OUT_DIR / "p1_wang_fine15_tme_summary.csv", index=False)
    print("\n=== Real-Data 15-Subtype Fine-Grained TME Audit (Wang HTMA r1 vs r2) ===")
    print(df_f.to_string(index=False))


if __name__ == "__main__":
    eval_table12_continuity()
    eval_fine15_wang()
