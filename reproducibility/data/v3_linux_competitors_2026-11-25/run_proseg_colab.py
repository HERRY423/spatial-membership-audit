#!/usr/bin/env python3
"""
Step 5B Remote Orchestrator on Colab Linux (v3linux):
Native Proseg v3.2.0 Cell Resegmentation + Full Downstream SingleR & Standardization Pipeline
on CosMx Breast X3c (54 FOVs: BR1 4066 GSM8513809 + BR2 4399 GSM8513811) with Same-Cell Protein IF.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import statsmodels.api as sm

from spatial_membership_audit.coverage_matching import build_split_half_curve, match_split_half_coverage
from spatial_membership_audit.extrapolation import extrapolate_ladder, summarize_thinning_ladder
from spatial_membership_audit.stability_weighting import cell_stability_weights, exact_weighted_perm_pair

WORK = Path("/content/v3work/proseg")
OUT_DIR = WORK / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ST_R = "/content/v3work/bin/micromamba"
R_ENV = "/content/v3work/mamba/st_r"
R_SCRIPT = WORK / "run_proseg_singler.R"
HPCA_RDA = Path("/content/v3work/hpca.rda")
HPCA_TXT = Path("/content/v3work/ntma/hpca_genes.txt")

SLIDES = [
    ("GSM8513809", "Patient_BR1_4066_27FOVs"),
    ("GSM8513811", "Patient_BR2_4399_27FOVs"),
]


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


def run_proseg_slide(gsm: str, sdir: Path):
    tx_gz = WORK / f"{gsm}_27fov_tx_file.csv.gz"
    zarr_tbl = sdir / "proseg-output.zarr" / "tables" / "table"
    exp_counts = sdir / "expected-counts.csv.gz"
    cell_meta = sdir / "cell-metadata.csv.gz"
    if zarr_tbl.exists() or (exp_counts.exists() and cell_meta.exists()):
        print(f"  [skip] Proseg outputs already exist for {gsm}", flush=True)
        return
    sdir.mkdir(parents=True, exist_ok=True)
    help_txt = subprocess.check_output(["proseg", "--help"], text=True)
    n_thr = str(os.cpu_count() or 8)
    cmd = ["proseg", "--cosmx", "--nthreads", n_thr]
    if "--output-path" in help_txt:
        cmd += ["--output-path", str(sdir)]
    elif "--output-prefix" in help_txt:
        cmd += ["--output-prefix", str(sdir) + "/"]
    cmd.append(str(tx_gz))
    print(f"  [proseg] Running: {' '.join(cmd)} (cwd={sdir})", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(sdir), text=True, capture_output=True)
    print(f"  [proseg] {gsm} exit={p.returncode} in {time.time()-t0:.1f}s", flush=True)
    if p.returncode != 0:
        print("STDOUT:", p.stdout[-2000:], flush=True)
        print("STDERR:", p.stderr[-2000:], flush=True)
        raise RuntimeError(f"proseg failed for {gsm}")
    print("  [proseg] generated files:", sorted(f.name for f in sdir.iterdir()), flush=True)


def eval_3axes_hard_fovs(fov_structs, codes, cmap):
    ce, cm, cs = cmap["Epithelial"], cmap["Myeloid"], cmap["Stroma"]
    cimm = np.array([cmap["T_cells"], cmap["B_cell"], cmap["Myeloid"]], dtype=np.uint8)
    F = len(fov_structs)

    def one(cv):
        out = np.full((F, 3), np.nan)
        for fi, (ix, edges, n, cc) in enumerate(fov_structs):
            if n < 2 or len(edges) == 0:
                continue
            loc = cv[ix]; ma = loc == ce; na = float(ma.sum())
            if na == 0:
                continue
            e0, e1 = edges[:, 0], edges[:, 1]
            for ai, mb in enumerate((loc == cm, np.isin(loc, cimm), loc == cs)):
                nb = float(mb.sum())
                if nb > 0:
                    out[fi, ai] = float(((ma[e0] & mb[e1]) | (ma[e1] & mb[e0])).sum()) / (cc * na * nb)
        return out.ravel()

    if codes.ndim == 1:
        return one(codes)
    st = np.vstack([one(codes[b]) for b in range(len(codes))])
    ok = np.isfinite(st).any(0)
    mu = np.full(st.shape[1], np.nan)
    mu[ok] = np.nanmean(st[:, ok], axis=0)
    return mu, st


def eval_3axes_soft_fovs(fov_structs, sc_df, score_cols, tau):
    S = sc_df[score_cols].to_numpy(np.float64)
    bad = ~np.isfinite(S).all(axis=1)
    Sc = np.where(bad[:, None], 0.0, S)
    ex = np.exp((Sc - Sc.max(axis=1, keepdims=True)) / float(tau))
    P = ex / ex.sum(axis=1, keepdims=True); P[bad] = 0.0
    ie, im, is_ = score_cols.index("Epithelial"), score_cols.index("Myeloid"), score_cols.index("Stroma")
    ii = [score_cols.index(c) for c in ("T_cells", "B_cell", "Myeloid")]
    out = np.full((len(fov_structs), 3), np.nan)
    for fi, (ix, edges, n, cc) in enumerate(fov_structs):
        if n < 2 or len(edges) == 0:
            continue
        Ps = P[ix]; wa = Ps[:, ie]; sa = float(wa.sum())
        e0, e1 = edges[:, 0], edges[:, 1]
        for ai, wb in enumerate((Ps[:, im], Ps[:, ii].sum(axis=1), Ps[:, is_])):
            sb = float(wb.sum()); d = float(sa * sb - np.dot(wa, wb))
            if sa > 0 and sb > 0 and d > 0:
                out[fi, ai] = float((wa[e0] * wb[e1] + wa[e1] * wb[e0]).sum()) / (cc * d)
    return out.ravel()


def eval_3axes_stab_w1(fov_structs, base_codes, draws_75, cmap):
    w = cell_stability_weights(base_codes, draws_75, power=1.0)
    ce, cm, cs = cmap["Epithelial"], cmap["Myeloid"], cmap["Stroma"]
    cimm = np.array([cmap["T_cells"], cmap["B_cell"], cmap["Myeloid"]], dtype=np.uint8)
    out = np.full((len(fov_structs), 3), np.nan)
    for fi, (ix, edges, n, cc) in enumerate(fov_structs):
        if n < 2 or len(edges) == 0:
            continue
        adj = sparse.csr_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
        loc = base_codes[ix]; wl = w[ix]
        wa = np.where(loc == ce, wl, 0.0)
        for ai, mb in enumerate((loc == cm, np.isin(loc, cimm), loc == cs)):
            out[fi, ai] = exact_weighted_perm_pair(adj, wa, np.where(mb, wl, 0.0)).r_weighted
    return out.ravel()


def eval_if_fovs(fov_structs, meta_df):
    F = len(fov_structs)
    r15 = np.full((F, 3), np.nan); rct = np.full((F, 3), np.nan)
    tum = meta_df["IF_PanCK_pos"].to_numpy() == 1
    bs = [
        meta_df["IF_Macrophage_z15"].to_numpy() == 1,
        meta_df["IF_Immune_z15"].to_numpy() == 1,
        meta_df["IF_Stroma_Other_z15"].to_numpy() == 1,
    ]
    wp = meta_df["w_PanCK"].to_numpy(float)
    wi = meta_df["w_Immune"].to_numpy(float)
    wbs = [meta_df["w_CD68"].to_numpy(float), wi, np.clip(1.0 - wp - wi, 0.0, 1.0)]
    for fi, (ix, edges, n, cc) in enumerate(fov_structs):
        if n < 2 or len(edges) == 0:
            continue
        e0, e1 = edges[:, 0], edges[:, 1]
        t_l = tum[ix]; na = float(t_l.sum())
        for ai, mb in enumerate(bs):
            m_l = mb[ix]; nb = float(m_l.sum())
            if na > 0 and nb > 0:
                r15[fi, ai] = float(((t_l[e0] & m_l[e1]) | (t_l[e1] & m_l[e0])).sum()) / (cc * na * nb)
        wa = wp[ix]; sa = float(wa.sum())
        for ai, wb_full in enumerate(wbs):
            wb = wb_full[ix]; sb = float(wb.sum()); d = float(sa * sb - np.dot(wa, wb))
            if sa > 0 and sb > 0 and d > 0:
                rct[fi, ai] = float((wa[e0] * wb[e1] + wa[e1] * wb[e0]).sum()) / (cc * d)
    return r15.ravel(), rct.ravel()


def main():
    t0 = time.time()
    hpca_genes = set(g.strip() for g in HPCA_TXT.read_text().splitlines() if g.strip())
    slide_results = {}
    match_diag = []
    flip_rows = []

    for gsm, plabel in SLIDES:
        print(f"\n================ Proseg v3.2.0 Pipeline: {gsm} ({plabel}) ================", flush=True)
        sdir = WORK / gsm
        run_proseg_slide(gsm, sdir)

        # Load Proseg v3.2.0 outputs (SpatialData proseg-output.zarr/tables/table OR legacy CSVs)
        zarr_tbl = sdir / "proseg-output.zarr" / "tables" / "table"
        if zarr_tbl.exists():
            import anndata as ad
            adata_p = ad.read_zarr(str(zarr_tbl))
            pmeta = adata_p.obs.reset_index(drop=True).copy()
            X_p = adata_p.X.toarray() if sparse.issparse(adata_p.X) else np.asarray(adata_p.X)
            pcounts = pd.DataFrame(X_p, columns=[str(g) for g in adata_p.var_names])
            if "fov" not in pmeta.columns and "original_cell_id" in pmeta.columns:
                # Parse c_<slide>_<fov>_<cell> -> fov integer
                def _parse_fov(s):
                    parts = str(s).split("_")
                    return int(parts[2]) if len(parts) >= 4 and parts[2].isdigit() else -1
                pmeta["fov"] = pmeta["original_cell_id"].map(_parse_fov).to_numpy(int)
        else:
            pmeta = pd.read_csv(sdir / "cell-metadata.csv.gz")
            pcounts = pd.read_csv(sdir / "expected-counts.csv.gz")
        vmeta = pd.read_parquet(WORK / f"{gsm}_meta_27fov.parquet").reset_index(drop=True)
        print(f"  Proseg cells={len(pmeta):,} (vendor={len(vmeta):,}), pmeta cols={pmeta.columns.tolist()[:12]}", flush=True)

        # Identify coordinate & FOV columns in Proseg metadata
        xcol = next(c for c in ("centroid_x", "x", "CenterX_global_px") if c in pmeta.columns)
        ycol = next(c for c in ("centroid_y", "y", "CenterY_global_px") if c in pmeta.columns)
        fcol = next((c for c in ("fov", "FOV") if c in pmeta.columns), None)

        # Detect unit scale (CosMx pixel = 0.12028 um; Proseg centroid_x/y are GLOBAL microns while vmeta x_um/y_um are FOV-local microns!)
        COSMX_UM_PER_PX = 0.12028
        vmeta["gx_um"] = vmeta["CenterX_global_px"].to_numpy(float) * COSMX_UM_PER_PX
        vmeta["gy_um"] = vmeta["CenterY_global_px"].to_numpy(float) * COSMX_UM_PER_PX
        px_span = float(pmeta[xcol].max() - pmeta[xcol].min())
        vx_um_span = float(vmeta["gx_um"].max() - vmeta["gx_um"].min())
        vx_px_span = float(vmeta["CenterX_global_px"].max() - vmeta["CenterX_global_px"].min())
        scale_um = COSMX_UM_PER_PX if abs(px_span - vx_px_span) < abs(px_span - vx_um_span) else 1.0
        pmeta["x_um"] = pmeta[xcol].to_numpy(float) * scale_um
        pmeta["y_um"] = pmeta[ycol].to_numpy(float) * scale_um
        print(f"  Proseg coord cols=({xcol},{ycol}), fcol={fcol}, scale_um={scale_um:.5f}", flush=True)

        # Match Proseg centroids to Vendor cells in GLOBAL microns (1-NN within FOV if valid fov, global 1-NN fallback for any unassigned fov)
        v_xy = vmeta[["gx_um", "gy_um"]].to_numpy(float)
        p_xy = pmeta[["x_um", "y_um"]].to_numpy(float)
        valid_fovs = set(vmeta["fov"].astype(int).unique().tolist())
        if fcol is not None:
            nn_dist = np.full(len(pmeta), np.inf)
            nn_vidx = np.full(len(pmeta), -1, dtype=int)
            for fv, df_p in pmeta.groupby(fcol):
                fv_int = int(fv)
                if fv_int not in valid_fovs:
                    continue
                vm_f = vmeta.index[vmeta["fov"] == fv_int].to_numpy()
                if len(vm_f) == 0:
                    continue
                d_f, i_f = cKDTree(v_xy[vm_f]).query(p_xy[df_p.index.to_numpy()], k=1)
                nn_dist[df_p.index.to_numpy()] = d_f
                nn_vidx[df_p.index.to_numpy()] = vm_f[i_f]
            # Global 1-NN fallback for any Proseg cell whose parsed fov was missing/-1
            unmatched = nn_vidx < 0
            if np.any(unmatched):
                d_g, i_g = cKDTree(v_xy).query(p_xy[unmatched], k=1)
                nn_dist[unmatched] = d_g
                nn_vidx[unmatched] = i_g
                pmeta.loc[unmatched, "fov"] = vmeta["fov"].to_numpy(int)[i_g]
            p_fov = pmeta["fov"].to_numpy(int)
        else:
            nn_dist, nn_vidx = cKDTree(v_xy).query(p_xy, k=1)
            p_fov = vmeta["fov"].to_numpy(int)[nn_vidx]
            pmeta["fov"] = p_fov

        m10 = float((nn_dist <= 10.0).mean())
        m5 = float((nn_dist <= 5.0).mean())
        med_d = float(np.median(nn_dist[np.isfinite(nn_dist)]))
        print(f"  Spatial Proseg->Vendor IF match: <=10um={100*m10:.1f}%, <=5um={100*m5:.1f}%, med_shift={med_d:.2f} um", flush=True)
        all_gene_cols = [c for c in pcounts.columns if c not in ("cell", "cell_id", "fov", "FOV", "index")]
        match_diag.append(dict(
            gsm=gsm, patient=plabel, n_vendor_cells=len(vmeta), n_proseg_cells=len(pmeta),
            match_le_10um_pct=100.0 * m10, match_le_5um_pct=100.0 * m5, median_centroid_shift_um=med_d,
            vendor_med_tx=float(vmeta["nCount_RNA"].median()),
            proseg_med_tx=float(pcounts[all_gene_cols].select_dtypes(include=[np.number]).sum(axis=1).median()),
        ))

        # Keep Proseg cells matched within <= 10 um (and record sensitivity mask <= 5 um)
        keep_p = (nn_dist <= 10.0) & (nn_vidx >= 0)
        pmeta_k = pmeta.loc[keep_p].copy().reset_index(drop=True)
        nn_v_k = nn_vidx[keep_p]
        for ifc in ("IF_PanCK_pos", "IF_Macrophage_z15", "IF_Immune_z15", "IF_Stroma_Other_z15", "w_PanCK", "w_CD68", "w_Immune"):
            pmeta_k[ifc] = vmeta[ifc].to_numpy()[nn_v_k]
        pmeta_k["matched_le_5um"] = nn_dist[keep_p] <= 5.0

        # Extract integer-rounded HPCA genes count matrix
        gene_cols = sorted([c for c in pcounts.columns if c in hpca_genes])
        print(f"  Matched Proseg cells (<=10um): {len(pmeta_k):,} / {len(pmeta):,}, shared HPCA genes={len(gene_cols)}", flush=True)
        assert len(pmeta_k) > 1000, f"Too few matched Proseg cells: {len(pmeta_k)}"
        mat_deep = np.rint(pcounts.loc[keep_p, gene_cols].to_numpy(np.float64)).astype(np.int32)
        mat_deep = np.maximum(mat_deep, 0)
        rng_sh = np.random.default_rng(2026112500 + (0 if gsm == "GSM8513809" else 1))
        mat_shal = rng_sh.binomial(mat_deep, 0.25).astype(np.int32)

        d_dp = float(np.median(mat_deep.sum(1)[mat_deep.sum(1) > 0]))
        d_sh = float(np.median(mat_shal.sum(1)[mat_shal.sum(1) > 0]))
        r_d = d_sh / d_dp
        q_mitm = min(1.0, math.sqrt(d_dp * d_sh) / d_dp)

        (sdir / "genes.txt").write_text("\n".join(gene_cols) + "\n")
        mat_deep.tofile(sdir / "deep_counts.bin")
        mat_shal.tofile(sdir / "shal_counts.bin")

        # Run SingleR probe -> solve frozen Split-half 2s -> run banks
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "probe", str(sdir), str(HPCA_RDA)], check=True)
        pr = json.loads((sdir / "probe.json").read_text())
        p2 = match_split_half_coverage(
            build_split_half_curve(pr["q_grid"], pr["deep"]["cond_psi"], unresolved_frac=pr["deep"]["unresolved_frac"]),
            build_split_half_curve(pr["q_grid"], pr["shal"]["cond_psi"], unresolved_frac=pr["shal"]["unresolved_frac"]),
            q0=0.75, pairing="both", min_q=0.05,
        )
        print(f"  [Proseg SingleR rates] deep={d_dp:.1f}, shal={d_sh:.1f}, r_d={r_d:.3f}, P2_2s=({p2.q_a:.3f},{p2.q_b:.3f})", flush=True)
        (sdir / "schedule.json").write_text(json.dumps({
            "deep": {"aq2": 0.75 * r_d, "p2": float(p2.q_a), "mitm": q_mitm, "q075": 0.75},
            "shal": {"q085": 0.85, "q075": 0.75, "q065": 0.65, "q050": 0.50, "p2": float(p2.q_b)},
        }, indent=2))
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "banks", str(sdir), str(HPCA_RDA)], check=True)

        # Build FOV graphs (r=25um) for Proseg cells AND Vendor cells in exact sorted FOV order!
        fovs = sorted(vmeta["fov"].unique().tolist())
        def build_fov_structs(df_m):
            out = []
            for fv in fovs:
                ix = df_m.index[df_m["fov"] == fv].to_numpy()
                n = len(ix)
                edges = cKDTree(df_m.iloc[ix][["x_um", "y_um"]].to_numpy(float)).query_pairs(25.0, output_type="ndarray") if n >= 2 else np.empty((0, 2), int)
                cc = 2.0 * len(edges) / (n * (n - 1.0)) if (n >= 2 and len(edges) > 0) else 0.0
                out.append((ix, edges, n, cc))
            return out

        p_fstructs = build_fov_structs(pmeta_k)
        v_fstructs = build_fov_structs(vmeta)

        lt = pd.read_csv(sdir / "label_codes.tsv", sep="\t")
        cmap = dict(zip(lt["label"], lt["code"]))
        sc_cols = sorted([c for c in cmap if c != "UNRESOLVED"])
        nc_p = len(pmeta_k)

        bd = np.fromfile(sdir / "base_deep.bin", dtype=np.uint8)
        bs = np.fromfile(sdir / "base_shal.bin", dtype=np.uint8)
        scd = pd.read_csv(sdir / "base_deep_scores.tsv.gz", sep="\t")
        scs = pd.read_csv(sdir / "base_shal_scores.tsv.gz", sep="\t")

        def ld_b(side, tag):
            return np.fromfile(sdir / f"bank_{side}_{tag}.bin", dtype=np.uint8).reshape(20, nc_p)

        r_if15, r_ifct = eval_if_fovs(v_fstructs, vmeta)  # Fixed vendor same-cell IF reference!
        # Vendor Naive (from uploaded baseline scores)
        v_lt = pd.read_csv(WORK / "cosmx_br_label_codes.tsv", sep="\t")
        v_cmap = dict(zip(v_lt["label"], v_lt["code"]))
        v_bd = pd.read_csv(WORK / f"{gsm}_vendor_deep_scores.tsv.gz", sep="\t")["label"].map(v_cmap).fillna(0).to_numpy(np.uint8)
        v_bs = pd.read_csv(WORK / f"{gsm}_vendor_shallow_scores.tsv.gz", sep="\t")["label"].map(v_cmap).fillna(0).to_numpy(np.uint8)
        r_v_dp = eval_3axes_hard_fovs(v_fstructs, v_bd, v_cmap)
        r_v_sh = eval_3axes_hard_fovs(v_fstructs, v_bs, v_cmap)

        # Proseg arms
        r_p_dp = eval_3axes_hard_fovs(p_fstructs, bd, cmap)
        r_p_sh = eval_3axes_hard_fovs(p_fstructs, bs, cmap)
        r_p_sf_dp = eval_3axes_soft_fovs(p_fstructs, scd, sc_cols, 0.05)
        r_p_sf_sh = eval_3axes_soft_fovs(p_fstructs, scs, sc_cols, 0.05)
        r_p_aq_dp, _ = eval_3axes_hard_fovs(p_fstructs, ld_b("deep", "aq2"), cmap)
        r_p_aq_sh, _ = eval_3axes_hard_fovs(p_fstructs, ld_b("shal", "q075"), cmap)
        r_p_p2_dp, _ = eval_3axes_hard_fovs(p_fstructs, ld_b("deep", "p2"), cmap)
        r_p_p2_sh, _ = eval_3axes_hard_fovs(p_fstructs, ld_b("shal", "p2"), cmap)
        r_p_sw_dp = eval_3axes_stab_w1(p_fstructs, bd, ld_b("deep", "q075"), cmap)
        r_p_sw_sh = eval_3axes_stab_w1(p_fstructs, bs, ld_b("shal", "q075"), cmap)

        # Proseg MITM geom EB
        r_p_mitm_dp, _ = eval_3axes_hard_fovs(p_fstructs, ld_b("deep", "mitm"), cmap)
        lad_q = [1.00, 0.85, 0.75, 0.65, 0.50]
        rb_sh = [r_p_sh] + [eval_3axes_hard_fovs(p_fstructs, ld_b("shal", f"q{int(q*100):03d}"), cmap)[1] for q in lad_q[1:]]
        zb_sh = [np.where(np.isfinite(x), 0.0, np.nan) for x in rb_sh]
        lad = summarize_thinning_ladder(rb_sh, zb_sh, lad_q, availability_threshold=0.90)
        ex = extrapolate_ladder(lad, depth_scale=d_sh, target_depth=math.sqrt(d_dp * d_sh), pair_ids=np.tile(np.arange(3), len(fovs)), method="hier_eb")
        r_p_mitm_sh = np.exp(ex.log_r_pred)

        # Heterotypic-interface boundary flip OR (Vendor vs Proseg v3.2.0!)
        for seg_lbl, df_m, fsts, b_dp, b_sh, tx_vec in [
            ("Vendor", vmeta, v_fstructs, v_bd, v_bs, vmeta["nCount_RNA"].to_numpy(float)),
            ("Proseg_v320", pmeta_k, p_fstructs, bd, bs, mat_deep.sum(1).astype(float)),
        ]:
            panck = df_m["IF_PanCK_pos"].to_numpy() == 1
            het_nb = np.zeros(len(df_m), dtype=bool)
            for ix, edges, n, _ in fsts:
                if len(edges) == 0:
                    continue
                pl = panck[ix]; het = pl[edges[:, 0]] != pl[edges[:, 1]]
                loc_h = np.zeros(n, dtype=bool); loc_h[edges[het, 0]] = True; loc_h[edges[het, 1]] = True
                het_nb[ix] = loc_h
            ok = (b_dp > 0) & (b_sh > 0)
            flp = (b_dp[ok] != b_sh[ok]).astype(float)
            hb = het_nb[ok].astype(float)
            X = sm.add_constant(np.column_stack([hb, np.log1p(tx_vec[ok])]))
            fit = sm.Logit(flp, X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_m["fov"].to_numpy()[ok]})
            ci = fit.conf_int()
            flip_rows.append(dict(
                gsm=gsm, patient=plabel, segmentation=seg_lbl, n_cells=int(ok.sum()),
                flip_interior_pct=100.0 * float(flp[hb == 0].mean()),
                flip_boundary_pct=100.0 * float(flp[hb == 1].mean()),
                boundary_or_depth_adj=float(math.exp(fit.params[1])),
                or_ci_lo=float(math.exp(ci[1, 0])), or_ci_hi=float(math.exp(ci[1, 1])),
                p_value=float(fit.pvalues[1]),
            ))

        slide_results[gsm] = dict(
            if15=r_if15, ifct=r_ifct,
            arms={
                "Vendor_Naive": (r_v_sh, r_v_dp),
                "Proseg_v320_Naive": (r_p_sh, r_p_dp),
                "Proseg_v320_Soft_tau005": (r_p_sf_sh, r_p_sf_dp),
                "Proseg_v320_Depth2s_q075": (r_p_aq_sh, r_p_aq_dp),
                "Proseg_v320_SplitHalf2s_default": (r_p_p2_sh, r_p_p2_dp),
                "Proseg_v320_MITM_geom_eb": (r_p_mitm_sh, r_p_mitm_dp),
                "Proseg_v320_Stability_w1": (r_p_sw_sh, r_p_sw_dp),
            },
        )
        # Clean large binary counts
        for p in (sdir / "deep_counts.bin", sdir / "shal_counts.bin"):
            if p.exists():
                p.unlink()

    # Evaluate strictly on ONE identical U_common per scope (POOLED 54 FOVs + BR1 4066 + BR2 4399)
    eval_rows = []
    for scope_lbl, gsms in [
        ("POOLED_54FOVs", ["GSM8513809", "GSM8513811"]),
        ("Patient_BR1_4066_27FOVs", ["GSM8513809"]),
        ("Patient_BR2_4399_27FOVs", ["GSM8513811"]),
    ]:
        l_if15 = safe_log(np.concatenate([slide_results[g]["if15"] for g in gsms]))
        l_ifct = safe_log(np.concatenate([slide_results[g]["ifct"] for g in gsms]))
        arm_logs = {
            k: (
                safe_log(np.concatenate([slide_results[g]["arms"][k][0] for g in gsms])),
                safe_log(np.concatenate([slide_results[g]["arms"][k][1] for g in gsms])),
            )
            for k in slide_results[gsms[0]]["arms"]
        }
        uc = np.isfinite(l_if15) & np.isfinite(l_ifct)
        for ls, ld in arm_logs.values():
            uc &= np.isfinite(ls) & np.isfinite(ld)
        n_uc = int(uc.sum())
        n_fov = len(l_if15) // 3
        fov_ids = np.repeat(np.arange(n_fov), 3)[uc]
        rng_b = np.random.default_rng(20261125)
        bw = np.array([np.bincount(rng_b.integers(0, n_fov, size=n_fov), minlength=n_fov) for _ in range(2000)], dtype=np.int16)

        li_u, lc_u = l_if15[uc], l_ifct[uc]
        lsh_vn, ldp_vn = arm_logs["Vendor_Naive"][0][uc], arm_logs["Vendor_Naive"][1][uc]
        diff_vn = np.abs(lsh_vn - ldp_vn); med_vn = float(np.median(diff_vn))
        lsh_pn, ldp_pn = arm_logs["Proseg_v320_Naive"][0][uc], arm_logs["Proseg_v320_Naive"][1][uc]
        diff_pn = np.abs(lsh_pn - ldp_pn); med_pn = float(np.median(diff_pn))
        sd_dp_vn, sd_sh_vn = float(np.std(ldp_vn, ddof=1)), float(np.std(lsh_vn, ddof=1))

        for ak, (ls_f, ld_f) in arm_logs.items():
            ls, ld = ls_f[uc], ld_f[uc]
            da = np.abs(ls - ld); med_a = float(np.median(da))
            red_vs_v = 100.0 * (1.0 - med_a / med_vn)
            red_vs_p = 100.0 * (1.0 - med_a / med_pn)
            bvs = [100.0 * (1.0 - weighted_med(da, bw[b][fov_ids]) / weighted_med(diff_vn, bw[b][fov_ids])) for b in range(len(bw))]
            lo, hi = float(np.quantile(bvs, 0.025)), float(np.quantile(bvs, 0.975))
            lavg = 0.5 * (ls + ld)
            eval_rows.append(dict(
                scope=scope_lbl, arm=ak, u_common_n=n_uc,
                comp_icc=icc_1_1(ls, ld), comp_pearson_r=float(np.corrcoef(ls, ld)[0, 1]),
                comp_mae=float(np.mean(da)), comp_med_disc=med_a,
                red_vs_vendor_naive_pct=red_vs_v, red_ci_lo=lo, red_ci_hi=hi,
                red_vs_proseg_naive_pct=red_vs_p,
                if15_spearman_avg=float(spearmanr(lavg, li_u).statistic),
                if15_spearman_deep=float(spearmanr(ld, li_u).statistic),
                if15_spearman_shal=float(spearmanr(ls, li_u).statistic),
                if15_mae_avg=0.5 * (float(np.mean(np.abs(ld - li_u))) + float(np.mean(np.abs(ls - li_u)))),
                ifcont_spearman_avg=float(spearmanr(lavg, lc_u).statistic),
                sd_ratio_deep_vs_vendor=float(np.std(ld, ddof=1) / max(1e-12, sd_dp_vn)),
                sd_ratio_shal_vs_vendor=float(np.std(ls, ddof=1) / max(1e-12, sd_sh_vn)),
            ))

    pd.DataFrame(eval_rows).to_csv(OUT_DIR / "proseg_v320_ucommon_summary.csv", index=False)
    pd.DataFrame(match_diag).to_csv(OUT_DIR / "proseg_v320_spatial_match_qc.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(OUT_DIR / "proseg_v320_boundary_flip_or.csv", index=False)

    print("\n=== PROSEG v3.2.0 + DOWNSTREAM PIPELINE SUMMARY (STRICT U_common) ===")
    print(pd.DataFrame(eval_rows)[["scope", "arm", "u_common_n", "comp_icc", "comp_mae", "red_vs_vendor_naive_pct", "red_vs_proseg_naive_pct", "if15_spearman_avg", "if15_mae_avg", "sd_ratio_deep_vs_vendor"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- Boundary Flip OR (Vendor vs Proseg v3.2.0) ---")
    print(pd.DataFrame(flip_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSTEP 5B COMPLETE IN {(time.time()-t0)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
