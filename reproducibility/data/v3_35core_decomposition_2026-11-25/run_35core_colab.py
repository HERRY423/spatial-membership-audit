#!/usr/bin/env python3
"""
Step 8 Remote Orchestrator on Colab Linux (v3linux):
35-Core Negative Probe Exact Reproduction + 3-Factor Factorial Ablation (Item 7).

Reproduces core_transport.py (+37.1% CosMx/Xenium, -36.3% CosMx/MERSCOPE, +25.8% MERSCOPE/Xenium,
12/35 MERSCOPE positive) and decomposes the divergence across:
  Factor 1 (Gene panel): 6_genes vs full_shared_HPCA
  Factor 2 (Labeling/scope): exclusive_2marker_1pair vs SingleR_1pair_EpiT vs SingleR_15pairs
  Factor 3 (q selector): Native vs 6g_median_q075 vs full_panel_depth_q075 vs Split_half_2s_default
"""
from __future__ import annotations

import csv
import json
import math
import struct
import subprocess
import time
import urllib.request
import zlib
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.spatial import cKDTree

from spatial_membership_audit.coverage_matching import build_split_half_curve, match_split_half_coverage

WORK = Path("/content/v3work/35core")
RAW_DIR = WORK / "raw"
OUT_DIR = WORK / "results"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ST_R = "/content/v3work/bin/micromamba"
R_ENV = "/content/v3work/mamba/st_r"
R_SCRIPT = WORK / "run_35core_singler.R"
HPCA_RDA = Path("/content/v3work/hpca.rda")
TOTAL_ZIP_BYTES = 743407896

GENES_6 = ["EPCAM", "CD3E", "CD4", "CD8A", "PPARG", "CD68"]
CLASSES = ["Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid"]
PAIRS_15 = [(i, j, f"{CLASSES[i]}--{CLASSES[j]}") for i in range(6) for j in range(i + 1, 6)]
EPI_T_IDX = [p[2] for p in PAIRS_15].index("Epithelial--T_cells")

PLATFORMS = {
    "CosMx": ("2024_cosmx_multitissue_htma", "2024_cosmx_multitissue_tTMA1.h5ad"),
    "MERSCOPE": ("2024_merscope_breast_htma", "2024_merscope_breast_tTMA1.h5ad"),
    "Xenium": ("2024_xenium_breast_htma", "2024_xenium_breast_tTMA1.h5ad"),
}


def icc_1_1(l0, l1):
    if len(l0) < 2:
        return math.nan
    w = float(np.mean((l0 - l1) ** 2) / 2.0)
    b = float(np.var(0.5 * (l0 + l1), ddof=1) - 0.5 * w)
    return float(b / (b + w)) if (b + w) > 0.0 else math.nan


def range_fetch_h5ad(url: str, inv_rec: dict) -> Path:
    name = inv_rec["name"]
    out = RAW_DIR / f"{name}.ziprange"
    size = int(inv_rec["compressed"]) + 300
    if not (out.exists() and out.stat().st_size == size):
        if out.exists():
            out.unlink()
        chunk = 1048576
        n_parts = (size + chunk - 1) // chunk
        with out.open("wb") as fout:
            for j in range(n_parts):
                lo = int(inv_rec["offset"]) + j * chunk
                hi = min(int(inv_rec["offset"]) + size, lo + chunk) - 1
                n = hi - lo + 1
                for att in range(5):
                    try:
                        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}", "Accept-Encoding": "identity"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            assert resp.status == 206
                            assert resp.headers.get("Content-Range") == f"bytes {lo}-{hi}/{TOTAL_ZIP_BYTES}"
                            b = resp.read(n + 1)
                            assert len(b) == n
                        fout.write(b)
                        break
                    except Exception:
                        if att == 4:
                            raise
                        time.sleep(1.5 * (2 ** att))
    print(f"[fetch] {name} ready ({out.stat().st_size/1e6:.2f} MB)", flush=True)
    return out


def _read_h5_str(grp, name):
    obj = grp[name]
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        cats = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in obj["categories"][:]])
        codes = obj["codes"][:]
        return np.where(codes >= 0, cats[np.maximum(codes, 0)], "")
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in obj[:]])


def extract_platform_matrix(h5_name: str, zp: Path, inv_rec: dict, df_meta: pd.DataFrame, genes_keep: list[str]):
    tmp = RAW_DIR / f"_tmp_{h5_name}"
    if tmp.exists():
        tmp.unlink()
    try:
        with zp.open("rb") as f:
            hdr = f.read(30)
            sig, _, _, method, _, _, _, _, _, fn_len, ex_len = struct.unpack("<4s5H3I2H", hdr)
            assert sig == b"PK\x03\x04" and method == 8 and f.read(fn_len).decode("utf-8") == h5_name
            _ = f.read(ex_len)
            rem = int(inv_rec["compressed"]); dec = zlib.decompressobj(-15); crc = 0; nb = 0
            with tmp.open("wb") as out_f:
                while rem > 0:
                    ch = f.read(min(1 << 20, rem))
                    if not ch:
                        break
                    rem -= len(ch)
                    d = dec.decompress(ch)
                    if d:
                        nb += len(d); crc = zlib.crc32(d, crc); out_f.write(d)
                t = dec.flush()
                if t:
                    nb += len(t); crc = zlib.crc32(t, crc); out_f.write(t)
        assert dec.eof and nb == int(inv_rec["size"]) and (crc & 0xFFFFFFFF) == int(inv_rec["crc"])

        with h5py.File(tmp, "r") as hf:
            obs = hf["obs"]
            ik = obs.attrs.get("_index", "_index")
            ik = ik.decode() if isinstance(ik, bytes) else str(ik)
            oidx = _read_h5_str(obs, ik)
            ocid = _read_h5_str(obs, "cell_id") if "cell_id" in obs else oidx
            var = hf["var"]
            vk = var.attrs.get("_index", "_index")
            vk = vk.decode() if isinstance(vk, bytes) else str(vk)
            vgenes = _read_h5_str(var, vk).tolist()
            if "gene" in var and len(set(vgenes)) < 10:
                vgenes = _read_h5_str(var, "gene").tolist()
            vmap = {g: i for i, g in enumerate(vgenes)}

            req = set(df_meta["cell_id"].tolist())
            h1, h2 = len(req & set(oidx.tolist())), len(req & set(ocid.tolist()))
            okeys = ocid if h2 > h1 else oidx
            pos_map = {cid: pos for pos, cid in enumerate(okeys) if cid in req}

            keep = df_meta["cell_id"].isin(pos_map).to_numpy()
            df_sub = df_meta.loc[keep].copy().reset_index(drop=True)
            ridx = np.array([pos_map[cid] for cid in df_sub["cell_id"].tolist()], dtype=np.int64)
            present_genes = [g for g in genes_keep if g in vmap]
            cidx = np.array([vmap[g] for g in present_genes], dtype=np.int64)

            x = hf["X"]
            if isinstance(x, h5py.Group):
                enc = str(x.attrs.get("encoding-type", ""))
                shp = tuple(int(v) for v in x.attrs["shape"])
                data = x["data"][:].astype(np.int32)
                sp = (sparse.csc_matrix((data, x["indices"][:], x["indptr"][:]), shape=shp) if "csc" in enc
                      else sparse.csr_matrix((data, x["indices"][:], x["indptr"][:]), shape=shp))
                mat = (sp[:, cidx][ridx, :] if "csc" in enc else sp[ridx, :][:, cidx]).toarray().astype(np.int32)
            else:
                n_obs = x.shape[0]
                sord = np.argsort(ridx); sr = ridx[sord]
                mat = np.empty((len(ridx), len(cidx)), dtype=np.int32)
                cur = 0
                for lo in range(0, n_obs, 25000):
                    hi = min(n_obs, lo + 25000)
                    hc = int(np.searchsorted(sr, hi, side="left"))
                    if hc > cur:
                        mat[sord[cur:hc]] = x[lo:hi, :][sr[cur:hc] - lo][:, cidx].astype(np.int32)
                    cur = hc
        return df_sub, mat, present_genes
    finally:
        if tmp.exists():
            tmp.unlink()


def excl_2m_ratio(marks: np.ndarray, edges: np.ndarray) -> float:
    epi = (marks[:, 0] > 0) & (marks[:, 1] == 0)
    tc = (marks[:, 1] > 0) & (marks[:, 0] == 0)
    n = len(marks)
    exp = len(edges) * 2.0 * int(epi.sum()) * int(tc.sum()) / (n * (n - 1.0)) if n > 1 else 0.0
    if exp <= 0.0:
        return math.nan
    a, b = edges[:, 0], edges[:, 1]
    return float(np.count_nonzero((epi[a] & tc[b]) | (tc[a] & epi[b])) / exp)


def hard_15_ratios(loc_codes: np.ndarray, edges: np.ndarray, lut: np.ndarray) -> np.ndarray:
    R = np.full(15, np.nan)
    n, m = len(loc_codes), len(edges)
    if n < 2 or m == 0:
        return R
    mapped = lut[loc_codes]
    cnt = np.bincount(mapped, minlength=7)[:6].astype(np.float64)
    c0, c1 = mapped[edges[:, 0]], mapped[edges[:, 1]]
    keep = (c0 < 6) & (c1 < 6) & (c0 != c1)
    o_mat = np.bincount(np.minimum(c0[keep], c1[keep]) * 6 + np.maximum(c0[keep], c1[keep]), minlength=36).astype(np.float64) if keep.any() else np.zeros(36)
    cc = 2.0 * m / (n * (n - 1.0))
    for pi, (a, b, _) in enumerate(PAIRS_15):
        if cnt[a] > 0 and cnt[b] > 0:
            R[pi] = o_mat[a * 6 + b] / (cc * cnt[a] * cnt[b])
    return R


def summarize_pair_comparison(r0: np.ndarray, r1: np.ndarray, nr0: np.ndarray, nr1: np.ndarray, seed: int = 20261122):
    d0 = np.isfinite(nr0) & np.isfinite(nr1) & (nr0 > 0) & (nr1 > 0)
    ok = d0 & np.isfinite(r0) & np.isfinite(r1) & (r0 > 0) & (r1 > 0)
    if ok.sum() < 3:
        return dict(n_d0=int(d0.sum()), n_ok=int(ok.sum()), icc=math.nan, pearson_r=math.nan,
                    mae=math.nan, med_disc=math.nan, red_pct_pen=math.nan, red_ci_lo=math.nan, red_ci_hi=math.nan,
                    red_pct_ok=math.nan, sd_ratio_a=math.nan, sd_ratio_b=math.nan)
    base = np.abs(np.log(nr0[d0] / nr1[d0]))
    new_pen = np.full(d0.sum(), np.inf)
    sub_ok = ok[d0]
    new_pen[sub_ok] = np.abs(np.log(r0[d0][sub_ok] / r1[d0][sub_ok]))
    med_b = float(np.median(base)); med_p = float(np.median(new_pen))
    red_pen = 100.0 * (1.0 - med_p / med_b) if (med_b > 0 and np.isfinite(med_p)) else -math.inf

    rng = np.random.default_rng(seed)
    idx = rng.integers(len(base), size=(2000, len(base)))
    den = np.median(base[idx], axis=1)
    num = np.median(new_pen[idx], axis=1)
    boot = np.where((den > 0) & np.isfinite(num), 100.0 * (1.0 - num / den), -np.inf)
    lo, hi = np.quantile(boot, [0.025, 0.975], method="inverted_cdf")

    la, lb = np.log(r0[ok]), np.log(r1[ok])
    na, nb = np.log(nr0[ok]), np.log(nr1[ok])
    return dict(
        n_d0=int(d0.sum()), n_ok=int(ok.sum()),
        icc=icc_1_1(la, lb), pearson_r=float(np.corrcoef(la, lb)[0, 1]),
        mae=float(np.mean(np.abs(la - lb))), med_disc=float(np.median(np.abs(la - lb))),
        red_pct_pen=float(red_pen), red_ci_lo=float(lo), red_ci_hi=float(hi),
        red_pct_ok=100.0 * (1.0 - float(np.median(np.abs(la - lb))) / float(np.median(np.abs(na - nb)))),
        sd_ratio_a=float(np.std(la, ddof=1) / max(1e-12, np.std(na, ddof=1))),
        sd_ratio_b=float(np.std(lb, ddof=1) / max(1e-12, np.std(nb, ddof=1))),
    )


def main():
    t0 = time.time()
    inv = json.loads((WORK / "broad_zip_inventory.json").read_text(encoding="utf-8"))
    inv_map = {m["name"]: m for m in inv["members"]}
    url = inv["url"]
    hpca_genes = set(g.strip() for g in (WORK / "hpca_genes.txt").read_text().splitlines() if g.strip())

    manifest = pd.read_csv(WORK / "ttma1_core_inventory.csv", dtype={"core": str})
    valid = manifest[manifest["labels_agree"] == True]
    cal_cores = sorted(valid[valid["historical_split"] == "calibration"]["core"].tolist(), key=int)
    test_cores = sorted(valid[valid["historical_split"] == "outside_split"]["core"].tolist(), key=int)
    all_keep_cores = set(cal_cores) | set(test_cores)
    print(f"Loaded exact manifest: {len(cal_cores)} cal cores, {len(test_cores)} outside-split test cores", flush=True)

    # Load panels, fetch H5ADs, extract full HPCA + 6-gene matrices for cal + test cores
    plat_data = {}
    marker_diag = []
    for pname, (stem, h5_name) in PLATFORMS.items():
        zp = range_fetch_h5ad(url, inv_map[h5_name])
        with (WORK / "stage" / f"gene_level_csv_{stem}.csv").open("r", encoding="utf-8-sig") as f:
            panel = {r["gene"] for r in csv.DictReader(f) if r.get("code_type") == "gene" and r.get("gene")}
        pf = pq.ParquetFile(WORK / "stage" / f"{stem}_cell_level.parquet.gzip")
        cols = ["cell_id", "core", "tissue_type", "x", "y"]
        if pname == "MERSCOPE":
            cols += [g for g in GENES_6 if g in pf.schema.names]
        df = pf.read(columns=cols).to_pandas()
        df["cell_id"] = df["cell_id"].astype(str)
        df["core"] = df["core"].astype("string").str.strip()
        df = df[df["core"].isin(all_keep_cores) & np.isfinite(df[["x", "y"]]).all(axis=1)].copy().reset_index(drop=True)
        if pname == "CosMx":
            df["x"] = df["x"].to_numpy(float) * 1000.0
            df["y"] = df["y"].to_numpy(float) * 1000.0

        genes_req = sorted((panel & hpca_genes) | set(GENES_6))
        df_j, mat_j, pgenes = extract_platform_matrix(h5_name, zp, inv_map[h5_name], df, genes_req)
        # Keep original MERSCOPE parquet 6-gene columns for exact core_transport.py replication check
        g6_idx = [pgenes.index(g) for g in GENES_6]
        mat6 = mat_j[:, g6_idx]
        if pname == "MERSCOPE" and all(g in df_j.columns for g in GENES_6):
            mat6_pq = df_j[GENES_6].to_numpy(np.int32)
        else:
            mat6_pq = mat6

        # Build test-core edges once
        test_mask = df_j["core"].isin(set(test_cores)).to_numpy()
        cal_mask = df_j["core"].isin(set(cal_cores)).to_numpy()
        cal6_tot = mat6_pq[cal_mask].sum(axis=1)
        med6_cal = float(np.median(cal6_tot[cal6_tot > 0]))

        # Per-core marker sparsity diagnostics on the 35 test cores
        ep_pos = float((mat6_pq[test_mask, 0] > 0).mean())
        cd3_pos = float((mat6_pq[test_mask, 1] > 0).mean())
        marker_diag.append(dict(
            platform=pname, n_test_cells=int(test_mask.sum()), n_hpca_genes=len(pgenes),
            cal_6g_medpos=med6_cal, epcam_pos_frac=ep_pos, cd3e_pos_frac=cd3_pos,
            epcam_mean_tx=float(mat6_pq[test_mask, 0].mean()), cd3e_mean_tx=float(mat6_pq[test_mask, 1].mean()),
        ))

        core_slices = {}
        core_edges = {}
        for c in test_cores:
            ix = np.flatnonzero(df_j["core"].to_numpy() == c)
            core_slices[c] = ix
            core_edges[c] = cKDTree(df_j.iloc[ix][["x", "y"]].to_numpy(float)).query_pairs(25.0, output_type="ndarray")

        plat_data[pname] = dict(
            df=df_j, mat=mat_j, mat6=mat6_pq, genes=pgenes, cal_mask=cal_mask, test_mask=test_mask,
            med6_cal=med6_cal, core_slices=core_slices, core_edges=core_edges,
        )
        print(f"[{pname}] joined={len(df_j):,}, hpca_genes={len(pgenes)}, cal_6g_medpos={med6_cal:.1f}, EPCAM+={100*ep_pos:.1f}%, CD3E+={100*cd3_pos:.2f}%", flush=True)

    pd.DataFrame(marker_diag).to_csv(OUT_DIR / "35core_marker_sparsity_diagnostics.csv", index=False)

    # Exact core_transport.py reproduction + 3-factor ablation across all 3 pairs!
    target_6g_global = min(plat_data[p]["med6_cal"] for p in PLATFORMS)
    ablation_rows = []
    repro_rows = []

    # Step A: Exact core_transport.py reproduction (6g_global_med_q, excl_2marker)
    excl_native = {}
    excl_6g_q075 = {}
    for pi, pname in enumerate(["Xenium", "MERSCOPE", "CosMx"]):
        pd_p = plat_data[pname]
        q_val = 0.75 * target_6g_global / pd_p["med6_cal"]
        nv, q75 = [], []
        for c in test_cores:
            ix = pd_p["core_slices"][c]; ed = pd_p["core_edges"][c]
            marks = pd_p["mat6"][ix, :2].astype(np.int64)
            nv.append(excl_2m_ratio(marks, ed))
            rng = np.random.default_rng([20261122, pi, int(c), 0])
            dr = np.array([excl_2m_ratio(rng.binomial(marks, q_val), ed) for _ in range(20)])
            fin = np.isfinite(dr)
            q75.append(float(dr[fin].mean()) if fin.sum() >= 18 else math.nan)
        excl_native[pname] = np.array(nv)
        excl_6g_q075[pname] = np.array(q75)

    # Step B: Per-pair 3-factor factorial ablation
    for p1, p2 in [("CosMx", "Xenium"), ("CosMx", "MERSCOPE"), ("MERSCOPE", "Xenium")]:
        pair_tag = f"{p1}_vs_{p2}"
        print(f"\n=== 3-Factor Ablation on 35 cores: {pair_tag} ===", flush=True)
        d1, d2 = plat_data[p1], plat_data[p2]
        shared_hpca = sorted(set(d1["genes"]) & set(d2["genes"]) & hpca_genes)
        print(f"  shared HPCA genes = {len(shared_hpca)}", flush=True)

        # Build test-core aligned matrices for SingleR on shared HPCA AND on 6-gene subset
        ix1_all = np.concatenate([d1["core_slices"][c] for c in test_cores])
        ix2_all = np.concatenate([d2["core_slices"][c] for c in test_cores])
        ptr1 = np.zeros(len(test_cores) + 1, dtype=int)
        ptr2 = np.zeros(len(test_cores) + 1, dtype=int)
        for ci, c in enumerate(test_cores):
            ptr1[ci + 1] = ptr1[ci] + len(d1["core_slices"][c])
            ptr2[ci + 1] = ptr2[ci] + len(d2["core_slices"][c])

        c1_hpca = [d1["genes"].index(g) for g in shared_hpca]
        c2_hpca = [d2["genes"].index(g) for g in shared_hpca]
        M1_full = np.ascontiguousarray(d1["mat"][ix1_all][:, c1_hpca], dtype=np.int32)
        M2_full = np.ascontiguousarray(d2["mat"][ix2_all][:, c2_hpca], dtype=np.int32)

        t1_full = M1_full.sum(axis=1); t2_full = M2_full.sum(axis=1)
        med_full_1 = float(np.median(t1_full[t1_full > 0]))
        med_full_2 = float(np.median(t2_full[t2_full > 0]))
        r_full = min(med_full_1, med_full_2) / max(med_full_1, med_full_2)
        q_full_1 = 0.75 * (r_full if med_full_1 > med_full_2 else 1.0)
        q_full_2 = 0.75 * (r_full if med_full_2 > med_full_1 else 1.0)

        # 6-gene median q (from core_transport.py)
        q_6g_1 = 0.75 * target_6g_global / d1["med6_cal"]
        q_6g_2 = 0.75 * target_6g_global / d2["med6_cal"]

        # Factor cell: Exclusive 2-marker under Native, 6g_med_q, full_panel_depth_q, and (after probe) split_half_2s
        # Run SingleR probe & banks on full_shared_HPCA first so we get Split-half 2s (q_p2_1, q_p2_2)!
        pdir_full = WORK / f"sr_{pair_tag}_full"
        pdir_full.mkdir(parents=True, exist_ok=True)
        (pdir_full / "genes.txt").write_text("\n".join(shared_hpca) + "\n")
        M1_full.tofile(pdir_full / "run_a_counts.bin")
        M2_full.tofile(pdir_full / "run_b_counts.bin")
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "probe", str(pdir_full), str(HPCA_RDA)], check=True)
        pr_full = json.loads((pdir_full / "probe.json").read_text())
        ca = build_split_half_curve(pr_full["q_grid"], pr_full["a"]["cond_psi"], unresolved_frac=pr_full["a"]["unresolved_frac"])
        cb = build_split_half_curve(pr_full["q_grid"], pr_full["b"]["cond_psi"], unresolved_frac=pr_full["b"]["unresolved_frac"])
        p2_full = match_split_half_coverage(ca, cb, q0=0.75, pairing="both", min_q=0.05)
        print(f"  {pair_tag} full-panel rates: med=({med_full_1:.1f},{med_full_2:.1f}) | 6g_q=({q_6g_1:.3f},{q_6g_2:.3f}) | "
              f"full_depth_q=({q_full_1:.3f},{q_full_2:.3f}) | split_half_2s=({p2_full.q_a:.3f},{p2_full.q_b:.3f})", flush=True)

        sched_full = {
            "a": {"q6g": q_6g_1, "qfull": q_full_1, "qp2": float(p2_full.q_a)},
            "b": {"q6g": q_6g_2, "qfull": q_full_2, "qp2": float(p2_full.q_b)},
        }
        (pdir_full / "schedule.json").write_text(json.dumps(sched_full, indent=2))
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "banks", str(pdir_full), str(HPCA_RDA)], check=True)

        # Also run SingleR on 6_genes ONLY (factor isolation: panel size vs labeling rule!)
        pdir_6g = WORK / f"sr_{pair_tag}_6g"
        pdir_6g.mkdir(parents=True, exist_ok=True)
        (pdir_6g / "genes.txt").write_text("\n".join(GENES_6) + "\n")
        np.ascontiguousarray(d1["mat6"][ix1_all], dtype=np.int32).tofile(pdir_6g / "run_a_counts.bin")
        np.ascontiguousarray(d2["mat6"][ix2_all], dtype=np.int32).tofile(pdir_6g / "run_b_counts.bin")
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "probe", str(pdir_6g), str(HPCA_RDA)], check=True)
        pr_6g = json.loads((pdir_6g / "probe.json").read_text())
        p2_6g = match_split_half_coverage(
            build_split_half_curve(pr_6g["q_grid"], pr_6g["a"]["cond_psi"], unresolved_frac=pr_6g["a"]["unresolved_frac"]),
            build_split_half_curve(pr_6g["q_grid"], pr_6g["b"]["cond_psi"], unresolved_frac=pr_6g["b"]["unresolved_frac"]),
            q0=0.75, pairing="both", min_q=0.05,
        )
        (pdir_6g / "schedule.json").write_text(json.dumps({
            "a": {"q6g": q_6g_1, "qfull": q_full_1, "qp2": float(p2_6g.q_a)},
            "b": {"q6g": q_6g_2, "qfull": q_full_2, "qp2": float(p2_6g.q_b)},
        }, indent=2))
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "banks", str(pdir_6g), str(HPCA_RDA)], check=True)

        # 1. Exclusive 2-marker under all 4 q rules (Native, 6g_med_q, full_panel_depth_q, split_half_2s)
        def run_excl_arm(q1_v, q2_v, seed_off):
            ra, rb = [], []
            for ci, c in enumerate(test_cores):
                m1 = d1["mat6"][d1["core_slices"][c], :2].astype(np.int64)
                m2 = d2["mat6"][d2["core_slices"][c], :2].astype(np.int64)
                e1, e2 = d1["core_edges"][c], d2["core_edges"][c]
                rng1 = np.random.default_rng([20261125, seed_off, ci, 1])
                rng2 = np.random.default_rng([20261125, seed_off, ci, 2])
                dr1 = np.array([excl_2m_ratio(rng1.binomial(m1, q1_v), e1) for _ in range(20)])
                dr2 = np.array([excl_2m_ratio(rng2.binomial(m2, q2_v), e2) for _ in range(20)])
                ra.append(float(dr1[np.isfinite(dr1)].mean()) if np.isfinite(dr1).sum() >= 18 else math.nan)
                rb.append(float(dr2[np.isfinite(dr2)].mean()) if np.isfinite(dr2).sum() >= 18 else math.nan)
            return np.array(ra), np.array(rb)

        e_n0, e_n1 = excl_native[p1], excl_native[p2]
        repro_m = summarize_pair_comparison(excl_6g_q075[p1], excl_6g_q075[p2], e_n0, e_n1)
        repro_rows.append(dict(comparison=f"{p1}/{p2}", q_a=q_6g_1, q_b=q_6g_2, **repro_m))

        for q_lbl, (r_a, r_b), qa, qb in [
            ("Native", (e_n0, e_n1), 1.0, 1.0),
            ("6g_median_q075", (excl_6g_q075[p1], excl_6g_q075[p2]), q_6g_1, q_6g_2),
            ("full_panel_depth_q075", run_excl_arm(q_full_1, q_full_2, 11), q_full_1, q_full_2),
            ("Split_half_2s_default", run_excl_arm(float(p2_full.q_a), float(p2_full.q_b), 12), float(p2_full.q_a), float(p2_full.q_b)),
        ]:
            m = summarize_pair_comparison(r_a, r_b, e_n0, e_n1)
            ablation_rows.append(dict(
                comparison=f"{p1}/{p2}", panel="6_genes", classifier_scope="exclusive_2marker_1pair",
                standardization=q_lbl, q_a=qa, q_b=qb, **m
            ))

        # 2. SingleR (both 6_genes panel & full_shared_HPCA panel) x (1-pair Epi--T & 15-pair) x (Native, 6g_q, full_depth_q, Split-half 2s)
        for panel_lbl, pdir, p2_obj in [
            ("6_genes", pdir_6g, p2_6g),
            (f"full_shared_HPCA_{len(shared_hpca)}g", pdir_full, p2_full),
        ]:
            lt = pd.read_csv(pdir / "label_codes.tsv", sep="\t")
            cmap = dict(zip(lt["label"], lt["code"]))
            lut = np.full(256, 6, dtype=np.int32)
            for k, cls in enumerate(CLASSES):
                lut[cmap[cls]] = k

            def eval_sr_codes(cvec, ptr, d_p):
                if cvec.ndim == 1:
                    out = np.full((len(test_cores), 15), np.nan)
                    for ci, c in enumerate(test_cores):
                        out[ci] = hard_15_ratios(cvec[ptr[ci]:ptr[ci+1]], d_p["core_edges"][c], lut)
                    return out
                B = len(cvec)
                st = np.full((B, len(test_cores), 15), np.nan)
                for b in range(B):
                    for ci, c in enumerate(test_cores):
                        st[b, ci] = hard_15_ratios(cvec[b, ptr[ci]:ptr[ci+1]], d_p["core_edges"][c], lut)
                fin = np.isfinite(st).sum(axis=0)
                mu = np.full((len(test_cores), 15), np.nan)
                ok = fin >= 18
                if ok.any():
                    mu[ok] = np.nanmean(st[:, ok], axis=0)
                return mu

            b1 = np.fromfile(pdir / "base_a.bin", dtype=np.uint8)
            b2 = np.fromfile(pdir / "base_b.bin", dtype=np.uint8)
            sr_n1 = eval_sr_codes(b1, ptr1, d1)
            sr_n2 = eval_sr_codes(b2, ptr2, d2)

            sr_arms = [("Native", sr_n1, sr_n2, 1.0, 1.0)]
            for q_lbl, btag, qa, qb in [
                ("6g_median_q075", "q6g", q_6g_1, q_6g_2),
                ("full_panel_depth_q075", "qfull", q_full_1, q_full_2),
                ("Split_half_2s_default", "qp2", float(p2_obj.q_a), float(p2_obj.q_b)),
            ]:
                k1 = np.fromfile(pdir / f"bank_a_{btag}.bin", dtype=np.uint8).reshape(20, len(b1))
                k2 = np.fromfile(pdir / f"bank_b_{btag}.bin", dtype=np.uint8).reshape(20, len(b2))
                sr_arms.append((q_lbl, eval_sr_codes(k1, ptr1, d1), eval_sr_codes(k2, ptr2, d2), qa, qb))

            for scope_lbl, slc in [("SingleR_1pair_EpiT", EPI_T_IDX), ("SingleR_15pairs", slice(None))]:
                nr0_s = sr_n1[:, slc].ravel()
                nr1_s = sr_n2[:, slc].ravel()
                for q_lbl, r1_m, r2_m, qa, qb in sr_arms:
                    m = summarize_pair_comparison(r1_m[:, slc].ravel(), r2_m[:, slc].ravel(), nr0_s, nr1_s)
                    ablation_rows.append(dict(
                        comparison=f"{p1}/{p2}", panel=panel_lbl, classifier_scope=scope_lbl,
                        standardization=q_lbl, q_a=qa, q_b=qb, **m
                    ))

    df_repro = pd.DataFrame(repro_rows)
    df_ab = pd.DataFrame(ablation_rows)
    df_repro.to_csv(OUT_DIR / "35core_exact_reproduction.csv", index=False)
    df_ab.to_csv(OUT_DIR / "35core_3factor_ablation.csv", index=False)

    print("\n=== EXACT REPRODUCTION OF core_transport.py (6g_median_q075, excl 2-marker) ===")
    print(df_repro[["comparison", "n_d0", "n_ok", "red_pct_pen", "red_ci_lo", "red_ci_hi"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== 3-FACTOR ABLATION SUMMARY (CosMx/MERSCOPE & CosMx/Xenium) ===")
    sub = df_ab[df_ab["standardization"].isin(["Native", "6g_median_q075", "full_panel_depth_q075", "Split_half_2s_default"])]
    print(sub[["comparison", "panel", "classifier_scope", "standardization", "n_d0", "n_ok", "icc", "mae", "red_pct_pen", "sd_ratio_a", "sd_ratio_b"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n35-CORE DECOMPOSITION COMPLETE IN {(time.time()-t0)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
