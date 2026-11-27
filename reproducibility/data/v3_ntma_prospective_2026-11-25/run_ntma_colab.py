#!/usr/bin/env python3
"""
Step 3 Remote Orchestrator on Colab Linux (v3linux):
Prospective Frozen-Protocol Evaluation on Wang 2025 Normal TMA (nTMA)
Tracks N-A (Xenium Breast vs CosMx+IF), N-B (Xenium Breast vs MERSCOPE), N-C (Xenium Breast vs Panhuman).

STRICT ZERO-TUNING POLICY:
  - Uses ONLY frozen DEFAULT_PROTOCOL_V3 from spatial-membership-audit@v3.0.0-protocol-freeze-20261125
    (Split-half 2s q0=0.75 + synchronized joint perm M=199, B=20)
  - Optional secondary frozen arm: MITM geom EB + guardrail
  - Fixed reference baselines: Naive, MNN rescaleBatches, Soft plug-in (tau=0.05 & 0.01), Depth-only 2s q0=0.75
  - Zero parameter sweeps or arm selection on nTMA.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
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
from scipy.stats import spearmanr, ttest_ind, ttest_rel, wilcoxon
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests

import spatial_membership_audit as sma
from spatial_membership_audit.coverage_matching import (
    build_split_half_curve,
    conditional_instability_metrics,
    match_split_half_coverage,
)
from spatial_membership_audit.dua_inference import joint_perm_pairs_from_draws, summarize_dua_draws
from spatial_membership_audit.extrapolation import (
    build_extrapolation_guardrail,
    extrapolate_ladder,
    select_mitm_target_depth,
    summarize_thinning_ladder,
)
from spatial_membership_audit.stability_weighting import cell_stability_weights, exact_weighted_perm_pair
from spatial_membership_audit.standardize import DEFAULT_PROTOCOL_V3, get_default_protocol_v3

WORK = Path("/content/v3work/ntma")
RAW_DIR = WORK / "raw"
DATA_DIR = WORK / "data"
OUT_DIR = WORK / "results"
for d in (RAW_DIR, DATA_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

ST_R = "/content/v3work/bin/micromamba"
R_ENV = "/content/v3work/mamba/st_r"
R_SCRIPT = WORK / "run_ntma_singler.R"
HPCA_RDA = Path("/content/v3work/hpca.rda")
ZIP_INV = WORK / "broad_zip_inventory.json"
HPCA_TXT = WORK / "hpca_genes.txt"

TOTAL_ZIP_BYTES = 743407896
FREEZE_TAG_UTC = "2026-11-25T14:12:08Z"

TRACK_SPECS = [
    ("Track_NA_Xenium_vs_CosMx_IF", "xenium_breast_normal", "cosmx_multitissue_normal"),
    ("Track_NB_Xenium_vs_MERSCOPE", "xenium_breast_normal", "merscope_breast_normal"),
    ("Track_NC_Xenium_Breast_vs_Panhuman", "xenium_breast_normal", "xenium_panhuman_normal"),
]

STEM_TO_H5AD = {
    "xenium_breast_normal": "2023_xenium_breast_nTMA.h5ad",
    "cosmx_multitissue_normal": "2023_cosmx_multitissue_nTMA.h5ad",
    "merscope_breast_normal": "2023_merscope_breast_nTMA.h5ad",
    "xenium_panhuman_normal": "2023_xenium_panhuman_nTMA.h5ad",
}

CLASSES = ["Epithelial", "Endothelial", "Stroma", "B_cell", "T_cells", "Myeloid"]
PAIRS = [(i, j, f"{CLASSES[i]}--{CLASSES[j]}") for i in range(6) for j in range(i + 1, 6)]
PAIR_NAMES = [p[2] for p in PAIRS]
KEY_4_AXES = ["Epithelial--T_cells", "Epithelial--Myeloid", "B_cell--T_cells", "Epithelial--Stroma"]


# =============================================================================
# Stage 1: Range-Fetch & Extract nTMA
# =============================================================================
def range_fetch_h5ad(url: str, inv_rec: dict) -> tuple[Path, dict]:
    name = inv_rec["name"]
    out = RAW_DIR / f"{name}.ziprange"
    size = int(inv_rec["compressed"]) + 300
    t0 = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not (out.exists() and out.stat().st_size == size):
        if out.exists():
            out.unlink()
        chunk = 1048576  # 1 MB chunks on fast Colab network
        n_parts = (size + chunk - 1) // chunk
        with out.open("wb") as fout:
            for j in range(n_parts):
                lo = int(inv_rec["offset"]) + j * chunk
                hi = min(int(inv_rec["offset"]) + size, lo + chunk) - 1
                n = hi - lo + 1
                ok = False
                for att in range(5):
                    try:
                        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}", "Accept-Encoding": "identity"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            assert resp.status == 206, resp.status
                            cr = resp.headers.get("Content-Range")
                            assert cr == f"bytes {lo}-{hi}/{TOTAL_ZIP_BYTES}", cr
                            b = resp.read(n + 1)
                            assert len(b) == n, (len(b), n)
                        fout.write(b)
                        ok = True
                        break
                    except Exception as e:
                        print(f"  [retry {att+1}] {name} part {j}: {e}", flush=True)
                        time.sleep(1.5 * (2 ** att))
                if not ok:
                    raise RuntimeError(f"Range-fetch failed on {name} part {j}")

    # Verify ZIP local file header
    hdr = out.read_bytes()[:256]
    fields = struct.unpack("<4s5H3I2H", hdr[:30])
    fn = hdr[30:30 + fields[-2]].decode("utf-8")
    assert hdr[:4] == b"PK\x03\x04" and fields[3] == 8 and fn == name, (fn, fields)
    finished_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = dict(
        name=name, started_utc=started_utc, finished_utc=finished_utc,
        postdates_freeze_tag=(started_utc > FREEZE_TAG_UTC),
        freeze_tag_utc=FREEZE_TAG_UTC,
        range_bytes=out.stat().st_size,
        range_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        seconds=round(time.monotonic() - t0, 2),
    )
    print(f"[fetch] {name}: {out.stat().st_size/1e6:.2f} MB in {rec['seconds']}s (UTC {finished_utc})", flush=True)
    return out, rec


def _read_h5_str(grp: h5py.Group, name: str) -> np.ndarray:
    obj = grp[name]
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        cats = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in obj["categories"][:]])
        codes = obj["codes"][:]
        return np.where(codes >= 0, cats[np.maximum(codes, 0)], "")
    vals = obj[:]
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in vals])


def decompress_and_extract_h5ad(h5_name: str, ziprange_path: Path, inv_rec: dict, targets: dict[str, tuple[pd.DataFrame, list[str]]]):
    tmp_h5 = RAW_DIR / f"_tmp_{h5_name}"
    if tmp_h5.exists():
        tmp_h5.unlink()
    try:
        with ziprange_path.open("rb") as f:
            hdr = f.read(30)
            sig, _, _, method, _, _, _, _, _, fn_len, ex_len = struct.unpack("<4s5H3I2H", hdr)
            fname = f.read(fn_len).decode("utf-8")
            _ = f.read(ex_len)
            assert sig == b"PK\x03\x04" and method == 8 and fname == h5_name
            comp_rem = int(inv_rec["compressed"])
            dec = zlib.decompressobj(-15)
            crc = 0
            n_bytes = 0
            with tmp_h5.open("wb") as out_f:
                while comp_rem > 0:
                    ch = f.read(min(1 << 20, comp_rem))
                    if not ch:
                        break
                    comp_rem -= len(ch)
                    data = dec.decompress(ch)
                    if data:
                        n_bytes += len(data)
                        crc = zlib.crc32(data, crc)
                        out_f.write(data)
                tail = dec.flush()
                if tail:
                    n_bytes += len(tail)
                    crc = zlib.crc32(tail, crc)
                    out_f.write(tail)
        assert dec.eof and n_bytes == int(inv_rec["size"]) and (crc & 0xFFFFFFFF) == int(inv_rec["crc"]), (
            dec.eof, n_bytes, inv_rec["size"], crc & 0xFFFFFFFF, inv_rec["crc"]
        )
        print(f"  [h5] {h5_name} CRC32 & size verified ({n_bytes:,} B)", flush=True)

        results = {}
        with h5py.File(tmp_h5, "r") as hf:
            obs = hf["obs"]
            idx_key = obs.attrs.get("_index", "_index")
            idx_key = idx_key.decode("utf-8") if isinstance(idx_key, bytes) else str(idx_key)
            obs_idx = _read_h5_str(obs, idx_key)
            obs_cid = _read_h5_str(obs, "cell_id") if "cell_id" in obs else obs_idx

            var = hf["var"]
            vidx_key = var.attrs.get("_index", "_index")
            vidx_key = vidx_key.decode("utf-8") if isinstance(vidx_key, bytes) else str(vidx_key)
            var_genes = _read_h5_str(var, vidx_key).tolist()
            if "gene" in var and len(set(var_genes)) < 10:
                var_genes = _read_h5_str(var, "gene").tolist()
            var_map = {g: idx for idx, g in enumerate(var_genes)}

            all_req = set()
            for df_req, _ in targets.values():
                all_req.update(df_req["cell_id"].tolist())
            hit_idx = len(all_req & set(obs_idx.tolist()))
            hit_cid = len(all_req & set(obs_cid.tolist()))
            obs_keys = obs_cid if hit_cid > hit_idx else obs_idx
            needed_pos = {cid: pos for pos, cid in enumerate(obs_keys) if cid in all_req}
            print(f"  [h5] matched {len(needed_pos):,}/{len(all_req):,} cells, n_var={len(var_genes)}", flush=True)

            x_node = hf["X"]
            is_sparse = isinstance(x_node, h5py.Group)
            if is_sparse:
                enc = str(x_node.attrs.get("encoding-type", ""))
                shape = tuple(int(v) for v in x_node.attrs["shape"])
                indptr = x_node["indptr"][:]
                indices = x_node["indices"][:]
                data = x_node["data"][:]
                assert np.isfinite(data).all() and (data >= 0).all() and np.all(data == np.floor(data))
                sp = (sparse.csc_matrix((data.astype(np.int32), indices, indptr), shape=shape) if "csc" in enc
                      else sparse.csr_matrix((data.astype(np.int32), indices, indptr), shape=shape))
                row_totals_all = np.asarray(sp.sum(axis=1)).ravel().astype(np.int64)
            else:
                sp = None
                row_totals_all = None

            for tag, (df_req, g_list) in targets.items():
                col_idx = np.array([var_map[g] for g in g_list], dtype=np.int64)
                keep = df_req["cell_id"].isin(needed_pos).to_numpy()
                df_sub = df_req.loc[keep].copy().reset_index(drop=True)
                row_idx = np.array([needed_pos[cid] for cid in df_sub["cell_id"].tolist()], dtype=np.int64)

                if is_sparse:
                    mat = (sp[:, col_idx][row_idx, :] if "csc" in enc else sp[row_idx, :][:, col_idx]).toarray().astype(np.int32)
                    tot_h5 = row_totals_all[row_idx]
                else:
                    n_obs, _ = x_node.shape
                    sort_ord = np.argsort(row_idx)
                    sorted_r = row_idx[sort_ord]
                    mat = np.empty((len(row_idx), len(col_idx)), dtype=np.int32)
                    tot_h5 = np.empty(len(row_idx), dtype=np.int64)
                    cursor = 0
                    for lo in range(0, n_obs, 25000):
                        hi = min(n_obs, lo + 25000)
                        hi_cur = int(np.searchsorted(sorted_r, hi, side="left"))
                        if hi_cur > cursor:
                            block = x_node[lo:hi, :]
                            sub_r = sorted_r[cursor:hi_cur] - lo
                            sub_rows = block[sub_r]
                            tot_h5[sort_ord[cursor:hi_cur]] = np.round(sub_rows.sum(axis=1)).astype(np.int64)
                            sub_b = sub_rows[:, col_idx]
                            assert np.isfinite(sub_b).all() and (sub_b >= 0).all() and np.all(sub_b == np.floor(sub_b))
                            mat[sort_ord[cursor:hi_cur]] = sub_b.astype(np.int32)
                        cursor = hi_cur

                # Repair merscope transcript_counts==0 directly from H5AD full-panel total counts!
                if float(df_sub["transcript_counts"].median()) == 0.0:
                    df_sub["transcript_counts"] = tot_h5.astype(np.float64)
                df_sub["tx_h5ad_total"] = tot_h5
                results[tag] = (df_sub, mat)
        return results
    finally:
        if tmp_h5.exists():
            tmp_h5.unlink()


def load_normal_meta(stem: str) -> pd.DataFrame:
    p = WORK / "stage" / f"{stem}_cell_level.parquet.gzip"
    pf = pq.ParquetFile(p)
    schema = pf.schema.names
    cols = ["cell_id", "core", "tissue_type", "x", "y", "transcript_counts"]
    for opt in ("fov", "Mean.PanCK", "Mean.CD45", "Mean.CD68"):
        if opt in schema:
            cols.append(opt)
    df = pf.read(columns=cols).to_pandas()
    df["cell_id"] = df["cell_id"].astype(str)
    df["core"] = df["core"].astype("string").str.strip()
    df["tissue_type"] = df["tissue_type"].astype("string").str.strip()
    df = df.dropna(subset=["core", "tissue_type", "x", "y"]).copy()
    # Exclude MARKER cores strictly per tTMA1 frozen qualification rule
    df = df[~df["tissue_type"].str.upper().str.contains("MARKER", na=False)].copy()
    is_mm = (float(df["x"].max()) - float(df["x"].min())) < 100.0
    scale = 1000.0 if is_mm else 1.0
    df["x_um"] = df["x"].to_numpy(np.float64) * scale
    df["y_um"] = df["y"].to_numpy(np.float64) * scale
    return df


def load_gene_panel(stem: str) -> set[str]:
    p = WORK / "stage" / f"gene_level_csv_{stem}.csv"
    with p.open("r", encoding="utf-8-sig") as f:
        return {r["gene"] for r in csv.DictReader(f) if r.get("code_type") == "gene" and r.get("gene")}


def build_edges_and_save(tdir: Path, side: str, df_cells: pd.DataFrame, mat: np.ndarray, ordered_cores: list[str]) -> dict:
    c2r = {c: i for i, c in enumerate(ordered_cores)}
    order = np.argsort(df_cells["core"].astype(str).map(c2r).to_numpy(), kind="mergesort")
    df_ord = df_cells.iloc[order].copy().reset_index(drop=True)
    mat_ord = np.ascontiguousarray(mat[order], dtype=np.int32)
    tot_hpca = mat_ord.sum(axis=1, dtype=np.int64)
    df_ord["tx_shared_hpca"] = tot_hpca

    nc = len(df_ord)
    n_cores = len(ordered_cores)
    cell_ptr = np.zeros(n_cores + 1, dtype=np.int64)
    cvals = df_ord["core"].astype(str).to_numpy()
    pos = 0
    for ci, cn in enumerate(ordered_cores):
        cell_ptr[ci] = pos
        while pos < nc and cvals[pos] == cn:
            pos += 1
    cell_ptr[n_cores] = pos
    assert pos == nc

    core_stats = {}
    for rad in (25.0, 50.0):
        u_p, v_p = [], []
        edge_ptr = np.zeros(n_cores + 1, dtype=np.int64)
        e_tot = 0
        for ci, cn in enumerate(ordered_cores):
            s0, s1 = int(cell_ptr[ci]), int(cell_ptr[ci + 1])
            xy = df_ord.iloc[s0:s1][["x_um", "y_um"]].to_numpy(np.float64)
            tree = cKDTree(xy)
            pairs = tree.query_pairs(r=rad, output_type="ndarray")
            if len(pairs):
                u_p.append(pairs[:, 0].astype(np.int32))
                v_p.append(pairs[:, 1].astype(np.int32))
                e_tot += len(pairs)
            edge_ptr[ci + 1] = e_tot
            if rad == 25.0:
                dists, _ = tree.query(xy, k=2)
                tc = tot_hpca[s0:s1]
                pt = tc[tc > 0]
                core_stats[cn] = dict(
                    n_cells=s1 - s0,
                    med_raw_total=float(np.median(df_ord.iloc[s0:s1]["transcript_counts"].to_numpy(np.float64))),
                    medpos_shared_hpca=float(np.median(pt)) if len(pt) else 0.0,
                    zero_frac=float((tc == 0).mean()),
                    med_1nn_um=float(np.median(dists[:, 1])),
                    n_edges_r25=int(len(pairs)),
                )
        np.savez_compressed(
            tdir / f"run_{side}_edges_r{int(rad)}.npz",
            cell_ptr=cell_ptr, edge_ptr=edge_ptr,
            u=np.concatenate(u_p) if u_p else np.empty(0, np.int32),
            v=np.concatenate(v_p) if v_p else np.empty(0, np.int32),
        )
    mat_ord.tofile(tdir / f"run_{side}_counts.bin")
    df_ord.to_parquet(tdir / f"run_{side}_cells.parquet", index=False)

    pt_all = tot_hpca[tot_hpca > 0]
    med_1nn = float(np.median([core_stats[c]["med_1nn_um"] for c in ordered_cores]))
    assert 3.5 <= med_1nn <= 20.0, f"Unexpected 1NN distance on {tdir.name} {side}: {med_1nn}"
    return dict(
        n_cells=nc,
        medpos_shared_hpca=float(np.median(pt_all)) if len(pt_all) else 0.0,
        mean_shared_hpca=float(np.mean(tot_hpca)),
        zero_frac_shared_hpca=float((tot_hpca == 0).mean()),
        med_1nn_um=med_1nn,
        core_stats=core_stats,
    )


def stage1_prepare_all_tracks():
    print("\n=== STAGE 1: Prospective Zenodo Range-Fetch & Track Extraction (Wang 2025 nTMA) ===", flush=True)
    inv = json.loads(ZIP_INV.read_text(encoding="utf-8"))
    inv_map = {m["name"]: m for m in inv["members"]}
    url = inv["url"]
    hpca_genes = set(line.strip() for line in HPCA_TXT.read_text(encoding="utf-8").splitlines() if line.strip())

    # Fetch all 4 nTMA H5ADs and record UTC fetch receipt
    receipts = {}
    zipranges = {}
    for stem, h5_name in STEM_TO_H5AD.items():
        zp, rec = range_fetch_h5ad(url, inv_map[h5_name])
        zipranges[stem] = zp
        receipts[h5_name] = rec
    (OUT_DIR / "ntma_fetch_receipt.json").write_text(
        json.dumps(dict(
            protocol_version=sma.__version__,
            frozen_protocol=get_default_protocol_v3(),
            freeze_tag="v3.0.0-protocol-freeze-20261125",
            freeze_tag_utc=FREEZE_TAG_UTC,
            files=receipts,
        ), indent=2), encoding="utf-8"
    )

    # Load metas & shared gene panels
    metas = {st: load_normal_meta(st) for st in STEM_TO_H5AD}
    panels = {st: load_gene_panel(st) for st in STEM_TO_H5AD}

    # Determine qualified shared cores (>=500 cells in both runs, identical tissue_type) per track
    track_plans = {}
    stem_targets: dict[str, dict[str, tuple[pd.DataFrame, list[str]]]] = {st: {} for st in STEM_TO_H5AD}
    for tname, sa, sb in TRACK_SPECS:
        tdir = DATA_DIR / tname
        tdir.mkdir(parents=True, exist_ok=True)
        shared_genes = sorted(panels[sa] & panels[sb] & hpca_genes)
        da, db = metas[sa], metas[sb]
        ca = da.groupby("core").agg(n_a=("cell_id", "size"), tt_a=("tissue_type", "first"))
        cb = db.groupby("core").agg(n_b=("cell_id", "size"), tt_b=("tissue_type", "first"))
        j = ca.join(cb, how="inner")
        j = j[(j["n_a"] >= 500) & (j["n_b"] >= 500)].copy()
        j = j.sort_index()
        cores_ord = j.index.astype(str).tolist()
        assert len(cores_ord) >= 15, f"{tname} has too few paired cores: {len(cores_ord)}"
        print(f"  [plan] {tname}: {len(cores_ord)} paired cores, {len(shared_genes)} shared HPCA genes, {j['tt_a'].nunique()} organs", flush=True)
        track_plans[tname] = (sa, sb, cores_ord, shared_genes, j)
        stem_targets[sa][f"{tname}__a"] = (da[da["core"].astype(str).isin(cores_ord)].copy(), shared_genes)
        stem_targets[sb][f"{tname}__b"] = (db[db["core"].astype(str).isin(cores_ord)].copy(), shared_genes)

    # Extract from each H5AD once (stream decompress -> extract -> immediately unlink tmp H5AD)
    extracted = {}
    for stem, h5_name in STEM_TO_H5AD.items():
        if not stem_targets[stem]:
            continue
        res = decompress_and_extract_h5ad(h5_name, zipranges[stem], inv_map[h5_name], stem_targets[stem])
        extracted.update(res)

    # Finalize per-track core intersections (ensure post-join >= 500 cells per core on both sides)
    for tname, (sa, sb, cores_cand, shared_genes, j) in track_plans.items():
        tdir = DATA_DIR / tname
        df_a, mat_a = extracted[f"{tname}__a"]
        df_b, mat_b = extracted[f"{tname}__b"]
        cnt_a = df_a["core"].astype(str).value_counts()
        cnt_b = df_b["core"].astype(str).value_counts()
        cores_ok = [c for c in cores_cand if cnt_a.get(c, 0) >= 500 and cnt_b.get(c, 0) >= 500]
        assert len(cores_ok) >= 15, f"{tname} dropped below 15 cores post-join: {len(cores_ok)}"

        ma = df_a["core"].astype(str).isin(cores_ok).to_numpy()
        mb = df_b["core"].astype(str).isin(cores_ok).to_numpy()
        df_a, mat_a = df_a.loc[ma].reset_index(drop=True), mat_a[ma]
        df_b, mat_b = df_b.loc[mb].reset_index(drop=True), mat_b[mb]

        (tdir / "genes.txt").write_text("\n".join(shared_genes) + "\n", encoding="utf-8")
        st_a = build_edges_and_save(tdir, "a", df_a, mat_a, cores_ok)
        st_b = build_edges_and_save(tdir, "b", df_b, mat_b, cores_ok)

        core_rows = []
        for c in cores_ok:
            core_rows.append(dict(
                core=c, tissue_type=str(j.loc[c, "tt_a"]),
                n_cells_a=st_a["core_stats"][c]["n_cells"], n_cells_b=st_b["core_stats"][c]["n_cells"],
                medpos_a=st_a["core_stats"][c]["medpos_shared_hpca"], medpos_b=st_b["core_stats"][c]["medpos_shared_hpca"],
            ))
        pd.DataFrame(core_rows).to_csv(tdir / "cores.csv", index=False)
        (tdir / "extract_summary.json").write_text(json.dumps(dict(
            track=tname, stem_a=sa, stem_b=sb, n_cores=len(cores_ok), n_genes=len(shared_genes),
            n_organs=int(pd.DataFrame(core_rows)["tissue_type"].nunique()),
            run_a={k: v for k, v in st_a.items() if k != "core_stats"},
            run_b={k: v for k, v in st_b.items() if k != "core_stats"},
        ), indent=2), encoding="utf-8")
        print(f"  [saved] {tname}: {len(cores_ok)} cores, medpos_a={st_a['medpos_shared_hpca']:.1f}, medpos_b={st_b['medpos_shared_hpca']:.1f}", flush=True)


# =============================================================================
# Stage 2: Run SingleR Probes, Solve Frozen Protocol Rates, Generate B=20 Banks
# =============================================================================
def dedup_banks(side_targets: list[tuple[str, float]], tol: float = 0.005) -> dict:
    scheduled = []
    mapping = {}
    for arm_tag, qv in side_targets:
        qv_f = float(qv)
        if qv_f >= 1.0 - 1e-6:
            mapping[arm_tag] = dict(q_target=qv_f, bank_tag="native", q_bank=1.0, reused=True)
            continue
        matched = next(((bt, bq) for bt, bq in scheduled if abs(qv_f - bq) <= tol), None)
        if matched is not None:
            mapping[arm_tag] = dict(q_target=qv_f, bank_tag=matched[0], q_bank=matched[1], reused=True)
        else:
            scheduled.append((arm_tag, qv_f))
            mapping[arm_tag] = dict(q_target=qv_f, bank_tag=arm_tag, q_bank=qv_f, reused=False)
    return dict(scheduled=[dict(tag=t, q=q) for t, q in scheduled], mapping=mapping)


def stage2_singler_and_rates():
    print("\n=== STAGE 2: Frozen Protocol Split-Half Probes & SingleR B=20 Banks ===", flush=True)
    all_rates = {}
    for tname, _, _ in TRACK_SPECS:
        tdir = DATA_DIR / tname
        # 2a. Probe
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "probe", str(tdir), str(HPCA_RDA)], check=True)
        pinfo = json.loads((tdir / "p2_probe_curves.json").read_text(encoding="utf-8"))
        ext = json.loads((tdir / "extract_summary.json").read_text(encoding="utf-8"))
        sa, sb = float(ext["run_a"]["medpos_shared_hpca"]), float(ext["run_b"]["medpos_shared_hpca"])
        r_depth = min(sa, sb) / max(sa, sb)
        sh_side = "a" if sa <= sb else "b"
        dp_side = "b" if sh_side == "a" else "a"

        # Solve ONLY frozen DEFAULT_PROTOCOL_V3 (Split-half 2s q0=0.75) + secondary MITM geom + reference Depth-only 2s q0=0.75
        proto_p = get_default_protocol_v3()["primary_cross_platform"]
        q0_frozen = float(proto_p["q0"])
        ca = build_split_half_curve(pinfo["q_grid"], pinfo["a"]["cond_psi"], unresolved_frac=pinfo["a"]["unresolved_frac"])
        cb = build_split_half_curve(pinfo["q_grid"], pinfo["b"]["cond_psi"], unresolved_frac=pinfo["b"]["unresolved_frac"])
        p2_both = match_split_half_coverage(ca, cb, q0=q0_frozen, pairing=str(proto_p["pairing"]), min_q=float(proto_p["split_half_min_q"]))

        d_geom = select_mitm_target_depth(sa, sb, mode="geometric")
        q_mitm_dp = min(1.0, d_geom / max(sa, sb))

        sh_req = [
            ("q085", 0.85), ("q075", q0_frozen), ("q065", 0.65), ("q050", 0.50),
            ("p2_both", p2_both.q_a if sh_side == "a" else p2_both.q_b),
        ]
        dp_req = [
            ("autoq_both", q0_frozen * r_depth),
            ("mitm_geom", q_mitm_dp),
            ("p2_both", p2_both.q_a if dp_side == "a" else p2_both.q_b),
        ]
        rec = dict(
            track=tname, medpos_a=sa, medpos_b=sb, shallower_side=sh_side, deeper_side=dp_side,
            r_depth=r_depth, d_mitm_geom=d_geom, q0_frozen=q0_frozen,
            p2_both=dict(q_a=p2_both.q_a, q_b=p2_both.q_b, target_psi=p2_both.target_psi,
                         noisier_side=p2_both.noisier_side, clamped=p2_both.clamped),
            schedules={sh_side: dedup_banks(sh_req), dp_side: dedup_banks(dp_req)},
        )
        (tdir / "track_rates.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        all_rates[tname] = rec
        print(f"  [rates] {tname}: sh={sh_side}({min(sa,sb):.1f}) vs dp={dp_side}({max(sa,sb):.1f}) | "
              f"P2_2s=({p2_both.q_a:.3f},{p2_both.q_b:.3f}, noisier={p2_both.noisier_side}) | MITM_dp={q_mitm_dp:.3f}", flush=True)

        # 2b. Generate native, B3, and deduplicated B=20 banks
        subprocess.run([ST_R, "run", "-p", R_ENV, "Rscript", str(R_SCRIPT), "banks", str(tdir), str(HPCA_RDA)], check=True)
    (OUT_DIR / "ntma_rates_and_audit.json").write_text(json.dumps(all_rates, indent=2), encoding="utf-8")


# =============================================================================
# Stage 3: Strict U_common Evaluation (Comparability, Paired Null, Reversals, Joint Perm, IF)
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
    tot = int(sw.sum())
    cs = np.cumsum(sw)
    if tot % 2 == 1:
        return float(sv[np.searchsorted(cs, tot // 2 + 1, side="left")])
    return float(0.5 * (sv[np.searchsorted(cs, tot // 2, side="left")] + sv[np.searchsorted(cs, tot // 2 + 1, side="left")]))


def cluster_boot_weights(clusters, n_boot=2000, seed=20261125):
    rng = np.random.default_rng(seed)
    uniq = sorted(np.unique(clusters).tolist())
    cl_idx = [np.flatnonzero(clusters == u) for u in uniq]
    k = len(uniq)
    w_mat = np.zeros((n_boot, len(clusters)), dtype=np.int16)
    for b in range(n_boot):
        for ci in rng.integers(0, k, size=k):
            w_mat[b, cl_idx[ci]] += 1
    return w_mat


def strat_core_boot_weights(tt, n_boot=2000, seed=2026112500):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tt == u) for u in sorted(np.unique(tt).tolist())]
    w_mat = np.zeros((n_boot, len(tt)), dtype=np.int16)
    for b in range(n_boot):
        for idxs in groups:
            for idx in rng.choice(idxs, size=len(idxs), replace=True):
                w_mat[b, idx] += 1
    return w_mat


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


def build_core_graphs(npz_path: Path) -> list[dict]:
    z = np.load(npz_path)
    cptr, eptr, u_all, v_all = z["cell_ptr"], z["edge_ptr"], z["u"], z["v"]
    graphs = []
    for ci in range(len(cptr) - 1):
        s0, s1 = int(cptr[ci]), int(cptr[ci + 1])
        e0p, e1p = int(eptr[ci]), int(eptr[ci + 1])
        n = s1 - s0
        eu, ev = u_all[e0p:e1p], v_all[e0p:e1p]
        m = len(eu)
        if n >= 2 and m > 0:
            deg = np.bincount(np.concatenate([eu, ev]), minlength=n).astype(np.float64)
            n_adj = float(0.5 * np.sum(deg * (deg - 1.0)))
            n_disj = float(m * (m - 1.0) * 0.5 - n_adj)
            c_const = 2.0 * m / (n * (n - 1.0))
        else:
            n_adj = n_disj = c_const = 0.0
        graphs.append(dict(s0=s0, s1=s1, n=n, m=m, e0=eu, e1=ev, n_adj=n_adj, n_disj=n_disj, c_const=c_const))
    return graphs


def exact_perm_15_core(g: dict, loc_codes: np.ndarray, lut: np.ndarray):
    R = np.full(15, np.nan); z = np.full(15, np.nan)
    n, m = g["n"], g["m"]
    if n < 2 or m == 0:
        return R, z
    mapped = lut[loc_codes]
    cnt = np.bincount(mapped, minlength=7)[:6].astype(np.float64)
    c0, c1 = mapped[g["e0"]], mapped[g["e1"]]
    keep = (c0 < 6) & (c1 < 6) & (c0 != c1)
    if keep.any():
        lo, hi = np.minimum(c0[keep], c1[keep]).astype(np.int32), np.maximum(c0[keep], c1[keep]).astype(np.int32)
        o_mat = np.bincount(lo * 6 + hi, minlength=36).astype(np.float64)
    else:
        o_mat = np.zeros(36, dtype=np.float64)
    c_const, n_adj, n_disj = g["c_const"], g["n_adj"], g["n_disj"]
    d3 = n * (n - 1.0) * (n - 2.0) if n >= 3 else 0.0
    d4 = d3 * (n - 3.0) if n >= 4 else 0.0
    for pi, (a, b, _) in enumerate(PAIRS):
        na, nb = cnt[a], cnt[b]
        if na <= 0 or nb <= 0:
            continue
        o = o_mat[a * 6 + b]
        ep = c_const * na * nb
        R[pi] = o / ep
        pa = (na * nb * (na + nb - 2.0)) / d3 if n >= 3 else 0.0
        pd_ = (4.0 * na * max(0.0, na - 1.0) * nb * max(0.0, nb - 1.0)) / d4 if (n >= 4 and na >= 2 and nb >= 2) else 0.0
        vp = max(0.0, ep + 2.0 * n_adj * pa + 2.0 * n_disj * pd_ - ep * ep)
        if vp > 0.0:
            z[pi] = (o - ep) / math.sqrt(vp)
    return R, z


def eval_codes(graphs, codes_or_draws, lut):
    U = len(graphs) * 15
    if codes_or_draws.ndim == 1:
        R = np.full(U, np.nan); z = np.full(U, np.nan)
        for ci, g in enumerate(graphs):
            R[ci*15:(ci+1)*15], z[ci*15:(ci+1)*15] = exact_perm_15_core(g, codes_or_draws[g["s0"]:g["s1"]], lut)
        return R, z
    B = len(codes_or_draws)
    Rb = np.full((B, U), np.nan); zb = np.full((B, U), np.nan)
    for b in range(B):
        for ci, g in enumerate(graphs):
            Rb[b, ci*15:(ci+1)*15], zb[b, ci*15:(ci+1)*15] = exact_perm_15_core(g, codes_or_draws[b, g["s0"]:g["s1"]], lut)
    return Rb, zb


def eval_soft_15(graphs, sc_df, score_cols, tau):
    S = sc_df[score_cols].to_numpy(np.float64)
    bad = ~np.isfinite(S).all(axis=1)
    Sc = np.where(bad[:, None], 0.0, S)
    ex = np.exp((Sc - Sc.max(axis=1, keepdims=True)) / float(tau))
    P_all = ex / ex.sum(axis=1, keepdims=True)
    cls_ix = [score_cols.index(c) for c in CLASSES]
    P = P_all[:, cls_ix]; P[bad] = 0.0
    out = np.full((len(graphs), 15), np.nan)
    for ci, g in enumerate(graphs):
        if g["n"] < 2 or g["m"] == 0:
            continue
        Ps = P[g["s0"]:g["s1"]]
        sums = Ps.sum(axis=0)
        e0, e1, cc = g["e0"], g["e1"], g["c_const"]
        for pi, (a, b, _) in enumerate(PAIRS):
            d = float(sums[a] * sums[b] - np.dot(Ps[:, a], Ps[:, b]))
            if d > 0.0:
                o = float((Ps[e0, a] * Ps[e1, b] + Ps[e0, b] * Ps[e1, a]).sum())
                out[ci, pi] = o / (cc * d)
    return out.ravel()


def eval_track_na_if(tdir: Path, gb: list[dict], cells_b: pd.DataFrame, cmap: dict, score_cols: list[str],
                     sc_b: pd.DataFrame, base_b: np.ndarray, b3_b: np.ndarray,
                     draws_autoq: np.ndarray, draws_p2: np.ndarray, draws_mitm: np.ndarray,
                     sh_side: str, ladder_sh_if_fn=None) -> list[dict]:
    """Exact same CosMx IF evaluation protocol as tTMA1 Track A strictly on U_common."""
    raw_pan = np.log1p(pd.to_numeric(cells_b["Mean.PanCK"], errors="coerce").fillna(0.0).to_numpy(np.float64))
    rng = np.random.default_rng(42)
    fit_x = rng.choice(raw_pan, size=min(100000, len(raw_pan)), replace=False).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(fit_x)
    hi = int(np.argmax(gmm.means_.ravel()))
    panck_prob = gmm.predict_proba(raw_pan.reshape(-1, 1))[:, hi]
    panck_pos = panck_prob >= 0.5

    raw_68 = np.log1p(pd.to_numeric(cells_b["Mean.CD68"], errors="coerce").fillna(0.0).to_numpy(np.float64))
    raw_45 = np.log1p(pd.to_numeric(cells_b["Mean.CD45"], errors="coerce").fillna(0.0).to_numpy(np.float64))
    z68 = np.zeros(len(cells_b)); z45 = np.zeros(len(cells_b))
    for g in gb:
        s0, s1 = g["s0"], g["s1"]
        for arr_in, z_out in ((raw_68, z68), (raw_45, z45)):
            sub = arr_in[s0:s1]; med = float(np.median(sub))
            mad = float(1.4826 * np.median(np.abs(sub - med)))
            z_out[s0:s1] = (sub - med) / max(mad, 1e-6)

    def make_if_vec(if_def: str):
        R_if = np.full((len(gb), 3), np.nan)
        for ci, g in enumerate(gb):
            s0, s1, n, m = g["s0"], g["s1"], g["n"], g["m"]
            if n < 2 or m == 0:
                continue
            e0, e1, cc = g["e0"], g["e1"], g["c_const"]
            if if_def == "z15_primary":
                tum = panck_pos[s0:s1]
                mac = (~tum) & (z68[s0:s1] > 1.5)
                imm = (~tum) & ((z68[s0:s1] > 1.5) | (z45[s0:s1] > 1.5))
                strom = (~tum) & (~imm)
                na = float(tum.sum())
                for ai, mb in enumerate((mac, imm, strom)):
                    nb = float(mb.sum())
                    if na > 0 and nb > 0:
                        R_if[ci, ai] = float(((tum[e0] & mb[e1]) | (tum[e1] & mb[e0])).sum()) / (cc * na * nb)
            else:
                wp = panck_prob[s0:s1]
                wc = np.clip(z68[s0:s1], 0.0, 5.0) / 5.0 * (1.0 - wp)
                wi = np.clip(np.maximum(z68[s0:s1], z45[s0:s1]), 0.0, 5.0) / 5.0 * (1.0 - wp)
                ws = np.clip(1.0 - wp - wi, 0.0, 1.0)
                sa = float(wp.sum())
                for ai, wb in enumerate((wc, wi, ws)):
                    sb = float(wb.sum()); d = float(sa * sb - np.dot(wp, wb))
                    if sa > 0 and sb > 0 and d > 0:
                        R_if[ci, ai] = float((wp[e0] * wb[e1] + wp[e1] * wb[e0]).sum()) / (cc * d)
        return R_if.ravel()

    r_if15 = make_if_vec("z15_primary")
    r_ifct = make_if_vec("continuous_w")

    c_epi, c_mye, c_str = cmap["Epithelial"], cmap["Myeloid"], cmap["Stroma"]
    c_imm = np.array([cmap["T_cells"], cmap["B_cell"], cmap["Myeloid"]], dtype=np.uint8)

    def eval_3_codes(arr, return_stack: bool = False):
        def one(cvec):
            out = np.full((len(gb), 3), np.nan)
            for ci, g in enumerate(gb):
                s0, s1 = g["s0"], g["s1"]
                if g["n"] < 2 or g["m"] == 0:
                    continue
                loc = cvec[s0:s1]; ma = loc == c_epi; na = float(ma.sum())
                if na == 0:
                    continue
                e0, e1, cc = g["e0"], g["e1"], g["c_const"]
                for ai, mb in enumerate((loc == c_mye, np.isin(loc, c_imm), loc == c_str)):
                    nb = float(mb.sum())
                    if nb > 0:
                        out[ci, ai] = float(((ma[e0] & mb[e1]) | (ma[e1] & mb[e0])).sum()) / (cc * na * nb)
            return out.ravel()
        if arr.ndim == 1:
            return one(arr)
        st = np.vstack([one(arr[b]) for b in range(len(arr))])
        if return_stack:
            return st
        ok = np.isfinite(st).any(axis=0)
        res = np.full(st.shape[1], np.nan)
        res[ok] = np.nanmean(st[:, ok], axis=0)
        return res

    def eval_3_soft(tau):
        S = sc_b[score_cols].to_numpy(np.float64)
        bad = ~np.isfinite(S).all(axis=1)
        Sc = np.where(bad[:, None], 0.0, S)
        ex = np.exp((Sc - Sc.max(axis=1, keepdims=True)) / float(tau))
        P = ex / ex.sum(axis=1, keepdims=True); P[bad] = 0.0
        ie, im, is_ = score_cols.index("Epithelial"), score_cols.index("Myeloid"), score_cols.index("Stroma")
        ii = [score_cols.index(c) for c in ("T_cells", "B_cell", "Myeloid")]
        out = np.full((len(gb), 3), np.nan)
        for ci, g in enumerate(gb):
            if g["n"] < 2 or g["m"] == 0:
                continue
            Ps = P[g["s0"]:g["s1"]]; wa = Ps[:, ie]; sa = float(wa.sum())
            e0, e1, cc = g["e0"], g["e1"], g["c_const"]
            for ai, wb in enumerate((Ps[:, im], Ps[:, ii].sum(axis=1), Ps[:, is_])):
                sb = float(wb.sum()); d = float(sa * sb - np.dot(wa, wb))
                if sa > 0 and sb > 0 and d > 0:
                    out[ci, ai] = float((wa[e0] * wb[e1] + wa[e1] * wb[e0]).sum()) / (cc * d)
        return out.ravel()

    def eval_3_stab_w1(draws_75):
        w = cell_stability_weights(base_b, draws_75, power=1.0)
        out = np.full((len(gb), 3), np.nan)
        for ci, g in enumerate(gb):
            if g["n"] < 2 or g["m"] == 0:
                continue
            adj = sparse.csr_matrix(
                (np.ones(2 * g["m"]), (np.r_[g["e0"], g["e1"]], np.r_[g["e1"], g["e0"]])),
                shape=(g["n"], g["n"]),
            )
            loc = base_b[g["s0"]:g["s1"]]
            wl = w[g["s0"]:g["s1"]]
            wa = np.where(loc == c_epi, wl, 0.0)
            for ai, mb in enumerate((loc == c_mye, np.isin(loc, c_imm), loc == c_str)):
                out[ci, ai] = exact_weighted_perm_pair(adj, wa, np.where(mb, wl, 0.0)).r_weighted
        return out.ravel()

    arm_vecs = {
        "Naive": safe_log(eval_3_codes(base_b)),
        "MNN_rescaleBatches": safe_log(eval_3_codes(b3_b)),
        "Soft_plugin_tau005": safe_log(eval_3_soft(0.05)),
        "Soft_plugin_tau001": safe_log(eval_3_soft(0.01)),
        "Stability_weighted_w1": safe_log(eval_3_stab_w1(draws_mitm if sh_side == "b" else draws_p2)),
        "Depth_only_2s_q075": safe_log(eval_3_codes(draws_autoq)),
        "Split_half_2s_default": safe_log(eval_3_codes(draws_p2)),
    }
    if ladder_sh_if_fn is not None and sh_side == "b":
        arm_vecs["MITM_geom_eb_guardrail"] = ladder_sh_if_fn(eval_3_codes)
    else:
        arm_vecs["MITM_geom_eb_guardrail"] = safe_log(eval_3_codes(draws_mitm))

    rows = []
    for if_name, r_if in (("z15_primary", r_if15), ("continuous_w", r_ifct)):
        l_if = safe_log(r_if)
        uc = np.isfinite(l_if)
        for lv in arm_vecs.values():
            uc &= np.isfinite(lv)
        li_u = l_if[uc]
        sd_if = float(np.std(li_u, ddof=1))
        for aname, lv in arm_vecs.items():
            la_u = lv[uc]
            rows.append(dict(
                track="Track_NA_Xenium_vs_CosMx_IF", if_def=if_name, arm=aname, u_common_n=int(uc.sum()),
                icc_vs_if=icc_1_1(la_u, li_u),
                pearson_r_vs_if=float(np.corrcoef(la_u, li_u)[0, 1]),
                spearman_rho_vs_if=float(spearmanr(la_u, li_u).statistic),
                mae_vs_if=float(np.mean(np.abs(la_u - li_u))),
                med_abs_vs_if=float(np.median(np.abs(la_u - li_u))),
                sd_ratio_vs_if=float(np.std(la_u, ddof=1) / max(1e-12, sd_if)),
                kappa_r_gt_1=float(cohen_kappa_score(la_u > 0.0, li_u > 0.0)),
            ))
    return rows


def stage3_evaluate_all_tracks():
    print("\n=== STAGE 3: Strict U_common Prospective Evaluation on Wang 2025 nTMA ===", flush=True)
    sum_rows, null_rows, rev_rows, jp_rows, if_rows = [], [], [], [], []

    for tname, _, _ in TRACK_SPECS:
        tdir = DATA_DIR / tname
        ddir = tdir / "draws"
        cores_df = pd.read_csv(tdir / "cores.csv")
        cores = cores_df["core"].astype(str).to_numpy()
        tt = cores_df["tissue_type"].astype(str).to_numpy()
        n_cores = len(cores)
        rates = json.loads((tdir / "track_rates.json").read_text(encoding="utf-8"))
        sh, dp = rates["shallower_side"], rates["deeper_side"]
        s_sh, s_dp = float(rates[f"medpos_{sh}"]), float(rates[f"medpos_{dp}"])

        lab_tbl = pd.read_csv(tdir / "label_codes.tsv", sep="\t")
        cmap = dict(zip(lab_tbl["label"], lab_tbl["code"]))
        lut = np.full(256, 6, dtype=np.int32)
        for k, cls in enumerate(CLASSES):
            lut[cmap[cls]] = k
        score_cols = sorted([c for c in cmap if c != "UNRESOLVED"])

        cells_a = pd.read_parquet(tdir / "run_a_cells.parquet")
        cells_b = pd.read_parquet(tdir / "run_b_cells.parquet")

        # Physical TMA 4x3 spatial blocks from run_a centroids
        cent = cells_a.groupby(cells_a["core"].astype(str))[["x_um", "y_um"]].median()
        xs, ys = cent.loc[cores, "x_um"].to_numpy(float), cent.loc[cores, "y_um"].to_numpy(float)
        bx = np.clip(np.floor((xs - xs.min()) / max(1e-6, xs.max() - xs.min()) * 4).astype(int), 0, 3)
        by = np.clip(np.floor((ys - ys.min()) / max(1e-6, ys.max() - ys.min()) * 3).astype(int), 0, 2)
        tma_blks = np.array([f"{ix}_{iy}" for ix, iy in zip(bx, by)])

        base_a = np.fromfile(ddir / "baseline_a_codes.bin", dtype=np.uint8)
        base_b = np.fromfile(ddir / "baseline_b_codes.bin", dtype=np.uint8)
        sc_a = pd.read_csv(ddir / "baseline_a_scores.tsv.gz", sep="\t")
        sc_b = pd.read_csv(ddir / "baseline_b_scores.tsv.gz", sep="\t")
        b3_a = np.fromfile(ddir / "b3_rescale_a.bin", dtype=np.uint8)
        b3_b = np.fromfile(ddir / "b3_rescale_b.bin", dtype=np.uint8)

        raw_cache = {}
        def get_draws(side, arm_tag):
            btag = "native" if arm_tag == "native" else rates["schedules"][side]["mapping"][arm_tag]["bank_tag"]
            if (side, btag) not in raw_cache:
                nc = len(cells_a) if side == "a" else len(cells_b)
                if btag == "native":
                    raw_cache[(side, btag)] = np.tile(base_a if side == "a" else base_b, (20, 1))
                else:
                    raw_cache[(side, btag)] = np.fromfile(ddir / f"run_{side}_{btag}.bin", dtype=np.uint8).reshape(20, nc)
            return raw_cache[(side, btag)]

        for rad in (25, 50):
            ga = build_core_graphs(tdir / f"run_a_edges_r{rad}.npz")
            gb = build_core_graphs(tdir / f"run_b_edges_r{rad}.npz")

            rz_cache = {}
            def get_rz(side, arm_tag):
                btag = "native" if arm_tag == "native" else rates["schedules"][side]["mapping"][arm_tag]["bank_tag"]
                if (side, btag) not in rz_cache:
                    rz_cache[(side, btag)] = eval_codes(ga if side == "a" else gb, get_draws(side, arm_tag), lut)
                return rz_cache[(side, btag)]

            nr_a, nz_a = eval_codes(ga, base_a, lut)
            nr_b, nz_b = eval_codes(gb, base_b, lut)
            rb3_a, _ = eval_codes(ga, b3_a, lut)
            rb3_b, _ = eval_codes(gb, b3_b, lut)
            rs05_a, rs05_b = eval_soft_15(ga, sc_a, score_cols, 0.05), eval_soft_15(gb, sc_b, score_cols, 0.05)
            rs01_a, rs01_b = eval_soft_15(ga, sc_a, score_cols, 0.01), eval_soft_15(gb, sc_b, score_cols, 0.01)

            # Depth-only 2s q0=0.75
            ta_aq, tb_aq = ("q075", "autoq_both") if sh == "a" else ("autoq_both", "q075")
            ra_aq, za_aq = get_rz("a", ta_aq); rb_aq, zb_aq = get_rz("b", tb_aq)
            da_aq = summarize_dua_draws(ra_aq, za_aq, availability_threshold=0.90)
            db_aq = summarize_dua_draws(rb_aq, zb_aq, availability_threshold=0.90)
            r_aq_a = np.where(da_aq.available, da_aq.r_mean, np.nan)
            r_aq_b = np.where(db_aq.available, db_aq.r_mean, np.nan)

            # Frozen PRIMARY: Split-half 2s default (q0=0.75)
            ra_p2, za_p2 = get_rz("a", "p2_both"); rb_p2, zb_p2 = get_rz("b", "p2_both")
            da_p2 = summarize_dua_draws(ra_p2, za_p2, availability_threshold=0.90)
            db_p2 = summarize_dua_draws(rb_p2, zb_p2, availability_threshold=0.90)
            r_p2_a = np.where(da_p2.available, da_p2.r_mean, np.nan)
            r_p2_b = np.where(db_p2.available, db_p2.r_mean, np.nan)

            # Frozen SECONDARY: MITM geom EB + guardrail (fallback to Split-half 2s default on abstained cores)
            lad_q = [1.00, 0.85, 0.75, 0.65, 0.50]
            sh_rb = [nr_a if sh == "a" else nr_b] + [get_rz(sh, qt)[0] for qt in ("q085", "q075", "q065", "q050")]
            sh_zb = [nz_a if sh == "a" else nz_b] + [get_rz(sh, qt)[1] for qt in ("q085", "q075", "q065", "q050")]
            lad_sh = summarize_thinning_ladder(sh_rb, sh_zb, lad_q, availability_threshold=0.90)
            cells_sh = cells_a if sh == "a" else cells_b
            sh_graphs = ga if sh == "a" else gb
            sh_base = base_a if sh == "a" else base_b
            tx_sh = cells_sh["tx_shared_hpca"].to_numpy(float)
            c_med_sh = np.array([float(np.median(tx_sh[g["s0"]:g["s1"]])) for g in sh_graphs])
            d_sh50 = get_draws(sh, "q050")
            c_flp50 = np.array([conditional_instability_metrics(sh_base[g["s0"]:g["s1"]], d_sh50[:, g["s0"]:g["s1"]], unresolved_label=cmap["UNRESOLVED"]).cond_flip_rate for g in sh_graphs])
            gmask = build_extrapolation_guardrail(np.repeat(c_med_sh, 15), np.repeat(c_flp50, 15), min_median_depth=2.0, max_cond_flip_rate=0.45)
            pids = np.tile(np.arange(15), n_cores)
            ex_res = extrapolate_ladder(lad_sh, depth_scale=s_sh, target_depth=float(rates["d_mitm_geom"]), pair_ids=pids, method="hier_eb", guardrail_mask=None)
            r_sh_mitm = np.exp(ex_res.log_r_pred)
            rd_m, zd_m = get_rz(dp, "mitm_geom")
            dd_m = summarize_dua_draws(rd_m, zd_m, availability_threshold=0.90)
            r_dp_mitm = np.where(dd_m.available, dd_m.r_mean, np.nan)
            r_mitm_a_raw, r_mitm_b_raw = (r_sh_mitm, r_dp_mitm) if sh == "a" else (r_dp_mitm, r_sh_mitm)
            r_mitm_a = np.where(gmask, r_mitm_a_raw, r_p2_a)
            r_mitm_b = np.where(gmask, r_mitm_b_raw, r_p2_b)

            arms_log = {
                "Naive": (safe_log(nr_a), safe_log(nr_b)),
                "MNN_rescaleBatches": (safe_log(rb3_a), safe_log(rb3_b)),
                "Soft_plugin_tau005": (safe_log(rs05_a), safe_log(rs05_b)),
                "Soft_plugin_tau001": (safe_log(rs01_a), safe_log(rs01_b)),
                "Depth_only_2s_q075": (safe_log(r_aq_a), safe_log(r_aq_b)),
                "Split_half_2s_default": (safe_log(r_p2_a), safe_log(r_p2_b)),
                "MITM_geom_eb_guardrail": (safe_log(r_mitm_a), safe_log(r_mitm_b)),
            }

            d0_mask = np.isfinite(arms_log["Naive"][0]) & np.isfinite(arms_log["Naive"][1])
            n_d0 = int(d0_mask.sum())
            uc = d0_mask.copy()
            raw_avail = {}
            for ak, (la, lb) in arms_log.items():
                ok = np.isfinite(la) & np.isfinite(lb)
                raw_avail[ak] = (int((ok & d0_mask).sum()), 100.0 * float((ok & d0_mask).sum()) / max(1, n_d0))
                uc &= ok
            n_uc = int(uc.sum())

            uc_idx = np.repeat(np.arange(n_cores), 15)[uc]
            pair_u = np.tile(PAIR_NAMES, n_cores)[uc]
            tt_u = np.repeat(tt, 15)[uc]
            bw_core = strat_core_boot_weights(tt, 2000, 2026112500)
            bw_tma = cluster_boot_weights(tma_blks, 2000, 2026112501)
            bw_org = cluster_boot_weights(tt, 2000, 2026112502)

            na_u, nb_u = arms_log["Naive"][0][uc], arms_log["Naive"][1][uc]
            diff_n = np.abs(na_u - nb_u); med_n = float(np.median(diff_n))
            sd_na, sd_nb = float(np.std(na_u, ddof=1)), float(np.std(nb_u, ddof=1))

            for ak, (la, lb) in arms_log.items():
                la_u, lb_u = la[uc], lb[uc]
                diff_a = np.abs(la_u - lb_u)
                med_a, mae_a = float(np.median(diff_a)), float(np.mean(diff_a))
                red = 100.0 * (1.0 - med_a / med_n) if med_n > 0 else math.nan

                def ci(bw):
                    vs = [100.0 * (1.0 - weighted_med(diff_a, bw[b][uc_idx]) / weighted_med(diff_n, bw[b][uc_idx])) for b in range(len(bw))]
                    vs = [v for v in vs if np.isfinite(v)]
                    return float(np.quantile(vs, 0.025)), float(np.quantile(vs, 0.975))

                c_lo, c_hi = ci(bw_core)
                t_lo, t_hi = ci(bw_tma)
                o_lo, o_hi = ci(bw_org)

                # Re-ID Top-5 & 4-axis kappa on U_common
                u_mat = uc.reshape(n_cores, 15)
                L0m, L1m = np.where(u_mat, la.reshape(n_cores, 15), np.nan), np.where(u_mat, lb.reshape(n_cores, 15), np.nan)
                bm = np.vstack([L0m, L1m])
                mc = np.where(np.isfinite(bm).any(0), np.nanmean(bm, 0), 0.0)
                sc_v = np.where(np.isfinite(bm).any(0), np.nanstd(bm, 0), 1.0); sc_v = np.where(sc_v > 0, sc_v, 1.0)
                Z0 = np.where(np.isfinite(L0m), (L0m - mc) / sc_v, 0.0)
                Z1 = np.where(np.isfinite(L1m), (L1m - mc) / sc_v, 0.0)
                n0 = np.where(np.linalg.norm(Z0, axis=1, keepdims=True) > 0, np.linalg.norm(Z0, axis=1, keepdims=True), 1.0)
                n1 = np.where(np.linalg.norm(Z1, axis=1, keepdims=True) > 0, np.linalg.norm(Z1, axis=1, keepdims=True), 1.0)
                sim = (Z0 / n0) @ (Z1 / n1).T
                reid5 = 100.0 * float(np.mean([int(np.sum(sim[i] > sim[i, i]) + 1) <= 5 for i in range(n_cores)]))
                kaps = []
                for ax in KEY_4_AXES:
                    pi = PAIR_NAMES.index(ax)
                    oka = u_mat[:, pi]
                    if oka.sum() >= 10 and len(np.unique(L0m[oka, pi] > 0)) > 1 and len(np.unique(L1m[oka, pi] > 0)) > 1:
                        kaps.append(float(cohen_kappa_score((L0m[oka, pi] > 0).astype(int), (L1m[oka, pi] > 0).astype(int))))

                sum_rows.append(dict(
                    track=tname, radius_um=rad, arm=ak, n_cores=n_cores, n_tma_blocks=int(len(np.unique(tma_blks))),
                    d0_n=n_d0, u_common_n=n_uc, raw_avail_n=raw_avail[ak][0], raw_avail_pct=raw_avail[ak][1],
                    icc=icc_1_1(la_u, lb_u), pearson_r=float(np.corrcoef(la_u, lb_u)[0, 1]),
                    spearman_rho=float(spearmanr(la_u, lb_u).statistic),
                    mae=mae_a, med_disc=med_a, red_pct=red,
                    red_ci_lo_core=c_lo, red_ci_hi_core=c_hi,
                    red_ci_lo_tma_block=t_lo, red_ci_hi_tma_block=t_hi,
                    red_ci_lo_organ=o_lo, red_ci_hi_organ=o_hi,
                    sd_ratio_a=float(np.std(la_u, ddof=1) / max(1e-12, sd_na)),
                    sd_ratio_b=float(np.std(lb_u, ddof=1) / max(1e-12, sd_nb)),
                    bio_eta2_a=compute_bio_eta2(la_u, pair_u, tt_u),
                    bio_eta2_b=compute_bio_eta2(lb_u, pair_u, tt_u),
                    reid_top5_pct=reid5, mean_4axis_kappa=float(np.mean(kaps)) if kaps else math.nan,
                    mitm_guardrail_pass_pct=100.0 * float(gmask.mean()) if ak == "MITM_geom_eb_guardrail" else math.nan,
                ))

            # Single-draw B=1 default reference at r=25
            if rad == 25:
                sd_rows = []
                for b in range(20):
                    la_b, lb_b = safe_log(ra_p2[b])[uc], safe_log(rb_p2[b])[uc]
                    okb = np.isfinite(la_b) & np.isfinite(lb_b)
                    if okb.sum() >= 10:
                        db_v = np.abs(la_b[okb] - lb_b[okb])
                        sd_rows.append(dict(
                            icc=icc_1_1(la_b[okb], lb_b[okb]), mae=float(np.mean(db_v)),
                            med_disc=float(np.median(db_v)), red=100.0 * (1.0 - float(np.median(db_v)) / med_n),
                            sd_a=float(np.std(la_b[okb], ddof=1) / max(1e-12, sd_na)),
                            sd_b=float(np.std(lb_b[okb], ddof=1) / max(1e-12, sd_nb)),
                        ))
                if sd_rows:
                    dfs = pd.DataFrame(sd_rows)
                    sum_rows.append(dict(
                        track=tname, radius_um=25, arm="Single_draw_B1_default_avg", n_cores=n_cores,
                        n_tma_blocks=int(len(np.unique(tma_blks))), d0_n=n_d0, u_common_n=n_uc,
                        raw_avail_n=math.nan, raw_avail_pct=math.nan,
                        icc=float(dfs["icc"].mean()), pearson_r=math.nan, spearman_rho=math.nan,
                        mae=float(dfs["mae"].mean()), med_disc=float(dfs["med_disc"].mean()),
                        red_pct=float(dfs["red"].mean()),
                        red_ci_lo_core=float(dfs["red"].quantile(0.025)), red_ci_hi_core=float(dfs["red"].quantile(0.975)),
                        red_ci_lo_tma_block=math.nan, red_ci_hi_tma_block=math.nan,
                        red_ci_lo_organ=math.nan, red_ci_hi_organ=math.nan,
                        sd_ratio_a=float(dfs["sd_a"].mean()), sd_ratio_b=float(dfs["sd_b"].mean()),
                        bio_eta2_a=math.nan, bio_eta2_b=math.nan, reid_top5_pct=math.nan,
                        mean_4axis_kappa=math.nan, mitm_guardrail_pass_pct=math.nan,
                    ))

                # 3-Tier Paired Zero-Effect Null Tests & Organ-vs-Organ Reversals at r=25um
                u_mat = uc.reshape(n_cores, 15)
                uniq_org = sorted(np.unique(tt).tolist())
                for ak, (la, lb) in arms_log.items():
                    L0 = np.where(u_mat, la.reshape(n_cores, 15), np.nan)
                    L1 = np.where(u_mat, lb.reshape(n_cores, 15), np.nan)

                    # Pan-organ paired t & Wilcoxon
                    pt, pw = [], []
                    for pi in range(15):
                        ok = np.isfinite(L0[:, pi]) & np.isfinite(L1[:, pi])
                        if ok.sum() >= 5:
                            d = L0[ok, pi] - L1[ok, pi]
                            pt.append(float(ttest_rel(L0[ok, pi], L1[ok, pi]).pvalue))
                            pw.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
                    qt = multipletests(pt, alpha=0.05, method="fdr_bh")[0] if pt else np.array([], bool)
                    qw = multipletests(pw, alpha=0.05, method="fdr_bh")[0] if pw else np.array([], bool)

                    # Within-organ paired t (>= 3 paired cores in an organ)
                    wpt = []
                    for o in uniq_org:
                        om = tt == o
                        for pi in range(15):
                            ok = om & np.isfinite(L0[:, pi]) & np.isfinite(L1[:, pi])
                            if ok.sum() >= 3:
                                wpt.append(float(ttest_rel(L0[ok, pi], L1[ok, pi]).pvalue))
                    wqt = multipletests(wpt, alpha=0.05, method="fdr_bh")[0] if wpt else np.array([], bool)

                    # 50/50 disjoint two-center Welch (20 stratified splits)
                    rng_sp = np.random.default_rng(2026112577)
                    fdr_d = []
                    for _ in range(20):
                        hm = np.zeros(n_cores, dtype=bool)
                        for o in uniq_org:
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
                        track=tname, arm=ak, u_common_n=n_uc,
                        pan_n_tests=len(pt), pan_paired_t_fdr_n=int(qt.sum()),
                        pan_paired_t_fdr_pct=100.0 * float(qt.mean()) if len(qt) else math.nan,
                        pan_wilcoxon_fdr_n=int(qw.sum()),
                        pan_wilcoxon_fdr_pct=100.0 * float(qw.mean()) if len(qw) else math.nan,
                        within_organ_n_tests=len(wpt), within_organ_paired_t_fdr_n=int(wqt.sum()),
                        within_organ_paired_t_fdr_pct=100.0 * float(wqt.mean()) if len(wqt) else math.nan,
                        disjoint_50_50_welch_fdr_pct=float(np.mean(fdr_d)) if fdr_d else math.nan,
                    ))

                    # Organ-vs-organ differential colocalization sign reversals (top-50 by run_a |delta|)
                    org_ok = [o for o in uniq_org if (tt == o).sum() >= 2]
                    c_recs = []
                    for o1, o2 in combinations(org_ok, 2):
                        m1, m2 = tt == o1, tt == o2
                        for pi in range(15):
                            a1, a2 = L0[m1, pi], L0[m2, pi]
                            b1, b2 = L1[m1, pi], L1[m2, pi]
                            a1, a2 = a1[np.isfinite(a1)], a2[np.isfinite(a2)]
                            b1, b2 = b1[np.isfinite(b1)], b2[np.isfinite(b2)]
                            if len(a1) >= 2 and len(a2) >= 2 and len(b1) >= 2 and len(b2) >= 2:
                                da, db = float(a1.mean() - a2.mean()), float(b1.mean() - b2.mean())
                                c_recs.append((da, db))
                    if c_recs:
                        ca_arr = np.array(c_recs)
                        top_k = min(50, len(ca_arr))
                        top_idx = np.argsort(-np.abs(ca_arr[:, 0]))[:top_k]
                        rev_pct = 100.0 * float(np.mean(np.sign(ca_arr[top_idx, 0]) != np.sign(ca_arr[top_idx, 1])))
                        c_r = float(np.corrcoef(ca_arr[:, 0], ca_arr[:, 1])[0, 1])
                        rev_rows.append(dict(track=tname, arm=ak, n_contrasts=len(ca_arr), top_k=top_k,
                                             organ_contrast_r=c_r, top_k_reversal_pct=rev_pct))

                # Synchronized Joint-Permutation (M=199) Jaccard & Type-S error at r=25um
                pair_tuples = [(a, b) for a, b, _ in PAIRS]
                n_perm = int(get_default_protocol_v3()["primary_cross_platform"]["n_permutations_cohort"])  # 199

                def run_jp(arm_lbl: str, draws_a: np.ndarray, draws_b: np.ndarray):
                    pa_all, pb_all, da_all, db_all = [], [], [], []
                    for ci, (g1, g2) in enumerate(zip(ga, gb)):
                        adj1 = sparse.csr_matrix((np.ones(2 * g1["m"]), (np.r_[g1["e0"], g1["e1"]], np.r_[g1["e1"], g1["e0"]])), shape=(g1["n"], g1["n"]))
                        adj2 = sparse.csr_matrix((np.ones(2 * g2["m"]), (np.r_[g2["e0"], g2["e1"]], np.r_[g2["e1"], g2["e0"]])), shape=(g2["n"], g2["n"]))
                        d1 = lut[draws_a[:, g1["s0"]:g1["s1"]]] if draws_a.ndim == 2 else np.tile(lut[draws_a[g1["s0"]:g1["s1"]]], (2, 1))
                        d2 = lut[draws_b[:, g2["s0"]:g2["s1"]]] if draws_b.ndim == 2 else np.tile(lut[draws_b[g2["s0"]:g2["s1"]]], (2, 1))
                        r1 = joint_perm_pairs_from_draws(adj1, d1, pair_tuples, availability_threshold=0.90, n_permutations=n_perm, seed=20261125 + ci)
                        r2 = joint_perm_pairs_from_draws(adj2, d2, pair_tuples, availability_threshold=0.90, n_permutations=n_perm, seed=20261125 + ci)
                        pa_all.append(np.array([r1[ab].permutation_p_z for ab in pair_tuples], dtype=np.float64))
                        pb_all.append(np.array([r2[ab].permutation_p_z for ab in pair_tuples], dtype=np.float64))
                        da_all.append(np.array([math.log(r1[ab].r_mean) if (np.isfinite(r1[ab].r_mean) and r1[ab].r_mean > 0) else math.nan for ab in pair_tuples], dtype=np.float64))
                        db_all.append(np.array([math.log(r2[ab].r_mean) if (np.isfinite(r2[ab].r_mean) and r2[ab].r_mean > 0) else math.nan for ab in pair_tuples], dtype=np.float64))
                    pa_v = np.concatenate(pa_all); pb_v = np.concatenate(pb_all)
                    da_v = np.concatenate(da_all); db_v = np.concatenate(db_all)
                    sa_m = np.isfinite(pa_v) & (pa_v <= 0.05) & uc
                    sb_m = np.isfinite(pb_v) & (pb_v <= 0.05) & uc
                    both = sa_m & sb_m
                    same = int((both & (np.sign(da_v) == np.sign(db_v))).sum())
                    opp = int((both & (np.sign(da_v) != np.sign(db_v))).sum())
                    union = int((sa_m | sb_m).sum())
                    jp_rows.append(dict(
                        track=tname, arm=arm_lbl, n_perm=n_perm, u_common_n=n_uc,
                        n_sig_a=int(sa_m.sum()), n_sig_b=int(sb_m.sum()), n_both=int(both.sum()),
                        n_same_sign=same, n_opp_sign=opp,
                        dir_jaccard=same / max(1, union),
                        type_s_pct=100.0 * opp / max(1, int(both.sum())) if both.sum() > 0 else 0.0,
                    ))

                print(f"  [joint-perm M=199] {tname}...", flush=True)
                run_jp("Naive", base_a, base_b)
                run_jp("Depth_only_2s_q075", get_draws("a", ta_aq), get_draws("b", tb_aq))
                run_jp("Split_half_2s_default", get_draws("a", "p2_both"), get_draws("b", "p2_both"))
                run_jp("Single_draw_B1_rep0", get_draws("a", "p2_both")[0], get_draws("b", "p2_both")[0])

                # Track N-A same-cell CosMx IF concordance!
                if tname == "Track_NA_Xenium_vs_CosMx_IF":
                    def _mitm_if_extrap(eval_fn):
                        r_nat = eval_fn(base_b)
                        r_st = [r_nat] + [eval_fn(get_draws("b", qt), return_stack=True) for qt in ("q085", "q075", "q065", "q050")]
                        z_st = [np.where(np.isfinite(x), 1.0, np.nan) for x in r_st]
                        lad3 = summarize_thinning_ladder(r_st, z_st, lad_q, availability_threshold=0.90)
                        ex3 = extrapolate_ladder(
                            lad3, depth_scale=s_sh, target_depth=float(rates["d_mitm_geom"]),
                            pair_ids=np.tile(np.arange(3), n_cores), method="hier_eb", guardrail_mask=None,
                        )
                        g3 = build_extrapolation_guardrail(
                            np.repeat(c_med_sh, 3), np.repeat(c_flp50, 3), min_median_depth=2.0, max_cond_flip_rate=0.45
                        )
                        lp2 = safe_log(eval_fn(get_draws("b", "p2_both")))
                        return np.where(g3, ex3.log_r_pred, lp2)

                    if_rows.extend(eval_track_na_if(
                        tdir, gb, cells_b, cmap, score_cols, sc_b, base_b, b3_b,
                        get_draws("b", tb_aq), get_draws("b", "p2_both"),
                        get_draws("b", "mitm_geom" if dp == "b" else "q075"), sh,
                        ladder_sh_if_fn=_mitm_if_extrap if sh == "b" else None,
                    ))

    pd.DataFrame(sum_rows).to_csv(OUT_DIR / "ntma_ucommon_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT_DIR / "ntma_paired_null_tests.csv", index=False)
    pd.DataFrame(rev_rows).to_csv(OUT_DIR / "ntma_organ_reversals.csv", index=False)
    pd.DataFrame(jp_rows).to_csv(OUT_DIR / "ntma_joint_perm_jaccard.csv", index=False)
    pd.DataFrame(if_rows).to_csv(OUT_DIR / "ntma_trackNA_if_concordance.csv", index=False)

    print("\n=== PROSPECTIVE WANG 2025 nTMA RESULTS (r=25um, STRICT U_common) ===", flush=True)
    df_s = pd.DataFrame(sum_rows)
    print(df_s[df_s["radius_um"] == 25][["track", "arm", "u_common_n", "icc", "mae", "red_pct", "red_ci_lo_tma_block", "red_ci_hi_tma_block", "sd_ratio_a", "sd_ratio_b", "bio_eta2_a"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- Paired Null Tests ---")
    print(pd.DataFrame(null_rows)[["track", "arm", "pan_paired_t_fdr_pct", "pan_wilcoxon_fdr_pct", "within_organ_paired_t_fdr_pct", "disjoint_50_50_welch_fdr_pct"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n--- Joint Perm Jaccard & Type-S ---")
    print(pd.DataFrame(jp_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- Track N-A Same-Cell CosMx IF Concordance ---")
    print(pd.DataFrame(if_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    t0 = time.time()
    print(f"spatial_membership_audit version={sma.__version__}, DEFAULT_PROTOCOL_V3={DEFAULT_PROTOCOL_V3}", flush=True)
    stage1_prepare_all_tracks()
    stage2_singler_and_rates()
    stage3_evaluate_all_tracks()
    print(f"\nALL PROSPECTIVE nTMA STAGES COMPLETE IN {(time.time()-t0)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
