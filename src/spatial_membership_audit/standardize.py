"""High-level scverse (AnnData / Squidpy) entry points for v3.0 colocalization standardization.

Exposes the frozen v3.0.0 default protocol and operational regimes of
``spatial-membership-audit`` through a single ergonomic interface operating on
raw count ``AnnData`` objects and any frozen cell-wise classifier callable:

  - ``mode="split_half"`` (alias ``"coverage"``; **v3.0 primary default**):
    unsupervised conditional Poisson split-half coverage matching
    (``match_split_half_coverage``, two-sided ``q0=0.75``, ``B=20``) with
    zero-inclusive ratio integration and synchronized joint-permutation DUA.
  - ``mode="mitm"`` (**optional secondary for moderate depth gaps <= 6-7x**):
    within-run ``1/sqrt(d)`` Rarefaction-Extrapolation to geometric-mean depth
    ``d* = sqrt(s_A * s_B)`` with empirical-Bayes shrinkage and low-depth /
    high-flip guardrails falling back to common-depth standardization.
  - ``mode="stability_weighted"`` (**single-specimen architectural map**):
    unthinned native-depth stability weighting (``w_i = 1 - f_i``) with
    exact closed-form weighted fixed-margin permutation moments.
  - ``mode="rarefaction"``: threshold-free post-QC depth-only rarefaction
    (``select_auto_q``) with zero-inclusive ``joint_perm_pairs_from_draws``.

Core numerical dependencies remain NumPy and SciPy; ``pandas`` and ``anndata``
are imported lazily so importing ``spatial_membership_audit`` never requires
the optional ``scverse`` extra.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from types import MappingProxyType
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse

from .adapters import validate_counts
from .coverage_matching import (
    build_split_half_curve,
    conditional_instability_metrics,
    match_split_half_coverage,
    split_half_disagreement,
)
from .depth_matching import select_auto_q
from .dua_inference import joint_perm_pairs_from_draws
from .ecosystem import build_graph
from .extrapolation import (
    build_extrapolation_guardrail,
    extrapolate_ladder,
    select_mitm_target_depth,
    summarize_thinning_ladder,
)
from .stability_weighting import cell_stability_weights, exact_weighted_perm_pair

_VALID_MODES = ("split_half", "coverage", "rarefaction", "mitm", "stability_weighted")

_DEFAULT_SPLIT_HALF_GRID = (1.00, 0.85, 0.75, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10)
_DEFAULT_LADDER_QS = (1.00, 0.85, 0.75, 0.65, 0.50)


def get_default_protocol_v3() -> dict[str, Any]:
    """Return a fresh JSON-serializable copy of the frozen v3.0 default protocol."""
    return {
        "protocol_version": "3.0.0",
        "primary_cross_platform": {
            "mode": "split_half",
            "pairing": "both",
            "q0": 0.75,
            "n_draws": 20,
            "split_half_grid": list(_DEFAULT_SPLIT_HALF_GRID),
            "split_half_max_cells": 10000,
            "split_half_reps": 3,
            "split_half_enforce_monotone": True,
            "split_half_min_q": 0.05,
            "ratio_estimator": "zero_inclusive_mean_o_over_mean_e",
            "availability_threshold": 0.90,
            "inference": "synchronized_joint_permutation",
            "n_permutations_cohort": 199,
            "n_permutations_single_unit": 999,
            "pvalue_formula": "phipson_smyth_plus_one",
            "radius_um": 25.0,
        },
        "secondary_moderate_gap": {
            "mode": "mitm",
            "mitm_mode": "geometric",
            "extrap_method": "hier_eb",
            "extrap_scale": "inv_sqrtd",
            "ladder_qs": list(_DEFAULT_LADDER_QS),
            "min_median_depth": 2.0,
            "recommended_min_median_depth": 25.0,
            "max_cond_flip_rate": 0.45,
            "fallback_mode": "rarefaction",
        },
        "single_specimen_map": {
            "mode": "stability_weighted",
            "stability_q": 0.75,
            "stability_power": 1.0,
            "n_draws": 20,
        },
    }


def _freeze_mapping(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze_mapping(v) for v in obj)
    return obj


DEFAULT_PROTOCOL_V3 = _freeze_mapping(get_default_protocol_v3())


@dataclass(frozen=True)
class StandardizedComparison:
    """Result container for standardized spatial colocalization across one or two runs."""

    mode: str
    table: Any  # pandas.DataFrame
    classes: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    rates: dict[str, Any]
    metadata: dict[str, Any]

    def to_squidpy_nhood_enrichment(
        self,
        *,
        run: str = "A",
        region: str | None = None,
        stat: str = "z_dua",
    ) -> dict[str, np.ndarray]:
        """Format pair statistics into Squidpy's ``(K, K)`` ``nhood_enrichment`` dict.

        Returns ``{"zscore": (K, K) float64, "count": (K, K) float64}`` ordered by
        ``self.classes``, suitable for ``squidpy.pl.nhood_enrichment`` or storing in
        ``adata.uns[f"{cluster_key}_nhood_enrichment"]``.
        """
        df = self.table
        sub = df[df["run"] == run]
        if region is not None:
            sub = sub[sub["region"] == str(region)]
        elif sub["region"].nunique() > 1:
            raise ValueError(
                f"Multiple regions present for run {run!r}; specify region=... explicitly."
            )
        k = len(self.classes)
        slot = {c: idx for idx, c in enumerate(self.classes)}
        zmat = np.full((k, k), np.nan, dtype=np.float64)
        rmat = np.full((k, k), np.nan, dtype=np.float64)
        np.fill_diagonal(zmat, 0.0)
        np.fill_diagonal(rmat, 1.0)
        for row in sub.itertuples(index=False):
            i = slot[str(row.label_a)]
            j = slot[str(row.label_b)]
            zv = float(getattr(row, stat))
            rv = float(row.r_est)
            zmat[i, j] = zmat[j, i] = zv
            rmat[i, j] = rmat[j, i] = rv
        return {"zscore": zmat, "count": rmat}


def _extract_aligned_counts(adata, count_layer: str | None, shared_genes: list[str] | None = None):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("unique cell (obs_names) and gene (var_names) identifiers required")
    if count_layer is not None:
        if count_layer not in adata.layers:
            raise ValueError(f"count_layer {count_layer!r} not found in adata.layers")
        mat = adata.layers[count_layer]
    elif "counts" in adata.layers:
        mat = adata.layers["counts"]
    elif adata.X is not None:
        mat = adata.X
    else:
        raise ValueError("No raw count matrix found in adata.layers['counts'] or adata.X")

    var_names = [str(g) for g in adata.var_names]
    if shared_genes is not None:
        idx_map = {g: i for i, g in enumerate(var_names)}
        missing = [g for g in shared_genes if g not in idx_map]
        if missing:
            raise ValueError(f"Missing shared genes in AnnData: {missing[:5]}")
        col_idx = [idx_map[g] for g in shared_genes]
        mat = mat[:, col_idx]
        var_names = list(shared_genes)

    if sparse.issparse(mat):
        arr = mat.tocsr()
        validate_counts(arr.data)
        arr = arr.astype(np.int64, copy=False)
    else:
        arr = np.asarray(mat)
        validate_counts(arr)
        arr = np.asarray(arr, dtype=np.int64)
    return arr, var_names


def _cell_totals(counts) -> np.ndarray:
    if sparse.issparse(counts):
        return np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()


def _binomial_thin(counts, q: float, rng: np.random.Generator):
    qf = float(q)
    if not math.isfinite(qf) or not 0.0 < qf <= 1.0:
        raise ValueError("retention probability q must lie in (0, 1]")
    if qf >= 1.0 - 1e-12:
        return counts.copy() if sparse.issparse(counts) else np.array(counts, copy=True)
    if sparse.issparse(counts):
        csr = counts.tocsr(copy=True)
        csr.data = rng.binomial(csr.data.astype(np.int64), qf).astype(np.int64)
        csr.eliminate_zeros()
        return csr
    out = np.zeros_like(counts, dtype=np.int64)
    nz = counts > 0
    if np.any(nz):
        out[nz] = rng.binomial(counts[nz].astype(np.int64), qf)
    return out


def _multinomial_split_half(counts, q: float, rng: np.random.Generator):
    th = _binomial_thin(counts, q, rng)
    if sparse.issparse(th):
        csr = th.tocsr(copy=True)
        h1_data = rng.binomial(csr.data.astype(np.int64), 0.5).astype(np.int64)
        h2_data = csr.data.astype(np.int64) - h1_data
        h1 = sparse.csr_matrix((h1_data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)
        h2 = sparse.csr_matrix((h2_data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)
        h1.eliminate_zeros()
        h2.eliminate_zeros()
        return h1, h2
    h1 = np.zeros_like(th, dtype=np.int64)
    nz = th > 0
    if np.any(nz):
        h1[nz] = rng.binomial(th[nz].astype(np.int64), 0.5)
    h2 = th - h1
    return h1, h2


def _resolve_regions_array(adata, region_key: str | None, default: str = "region_0") -> np.ndarray:
    if region_key is None:
        return np.full(adata.n_obs, default, dtype=object)
    if region_key not in adata.obs:
        raise ValueError(f"region_key {region_key!r} not found in adata.obs")
    col = adata.obs[region_key]
    if col.isna().any():
        raise ValueError(f"adata.obs[{region_key!r}] contains missing values")
    return col.astype(str).to_numpy()


def _resolve_region_adjacencies(
    adata,
    regions: np.ndarray,
    *,
    adjacency_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    radius_um: float | None = 25.0,
) -> dict[str, tuple[np.ndarray, sparse.csr_matrix]]:
    uniq = sorted(set(str(r) for r in regions))
    if adjacency_key in adata.obsp:
        g_all = sparse.csr_matrix(adata.obsp[adjacency_key], dtype=float).copy()
        g_all.sum_duplicates()
        g_all.eliminate_zeros()
        g_all = ((g_all + g_all.T) > 0).astype(float).tocsr()
        g_all.setdiag(0.0)
        g_all.eliminate_zeros()
        out = {}
        for r in uniq:
            idx = np.flatnonzero(regions == r)
            out[r] = (idx, g_all[idx][:, idx].tocsr())
        return out

    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Neither adata.obsp[{adjacency_key!r}] nor adata.obsm[{spatial_key!r}] found"
        )
    if radius_um is None or float(radius_um) <= 0.0:
        raise ValueError("radius_um must be positive when constructing radius graphs")
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    sg = build_graph(
        xy,
        tuple(str(c) for c in adata.obs_names),
        tuple(str(r) for r in regions),
        method="radius",
        radius_um=float(radius_um),
    )
    out = {}
    for r in uniq:
        idx = np.flatnonzero(regions == r)
        out[r] = (idx, sg.adjacency[idx][:, idx].tocsr())
    return out


def _call_classifier(
    classifier_fn: Callable,
    counts,
    gene_names: list[str],
    unresolved_label: str = "UNRESOLVED",
) -> np.ndarray:
    totals = _cell_totals(counts)
    raw_labels = np.asarray(classifier_fn(counts, gene_names), dtype=object)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != counts.shape[0]:
        raise ValueError(
            f"classifier_fn must return a 1D label sequence of length {counts.shape[0]}"
        )
    out = np.where(totals > 0.0, raw_labels.astype(str), str(unresolved_label))
    return out


def _draw_bank(
    counts,
    gene_names: list[str],
    q: float,
    n_draws: int,
    classifier_fn: Callable,
    rng: np.random.Generator,
    unresolved_label: str = "UNRESOLVED",
) -> np.ndarray:
    return np.vstack(
        [
            _call_classifier(
                classifier_fn,
                _binomial_thin(counts, q, rng),
                gene_names,
                unresolved_label=unresolved_label,
            )
            for _ in range(int(n_draws))
        ]
    )


def _encode_labels(arr: np.ndarray, code_map: dict[str, int], other_code: int) -> np.ndarray:
    flat = np.asarray(arr, dtype=object).ravel()
    enc = np.fromiter((code_map.get(str(v), other_code) for v in flat), dtype=np.int32, count=flat.size)
    return enc.reshape(arr.shape)


def _eval_run_joint_perm(
    run_name: str,
    reg_graphs: dict[str, tuple[np.ndarray, sparse.csr_matrix]],
    draw_labels_str: np.ndarray,
    classes: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    *,
    availability_threshold: float,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    code_map = {c: idx for idx, c in enumerate(classes)}
    other_code = len(classes)
    draws_enc = _encode_labels(draw_labels_str, code_map, other_code)
    pair_codes = [(code_map[a], code_map[b]) for a, b in pairs]

    rows: list[dict[str, Any]] = []
    for ri, (reg, (idx, adj)) in enumerate(reg_graphs.items()):
        sub_d = draws_enc[:, idx]
        res_map = joint_perm_pairs_from_draws(
            adj,
            sub_d,
            pair_codes,
            availability_threshold=availability_threshold,
            n_permutations=n_permutations,
            seed=int(seed) + ri * 97,
        )
        for (la, lb), (ca, cb) in zip(pairs, pair_codes):
            res = res_map[(ca, cb)]
            r_est = float(res.r_mean) if bool(res.available) else math.nan
            log_r = math.log(r_est) if (math.isfinite(r_est) and r_est > 0.0) else math.nan
            rows.append(
                dict(
                    run=run_name,
                    region=reg,
                    label_a=la,
                    label_b=lb,
                    r_est=r_est,
                    log_r_est=log_r,
                    p_plus=float(res.p_plus),
                    available=bool(res.available),
                    mc_se_logr=float(res.mc_se_logr),
                    z_unadj=float(res.z_unadj),
                    z_dua=float(res.z_dua),
                    permutation_p_z=float(res.permutation_p_z),
                    null_sd_z=float(res.null_sd_z),
                    null_corr_z=float(res.null_corr_z),
                    guardrail_triggered=False,
                )
            )
    return rows


def _stratified_probe_indices(
    regions: np.ndarray,
    max_cells: int | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Select deterministic region-stratified indices for split-half probing."""
    n_obs = int(regions.shape[0])
    if max_cells is None or n_obs <= int(max_cells):
        return None
    cap = int(max_cells)
    if cap < 1:
        raise ValueError("split_half_max_cells must be >= 1 when provided")
    uniq = sorted(set(str(r) for r in regions))
    chosen: list[np.ndarray] = []
    # Proportional allocation with floor 1 when cap >= n_regions, trimmed deterministically
    per_reg: dict[str, int] = {}
    rem = cap
    for r in uniq:
        cnt = int(np.sum(regions == r))
        alloc = max(1, int(math.floor(cap * cnt / n_obs))) if cap >= len(uniq) else 0
        alloc = min(cnt, alloc)
        per_reg[r] = alloc
        rem -= alloc
    if rem > 0:
        for r in uniq:
            if rem <= 0:
                break
            cnt = int(np.sum(regions == r))
            room = cnt - per_reg[r]
            add = min(room, rem)
            per_reg[r] += add
            rem -= add
    elif rem < 0:
        for r in reversed(uniq):
            if rem >= 0:
                break
            drop = min(per_reg[r], -rem)
            per_reg[r] -= drop
            rem += drop
    for r in uniq:
        k = per_reg[r]
        if k <= 0:
            continue
        idx_r = np.flatnonzero(regions == r)
        if k >= len(idx_r):
            chosen.append(idx_r)
        else:
            sel = np.sort(rng.choice(idx_r, size=k, replace=False))
            chosen.append(sel)
    return np.sort(np.concatenate(chosen))


def standardize_colocalization_anndata(
    adata_a,
    adata_b=None,
    *,
    classifier_fn: Callable[[Any, Sequence[str]], Sequence[str]],
    mode: str = "split_half",
    classes: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    count_layer: str | None = None,
    region_key: str | None = None,
    adjacency_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    radius_um: float = 25.0,
    q0: float = 0.75,
    pairing: str = "both",
    n_draws: int = 20,
    n_permutations: int = 199,
    availability_threshold: float = 0.90,
    mitm_mode: str = "geometric",
    extrap_method: str = "hier_eb",
    extrap_scale: str = "inv_sqrtd",
    ladder_qs: Sequence[float] = _DEFAULT_LADDER_QS,
    min_median_depth: float = 2.0,
    max_cond_flip_rate: float = 0.45,
    split_half_grid: Sequence[float] = _DEFAULT_SPLIT_HALF_GRID,
    split_half_reps: int = 3,
    split_half_max_cells: int | None = 10000,
    split_half_enforce_monotone: bool = True,
    split_half_min_q: float = 0.05,
    stability_q: float = 0.75,
    stability_power: float = 1.0,
    unresolved_label: str = "UNRESOLVED",
    seed: int = 0,
) -> StandardizedComparison:
    """Standardize spatial colocalization across one or two ``AnnData`` objects.

    Parameters
    ----------
    adata_a, adata_b:
        One ``AnnData`` (for ``mode="stability_weighted"``) or two ``AnnData``
        objects with raw nonnegative integer counts.
    classifier_fn:
        Frozen cell-wise classifier ``classifier_fn(counts, gene_names) -> labels``.
    mode:
        One of ``"split_half"`` (default; alias ``"coverage"``), ``"mitm"``,
        ``"stability_weighted"``, or ``"rarefaction"``.
    """
    import pandas as pd

    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}")
    if int(n_draws) < 2:
        raise ValueError("n_draws must be >= 2")
    if mode != "stability_weighted" and adata_b is None:
        raise ValueError(f"adata_b is required for cross-run mode={mode!r}")

    rng = np.random.default_rng(int(seed))

    if adata_b is not None:
        genes_a = [str(g) for g in adata_a.var_names]
        genes_b_set = {str(g) for g in adata_b.var_names}
        shared_genes = [g for g in genes_a if g in genes_b_set]
        if not shared_genes:
            raise ValueError("No overlapping genes found between adata_a and adata_b")
        counts_a, gene_names = _extract_aligned_counts(adata_a, count_layer, shared_genes)
        counts_b, _ = _extract_aligned_counts(adata_b, count_layer, shared_genes)
    else:
        counts_a, gene_names = _extract_aligned_counts(adata_a, count_layer, None)
        counts_b = None

    reg_a = _resolve_regions_array(adata_a, region_key, default="region_0")
    graphs_a = _resolve_region_adjacencies(
        adata_a, reg_a, adjacency_key=adjacency_key, spatial_key=spatial_key, radius_um=radius_um
    )
    base_a = _call_classifier(classifier_fn, counts_a, gene_names, unresolved_label=unresolved_label)

    if adata_b is not None:
        reg_b = _resolve_regions_array(adata_b, region_key, default="region_0")
        graphs_b = _resolve_region_adjacencies(
            adata_b, reg_b, adjacency_key=adjacency_key, spatial_key=spatial_key, radius_um=radius_um
        )
        base_b = _call_classifier(classifier_fn, counts_b, gene_names, unresolved_label=unresolved_label)
    else:
        reg_b, graphs_b, base_b = None, None, None

    # Resolve target classes & unordered pairs
    if classes is None:
        seen = set(str(x) for x in base_a)
        if base_b is not None:
            seen |= set(str(x) for x in base_b)
        seen.discard(str(unresolved_label))
        seen.discard("OTHER")
        class_tuple = tuple(sorted(seen))
    else:
        class_tuple = tuple(str(c) for c in classes)
    if len(class_tuple) < 2:
        raise ValueError("At least two distinct target classes are required")

    if pairs is None:
        pair_tuple = tuple(itertools.combinations(class_tuple, 2))
    else:
        pair_tuple = tuple((str(a), str(b)) for a, b in pairs)
        for a, b in pair_tuple:
            if a == b or a not in class_tuple or b not in class_tuple:
                raise ValueError(f"Invalid pair {(a, b)!r} for classes {class_tuple!r}")

    # Mode 1: Single-run native-depth stability weighting
    if mode == "stability_weighted":
        draws_a = _draw_bank(
            counts_a, gene_names, stability_q, n_draws, classifier_fn, rng, unresolved_label=unresolved_label
        )
        weights = cell_stability_weights(base_a, draws_a, power=stability_power)
        rows = []
        for reg, (idx, adj) in graphs_a.items():
            loc_b = base_a[idx]
            loc_w = weights[idx]
            for la, lb in pair_tuple:
                wa = np.where(loc_b == la, loc_w, 0.0)
                wb = np.where(loc_b == lb, loc_w, 0.0)
                sw = exact_weighted_perm_pair(adj, wa, wb)
                log_r = math.log(sw.r_weighted) if (math.isfinite(sw.r_weighted) and sw.r_weighted > 0.0) else math.nan
                rows.append(
                    dict(
                        run="A",
                        region=reg,
                        label_a=la,
                        label_b=lb,
                        r_est=float(sw.r_weighted),
                        log_r_est=log_r,
                        p_plus=1.0 if math.isfinite(sw.r_weighted) else 0.0,
                        available=bool(math.isfinite(sw.r_weighted)),
                        mc_se_logr=math.nan,
                        z_unadj=float(sw.z_weighted),
                        z_dua=float(sw.z_weighted),
                        permutation_p_z=math.nan,
                        null_sd_z=1.0,
                        null_corr_z=math.nan,
                        guardrail_triggered=False,
                        effective_n_a=float(sw.effective_n_a),
                        effective_n_b=float(sw.effective_n_b),
                    )
                )
        return StandardizedComparison(
            mode=mode,
            table=pd.DataFrame(rows),
            classes=class_tuple,
            pairs=pair_tuple,
            rates={"stability_q": float(stability_q), "stability_power": float(stability_power)},
            metadata={"n_shared_genes": len(gene_names), "n_draws": int(n_draws)},
        )

    # Cross-run post-QC positive transcript medians
    tot_a = _cell_totals(counts_a)
    tot_b = _cell_totals(counts_b)
    pos_a = tot_a[tot_a > 0.0]
    pos_b = tot_b[tot_b > 0.0]
    if pos_a.size == 0 or pos_b.size == 0:
        raise ValueError("Both runs must contain cells with positive shared-panel counts")
    s_a = float(np.median(pos_a))
    s_b = float(np.median(pos_b))

    # Mode 2: Threshold-free post-QC common-depth rarefaction
    if mode == "rarefaction":
        sel = select_auto_q(tot_a, tot_b, pairing=pairing, sym_q=q0, balance_threshold=1.0)
        draws_a = (
            np.tile(base_a[None, :], (n_draws, 1))
            if sel.q_a >= 1.0 - 1e-12
            else _draw_bank(counts_a, gene_names, sel.q_a, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
        )
        draws_b = (
            np.tile(base_b[None, :], (n_draws, 1))
            if sel.q_b >= 1.0 - 1e-12
            else _draw_bank(counts_b, gene_names, sel.q_b, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
        )
        rows = _eval_run_joint_perm(
            "A", graphs_a, draws_a, class_tuple, pair_tuple,
            availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 101,
        ) + _eval_run_joint_perm(
            "B", graphs_b, draws_b, class_tuple, pair_tuple,
            availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 202,
        )
        return StandardizedComparison(
            mode=mode,
            table=pd.DataFrame(rows),
            classes=class_tuple,
            pairs=pair_tuple,
            rates={"q_a": sel.q_a, "q_b": sel.q_b, "medpos_a": s_a, "medpos_b": s_b, "depth_ratio": sel.ratio_medpos},
            metadata={"n_shared_genes": len(gene_names), "n_draws": int(n_draws), "pairing": pairing, "q0": float(q0)},
        )

    # Mode 3: Conditional split-half coverage matching (v3 primary default)
    if mode in ("split_half", "coverage"):
        q_grid = sorted({float(q) for q in split_half_grid} | {1.0, float(q0)})
        psi_a_vals, psi_b_vals = [], []
        idx_probe_a = _stratified_probe_indices(reg_a, split_half_max_cells, rng)
        idx_probe_b = _stratified_probe_indices(reg_b, split_half_max_cells, rng)
        probe_counts_a = counts_a[idx_probe_a] if idx_probe_a is not None else counts_a
        probe_counts_b = counts_b[idx_probe_b] if idx_probe_b is not None else counts_b
        mask_a = (tot_a[idx_probe_a] if idx_probe_a is not None else tot_a) > 0.0
        mask_b = (tot_b[idx_probe_b] if idx_probe_b is not None else tot_b) > 0.0
        for qv in q_grid:
            ha_list, hb_list = [], []
            for _ in range(int(split_half_reps)):
                c1, c2 = _multinomial_split_half(probe_counts_a, qv, rng)
                ha_list.append(_call_classifier(classifier_fn, c1, gene_names, unresolved_label=unresolved_label))
                hb_list.append(_call_classifier(classifier_fn, c2, gene_names, unresolved_label=unresolved_label))
            psi_a, u_a, _ = split_half_disagreement(np.vstack(ha_list), np.vstack(hb_list), classifiable_mask=mask_a)
            psi_a_vals.append(psi_a)

            ha_list, hb_list = [], []
            for _ in range(int(split_half_reps)):
                c1, c2 = _multinomial_split_half(probe_counts_b, qv, rng)
                ha_list.append(_call_classifier(classifier_fn, c1, gene_names, unresolved_label=unresolved_label))
                hb_list.append(_call_classifier(classifier_fn, c2, gene_names, unresolved_label=unresolved_label))
            psi_b, u_b, _ = split_half_disagreement(np.vstack(ha_list), np.vstack(hb_list), classifiable_mask=mask_b)
            psi_b_vals.append(psi_b)

        curve_a = build_split_half_curve(
            q_grid,
            psi_a_vals,
            unresolved_frac=float(1.0 - mask_a.mean()),
            enforce_monotone=split_half_enforce_monotone,
        )
        curve_b = build_split_half_curve(
            q_grid,
            psi_b_vals,
            unresolved_frac=float(1.0 - mask_b.mean()),
            enforce_monotone=split_half_enforce_monotone,
        )
        cm = match_split_half_coverage(
            curve_a,
            curve_b,
            pairing=pairing,
            q0=q0,
            min_q=split_half_min_q,
        )

        draws_a = (
            np.tile(base_a[None, :], (n_draws, 1))
            if cm.q_a >= 1.0 - 1e-12
            else _draw_bank(counts_a, gene_names, cm.q_a, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
        )
        draws_b = (
            np.tile(base_b[None, :], (n_draws, 1))
            if cm.q_b >= 1.0 - 1e-12
            else _draw_bank(counts_b, gene_names, cm.q_b, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
        )
        rows = _eval_run_joint_perm(
            "A", graphs_a, draws_a, class_tuple, pair_tuple,
            availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 303,
        ) + _eval_run_joint_perm(
            "B", graphs_b, draws_b, class_tuple, pair_tuple,
            availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 404,
        )
        return StandardizedComparison(
            mode=mode,
            table=pd.DataFrame(rows),
            classes=class_tuple,
            pairs=pair_tuple,
            rates={
                "q_a": cm.q_a,
                "q_b": cm.q_b,
                "target_psi": cm.target_psi,
                "noisier_side": cm.noisier_side,
                "medpos_a": s_a,
                "medpos_b": s_b,
                "clamped": cm.clamped,
            },
            metadata={
                "n_shared_genes": len(gene_names),
                "n_draws": int(n_draws),
                "pairing": pairing,
                "q0": float(q0),
                "n_probe_cells_a": int(probe_counts_a.shape[0]),
                "n_probe_cells_b": int(probe_counts_b.shape[0]),
            },
        )

    # Mode 4: Rarefaction-Extrapolation MITM with automatic low-depth / high-flip guardrail (P1)
    d_target = select_mitm_target_depth(s_a, s_b, mode=mitm_mode)
    sh_is_a = s_a <= s_b
    sh_run, dp_run = ("A", "B") if sh_is_a else ("B", "A")
    sh_counts, dp_counts = (counts_a, counts_b) if sh_is_a else (counts_b, counts_a)
    sh_base, dp_base = (base_a, base_b) if sh_is_a else (base_b, base_a)
    sh_tot = tot_a if sh_is_a else tot_b
    sh_graphs, dp_graphs = (graphs_a, graphs_b) if sh_is_a else (graphs_b, graphs_a)
    s_sh, s_dp = (s_a, s_b) if sh_is_a else (s_b, s_a)

    # Deep run: rarefy down to MITM target depth d_target + prepare rarefaction fallback at q0 * s_sh / s_dp
    q_dp_mitm = min(1.0, d_target / s_dp)
    q_dp_fb = min(1.0, float(q0) * s_sh / s_dp)
    dp_mitm_draws = _draw_bank(dp_counts, gene_names, q_dp_mitm, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
    dp_fb_draws = _draw_bank(dp_counts, gene_names, q_dp_fb, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)

    dp_mitm_rows = _eval_run_joint_perm(
        dp_run, dp_graphs, dp_mitm_draws, class_tuple, pair_tuple,
        availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 505,
    )
    dp_fb_rows = _eval_run_joint_perm(
        dp_run, dp_graphs, dp_fb_draws, class_tuple, pair_tuple,
        availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 606,
    )

    # Shallow run: build thinning ladder [1.0, 0.85, 0.75, 0.65, 0.50]
    q_lad = [float(q) for q in ladder_qs]
    if not math.isclose(q_lad[0], 1.0, abs_tol=1e-12):
        raise ValueError("ladder_qs[0] must be 1.0")
    code_map = {c: idx for idx, c in enumerate(class_tuple)}
    other_code = len(class_tuple)
    pair_codes = [(code_map[a], code_map[b]) for a, b in pair_tuple]
    sh_regs = list(sh_graphs.keys())
    n_p = len(pair_tuple)
    u_tot = len(sh_regs) * n_p

    sh_r_banks = []
    sh_z_banks = []
    sh_draws_by_q: dict[float, np.ndarray] = {}

    for qv in q_lad:
        if qv >= 1.0 - 1e-12:
            d_str = np.tile(sh_base[None, :], (2, 1))
            b_eff = 1
        else:
            d_str = _draw_bank(sh_counts, gene_names, qv, n_draws, classifier_fn, rng, unresolved_label=unresolved_label)
            sh_draws_by_q[qv] = d_str
            b_eff = int(n_draws)
        d_enc = _encode_labels(d_str, code_map, other_code)
        r_mat = np.full((b_eff, u_tot), np.nan, dtype=np.float64)
        z_mat = np.full((b_eff, u_tot), np.nan, dtype=np.float64)
        for ri, reg in enumerate(sh_regs):
            idx, adj = sh_graphs[reg]
            # Fast single-draw moments without permutation bank for ladder regression
            res_map = joint_perm_pairs_from_draws(
                adj, d_enc[:, idx], pair_codes, availability_threshold=0.01, n_permutations=0, seed=0
            )
            # Re-extract per-draw exact ratios via direct vectorized pass for ladder MC variance
            sub_enc = d_enc[:b_eff, idx]
            n_loc = len(idx)
            g_up = sparse.triu(adj, k=1).tocoo()
            eu, ev = g_up.row, g_up.col
            m_loc = len(eu)
            c_const = (2.0 * m_loc) / (n_loc * (n_loc - 1.0)) if n_loc >= 2 and m_loc > 0 else 0.0
            for b_i in range(b_eff):
                lab_b = sub_enc[b_i]
                for pi, (ca, cb) in enumerate(pair_codes):
                    ma = lab_b == ca
                    mb = lab_b == cb
                    na = float(ma.sum())
                    nb = float(mb.sum())
                    if na > 0.0 and nb > 0.0 and c_const > 0.0:
                        obs = float(np.count_nonzero((ma[eu] & mb[ev]) | (mb[eu] & ma[ev])))
                        exp_v = c_const * na * nb
                        r_mat[b_i, ri * n_p + pi] = obs / exp_v
                        z_mat[b_i, ri * n_p + pi] = 1.0 if res_map[(ca, cb)].available_z else np.nan
        sh_r_banks.append(r_mat[0] if b_eff == 1 else r_mat)
        sh_z_banks.append(z_mat[0] if b_eff == 1 else z_mat)

    ladder = summarize_thinning_ladder(sh_r_banks, sh_z_banks, q_lad, availability_threshold=availability_threshold)

    # Guardrail on shallow run per region: median depth & conditional q=0.50 flip rate
    q_flip = min(sh_draws_by_q.keys(), key=lambda x: abs(x - 0.50))
    draws_flip = sh_draws_by_q[q_flip]
    reg_med = np.array([float(np.median(sh_tot[sh_graphs[r][0]])) for r in sh_regs], dtype=np.float64)
    reg_flp = np.array(
        [
            conditional_instability_metrics(
                sh_base[sh_graphs[r][0]],
                draws_flip[:, sh_graphs[r][0]],
                unresolved_label=str(unresolved_label),
            ).cond_flip_rate
            for r in sh_regs
        ],
        dtype=np.float64,
    )
    gmask = build_extrapolation_guardrail(
        np.repeat(reg_med, n_p),
        np.repeat(reg_flp, n_p),
        min_median_depth=min_median_depth,
        max_cond_flip_rate=max_cond_flip_rate,
    )

    q_fb_sh = min(sh_draws_by_q.keys(), key=lambda x: abs(x - float(q0)))
    sh_fb_rows = _eval_run_joint_perm(
        sh_run, sh_graphs, sh_draws_by_q[q_fb_sh], class_tuple, pair_tuple,
        availability_threshold=availability_threshold, n_permutations=n_permutations, seed=seed + 707,
    )
    sh_fb_r = np.array([row["r_est"] for row in sh_fb_rows], dtype=np.float64)

    pair_ids = np.tile(np.arange(n_p), len(sh_regs))
    ext = extrapolate_ladder(
        ladder,
        depth_scale=s_sh,
        target_depth=d_target,
        pair_ids=pair_ids,
        method=extrap_method,
        scale=extrap_scale,
        guardrail_mask=gmask,
        fallback_r=sh_fb_r,
    )

    sh_rows = []
    reg_triggered = {}
    for u_idx, fb_row in enumerate(sh_fb_rows):
        trig = bool(ext.guardrail_triggered[u_idx])
        reg_triggered[fb_row["region"]] = trig
        r_val = float(ext.r_pred[u_idx])
        log_r = float(ext.log_r_pred[u_idx])
        sh_rows.append(
            dict(
                **{k: v for k, v in fb_row.items() if k not in ("r_est", "log_r_est", "mc_se_logr", "guardrail_triggered")},
                r_est=r_val,
                log_r_est=log_r,
                mc_se_logr=float(ext.extrap_se_logr[u_idx]) if not trig else float(fb_row["mc_se_logr"]),
                guardrail_triggered=trig,
            )
        )

    # Apply matched fallback on deep run for any region where guardrail fired (when regions share names)
    dp_fb_map = {(r["region"], r["label_a"], r["label_b"]): r for r in dp_fb_rows}
    dp_rows = []
    for row in dp_mitm_rows:
        if reg_triggered.get(row["region"], False):
            fb = dict(dp_fb_map[(row["region"], row["label_a"], row["label_b"])])
            fb["guardrail_triggered"] = True
            dp_rows.append(fb)
        else:
            dp_rows.append(row)

    all_rows = (sh_rows + dp_rows) if sh_is_a else (dp_rows + sh_rows)
    return StandardizedComparison(
        mode=mode,
        table=pd.DataFrame(all_rows),
        classes=class_tuple,
        pairs=pair_tuple,
        rates={
            "medpos_a": s_a,
            "medpos_b": s_b,
            "shallower_run": sh_run,
            "target_depth": float(d_target),
            "q_deep_mitm": float(q_dp_mitm),
            "q_deep_fallback": float(q_dp_fb),
            "guardrail_triggered_units": int(ext.guardrail_triggered.sum()),
        },
        metadata={
            "n_shared_genes": len(gene_names),
            "n_draws": int(n_draws),
            "mitm_mode": mitm_mode,
            "extrap_method": extrap_method,
            "extrap_scale": extrap_scale,
        },
    )
