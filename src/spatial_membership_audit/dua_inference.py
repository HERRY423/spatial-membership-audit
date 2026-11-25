"""Detection uncertainty for ratios and correlated spatial-permutation ensembles.

The fixed-sample Monte Carlo error, the spatial randomization null and the
error on a future repeat have separate inputs and separate interpretations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

import numpy as np
from scipy import sparse
from .permutation_bridge import fixed_size_edge_moments


@dataclass(frozen=True)
class DUAResult:
    p_plus: Any
    r_mean: Any
    z_unadj: Any
    z_dua: Any
    z_dua_logr: Any
    z_var: Any
    det_var_logr: Any
    spatial_var_logr: Any
    se_total_logr: Any
    ci95_logr_lo: Any
    ci95_logr_hi: Any
    available: Any
    note: str
    mc_se_logr: Any = None
    ci95_width_logr: Any = None
    available_z: Any = None
    null_sd_z: Any = None
    permutation_p_z: Any = None
    n_defined: Any = None
    n_defined_z: Any = None
    null_corr_z: Any = None


def _draw_summary(values):
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    denom = np.maximum(n, 1)
    mean = np.where(n > 0, np.where(valid, values, 0).sum(axis=0) / denom, np.nan)
    deviations = np.where(valid, values - mean, 0)
    variance = np.where(n >= 2, (deviations ** 2).sum(axis=0) / np.maximum(n - 1, 1), np.nan)
    return n, mean, variance


def _bank(values, label, observed):
    bank = np.asarray(values, dtype=float)
    if bank.ndim != observed.ndim or bank.shape[1:] != observed.shape[1:] or len(bank) < 2:
        raise ValueError(f"{label} must have shape (K, ...) matching observed units, K >= 2")
    if np.isinf(bank).any():
        raise ValueError(f"{label} must contain finite values or NaN")
    return bank.reshape(len(bank), -1)


def summarize_dua_draws(r_draws, z_draws, vlog_draws=None, *, availability_threshold=0.90,
                        null_mean_z=None, repeat_log_errors=None, ratio_null_var=None) -> DUAResult:
    """Summarize independent conditional thinning draws on a fixed input.

    r_draws contains O/E for every E>0 (zero O is valid), NaN otherwise.
    z_draws may be undefined even when R exists. Both support (B,) or (B,U).
    MC SE for log(mean R) uses the delta method on all defined ratios.
    det_var_logr is Var(R)/mean(R)^2, a draw-dispersion proxy.

    null_mean_z[k] is the mean of eligible draw z scores under ONE joint
    permutation applied to every draw in replicate k, preserving their
    dependence. z_dua scales the observed mean about its known null mean zero
    by the sample SD of this bank. permutation_p_z is the two-sided rank
    p-value with the Monte Carlo plus-one correction. A zero-variance bank
    cannot supply a standardized score. No assumed unit null variance is used.

    Optional ratio_null_var contains per-draw Var(O/E) under label permutation.
    Alternatively vlog_draws approximates it as R^2 Var(log R) only for
    strictly positive draws. spatial_var_logr is a descriptive average on the
    delta scale. It is not the null variance of the ensemble mean.

    Optional repeat_log_errors are held-out signed differences, future minus
    current log(mean R), from exchangeable independent calibration units at
    the same level as the intended prediction. Using the conformal 95% order
    statistic of their absolute values yields a marginal prediction band.
    With fewer than 19 calibration errors the finite-sample band is unbounded.
    No calibration data means no band. se_total_logr and z_dua_logr are retired
    v1 fields and return NaN. v1 results require the archived v1 package.
    """
    thr = float(availability_threshold)
    if not 0.0 < thr <= 1.0:
        raise ValueError("availability_threshold must be in (0, 1]")
    r = np.asarray(r_draws, dtype=float)
    z = np.asarray(z_draws, dtype=float)
    if r.ndim not in (1, 2) or z.shape != r.shape or len(r) < 2:
        raise ValueError("r_draws and z_draws must share shape (B,) or (B,U), B >= 2")
    if np.any(r < 0) or np.isinf(r).any() or np.isinf(z).any():
        raise ValueError("R must be nonnegative; use NaN for an undefined R or z")
    if (np.isfinite(z) & ~np.isfinite(r)).any():
        raise ValueError("a finite z requires a defined ratio")
    rr = r.reshape(len(r), -1)
    zz = z.reshape(len(r), -1)
    nr, mu, vr = _draw_summary(rr)
    nz, zm, vz = _draw_summary(zz)
    p_plus = nr / len(r)
    gate_r = (nr >= 2) & (p_plus >= thr)
    gate_z = (nz >= 2) & (nz / len(r) >= thr)
    empty = np.full(mu.shape, np.nan)
    positive_mu = np.where(mu > 0, mu, np.nan)
    dispersion = vr / positive_mu ** 2
    mc_se = np.sqrt(dispersion / np.maximum(nr, 1))
    spatial = empty.copy()
    if vlog_draws is not None and ratio_null_var is not None:
        raise ValueError("supply only one per-draw spatial variance representation")
    var_input = ratio_null_var if ratio_null_var is not None else vlog_draws
    if var_input is not None:
        v = np.asarray(var_input, dtype=float)
        if v.shape != r.shape or np.any(v < 0) or np.isinf(v).any():
            raise ValueError("per-draw variances must match R and be nonnegative or NaN")
        vv = v.reshape(rr.shape)
        if ratio_null_var is None:
            vv = np.where(rr > 0, vv * rr ** 2, np.nan)
        nv, vbar, _ = _draw_summary(np.where(np.isfinite(rr), vv, np.nan))
        spatial = np.where(nv == nr, vbar / positive_mu ** 2, np.nan)

    sd_null, z_cal, p_perm, rho_null = empty.copy(), empty.copy(), empty.copy(), empty.copy()
    if null_mean_z is not None:
        bank = _bank(null_mean_z, "null_mean_z", z)
        n_null, _, null_variance = _draw_summary(bank)
        complete = (n_null == len(bank)) & gate_z
        sd_null[complete] = np.sqrt(null_variance[complete])
        nondegenerate = complete & (sd_null > 0)
        np.divide(zm, sd_null, out=z_cal, where=nondegenerate)
        corr_mask = nondegenerate & (nz >= 2)
        if corr_mask.any():
            nz_c = nz[corr_mask].astype(float)
            raw_rho = (nz_c * null_variance[corr_mask] - 1.0) / (nz_c - 1.0)
            rho_null[corr_mask] = np.clip(raw_rho, -1.0 / (nz_c - 1.0), 1.0)
        # Treat numerical ties in the randomization statistic as ties.
        cutoff = np.abs(zm) - 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(zm))
        exceed = (np.abs(bank) >= cutoff).sum(axis=0)
        p_perm[complete] = (1 + exceed[complete]) / (len(bank) + 1)

    lo, hi = empty.copy(), empty.copy()
    if repeat_log_errors is not None:
        errors = _bank(repeat_log_errors, "repeat_log_errors", r)
        pts = np.where(gate_r & (mu > 0), np.log(positive_mu), np.nan)
        lo, hi, _ = conformal_prediction_band(pts, errors, alpha=0.05)

    def output(arr):
        if r.ndim == 1:
            return arr[0].item()
        arr = np.array(arr, copy=True)
        arr.setflags(write=False)
        return arr

    return DUAResult(
        p_plus=output(p_plus), r_mean=output(mu), z_unadj=output(np.where(gate_z, zm, np.nan)),
        z_dua=output(z_cal), z_dua_logr=output(empty), z_var=output(vz),
        det_var_logr=output(dispersion), spatial_var_logr=output(spatial),
        se_total_logr=output(empty), ci95_logr_lo=output(lo), ci95_logr_hi=output(hi),
        available=output(gate_r), mc_se_logr=output(mc_se), ci95_width_logr=output(hi-lo),
        available_z=output(gate_z), null_sd_z=output(sd_null), permutation_p_z=output(p_perm),
        n_defined=output(nr), n_defined_z=output(nz), null_corr_z=output(rho_null),
        note="MC precision is conditional on fixed inputs. Spatial inference requires a joint permutation bank. Prediction requires independent repeat errors."
    )


def conformal_prediction_band(point_logr, repeat_log_errors, *, alpha: float = 0.05):
    """Finite-sample split-conformal prediction band ``(lo, hi, width)`` around ``point_logr``.

    ``repeat_log_errors`` holds out-of-sample signed errors ``future - current``
    from ``K`` exchangeable calibration units. Missing calibration entries make
    that unit's band ``NaN``; when ``K < ceil((1 - alpha) * (K + 1))`` the
    finite-sample band is unbounded (``(-inf, +inf, +inf)``).
    """
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    pts_in = np.asarray(point_logr, dtype=float)
    if pts_in.ndim not in (0, 1):
        raise ValueError("point_logr must be a scalar or 1D array")
    pts = pts_in.reshape(1) if pts_in.ndim == 0 else pts_in
    if np.isinf(pts).any():
        raise ValueError("point_logr must be finite or NaN")

    errs = np.asarray(repeat_log_errors, dtype=float)
    if np.isinf(errs).any():
        raise ValueError("repeat_log_errors must contain finite values or NaN")
    if pts_in.ndim == 0:
        if errs.ndim != 1 or len(errs) < 1:
            raise ValueError("repeat_log_errors must have shape (K,) with K >= 1 for scalar point_logr")
        err_mat = errs[:, None]
    else:
        if errs.ndim == 1:
            if len(errs) < 1:
                raise ValueError("repeat_log_errors must have length K >= 1")
            err_mat = np.broadcast_to(errs[:, None], (len(errs), len(pts)))
        elif errs.ndim == 2 and errs.shape[0] >= 1 and errs.shape[1] == len(pts):
            err_mat = errs
        else:
            raise ValueError(f"repeat_log_errors must have shape (K,) or (K, {len(pts)}) with K >= 1")

    k_ord = math.ceil((1.0 - alpha_f) * (len(err_mat) + 1) - 1e-12)
    lo = np.full(len(pts), np.nan, dtype=float)
    hi = np.full(len(pts), np.nan, dtype=float)
    for u in np.flatnonzero(np.isfinite(pts)):
        residuals = np.abs(err_mat[:, u])
        if not np.isfinite(residuals).all():
            continue
        radius = float(np.partition(residuals, k_ord - 1)[k_ord - 1]) if k_ord <= len(residuals) else math.inf
        lo[u], hi[u] = pts[u] - radius, pts[u] + radius
    width = hi - lo

    if pts_in.ndim == 0:
        return lo[0].item(), hi[0].item(), width[0].item()
    for arr in (lo, hi, width):
        arr.setflags(write=False)
    return lo, hi, width


def _exact_pair_moments_fraction(
    n: int,
    m: int,
    shared: int,
    disjoint: int,
    na: int,
    nb: int,
) -> tuple[float, float]:
    """Closed-form exact without-replacement edge moments for fixed sizes."""
    if na <= 0 or nb <= 0 or m <= 0 or n < 2:
        return math.nan, math.nan
    mean = Fraction(2 * na * nb * m, n * (n - 1))
    adj = Fraction(na * nb * (na + nb - 2), n * (n - 1) * (n - 2)) if n >= 3 else 0
    sep = Fraction(4 * na * (na - 1) * nb * (nb - 1), n * (n - 1) * (n - 2) * (n - 3)) if n >= 4 else 0
    var = mean + 2 * shared * adj + 2 * disjoint * sep - mean * mean
    if var < 0:
        raise ArithmeticError("negative exact permutation variance")
    return float(mean), float(var)


def _prepare_perm_inputs(adjacency, draw_labels, availability_threshold: float, n_permutations: int):
    thr = float(availability_threshold)
    if not math.isfinite(thr) or not 0.0 < thr <= 1.0:
        raise ValueError("availability_threshold must be in (0, 1]")
    if (
        isinstance(n_permutations, (bool, np.bool_))
        or not isinstance(n_permutations, (int, np.integer))
        or n_permutations < 0
        or n_permutations == 1
    ):
        raise ValueError("n_permutations must be zero or an integer >= 2")
    graph = sparse.csr_matrix(adjacency, dtype=float, copy=True)
    graph.sum_duplicates()
    graph.eliminate_zeros()
    meta = fixed_size_edge_moments(graph, 0, 0)
    n, m = int(meta["n"]), int(meta["m"])
    labels = np.asarray(draw_labels)
    if labels.ndim != 2 or labels.shape[0] < 2 or labels.shape[1] != n:
        raise ValueError(f"draw_labels must have shape (B, {n}) with B >= 2")
    return graph, meta, labels, n, m


def exact_perm_z_from_draws(adjacency, draw_labels, *, a=0, b=1,
                            availability_threshold=0.90, n_permutations=199, seed=0) -> DUAResult:
    """Exact single-draw moments and Monte Carlo joint-permutation inference for mean z.

    Arrays scale with the graph and the number of draws. Set n_permutations=0
    for the exact per-draw moments and MC precision alone.
    """
    if a == b:
        raise ValueError("distinct class codes a != b required")
    graph, meta, labels, n, m = _prepare_perm_inputs(
        adjacency, draw_labels, availability_threshold, n_permutations
    )
    g_up = sparse.triu(graph, k=1).tocoo()
    eu, ev = g_up.row, g_up.col
    B = len(labels)
    means, variances = np.full(B, np.nan), np.full(B, np.nan)
    ia, ib = labels == a, labels == b
    for i in range(B):
        means[i], variances[i] = _exact_pair_moments_fraction(
            n, m, meta["shared_edge_pairs"], meta["disjoint_edge_pairs"],
            int(ia[i].sum()), int(ib[i].sum()),
        )
    valid_r = np.isfinite(means) & (means > 0.0)
    valid_z = valid_r & np.isfinite(variances) & (variances > 0.0)
    sd = np.where(valid_z, np.sqrt(variances), np.nan)
    def observed(left, right):
        # Process one draw at a time to avoid a B-by-edges temporary.
        return np.array([np.count_nonzero((i[left] & j[right]) | (j[left] & i[right])) for i, j in zip(ia, ib)], dtype=float)
    obs = observed(eu, ev)
    r_draws = np.full(B, np.nan, dtype=float)
    np.divide(obs, means, out=r_draws, where=valid_r)
    z = np.full(B, np.nan, dtype=float)
    np.divide(obs - means, sd, out=z, where=valid_z)
    rnv = np.full(B, np.nan, dtype=float)
    np.divide(variances, means ** 2, out=rnv, where=valid_r)
    bank = None
    if n_permutations and int(valid_z.sum()) >= max(2, B * availability_threshold):
        rng = np.random.default_rng(seed)
        bank = np.empty(n_permutations)
        for j in range(n_permutations):
            perm = rng.permutation(n)
            obs_p = observed(perm[eu], perm[ev])
            bank[j] = float(np.mean((obs_p[valid_z] - means[valid_z]) / sd[valid_z]))
    return summarize_dua_draws(r_draws, z, ratio_null_var=rnv,
                               null_mean_z=bank, availability_threshold=availability_threshold)


def joint_perm_pairs_from_draws(
    adjacency,
    draw_labels,
    pairs,
    *,
    availability_threshold: float = 0.90,
    n_permutations: int = 199,
    seed: int = 0,
) -> dict[tuple[int, int], DUAResult]:
    """Vectorized multi-pair exact moments and shared joint-permutation DUA inference.

    Shares one joint vertex-permutation stream across all ``(a, b)`` pairs on a
    fixed graph so matrix multiplication cost is amortized across pairs while
    matching ``exact_perm_z_from_draws`` numerically.
    """
    pair_list = []
    seen = set()
    for item in pairs:
        if len(item) != 2:
            raise ValueError("each entry in pairs must be a 2-tuple (a, b)")
        a_raw, b_raw = item
        if isinstance(a_raw, (bool, np.bool_)) or isinstance(b_raw, (bool, np.bool_)):
            raise ValueError("class codes in pairs must be integers, not bool")
        a, b = int(a_raw), int(b_raw)
        if a == b:
            raise ValueError(f"distinct class codes a != b required, got ({a}, {b})")
        if (a, b) in seen:
            raise ValueError(f"duplicate pair ({a}, {b}) in pairs")
        seen.add((a, b))
        pair_list.append((a, b))
    if not pair_list:
        raise ValueError("pairs must contain at least one (a, b) tuple")

    graph, meta, labels, n, m = _prepare_perm_inputs(
        adjacency, draw_labels, availability_threshold, n_permutations
    )
    B = len(labels)
    used_codes = sorted({c for ab in pair_list for c in ab})
    C = len(used_codes)
    code_slot = {c: idx for idx, c in enumerate(used_codes)}

    # Indicator matrix Y of shape (n, C * B): exact integer counts via graph @ Y
    Y = np.empty((n, C * B), dtype=np.float64)
    for ci, code in enumerate(used_codes):
        Y[:, ci * B:(ci + 1) * B] = (labels == code).T
    na_cb = Y.sum(axis=0, dtype=np.float64).astype(np.int64).reshape(C, B)

    gy = graph @ Y
    obs_all = np.rint(
        np.einsum("nci,ndi->cdi", Y.reshape(n, C, B), gy.reshape(n, C, B))
    )

    shared = int(meta["shared_edge_pairs"])
    disjoint = int(meta["disjoint_edge_pairs"])
    moment_cache: dict[tuple[int, int], tuple[float, float]] = {}

    pair_data = []
    eligible_indices = []
    min_req = max(2, B * float(availability_threshold))

    for idx_p, (a, b) in enumerate(pair_list):
        ca, cb = code_slot[a], code_slot[b]
        means = np.full(B, np.nan, dtype=np.float64)
        variances = np.full(B, np.nan, dtype=np.float64)
        na_vec = na_cb[ca]
        nb_vec = na_cb[cb]
        for i in range(B):
            na_i, nb_i = int(na_vec[i]), int(nb_vec[i])
            key = (na_i, nb_i) if na_i <= nb_i else (nb_i, na_i)
            cached = moment_cache.get(key)
            if cached is None:
                cached = _exact_pair_moments_fraction(n, m, shared, disjoint, na_i, nb_i)
                moment_cache[key] = cached
            means[i], variances[i] = cached

        obs_p = obs_all[ca, cb]
        valid_r = np.isfinite(means) & (means > 0.0)
        r_p = np.full(B, np.nan, dtype=np.float64)
        np.divide(obs_p, means, out=r_p, where=valid_r)

        valid_z = valid_r & np.isfinite(variances) & (variances > 0.0)
        sd_p = np.where(valid_z, np.sqrt(variances), np.nan)
        z_p = np.full(B, np.nan, dtype=np.float64)
        np.divide(obs_p - means, sd_p, out=z_p, where=valid_z)

        rnv_p = np.full(B, np.nan, dtype=np.float64)
        np.divide(variances, means ** 2, out=rnv_p, where=valid_r)

        is_elig = n_permutations > 0 and int(valid_z.sum()) >= min_req
        if is_elig:
            eligible_indices.append(idx_p)
        pair_data.append(
            dict(
                ab=(a, b),
                ca=ca,
                cb=cb,
                means=means,
                sd=sd_p,
                valid_z=valid_z,
                r=r_p,
                z=z_p,
                rnv=rnv_p,
                bank=np.empty(n_permutations, dtype=np.float64) if is_elig else None,
            )
        )

    if n_permutations > 0 and eligible_indices:
        rng = np.random.default_rng(seed)
        for j in range(n_permutations):
            perm = rng.permutation(n)
            y_perm = Y[perm, :]
            gy_perm = graph @ y_perm
            obs_perm = np.rint(
                np.einsum("nci,ndi->cdi", y_perm.reshape(n, C, B), gy_perm.reshape(n, C, B))
            )
            for idx_p in eligible_indices:
                pd_i = pair_data[idx_p]
                vz = pd_i["valid_z"]
                op = obs_perm[pd_i["ca"], pd_i["cb"], vz]
                pd_i["bank"][j] = float(np.mean((op - pd_i["means"][vz]) / pd_i["sd"][vz]))

    results: dict[tuple[int, int], DUAResult] = {}
    for pd_i in pair_data:
        results[pd_i["ab"]] = summarize_dua_draws(
            pd_i["r"],
            pd_i["z"],
            ratio_null_var=pd_i["rnv"],
            null_mean_z=pd_i["bank"],
            availability_threshold=availability_threshold,
        )
    return results


def decompose_dua_uncertainty(
    r_draws,
    z_draws,
    *,
    null_mean_z=None,
    repeat_log_errors=None,
    ratio_null_var=None,
    availability_threshold: float = 0.90,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Explicitly decompose DUA inference into its three statistical regimes.

    Separates:
      1. Fixed-input Monte Carlo precision for ``log(E[R | E > 0])`` (Proposition 1);
      2. Within-run synchronized joint-permutation spatial null calibration (Proposition 2);
      3. Out-of-sample split-conformal repeat prediction band at level ``1 - alpha`` (Proposition 3).
    """
    res = summarize_dua_draws(
        r_draws,
        z_draws,
        ratio_null_var=ratio_null_var,
        null_mean_z=null_mean_z,
        repeat_log_errors=None,
        availability_threshold=availability_threshold,
    )
    r_arr = np.asarray(res.r_mean, dtype=float)
    avail_arr = np.asarray(res.available, dtype=bool)
    pos_mu = np.where(avail_arr & np.isfinite(r_arr) & (r_arr > 0.0), r_arr, np.nan)
    pts = np.log(pos_mu)
    if pts.ndim == 0:
        pts_in: Any = float(pts.item())
    else:
        pts_in = pts

    if repeat_log_errors is not None:
        ci_lo, ci_hi, ci_w = conformal_prediction_band(pts_in, repeat_log_errors, alpha=alpha)
    else:
        nan_like = math.nan if np.ndim(pts_in) == 0 else np.full_like(pts, np.nan)
        ci_lo, ci_hi, ci_w = nan_like, nan_like, nan_like

    return dict(
        regime_1_fixed_input_mc=dict(
            estimand="log(E[R | E > 0])",
            r_mean=res.r_mean,
            log_r_mean=pts_in,
            p_plus=res.p_plus,
            n_defined=res.n_defined,
            available=res.available,
            det_var_logr=res.det_var_logr,
            mc_se_logr=res.mc_se_logr,
        ),
        regime_2_spatial_permutation_null=dict(
            z_unadj=res.z_unadj,
            z_dua=res.z_dua,
            null_sd_z=res.null_sd_z,
            null_corr_z=res.null_corr_z,
            permutation_p_z=res.permutation_p_z,
            n_defined_z=res.n_defined_z,
            available_z=res.available_z,
            spatial_var_logr=res.spatial_var_logr,
        ),
        regime_3_conformal_repeat_prediction=dict(
            alpha=float(alpha),
            ci_logr_lo=ci_lo,
            ci_logr_hi=ci_hi,
            ci_width_logr=ci_w,
        ),
        dua_result=res,
    )
