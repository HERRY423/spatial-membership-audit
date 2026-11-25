"""Within-run Rarefaction-Extrapolation (Poisson-SIMEX) and guardrail helpers.

All estimators operate strictly within a single run's conditional thinning
ladder without using information from the comparison run. Noise coordinates
map retained depth ``d = q * s_k`` onto ``inv_sqrtd = 1 / sqrt(d)`` (default,
matching classification shot-noise SD scaling) or ``inv_d = 1 / d``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


_VALID_SCALES = ("inv_sqrtd", "inv_d", "log_inv_d")
_VALID_METHODS = ("prior_only", "hier_eb", "shrunk_step", "unit_linear", "attenuation_exp")
_VALID_MITM_MODES = ("geometric", "half_deep", "high_frac")


@dataclass(frozen=True)
class ThinningLadder:
    q_vals: np.ndarray
    r_mean: np.ndarray
    y_log: np.ndarray
    v_mc: np.ndarray
    p_plus: np.ndarray
    valid: np.ndarray
    availability_threshold: float
    n_draws: np.ndarray


@dataclass(frozen=True)
class ExtrapolationResult:
    r_pred: np.ndarray
    log_r_pred: np.ndarray
    slope_used: np.ndarray
    slope_se: np.ndarray
    extrap_se_logr: np.ndarray
    guardrail_triggered: np.ndarray
    target_depth: float
    depth_scale: float
    delta_xi: float
    method: str
    scale: str
    note: str


def _readonly(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def noise_coordinate(q, depth_scale: float, scale: str = "inv_sqrtd"):
    """Map retention rate(s) ``q`` at depth ``depth_scale`` to noise coordinate."""
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {_VALID_SCALES}")
    sk = float(depth_scale)
    if not math.isfinite(sk) or sk <= 0.0:
        raise ValueError("depth_scale must be finite and > 0")
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.size == 0 or not np.isfinite(q_arr).all() or np.any(q_arr <= 0.0):
        raise ValueError("q must contain finite positive values")
    d = q_arr * sk
    if scale == "inv_sqrtd":
        val = 1.0 / np.sqrt(d)
    elif scale == "inv_d":
        val = 1.0 / d
    else:
        val = np.log1p(1.0 / d)
    return float(val.item()) if np.ndim(q) == 0 else val


def target_noise_coordinate(target_depth: float, scale: str = "inv_sqrtd") -> float:
    """Map finite positive or infinite ``target_depth`` onto noise coordinate."""
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {_VALID_SCALES}")
    d = float(target_depth)
    if math.isinf(d) and d > 0.0:
        return 0.0
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("target_depth must be > 0 (or +inf)")
    if scale == "inv_sqrtd":
        return 1.0 / math.sqrt(d)
    if scale == "inv_d":
        return 1.0 / d
    return math.log1p(1.0 / d)


def _finite_mean_var(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    denom = np.maximum(n, 1)
    mean = np.where(n > 0, np.where(valid, values, 0.0).sum(axis=0) / denom, np.nan)
    dev = np.where(valid, values - np.where(np.isfinite(mean), mean, 0.0), 0.0)
    var = np.where(n >= 2, (dev ** 2).sum(axis=0) / np.maximum(n - 1, 1), np.nan)
    return n, mean, var


def summarize_thinning_ladder(
    r_banks,
    z_banks,
    q_vals,
    *,
    availability_threshold: float = 0.90,
) -> ThinningLadder:
    """Summarize native (``q=1.0``) + thinned replicate banks along a retention ladder.

    Native element ``m=0`` may have shape ``(U,)`` or ``(B, U)``; thinned
    elements ``q < 1.0`` must have shape ``(B, U)`` with ``B >= 2`` so Monte Carlo
    variance is identifiable. Valid zeros (``R == 0``) contribute to ``r_mean``
    and to the delta-method variance ``Var(R) / (n_fin * r_mean^2)``.
    """
    thr = float(availability_threshold)
    if not math.isfinite(thr) or not 0.0 < thr <= 1.0:
        raise ValueError("availability_threshold must be in (0, 1]")
    q_arr = np.asarray(q_vals, dtype=np.float64)
    if q_arr.ndim != 1 or q_arr.size < 1 or not np.isfinite(q_arr).all():
        raise ValueError("q_vals must be a nonempty 1D array of finite floats")
    if np.any(q_arr <= 0.0) or np.any(q_arr > 1.0) or not math.isclose(float(q_arr[0]), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("q_vals must lie in (0, 1] with q_vals[0] == 1.0")
    if len(np.unique(q_arr)) != len(q_arr):
        raise ValueError("q_vals must not contain duplicate values")
    if len(r_banks) != len(q_arr) or len(z_banks) != len(q_arr):
        raise ValueError("r_banks, z_banks, and q_vals must have equal length")

    m_len = len(q_arr)
    first_r = np.asarray(r_banks[0], dtype=np.float64)
    if first_r.ndim not in (1, 2) or first_r.shape[-1] < 1:
        raise ValueError("each entry in r_banks must have shape (U,) or (B, U) with U >= 1")
    u_len = first_r.shape[-1]

    r_mean = np.full((m_len, u_len), np.nan, dtype=np.float64)
    y_log = np.full((m_len, u_len), np.nan, dtype=np.float64)
    v_mc = np.full((m_len, u_len), np.nan, dtype=np.float64)
    p_plus = np.zeros((m_len, u_len), dtype=np.float64)
    valid = np.zeros((m_len, u_len), dtype=bool)
    n_draws = np.zeros(m_len, dtype=np.int64)

    for m, (rb, zb, qv) in enumerate(zip(r_banks, z_banks, q_arr)):
        rb_arr = np.asarray(rb, dtype=np.float64)
        zb_arr = np.asarray(zb, dtype=np.float64)
        if rb_arr.ndim not in (1, 2) or zb_arr.shape != rb_arr.shape or rb_arr.shape[-1] != u_len:
            raise ValueError("r_banks and z_banks entries must match unit count U")
        if np.any(rb_arr < 0.0) or np.isinf(rb_arr).any() or np.isinf(zb_arr).any():
            raise ValueError("ratios must be nonnegative and finite or NaN; z must be finite or NaN")
        if (np.isfinite(zb_arr) & ~np.isfinite(rb_arr)).any():
            raise ValueError("a finite z requires a defined ratio")

        if rb_arr.ndim == 1:
            if qv < 1.0 - 1e-12:
                raise ValueError("thinned ladder points (q < 1.0) must have shape (B, U) with B >= 2")
            n_draws[m] = 1
            r_mean[m] = rb_arr
            p_plus[m] = np.isfinite(zb_arr).astype(np.float64)
            ok = np.isfinite(rb_arr) & (rb_arr > 0.0)
            y_log[m, ok] = np.log(rb_arr[ok])
            valid[m] = ok
            continue

        if rb_arr.shape[0] < 2 and qv < 1.0 - 1e-12:
            raise ValueError("thinned ladder points (q < 1.0) require B >= 2 draws")
        n_draws[m] = rb_arr.shape[0]
        p_plus[m] = np.isfinite(zb_arr).mean(axis=0, dtype=np.float64)
        n_fin, mu, vr = _finite_mean_var(rb_arr)
        ok_m = n_fin >= 1
        r_mean[m, ok_m] = mu[ok_m]

        ok_v = (n_fin >= 2) & np.isfinite(mu) & (mu > 0.0)
        if ok_v.any():
            v_mc[m, ok_v] = vr[ok_v] / (n_fin[ok_v] * (mu[ok_v] ** 2))

        ok_reg = np.isfinite(mu) & (mu > 0.0) & (p_plus[m] >= thr)
        y_log[m, ok_reg] = np.log(mu[ok_reg])
        valid[m] = ok_reg

    thinned_idx = [m for m, qv in enumerate(q_arr) if qv < 1.0 - 1e-9]
    for m, qv in enumerate(q_arr):
        if qv >= 1.0 - 1e-9 and np.isnan(v_mc[m]).all():
            if thinned_idx:
                sub_v = v_mc[thinned_idx]
                for u in range(u_len):
                    fin = sub_v[np.isfinite(sub_v[:, u]), u]
                    v_mc[m, u] = 0.5 * float(np.median(fin)) if fin.size > 0 else 0.0
            else:
                v_mc[m] = 0.0

    return ThinningLadder(
        q_vals=_readonly(q_arr),
        r_mean=_readonly(r_mean),
        y_log=_readonly(y_log),
        v_mc=_readonly(v_mc),
        p_plus=_readonly(p_plus),
        valid=_readonly(valid),
        availability_threshold=thr,
        n_draws=_readonly(n_draws),
    )


def build_extrapolation_guardrail(
    region_median_depth,
    region_cond_flip_rate=None,
    *,
    min_median_depth: float = 2.0,
    max_cond_flip_rate: float = 0.45,
) -> np.ndarray:
    """Identify units/regions with sufficient signal depth for extrapolation.

    Units with non-finite or insufficient median transcript depth or excessive
    conditional flip rate are marked ``False`` so extrapolation safely falls back
    to rarefaction or native estimates.
    """
    min_d = float(min_median_depth)
    max_f = float(max_cond_flip_rate)
    if not math.isfinite(min_d) or min_d < 0.0:
        raise ValueError("min_median_depth must be finite and >= 0")
    if not math.isfinite(max_f) or not 0.0 < max_f <= 1.0:
        raise ValueError("max_cond_flip_rate must be in (0, 1]")

    depths = np.asarray(region_median_depth, dtype=np.float64)
    if depths.ndim != 1 or depths.size == 0 or np.isinf(depths).any():
        raise ValueError("region_median_depth must be a nonempty 1D array without +/-inf")
    fin_d = np.isfinite(depths)
    if np.any(depths[fin_d] < 0.0):
        raise ValueError("region_median_depth must be nonnegative")
    ok = fin_d & (depths >= min_d)

    if region_cond_flip_rate is not None:
        flips = np.asarray(region_cond_flip_rate, dtype=np.float64)
        if flips.shape != depths.shape or np.isinf(flips).any():
            raise ValueError("region_cond_flip_rate must match region_median_depth shape without +/-inf")
        fin_f = np.isfinite(flips)
        if np.any((flips[fin_f] < 0.0) | (flips[fin_f] > 1.0)):
            raise ValueError("region_cond_flip_rate values must lie in [0, 1]")
        ok = ok & fin_f & (flips <= max_f)

    return _readonly(ok)


def select_mitm_target_depth(
    depth_a: float,
    depth_b: float,
    *,
    mode: str = "geometric",
    high_frac: float = 0.75,
) -> float:
    """Choose a symmetric meet-in-the-middle target depth between two runs."""
    da = float(depth_a)
    db = float(depth_b)
    if not math.isfinite(da) or not math.isfinite(db) or da <= 0.0 or db <= 0.0:
        raise ValueError("depth_a and depth_b must be finite and > 0")
    if mode not in _VALID_MITM_MODES:
        raise ValueError(f"mode must be one of {_VALID_MITM_MODES}")
    hf = float(high_frac)
    if not math.isfinite(hf) or not 0.0 < hf <= 1.0:
        raise ValueError("high_frac must be in (0, 1]")
    lo, hi = min(da, db), max(da, db)
    if mode == "geometric":
        return math.sqrt(da * db)
    if mode == "half_deep":
        return max(lo, 0.5 * hi)
    return max(lo, hf * hi)


def _wls_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Fit weighted linear regression ``y = a + b * x``; return ``(a, b, se_b2)``."""
    sw = float(np.sum(w))
    if sw <= 0.0 or len(x) < 2:
        return float(np.mean(y)), 0.0, 1e6
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    dx = x - xm
    sxx = float(np.sum(w * dx * dx))
    if sxx <= 1e-18:
        return ym, 0.0, 1e6
    b = float(np.sum(w * dx * (y - ym)) / sxx)
    a = ym - b * xm
    resid = y - (a + b * x)
    dof = max(1, len(x) - 2)
    rss_w = float(np.sum(w * resid * resid))
    scale = max(1.0, rss_w / dof)
    return a, b, scale / sxx


def _fit_unit_slopes(ladder: ThinningLadder, xi_vals: np.ndarray):
    y_mat = ladder.y_log
    v_mat = ladder.v_mc
    val_mat = ladder.valid
    _, u_len = y_mat.shape
    dx_all = xi_vals - float(xi_vals[0])

    y0_hat = y_mat[0].copy()
    beta = np.full(u_len, np.nan, dtype=np.float64)
    se_b2 = np.full(u_len, np.nan, dtype=np.float64)
    for u in range(u_len):
        mask = val_mat[:, u]
        if mask.sum() < 3:
            continue
        x = dx_all[mask]
        y = y_mat[mask, u]
        vm = np.where(np.isfinite(v_mat[mask, u]), np.maximum(0.0, v_mat[mask, u]), 0.0)
        w = 1.0 / (vm + 1e-4)
        a, b, sb2 = _wls_line(x, y, w)
        y0_hat[u] = a
        beta[u] = b
        se_b2[u] = sb2
    return y0_hat, beta, se_b2


def _extrapolate_hierarchical_eb(
    y0_hat: np.ndarray,
    y0_native: np.ndarray,
    beta_u: np.ndarray,
    se_b2_u: np.ndarray,
    pair_arr: np.ndarray,
    delta_star: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_len = len(y0_native)
    y_pred = y0_native.copy()
    slope_used = np.full(u_len, np.nan, dtype=np.float64)
    slope_se = np.full(u_len, np.nan, dtype=np.float64)

    for pid in np.unique(pair_arr):
        pmask = pair_arr == pid
        fit_mask = pmask & np.isfinite(beta_u) & np.isfinite(y0_hat) & np.isfinite(se_b2_u)
        if fit_mask.sum() < 3:
            idx_p = np.flatnonzero(fit_mask)
            slope_used[idx_p] = beta_u[idx_p]
            slope_se[idx_p] = np.sqrt(np.maximum(0.0, se_b2_u[idx_p]))
            y_pred[idx_p] = y0_hat[idx_p] + beta_u[idx_p] * delta_star
            continue

        idx_f = np.flatnonzero(fit_mask)
        b_vec = beta_u[idx_f]
        y0_vec = y0_hat[idx_f]
        sb2_vec = np.maximum(1e-6, se_b2_u[idx_f])
        w_pr = 1.0 / (sb2_vec + float(np.median(sb2_vec)) + 1e-6)
        xp = np.column_stack([np.ones_like(y0_vec), y0_vec])
        xtwp = xp.T * w_pr
        ap = xtwp @ xp + np.diag([0.0, 1e-3])
        try:
            gam = np.linalg.solve(ap, xtwp @ b_vec)
        except np.linalg.LinAlgError:
            gam = np.array([float(np.median(b_vec)), 0.0])

        b_prior = gam[0] + gam[1] * y0_vec
        resid = b_vec - b_prior
        w_dl = 1.0 / sb2_vec
        q_dl = float(np.sum(w_dl * resid * resid))
        sw_dl = float(np.sum(w_dl))
        c_dl = sw_dl - float(np.sum(w_dl * w_dl)) / max(sw_dl, 1e-12)
        dof_dl = max(1, len(idx_f) - 2)
        tau2 = max(0.0, (q_dl - dof_dl) / max(c_dl, 1e-12)) if c_dl > 0.0 else 0.0

        w_eb = tau2 / (tau2 + sb2_vec)
        b_eb = w_eb * b_vec + (1.0 - w_eb) * b_prior
        var_prior_est = float(np.median(sb2_vec)) / max(1, len(idx_f))
        var_b_post = w_eb * sb2_vec + ((1.0 - w_eb) ** 2) * var_prior_est

        if method == "prior_only":
            slope_used[idx_f] = b_prior
            slope_se[idx_f] = math.sqrt(max(0.0, var_prior_est))
            y_pred[idx_f] = y0_vec + b_prior * delta_star
        elif method == "hier_eb":
            slope_used[idx_f] = b_eb
            slope_se[idx_f] = np.sqrt(np.maximum(0.0, var_b_post))
            y_pred[idx_f] = y0_vec + b_eb * delta_star
        else:
            step_eb = b_eb * delta_star
            var_step = (delta_star ** 2) * var_b_post
            shrink_snr = (step_eb * step_eb) / np.maximum(1e-12, step_eb * step_eb + var_step)
            slope_used[idx_f] = shrink_snr * b_eb
            slope_se[idx_f] = shrink_snr * np.sqrt(np.maximum(0.0, var_b_post))
            y_pred[idx_f] = y0_vec + shrink_snr * step_eb

        miss_idx = np.flatnonzero(pmask & (~fit_mask) & np.isfinite(y0_native))
        if len(miss_idx):
            bp_m = gam[0] + gam[1] * y0_native[miss_idx]
            slope_used[miss_idx] = bp_m
            extra_var = 0.0 if method == "prior_only" else tau2
            slope_se[miss_idx] = math.sqrt(max(0.0, var_prior_est + extra_var))
            y_pred[miss_idx] = y0_native[miss_idx] + bp_m * delta_star

    return y_pred, slope_used, slope_se


def _extrapolate_attenuation_exp(
    ladder: ThinningLadder,
    xi_vals: np.ndarray,
    delta_star: float,
    pair_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_mat = ladder.y_log
    val_mat = ladder.valid
    m_len, u_len = y_mat.shape
    xi0 = float(xi_vals[0])
    step_up = -delta_star
    y_pred = y_mat[0].copy()
    slope_used = np.full(u_len, np.nan, dtype=np.float64)
    slope_se = np.full(u_len, np.nan, dtype=np.float64)

    for pid in np.unique(pair_arr):
        pmask = pair_arr == pid
        ok0 = pmask & val_mat[0]
        if ok0.sum() < 4:
            continue
        common = pmask & val_mat.all(axis=0)
        if common.sum() < 4:
            common = ok0 & (val_mat.sum(axis=0) >= max(2, m_len - 1))
        if common.sum() < 4:
            continue

        y0_c = y_mat[0, common]
        sd0_c = math.sqrt(max(0.0, float(np.var(y0_c, ddof=1))))
        if sd0_c <= 1e-9:
            continue

        mu_pts, dxi_pts, a_sd_pts = [], [], []
        for m in range(m_len):
            vm_c = ok0 & val_mat[m]
            sub_m = common if val_mat[m, common].all() else vm_c
            if sub_m.sum() < 4:
                continue
            y0_s = y_mat[0, sub_m]
            ym_s = y_mat[m, sub_m]
            mu_pts.append(float(np.mean(ym_s)))
            dxi_pts.append(float(xi_vals[m] - xi0))
            if m == 0:
                a_sd_pts.append(1.0)
            else:
                sd0 = max(1e-9, float(np.std(y0_s, ddof=1)))
                sdm = float(np.std(ym_s, ddof=1))
                a_sd_pts.append(sdm / sd0)

        if len(dxi_pts) < 3:
            continue
        dxi_arr = np.asarray(dxi_pts, dtype=np.float64)
        mu_arr = np.asarray(mu_pts, dtype=np.float64)
        _, beta_mu, sb2_mu = _wls_line(dxi_arr, mu_arr, np.ones_like(dxi_arr))
        mu0 = float(np.mean(y0_c))
        mu_star = mu0 + beta_mu * delta_star

        pos_dx = dxi_arr > 1e-12
        if pos_dx.any():
            dx_sub = dxi_arr[pos_dx]
            a_sub = np.clip(np.asarray(a_sd_pts, dtype=np.float64)[pos_dx], 0.05, 2.0)
            kappa = max(0.0, float(np.dot(dx_sub, -np.log(a_sub)) / max(np.dot(dx_sub, dx_sub), 1e-12)))
            amp = float(min(2.5, max(0.4, math.exp(kappa * step_up))))
        else:
            amp = 1.0

        idx_ok = np.flatnonzero(ok0)
        y_new = mu_star + amp * (y_mat[0, idx_ok] - mu0)
        y_pred[idx_ok] = y_new
        if abs(delta_star) > 1e-12:
            slope_used[idx_ok] = (y_new - y_mat[0, idx_ok]) / delta_star
        else:
            slope_used[idx_ok] = 0.0
        slope_se[idx_ok] = math.sqrt(max(0.0, sb2_mu))

    return y_pred, slope_used, slope_se


def extrapolate_ladder(
    ladder: ThinningLadder,
    depth_scale: float,
    target_depth: float,
    pair_ids,
    *,
    method: str = "prior_only",
    scale: str = "inv_sqrtd",
    guardrail_mask=None,
    fallback_r=None,
    clip_log_r: float = 12.0,
) -> ExtrapolationResult:
    """Extrapolate within-run colocalization ratios along the thinning ladder.

    Supports ``'prior_only'`` (default), ``'hier_eb'``, ``'shrunk_step'``,
    ``'unit_linear'``, and ``'attenuation_exp'``. Units with ``guardrail_mask``
    equal to ``False`` safely revert to ``fallback_r`` (when provided) or native
    ``ladder.r_mean[0]``.
    """
    if not isinstance(ladder, ThinningLadder):
        raise ValueError("ladder must be a ThinningLadder instance")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}")
    clip = float(clip_log_r)
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip_log_r must be finite and > 0")

    u_len = ladder.r_mean.shape[1]
    pair_arr = np.asarray(pair_ids)
    if pair_arr.ndim != 1 or pair_arr.shape[0] != u_len:
        raise ValueError(f"pair_ids must be a 1D array of length {u_len}")

    if guardrail_mask is not None:
        gmask = np.asarray(guardrail_mask, dtype=bool)
        if gmask.shape != (u_len,):
            raise ValueError(f"guardrail_mask must have shape ({u_len},)")
    else:
        gmask = np.ones(u_len, dtype=bool)

    if fallback_r is not None:
        fb = np.asarray(fallback_r, dtype=np.float64)
        if fb.shape != (u_len,) or np.any(fb < 0.0) or np.isinf(fb).any():
            raise ValueError(f"fallback_r must match shape ({u_len},) and contain nonnegative finite or NaN values")
    else:
        fb = ladder.r_mean[0].copy()

    xi_vals = noise_coordinate(ladder.q_vals, depth_scale, scale=scale)
    xi0 = float(xi_vals[0])
    xi_star = target_noise_coordinate(target_depth, scale=scale)
    delta_star = xi_star - xi0

    y0_native = ladder.y_log[0].copy()
    if method == "unit_linear":
        y0_hat, beta_u, se_b2_u = _fit_unit_slopes(ladder, xi_vals)
        ok_fit = np.isfinite(beta_u) & np.isfinite(y0_hat)
        y_pred = np.where(ok_fit, y0_hat + beta_u * delta_star, y0_native)
        slope_used = np.where(ok_fit, beta_u, np.nan)
        slope_se = np.where(ok_fit, np.sqrt(np.maximum(0.0, se_b2_u)), np.nan)
    elif method in ("prior_only", "hier_eb", "shrunk_step"):
        y0_hat, beta_u, se_b2_u = _fit_unit_slopes(ladder, xi_vals)
        y_pred, slope_used, slope_se = _extrapolate_hierarchical_eb(
            y0_hat, y0_native, beta_u, se_b2_u, pair_arr, delta_star, method
        )
    else:
        y_pred, slope_used, slope_se = _extrapolate_attenuation_exp(
            ladder, xi_vals, delta_star, pair_arr
        )

    native_r0 = ladder.r_mean[0]
    r_out = np.full(u_len, np.nan, dtype=np.float64)
    log_r_out = np.full(u_len, np.nan, dtype=np.float64)
    triggered = ~gmask

    # Valid extrapolated units where guardrail passed
    extrap_ok = gmask & np.isfinite(y_pred)
    clipped_y = np.clip(y_pred[extrap_ok], -clip, clip)
    log_r_out[extrap_ok] = clipped_y
    r_out[extrap_ok] = np.exp(clipped_y)

    # Units passing guardrail without valid regression slope keep native finite ratios (including 0.0)
    keep_nat = gmask & (~extrap_ok) & np.isfinite(native_r0)
    r_out[keep_nat] = native_r0[keep_nat]
    pos_nat = keep_nat & (native_r0 > 0.0)
    log_r_out[pos_nat] = np.log(native_r0[pos_nat])

    # Guardrail-triggered fallback units
    fb_fin = triggered & np.isfinite(fb)
    r_out[fb_fin] = fb[fb_fin]
    fb_pos = triggered & np.isfinite(fb) & (fb > 0.0)
    log_r_out[fb_pos] = np.log(fb[fb_pos])

    slope_out = slope_used.copy()
    se_out = slope_se.copy()
    slope_out[triggered] = 0.0
    se_out[triggered] = 0.0
    extrap_se = np.where(np.isfinite(se_out), abs(delta_star) * se_out, np.nan)
    extrap_se[triggered] = 0.0

    note = (
        f"Within-run {method} extrapolation ({scale}) from depth_scale={float(depth_scale):.3g} "
        f"to target_depth={float(target_depth):.3g} (delta_xi={delta_star:.4f}); "
        f"guardrail triggered on {int(triggered.sum())}/{u_len} units."
    )
    return ExtrapolationResult(
        r_pred=_readonly(r_out),
        log_r_pred=_readonly(log_r_out),
        slope_used=_readonly(slope_out),
        slope_se=_readonly(se_out),
        extrap_se_logr=_readonly(extrap_se),
        guardrail_triggered=_readonly(triggered),
        target_depth=float(target_depth),
        depth_scale=float(depth_scale),
        delta_xi=float(delta_star),
        method=method,
        scale=scale,
        note=note,
    )
