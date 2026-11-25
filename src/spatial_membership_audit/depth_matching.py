"""Bidirectional matching of observed gene means and thinning sensitivity.

These operations do not identify physical efficiency. Matching means used to
fit the rates is an identity, not a validation result. Beta-binomial summaries
are conditional on a supplied histogram and rho; rho is not estimated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BidirectionalMatch:
    retention_a: np.ndarray
    retention_b: np.ndarray
    common_mean: np.ndarray
    status: np.ndarray


def bidirectional_retention(mean_a, mean_b) -> BidirectionalMatch:
    """Thin only the larger observed mean, independently for each aligned gene.

    Both zero: retain both sides, status BOTH_ZERO (no efficiency information).
    One zero: remove counts only from the positive side, COMMON_TARGET_ZERO.
    No pseudocounts, filtering, ratio clipping, or imputation is performed.
    Caller is responsible for aligning gene identities and measurement units.
    Applying calibration-fitted rates to new samples need not thin the larger
    mean in the new samples; do not silently refit a claimed transfer test.
    """
    a, b = np.asarray(mean_a, dtype=float), np.asarray(mean_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("aligned nonempty one-dimensional gene means required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("gene means must be finite and nonnegative")
    common = np.minimum(a, b)
    pa, pb = np.ones_like(a), np.ones_like(b)
    np.divide(common, a, out=pa, where=a > 0)
    np.divide(common, b, out=pb, where=b > 0)
    status = np.full(a.shape, "TIED", dtype=object)
    status[a > b], status[b > a] = "THIN_A", "THIN_B"
    status[(a == 0) ^ (b == 0)] = "COMMON_TARGET_ZERO"
    status[(a == 0) & (b == 0)] = "BOTH_ZERO"
    for value in (pa, pb, common, status):
        value.setflags(write=False)
    return BidirectionalMatch(pa, pb, common, status)


def thin_histogram_summary(histogram, retention: float, *, rho: float = 0.0) -> dict:
    """Analytic mean, population variance, zero fraction of a thinned histogram.

    Histogram[k] is the number of deposited cells with integer count k. For
    rho>0, each cell/gene has P~Beta(p*kappa,(1-p)*kappa), kappa=(1-rho)/rho,
    then Y|X,P~Binomial(X,P). Conditional variance inflation is 1+(X-1)*rho.
    This is a retention-mixture stress test, not independent NB data splitting.
    p=0, p=1 and rho=0 are evaluated as their degenerate/binomial limits.
    Variance is across the empirical cell distribution, with denominator N.
    It is NOT a standard error or variance of an aggregate spatial endpoint.
    """
    h = np.asarray(histogram, dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all() or (h < 0).any() or (h != np.floor(h)).any():
        raise ValueError("histogram must contain finite nonnegative integer frequencies")
    n = float(h.sum())
    if not np.isfinite(n) or not 0 < n <= 2**53:
        raise ValueError("histogram must have a positive exactly representable cell count")
    p = float(retention)
    rho = float(rho)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("retention must be in [0,1]")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be in [0,1)")
    x = np.arange(h.size, dtype=float)
    w = h/n
    mean_x = float(w @ x)
    var_x = float(w @ ((x-mean_x)**2))
    factorial2 = float(w @ (x*(x-1)))
    mean = p*mean_x
    conditional_var = p*(1-p)*(mean_x+rho*factorial2)
    variance = p*p*var_x+conditional_var
    if p == 0:
        zero = 1.0
    elif p == 1:
        zero = float(w[0])
    elif rho == 0:
        zero = float(w @ np.exp(x*np.log1p(-p)))
    else:
        # P0(k+1)/P0(k) = (beta+k)/(alpha+beta+k).
        # Work in log space, avoiding subtraction of two large log-beta values.
        j = np.arange(h.size-1, dtype=float)
        log_step = np.log1p(-p*((1-rho)/(1-rho+j*rho)))
        log_zero = np.r_[0.0, np.cumsum(log_step)]
        zero = float(w @ np.exp(log_zero))
    if not np.isfinite([mean, variance, zero]).all() or not 0 <= zero <= 1+1e-14:
        raise ArithmeticError("nonfinite/out-of-range thinning summary")
    return dict(n_cells=int(n), mean=mean, variance=variance,
                zero_fraction=zero, conditional_variance_mean=conditional_var,
                source_mean=mean_x, source_variance=var_x,
                source_factorial_moment2=factorial2, retention=p, rho=rho,
                model="binomial" if rho == 0 else "beta_binomial",
                variance_scope="empirical_cell_distribution_not_estimation_uncertainty")


@dataclass(frozen=True)
class AutoQSelection:
    mode: str
    ratio_medpos: float
    ratio_mean: float
    q_a: float
    q_b: float
    deeper_side: str
    balance_threshold: float
    pairing: str
    note: str


def _validate_totals_vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{name} must be a nonempty 1D array of finite nonnegative totals")
    return arr


def marginal_depth_ratio(totals_a, totals_b, *, method: str = "medpos") -> tuple[float, str, float, float]:
    """Compute unsupervised marginal cell-depth ratio min/max across two runs.

    ``totals_a`` and ``totals_b`` MUST be per-cell transcript totals computed on
    **post-QC eligible cells over aligned classifier-input genes**.
    ``method='medpos'`` uses the positive-cell median (``median(x[x > 0])``) on
    both cohorts to remain robust to zero inflation, falling back to arithmetic
    means when either cohort has no positive entries. Returns
    ``(ratio, deeper_side, stat_a, stat_b)``.
    """
    if method not in ("medpos", "mean"):
        raise ValueError("method must be 'medpos' or 'mean'")
    a = _validate_totals_vector(totals_a, "totals_a")
    b = _validate_totals_vector(totals_b, "totals_b")
    if method == "medpos":
        pos_a = a[a > 0]
        pos_b = b[b > 0]
        if pos_a.size > 0 and pos_b.size > 0:
            stat_a = float(np.median(pos_a))
            stat_b = float(np.median(pos_b))
        else:
            stat_a = float(np.mean(a))
            stat_b = float(np.mean(b))
    else:
        stat_a = float(np.mean(a))
        stat_b = float(np.mean(b))
    hi = max(stat_a, stat_b)
    lo = min(stat_a, stat_b)
    ratio = 1.0 if hi == 0.0 else float(lo / hi)
    if stat_a > stat_b:
        deeper_side = "A"
    elif stat_b > stat_a:
        deeper_side = "B"
    else:
        deeper_side = "TIED"
    return ratio, deeper_side, stat_a, stat_b


def select_auto_q(
    totals_a,
    totals_b,
    *,
    balance_threshold: float = 1.0,
    sym_q: float = 0.75,
    pairing: str = "both",
    method: str = "medpos",
) -> AutoQSelection:
    """Unsupervised common-depth rarefaction selector requiring zero labels or splits.

    ``totals_a`` and ``totals_b`` MUST be per-cell transcript totals computed on
    **post-QC eligible cells over aligned classifier-input genes**.

    By default (``balance_threshold=1.0``), applies a smooth, threshold-free
    common-depth rarefaction formula without discontinuous routing jumps:
    ``q_k = sym_q * min(s_A, s_B) / s_k`` when ``pairing='both'`` (default
    ``sym_q=0.75``), or ``q_deep = ratio, q_shallow = 1.0`` when
    ``pairing='onesided'``. If a caller explicitly passes ``balance_threshold < 1.0``,
    preserves legacy thresholded routing (symmetric ``sym_q`` when
    ``ratio >= balance_threshold`` and matched thinning below).
    Preserves exact exchange symmetry: ``select_auto_q(a, b).q_a == select_auto_q(b, a).q_b``.
    """
    thr = float(balance_threshold)
    sq = float(sym_q)
    if not np.isfinite(thr) or not 0.0 < thr <= 1.0:
        raise ValueError("balance_threshold must be in (0, 1]")
    if not np.isfinite(sq) or not 0.0 < sq <= 1.0:
        raise ValueError("sym_q must be in (0, 1]")
    if pairing not in ("both", "onesided"):
        raise ValueError("pairing must be 'both' or 'onesided'")
    if method not in ("medpos", "mean"):
        raise ValueError("method must be 'medpos' or 'mean'")

    ratio_medpos, deeper_med, _, _ = marginal_depth_ratio(totals_a, totals_b, method="medpos")
    ratio_mean, deeper_mean, _, _ = marginal_depth_ratio(totals_a, totals_b, method="mean")
    ratio_sel, deeper_side = (ratio_medpos, deeper_med) if method == "medpos" else (ratio_mean, deeper_mean)

    if thr >= 1.0:
        if pairing == "both":
            mode = "COMMON_DEPTH_BOTH"
            if ratio_sel == 1.0:
                q_a, q_b = sq, sq
            else:
                q_a, q_b = (sq * ratio_sel, sq) if deeper_side == "A" else (sq, sq * ratio_sel)
        else:
            mode = "COMMON_DEPTH_ONESIDED"
            if deeper_side == "A":
                q_a, q_b = ratio_sel, 1.0
            elif deeper_side == "B":
                q_a, q_b = 1.0, ratio_sel
            else:
                q_a, q_b = 1.0, 1.0
    elif ratio_sel >= thr:
        if pairing == "both":
            mode = "BALANCED_SYM"
            q_a, q_b = sq, sq
        else:
            mode = "BALANCED_ONESIDED"
            if deeper_side == "A":
                q_a, q_b = ratio_sel, 1.0
            elif deeper_side == "B":
                q_a, q_b = 1.0, ratio_sel
            else:
                q_a, q_b = 1.0, 1.0
    else:
        if pairing == "both":
            mode = "AUTO_Q_BOTH"
            q_a, q_b = (sq * ratio_sel, sq) if deeper_side == "A" else (sq, sq * ratio_sel)
        else:
            mode = "AUTO_Q_ONESIDED"
            q_a, q_b = (ratio_sel, 1.0) if deeper_side == "A" else (1.0, ratio_sel)

    note = (
        f"Unsupervised {method} ratio={ratio_sel:.4f} (threshold={thr:.2f}); "
        f"selected {mode} (q_a={q_a:.4f}, q_b={q_b:.4f}, deeper={deeper_side})."
    )
    return AutoQSelection(
        mode=mode,
        ratio_medpos=ratio_medpos,
        ratio_mean=ratio_mean,
        q_a=float(q_a),
        q_b=float(q_b),
        deeper_side=deeper_side,
        balance_threshold=thr,
        pairing=pairing,
        note=note,
    )
