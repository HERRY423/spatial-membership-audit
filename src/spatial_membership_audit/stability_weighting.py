"""Stability-weighted spatial colocalization with exact closed-form permutation moments.

Repurposes conditional thinning draws at native depth as per-cell reliability
weights w_i = (1 - f_i)^p applied to the unthinned baseline cell-type labels,
improving single-run fidelity to biological ground truth without sacrificing
transcript depth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse

from .permutation_bridge import fixed_size_edge_moments


@dataclass(frozen=True)
class StabilityWeightedResult:
    r_weighted: float
    z_weighted: float
    observed_weighted: float
    expected_weighted: float
    var_weighted: float
    sum_wa: float
    sum_wb: float
    effective_n_a: float
    effective_n_b: float
    mean_weight_a: float
    mean_weight_b: float
    note: str


def cell_stability_weights(
    base_labels,
    draw_labels,
    *,
    power: float = 1.0,
) -> np.ndarray:
    """Compute per-cell label stability weights (1 - flip_rate) ** power.

    Parameters
    ----------
    base_labels:
        1D sequence of unthinned baseline cell-type labels of length N.
    draw_labels:
        2D sequence of shape (B, N) containing B replicate thinning draws.
    power:
        Nonnegative exponent applied to the stability probability (default 1.0).
    """
    pwr = float(power)
    if not math.isfinite(pwr) or pwr < 0.0:
        raise ValueError("power must be finite and >= 0")
    base = np.asarray(base_labels)
    draws = np.asarray(draw_labels)
    if base.ndim != 1 or base.shape[0] < 1:
        raise ValueError("base_labels must be a non-empty 1D array")
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] != base.shape[0]:
        raise ValueError(f"draw_labels must have shape (B, {base.shape[0]}) with B >= 1")
    stability = (draws == base[None, :]).mean(axis=0, dtype=np.float64)
    return np.asarray(stability ** pwr, dtype=np.float64)


def exact_weighted_perm_pair(adjacency, wa, wb) -> StabilityWeightedResult:
    """Exact permutation mean, variance, R = O/E, and z for disjoint weights wa, wb.

    Both wa and wb must be finite, nonnegative, 1D arrays aligned to ``adjacency``
    with disjoint positive supports (``wa * wb == 0`` everywhere).
    """
    moments_meta = fixed_size_edge_moments(adjacency, 0, 0)
    n = int(moments_meta["n"])
    m = int(moments_meta["m"])
    n_adj = int(moments_meta["shared_edge_pairs"])
    n_disj = int(moments_meta["disjoint_edge_pairs"])

    wa_arr = np.asarray(wa, dtype=np.float64)
    wb_arr = np.asarray(wb, dtype=np.float64)
    if wa_arr.ndim != 1 or wb_arr.ndim != 1 or wa_arr.shape[0] != n or wb_arr.shape[0] != n:
        raise ValueError(f"wa and wb must be 1D arrays of length {n}")
    if not np.isfinite(wa_arr).all() or not np.isfinite(wb_arr).all():
        raise ValueError("wa and wb must be finite")
    if (wa_arr < 0.0).any() or (wb_arr < 0.0).any():
        raise ValueError("wa and wb must be nonnegative")
    if ((wa_arr > 0.0) & (wb_arr > 0.0)).any():
        raise ValueError("wa and wb must have disjoint positive supports (wa * wb == 0)")

    sa = float(wa_arr.sum())
    sb = float(wb_arr.sum())
    sa2 = float(np.dot(wa_arr, wa_arr))
    sb2 = float(np.dot(wb_arr, wb_arr))
    eff_na = (sa * sa / sa2) if sa2 > 0.0 else 0.0
    eff_nb = (sb * sb / sb2) if sb2 > 0.0 else 0.0
    supp_a = int((wa_arr > 0.0).sum())
    supp_b = int((wb_arr > 0.0).sum())
    mean_wa = (sa / supp_a) if supp_a > 0 else math.nan
    mean_wb = (sb / supp_b) if supp_b > 0 else math.nan

    g_up = sparse.triu(sparse.csr_matrix(adjacency), k=1).tocoo()
    eu = g_up.row
    ev = g_up.col
    obs = float((wa_arr[eu] * wb_arr[ev] + wa_arr[ev] * wb_arr[eu]).sum()) if m > 0 else 0.0

    note = (
        "Exact weighted fixed-graph permutation moments for disjoint cell reliability weights; "
        "targets native-depth biological accuracy within a run rather than cross-depth equalization."
    )
    if n < 2 or m == 0 or sa <= 0.0 or sb <= 0.0:
        return StabilityWeightedResult(
            r_weighted=math.nan,
            z_weighted=math.nan,
            observed_weighted=obs,
            expected_weighted=0.0,
            var_weighted=0.0,
            sum_wa=sa,
            sum_wb=sb,
            effective_n_a=eff_na,
            effective_n_b=eff_nb,
            mean_weight_a=mean_wa,
            mean_weight_b=mean_wb,
            note=note,
        )

    c_const = (2.0 * m) / (n * (n - 1.0))
    exp_val = c_const * sa * sb
    t_self = c_const * sa2 * sb2
    denom3 = n * (n - 1.0) * (n - 2.0) if n >= 3 else 0.0
    denom4 = denom3 * (n - 3.0) if n >= 4 else 0.0
    p_adj = (
        (sa2 * max(0.0, sb * sb - sb2) + sb2 * max(0.0, sa * sa - sa2)) / denom3
        if n >= 3
        else 0.0
    )
    p_disj = (
        4.0 * max(0.0, sa * sa - sa2) * max(0.0, sb * sb - sb2) / denom4
        if n >= 4
        else 0.0
    )
    second_moment = t_self + 2.0 * n_adj * p_adj + 2.0 * n_disj * p_disj
    var_val = second_moment - exp_val * exp_val
    if var_val <= 1e-12 * max(second_moment, exp_val * exp_val):
        var_val = 0.0

    r_val = (obs / exp_val) if exp_val > 0.0 else math.nan
    z_val = ((obs - exp_val) / math.sqrt(var_val)) if var_val > 0.0 else math.nan

    return StabilityWeightedResult(
        r_weighted=r_val,
        z_weighted=z_val,
        observed_weighted=obs,
        expected_weighted=exp_val,
        var_weighted=var_val,
        sum_wa=sa,
        sum_wb=sb,
        effective_n_a=eff_na,
        effective_n_b=eff_nb,
        mean_weight_a=mean_wa,
        mean_weight_b=mean_wb,
        note=note,
    )


def stability_weighted_colocalization(
    adjacency,
    base_labels,
    draw_labels,
    *,
    a=0,
    b=1,
    power: float = 1.0,
) -> StabilityWeightedResult:
    """Evaluate stability-weighted spatial colocalization for baseline classes ``a`` and ``b``."""
    if a == b:
        raise ValueError("distinct class codes a != b required")
    base = np.asarray(base_labels)
    weights = cell_stability_weights(base, draw_labels, power=power)
    wa = np.where(base == a, weights, 0.0)
    wb = np.where(base == b, weights, 0.0)
    return exact_weighted_perm_pair(adjacency, wa, wb)
