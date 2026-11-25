"""Conditional label instability and Poisson split-half coverage matching.

Cross-panel and cross-platform runs can suffer zero inflation where empty
(UNRESOLVED) cells cannot flip across thinned draws, biasing unconditional flip
rates downward. All coverage matching in this module conditions strictly on
classifiable cells while reporting ``unresolved_frac`` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


_VALID_PAIRINGS = ("both", "onesided")


@dataclass(frozen=True)
class ConditionalInstability:
    n_cells: int
    n_draws: int
    n_classifiable: int
    unresolved_frac: float
    cond_flip_rate: float
    cond_pair_disagreement: float
    uncond_flip_rate: float
    uncond_pair_disagreement: float
    effective_stability: float
    per_cell_flip: np.ndarray
    per_cell_pair_dis: np.ndarray
    note: str


@dataclass(frozen=True)
class SplitHalfCurve:
    q_vals: np.ndarray
    psi_vals: np.ndarray
    raw_psi_vals: np.ndarray
    unresolved_frac: float
    monotone_enforced: bool


@dataclass(frozen=True)
class CoverageMatchResult:
    q_a: float
    q_b: float
    target_psi: float
    psi_a_at_ref: float
    psi_b_at_ref: float
    effective_ratio: float
    noisier_side: str
    pairing: str
    q0: float
    unresolved_frac_a: float
    unresolved_frac_b: float
    clamped: bool
    note: str


def _readonly(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _resolve_classifiable_mask(n_cells: int, base_labels=None, *, classifiable_mask=None, unresolved_label=None) -> np.ndarray:
    if classifiable_mask is not None:
        mask = np.asarray(classifiable_mask, dtype=bool)
        if mask.shape != (n_cells,):
            raise ValueError(f"classifiable_mask must have shape ({n_cells},)")
    else:
        mask = np.ones(n_cells, dtype=bool)
    if base_labels is not None and unresolved_label is not None:
        mask = mask & (np.asarray(base_labels) != unresolved_label)
    return mask


def conditional_instability_metrics(
    base_labels,
    draw_labels,
    *,
    classifiable_mask=None,
    unresolved_label=None,
) -> ConditionalInstability:
    """Measure label flip rate and unbiased pairwise disagreement across draws.

    Conditioning on ``classifiable_mask`` and excluding ``unresolved_label``
    prevents zero-count cells (which trivially remain UNRESOLVED) from
    diluting cross-run instability estimates.
    """
    base = np.asarray(base_labels)
    draws = np.asarray(draw_labels)
    if base.ndim != 1 or base.shape[0] < 1:
        raise ValueError("base_labels must be a nonempty 1D array")
    n_cells = base.shape[0]
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] != n_cells:
        raise ValueError(f"draw_labels must have shape (B, {n_cells}) with B >= 1")
    b_draws = draws.shape[0]

    mask = _resolve_classifiable_mask(
        n_cells, base, classifiable_mask=classifiable_mask, unresolved_label=unresolved_label
    )
    n_classifiable = int(mask.sum())
    unresolved_frac = float(1.0 - n_classifiable / n_cells)

    stacked = np.vstack([base[None, :], draws])
    _, inv = np.unique(stacked, return_inverse=True)
    inv = inv.reshape(b_draws + 1, n_cells)
    base_codes = inv[0]
    draw_codes = inv[1:]
    n_codes = int(inv.max()) + 1

    per_cell_flip = (draw_codes != base_codes[None, :]).mean(axis=0, dtype=np.float64)
    if b_draws >= 2:
        cmat = np.zeros((n_codes, n_cells), dtype=np.int64)
        for code in range(n_codes):
            cmat[code] = (draw_codes == code).sum(axis=0)
        same_pairs = np.sum(cmat * (cmat - 1), axis=0, dtype=np.float64)
        per_cell_pair_dis = 1.0 - same_pairs / float(b_draws * (b_draws - 1))
    else:
        per_cell_pair_dis = np.full(n_cells, np.nan, dtype=np.float64)

    uncond_flip = float(per_cell_flip.mean())
    uncond_pair = float(per_cell_pair_dis.mean()) if b_draws >= 2 else math.nan
    if n_classifiable > 0:
        cond_flip = float(per_cell_flip[mask].mean())
        cond_pair = float(per_cell_pair_dis[mask].mean()) if b_draws >= 2 else math.nan
        eff_stab = 1.0 - cond_flip
    else:
        cond_flip = math.nan
        cond_pair = math.nan
        eff_stab = math.nan

    return ConditionalInstability(
        n_cells=n_cells,
        n_draws=b_draws,
        n_classifiable=n_classifiable,
        unresolved_frac=unresolved_frac,
        cond_flip_rate=cond_flip,
        cond_pair_disagreement=cond_pair,
        uncond_flip_rate=uncond_flip,
        uncond_pair_disagreement=uncond_pair,
        effective_stability=eff_stab,
        per_cell_flip=_readonly(per_cell_flip),
        per_cell_pair_dis=_readonly(per_cell_pair_dis),
        note=(
            "Conditional instability excludes unclassifiable/zero cells from the denominator "
            "and reports unresolved_frac separately."
        ),
    )


def split_half_disagreement(
    half_a_labels,
    half_b_labels,
    *,
    classifiable_mask=None,
    unresolved_label=None,
) -> tuple[float, float, int]:
    """Compute conditional disagreement ``P(hat{Z}(Y_1) != hat{Z}(Y_2) | classifiable)``.

    Accepts ``(N,)`` or ``(R, N)`` label arrays for independent paired Poisson
    split-half replicates at total retention ``q``. Pass ``classifiable_mask``
    for the baseline positive-count cells (``total > 0`` at native depth). If
    only ``unresolved_label`` is supplied, cells that are ``unresolved_label`` in
    every split-half observation are excluded from the conditional denominator.
    Returns ``(cond_psi, unresolved_frac, n_classifiable)``.
    """
    a = np.asarray(half_a_labels)
    b = np.asarray(half_b_labels)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] < 1 or a.shape[0] < 1:
        raise ValueError("half_a_labels and half_b_labels must share shape (N,) or (R, N) with N >= 1, R >= 1")
    n_cells = a.shape[1]

    if classifiable_mask is not None:
        mask = np.asarray(classifiable_mask, dtype=bool)
        if mask.shape != (n_cells,):
            raise ValueError(f"classifiable_mask must have shape ({n_cells},)")
    elif unresolved_label is not None:
        mask = np.any((a != unresolved_label) | (b != unresolved_label), axis=0)
    else:
        mask = np.ones(n_cells, dtype=bool)

    n_classifiable = int(mask.sum())
    unresolved_frac = float(1.0 - n_classifiable / n_cells)
    if n_classifiable == 0:
        return math.nan, unresolved_frac, 0

    disagree = a[:, mask] != b[:, mask]
    return float(disagree.mean(dtype=np.float64)), unresolved_frac, n_classifiable


def _pava_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm enforcing monotone non-increasing sequence."""
    blocks: list[list[float]] = []
    for v in values:
        blocks.append([float(v), 1.0])
        while len(blocks) >= 2:
            prev_mean = blocks[-2][0] / blocks[-2][1]
            curr_mean = blocks[-1][0] / blocks[-1][1]
            if prev_mean < curr_mean:
                blocks[-2][0] += blocks[-1][0]
                blocks[-2][1] += blocks[-1][1]
                blocks.pop()
            else:
                break
    out = np.empty(len(values), dtype=np.float64)
    idx = 0
    for total, count in blocks:
        cnt = int(count)
        out[idx:idx + cnt] = total / count
        idx += cnt
    return out


def build_split_half_curve(
    q_vals,
    psi_vals,
    *,
    unresolved_frac: float = 0.0,
    enforce_monotone: bool = True,
) -> SplitHalfCurve:
    """Construct a validated conditional split-half instability curve ``psi(q)``."""
    q = np.asarray(q_vals, dtype=np.float64)
    psi = np.asarray(psi_vals, dtype=np.float64)
    if q.ndim != 1 or psi.ndim != 1 or q.shape != psi.shape or q.size < 2:
        raise ValueError("q_vals and psi_vals must be aligned 1D arrays of length >= 2")
    if not np.isfinite(q).all() or np.any(q <= 0.0) or np.any(q > 1.0):
        raise ValueError("q_vals must be finite and lie in (0, 1]")
    if len(np.unique(q)) != len(q):
        raise ValueError("q_vals must be strictly distinct")
    if not np.isfinite(psi).all() or np.any(psi < 0.0) or np.any(psi > 1.0):
        raise ValueError("psi_vals must be finite probabilities in [0, 1]")
    ufrac = float(unresolved_frac)
    if not math.isfinite(ufrac) or not 0.0 <= ufrac <= 1.0:
        raise ValueError("unresolved_frac must lie in [0, 1]")

    order = np.argsort(q)
    q_sorted = q[order]
    psi_raw = psi[order]
    psi_mono = np.clip(_pava_nonincreasing(psi_raw), 0.0, 1.0) if enforce_monotone else psi_raw.copy()

    return SplitHalfCurve(
        q_vals=_readonly(q_sorted),
        psi_vals=_readonly(psi_mono),
        raw_psi_vals=_readonly(psi_raw),
        unresolved_frac=ufrac,
        monotone_enforced=bool(enforce_monotone),
    )


def _eval_curve(curve: SplitHalfCurve, q: float) -> float:
    return float(np.interp(math.log(q), np.log(curve.q_vals), curve.psi_vals))


def _solve_q_for_psi(curve: SplitHalfCurve, target_psi: float, q_upper: float, min_q: float) -> tuple[float, bool]:
    q_lo = max(min_q, float(curve.q_vals[0]))
    q_hi = min(q_upper, float(curve.q_vals[-1]), 1.0)
    psi_at_lo = _eval_curve(curve, q_lo)
    psi_at_hi = _eval_curve(curve, q_hi)

    if target_psi > psi_at_lo + 1e-14:
        return float(q_lo), True
    if target_psi < psi_at_hi - 1e-14:
        return float(q_hi), True

    # Restrict monotone grid to [q_lo, q_hi] and interpolate in (log q, psi)
    qs = curve.q_vals
    ps = curve.psi_vals
    inside = (qs > q_lo) & (qs < q_hi)
    q_pts = np.concatenate([[q_lo], qs[inside], [q_hi]])
    p_pts = np.concatenate([[psi_at_lo], ps[inside], [psi_at_hi]])

    for i in range(len(q_pts) - 1):
        hi_p, lo_p = p_pts[i], p_pts[i + 1]
        if lo_p <= target_psi <= hi_p:
            if abs(hi_p - lo_p) <= 1e-14:
                log_q = 0.5 * (math.log(float(q_pts[i])) + math.log(float(q_pts[i + 1])))
            else:
                frac = (hi_p - target_psi) / (hi_p - lo_p)
                log_q = math.log(float(q_pts[i])) + frac * (math.log(float(q_pts[i + 1])) - math.log(float(q_pts[i])))
            q_sol = math.exp(log_q)
            clamped = q_sol < min_q
            return float(min(q_hi, max(q_lo, q_sol))), clamped
    return float(q_lo), True


def match_split_half_coverage(
    curve_a: SplitHalfCurve,
    curve_b: SplitHalfCurve,
    *,
    q0: float = 0.75,
    pairing: str = "both",
    min_q: float = 0.05,
) -> CoverageMatchResult:
    """Match conditional split-half classifier instability across two runs.

    Evaluates conditional split-half disagreement ``psi_k(q_ref)`` at
    ``q_ref = 1.0`` (``pairing='onesided'``) or ``q_ref = q0`` (``pairing='both'``),
    retains ``q_ref`` on the noisier run, and solves ``psi_clean(q*) = psi_noisy(q_ref)``
    on the cleaner run. Preserves exact exchange symmetry between A and B.
    """
    if not isinstance(curve_a, SplitHalfCurve) or not isinstance(curve_b, SplitHalfCurve):
        raise ValueError("curve_a and curve_b must be SplitHalfCurve instances")
    if pairing not in _VALID_PAIRINGS:
        raise ValueError(f"pairing must be one of {_VALID_PAIRINGS}")
    q0_f = float(q0)
    min_q_f = float(min_q)
    if not math.isfinite(q0_f) or not 0.0 < q0_f <= 1.0:
        raise ValueError("q0 must be in (0, 1]")
    if not math.isfinite(min_q_f) or not 0.0 < min_q_f <= 1.0:
        raise ValueError("min_q must be in (0, 1]")

    q_ref = 1.0 if pairing == "onesided" else q0_f
    psi_a = _eval_curve(curve_a, q_ref)
    psi_b = _eval_curve(curve_b, q_ref)

    if psi_a > psi_b + 1e-14:
        noisier = "A"
        target_psi = psi_a
        q_a = q_ref
        q_b, clamped = _solve_q_for_psi(curve_b, target_psi, q_upper=q_ref, min_q=min_q_f)
    elif psi_b > psi_a + 1e-14:
        noisier = "B"
        target_psi = psi_b
        q_b = q_ref
        q_a, clamped = _solve_q_for_psi(curve_a, target_psi, q_upper=q_ref, min_q=min_q_f)
    else:
        noisier = "TIED"
        target_psi = 0.5 * (psi_a + psi_b)
        q_a = q_ref
        q_b = q_ref
        clamped = False

    eff_ratio = min(q_a, q_b) / max(q_a, q_b)
    note = (
        f"Conditional split-half coverage matching ({pairing}, q_ref={q_ref:.4f}): "
        f"psi_A={psi_a:.4f}, psi_B={psi_b:.4f} -> noisier={noisier}, "
        f"q_a={q_a:.4f}, q_b={q_b:.4f}, clamped={clamped}."
    )
    return CoverageMatchResult(
        q_a=float(q_a),
        q_b=float(q_b),
        target_psi=float(target_psi),
        psi_a_at_ref=float(psi_a),
        psi_b_at_ref=float(psi_b),
        effective_ratio=float(eff_ratio),
        noisier_side=noisier,
        pairing=pairing,
        q0=q0_f,
        unresolved_frac_a=curve_a.unresolved_frac,
        unresolved_frac_b=curve_b.unresolved_frac,
        clamped=bool(clamped),
        note=note,
    )
