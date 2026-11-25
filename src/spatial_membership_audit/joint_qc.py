"""Joint retained-count QC and four-marker veto probabilities on a fixed graph.

This opt-in operation rechecks QC after independent molecule retention. It does
not alter the original fixed-QC classifier. Computation uses capped probability
generating functions in binary64, not a certified relative-error tail algorithm.
QC failures are OTHER nodes; callers must not remove them and silently change
the graph denominator. Supply every measured Gene Expression feature for QC.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import binom


def _gene_terms(n, p, cap):
    """P(Y=0), and (min(Y,cap),P) terms for positive Y; no tail subtraction."""
    if n == 0 or p == 0:
        return 1., []
    if p == 1:
        return 0., [(min(n, cap), 1.)]
    log_zero = n * math.log1p(-p)
    zero = math.exp(log_zero)
    if cap == 0:
        return zero, [(0, -math.expm1(log_zero))]
    terms = [(k, float(binom.pmf(k, n, p))) for k in range(1, min(n, cap - 1) + 1)]
    if n >= cap:
        terms.append((cap, float(binom.sf(cap - 1, n, p))))
    return zero, terms


def _shift_total(array, k):
    cap = array.shape[0] - 1
    result = np.zeros_like(array)
    if k >= cap:
        result[cap] = array.sum(axis=0)
    else:
        result[k:cap] = array[:cap-k]
        result[cap] = array[cap-k:].sum(axis=0)
    return result


def _advance_background(dp, zero, positive):
    out = zero * dp
    for k, mass in positive:
        moved = _shift_total(dp, k) * mass
        if dp.shape[1] == 1:
            out += moved
        else:
            out[:, 1:] += moved[:, :-1]
            out[:, -1] += moved[:, -1]
    return out


def _label(mask):
    a = (mask & 3) == 3
    b = (mask & 12) == 12
    return 0 if a and not b else 1 if b and not a else 2


def _probability_state(value, possible):
    if not possible:
        if value != 0:
            raise ArithmeticError('nonzero mass for a structurally impossible event')
        return 'STRUCTURAL_ZERO'
    if value == 0:
        return 'NUMERICAL_UNDERFLOW'
    return 'POSITIVE_SUBNORMAL' if value < np.finfo(float).tiny else 'POSITIVE_FINITE'


def four_gene_joint_qc(counts, retention, *, marker_indices=(0, 1, 2, 3),
                       min_total=50, min_detected=20):
    """One cell: joint A/QC, B/QC, and QC probabilities, with raw support states.

    A requires markers 0 AND 1, B markers 2 AND 3; simultaneous A/B is vetoed.
    Marker indices refer to the complete supplied panel. Gene names/panel
    completeness must be established by the caller; missing genes are not zero.
    P(retained total>=min_total, detected>=min_detected, label) is accumulated
    jointly. Scalar retention is broadcast; a gene vector is also accepted.
    """
    x = np.asarray(counts)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError('one aligned vector of nonnegative integer raw counts required')
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)) or np.any(x > 2**53):
        raise ValueError('finite nonnegative integer counts <=2**53 required')
    if len(marker_indices) != 4 or len(set(marker_indices)) != 4:
        raise ValueError('four distinct measured marker indices required')
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0 or k >= len(x) for k in marker_indices):
        raise ValueError('marker indices must point to four measured genes')
    for threshold in (min_total, min_detected):
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, np.integer)) or threshold < 0:
            raise ValueError('nonnegative integer QC thresholds required')
    if (min_total+1)*(min_detected+1) > 2_000_000:
        raise ValueError('QC state space exceeds the declared development implementation limit')
    p = np.asarray(retention, dtype=float)
    if p.ndim == 0:
        p = np.full(len(x), float(p))
    if p.shape != x.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('retention must be scalar or an aligned vector in [0,1]')
    x = [int(v) for v in x]
    markers = list(marker_indices)
    background = [i for i in range(len(x)) if i not in markers]
    cap_t, cap_d = int(min_total), int(min_detected)
    bg = np.zeros((cap_t + 1, cap_d + 1)); bg[0, 0] = 1.
    for i in background:
        zero, positive = _gene_terms(x[i], float(p[i]), cap_t)
        bg = _advance_background(bg, zero, positive)
    # The marker mask keeps the dependence between both gates and QC.
    md = np.zeros((16, cap_t + 1)); md[0, 0] = 1.
    for bit, i in enumerate(markers):
        zero, positive = _gene_terms(x[i], float(p[i]), cap_t)
        nxt = zero * md
        for mask in range(1 << bit):
            for k, mass in positive:
                nxt[mask | (1 << bit)] += mass * _shift_total(md[mask], k)
        md = nxt
    bg_mass, marker_mass = math.fsum(bg.ravel()), math.fsum(md.ravel())
    residual = max(abs(bg_mass-1.), abs(marker_mass-1.))
    if residual > 2e-12:
        raise ArithmeticError(f'probability mass residual {residual} exceeds 2e-12')
    tails = bg[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    class_joint = np.zeros(3)
    class_base = np.zeros(3)
    class_possible = [False]*3
    bg_max_t = sum(x[i] for i in background if p[i] > 0)
    bg_max_d = sum(x[i] > 0 and p[i] > 0 for i in background)
    for mask in range(16):
        label = _label(mask)
        marker_d = mask.bit_count()
        class_base[label] += math.fsum(md[mask])
        d_need = max(0, cap_d-marker_d)
        class_joint[label] += math.fsum(float(mass)*float(tails[max(0, cap_t-t), d_need])
                                       for t, mass in enumerate(md[mask]))
        supported = all((x[i] > 0 and p[i] > 0) if mask & (1 << bit)
                        else not (x[i] > 0 and p[i] == 1)
                        for bit, i in enumerate(markers))
        max_t = bg_max_t + sum(x[i] for bit, i in enumerate(markers) if mask & (1 << bit))
        if supported and max_t >= cap_t and bg_max_d+marker_d >= cap_d:
            class_possible[label] = True
    qc = math.fsum(class_joint)
    if min(class_joint) < 0 or qc > 1 + 2e-12:
        raise ArithmeticError('joint event probabilities left [0,1] beyond tolerance')
    return {
        'p_a_and_qc': float(class_joint[0]), 'p_b_and_qc': float(class_joint[1]),
        'p_other_and_qc': float(class_joint[2]), 'p_qc': qc,
        'p_a_without_qc': float(class_base[0]), 'p_b_without_qc': float(class_base[1]),
        'a_status': _probability_state(class_joint[0], class_possible[0]),
        'b_status': _probability_state(class_joint[1], class_possible[1]),
        'qc_status': _probability_state(qc, any(class_possible)),
        'mass_residual': residual, 'qc_thresholds': [cap_t, cap_d],
        'operation': 'joint_rechecked_QC_fixed_graph_OTHER_for_failures',
        'precision_scope': 'binary64_probability_DP; not certified relative tail accuracy',
    }
