"""Conditional thinning moments on a fixed binary undirected graph.

Exact first/second moments under independent categorical node labels; ratio
moments use a delta approximation, NOT a Poisson assumption. Labels must be
mutually exclusive within a node. Classifier posteriors are not automatically
thinning probabilities. No independence across cells is implied for refitted
or spatially smoothed classifiers.
"""
import numpy as np
from scipy import sparse
from scipy.stats import binom
from .definitions import REQUIRED, panel_contract
from .availability import availability_tails

CLASSES = ('B_CANDIDATE', 'CD8_T_CANDIDATE', 'CD4_T_CANDIDATE', 'MYELOID_CANDIDATE')


def marker_probabilities(counts, genes, eligible, p):
    """Exact probabilities for common125_candidate_v1, including new entrants.

    p may be one scalar or a measured-gene-aligned retention vector. Missing
    genes are rejected. A vector specifies a conditional deletion scenario;
    this function does not estimate or validate real measurement efficiencies.
    """
    x = np.asarray(counts)
    mask = np.asarray(eligible)
    genes = list(genes)
    if x.ndim != 2 or x.shape[1] != len(genes):
        raise ValueError('counts/genes shape mismatch')
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError('finite nonnegative integer counts required')
    if mask.dtype != bool or mask.shape != (len(x),):
        raise ValueError('fixed baseline boolean mask required')
    contract = panel_contract(genes)
    if contract['missing_genes']:
        raise ValueError('NOT_IDENTIFIABLE: missing measured genes')
    retention = np.asarray(p, dtype=float)
    if retention.ndim == 0:
        retention = np.full(len(genes), float(retention))
    if retention.shape != (len(genes),) or not np.all(np.isfinite(retention)) or np.any((retention < 0) | (retention > 1)):
        raise ValueError('p must be in [0,1], scalar or aligned with measured genes')
    ix = [genes.index(g) for g in REQUIRED]
    x, retention = x[:, ix], retention[ix]
    z = binom.pmf(0, x, retention)
    one = binom.pmf(1, x, retention)
    pos = binom.sf(0, x, retention)
    two = binom.sf(1, x, retention)
    b = pos[:, 0] * (pos[:, 1] + z[:, 1] * pos[:, 2])
    nb = z[:, 0] + pos[:, 0] * z[:, 1] * z[:, 2]
    t = two[:, 3] * pos[:, 4] + one[:, 3] * two[:, 4]
    nt = z[:, 3] + pos[:, 3] * z[:, 4] + one[:, 3] * one[:, 4]
    m = pos[:, 7] * pos[:, 8]
    nm = z[:, 7] + pos[:, 7] * z[:, 8]
    npl = z[:, 9] + pos[:, 9] * z[:, 10]
    u, nu, v, nv = two[:, 5], z[:, 5] + one[:, 5], pos[:, 6], z[:, 6]
    out = np.column_stack([b * nt * nm * npl, t * nb * nm * npl * u * nv,
                           t * nb * nm * npl * nu * v, m * nb * nt * npl])
    return out * mask[:, None]


def _availability(a, b):
    # Four-state dynamic program: no target, only A, only B, both.
    # Unlike inclusion-exclusion, stable even when P(both) is extremely small.
    none, only_a, only_b, both = 1., 0., 0., 0.
    for ai, bi in zip(a, b):
        both += only_a * bi + only_b * ai
        only_a, only_b, none = (only_a * (1-bi) + none * ai,
                               only_b * (1-ai) + none * bi,
                               none * max(0., 1-ai-bi))
    return min(1., max(0., both))


def _nonnegative(value, scale):
    if value < -1e-10 * max(1., scale):
        raise ArithmeticError('negative variance exceeds roundoff tolerance')
    return max(0., float(value))


def _exact_support_ratio(out, pa, pb, i, j, c, mu_d, var_d):
    """Resolve structural constants without treating small positives as zero.

    If all possible cross-class node pairs are edges, O=D=n_A*n_B
    pointwise, hence R=1/c on E>0. With only two active nodes D=1
    conditional on E>0, even when their connecting edge is absent.
    Integer support counts avoid a tolerance-based numerical classification.
    """
    a, b = pa > 0, pb > 0
    possible = int(a.sum()) * int(b.sum()) - int(np.count_nonzero(a & b))
    present = int(np.count_nonzero(a[i] & b[j])) + int(np.count_nonzero(b[i] & a[j]))
    two = np.count_nonzero(a | b) == 2
    if not two and present != possible:
        return False
    probability = out['p_finite']
    ed = 1. if two else mu_d / probability
    vd = 0. if two else _nonnegative(
        (var_d + mu_d**2) / probability - ed**2,
        (var_d + mu_d**2) / probability)
    connected = present == possible
    eo, ee = (ed if connected else 0.), c * ed
    cvd = np.sqrt(vd) / ed
    out.update(mu_O_finite=eo, mu_E_finite=ee,
               var_O_finite=vd if connected else 0., var_E_finite=c*c*vd,
               cov_O_E_finite=c*vd if connected else 0.,
               ratio_of_means=1./c if connected else 0.,
               mean_R_delta2=1./c if connected else 0., var_R_delta=0.,
               cv_R_delta=0. if connected else None,
               cv_E_finite=cvd, cv_O_finite=cvd if connected else None,
               status='DELTA_APPROXIMATION' if connected else 'ZERO_NUMERATOR_CV_UNDEFINED')
    # Retain the established status/warning contract, even for an exact subcase.
    if connected and (probability < .95 or cvd > .25):
        out['approximation_warning'] = 'SPARSE_OR_VARIABLE_DENOMINATOR_NO_ACCURACY_GUARANTEE'
    return True


def pair_moments(probabilities, adjacency, a=0, b=1):
    """Return exact O/E moments, availability and conditional delta CV.

    O counts cross-class undirected edges. E=2M*n_A*n_B/[N(N-1)].
    O=E=0 whenever E is undefined as a ratio denominator. Conditional first
    and raw second moments therefore divide by P(E>0) before centering.
    Complexity O(N+M); no dense complete graph is constructed.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or a == b or min(a,b) < 0 or max(a,b) >= p.shape[1]:
        raise ValueError('distinct valid class indices and 2D probabilities required')
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or np.any(p.sum(axis=1) > 1+1e-12):
        raise ValueError('mutually exclusive categorical probabilities required')
    g = sparse.csr_matrix(adjacency, dtype=float)
    g.sum_duplicates(); g.eliminate_zeros()
    n = len(p)
    if g.shape != (n,n) or np.any(g.data != 1) or np.any(g.diagonal() != 0) or (g != g.T).nnz:
        raise ValueError('fixed binary symmetric graph without self edges required')
    m = g.nnz // 2
    pa, pb = p[:, a], p[:, b]
    c = 2*m/(n*(n-1)) if n > 1 else 0.
    # Stable sums over all pairs exclude each same-node categorical conflict.
    da = np.cumsum(pa) - pa + np.cumsum(pa[::-1])[::-1] - pa
    db = np.cumsum(pb) - pb + np.cumsum(pb[::-1])[::-1] - pb
    oa, ob = g @ pa, g @ pb
    mu_d, mu_o = float(pa @ db), float(pa @ ob)
    mu_e = c * mu_d
    va, vb, ab = pa*(1-pa), pb*(1-pb), pa*pb
    # First-order Hoeffding projections; centered categorical covariance.
    def projection(sa, sb, ta, tb):
        return float(np.sum(va*sb*tb + vb*sa*ta - ab*(sb*ta + sa*tb)))
    edges = sparse.triu(g, k=1, format='coo')
    i, j = edges.row, edges.col
    residual_o = float(np.sum(va[i]*vb[j] + vb[i]*va[j] + 2*ab[i]*ab[j]))
    # Sum complete-graph residual terms by prefix sums, avoiding cancellation.
    pva = np.r_[0., np.cumsum(va[:-1])] if n else np.empty(0)
    pvb = np.r_[0., np.cumsum(vb[:-1])] if n else np.empty(0)
    pab = np.r_[0., np.cumsum(ab[:-1])] if n else np.empty(0)
    residual_d = float(va @ pvb + vb @ pva + 2*ab @ pab)
    var_o = _nonnegative(projection(oa,ob,oa,ob)+residual_o, mu_o**2)
    var_d = _nonnegative(projection(da,db,da,db)+residual_d, mu_d**2)
    cov_od = projection(oa,ob,da,db)+residual_o
    var_e, cov_oe = c*c*var_d, c*cov_od
    pplus = _availability(pa,pb) if m else 0.
    out = dict(n=n, m=m, expected_n_a=float(pa.sum()), expected_n_b=float(pb.sum()),
               mu_O=mu_o, mu_E=mu_e, var_O=var_o, var_E=var_e, cov_O_E=cov_oe,
               p_finite=pplus, mu_O_finite=None, mu_E_finite=None, var_O_finite=None,
               var_E_finite=None, cov_O_E_finite=None, ratio_of_means=None,
               mean_R_delta2=None, var_R_delta=None, cv_R_delta=None,
               cv_E_finite=None, cv_O_finite=None,
               status='NO_FINITE_RATIO', approximation_warning=None)
    out.update(availability_tails(pa, pb, has_edges=bool(m)))
    if pplus == 0 or mu_e == 0:
        # Preserve inability to distinguish a numerically underflowed support.
        possible = m > 0 and np.count_nonzero((pa>0)|(pb>0)) >= 2 and np.any(pa>0) and np.any(pb>0)
        if possible: out['status'] = 'NUMERICAL_UNDERFLOW'
        return out
    if _exact_support_ratio(out, pa, pb, i, j, c, mu_d, var_d):
        return out
    eo, ee = mu_o/pplus, mu_e/pplus
    vo = _nonnegative((var_o+mu_o**2)/pplus-eo**2, (var_o+mu_o**2)/pplus)
    ve = _nonnegative((var_e+mu_e**2)/pplus-ee**2, (var_e+mu_e**2)/pplus)
    co = (cov_oe+mu_o*mu_e)/pplus-eo*ee
    out.update(mu_O_finite=eo, mu_E_finite=ee, var_O_finite=vo,
               var_E_finite=ve, cov_O_E_finite=co, cv_E_finite=np.sqrt(ve)/ee,
               ratio_of_means=eo/ee, status='ZERO_NUMERATOR_CV_UNDEFINED')
    if eo == 0:
        out.update(mean_R_delta2=0., var_R_delta=0.)
        return out
    vr = _nonnegative(vo/ee**2 + eo**2*ve/ee**4 - 2*eo*co/ee**3,
                      vo/ee**2 + eo**2*ve/ee**4 + abs(2*eo*co/ee**3))
    out.update(mean_R_delta2=eo/ee + eo*ve/ee**3-co/ee**2,
               var_R_delta=vr, cv_R_delta=np.sqrt(vr)/(eo/ee),
               cv_O_finite=np.sqrt(vo)/eo, status='DELTA_APPROXIMATION')
    if pplus < .95 or out['cv_E_finite'] > .25:
        out['approximation_warning'] = 'SPARSE_OR_VARIABLE_DENOMINATOR_NO_ACCURACY_GUARANTEE'
    return out
