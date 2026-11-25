"""Optional prepared-graph implementation; no performance claim on import.

The analytic.py implementation provides the reference contract. Preparation validates and
owns the graph once; each call still validates all categorical probabilities.
Only exactly zero pair support is removed. Original N and M always define E,
including edges to OTHER nodes and isolated nodes that affect class counts.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from .analytic import _nonnegative, _exact_support_ratio
from .availability import availability_tails


def availability_tree(a, b):
    """Stable four-state reduction without subtracting near-one probabilities.

    Merge independent blocks (NONE, A_ONLY, B_ONLY, BOTH) in a balanced tree.
    All additions/multiplications are nonnegative. O(N) work and storage;
    no fitted thresholds, inclusion-exclusion, or probability clipping at zero.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0) or np.any(a+b > 1+1e-12):
        raise ValueError('aligned mutually exclusive probabilities required')
    active = (a > 0) | (b > 0)
    a, b = a[active], b[active]
    if len(a) < 2:
        return 0.
    state = np.column_stack([np.maximum(0., 1-a-b), a, b, np.zeros(len(a))])
    while len(state) > 1:
        size = len(state)//2
        left, right = state[:2*size:2], state[1:2*size:2]
        n, x, y, z = left.T
        nn, xx, yy, zz = right.T
        merged = np.column_stack([n*nn, x*(nn+xx)+n*xx,
                                  y*(nn+yy)+n*yy,
                                  z + zz*(n+x+y) + x*yy+y*xx])
        state = np.vstack([merged, state[-1:]]) if len(state)%2 else merged
    return min(1., max(0., float(state[0, 3])))


@dataclass(frozen=True)
class PreparedPairGraph:
    """Owned validated graph; use prepare_pair_graph, not direct construction."""
    graph: object
    row: object
    col: object
    n: int
    m: int


def prepare_pair_graph(adjacency):
    g = sparse.csr_matrix(adjacency, dtype=float, copy=True)
    g.sum_duplicates()
    g.eliminate_zeros()
    if g.shape[0] != g.shape[1] or np.any(g.data != 1) or np.any(g.diagonal() != 0) or (g != g.T).nnz:
        raise ValueError('fixed binary symmetric graph without self edges required')
    upper = sparse.triu(g, k=1, format='coo')
    for array in [g.data, g.indices, g.indptr, upper.row, upper.col]:
        array.flags.writeable = False
    return PreparedPairGraph(g, upper.row, upper.col, g.shape[0], g.nnz//2)


def pair_moments_prepared(probabilities, prepared, a=0, b=1):
    """Same output contract as pair_moments on an explicitly prepared graph.

    Graph validation/copying belongs to preparation, never an unreported cost.
    Pair support is compressed on every call, so changing p cannot reuse a stale
    support mask. No zeroing of small positive probabilities is permitted.
    """
    if not isinstance(prepared, PreparedPairGraph):
        raise ValueError('prepare_pair_graph result required')
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or a == b or min(a,b) < 0 or max(a,b) >= p.shape[1] or len(p) != prepared.n:
        raise ValueError('distinct valid class indices and aligned 2D probabilities required')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any(p.sum(axis=1) > 1+1e-12):
        raise ValueError('mutually exclusive categorical probabilities required')
    n, m = prepared.n, prepared.m
    g = prepared.graph
    # Sparse graph multiplication stays on the original graph; zero-probability
    # nodes contribute nothing. Compress the scalar recurrences and prefix sums,
    # avoiding a new sparse graph and its allocation for each query.
    pa_full, pb_full = p[:, a], p[:, b]
    active = (pa_full > 0) | (pb_full > 0)
    pa, pb = pa_full[active], pb_full[active]
    k = len(pa)
    oa, ob = (g @ pa_full)[active], (g @ pb_full)[active]
    da = np.cumsum(pa)-pa + np.cumsum(pa[::-1])[::-1]-pa
    db = np.cumsum(pb)-pb + np.cumsum(pb[::-1])[::-1]-pb
    c = 2*m/(n*(n-1)) if n > 1 else 0.
    mu_d, mu_o = float(pa @ db), float(pa @ ob)
    mu_e = c*mu_d
    va, vb, ab = pa*(1-pa), pb*(1-pb), pa*pb
    def projection(sa, sb, ta, tb):
        return float(np.sum(va*sb*tb + vb*sa*ta - ab*(sb*ta+sa*tb)))
    # Upper-edge indices are compiled once. Inactive endpoint variances are zero.
    i, j = prepared.row, prepared.col
    vaf, vbf, abf = pa_full*(1-pa_full), pb_full*(1-pb_full), pa_full*pb_full
    residual_o = float(np.sum(vaf[i]*vbf[j]+vbf[i]*vaf[j]+2*abf[i]*abf[j]))
    pva = np.r_[0., np.cumsum(va[:-1])] if k else np.empty(0)
    pvb = np.r_[0., np.cumsum(vb[:-1])] if k else np.empty(0)
    pab = np.r_[0., np.cumsum(ab[:-1])] if k else np.empty(0)
    residual_d = float(va @ pvb+vb @ pva+2*ab @ pab)
    var_o = _nonnegative(projection(oa,ob,oa,ob)+residual_o, mu_o**2)
    var_d = _nonnegative(projection(da,db,da,db)+residual_d, mu_d**2)
    cov_od = projection(oa,ob,da,db)+residual_o
    var_e, cov_oe = c*c*var_d, c*cov_od
    pplus = availability_tree(pa,pb) if m else 0.
    out = dict(n=n, m=m, expected_n_a=float(pa.sum()), expected_n_b=float(pb.sum()),
               mu_O=mu_o, mu_E=mu_e, var_O=var_o, var_E=var_e, cov_O_E=cov_oe,
               p_finite=pplus, mu_O_finite=None, mu_E_finite=None, var_O_finite=None,
               var_E_finite=None, cov_O_E_finite=None, ratio_of_means=None,
               mean_R_delta2=None, var_R_delta=None, cv_R_delta=None,
               cv_E_finite=None, cv_O_finite=None,
               status='NO_FINITE_RATIO', approximation_warning=None)
    out.update(availability_tails(pa_full, pb_full, has_edges=bool(m)))
    if pplus == 0 or mu_e == 0:
        possible = m > 0 and k >= 2 and np.any(pa>0) and np.any(pb>0)
        if possible: out['status'] = 'NUMERICAL_UNDERFLOW'
        return out
    if _exact_support_ratio(out, pa_full, pb_full, i, j, c, mu_d, var_d):
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
    vr = _nonnegative(vo/ee**2+eo**2*ve/ee**4-2*eo*co/ee**3,
                      vo/ee**2+eo**2*ve/ee**4+abs(2*eo*co/ee**3))
    out.update(mean_R_delta2=eo/ee+eo*ve/ee**3-co/ee**2,
               var_R_delta=vr, cv_R_delta=np.sqrt(vr)/(eo/ee),
               cv_O_finite=np.sqrt(vo)/eo, status='DELTA_APPROXIMATION')
    if pplus < .95 or out['cv_E_finite'] > .25:
        out['approximation_warning'] = 'SPARSE_OR_VARIABLE_DENOMINATOR_NO_ACCURACY_GUARANTEE'
    return out
