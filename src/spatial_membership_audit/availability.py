"""Direct disjoint availability tails, retaining log probabilities and zero causes.

Generic probability input refers to the supplied binary64 distribution only.
The half-retention marker path keeps rare failures from integer counts in log
space, including failures hidden by a rounded class probability of one.
"""
import itertools
import math
import numpy as np
from .definitions import REQUIRED, panel_contract


def _record(logp):
    if np.isneginf(logp):
        return 0., 'STRUCTURAL_ZERO'
    value = math.exp(min(0., float(logp)))
    if value == 0.:
        return 0., 'NUMERICAL_UNDERFLOW'
    return value, 'POSITIVE_SUBNORMAL' if value < np.finfo(float).tiny else 'POSITIVE_FINITE'


def availability_from_log_states(log_states, *, has_edges=True, basis='supplied_log_states'):
    """States per cell are (neither, A, B); compute NA without 1-P(finite).

    Structural zero means impossible in the explicitly supplied distribution.
    A numerical zero with a finite log probability is underflow, not absence.
    No threshold truncation or subtraction of near-one aggregate tails occurs.
    """
    s = np.asarray(log_states, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3 or np.isnan(s).any() or np.isposinf(s).any() or np.any(s > 1e-12):
        raise ValueError('N by 3 log categorical probabilities required')
    if len(s) and np.any(np.abs(np.logaddexp.reduce(s, axis=1)) > 2e-12):
        raise ValueError('log categorical probabilities must sum to one')
    if not has_edges:
        lf, ln = -np.inf, 0.
    else:
        # Four disjoint states; all arithmetic uses nonnegative probability sums.
        n, a, b, both = 0., -np.inf, -np.inf, -np.inf
        for other, ai, bi in s:
            new_both = np.logaddexp(both, np.logaddexp(a+bi, b+ai))
            a, b, n = (np.logaddexp(a+np.logaddexp(other, ai), n+ai),
                       np.logaddexp(b+np.logaddexp(other, bi), n+bi), n+other)
            both = new_both
        lf, ln = min(0., float(both)), min(0., float(np.logaddexp.reduce([n,a,b])))
    finite, fs = _record(lf)
    na, ns = _record(ln)
    # JSON null represents log(0); the status retains its structural meaning.
    return dict(p_finite_direct=finite, p_na=na,
                log_p_finite=None if np.isneginf(lf) else float(lf),
                log_p_na=None if np.isneginf(ln) else float(ln),
                finite_probability_status=fs, na_probability_status=ns, probability_basis=basis)


def availability_tails(a, b, *, has_edges=True):
    """Direct tails for *supplied* categorical probabilities, not raw counts."""
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0) or np.any(b<0) or np.any(a+b>1):
        raise ValueError('aligned mutually exclusive probabilities required')
    with np.errstate(divide='ignore'):
        s = np.column_stack([np.log1p(-(a+b)), np.log(a), np.log(b)])
    return availability_from_log_states(s, has_edges=has_edges, basis='supplied_binary64_categorical_values')


def marker_log_categories_half(counts, genes, eligible):
    """Log probabilities for the frozen complete 11-marker rule at p=0.5 only.

    Returns B, CD8, CD4, myeloid, OTHER. Sixty-four independent gate states
    form a disjoint partition; OTHER and rare gate failures are never 1-class.
    No physical efficiency or identity interpretation is implied.
    """
    x, mask, genes = np.asarray(counts), np.asarray(eligible), list(genes)
    if x.ndim!=2 or x.shape[1]!=len(genes) or not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('finite nonnegative integer counts required')
    if mask.dtype!=bool or mask.shape!=(len(x),):
        raise ValueError('fixed baseline boolean mask required')
    if len(genes)!=len(set(genes)) or panel_contract(genes)['missing_genes']:
        raise ValueError('all required markers must be measured uniquely')
    x=x[:,[genes.index(g) for g in REQUIRED]].astype(float)
    z=-x*math.log(2.)
    def complement(logp):
        # Log(1-exp(logp)); the complementary small probability was computed first.
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.where(logp < -math.log(2.), np.log1p(-np.exp(logp)), np.log(-np.expm1(logp)))
    pos=complement(z)
    with np.errstate(divide='ignore'):
        one=np.log(x)+z
    nottwo=np.log1p(x)+z
    nottwo[x<=1]=0.
    two=complement(nottwo)
    # Gate probability and its directly evaluated complement, on disjoint genes.
    B=pos[:,0]+complement(z[:,1]+z[:,2])
    nB=np.logaddexp(z[:,0],pos[:,0]+z[:,1]+z[:,2])
    T=np.logaddexp(two[:,3]+pos[:,4],one[:,3]+two[:,4])
    nT=np.logaddexp.reduce(np.column_stack([z[:,3],pos[:,3]+z[:,4],one[:,3]+one[:,4]]),axis=1)
    M=pos[:,7]+pos[:,8]; nM=np.logaddexp(z[:,7],pos[:,7]+z[:,8])
    P=pos[:,9]+pos[:,10]; nP=np.logaddexp(z[:,9],pos[:,9]+z[:,10])
    gates=[(nB,B),(nT,T),(nM,M),(nP,P),(nottwo[:,5],two[:,5]),(z[:,6],pos[:,6])]
    out=np.full((len(x),5),-np.inf)
    for flags in itertools.product((0,1),repeat=6):
        b,t,m,pl,u,v=flags
        label=(0 if b and not(t or m or pl) else
               1 if t and not(b or m or pl) and u and not v else
               2 if t and not(b or m or pl or u) and v else
               3 if m and not(b or t or pl) else 4)
        weight=sum(gates[j][flag] for j,flag in enumerate(flags))
        out[:,label]=np.logaddexp(out[:,label],weight)
    out[~mask,:]=-np.inf;out[~mask,4]=0.
    return out


def marker_pair_availability_half(counts, genes, eligible, a, b, *, has_edges=True):
    if a==b or min(a,b)<0 or max(a,b)>3:
        raise ValueError('distinct candidate indices in 0..3 required')
    cats=marker_log_categories_half(counts,genes,eligible)
    other=np.logaddexp.reduce(cats[:,[j for j in range(5) if j not in (a,b)]],axis=1)
    return availability_from_log_states(np.column_stack([other,cats[:,a],cats[:,b]]),
        has_edges=has_edges,basis='integer_counts_complete_rule_p_half_fixed_QC')
