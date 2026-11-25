"""Direct categorical MC baseline with explicit graph preparation and support.

Same independent node-label model as analytic_prepared. Deterministic OTHER
nodes are skipped; positive-support isolates remain, and original N/M define E.
No automatic method selection or guarantee about adequate draw counts.
"""
import numpy as np
from .analytic_prepared import PreparedPairGraph


def sample_pair_prepared(probabilities, prepared, draws, seed, *, return_draws=False):
    if not isinstance(prepared, PreparedPairGraph):
        raise ValueError('prepare_pair_graph result required')
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(prepared.n,2) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or np.any(p.sum(1)>1+1e-12):
        raise ValueError('aligned two-class exclusive probabilities required')
    if isinstance(draws,bool) or not isinstance(draws,(int,np.integer)) or draws<2:
        raise ValueError('integer draws >= 2 required')
    active=np.flatnonzero(np.any(p>0,axis=1))
    g=prepared.graph
    if len(active)!=prepared.n:
        p=p[active];g=g[active][:,active]
    n,m=prepared.n,prepared.m
    c=2*m/(n*(n-1)) if n>1 else 0.
    rng=np.random.default_rng(seed);values=np.full(draws,np.nan)
    cumulative=p.sum(axis=1)
    for start in range(0,draws,64):
        end=min(draws,start+64);u=rng.random((end-start,len(p)))
        a=(u<p[:,0]).astype(float);b=((u>=p[:,0])&(u<cumulative)).astype(float)
        o=np.sum((a@g)*b,axis=1);e=c*a.sum(1)*b.sum(1)
        valid=e>0;values[start:end][valid]=o[valid]/e[valid]
    finite=values[np.isfinite(values)];mean=float(finite.mean()) if len(finite) else None
    cv=float(finite.std(ddof=1)/mean) if len(finite)>1 and mean>0 else None
    result=dict(cv=cv,p_finite=len(finite)/draws,status='FINITE_SAMPLE_ESTIMATE' if cv is not None else 'SAMPLE_CV_UNDEFINED')
    return (result,values) if return_draws else result
