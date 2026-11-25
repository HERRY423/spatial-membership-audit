"""Direct QC-failure log probability under independent binomial retention.

Uses the disjoint union D<d OR (D>=d AND T<t). Neither term is obtained
by subtracting a rounded near-one pass probability. Counts need the complete
measured panel. This binary64 log DP has no universal relative-error certificate.
"""
import math
import numpy as np
from scipy.special import logsumexp


def qc_failure_log_probability(counts, retention, *, min_total=50, min_detected=20):
    """Return finite natural/base-10 logs for reachable tails, including underflow.

    Structural zero has log_probability/log10_probability=None (log(0)=-inf).
    The float probability and its status remain separate from the log output.
    """
    x=np.asarray(counts)
    if x.ndim!=1 or not np.issubdtype(x.dtype,np.number) or np.iscomplexobj(x):
        raise ValueError('one-dimensional integer counts required')
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)) or np.any(x>2**53):
        raise ValueError('finite nonnegative integer counts <=2**53 required')
    for threshold in (min_total,min_detected):
        if isinstance(threshold,(bool,np.bool_)) or not isinstance(threshold,(int,np.integer)) or threshold<0:
            raise ValueError('nonnegative integer QC thresholds required')
    t,d=int(min_total),int(min_detected)
    if (t+1)*(d+1)>2_000_000:
        raise ValueError('QC state space exceeds declared limit')
    p=np.asarray(retention,dtype=float)
    if p.ndim==0:p=np.full(len(x),float(p))
    if p.shape!=x.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('retention must be scalar or aligned vector in [0,1]')
    x=[int(v) for v in x]
    certain_t=sum(n for n,q in zip(x,p) if q==1)
    certain_d=sum(n>0 and q==1 for n,q in zip(x,p))
    maximum_t=sum(n for n,q in zip(x,p) if q>0)
    maximum_d=sum(n>0 and q>0 for n,q in zip(x,p))
    base=dict(method='log_domain_disjoint_QC_failure_DP',
              precision_scope='binary64 log arithmetic; no universal relative-error certificate')
    if certain_t>=t and certain_d>=d:
        return dict(base,log_probability=None,log10_probability=None,probability_float=0.,
                    float_status='STRUCTURAL_ZERO',log_status='STRUCTURAL_ZERO')
    if maximum_t<t or maximum_d<d:
        return dict(base,log_probability=0.,log10_probability=0.,probability_float=1.,
                    float_status='POSITIVE_FINITE',log_status='POSITIVE_LOG_FINITE')

    # Marginal D<d: no total-count dimension and no upper-tail subtraction.
    detect=np.full(d,-np.inf)
    if d:detect[0]=0.
    # Joint T<t,D>=d: retain t=0,...,t-1 and cap only detected count at d.
    low_total=np.full((t,d+1),-np.inf)
    if t:low_total[0,0]=0.
    for n,q in zip(x,p):
        if n==0 or q==0:continue
        if q==1:
            if d:detect=np.concatenate(([-np.inf],detect[:-1]))
            if t:
                nxt=np.full_like(low_total,-np.inf)
                if n<t:
                    if d:
                        nxt[n:,1:]=low_total[:-n,:-1]
                        nxt[n:,d]=np.logaddexp(nxt[n:,d],low_total[:-n,d])
                    else:nxt[n:,0]=low_total[:-n,0]
                low_total=nxt
            continue
        log_zero=n*math.log1p(-float(q))
        log_positive=math.log(-math.expm1(log_zero))
        if d:
            updated=detect+log_zero
            updated[1:]=np.logaddexp(updated[1:],detect[:-1]+log_positive)
            detect=updated
        if t:
            updated=low_total+log_zero
            log_mass=log_zero
            log_odds=math.log(float(q))-math.log1p(-float(q))
            for k in range(1,min(n,t-1)+1):
                log_mass+=math.log(n-k+1)-math.log(k)+log_odds
                if d:
                    updated[k:,1:]=np.logaddexp(updated[k:,1:],low_total[:-k,:-1]+log_mass)
                    updated[k:,d]=np.logaddexp(updated[k:,d],low_total[:-k,d]+log_mass)
                else:updated[k:,0]=np.logaddexp(updated[k:,0],low_total[:-k,0]+log_mass)
            low_total=updated
    log_d=float(logsumexp(detect)) if d else -math.inf
    log_t=float(logsumexp(low_total[:,d])) if t else -math.inf
    log_failure=float(np.logaddexp(log_d,log_t))
    if not math.isfinite(log_failure) or log_failure>0:
        raise ArithmeticError('reachable QC failure log left its finite nonpositive domain')
    value=math.exp(log_failure)
    state='NUMERICAL_UNDERFLOW' if value==0 else 'POSITIVE_SUBNORMAL' if value<np.finfo(float).tiny else 'POSITIVE_FINITE'
    return dict(base,log_probability=log_failure,log10_probability=log_failure/math.log(10),
                probability_float=value,float_status=state,log_status='POSITIVE_LOG_FINITE',
                log_detected_failure=log_d if math.isfinite(log_d) else None,
                log_total_failure_with_detected_pass=log_t if math.isfinite(log_t) else None)
