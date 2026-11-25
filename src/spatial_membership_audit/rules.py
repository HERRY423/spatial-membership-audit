"""A fixed, explicitly named second marker rule; not cell identity truth."""
import numpy as np


def stromal_myeloid_probabilities(counts, eligible, p):
    """COL1A1/FBN1 versus CSF1R/TYROBP, with mutual vetoes.

Stable complements avoid false structural zero from subtracting nearly-one
joint detection probabilities. Floating underflow of a supported class raises.
"""
    x=np.asarray(counts);mask=np.asarray(eligible)
    if x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('four aligned nonnegative integer raw counts required')
    if mask.dtype!=bool or mask.shape!=(len(x),) or not np.isscalar(p) or not np.isfinite(p) or not 0<=p<=1:
        raise ValueError('fixed boolean eligibility and scalar p in [0,1] required')
    if p==1:
        pos=x>0;f=pos[:,0]&pos[:,1];m=pos[:,2]&pos[:,3]
        return np.column_stack([f&~m,m&~f]).astype(float)*mask[:,None]
    z=np.exp(x*np.log1p(-p));pos=-np.expm1(x*np.log1p(-p))
    f=pos[:,0]*pos[:,1];m=pos[:,2]*pos[:,3]
    not_f=z[:,0]+pos[:,0]*z[:,1]
    not_m=z[:,2]+pos[:,2]*z[:,3]
    out=np.column_stack([f*not_m,m*not_f])*mask[:,None]
    support=np.column_stack([(x[:,0]>0)&(x[:,1]>0),(x[:,2]>0)&(x[:,3]>0)])&mask[:,None]&(p>0)
    if np.any(support&(out==0)):
        raise FloatingPointError('NUMERICAL_UNDERFLOW: raw-count class support is positive')
    return out
