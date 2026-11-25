"""Explicit AnnData and supplied-label trajectory interfaces.

No classifier is trained or downloaded here. Native Squidpy/Giotto scores
remain supplied outcomes; their null distributions are not interchangeable.
"""
import itertools
import numpy as np
from scipy import sparse
from .analytic import marker_probabilities, pair_moments, CLASSES
from .definitions import REQUIRED


def predict_anndata(adata, *, p, count_layer, eligible_key, region_key,
                    adjacency_key='spatial_connectivities'):
    """Read raw counts layer and an existing graph without modifying AnnData.

    Explicit region keys and fixed eligibility are required. Cross-region
    edges are rejected rather than silently discarded. Counts are read only
    for the 11 measured markers. X/normalized values are never guessed.
    """
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('unique cell and gene names required')
    names=list(adata.var_names)
    if not set(REQUIRED).issubset(names):raise ValueError('NOT_IDENTIFIABLE: missing panel genes')
    if adata.obs[region_key].isna().any():raise ValueError('explicit region identity required')
    mask=adata.obs[eligible_key].to_numpy()
    if mask.dtype != bool:raise ValueError('fixed eligibility must be boolean')
    ix=[names.index(g) for g in REQUIRED]
    counts=adata.layers[count_layer][:,ix]
    if sparse.issparse(counts):counts=counts.toarray()
    graph=sparse.csr_matrix(adata.obsp[adjacency_key]);graph.eliminate_zeros()
    region=adata.obs[region_key].to_numpy()
    i,j=graph.nonzero()
    if np.any(region[i]!=region[j]):raise ValueError('cross-region edges forbidden')
    retention=np.asarray(p)
    if retention.ndim==1:
        if len(retention)!=len(names):raise ValueError('gene retention must align with all var_names')
        retention=retention[ix]
    probabilities=marker_probabilities(counts,REQUIRED,mask,retention)
    rows=[]
    for name in dict.fromkeys(region):
        nodes=np.flatnonzero(region==name);local=graph[nodes][:,nodes]
        for a,b in itertools.combinations(range(4),2):
            rows.append(dict(region=str(name),label_a=CLASSES[a],label_b=CLASSES[b],
                             **pair_moments(probabilities[nodes],local,a,b)))
    return rows


def summarize_trajectory(values, *, nonnegative_ratio):
    """Describe saved official/statistic outputs, preserving nonfinite states.

    Signed scores (nhood z, Giotto log enrichment) use SD and sign frequencies;
    CV is only supplied for explicitly nonnegative ratios with positive mean.
    This does not estimate biological accuracy or a null-test p value.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=1:raise ValueError('one endpoint over supplied replicates required')
    finite=x[np.isfinite(x)]
    if nonnegative_ratio and np.any(finite<0):raise ValueError('negative ratio')
    mean=float(finite.mean()) if len(finite) else None
    sd=float(finite.std(ddof=1)) if len(finite)>1 else None
    return dict(total=len(x),finite=len(finite),nonfinite=int((~np.isfinite(x)).sum()),
        zero=int((finite==0).sum()),positive=int((finite>0).sum()),negative=int((finite<0).sum()),
        mean=mean,sd=sd,cv=sd/mean if nonnegative_ratio and mean and sd is not None else None,
        interpretation='conditional technical trajectory summary; no independent samples added')
