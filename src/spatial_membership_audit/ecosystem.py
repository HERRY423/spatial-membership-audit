"""Graph/point-pattern interoperability with explicit statistical contracts.

No downloads, classifier fitting, or biological significance decisions.
Native scores and analytic conditional moments are separate outputs.
"""
from dataclasses import dataclass
import hashlib
import inspect
import warnings
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree, Delaunay
from .analytic import _availability, _nonnegative


def _ids(values, name):
    values = tuple(values)
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f'{name}: nonempty string identifiers required')
    if len(set(values)) != len(values):
        raise ValueError(f'{name}: duplicate identifiers')
    return values


@dataclass(frozen=True)
class SpatialGraph:
    """Cell order, region isolation and graph semantics travel together.

    ``native`` declares uncalibrated source coordinates for a supplied graph.
    Graph-only statistics use adjacency, not distances. It does not authorize
    a physical length interpretation; build_graph still takes xy_um explicitly.
    """
    cell_ids: tuple
    regions: tuple
    adjacency: object
    directed: bool
    provenance: str
    coordinate_unit: str = 'um'

    def __post_init__(self):
        ids = _ids(self.cell_ids, 'cells')
        regions = tuple(self.regions)
        if len(regions) != len(ids) or any(not isinstance(x, str) or not x for x in regions):
            raise ValueError('explicit region for every cell required')
        if self.coordinate_unit not in ('um', 'native') or not self.provenance:
            raise ValueError('explicit um or uncalibrated native units and graph provenance required')
        g = sparse.csr_matrix(self.adjacency, dtype=float).copy()
        g.sum_duplicates(); g.eliminate_zeros(); g.sort_indices()
        if g.shape != (len(ids), len(ids)) or not np.isfinite(g.data).all() or np.any(g.data < 0):
            raise ValueError('finite nonnegative cell-aligned adjacency required')
        if np.any(g.diagonal()):
            raise ValueError('self edges forbidden')
        if not self.directed and (g != g.T).nnz:
            raise ValueError('undirected graph must be symmetric; no implicit symmetrization')
        i, j = g.nonzero()
        if np.any(np.asarray(regions)[i] != np.asarray(regions)[j]):
            raise ValueError('cross-region edges forbidden')
        object.__setattr__(self, 'cell_ids', ids)
        object.__setattr__(self, 'regions', regions)
        object.__setattr__(self, 'adjacency', g)


def build_graph(xy_um, cell_ids, regions, *, method, radius_um=None, k=None,
                knn_symmetry=None):
    """Per-region radius/Delaunay/kNN graphs; kNN semantics must be explicit.

    Delaunay has no implicit distance pruning; coincident coordinates rejected.
    These are SciPy graphs, never labelled native Squidpy/Giotto defaults.
    """
    ids = _ids(cell_ids, 'cells'); regions = tuple(regions)
    xy = np.asarray(xy_um, dtype=float)
    if xy.shape != (len(ids), 2) or not np.isfinite(xy).all() or len(regions) != len(ids):
        raise ValueError('finite N by 2 coordinates and aligned regions required')
    if method not in ('radius', 'knn', 'delaunay'):
        raise ValueError('unknown graph method')
    if method == 'radius' and (radius_um is None or not np.isfinite(radius_um) or radius_um <= 0):
        raise ValueError('positive finite radius required')
    if method == 'knn' and (not isinstance(k, int) or k < 1 or knn_symmetry not in ('directed','union','mutual')):
        raise ValueError('positive integer k and directed/union/mutual policy required')
    edges = []
    for region in dict.fromkeys(regions):
        ix = np.flatnonzero(np.asarray(regions) == region); local = xy[ix]
        if len(np.unique(local, axis=0)) != len(local):
            raise ValueError('coincident coordinates require an explicit upstream policy')
        if method == 'radius':
            pairs = cKDTree(local).query_pairs(radius_um, output_type='ndarray')
            for i, j in pairs: edges.extend(((ix[i],ix[j]), (ix[j],ix[i])))
        elif method == 'knn':
            if len(ix) <= k: raise ValueError('k must be smaller than every region size')
            _, neighbors = cKDTree(local).query(local, k=k+1)
            local_edges = {(i,int(j)) for i, row in enumerate(neighbors) for j in row if i != j}
            if knn_symmetry == 'union': local_edges |= {(j,i) for i,j in local_edges}
            if knn_symmetry == 'mutual': local_edges = {(i,j) for i,j in local_edges if (j,i) in local_edges}
            edges.extend((ix[i],ix[j]) for i,j in local_edges)
        else:
            if len(ix) < 3: raise ValueError('Delaunay requires at least three noncollinear points')
            triangles = Delaunay(local).simplices
            local_edges = {(int(i),int(j)) for tri in triangles for i in tri for j in tri if i != j}
            edges.extend((ix[i],ix[j]) for i,j in local_edges)
    ij = np.asarray(edges, dtype=int).reshape(-1,2)
    g = sparse.csr_matrix((np.ones(len(ij)), (ij[:,0],ij[:,1])), shape=(len(ids),len(ids)))
    return SpatialGraph(ids, regions, g, method == 'knn' and knn_symmetry == 'directed',
                        f'SciPy:{method};radius_um={radius_um};k={k};symmetry={knn_symmetry}')


def _symmetric_weights(adjacency, n):
    g = sparse.csr_matrix(adjacency, dtype=float).copy()
    g.sum_duplicates(); g.eliminate_zeros()
    if g.shape != (n,n) or not np.isfinite(g.data).all() or np.any(g.data < 0) or np.any(g.diagonal()) or (g != g.T).nnz:
        raise ValueError('nonnegative symmetric weights, no self edges required')
    return g


def graph_from_anndata(adata, *, region_key, adjacency_key='spatial_connectivities',
                       directed, coordinate_unit):
    """Read an existing AnnData graph with explicit orientation and units."""
    if adata.obs[region_key].isna().any(): raise ValueError('missing region identity')
    return SpatialGraph(tuple(adata.obs_names),tuple(adata.obs[region_key]),
                        adata.obsp[adjacency_key],directed,f'AnnData.obsp[{adjacency_key}]',coordinate_unit)


def giotto_edge_table(graph, xy_um):
    """Export unique binary undirected edges with native Giotto required columns."""
    import pandas as pd
    xy=np.asarray(xy_um,dtype=float)
    if graph.directed or np.any(graph.adjacency.data!=1):
        raise ValueError('explicit binary undirected graph required for Giotto export')
    if xy.shape!=(len(graph.cell_ids),2) or not np.isfinite(xy).all():
        raise ValueError('finite cell-aligned micrometre coordinates required')
    i,j=sparse.triu(graph.adjacency,k=1).nonzero(); ids=np.asarray(graph.cell_ids)
    return pd.DataFrame({'from':ids[i],'to':ids[j],
        'sdimx_begin':xy[i,0],'sdimy_begin':xy[i,1],'sdimx_end':xy[j,0],'sdimy_end':xy[j,1],
        'distance':np.linalg.norm(xy[i]-xy[j],axis=1),'weight':np.ones(len(i))})


def random_label_moments(adjacency, n_a, n_b):
    """Exact moments of weighted cross-type count under fixed-count node permutation.

    Includes shared-node and disjoint-edge dependence. One region only.
    This is NOT Giotto's edge-endpoint permutation or thinning variance.
    """
    n = adjacency.shape[0]; g = _symmetric_weights(adjacency, n)
    if any(not isinstance(v, (int,np.integer)) or v < 0 for v in (n_a,n_b)) or n_a+n_b > n:
        raise ValueError('valid exclusive class counts required')
    w = sparse.triu(g, k=1).data
    s, s2 = float(w.sum()), float(w@w)
    degree = np.asarray(g.sum(axis=1)).ravel()
    shared = max(0., float((degree@degree-2*s2)/2))
    disjoint = max(0., (s*s-s2)/2-shared)
    a,b = n_a,n_b
    q2 = 2*a*b/(n*(n-1)) if n >= 2 else 0.
    q3 = a*b*(a+b-2)/(n*(n-1)*(n-2)) if n >= 3 else 0.
    q4 = 4*a*(a-1)*b*(b-1)/(n*(n-1)*(n-2)*(n-3)) if n >= 4 else 0.
    var = s2*(q2-q2*q2)+2*shared*(q3-q2*q2)+2*disjoint*(q4-q2*q2)
    var = _nonnegative(var, (s*q2)**2)
    return dict(mean=s*q2, variance=var, sd=float(np.sqrt(var)),
                status='ZERO_NULL_VARIANCE' if var == 0 else 'FINITE',
                null_model='uniform_fixed_count_node_labels', weight_sum=s,
                weight_squared_sum=s2, shared_edge_weight_products=shared,
                disjoint_edge_weight_products=disjoint)


def weighted_pair_moments(probabilities, adjacency, a=0, b=1, *, independent_nodes=False):
    """Exact moments of T=sum w_ij I(cross A,B), D=n_A*n_B.

    Within-node covariance diag(p)-pp' is included. Cross-node covariance is
    assumed zero ONLY after caller acknowledgement. Ratio results are delta
    approximations conditional on D>0, not an exact ratio distribution.
    """
    if independent_nodes is not True:
        raise ValueError('independent_nodes=True required; correlated classifiers need joint label trajectories')
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or a == b or min(a,b) < 0 or max(a,b) >= p.shape[1]:
        raise ValueError('distinct valid class indices required')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or np.any(p.sum(axis=1)>1+1e-12):
        raise ValueError('exclusive categorical probabilities required; scores are not probabilities')
    g = _symmetric_weights(adjacency, len(p))
    pa,pb = p[:,a],p[:,b]; va,vb,ab=pa*(1-pa),pb*(1-pb),pa*pb
    da = np.cumsum(pa)-pa+np.cumsum(pa[::-1])[::-1]-pa
    db = np.cumsum(pb)-pb+np.cumsum(pb[::-1])[::-1]-pb
    ga,gb = g@pa,g@pb
    def proj(sa,sb,ta,tb):
        return float(np.sum(va*sb*tb+vb*sa*ta-ab*(sb*ta+sa*tb)))
    e = sparse.triu(g,k=1,format='coo'); i,j,w=e.row,e.col,e.data
    residual=va[i]*vb[j]+vb[i]*va[j]+2*ab[i]*ab[j]
    prefix=lambda x: np.r_[0.,np.cumsum(x[:-1])] if len(x) else np.empty(0)
    rd=float(va@prefix(vb)+vb@prefix(va)+2*ab@prefix(ab))
    mt,md=float(pa@gb),float(pa@db)
    vt=_nonnegative(proj(ga,gb,ga,gb)+float((w*w)@residual),mt*mt)
    vd=_nonnegative(proj(da,db,da,db)+rd,md*md)
    cov=proj(ga,gb,da,db)+float(w@residual)
    available=_availability(pa,pb)
    out=dict(mu_T=mt,mu_D=md,var_T=vt,var_D=vd,cov_T_D=cov,p_D_positive=available,
             ratio_of_conditional_means=None,var_ratio_delta=None,mean_ratio_delta2=None,
             status='NO_FINITE_DENOMINATOR',assumption='independent categorical nodes; fixed weights')
    if available and md:
        et,ed=mt/available,md/available
        ct=(cov+mt*md)/available-et*ed
        vtc=_nonnegative((vt+mt*mt)/available-et*et,(vt+mt*mt)/available)
        vdc=_nonnegative((vd+md*md)/available-ed*ed,(vd+md*md)/available)
        vr=_nonnegative(vtc/ed**2+et**2*vdc/ed**4-2*et*ct/ed**3,
                        vtc/ed**2+et**2*vdc/ed**4+abs(2*et*ct/ed**3))
        out.update(ratio_of_conditional_means=et/ed,var_ratio_delta=vr,
                   mean_ratio_delta2=et/ed+et*vdc/ed**3-ct/ed**2,status='DELTA_APPROXIMATION')
    return out


def giotto_endpoint_null_moments(adjacency, labels, pair):
    """Pinned Giotto 4.2.3 edge-endpoint shuffle, not a node-label shuffle.

    After unique undirected edges, stubs carry degree-weighted label totals.
    Native finite Monte Carlo means need not equal this exhaustive mean.
    """
    labels=np.asarray(labels)
    if labels.ndim!=1 or len(pair)!=2 or pair[0]==pair[1]:
        raise ValueError('one label per node and distinct pair required')
    g=_symmetric_weights(adjacency,len(labels))
    if np.any(g.data!=1): raise ValueError('Giotto endpoint null requires a binary graph')
    degree=np.asarray(g.sum(axis=1)).ravel()
    a,b=(int(degree[labels==label].sum()) for label in pair)
    m=g.nnz//2; total=2*m
    q2=2*a*b/(total*(total-1)) if total>=2 else 0.
    q4=4*a*(a-1)*b*(b-1)/(total*(total-1)*(total-2)*(total-3)) if total>=4 else 0.
    variance=_nonnegative(m*(q2-q2*q2)+m*(m-1)*(q4-q2*q2),(m*q2)**2)
    return dict(mean=m*q2,variance=variance,stub_a=a,stub_b=b,edges=m,
                status='NO_EDGES' if not m else ('ZERO_NULL_VARIANCE' if variance==0 else 'FINITE'),
                null_model='edge_endpoint_label_multiset_permutation_Giotto_4.2.3')


def ripley_weights(xy_um, radius_um, *, window, correction):
    """Rectangular window only. window=(xmin,xmax,ymin,ymax), supplied explicitly.

    Translation weights are area/overlap(window, window+displacement).
    Never infer a tissue/acquisition boundary from extrema of observed cells.
    """
    xy=np.asarray(xy_um,dtype=float); bounds=np.asarray(window,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or bounds.shape!=(4,) or not np.isfinite(bounds).all():
        raise ValueError('finite coordinates and explicit rectangular window required')
    xmin,xmax,ymin,ymax=bounds; width,height=xmax-xmin,ymax-ymin
    if width<=0 or height<=0 or not np.isfinite(radius_um) or radius_um<0 or radius_um>min(width,height)/2:
        raise ValueError('positive window and radius within window inradius required')
    if np.any(xy[:,0]<xmin) or np.any(xy[:,0]>xmax) or np.any(xy[:,1]<ymin) or np.any(xy[:,1]>ymax):
        raise ValueError('points outside explicit observation window')
    if correction not in ('none','translation'): raise ValueError('supported corrections: none, translation')
    edges=cKDTree(xy).query_pairs(radius_um,output_type='ndarray')
    i,j=edges[:,0],edges[:,1]; weights=np.ones(len(edges)); area=width*height
    if correction=='translation':
        delta=np.abs(xy[i]-xy[j]); overlap=(width-delta[:,0])*(height-delta[:,1])
        if np.any(overlap<=0): raise ValueError('undefined translation correction')
        weights=area/overlap
    g=sparse.csr_matrix((np.r_[weights,weights],(np.r_[i,j],np.r_[j,i])),shape=(len(xy),len(xy)))
    return g,float(area)


def cross_k(xy_um, labels, pair, radii_um, *, window, correction):
    """Cross K/L for one region with observed intensities, retaining absent types.

    This estimator is not a hypothesis test. pi*r^2 assumes homogeneous
    independent types and is not the fixed-location random-label reference.
    """
    labels=np.asarray(labels)
    if len(labels)!=len(xy_um) or labels.ndim!=1 or len(pair)!=2 or pair[0]==pair[1]:
        raise ValueError('aligned labels and distinct pair required')
    if any(not isinstance(v,str) or not v for v in labels):
        raise ValueError('explicit nonempty string labels; use UNKNOWN for missing classifications')
    radii=np.asarray(radii_um,dtype=float)
    if radii.ndim!=1 or not len(radii) or not np.isfinite(radii).all() or np.any(np.diff(radii)<=0):
        raise ValueError('increasing finite radii required')
    a,b=(labels==v for v in pair); d=int(a.sum())*int(b.sum()); rows=[]
    for r in radii:
        g,area=ripley_weights(xy_um,float(r),window=window,correction=correction)
        t=float(a.astype(float)@g@b.astype(float)); kval=area*t/d if d else None
        rows.append(dict(radius_um=float(r),weighted_count=t,n_a=int(a.sum()),n_b=int(b.sum()),
            K=kval,L=float(np.sqrt(kval/np.pi)) if d else None,
            L_minus_r=float(np.sqrt(kval/np.pi)-r) if d else None,
            poisson_reference=float(np.pi*r*r),correction=correction,
            status='FINITE' if d else 'ABSENT_TYPE'))
    return rows


def predict_cross_k(probabilities, xy_um, radii_um, *, window, correction,
                    a=0, b=1, independent_nodes=False):
    """Fixed-location categorical propagation to cross K and L.

    Exact T,D covariance; K and L moments are delta approximations. At K=0,
    the L derivative is singular and its variance is deliberately not returned.
    """
    if len(probabilities)!=len(xy_um): raise ValueError('probabilities must align with coordinates')
    radii=np.asarray(radii_um,dtype=float)
    if radii.ndim!=1 or not len(radii) or not np.isfinite(radii).all() or np.any(np.diff(radii)<=0):
        raise ValueError('increasing finite radii required')
    rows=[]
    for r in radii:
        g,area=ripley_weights(xy_um,float(r),window=window,correction=correction)
        out=weighted_pair_moments(probabilities,g,a,b,independent_nodes=independent_nodes)
        ratio=out['ratio_of_conditional_means']
        kval=area*ratio if ratio is not None else None
        var_k=area**2*out['var_ratio_delta'] if ratio is not None else None
        rows.append(dict(radius_um=float(r),correction=correction,moments=out,
            K_at_conditional_means=kval,var_K_delta=var_k,
            L_at_conditional_means=float(np.sqrt(kval/np.pi)) if kval is not None else None,
            var_L_delta=var_k/(4*np.pi*kval) if kval is not None and kval>0 else None,
            L_status='DELTA_APPROXIMATION' if kval is not None and kval>0 else
                     ('ZERO_K_DERIVATIVE_UNDEFINED' if kval==0 else 'ABSENT_TYPE'),
            scientific_calibration='NOT_ESTABLISHED'))
    return rows


def native_squidpy(graph, labels, *, n_perms, seed, categories=None):
    """Run official nhood_enrichment per region, retain raw native z/count.

    Binary topology required since the native function ignores edge weights.
    Supports 1.6 tuple and 1.8 named result. Does not mutate caller AnnData.
    """
    import anndata as ad
    import pandas as pd
    import squidpy as sq
    labels=tuple(labels)
    if len(labels)!=len(graph.cell_ids) or any(not isinstance(x,str) or not x for x in labels):
        raise ValueError('explicit aligned labels required; use UNKNOWN instead of dropping cells')
    if np.any(graph.adjacency.data!=1): raise ValueError('Squidpy nhood counts topology; weighted input refused')
    if not isinstance(n_perms,int) or n_perms<2 or not isinstance(seed,int): raise ValueError('explicit permutations >=2 and integer seed required')
    rows=[]; categories=sorted(set(labels)) if categories is None else list(_ids(categories,'categories'))
    if len(categories)<2 or not set(labels).issubset(categories):
        raise ValueError('at least two explicit categories covering all labels required')
    region_array=np.asarray(graph.regions); captured_warnings=[]
    for region in dict.fromkeys(graph.regions):
        ix=np.flatnonzero(region_array==region); local_labels=np.asarray(labels)[ix]
        obs=pd.DataFrame({'label':pd.Categorical(local_labels,categories=categories)},index=np.asarray(graph.cell_ids)[ix])
        obj=ad.AnnData(X=sparse.csr_matrix((len(ix),0)),obs=obs)
        g=graph.adjacency[ix][:,ix]; obj.obsp['spatial_connectivities']=g.copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',RuntimeWarning)
            result=sq.gr.nhood_enrichment(obj,cluster_key='label',copy=True,n_perms=n_perms,
                seed=seed,n_jobs=1,backend='threading',numba_parallel=False,show_progress_bar=False)
        if hasattr(result,'zscore'): z,count=result.zscore,result.counts
        else: z,count=result
        for i,a in enumerate(categories):
            for j,b in enumerate(categories):
                if i==j: continue
                present=np.any(local_labels==a) and np.any(local_labels==b)
                row=dict(region=region,label_a=a,label_b=b,count=int(count[i,j]),zscore=float(z[i,j]),
                    status='ABSENT_TYPE' if not present else ('FINITE' if np.isfinite(z[i,j]) else 'NONFINITE_NATIVE_Z'),
                    exact_node_null=None)
                if not graph.directed:
                    row['exact_node_null']=random_label_moments(g,int(sum(local_labels==a)),int(sum(local_labels==b)))
                rows.append(row)
        captured_warnings.extend(str(w.message) for w in caught)
    source=inspect.getsourcefile(sq.gr.nhood_enrichment)
    with open(source,'rb') as handle: digest=hashlib.sha256(handle.read()).hexdigest()
    return dict(rows=rows,tool='squidpy.gr.nhood_enrichment',version=sq.__version__,
                source_sha256=digest,n_perms=n_perms,seed=seed,
                graph=graph.provenance,warnings=sorted(set(captured_warnings)),
                null_model='within_region_uniform_node_label_permutation')


def align_annotation(table, cell_ids, *, id_column, label_column, method):
    """Common import for native label exports. Never turns scores into posteriors."""
    ids=_ids(cell_ids,'cells')
    supplied=_ids(table[id_column].tolist(),'annotation cells')
    if set(ids)!=set(supplied): raise ValueError('annotation must cover exactly the fixed cell universe')
    ordered=table.set_index(id_column).loc[list(ids)]
    # Native CellTypist returns pandas Categorical; inserting UNKNOWN directly
    # would fail even without missing values when UNKNOWN is not a category.
    labels=ordered[label_column].astype(object).fillna('UNKNOWN').astype(str).to_numpy()
    if np.any(labels==''): raise ValueError('empty labels; use explicit UNKNOWN')
    return dict(cell_ids=ids,labels=labels,method=method,
                probability_semantics='NOT_INFERRED_FROM_LABELS_OR_SCORES')


def from_celltypist(result, cell_ids, *, label_column='predicted_labels'):
    """Consume AnnotationResult; majority voting must be selected explicitly."""
    table=result.predicted_labels.copy()
    table.insert(0,'cell_id',table.index.astype(str))
    return align_annotation(table,cell_ids,id_column='cell_id',label_column=label_column,method='CellTypist')


def categorical_input(table, cell_ids, *, id_column, class_columns, semantics,
                      independent_nodes, provenance):
    """Explicit gate for probabilities from a supplied model, never auto-normalize.

    CellTypist one-vs-rest probabilities and SingleR scores are not a categorical
    posterior. Transition frequencies from joint replicates still do not justify
    cross-cell independence. Caller records the intended probability mechanism.
    """
    if semantics not in ('categorical_posterior','conditional_reassignment_probability'):
        raise ValueError('unknown probability semantics; raw classifier scores are forbidden')
    if independent_nodes is not True or not provenance:
        raise ValueError('explicit independent-node assumption and provenance required')
    ids=_ids(cell_ids,'cells'); supplied=_ids(table[id_column].tolist(),'probability cells')
    if set(ids)!=set(supplied) or len(set(class_columns))!=len(class_columns) or not class_columns:
        raise ValueError('complete cell universe and distinct class columns required')
    p=table.set_index(id_column).loc[list(ids),list(class_columns)].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1,atol=1e-10,rtol=0):
        raise ValueError('full mutually exclusive categorical distribution required; no normalization or clipping')
    return dict(probabilities=p,cell_ids=ids,classes=tuple(class_columns),semantics=semantics,
                independent_nodes=True,provenance=provenance,
                calibration_status='NOT_ESTABLISHED_BY_IMPORT')
