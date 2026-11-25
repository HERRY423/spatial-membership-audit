"""Supplied joint realizations and continuous spatial fields, without refitting.

These paths do not assume independent nodes. They describe supplied realizations,
not calibrated posterior coverage, biological replication, or thinning forecasts.
The continuous-field endpoint is explicitly distinct from categorical cell O/E.
"""
import numpy as np
from scipy import sparse
from .ecosystem import _ids


def _align(values, node_ids, graph):
    ids = _ids(node_ids, 'supplied nodes')
    if set(ids) != set(graph.cell_ids):
        raise ValueError('exact fixed graph node universe required')
    if values.shape[-2] != len(ids):
        raise ValueError('node axis mismatch')
    lookup = {key: i for i, key in enumerate(ids)}
    return np.take(values, [lookup[key] for key in graph.cell_ids], axis=-2)


def _graph(graph):
    if graph.directed or np.any(graph.adjacency.data != 1):
        raise ValueError('fixed binary undirected graph required')
    # One normalization per region. Never pool disconnected specimens.
    if len(set(graph.regions)) != 1:
        raise ValueError('exactly one region required per endpoint evaluation')


def _summary(rows, mechanism):
    values = np.array([np.nan if r['R'] is None else r['R'] for r in rows])
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else None
    sd = float(finite.std(ddof=1)) if len(finite) > 1 else None
    return dict(total=len(rows), finite=len(finite), undefined=len(rows)-len(finite),
                zero=int(np.sum(finite == 0)), positive=int(np.sum(finite > 0)),
                finite_fraction=len(finite)/len(rows), mean_R_finite=mean,
                sd_R_finite=sd, cv_R_finite=sd/mean if sd is not None and mean > 0 else None,
                mechanism=mechanism, interval_coverage='NOT_ESTABLISHED',
                interpretation='supplied realization summary; no independent biological replicates added')


def joint_label_summary(labels, node_ids, graph, *, classes, pair,
                        realization_ids, mechanism, provenance):
    """Evaluate whole label vectors; never resample their cellwise marginals.

Labels must include an explicit UNKNOWN/OTHER category when present. Missing
nodes, unknown labels and failed realizations raise, rather than being dropped.
"""
    _graph(graph)
    if mechanism not in ('posterior_joint_labels', 'refit_technical_labels') or not provenance:
        raise ValueError('explicit mechanism and provenance required')
    cats = _ids(classes, 'classes'); reps = _ids(realization_ids, 'realizations')
    if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(cats):
        raise ValueError('distinct pair in explicit class universe required')
    x = np.asarray(labels, dtype=object)
    if not reps or x.ndim != 2 or x.shape[0] != len(reps):
        raise ValueError('nonempty realizations by nodes required')
    if any(not isinstance(v, str) or v not in cats for v in x.flat):
        raise ValueError('all labels must belong to explicit classes; no implicit missing label conversion')
    x = _align(x[..., None], node_ids, graph)[..., 0]
    n = len(graph.cell_ids); g = graph.adjacency; m = g.nnz//2
    scale = 2*m/(n*(n-1)) if n > 1 else 0.
    rows = []
    for rid, lab in zip(reps, x):
        a = (lab == pair[0]).astype(float); b = (lab == pair[1]).astype(float)
        o = float(a @ (g @ b)); e = scale*float(a.sum()*b.sum())
        rows.append(dict(realization_id=rid, n_a=int(a.sum()), n_b=int(b.sum()), O=o, E=e,
                         R=o/e if e > 0 else None,
                         status='FINITE_ZERO' if e > 0 and o == 0 else ('FINITE' if e > 0 else 'NO_FINITE_RATIO')))
    return dict(endpoint='categorical_cell_O_over_E', rows=rows,
                summary=_summary(rows, mechanism), provenance=provenance,
                independence_assumption='NOT_REQUIRED_FOR_SUPPLIED_JOINT_LABELS',
                target='distribution of supplied whole-label realizations; not independent-node analytic moments')


def continuous_field_summary(values, node_ids, graph, *, classes, pair,
                             realization_ids, mechanism, value_semantics, provenance):
    """Describe nonnegative abundance/composition fields on spatial nodes.

For A,B fields: O=A.T G B and E=2M/[N(N-1)] sum_(i!=j) A_i B_j.
E is the uniform whole-node permutation expectation of O for fixed fields.
Same-node products are excluded, even for mixtures. No category draws occur.
This is between-node co-abundance, not a count of individual cell contacts.
"""
    _graph(graph)
    if mechanism not in ('posterior_joint_fields', 'point_estimate', 'refit_technical_fields'):
        raise ValueError('explicit field-generating mechanism required')
    if value_semantics not in ('cell_abundance', 'composition', 'expected_mapped_mass') or not provenance:
        raise ValueError('explicit field meaning and provenance required')
    cats = _ids(classes, 'classes'); reps = _ids(realization_ids, 'realizations')
    if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(cats):
        raise ValueError('distinct pair in explicit class universe required')
    x = np.asarray(values, dtype=float)
    if not reps or x.ndim != 3 or x.shape[0] != len(reps) or x.shape[2] != len(cats):
        raise ValueError('nonempty realization by node by class array required; summaries are not draws')
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('finite nonnegative fields required; no clipping or missing-node omission')
    if mechanism == 'point_estimate' and len(reps) != 1:
        raise ValueError('one point estimate required; repeated copies do not supply uncertainty')
    if value_semantics == 'composition' and not np.allclose(x.sum(axis=2), 1, atol=1e-10, rtol=0):
        raise ValueError('composition must already sum to one; no automatic normalization')
    x = _align(x, node_ids, graph)
    aidx, bidx = cats.index(pair[0]), cats.index(pair[1]); n = len(graph.cell_ids)
    scale = graph.adjacency.nnz/(n*(n-1)) if n > 1 else 0.
    rows = []
    for rid, field in zip(reps, x):
        a, b = field[:, aidx], field[:, bidx]
        # Prefix/suffix sum avoids cancellation of large same-node contributions.
        others = np.r_[0., np.cumsum(b[:-1])] + np.r_[np.cumsum(b[:0:-1])[::-1], 0.]
        with np.errstate(over='ignore', invalid='ignore'):
            o = float(a @ (graph.adjacency @ b)); e = scale*float(a @ others)
        if not np.isfinite(o) or not np.isfinite(e):
            raise ArithmeticError('nonfinite continuous-field endpoint')
        ratio = o/e if e > 0 else None
        if ratio is not None and not np.isfinite(ratio):
            raise ArithmeticError('nonfinite continuous-field ratio')
        rows.append(dict(realization_id=rid, O=o, E=e, R=ratio,
                         status='FINITE_ZERO' if e > 0 and o == 0 else ('FINITE' if e > 0 else 'NO_FINITE_RATIO')))
    return dict(endpoint='between_node_continuous_coabundance_O_over_permutation_E',
                value_semantics=value_semantics, rows=rows, summary=_summary(rows, mechanism),
                provenance=provenance, independence_assumption='NOT_REQUIRED_FOR_SUPPLIED_JOINT_FIELDS',
                uncertainty_status='NOT_AVAILABLE_FROM_POINT_ESTIMATE' if mechanism == 'point_estimate' else 'SUPPLIED_REALIZATIONS_ONLY',
                interpretation='distinct spatial-field endpoint; not categorical cell-contact O/E or molecular-thinning forecast')


def cell2location_field_samples(w_sf, *, sample_ids, spot_ids, class_names, joint_draw_provenance):
    """Wrap explicit native w_sf draws (S, spots, types), not means/q05/q95.

The exporter must preserve a common joint draw across all spots. Concatenating
independently sampled minibatches does not establish shared-parameter dependence.
"""
    x = np.asarray(w_sf, dtype=float)
    sid = _ids(sample_ids, 'draws'); nodes = _ids(spot_ids, 'spots'); cats = _ids(class_names, 'classes')
    if x.shape != (len(sid), len(nodes), len(cats)) or not sid or not joint_draw_provenance:
        raise ValueError('documented joint w_sf draws with explicit axes required')
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('invalid native abundance draws')
    return dict(values=x.copy(), node_ids=nodes, classes=cats, realization_ids=sid,
                mechanism='posterior_joint_fields', value_semantics='cell_abundance',
                provenance=joint_draw_provenance)


def rctd_composition_field(weights, *, provenance):
    """Consume already normalized native RCTD pixel-by-type weights DataFrame.

Raw full-mode weights must be normalized by the native routine beforehand.
Neither proportions nor repeated identical values imply a posterior distribution.
"""
    nodes = _ids(weights.index.tolist(), 'pixels'); cats = _ids(weights.columns.tolist(), 'types')
    x = weights.to_numpy(dtype=float)
    if not provenance or not nodes or not cats or not np.isfinite(x).all() or np.any(x < 0) or not np.allclose(x.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError('finite normalized native weights and provenance required; rejects retained as explicit state by caller')
    return dict(values=x[None, ...], node_ids=nodes, classes=cats, realization_ids=('point',),
                mechanism='point_estimate', value_semantics='composition', provenance=provenance)


def tangram_cell_mapping_field(adata_map, source_labels, *, provenance, mode):
    """Project cell-mode source-to-spot mapping to expected mapped source mass.

This does not create physical cell counts or cell-label posteriors. Cluster and
constrained modes require additional weighting/filter contracts and are refused.
"""
    if mode != 'cells' or not provenance:
        raise ValueError('only explicit cell-mode mapping is implemented')
    cells = _ids(adata_map.obs_names.tolist(), 'source cells')
    spots = _ids(adata_map.var_names.tolist(), 'spots')
    supplied = _ids(source_labels.index.tolist(), 'source label IDs')
    if not cells or not spots or set(cells) != set(supplied):
        raise ValueError('complete source-cell labels required')
    labs = source_labels.loc[list(cells)].tolist()
    if any(not isinstance(x, str) or not x for x in labs):
        raise ValueError('explicit source type labels required')
    cats = tuple(dict.fromkeys(labs))
    m = sparse.csr_matrix(adata_map.X, dtype=float)
    if m.shape != (len(cells), len(spots)) or not np.isfinite(m.data).all() or np.any(m.data < 0) or not np.allclose(np.asarray(m.sum(1)).ravel(), 1, atol=1e-6, rtol=0):
        raise ValueError('valid cell-by-spot native mapping probabilities required')
    membership = sparse.csr_matrix((np.ones(len(cells)), (np.arange(len(cells)), [cats.index(x) for x in labs])), shape=(len(cells), len(cats)))
    field = (m.T @ membership).toarray()
    return dict(values=field[None, ...], node_ids=spots, classes=cats, realization_ids=('point',),
                mechanism='point_estimate', value_semantics='expected_mapped_mass', provenance=provenance)
