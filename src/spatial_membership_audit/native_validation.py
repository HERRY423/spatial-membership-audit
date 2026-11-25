"""Validate native whole-output records against a fixed graph and run plan.

Each record carries its own node and class axes. Invalid outputs, failed fits
and unstarted fits remain different scheduled states. This checks interfaces
and observed endpoint arithmetic, never classifier accuracy or independence.
"""
import math
import numpy as np
from scipy import sparse
from .ecosystem import _ids
from .posterior_bridge import joint_label_summary, continuous_field_summary


SEMANTICS = ('hard_labels', 'composition', 'cell_abundance', 'expected_mapped_mass')


def _aligned(record, graph, classes, semantics):
    if not isinstance(record, dict) or not isinstance(record.get('provenance'), str) or not record['provenance'].strip():
        raise ValueError('each native output requires source provenance')
    if record.get('semantics') != semantics:
        raise ValueError('output semantics must match the declared route; no implicit posterior/weight conversion')
    nodes = _ids(record['node_ids'], 'native output nodes')
    cats = _ids(record['classes'], 'native output classes')
    if set(nodes) != set(graph.cell_ids):
        raise ValueError('exact fixed node universe required for every output')
    if set(cats) != set(classes):
        raise ValueError('exact class universe required; no silent lost-class zero filling')
    values = np.asarray(record['values'], dtype=object if semantics == 'hard_labels' else float)
    expected = (len(nodes),) if semantics == 'hard_labels' else (len(nodes), len(cats))
    if values.shape != expected:
        raise ValueError('native axis dimensions do not match identifiers')
    if semantics == 'hard_labels':
        if any(not isinstance(v, str) or v not in classes for v in values):
            raise ValueError('explicit label from the class universe required; no argmax or missing conversion')
    else:
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('finite nonnegative field required; no clipping or imputation')
        if semantics == 'composition' and not np.allclose(values.sum(1), 1., rtol=0, atol=1e-10):
            raise ValueError('composition must be normalized upstream by its native procedure')
    lookup = {v:i for i,v in enumerate(nodes)}
    values = values[[lookup[v] for v in graph.cell_ids]]
    if semantics != 'hard_labels':
        lookup = {v:i for i,v in enumerate(cats)}
        values = values[:, [lookup[v] for v in classes]]
    return values


def _reference(values, graph, classes, pair, semantics):
    """Edge-list O and direct off-diagonal node-pair E; no matrix endpoint call."""
    n = len(graph.cell_ids)
    edges = sparse.triu(graph.adjacency, k=1, format='coo')
    if semantics == 'hard_labels':
        a, b = values == pair[0], values == pair[1]
        o = float(sum(int(a[i] and b[j]) + int(b[i] and a[j]) for i,j in zip(edges.row, edges.col)))
        off_diagonal = int(a.sum()) * int(b.sum())
    else:
        a = values[:, classes.index(pair[0])]
        b = values[:, classes.index(pair[1])]
        o = math.fsum(float(a[i])*float(b[j]) + float(b[i])*float(a[j]) for i,j in zip(edges.row, edges.col))
        off_diagonal = math.fsum(float(a[i])*math.fsum(float(b[j]) for j in range(n) if j != i) for i in range(n))
    e = (2*len(edges.data)/(n*(n-1))) * off_diagonal if n > 1 else 0.
    if not math.isfinite(o) or not math.isfinite(e):
        raise ArithmeticError('independent reference is nonfinite')
    r = o/e if e > 0 else None
    if r is not None and not math.isfinite(r):
        raise ArithmeticError('independent reference ratio is nonfinite')
    return {'O':o, 'E':e, 'R':r,
            'status':'NO_FINITE_RATIO' if e == 0 else ('FINITE_ZERO' if o == 0 else 'FINITE')}


def validate_native_output(record, *, graph, classes, pair, semantics,
                           realization_id, rtol=1e-10, atol=1e-12):
    """Validate one complete native output and compare two endpoint calculations.

The numerical tolerances check finite arithmetic agreement. They do not certify
model inference, input precision, probability calibration or a physical null.
"""
    classes = _ids(classes, 'declared classes')
    if semantics not in SEMANTICS or len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(classes):
        raise ValueError('explicit supported semantics and distinct declared pair required')
    if not graph.cell_ids or graph.directed or np.any(graph.adjacency.data != 1) or len(set(graph.regions)) != 1:
        raise ValueError('nonempty single-region binary undirected fixed graph required')
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError('finite nonnegative comparison tolerances required')
    values = _aligned(record, graph, classes, semantics)
    kw = dict(node_ids=graph.cell_ids, graph=graph, classes=classes, pair=pair,
              realization_ids=[realization_id], provenance=record['provenance'])
    if semantics == 'hard_labels':
        result = joint_label_summary([values], mechanism='refit_technical_labels', **kw)
    else:
        result = continuous_field_summary([values], mechanism='point_estimate', value_semantics=semantics, **kw)
    got = result['rows'][0]
    ref = _reference(values, graph, classes, pair, semantics)
    for key in ('O','E','R'):
        if got[key] is None or ref[key] is None:
            if got[key] is not None or ref[key] is not None:
                raise ArithmeticError(f'independent reference disagrees on definedness: {key}')
        elif not math.isclose(got[key], ref[key], rel_tol=rtol, abs_tol=atol):
            raise ArithmeticError(f'independent reference disagrees: {key}')
    if got['status'] != ref['status']:
        raise ArithmeticError('independent reference disagrees on zero/undefined state')
    return dict(got, independent_reference=ref, endpoint=result['endpoint'], semantics=semantics,
                source_provenance=record['provenance'],
                validation='NATIVE_OUTPUT_CONTRACT_AND_ENDPOINT_ARITHMETIC',
                identity_accuracy='NOT_ASSESSED', predictive_calibration='NOT_ASSESSED')


def audit_native_outputs(*, baseline, outputs, failed, not_run, planned_ids,
                         graph, classes, pair, semantics, provenance):
    """Audit independently indexed outputs with an exhaustive scheduled ledger.

``outputs`` maps each ID to its own values, node_ids, classes and provenance.
Only one prespecified label vector/field is accepted per scheduled refit.
Posterior draws must not be flattened into extra refits. ``not_run`` differs
from a failed fit; invalid supplied outputs are retained as INVALID_OUTPUT.
"""
    plan = _ids(planned_ids, 'planned outputs')
    if not plan or not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('nonempty scheduled plan and provenance required')
    sets = [set(x) for x in (outputs, failed, not_run)]
    if set.union(*sets) != set(plan) or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('supplied, failed and not-run IDs must partition the plan exactly')
    if any(not isinstance(v,str) or not v.strip() for d in (failed,not_run) for v in d.values()):
        raise ValueError('failed and not-run outputs need explicit reasons')
    # Global contract errors must not be disguised as failures of individual fits.
    classes = _ids(classes, 'declared classes')
    if semantics not in SEMANTICS or len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(classes):
        raise ValueError('invalid global output semantics or pair')
    if not graph.cell_ids or graph.directed or np.any(graph.adjacency.data != 1) or len(set(graph.regions)) != 1:
        raise ValueError('nonempty single-region binary undirected fixed graph required')
    def evaluate(record, rid):
        try:
            return validate_native_output(record, graph=graph, classes=classes, pair=pair,
                semantics=semantics, realization_id=rid)
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            return dict(realization_id=rid,status='INVALID_OUTPUT',reason=f'{type(error).__name__}: {error}',
                        O=None,E=None,R=None)
    base = evaluate(baseline, 'baseline')
    rows = []
    for rid in plan:
        if rid in failed or rid in not_run:
            row = dict(realization_id=rid,status='FIT_FAILED' if rid in failed else 'NOT_RUN',
                       reason=failed.get(rid,not_run.get(rid)),O=None,E=None,R=None)
        else:
            row = evaluate(outputs[rid],rid)
        row['delta_R'] = row['R']-base['R'] if row['R'] is not None and base['R'] is not None else None
        rows.append(row)
    finite = np.asarray([r['R'] for r in rows if r['R'] is not None], dtype=float)
    accepted = sum(r['status'] in ('FINITE','FINITE_ZERO','NO_FINITE_RATIO') for r in rows)
    mean = float(finite.mean()) if len(finite) else None
    sd = float(finite.std(ddof=1)) if len(finite)>1 else None
    summary = dict(planned=len(plan),supplied=len(outputs),fit_failed=len(failed),not_run=len(not_run),
        invalid_outputs=sum(r['status']=='INVALID_OUTPUT' for r in rows),accepted_outputs=accepted,
        finite=len(finite),undefined_accepted=accepted-len(finite),
        finite_zero=int(np.sum(finite==0)),finite_positive=int(np.sum(finite>0)),
        finite_fraction_of_planned=len(finite)/len(plan),
        finite_fraction_of_accepted=len(finite)/accepted if accepted else None,
        mean_R_finite=mean,sd_R_finite=sd,cv_R_finite=sd/mean if sd is not None and mean>0 else None)
    return dict(baseline=base,rows=rows,summary=summary,semantics=semantics,provenance=provenance,
        node_axis_policy='each native realization aligned independently; exact node universe',
        class_axis_policy='each numeric field aligned independently; exact class universe',
        dependence='whole native output preserved; no independent-node approximation',
        sampling_unit='scheduled refit, not node, posterior draw or biological replicate',
        posterior_draw_policy='one prespecified output per refit; nested draws require a separate design',
        structural_unavailability='NOT_ESTABLISHED_FROM_FINITE_REALIZATIONS',
        identity_accuracy='NOT_ASSESSED',predictive_calibration='NOT_ASSESSED')
