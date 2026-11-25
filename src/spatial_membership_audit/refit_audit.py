"""Audit supplied whole-refit realizations against an explicit run schedule.

No fitting, posterior sampling, independence approximation or predictive claims.
Missing/failed runs remain in the denominator; structural NA requires a proof
outside this module, not a finite collection of empty realizations.
"""
import math
import numpy as np
from .ecosystem import _ids
from .posterior_bridge import joint_label_summary, continuous_field_summary


def audit_refits(*, baseline, realizations, failures, planned_ids, node_ids,
                 graph, classes, pair, provenance, kind='labels', value_semantics=None):
    """Each realization is one complete label vector or node-by-type field.

``failures`` maps scheduled IDs to nonempty reasons. Successful and failed IDs
must partition the plan exactly. Baseline is evaluated separately. One field
per refit can be a posterior mean; nested posterior draws must NOT be flattened
into extra refits. This caller responsibility is recorded in the output.
"""
    plan = _ids(planned_ids, 'planned refits')
    if not plan or kind not in ('labels', 'fields'):
        raise ValueError('nonempty schedule and explicit labels/fields kind required')
    ok, failed = set(realizations), set(failures)
    if ok & failed or ok | failed != set(plan):
        raise ValueError('successes and failures must exactly partition the schedule')
    if any(not isinstance(s, str) or not s.strip() for s in failures.values()):
        raise ValueError('each failed refit needs an explicit reason')
    def evaluate(value, rid):
        args = dict(node_ids=node_ids, graph=graph, classes=classes, pair=pair,
                    realization_ids=[rid], provenance=provenance)
        if kind == 'labels':
            return joint_label_summary([value], mechanism='refit_technical_labels', **args)['rows'][0]
        return continuous_field_summary([value], mechanism='refit_technical_fields',
                    value_semantics=value_semantics, **args)['rows'][0]
    base = evaluate(baseline, 'baseline')
    rows = []
    for rid in plan:
        if rid in failures:
            rows.append(dict(realization_id=rid, status='FIT_FAILED', reason=failures[rid],
                             O=None, E=None, R=None, delta_R=None))
            continue
        row = evaluate(realizations[rid], rid)
        row['delta_R'] = row['R'] - base['R'] if row['R'] is not None and base['R'] is not None else None
        row['reason'] = None
        rows.append(row)
    finite = np.array([r['R'] for r in rows if r['R'] is not None], float)
    delta = np.array([r['delta_R'] for r in rows if r['delta_R'] is not None], float)
    mean = float(finite.mean()) if len(finite) else None
    sd = float(finite.std(ddof=1)) if len(finite) > 1 else None
    successful = len(ok)
    summary = dict(planned=len(plan), successful=successful, failed=len(failed),
        finite=len(finite), undefined_successful=successful-len(finite),
        zero=int(np.sum(finite == 0)), positive=int(np.sum(finite > 0)),
        finite_fraction_of_planned=len(finite)/len(plan),
        finite_fraction_given_success=len(finite)/successful if successful else None,
        mean_R_finite=mean, sd_R_finite=sd,
        cv_R_finite=sd/mean if sd is not None and mean > 0 else None,
        mcse_mean_R_finite=sd/math.sqrt(len(finite)) if sd is not None else None,
        baseline_R=base['R'], baseline_status=base['status'],
        finite_delta=len(delta), mean_delta_R_finite=float(delta.mean()) if len(delta) else None,
        newly_undefined_if_baseline_finite=successful-len(finite) if base['R'] is not None else None)
    return dict(baseline=base, rows=rows, summary=summary, kind=kind,
        value_semantics=value_semantics, provenance=provenance,
        dependence='whole-refit realizations preserved; no independent-node formula',
        uncertainty='conditional empirical dispersion; not identity calibration or biological replication',
        posterior_draw_policy='one prespecified summary per refit; nested draws are not extra refits',
        structural_unavailability='NOT_ESTABLISHED_FROM_FINITE_REALIZATIONS',
        predictive_validation='NOT_PERFORMED')
