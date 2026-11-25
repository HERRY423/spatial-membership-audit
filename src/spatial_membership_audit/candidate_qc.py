"""Joint QC for common125_candidate_v1, using nonnegative capped polynomials.

Complete measured Gene Expression counts must be supplied by the caller.
No cell is removed from a spatial graph. Failures, conflicts and unresolved
labels contribute to its OTHER state. This is an opt-in operation, not a
change to the original fixed-QC classifier. Tail relative accuracy is not
certified; raw support is tracked separately from floating probability mass.
"""
from itertools import product
import math

import numpy as np
from scipy.stats import binom

from .analytic import CLASSES
from .definitions import REQUIRED, panel_contract
from .joint_qc import _gene_terms, _advance_background, _probability_state
from .qc_tail import qc_failure_log_probability


def _convolve(a, b, cap):
    raw = np.convolve(a, b)
    out = raw[:cap+1].copy()
    out[cap] = math.fsum(raw[cap:])
    return out


def _components(n, p, cap):
    """0 / 1 / >=2 retained molecules, each with total-count distribution.

    The maximum attainable count is integer support information, independent
    of pmf/sf rounding or underflow. It also preserves the classification
    threshold when the QC total cap is smaller than two.
    """
    arrays = [np.zeros(cap+1) for _ in range(3)]
    if p == 0 or n == 0:
        arrays[0][0] = 1.
        return arrays, [0, None, None]
    if p == 1:
        category = min(n, 2)
        arrays[category][min(n, cap)] = 1.
        maximum = [None]*3; maximum[category] = n
        return arrays, maximum
    arrays[0][0] = math.exp(n*math.log1p(-p))
    arrays[1][min(1, cap)] = float(binom.pmf(1, n, p))
    if n >= 2:
        if cap < 2:
            arrays[2][cap] = float(binom.sf(1, n, p))
        else:
            for k in range(2, min(n, cap-1)+1):
                arrays[2][k] = float(binom.pmf(k, n, p))
            if n >= cap:
                arrays[2][cap] = float(binom.sf(cap-1, n, p))
    return arrays, [0, 1, n if n >= 2 else None]


def _group(components, indices, predicate, cap):
    k = len(indices)
    probabilities = np.zeros((2, cap+1, k+1))
    support = [{}, {}]
    for levels in product(range(3), repeat=k):
        maximum = [components[i][1][level] for i,level in zip(indices,levels)]
        if any(n is None for n in maximum):
            continue
        event, detected = int(predicate(levels)), sum(v > 0 for v in levels)
        total = np.zeros(cap+1); total[0] = 1.
        for i,level in zip(indices,levels):
            total = _convolve(total, components[i][0][level], cap)
        probabilities[event, :, detected] += total
        support[event][detected] = max(support[event].get(detected, -1), sum(maximum))
    return probabilities, support


def _combine(left, right, cap_t, cap_d):
    out = np.zeros((cap_t+1, min(cap_d, left.shape[1]+right.shape[1]-2)+1))
    for i in range(left.shape[1]):
        if not np.any(left[:,i]):
            continue
        for j in range(right.shape[1]):
            if not np.any(right[:,j]):
                continue
            out[:, min(i+j, cap_d)] += _convolve(left[:,i], right[:,j], cap_t)
    return out


def _support_combine(left, right):
    out = {}
    for dl,tl in left.items():
        for dr,tr in right.items():
            out[dl+dr] = max(out.get(dl+dr, -1), tl+tr)
    return out


def candidate_joint_qc(counts, genes, retention, *, min_total=50, min_detected=20,
                       include_partition=False):
    """One cell: four full-rule class AND QC probabilities plus QC probability.

    Genes must be unique and include all REQUIRED markers. Their completeness
    as a measured panel is a caller/provenance obligation, not inferred here.
    Probabilities describe an eligible baseline cell; the batch adapter below
    requires the original eligibility mask and holds ineligible cells OTHER.
    """
    x = np.asarray(counts); genes = list(genes)
    if x.ndim != 1 or x.shape != (len(genes),) or not np.issubdtype(x.dtype,np.number) or np.iscomplexobj(x):
        raise ValueError('aligned one-dimensional integer counts and measured genes required')
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)) or np.any(x > 2**53):
        raise ValueError('finite nonnegative integer counts <=2**53 required')
    if panel_contract(genes)['missing_genes']:
        raise ValueError('NOT_IDENTIFIABLE: missing measured rule genes')
    for threshold in (min_total,min_detected):
        if isinstance(threshold,(bool,np.bool_)) or not isinstance(threshold,(int,np.integer)) or threshold < 0:
            raise ValueError('nonnegative integer QC thresholds required')
    if (min_total+1)*(min_detected+1) > 2_000_000:
        raise ValueError('QC state space exceeds declared limit')
    p = np.asarray(retention,dtype=float)
    if p.ndim == 0: p = np.full(len(x),float(p))
    if p.shape != x.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('retention must be scalar or aligned vector in [0,1]')
    x = [int(v) for v in x]; ct,cd = int(min_total),int(min_detected)
    marker = [genes.index(g) for g in REQUIRED]
    background = [i for i in range(len(x)) if i not in marker]
    bg = np.zeros((ct+1,cd+1)); bg[0,0] = 1.
    for i in background:
        bg = _advance_background(bg,*_gene_terms(x[i],float(p[i]),ct))
    components = [_components(x[i],float(p[i]),ct) for i in marker]
    groups = [
        _group(components,[0,1,2],lambda q:q[0]>0 and (q[1]>0 or q[2]>0),ct),
        _group(components,[3,4],lambda q:q[0]>0 and q[1]>0 and sum(q)>=3,ct),
        _group(components,[7,8],lambda q:q[0]>0 and q[1]>0,ct),
        _group(components,[9,10],lambda q:q[0]>0 and q[1]>0,ct),
        _group(components,[5],lambda q:q[0]>=2,ct),
        _group(components,[6],lambda q:q[0]>=1,ct),
    ]
    residual = max([abs(math.fsum(bg.ravel())-1.)]+
                   [abs(math.fsum(g[0].ravel())-1.) for g in groups])
    if residual > 2e-12:
        raise ArithmeticError('probability mass residual exceeds 2e-12')
    bg_tails = bg[::-1,::-1].cumsum(axis=0).cumsum(axis=1)[::-1,::-1]
    bg_max_t = sum(x[i] for i in background if p[i]>0)
    bg_max_d = sum(x[i]>0 and p[i]>0 for i in background)
    # B, CD8, CD4, myeloid, then QC irrespective of label. None is either gate.
    constraints = [(1,0,0,0,None,None),(0,1,0,0,1,0),
                   (0,1,0,0,0,1),(0,0,1,0,None,None),(None,)*6]
    # Seven disjoint rectangles of gate states cover OTHER while QC passes.
    # This complement is evaluated directly, never as 1-sum(class masses).
    if include_partition:
        constraints += [(None,None,None,1,None,None),(0,0,0,0,None,None),
                        (1,1,None,0,None,None),(1,0,1,0,None,None),
                        (0,1,1,0,None,None),(0,1,0,0,0,0),(0,1,0,0,1,1)]
    values, states = [], []
    qc_failure = None
    for desired in constraints:
        poly = np.zeros((ct+1,1)); poly[0,0]=1.
        reachable = {0:0}
        for (gp,gs),flag in zip(groups,desired):
            if flag is None:
                factor=gp.sum(axis=0)
                allowed={d:max(gs[0].get(d,-1),gs[1].get(d,-1)) for d in gs[0].keys()|gs[1].keys()}
            else:
                factor,allowed=gp[flag],gs[flag]
            poly=_combine(poly,factor,ct,cd)
            reachable=_support_combine(reachable,allowed)
        terms=[float(poly[t,d])*float(bg_tails[max(0,ct-t),max(0,cd-d)])
               for t in range(ct+1) for d in range(poly.shape[1])]
        value=math.fsum(terms)
        possible=any(t+bg_max_t>=ct and d+bg_max_d>=cd for d,t in reachable.items())
        values.append(value);states.append(_probability_state(value,possible))
        if desired == (None,)*6:
            # Direct, disjoint complement states. Never obtain rare QC failure
            # as 1 minus a rounded near-one pass probability.
            full=_combine(bg,poly,ct,cd)
            qc_failure=math.fsum([*full[:ct,:].ravel(),*full[ct,:cd]])
            partition_mass=math.fsum(full.ravel())
            if abs(partition_mass-1.)>2e-12:
                raise ArithmeticError('QC partition mass residual exceeds 2e-12')
    classes,qc_direct=values[:4],values[4]
    # Use the smaller directly accumulated side. This is the complement
    # identity, not clipping or renormalizing target-class probabilities.
    failure_direct=qc_failure
    from_failure=failure_direct<=.5
    qc=1.-failure_direct if from_failure else qc_direct
    qc_failure=failure_direct if from_failure else 1.-qc_direct
    certain_t=sum(n for n,q in zip(x,p) if q==1.)
    certain_d=sum(n>0 and q==1. for n,q in zip(x,p))
    failure_possible=certain_t<ct or certain_d<cd
    failure_state=_probability_state(qc_failure,failure_possible)
    if failure_state in ('NUMERICAL_UNDERFLOW','POSITIVE_SUBNORMAL'):
        tail=qc_failure_log_probability(x,p,min_total=ct,min_detected=cd)
        log_failure,log10_failure=tail['log_probability'],tail['log10_probability']
        log_status,log_method=tail['log_status'],tail['method']
    elif failure_state=='STRUCTURAL_ZERO':
        log_failure=log10_failure=None
        log_status='STRUCTURAL_ZERO';log_method='raw_support'
    else:
        log_failure=math.log(qc_failure);log10_failure=log_failure/math.log(10)
        log_status='POSITIVE_LOG_FINITE';log_method='log_of_direct_binary64_failure'
    if min(values)<0 or max(values)>1+2e-12 or math.fsum(classes)>qc+2e-12:
        raise ArithmeticError('joint class/QC probability inconsistency')
    result = dict(classes=list(CLASSES),probabilities=classes,p_qc=qc,
                p_qc_direct=qc_direct,p_qc_failure_direct=failure_direct,
                p_qc_failure=qc_failure,qc_failure_status=failure_state,
                qc_failure_log=log_failure,qc_failure_log10=log10_failure,
                qc_failure_log_status=log_status,qc_failure_log_method=log_method,
                qc_evaluation='complement_of_direct_failure' if from_failure else 'direct_pass',
                class_statuses=states[:4],qc_status=states[4],mass_residual=residual,
                qc_thresholds=[ct,cd],rule_id='common125_candidate_v1',
                operation='joint_rechecked_QC_fixed_graph_OTHER_for_failures',
                precision_scope='binary64_nonnegative_DP; no relative tail certificate')
    if include_partition:
        other_pass=math.fsum(values[5:])
        other_possible=any(s!='STRUCTURAL_ZERO' for s in states[5:])
        other_state=_probability_state(other_pass,other_possible)
        # Do not turn an unresolved class/complement underflow into support zero.
        unresolved=any(s=='NUMERICAL_UNDERFLOW' for s in states[:4]) or other_state=='NUMERICAL_UNDERFLOW'
        if unresolved:
            logs=None
        else:
            lo=-math.inf if other_pass==0 else math.log(other_pass)
            lf=-math.inf if log_failure is None else log_failure
            logs=[None if v==0 else math.log(v) for v in classes]
            combined=float(np.logaddexp(lo,lf))
            logs.append(None if combined==-math.inf else combined)
        result.update(p_other_and_qc=other_pass,other_and_qc_status=other_state,
                      categorical_log_probabilities=logs,
                      partition_status='UNRESOLVED_COMPONENT_UNDERFLOW' if unresolved else 'AVAILABLE',
                      partition_basis='direct_disjoint_joint_QC_and_11_marker_classes')
    return result


def candidate_joint_probabilities(counts, genes, baseline_eligible, retention, *, min_total=50, min_detected=20,
                                  include_partition=False):
    """Batch adapter with no node deletion and no silent underflow propagation.

    Returns an n-by-4 matrix compatible with fixed-graph probability APIs plus
    QC probabilities/states. If an attainable class mass underflows, inspect
    the per-cell API; do not pass a zero to a downstream structural classifier.
    """
    x=np.asarray(counts); mask=np.asarray(baseline_eligible)
    if x.ndim!=2 or x.shape[1]!=len(genes) or mask.dtype!=bool or mask.shape!=(len(x),):
        raise ValueError('aligned count matrix, measured genes and fixed boolean mask required')
    # Validate every row, including baseline-ineligible rows, without dropping
    # panel or integer checks. They remain present with structural-zero masses.
    result=[];probabilities=np.zeros((len(x),4));qc=np.zeros(len(x))
    for i,row in enumerate(x):
        record=candidate_joint_qc(row,genes,retention,min_total=min_total,min_detected=min_detected,
                                  include_partition=include_partition)
        if mask[i]:
            if 'NUMERICAL_UNDERFLOW' in record['class_statuses'] or record['qc_status']=='NUMERICAL_UNDERFLOW':
                raise ArithmeticError(f'NUMERICAL_UNDERFLOW at row {i}; retain per-cell support states')
            if math.fsum(record['probabilities'])>1. or record['p_qc']>1.:
                raise ArithmeticError(f'NUMERICAL_DOMAIN_EXCESS at row {i}; no silent normalization')
            probabilities[i]=record['probabilities'];qc[i]=record['p_qc']
        if not mask[i] and include_partition:
            record=dict(record,categorical_log_probabilities=[None,None,None,None,0.],
                        p_other_and_qc=0.,other_and_qc_status='STRUCTURAL_ZERO',partition_status='AVAILABLE')
        result.append(record if mask[i] else dict(record,probabilities=[0.]*4,p_qc=0.,p_qc_failure=1.,
                       qc_failure_status='POSITIVE_FINITE',
                       qc_failure_log=0.,qc_failure_log10=0.,
                       qc_failure_log_status='POSITIVE_LOG_FINITE',qc_failure_log_method='fixed_baseline_ineligible',
                       class_statuses=['STRUCTURAL_ZERO']*4,qc_status='STRUCTURAL_ZERO',
                       reason='FIXED_BASELINE_INELIGIBLE'))
    return dict(probabilities=probabilities,p_qc=qc,cells=result,classes=list(CLASSES),
                operation='joint_rechecked_QC_fixed_graph_OTHER_for_failures')
