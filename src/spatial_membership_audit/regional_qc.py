"""Fixed-graph O/E, direct NA tails, and conditional CV after joint marker/QC.

The full measured panel and frozen baseline eligibility are mandatory. No node
is removed. Log availability retains QC failures hidden by rounded class-one
probabilities; CV enclosures only cover the supplied binary64 class masses.
"""
import math
import numpy as np
from scipy import sparse
from .analytic import CLASSES, pair_moments
from .availability import availability_from_log_states
from .candidate_qc import candidate_joint_probabilities
from .primary_cv import estimate_cv
from .direct_cv import point, mul, div, down, up


def regional_joint_qc(counts, genes, baseline_eligible, retention, adjacency, *,
                      pairs=((0,1),), min_total=50, min_detected=20,
                      accuracy_factor, max_seconds_per_pair=6.):
    """Return cell evidence and region summaries; caller must impose a hard cap."""
    joint=candidate_joint_probabilities(counts,genes,baseline_eligible,retention,
              min_total=min_total,min_detected=min_detected,include_partition=True)
    return summarize_joint_region(joint,adjacency,pairs=pairs,
              accuracy_factor=accuracy_factor,max_seconds_per_pair=max_seconds_per_pair)


def summarize_joint_region(joint, adjacency, *, pairs=((0,1),), accuracy_factor,
                           max_seconds_per_pair=6.):
    """Reuse one cell probability calculation across prespecified pairs.

    mean_R_interval is E[O/E | E>0], not E[O]/E[E]. Intervals and CV refer to
    exact binary64 input class probabilities, with separate raw-count log tails.
    Hidden complement mass triggers an explicit upstream-precision guard.
    """
    p=np.asarray(joint['probabilities'],dtype=float)
    if p.ndim!=2 or p.shape[1]!=4 or len(joint['cells'])!=len(p):
        raise ValueError('aligned full-rule joint cell records required')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or any(math.fsum([1.,*(-float(v) for v in row)])<0 for row in p):
        raise ArithmeticError('NUMERICAL_DOMAIN_EXCESS: no normalization permitted')
    logs=[]
    for i,record in enumerate(joint['cells']):
        if record.get('partition_status')!='AVAILABLE' or record.get('categorical_log_probabilities') is None:
            raise ArithmeticError(f'UNRESOLVED_COMPONENT_UNDERFLOW at row {i}')
        if not np.array_equal(p[i],record['probabilities']):
            raise ValueError('cell records and probabilities disagree')
        logs.append([-math.inf if v is None else v for v in record['categorical_log_probabilities']])
    logs=np.asarray(logs,dtype=float).reshape((-1,5))
    if np.isnan(logs).any() or np.isposinf(logs).any() or np.any(np.abs(np.logaddexp.reduce(logs,axis=1))>2e-12):
        raise ArithmeticError('direct categorical partition is not normalized within tolerance')
    if not np.allclose(np.exp(logs[:,:4]),p,rtol=2e-12,atol=0.):
        raise ValueError('log class probabilities disagree with supplied masses')
    g=sparse.csr_matrix(adjacency,dtype=float);g.sum_duplicates();g.eliminate_zeros()
    if g.shape!=(len(p),len(p)) or np.any(g.data!=1) or np.any(g.diagonal()!=0) or (g!=g.T).nnz:
        raise ValueError('binary symmetric fixed graph without self edges required')
    pairs=list(pairs)
    if any(len(pair)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in pair)
           or pair[0]==pair[1] or min(pair)<0 or max(pair)>3 for pair in pairs):
        raise ValueError('distinct full-rule class indices in 0..3 required')
    n=len(p);m=g.nnz//2;results=[]
    for a,b in pairs:
        neither=np.logaddexp.reduce(logs[:,[j for j in range(5) if j not in (a,b)]],axis=1)
        tails=availability_from_log_states(np.column_stack([neither,logs[:,a],logs[:,b]]),
                    has_edges=m>0,basis='direct_disjoint_11_marker_joint_QC_with_log_failure')
        raw=pair_moments(p,g,a=a,b=b)
        # A finite direct complement can be invisible to a rounded unit pair
        # mass. Do not certify raw-count constant CV from that rounded model.
        hidden=[i for i in range(n) if math.fsum([1.,-float(p[i,a]),-float(p[i,b])])==0 and math.isfinite(neither[i])]
        if hidden and tails['finite_probability_status']!='STRUCTURAL_ZERO':
            cv=dict(status='UNRESOLVED_UPSTREAM_PRECISION',cv=None,cv_interval=None,
                    reason='positive direct complement hidden by unit binary64 pair mass',
                    affected_rows=hidden,fallback_used=False,
                    accuracy_scope='raw-count CV not certified after upstream rounding loss')
        else:
            cv=estimate_cv(p,g,a=a,b=b,accuracy_factor=accuracy_factor,max_seconds=max_seconds_per_pair)
        mean=second=variance=None
        status=cv['status']
        if 'first_ratio_moment_interval' in cv and cv['p_finite_interval'][0]>0:
            scale=div(point(float(n*(n-1))),point(float(2*m)))
            mean=mul(scale,div(cv['first_ratio_moment_interval'],cv['p_finite_interval']))
            second=mul(mul(scale,scale),div(cv['second_ratio_moment_interval'],cv['p_finite_interval']))
            squared=mul(mean,mean)
            variance=(down(second[0]-squared[1]),up(second[1]-squared[0]))
            if variance[1]<0:raise ArithmeticError('inconsistent conditional variance enclosure')
        elif status in ('EXACT_ZERO_CV','ZERO_NUMERATOR_CV_UNDEFINED'):
            # Compute the constant from graph integers with outward operations,
            # not from already rounded raw moment ratios.
            if status=='ZERO_NUMERATOR_CV_UNDEFINED':
                mean=(0.,0.)
            else:
                mean=div(point(float(n*(n-1))),point(float(2*m)))
                if np.all((p[:,a]==0)|(p[:,a]==1)) and np.all((p[:,b]==0)|(p[:,b]==1)):
                    na=int(p[:,a].sum());nb=int(p[:,b].sum())
                    observed=int(p[:,a] @ (g @ p[:,b]))
                    mean=mul(mean,div(point(float(observed)),point(float(na*nb))))
            second=mul(mean,mean);variance=(0.,0.)
        fields=lambda z:None if z is None else [float(z[0]),float(z[1])]
        results.append(dict(class_indices=[int(a),int(b)],classes=[CLASSES[a],CLASSES[b]],
                     availability=tails,direct_cv=cv,mean_R_interval=fields(mean),
                     second_moment_R_interval=fields(second),variance_R_interval=fields(variance),
                     ratio_of_means=raw['ratio_of_means'],raw_moments={k:raw[k] for k in ('mu_O','mu_E','var_O','var_E','cov_O_E')},
                     numerical_scope='moments/CV condition on binary64 class masses; log availability retains direct QC complement; upstream error not certified',
                     mean_target='E[O/E | E>0]',status=status))
    return dict(n=n,m=m,pairs=results,cells=joint['cells'],
                probabilities=p,classes=list(CLASSES),
                operation='11_marker_joint_rechecked_QC_fixed_graph_OTHER_for_failures')
