"""Whole-state pushforwards, including endpoint-specific undefined atoms.

Exact mode has rational probability masses on exhaustive finite label states.
Non-rational endpoint values retain symbolic keys and numeric evaluations;
this is not an interval-arithmetic certificate. Empirical mode represents
complete supplied realizations, never a population-exact distribution.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
import hashlib
import math
import numpy as np
from scipy import sparse
from .ecosystem import SpatialGraph, _ids, ripley_weights

METRICS = ('O', 'E', 'R', 'z_node', 'z_node_mc', 'co_occurrence',
           'giotto_enrichment', 'giotto_enrichment_mc', 'G_a_to_b', 'G_b_to_a',
           'K_none', 'L_none', 'L_minus_r_none',
           'K_translation', 'L_translation', 'L_minus_r_translation')
SIGNED = {'z_node', 'z_node_mc', 'giotto_enrichment', 'giotto_enrichment_mc',
          'L_minus_r_none', 'L_minus_r_translation'}


class StateBudgetExceeded(RuntimeError):
    """No distribution is returned from an incomplete positive-mass prefix."""


def _fraction(x):
    if isinstance(x, (bool, np.bool_)):
        raise ValueError('Boolean is not a numeric mass')
    if isinstance(x,np.integer):return F(int(x))
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)): raise ValueError('finite number required')
        return F(float(x))  # Exact supplied binary value, without normalization.
    q=F(x)
    # Fraction(np.int64(...), ...) can retain fixed-width NumPy payloads.
    # Canonicalize supplied rational objects as well as bare NumPy integers.
    return F(int(q.numerator),int(q.denominator))


def _missing(reason):
    return dict(value=None, status=reason, key=reason, order=None)


def _atom(value, key, order):
    if not math.isfinite(value):
        raise ArithmeticError('finite symbolic endpoint overflowed numerically')
    return dict(value=float(value), status='FINITE_ZERO' if value == 0 else
                'FINITE_POSITIVE' if value > 0 else 'FINITE_NEGATIVE', key=key, order=str(order))


def _rational(q):
    q=F(q)
    if q and not float(q):raise ArithmeticError('nonzero rational endpoint underflow')
    return _atom(float(q),'q:'+str(q),q)


def _z(observed, mean, variance, *, zero_reason='ZERO_NULL_VARIANCE', native=False):
    d=F(observed)-mean
    if not variance:
        if native and d:
            return dict(value=None,status='POSITIVE_INFINITY' if d>0 else 'NEGATIVE_INFINITY',
                        key='+inf' if d>0 else '-inf',order=None)
        return _missing('UNDEFINED_0_OVER_0' if native else zero_reason)
    if variance<0:raise ArithmeticError('negative exact null variance')
    if not d:return _rational(0)
    q=d*d/variance;sign=1 if d>0 else -1
    return _atom(sign*math.sqrt(float(q)),f'{sign}sqrt({q})',sign*q)


def _log2(q):
    q=F(q)
    if q<=0:raise ValueError('positive log ratio required')
    # Every log atom orders by its argument, including log2(1)=0.
    if q==1:return _atom(0,'log2(1)',q)
    value=math.log1p(float(q-1))/math.log(2) if F(1,2)<q<F(2) else math.log2(q.numerator)-math.log2(q.denominator)
    if not value:raise ArithmeticError('nonzero log endpoint lost numerically')
    return _atom(value,f'log2({q})',q)


def _L(k, radius=F(0)):
    k=F(k);radius=F(radius)
    # Order is exact because sqrt(K/pi)-r is monotone in K at a fixed radius.
    if k==0:
        return _atom(-float(radius),f'sqrt(0/pi)-{radius}',k)
    return _atom(math.sqrt(float(k)/math.pi)-float(radius),f'sqrt({k}/pi)-{radius}',k)


class EndpointScale:
    """One fixed binary graph, optionally verified against a rectangular window.

Supplying radius without coordinates attests a saved radius-graph interpretation
for G only. K/L require explicit coordinates, window, units and provenance.
"""
    def __init__(self, name, graph, *, radius_um=None, xy_um=None, window=None,
                 window_provenance=None):
        if not isinstance(name,str) or not name.strip():raise ValueError('scale name required')
        if not isinstance(graph,SpatialGraph) or not graph.cell_ids or graph.directed or len(set(graph.regions))!=1:
            raise ValueError('nonempty single-region fixed undirected graph required')
        g=sparse.csr_matrix(graph.adjacency,dtype=np.int64).copy()
        if np.any(graph.adjacency.data!=1) or np.any(g.diagonal()) or (g!=g.T).nnz:
            raise ValueError('binary symmetric loop-free graph required')
        self.name=name;self.graph=graph;self.g=g;self.n=len(graph.cell_ids);self.m=g.nnz//2
        self.degree=np.diff(g.indptr);self.shared=sum(math.comb(int(d),2) for d in self.degree)
        self.edges=sparse.triu(g,k=1,format='coo');self.radius=radius_um
        self.area=None;self.translation=None;self.geometry='NOT_SUPPLIED';self._null_cache={}
        if radius_um is not None:
            if isinstance(radius_um,bool) or not math.isfinite(radius_um) or radius_um<0 or graph.coordinate_unit!='um':
                raise ValueError('nonnegative micrometre radius required')
            self.geometry='CALLER_ATTESTED_SAVED_RADIUS_GRAPH'
        if any(x is not None for x in (xy_um,window,window_provenance)):
            if any(x is None for x in (xy_um,window,window_provenance,radius_um)) or not window_provenance.strip():
                raise ValueError('coordinates, radius, explicit window and window provenance required together')
            xy=np.asarray(xy_um,dtype=float)
            if xy.shape!=(self.n,2) or len(np.unique(xy,axis=0))!=self.n:
                raise ValueError('aligned noncoincident coordinates required')
            check,_=ripley_weights(xy,radius_um,window=window,correction='none')
            if (check!=g).nnz:raise ValueError('radius graph differs from explicit coordinates')
            lo,hi,bot,top=map(_fraction,window);width,height=hi-lo,top-bot
            self.area=width*height
            coords=[[F(float(v)) for v in row] for row in xy]
            self.translation=[]
            for i,j in zip(self.edges.row,self.edges.col):
                dx=abs(coords[i][0]-coords[j][0]);dy=abs(coords[i][1]-coords[j][1])
                overlap=(width-dx)*(height-dy)
                if overlap<=0:raise ValueError('undefined translation overlap')
                self.translation.append(self.area/overlap)
            self.geometry='VERIFIED_EXPLICIT_RECTANGULAR_ANALYSIS_WINDOW'
            self.window_provenance=window_provenance

    def null(self,a,b):
        key=(a,b)
        if key not in self._null_cache:
            n,m,s=self.n,self.m,self.shared;d=math.comb(m,2)-s
            q2=F(2*a*b,n*(n-1)) if n>=2 else F(0)
            q3=F(a*b*(a+b-2),n*(n-1)*(n-2)) if n>=3 else F(0)
            q4=F(4*a*(a-1)*b*(b-1),n*(n-1)*(n-2)*(n-3)) if n>=4 else F(0)
            mean=m*q2;var=mean+2*s*q3+2*d*q4-mean*mean
            if var<0:raise ArithmeticError('negative node-null variance')
            self._null_cache[key]=(mean,var)
        return self._null_cache[key]

    def evaluate(self, labels, pair, *, null_counts=None):
        if len(labels)!=self.n or len(pair)!=2 or pair[0]==pair[1]:raise ValueError('aligned labels and distinct pair required')
        lab=np.asarray(labels);a=(lab==pair[0]);b=(lab==pair[1]);na,nb=int(a.sum()),int(b.sum())
        i,j=self.edges.row,self.edges.col;cross=(a[i]&b[j])|(b[i]&a[j]);o=int(cross.sum())
        da,db=int(self.degree[a].sum()),int(self.degree[b].sum());mu,var=self.null(na,nb)
        missing='ABSENT_TYPE' if not na or not nb else 'NO_EDGES'
        giotto_mu=F(da*db,2*self.m-1) if self.m else F(0)
        out=dict(O=_rational(o),E=_rational(mu),R=_rational(F(o)/mu) if mu else _missing(missing),
            z_node=_z(o,mu,var,zero_reason=missing if not mu else 'ZERO_NULL_VARIANCE'),
            co_occurrence=_rational(F(2*self.m*o,da*db)) if da*db else _missing('NO_ENDPOINT_SUPPORT'),
            giotto_enrichment=_log2(F(o+1)/(giotto_mu+1)) if da*db else _missing('NO_ENDPOINT_SUPPORT'),
            z_node_mc=_missing('NOT_REQUESTED'),giotto_enrichment_mc=_missing('NOT_REQUESTED'))
        if self.radius is None:
            out.update(G_a_to_b=_missing('RADIUS_NOT_DECLARED'),G_b_to_a=_missing('RADIUS_NOT_DECLARED'))
        elif not na or not nb:
            out.update(G_a_to_b=_missing('ABSENT_TYPE'),G_b_to_a=_missing('ABSENT_TYPE'))
        else:
            out['G_a_to_b']=_rational(F(int(np.count_nonzero(a&(self.g@b.astype(np.int64)>0))),na))
            out['G_b_to_a']=_rational(F(int(np.count_nonzero(b&(self.g@a.astype(np.int64)>0))),nb))
        for correction in ('none','translation'):
            if self.area is None:reason='GEOMETRY_NOT_SUPPLIED'
            elif not na or not nb:reason='ABSENT_TYPE'
            else:reason=None
            if reason:
                for metric in ('K','L','L_minus_r'):out[f'{metric}_{correction}']=_missing(reason)
            else:
                t=F(o) if correction=='none' else sum((w for w,c in zip(self.translation,cross) if c),F(0))
                k=self.area*t/(na*nb)
                out[f'K_{correction}']=_rational(k)
                out[f'L_{correction}']=_L(k)
                out[f'L_minus_r_{correction}']=_L(k,_fraction(self.radius))
        if null_counts is not None:
            for key in ('node','stub'):
                v=null_counts[key]
                if len(v)<2 or any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or not 0<=x<=self.m for x in v):
                    raise ValueError('at least two valid integer null counts required')
            v=list(map(int,null_counts['node']));mean=F(sum(v),len(v));variance=F(sum(x*x for x in v),len(v))-mean*mean
            out['z_node_mc']=_z(o,mean,variance,native=True)
            if da*db:out['giotto_enrichment_mc']=_log2(F(o+1)/(F(sum(null_counts['stub']),len(null_counts['stub']))+1))
            else:out['giotto_enrichment_mc']=_missing('NO_ENDPOINT_SUPPORT')
        return out

    def simulate_null_counts(self, labels, pair, *, permutations, seed):
        if isinstance(permutations,bool) or not isinstance(permutations,int) or permutations<2:
            raise ValueError('integer permutations >=2 required')
        lab=np.asarray(labels);codes=np.where(lab==pair[0],0,np.where(lab==pair[1],1,2))
        i,j=self.edges.row,self.edges.col;m=self.m
        if m==0 or not np.any(codes==0) or not np.any(codes==1):
            return dict(node=[0]*permutations,stub=[0]*permutations)
        node_rng=np.random.default_rng(seed);stub_rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
        stubs=np.r_[codes[i],codes[j]];node=[];stub=[]
        for _ in range(permutations):
            v=node_rng.permutation(codes)
            node.append(int(np.count_nonzero(((v[i]==0)&(v[j]==1))|((v[i]==1)&(v[j]==0)))))
            v=stub_rng.permutation(stubs);left,right=v[:m],v[m:]
            stub.append(int(np.count_nonzero(((left==0)&(right==1))|((left==1)&(right==0)))))
        return dict(node=node,stub=stub)


def _count_convolution(law, draws, *, square, max_states):
    out={(0,0):F(1)}
    for _ in range(draws):
        nxt=defaultdict(F)
        for (s,q),mass in out.items():
            for value,p in law.items():nxt[s+value,q+value*value if square else 0]+=mass*p
        if len(nxt)>max_states:raise StateBudgetExceeded('inner null convolution state budget exceeded')
        out=dict(nxt)
    return out


@lru_cache(maxsize=4096)
def _stub_count_law(a,b,c):
    total=a+b+c
    if not total:return ((0,F(1)),)
    counts=[a,b,c];first=next(i for i,x in enumerate(counts) if x);counts[first]-=1
    result=defaultdict(F)
    for second,count in enumerate(counts):
        if not count:continue
        child=counts.copy();child[second]-=1;inc=int({first,second}=={0,1})
        for value,mass in _stub_count_law(*child):result[value+inc]+=F(count,total-1)*mass
    return tuple(sorted(result.items()))


def _exact_inner(scale,labels,pair,permutations,max_states):
    n=scale.n;a=labels.count(pair[0]);b=labels.count(pair[1]);total=math.comb(n,a)*math.comb(n-a,b)
    if total>max_states or 2*scale.m>30:
        raise StateBudgetExceeded('exact inner null too large; use explicitly empirical mode')
    law=defaultdict(F);universe=set(range(n));i,j=scale.edges.row,scale.edges.col
    for aa in combinations(range(n),a):
        for bb in combinations(sorted(universe-set(aa)),b):
            v=np.full(n,2);v[list(aa)]=0;v[list(bb)]=1
            o=int(np.count_nonzero(((v[i]==0)&(v[j]==1))|((v[i]==1)&(v[j]==0))))
            law[o]+=F(1,total)
    lab=np.asarray(labels);da=int(scale.degree[lab==pair[0]].sum());db=int(scale.degree[lab==pair[1]].sum())
    node=_count_convolution(law,permutations,square=True,max_states=max_states)
    stub=_count_convolution(dict(_stub_count_law(da,db,2*scale.m-da-db)),permutations,square=False,max_states=max_states)
    base=scale.evaluate(labels,pair);observed=int(base['O']['value']);out=[]
    for (ns,nq),np_ in node.items():
        mean=F(ns,permutations);var=F(nq,permutations)-mean*mean
        z=_z(observed,mean,var,native=True)
        for (ss,_),sp in stub.items():
            enrich=_log2(F(observed+1)/(F(ss,permutations)+1)) if da*db else _missing('NO_ENDPOINT_SUPPORT')
            out.append(({'z_node_mc':z,'giotto_enrichment_mc':enrich},np_*sp))
            if len(out)>max_states:raise StateBudgetExceeded('inner joint atom budget exceeded')
    return out


def _plan(scales,classes,pair):
    scales=tuple(scales);classes=_ids(classes,'classes')
    if not scales or len({s.name for s in scales})!=len(scales):raise ValueError('unique nonempty scale list required')
    if any(s.graph.cell_ids!=scales[0].graph.cell_ids for s in scales):raise ValueError('shared fixed node order across scales required')
    if len(pair)!=2 or pair[0]==pair[1] or not set(pair)<=set(classes):raise ValueError('distinct declared pair required')
    return scales,classes


def _labels(record,ids,classes):
    if record.get('semantics')!='hard_labels' or not record.get('provenance'):
        raise ValueError('explicit hard-label semantics and source required')
    nodes=_ids(record['node_ids'],'native nodes');cats=_ids(record['classes'],'native classes')
    if set(nodes)!=set(ids) or set(cats)!=set(classes):raise ValueError('complete fixed nodes and classes required')
    labels=record['values']
    if len(labels)!=len(nodes) or any(not isinstance(x,str) or x not in classes for x in labels):raise ValueError('one declared label per node required')
    lookup=dict(zip(nodes,labels));return [lookup[x] for x in ids]


def _flatten(scale,atoms):return {f'{scale.name}/{metric}':atoms[metric] for metric in METRICS}


def _summary(joint, dimensions, *, mode, planned, independent_outer):
    """Retain exact masses, full sub-CDFs and endpoint-specific conditioning."""
    total=sum((F(r['mass']) for r in joint),F(0))
    if total!=1:raise ArithmeticError('joint probability mass must be exactly one')
    marginals={}
    for dim in dimensions:
        atoms={};states=defaultdict(F)
        for row in joint:
            a=row['endpoints'][dim];p=F(row['mass']);key=a['key']
            if key not in atoms:atoms[key]=dict(a,mass=F(0))
            atoms[key]['mass']+=p;states[a['status']]+=p
        finite=[a for a in atoms.values() if a['value'] is not None]
        # Each dimension has a fixed monotone rational order, even for sqrt/log.
        finite.sort(key=lambda a:F(a['order']))
        fm=sum((a['mass'] for a in finite),F(0));cdf=F(0);table=[]
        for a in finite:
            cdf+=a['mass'];table.append(dict(a,mass=str(a['mass']),subcdf=str(cdf),conditional_cdf=str(cdf/fm)))
        mean=math.fsum(float(a['mass']/fm)*a['value'] for a in finite) if fm else None
        variance=math.fsum(float(a['mass']/fm)*(a['value']-mean)**2 for a in finite) if fm else None
        quantiles={}
        for q in (F(0),F(1,40),F(1,2),F(39,40),F(1)):
            chosen=next((a for a in table if F(a['conditional_cdf'])>=q),None)
            quantiles[str(q)]=None if chosen is None else {'value':chosen['value'],'key':chosen['key']}
        metric=dim.rsplit('/',1)[1];signed=metric in SIGNED
        marginals[dim]=dict(atoms=table,nonfinite_atoms=[dict(a,mass=str(a['mass'])) for a in atoms.values() if a['value'] is None],
            status_mass={s:str(p) for s,p in states.items()},finite_mass=str(fm),
            finite_mass_numeric=float(fm),conditional_mean=mean,conditional_variance=variance,
            conditional_sd=math.sqrt(variance) if variance is not None else None,
            conditional_cv=math.sqrt(variance)/mean if not signed and mean is not None and mean>0 else None,
            cv_status='NOT_DEFINED_FOR_SIGNED_SCORE' if signed else 'DEFINED' if mean is not None and mean>0 else 'NO_POSITIVE_FINITE_MEAN',
            quantiles_given_finite=quantiles,
            moment_scope='numeric moments of the declared law; not interval certified',
            cdf_scope='finite sub-CDF and finite-conditional CDF; undefined/infinite atoms separate')
    covariance=[]
    for k,left in enumerate(dimensions):
        for right in dimensions[k:]:
            valid=[r for r in joint if r['endpoints'][left]['value'] is not None and r['endpoints'][right]['value'] is not None]
            mass=sum((F(r['mass']) for r in valid),F(0))
            if mass:
                weights=[float(F(r['mass'])/mass) for r in valid]
                x=[r['endpoints'][left]['value'] for r in valid];y=[r['endpoints'][right]['value'] for r in valid]
                mx=math.fsum(w*v for w,v in zip(weights,x));my=math.fsum(w*v for w,v in zip(weights,y))
                cov=math.fsum(w*(u-mx)*(v-my) for w,u,v in zip(weights,x,y))
            else:cov=None
            covariance.append(dict(left=left,right=right,joint_finite_mass=str(mass),conditional_covariance=cov,
                conditioning='both endpoints finite; may differ between pairs'))
    return dict(mode=mode,joint_atoms=joint,marginals=marginals,pairwise_covariance=covariance,
        total_mass='1',planned_outer_realizations=planned,independent_outer_realizations=independent_outer,
        conditioning='endpoint-specific finite domains; no common-success filtering',
        interval_coverage='NOT_ESTABLISHED',biological_replication='NOT_ADDED',
        exactness='exact rational state masses; symbolic endpoints with floating evaluation' if mode=='EXACT_FINITE_STATE' else
                  'complete empirical measure of supplied records; population distribution is approximate')


def exact_endpoint_distribution(probabilities, *, scales, classes, pair,
                                independent_nodes, provenance, max_states,
                                null_permutations=None):
    """Exhaustive independent-label pushforward, with optional exact inner noise.

Rows must sum EXACTLY to one as rationals. Binary64 inputs retain their exact
binary values. No clipping, renormalization, positive-mass pruning or fallback.
Inner finite-permutation streams are independent across tools/scales given the
outer label state. State limits are computational, not statistical truncation.
"""
    scales,classes=_plan(scales,classes,pair)
    if independent_nodes is not True or not provenance:raise ValueError('explicit independent-node assumption and provenance required')
    if isinstance(max_states,bool) or not isinstance(max_states,int) or max_states<1:raise ValueError('positive integer state cap required')
    if null_permutations is not None and (isinstance(null_permutations,bool) or not isinstance(null_permutations,int) or null_permutations<2):
        raise ValueError('at least two inner permutations required')
    p=[[ _fraction(v) for v in row] for row in probabilities]
    if len(p)!=scales[0].n or any(len(row)!=len(classes) or any(v<0 or v>1 for v in row) or sum(row)!=1 for row in p):
        raise ValueError('full exact categorical rows required; no normalization')
    support=[[(classes[j],v) for j,v in enumerate(row) if v>0] for row in p]
    count=math.prod(map(len,support))
    if count>max_states:raise StateBudgetExceeded(f'{count} positive outer states exceed {max_states}')
    dimensions=[f'{s.name}/{m}' for s in scales for m in METRICS];combined={};raw_atoms=0
    for state in product(*support):
        labels=[x[0] for x in state];mass=math.prod(x[1] for x in state);base={};branches=[({},F(1))]
        for scale in scales:
            base.update(_flatten(scale,scale.evaluate(labels,pair)))
            if null_permutations is not None:
                inner=_exact_inner(scale,labels,pair,null_permutations,max_states)
                if len(branches)*len(inner)>max_states:raise StateBudgetExceeded('multiscale inner joint state limit exceeded')
                branches=[(dict(prev,**{f'{scale.name}/{k}':v for k,v in vals.items()}),w*q) for prev,w in branches for vals,q in inner]
                if len(branches)>max_states:raise StateBudgetExceeded('multiscale inner joint state limit exceeded')
        for override,inner_mass in branches:
            raw_atoms+=1
            if raw_atoms>max_states:raise StateBudgetExceeded('complete joint enumeration exceeded state cap; no partial law returned')
            vector=dict(base,**override);key=tuple(vector[d]['key'] for d in dimensions)
            if key not in combined:combined[key]={'endpoints':vector,'mass':F(0)}
            combined[key]['mass']+=mass*inner_mass
    joint=[dict(r,mass=str(r['mass'])) for r in combined.values()]
    result=_summary(joint,dimensions,mode='EXACT_FINITE_STATE',planned=count,independent_outer=None)
    result.update(provenance=provenance,positive_outer_states=count,enumerated_joint_records=raw_atoms,
        unique_joint_atoms=len(joint),null_permutations=null_permutations,
        inner_dependence='independent tool/scale permutation streams conditional on common complete labels')
    return result


def empirical_endpoint_distribution(*, outputs, failed, not_run, planned_ids,
                                    scales, classes, pair, provenance,
                                    null_permutations=None, seed=None):
    """One whole realized label vector per planned outer perturbation.

All endpoints/scales are recomputed together. Optional finite-null simulations
use fresh deterministic, documented streams for each outer ID/tool/scale. They
are algorithm draws, never extra outer perturbations or biological replicates.
"""
    scales,classes=_plan(scales,classes,pair);ids=_ids(planned_ids,'planned outer IDs')
    if not ids or not provenance:raise ValueError('nonempty schedule and provenance required')
    sets=[set(d) for d in (outputs,failed,not_run)]
    if set.union(*sets)!=set(ids) or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('outputs, failures and not-run IDs must partition schedule')
    if any(not isinstance(v,str) or not v.strip() for d in (failed,not_run) for v in d.values()):raise ValueError('failure reasons required')
    if null_permutations is not None and (isinstance(seed,bool) or not isinstance(seed,int) or seed<0):raise ValueError('explicit nonnegative seed required')
    dimensions=[f'{s.name}/{m}' for s in scales for m in METRICS];joint=[];null_records={}
    for rid in ids:
        reason=failed.get(rid,not_run.get(rid));state='FIT_FAILED' if rid in failed else 'NOT_RUN' if rid in not_run else None
        vector={}
        if state is None:
            try:labels=_labels(outputs[rid],scales[0].graph.cell_ids,classes)
            except (KeyError,ValueError,TypeError) as e:state='INVALID_OUTPUT';reason=str(e)
        if state is not None:
            vector={d:_missing(state) for d in dimensions}
        else:
            for scale in scales:
                counts=None
                if null_permutations is not None:
                    key=f'{seed}|{rid}|{scale.name}'
                    local_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                    counts=scale.simulate_null_counts(labels,pair,permutations=null_permutations,seed=local_seed)
                    null_records[f'{rid}/{scale.name}']=dict(seed=local_seed,**counts)
                vector.update(_flatten(scale,scale.evaluate(labels,pair,null_counts=counts)))
        joint.append(dict(realization_id=rid,mass=str(F(1,len(ids))),endpoints=vector,execution_reason=reason))
    result=_summary(joint,dimensions,mode='EMPIRICAL_WHOLE_REALIZATIONS',planned=len(ids),independent_outer=None)
    result.update(provenance=provenance,null_permutations=null_permutations,null_count_records=null_records,
        outer_samples=len(ids),supplied=len(outputs),failed=len(failed),not_run=len(not_run),
        sampling_precision='no iid claim inferred from filenames; no automatic confidence bands',
        inner_dependence='independent tool/scale streams conditional on each complete outer label vector')
    return result
