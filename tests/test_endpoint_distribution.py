import unittest,itertools,math
from collections import defaultdict
from fractions import Fraction as F
import numpy as np
from spatial_membership_audit.ecosystem import SpatialGraph,build_graph,cross_k
from spatial_membership_audit.endpoint_distribution import (EndpointScale,exact_endpoint_distribution,
    empirical_endpoint_distribution,StateBudgetExceeded,_stub_count_law,_exact_inner,_log2,_summary)

CLASSES=('A','B','OTHER')
def fixture(radius=1.5):
    xy=np.array([[1,1],[1.2,2.1],[2.5,1.3],[3.6,2.2]])
    g=build_graph(xy,list('abcd'),['r']*4,method='radius',radius_um=radius)
    return EndpointScale(str(radius),g,radius_um=radius,xy_um=xy,window=[0,10,0,10],window_provenance='fixed synthetic square'),xy

def exact(p,scales=None,**kwargs):
    return exact_endpoint_distribution(p,scales=scales or [fixture()[0]],classes=CLASSES,pair=('A','B'),
        independent_nodes=kwargs.pop('independent_nodes',True),provenance='engineering fixture',
        max_states=kwargs.pop('max_states',100000),**kwargs)

def record(labels,ids=tuple('abcd')):
    return dict(semantics='hard_labels',provenance='fixture',node_ids=ids,classes=CLASSES,values=labels)

class EndpointDistributionTests(unittest.TestCase):
    def test_fixed_label_observations_and_native_K_definition(self):
        s,xy=fixture(2.8);labels=['A','B','OTHER','B'];out=s.evaluate(labels,('A','B'))
        i,j=s.edges.row,s.edges.col
        self.assertEqual(out['O']['value'],sum({labels[a],labels[b]}=={'A','B'} for a,b in zip(i,j)))
        for corr in ('none','translation'):
            expected=cross_k(xy,labels,('A','B'),[2.8],window=[0,10,0,10],correction=corr)[0]
            self.assertAlmostEqual(out['K_'+corr]['value'],expected['K'])
            self.assertAlmostEqual(out['L_'+corr]['value'],expected['L'])
        self.assertEqual(out['G_a_to_b']['value'],1)

    def test_null_moments_independent_exhaustive_permutations(self):
        for r in (1.5,2.8):
            s,_=fixture(r)
            for a in range(5):
                for b in range(5-a):
                    labels=['A']*a+['B']*b+['OTHER']*(4-a-b)
                    counts=[]
                    for v in set(itertools.permutations(labels)):
                        counts.append(sum({v[i],v[j]}=={'A','B'} for i,j in zip(s.edges.row,s.edges.col)))
                    mu=F(sum(counts),len(counts));var=sum((F(x)-mu)**2 for x in counts)/len(counts)
                    self.assertEqual(s.null(a,b),(mu,var))

    def test_endpoint_stub_null_matches_all_stub_permutations(self):
        for a,b,c in [(1,1,2),(2,2,2),(3,1,2),(0,2,2)]:
            stubs=[0]*a+[1]*b+[2]*c;m=len(stubs)//2;counts=defaultdict(int)
            perms=set(itertools.permutations(stubs))
            for v in perms:counts[sum({v[i],v[i+m]}=={0,1} for i in range(m))]+=1
            self.assertEqual(dict(_stub_count_law(a,b,c)),{k:F(v,len(perms)) for k,v in counts.items()})

    def test_finite_permutation_scores_exact_independent_bruteforce(self):
        s,_=fixture(1.5);labels=['A','B','OTHER','OTHER'];o=s.evaluate(labels,('A','B'))['O']['value']
        node=[]
        for v in set(itertools.permutations(labels)):
            node.append(sum({v[i],v[j]}=={'A','B'} for i,j in zip(s.edges.row,s.edges.col)))
        stubs=[labels[i] for i in s.edges.row]+[labels[j] for j in s.edges.col];stub=[]
        for v in set(itertools.permutations(stubs)):
            stub.append(sum({v[i],v[i+s.m]}=={'A','B'} for i in range(s.m)))
        expected=defaultdict(F)
        for n1,n2 in itertools.product(node,repeat=2):
            mean=F(n1+n2,2);var=F((n1-n2)**2,4)
            z=(o-float(mean))/math.sqrt(float(var)) if var else ('UNDEFINED_0_OVER_0' if o==mean else 'POSITIVE_INFINITY' if o>mean else 'NEGATIVE_INFINITY')
            for t1,t2 in itertools.product(stub,repeat=2):
                enrich=math.log2((o+1)/(1+(t1+t2)/2))
                expected[str(z),round(enrich,12)]+=F(1,len(node)**2*len(stub)**2)
        actual=defaultdict(F)
        for atoms,mass in _exact_inner(s,labels,('A','B'),2,100000):
            z=atoms['z_node_mc'];actual[str(z['value']) if z['value'] is not None else z['status'],round(atoms['giotto_enrichment_mc']['value'],12)]+=mass
        self.assertEqual(actual,expected)

    def test_exact_mass_cdf_and_nonlinearity(self):
        result=exact([[F(1,4),F(1,4),F(1,2)]]*4)
        self.assertEqual(sum(F(x['mass']) for x in result['joint_atoms']),1)
        self.assertEqual(result['positive_outer_states'],81)
        k=result['marginals']['1.5/K_none'];l=result['marginals']['1.5/L_none']
        self.assertLess(l['conditional_mean'],math.sqrt(k['conditional_mean']/math.pi))
        for marginal in result['marginals'].values():
            self.assertEqual(sum(F(v) for v in marginal['status_mass'].values()),1)
            if marginal['atoms']:self.assertEqual(marginal['atoms'][-1]['conditional_cdf'],'1')
        self.assertIsNone(result['marginals']['1.5/giotto_enrichment']['conditional_cv'])

    def test_log_zero_order_is_between_negative_and_positive(self):
        joint=[dict(mass='1/3',endpoints={'r/giotto_enrichment':_log2(q)}) for q in (F(2),F(1),F(1,2))]
        m=_summary(joint,['r/giotto_enrichment'],mode='EXACT_FINITE_STATE',planned=3,independent_outer=None)['marginals']['r/giotto_enrichment']
        self.assertEqual([a['value'] for a in m['atoms']],[-1,0,1])
        self.assertEqual(m['quantiles_given_finite']['1/2']['value'],0)

    def test_no_pruning_no_normalization_no_silent_approximation(self):
        p=F(1,10**350)
        result=exact([[p,1-p,0],[0,1,0],[0,0,1],[0,0,1]])
        self.assertEqual(result['positive_outer_states'],2)
        self.assertTrue(any(F(x['mass'])==p for x in result['joint_atoms']))
        with self.assertRaises(ValueError):exact([[.1,.2,.7]]*4)
        with self.assertRaises(ValueError):exact([[1,0,0]]*4,independent_nodes=False)
        with self.assertRaises(StateBudgetExceeded):exact([[F(1,3)]*3]*4,max_states=80)

    def test_domain_zero_empty_and_absence(self):
        g=SpatialGraph(tuple('abcd'),('r',)*4,np.zeros((4,4)),False,'fixture')
        xy=np.array([[1,1],[2,2],[3,3],[4,4]])
        s=EndpointScale('empty',g,radius_um=.1,xy_um=xy,window=[0,10,0,10],window_provenance='square')
        v=s.evaluate(['A','B','OTHER','OTHER'],('A','B'))
        self.assertEqual(v['R']['status'],'NO_EDGES');self.assertEqual(v['K_none']['status'],'FINITE_ZERO')
        self.assertEqual(v['G_a_to_b']['value'],0)
        self.assertEqual(v['co_occurrence']['status'],'NO_ENDPOINT_SUPPORT')
        self.assertEqual(s.evaluate(['A']*4,('A','B'))['K_none']['status'],'ABSENT_TYPE')

    def test_missing_geometry_not_fabricated(self):
        s,_=fixture();plain=EndpointScale('g',s.graph,radius_um=1.5)
        v=plain.evaluate(['A','B','OTHER','OTHER'],('A','B'))
        self.assertEqual(v['K_none']['status'],'GEOMETRY_NOT_SUPPLIED')
        self.assertEqual(v['G_a_to_b']['value'],1)
        with self.assertRaises(ValueError):EndpointScale('g',s.graph,radius_um=2,xy_um=np.ones((4,2)),window=[0,10,0,10],window_provenance='bad')

    def test_numpy_integer_window_cannot_overflow_fraction(self):
        from spatial_membership_audit.endpoint_distribution import _fraction
        wrapped=F(np.int64(1),np.int64(10**12))
        normalized=_fraction(wrapped)
        self.assertIs(type(normalized.numerator),int)
        self.assertEqual(normalized**4,F(1,10**48))
        mass=exact([[wrapped,1-wrapped,0]]*4)
        self.assertEqual(sum(F(r['mass']) for r in mass['joint_atoms']),1)
        s,xy=fixture();window=np.array([0,500,0,500],dtype=np.int64)
        integer=EndpointScale('integer',s.graph,radius_um=1.5,xy_um=xy,window=window,window_provenance='fixed square')
        floating=EndpointScale('floating',s.graph,radius_um=1.5,xy_um=xy,window=window.astype(float),window_provenance='same square')
        self.assertEqual(integer.translation,floating.translation)
        self.assertEqual(integer.evaluate(['A','B','OTHER','OTHER'],('A','B'))['K_translation'],floating.evaluate(['A','B','OTHER','OTHER'],('A','B'))['K_translation'])

    def test_failure_partition_alignment_and_null_records(self):
        s,_=fixture();outputs={'ok':record(['OTHER','OTHER','B','A'],tuple('dcba')),'bad':record(['A'])}
        result=empirical_endpoint_distribution(outputs=outputs,failed={'f':'native failure'},not_run={'n':'not started'},planned_ids=['ok','bad','f','n'],
            scales=[s],classes=CLASSES,pair=('A','B'),provenance='fixture',null_permutations=2,seed=17)
        m=result['marginals']['1.5/O'];self.assertEqual(m['finite_mass'],'1/4')
        self.assertEqual(result['joint_atoms'][0]['endpoints']['1.5/O']['value'],1)
        self.assertEqual(m['status_mass']['INVALID_OUTPUT'],'1/4');self.assertEqual(m['status_mass']['FIT_FAILED'],'1/4')
        self.assertEqual(len(result['null_count_records']['ok/1.5']['node']),2)
        with self.assertRaises(ValueError):empirical_endpoint_distribution(outputs=outputs,failed={},not_run={},planned_ids=['ok'],scales=[s],classes=CLASSES,pair=('A','B'),provenance='fixture')

    def test_joint_covariance_uses_same_outer_vectors(self):
        s,_=fixture();r=exact([[F(1,4),F(1,4),F(1,2)]]*4,scales=[s,fixture(2.8)[0]])
        c=next(v for v in r['pairwise_covariance'] if v['left']=='1.5/O' and v['right']=='2.8/O')
        self.assertGreater(c['conditional_covariance'],0)
        self.assertEqual(c['joint_finite_mass'],'1')

if __name__=='__main__':unittest.main()
