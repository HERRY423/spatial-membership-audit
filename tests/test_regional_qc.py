import itertools
import math
import unittest
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from spatial_membership_audit.definitions import REQUIRED, candidate_labels
from spatial_membership_audit.candidate_qc import candidate_joint_probabilities
from spatial_membership_audit.regional_qc import regional_joint_qc, summarize_joint_region


def independent_category(y, ct, cd):
    if sum(y)<ct or sum(v>0 for v in y)<cd:return 4
    b=y[0]>0 and (y[1]>0 or y[2]>0)
    t=y[3]>0 and y[4]>0 and y[3]+y[4]>=3
    m=y[7]>0 and y[8]>0
    plasma=y[9]>0 and y[10]>0
    if plasma or int(b)+int(t)+int(m)!=1:return 4
    if b:return 0
    if m:return 3
    cd8=y[5]>=2;cd4=y[6]>=1
    return 1 if cd8 and not cd4 else 2 if cd4 and not cd8 else 4


def molecular_categories(row, ps, ct, cd):
    cat=[F(0)]*5
    for y in itertools.product(*(range(int(n)+1) for n in row)):
        w=F(1)
        for n,k,p in zip(row,y,ps):w*=math.comb(int(n),k)*p**k*(1-p)**(int(n)-k)
        cat[independent_category(y,ct,cd)]+=w
    return cat


def distribution(cats, edges, a, b):
    n=len(cats);c=F(2*len(edges),n*(n-1));finite=mean=second=F(0)
    for labels in itertools.product(range(5),repeat=n):
        w=math.prod(cats[i][v] for i,v in enumerate(labels))
        na=labels.count(a);nb=labels.count(b)
        if not na or not nb:continue
        o=sum({labels[i],labels[j]}=={a,b} for i,j in edges)
        r=F(o)/(c*na*nb);finite+=w;mean+=w*r;second+=w*r*r
    return finite,mean/finite,second/finite


class RegionalQCTests(unittest.TestCase):
    def fixture(self):
        x=np.zeros((4,12),dtype=int)
        x[0,[0,1,7,11]]=1
        x[1,[3,4,5,11]]=[2,1,2,1]
        x[2,[0,1,9,10,11]]=1
        x[3,[7,8,3,4,6,11]]=[1,1,1,2,1,1]
        edges=[(0,1),(1,2),(2,3)]
        g=np.zeros((4,4))
        for i,j in edges:g[i,j]=g[j,i]=1
        return x,list(REQUIRED)+['background'],np.ones(4,dtype=bool),g,edges

    def test_full_molecular_enumeration_scalar_and_vector(self):
        x,genes,mask,g,edges=self.fixture()
        for ps in ([F(1,2)]*12,[F(1,2) if j%2 else F(3,4) for j in range(12)]):
            cats=[molecular_categories(row,ps,3,2) for row in x]
            out=regional_joint_qc(x,genes,mask,np.array(ps,dtype=float),g,pairs=((0,1),(0,3)),
                      min_total=3,min_detected=2,accuracy_factor=1.05,max_seconds_per_pair=3)
            for row,cat in zip(out['cells'],cats):
                self.assertTrue(np.allclose(row['probabilities'],[float(q) for q in cat[:4]],rtol=2e-12,atol=1e-15))
                decoded=[0 if v is None else math.exp(v) for v in row['categorical_log_probabilities']]
                self.assertTrue(np.allclose(decoded,[float(q) for q in cat],rtol=2e-12,atol=1e-15))
            for result,(a,b) in zip(out['pairs'],((0,1),(0,3))):
                f,mean,second=distribution(cats,edges,a,b)
                self.assertAlmostEqual(result['availability']['p_finite_direct'],float(f),places=12)
                self.assertAlmostEqual(result['availability']['p_na'],float(1-f),places=12)
                self.assertEqual(result['status'],'EXACT_ZERO_CV' if second==mean*mean else 'RESOLVED')
                refcv=math.sqrt(float(second-mean*mean))/float(mean)
                for key,val in [('mean_R_interval',float(mean)),('second_moment_R_interval',float(second))]:
                    lo,hi=result[key];self.assertLessEqual(lo,val);self.assertGreaterEqual(hi,val)
                lo,hi=result['direct_cv']['cv_interval'];self.assertLessEqual(lo,refcv);self.assertGreaterEqual(hi,refcv)

    def test_qc_log_tail_reaches_region_without_subtraction(self):
        genes=list(REQUIRED)+['background'];x=np.zeros((2,12),dtype=int)
        x[0,[0,1,11]]=[1,1,1100];x[1,[3,4,5]]=[2,1,2]
        p=np.ones(12);p[11]=.5
        out=regional_joint_qc(x,genes,np.ones(2,dtype=bool),p,[[0,1],[1,0]],
                    min_total=3,min_detected=2,accuracy_factor=1.25)
        r=out['pairs'][0]
        self.assertEqual(r['availability']['na_probability_status'],'NUMERICAL_UNDERFLOW')
        self.assertAlmostEqual(r['availability']['log_p_na'],-1100*math.log(2),places=10)
        self.assertEqual(r['status'],'UNRESOLVED_UPSTREAM_PRECISION')
        self.assertIsNone(r['direct_cv']['cv'])
        empty=regional_joint_qc(x,genes,np.ones(2,dtype=bool),p,[[0,0],[0,0]],
                    min_total=3,min_detected=2,accuracy_factor=1.25)['pairs'][0]
        self.assertEqual(empty['status'],'NO_FINITE_RATIO')
        # Representable tiny probability is also not rounded away by 1-P+.
        x[0,11]=110
        r=regional_joint_qc(x,genes,np.ones(2,dtype=bool),p,[[0,1],[1,0]],
                    min_total=3,min_detected=2,accuracy_factor=1.25)['pairs'][0]
        self.assertEqual(r['availability']['na_probability_status'],'POSITIVE_FINITE')
        self.assertAlmostEqual(r['availability']['p_na']/2.**-110,1.,places=11)

    def test_deterministic_and_fixed_ineligible_and_no_edges(self):
        x,genes,mask,g,edges=self.fixture()
        for p in (0.,1.):
            result=regional_joint_qc(x,genes,mask,p,g,min_total=3,min_detected=2,accuracy_factor=1.25)
            labels=candidate_labels(x*p,genes,mask & ((x*p).sum(1)>=3) & ((x*p>0).sum(1)>=2))
            expected=np.column_stack([labels==c for c in result['classes']]).astype(float)
            np.testing.assert_array_equal(result['probabilities'],expected)
        mask[0]=False
        result=regional_joint_qc(x,genes,mask,.5,g*0,min_total=3,min_detected=2,accuracy_factor=1.25)
        self.assertEqual(result['n'],4);self.assertEqual(result['m'],0)
        self.assertEqual(result['cells'][0]['categorical_log_probabilities'],[None,None,None,None,0.])
        self.assertEqual(result['pairs'][0]['availability']['finite_probability_status'],'STRUCTURAL_ZERO')
        self.assertEqual(result['pairs'][0]['status'],'NO_FINITE_RATIO')

    def test_structural_zero_and_unresolved_components_remain_distinct(self):
        x,genes,mask,g,edges=self.fixture()
        joint=candidate_joint_probabilities(x,genes,mask,.5,min_total=3,min_detected=2,include_partition=True)
        joint['cells'][0]['partition_status']='UNRESOLVED_COMPONENT_UNDERFLOW'
        with self.assertRaisesRegex(ArithmeticError,'UNRESOLVED_COMPONENT_UNDERFLOW'):
            summarize_joint_region(joint,g,accuracy_factor=1.25)
        with self.assertRaisesRegex(ValueError,'missing measured'):
            regional_joint_qc(x[:,:-2],genes[:-2],mask,.5,g,accuracy_factor=1.25)

if __name__=='__main__':unittest.main()
