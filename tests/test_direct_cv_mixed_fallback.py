"""Independent exact moments for mixed-node residual boundary regressions."""
from fractions import Fraction as F
from itertools import product
from math import comb,prod
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from spatial_membership_audit import direct_cv as cv

def graph(n,edges):
    edges=sorted({tuple(sorted(e)) for e in edges});i,j=np.array(edges).T
    return sparse.csr_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=(n,n))

class MixedFallbackRegression(unittest.TestCase):
    def assert_moments(self,result,p,j1,j2):
        self.assertEqual(result['status'],'RESOLVED')
        for key,value in [('first_ratio_moment_interval',j1),('second_ratio_moment_interval',j2)]:
            lo,hi=result[key];self.assertLessEqual(F(lo),value);self.assertGreaterEqual(F(hi),value)
        cv2=p*j2/(j1*j1)-1;lo,hi=result['cv_interval']
        self.assertLessEqual(F(lo)**2,cv2);self.assertGreaterEqual(F(hi)**2,cv2)

    def test_deterministic_and_mixed_nodes_exact_enumeration(self):
        n=64;edges={(i,(i+1)%n) for i in range(n)}|{(0,i) for i in range(60,64)}|{(35,i) for i in range(60,64)}
        edges={tuple(sorted(e)) for e in edges};g=graph(n,edges)
        a=np.r_[np.ones(30),np.zeros(30),np.full(4,.25)]
        b=np.r_[np.zeros(30),np.ones(30),np.full(4,.5)]
        p=j1=j2=F(0)
        for state in product(range(3),repeat=4):
            labels=[0]*30+[1]*30+list(state);w=prod(F(1,2) if s==1 else F(1,4) for s in state)
            o=sum({labels[i],labels[j]}=={0,1} for i,j in edges)
            t=F(o,labels.count(0)*labels.count(1));p+=w;j1+=w*t;j2+=w*t*t
        result=cv.estimate(a,b,g,max_seconds=5)
        self.assert_moments(result,p,j1,j2)
        prepare=cv.prepare_graph_parts
        def force_interval(*args):
            parts=prepare(*args);parts['single_point_safe']=False;return parts
        with patch.object(cv,'prepare_graph_parts',side_effect=force_interval):
            fallback=cv.estimate(a,b,g,max_seconds=5)
        self.assertEqual(fallback['moment_backend'],'interval_active_grid_fallback')
        self.assert_moments(fallback,p,j1,j2)

    def test_all_mixed_zero_residual_cycle_exact_combinatorics(self):
        n=64;m=n;g=graph(n,{(i,(i+1)%n) for i in range(n)})
        p=1-F(2,2**n);j1=j2=F(0);shared=n;disjoint=comb(m,2)-shared
        for k in range(1,n):
            mass=F(comb(n,k),2**n);e1=F(2*k*(n-k),n*(n-1))
            e_shared=e1/2;e_disjoint=F(4*k*(k-1)*(n-k)*(n-k-1),n*(n-1)*(n-2)*(n-3))
            j1+=mass*m*e1/(k*(n-k))
            j2+=mass*(m*e1+2*shared*e_shared+2*disjoint*e_disjoint)/(k*k*(n-k)*(n-k))
        result=cv.estimate(np.full(n,.5),np.full(n,.5),g,max_seconds=5)
        self.assert_moments(result,p,j1,j2)

    def test_near_zero_residual_encloses_exact_binary_inputs(self):
        a=np.array([1.,1.-2.**-40,1.-2.**-52,.5])
        b=np.array([0.,2.**-42,2.**-60,.5])
        lo,hi=cv.residual_probability(a,b)
        for i in range(len(a)):
            exact=F(1)-F(float(a[i]))-F(float(b[i]))
            self.assertLessEqual(F(float(lo[i])),exact)
            self.assertGreaterEqual(F(float(hi[i])),exact)
        self.assertLessEqual(cv.residual_ratio_bound((lo,hi)),1.+1e-7)
        self.assertEqual(cv.residual_ratio_bound((np.array([0.]),np.array([.1]))),np.inf)

    def test_disjoint_support_with_denominator_only_nodes(self):
        # A is deterministic, B count is Binomial(33,.5); only one actual edge.
        # Conditional on B count=k, the edge is present with probability k/33.
        n=64;a=np.r_[np.ones(31),np.zeros(33)];b=np.r_[np.zeros(31),np.full(33,.5)]
        p=1-F(1,2**33);j1=p/(31*33)
        j2=sum((F(comb(33,k),2**33*33*31**2*k) for k in range(1,34)),F(0))
        result=cv.estimate(a,b,graph(n,{(0,31)}),max_seconds=5)
        self.assertEqual(result['moment_backend'],'separable_nonnegative')
        self.assert_moments(result,p,j1,j2)

if __name__=='__main__':unittest.main()
