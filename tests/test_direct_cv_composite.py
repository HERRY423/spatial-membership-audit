"""Regression for composite quadrature and its positive-sum error bound."""
from fractions import Fraction
import unittest
import numpy as np
from scipy import sparse
from spatial_membership_audit import direct_cv as cv


class CompositeCVRegression(unittest.TestCase):
    def test_float_monomials_on_unequal_composite_rules(self):
        for support in [2,5,33,222]:
            x,w1,w2,bounds=cv.rule(4,support)
            for degree in sorted({0,1,support//2,support-1}):
                for power,weights in enumerate([w1,w2],1):
                    exact=sum((Fraction(float(w))*Fraction(float(t))**degree for t,w in zip(x,weights)),Fraction(0))*(degree+1)**power
                    self.assertLessEqual(Fraction(bounds[power-1][0]),exact)
                    self.assertGreaterEqual(Fraction(bounds[power-1][1]),exact)
        self.assertNotEqual(len(cv.rule(4,5)[0]),len(cv.rule(4,2)[0]))

    def test_sum_encloses_small_terms_and_subnormals(self):
        for values in [np.r_[1.,np.full(2051,2.**-53)],np.full(2051,np.nextafter(0.,1.)),np.zeros(65)]:
            exact=sum((Fraction(float(v)) for v in values),Fraction(0))
            lo,hi=cv.reduce_interval(cv.point(values))
            self.assertLessEqual(Fraction(float(lo)),exact)
            self.assertGreaterEqual(Fraction(float(hi)),exact)

    def test_rectangular_tensor_has_exact_ratio_moments(self):
        # Five independent A candidates, one always-B vertex, one actual edge.
        # n_A=k>0: O is Bernoulli(k/5), T=O/k; p=31/32,
        # J1=p/5, J2=sum_{k=1}^5 binom(5,k)/(32*5*k).
        from math import comb
        a=np.r_[np.full(5,.5),0.];b=np.r_[np.zeros(5),1.]
        g=sparse.csr_matrix(([1.,1.],([0,5],[5,0])),shape=(6,6))
        p=Fraction(31,32);j1=p/5
        j2=sum((Fraction(comb(5,k),32*5*k) for k in range(1,6)),Fraction(0))
        result=cv.estimate(a,b,g,accuracy_factor=1.25)
        self.assertEqual(result['status'],'RESOLVED')
        for key,value in [('first_ratio_moment_interval',j1),('second_ratio_moment_interval',j2)]:
            lo,hi=result[key]
            self.assertLessEqual(Fraction(lo),value);self.assertGreaterEqual(Fraction(hi),value)
        lo,hi=result['cv_interval'];cv2=p*j2/(j1*j1)-1
        self.assertLessEqual(Fraction(lo)**2,cv2);self.assertGreaterEqual(Fraction(hi)**2,cv2)


if __name__=='__main__':unittest.main()
