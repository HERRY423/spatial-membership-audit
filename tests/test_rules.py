import unittest
from fractions import Fraction as F
import numpy as np
from spatial_membership_audit.rules import stromal_myeloid_probabilities

class FixedRuleTests(unittest.TestCase):
    def test_stable_against_exact_rational_including_high_counts(self):
        counts=np.array([[0,0,0,0],[1,1,0,0],[2,2,2,2],[100,100,100,100],[100,1,1,100]])
        for p in (F(1,4),F(1,2),F(3,4)):
            expected=[]
            for row in counts:
                q=[1-(1-p)**int(v) for v in row];f=q[0]*q[1];m=q[2]*q[3]
                expected.append([float(f*(1-m)),float(m*(1-f))])
            got=stromal_myeloid_probabilities(counts,np.ones(len(counts),bool),float(p))
            np.testing.assert_allclose(got,expected,rtol=5e-14,atol=0)
            self.assertTrue(np.all(got[3]>0))
    def test_underflow_not_structural_zero(self):
        with self.assertRaises(FloatingPointError):
            stromal_myeloid_probabilities(np.full((1,4),2000),np.ones(1,bool),.5)
    def test_endpoints_and_fixed_mask(self):
        x=np.array([[1,1,0,0],[1,1,1,1],[0,0,1,1]])
        np.testing.assert_array_equal(stromal_myeloid_probabilities(x,np.ones(3,bool),1),[[1,0],[0,0],[0,1]])
        self.assertFalse(stromal_myeloid_probabilities(x,np.zeros(3,bool),.5).any())
        self.assertFalse(stromal_myeloid_probabilities(x,np.ones(3,bool),0).any())

if __name__=='__main__':unittest.main()
