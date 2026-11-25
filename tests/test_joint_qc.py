import unittest
from fractions import Fraction
import numpy as np
from spatial_membership_audit.joint_qc import four_gene_joint_qc


class JointQCTests(unittest.TestCase):
    def test_shared_molecules_preclude_independent_qc_factor(self):
        r=four_gene_joint_qc([1,1,0,0],.5,min_total=2,min_detected=2)
        self.assertEqual(r['p_a_and_qc'],.25)
        self.assertEqual(r['p_qc']*r['p_a_without_qc'],.0625)

    def test_gene_vector_and_background_qc(self):
        # A is deterministic; passing QC needs the two background detections.
        r=four_gene_joint_qc([1,1,0,0,1,1],[1,1,0,0,.25,.75],min_total=4,min_detected=4)
        self.assertEqual(r['p_a_and_qc'],3/16)
        self.assertEqual(r['p_qc'],3/16)
        self.assertEqual(r['b_status'],'STRUCTURAL_ZERO')
        impossible=four_gene_joint_qc([1,1,0,0],.5,min_total=3,min_detected=2)
        self.assertEqual(impossible['qc_status'],'STRUCTURAL_ZERO')

    def test_underflow_is_not_structural(self):
        tiny=four_gene_joint_qc([1,1,300,300],.5,min_total=0,min_detected=0)
        ref=Fraction(1,4)*(2*Fraction(1,2**300)-Fraction(1,2**600))
        self.assertLess(abs(tiny['p_a_and_qc']/float(ref)-1),1e-11)
        under=four_gene_joint_qc([1,1,2000,2000],.5,min_total=0,min_detected=0)
        self.assertEqual(under['a_status'],'NUMERICAL_UNDERFLOW')
        zero=four_gene_joint_qc([0,1,2000,2000],.5,min_total=0,min_detected=0)
        self.assertEqual(zero['a_status'],'STRUCTURAL_ZERO')

    def test_no_qc_reduces_to_marker_operation(self):
        from spatial_membership_audit.rules import stromal_myeloid_probabilities
        x=np.array([[2,1,1,2],[1,1,0,0],[0,0,1,1]])
        for p in [0,.25,.5,.75,1]:
            expected=stromal_myeloid_probabilities(x,np.ones(3,bool),p)
            got=[four_gene_joint_qc(row,p,min_total=0,min_detected=0) for row in x]
            np.testing.assert_allclose([[r['p_a_and_qc'],r['p_b_and_qc']] for r in got],expected,rtol=2e-13,atol=0)


if __name__=='__main__': unittest.main()
