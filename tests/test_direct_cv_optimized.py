"""Exact-arithmetic regressions for the optimized certified-CV engine."""
from fractions import Fraction as F
import unittest

import numpy as np
from scipy import sparse
from spatial_membership_audit import direct_cv as cv


class OptimizedCVRegression(unittest.TestCase):
    def test_monomial_certificate_includes_bit_and_block_boundaries(self):
        x,w,w2,bounds=cv.rule(4,2050)
        for r,weights in enumerate((w,w2),start=1):
            for k in (*range(65),1023,1024,2047,2048,2049):
                exact=sum((F(float(wi))*F(float(xi))**k for xi,wi in zip(x,weights)),F(0))*(k+1)**r
                self.assertLessEqual(F(bounds[r-1][0]),exact)
                self.assertGreaterEqual(F(bounds[r-1][1]),exact)

    def test_denominator_only_node_is_not_deleted(self):
        # Node 0 is A. Node 1 is B/other; node 2 is always B and isolated.
        # T is 0 or 1/2, hence CV=1; deleting node 2 changes its mean.
        a=np.array([1.,0.,0.]);b=np.array([0.,.5,1.])
        graph=sparse.csr_matrix([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
        out=cv.estimate(a,b,graph,accuracy_factor=1.25)
        self.assertEqual(out['status'],'RESOLVED')
        self.assertEqual(out['moment_nodes'],2)
        self.assertEqual(out['n'],3)
        self.assertLessEqual(out['first_ratio_moment_interval'][0],.25)
        self.assertGreaterEqual(out['first_ratio_moment_interval'][1],.25)
        self.assertLessEqual(out['cv_interval'][0],1.)
        self.assertGreaterEqual(out['cv_interval'][1],1.)

    def test_original_edge_count_and_zero_numerator_state(self):
        graph=sparse.csr_matrix([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
        out=cv.estimate([1.,0.,0.],[0.,0.,1.],graph)
        self.assertEqual(out['status'],'ZERO_NUMERATOR_CV_UNDEFINED')
        self.assertEqual(out['moment_edges'],0)
        self.assertEqual(out['m'],1)


if __name__=='__main__':unittest.main()
