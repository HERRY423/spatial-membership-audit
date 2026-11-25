"""Tail regressions independently checked against known event probabilities."""
import math
import json
import unittest
import numpy as np
from spatial_membership_audit.availability import availability_tails,marker_pair_availability_half
from spatial_membership_audit.analytic import marker_probabilities,pair_moments
from spatial_membership_audit.definitions import REQUIRED


class TailTests(unittest.TestCase):
    def test_tiny_complement_survives_rounded_availability(self):
        # A absent only when all 54 Bernoulli(.5) nodes fail; B is certain.
        out=availability_tails([.5]*54+[0],[0]*54+[1])
        self.assertAlmostEqual(out['p_na']/2**-54,1.,places=12)
        self.assertEqual(out['na_probability_status'],'POSITIVE_FINITE')

    def test_underflow_is_not_structural_zero(self):
        out=availability_tails([.5]*1329+[0],[0]*1329+[1])
        self.assertEqual(out['p_na'],0.)
        self.assertEqual(out['na_probability_status'],'NUMERICAL_UNDERFLOW')
        self.assertAlmostEqual(out['log_p_na'],-1329*math.log(2),places=8)
        zero=availability_tails([1,0],[0,1])
        self.assertEqual(zero['na_probability_status'],'STRUCTURAL_ZERO')
        self.assertIsNone(zero['log_p_na'])
        json.dumps(zero,allow_nan=False)

    def test_subnormal_is_positive(self):
        out=availability_tails([.5]*1050+[0],[0]*1050+[1])
        self.assertGreater(out['p_na'],0)
        self.assertEqual(out['na_probability_status'],'POSITIVE_SUBNORMAL')

    def test_count_input_keeps_rounded_cell_failure(self):
        x=np.zeros((2,11),dtype=int);x[0,[0,1]]=600;x[1,[7,8]]=600
        mask=np.ones(2,bool)
        p=marker_probabilities(x,REQUIRED,mask,.5)
        rounded=availability_tails(p[:,0],p[:,3])
        raw=marker_pair_availability_half(x,REQUIRED,mask,0,3)
        self.assertEqual(rounded['na_probability_status'],'STRUCTURAL_ZERO')
        self.assertEqual(rounded['probability_basis'],'supplied_binary64_categorical_values')
        # Four required molecules groups; q=1-(1-2^-600)^4 ~4*2^-600.
        self.assertAlmostEqual(raw['p_na']/(4*2.**-600),1.,places=10)
        self.assertEqual(raw['na_probability_status'],'POSITIVE_FINITE')

    def test_geometry_and_missing_inputs(self):
        out=availability_tails([1,0],[0,1],has_edges=False)
        self.assertEqual(out['p_na'],1.)
        self.assertEqual(out['finite_probability_status'],'STRUCTURAL_ZERO')
        with self.assertRaises(ValueError):availability_tails([.8],[.4])
        with self.assertRaises(ValueError):marker_pair_availability_half([[1]],['MS4A1'],np.ones(1,bool),0,1)

    def test_existing_moment_interface_adds_direct_na(self):
        out=pair_moments([[.5,0],[0,1]],[[0,1],[1,0]])
        self.assertEqual(out['p_na'],.5)
        self.assertEqual(out['p_finite'],.5)
        self.assertEqual(out['cv_R_delta'],0.)

if __name__=='__main__':unittest.main()
