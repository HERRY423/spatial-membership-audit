import itertools
import math
import unittest
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
from spatial_membership_audit.qc_tail import qc_failure_log_probability
from spatial_membership_audit.candidate_qc import candidate_joint_qc,candidate_joint_probabilities
from spatial_membership_audit.definitions import REQUIRED


def exhaustive_failure(counts,p,t,d):
    probability=Fraction(0)
    for y in itertools.product(*(range(n+1) for n in counts)):
        if sum(y)>=t and sum(k>0 for k in y)>=d:continue
        probability+=math.prod(Fraction(math.comb(n,k))*q**k*(1-q)**(n-k) for n,k,q in zip(counts,y,p))
    return probability


class QCTailTests(unittest.TestCase):
    def test_full_count_enumeration_scalar_and_vector(self):
        counts=[2,3,1]
        for values in [[Fraction(1,4)]*3,[Fraction(3,4)]*3,[Fraction(1),Fraction(1,2),Fraction(0)]]:
            for t,d in [(0,0),(0,2),(2,0),(3,2),(5,3),(9,3)]:
                with self.subTest(p=values,t=t,d=d):
                    exact=exhaustive_failure(counts,values,t,d)
                    result=qc_failure_log_probability(counts,list(map(float,values)),min_total=t,min_detected=d)
                    if exact==0:
                        self.assertEqual(result['float_status'],'STRUCTURAL_ZERO')
                        self.assertIsNone(result['log_probability'])
                    else:
                        self.assertAlmostEqual(result['probability_float'],float(exact),places=12)
                        with localcontext() as ctx:
                            ctx.prec=80
                            log_reference=(Decimal(exact.numerator)/Decimal(exact.denominator)).ln()
                        self.assertLess(abs(result['log_probability']-float(log_reference)),2e-12)

    def test_structural_and_certain_endpoints(self):
        for p,status,log in [(1,'STRUCTURAL_ZERO',None),(0,'POSITIVE_FINITE',0.)]:
            r=qc_failure_log_probability([2000],p,min_total=1,min_detected=1)
            self.assertEqual(r['float_status'],status);self.assertEqual(r['log_probability'],log)
        self.assertEqual(qc_failure_log_probability([],0,min_total=1)['log_probability'],0.)

    def test_underflow_log_has_independent_decimal_reference(self):
        for n in [1000,1040,2000]:
            r=qc_failure_log_probability([n],.5,min_total=1,min_detected=1)
            with localcontext() as ctx:
                ctx.prec=100;reference=-Decimal(n)*Decimal(2).ln()
            self.assertLess(abs(r['log_probability']-float(reference)),1e-11)
            self.assertEqual(r['log_status'],'POSITIVE_LOG_FINITE')
        self.assertEqual(r['float_status'],'NUMERICAL_UNDERFLOW')
        self.assertEqual(r['probability_float'],0.)

    def test_candidate_fallback_preserves_float_state(self):
        genes=list(REQUIRED)+['BG'];counts=[0]*11+[2000]
        r=candidate_joint_qc(counts,genes,.5,min_total=1,min_detected=1)
        self.assertEqual(r['p_qc_failure'],0.)
        self.assertEqual(r['qc_failure_status'],'NUMERICAL_UNDERFLOW')
        self.assertTrue(math.isfinite(r['qc_failure_log10']))
        self.assertEqual(r['qc_failure_log_method'],'log_domain_disjoint_QC_failure_DP')
        vector=candidate_joint_qc(counts,genes,[.5]*12,min_total=1,min_detected=1)
        self.assertEqual(vector['qc_failure_log'],r['qc_failure_log'])
        exact=candidate_joint_qc(counts,genes,1,min_total=1,min_detected=1)
        self.assertIsNone(exact['qc_failure_log']);self.assertEqual(exact['qc_failure_log_status'],'STRUCTURAL_ZERO')

    def test_ineligible_mask_overrides_effective_log(self):
        counts=[[0]*11+[2000]];genes=list(REQUIRED)+['BG']
        r=candidate_joint_probabilities(counts,genes,np.array([False]),.5,min_total=1,min_detected=1)['cells'][0]
        self.assertEqual(r['p_qc_failure'],1.)
        self.assertEqual(r['qc_failure_log'],0.)
        self.assertEqual(r['qc_failure_log_method'],'fixed_baseline_ineligible')

    def test_invalid_inputs_rejected(self):
        for counts,p in [([.5],.5),([1],1.5),([1],[.1,.2]),([1],float('nan'))]:
            with self.assertRaises(ValueError):qc_failure_log_probability(counts,p)
        with self.assertRaises(ValueError):qc_failure_log_probability([1],.5,min_total=True)
