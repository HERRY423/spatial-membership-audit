import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from spatial_membership_audit.analytic import CLASSES, marker_probabilities
from spatial_membership_audit.candidate_qc import candidate_joint_qc, candidate_joint_probabilities
from spatial_membership_audit.definitions import REQUIRED, candidate_labels


def exact_reference(x, genes, p, total, detected):
    p=[Fraction(v) for v in p] if isinstance(p,(tuple,list)) else [Fraction(p)]*len(x)
    possibilities=[list(range(n+1)) if 0<q<1 else [n if q==1 else 0] for n,q in zip(x,p)]
    y=np.array(list(itertools.product(*possibilities)),dtype=np.int64)
    labels=candidate_labels(y,genes,np.ones(len(y),bool))
    tables=[{k:Fraction(math.comb(n,k))*q**k*(1-q)**(n-k) for k in rr}
            for n,q,rr in zip(x,p,possibilities)]
    result=[Fraction(0)]*5
    for row,label in zip(y,labels):
        if row.sum()<total or np.count_nonzero(row)<detected: continue
        probability=math.prod(table[int(k)] for table,k in zip(tables,row))
        result[4]+=probability
        if label in CLASSES:result[CLASSES.index(label)]+=probability
    return result


class FullRuleJointQCTests(unittest.TestCase):
    def test_complete_count_enumeration(self):
        # Includes B OR, T sum=3, CD8=2, CD4, myeloid/plasma conflicts and background.
        x=[1,1,1,2,1,2,1,1,1,1,1,1]; genes=list(REQUIRED)+['BACKGROUND']
        cases=[(.5,5,4),(.25,3,3),(.75,8,6),
               ([0,1,.25,.5,.75,.5,0,.25,1,.5,.75,.25],4,3)]
        for p,t,d in cases:
            with self.subTest(p=p,total=t,detected=d):
                expected=exact_reference(x,genes,p,t,d)
                result=candidate_joint_qc(x,genes,p,min_total=t,min_detected=d)
                np.testing.assert_allclose(result['probabilities']+[result['p_qc']],
                                           list(map(float,expected)),atol=2e-12,rtol=0)
                self.assertAlmostEqual(result['p_qc_failure'],float(1-expected[4]),places=12)
                self.assertTrue(0<=result['p_qc']<=1 and 0<=result['p_qc_failure']<=1)

    def test_qc_caps_below_classification_thresholds(self):
        x=[0,0,0,2,1,2,0,0,0,0,0]
        for t,d in [(0,0),(1,1),(2,1),(3,2),(6,4)]:
            result=candidate_joint_qc(x,REQUIRED,.5,min_total=t,min_detected=d)
            expected=exact_reference(x,REQUIRED,.5,t,d)
            np.testing.assert_allclose(result['probabilities']+[result['p_qc']],list(map(float,expected)),atol=2e-12,rtol=0)

    def test_without_qc_matches_full_fixed_classifier(self):
        rng=np.random.default_rng(1729);genes=list(REQUIRED)+['BG1','BG2']
        x=rng.integers(0,7,size=(9,len(genes)))
        for p in [0,.25,.5,.75,1,np.linspace(0,1,len(genes))]:
            old=marker_probabilities(x,genes,np.ones(len(x),bool),p)
            for row,expected in zip(x,old):
                result=candidate_joint_qc(row,genes,p,min_total=0,min_detected=0)
                np.testing.assert_allclose(result['probabilities'],expected,atol=2e-12,rtol=2e-12)
                self.assertAlmostEqual(result['p_qc'],1.,places=12)

    def test_p_one_hard_labels_conflicts_and_baseline_mask(self):
        x=np.zeros((7,12),dtype=int);genes=list(REQUIRED)+['BG']
        x[0,[0,2]]=[1,1]              # B through the second OR branch
        x[1,[3,4,5]]=[2,1,2]          # CD8
        x[2,[3,4,6]]=[1,2,1]          # CD4
        x[3,[7,8]]=[1,1]              # myeloid
        x[4,[0,1,9,10]]=[1,1,1,1]     # plasma veto
        x[5,[3,4,5,6]]=[2,1,2,1]      # CD4/CD8 conflict
        x[6,[0,1]]=[1,1]
        mask=np.array([True]*6+[False])
        for t,d in [(0,0),(3,2),(5,3)]:
            qc=(x.sum(axis=1)>=t)&((x>0).sum(axis=1)>=d)&mask
            expected=candidate_labels(x,genes,qc)
            result=candidate_joint_probabilities(x,genes,mask,1,min_total=t,min_detected=d)
            np.testing.assert_array_equal(result['p_qc'],qc.astype(float))
            np.testing.assert_array_equal(result['probabilities'],np.column_stack([expected==c for c in CLASSES]).astype(float))
            self.assertEqual(result['probabilities'].shape,(7,4))

    def test_tiny_underflow_structural_and_batch_guard(self):
        x=[1,1,0,0,0,0,0,300,300,0,0]
        result=candidate_joint_qc(x,REQUIRED,.5,min_total=0,min_detected=0)
        ref=Fraction(1,4)*(2*Fraction(1,2**300)-Fraction(1,2**600))
        self.assertLess(abs(result['probabilities'][0]/float(ref)-1),1e-11)
        self.assertEqual(result['class_statuses'][0],'POSITIVE_FINITE')
        x[7]=x[8]=1040
        sub=candidate_joint_qc(x,REQUIRED,.5,min_total=0,min_detected=0)
        self.assertGreater(sub['probabilities'][0],0.)
        self.assertEqual(sub['class_statuses'][0],'POSITIVE_SUBNORMAL')
        x[7]=x[8]=2000
        under=candidate_joint_qc(x,REQUIRED,.5,min_total=0,min_detected=0)
        self.assertEqual(under['class_statuses'][0],'NUMERICAL_UNDERFLOW')
        with self.assertRaisesRegex(ArithmeticError,'NUMERICAL_UNDERFLOW'):
            candidate_joint_probabilities([x],REQUIRED,np.array([True]),.5,min_total=0,min_detected=0)
        x[0]=0
        zero=candidate_joint_qc(x,REQUIRED,.5,min_total=0,min_detected=0)
        self.assertEqual(zero['class_statuses'][0],'STRUCTURAL_ZERO')

    def test_invalid_inputs_and_missing_panel_are_not_zeros(self):
        x=[0]*11
        cases=[(x[:-1],REQUIRED[:-1],.5),([.5]+x[1:],REQUIRED,.5),
               (x,REQUIRED,[.5]*10),(x,REQUIRED,float('nan'))]
        for counts,genes,p in cases:
            with self.subTest(genes=len(genes),p=p),self.assertRaises(ValueError):
                candidate_joint_qc(counts,genes,p)
        with self.assertRaises(ValueError):candidate_joint_qc(x,REQUIRED,.5,min_total=True)

    def test_direct_qc_complement_preserves_rare_failure(self):
        genes=list(REQUIRED)+['BG']
        result=candidate_joint_qc([0]*11+[1000],genes,.5,min_total=1,min_detected=1)
        self.assertEqual(result['p_qc'],1.)
        self.assertGreater(result['p_qc_failure'],0.)
        self.assertLess(abs(result['p_qc_failure']/float(Fraction(1,2**1000))-1),1e-10)
        self.assertEqual(result['qc_failure_status'],'POSITIVE_FINITE')
        under=candidate_joint_qc([0]*11+[2000],genes,.5,min_total=1,min_detected=1)
        self.assertEqual(under['qc_failure_status'],'NUMERICAL_UNDERFLOW')
        structural=candidate_joint_qc([0]*11+[2000],genes,1,min_total=1,min_detected=1)
        self.assertEqual(structural['qc_failure_status'],'STRUCTURAL_ZERO')


if __name__=='__main__':unittest.main()
