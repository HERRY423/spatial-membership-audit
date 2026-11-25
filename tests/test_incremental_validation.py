import itertools
import unittest
import numpy as np
from spatial_membership_audit.incremental_validation import (
    shrink_depth_retention, train_only_design, fit_predict_fixed,
    group_bootstrap_difference, incremental_gate)


class IncrementalValidationTests(unittest.TestCase):
    def test_retention_endpoints_and_half_log_contrast(self):
        a=np.array([2., 8., 4.]); b=np.array([4., 2., 2.])
        p,q,_=shrink_depth_retention(a,b,1)
        np.testing.assert_allclose(a*p,b*q)
        p,q,_=shrink_depth_retention(a,b,0)
        np.testing.assert_allclose(p, np.full(3,b.sum()/a.sum()))
        np.testing.assert_array_equal(q,np.ones(3))
        p,q,_=shrink_depth_retention(a,b,.5)
        np.testing.assert_allclose(np.log(q/p),.5*(np.log(a/b)+np.log(a.sum()/b.sum())))

    def test_zero_boundary_no_pseudocount(self):
        p,q,s=shrink_depth_retention([0,1,0],[0,0,3],.5)
        np.testing.assert_array_equal(p,[1,0,1]);np.testing.assert_array_equal(q,[1,1,0])
        self.assertEqual(s.tolist(),['BOTH_ABSENT','COMMON_DEPTH_ZERO','COMMON_DEPTH_ZERO'])

    def test_bad_means_fail_closed(self):
        for a,b,s in [([1],[2],2),([float('nan')],[1],.5),([-1],[1],.5),([] ,[],.5)]:
            with self.assertRaises(ValueError): shrink_depth_retention(a,b,s)

    def test_test_features_do_not_change_transform(self):
        x=np.array([[1,2],[3,2],[5,2.]])
        a,_,t=train_only_design(x,[[1e10,-1]],['a']*3,['b'],['a','b'])
        aa,_,tt=train_only_design(x,[[-1e10,10]],['a']*3,['b'],['a','b'])
        np.testing.assert_array_equal(a,aa);self.assertEqual(t,tt)
        self.assertEqual(t['numeric_mean'],[3,2]);self.assertEqual(t['constant_numeric'],[False,True])

    def test_missing_features_and_unknown_classes_rejected(self):
        with self.assertRaises(ValueError): train_only_design([[1]],[[float('nan')]],['a'],['a'],['a'])
        with self.assertRaises(ValueError): train_only_design([[1]],[[2]],['a'],['z'],['a'])

    def test_ridge_matches_augmented_least_squares(self):
        x=np.column_stack([np.ones(5),np.arange(5)]);y=np.array([1,2,1,3,5.])
        pred, fit=fit_predict_fixed(x,y,x,task='continuous')
        aug=np.vstack([x,[0,1]]);target=np.r_[y,0]
        ref=np.linalg.lstsq(aug,target,rcond=None)[0]
        np.testing.assert_allclose(fit['coefficients'],ref,atol=1e-12)
        np.testing.assert_allclose(pred,np.maximum(x@ref,0))

    def test_logistic_probability_and_stationarity(self):
        x=np.column_stack([np.ones(8),np.arange(8)-3.5]);y=np.array([0,0,1,0,1,0,1,1])
        pred, fit=fit_predict_fixed(x,y,x,task='binary')
        self.assertTrue(((pred>0)&(pred<1)).all())
        grad=x.T@(pred-y)+np.array([0,fit['coefficients'][1]])
        self.assertLess(abs(grad).max(),1e-5)
        self.assertLess(pred[0],pred[-1])

    def test_single_class_does_not_become_default_probability(self):
        with self.assertRaises(ValueError): fit_predict_fixed(np.ones((3,1)),np.ones(3),np.ones((1,1)),task='binary')

    def test_cluster_bootstrap_has_correct_paired_mean_and_keeps_repeated_groups(self):
        result=group_bootstrap_difference([0,2,9],[0,0,0],['a','a','b'],draws=2000)
        self.assertEqual(result['value'],11/3);self.assertEqual(result['groups'],2)
        self.assertEqual(result['low'],1);self.assertEqual(result['high'],9)

    def test_gate_cannot_pass_missing_or_harmful_stratum(self):
        e={'value':-.2,'high':-.01}
        self.assertTrue(incremental_gate(e,[-.1,-.2],coverage_not_worse=True)['passed'])
        for d in [[-.1,0],[-.1,None],[]]:
            self.assertFalse(incremental_gate(e,d,coverage_not_worse=True)['passed'])
        self.assertFalse(incremental_gate(e,[-.1],coverage_not_worse=False)['passed'])


if __name__=='__main__': unittest.main()
