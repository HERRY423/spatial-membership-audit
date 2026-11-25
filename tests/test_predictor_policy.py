import math
import sys
import unittest
import numpy as np
from spatial_membership_audit.predictor_policy import (
    CVInterval, assess_predictors, assess_moment_output, bounded_cv_interval)


class PredictorPolicyTests(unittest.TestCase):
    def policy(self, d=1., s=2., **kwargs):
        return assess_predictors(d, s, regret_factor=1.1, accuracy_factor=1.25, **kwargs)

    def test_regret_theorem_and_sharpness(self):
        # Targets far below/above attain the bound; interior targets can improve
        # on both candidates. This checks the actual loss, not copied branches.
        for d, s in [(1., 4.), (4., 1.), (1e-150, 1e150), (.9, 1.1)]:
            m = math.exp((math.log(d)+math.log(s))/2)
            bound = abs(math.log(d)-math.log(s))/2
            for logt in np.linspace(-700, 700, 701):
                regret = abs(math.log(m)-logt)-min(abs(math.log(d)-logt), abs(math.log(s)-logt))
                self.assertLessEqual(regret, bound+1e-12)
            target = min(math.log(d), math.log(s))-10
            self.assertAlmostEqual(abs(math.log(m)-target)-min(abs(math.log(d)-target), abs(math.log(s)-target)), bound)

    def test_agreement_does_not_certify_accuracy(self):
        r = self.policy(1., 1.01)
        self.assertEqual(r["status"], "RELATIVE_REGRET_ONLY")
        self.assertEqual(r["accuracy_status"], "UNRESOLVED")
        self.assertIsNone(r["absolute_error_factor_bound"])

    def test_ranking_both_directions_and_crossing(self):
        for d, s, lo, hi, winner in [(1,2,.8,.9,"DELTA"),
                                    (2,1,.8,.9,"EXPECTED_O"),
                                    (1,2,3,4,"EXPECTED_O"),
                                    (2,1,3,4,"DELTA"),
                                    (1,2,1,2,"UNRESOLVED")]:
            r = self.policy(d, s, reference=CVInterval(lo, hi, "exact_enclosure", "fixture"))
            self.assertEqual(r["ranking"], winner)

    def test_certified_winner_can_still_be_inaccurate(self):
        r = self.policy(reference=CVInterval(10, 11, "exact_enclosure", "fixture"))
        self.assertEqual(r["ranking"], "EXPECTED_O")
        self.assertFalse(any(r["candidate_accuracy_pass"].values()))
        self.assertEqual(r["recommended_method"], "model_reference_center")

    def test_reference_recommendation_does_not_inherit_compromise_bound(self):
        r=self.policy(1,1,reference=CVInterval(1,1.44,"exact_enclosure","fixture"))
        self.assertAlmostEqual(r["recommended_value"],1.2)
        self.assertEqual(r["compromise_regret_factor_bound"],1)
        self.assertEqual(r["compromise_regret_bound_output"],"geometric_compromise")
        self.assertIsNone(r["recommended_regret_factor_bound"])

    def test_extreme_positive_values_return_states_without_overflow(self):
        small=float.fromhex("0x0.0000000000001p-1022")
        large=sys.float_info.max
        r=self.policy(small,large)
        self.assertEqual(r["status"],"NEEDS_MODEL_REFERENCE")
        self.assertIsNone(r["compromise_regret_factor_bound"])
        self.assertTrue(math.isfinite(r["compromise_regret_log_bound"]))
        r=self.policy(small,large,reference=CVInterval(1,1.01,"exact_enclosure","fixture"))
        self.assertEqual(r["status"],"MODEL_REFERENCE_RESOLVED")
        self.assertIsNone(r["candidate_error_factor_bounds"]["delta"])

    def test_evaluation_reference_never_acts_as_deployment_oracle(self):
        r = self.policy(reference=CVInterval(10,11,"exact_enclosure","test target"),
                        reference_role="evaluation_only")
        self.assertEqual(r["ranking"], "UNRESOLVED")
        self.assertEqual(r["status"], "NEEDS_MODEL_REFERENCE")

    def test_approximate_reference_not_certified(self):
        r = self.policy(reference=CVInterval(10,11,"approximate_mc","jackknife"))
        self.assertEqual(r["ranking"], "UNRESOLVED")
        self.assertEqual(r["accuracy_status"], "UNRESOLVED")

    def test_zero_na_and_nonpositive_are_distinct(self):
        for status in ["NO_FINITE_RATIO", "ZERO_NUMERATOR_CV_UNDEFINED", "NUMERICAL_UNDERFLOW"]:
            self.assertEqual(assess_moment_output({"status":status},regret_factor=1.1,
                                                 accuracy_factor=1.25)["status"],status)
        for d in [0, None, math.nan, math.inf, -1]:
            self.assertEqual(self.policy(d, 1)["status"],"NONPOSITIVE_OR_UNAVAILABLE_CANDIDATE")
        interval, stats = bounded_cv_interval([0,0,math.nan],1,alpha=.05,evidence_id="fixed")
        self.assertEqual((stats["finite_n"],stats["zero_n"],stats["positive_n"]),(2,2,0))
        self.assertEqual(interval.upper,math.inf)
        self.assertEqual(interval.lower,0)

    def test_fixed_budget_coverage_enumerated_bernoulli(self):
        # Exact binomial probabilities; no random test or new scientific draws.
        n, p, alpha = 50, .2, .1
        truth = math.sqrt((1-p)/p)
        miss = 0.
        for k in range(n+1):
            ci, _ = bounded_cv_interval([1]*k+[0]*(n-k),1,alpha=alpha,evidence_id="synthetic")
            if not ci.lower <= truth <= ci.upper:
                miss += math.comb(n,k)*p**k*(1-p)**(n-k)
        self.assertLessEqual(miss, alpha)

    def test_invalid_input_and_infinite_interval(self):
        for values, bound in [([math.inf],1), ([-1],1), ([2],1), ([1],0)]:
            with self.assertRaises(ValueError):
                bounded_cv_interval(values,bound,alpha=.05,evidence_id="fixture")
        ci, stats = bounded_cv_interval([math.nan],1,alpha=.05,evidence_id="fixture")
        self.assertEqual(stats["finite_n"],0)
        self.assertEqual(self.policy(reference=ci)["status"],"NEEDS_MODEL_REFERENCE")
        with self.assertRaises(ValueError):
            CVInterval(2,1,"exact_enclosure","fixture")
        with self.assertRaises(ValueError):
            self.policy(reference_role="test")

    def test_adapter(self):
        r=assess_moment_output({"status":"DELTA_APPROXIMATION","mu_O_finite":4,
                               "cv_R_delta":.51},regret_factor=1.1,accuracy_factor=1.25)
        self.assertEqual(r["expected_O"],.5)
        self.assertEqual(r["status"],"RELATIVE_REGRET_ONLY")


if __name__ == "__main__":
    unittest.main()
