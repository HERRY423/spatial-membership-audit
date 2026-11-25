"""Contract tests on synthetic fixtures; no performance/coverage validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from spatial_membership_audit.auto_routing import route_cv, main, _sample


class AutoRoutingTests(unittest.TestCase):
    def setUp(self):
        # Node2 is an isolated B that must remain in the denominator.
        # Enumerating node1's two labels gives R=0 or1.5, both mass1/2: CV=1.
        self.p = np.array([[1., 0.], [0., .5], [0., 1.]])
        self.g = sparse.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    def screen(self, **kwargs):
        options = dict(requirement='screening', direct_support_hint=0,
                       draws=128, replicates=2, max_seconds=5.)
        options.update(kwargs)
        return route_cv(self.p, self.g, **options)

    def test_enclosure_has_real_enclosure(self):
        out = route_cv(self.p, self.g, requirement='enclosure', max_seconds=5.)
        self.assertEqual(out['engine'], 'D')
        self.assertEqual(out['status'], 'RESOLVED')
        self.assertLessEqual(out['cv_interval'][0], 1.)
        self.assertGreaterEqual(out['cv_interval'][1], 1.)
        self.assertTrue(out['factor_target_assessed'])
        self.assertEqual(out['guarantee'], 'CONDITIONAL_NUMERICAL_ENCLOSURE')

    def test_sobol_matches_two_state_enumeration_retains_isolate(self):
        out = self.screen()
        self.assertEqual(out['engine'], 'SOBOL')
        self.assertEqual(out['K_union'], 3)
        self.assertAlmostEqual(out['cv'], 1.)
        self.assertAlmostEqual(out['output']['finite_mean'], .75)
        self.assertEqual(out['output']['counts'], dict(finite=256, unavailable=0, finite_zero=128, positive=128))
        self.assertIsNone(out['cv_interval'])
        self.assertFalse(out['factor_target_assessed'])
        self.assertEqual(out['coverage_calibration'], 'NOT_ESTABLISHED')

    def test_mc_seed_replay_and_independent_scalar_reference(self):
        out = self.screen(screening_engine='mc', seed=5)
        other = self.screen(screening_engine='mc', seed=5)
        self.assertEqual(out['output'], other['output'])
        vals=[]
        for child in np.random.SeedSequence(5).spawn(2):
            rng=np.random.default_rng(int(child.generate_state(1)[0]))
            for u in rng.random((128,3)):
                labels=[1 if u[i]<self.p[i,0] else 2 if u[i]<sum(self.p[i]) else 0 for i in range(3)]
                na=labels.count(1); nb=labels.count(2)
                contacts=int(set(labels[:2])=={1,2})
                vals.append(contacts/((1/3)*na*nb))
        self.assertAlmostEqual(out['cv'], np.std(vals)/np.mean(vals))

    def test_original_n_m_include_other_nodes_and_edges(self):
        p=np.vstack([self.p,[0,0],[0,0]])
        g=sparse.block_diag((self.g,[[0,1],[1,0]]),format='csr')
        out=route_cv(p,g,requirement='screening',direct_support_hint=0,draws=128,replicates=2,max_seconds=5.)
        self.assertEqual(out['n'],5); self.assertEqual(out['m'],2)
        self.assertAlmostEqual(out['output']['finite_mean'],1.25)
        self.assertAlmostEqual(out['cv'],1.)

    def test_no_silent_downgrade_on_enclosure_failure(self):
        failure=dict(status='UNRESOLVED', reason='BUDGET_EXHAUSTED',cv=None,cv_interval=None)
        with patch('spatial_membership_audit.auto_routing.estimate_cv',return_value=failure):
            out=route_cv(self.p,self.g,requirement='enclosure',max_seconds=5.)
        self.assertEqual(out['engine'],'D'); self.assertIsNone(out['cv'])
        self.assertEqual(len(out['attempts']),1); self.assertFalse(out['fallback_used'])
        self.assertEqual(out['guarantee'],'NONE')
        with self.assertRaises(ValueError):
            route_cv(self.p,self.g,requirement='enclosure',allow_approximate_fallback=True)

    def test_explicit_screening_fallback_retains_failed_d_attempt(self):
        failure=dict(status='UNRESOLVED',reason='ORDER_LIMIT',cv=None,cv_interval=[0,2])
        with patch('spatial_membership_audit.auto_routing.estimate_cv',return_value=failure):
            out=self.screen(direct_support_hint=1000,allow_approximate_fallback=True)
        self.assertTrue(out['fallback_used']); self.assertEqual(len(out['attempts']),2)
        self.assertEqual(out['attempts'][0]['output'],failure)
        self.assertIsNone(out['cv_interval']); self.assertFalse(out['factor_target_assessed'])
        self.assertEqual(out['guarantee'],'APPROXIMATE_POINT_ESTIMATE')

    def test_numerical_inconsistency_does_not_get_hidden_by_fallback(self):
        with patch('spatial_membership_audit.auto_routing.estimate_cv',return_value=dict(status='NUMERICAL_INCONSISTENCY',cv=None)):
            out=self.screen(direct_support_hint=1000,allow_approximate_fallback=True)
        self.assertEqual(out['status'],'NUMERICAL_INCONSISTENCY');self.assertFalse(out['fallback_used'])

    def test_strict_factor_routes_to_d_independent_of_size_hint(self):
        out=self.screen(accuracy_factor=1.01)
        self.assertEqual(out['engine'],'D')
        self.assertIn('STRICT_FACTOR_HEURISTIC',out['route_trace'])

    def test_sobol_dimension_limit_is_reported_and_routes_mc(self):
        with patch('spatial_membership_audit.auto_routing.qmc.Sobol.MAXDIM',2):
            out=self.screen(screening_engine='sobol')
        self.assertEqual(out['engine'],'MC')
        self.assertIn('SOBOL_DIMENSION_LIMIT',out['route_trace'])

    def test_structural_zero_and_unavailability_distinct(self):
        p=self.p.copy();p[1]=0
        zero=route_cv(p,self.g,requirement='screening')
        self.assertEqual(zero['status'],'ZERO_NUMERATOR_CV_UNDEFINED');self.assertIsNone(zero['cv'])
        p[2]=0
        na=route_cv(p,self.g,requirement='screening')
        self.assertEqual(na['status'],'NO_FINITE_RATIO')
        self.assertEqual(na['output']['p_finite_interval'],[0,0])

    def test_finite_zero_sample_does_not_become_structural_zero(self):
        p=self.p.copy();p[1]=[0,1e-30]
        out=route_cv(p,self.g,requirement='screening',direct_support_hint=0,screening_engine='mc',draws=32,replicates=2,max_seconds=5.)
        self.assertEqual(out['status'],'SAMPLE_NO_POSITIVE_CONTACTS')
        self.assertIsNone(out['cv']);self.assertTrue(out['rare_contact_hint'])
        self.assertEqual(out['output']['counts']['finite_zero'],64)
        self.assertEqual(out['guarantee'],'NONE')

    def test_sample_no_finite_is_not_structural_no_finite(self):
        p=np.full((3,2),1e-30)
        out=route_cv(p,self.g,requirement='screening',direct_support_hint=0,draws=32,replicates=2,max_seconds=5.)
        self.assertEqual(out['status'],'SAMPLE_NO_FINITE_OBSERVATIONS')
        self.assertEqual(out['output']['counts']['unavailable'],64)

    def test_soft_deadline_and_memory_failure_return_no_invented_estimate(self):
        out=self.screen(max_seconds=1e-12)
        self.assertEqual(out['status'],'BUDGET_EXHAUSTED');self.assertIsNone(out['cv'])
        self.assertEqual(len(out['sampling_plan']),2)
        self.assertTrue(all(x['status']=='NOT_RUN_BUDGET' for x in out['sampling_plan']))
        out=self.screen(max_batch_bytes=1)
        self.assertEqual(out['status'],'RESOURCE_LIMIT');self.assertIsNone(out['cv'])
        self.assertEqual(len(out['sampling_plan']),2)
        self.assertTrue(all(x['status']=='NOT_RUN_RESOURCE_LIMIT' for x in out['sampling_plan']))
        expected=[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(0).spawn(2)]
        self.assertEqual([x['seed'] for x in out['sampling_plan']],expected)

    def test_already_known_structural_states_survive_expired_budget(self):
        p=self.p.copy();p[1]=0
        out=route_cv(p,self.g,requirement='screening',max_seconds=1e-12)
        self.assertEqual(out['status'],'ZERO_NUMERATOR_CV_UNDEFINED')
        self.assertFalse(out['within_soft_budget'])
        p[2]=0
        out=route_cv(p,self.g,requirement='screening',max_seconds=1e-12)
        self.assertEqual(out['status'],'NO_FINITE_RATIO')
        self.assertTrue(all(x['status']=='NOT_NEEDED_STRUCTURAL' for x in out['sampling_plan']))

    def test_no_sampling_budget_is_not_observed_zero_or_missing_plan(self):
        out=_sample(self.p,self.g,3,1,engine='MC',draws=32,replicates=2,seed=1,
                    deadline=-1.,batch_size=16,max_batch_bytes=1024**2)
        self.assertEqual(out['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(out['completed_draws'],0)
        self.assertEqual(len(out['replicates']),2)
        self.assertTrue(all(x['status']=='NOT_RUN_BUDGET' for x in out['replicates']))

    def test_invalid_inputs(self):
        for kwargs in [dict(draws=3),dict(replicates=1),dict(seed=-1),dict(accuracy_factor=1),dict(requirement='calibrated'),dict(a=True),dict(allow_approximate_fallback=1)]:
            with self.assertRaises(ValueError): self.screen(**kwargs)
        p=np.pad(self.p,((0,0),(0,1)));p[0,2]=1e-30
        with self.assertRaises(ValueError):route_cv(p,self.g,requirement='screening')

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td); np.save(folder/'p.npy',self.p);sparse.save_npz(folder/'g.npz',self.g)
            argv=['--probabilities',str(folder/'p.npy'),'--adjacency',str(folder/'g.npz'),
                  '--requirement','screening','--support-hint','0','--draws','32',
                  '--replicates','2','--out',str(folder/'out.json')]
            main(argv); out=json.loads((folder/'out.json').read_text())
            self.assertEqual(out['engine'],'SOBOL')
            with self.assertRaises(FileExistsError):main(argv)


if __name__=='__main__':unittest.main()
