"""Regression tests for the E4 MC-zero and vector-retention audit failures.

All inputs are synthetic software fixtures. These tests make no biological
accuracy, uncertainty-coverage or end-to-end inference claim.
"""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatial_membership_audit.custom_rules import frozen_classifier_probabilities
from spatial_membership_audit.standard_ecosystem import audit_anndata
from spatial_membership_audit.action_guide import triage_audit_record, triage_audit_table


def rare_classifier(x, genes):
    return np.where(x[:, 0] == 20, 'A', np.where(x[:, 1] > 0, 'B', 'OTHER'))


def nonlinear_classifier(x, genes):
    g0, g1 = x[:, 0], x[:, 1]
    return np.where(g0 + g1 == 0, 'OTHER',
                    np.where((g0 * g1 + g0) % 2 == 0, 'A', 'B'))


def make_adata(counts=None, adjacency=None):
    counts = np.array([[20, 0], [0, 20]]) if counts is None else np.asarray(counts)
    n, g = counts.shape
    if adjacency is None:
        adjacency = np.zeros((n, n))
        for i in range(n - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    obj = ad.AnnData(sparse.csr_matrix(counts),
        obs=pd.DataFrame({'region': ['R'] * n}, index=[f'c{i}' for i in range(n)]),
        var=pd.DataFrame(index=[f'G{i}' for i in range(g)]))
    obj.obsp['spatial_connectivities'] = sparse.csr_matrix(adjacency)
    return obj


class ClassifierMCProvenanceTests(unittest.TestCase):
    def audit(self, obj=None, **kwargs):
        options = dict(classifier_fn=rare_classifier, classifier_assume_cellwise=True,
                       classes=['A', 'B'], region_key='region', max_seconds=3.0)
        options.update(kwargs)
        return audit_anndata(make_adata() if obj is None else obj, **options)

    def test_rare_positive_probability_is_unresolved_not_structurally_absent(self):
        # Independent analytic reference: the two genes reside on different
        # nodes; A requires all 20 retained, B at least one retained.
        true_p_plus = 2 ** -20 * (1 - 2 ** -20)
        self.assertGreater(true_p_plus, 0)
        for mode in ('direct', 'auto', 'delta'):
            with self.subTest(mode=mode):
                row = self.audit(p=0.5, cv_mode=mode).iloc[0]
                self.assertEqual(row['p_finite'], 0.0)  # preserve the observed MC estimate
                self.assertEqual(row['probability_support_status'], 'UNRESOLVED_MC_SUPPORT')
                self.assertEqual(row['cv_status'], 'UNRESOLVED_UPSTREAM_SAMPLING')
                self.assertEqual(row['regime'], 'NUMERICAL_UNRESOLVED')
                self.assertEqual(row['guarantee'], 'NONE')
                self.assertTrue(pd.isna(row['cv']))
                self.assertTrue(pd.isna(row['cv_lower']))
                self.assertEqual(row['action_code'], 'RESOLVE_CLASSIFIER_MC_SUPPORT')
                self.assertNotIn('两类不能同时出现', row['action_zh'])
                self.assertIn('抽样未解', row['action_zh'])

    def test_positive_mc_result_keeps_conditional_scope_and_provenance(self):
        obj = make_adata([[2, 1], [1, 2], [2, 2], [1, 1]])
        df = self.audit(obj, p=0.5, classifier_fn=nonlinear_classifier,
                        classifier_draws=256, seed=42, inplace=True)
        row = df.iloc[0]
        self.assertEqual(row['cv_status'], 'RESOLVED')
        self.assertEqual(row['guarantee'], 'MC_PLUGIN_ONLY')
        self.assertEqual(row['conditional_guarantee'], 'CONDITIONAL_NUMERICAL_ENCLOSURE')
        self.assertEqual(row['cv_interval_scope'], 'CONDITIONAL_ON_MC_PROBABILITIES_EXCLUDES_UPSTREAM_ERROR')
        self.assertTrue(row['p_finite_is_estimate'])
        self.assertIn('not included', row['upstream_probability_uncertainty'])
        for meta in (row, df.attrs['provenance']['probability_runs']['retention_0'],
                     obj.uns['spatial_membership_audit']['provenance']['probability_runs']['retention_0']):
            self.assertEqual(meta['n_draws'], 256)
            self.assertEqual(meta['seed'], 42)
            self.assertEqual(meta['semantics'], 'frozen_cellwise_classifier_thinning_mc')
            self.assertEqual(meta['probability_method'], 'monte_carlo')
            self.assertEqual(meta['independence_basis'], 'caller_acknowledged_assumption_not_proved')

    def test_vector_retention_runs_and_preserves_values_and_gene_order(self):
        obj = make_adata([[2, 1], [1, 2], [2, 2], [1, 1]])
        before = obj.X.copy()
        df = self.audit(obj, p=[0.5, 0.7], classifier_fn=nonlinear_classifier,
                        classifier_draws=256, inplace=True)
        row = df.iloc[0]
        self.assertEqual(row['cv_status'], 'RESOLVED')
        self.assertEqual(row['retention_kind'], 'gene_specific')
        self.assertEqual(json.loads(row['p'].split(':', 1)[1]), [0.5, 0.7])
        self.assertEqual(row['retention_curve_status'], 'NOT_EVALUATED_NONSCALAR_RETENTION')
        self.assertFalse(row['is_non_monotonic_trajectory'])
        self.assertTrue(pd.isna(row['max_adjacent_p_plus_increase']))
        run = obj.uns['spatial_membership_audit']['provenance']['probability_runs']['retention_0']
        np.testing.assert_array_equal(run['values'], [0.5, 0.7])
        self.assertEqual(run['genes'], ['G0', 'G1'])
        self.assertEqual((before != obj.X).nnz, 0)

    def test_downstream_unresolved_is_not_promoted_to_a_guarantee(self):
        obj = make_adata([[2, 1], [1, 2], [2, 2], [1, 1]])
        with patch('spatial_membership_audit.standard_ecosystem.estimate_cv',
                   return_value=dict(status='UNRESOLVED', cv=None, cv_interval=None)):
            row = self.audit(obj, p=0.5, classifier_fn=nonlinear_classifier).iloc[0]
        self.assertEqual(row['cv_status'], 'UNRESOLVED')
        self.assertEqual(row['conditional_guarantee'], 'NONE')
        self.assertEqual(row['guarantee'], 'NONE')
        self.assertEqual(row['regime'], 'NUMERICAL_UNRESOLVED')

    def test_scalar_grid_records_each_actual_seed_and_exact_endpoints(self):
        df = self.audit(p_grid=[1.0, 0.5, 0.0], seed=41)
        self.assertEqual(df['seed'].tolist(), [41, 10048, 20055])
        self.assertEqual(df['n_draws'].tolist(), [0, 512, 0])
        self.assertEqual(df['probability_method'].tolist(), ['deterministic', 'monte_carlo', 'deterministic'])
        runs = df.attrs['provenance']['probability_runs']
        self.assertEqual(len(runs), 3)
        for i, row in df.iterrows():
            self.assertEqual(runs[f'retention_{i}']['seed'], row['seed'])
            self.assertEqual(runs[f'retention_{i}']['n_draws'], row['n_draws'])

    def test_vector_result_tables_can_be_combined(self):
        obj = make_adata([[2, 1], [1, 2], [2, 2], [1, 1]])
        frames = [self.audit(obj, p=p, classifier_fn=nonlinear_classifier,
                             classifier_draws=128, cv_mode='delta')
                  for p in ([0.5, 0.7], [0.3, 0.4])]
        combined = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(combined), 2)
        self.assertNotEqual(combined.iloc[0]['p'], combined.iloc[1]['p'])
        # Different runs retain their own provenance; pandas may drop unequal
        # attrs on concat, so row metadata and original tables remain available.
        self.assertEqual(frames[0].attrs['provenance']['probability_runs']['retention_0']['values'], [0.5, 0.7])

    def test_exact_scalar_and_vector_endpoint_zeros_stay_exact(self):
        for p in (0.0, [0.0, 0.0]):
            row = self.audit(p=p).iloc[0]
            self.assertEqual(row['n_draws'], 0)
            self.assertFalse(row['p_finite_is_estimate'])
            self.assertEqual(row['regime'], 'STRUCTURAL_ABSENT_POPULATION')
            self.assertEqual(row['guarantee'], 'STRUCTURAL_STATE_ONLY')
        for p in (1.0, [1.0, 1.0]):
            row = self.audit(p=p).iloc[0]
            self.assertEqual(row['n_draws'], 0)
            self.assertEqual(row['engine'], 'BASELINE')
            self.assertEqual(row['cv_status'], 'SKIPPED_P1')
            self.assertEqual(row['cv'], 0.0)
            self.assertEqual(row['guarantee'], 'BASELINE_OBSERVED')

    def test_no_graph_edges_is_still_a_known_structural_state(self):
        row = self.audit(make_adata(adjacency=np.zeros((2, 2))), p=0.5).iloc[0]
        self.assertEqual(row['regime'], 'STRUCTURAL_NO_GRAPH_EDGES')
        self.assertNotEqual(row['cv_status'], 'UNRESOLVED_UPSTREAM_SAMPLING')

    def test_unobserved_contact_is_not_certified_zero(self):
        def classify(x, genes):
            return np.where((x[:, 0] == 20) | (x[:, 2] > 0), 'A',
                            np.where(x[:, 1] > 0, 'B', 'OTHER'))
        # The isolated third node ensures P+ is positive even when no A is
        # observed at node 0, which is the only possible A endpoint of the edge.
        obj = make_adata([[20, 0, 0], [0, 20, 0], [0, 0, 1]],
                         [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        row = self.audit(obj, p=[0.5, 0.5, 1.0], classifier_fn=classify).iloc[0]
        self.assertGreater(row['p_finite'], 0)
        self.assertEqual(row['mu_O_finite'], 0.0)
        self.assertEqual(row['conditional_cv_status'], 'ZERO_NUMERATOR_CV_UNDEFINED')
        self.assertEqual(row['cv_status'], 'UNRESOLVED_UPSTREAM_SAMPLING')
        self.assertEqual(row['regime'], 'NUMERICAL_UNRESOLVED')

    def test_invalid_retention_and_vector_grids_fail_before_classification(self):
        def must_not_be_called(x, genes):
            self.fail('invalid retention should be rejected before classifier evaluation')
        for kwargs in ({'p': [0.5]}, {'p': [0.5, 1.1]}, {'p': [np.nan, 0.5]},
                       {'p_grid': [[0.5, 0.7], [0.2, 0.3]]}, {'p_grid': []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.audit(classifier_fn=must_not_be_called, **kwargs)

    def test_guide_respects_mc_metadata_even_with_raw_conditional_status(self):
        for status in ('NO_FINITE_RATIO', 'ZERO_NUMERATOR_CV_UNDEFINED'):
            row = triage_audit_record(dict(m=1, p_finite=0.0, cv=None,
                cv_status=status, probability_method='monte_carlo'))
            self.assertEqual(row['regime'], 'NUMERICAL_UNRESOLVED')
            self.assertEqual(row['action_code'], 'RESOLVE_CLASSIFIER_MC_SUPPORT')
        # A vector record must not inherit a non-monotonic scalar curve flag.
        rows = [dict(region='R', label_a='A', label_b='B', p=p, p_finite=prob,
                     cv=0.3, cv_status='RESOLVED') for p, prob in [(1.0, 0.2), (0.5, 0.8)]]
        rows.append(dict(rows[-1], p='vector:[0.5,0.7]', retention_kind='gene_specific'))
        triaged = triage_audit_table(rows)
        self.assertTrue(triaged[1]['is_non_monotonic_trajectory'])
        self.assertFalse(triaged[2]['is_non_monotonic_trajectory'])
        self.assertEqual(triaged[2]['retention_curve_status'], 'NOT_EVALUATED_NONSCALAR_RETENTION')

    def test_anndata_roundtrip_preserves_uncertainty_and_vector_provenance(self):
        # Saving an all-NA CV column must not turn it into object/string data.
        cases = [(make_adata(), dict(p=0.5)),
                 (make_adata([[2, 1], [1, 2], [2, 2], [1, 1]]),
                  dict(p=[0.5, 0.7], classifier_fn=nonlinear_classifier))]
        for obj, kwargs in cases:
            self.audit(obj, inplace=True, **kwargs)
            with tempfile.TemporaryDirectory() as folder:
                path = pathlib.Path(folder) / 'audit.h5ad'
                obj.write_h5ad(path)
                saved = ad.read_h5ad(path)
            info = saved.uns['spatial_membership_audit']
            run = info['provenance']['probability_runs']['retention_0']
            self.assertEqual(run['n_draws'], 512)
            self.assertEqual(run['seed'], 1729)
            self.assertIn('not included', run['upstream_probability_uncertainty'])
            if np.ndim(kwargs['p']) == 0:
                self.assertEqual(info['table'].iloc[0]['cv_status'], 'UNRESOLVED_UPSTREAM_SAMPLING')
            else:
                np.testing.assert_array_equal(run['values'], [0.5, 0.7])
                np.testing.assert_array_equal(run['genes'], ['G0', 'G1'])

    def test_low_level_exact_and_mc_metadata_are_explicit(self):
        for p in (0.0, 0.5, 1.0):
            result = frozen_classifier_probabilities(np.array([[20, 0], [0, 20]]),
                ['G0', 'G1'], np.ones(2, bool), p, classifier_fn=rare_classifier,
                classes=['A', 'B'], assume_cellwise=True, n_draws=32, seed=7)
            self.assertEqual(result['probability_method'], 'monte_carlo' if p == 0.5 else 'deterministic')
            self.assertEqual(result['n_draws'], 32 if p == 0.5 else 0)
            self.assertEqual(result['seed'], 7)
            self.assertIn('upstream_probability_uncertainty', result)


if __name__ == '__main__':
    unittest.main()
