import itertools
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse
from spatial_membership_audit.ecosystem import SpatialGraph, categorical_input
from spatial_membership_audit.posterior_bridge import (
    joint_label_summary, continuous_field_summary, cell2location_field_samples,
    rctd_composition_field, tangram_cell_mapping_field)


class PosteriorBridgeTests(unittest.TestCase):
    def test_supplied_native_coordinate_graph_has_no_length_claim(self):
        adjacency = [[0,1,0],[1,0,1],[0,1,0]]
        kw = dict(classes=['A','B'],pair=('A','B'),realization_ids=['one'],
                  mechanism='point_estimate',value_semantics='composition',
                  provenance='synthetic contract test')
        values = np.array([[[.8,.2],[.3,.7],[.6,.4]]])
        outputs=[]
        for unit in ['um','native']:
            g=SpatialGraph(('x','y','z'),('r',)*3,adjacency,False,
                           'supplied fixed adjacency with explicit unit',unit)
            self.assertEqual(g.coordinate_unit,unit)
            outputs.append(continuous_field_summary(values,['x','y','z'],g,**kw))
        self.assertEqual(outputs[0],outputs[1])
        for unit in ['',None,'unknown']:
            with self.assertRaises(ValueError):
                SpatialGraph(('x','y','z'),('r',)*3,adjacency,False,'supplied',unit)

    def setUp(self):
        self.g = SpatialGraph(('x','y','z'), ('r',)*3, [[0,1,0],[1,0,1],[0,1,0]], False, 'synthetic')
        self.kw = dict(classes=['A','B','OTHER'], pair=('A','B'),
                       realization_ids=['d0','d1'], mechanism='posterior_joint_labels', provenance='synthetic')

    def test_joint_dependence_not_replaced_by_marginals(self):
        # Both distributions have exactly the same per-node marginals.
        g = SpatialGraph(('x','y'), ('r',)*2, [[0,1],[1,0]], False, 'synthetic')
        same = joint_label_summary([['A','A'],['B','B']], ['x','y'], g, **self.kw)
        opposite = joint_label_summary([['A','B'],['B','A']], ['x','y'], g, **self.kw)
        self.assertEqual(same['summary']['finite_fraction'], 0)
        self.assertEqual(opposite['summary']['finite_fraction'], 1)
        self.assertEqual(opposite['summary']['cv_R_finite'], 0)

    def test_alignment_and_finite_zero(self):
        out = joint_label_summary([['B','OTHER','A'],['A','OTHER','A']], ['z','y','x'], self.g, **self.kw)
        self.assertEqual(out['rows'][0]['status'], 'FINITE_ZERO')
        self.assertEqual(out['rows'][1]['status'], 'NO_FINITE_RATIO')
        self.assertIsNone(out['summary']['cv_R_finite'])
        self.assertEqual(out['summary']['undefined'], 1)

    def test_no_silent_drop_or_unknown(self):
        for labels, ids in [([['A','B'],['A','B']], ['x','y']),
                            ([['A',None,'B'],['A','B','A']], ['x','y','z'])]:
            with self.assertRaises(ValueError): joint_label_summary(labels,ids,self.g,**self.kw)

    def test_region_isolation(self):
        g = SpatialGraph(('x','y'), ('r','s'), [[0,0],[0,0]], False, 'synthetic')
        with self.assertRaises(ValueError): joint_label_summary([['A','B'],['B','A']], ['x','y'], g, **self.kw)

    def field(self, x, mechanism='posterior_joint_fields'):
        return continuous_field_summary(x, ['x','y','z'], self.g, classes=['A','B'], pair=('A','B'),
            realization_ids=[f'd{i}' for i in range(len(x))], mechanism=mechanism,
            value_semantics='cell_abundance', provenance='synthetic')

    def test_continuous_denominator_by_full_permutation(self):
        x = np.array([[2.,3.],[.5,1.],[4.,.25]])
        observed = self.field(x[None])['rows'][0]
        all_o = [x[list(ix),0] @ self.g.adjacency @ x[list(ix),1]
                 for ix in itertools.permutations(range(3))]
        self.assertAlmostEqual(observed['E'], np.mean(all_o))
        self.assertAlmostEqual(observed['O'], x[:,0] @ self.g.adjacency @ x[:,1])

    def test_one_hot_reduces_to_label_endpoint(self):
        x = np.array([[[1,0],[0,1],[1,0]], [[0,0],[0,0],[0,0]]])
        field = self.field(x)
        labels = joint_label_summary([['A','B','A'],['OTHER']*3], ['x','y','z'], self.g, **self.kw)
        for l, f in zip(labels['rows'],field['rows']):
            self.assertEqual((l['O'],l['E'],l['R']), (f['O'],f['E'],f['R']))

    def test_cell2location_native_draw_contract(self):
        x=np.array([[[2,1],[3,4],[1,2]], [[4,2],[6,8],[2,4]]],float)
        wrapped=cell2location_field_samples(x,sample_ids=['s0','s1'],spot_ids=['z','y','x'],class_names=['A','B'],joint_draw_provenance='synthetic joint draws')
        out=continuous_field_summary(graph=self.g,pair=('A','B'),**wrapped)
        self.assertAlmostEqual(out['rows'][0]['R'],out['rows'][1]['R'])
        self.assertEqual(out['summary']['mechanism'],'posterior_joint_fields')
        with self.assertRaises(ValueError): cell2location_field_samples(x.mean(0),sample_ids=['mean'],spot_ids=['z','y','x'],class_names=['A','B'],joint_draw_provenance='not draws')

    def test_rctd_point_is_not_posterior(self):
        df=pd.DataFrame([[.2,.8],[.6,.4],[.5,.5]],index=['z','y','x'],columns=['A','B'])
        wrap=rctd_composition_field(df,provenance='synthetic schema')
        out=continuous_field_summary(graph=self.g,pair=('A','B'),**wrap)
        self.assertEqual(out['uncertainty_status'],'NOT_AVAILABLE_FROM_POINT_ESTIMATE')
        self.assertIsNone(out['summary']['cv_R_finite'])
        with self.assertRaises(ValueError): rctd_composition_field(df*2,provenance='raw weights')

    def test_tangram_orientation_and_mass(self):
        obj=SimpleNamespace(X=sparse.csr_matrix([[.8,.1,.1],[.2,.3,.5]]),obs_names=pd.Index(['c1','c2']),var_names=pd.Index(['x','y','z']))
        labels=pd.Series(['B','A'],index=['c2','c1'])
        wrap=tangram_cell_mapping_field(obj,labels,provenance='synthetic cell-mode',mode='cells')
        np.testing.assert_allclose(wrap['values'][0],[[.8,.2],[.1,.3],[.1,.5]])
        np.testing.assert_allclose(wrap['values'].sum(axis=1),[[1,1]])
        with self.assertRaises(ValueError): tangram_cell_mapping_field(obj,labels,provenance='synthetic',mode='clusters')

    def test_duplicate_and_nonfinite_refused(self):
        with self.assertRaises(ValueError): self.field(np.full((2,3,2),np.nan))
        with self.assertRaises(ValueError): self.field(-np.ones((2,3,2)))
        with self.assertRaises(ValueError): self.field(np.ones((2,3,2)),mechanism='point_estimate')


if __name__ == '__main__': unittest.main()
