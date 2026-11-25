import unittest
from unittest.mock import patch
from types import SimpleNamespace
import anndata as ad
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from spatialdata.models import ShapesModel
from spatialdata.transformations import Translation
from spatial_membership_audit.standard_ecosystem import audit_spatialdata, extract_anndata_from_spatialdata
from spatial_membership_audit.action_guide import triage_audit_record
from spatial_membership_audit.custom_rules import frozen_classifier_probabilities

class ResolutionRegressions(unittest.TestCase):
    def make(self):
        a=ad.AnnData(np.ones((2,1)),obs=pd.DataFrame({'instance_id':[7,3]},index=['a','b']))
        a.obsm['spatial']=np.zeros((2,2))
        shapes=ShapesModel.parse(gpd.GeoDataFrame({'radius':[1.,1.]},geometry=[Point(1,2),Point(4,6)],index=[3,7]),
                                transformations={'translated':Translation([10,20],axes=('x','y'))})
        return SimpleNamespace(tables={'table':a},shapes={'cells':shapes})
    def test_coordinate_request_fails_closed(self):
        s=self.make()
        with self.assertRaisesRegex(ValueError,'Cannot transform'):
            extract_anndata_from_spatialdata(s,shapes_key='cells',coordinate_system='missing')
        with self.assertRaisesRegex(ValueError,'cannot be resolved'):
            extract_anndata_from_spatialdata(s,shapes_key='absent',coordinate_system='translated')
        np.testing.assert_array_equal(s.tables['table'].obsm['spatial'],0)
    def test_missing_coordinate_system_with_cached_spatial_data(self):
        # Cached coordinates/graph must not bypass an explicitly requested transform.
        for has_coordinates in (False, True):
            for shapes_key in (None, 'cells'):
                with self.subTest(has_coordinates=has_coordinates, shapes_key=shapes_key):
                    s = self.make()
                    if not has_coordinates:
                        del s.tables['table'].obsm['spatial']
                    graph = np.array([[0., 1.], [1., 0.]])
                    s.tables['table'].obsp['spatial_connectivities'] = graph.copy()
                    with self.assertRaisesRegex(
                        ValueError, "Cannot transform.*'cells'.*'missing'"
                    ) as caught:
                        extract_anndata_from_spatialdata(
                            s, shapes_key=shapes_key, coordinate_system='missing'
                        )
                    self.assertIsNotNone(caught.exception.__cause__)
                    self.assertEqual('spatial' in s.tables['table'].obsm, has_coordinates)
                    if has_coordinates:
                        np.testing.assert_array_equal(s.tables['table'].obsm['spatial'], 0)
                    np.testing.assert_array_equal(s.tables['table'].obsp['spatial_connectivities'], graph)
                    self.assertNotIn('_sma_coordinate_provenance', s.tables['table'].uns)
    def test_transform_failure_preserves_original_error(self):
        s = self.make()
        original = RuntimeError('synthetic coordinate conversion failure')
        with patch('spatialdata.transform', side_effect=original) as transform:
            with self.assertRaisesRegex(
                ValueError, "Cannot transform.*'cells'.*'translated'"
            ) as caught:
                extract_anndata_from_spatialdata(
                    s, shapes_key='cells', coordinate_system='translated'
                )
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual(transform.call_count, 1)
        self.assertIs(transform.call_args.args[0], s.shapes['cells'])
        self.assertEqual(transform.call_args.kwargs['to_coordinate_system'], 'translated')
        np.testing.assert_array_equal(s.tables['table'].obsm['spatial'], 0)
        np.testing.assert_array_equal(s.shapes['cells'].geometry.x, [1., 4.])
        np.testing.assert_array_equal(s.shapes['cells'].geometry.y, [2., 6.])
    def test_audit_stops_before_analysis_on_coordinate_failure(self):
        s = self.make()
        original = RuntimeError('synthetic coordinate conversion failure')
        with patch('spatialdata.transform', side_effect=original):
            with patch('spatial_membership_audit.standard_ecosystem.audit_anndata') as audit:
                with self.assertRaisesRegex(ValueError, 'Cannot transform') as caught:
                    audit_spatialdata(
                        s, shapes_key='cells', coordinate_system='translated', inplace=True
                    )
                audit.assert_not_called()
        self.assertIs(caught.exception.__cause__, original)
        self.assertNotIn('spatial_membership_audit', s.tables['table'].uns)
        self.assertNotIn('_sma_coordinate_provenance', s.tables['table'].uns)
        np.testing.assert_array_equal(s.tables['table'].obsm['spatial'], 0)
    def test_real_transform_alignment_and_no_mutation(self):
        s=self.make()
        a=extract_anndata_from_spatialdata(s,coordinate_system='translated')
        np.testing.assert_allclose(a.obsm['spatial'],[[14,26],[11,22]])
        self.assertTrue(a.uns['_sma_coordinate_provenance']['transform_applied'])
        np.testing.assert_array_equal(s.tables['table'].obsm['spatial'],0)
        s.tables['table'].obs['instance_id']=[7,7]
        with self.assertRaisesRegex(ValueError,'unique'):
            extract_anndata_from_spatialdata(s,shapes_key='cells')
    def test_unknown_availability_is_not_robust(self):
        self.assertEqual(triage_audit_record({'cv':.01,'status':'RESOLVED'})['regime'],'NUMERICAL_UNRESOLVED')
        self.assertEqual(triage_audit_record({'cv':.01,'P_plus':1e-14,'status':'RESOLVED'})['regime'],'TIER_5A_CONDITIONING_TRAP')
    def test_middle_cell_dependency_detected_even_when_acknowledged(self):
        def clf(x,genes):
            v=x[:,0]
            return np.where(v<=1,'B',np.where(v>=9,'A',np.where(v>v.mean(),'A','B')))
        kwargs=dict(counts=np.array([[0],[4],[6],[10]]),genes=['G'],eligible=np.ones(4,bool),
                    p=.5,classifier_fn=clf,classes=['A','B'],n_draws=16)
        with self.assertRaisesRegex(ValueError,'assume_cellwise'):frozen_classifier_probabilities(**kwargs)
        with self.assertRaisesRegex(ValueError,'independence'):frozen_classifier_probabilities(**kwargs,assume_cellwise=True)

if __name__=='__main__':unittest.main()
