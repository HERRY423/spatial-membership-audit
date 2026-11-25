import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from spatial_membership_audit.definitions import REQUIRED
from spatial_membership_audit.analytic_adapters import predict_anndata,summarize_trajectory


class AdapterTests(unittest.TestCase):
    def test_anndata_preserves_regions(self):
        import anndata as ad
        x=np.ones((4,11),dtype=int)
        a=ad.AnnData(X=sparse.csr_matrix(x),obs=pd.DataFrame({'region':['x','x','y','y'],
            'eligible':[True]*4},index=['a','b','c','d']),var=pd.DataFrame(index=REQUIRED))
        a.layers['counts']=a.X.copy()
        a.obsp['spatial_connectivities']=sparse.block_diag([[[0,1],[1,0]]]*2,format='csr')
        result=predict_anndata(a,p=.5,count_layer='counts',eligible_key='eligible',region_key='region')
        self.assertEqual(len(result),12)
        self.assertTrue(all(x['n']==2 for x in result))
        graph=a.obsp['spatial_connectivities'].tolil();graph[0,2]=graph[2,0]=1
        a.obsp['spatial_connectivities']=graph.tocsr()
        with self.assertRaises(ValueError):
            predict_anndata(a,p=.5,count_layer='counts',eligible_key='eligible',region_key='region')

    def test_signed_scores_and_missingness(self):
        r=summarize_trajectory([-1,0,1,np.nan],nonnegative_ratio=False)
        self.assertEqual((r['finite'],r['nonfinite'],r['zero']),(3,1,1))
        self.assertIsNone(r['cv'])
        self.assertIsNone(summarize_trajectory([0,0],nonnegative_ratio=True)['cv'])
        with self.assertRaises(ValueError):summarize_trajectory([-1],nonnegative_ratio=True)
