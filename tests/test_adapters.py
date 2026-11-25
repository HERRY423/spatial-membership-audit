import tempfile,unittest
from pathlib import Path
import numpy as np
import h5py
from spatial_membership_audit.adapters import validate_counts,tenx_metadata,iter_tenx_cells,iter_cosmx_cells
from spatial_membership_audit.definitions import REQUIRED,candidate_labels,panel_contract

class AdapterTests(unittest.TestCase):
    def fixture(self,p):
        with h5py.File(p,'w') as f:
            g=f.create_group('matrix');g['shape']=[3,2];g['data']=[2,1,4];g['indices']=[0,2,1];g['indptr']=[0,2,3]
            g['barcodes']=np.array(['a','b'],dtype='S');g['features/name']=np.array(['CD3D','MS4A1','NegPrb1'],dtype='S');g['features/feature_type']=np.array(['Gene Expression','Gene Expression','Negative Control Probe'],dtype='S')

    def test_tenx_orientation_and_control_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5';self.fixture(p);m=tenx_metadata(p)
            self.assertEqual(m['barcodes'].tolist(),['a','b'])
            actual=np.vstack([c.toarray() for _,_,c in iter_tenx_cells(p,batch_cells=1)])
            np.testing.assert_array_equal(actual,[[2,0,1],[0,4,0]])
            self.assertEqual(int(actual.sum()),7)

    def test_corrupt_sparse_layout_and_duplicate_barcode_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5';self.fixture(p)
            with h5py.File(p,'r+') as f:f['matrix/indptr'][2]=100
            with self.assertRaises(ValueError):list(iter_tenx_cells(p))
            with h5py.File(p,'r+') as f:f['matrix/indptr'][2]=3;f['matrix/barcodes'][1]=b'a'
            with self.assertRaises(ValueError):tenx_metadata(p)

    def test_invalid_counts_rejected(self):
        for data in [[-1],[.5],[np.nan],[np.inf],[float(2**63)],np.array([2**63],dtype=np.uint64)]:
            with self.assertRaises(ValueError):validate_counts(data)
        validate_counts(np.array([0,2**63-1],dtype=np.int64))

    def test_cosmx_controls_keys_and_header_collision(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.csv';p.write_text('fov,cell_ID,cell,CD3D,NegPrb1,SystemControl1\n1,2,c_1_2,3,1,5\n')
            c,f=next(iter_cosmx_cells(p));self.assertEqual(f['genes'],['CD3D']);self.assertEqual(c.cell.iloc[0],'c_1_2')
            p.write_text('fov,cell_ID,CD3D,CD3D\n1,2,3,4\n')
            with self.assertRaises(ValueError):list(iter_cosmx_cells(p))

    def test_missing_gene_is_not_zero_expression(self):
        genes=list(REQUIRED);genes.remove('CD4')
        self.assertEqual(panel_contract(genes)['status'],'NOT_IDENTIFIABLE')
        with self.assertRaises(ValueError):candidate_labels(np.zeros((1,len(genes))),genes,np.array([True]))

    def test_positive_cd4_and_conflicts_and_frozen_eligibility(self):
        genes=list(REQUIRED);x=np.zeros((5,len(genes)),dtype=int)
        def setrow(i,values):
            for k,v in values.items():x[i,genes.index(k)]=v
        setrow(0,dict(CD3D=2,CD3E=1));setrow(1,dict(CD3D=2,CD3E=1,CD4=1));setrow(2,dict(CD3D=2,CD3E=1,CD4=1,CD8A=2));setrow(3,dict(MS4A1=1,CD79A=1,MZB1=1,TNFRSF17=1));setrow(4,dict(CD14=2,CD68=2))
        actual=candidate_labels(x,genes,np.array([True,True,True,True,False]))
        self.assertEqual(actual.tolist(),['T_UNRESOLVED','CD4_T_CANDIDATE','AMBIGUOUS','AMBIGUOUS','LOW_DETECTION_BASELINE'])
        self.assertEqual(candidate_labels(x*0,genes,np.array([True,True,True,True,False]))[-1],'LOW_DETECTION_BASELINE')

if __name__=='__main__':unittest.main()
