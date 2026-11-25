import csv,itertools,json,tempfile,unittest
from pathlib import Path
import numpy as np
from spatial_membership_audit.core import endpoints,membership,run

class AuditTests(unittest.TestCase):
    def test_expectation_against_exhaustive_label_assignments(self):
        xy=np.array([[0,0],[1,0],[2,0],[10,0]],float)
        assignments=set(itertools.permutations(['A','B','B','X']))
        values=[endpoints(xy,np.array(a),'A','B',1.1) for a in assignments]
        self.assertAlmostEqual(sum(x['O'] for x in values)/len(values),values[0]['E'])
        self.assertAlmostEqual(values[0]['E'],2/3)

    def test_nonzero_targets_without_edges(self):
        e=endpoints(np.array([[0,0],[100,0]],float),np.array(['A','B']),'A','B',1)
        self.assertIsNone(e['OE']);self.assertEqual(e['G_a_to_b'],0)
        self.assertEqual(e['OE_status'],'NO_GRAPH_EDGES')

    def test_absent_target_is_not_zero_proximity(self):
        e=endpoints(np.array([[0,0],[1,0]],float),np.array(['A','A']),'A','B',2)
        self.assertIsNone(e['OE']);self.assertIsNone(e['G_a_to_b'])
        self.assertIn('NO_B',e['OE_status'])

    def test_full_membership_replacement_with_unchanged_abundance(self):
        m=membership({'a','b'},{'c','d'})
        self.assertEqual((m['baseline_count'],m['current_count'],m['jaccard'],m['lost'],m['gained']),(2,2,0,2,2))
        self.assertIsNone(membership(set(),set())['jaccard'])

    def test_directed_G_is_not_OE_or_symmetric(self):
        xy=np.array([[0,0],[10,0],[.2,0]],float)
        e=endpoints(xy,np.array(['A','A','B']),'A','B',1)
        self.assertEqual(e['G_a_to_b'],.5);self.assertEqual(e['G_b_to_a'],1)
        self.assertAlmostEqual(e['OE'],1.5)

    def fixture(self,p,missing=False):
        (p/'cells.tsv').write_text('cell_id\tregion_id\tx_um\ty_um\na\tr1\t0\t0\nb\tr1\t1\t0\nc\tr2\t0\t0\nd\tr2\t1\t0\n',encoding='utf-8')
        records=[('a','baseline','A'),('b','baseline','B'),('c','baseline','A'),('d','baseline','B'),('a','r0','A'),('b','r0','UNKNOWN'),('c','r0','A'),('d','r0','B')]
        if missing:records.pop()
        (p/'labels.tsv').write_text('cell_id\treplicate\tlabel\n'+'\n'.join('\t'.join(r) for r in reversed(records))+'\n',encoding='utf-8')

    def test_cli_contract_na_and_region_separation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            rows=run(p/'cells.tsv',p/'labels.tsv',['A,B'],2,p/'output')
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['n_edges']==1 for r in rows))
            affected=next(r for r in rows if r['region_id']=='r1' and r['replicate']=='r0')
            self.assertEqual(affected['paired_OE_status'],'UNESTIMABLE_PAIR')
            self.assertIsNone(affected['delta_OE'])
            self.assertIn('NA',(p/'output/endpoints.tsv').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):run(p/'cells.tsv',p/'labels.tsv',['A,B'],2,p/'output')

    def test_missing_labels_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,True)
            with self.assertRaises(ValueError):run(p/'cells.tsv',p/'labels.tsv',['A,B'],2,p/'output')
            self.assertFalse((p/'output').exists())

    def test_invalid_radius_and_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            for radius in [0,-1,float('nan'),float('inf')]:
                with self.assertRaises(ValueError):run(p/'cells.tsv',p/'labels.tsv',['A,B'],radius,p/'output')
            with self.assertRaises(ValueError):run(p/'cells.tsv',p/'labels.tsv',['A,B','B,A'],2,p/'output')

if __name__=='__main__':unittest.main()
