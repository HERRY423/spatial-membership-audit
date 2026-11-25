import copy
import unittest
import numpy as np
from spatial_membership_audit.ecosystem import SpatialGraph
from spatial_membership_audit.native_validation import validate_native_output, audit_native_outputs


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.g=SpatialGraph(('x','y','z'),('r',)*3,[[0,1,0],[1,0,1],[0,1,0]],False,'test')
        self.classes=('A','B','OTHER')
        self.record={'node_ids':('x','y','z'),'classes':self.classes,'values':['A','B','OTHER'],
                     'semantics':'hard_labels','provenance':'fixture, not native scientific evidence'}
        self.kw=dict(graph=self.g,classes=self.classes,pair=('A','B'),semantics='hard_labels')
    def test_independent_node_order_per_realization(self):
        changed=dict(self.record,node_ids=('z','x','y'),values=['OTHER','A','B'])
        r=audit_native_outputs(baseline=self.record,outputs={'r':changed},failed={},not_run={},
            planned_ids=['r'],provenance='test',**self.kw)
        self.assertEqual(r['rows'][0]['delta_R'],0)
        self.assertEqual(r['rows'][0]['O'],1)
        self.assertAlmostEqual(r['rows'][0]['E'],2/3)
    def test_failed_not_run_invalid_and_undefined_have_separate_counts(self):
        bad=dict(self.record,node_ids=('x','y','missing'))
        empty=dict(self.record,values=['A','A','A'])
        zero=dict(self.record,values=['A','OTHER','B'])
        r=audit_native_outputs(baseline=self.record,outputs={'bad':bad,'na':empty,'zero':zero},
            failed={'fail':'fit failed'},not_run={'pending':'budget not authorized'},
            planned_ids=['bad','na','zero','fail','pending'],provenance='test',**self.kw)
        s=r['summary']
        self.assertEqual((s['planned'],s['invalid_outputs'],s['fit_failed'],s['not_run'],s['undefined_accepted'],s['finite_zero']), (5,1,1,1,1,1))
        self.assertEqual(s['finite_fraction_of_planned'],.2)
        self.assertEqual(s['finite_fraction_of_accepted'],.5)
        self.assertIsNone(s['cv_R_finite'])
    def test_no_silent_schedule_omission(self):
        with self.assertRaises(ValueError):
            audit_native_outputs(baseline=self.record,outputs={},failed={},not_run={},planned_ids=['r'],provenance='test',**self.kw)
    def test_invalid_baseline_does_not_invent_delta(self):
        bad=dict(self.record,node_ids=('x','y','wrong'))
        r=audit_native_outputs(baseline=bad,outputs={'r':self.record},failed={},not_run={},planned_ids=['r'],provenance='test',**self.kw)
        self.assertEqual(r['baseline']['status'],'INVALID_OUTPUT')
        self.assertEqual(r['rows'][0]['status'],'FINITE')
        self.assertIsNone(r['rows'][0]['delta_R'])
    def test_duplicate_nodes_unknown_labels_missing_class_and_semantics_rejected(self):
        bads=[dict(self.record,node_ids=('x','x','z')),dict(self.record,values=['A','UNKNOWN','B']),
              dict(self.record,classes=('A','B')),dict(self.record,semantics='composition')]
        for bad in bads:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_native_output(bad,realization_id='r',**self.kw)
    def test_continuous_field_axes_and_off_diagonal_reference(self):
        g=SpatialGraph(('x','y','z'),('r',)*3,[[0,1,0],[1,0,1],[0,1,0]],False,'test')
        values=np.array([[.8,.2],[.2,.8],[.5,.5]])
        record=dict(node_ids=('x','y','z'),classes=('A','B'),values=values,semantics='composition',provenance='test')
        kw=dict(graph=g,classes=('A','B'),pair=('A','B'),semantics='composition',realization_id='r')
        a=validate_native_output(record,**kw)
        b=validate_native_output(dict(record,node_ids=('z','x','y'),classes=('B','A'),values=values[[2,0,1]][:,[1,0]]),**kw)
        self.assertAlmostEqual(a['O'],1.18)
        self.assertAlmostEqual(a['E'],1.12)
        self.assertAlmostEqual(a['R'],b['R'])
        self.assertNotEqual(a['O'],1)  # No implicit argmax categorization.
    def test_bad_composition_not_normalized_automatically(self):
        for values in ([[.8,.3],[.2,.8],[.5,.5]], [[1,0],[0,1],[float('nan'),0]], [[1,0],[0,1],[-.1,1.1]]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                validate_native_output(dict(node_ids=self.g.cell_ids,classes=('A','B'),values=values,semantics='composition',provenance='test'),
                    graph=self.g,classes=('A','B'),pair=('A','B'),semantics='composition',realization_id='r')
    def test_joint_dependence_is_not_replaced_by_marginals(self):
        g=SpatialGraph(('x','y'),('r','r'),[[0,1],[1,0]],False,'test')
        def rec(v): return dict(node_ids=g.cell_ids,classes=('A','B'),values=v,semantics='hard_labels',provenance='test')
        kw=dict(baseline=rec(['A','B']),graph=g,classes=('A','B'),pair=('A','B'),semantics='hard_labels',provenance='test',planned_ids=['0','1'],failed={},not_run={})
        same=audit_native_outputs(outputs={'0':rec(['A','A']),'1':rec(['B','B'])},**kw)
        split=audit_native_outputs(outputs={'0':rec(['A','B']),'1':rec(['B','A'])},**kw)
        self.assertEqual(same['summary']['finite'],0)
        self.assertEqual(split['summary']['finite'],2)
    def test_one_field_cannot_supply_sd(self):
        rec=dict(node_ids=self.g.cell_ids,classes=('A','B'),values=[[.8,.2],[.2,.8],[.5,.5]],semantics='composition',provenance='test')
        r=audit_native_outputs(baseline=rec,outputs={'r':rec},failed={},not_run={},planned_ids=['r'],graph=self.g,
             classes=('A','B'),pair=('A','B'),semantics='composition',provenance='test')
        self.assertIsNone(r['summary']['sd_R_finite'])
        self.assertIsNone(r['summary']['cv_R_finite'])


if __name__=='__main__':unittest.main()
