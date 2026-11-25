import unittest
from spatial_membership_audit.ecosystem import SpatialGraph
from spatial_membership_audit.refit_audit import audit_refits


class RefitAuditTests(unittest.TestCase):
    def setUp(self):
        self.g = SpatialGraph(('x','y','z'), ('r',)*3, [[0,1,0],[1,0,1],[0,1,0]], False, 'synthetic')
        self.kw = dict(baseline=['A','B','OTHER'], node_ids=self.g.cell_ids, graph=self.g,
                       classes=['A','B','OTHER'], pair=('A','B'), provenance='synthetic')

    def test_failure_na_and_zero_have_distinct_denominators(self):
        r = audit_refits(realizations={'zero':['A','OTHER','B'], 'na':['A']*3},
                         failures={'fail':'native convergence failure'}, planned_ids=['zero','na','fail'], **self.kw)
        s=r['summary']
        self.assertEqual((s['planned'],s['failed'],s['finite'],s['undefined_successful'],s['zero']), (3,1,1,1,1))
        self.assertEqual(s['finite_fraction_of_planned'],1/3)
        self.assertEqual(s['finite_fraction_given_success'],1/2)
        self.assertIsNone(s['cv_R_finite'])

    def test_dropped_or_overlapping_runs_rejected(self):
        for runs,fails in [({},{}),({'r':['A','B','OTHER']},{'r':'failed'})]:
            with self.assertRaises(ValueError): audit_refits(realizations=runs,failures=fails,planned_ids=['r'],**self.kw)

    def test_all_failures_retained(self):
        r=audit_refits(realizations={},failures={'r':'timeout'},planned_ids=['r'],**self.kw)
        self.assertEqual(r['summary']['finite_fraction_of_planned'],0)
        self.assertIsNone(r['summary']['finite_fraction_given_success'])
        self.assertEqual(r['structural_unavailability'],'NOT_ESTABLISHED_FROM_FINITE_REALIZATIONS')

    def test_joint_dependence_retained(self):
        g=SpatialGraph(('x','y'),('r','r'),[[0,1],[1,0]],False,'synthetic')
        kw=dict(self.kw,graph=g,node_ids=g.cell_ids,baseline=['A','B'])
        a=audit_refits(realizations={'0':['A','A'],'1':['B','B']},failures={},planned_ids=['0','1'],**kw)
        b=audit_refits(realizations={'0':['A','B'],'1':['B','A']},failures={},planned_ids=['0','1'],**kw)
        self.assertEqual(a['summary']['finite'],0)
        self.assertEqual(b['summary']['finite'],2)

    def test_continuous_compositions_not_argmax_labels(self):
        kw=dict(self.kw,baseline=[[.8,.2],[.2,.8],[.5,.5]],classes=['A','B'],kind='fields',value_semantics='composition')
        r=audit_refits(realizations={'0':[[.6,.4],[.4,.6],[.5,.5]]},failures={},planned_ids=['0'],**kw)
        self.assertEqual(r['summary']['finite'],1)
        self.assertNotEqual(r['rows'][0]['O'],1)
        self.assertIsNone(r['summary']['sd_R_finite'])

    def test_baseline_na_not_proof_of_structural_na(self):
        kw=dict(self.kw,baseline=['A']*3)
        r=audit_refits(realizations={'0':['A','B','OTHER']},failures={},planned_ids=['0'],**kw)
        self.assertIsNone(r['summary']['baseline_R'])
        self.assertEqual(r['summary']['finite'],1)
        self.assertEqual(r['summary']['finite_delta'],0)


if __name__ == '__main__': unittest.main()
