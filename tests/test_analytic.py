import itertools
import unittest
import numpy as np
from scipy import sparse
from scipy.stats import binom
from spatial_membership_audit.analytic import marker_probabilities, pair_moments, CLASSES
from spatial_membership_audit.definitions import candidate_labels, REQUIRED


class AnalyticTests(unittest.TestCase):
    def test_exact_categorical_enumeration(self):
        rng = np.random.default_rng(1701)
        for n in (2,3,5):
            for case in range(5):
                p = rng.dirichlet([1,1,1], size=n)
                upper = np.triu((rng.random((n,n)) < .6).astype(float),1)
                g = upper+upper.T
                rows=[]
                for labels in itertools.product(range(3),repeat=n):
                    labels=np.array(labels); w=np.prod(p[np.arange(n), labels])
                    a=(labels==0).astype(float); b=(labels==1).astype(float)
                    o=float(a@g@b); e=g.sum()*a.sum()*b.sum()/(n*(n-1))
                    rows.append([w,o,e])
                rows=np.array(rows); w,o,e=rows.T
                result=pair_moments(p,g)
                for key,expected in [('mu_O',w@o),('mu_E',w@e),
                    ('var_O',w@(o*o)-(w@o)**2),('var_E',w@(e*e)-(w@e)**2),
                    ('cov_O_E',w@(o*e)-(w@o)*(w@e)),('p_finite',w[e>0].sum())]:
                    self.assertAlmostEqual(result[key],expected,places=11,msg=(n,case,key))
                if result['p_finite']:
                    wc=w[e>0]/w[e>0].sum(); oc=o[e>0]; ec=e[e>0]
                    for key,expected in [('mu_O_finite',wc@oc),('mu_E_finite',wc@ec),
                        ('var_O_finite',wc@(oc*oc)-(wc@oc)**2),
                        ('var_E_finite',wc@(ec*ec)-(wc@ec)**2),
                        ('cov_O_E_finite',wc@(oc*ec)-(wc@oc)*(wc@ec))]:
                        self.assertAlmostEqual(result[key],expected,places=11,msg=key)

    def test_complete_graph_ratio_exactly_one(self):
        p=np.array([[.2,.3],[.8,.1],[.4,.4],[.1,.7]])
        result=pair_moments(p,np.ones((4,4))-np.eye(4))
        self.assertAlmostEqual(result['mean_R_delta2'],1.)
        self.assertLess(result['cv_R_delta'],1e-7)

    def test_empty_sparse_and_deterministic(self):
        self.assertEqual(pair_moments(np.empty((0,2)),sparse.csr_matrix((0,0)))['p_finite'],0)
        self.assertEqual(pair_moments([[.4,.6]],[[0]])['status'],'NO_FINITE_RATIO')
        r=pair_moments([[1,0],[0,1]],[[0,1],[1,0]])
        self.assertEqual(r['cv_R_delta'],0)
        r=pair_moments([[1,0],[0,1],[0,0]],[[0,0,1],[0,0,0],[1,0,0]])
        self.assertEqual(r['status'],'ZERO_NUMERATOR_CV_UNDEFINED')

    def test_invalid_inputs(self):
        for p,g in [([[.8,.8],[.1,.1]],[[0,1],[1,0]]),
                    ([[.2,.3],[.3,.4]],[[0,1],[0,0]]),
                    ([[.2,.3],[.3,.4]],[[1,1],[1,0]])]:
            with self.assertRaises(ValueError):pair_moments(p,g)

    def test_gate_full_enumeration_and_gene_retention(self):
        # Counts span positive thresholds, competing lineages and veto release.
        source=np.array([1,1,0,2,1,2,1,1,1,1,1])
        outcomes=np.array(list(itertools.product(*[range(x+1) for x in source])))
        labels=candidate_labels(outcomes,list(REQUIRED),np.ones(len(outcomes),bool))
        for retention in [.1,.5,.9, np.linspace(.1,.9,11),0.,1.]:
            weights=binom.pmf(outcomes,source,retention).prod(axis=1)
            expected=np.array([weights[labels==c].sum() for c in CLASSES])
            predicted=marker_probabilities(source[None,:],REQUIRED,np.array([True]),retention)[0]
            np.testing.assert_allclose(predicted,expected,atol=2e-14,rtol=1e-12)
        with self.assertRaises(ValueError):
            marker_probabilities(source[None,:],REQUIRED,np.array([True]),1.1)


if __name__ == '__main__':unittest.main()
