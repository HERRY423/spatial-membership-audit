"""Engineering checks: enumeration, numerical tails and graph ownership."""
import itertools
import unittest
import numpy as np
from scipy import sparse
from spatial_membership_audit.analytic import pair_moments, _availability
from spatial_membership_audit.analytic_prepared import (
    availability_tree, prepare_pair_graph, pair_moments_prepared)


class PreparedTests(unittest.TestCase):
    def assert_contract(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key, value in expected.items():
            if value is None or isinstance(value, str):
                self.assertEqual(value, actual[key], key)
            else:
                np.testing.assert_allclose(actual[key], value, rtol=2e-10, atol=2e-12, err_msg=key)

    def test_small_graph_full_enumeration(self):
        rng = np.random.default_rng(28109)
        for n in range(2, 7):
            for case in range(4):
                p = rng.dirichlet([1., 1., 2.], size=n)
                p[case % n] = [0, 0, 1]  # deterministic OTHER
                upper = np.triu(rng.random((n,n)) < .45, 1)
                graph = (upper | upper.T).astype(float)
                got = pair_moments_prepared(p, prepare_pair_graph(graph))
                self.assert_contract(pair_moments(p,graph), got)
                rows=[]
                for labels in itertools.product(range(3), repeat=n):
                    labels=np.asarray(labels)
                    w=float(p[np.arange(n),labels].prod())
                    a=(labels==0).astype(float);b=(labels==1).astype(float)
                    o=float(a@graph@b);e=graph.sum()*a.sum()*b.sum()/(n*(n-1))
                    rows.append((w,o,e))
                w,o,e=np.asarray(rows).T
                for key, value in [('mu_O',w@o),('mu_E',w@e),
                                   ('var_O',w@(o*o)-(w@o)**2),
                                   ('var_E',w@(e*e)-(w@e)**2),
                                   ('cov_O_E',w@(o*e)-(w@o)*(w@e)),
                                   ('p_finite',w[e>0].sum())]:
                    self.assertAlmostEqual(got[key],value,places=11,msg=key)

    def test_rare_availability_and_endpoints(self):
        for a,b in [([],[]),([.3],[.4]),([1.,0],[0,1.]),
                    ([1e-140,0],[0,2e-140]),
                    ([.5,0,0],[0,1e-100,0])]:
            self.assertAlmostEqual(availability_tree(a,b), _availability(a,b), delta=max(1e-320, abs(_availability(a,b))*2e-14))
        self.assertGreater(availability_tree([1e-140,0],[0,2e-140]),0.)
        rng=np.random.default_rng(11)
        for n in [3,31,1000]:
            p=rng.dirichlet([1,1,1],size=n)
            self.assertAlmostEqual(availability_tree(p[:,0],p[:,1]),_availability(p[:,0],p[:,1]),places=12)

    def test_structural_constant_ratio_is_exact(self):
        p = np.array([[0., 0., 1.],
                      [.1430987698689457, .03818261411074578, .8187186160203086],
                      [.4804770074392417, .07814553224145743, .4413774603193008]])
        for graph, ratio in [(np.ones((3,3))-np.eye(3), 1.),
                             (np.array([[0,0,0],[0,0,1],[0,1,0]]), 3.)]:
            for got in [pair_moments(p, graph),
                        pair_moments_prepared(p, prepare_pair_graph(graph))]:
                self.assertEqual(got['mean_R_delta2'], ratio)
                for key in ['var_R_delta','cv_R_delta','var_O_finite','var_E_finite',
                            'cov_O_E_finite','cv_O_finite','cv_E_finite']:
                    self.assertEqual(got[key], 0., key)
        # More than two active nodes: D varies, but R remains exactly constant.
        p = np.array([[.2,.3],[.4,.1],[.1,.2],[.3,.2]])
        graph = np.ones((4,4))-np.eye(4)
        for got in [pair_moments(p,graph),
                    pair_moments_prepared(p,prepare_pair_graph(graph))]:
            self.assertEqual(got['cv_R_delta'], 0.)
            self.assertEqual(got['mean_R_delta2'], 1.)
            self.assertGreater(got['var_E_finite'], 0.)

    def test_small_positive_nonconstant_variance_is_preserved(self):
        # A fixed A, a fixed B connected to it, and a rare extra B off-edge.
        # D=1+Bernoulli(q), O=1; delta var=q(1-q)/(c^2*(1+q)^4).
        graph = np.array([[0,1,0],[1,0,0],[0,0,0]])
        for q in [1e-6, 1e-9, 1e-12]:
            p = np.array([[1.,0.],[0.,1.],[0.,q]])
            expected = q*(1-q)/((1/3)**2*(1+q)**4)
            for got in [pair_moments(p,graph),
                        pair_moments_prepared(p,prepare_pair_graph(graph))]:
                self.assertGreater(got['var_R_delta'],0.)
                self.assertAlmostEqual(got['var_R_delta']/expected,1.,delta=.002)

    def test_original_denominator_isolates_and_other_edges(self):
        graph=np.zeros((7,7))
        for i,j in [(0,1),(1,2),(3,4),(4,5)]: graph[i,j]=graph[j,i]=1
        p=np.zeros((7,2));p[0]=[.3,.2];p[1]=[.1,.4];p[6]=[.2,.1]
        got=pair_moments_prepared(p,prepare_pair_graph(graph))
        self.assert_contract(pair_moments(p,graph),got)
        self.assertEqual((got['n'],got['m']),(7,4))
        self.assertNotEqual(got['mu_E'],pair_moments(p[:2],graph[:2,:2])['mu_E'])

    def test_graph_owned_and_probability_mask_not_cached(self):
        graph=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        old=graph.copy();prepared=prepare_pair_graph(graph)
        graph.data[:]=0
        for p in [np.array([[.2,.3],[.4,0],[0,.1]]),np.array([[0,0],[1,0],[0,1]])]:
            self.assert_contract(pair_moments(p,old),pair_moments_prepared(p,prepared))
        with self.assertRaises(ValueError): prepared.graph.data[0]=2

    def test_empty_deterministic_and_rejections(self):
        for p,g in [(np.empty((0,2)),np.empty((0,0))),
                    ([[.3,.2]],[[0]]),([[1,0],[0,1]],[[0,1],[1,0]]),
                    ([[1,0],[0,1],[0,0]],[[0,0,1],[0,0,0],[1,0,0]])]:
            self.assert_contract(pair_moments(p,g),pair_moments_prepared(p,prepare_pair_graph(g)))
        for graph in [[[0,1],[0,0]],[[1,0],[0,0]],[[0,2],[2,0]],np.zeros((2,3))]:
            with self.assertRaises(ValueError):prepare_pair_graph(graph)
        prepared=prepare_pair_graph([[0,1],[1,0]])
        for p in [[[.8,.8],[0,0]],[[float('nan'),0],[0,0]],[[0,0]]]:
            with self.assertRaises(ValueError):pair_moments_prepared(p,prepared)


if __name__ == '__main__': unittest.main()
