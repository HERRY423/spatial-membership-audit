import unittest
import numpy as np
from spatial_membership_audit.analytic_prepared import prepare_pair_graph
from spatial_membership_audit.monte_carlo_prepared import sample_pair_prepared


class MonteCarloPreparedTests(unittest.TestCase):
    def test_compact_seed_replay_original_scale_and_isolates(self):
        g=np.zeros((7,7))
        for i,j in [(0,2),(1,3),(3,4)]:g[i,j]=g[j,i]=1
        p=np.zeros((7,2));p[0]=[.3,.2];p[2]=[.1,.4];p[6]=[.3,.3]
        prepared=prepare_pair_graph(g)
        summary,values=sample_pair_prepared(p,prepared,128,5,return_draws=True)
        # Independent scalar evaluator of the same compact random numbers.
        ids=[0,2,6];rng=np.random.default_rng(5);expected=[]
        for start in range(0,128,64):
            for u in rng.random((64,3)):
                labels=np.zeros(7,dtype=int)
                for k,i in enumerate(ids):
                    labels[i]=1 if u[k]<p[i,0] else (2 if u[k]<p[i].sum() else 0)
                na=int((labels==1).sum());nb=int((labels==2).sum())
                o=sum(int(labels[i]==1 and labels[j]==2) for i in range(7) for j in range(7) if g[i,j])
                e=6/(7*6)*na*nb
                expected.append(o/e if e>0 else np.nan)
        np.testing.assert_array_equal(values,np.array(expected))
        finite=np.asarray(expected);finite=finite[np.isfinite(finite)]
        self.assertAlmostEqual(summary['cv'],finite.std(ddof=1)/finite.mean())

    def test_structural_zero_and_no_finite(self):
        prepared=prepare_pair_graph([[0,1,0],[1,0,0],[0,0,0]])
        out=sample_pair_prepared([[1,0],[0,0],[0,1]],prepared,32,8)
        self.assertIsNone(out['cv']);self.assertEqual(out['p_finite'],1.)
        out=sample_pair_prepared([[0,0],[0,0],[0,0]],prepared,32,8)
        self.assertIsNone(out['cv']);self.assertEqual(out['p_finite'],0.)

    def test_invalid_contracts(self):
        prepared=prepare_pair_graph([[0,1],[1,0]])
        for p,n in [([[.8,.8],[0,0]],32),([[.1,.1]],32),([[.1,.1],[.1,.1]],1)]:
            with self.assertRaises(ValueError):sample_pair_prepared(p,prepared,n,1)


if __name__=='__main__':unittest.main()
