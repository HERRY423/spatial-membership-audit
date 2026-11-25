import itertools
import math
import unittest
import numpy as np
from spatial_membership_audit.ecosystem import SpatialGraph, cross_k, build_graph
from spatial_membership_audit.endpoint_bridge import observed_endpoint_bridge


def graph(g, regions=None):
    return SpatialGraph(tuple(map(str, range(len(g)))), tuple(regions or ['r']*len(g)),
                        g, False, 'synthetic explicit graph')


class EndpointBridgeTests(unittest.TestCase):
    def test_all_four_node_graphs_and_labels_against_pair_loop(self):
        # Independent definition using directed pair counts and endpoint margins.
        edges=list(itertools.combinations(range(4),2))
        for bits in itertools.product((0,1),repeat=6):
            g=np.zeros((4,4),int)
            for (i,j), bit in zip(edges,bits): g[i,j]=g[j,i]=bit
            for lab in itertools.product(('A','B','OTHER'),repeat=4):
                out=observed_endpoint_bridge(graph(g),lab,('A','B'))
                counts={}
                for i in range(4):
                    for j in range(4):
                        if g[i,j]: counts[lab[i],lab[j]]=counts.get((lab[i],lab[j]),0)+1
                da=sum(v for (a,b),v in counts.items() if a=='A')
                db=sum(v for (a,b),v in counts.items() if b=='B')
                target=counts.get(('A','B'),0)*sum(counts.values())/(da*db) if da*db else None
                self.assertEqual(out['co_occurrence'],target)
                if out['R'] is not None and target is not None:
                    self.assertAlmostEqual(out['R']*out['co_occurrence_over_R_factor'],target)

    def test_same_R_different_cooccurrence(self):
        g1=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,0],[1,0,0,0]])
        g2=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
        a=observed_endpoint_bridge(graph(g1),['A','B','OTHER','OTHER'],('A','B'))
        b=observed_endpoint_bridge(graph(g2),['A','B','OTHER','OTHER'],('A','B'))
        self.assertEqual(a['R'],b['R'])
        self.assertNotEqual(a['co_occurrence'],b['co_occurrence'])

    def test_zero_absent_and_isolated_are_separate(self):
        g=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
        out=observed_endpoint_bridge(graph(g),['A','B','OTHER','OTHER'],('A','B'))
        self.assertEqual(out['R'],0)
        self.assertEqual(out['co_occurrence'],0)
        isolated=np.array([[0,0,0],[0,0,1],[0,1,0]])
        out=observed_endpoint_bridge(graph(isolated),['A','B','OTHER'],('A','B'))
        self.assertEqual(out['R'],0)
        self.assertIsNone(out['co_occurrence'])
        self.assertEqual(out['co_occurrence_status'],'NO_ENDPOINT_SUPPORT')

    def test_complete_graph_and_empty_graph(self):
        out=observed_endpoint_bridge(graph(np.ones((4,4))-np.eye(4)),['A','B','OTHER','OTHER'],('A','B'))
        self.assertEqual(out['R'],1)
        self.assertIsNone(out['z'])
        self.assertAlmostEqual(out['co_occurrence'],4/3)
        out=observed_endpoint_bridge(graph(np.zeros((3,3))),['A','B','OTHER'],('A','B'),window_area=4)
        self.assertIsNone(out['R'])
        self.assertEqual(out['K'],0)

    def test_K_L_match_coordinate_definition(self):
        xy=np.array([[1,1],[1,2],[3,2],[3,3]])
        lab=['A','B','OTHER','B']
        g=build_graph(xy,('0','1','2','3'),('r',)*4,method='radius',radius_um=2)
        out=observed_endpoint_bridge(g,lab,('A','B'),window_area=16)
        ref=cross_k(xy,lab,('A','B'),[2],window=(0,4,0,4),correction='none')[0]
        self.assertEqual(out['K'],ref['K'])
        self.assertEqual(out['L'],ref['L'])
        self.assertAlmostEqual(out['R']*out['K_over_R_factor'],out['K'])

    def test_reject_incompatible_contracts(self):
        with self.assertRaises(ValueError):
            observed_endpoint_bridge(graph(np.zeros((2,2)),['r','s']),['A','B'],('A','B'))
        with self.assertRaises(ValueError):
            observed_endpoint_bridge(graph([[0,2],[2,0]]),['A','B'],('A','B'))
        for bad in (True,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                observed_endpoint_bridge(graph([[0,1],[1,0]]),['A','B'],('A','B'),window_area=bad)


if __name__=='__main__': unittest.main()
