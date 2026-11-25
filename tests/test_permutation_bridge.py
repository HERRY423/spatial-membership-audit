import itertools
import unittest
from fractions import Fraction
import numpy as np
from spatial_membership_audit.permutation_bridge import fixed_size_edge_moments, observed_permutation_bridge


class PermutationBridgeTests(unittest.TestCase):
    def test_exact_fixed_count_label_enumeration(self):
        g=np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]])
        values=[]
        for a in itertools.combinations(range(4),2):
            for b in set(range(4))-set(a):
                labels=np.full(4,2);labels[list(a)]=0;labels[b]=1
                values.append(int((labels==0).astype(int)@g@(labels==1).astype(int)))
        mean=Fraction(sum(values),len(values));var=sum((Fraction(v)-mean)**2 for v in values)/len(values)
        result=fixed_size_edge_moments(g,2,1)
        self.assertEqual(result['mean_fraction'],mean)
        self.assertEqual(result['variance_fraction'],var)

    def test_same_ratio_does_not_guarantee_defined_z(self):
        complete=np.ones((4,4))-np.eye(4)
        out=observed_permutation_bridge(complete,[0,0,1,2])
        self.assertEqual(out['ratio'],1.)
        self.assertEqual(out['status'],'ZERO_NULL_VARIANCE')
        self.assertIsNone(out['z'])

    def test_graph_validation(self):
        with self.assertRaises(ValueError): fixed_size_edge_moments([[0,1],[0,0]],1,1)
        with self.assertRaises(ValueError): fixed_size_edge_moments([[0,2],[2,0]],1,1)


if __name__=='__main__': unittest.main()
