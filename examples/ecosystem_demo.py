"""Small synthetic end-to-end example. Uses no native optional dependencies."""
import numpy as np
from spatial_membership_audit.ecosystem import (
    build_graph, cross_k, random_label_moments, weighted_pair_moments,
    predict_cross_k, giotto_endpoint_null_moments,
)

xy = np.array([[1.,1.],[2.,1.],[2.,3.],[5.,4.]])
ids = ('c1','c2','c3','c4')
labels = np.array(['B','CD8','B','OTHER'])
graph = build_graph(xy, ids, ('region1',)*4, method='radius', radius_um=2.5)
# Different null models kept separate even when observed graph is identical.
print('Squidpy symmetric-graph exact node-label null:',
      random_label_moments(graph.adjacency, 2, 1))
print('Giotto 4.2.3 verified endpoint-label null:',
      giotto_endpoint_null_moments(graph.adjacency, labels, ('B','CD8')))
print('Observed cross K/L:', cross_k(xy, labels, ('B','CD8'), [0,1.5,2.5],
                                    window=(0,10,0,10), correction='translation'))
# Explicit synthetic categorical model, NOT arbitrary classifier scores.
p = np.array([[.8,.1,.1],[.1,.7,.2],[.6,.2,.2],[.1,.2,.7]])
print('Exact weighted numerator/denominator moments:',
      weighted_pair_moments(p, graph.adjacency, independent_nodes=True))
print('Cross K/L delta propagation:', predict_cross_k(p, xy, [0,1.5,2.5],
      window=(0,10,0,10), correction='translation', independent_nodes=True))
