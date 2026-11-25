"""Exact fixed-size random-label moments for cross-class edges on a fixed graph.

These moments define a permutation null, not the rarefaction distribution.
They provide a graph- and class-size-dependent O/E to z mapping when variance
is positive. No universal conversion from R alone exists.
"""
from fractions import Fraction
from math import comb, sqrt
import numpy as np
from scipy import sparse


def fixed_size_edge_moments(adjacency, n_a, n_b):
    g = sparse.csr_matrix(adjacency, dtype=float)
    if len(g.shape) != 2 or g.shape[0] != g.shape[1]:
        raise ValueError('square graph required')
    g.sum_duplicates(); g.eliminate_zeros()
    if not np.isfinite(g.data).all() or np.any(g.data != 1) or np.any(g.diagonal() != 0) or (g != g.T).nnz:
        raise ValueError('symmetric simple unweighted graph without self loops required')
    n = g.shape[0]
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0 for k in (n_a, n_b)) or n_a+n_b > n:
        raise ValueError('nonnegative integer disjoint class sizes required')
    a, b = int(n_a), int(n_b)
    m = g.nnz//2
    shared = sum(comb(int(d), 2) for d in np.diff(g.indptr))
    disjoint = comb(m, 2)-shared
    edge = Fraction(2*a*b, n*(n-1)) if n >= 2 else Fraction(0)
    adjacent = Fraction(a*b*(a+b-2), n*(n-1)*(n-2)) if n >= 3 else Fraction(0)
    separate = Fraction(4*a*(a-1)*b*(b-1), n*(n-1)*(n-2)*(n-3)) if n >= 4 else Fraction(0)
    mean = m*edge
    second = mean+2*shared*adjacent+2*disjoint*separate
    variance = second-mean*mean
    if variance < 0:
        raise ArithmeticError('negative exact permutation variance')
    return {'n': n, 'm': m, 'shared_edge_pairs': shared, 'disjoint_edge_pairs': disjoint,
            'mean_fraction': mean, 'variance_fraction': variance,
            'mean': float(mean), 'variance': float(variance),
            'status': 'NO_EXPECTED_EDGES' if mean == 0 else 'ZERO_NULL_VARIANCE' if variance == 0 else 'DEFINED',
            'estimand': 'uniform_label_permutation_fixed_class_sizes'}


def observed_permutation_bridge(adjacency, labels, *, a=0, b=1):
    labels = np.asarray(labels)
    g = sparse.csr_matrix(adjacency)
    if labels.shape != (g.shape[0],) or a == b:
        raise ValueError('aligned labels and distinct class codes required')
    aa, bb = labels == a, labels == b
    moments = fixed_size_edge_moments(g, int(aa.sum()), int(bb.sum()))
    observed = int(aa.astype(np.int64) @ g @ bb.astype(np.int64))
    mean, variance = moments['mean_fraction'], moments['variance_fraction']
    ratio = Fraction(observed, 1)/mean if mean else None
    z = float(Fraction(observed, 1)-mean)/sqrt(float(variance)) if variance else None
    return {**moments, 'observed': observed, 'ratio': float(ratio) if ratio is not None else None, 'z': z}
