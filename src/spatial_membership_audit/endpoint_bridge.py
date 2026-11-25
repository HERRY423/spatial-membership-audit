"""Observed endpoint mappings on an explicitly supplied single-region graph.

These are algebraic mappings for one realized label vector, not propagated
rarefaction distributions. Whole refits must be evaluated one vector at a time.
"""
import math
import numpy as np
from .permutation_bridge import observed_permutation_bridge


def observed_endpoint_bridge(graph, labels, pair, *, window_area=None):
    """Return O/E, exact fixed-size permutation z, edge co-occurrence and K/L.

``graph`` is a SpatialGraph for ONE region. Its adjacency must be symmetric,
binary and loop-free. Co-occurrence uses endpoint marginals, not cell fractions.
K/L require an explicitly supplied observation-window area and an uncorrected
radius graph. The caller attests these geometry semantics by supplying area;
use ecosystem.cross_k for coordinate-checked/translation-corrected K/L.
Undefined co-occurrence is None (native Squidpy may encode it as zero).
"""
    if len(set(graph.regions)) != 1 or graph.directed:
        raise ValueError('one explicit region and an undirected graph required')
    lab = np.asarray(labels)
    if lab.shape != (len(graph.cell_ids),) or len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError('aligned labels and distinct pair required')
    if any(not isinstance(v, str) or not v.strip() for v in lab):
        raise ValueError('explicit string labels required; retain UNKNOWN/OTHER')
    if any(not isinstance(v, str) or not v.strip() for v in pair):
        raise ValueError('explicit string pair required')
    if window_area is not None:
        if isinstance(window_area, (bool, np.bool_)) or not np.isscalar(window_area):
            raise ValueError('positive finite observation-window area required')
        window_area = float(window_area)
        if not math.isfinite(window_area) or window_area <= 0 or graph.coordinate_unit != 'um':
            raise ValueError('positive finite area in um squared required')
    null = observed_permutation_bridge(graph.adjacency, lab, a=pair[0], b=pair[1])
    aa, bb = lab == pair[0], lab == pair[1]
    n_a, n_b = int(aa.sum()), int(bb.sum())
    degree = np.asarray(graph.adjacency.sum(axis=1)).ravel()
    d_a, d_b = int(degree[aa].sum()), int(degree[bb].sum())
    total = 2 * null['m']
    co = null['observed'] * total / (d_a*d_b) if d_a*d_b else None
    n = null['n']
    # C_AB / R = (2M)^2 n_A n_B / [N(N-1) d_A d_B].
    co_factor = total**2*n_a*n_b/(n*(n-1)*d_a*d_b) if d_a*d_b else None
    kval = window_area*null['observed']/(n_a*n_b) if window_area is not None and n_a*n_b else None
    lval = math.sqrt(kval/math.pi) if kval is not None else None
    k_factor = window_area*total/(n*(n-1)) if window_area is not None and n > 1 else None
    return dict(O=null['observed'], E=null['mean'], R=null['ratio'],
        ratio_status='FINITE' if null['ratio'] is not None else 'NO_EXPECTED_EDGES',
        z=null['z'], z_status=null['status'], null_variance=null['variance'],
        n_a=n_a, n_b=n_b, N=n, M=null['m'], endpoint_degree_a=d_a, endpoint_degree_b=d_b,
        co_occurrence=co, co_occurrence_status='FINITE' if co is not None else 'NO_ENDPOINT_SUPPORT',
        co_occurrence_over_R_factor=co_factor,
        K=kval, L=lval, K_over_R_factor=k_factor,
        K_status='WINDOW_NOT_SUPPLIED' if window_area is None else 'FINITE' if kval is not None else 'ABSENT_TYPE',
        graph_provenance=graph.provenance,
        K_geometry='CALLER_SUPPLIED_UNCORRECTED_RADIUS_GRAPH' if window_area is not None else 'NOT_REQUESTED',
        null='fixed_class_size_uniform_node_permutation',
        uncertainty='OBSERVED_MAPPING_ONLY_NOT_RAREFACTION_MOMENTS',
        native_equivalence='ALGEBRA_ONLY_REQUIRES_IDENTICAL_PAIR_SUPPORT')
