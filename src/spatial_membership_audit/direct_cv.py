"""Direct conditional ratio moments with explicit numerical enclosures.

An enclosure concerns the exact dyadic input probabilities on a fixed graph.
It assumes IEEE-754 binary64 round-to-nearest with gradual underflow, and a
nonnegative CSR sum with at most degree additions. It is not a formal machine
proof or an enclosure for upstream probability-estimation error.
"""
from functools import lru_cache
from fractions import Fraction
import math
import time
import threading

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy import sparse
from scipy.sparse import _sparsetools
from scipy.special import roots_legendre

INF = np.inf
EPS = np.finfo(float).eps
ETA = float(np.nextafter(0., 1.))
TERM_UPPER_LIMIT = 1e-20
_CV_CONTEXT = threading.local()


def point(x):
    x = np.asarray(x, dtype=float)
    return x, x


def down_inplace(x):
    if np.ndim(x) == 0:
        v = float(x)
        return np.float64(0.0 if v <= 0.0 else np.nextafter(v, -INF))
    b = x.view(np.uint64)
    b -= (x > 0.)
    x[x < 0.] = 0.
    return x


def up_inplace(x):
    if np.ndim(x) == 0:
        return np.nextafter(x, INF)
    zero = (x == 0.)
    if np.any(zero):
        x[zero] = 0.
    b = x.view(np.uint64)
    b += (x < INF)
    return x


def down(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        v = float(x)
        return np.float64(0.0 if v <= 0.0 else np.nextafter(v, -INF))
    return down_inplace(x.copy())


def up(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.nextafter(x, INF)
    return up_inplace(x.copy())


def add(x, y):
    lo = x[0] + y[0]
    hi = x[1] + y[1]
    if np.ndim(lo) == 0:
        return np.maximum(0., np.nextafter(lo, -INF)), np.where((x[1] == 0) & (y[1] == 0), 0., np.nextafter(hi, INF))
    blo = lo.view(np.uint64); blo -= (lo > 0.)
    bhi = hi.view(np.uint64); bhi += (hi > 0.) & (hi < INF)
    return lo, hi


def mul(x, y):
    lo = x[0] * y[0]
    hi = x[1] * y[1]
    if np.ndim(lo) == 0:
        return np.maximum(0., np.nextafter(lo, -INF)), np.where((x[1] == 0) | (y[1] == 0), 0., np.nextafter(hi, INF))
    blo = lo.view(np.uint64); blo -= (lo > 0.)
    bhi = hi.view(np.uint64)
    bhi += ((x[1] != 0.) & (y[1] != 0.)) & (hi < INF)
    return lo, hi


def div(x, y):
    if np.any(y[0] <= 0):
        raise ArithmeticError('nonpositive divisor interval')
    lo = x[0] / y[1]
    hi = x[1] / y[0]
    if np.ndim(lo) == 0:
        return np.maximum(0., np.nextafter(lo, -INF)), np.where(x[1] == 0, 0., np.nextafter(hi, INF))
    blo = lo.view(np.uint64); blo -= (lo > 0.)
    bhi = hi.view(np.uint64); bhi += (x[1] != 0.) & (hi < INF)
    return lo, hi


def square_difference(x, y):
    lo, hi = np.nextafter(x[0]-y[1], -INF), np.nextafter(x[1]-y[0], INF)
    low_abs = np.where((lo <= 0) & (hi >= 0), 0., np.minimum(abs(lo), abs(hi)))
    high_abs = np.maximum(abs(lo), abs(hi))
    return down(low_abs*low_abs), up(high_abs*high_abs)


def gamma_factors(k):
    ke = float(np.nextafter(k * EPS, INF))
    if ke >= 0.5:
        raise ArithmeticError("rounding bound unavailable")
    g = float(np.nextafter(ke / np.nextafter(1. - ke, -INF), INF))
    lo_div = float(np.nextafter(1. + g, INF))
    hi_div = float(np.nextafter(1. - g, -INF))
    sub_err = float(np.nextafter(k * ETA, INF))
    return lo_div, hi_div, sub_err


def outward_round(lo_raw, hi_raw, k, rho_lo=1.0, rho_hi=1.0):
    lo_div, hi_div, sub_err = gamma_factors(k)
    if rho_lo != 1.0:
        lo_div = float(np.nextafter(lo_div * rho_lo, INF))
    if rho_hi != 1.0:
        hi_div = float(np.nextafter(hi_div / rho_hi, -INF))
    if np.ndim(lo_raw) == 0:
        lo = 0.0 if lo_raw <= sub_err else float(np.nextafter(float(np.nextafter(lo_raw - sub_err, -INF)) / lo_div, -INF))
        hi = 0.0 if hi_raw == 0.0 else float(np.nextafter(float(np.nextafter(hi_raw + sub_err, INF)) / hi_div, INF))
        return np.float64(lo), np.float64(hi)
    low = down_inplace(np.maximum(0., np.nextafter(lo_raw - sub_err, -INF)) / lo_div)
    high = up_inplace(np.nextafter(hi_raw + sub_err, INF) / hi_div)
    high[hi_raw == 0.] = 0.
    return low, high


def reduce_sum_outward(x, axis=-1, extra_ops=0):
    lo, hi = x[0], x[1]
    n = lo.shape[axis]
    if n == 0:
        sh = lo.shape[:axis] + lo.shape[axis+1:] if axis != -1 else lo.shape[:-1]
        return point(np.zeros(sh, dtype=np.float64))
    if n == 1 and extra_ops == 0:
        return np.take(lo, 0, axis=axis), np.take(hi, 0, axis=axis)
    slo, shi = np.sum(lo, axis=axis), np.sum(hi, axis=axis)
    return outward_round(slo, shi, n + extra_ops)


_LO_DIV16, _HI_DIV16, _ = gamma_factors(16)


def reduce_mul_fast(x):
    lo, hi = x[0], x[1]
    if lo.shape[-1] == 0:
        return point(np.ones(lo.shape[:-1], dtype=np.float64))
    while lo.shape[-1] >= 32:
        n = lo.shape[-1]
        m = (n // 16) * 16
        lo_main = lo[..., :m].reshape(lo.shape[:-1] + (n // 16, 16))
        hi_main = hi[..., :m].reshape(hi.shape[:-1] + (n // 16, 16))
        p_lo = np.prod(lo_main, axis=-1)
        p_hi = np.prod(hi_main, axis=-1)
        if np.any(p_lo < 1e-290):
            break
        p_lo = down_inplace(p_lo / _LO_DIV16)
        p_hi = up_inplace(p_hi / _HI_DIV16)
        if n > m:
            lo = np.concatenate([p_lo, lo[..., m:]], axis=-1)
            hi = np.concatenate([p_hi, hi[..., m:]], axis=-1)
        else:
            lo, hi = p_lo, p_hi
    while lo.shape[-1] > 1:
        n = lo.shape[-1]
        even = n - (n % 2)
        z = mul((lo[..., :even:2], hi[..., :even:2]),
                (lo[..., 1:even:2], hi[..., 1:even:2]))
        if n % 2:
            lo = np.concatenate([z[0], lo[..., -1:]], axis=-1)
            hi = np.concatenate([z[1], hi[..., -1:]], axis=-1)
        else:
            lo, hi = z
    return lo[..., 0], hi[..., 0]


def reduce_interval(x, op=add, identity=0.):
    """Nonnegative sum bound, or balanced outward product, on the last axis."""
    lo, hi = np.broadcast_arrays(x[0], x[1])
    if lo.shape[-1] == 0:
        return point(np.full(lo.shape[:-1], identity))
    if op is add and lo.shape[-1] > 1:
        return reduce_sum_outward(x)
    while lo.shape[-1] > 1:
        n = lo.shape[-1]
        even = n-n%2
        z = op((lo[..., :even:2], hi[..., :even:2]),
               (lo[..., 1:even:2], hi[..., 1:even:2]))
        if n%2:
            lo = np.concatenate([z[0], lo[..., -1:]], axis=-1)
            hi = np.concatenate([z[1], hi[..., -1:]], axis=-1)
        else:
            lo, hi = z
    return lo[..., 0], hi[..., 0]


def availability(a, b, c):
    """Merge disjoint blocks with states none, A-only, B-only, both."""
    n = len(a)
    if n == 0: return 0., 0.
    states = [c, point(a), point(b), point(np.zeros(n))]; levels = 0
    while n > 1:
        even = n - n % 2
        left = [(v[0][:even:2], v[1][:even:2]) for v in states]
        right = [(v[0][1:even:2], v[1][1:even:2]) for v in states]
        none = mul(left[0], right[0])
        only_a = add(mul(left[1], add(right[0], right[1])), mul(left[0], right[1]))
        only_b = add(mul(left[2], add(right[0], right[2])), mul(left[0], right[2]))
        not_both_left = add(add(left[0], left[1]), left[2])
        cross = add(mul(left[1], right[2]), mul(left[2], right[1]))
        both = add(left[3], add(mul(not_both_left, right[3]), cross))
        merged = [none, only_a, only_b, both]
        if n % 2:
            merged = [tuple(np.concatenate([v[k], old[k][-1:]]) for k in (0, 1))
                      for v, old in zip(merged, states)]
        states = merged; n = len(states[0][0]); levels += 1
    return float(states[3][0][0]), min(1., float(states[3][1][0]))


def residual_probability(a, b):
    # fsum is used to reject invalid probabilities, never to justify an enclosure.
    s = a + b
    slack = 2.0 * EPS
    suspicious = np.flatnonzero(s > 1.0 - slack)
    exact_zero = np.zeros(len(a), dtype=bool)
    if len(suspicious) > 0:
        for idx in suspicious:
            rem = math.fsum([1., -float(a[idx]), -float(b[idx])])
            if rem < 0.:
                raise ValueError('a+b exceeds one as exact binary input')
            elif rem == 0.:
                exact_zero[idx] = True
    lo = np.nextafter(np.nextafter(1.-a, -INF)-b, -INF)
    hi = np.nextafter(np.nextafter(1.-a, INF)-b, INF)
    lo = np.where(exact_zero, 0., np.maximum(0., lo))
    hi = np.where(exact_zero, 0., np.minimum(1., hi))
    # Near one, two outward subtractions can make a tiny positive residual
    # interval much wider than the exact dyadic value. Tighten only those rows
    # using exact rational subtraction, then compare the rounded float to that
    # rational before selecting adjacent endpoints. fsum is not the certificate.
    wide = (~exact_zero) & ((lo <= 0.) | (hi > lo * (1. + 1e-7)))
    for idx in np.flatnonzero(wide):
        exact = Fraction(1) - Fraction(float(a[idx])) - Fraction(float(b[idx]))
        if exact < 0:
            raise ValueError('a+b exceeds one as exact binary input')
        rounded = float(exact)
        represented = Fraction(rounded)
        lo[idx] = np.nextafter(rounded, -INF) if represented > exact else rounded
        hi[idx] = np.nextafter(rounded, INF) if represented < exact else rounded
    return lo, hi


def residual_ratio_bound(c):
    """Multiplicative width; exact zero residuals have no perturbation.

    A zero lower bound with a positive upper bound remains unsafe. An exact
    zero does not enter a relative division and all factors containing it are
    exactly zero. Supported nodes still have a*x+b*y>0 at quadrature points.
    """
    positive = c[0] > 0.
    if np.any(~positive & (c[1] != 0.)):
        return INF
    ratio = float(np.max(up(c[1][positive] / c[0][positive]))) if np.any(positive) else 1.
    return float(np.nextafter(max(1., ratio), INF))


def matvec_interval(g, x):
    """Reuse only graph-dependent error factors; identical outward operations."""
    cache = getattr(_CV_CONTEXT, "matrix", {})
    key = id(g)
    if key not in cache:
        d = np.diff(g.indptr).astype(float)
        de = up(d*EPS); gamma = up(de/np.nextafter(1.-de, -INF))
        if np.any(gamma >= 1): raise ArithmeticError('CSR rounding bound unavailable')
        cache[key] = (g, d == 0, up(1.+gamma), np.nextafter(1.-gamma, -INF))
    graph, zero, lower_denominator, upper_denominator = cache[key]
    assert graph is g
    lo, hi = (g@x[0].T).T, (g@x[1].T).T
    low = down(lo/lower_denominator); high = up(hi/upper_denominator)
    return np.where(zero, 0., low), np.where(zero, 0., high)


def _tilted_polynomial_full(a, b, c, g, edges, x, y, moment_nodes=None):
    """Both F values for a batch of positive tensor-grid points."""
    x, y = point(np.asarray(x)[:, None]), point(np.asarray(y)[:, None])
    ax, by = mul(point(a), x), mul(point(b), y)
    z = add(add(ax, by), c)
    product = reduce_interval(z, op=mul, identity=1.)
    if moment_nodes is not None:
        ax = ax[0][:, moment_nodes], ax[1][:, moment_nodes]
        by = by[0][:, moment_nodes], by[1][:, moment_nodes]
        z = z[0][:, moment_nodes], z[1][:, moment_nodes]
        c = c[0][moment_nodes], c[1][moment_nodes]
    aa, bb, cc = div(ax, z), div(by, z), div(c, z)
    u, v = matvec_interval(g, aa), matvec_interval(g, bb)
    i, j = edges
    ai, aj = (aa[0][:, i], aa[1][:, i]), (aa[0][:, j], aa[1][:, j])
    bi, bj = (bb[0][:, i], bb[1][:, i]), (bb[0][:, j], bb[1][:, j])
    ci, cj = (cc[0][:, i], cc[1][:, i]), (cc[0][:, j], cc[1][:, j])
    mean = reduce_interval(add(mul(ai, bj), mul(bi, aj)))
    first = add(add(mul(mul(aa, cc), mul(v, v)), mul(mul(bb, cc), mul(u, u))),
                mul(mul(aa, bb), square_difference(v, u)))
    residual = add(add(mul(mul(ai, add(ci, bi)), mul(bj, add(cj, aj))),
                       mul(mul(bi, add(ci, ai)), mul(aj, add(cj, bj)))),
                   mul(point(2.), mul(mul(ai, bi), mul(aj, bj))))
    raw2 = add(mul(mean, mean), add(reduce_interval(first), reduce_interval(residual)))
    xy = mul((x[0][:, 0], x[1][:, 0]), (y[0][:, 0], y[1][:, 0]))
    scale = div(product, xy)
    return mul(scale, mean), mul(scale, raw2)


def matmul_nonneg_interval(x, y, extra_ops=0):
    k = x[0].shape[1]
    nx, ny = x[0].shape[0], y[0].shape[0]
    if k == 0:
        return np.zeros((nx, ny), dtype=np.float64), np.zeros((nx, ny), dtype=np.float64)
    lo_raw = x[0] @ y[0].T
    hi_raw = x[1] @ y[1].T
    return outward_round(lo_raw, hi_raw, k + extra_ops)


@lru_cache(maxsize=64)
def rule(order, support):
    """Certify *every* monomial degree for the actual positive float rule."""
    if not isinstance(support, (int, np.integer)) or not 1 <= support <= 2**53:
        raise ValueError('positive exactly representable support size required')
    panels, endpoint = [0.], .5
    while (1.-endpoint)*support > 1.:
        panels.append(endpoint)
        endpoint = 1.-(1.-endpoint)/2.
    panels.append(1.)
    nodes, weights, log_weights = [], [], []
    for left, right in zip(panels[:-1], panels[1:]):
        q = 2*order if left == 0 else order
        xx, ww = roots_legendre(q)
        local, weight = (xx+1.)/2., ww/2.
        width = right-left
        x_panel = left+width*local
        w_panel = width*weight
        if left == 0:
            degrees = np.arange(q)
            moments = np.ones(q)
            moments[1:] = (-1.)**degrees[1:]/(degrees[1:]*(degrees[1:]+1))
            log_weight = weight*(legvander(xx, q-1) @ ((2*degrees+1)*moments))
            v_panel = width*(log_weight-np.log(width)*weight)
        else:
            v_panel = w_panel*(-np.log(x_panel))
        nodes.extend(x_panel); weights.extend(w_panel); log_weights.extend(v_panel)
    x, w, w2 = np.array(nodes), np.array(weights), np.array(log_weights)
    if not (np.all((x > 0) & (x < 1)) and np.all(w > 0) and np.all(w2 > 0)):
        raise ArithmeticError('nonpositive quadrature rule')

    q_len = len(x)
    bases = []
    base = point(x)
    for bit in range((int(support) - 1).bit_length()):
        bases.append(base)
        base = mul(base, base)

    block_len = min(support, 2048)
    deg0 = np.arange(block_len, dtype=np.int64)
    lo0 = np.ones((block_len, q_len), dtype=np.float64)
    hi0 = np.ones((block_len, q_len), dtype=np.float64)
    for bit in range(min(11, len(bases))):
        sel = ((deg0 >> bit) & 1).astype(bool)
        if np.any(sel):
            lo0[sel], hi0[sel] = mul((lo0[sel], hi0[sel]), bases[bit])

    starts = np.arange(0, support, 2048, dtype=np.int64)
    n_blocks = len(starts)
    s_lo = np.ones((n_blocks, q_len), dtype=np.float64)
    s_hi = np.ones((n_blocks, q_len), dtype=np.float64)
    for bit in range(11, len(bases)):
        sel = ((starts >> bit) & 1).astype(bool)
        if np.any(sel):
            s_lo[sel], s_hi[sel] = mul((s_lo[sel], s_hi[sel]), bases[bit])

    w1_blocks = mul((s_lo, s_hi), point(w[None, :]))
    w2_blocks = mul((s_lo, s_hi), point(w2[None, :]))

    sum1_lo, sum1_hi = matmul_nonneg_interval(w1_blocks, (lo0, hi0))
    sum2_lo, sum2_hi = matmul_nonneg_interval(w2_blocks, (lo0, hi0))

    sum1_flat = (sum1_lo.ravel()[:support], sum1_hi.ravel()[:support])
    sum2_flat = (sum2_lo.ravel()[:support], sum2_hi.ravel()[:support])
    d1 = point(np.arange(1, support + 1, dtype=np.float64))
    d2 = mul(d1, d1)
    r1 = mul(sum1_flat, d1)
    r2 = mul(sum2_flat, d2)
    c0 = (float(np.min(r1[0])), float(np.max(r1[1])))
    c1 = (float(np.min(r2[0])), float(np.max(r2[1])))
    if c0[0] <= 0 or c1[0] <= 0:
        raise ArithmeticError('monomial certificate has nonpositive lower bound')
    return x, w, w2, (c0, c1)


def sqrt_interval(x):
    return float(down(np.sqrt(x[0]))), float(up(np.sqrt(x[1])))


def cv_interval(p, j1, j2):
    def intersect(value, hi=INF):
        lo, upper = max(0., float(value[0])), min(hi, float(value[1]))
        if not np.isfinite(lo) or np.isnan(upper) or lo > upper:
            raise ArithmeticError('invalid or empty moment interval')
        return lo, upper
    p, j1, j2 = intersect(p, 1.), intersect(j1), intersect(j2)
    if j1[0] <= 0:
        return 0., INF
    ratio = div(mul(p, j2), mul(j1, j1))
    upper_squared = float(np.nextafter(ratio[1]-1., INF))
    if upper_squared < 0.:
        raise InconsistentEnclosure('negative certified CV-squared upper bound')
    squared = (max(0., float(np.nextafter(ratio[0]-1., -INF))), upper_squared)
    return sqrt_interval(squared)


class InconsistentEnclosure(ArithmeticError):
    pass


def intersect_enclosures(old, new):
    result = new if old is None else (max(old[0], new[0]), min(old[1], new[1]))
    if result[0] > result[1] or np.any(np.isnan(result)):
        raise InconsistentEnclosure('disjoint certified enclosures')
    return result


def matvec_prep_csr(g):
    d = np.diff(g.indptr).astype(float)
    max_d = int(np.max(d)) if len(d) else 0
    de = float(np.nextafter(max_d * EPS, INF)) if len(d) else 0.
    if de >= 1.:
        raise ArithmeticError("CSR rounding bound unavailable")
    de_arr = up(d * EPS)
    gamma = up(de_arr / np.nextafter(1. - de_arr, -INF))
    if np.any(gamma >= 1.):
        raise ArithmeticError("CSR rounding bound unavailable")
    inv_lo = down(1.0 / up(1. + gamma))
    inv_hi = up(1.0 / np.nextafter(1. - gamma, -INF))
    inv_lo[d == 0] = 0.0
    inv_hi[d == 0] = 0.0
    return g, inv_lo, inv_hi, max_d


def split_csr_rows(mat, chunk_rows):
    m = mat.shape[0]
    chunks = []
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data
    for r0 in range(0, m, chunk_rows):
        r1 = min(m, r0 + chunk_rows)
        p0, p1 = int(indptr[r0]), int(indptr[r1])
        if p1 > p0:
            ip_c = (indptr[r0:r1+1] - p0).copy()
            idx_c = indices[p0:p1]
            dat_c = data[p0:p1]
        else:
            ip_c = None
            idx_c = None
            dat_c = None
        chunks.append((r0, r1, r1 - r0, ip_c, idx_c, dat_c))
    return chunks


def fast_linear_prod_1d(a_vec, c_int, x_vec):
    n = len(a_vec)
    if n == 0:
        return point(np.ones(len(x_vec), dtype=np.float64))
    lo_acc = np.ones(len(x_vec), dtype=np.float64)
    hi_acc = np.ones(len(x_vec), dtype=np.float64)
    lo_d2, hi_d2, _ = gamma_factors(3)
    for off in range(0, n, 16384):
        sl = slice(off, min(n, off + 16384))
        av = a_vec[sl][None, :]
        clo, chi = c_int[0][sl][None, :], c_int[1][sl][None, :]
        z_lo = (av * x_vec[:, None] + clo) / lo_d2
        z_hi = (av * x_vec[:, None] + chi) / hi_d2
        b_lo, b_hi = reduce_mul_fast((z_lo, z_hi))
        lo_acc, hi_acc = mul((lo_acc, hi_acc), (b_lo, b_hi))
    return lo_acc, hi_acc


def prepare_graph_parts(a, b, c, g, moment_nodes):
    """Precompute order-independent graph partitions once per pair."""
    gm = g if g.shape[0] == len(moment_nodes) else g[moment_nodes][:, moment_nodes].tocsr()
    am, bm = a[moment_nodes], b[moment_nodes]
    cm = (c[0][moment_nodes], c[1][moment_nodes])
    residual_ratio = residual_ratio_bound(cm)
    single_point_safe = residual_ratio <= 1.0 + 1e-7
    idx_a = np.flatnonzero((am > 0) & (bm == 0))
    idx_b = np.flatnonzero((bm > 0) & (am == 0))
    idx_w = np.flatnonzero((am > 0) & (bm > 0))
    _, inv_lo_full, inv_hi_full, max_d_gm = matvec_prep_csr(gm)

    g_AB = gm[idx_a][:, idx_b].tocsr()
    g_BA = gm[idx_b][:, idx_a].tocsr()
    g_WA = gm[idx_w][:, idx_a].tocsr()
    g_WB = gm[idx_w][:, idx_b].tocsr()
    g_WW = gm[idx_w][:, idx_w].tocsr()

    g_AW_full = gm[idx_a][:, idx_w].tocsr()
    g_BW_full = gm[idx_b][:, idx_w].tocsr()
    mask_aw = np.diff(g_AW_full.indptr) > 0
    mask_bw = np.diff(g_BW_full.indptr) > 0
    mask_a0 = ~mask_aw
    mask_b0 = ~mask_bw
    idx_aw_global = idx_a[mask_aw]
    idx_bw_global = idx_b[mask_bw]
    g_AW = g_AW_full[mask_aw].tocsr()
    g_BW = g_BW_full[mask_bw].tocsr()

    nw = len(idx_w)
    naw = len(idx_aw_global)
    nbw = len(idx_bw_global)
    ww_chunks = split_csr_rows(g_WW, min(max(nw, 1), 4096)) if nw > 0 else []
    aw_chunks = split_csr_rows(g_AW, min(max(naw, 1), 8192)) if naw > 0 else []
    bw_chunks = split_csr_rows(g_BW, min(max(nbw, 1), 8192)) if nbw > 0 else []

    return dict(
        gm=gm, am=am, bm=bm, cm=cm, single_point_safe=single_point_safe,
        residual_ratio=residual_ratio,
        idx_a=idx_a, idx_b=idx_b, idx_w=idx_w,
        inv_lo_full=inv_lo_full, inv_hi_full=inv_hi_full, max_d_gm=max_d_gm,
        g_AB=g_AB, g_BA=g_BA, g_WA=g_WA, g_WB=g_WB, g_WW=g_WW,
        mask_aw=mask_aw, mask_bw=mask_bw, mask_a0=mask_a0, mask_b0=mask_b0,
        idx_aw_global=idx_aw_global, idx_bw_global=idx_bw_global,
        g_AW=g_AW, g_BW=g_BW,
        ww_chunks=ww_chunks, aw_chunks=aw_chunks, bw_chunks=bw_chunks,
        is_a_all=(a > 0) & (b == 0),
        is_b_all=(b > 0) & (a == 0),
        is_w_all=(a > 0) & (b > 0),
    )


def integrate_order_separable(a, b, c, g, edges, order, deadline, moment_nodes, parts=None):
    if parts is None:
        parts = prepare_graph_parts(a, b, c, g, moment_nodes)
    ka, kb = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    xa, wa, wa2, ca = rule(order, ka)
    xb, wb, wb2, cb = rule(order, kb)
    na, nb = len(xa), len(xb)

    is_a_all = parts['is_a_all']
    is_b_all = parts['is_b_all']
    is_w_all = parts['is_w_all']

    prod_a_1d = fast_linear_prod_1d(a[is_a_all], (c[0][is_a_all], c[1][is_a_all]), xa)
    prod_b_1d = fast_linear_prod_1d(b[is_b_all], (c[0][is_b_all], c[1][is_b_all]), xb)

    if np.any(is_w_all):
        aw_all, bw_all = a[is_w_all], b[is_w_all]
        cw_all = (c[0][is_w_all], c[1][is_w_all])
        cwb_hi = up(cw_all[1] + bw_all)
        cwa_hi = up(cw_all[1] + aw_all)
        _, pw_x1_hi = fast_linear_prod_1d(aw_all, (cwb_hi, cwb_hi), xa)
        _, pw_1y_hi = fast_linear_prod_1d(bw_all, (cwa_hi, cwa_hi), xb)
        pw_x1_hi = np.minimum(1., pw_x1_hi)
        pw_1y_hi = np.minimum(1., pw_1y_hi)
        pw_ub_2d = np.minimum(pw_x1_hi[:, None], np.minimum(pw_1y_hi[None, :], up(pw_x1_hi[:, None] * pw_1y_hi[None, :])))
    else:
        pw_ub_2d = np.ones((na, nb), dtype=np.float64)

    x_2d = np.repeat(xa, nb)
    y_2d = np.tile(xb, na)
    xy_2d = mul(point(x_2d), point(y_2d))
    m_edges = float(len(edges[0]))
    m_pt = point(m_edges)

    prod_ab_base = mul((np.repeat(prod_a_1d[0], nb), np.repeat(prod_a_1d[1], nb)),
                       (np.tile(prod_b_1d[0], na), np.tile(prod_b_1d[1], na)))
    ub_init = up(prod_ab_base[1] * pw_ub_2d.ravel())
    scale_ub = div((np.zeros_like(ub_init), ub_init), xy_2d)
    b1_init = mul(scale_ub, m_pt)
    b2_init = mul(b1_init, m_pt)
    bypass = (b1_init[1] <= TERM_UPPER_LIMIT) & (b2_init[1] <= TERM_UPPER_LIMIT)

    one = (np.zeros(len(x_2d), dtype=np.float64), b1_init[1].copy())
    two = (np.zeros(len(x_2d), dtype=np.float64), b2_init[1].copy())

    cand_idx = np.flatnonzero(~bypass)
    if len(cand_idx) > 0:
        if np.any(is_w_all):
            nw_all = len(aw_all)
            pw_lo = np.ones(len(cand_idx), dtype=np.float64)
            pw_hi = np.ones(len(cand_idx), dtype=np.float64)
            lo_d3, hi_d3, _ = gamma_factors(4)
            ia_cand = cand_idx // nb
            ib_cand = cand_idx % nb
            for off_v in range(0, nw_all, 8192):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("BUDGET_EXHAUSTED")
                sl_v = slice(off_v, min(nw_all, off_v + 8192))
                ax_1d = xa[:, None] * aw_all[None, sl_v]
                by_1d = xb[:, None] * bw_all[None, sl_v]
                ax_c_lo = ax_1d + cw_all[0][None, sl_v]
                ax_c_hi = ax_1d + cw_all[1][None, sl_v]
                for off_pt in range(0, len(cand_idx), 256):
                    sl_pt = slice(off_pt, min(len(cand_idx), off_pt + 256))
                    ia_b = ia_cand[sl_pt]
                    ib_b = ib_cand[sl_pt]
                    by_b = by_1d[ib_b]
                    z_lo = (ax_c_lo[ia_b] + by_b) / lo_d3
                    z_hi = (ax_c_hi[ia_b] + by_b) / hi_d3
                    b_lo, b_hi = reduce_mul_fast((z_lo, z_hi))
                    pw_lo[sl_pt], pw_hi[sl_pt] = mul((pw_lo[sl_pt], pw_hi[sl_pt]), (b_lo, b_hi))
            prod_cand = mul((prod_ab_base[0][cand_idx], prod_ab_base[1][cand_idx]), (pw_lo, pw_hi))
        else:
            prod_cand = (prod_ab_base[0][cand_idx], prod_ab_base[1][cand_idx])

        scale_cand = div(prod_cand, (xy_2d[0][cand_idx], xy_2d[1][cand_idx]))
        b1_cand = mul(scale_cand, m_pt)
        b2_cand = mul(b1_cand, m_pt)
        one[1][cand_idx] = b1_cand[1]
        two[1][cand_idx] = b2_cand[1]
        keep = ~((b1_cand[1] <= TERM_UPPER_LIMIT) & (b2_cand[1] <= TERM_UPPER_LIMIT))
        act_idx = cand_idx[keep]
        prod_act = (prod_cand[0][keep], prod_cand[1][keep])
    else:
        act_idx = cand_idx
        prod_act = (np.empty(0), np.empty(0))

    if len(act_idx) > 0:
        if len(parts['idx_w']) and not parts['single_point_safe']:
            # The reduced-point error model needs narrow, positive residual
            # intervals. If that precondition fails, every mixed-node mean,
            # variance and edge-residual contribution still has to be included.
            # Evaluate the existing full interval identity on active grid
            # points only. prod_act already includes denominator-only nodes;
            # graph/moment factors below contain just incident nodes.
            am, bm, cm = parts['am'], parts['bm'], parts['cm']
            for offset in range(0, len(act_idx), 32):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('BUDGET_EXHAUSTED')
                sl = slice(offset, offset + 32)
                indices = act_idx[sl]
                xx, yy = x_2d[indices], y_2d[indices]
                ax = mul(point(am), point(xx[:, None]))
                by = mul(point(bm), point(yy[:, None]))
                z = add(add(ax, by), cm)
                actual = _moments_from_factors(ax, by, z,
                    (prod_act[0][sl], prod_act[1][sl]), cm, g, edges, xx, yy)
                for target, values in zip((one, two), actual):
                    for k in (0, 1):
                        target[k][indices] = values[k]
            weights1 = mul(point(np.repeat(wa, nb)), point(np.tile(wb, na)))
            weights2 = mul(point(np.repeat(wa2, nb)), point(np.tile(wb2, na)))
            q1 = reduce_sum_outward(mul(one, weights1))
            q2 = reduce_sum_outward(mul(two, weights2))
            return div(q1, mul(ca[0], cb[0])), div(q2, mul(ca[1], cb[1]))

        am, bm, cm = parts['am'], parts['bm'], parts['cm']
        idx_a, idx_b, idx_w = parts['idx_a'], parts['idx_b'], parts['idx_w']
        inv_lo_full, inv_hi_full, max_d_gm = parts['inv_lo_full'], parts['inv_hi_full'], parts['max_d_gm']
        g_AB, g_BA, g_WA, g_WB, g_WW = parts['g_AB'], parts['g_BA'], parts['g_WA'], parts['g_WB'], parts['g_WW']
        mask_aw, mask_bw, mask_a0, mask_b0 = parts['mask_aw'], parts['mask_bw'], parts['mask_a0'], parts['mask_b0']
        idx_aw_global, idx_bw_global = parts['idx_aw_global'], parts['idx_bw_global']
        g_AW, g_BW = parts['g_AW'], parts['g_BW']
        ww_chunks, aw_chunks, bw_chunks = parts['ww_chunks'], parts['aw_chunks'], parts['bw_chunks']

        ia_all = act_idx // nb
        ib_all = act_idx % nb
        ux, ia_inv = np.unique(ia_all, return_inverse=True)
        uy, ib_inv = np.unique(ib_all, return_inverse=True)
        xa_act, xb_act = xa[ux], xb[uy]
        nx_act, ny_act = len(ux), len(uy)

        aa_A = mul(point(am[idx_a]), point(xa_act[:, None]))
        z_A = add(aa_A, (cm[0][idx_a], cm[1][idx_a]))
        alpha_A = div(aa_A, z_A)
        gamma_A = div((cm[0][idx_a], cm[1][idx_a]), z_A)
        p_A = mul(alpha_A, gamma_A)
        # Large quadrature-by-node temporaries have no later consumers.
        # Release them before constructing the second population's intervals.
        del aa_A, z_A, gamma_A

        bb_B = mul(point(bm[idx_b]), point(xb_act[:, None]))
        z_B = add(bb_B, (cm[0][idx_b], cm[1][idx_b]))
        beta_B = div(bb_B, z_B)
        gamma_B = div((cm[0][idx_b], cm[1][idx_b]), z_B)
        q_B = mul(beta_B, gamma_B)
        del bb_B, z_B, gamma_B

        def sub_matvec_col(sub_g, rows_idx, x_col):
            if sub_g.nnz == 0 or x_col[0].shape[1] == 0:
                sh = (len(rows_idx), x_col[0].shape[1])
                return np.zeros(sh, dtype=np.float64), np.zeros(sh, dtype=np.float64)
            lo = sub_g @ x_col[0]
            hi = sub_g @ x_col[1]
            np.multiply(lo, inv_lo_full[rows_idx, None], out=lo)
            np.multiply(hi, inv_hi_full[rows_idx, None], out=hi)
            down_inplace(lo)
            up_inplace(hi)
            return lo, hi

        alpha_A_col = (np.ascontiguousarray(alpha_A[0].T), np.ascontiguousarray(alpha_A[1].T))
        p_A_col = (np.ascontiguousarray(p_A[0].T), np.ascontiguousarray(p_A[1].T))
        beta_B_col = (np.ascontiguousarray(beta_B[0].T), np.ascontiguousarray(beta_B[1].T))
        q_B_col = (np.ascontiguousarray(q_B[0].T), np.ascontiguousarray(q_B[1].T))

        v_A_from_B_col = sub_matvec_col(g_AB, idx_a, beta_B_col)
        gq_A_from_B_col = sub_matvec_col(g_AB, idx_a, q_B_col)
        u_B_from_A_col = sub_matvec_col(g_BA, idx_b, alpha_A_col)

        u_W_from_A_col = sub_matvec_col(g_WA, idx_w, alpha_A_col)
        gp_W_from_A_col = sub_matvec_col(g_WA, idx_w, p_A_col)
        v_W_from_B_col = sub_matvec_col(g_WB, idx_w, beta_B_col)
        gq_W_from_B_col = sub_matvec_col(g_WB, idx_w, q_B_col)
        del alpha_A_col, p_A_col, beta_B_col

        mean_AB_2d = matmul_nonneg_interval(alpha_A, (v_A_from_B_col[0].T, v_A_from_B_col[1].T))
        res_AB_2d = matmul_nonneg_interval(p_A, (gq_A_from_B_col[0].T, gq_A_from_B_col[1].T))
        del gq_A_from_B_col, alpha_A, beta_B

        v_A0_T = (np.ascontiguousarray(v_A_from_B_col[0][mask_a0].T), np.ascontiguousarray(v_A_from_B_col[1][mask_a0].T))
        v_A0_sq = mul(v_A0_T, v_A0_T)
        first_A0_2d = matmul_nonneg_interval((p_A[0][:, mask_a0], p_A[1][:, mask_a0]), v_A0_sq)
        del v_A0_T, v_A0_sq

        u_B0_T = (np.ascontiguousarray(u_B_from_A_col[0][mask_b0].T), np.ascontiguousarray(u_B_from_A_col[1][mask_b0].T))
        u_B0_sq = mul(u_B0_T, u_B0_T)
        first_B0_2d = matmul_nonneg_interval(u_B0_sq, (q_B[0][:, mask_b0], q_B[1][:, mask_b0]))
        del u_B0_T, u_B0_sq

        p_AW_row = (np.ascontiguousarray(p_A[0][:, mask_aw]), np.ascontiguousarray(p_A[1][:, mask_aw]))
        v_AW_from_B_col = (np.ascontiguousarray(v_A_from_B_col[0][mask_aw]), np.ascontiguousarray(v_A_from_B_col[1][mask_aw]))
        q_BW_col = (np.ascontiguousarray(q_B_col[0][mask_bw]), np.ascontiguousarray(q_B_col[1][mask_bw]))
        u_BW_from_A_col = (np.ascontiguousarray(u_B_from_A_col[0][mask_bw]), np.ascontiguousarray(u_B_from_A_col[1][mask_bw]))
        del p_A, q_B, q_B_col, v_A_from_B_col, u_B_from_A_col

        m_rect = (mean_AB_2d[0].copy(), mean_AB_2d[1].copy())
        r_rect = (res_AB_2d[0].copy(), res_AB_2d[1].copy())
        f_rect = add(first_A0_2d, first_B0_2d)

        nw = len(idx_w)
        naw = len(idx_aw_global)
        nbw = len(idx_bw_global)
        if nw > 0:
            aw_m, bw_m = am[idx_w], bm[idx_w]
            cw_lo = cm[0][idx_w, None]
            cw_hi = cm[1][idx_w, None]

            can_single = parts['single_point_safe']
            if can_single:
                rho_c = parts['residual_ratio']
                rho_c2 = float(np.nextafter(rho_c * rho_c, INF))
                rho_c3 = float(np.nextafter(rho_c2 * rho_c, INF))
                rho_c4 = float(np.nextafter(rho_c2 * rho_c2, INF))
                lo_duv, hi_duv, sub_uv = gamma_factors(max_d_gm + 8)
                rho_uv_lo = float(np.nextafter(rho_c * lo_duv, INF))
                rho_uv_hi = float(np.nextafter(1.0 / hi_duv, INF))
                delta_uv = float(np.nextafter(max(rho_uv_hi - 1.0, 1.0 - 1.0 / rho_uv_lo), INF))
                young_coef = float(np.nextafter(4.0 * delta_uv * (1.0 + 2.0 * delta_uv), INF))
                young_sub = float(np.nextafter(8.0 * (1.0 + 1.0 / max(delta_uv, 1e-16)) * (sub_uv * sub_uv), INF))

                by_w = bw_m[:, None] * xb_act[None, :]
                ax_all = aw_m[:, None] * xa_act[None, :]
                u_W_from_A_T = np.ascontiguousarray(u_W_from_A_col[0].T)
                gp_W_from_A_T = np.ascontiguousarray(gp_W_from_A_col[0].T)

                pack_in = np.empty((nw, 4 * ny_act), dtype=np.float64)
                al_w = pack_in[:, 0*ny_act : 1*ny_act]
                be_w = pack_in[:, 1*ny_act : 2*ny_act]
                qw_w = pack_in[:, 2*ny_act : 3*ny_act]
                rw_w = pack_in[:, 3*ny_act : 4*ny_act]
                ga_w = np.empty((nw, ny_act), dtype=np.float64)
                al_w_c = np.empty((nw, ny_act), dtype=np.float64)
                be_w_c = np.empty((nw, ny_act), dtype=np.float64)

                CHUNK = min(nw, 4096)
                c_inv_z = np.empty((CHUNK, ny_act), dtype=np.float64)
                c_pack_out = np.empty((CHUNK, 4 * ny_act), dtype=np.float64)
                c_pw = np.empty((CHUNK, ny_act), dtype=np.float64)
                c_tmp1 = np.empty((CHUNK, ny_act), dtype=np.float64)
                c_tmp2 = np.empty((CHUNK, ny_act), dtype=np.float64)
                c_slo = np.empty((CHUNK, ny_act), dtype=np.float64)

                acc_m = np.empty((CHUNK, ny_act), dtype=np.float64)
                acc_r = np.empty((CHUNK, ny_act), dtype=np.float64)
                acc_f_pos = np.empty((CHUNK, ny_act), dtype=np.float64)
                acc_f_sq = np.empty((CHUNK, ny_act), dtype=np.float64)
                acc_f_vu = np.empty((CHUNK, ny_act), dtype=np.float64)

                CHUNK_AB = 8192
                if naw > 0:
                    c_aw = np.empty((min(naw, CHUNK_AB), ny_act), dtype=np.float64)
                if nbw > 0:
                    c_bw = np.empty((min(nbw, CHUNK_AB), ny_act), dtype=np.float64)
                    acc_bw = np.empty((min(nbw, CHUNK_AB), ny_act), dtype=np.float64)

                for p_idx in range(nx_act):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("BUDGET_EXHAUSTED")
                    ax_p = ax_all[:, p_idx:p_idx+1]
                    u_W_A_p = u_W_from_A_col[0][:, p_idx:p_idx+1]

                    for r0, r1, n_r, _, _, _ in ww_chunks:
                        sl = slice(r0, r1)
                        iz = c_inv_z[:n_r]
                        al = al_w[sl]
                        be = be_w[sl]
                        qw = qw_w[sl]
                        rw = rw_w[sl]
                        ga = ga_w[sl]

                        np.add(ax_p[sl], by_w[sl], out=iz)
                        np.add(iz, cw_lo[sl], out=iz)
                        np.reciprocal(iz, out=iz)
                        np.multiply(ax_p[sl], iz, out=al)
                        np.multiply(by_w[sl], iz, out=be)
                        np.copyto(al_w_c[sl], al)
                        np.copyto(be_w_c[sl], be)
                        np.multiply(cw_lo[sl], iz, out=ga)
                        np.multiply(al, be, out=rw)
                        np.add(ga, al, out=qw)
                        np.multiply(be, qw, out=qw)

                    mw_raw = u_W_from_A_T[p_idx] @ be_w_c
                    rw_raw = gp_W_from_A_T[p_idx] @ qw_w

                    acc_m.fill(0.0)
                    acc_r.fill(0.0)
                    acc_f_pos.fill(0.0)
                    acc_f_sq.fill(0.0)
                    acc_f_vu.fill(0.0)

                    for r0, r1, n_r, ip_c, idx_c, dat_c in ww_chunks:
                        sl = slice(r0, r1)
                        al = al_w[sl]
                        be = be_w[sl]
                        rw = rw_w[sl]
                        ga = ga_w[sl]
                        pw = c_pw[:n_r]
                        t1 = c_tmp1[:n_r]
                        t2 = c_tmp2[:n_r]
                        slo = c_slo[:n_r]

                        po = c_pack_out[:n_r]
                        uw = po[:, 0*ny_act : 1*ny_act]
                        vw = po[:, 1*ny_act : 2*ny_act]
                        gqw = po[:, 2*ny_act : 3*ny_act]
                        grw = po[:, 3*ny_act : 4*ny_act]

                        np.copyto(uw, u_W_A_p[sl])
                        np.copyto(vw, v_W_from_B_col[0][sl])
                        np.copyto(gqw, gq_W_from_B_col[0][sl])
                        grw.fill(0.0)
                        if ip_c is not None:
                            _sparsetools.csr_matvecs(n_r, nw, 4 * ny_act, ip_c, idx_c, dat_c, pack_in, po)

                        np.multiply(al, vw, out=t1)
                        acc_m[:n_r] += t1

                        np.multiply(al, ga, out=t1)
                        np.add(t1, rw, out=pw)
                        np.multiply(pw, gqw, out=t2)
                        if ip_c is not None:
                            np.multiply(rw, grw, out=slo)
                            t2 += slo
                        acc_r[:n_r] += t2

                        np.square(vw, out=t2)
                        np.square(uw, out=slo)
                        t1 *= t2
                        np.multiply(be, ga, out=pw)
                        pw *= slo
                        t1 += pw
                        acc_f_pos[:n_r] += t1

                        t2 += slo
                        t2 *= rw
                        acc_f_vu[:n_r] += t2

                        np.subtract(vw, uw, out=t1)
                        np.square(t1, out=t1)
                        t1 *= rw
                        acc_f_sq[:n_r] += t1

                    mw_raw += np.sum(acc_m, axis=0)
                    rw_raw += np.sum(acc_r, axis=0)
                    fw_pos_raw = np.sum(acc_f_pos, axis=0)
                    fw_sq_raw = np.sum(acc_f_sq, axis=0)
                    fw_vu_raw = np.sum(acc_f_vu, axis=0)

                    mw_lo, mw_hi = outward_round(mw_raw, mw_raw, nw + max_d_gm + 10, rho_lo=rho_c2, rho_hi=rho_c)
                    m_rect[0][p_idx], m_rect[1][p_idx] = add((m_rect[0][p_idx], m_rect[1][p_idx]), (mw_lo, mw_hi))

                    rw_sum_lo, rw_sum_hi = outward_round(rw_raw, rw_raw, nw + max_d_gm + 14, rho_lo=rho_c4, rho_hi=rho_c3)
                    r_rect[0][p_idx], r_rect[1][p_idx] = add((r_rect[0][p_idx], r_rect[1][p_idx]), (rw_sum_lo, rw_sum_hi))

                    fw_pos_lo, fw_pos_hi = outward_round(fw_pos_raw, fw_pos_raw, nw + 2*max_d_gm + 16, rho_lo=rho_c4, rho_hi=rho_c2)
                    _, fw_vu_hi = outward_round(fw_vu_raw, fw_vu_raw, nw + 2*max_d_gm + 16, rho_lo=rho_c4, rho_hi=rho_c2)
                    pert = up(young_coef * fw_vu_hi + float(nw) * young_sub)
                    sq_lo_in = np.maximum(0.0, down(fw_sq_raw * (1.0 - 2.0 * delta_uv) - pert))
                    sq_hi_in = up(fw_sq_raw * (1.0 + 2.0 * delta_uv) + pert)
                    fw_sq_lo, fw_sq_hi = outward_round(sq_lo_in, sq_hi_in, nw + 16, rho_lo=rho_c2, rho_hi=rho_c)
                    fw_lo, fw_hi = add((fw_pos_lo, fw_pos_hi), (fw_sq_lo, fw_sq_hi))
                    f_rect[0][p_idx], f_rect[1][p_idx] = add((f_rect[0][p_idx], f_rect[1][p_idx]), (fw_lo, fw_hi))

                    if naw > 0:
                        f_aw_raw = np.zeros(ny_act, dtype=np.float64)
                        p_aw_p = p_AW_row[0][p_idx]
                        for r0, r1, n_r, ip_c, idx_c, dat_c in aw_chunks:
                            sl = slice(r0, r1)
                            buf = c_aw[:n_r]
                            np.copyto(buf, v_AW_from_B_col[0][sl])
                            if ip_c is not None:
                                _sparsetools.csr_matvecs(n_r, nw, ny_act, ip_c, idx_c, dat_c, be_w_c, buf)
                            np.square(buf, out=buf)
                            f_aw_raw += p_aw_p[sl] @ buf
                        f_aw_lo, f_aw_hi = outward_round(f_aw_raw, f_aw_raw, naw + 2*max_d_gm + 10, rho_lo=rho_c4, rho_hi=rho_c2)
                        f_rect[0][p_idx], f_rect[1][p_idx] = add((f_rect[0][p_idx], f_rect[1][p_idx]), (f_aw_lo, f_aw_hi))

                    if nbw > 0:
                        acc_bw.fill(0.0)
                        u_bw_p = u_BW_from_A_col[0][:, p_idx:p_idx+1]
                        for r0, r1, n_r, ip_c, idx_c, dat_c in bw_chunks:
                            sl = slice(r0, r1)
                            buf = c_bw[:n_r]
                            np.copyto(buf, u_bw_p[sl])
                            if ip_c is not None:
                                _sparsetools.csr_matvecs(n_r, nw, ny_act, ip_c, idx_c, dat_c, al_w_c, buf)
                            np.square(buf, out=buf)
                            buf *= q_BW_col[0][sl]
                            acc_bw[:n_r] += buf
                        f_bw_raw = np.sum(acc_bw, axis=0)
                        f_bw_lo, f_bw_hi = outward_round(f_bw_raw, f_bw_raw, nbw + 2*max_d_gm + 10, rho_lo=rho_c4, rho_hi=rho_c2)
                        f_rect[0][p_idx], f_rect[1][p_idx] = add((f_rect[0][p_idx], f_rect[1][p_idx]), (f_bw_lo, f_bw_hi))

        m_act = (m_rect[0][ia_inv, ib_inv], m_rect[1][ia_inv, ib_inv])
        r_act = (r_rect[0][ia_inv, ib_inv], r_rect[1][ia_inv, ib_inv])
        f_act = (f_rect[0][ia_inv, ib_inv], f_rect[1][ia_inv, ib_inv])
        raw2 = add(mul(m_act, m_act), add(f_act, r_act))
        scale = div(prod_act, (xy_2d[0][act_idx], xy_2d[1][act_idx]))
        f1_act = mul(scale, m_act)
        f2_act = mul(scale, raw2)
        for k in (0, 1):
            one[k][act_idx] = f1_act[k]
            two[k][act_idx] = f2_act[k]

    weights1 = mul(point(np.repeat(wa, nb)), point(np.tile(wb, na)))
    weights2 = mul(point(np.repeat(wa2, nb)), point(np.tile(wb2, na)))
    q1 = reduce_sum_outward(mul(one, weights1))
    q2 = reduce_sum_outward(mul(two, weights2))
    return div(q1, mul(ca[0], cb[0])), div(q2, mul(ca[1], cb[1]))


def integrate_order(a, b, c, g, edges, order, batch_size, deadline, moment_nodes=None, parts=None):
    if len(a) < 64 or len(edges[0]) == 0 or len(edges[0]) > 2**53:
        ka, kb = int(np.count_nonzero(a)), int(np.count_nonzero(b))
        xa, wa, wa2, ca = rule(order, ka)
        xb, wb, wb2, cb = rule(order, kb)
        na, nb = len(xa), len(xb)
        x, y = np.repeat(xa, nb), np.tile(xb, na)
        weights1 = mul(point(np.repeat(wa, nb)), point(np.tile(wb, na)))
        weights2 = mul(point(np.repeat(wa2, nb)), point(np.tile(wb2, na)))
        q1, q2 = point(0.), point(0.)
        for offset in range(0, len(x), batch_size):
            if time.perf_counter() >= deadline:
                raise TimeoutError('BUDGET_EXHAUSTED')
            sl = slice(offset, offset+batch_size)
            f1, f2 = tilted_polynomial(a, b, c, g, edges, x[sl], y[sl], moment_nodes)
            q1 = add(q1, reduce_interval(mul(f1, (weights1[0][sl], weights1[1][sl]))))
            q2 = add(q2, reduce_interval(mul(f2, (weights2[0][sl], weights2[1][sl]))))
        return div(q1, mul(ca[0], cb[0])), div(q2, mul(ca[1], cb[1]))
    return integrate_order_separable(a, b, c, g, edges, order, deadline, moment_nodes, parts=parts)


def _estimate_impl(a, b, adjacency, *, accuracy_factor=1.25, orders=(4, 8, 16, 32, 64),
             max_seconds=30., batch_size=32):
    """One target, a certified point if available, otherwise explicit unresolved."""
    start = time.perf_counter()
    if not np.isfinite(accuracy_factor) or accuracy_factor <= 1:
        raise ValueError('accuracy_factor must be finite and greater than one')
    if not orders or any(not isinstance(q, int) or q < 1 for q in orders):
        raise ValueError('positive integer quadrature orders required')
    if not np.isfinite(max_seconds) or max_seconds <= 0 or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('positive finite runtime and integer batch size required')
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a < 0) or np.any(b < 0) or np.any(a > 1) or np.any(b > 1):
        raise ValueError('finite aligned probabilities in [0,1] required')
    c = residual_probability(a, b)
    g = sparse.csr_matrix(adjacency, dtype=float)
    g.sum_duplicates(); g.eliminate_zeros(); g.sort_indices()
    if g.shape != (len(a), len(a)) or np.any(g.data != 1) or np.any(g.diagonal() != 0) or (g != g.T).nnz:
        raise ValueError('binary symmetric graph without self edges required')
    out = dict(method='direct_conditional_ratio_moments', target='CV(O/E | E>0)',
               accuracy_factor=accuracy_factor, status='UNRESOLVED', cv=None, cv_interval=None,
               achieved_factor=None, p_finite_interval=None, orders_attempted=[],
               n=len(a), m=g.nnz//2, inference='conditional_model_only', elapsed_seconds=None,
               quadrature='dyadic_panels_log_interpolation_v3')

    def finish(status, reason=None):
        out.update(status=status, reason=reason, elapsed_seconds=time.perf_counter()-start)
        return out

    supports_a, supports_b = a > 0, b > 0
    possible = int(supports_a.sum())*int(supports_b.sum())-int(np.count_nonzero(supports_a & supports_b))
    if not out['m'] or possible == 0:
        return finish('NO_FINITE_RATIO')
    active = np.flatnonzero(supports_a | supports_b)
    a, b, c = a[active], b[active], (c[0][active], c[1][active])
    g = g[active][:, active].tocsr()
    ee = sparse.triu(g, k=1, format='coo')
    edges = ee.row, ee.col
    i, j = edges
    present = int(np.count_nonzero((a[i] > 0) & (b[j] > 0)))+int(np.count_nonzero((b[i] > 0) & (a[j] > 0)))
    possible_edge = ((a[i] > 0) & (b[j] > 0)) | ((b[i] > 0) & (a[j] > 0))
    i, j = i[possible_edge], j[possible_edge]
    edges = i, j
    g = sparse.csr_matrix((np.ones(2*len(i)), (np.r_[i,j], np.r_[j,i])), shape=g.shape)
    out['moment_edges'] = len(i)
    out['p_finite_interval'] = p = availability(a, b, c)
    if present == 0:
        return finish('ZERO_NUMERATOR_CV_UNDEFINED')
    deterministic = np.all((a == 0) | (a == 1)) and np.all((b == 0) | (b == 1))
    if present == possible or deterministic:
        out.update(cv=0., cv_interval=(0., 0.))
        return finish('EXACT_ZERO_CV', 'pointwise_constant_positive_ratio')
    if p[0] <= 0:
        return finish('NUMERICAL_UNDERFLOW', 'positive finite support not numerically enclosed away from zero')
    moment_nodes = np.union1d(i, j)
    g = g[moment_nodes][:, moment_nodes].tocsr()
    edges = np.searchsorted(moment_nodes, i), np.searchsorted(moment_nodes, j)
    out['moment_nodes'] = len(moment_nodes)

    parts = prepare_graph_parts(a, b, c, g, moment_nodes) if len(a) >= 64 else None
    out['moment_backend'] = ('full_interval_grid' if parts is None else
        'interval_active_grid_fallback' if len(parts['idx_w']) and not parts['single_point_safe'] else
        'separable_nonnegative')
    j1, j2 = None, None
    for order in orders:
        if time.perf_counter()-start >= max_seconds:
            return finish('UNRESOLVED', 'BUDGET_EXHAUSTED')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                new1, new2 = integrate_order(a, b, c, g, edges, order, batch_size, start+max_seconds, moment_nodes, parts=parts)
                j1 = intersect_enclosures(j1, new1)
                j2 = intersect_enclosures(j2, new2)
                interval = intersect_enclosures(out['cv_interval'], cv_interval(p, j1, j2))
            out.update(cv_interval=interval, first_ratio_moment_interval=tuple(map(float, j1)),
                       second_ratio_moment_interval=tuple(map(float, j2)))
            out['orders_attempted'].append(dict(order=order, status='ENCLOSED', cv_interval=interval,
                                               elapsed_at_order=time.perf_counter()-start))
            lo, hi = interval
            if lo > 0 and np.isfinite(hi):
                center = float(np.sqrt(lo)*np.sqrt(hi))
                factor = float(max(up(center/lo), up(hi/center)))
                if factor <= accuracy_factor:
                    out.update(cv=center, achieved_factor=factor)
                    return finish('RESOLVED')
        except TimeoutError:
            out['orders_attempted'].append(dict(order=order, status='BUDGET_EXHAUSTED',
                                               elapsed_at_order=time.perf_counter()-start))
            return finish('UNRESOLVED', 'BUDGET_EXHAUSTED')
        except InconsistentEnclosure as exc:
            return finish('NUMERICAL_INCONSISTENCY', str(exc))
        except ArithmeticError as exc:
            out['orders_attempted'].append(dict(order=order, status='NUMERICAL_UNRESOLVED', reason=str(exc),
                                               elapsed_at_order=time.perf_counter()-start))
    return finish('UNRESOLVED' if out['cv_interval'] is not None else 'NUMERICAL_UNRESOLVED', 'ORDER_LIMIT')


def _moments_from_factors(ax, by, z, product, c, g, edges, x, y, moment_nodes=None):
    """Original moment identities and outward operations, with existing Z factors."""
    if moment_nodes is not None:
        ax = ax[0][:, moment_nodes], ax[1][:, moment_nodes]
        by = by[0][:, moment_nodes], by[1][:, moment_nodes]
        z = z[0][:, moment_nodes], z[1][:, moment_nodes]
        c = c[0][moment_nodes], c[1][moment_nodes]
    aa, bb, cc = div(ax, z), div(by, z), div(c, z)
    u, v = matvec_interval(g, aa), matvec_interval(g, bb)
    i, j = edges
    ai, aj = (aa[0][:, i], aa[1][:, i]), (aa[0][:, j], aa[1][:, j])
    bi, bj = (bb[0][:, i], bb[1][:, i]), (bb[0][:, j], bb[1][:, j])
    ci, cj = (cc[0][:, i], cc[1][:, i]), (cc[0][:, j], cc[1][:, j])
    mean = reduce_interval(mul(aa, v))
    first = add(add(mul(mul(aa, cc), mul(v, v)), mul(mul(bb, cc), mul(u, u))),
                mul(mul(aa, bb), square_difference(v, u)))
    residual = add(add(mul(mul(ai, add(ci, bi)), mul(bj, add(cj, aj))),
                       mul(mul(bi, add(ci, ai)), mul(aj, add(cj, bj)))),
                   mul(point(2.), mul(mul(ai, bi), mul(aj, bj))))
    raw2 = add(mul(mean, mean), add(reduce_interval(first), reduce_interval(residual)))
    scale = div(product, mul(point(x), point(y)))
    return mul(scale, mean), mul(scale, raw2)


def tilted_polynomial(a, b, c, g, edges, x, y, moment_nodes=None):
    x = np.asarray(x); y = np.asarray(y)
    if len(a) < 64 or len(edges[0]) > 2**53:
        return _tilted_polynomial_full(a, b, c, g, edges, x, y, moment_nodes)
    ax = mul(point(a), point(x[:, None])); by = mul(point(b), point(y[:, None]))
    z = add(add(ax, by), c)
    product = reduce_interval(z, op=mul, identity=1.)
    scale = div(product, mul(point(x), point(y)))
    m = point(float(len(edges[0])))
    bound1 = mul(scale, m); bound2 = mul(bound1, m)
    bypass = (bound1[1] <= TERM_UPPER_LIMIT) & (bound2[1] <= TERM_UPPER_LIMIT)
    one = (np.zeros(len(x)), bound1[1].copy()); two = (np.zeros(len(x)), bound2[1].copy())
    active = ~bypass
    if np.any(active):
        select = lambda v: (v[0][active], v[1][active])
        actual1, actual2 = _moments_from_factors(select(ax), select(by), select(z), select(product),
            c, g, edges, x[active], y[active], moment_nodes)
        for k in (0, 1): one[k][active] = actual1[k]; two[k][active] = actual2[k]
    return one, two


def estimate(a, b, adjacency, *, accuracy_factor=1.25, orders=(4, 8, 16, 32, 64), max_seconds=30., batch_size=32):
    """Certified primary target with a per-call, per-thread fixed-graph cache."""
    previous = getattr(_CV_CONTEXT, 'matrix', None)
    _CV_CONTEXT.matrix = {}
    try:
        return _estimate_impl(a, b, adjacency, accuracy_factor=accuracy_factor, orders=orders,
                              max_seconds=max_seconds, batch_size=batch_size)
    finally:
        if previous is None: del _CV_CONTEXT.matrix
        else: _CV_CONTEXT.matrix = previous
