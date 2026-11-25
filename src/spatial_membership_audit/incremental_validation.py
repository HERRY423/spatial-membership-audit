"""Small, explicit building blocks for retrospective incremental-value checks.

These functions do not confer external validation or recommend a correction.
The caller must freeze groups, features, losses and comparators beforehand.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def shrink_depth_retention(mean_a, mean_b, strength=0.5):
    """Shrink gene log contrasts towards the panel-total log contrast.

    strength=0 is scalar, strength=1 is gene-wise for positive genes.
    Zero-boundary genes retain explicit original deletion semantics at every
    strength; they are not stabilized by a pseudocount.
    """
    a, b = np.asarray(mean_a, float), np.asarray(mean_b, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("aligned nonempty gene means required")
    if not np.isfinite(np.r_[a, b]).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("finite nonnegative means required")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    pa, pb = np.ones_like(a), np.ones_like(b)
    status = np.full(len(a), "BOTH_ABSENT", dtype=object)
    only_a, only_b = (a > 0) & (b == 0), (a == 0) & (b > 0)
    pa[only_a], pb[only_b] = 0, 0
    status[only_a | only_b] = "COMMON_DEPTH_ZERO"
    both = (a > 0) & (b > 0)
    if both.any():
        total = np.log(a.sum()) - np.log(b.sum())
        contrast = strength * (np.log(a[both]) - np.log(b[both])) + (1-strength)*total
        pa[both] = np.exp(-np.maximum(contrast, 0))
        pb[both] = np.exp(np.minimum(contrast, 0))
        status[both] = np.where(contrast > 0, "THIN_A", np.where(contrast < 0, "THIN_B", "TIED"))
    return pa, pb, status


def train_only_design(train_numeric, test_numeric, train_classes, test_classes, classes):
    """Fit numerical centering/scaling on training rows only; no imputation."""
    train, test = np.asarray(train_numeric, float), np.asarray(test_numeric, float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1] or not len(train):
        raise ValueError("nonempty training matrix and aligned test matrix required")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("missing/nonfinite features must be retained as unavailable")
    names = list(classes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate class names")
    trc, tec = np.asarray(train_classes), np.asarray(test_classes)
    if len(trc) != len(train) or len(tec) != len(test):
        raise ValueError("class vectors must align with rows")
    if not set(trc).issubset(names) or not set(tec).issubset(names):
        raise ValueError("unknown class")
    mean, scale = train.mean(axis=0), train.std(axis=0)
    constant = scale == 0
    scale[constant] = 1
    def form(x, c):
        onehot = np.column_stack([c == name for name in names]).astype(float)
        return np.column_stack([np.ones(len(x)), onehot, (x-mean)/scale])
    return form(train, trc), form(test, tec), dict(
        numeric_mean=mean.tolist(), numeric_scale=scale.tolist(),
        constant_numeric=constant.tolist(), classes=names,
        intercept_penalized=False, onehot_penalized=True)


def fit_predict_fixed(train_x, train_y, test_x, *, task, penalty=1.0):
    """Fixed ridge or logistic fit. No held-out labels or tuning accepted."""
    x, y, test = np.asarray(train_x, float), np.asarray(train_y, float), np.asarray(test_x, float)
    if x.ndim != 2 or y.shape != (len(x),) or test.ndim != 2 or test.shape[1] != x.shape[1]:
        raise ValueError("design and response shape mismatch")
    if not len(x) or not np.isfinite(np.r_[x.ravel(), y, test.ravel()]).all():
        raise ValueError("finite training and test values required")
    if not np.all(x[:, 0] == 1) or not np.all(test[:, 0] == 1):
        raise ValueError("first design column must be intercept")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("fixed positive penalty required")
    reg = np.ones(x.shape[1]); reg[0] = 0
    if task == "continuous":
        coef = np.linalg.solve(x.T@x + penalty*np.diag(reg), x.T@y)
        pred = np.maximum(test@coef, 0)
        detail = {"status": "OK", "iterations": 1}
    elif task == "binary":
        if not set(y).issubset({0, 1}) or len(np.unique(y)) != 2:
            raise ValueError("binary training requires both outcome classes")
        def objective(coef):
            z = x@coef
            loss = np.logaddexp(0, z).sum() - y@z + penalty/2*np.dot(coef*reg, coef)
            grad = x.T@(expit(z)-y) + penalty*coef*reg
            return loss, grad
        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success:
            raise RuntimeError("FIT_FAILED: " + str(result.message))
        coef, pred = result.x, expit(test@result.x)
        detail = {"status": "OK", "iterations": int(result.nit), "gradient_max": float(np.max(np.abs(result.jac)))}
    else:
        raise ValueError("task must be binary or continuous")
    return pred, dict(**detail, task=task, penalty=penalty, coefficients=coef.tolist())


def group_bootstrap_difference(values_new, values_base, groups, *, statistic="mean", draws=2000, seed=2026102609):
    """Paired group bootstrap, conditional on saved fits; groups stay intact."""
    a, b, g = np.asarray(values_new, float), np.asarray(values_base, float), np.asarray(groups)
    if a.ndim != 1 or a.shape != b.shape or g.shape != a.shape or not len(a):
        raise ValueError("aligned nonempty paired values and groups required")
    if not np.isfinite(np.r_[a, b]).all():
        raise ValueError("declare common finite support before bootstrapping")
    if statistic not in ("mean", "median") or not isinstance(draws, int) or draws < 1:
        raise ValueError("invalid statistic or draws")
    levels = np.unique(g); indices = [np.flatnonzero(g == k) for k in levels]
    rng = np.random.default_rng(seed); delta = a-b
    function = np.mean if statistic == "mean" else np.median
    result = np.empty(draws)
    for i in range(draws):
        picked = rng.integers(len(levels), size=len(levels))
        result[i] = function(delta[np.concatenate([indices[j] for j in picked])])
    lo, hi = np.quantile(result, [.025, .975])
    return dict(value=float(function(delta)), low=float(lo), high=float(hi),
                draws=draws, groups=len(levels), rows=len(a), seed=seed,
                uncertainty="descriptive group resampling conditional on saved fit; not patient inference")


def incremental_gate(primary_effect, stratum_differences, *, coverage_not_worse):
    """Strict internal gate; does not establish external utility."""
    diffs = list(stratum_differences)
    finite = all(v is not None and np.isfinite(v) for v in diffs)
    effect_ok = all(primary_effect.get(k) is not None and np.isfinite(primary_effect[k])
                    for k in ("value", "high"))
    passed = bool(effect_ok and primary_effect["value"] < 0 and primary_effect["high"] < 0
                  and finite and len(diffs) > 0 and all(v < 0 for v in diffs) and coverage_not_worse)
    return dict(status="INTERNAL_INCREMENT_SUPPORTED" if passed else "INTERNAL_INCREMENT_NOT_ESTABLISHED",
                passed=passed, coverage_not_worse=bool(coverage_not_worse),
                all_prespecified_strata_improve=bool(finite and diffs and all(v < 0 for v in diffs)),
                external_validation="NOT_ESTABLISHED")
