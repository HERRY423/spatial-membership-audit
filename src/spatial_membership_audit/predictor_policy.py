"""Optional decision layer. No fitting, simulation, or implicit budget changes.

Loss is absolute log error on a strictly positive population CV. Agreement of
two approximations controls regret against those approximations, NOT accuracy.
Reference enclosures must refer to the same fixed-input conditional model.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np


@dataclass(frozen=True)
class CVInterval:
    lower: float
    upper: float
    kind: str
    evidence_id: str
    alpha: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.lower) or self.lower < 0:
            raise ValueError("finite nonnegative lower bound required")
        if math.isnan(self.upper) or self.upper < self.lower:
            raise ValueError("ordered nonnegative bounds required")
        if self.kind not in {"exact_enclosure", "fixed_budget_hoeffding", "approximate_mc"}:
            raise ValueError("unknown reference kind")
        if not self.evidence_id:
            raise ValueError("reference provenance required")
        if self.kind == "fixed_budget_hoeffding" and (
            self.alpha is None or not 0 < self.alpha < 1
        ):
            raise ValueError("fixed-budget failure probability required")


def bounded_cv_interval(values, upper_bound, *, alpha, evidence_id):
    """Pointwise, fixed-budget Hoeffding enclosure, conditional on finite count.

    Input is iid R=O/E at a FIXED predeclared draw budget. NaN represents E=0;
    finite zeros MUST be retained. No data-dependent stopping/retry is allowed.
    Caller supplies a proven deterministic R bound (not the sample maximum).
    For a simple binary graph, R <= N*(N-1)/(2*M). On the finite subset,
    X=R/upper_bound lies in [0,1]; simultaneous bounds for E[X], E[X^2]
    yield a CV enclosure. Missing/zero samples never prove structural zero.

    Coverage is a mathematical statement under iid/bound assumptions. The
    floating implementation includes a roundoff guard, not a formal verified
    interval-arithmetic proof. Multiple records/looks need allocated alpha.
    """
    if not math.isfinite(upper_bound) or upper_bound <= 0 or not 0 < alpha < 1:
        raise ValueError("positive finite support bound and alpha in (0,1) required")
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or np.isinf(v).any():
        raise ValueError("one dimensional samples; only NaN may mark E=0")
    finite = v[~np.isnan(v)]
    if np.any(finite < 0) or np.any(finite > upper_bound):
        raise ValueError("sample outside declared model support")
    n = len(finite)
    metadata = dict(draws=len(v), finite_n=n, zero_n=int(np.sum(finite == 0)),
                    positive_n=int(np.sum(finite > 0)), optional_stopping_allowed=False)
    if not n:
        return CVInterval(0., math.inf, "fixed_budget_hoeffding", evidence_id, alpha), metadata
    x = finite / upper_bound
    mean, second = float(np.mean(x)), float(np.mean(x*x))
    # Union of two two-sided Hoeffding inequalities: 4 exp(-2 n eps^2).
    eps = math.sqrt(math.log(4/alpha)/(2*n)) + 64*np.finfo(float).eps
    ml, mu = max(0., mean-eps), min(1., mean+eps)
    ql, qu = max(0., second-eps), min(1., second+eps)
    lo2 = max(0., ql/(mu*mu)-1.) if mu else 0.
    hi2 = max(0., qu/(ml*ml)-1.) if ml else math.inf
    lo = max(0., math.nextafter(math.sqrt(lo2), -math.inf))
    hi = math.nextafter(math.sqrt(hi2), math.inf)
    metadata.update(mean_normalized=mean, second_normalized=second, epsilon=eps)
    return CVInterval(lo, hi, "fixed_budget_hoeffding", evidence_id, alpha), metadata


def _positive(value):
    return value is not None and math.isfinite(value) and value > 0


def _geomean(a, b):
    # Avoid a*b overflow/underflow.
    return math.exp(math.log(a)/2 + math.log(b)/2)


def _exp_display(log_value):
    """None means finite mathematical factor exceeds floating display range."""
    try:
        return math.exp(log_value)
    except OverflowError:
        return None


def _fraction_display(value):
    try:
        return float(value)
    except OverflowError:
        return None


def assess_predictors(delta, expected_o, *, regret_factor, accuracy_factor,
                      reference=None, reference_role="deployment"):
    """Assess candidates with an explicit user-chosen tolerance.

    regret_factor: allowed multiplicative excess error versus the better of
      these TWO candidates. It is not an absolute-CV accuracy tolerance.
    accuracy_factor: desired per-record multiplicative error against the
      conditional population CV (stronger than a collection-average F gate).
    reference_role='evaluation_only' NEVER enables deployment certification.
    approximate MC bounds are displayed but NEVER promoted to rigorous bounds.
    An accepted interval is caller-supplied evidence, not verified by its tag.
    """
    if not all(math.isfinite(x) and x >= 1 for x in (regret_factor, accuracy_factor)):
        raise ValueError("finite tolerance factors >= 1 required")
    if reference_role not in {"deployment", "evaluation_only"}:
        raise ValueError("explicit reference role required")
    out = dict(delta=delta, expected_O=expected_o, policy_version="2",
               ranking="UNRESOLVED", recommended_value=None, recommended_method=None,
               accuracy_status="UNRESOLVED", absolute_error_factor_bound=None,
               compromise_regret_status="NOT_AVAILABLE",
               compromise_regret_factor_bound=None, compromise_regret_log_bound=None,
               recommended_regret_factor_bound=None,
               reference_kind=reference.kind if reference else None,
               reference_role=reference_role, reference_id=reference.evidence_id if reference else None)
    if not (_positive(delta) and _positive(expected_o)):
        out["status"] = "NONPOSITIVE_OR_UNAVAILABLE_CANDIDATE"
        return out
    d, s = float(delta), float(expected_o)
    fd, fs = Fraction(d), Fraction(s)
    gap = abs(math.log(d)-math.log(s))
    out.update(log_disagreement=gap, geometric_compromise=_geomean(d, s),
               compromise_regret_factor_bound=_exp_display(gap/2),
               compromise_regret_log_bound=gap/2)
    out["compromise_regret_bound_output"] = "geometric_compromise"
    out["compromise_factor_display_status"] = (
        "FINITE" if out["compromise_regret_factor_bound"] is not None else "OUT_OF_FLOAT_RANGE")
    # Compare rational numbers to avoid rounding-dependent threshold decisions.
    within = max(fd, fs) <= min(fd, fs)*Fraction(regret_factor)**2
    out["compromise_regret_status"] = "WITHIN_TOLERANCE" if within else "EXCEEDS_TOLERANCE"
    if within:
        out.update(recommended_value=out["geometric_compromise"],
                   recommended_method="geometric_compromise",
                   recommended_regret_factor_bound=out["compromise_regret_factor_bound"],
                   status="RELATIVE_REGRET_ONLY")
    else:
        out["status"] = "NEEDS_MODEL_REFERENCE"
    if d == s:
        out["ranking"] = "IDENTICAL_PREDICTIONS"
    if reference is None or reference_role == "evaluation_only":
        return out
    if reference.kind == "approximate_mc":
        out["reference_use"] = "APPROXIMATE_ONLY_NOT_CERTIFIED"
        return out
    lo, hi = reference.lower, reference.upper
    if lo <= 0 or not math.isfinite(hi):
        out["reference_use"] = "INSUFFICIENT_POSITIVE_ENCLOSURE"
        return out
    fl, fu = Fraction(lo), Fraction(hi)
    midpoint2 = fd*fs
    if d != s:
        if fu*fu < midpoint2:
            out["ranking"] = "DELTA" if d < s else "EXPECTED_O"
        elif fl*fl > midpoint2:
            out["ranking"] = "DELTA" if d > s else "EXPECTED_O"
    out["reference_use"] = "CONDITIONAL_ON_SUPPLIED_ENCLOSURE"
    out["coverage_alpha"] = reference.alpha
    # Ranking and absolute accuracy are separate. If an interval is available,
    # its own geometric center has the smallest worst-case log error over it.
    candidates = {name: max(value/fl, fu/value)
                  for name, value in (("delta", fd), ("expected_O", fs))}
    out["candidate_accuracy_pass"] = {
        name: bound <= Fraction(accuracy_factor) for name, bound in candidates.items()}
    out["candidate_error_factor_bounds"] = {k: _fraction_display(v) for k, v in candidates.items()}
    out["candidate_error_factor_display_status"] = {
        k: "FINITE" if out["candidate_error_factor_bounds"][k] is not None else "OUT_OF_FLOAT_RANGE"
        for k in candidates}
    if fu <= fl*Fraction(accuracy_factor)**2:
        out.update(recommended_value=_geomean(lo, hi), recommended_method="model_reference_center",
                   recommended_regret_factor_bound=None,
                   absolute_error_factor_bound=_exp_display((math.log(hi)-math.log(lo))/2),
                   accuracy_status="ENCLOSURE_PASSES_REQUESTED_FACTOR",
                   status="MODEL_REFERENCE_RESOLVED")
    return out


def assess_moment_output(moments, *, regret_factor, accuracy_factor, **kwargs):
    """Adapter for either existing pair_moments implementation; keeps NA states."""
    if moments.get("status") != "DELTA_APPROXIMATION":
        return dict(status=moments.get("status", "UNKNOWN"),
                    recommended_value=None, ranking="NOT_APPLICABLE",
                    accuracy_status="UNRESOLVED")
    mu = moments.get("mu_O_finite")
    simple = 1/math.sqrt(mu) if _positive(mu) else None
    return assess_predictors(moments.get("cv_R_delta"), simple,
                             regret_factor=regret_factor,
                             accuracy_factor=accuracy_factor, **kwargs)
