"""Biologist's operational decision guide for P+ and conditional CV diagnostics.

Translates availability P+ = Pr(E > 0), conditional dispersion CV(R | E > 0),
and structural/numerical status codes into transparent diagnostic regimes and
concrete experimental/analytical actions.

Scope and Guardrails
--------------------
1. Diagnostic Triage vs. Hypothesis-Test Filtering:
   The regimes below are pre-interpretation sensitivity and reporting conventions
   for a specified count-retention operation. As shown in the frozen four-arm
   simulation (QC-FILTER-MIN-01), naive pre-test filtering on P+ and CV does
   not automatically reduce family-wise error rates (FWER) or increase
   permutation-test power (Bourgon et al., 2010).
2. Configurable Conventions:
   Default cutoffs (P+ in {0.50, 0.95}; conditional CV in {0.25, 0.50, 1.00})
   are operational reporting conventions, not universal physical phase transitions.
"""
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


INFERENTIAL_GUARDRAIL = (
    'Operational sensitivity triage only; do not use P+/CV cutoffs as an '
    'uncalibrated pre-test p-value filter claiming automatic FWER reduction '
    'or power gain (see QC-FILTER-MIN-01 and Bourgon et al., 2010).'
)


@dataclass(frozen=True)
class ActionThresholds:
    """Configurable diagnostic thresholds for availability P+ and conditional CV."""
    p_robust: float = 0.95
    p_fragile: float = 0.50
    cv_low: float = 0.25
    cv_moderate: float = 0.50
    cv_high: float = 1.00
    high_ratio_flag: float = 2.0
    non_monotonic_tol: float = 1e-12

    def __post_init__(self):
        if not (0.0 < self.p_fragile < self.p_robust <= 1.0):
            raise ValueError('Require 0 < p_fragile < p_robust <= 1')
        if not (0.0 < self.cv_low < self.cv_moderate <= self.cv_high):
            raise ValueError('Require 0 < cv_low < cv_moderate <= cv_high')
        if self.high_ratio_flag <= 0 or self.non_monotonic_tol < 0:
            raise ValueError('Positive high_ratio_flag and nonnegative non_monotonic_tol required')


ACTION_REGIME_CATALOG: dict[str, dict[str, str]] = {'STRUCTURAL_NO_GRAPH_EDGES': {'severity': 'STRUCTURAL',
                               'range_summary': 'M = 0 (no edges in spatial graph)',
                               'action_code': 'VERIFY_COORDINATE_SCALE_AND_RADIUS',
                               'action_en': 'Verify coordinate units and the specified graph '
                                            'construction. With M=0 the ratio is undefined; retain '
                                            'NA.',
                               'action_zh': '核对坐标单位与预设建图方式；M=0 时比值不可定义，保留 NA。'},
 'STRUCTURAL_ABSENT_POPULATION': {'severity': 'STRUCTURAL',
                                  'range_summary': 'P+ = 0 (NO_FINITE_RATIO / baseline '
                                                   'STRUCTURAL_NA)',
                                  'action_code': 'RECORD_STRUCTURAL_NA_DO_NOT_IMPUTE_ZERO',
                                  'action_en': 'The two exclusive classes cannot coexist under '
                                               'this specified model. Preserve NA; inspect marker '
                                               'coverage and class support.',
                                  'action_zh': '在指定互斥分类模型下，两类不能同时出现；保留 NA，核对 marker 覆盖和类别支持。'},
 'NON_MONOTONIC_VETO_ENTRY': {'severity': 'WARNING',
                              'range_summary': 'Baseline STRUCTURAL_NA with P+(p) > 0 or dP+/dp < '
                                               '0',
                              'action_code': 'INSPECT_EXCLUSION_VETO_MARKERS',
                              'action_en': 'Inspect the complete retention curve and rule '
                                           'transitions. Veto dropout is one possible explanation; '
                                           'the trajectory alone does not establish it.',
                              'action_zh': '展示完整保留率曲线并核查分类规则变化；否决基因脱落是一种待核对解释，非单调轨迹本身不能证明原因。'},
 'DEFINED_ZERO_CONTACT': {'severity': 'CAUTION',
                          'range_summary': 'P+ > 0, E[O | E>0] = 0 (ZERO_NUMERATOR_CV_UNDEFINED)',
                          'action_code': 'REPORT_ZERO_CONTACT_WITH_AVAILABILITY',
                          'action_en': 'Under the fixed graph and label model, conditional contact '
                                       'is zero and relative CV is undefined. Report R=0 with P+; '
                                       'biological repulsion requires a separate null comparison.',
                          'action_zh': '在固定图和标签模型下条件接触为零，相对 CV 不可定义；并列报告 R=0 与 P+，生物学排斥须另有零假设比较。'},
 'NUMERICAL_UNRESOLVED': {'severity': 'UNRESOLVED',
                          'range_summary': 'CV status UNRESOLVED / BUDGET_EXHAUSTED / '
                                           'NUMERICAL_UNDERFLOW',
                          'action_code': 'ROUTE_TO_SCREENING_OR_EXPAND_BUDGET',
                          'action_en': 'Preserve unresolved or missing values. Check inputs, '
                                       'numerical status and declared budget; label any MC/Sobol '
                                       'estimate and its sampling uncertainty.',
                          'action_zh': '保留未解决或缺失状态；核对输入、数值状态与预算，若使用 MC/Sobol，注明近似方法及抽样不确定性。'},
 'TIER_1_ROBUST': {'severity': 'OK',
                   'range_summary': 'P+ >= 0.95 and conditional CV <= 0.25',
                   'action_code': 'PROCEED_STANDARD_INTERPRETATION',
                   'action_en': 'High availability and low conditional dispersion under the '
                                'specified retention model. Report P+, CV and method; this '
                                'diagnostic does not validate biological interpretation or '
                                'cross-sample exchangeability.',
                   'action_zh': '在指定保留模型下可用率高、条件离散度低；报告 P+、CV 及计算方法，这一诊断不验证生物学解释或跨样本可比性。'},
 'TIER_2_MODERATE_DISPERSION': {'severity': 'CAUTION',
                                'range_summary': 'P+ >= 0.95 and 0.25 < conditional CV <= 0.50',
                                'action_code': 'USE_ANALYTIC_EXPECTATION_AVOID_SINGLE_DRAW',
                                'action_en': 'Report availability and conditional dispersion; '
                                             'include the conditional mean when relevant. '
                                             'Numerical enclosures and biological sampling '
                                             'intervals answer different questions.',
                                'action_zh': '并列报告可用率、条件离散度及必要的条件均值；数值包围区间与生物学抽样区间回答不同问题。'},
 'TIER_3_CONTACT_VOLATILE': {'severity': 'WARNING',
                             'range_summary': 'P+ >= 0.95 and conditional CV > 0.50 (extreme if CV '
                                              '> 1.0)',
                             'action_code': 'INSPECT_SPARSE_CONTACTS_AND_BOUNDARY_CELLS',
                             'action_en': 'High conditional dispersion warrants inspecting contact '
                                          'counts and rule-boundary cells. CV alone identifies '
                                          'neither the cause nor the number of influential cells.',
                             'action_zh': '条件离散度较高，可核查接触计数与门控边界细胞；仅凭 CV 不能确定原因或关键细胞数量。'},
 'TIER_4_AVAILABILITY_WARNING': {'severity': 'WARNING',
                                 'range_summary': '0.50 <= P+ < 0.95 (moderate denominator dropout '
                                                  'risk)',
                                 'action_code': 'REPORT_AVAILABILITY_AND_POOL_WINDOWS_IF_NEEDED',
                                 'action_en': 'Report P+ alongside conditional CV. Treat changes '
                                              'to ROI size, radius or thresholds as explicitly '
                                              'labelled sensitivity analyses with a potentially '
                                              'different estimand.',
                                 'action_zh': '并列报告 P+ 与条件 CV；改变 ROI、半径或门槛须明确标为敏感性分析，并说明目标量可能改变。'},
 'TIER_5A_CONDITIONING_TRAP': {'severity': 'CRITICAL',
                               'range_summary': 'P+ < 0.50 and conditional CV <= 0.50 (The '
                                                'Conditioning Trap)',
                               'action_code': 'DO_NOT_TRUST_LOW_CV_ALONE_FLAG_DENOMINATOR_FRAGILE',
                               'action_en': 'A low conditional CV can coexist with frequent '
                                            'undefined ratios. Retain both summaries; low CV alone '
                                            'is insufficient evidence of availability or '
                                            'biological robustness.',
                               'action_zh': '低条件 CV 可以与频繁不可定义同时出现；保留两项结果，不能仅凭低 CV 推断高可用性或生物学稳健性。'},
 'TIER_5B_SEVERE_FRAGILITY': {'severity': 'CRITICAL',
                              'range_summary': 'P+ < 0.50 and conditional CV > 0.50 (Severe '
                                               'Availability & Ratio Fragility)',
                              'action_code': 'DO_NOT_INTERPRET_SHALLOWER_ABSENCE_AS_BIOLOGY',
                              'action_en': 'Report frequent undefined ratios and high conditional '
                                           'dispersion. Do not infer biological absence from '
                                           'thinning dropout; any pooling or exclusion rule needs '
                                           'separate validation.',
                              'action_zh': '报告频繁不可定义与高条件离散度；不能将抽稀失定义推作生物学缺失，合并或剔除规则须另行验证。'}}


def _extract_p_plus(record: Mapping) -> float | None:
    for key in ('p_finite', 'P_plus', 'p_finite_direct', 'p_D_positive', 'p_finite_estimate'):
        if key in record and record[key] is not None and record[key] != '':
            val = float(record[key])
            if math.isfinite(val):
                return val
    if 'availability' in record and isinstance(record['availability'], Mapping):
        return _extract_p_plus(record['availability'])
    if 'p_finite_interval' in record and record['p_finite_interval'] is not None:
        iv = record['p_finite_interval']
        if len(iv) == 2 and iv[0] is not None and iv[1] is not None:
            return 0.5 * (float(iv[0]) + float(iv[1]))
    return None


def _extract_cv(record: Mapping) -> float | None:
    for key in ('cv', 'cv_R_delta', 'cv_R_finite'):
        if key in record and record[key] is not None and record[key] != '':
            val = float(record[key])
            if math.isfinite(val):
                return val
    if 'direct_cv' in record and isinstance(record['direct_cv'], Mapping):
        return _extract_cv(record['direct_cv'])
    return None


def _extract_status(record: Mapping) -> str:
    for key in ('cv_status', 'status'):
        if key in record and record[key]:
            return str(record[key])
    if 'direct_cv' in record and isinstance(record['direct_cv'], Mapping):
        return _extract_status(record['direct_cv'])
    return 'UNKNOWN'


def _extract_baseline_r(record: Mapping) -> float | None:
    for key in ('baseline_R', 'R_baseline', 'R0'):
        if key in record and record[key] is not None and record[key] != '':
            val = float(record[key])
            if math.isfinite(val):
                return val
    return None


def triage_audit_record(
    record: Mapping,
    *,
    thresholds: ActionThresholds = ActionThresholds(),
    baseline_state: str | None = None,
    is_non_monotonic: bool = False,
) -> dict:
    """Classify one spatial audit record into an actionable operational regime.

    Parameters
    ----------
    record : Mapping
        Output dictionary from ``pair_moments``, ``estimate_cv``, ``route_cv``,
        ``regional_joint_qc``, or a saved retention-curve row.
    thresholds : ActionThresholds
        Configurable diagnostic thresholds for P+ and conditional CV.
    baseline_state : str, optional
        Baseline state at p=1.0 (e.g. ``'FINITE_POSITIVE'``, ``'ZERO_R'``,
        ``'STRUCTURAL_NA'``, ``'NO_EDGES'``).
    is_non_monotonic : bool, default False
        Whether P+(p) increased over at least one downward retention step.

    Returns
    -------
    triage : dict
        Contains ``regime``, ``severity``, ``action_code``, ``action_en``,
        ``action_zh``, ``conditioning_trap``, ``high_ratio_fragility_warning``,
        ``p_plus``, ``cv``, ``status``, and ``inferential_guardrail``.
    """
    status = _extract_status(record)
    b_state = str(baseline_state or record.get('baseline_state') or '')
    p_plus = _extract_p_plus(record)
    cv = _extract_cv(record)
    base_r = _extract_baseline_r(record)
    edges = record.get('m', record.get('edges', None))
    mc_input = (record.get('probability_method') == 'monte_carlo' or
                record.get('semantics') == 'frozen_cellwise_classifier_thinning_mc')
    upstream_sampling_unresolved = (
        status == 'UNRESOLVED_UPSTREAM_SAMPLING' or
        record.get('probability_support_status') == 'UNRESOLVED_MC_SUPPORT' or
        (mc_input and (p_plus == 0.0 or status in (
            'NO_FINITE_RATIO', 'NO_FINITE_DENOMINATOR', 'ABSENT_TYPE',
            'ZERO_NUMERATOR_CV_UNDEFINED',
        )))
    )

    # 1. Graph has no edges
    if b_state == 'NO_EDGES' or status == 'NO_EDGES' or (edges is not None and int(edges) == 0):
        regime = 'STRUCTURAL_NO_GRAPH_EDGES'
    elif upstream_sampling_unresolved:
        # This must precede the structural-zero branches, including when the
        # caller passes a raw conditional graph-engine status with MC metadata.
        regime = 'NUMERICAL_UNRESOLVED'
    # 2. Non-monotonic rule-boundary / veto entry
    elif is_non_monotonic or (b_state == 'STRUCTURAL_NA' and p_plus is not None and p_plus > thresholds.non_monotonic_tol):
        regime = 'NON_MONOTONIC_VETO_ENTRY'
    # 3. Structural absence / zero denominator availability
    elif status in ('NO_FINITE_RATIO', 'NO_FINITE_DENOMINATOR', 'ABSENT_TYPE') or (p_plus is not None and p_plus == 0.0):
        regime = 'STRUCTURAL_ABSENT_POPULATION'
    # 4. Positive denominator availability, but conditional numerator mean is zero
    elif status == 'ZERO_NUMERATOR_CV_UNDEFINED':
        regime = 'DEFINED_ZERO_CONTACT'
    # 5. Baseline p=1 skipped CV calculation
    elif status == 'SKIPPED_P1' or (p_plus is not None and p_plus == 1.0 and cv is None and b_state in ('FINITE_POSITIVE', 'ZERO_R')):
        regime = 'DEFINED_ZERO_CONTACT' if b_state == 'ZERO_R' else 'TIER_1_ROBUST'
    # 6. Numerical or budget unresolved states
    elif p_plus is None or not 0 <= p_plus <= 1 or cv is None or cv < 0 or status in (
        'UNRESOLVED', 'BUDGET_EXHAUSTED', 'RESOURCE_LIMIT',
        'NUMERICAL_UNDERFLOW', 'NUMERICAL_UNRESOLVED', 'NUMERICAL_INCONSISTENCY',
        'UNRESOLVED_UPSTREAM_PRECISION', 'SAMPLE_NO_FINITE_OBSERVATIONS',
        'SAMPLE_NO_POSITIVE_CONTACTS', 'SAMPLE_INSUFFICIENT_FINITE_OBSERVATIONS',
    ):
        regime = 'NUMERICAL_UNRESOLVED'
    else:
        # Both P+ and CV are available (or CV is resolved on E>0)
        eff_p = p_plus
        if eff_p < thresholds.p_fragile:
            if cv <= thresholds.cv_moderate:
                regime = 'TIER_5A_CONDITIONING_TRAP'
            else:
                regime = 'TIER_5B_SEVERE_FRAGILITY'
        elif eff_p < thresholds.p_robust:
            regime = 'TIER_4_AVAILABILITY_WARNING'
        else:
            if cv <= thresholds.cv_low:
                regime = 'TIER_1_ROBUST'
            elif cv <= thresholds.cv_moderate:
                regime = 'TIER_2_MODERATE_DISPERSION'
            else:
                regime = 'TIER_3_CONTACT_VOLATILE'

    info = ACTION_REGIME_CATALOG[regime]
    if regime == 'NUMERICAL_UNRESOLVED' and upstream_sampling_unresolved:
        info = dict(info, action_code='RESOLVE_CLASSIFIER_MC_SUPPORT',
                    range_summary='Finite classifier MC does not certify zero support',
                    action_en='Preserve the MC point estimate and unresolved state. '
                              'An unobserved class/contact is not structurally impossible. '
                              'Check classifier support or increase draws within the declared budget; '
                              'downstream numerical enclosures exclude upstream MC error.',
                    action_zh='保留 MC 点估计和抽样未解状态；未抽中的类别或接触不能判为结构不可能。'
                              '核对分类器支持范围，或在既定预算内增加抽样；下游数值区间不包含上游 MC 误差。')
    conditioning_trap = bool(
        not upstream_sampling_unresolved and p_plus is not None and cv is not None and (
            (p_plus < thresholds.p_fragile and cv <= thresholds.cv_moderate)
            or (p_plus < thresholds.p_robust and cv <= thresholds.cv_low)
        )
    )
    high_ratio_fragility = bool(
        not upstream_sampling_unresolved and base_r is not None and base_r >= thresholds.high_ratio_flag
        and p_plus is not None and p_plus < thresholds.p_fragile
    )
    sub_note = None
    if regime == 'TIER_3_CONTACT_VOLATILE' and cv is not None and cv > thresholds.cv_high:
        sub_note = 'EXTREME_CV_ABOVE_1'
    elif regime == 'TIER_4_AVAILABILITY_WARNING' and conditioning_trap:
        sub_note = 'LOW_CV_WITH_MODERATE_AVAILABILITY_DROPOUT'

    return dict(
        regime=regime,
        severity=info['severity'],
        action_code=info['action_code'],
        range_summary=info['range_summary'],
        action_en=info['action_en'],
        action_zh=info['action_zh'],
        conditioning_trap=conditioning_trap,
        high_ratio_fragility_warning=high_ratio_fragility,
        sub_flag=sub_note,
        p_plus=p_plus,
        cv=cv,
        status=status,
        baseline_state=b_state if b_state else None,
        is_non_monotonic=bool(is_non_monotonic),
        inferential_guardrail=INFERENTIAL_GUARDRAIL,
    )


def _scalar_retention(record: Mapping, p_key: str) -> float | None:
    """Return an orderable scalar retention; vectors have no implicit depth order."""
    if record.get('retention_kind') == 'gene_specific':
        return None
    value = record.get(p_key)
    if value is None or np.ndim(value) != 0:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None


def detect_retention_non_monotonicity(
    records: Sequence[Mapping],
    *,
    group_keys: Sequence[str] = ('region', 'label_a', 'label_b'),
    p_key: str = 'p',
    tol: float = 1e-12,
) -> dict[tuple, dict]:
    """Identify (region, pair) trajectories where P+(p) increases as p decreases.

    Returns a mapping from group key tuple to a summary dict with
    ``is_non_monotonic``, ``max_adjacent_increase``, and ``max_increase_from_p1``.
    """
    grouped: dict[tuple, list[tuple[float, float]]] = {}
    for row in records:
        p_val = _scalar_retention(row, p_key)
        if p_val is None:
            continue
        p_plus = _extract_p_plus(row)
        if p_plus is None:
            continue
        if all(k in row for k in group_keys):
            key = tuple(row[k] for k in group_keys)
        elif 'index' in row:
            key = (str(row['index']),)
        else:
            continue
        grouped.setdefault(key, []).append((p_val, p_plus))

    out: dict[tuple, dict] = {}
    for key, pts in grouped.items():
        # Sort from highest retention (e.g. p=1.0) down to lowest retention (e.g. p=0.1)
        ordered = sorted(pts, key=lambda x: -x[0])
        if len(ordered) < 2:
            out[key] = dict(is_non_monotonic=False, max_adjacent_increase=0.0, max_increase_from_p1=0.0,
                            n_scalar_points=len(ordered))
            continue
        vals = [v for _, v in ordered]
        adj_inc = max(vals[i + 1] - vals[i] for i in range(len(vals) - 1))
        from_top = max(v - vals[0] for v in vals[1:])
        out[key] = dict(
            is_non_monotonic=bool(adj_inc > tol),
            max_adjacent_increase=max(0.0, float(adj_inc)),
            max_increase_from_p1=max(0.0, float(from_top)),
            n_scalar_points=len(ordered),
        )
    return out


def triage_audit_table(
    records_or_df,
    *,
    thresholds: ActionThresholds = ActionThresholds(),
    group_keys: Sequence[str] = ('region', 'label_a', 'label_b'),
    p_key: str = 'p',
):
    """Attach operational triage columns to a list of audit dicts or a pandas DataFrame.

    Automatically detects non-monotonic P+(p) trajectories when multiple retention
    levels ``p`` are present for the same ``group_keys`` (or ``index``).
    """
    import pandas as pd

    is_df = isinstance(records_or_df, pd.DataFrame)
    records = records_or_df.to_dict(orient='records') if is_df else [dict(r) for r in records_or_df]
    if not records:
        return pd.DataFrame() if is_df else []

    active_keys = tuple(group_keys) if all(k in records[0] for k in group_keys) else (
        ('index',) if 'index' in records[0] else ()
    )
    non_mono_map = (
        detect_retention_non_monotonicity(records, group_keys=active_keys, p_key=p_key, tol=thresholds.non_monotonic_tol)
        if active_keys and p_key in records[0]
        else {}
    )

    enriched = []
    for row in records:
        key = tuple(row[k] for k in active_keys) if active_keys and all(k in row for k in active_keys) else None
        nm_info = non_mono_map.get(key, {}) if key is not None else {}
        p_val = _scalar_retention(row, p_key)
        is_nm = p_val is not None and bool(nm_info.get('is_non_monotonic', False))
        # Only flag non-monotonicity at thinned points p < 1 where P+ > 0 or along the trajectory
        flag_nm = is_nm and p_val < 1.0
        triage = triage_audit_record(
            row,
            thresholds=thresholds,
            baseline_state=row.get('baseline_state'),
            is_non_monotonic=flag_nm,
        )
        merged = dict(row)
        merged.update(
            regime=triage['regime'],
            severity=triage['severity'],
            action_code=triage['action_code'],
            conditioning_trap=triage['conditioning_trap'],
            high_ratio_fragility_warning=triage['high_ratio_fragility_warning'],
            is_non_monotonic_trajectory=is_nm,
            max_adjacent_p_plus_increase=nm_info.get('max_adjacent_increase', 0.0) if p_val is not None else None,
            retention_curve_status=(
                'NOT_EVALUATED_NONSCALAR_RETENTION' if p_val is None else
                'EVALUATED_SCALAR_CURVE' if nm_info.get('n_scalar_points', 0) > 1 else
                'NOT_EVALUATED_SINGLE_POINT'
            ),
            action_en=triage['action_en'],
            action_zh=triage['action_zh'],
            inferential_guardrail=triage['inferential_guardrail'],
        )
        enriched.append(merged)

    return pd.DataFrame(enriched) if is_df else enriched


def summarize_operational_actions(triaged_records_or_df, *, p_key: str = 'p') -> dict:
    """Summarize regime counts, conditioning-trap incidence, and action priorities."""
    import pandas as pd

    rows = (
        triaged_records_or_df.to_dict(orient='records')
        if isinstance(triaged_records_or_df, pd.DataFrame)
        else list(triaged_records_or_df)
    )
    if rows and 'regime' not in rows[0]:
        rows = triage_audit_table(rows, p_key=p_key)

    regime_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    by_p: dict[str, dict[str, int]] = {}
    trap_count = 0
    high_r_fragile_count = 0
    non_mono_count = 0

    for r in rows:
        reg = str(r['regime'])
        sev = str(r['severity'])
        regime_counts[reg] = regime_counts.get(reg, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        if r.get('conditioning_trap'):
            trap_count += 1
        if r.get('high_ratio_fragility_warning'):
            high_r_fragile_count += 1
        if r.get('is_non_monotonic_trajectory'):
            non_mono_count += 1
        if p_key in r and r[p_key] is not None:
            pk = str(r[p_key])
            by_p.setdefault(pk, {})
            by_p[pk][reg] = by_p[pk].get(reg, 0) + 1

    return dict(
        total_records=len(rows),
        regime_counts=regime_counts,
        severity_counts=severity_counts,
        regime_counts_by_p=by_p,
        conditioning_trap_records=trap_count,
        high_ratio_fragility_records=high_r_fragile_count,
        non_monotonic_trajectory_records=non_mono_count,
        inferential_guardrail=INFERENTIAL_GUARDRAIL,
    )
