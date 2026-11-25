"""Depth-standardized rarefaction-extrapolation, conditional coverage matching, DUA inference, and stability weighting."""
from .coverage_matching import (
    ConditionalInstability,
    CoverageMatchResult,
    SplitHalfCurve,
    build_split_half_curve,
    conditional_instability_metrics,
    match_split_half_coverage,
    split_half_disagreement,
)
from .depth_matching import (
    AutoQSelection,
    BidirectionalMatch,
    bidirectional_retention,
    marginal_depth_ratio,
    select_auto_q,
    thin_histogram_summary,
)
from .dua_inference import (
    DUAResult,
    conformal_prediction_band,
    decompose_dua_uncertainty,
    exact_perm_z_from_draws,
    joint_perm_pairs_from_draws,
    summarize_dua_draws,
)
from .extrapolation import (
    ExtrapolationResult,
    ThinningLadder,
    build_extrapolation_guardrail,
    extrapolate_ladder,
    noise_coordinate,
    select_mitm_target_depth,
    summarize_thinning_ladder,
    target_noise_coordinate,
)
from .stability_weighting import (
    StabilityWeightedResult,
    cell_stability_weights,
    exact_weighted_perm_pair,
    stability_weighted_colocalization,
)
from .standardize import (
    DEFAULT_PROTOCOL_V3,
    StandardizedComparison,
    get_default_protocol_v3,
    standardize_colocalization_anndata,
)

__version__ = "3.0.0"

__all__ = [
    "__version__",
    "AutoQSelection",
    "BidirectionalMatch",
    "ConditionalInstability",
    "CoverageMatchResult",
    "DEFAULT_PROTOCOL_V3",
    "DUAResult",
    "ExtrapolationResult",
    "SplitHalfCurve",
    "StabilityWeightedResult",
    "StandardizedComparison",
    "ThinningLadder",
    "bidirectional_retention",
    "build_extrapolation_guardrail",
    "build_split_half_curve",
    "cell_stability_weights",
    "conditional_instability_metrics",
    "conformal_prediction_band",
    "decompose_dua_uncertainty",
    "exact_perm_z_from_draws",
    "exact_weighted_perm_pair",
    "extrapolate_ladder",
    "get_default_protocol_v3",
    "joint_perm_pairs_from_draws",
    "marginal_depth_ratio",
    "match_split_half_coverage",
    "noise_coordinate",
    "select_auto_q",
    "select_mitm_target_depth",
    "split_half_disagreement",
    "stability_weighted_colocalization",
    "standardize_colocalization_anndata",
    "summarize_dua_draws",
    "summarize_thinning_ladder",
    "target_noise_coordinate",
    "thin_histogram_summary",
]
