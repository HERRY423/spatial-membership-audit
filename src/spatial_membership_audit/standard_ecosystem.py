"""Out-of-the-box integration with AnnData, SpatialData, Squidpy, and Giotto.

Connects standard scverse/Giotto spatial objects directly to:
  1. Declarative marker rules (arbitrary gene panels), frozen cell-wise
     classifiers, or pre-supplied categorical probability matrices;
  2. Exact availability P+ = Pr(E > 0) and conditional ratio CV(R | E > 0)
     via direct deterministic enclosure (D), auto-routing (D/Sobol/MC), or
     fast analytical delta moments;
  3. The Biologist's Operational Decision Guide (regime, action_code,
     conditioning_trap warning, and bilingual actionable instructions).

Input objects (AnnData, SpatialData) are never modified in place unless
``inplace=True`` is explicitly requested.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .adapters import validate_counts
from .analytic import pair_moments
from .auto_routing import route_cv
from .custom_rules import (
    COMMON125_11MARKER_SPEC,
    MarkerRuleSpec,
    evaluate_marker_rule_labels,
    evaluate_marker_rule_probabilities,
    frozen_classifier_probabilities,
    resolve_rule_spec,
)
from .ecosystem import (
    SpatialGraph,
    build_graph,
    giotto_edge_table,
    giotto_endpoint_null_moments,
    graph_from_anndata,
    native_squidpy,
)
from .primary_cv import estimate_cv
from .action_guide import (
    ActionThresholds,
    summarize_operational_actions,
    triage_audit_table,
)


def _resolve_anndata_counts(adata, count_layer: str | None):
    """Extract and validate nonnegative integer counts without guessing."""
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('unique cell (obs_names) and gene (var_names) identifiers required')
    if count_layer is not None:
        if count_layer not in adata.layers:
            raise ValueError(f'count_layer {count_layer!r} not found in adata.layers')
        mat = adata.layers[count_layer]
        source = f'adata.layers[{count_layer!r}]'
    elif 'counts' in adata.layers:
        mat = adata.layers['counts']
        source = "adata.layers['counts']"
    elif adata.X is not None:
        mat = adata.X
        source = 'adata.X'
    else:
        raise ValueError('No count matrix found in adata.layers or adata.X')
    validate_counts(mat.data if sparse.issparse(mat) else mat)
    return mat, list(adata.var_names), source


def _resolve_eligibility(adata, counts_mat, eligible_key: str | None,
                         min_total_counts: int | None, min_detected_genes: int | None):
    """Resolve fixed baseline boolean eligibility mask and its provenance."""
    n = adata.n_obs
    if eligible_key is not None:
        if eligible_key not in adata.obs:
            raise ValueError(f'eligible_key {eligible_key!r} not found in adata.obs')
        col = adata.obs[eligible_key]
        if col.isna().any():
            raise ValueError(f'adata.obs[{eligible_key!r}] contains NA values')
        mask = col.to_numpy()
        if mask.dtype != bool:
            raise ValueError(f'adata.obs[{eligible_key!r}] must be boolean')
        return mask, f'adata.obs[{eligible_key!r}]'
    if min_total_counts is not None or min_detected_genes is not None:
        mt = 0 if min_total_counts is None else int(min_total_counts)
        md = 0 if min_detected_genes is None else int(min_detected_genes)
        if sparse.issparse(counts_mat):
            totals = np.asarray(counts_mat.sum(axis=1)).ravel()
            detected = np.asarray((counts_mat > 0).sum(axis=1)).ravel()
        else:
            x = np.asarray(counts_mat)
            totals = x.sum(axis=1)
            detected = (x > 0).sum(axis=1)
        mask = (totals >= mt) & (detected >= md)
        return mask.astype(bool), f'baseline_qc(total>={mt},detected>={md})'
    return np.ones(n, dtype=bool), 'all_cells_eligible_default'


def _resolve_regions(adata, region_key: str | None, default_region: str = 'region_0') -> tuple[np.ndarray, str]:
    if region_key is not None:
        if region_key not in adata.obs:
            raise ValueError(f'region_key {region_key!r} not found in adata.obs')
        if adata.obs[region_key].isna().any():
            raise ValueError(f'adata.obs[{region_key!r}] contains missing region values')
        reg = adata.obs[region_key].astype(str).to_numpy()
        if np.any(reg == ''):
            raise ValueError('empty region identifiers forbidden')
        return reg, f'adata.obs[{region_key!r}]'
    return np.full(adata.n_obs, str(default_region), dtype=object), f'single_region_default({default_region})'


def _resolve_spatial_graph(
    adata,
    regions: np.ndarray,
    *,
    adjacency_key: str = 'spatial_connectivities',
    spatial_key: str = 'spatial',
    graph_method: str | None = None,
    radius_um: float | None = None,
    k: int | None = None,
    knn_symmetry: str = 'union',
    coordinate_unit: str = 'um',
) -> SpatialGraph:
    """Resolve an undirected binary SpatialGraph from adata.obsp or adata.obsm."""
    cell_ids = tuple(str(x) for x in adata.obs_names)
    reg_tuple = tuple(str(x) for x in regions)
    if graph_method is None and adjacency_key in adata.obsp:
        g_raw = sparse.csr_matrix(adata.obsp[adjacency_key], dtype=float).copy()
        g_raw.sum_duplicates()
        g_raw.eliminate_zeros()
        # If Squidpy created binary symmetric connectivities, preserve directly
        if np.any(g_raw.data != 1.0):
            raise ValueError(
                f'adata.obsp[{adjacency_key!r}] contains non-binary edge weights; '
                'supply a binary connectivity matrix or specify graph_method="radius"/"knn"/"delaunay"'
            )
        return SpatialGraph(
            cell_ids=cell_ids,
            regions=reg_tuple,
            adjacency=g_raw,
            directed=False,
            provenance=f'adata.obsp[{adjacency_key!r}]',
            coordinate_unit=coordinate_unit,
        )
    method = graph_method or 'radius'
    if spatial_key not in adata.obsm:
        raise ValueError(
            f'Neither adata.obsp[{adjacency_key!r}] nor adata.obsm[{spatial_key!r}] found; '
            'provide spatial coordinates or a precomputed adjacency matrix'
        )
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f'adata.obsm[{spatial_key!r}] must have at least 2 coordinate columns')
    xy2 = xy[:, :2]
    if method == 'radius' and radius_um is None:
        raise ValueError('radius_um must be explicitly provided when building a radius graph')
    return build_graph(
        xy2,
        cell_ids,
        reg_tuple,
        method=method,
        radius_um=radius_um,
        k=k,
        knn_symmetry=knn_symmetry if method == 'knn' else None,
    )


def _baseline_pair_observation(p_base: np.ndarray, local_graph: sparse.csr_matrix, a: int, b: int) -> dict:
    """Compute baseline (p=1) observed O, E, R, and structural state."""
    n = len(p_base)
    m = local_graph.nnz // 2
    pa = p_base[:, a]
    pb = p_base[:, b]
    na = float(pa.sum())
    nb = float(pb.sum())
    c = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    o_val = float(pa @ (local_graph @ pb))
    # Off-diagonal expectation in case p_base is soft categorical probabilities
    same_node = float(np.sum(pa * pb))
    d_val = max(0.0, na * nb - same_node)
    e_val = c * d_val
    if m == 0:
        state = 'NO_EDGES'
        r_val = None
    elif e_val <= 0.0:
        state = 'STRUCTURAL_NA'
        r_val = None
    elif o_val == 0.0:
        state = 'ZERO_R'
        r_val = 0.0
    else:
        state = 'FINITE_POSITIVE'
        r_val = o_val / e_val
    return dict(
        baseline_n_a=na,
        baseline_n_b=nb,
        baseline_O=o_val,
        baseline_E=e_val,
        baseline_R=r_val,
        baseline_state=state,
    )


def _retention_schedule(p, p_grid, n_genes: int):
    """Validate scalar curves or one gene-aligned vector, without ordering vectors."""
    values = list(p_grid) if p_grid is not None else [p]
    if not values:
        raise ValueError('at least one retention value p is required')
    retentions = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        if p_grid is not None and arr.ndim != 0:
            raise ValueError('p_grid supports scalar retention levels only; '
                             'pass a single gene-specific vector as p')
        if arr.ndim != 0 and (arr.ndim != 1 or arr.shape != (n_genes,)):
            raise ValueError('p must be scalar or a vector aligned with adata.var_names')
        if not np.all(np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
            raise ValueError('retention values must be finite and in [0,1]')
        retentions.append(float(arr) if arr.ndim == 0 else arr.copy())
    return retentions


def audit_anndata(
    adata,
    *,
    p: float | Sequence[float] = 0.5,
    p_grid: Sequence[float] | None = None,
    rule: str | Mapping | MarkerRuleSpec | None = 'common125_candidate_v1',
    classifier_fn: Callable[[np.ndarray, Sequence[str]], Sequence[str]] | None = None,
    probability_key: str | None = None,
    classes: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    count_layer: str | None = None,
    eligible_key: str | None = None,
    min_total_counts: int | None = None,
    min_detected_genes: int | None = None,
    region_key: str | None = None,
    adjacency_key: str = 'spatial_connectivities',
    spatial_key: str = 'spatial',
    graph_method: str | None = None,
    radius_um: float | None = None,
    k: int | None = None,
    knn_symmetry: str = 'union',
    coordinate_unit: str = 'um',
    cv_mode: str = 'direct',
    accuracy_factor: float = 1.25,
    max_seconds: float = 6.0,
    direct_support_hint: int = 1000,
    allow_approximate_fallback: bool = False,
    classifier_draws: int = 512,
    classifier_assume_cellwise: bool = False,
    seed: int = 1729,
    thresholds: ActionThresholds = ActionThresholds(),
    inplace: bool = False,
    uns_key: str = 'spatial_membership_audit',
) -> pd.DataFrame:
    """Run an end-to-end spatial membership and colocalization audit on AnnData.

    Supports:
      - Any declarative ``MarkerRuleSpec`` or preset (``'common125_candidate_v1'``,
        ``'stromal_myeloid_4gene'``, ``'generic_3lineage_6marker'``, or custom dict);
      - Any frozen cell-wise classifier ``classifier_fn(counts, genes) -> labels``;
      - Precomputed per-cell categorical probability matrices in ``adata.obsm[probability_key]``;
      - Scalar or gene-aligned vector ``p``; scalar-only curves ``p_grid``;
      - CV calculation via ``cv_mode='direct'`` (certified numerical enclosure),
        ``cv_mode='auto'`` (guarantee-aware D / Sobol / MC routing), or
        ``cv_mode='delta'`` (fast analytical moments + delta approximation);
      - Automatic operational triage and concrete biologist guidance on every row.

    Classifier MC provenance is retained in each row and in ``df.attrs['provenance']``
    (also ``adata.uns[uns_key]['provenance']`` with ``inplace=True``). For MC inputs,
    ``guarantee`` is at most ``MC_PLUGIN_ONLY``; ``conditional_guarantee`` describes
    the graph engine conditional on the estimated cell probabilities. Neither its
    CV enclosure nor its availability point estimate includes upstream MC error.
    Gene-specific vectors are retained with their gene order and are not treated
    as an ordered scalar retention curve.
    """
    if cv_mode not in ('direct', 'auto', 'delta'):
        raise ValueError("cv_mode must be 'direct', 'auto', or 'delta'")

    regions, region_source = _resolve_regions(adata, region_key)
    graph = _resolve_spatial_graph(
        adata,
        regions,
        adjacency_key=adjacency_key,
        spatial_key=spatial_key,
        graph_method=graph_method,
        radius_um=radius_um,
        k=k,
        knn_symmetry=knn_symmetry,
        coordinate_unit=coordinate_unit,
    )

    retentions = _retention_schedule(p, p_grid, adata.n_vars)

    # Determine probability source mode
    if probability_key is not None:
        if probability_key not in adata.obsm:
            raise ValueError(f'probability_key {probability_key!r} not found in adata.obsm')
        p_mat = np.asarray(adata.obsm[probability_key], dtype=float)
        if p_mat.ndim != 2 or len(p_mat) != adata.n_obs:
            raise ValueError(f'adata.obsm[{probability_key!r}] must be a 2D array aligned with cells')
        if not np.all(np.isfinite(p_mat)) or np.any((p_mat < 0) | (p_mat > 1)) or np.any(p_mat.sum(axis=1) > 1.0 + 1e-10):
            raise ValueError('supplied probabilities must be mutually exclusive and row-sum <= 1')
        if classes is None:
            if isinstance(adata.obsm[probability_key], pd.DataFrame):
                class_names = tuple(str(c) for c in adata.obsm[probability_key].columns)
            else:
                class_names = tuple(f'CLASS_{i}' for i in range(p_mat.shape[1]))
        else:
            class_names = tuple(str(c) for c in classes)
        if len(class_names) != p_mat.shape[1]:
            raise ValueError('classes length must match columns of probability_key matrix')
        prob_mode = f'supplied_obsm[{probability_key!r}]'
        counts_mat = genes = count_source = mask = elig_source = spec = None
    else:
        counts_mat, genes, count_source = _resolve_anndata_counts(adata, count_layer)
        mask, elig_source = _resolve_eligibility(adata, counts_mat, eligible_key, min_total_counts, min_detected_genes)
        if classifier_fn is not None:
            if classes is None or len(classes) < 2:
                raise ValueError('explicit classes (length >= 2) required when using classifier_fn')
            class_names = tuple(str(c) for c in classes)
            prob_mode = 'frozen_cellwise_classifier'
            spec = None
        else:
            if rule is None:
                raise ValueError('Specify rule, classifier_fn, or probability_key')
            spec = resolve_rule_spec(rule)
            class_names = spec.classes
            prob_mode = f'marker_rule:{spec.rule_id}'

    # Resolve candidate pairs to evaluate
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if pairs is None:
        pair_indices = list(itertools.combinations(range(len(class_names)), 2))
    else:
        pair_indices = []
        for ca, cb in pairs:
            if ca not in class_to_idx or cb not in class_to_idx or ca == cb:
                raise ValueError(f'Invalid pair {(ca, cb)!r} for classes {class_names}')
            pair_indices.append((class_to_idx[ca], class_to_idx[cb]))

    # Compute baseline (p=1.0) probabilities for baseline R_0 and baseline_state
    if probability_key is not None:
        p_base_all = np.asarray(adata.obsm[probability_key], dtype=float)
    elif classifier_fn is not None:
        base_res = frozen_classifier_probabilities(
            counts_mat, genes, mask, 1.0,
            classifier_fn=classifier_fn, classes=class_names,
            n_draws=classifier_draws, seed=seed,
            assume_cellwise=classifier_assume_cellwise,
        )
        p_base_all = base_res['probabilities']
    else:
        p_base_all = evaluate_marker_rule_probabilities(counts_mat, genes, mask, 1.0, spec)

    raw_rows = []
    probability_runs = {}
    for p_idx, p_val in enumerate(retentions):
        is_scalar = np.ndim(p_val) == 0
        p_label = float(p_val) if is_scalar else 'vector:' + json.dumps(p_val.tolist(), separators=(',', ':'))
        retention_kind = 'scalar' if is_scalar else 'gene_specific'
        retention_key = f'retention_{p_idx}'
        probability_metadata = {}
        mc_input = False
        if probability_key is not None:
            probs_all = p_base_all
        elif classifier_fn is not None:
            res_p = frozen_classifier_probabilities(
                counts_mat, genes, mask, p_val,
                classifier_fn=classifier_fn, classes=class_names,
                n_draws=classifier_draws, seed=seed + p_idx * 10007,
                assume_cellwise=classifier_assume_cellwise,
            )
            probs_all = res_p['probabilities']
            probability_metadata = {key: res_p[key] for key in (
                'semantics', 'probability_method', 'n_draws', 'seed',
                'independent_nodes', 'independence_basis', 'upstream_probability_uncertainty',
            )}
            mc_input = res_p['probability_method'] == 'monte_carlo'
            probability_metadata.update(
                p_finite_is_estimate=mc_input,
                cv_interval_scope=('CONDITIONAL_ON_MC_PROBABILITIES_EXCLUDES_UPSTREAM_ERROR'
                                   if mc_input else 'CONDITIONAL_ON_SPECIFIED_PROBABILITIES'),
            )
        else:
            probs_all = evaluate_marker_rule_probabilities(counts_mat, genes, mask, p_val, spec)

        probability_runs[retention_key] = dict(
            probability_source=prob_mode, retention_kind=retention_kind,
            values=np.atleast_1d(p_val).astype(float).tolist(),
            genes=[] if is_scalar else list(adata.var_names),
            **probability_metadata,
        )

        for reg_name in dict.fromkeys(regions):
            nodes = np.flatnonzero(regions == reg_name)
            local_g = graph.adjacency[nodes][:, nodes]
            local_p = probs_all[nodes]
            local_base = p_base_all[nodes]

            for a_idx, b_idx in pair_indices:
                base_obs = _baseline_pair_observation(local_base, local_g, a_idx, b_idx)
                pm = pair_moments(local_p, local_g, a_idx, b_idx)

                cv_val = pm['cv_R_delta']
                cv_interval = None
                cv_status = pm['status']
                guarantee = 'DELTA_APPROXIMATION_ONLY'
                engine_used = 'DELTA'

                if np.all(np.asarray(p_val) == 1.0) and probability_key is None:
                    # At p=1.0 under deterministic baseline rules, CV is not a thinned target
                    cv_val = 0.0 if base_obs['baseline_state'] == 'FINITE_POSITIVE' else None
                    cv_interval = [0.0, 0.0] if cv_val == 0.0 else None
                    cv_status = 'SKIPPED_P1' if base_obs['baseline_state'] in ('FINITE_POSITIVE', 'ZERO_R') else pm['status']
                    guarantee = 'BASELINE_OBSERVED'
                    engine_used = 'BASELINE'
                elif cv_mode == 'direct':
                    d_res = estimate_cv(
                        local_p, local_g, a=a_idx, b=b_idx,
                        accuracy_factor=accuracy_factor, max_seconds=max_seconds,
                    )
                    cv_val = d_res.get('cv')
                    cv_interval = d_res.get('cv_interval')
                    cv_status = d_res['status']
                    guarantee = (
                        'CONDITIONAL_NUMERICAL_ENCLOSURE'
                        if cv_status in ('RESOLVED', 'EXACT_ZERO_CV')
                        else 'STRUCTURAL_STATE_ONLY' if cv_status in ('NO_FINITE_RATIO', 'ZERO_NUMERATOR_CV_UNDEFINED')
                        else 'NONE'
                    )
                    engine_used = 'D'
                elif cv_mode == 'auto':
                    r_res = route_cv(
                        local_p, local_g,
                        requirement='screening' if allow_approximate_fallback or direct_support_hint >= 0 else 'enclosure',
                        accuracy_factor=accuracy_factor,
                        a=a_idx, b=b_idx,
                        max_seconds=max_seconds,
                        direct_support_hint=direct_support_hint,
                        allow_approximate_fallback=allow_approximate_fallback,
                        seed=seed,
                    )
                    cv_val = r_res.get('cv')
                    cv_interval = r_res.get('cv_interval')
                    cv_status = r_res['status']
                    guarantee = r_res['guarantee']
                    engine_used = r_res['engine']

                conditional_cv_status = cv_status
                conditional_guarantee = guarantee
                probability_support_status = 'NOT_ASSESSED'
                if mc_input:
                    probability_support_status = 'MC_POINT_ESTIMATES_ONLY'
                    if pm['m'] > 0:
                        guarantee = ('MC_PLUGIN_ONLY' if conditional_guarantee != 'NONE'
                                     and cv_val is not None and np.isfinite(cv_val) else 'NONE')
                        # A finite simulation cannot certify that a class or a
                        # possible contact has zero probability under thinning.
                        if (pm['p_finite'] == 0.0 or pm['mu_O_finite'] == 0.0 or
                                cv_status in ('NO_FINITE_RATIO', 'NO_FINITE_DENOMINATOR',
                                              'ZERO_NUMERATOR_CV_UNDEFINED', 'ABSENT_TYPE')):
                            probability_support_status = 'UNRESOLVED_MC_SUPPORT'
                            cv_status = 'UNRESOLVED_UPSTREAM_SAMPLING'
                            guarantee = 'NONE'
                            cv_val = None
                            cv_interval = None

                raw_rows.append(dict(
                    region=str(reg_name),
                    label_a=class_names[a_idx],
                    label_b=class_names[b_idx],
                    p=p_label,
                    retention_kind=retention_kind,
                    retention_key=retention_key,
                    n=pm['n'],
                    m=pm['m'],
                    baseline_state=base_obs['baseline_state'],
                    baseline_n_a=base_obs['baseline_n_a'],
                    baseline_n_b=base_obs['baseline_n_b'],
                    baseline_O=base_obs['baseline_O'],
                    baseline_E=base_obs['baseline_E'],
                    baseline_R=base_obs['baseline_R'],
                    expected_n_a=pm['expected_n_a'],
                    expected_n_b=pm['expected_n_b'],
                    p_finite=pm['p_finite'],
                    p_na=pm['p_na'],
                    log_p_finite=pm['log_p_finite'],
                    log_p_na=pm['log_p_na'],
                    mu_O_finite=pm['mu_O_finite'],
                    mu_E_finite=pm['mu_E_finite'],
                    ratio_of_means=pm['ratio_of_means'],
                    mean_R_delta2=pm['mean_R_delta2'],
                    cv_R_delta=pm['cv_R_delta'],
                    cv=cv_val,
                    cv_lower=cv_interval[0] if cv_interval is not None else None,
                    cv_upper=cv_interval[1] if cv_interval is not None else None,
                    cv_status=cv_status,
                    conditional_cv_status=conditional_cv_status,
                    engine=engine_used,
                    guarantee=guarantee,
                    conditional_guarantee=conditional_guarantee,
                    probability_support_status=probability_support_status,
                    probability_source=prob_mode,
                    **probability_metadata,
                ))

    triaged_rows = triage_audit_table(raw_rows, thresholds=thresholds, group_keys=('region', 'label_a', 'label_b'), p_key='p')
    df = pd.DataFrame(triaged_rows)
    # All-missing numeric columns must stay numeric (NaN), including a single
    # unresolved MC row, so the attached audit can round-trip through .h5ad.
    for column in ('baseline_R', 'mu_O_finite', 'mu_E_finite', 'ratio_of_means',
                   'mean_R_delta2', 'cv_R_delta', 'cv', 'cv_lower', 'cv_upper',
                   'log_p_finite', 'log_p_na', 'max_adjacent_p_plus_increase'):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors='raise').astype(float)

    provenance = dict(
        probability_source=prob_mode,
        count_source=count_source if probability_key is None else None,
        eligibility_source=elig_source if probability_key is None else None,
        region_source=region_source,
        graph_provenance=graph.provenance,
        cv_mode=cv_mode,
        accuracy_factor=accuracy_factor,
        probability_runs=probability_runs,
    )
    df.attrs['provenance'] = provenance
    if inplace:
        summary = summarize_operational_actions(df, p_key='p')
        adata.uns[uns_key] = dict(
            table=df,
            summary=summary,
            provenance=provenance,
        )
    return df


def extract_anndata_from_spatialdata(
    sdata,
    *,
    table_key: str = 'table',
    shapes_key: str | None = None,
    spatial_key: str = 'spatial',
    coordinate_system: str | None = None,
):
    """Extract a non-mutating AnnData view/copy with aligned 2D spatial coordinates
    from a SpatialData object.

    A requested coordinate system must be applied successfully, even when the
    table already has coordinates or a graph. Transformation failures raise
    ValueError with the original exception as their cause.
    """
    if hasattr(sdata, 'tables') and table_key in sdata.tables:
        adata = sdata.tables[table_key].copy()
    elif hasattr(sdata, '__getitem__'):
        adata = sdata[table_key].copy()
    else:
        raise ValueError(f'table_key {table_key!r} not found in SpatialData object')

    if spatial_key not in adata.obsm or shapes_key is not None or coordinate_system is not None:
        target_shapes_key = shapes_key
        if target_shapes_key is None and hasattr(sdata, 'shapes') and len(sdata.shapes) == 1:
            target_shapes_key = next(iter(sdata.shapes.keys()))
        if target_shapes_key is not None and hasattr(sdata, 'shapes') and target_shapes_key in sdata.shapes:
            gdf = sdata.shapes[target_shapes_key]
            if coordinate_system is not None:
                try:
                    import spatialdata as sd
                    gdf = sd.transform(gdf, to_coordinate_system=coordinate_system)
                except Exception as exc:
                    raise ValueError(
                        f'Cannot transform SpatialData shapes {target_shapes_key!r} '
                        f'to requested coordinate system {coordinate_system!r}; '
                        'no untransformed fallback was used'
                    ) from exc
            # Align shape centroids with adata.obs_names (or instance_id column if present)
            attrs = adata.uns.get('spatialdata_attrs', {})
            instance_key = attrs.get('instance_key', 'instance_id')
            region_key = attrs.get('region_key')
            if region_key in adata.obs and not (
                adata.obs[region_key].astype(str) == target_shapes_key
            ).all():
                raise ValueError('Table contains observations bound to other SpatialData elements')
            if instance_key in adata.obs:
                if adata.obs[instance_key].isna().any():
                    raise ValueError('SpatialData instance identifiers contain missing values')
                lookup_ids = adata.obs[instance_key].astype(str).tolist()
            else:
                lookup_ids = [str(x) for x in adata.obs_names]
            gdf_indexed = gdf.copy()
            gdf_indexed.index = gdf_indexed.index.astype(str)
            if not gdf_indexed.index.is_unique or len(set(lookup_ids)) != len(lookup_ids):
                raise ValueError('SpatialData shapes and table instance identifiers must be unique')
            if not set(lookup_ids).issubset(set(gdf_indexed.index)):
                raise ValueError('SpatialData shapes index does not cover all AnnData table observations')
            aligned_geom = gdf_indexed.loc[lookup_ids].geometry
            centroids = np.column_stack([aligned_geom.centroid.x.to_numpy(dtype=float),
                                         aligned_geom.centroid.y.to_numpy(dtype=float)])
            if centroids.shape != (adata.n_obs, 2) or not np.isfinite(centroids).all():
                raise ValueError('SpatialData shape centroids must be finite aligned 2D coordinates')
            adata.obsm[spatial_key] = centroids
            adata.uns['_sma_coordinate_provenance'] = dict(
                shapes_key=target_shapes_key,
                coordinate_system=coordinate_system or 'stored_element_coordinates',
                transform_applied=coordinate_system is not None,
                physical_unit='not_inferred_from_coordinate_system_name',
            )
        elif shapes_key is not None or coordinate_system is not None:
            raise ValueError(
                'Requested shapes/coordinate system cannot be resolved; provide a valid '
                'shapes_key with registered SpatialData transformations'
            )

    if spatial_key not in adata.obsm and 'spatial_connectivities' not in adata.obsp:
        raise ValueError(
            f'SpatialData table {table_key!r} has neither obsm[{spatial_key!r}] nor '
            'obsp["spatial_connectivities"], and no shapes centroids could be resolved'
        )
    return adata


def audit_spatialdata(
    sdata,
    *,
    table_key: str = 'table',
    shapes_key: str | None = None,
    spatial_key: str = 'spatial',
    coordinate_system: str | None = None,
    inplace: bool = False,
    uns_key: str = 'spatial_membership_audit',
    **kwargs,
) -> pd.DataFrame:
    """Run an end-to-end spatial membership and colocalization audit on a SpatialData object.

    Extracts ``sdata.tables[table_key]`` (and resolves centroids from ``sdata.shapes``
    if needed), runs ``audit_anndata``, and attaches operational triage guidance.
    """
    adata = extract_anndata_from_spatialdata(
        sdata,
        table_key=table_key,
        shapes_key=shapes_key,
        spatial_key=spatial_key,
        coordinate_system=coordinate_system,
    )
    df = audit_anndata(adata, spatial_key=spatial_key, inplace=inplace, uns_key=uns_key, **kwargs)
    if inplace and hasattr(sdata, 'tables') and table_key in sdata.tables:
        sdata.tables[table_key].uns[uns_key] = adata.uns[uns_key]
    return df


def audit_squidpy_nhood(
    adata,
    *,
    p: float | Sequence[float] = 0.5,
    rule: str | Mapping | MarkerRuleSpec | None = 'common125_candidate_v1',
    classifier_fn: Callable[[np.ndarray, Sequence[str]], Sequence[str]] | None = None,
    cluster_key: str | None = None,
    n_perms: int = 100,
    seed: int = 1729,
    z_threshold: float = 1.96,
    count_layer: str | None = None,
    eligible_key: str | None = None,
    min_total_counts: int | None = None,
    min_detected_genes: int | None = None,
    region_key: str | None = None,
    adjacency_key: str = 'spatial_connectivities',
    spatial_key: str = 'spatial',
    graph_method: str | None = None,
    radius_um: float | None = None,
    k: int | None = None,
    knn_symmetry: str = 'union',
    coordinate_unit: str = 'um',
    cv_mode: str = 'direct',
    accuracy_factor: float = 1.25,
    max_seconds: float = 6.0,
    thresholds: ActionThresholds = ActionThresholds(),
    inplace: bool = False,
    uns_key: str = 'spatial_membership_audit_squidpy',
) -> pd.DataFrame:
    """Run native Squidpy neighborhood enrichment side-by-side with P+ and CV audit.

    Joins Squidpy's empirical permutation z-scores and exact node-permutation
    null moments with analytical availability P+ and conditional CV(R | E > 0)
    under molecule thinning, flagging high-|z| comparisons that are fragile or
    fall into the conditioning trap.
    """
    regions, _ = _resolve_regions(adata, region_key)
    graph = _resolve_spatial_graph(
        adata,
        regions,
        adjacency_key=adjacency_key,
        spatial_key=spatial_key,
        graph_method=graph_method,
        radius_um=radius_um,
        k=k,
        knn_symmetry=knn_symmetry,
        coordinate_unit=coordinate_unit,
    )
    counts_mat, genes, _ = _resolve_anndata_counts(adata, count_layer)
    mask, _ = _resolve_eligibility(adata, counts_mat, eligible_key, min_total_counts, min_detected_genes)

    if cluster_key is not None and cluster_key in adata.obs:
        base_labels = adata.obs[cluster_key].astype(object).fillna('UNKNOWN').astype(str).to_numpy()
    elif classifier_fn is not None:
        base_labels = np.asarray(classifier_fn(
            counts_mat.toarray() if sparse.issparse(counts_mat) else np.asarray(counts_mat),
            genes,
        ), dtype=object)
        base_labels[~mask] = 'LOW_DETECTION_BASELINE'
        base_labels = base_labels.astype(str)
    else:
        spec = resolve_rule_spec(rule)
        base_labels = evaluate_marker_rule_labels(counts_mat, genes, mask, spec).astype(str)

    audit_df = audit_anndata(
        adata,
        p=p,
        rule=rule,
        classifier_fn=classifier_fn,
        count_layer=count_layer,
        eligible_key=eligible_key,
        min_total_counts=min_total_counts,
        min_detected_genes=min_detected_genes,
        region_key=region_key,
        adjacency_key=adjacency_key,
        spatial_key=spatial_key,
        graph_method=graph_method,
        radius_um=radius_um,
        k=k,
        knn_symmetry=knn_symmetry,
        coordinate_unit=coordinate_unit,
        cv_mode=cv_mode,
        accuracy_factor=accuracy_factor,
        max_seconds=max_seconds,
        seed=seed,
        thresholds=thresholds,
        inplace=False,
    )

    target_classes = sorted(set(audit_df['label_a']).union(set(audit_df['label_b'])))
    all_cats = sorted(set(target_classes).union(set(base_labels)))
    sq_out = native_squidpy(graph, base_labels, n_perms=n_perms, seed=seed, categories=all_cats)
    sq_lookup = {
        (str(r['region']), str(r['label_a']), str(r['label_b'])): r
        for r in sq_out['rows']
    }

    merged_rows = []
    for row in audit_df.to_dict(orient='records'):
        key = (str(row['region']), str(row['label_a']), str(row['label_b']))
        sq_row = sq_lookup.get(key, {})
        null_info = sq_row.get('exact_node_null') or {}
        z_val = sq_row.get('zscore')
        sq_status = sq_row.get('status', 'NOT_FOUND')
        null_status = null_info.get('status')

        p_plus = row.get('p_finite')
        z_finite = z_val is not None and np.isfinite(z_val)
        high_z = bool(z_finite and abs(float(z_val)) >= z_threshold)

        if null_status == 'ZERO_NULL_VARIANCE':
            sq_flag = 'SQUIDPY_ZERO_PERMUTATION_VARIANCE'
        elif high_z and p_plus is not None and p_plus < thresholds.p_fragile:
            sq_flag = 'HIGH_SQUIDPY_Z_BUT_SEVERE_AVAILABILITY_LOSS'
        elif high_z and row.get('conditioning_trap'):
            sq_flag = 'HIGH_SQUIDPY_Z_WITH_CONDITIONING_TRAP'
        elif high_z and row.get('regime') == 'TIER_1_ROBUST':
            sq_flag = 'HIGH_SQUIDPY_Z_ROBUST_UNDER_THINNING'
        else:
            sq_flag = f"SQUIDPY_{sq_status}_AUDIT_{row.get('regime')}"

        merged_rows.append(dict(
            **row,
            squidpy_count=sq_row.get('count'),
            squidpy_zscore=z_val,
            squidpy_status=sq_status,
            exact_null_mean=null_info.get('mean'),
            exact_null_sd=null_info.get('sd'),
            exact_null_status=null_status,
            squidpy_audit_flag=sq_flag,
        ))

    out_df = pd.DataFrame(merged_rows)
    if inplace:
        adata.uns[uns_key] = dict(
            table=out_df,
            squidpy_version=sq_out['version'],
            squidpy_source_sha256=sq_out['source_sha256'],
            summary=summarize_operational_actions(out_df, p_key='p'),
        )
    return out_df


def prepare_giotto_audit_bundle(
    adata,
    *,
    p: float = 0.5,
    rule: str | Mapping | MarkerRuleSpec = 'common125_candidate_v1',
    count_layer: str | None = None,
    eligible_key: str | None = None,
    min_total_counts: int | None = None,
    min_detected_genes: int | None = None,
    region_key: str | None = None,
    adjacency_key: str = 'spatial_connectivities',
    spatial_key: str = 'spatial',
    graph_method: str | None = None,
    radius_um: float | None = None,
    cv_mode: str = 'direct',
    accuracy_factor: float = 1.25,
    max_seconds: float = 6.0,
    thresholds: ActionThresholds = ActionThresholds(),
    out_dir: str | Path | None = None,
) -> dict:
    """Prepare a complete Giotto-compatible audit bundle from AnnData.

    Produces:
      1. ``audit_table``: DataFrame with P+, conditional CV, Giotto exact
         edge-endpoint permutation null moments (``giotto_stub_null_mean``,
         ``giotto_stub_null_sd``, ``giotto_stub_null_status``), and operational
         triage;
      2. ``edge_table``: DataFrame formatted for ``Giotto::createSpatNetObj``;
      3. ``cell_metadata``: DataFrame with ``cell_ID``, ``region``, and baseline
         ``sma_label`` for ``Giotto::addCellMetadata``.
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f'adata.obsm[{spatial_key!r}] micrometre coordinates required for Giotto bundle')
    xy_um = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    regions, _ = _resolve_regions(adata, region_key)
    graph = _resolve_spatial_graph(
        adata,
        regions,
        adjacency_key=adjacency_key,
        spatial_key=spatial_key,
        graph_method=graph_method,
        radius_um=radius_um,
        coordinate_unit='um',
    )
    counts_mat, genes, _ = _resolve_anndata_counts(adata, count_layer)
    mask, _ = _resolve_eligibility(adata, counts_mat, eligible_key, min_total_counts, min_detected_genes)
    spec = resolve_rule_spec(rule)
    base_labels = evaluate_marker_rule_labels(counts_mat, genes, mask, spec).astype(str)

    audit_df = audit_anndata(
        adata,
        p=p,
        rule=spec,
        count_layer=count_layer,
        eligible_key=eligible_key,
        min_total_counts=min_total_counts,
        min_detected_genes=min_detected_genes,
        region_key=region_key,
        adjacency_key=adjacency_key,
        spatial_key=spatial_key,
        graph_method=graph_method,
        radius_um=radius_um,
        cv_mode=cv_mode,
        accuracy_factor=accuracy_factor,
        max_seconds=max_seconds,
        thresholds=thresholds,
        inplace=False,
    )

    # Attach exact Giotto 4.2.3 edge-endpoint stub permutation null moments per region & pair
    giotto_rows = []
    for row in audit_df.to_dict(orient='records'):
        reg_name = str(row['region'])
        nodes = np.flatnonzero(regions == reg_name)
        local_g = graph.adjacency[nodes][:, nodes]
        local_labs = base_labels[nodes]
        stub = giotto_endpoint_null_moments(local_g, local_labs, (row['label_a'], row['label_b']))
        giotto_rows.append(dict(
            **row,
            giotto_stub_null_mean=stub['mean'],
            giotto_stub_null_variance=stub['variance'],
            giotto_stub_null_sd=float(np.sqrt(stub['variance'])),
            giotto_stub_a=stub['stub_a'],
            giotto_stub_b=stub['stub_b'],
            giotto_stub_null_status=stub['status'],
        ))

    audit_giotto_df = pd.DataFrame(giotto_rows)
    edges_df = giotto_edge_table(graph, xy_um)
    meta_df = pd.DataFrame({
        'cell_ID': list(graph.cell_ids),
        'region': list(graph.regions),
        'sma_label': list(base_labels),
        'sdimx': xy_um[:, 0],
        'sdimy': xy_um[:, 1],
    })

    if out_dir is not None:
        folder = Path(out_dir)
        if folder.exists():
            raise FileExistsError(f'out_dir {folder} already exists; refusing to overwrite')
        folder.mkdir(parents=True, exist_ok=False)
        audit_giotto_df.to_csv(folder / 'giotto_audit_triage.csv', index=False)
        edges_df.to_csv(folder / 'giotto_spatial_network.csv', index=False)
        meta_df.to_csv(folder / 'giotto_cell_metadata.csv', index=False)

    return dict(
        audit_table=audit_giotto_df,
        edge_table=edges_df,
        cell_metadata=meta_df,
        summary=summarize_operational_actions(audit_giotto_df, p_key='p'),
    )
