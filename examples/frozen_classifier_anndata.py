"""Runnable frozen-classifier AnnData tutorial, using only synthetic inputs.

Run from any working directory:
    python frozen_classifier_anndata.py --out-dir <new-output-directory>

No training, external data, model downloads, or installation are performed.
See tutorials/frozen_classifier_anndata.md for the walkthrough and boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from spatial_membership_audit.custom_rules import (  # noqa: E402
    FrozenLinearCentroidClassifier, frozen_classifier_probabilities,
)
from spatial_membership_audit.ecosystem import build_graph  # noqa: E402
from spatial_membership_audit.standard_ecosystem import audit_anndata  # noqa: E402

GENES = ['GENE_A', 'GENE_B']
CLASSES = ['TYPE_A', 'TYPE_B']
SEED = 1729
DRAWS = 4096
P_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]
SCALAR_KEY = 'frozen_classifier_scalar'
VECTOR_KEY = 'frozen_classifier_vector'
CALLBACK_KEY = 'frozen_classifier_callback'


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NA' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def build_input():
    # Illustrative counts/coordinates, not an observed tissue or cell identity.
    counts = np.array([[4, 0], [0, 4], [2, 1], [1, 2], [1, 1], [0, 0]], dtype=np.int64)
    xy = np.array([[0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10]], dtype=float)
    obj = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({'roi': ['synthetic_roi'] * len(counts),
                          'qc_pass': counts.sum(axis=1) >= 1},
                         index=[f'cell_{i}' for i in range(len(counts))]),
        var=pd.DataFrame(index=GENES),
    )
    obj.layers['counts'] = obj.X.copy()
    obj.obsm['spatial'] = xy
    graph = build_graph(xy, tuple(obj.obs_names), tuple(obj.obs['roi']),
                        method='radius', radius_um=12.0)
    obj.obsp['spatial_connectivities'] = graph.adjacency.copy()
    obj.uns['tutorial_input'] = dict(
        source='SYNTHETIC_ENGINEERING_EXAMPLE', coordinate_unit='um',
        radius_um=12.0, qc='fixed baseline total counts >= 1; keep all nodes',
        graph='fixed symmetric binary radius graph',
    )
    return obj, counts


def make_frozen_model():
    model = FrozenLinearCentroidClassifier(
        genes=GENES, classes=CLASSES,
        weights=[[1.0, -1.0], [-1.0, 1.0]], intercept=[0.0, 0.0],
        scale_factor=1e4, min_margin=0.2, min_positive_markers=1,
    )
    # The dataclass freezes attributes; protect its arrays as well in this example.
    model.weights.setflags(write=False)
    model.intercept.setflags(write=False)
    return model


def model_config(model):
    return dict(family='fixed_linear_scores_on_rowwise_log1p_normalized_counts',
                source='illustrative fixed coefficients; not fitted to this tutorial',
                genes=model.genes, classes=model.classes, weights=model.weights,
                intercept=model.intercept, scale_factor=model.scale_factor,
                min_margin=model.min_margin, min_positive_markers=model.min_positive_markers,
                other_label=model.other_label)


def fingerprint(value):
    data = json.dumps(json_value(value), sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def enumerate_cell_probabilities(counts, eligible, classifier, p):
    """Small-input reference: sum binomial count-state weights, no MC."""
    result = np.zeros((len(counts), len(CLASSES)))
    for i, row in enumerate(counts):
        if not eligible[i]:
            continue
        for state in itertools.product(*(range(int(n) + 1) for n in row)):
            weight = math.prod(math.comb(int(n), int(k)) * p ** k * (1 - p) ** (n - k)
                               for n, k in zip(row, state))
            label = classifier(np.array([state]), GENES)[0]
            if label in CLASSES:
                result[i, CLASSES.index(label)] += weight
    return result


def enumerate_graph(probabilities, graph):
    """Independent small-input label-state sum for the fixed graph O/E ratio."""
    n = len(probabilities)
    edges = list(zip(*sparse.triu(graph, k=1).nonzero()))
    all_probs = np.column_stack([probabilities, 1 - probabilities.sum(axis=1)])
    accum = np.zeros(3)
    for labels in itertools.product(range(3), repeat=n):
        weight = math.prod(all_probs[i, label] for i, label in enumerate(labels))
        na, nb = labels.count(0), labels.count(1)
        if weight == 0 or na == 0 or nb == 0:
            continue
        observed = sum({labels[i], labels[j]} == {0, 1} for i, j in edges)
        expected = 2 * len(edges) * na * nb / (n * (n - 1))
        ratio = observed / expected
        accum += weight * np.array([1, ratio, ratio * ratio])
    mean = accum[1] / accum[0]
    cv = math.sqrt(max(0, accum[2] / accum[0] - mean * mean)) / mean
    return dict(p_plus=float(accum[0]), conditional_mean_r=float(mean), conditional_cv=float(cv))


def run_tutorial(out):
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        write_target = out / 'checks_progress.json'
        write_target.write_text(json.dumps(checks, indent=2), encoding='utf-8')
        if not condition:
            raise AssertionError(name)

    adata, counts = build_input()
    eligible = adata.obs['qc_pass'].to_numpy()
    graph_before = adata.obsp['spatial_connectivities'].copy()
    model = make_frozen_model()
    config = model_config(model)
    model_sha = fingerprint(config)
    write_json(out / 'frozen_model.json', dict(parameters=config, sha256=model_sha))
    adata.uns['frozen_model_sha256'] = model_sha
    adata.obs['baseline_candidate'] = model(counts, GENES)
    adata.obs.loc[~eligible, 'baseline_candidate'] = 'LOW_DETECTION_BASELINE'
    adata.write_h5ad(out / 'input.h5ad')

    common = dict(classifier_fn=model, classes=CLASSES, count_layer='counts',
                  eligible_key='qc_pass', region_key='roi', coordinate_unit='um',
                  cv_mode='direct', accuracy_factor=1.05, max_seconds=3.0,
                  classifier_draws=DRAWS, seed=SEED, inplace=True)
    scalar = audit_anndata(adata, p_grid=P_GRID, uns_key=SCALAR_KEY, **common)
    scalar.to_csv(out / 'scalar_audit.csv', index=False, na_rep='NA')
    thinned = scalar[(scalar['p'] > 0) & (scalar['p'] < 1)]
    check('scalar_curve_has_five_rows', len(scalar) == len(P_GRID))
    check('scalar_mc_rows_resolved_conditionally', (thinned['cv_status'] == 'RESOLVED').all())
    check('mc_output_not_end_to_end_certificate', (thinned['guarantee'] == 'MC_PLUGIN_ONLY').all())
    check('conditional_engine_certificate_retained',
          (thinned['conditional_guarantee'] == 'CONDITIONAL_NUMERICAL_ENCLOSURE').all())
    check('actual_draws_and_seeds_retained', scalar['n_draws'].tolist() == [0, DRAWS, DRAWS, DRAWS, 0]
          and scalar['seed'].tolist() == [SEED + i * 10007 for i in range(len(P_GRID))])

    vector = audit_anndata(adata, p=np.array([0.4, 0.8]), uns_key=VECTOR_KEY, **common)
    vector.to_csv(out / 'vector_audit.csv', index=False, na_rep='NA')
    vector_meta = adata.uns[VECTOR_KEY]['provenance']['probability_runs']['retention_0']
    check('vector_values_and_genes_preserved', vector_meta['values'] == [0.4, 0.8]
          and vector_meta['genes'] == GENES)
    check('vector_not_treated_as_scalar_curve', vector.iloc[0]['retention_curve_status'] ==
          'NOT_EVALUATED_NONSCALAR_RETENTION' and not vector.iloc[0]['is_non_monotonic_trajectory'])
    check('vector_classifier_chain_resolved', vector.iloc[0]['cv_status'] == 'RESOLVED')

    # The public protocol also accepts a user-defined callable. Its entire
    # preprocessing/prediction path must be frozen and rowwise, as this one is.
    def frozen_callback(x, genes):
        return model(x, genes)

    callback_options = dict(common, classifier_fn=frozen_callback,
                            classifier_assume_cellwise=True)
    callback = audit_anndata(adata, p=0.5, uns_key=CALLBACK_KEY, **callback_options)
    callback.to_csv(out / 'callback_audit.csv', index=False, na_rep='NA')
    check('custom_callable_path_resolved', callback.iloc[0]['cv_status'] == 'RESOLVED')
    check('custom_callable_assumption_recorded', callback.iloc[0]['independence_basis'] ==
          'caller_acknowledged_assumption_not_proved')

    guards = {}
    try:
        frozen_classifier_probabilities(counts, GENES, eligible, 0.5,
            classifier_fn=frozen_callback, classes=CLASSES, n_draws=16)
    except ValueError as error:
        guards['custom_callable_without_ack'] = str(error)
    check('unacknowledged_callable_rejected', 'assume_cellwise' in guards.get('custom_callable_without_ack', ''))

    def cohort_dependent_classifier(x, genes):
        return np.where(x[:, 0] > x[:, 0].mean(), CLASSES[0], CLASSES[1])

    try:
        frozen_classifier_probabilities(counts, GENES, eligible, 0.5,
            classifier_fn=cohort_dependent_classifier, classes=CLASSES,
            assume_cellwise=True, n_draws=16)
    except ValueError as error:
        guards['cohort_dependent_classifier'] = str(error)
    check('cohort_dependent_example_rejected', 'independence' in guards.get('cohort_dependent_classifier', ''))
    write_json(out / 'guard_examples.json', guards)

    # The selected seed reproduces exactly the probabilities fed to the p=.5
    # scalar row. The reference below enumerates small states, not simulations.
    selected_seed = SEED + P_GRID.index(0.5) * 10007
    mc = frozen_classifier_probabilities(counts, GENES, eligible, 0.5,
        classifier_fn=model, classes=CLASSES, n_draws=DRAWS, seed=selected_seed)
    exact_probs = enumerate_cell_probabilities(counts, eligible, model, 0.5)
    tolerance = 6 * np.sqrt(exact_probs * (1 - exact_probs) / DRAWS) + 0.002
    check('cell_probability_reference_within_engineering_tolerance',
          np.all(np.abs(mc['probabilities'] - exact_probs) <= tolerance))
    per_cell = pd.DataFrame(mc['probabilities'], index=adata.obs_names, columns=CLASSES)
    per_cell['OTHER_OR_INELIGIBLE'] = 1 - per_cell[CLASSES].sum(axis=1)
    per_cell.to_csv(out / 'per_cell_mc_probabilities.csv', index_label='cell_id')
    true_reference = enumerate_graph(exact_probs, graph_before)
    plugin_reference = enumerate_graph(mc['probabilities'], graph_before)
    selected_row = scalar.loc[scalar['p'] == 0.5].iloc[0]
    check('graph_availability_matches_same_probability_reference',
          abs(float(selected_row['p_finite']) - plugin_reference['p_plus']) < 1e-12)
    check('graph_cv_encloses_same_probability_reference',
          selected_row['cv_lower'] <= plugin_reference['conditional_cv'] <= selected_row['cv_upper'])
    reference = dict(exact_cell_probabilities=exact_probs, mc_cell_probabilities=mc['probabilities'],
                     p=0.5, n_draws=DRAWS, seed=selected_seed,
                     max_absolute_probability_error=float(np.max(abs(mc['probabilities'] - exact_probs))),
                     engineering_tolerance='6 binomial SE + 0.002; not a confidence coverage assertion',
                     exact_thinning_reference=true_reference, mc_plugin_reference=plugin_reference,
                     reported_cv_interval=[selected_row['cv_lower'], selected_row['cv_upper']],
                     interval_scope='Only the MC-plugin reference is required to lie in this numerical interval')
    write_json(out / 'small_state_reference.json', reference)

    # A deliberately rare event shows why finite MC must not certify zeros.
    rare = ad.AnnData(sparse.csr_matrix([[20, 0], [0, 20]]),
        obs=pd.DataFrame(index=['rare_a', 'rare_b']), var=pd.DataFrame(index=GENES))
    rare.obsp['spatial_connectivities'] = sparse.csr_matrix([[0, 1], [1, 0]])

    def rare_classifier(x, genes):
        return np.where(x[:, 0] == 20, CLASSES[0], np.where(x[:, 1] > 0, CLASSES[1], 'OTHER'))

    rare_result = audit_anndata(rare, p=0.5, classifier_fn=rare_classifier,
        classifier_assume_cellwise=True, classes=CLASSES, classifier_draws=512, seed=SEED,
        cv_mode='direct', max_seconds=3.0, inplace=True)
    rare_result.to_csv(out / 'rare_event_audit.csv', index=False, na_rep='NA')
    check('rare_positive_event_stays_sampling_unresolved',
          rare_result.iloc[0]['p_finite'] == 0 and
          rare_result.iloc[0]['cv_status'] == 'UNRESOLVED_UPSTREAM_SAMPLING' and
          rare_result.iloc[0]['regime'] == 'NUMERICAL_UNRESOLVED')
    write_json(out / 'rare_event_reference.json', dict(
        exact_p_plus=2 ** -20 * (1 - 2 ** -20), n_draws=512, seed=SEED,
        mc_point_estimate=float(rare_result.iloc[0]['p_finite']),
        reported_status=rare_result.iloc[0]['cv_status'],
        derivation='P(A at cell 0)=2^-20; P(B at cell 1)=1-2^-20; independent cells'))

    adata.write_h5ad(out / 'audited.h5ad')
    rare.write_h5ad(out / 'rare_event.h5ad')
    reloaded = ad.read_h5ad(out / 'audited.h5ad')
    reload_rare = ad.read_h5ad(out / 'rare_event.h5ad')
    persisted = reloaded.uns[SCALAR_KEY]['provenance']['probability_runs']['retention_2']
    check('h5ad_mc_provenance_roundtrip', int(persisted['n_draws']) == DRAWS
          and int(persisted['seed']) == selected_seed and
          'not included' in persisted['upstream_probability_uncertainty'])
    saved_vector = reloaded.uns[VECTOR_KEY]['provenance']['probability_runs']['retention_0']
    check('h5ad_vector_roundtrip', np.array_equal(saved_vector['values'], [0.4, 0.8])
          and list(saved_vector['genes']) == GENES)
    check('h5ad_unresolved_state_roundtrip', reload_rare.uns['spatial_membership_audit']['table'].iloc[0]['cv_status']
          == 'UNRESOLVED_UPSTREAM_SAMPLING')
    check('raw_counts_and_fixed_graph_preserved', np.array_equal(reloaded.layers['counts'].toarray(), counts)
          and (reloaded.obsp['spatial_connectivities'] != graph_before).nnz == 0)
    check('frozen_model_unchanged', fingerprint(model_config(model)) == model_sha)
    summary = dict(status='PASS', fixture='SYNTHETIC_ENGINEERING_EXAMPLE',
        n_cells=adata.n_obs, n_genes=adata.n_vars, n_eligible=int(eligible.sum()),
        n_edges=graph_before.nnz // 2, model_sha256=model_sha, checks=checks,
        scalar_rows=len(scalar), vector_rows=len(vector), callback_rows=len(callback), rare_rows=len(rare_result),
        exact_reference=reference, versions={name: version(name) for name in ['anndata', 'numpy', 'scipy', 'pandas', 'h5py']},
        interpretation='Software workflow only; no biological accuracy, physical calibration or end-to-end MC coverage claim')
    write_json(out / 'summary.json', summary)
    write_json(out / 'artifact_manifest.json', {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(scalar[['p', 'p_finite', 'cv', 'cv_status', 'n_draws', 'guarantee']].to_string(index=False))
    print(json.dumps(dict(status='PASS', checks=len(checks), output_directory=str(out)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path, help='New directory; existing outputs are never overwritten')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        run_tutorial(out)
    except Exception as error:
        write_json(out / 'failure.json', dict(error_type=type(error).__name__, message=str(error),
                                             traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
