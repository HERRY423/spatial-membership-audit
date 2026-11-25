"""Experimental, guarantee-aware D / scrambled-Sobol / MC routing.

Routing thresholds are configurable engineering heuristics, not validated
performance boundaries. Sampling never returns a deterministic enclosure or a
calibrated confidence interval. Original N/M and positive-support isolates are
retained. See AUTO_ROUTING.md for scope and resource limitations.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import qmc

from .analytic_prepared import prepare_pair_graph
from .primary_cv import estimate_cv, json_safe

POLICY_VERSION = 'experimental_v1'


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _sampling_engine(requested, dimensions):
    if requested == 'mc':
        return 'MC', 'MC_REQUESTED'
    if dimensions > qmc.Sobol.MAXDIM:
        return 'MC', 'SOBOL_DIMENSION_LIMIT'
    return 'SOBOL', 'SCRAMBLED_SOBOL_WITHIN_DIMENSION_LIMIT'


def _plan(draws, replicates, seed, status):
    return [dict(replicate=i, seed=int(child.generate_state(1)[0]),
                 requested_draws=draws, completed_draws=0, complete=False,
                 status=status, counts=dict(finite=0, unavailable=0, finite_zero=0, positive=0), cv=None)
            for i, child in enumerate(np.random.SeedSequence(seed).spawn(replicates))]


def _sample(p, graph, original_n, original_m, *, engine, draws, replicates,
            seed, deadline, batch_size, max_batch_bytes):
    """Bounded batches; moments via mergeable Welford updates, no CI claim."""
    k = len(p)
    # A conservative working-array estimate, NOT a process memory hard limit.
    allowed = max_batch_bytes // max(1, 64 * k)
    if allowed < 1:
        return dict(status='RESOURCE_LIMIT', cv=None, cv_interval=None,
                    reason='ESTIMATED_SINGLE_DRAW_WORKSPACE_EXCEEDS_LIMIT',
                    completed_draws=0, requested_draws=draws * replicates,
                    replicates=_plan(draws, replicates, seed, 'NOT_RUN_RESOURCE_LIMIT'),
                    counts={'finite': 0, 'unavailable': 0, 'finite_zero': 0, 'positive': 0})
    chunk = 2 ** int(math.floor(math.log2(min(batch_size, allowed, draws))))
    scale = 2. * original_m / (original_n * (original_n - 1))
    cumulative = p[:, 0] + p[:, 1]
    seeds = np.random.SeedSequence(seed).spawn(replicates)
    counts = dict(finite=0, unavailable=0, finite_zero=0, positive=0)
    total = 0; mean = 0.; m2 = 0.; completed = 0
    summaries = []
    for rep, child in enumerate(seeds):
        if time.perf_counter() >= deadline:
            break
        rep_seed = int(child.generate_state(1)[0])
        generator = (qmc.Sobol(d=k, scramble=True, bits=53, seed=rep_seed)
                     if engine == 'SOBOL' else np.random.default_rng(rep_seed))
        nr = 0; rep_count = 0; rep_mean = 0.; rep_m2 = 0.
        rep_counts = dict(finite=0, unavailable=0, finite_zero=0, positive=0)
        for start in range(0, draws, chunk):
            if time.perf_counter() >= deadline:
                break
            size = min(chunk, draws - start)
            u = generator.random(n=size) if engine == 'SOBOL' else generator.random((size, k))
            a = (u < p[:, 0]).astype(float)
            b = ((u >= p[:, 0]) & (u < cumulative)).astype(float)
            denominator = scale * a.sum(axis=1) * b.sum(axis=1)
            contacts = np.sum((a @ graph) * b, axis=1)
            valid = denominator > 0
            values = contacts[valid] / denominator[valid]
            increments = dict(finite=int(valid.sum()), unavailable=int((~valid).sum()),
                              finite_zero=int(np.sum(values == 0)), positive=int(np.sum(values > 0)))
            for name, number in increments.items():
                counts[name] += number; rep_counts[name] += number
            n = len(values)
            if n:
                bm = float(values.mean()); bm2 = float(np.sum((values - bm) ** 2))
                delta = bm - mean
                m2 += bm2 + delta * delta * total * n / (total + n)
                mean += delta * n / (total + n); total += n
                delta = bm - rep_mean
                rep_m2 += bm2 + delta * delta * rep_count * n / (rep_count + n)
                rep_mean += delta * n / (rep_count + n); rep_count += n
            nr += size; completed += size
        rep_cv = math.sqrt(rep_m2 / rep_count) / rep_mean if rep_count >= 2 and rep_mean > 0 else None
        summaries.append(dict(replicate=rep, seed=rep_seed, requested_draws=draws,
                              completed_draws=nr, complete=nr == draws,
                              status='COMPLETE' if nr == draws else 'PARTIAL_BUDGET',
                              counts=rep_counts, cv=rep_cv))
        if nr < draws:
            break
    # Retain planned but unexecuted streams; they are not observed zero draws.
    for rep in range(len(summaries), replicates):
        summaries.append(dict(replicate=rep, seed=int(seeds[rep].generate_state(1)[0]), requested_draws=draws,
                              completed_draws=0, complete=False, status='NOT_RUN_BUDGET',
                              counts=dict(finite=0, unavailable=0, finite_zero=0, positive=0), cv=None))
    cv = math.sqrt(m2 / total) / mean if total >= 2 and mean > 0 else None
    complete = completed == draws * replicates
    if completed == 0:
        state = 'BUDGET_EXHAUSTED'
    elif counts['finite'] == 0:
        state = 'SAMPLE_NO_FINITE_OBSERVATIONS'
    elif counts['positive'] == 0:
        state = 'SAMPLE_NO_POSITIVE_CONTACTS'
    elif total < 2:
        state = 'SAMPLE_INSUFFICIENT_FINITE_OBSERVATIONS'
    else:
        state = 'APPROXIMATE_ESTIMATE' if complete else 'PARTIAL_SAMPLE_ESTIMATE'
    rep_values = [x['cv'] for x in summaries if x['complete'] and x['cv'] is not None]
    return dict(status=state, cv=cv, cv_interval=None,
                reason=None if complete else 'SOFT_BUDGET_EXHAUSTED',
                completed_draws=completed, requested_draws=draws * replicates,
                complete=complete, counts=counts,
                p_finite_estimate=counts['finite'] / completed if completed else None,
                finite_mean=mean if total else None, variance_ddof=0,
                sampling_uncertainty=dict(kind='DESCRIPTIVE_REPLICATE_SPREAD',
                    complete_replicates_with_cv=len(rep_values),
                    cv_min=min(rep_values) if rep_values else None,
                    cv_max=max(rep_values) if rep_values else None,
                    confidence_interval=None, coverage_calibration='NOT_ESTABLISHED'),
                replicates=summaries, seed=seed, batch_size=chunk,
                sobol_bits=53 if engine == 'SOBOL' else None,
                full_power_of_two_scrambles=complete if engine == 'SOBOL' else None)


def route_cv(probabilities, adjacency, *, requirement, accuracy_factor=1.25,
             a=0, b=1, max_seconds=6., direct_support_hint=1000,
             strict_factor_hint=1.01, screening_engine='auto', draws=1024,
             replicates=4, seed=0, allow_approximate_fallback=False,
             batch_size=32, max_batch_bytes=64 * 1024 ** 2):
    """Compute a conditional CV with an explicit requirement and route trace.

    requirement='enclosure' never falls back to a stochastic estimate.
    requirement='screening' permits approximate output; optional fallback is
    explicit and retains the complete D attempt. max_seconds is a soft total
    deadline, including validation/preparation, not a hard OS resource cap.
    K means the union of nodes with positive A or B probability, including
    isolates; it is not a tissue-cell-count or universal complexity boundary.
    """
    started = time.perf_counter()
    if requirement not in ('enclosure', 'screening'):
        raise ValueError('requirement must be enclosure or screening')
    if screening_engine not in ('auto', 'sobol', 'mc'):
        raise ValueError('screening_engine must be auto, sobol or mc')
    if not isinstance(allow_approximate_fallback, bool):
        raise ValueError('allow_approximate_fallback must be boolean')
    if requirement == 'enclosure' and allow_approximate_fallback:
        raise ValueError('enclosure requests cannot allow approximate fallback')
    for name, value in [('accuracy_factor', accuracy_factor), ('strict_factor_hint', strict_factor_hint)]:
        if isinstance(value, bool) or not np.isfinite(value) or value <= 1:
            raise ValueError(f'{name} must be finite and > 1')
    if isinstance(max_seconds, bool) or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be positive and finite')
    direct_support_hint = _integer('direct_support_hint', direct_support_hint, 0)
    draws = _integer('draws', draws, 2); replicates = _integer('replicates', replicates, 2)
    seed = _integer('seed', seed, 0); batch_size = _integer('batch_size', batch_size, 1)
    max_batch_bytes = _integer('max_batch_bytes', max_batch_bytes, 1)
    if draws & (draws - 1) or draws > 2 ** 53:
        raise ValueError('draws must be a power of two between 2 and 2**53')
    p = np.asarray(probabilities, dtype=float)
    a = _integer('a', a, 0); b = _integer('b', b, 0)
    if p.ndim != 2 or a == b or max(a, b) >= p.shape[1]:
        raise ValueError('2D probabilities and distinct valid class indices required')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('finite probabilities in [0,1] required')
    if any(math.fsum([1., *(-float(v) for v in row)]) < 0 for row in p):
        raise ValueError('categorical row sum exceeds one as exact input')
    prepared = prepare_pair_graph(adjacency)
    if len(p) != prepared.n:
        raise ValueError('probabilities and graph must align')
    pa, pb = p[:, a], p[:, b]
    active = (pa > 0) | (pb > 0); k = int(active.sum())
    ka = int((pa > 0).sum()); kb = int((pb > 0).sum())
    possible = ka * kb - int(np.sum((pa > 0) & (pb > 0)))
    i, j = prepared.row, prepared.col
    contacts_possible = bool(np.any(((pa[i] > 0) & (pb[j] > 0)) | ((pb[i] > 0) & (pa[j] > 0))))
    expected_contacts = float(pa @ (prepared.graph @ pb))
    rare_hint = contacts_possible and draws * expected_contacts < 1.
    trace = []
    result = dict(policy=POLICY_VERSION, policy_validated=False,
                  requirement=requirement, target='CV(O/E | E>0)',
                  accuracy_scope='fixed graph and supplied categorical probabilities only',
                  n=prepared.n, m=prepared.m, K_A=ka, K_B=kb, K_union=k,
                  route_trace=trace, rare_contact_hint=rare_hint,
                  rare_hint_basis='draws_per_replicate * floating_expected_contacts < 1; heuristic only',
                  expected_contacts_diagnostic=expected_contacts,
                  direct_support_hint=direct_support_hint, strict_factor_hint=strict_factor_hint,
                  requested_factor=accuracy_factor, factor_achieved=None,
                  factor_target_assessed=False, coverage_calibration='NOT_ESTABLISHED',
                  floating_rng_resolution_bits=53,
                  sub_resolution_probability_warning=bool(np.any(((pa > 0) & (pa < 2.**-53)) | ((pb > 0) & (pb < 2.**-53)))),
                  sampling_plan=_plan(draws, replicates, seed, 'NOT_SELECTED'),
                  fallback_used=False, attempts=[], max_seconds=max_seconds,
                  resource_scope='soft elapsed deadline; estimated batch workspace, no process hard cap')
    def finish(engine, output, guarantee):
        if engine in ('MC', 'SOBOL'):
            result['sampling_plan'] = output['replicates']
        else:
            state = 'NOT_RUN_BUDGET' if engine == 'NONE' else 'NOT_SELECTED_D' if engine == 'D' else 'NOT_NEEDED_STRUCTURAL'
            for entry in result['sampling_plan']:
                entry['status'] = state
        elapsed = time.perf_counter() - started
        result.update(engine=engine, output=output, status=output['status'],
                      cv=output.get('cv'), cv_interval=output.get('cv_interval'),
                      guarantee=guarantee, elapsed_seconds=elapsed,
                      within_soft_budget=elapsed <= max_seconds)
        return result
    deadline = started + max_seconds
    # Exact logical support checks cannot be replaced by observed sample zeros.
    if not prepared.m or not possible:
        trace.append('STRUCTURAL_NO_FINITE_RATIO')
        return finish('STRUCTURAL', dict(status='NO_FINITE_RATIO', cv=None,
                      p_finite_interval=[0., 0.]), 'STRUCTURAL_STATE_ONLY')
    if not contacts_possible:
        trace.append('STRUCTURAL_NO_POSSIBLE_CONTACT')
        return finish('STRUCTURAL', dict(status='ZERO_NUMERATOR_CV_UNDEFINED', cv=None),
                      'STRUCTURAL_STATE_ONLY')
    if time.perf_counter() >= deadline:
        return finish('NONE', dict(status='BUDGET_EXHAUSTED', cv=None), 'NONE')
    use_direct = requirement == 'enclosure' or accuracy_factor <= strict_factor_hint or k <= direct_support_hint
    if requirement == 'enclosure':
        trace.append('ENCLOSURE_REQUIRED')
    elif accuracy_factor <= strict_factor_hint:
        trace.append('STRICT_FACTOR_HEURISTIC')
    elif k <= direct_support_hint:
        trace.append('SMALL_SUPPORT_HEURISTIC')
    else:
        trace.append('SCREENING_LARGE_SUPPORT_HEURISTIC')
    if rare_hint:
        trace.append('RARE_CONTACT_WARNING_D_NOT_EXCLUSIVE; importance_sampling_not_implemented_in_router')
    if use_direct:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return finish('NONE', dict(status='BUDGET_EXHAUSTED', cv=None), 'NONE')
        # Explicit screening fallback reserves half the remaining soft budget.
        d_budget = remaining / 2 if allow_approximate_fallback else remaining
        # The prepared graph is read-only; the legacy D validator canonicalizes
        # its CSR arrays in place. Own a writable copy and charge it to this call.
        d = estimate_cv(p, prepared.graph.copy(), accuracy_factor=accuracy_factor, a=a, b=b,
                        max_seconds=d_budget, batch_size=batch_size)
        result['attempts'].append(dict(engine='D', output=d))
        if d['status'] in ('RESOLVED', 'EXACT_ZERO_CV'):
            result.update(factor_achieved=1. if d['status']=='EXACT_ZERO_CV' else d.get('achieved_factor'), factor_target_assessed=True)
            return finish('D', d, 'CONDITIONAL_NUMERICAL_ENCLOSURE')
        if d['status'] in ('NO_FINITE_RATIO', 'ZERO_NUMERATOR_CV_UNDEFINED'):
            return finish('D', d, 'STRUCTURAL_STATE_ONLY')
        if not allow_approximate_fallback or d['status'] != 'UNRESOLVED':
            return finish('D', d, 'NONE')
        if time.perf_counter() >= deadline:
            trace.append('FALLBACK_NOT_RUN_NO_BUDGET')
            return finish('D', d, 'NONE')
        result['fallback_used'] = True
        trace.append('EXPLICIT_APPROXIMATE_FALLBACK_AFTER_D_UNRESOLVED')
    engine, reason = _sampling_engine(screening_engine, k)
    trace.append(reason)
    pair = p[active][:, [a, b]]
    graph = prepared.graph[active][:, active].tocsr()
    sample = _sample(pair, graph, prepared.n, prepared.m, engine=engine,
                     draws=draws, replicates=replicates, seed=seed,
                     deadline=deadline, batch_size=batch_size, max_batch_bytes=max_batch_bytes)
    result['attempts'].append(dict(engine=engine, output=sample))
    return finish(engine, sample, 'APPROXIMATE_POINT_ESTIMATE' if sample.get('cv') is not None else 'NONE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probabilities', required=True, type=Path)
    parser.add_argument('--adjacency', required=True, type=Path)
    parser.add_argument('--requirement', required=True, choices=['enclosure', 'screening'])
    parser.add_argument('--factor', type=float, default=1.25)
    parser.add_argument('--a', type=int, default=0); parser.add_argument('--b', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--support-hint', type=int, default=1000)
    parser.add_argument('--screening-engine', choices=['auto', 'sobol', 'mc'], default='auto')
    parser.add_argument('--draws', type=int, default=1024)
    parser.add_argument('--replicates', type=int, default=4); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-approximate-fallback', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('output exists; refusing to overwrite')
    p = np.load(args.probabilities, allow_pickle=False); g = sparse.load_npz(args.adjacency)
    out = route_cv(p, g, requirement=args.requirement, accuracy_factor=args.factor,
                   a=args.a, b=args.b, max_seconds=args.seconds,
                   direct_support_hint=args.support_hint, screening_engine=args.screening_engine,
                   draws=args.draws, replicates=args.replicates, seed=args.seed,
                   allow_approximate_fallback=args.allow_approximate_fallback)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(json_safe(out), f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
