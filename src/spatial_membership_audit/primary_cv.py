"""One primary conditional-CV output, with an explicit numerical accuracy goal.

Real-input coverage is recorded separately in DIRECT_CV.md. This module never
chooses Delta/Expected-O using an evaluation target or fills an unresolved CV.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from .direct_cv import estimate


def estimate_cv(probabilities, adjacency, *, accuracy_factor, a=0, b=1,
                max_seconds=6., orders=(4, 8, 16, 32, 64), batch_size=32):
    """Return CV(O/E | E>0), a numerical enclosure and an explicit status.

    accuracy_factor is required: no scientific tolerance is assumed for the
    caller. max_seconds is a soft computational deadline checked at boundaries;
    a hard total wall-clock cap requires an external worker, as in the pilot.
    Probabilities are independent categorical probabilities, not arbitrary
    classifier scores. The certificate conditions on their exact float values.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or isinstance(a, bool) or isinstance(b, bool):
        raise ValueError('2D categorical probabilities and integer class indices required')
    if a == b or min(a,b) < 0 or max(a,b) >= p.shape[1]:
        raise ValueError('distinct valid class indices required')
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('finite categorical probabilities in [0,1] required')
    row_sums = np.sum(p, axis=1)
    slack = float(p.shape[1] * np.finfo(float).eps)
    suspicious = np.flatnonzero(row_sums > 1.0 - slack)
    if len(suspicious) > 0:
        for idx in suspicious:
            if math.fsum([1., *(-float(v) for v in p[idx])]) < 0.:
                raise ValueError('categorical row sum exceeds one')
    out = estimate(p[:,a], p[:,b], adjacency, accuracy_factor=accuracy_factor,
                   orders=orders, max_seconds=max_seconds, batch_size=batch_size)
    out['class_indices'] = [int(a), int(b)]
    out['primary_output'] = 'cv'
    out['fallback_used'] = False
    out['accuracy_scope'] = 'exact_float_input_conditional_model; not upstream probability or physical measurement error'
    return out


def json_safe(value):
    if isinstance(value, dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probabilities', type=Path, required=True, help='2D .npy categorical probabilities')
    parser.add_argument('--adjacency', type=Path, required=True, help='SciPy sparse .npz binary symmetric graph')
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=1)
    parser.add_argument('--accuracy-factor', type=float, required=True)
    parser.add_argument('--compute-seconds', type=float, default=6., help='soft algorithm time limit; excludes file I/O')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('output already exists; refusing to overwrite')
    p = np.load(args.probabilities, allow_pickle=False)
    g = sparse.load_npz(args.adjacency)
    result = estimate_cv(p, g, a=args.a, b=args.b, accuracy_factor=args.accuracy_factor,
                         max_seconds=args.compute_seconds)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
