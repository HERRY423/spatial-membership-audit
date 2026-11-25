"""Declarative marker-gating rules and frozen cell-wise classifier adapters.

Extends spatial_membership_audit beyond the hardcoded 11-marker
(common125_candidate_v1) and 4-gene (stromal_myeloid) rules to arbitrary
user-defined panels, count thresholds, multi-gene sum gates, exclusion vetoes,
and frozen cell-wise classifiers under independent binomial molecule thinning.

All probabilities condition on observed integer counts X and specified
retention p. Candidate labels and probabilities describe operational rules,
not ground-truth biological identities.
"""
from dataclasses import dataclass, field
from itertools import product
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import binom

from .adapters import validate_counts
from .definitions import REQUIRED as REQUIRED_11, RULE_ID as RULE_11_ID
from .analytic import CLASSES as CLASSES_11


@dataclass(frozen=True)
class GeneThreshold:
    """Single-gene retained-count condition: min_count <= Y_g <= max_count."""
    gene: str
    min_count: int = 1
    max_count: int | None = None

    def __post_init__(self):
        if not isinstance(self.gene, str) or not self.gene.strip():
            raise ValueError('gene must be a nonempty string')
        if isinstance(self.min_count, bool) or not isinstance(self.min_count, (int, np.integer)) or self.min_count < 0:
            raise ValueError('min_count must be a nonnegative integer')
        if self.max_count is not None:
            if isinstance(self.max_count, bool) or not isinstance(self.max_count, (int, np.integer)) or self.max_count < self.min_count:
                raise ValueError('max_count must be an integer >= min_count or None')

    def genes(self) -> tuple[str, ...]:
        return (self.gene,)

    def cap_for(self, gene: str) -> int:
        if gene != self.gene:
            return 0
        if self.max_count is None:
            return int(self.min_count)
        return int(self.max_count) + 1

    def evaluate_state(self, state: Mapping[str, int]) -> bool:
        val = state[self.gene]
        if val < self.min_count:
            return False
        if self.max_count is not None and val > self.max_count:
            return False
        return True


@dataclass(frozen=True)
class GeneSumThreshold:
    """Multi-gene retained-count sum condition: sum_{g in genes} Y_g >= min_sum
    and each Y_g >= min_each.
    """
    genes_tuple: tuple[str, ...]
    min_sum: int
    min_each: int = 0

    def __init__(self, genes: Sequence[str], min_sum: int, min_each: int = 0):
        g_tuple = tuple(genes)
        if not g_tuple or any(not isinstance(g, str) or not g.strip() for g in g_tuple):
            raise ValueError('genes must be a nonempty sequence of gene names')
        if len(set(g_tuple)) != len(g_tuple):
            raise ValueError('duplicate genes in GeneSumThreshold')
        if isinstance(min_sum, bool) or not isinstance(min_sum, (int, np.integer)) or min_sum < 0:
            raise ValueError('min_sum must be a nonnegative integer')
        if isinstance(min_each, bool) or not isinstance(min_each, (int, np.integer)) or min_each < 0:
            raise ValueError('min_each must be a nonnegative integer')
        object.__setattr__(self, 'genes_tuple', g_tuple)
        object.__setattr__(self, 'min_sum', int(min_sum))
        object.__setattr__(self, 'min_each', int(min_each))

    def genes(self) -> tuple[str, ...]:
        return self.genes_tuple

    def cap_for(self, gene: str) -> int:
        if gene not in self.genes_tuple:
            return 0
        # Saturate each gene at max(min_each, min_sum - (k-1)*min_each)
        k = len(self.genes_tuple)
        needed = max(self.min_each, self.min_sum - (k - 1) * self.min_each)
        return max(1, int(needed))

    def evaluate_state(self, state: Mapping[str, int]) -> bool:
        vals = [state[g] for g in self.genes_tuple]
        if any(v < self.min_each for v in vals):
            return False
        return sum(vals) >= self.min_sum


@dataclass(frozen=True)
class AllOf:
    """Logical conjunction (AND) of sub-predicates."""
    predicates: tuple

    def __init__(self, predicates: Iterable):
        preds = tuple(predicates)
        if not preds:
            raise ValueError('AllOf requires at least one predicate')
        object.__setattr__(self, 'predicates', preds)

    def genes(self) -> tuple[str, ...]:
        out = []
        for p in self.predicates:
            for g in p.genes():
                if g not in out:
                    out.append(g)
        return tuple(out)

    def cap_for(self, gene: str) -> int:
        return max((p.cap_for(gene) for p in self.predicates), default=0)

    def evaluate_state(self, state: Mapping[str, int]) -> bool:
        return all(p.evaluate_state(state) for p in self.predicates)


@dataclass(frozen=True)
class AnyOf:
    """Logical disjunction (OR) of sub-predicates."""
    predicates: tuple

    def __init__(self, predicates: Iterable):
        preds = tuple(predicates)
        if not preds:
            raise ValueError('AnyOf requires at least one predicate')
        object.__setattr__(self, 'predicates', preds)

    def genes(self) -> tuple[str, ...]:
        out = []
        for p in self.predicates:
            for g in p.genes():
                if g not in out:
                    out.append(g)
        return tuple(out)

    def cap_for(self, gene: str) -> int:
        return max((p.cap_for(gene) for p in self.predicates), default=0)

    def evaluate_state(self, state: Mapping[str, int]) -> bool:
        return any(p.evaluate_state(state) for p in self.predicates)


@dataclass(frozen=True)
class NotRule:
    """Logical negation (NOT) of a predicate."""
    predicate: object

    def genes(self) -> tuple[str, ...]:
        return self.predicate.genes()

    def cap_for(self, gene: str) -> int:
        return self.predicate.cap_for(gene)

    def evaluate_state(self, state: Mapping[str, int]) -> bool:
        return not self.predicate.evaluate_state(state)


@dataclass(frozen=True)
class CandidateClassSpec:
    """One operational candidate class within a mutually exclusive rule.

    A cell belongs to ``name`` iff:
      1. Its baseline eligibility is True,
      2. ``predicate`` evaluates to True,
      3. All ``excluded_gates`` (e.g. competing lineages and global vetoes)
         evaluate to False.
    """
    name: str
    positive_gate: str
    required_subgates: tuple[str, ...] = ()
    forbidden_gates: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('candidate class name must be a nonempty string')
        if not isinstance(self.positive_gate, str) or not self.positive_gate.strip():
            raise ValueError('positive_gate must be a nonempty gate name')
        object.__setattr__(self, 'required_subgates', tuple(self.required_subgates))
        object.__setattr__(self, 'forbidden_gates', tuple(self.forbidden_gates))


@dataclass(frozen=True)
class MarkerRuleSpec:
    """Complete declarative specification of a mutually exclusive marker rule.

    Parameters
    ----------
    rule_id : str
        Explicit identifier for provenance tracking.
    gates : Mapping[str, predicate]
        Named boolean predicates over marker genes. Gates that do not share
        genes are evaluated on independent binomial components; gates sharing
        genes are automatically grouped into joint connected components so
        within-cell correlations are exact.
    candidates : Sequence[CandidateClassSpec]
        Ordered candidate classes. Mutual exclusivity across candidates is
        verified exhaustively over all gate truth assignments at initialization.
    other_label : str
        Label assigned to cells that fail QC, trigger vetoes, have conflicting
        lineage evidence, or match no candidate class.
    """
    rule_id: str
    gates: Mapping[str, object]
    candidates: tuple[CandidateClassSpec, ...]
    other_label: str = 'OTHER'

    def __init__(self, rule_id: str, gates: Mapping[str, object],
                 candidates: Sequence[CandidateClassSpec], other_label: str = 'OTHER'):
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError('rule_id must be a nonempty string')
        if not gates:
            raise ValueError('at least one named gate is required')
        gates_dict = dict(gates)
        cand_tuple = tuple(candidates)
        if len(cand_tuple) < 2:
            raise ValueError('at least two candidate classes required')
        names = [c.name for c in cand_tuple]
        if len(set(names)) != len(names) or other_label in names:
            raise ValueError('candidate class names must be unique and distinct from other_label')
        for c in cand_tuple:
            for gname in (c.positive_gate, *c.required_subgates, *c.forbidden_gates):
                if gname not in gates_dict:
                    raise ValueError(f'candidate {c.name} references unknown gate {gname!r}')
        # Verify mutual exclusivity across all 2^G gate truth assignments
        gate_names = tuple(gates_dict.keys())
        if len(gate_names) <= 20:
            for bits in product((False, True), repeat=len(gate_names)):
                gmap = dict(zip(gate_names, bits))
                active = [
                    c.name for c in cand_tuple
                    if gmap[c.positive_gate]
                    and all(gmap[req] for req in c.required_subgates)
                    and not any(gmap[forb] for forb in c.forbidden_gates)
                ]
                if len(active) > 1:
                    raise ValueError(
                        f'rule {rule_id!r} is not mutually exclusive: '
                        f'candidates {active} simultaneously active under gate state {gmap}'
                    )
        object.__setattr__(self, 'rule_id', rule_id)
        object.__setattr__(self, 'gates', gates_dict)
        object.__setattr__(self, 'candidates', cand_tuple)
        object.__setattr__(self, 'other_label', other_label)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.candidates)

    @property
    def required_genes(self) -> tuple[str, ...]:
        seen = []
        for pred in self.gates.values():
            for g in pred.genes():
                if g not in seen:
                    seen.append(g)
        return tuple(seen)

    def panel_contract(self, measured_genes: Sequence[str]) -> dict:
        genes_list = list(measured_genes)
        if len(genes_list) != len(set(genes_list)):
            raise ValueError('Duplicate gene names')
        req = self.required_genes
        missing = sorted(set(req) - set(genes_list))
        return dict(
            rule_id=self.rule_id,
            required_genes=list(req),
            classes=list(self.classes),
            missing_genes=missing,
            status='NOT_IDENTIFIABLE' if missing else 'CANDIDATE_ONLY_IDENTITY_NOT_VALIDATED',
        )


def predicate_from_dict(spec: Mapping) -> object:
    """Build a gate predicate from a JSON/YAML-compatible dictionary."""
    kind = spec.get('op', 'threshold')
    if kind == 'threshold':
        return GeneThreshold(
            gene=spec['gene'],
            min_count=spec.get('min_count', 1),
            max_count=spec.get('max_count', None),
        )
    if kind == 'sum':
        return GeneSumThreshold(
            genes=spec['genes'],
            min_sum=spec['min_sum'],
            min_each=spec.get('min_each', 0),
        )
    if kind == 'all':
        return AllOf(predicate_from_dict(sub) for sub in spec['predicates'])
    if kind == 'any':
        return AnyOf(predicate_from_dict(sub) for sub in spec['predicates'])
    if kind == 'not':
        return NotRule(predicate_from_dict(spec['predicate']))
    raise ValueError(f'Unknown predicate op: {kind!r}')


def marker_rule_from_dict(spec: Mapping) -> MarkerRuleSpec:
    """Construct a MarkerRuleSpec from a dictionary specification.

    Supports both explicit ``gates`` + ``candidates`` dicts and a simplified
    ``lineages`` dict where all lineages mutually exclude each other and
    optional ``vetoes`` exclude all lineages.
    """
    rule_id = spec.get('rule_id', 'custom_marker_rule')
    other_label = spec.get('other_label', 'OTHER')
    if 'lineages' in spec:
        gates = {}
        lineage_names = []
        for lname, lpred in spec['lineages'].items():
            gates[lname] = predicate_from_dict(lpred) if isinstance(lpred, Mapping) else lpred
            lineage_names.append(lname)
        veto_names = []
        for vname, vpred in spec.get('vetoes', {}).items():
            gates[vname] = predicate_from_dict(vpred) if isinstance(vpred, Mapping) else vpred
            veto_names.append(vname)
        candidates = []
        for lname in lineage_names:
            forbidden = [other for other in lineage_names if other != lname] + veto_names
            candidates.append(CandidateClassSpec(
                name=lname,
                positive_gate=lname,
                forbidden_gates=tuple(forbidden),
            ))
        return MarkerRuleSpec(rule_id=rule_id, gates=gates, candidates=candidates, other_label=other_label)

    gates = {
        k: (predicate_from_dict(v) if isinstance(v, Mapping) else v)
        for k, v in spec['gates'].items()
    }
    candidates = [
        CandidateClassSpec(
            name=c['name'],
            positive_gate=c['positive_gate'],
            required_subgates=tuple(c.get('required_subgates', ())),
            forbidden_gates=tuple(c.get('forbidden_gates', ())),
        )
        for c in spec['candidates']
    ]
    return MarkerRuleSpec(rule_id=rule_id, gates=gates, candidates=candidates, other_label=other_label)


# Built-in presets demonstrating that the hardcoded 11-marker and 4-gene rules
# are exact special cases of MarkerRuleSpec, alongside general multi-lineage rules.
COMMON125_11MARKER_SPEC = MarkerRuleSpec(
    rule_id=RULE_11_ID,
    gates={
        'B': AllOf([
            GeneThreshold('MS4A1', min_count=1),
            AnyOf([GeneThreshold('CD19', min_count=1), GeneThreshold('CD79A', min_count=1)]),
        ]),
        'T': GeneSumThreshold(('CD3D', 'CD3E'), min_sum=3, min_each=1),
        'MYELOID': AllOf([GeneThreshold('CD14', min_count=1), GeneThreshold('CD68', min_count=1)]),
        'PLASMA_VETO': AllOf([GeneThreshold('MZB1', min_count=1), GeneThreshold('TNFRSF17', min_count=1)]),
        'CD8_HIGH': GeneThreshold('CD8A', min_count=2),
        'CD4_POS': GeneThreshold('CD4', min_count=1),
    },
    candidates=(
        CandidateClassSpec('B_CANDIDATE', positive_gate='B', forbidden_gates=('T', 'MYELOID', 'PLASMA_VETO')),
        CandidateClassSpec('CD8_T_CANDIDATE', positive_gate='T', required_subgates=('CD8_HIGH',),
                           forbidden_gates=('B', 'MYELOID', 'PLASMA_VETO', 'CD4_POS')),
        CandidateClassSpec('CD4_T_CANDIDATE', positive_gate='T', required_subgates=('CD4_POS',),
                           forbidden_gates=('B', 'MYELOID', 'PLASMA_VETO', 'CD8_HIGH')),
        CandidateClassSpec('MYELOID_CANDIDATE', positive_gate='MYELOID', forbidden_gates=('B', 'T', 'PLASMA_VETO')),
    ),
)

STROMAL_MYELOID_4GENE_SPEC = MarkerRuleSpec(
    rule_id='stromal_myeloid_4gene_v1',
    gates={
        'STROMAL': AllOf([GeneThreshold('COL1A1', min_count=1), GeneThreshold('FBN1', min_count=1)]),
        'MYELOID': AllOf([GeneThreshold('CSF1R', min_count=1), GeneThreshold('TYROBP', min_count=1)]),
    },
    candidates=(
        CandidateClassSpec('STROMAL_CANDIDATE', positive_gate='STROMAL', forbidden_gates=('MYELOID',)),
        CandidateClassSpec('MYELOID_CANDIDATE', positive_gate='MYELOID', forbidden_gates=('STROMAL',)),
    ),
)

EPCAM_STROMAL_4GENE_SPEC = MarkerRuleSpec(
    rule_id='epithelial_stromal_4gene_v1',
    gates={
        'EPITHELIAL': AllOf([GeneThreshold('EPCAM', min_count=1), GeneThreshold('KRT7', min_count=1)]),
        'STROMAL': AllOf([GeneThreshold('PDGFRA', min_count=1), GeneThreshold('COL5A2', min_count=1)]),
    },
    candidates=(
        CandidateClassSpec('EPITHELIAL_CANDIDATE', positive_gate='EPITHELIAL', forbidden_gates=('STROMAL',)),
        CandidateClassSpec('STROMAL_CANDIDATE', positive_gate='STROMAL', forbidden_gates=('EPITHELIAL',)),
    ),
)

GENERIC_3LINEAGE_6MARKER_SPEC = MarkerRuleSpec(
    rule_id='generic_tme_3lineage_6marker_v1',
    gates={
        'EPITHELIAL': AnyOf([GeneThreshold('EPCAM', min_count=1), GeneThreshold('KRT8', min_count=1)]),
        'T_CELL': AnyOf([GeneThreshold('CD3D', min_count=1), GeneThreshold('CD3E', min_count=1)]),
        'MACROPHAGE': AllOf([GeneThreshold('CD68', min_count=1), GeneThreshold('CD163', min_count=1)]),
    },
    candidates=(
        CandidateClassSpec('EPITHELIAL', positive_gate='EPITHELIAL', forbidden_gates=('T_CELL', 'MACROPHAGE')),
        CandidateClassSpec('T_CELL', positive_gate='T_CELL', forbidden_gates=('EPITHELIAL', 'MACROPHAGE')),
        CandidateClassSpec('MACROPHAGE', positive_gate='MACROPHAGE', forbidden_gates=('EPITHELIAL', 'T_CELL')),
    ),
)

PRESET_RULES: dict[str, MarkerRuleSpec] = {
    RULE_11_ID: COMMON125_11MARKER_SPEC,
    'common125_candidate_v1': COMMON125_11MARKER_SPEC,
    '11_marker': COMMON125_11MARKER_SPEC,
    'stromal_myeloid_4gene': STROMAL_MYELOID_4GENE_SPEC,
    'stromal_myeloid_4gene_v1': STROMAL_MYELOID_4GENE_SPEC,
    'epithelial_stromal_4gene': EPCAM_STROMAL_4GENE_SPEC,
    'generic_tme_3lineage_6marker_v1': GENERIC_3LINEAGE_6MARKER_SPEC,
    'generic_3lineage_6marker': GENERIC_3LINEAGE_6MARKER_SPEC,
}


def resolve_rule_spec(rule: str | Mapping | MarkerRuleSpec) -> MarkerRuleSpec:
    """Resolve a preset name, dictionary specification, or MarkerRuleSpec."""
    if isinstance(rule, MarkerRuleSpec):
        return rule
    if isinstance(rule, str):
        if rule not in PRESET_RULES:
            raise ValueError(f'Unknown preset rule {rule!r}; available presets: {sorted(PRESET_RULES)}')
        return PRESET_RULES[rule]
    if isinstance(rule, Mapping):
        return marker_rule_from_dict(rule)
    raise TypeError('rule must be a preset name, dict specification, or MarkerRuleSpec')


def _single_gene_capped_pmf(x_col: np.ndarray, p_val: float, cap: int) -> np.ndarray:
    """Return exact (N, cap+1) probabilities for states 0, 1, ..., cap-1, >=cap.

    Uses exp(x*log1p(-p)) and -expm1(x*log1p(-p)) for cap=1 to avoid
    catastrophic cancellation when survival probability is near 0 or 1.
    """
    n = len(x_col)
    out = np.zeros((n, cap + 1), dtype=float)
    if p_val == 0.0:
        out[:, 0] = 1.0
        return out
    if p_val == 1.0:
        idx = np.minimum(x_col.astype(int), cap)
        out[np.arange(n), idx] = 1.0
        return out
    if cap == 1:
        log_q = np.log1p(-p_val)
        out[:, 0] = np.exp(x_col * log_q)
        out[:, 1] = -np.expm1(x_col * log_q)
        return out
    for k in range(cap):
        out[:, k] = binom.pmf(k, x_col, p_val)
    out[:, cap] = binom.sf(cap - 1, x_col, p_val)
    return out


def _connected_gate_components(rule_spec: MarkerRuleSpec) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Partition gates and their referenced genes into connected components.

    Two gates belong to the same component if they share at least one gene.
    Gates in distinct components depend on disjoint gene sets and are therefore
    conditionally independent given (X, p).
    """
    gate_names = list(rule_spec.gates.keys())
    gate_genes = {gname: set(pred.genes()) for gname, pred in rule_spec.gates.items()}
    visited_gates = set()
    components = []
    for gname in gate_names:
        if gname in visited_gates:
            continue
        comp_gates = []
        comp_genes = set()
        queue = [gname]
        visited_gates.add(gname)
        while queue:
            curr = queue.pop(0)
            comp_gates.append(curr)
            curr_genes = gate_genes[curr]
            comp_genes |= curr_genes
            for other in gate_names:
                if other not in visited_gates and (gate_genes[other] & comp_genes):
                    visited_gates.add(other)
                    queue.append(other)
        ordered_genes = tuple(g for g in rule_spec.required_genes if g in comp_genes)
        components.append((tuple(comp_gates), ordered_genes))
    return components


def evaluate_marker_rule_probabilities(
    counts,
    genes: Sequence[str],
    eligible,
    p,
    rule: str | Mapping | MarkerRuleSpec = COMMON125_11MARKER_SPEC,
) -> np.ndarray:
    """Compute exact per-cell candidate probabilities under binomial thinning.

    Parameters
    ----------
    counts : array-like or sparse matrix of shape (N, G)
        Observed nonnegative integer transcript counts.
    genes : sequence of str of length G
        Measured gene names aligned with columns of ``counts``.
    eligible : boolean array-like of shape (N,)
        Fixed baseline QC eligibility mask. Ineligible cells receive zero
        probability for all candidate classes (remaining in OTHER).
    p : float or array-like of shape (G,)
        Scalar retention rate in [0, 1] or gene-aligned retention vector.
    rule : str, dict, or MarkerRuleSpec
        Marker rule specification or preset name.

    Returns
    -------
    probabilities : np.ndarray of shape (N, K)
        Exact probabilities for each of the K mutually exclusive candidate
        classes defined in ``rule.classes``.
    """
    spec = resolve_rule_spec(rule)
    genes_list = list(genes)
    contract = spec.panel_contract(genes_list)
    if contract['missing_genes']:
        raise ValueError('NOT_IDENTIFIABLE: missing measured genes: ' + ','.join(contract['missing_genes']))
    mask = np.asarray(eligible)
    if mask.dtype != bool or mask.ndim != 1:
        raise ValueError('fixed baseline boolean mask required')
    if sparse.issparse(counts):
        if counts.shape != (len(mask), len(genes_list)):
            raise ValueError('counts/genes/eligible shape mismatch')
        # Extract only required genes into dense array after validating full matrix
        validate_counts(counts.data)
    else:
        x_full = np.asarray(counts)
        if x_full.ndim != 2 or x_full.shape != (len(mask), len(genes_list)):
            raise ValueError('counts/genes/eligible shape mismatch')
        validate_counts(x_full)

    retention = np.asarray(p, dtype=float)
    if retention.ndim == 0:
        retention = np.full(len(genes_list), float(retention))
    if retention.shape != (len(genes_list),) or not np.all(np.isfinite(retention)) or np.any((retention < 0) | (retention > 1)):
        raise ValueError('p must be in [0,1], scalar or aligned with measured genes')

    req_genes = spec.required_genes
    ix = [genes_list.index(g) for g in req_genes]
    if sparse.issparse(counts):
        x = counts[:, ix].toarray().astype(float)
    else:
        x = np.asarray(counts[:, ix], dtype=float)
    ret_req = retention[ix]
    gene_to_local = {g: idx for idx, g in enumerate(req_genes)}

    n_cells = len(mask)
    components = _connected_gate_components(spec)

    # For each connected component of gates, compute exact probabilities for all
    # 2^(len(comp_gates)) boolean truth assignments by summing disjoint count states.
    comp_truth_probs: list[dict[tuple[bool, ...], np.ndarray]] = []
    comp_gate_lists: list[tuple[str, ...]] = []

    for comp_gates, comp_genes in components:
        caps = {
            g: max(spec.gates[gname].cap_for(g) for gname in comp_gates)
            for g in comp_genes
        }
        pmfs = {
            g: _single_gene_capped_pmf(x[:, gene_to_local[g]], float(ret_req[gene_to_local[g]]), caps[g])
            for g in comp_genes
        }
        truth_accum: dict[tuple[bool, ...], np.ndarray] = {
            bits: np.zeros(n_cells, dtype=float)
            for bits in product((False, True), repeat=len(comp_gates))
        }
        state_ranges = [range(caps[g] + 1) for g in comp_genes]
        for count_tuple in product(*state_ranges):
            state_map = dict(zip(comp_genes, count_tuple))
            bits = tuple(spec.gates[gname].evaluate_state(state_map) for gname in comp_gates)
            prob_state = np.ones(n_cells, dtype=float)
            for g, k_val in zip(comp_genes, count_tuple):
                prob_state = prob_state * pmfs[g][:, k_val]
            truth_accum[bits] += prob_state
        comp_truth_probs.append(truth_accum)
        comp_gate_lists.append(comp_gates)

    # Evaluate each candidate class using its exact gate constraints across components
    n_classes = len(spec.candidates)
    out = np.zeros((n_cells, n_classes), dtype=float)

    for c_idx, cand in enumerate(spec.candidates):
        req_true = {cand.positive_gate, *cand.required_subgates}
        req_false = set(cand.forbidden_gates)
        if req_true & req_false:
            continue
        class_prob = np.ones(n_cells, dtype=float)
        for comp_gates, truth_accum in zip(comp_gate_lists, comp_truth_probs):
            # Sum over all truth assignments in this component consistent with candidate
            matching_arrays = []
            for bits, arr in truth_accum.items():
                gmap = dict(zip(comp_gates, bits))
                if all(gmap[g] for g in req_true if g in gmap) and not any(gmap[g] for g in req_false if g in gmap):
                    matching_arrays.append(arr)
            if not matching_arrays:
                class_prob = np.zeros(n_cells, dtype=float)
                break
            if len(matching_arrays) == 1:
                comp_prob = matching_arrays[0]
            else:
                comp_prob = np.sum(matching_arrays, axis=0)
            class_prob = class_prob * comp_prob
        out[:, c_idx] = class_prob

    out = out * mask[:, None]
    # Check for floating underflow when raw-count support at p=1 is positive and p>0
    if np.all(ret_req > 0):
        p1 = evaluate_marker_rule_probabilities(x, req_genes, mask, 1.0, spec) if np.any(ret_req < 1.0) else out
        support = (p1 > 0) & mask[:, None]
        if np.any(support & (out == 0)):
            raise FloatingPointError('NUMERICAL_UNDERFLOW: raw-count class support is positive')
    return out


def evaluate_marker_rule_labels(
    counts,
    genes: Sequence[str],
    baseline_eligible,
    rule: str | Mapping | MarkerRuleSpec = COMMON125_11MARKER_SPEC,
    ineligible_label: str = 'LOW_DETECTION_BASELINE',
) -> np.ndarray:
    """Assign deterministic candidate labels at baseline (p=1.0) for any MarkerRuleSpec."""
    spec = resolve_rule_spec(rule)
    mask = np.asarray(baseline_eligible)
    probs = evaluate_marker_rule_probabilities(counts, genes, mask, 1.0, spec)
    labels = np.full(len(mask), spec.other_label, dtype=object)
    for idx, cname in enumerate(spec.classes):
        labels[probs[:, idx] > 0.5] = cname
    labels[~mask] = ineligible_label
    return labels


@dataclass(frozen=True)
class FrozenLinearCentroidClassifier:
    """A frozen cell-independent linear/centroid classifier on log1p-normalized counts.

    Because the reference weights and intercepts are frozen prior to thinning,
    each cell's classification under binomial molecule thinning depends only on
    its own retained count vector Y_i, satisfying cross-cell conditional
    independence given (X, p).
    """
    genes: tuple[str, ...]
    classes: tuple[str, ...]
    weights: np.ndarray
    intercept: np.ndarray
    scale_factor: float = 1e4
    min_margin: float = 0.0
    min_positive_markers: int = 1
    other_label: str = 'OTHER'

    def __init__(self, genes: Sequence[str], classes: Sequence[str], weights,
                 intercept=None, *, scale_factor: float = 1e4,
                 min_margin: float = 0.0, min_positive_markers: int = 1,
                 other_label: str = 'OTHER'):
        g_tuple = tuple(genes)
        c_tuple = tuple(classes)
        w = np.asarray(weights, dtype=float)
        if len(g_tuple) < 1 or len(c_tuple) < 2:
            raise ValueError('at least 1 gene and 2 classes required')
        if len(set(g_tuple)) != len(g_tuple) or len(set(c_tuple)) != len(c_tuple):
            raise ValueError('unique genes and classes required')
        if w.shape != (len(g_tuple), len(c_tuple)) or not np.all(np.isfinite(w)):
            raise ValueError('finite weights of shape (len(genes), len(classes)) required')
        b = np.zeros(len(c_tuple), dtype=float) if intercept is None else np.asarray(intercept, dtype=float)
        if b.shape != (len(c_tuple),) or not np.all(np.isfinite(b)):
            raise ValueError('finite intercept of shape (len(classes),) required')
        object.__setattr__(self, 'genes', g_tuple)
        object.__setattr__(self, 'classes', c_tuple)
        object.__setattr__(self, 'weights', w.copy())
        object.__setattr__(self, 'intercept', b.copy())
        object.__setattr__(self, 'scale_factor', float(scale_factor))
        object.__setattr__(self, 'min_margin', float(min_margin))
        object.__setattr__(self, 'min_positive_markers', int(min_positive_markers))
        object.__setattr__(self, 'other_label', str(other_label))

    def __call__(self, counts_batch: np.ndarray, genes: Sequence[str]) -> np.ndarray:
        genes_list = list(genes)
        ix = [genes_list.index(g) for g in self.genes]
        sub = np.asarray(counts_batch[:, ix], dtype=float)
        totals = sub.sum(axis=1, keepdims=True)
        detected = (sub > 0).sum(axis=1)
        normed = np.log1p(np.where(totals > 0, sub / np.maximum(totals, 1.0) * self.scale_factor, 0.0))
        scores = normed @ self.weights + self.intercept[None, :]
        order = np.argsort(scores, axis=1)
        top_idx = order[:, -1]
        second_idx = order[:, -2]
        margin = scores[np.arange(len(sub)), top_idx] - scores[np.arange(len(sub)), second_idx]
        top_score = scores[np.arange(len(sub)), top_idx]
        valid = (detected >= self.min_positive_markers) & (margin >= self.min_margin) & (top_score > 0)
        labels = np.full(len(sub), self.other_label, dtype=object)
        for c_i, cname in enumerate(self.classes):
            labels[valid & (top_idx == c_i)] = cname
        return labels


def frozen_classifier_probabilities(
    counts,
    genes: Sequence[str],
    eligible,
    p,
    *,
    classifier_fn: Callable[[np.ndarray, Sequence[str]], Sequence[str]],
    classes: Sequence[str],
    n_draws: int = 512,
    seed: int = 1729,
    batch_draws: int = 64,
    verify_cell_independence: bool = True,
    assume_cellwise: bool = False,
) -> dict:
    """Estimate per-cell thinning reassignment probabilities for a frozen cell-wise classifier.

    Unlike unsupervised clustering or cohort-refitted annotations, a frozen
    cell-wise classifier maps each cell's retained count vector Y_i to a label
    independently of other cells. Consequently, conditional on (X, p), cell
    classifications remain independent across nodes and can be propagated into
    pair_moments, estimate_cv, and route_cv.

    Arbitrary callables require an explicit ``assume_cellwise=True`` acknowledgement.
    Finite invariance checks can falsify this assumption but cannot prove it.
    Returns a dictionary containing ``probabilities`` (N, K), ``classes``,
    ``baseline_labels``, and provenance metadata. MC zeros are unobserved
    outcomes, not certificates of structural absence. Downstream numerical
    enclosures condition on the estimated probabilities and exclude MC error.
    """
    genes_list = list(genes)
    if len(genes_list) != len(set(genes_list)):
        raise ValueError('unique gene names required')
    classes_tuple = tuple(classes)
    if len(classes_tuple) < 2 or len(set(classes_tuple)) != len(classes_tuple):
        raise ValueError('at least two unique target classes required')
    mask = np.asarray(eligible)
    if mask.dtype != bool or mask.ndim != 1:
        raise ValueError('fixed baseline boolean mask required')
    x = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    if x.ndim != 2 or x.shape != (len(mask), len(genes_list)):
        raise ValueError('counts/genes/eligible shape mismatch')
    validate_counts(x)
    x_int = x.astype(np.int64, copy=False)

    retention = np.asarray(p, dtype=float)
    if retention.ndim == 0:
        retention = np.full(len(genes_list), float(retention))
    if retention.shape != (len(genes_list),) or not np.all(np.isfinite(retention)) or np.any((retention < 0) | (retention > 1)):
        raise ValueError('p must be in [0,1], scalar or aligned with measured genes')
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or n_draws < 16:
        raise ValueError('n_draws must be an integer >= 16')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(batch_draws, bool) or not isinstance(batch_draws, (int, np.integer)) or batch_draws < 1:
        raise ValueError('batch_draws must be a positive integer')

    n_cells = len(mask)
    built_in = type(classifier_fn) is FrozenLinearCentroidClassifier
    if not built_in and assume_cellwise is not True:
        raise ValueError('Custom classifier requires assume_cellwise=True after verifying '
                         'frozen, deterministic, per-cell semantics; finite checks are not a proof')
    independence_basis = ('built_in_cellwise_definition' if built_in else
                          'caller_acknowledged_assumption_not_proved')
    if verify_cell_independence and n_cells >= 2 and not built_in:
        base_out = np.asarray(classifier_fn(x_int, genes_list), dtype=object)
        if base_out.shape != (n_cells,):
            raise ValueError('classifier_fn must return one label per cell')
        rev_out = np.asarray(classifier_fn(x_int[::-1], genes_list), dtype=object)[::-1]
        single_outputs = [np.asarray(classifier_fn(x_int[i:i+1], genes_list), dtype=object)
                          for i in range(n_cells)]
        if (
            not np.array_equal(base_out, rev_out)
            or any(v.shape != (1,) or v[0] != base_out[i]
                   for i, v in enumerate(single_outputs))
        ):
            raise ValueError(
                'classifier_fn failed cell-independence check (order or cohort-subset invariance); '
                'classifiers that refit, normalize, or smooth across cells violate independent-node assumptions'
            )
    else:
        base_out = np.asarray(classifier_fn(x_int, genes_list), dtype=object)
        if base_out.shape != (n_cells,):
            raise ValueError('classifier_fn must return one label per cell')

    baseline_labels = base_out.copy()
    baseline_labels[~mask] = 'LOW_DETECTION_BASELINE'

    n_classes = len(classes_tuple)
    class_to_idx = {c: i for i, c in enumerate(classes_tuple)}

    if np.all(retention == 1.0):
        probs = np.zeros((n_cells, n_classes), dtype=float)
        for i, lab in enumerate(base_out):
            if mask[i] and lab in class_to_idx:
                probs[i, class_to_idx[lab]] = 1.0
        return dict(
            probabilities=probs,
            classes=classes_tuple,
            baseline_labels=baseline_labels,
            n_draws=0,
            seed=int(seed),
            semantics='frozen_cellwise_classifier_exact_p1',
            probability_method='deterministic',
            upstream_probability_uncertainty='None; deterministic p=1 classification',
            independent_nodes=True,
            independence_basis=independence_basis,
        )

    if np.all(retention == 0.0):
        zero_out = np.asarray(classifier_fn(np.zeros_like(x_int), genes_list), dtype=object)
        probs = np.zeros((n_cells, n_classes), dtype=float)
        for i, lab in enumerate(zero_out):
            if mask[i] and lab in class_to_idx:
                probs[i, class_to_idx[lab]] = 1.0
        return dict(
            probabilities=probs,
            classes=classes_tuple,
            baseline_labels=baseline_labels,
            n_draws=0,
            seed=int(seed),
            semantics='frozen_cellwise_classifier_exact_p0',
            probability_method='deterministic',
            upstream_probability_uncertainty='None; deterministic p=0 classification',
            independent_nodes=True,
            independence_basis=independence_basis,
        )

    rng = np.random.default_rng(int(seed))
    counts_accum = np.zeros((n_cells, n_classes), dtype=np.int64)
    eligible_idx = np.flatnonzero(mask)
    if len(eligible_idx) > 0:
        x_elig = x_int[eligible_idx]
        for start in range(0, int(n_draws), max(1, int(batch_draws))):
            bsize = min(int(batch_draws), int(n_draws) - start)
            # Draw (bsize, N_elig, G) thinned counts and evaluate frozen classifier
            thinned = rng.binomial(
                x_elig[None, :, :],
                retention[None, None, :],
                size=(bsize, len(eligible_idx), len(genes_list)),
            )
            flat = thinned.reshape(bsize * len(eligible_idx), len(genes_list))
            pred = np.asarray(classifier_fn(flat, genes_list), dtype=object).reshape(bsize, len(eligible_idx))
            for c_name, c_i in class_to_idx.items():
                counts_accum[eligible_idx, c_i] += np.sum(pred == c_name, axis=0)

    probs = counts_accum.astype(float) / float(n_draws)
    return dict(
        probabilities=probs,
        classes=classes_tuple,
        baseline_labels=baseline_labels,
        n_draws=int(n_draws),
        seed=int(seed),
        semantics='frozen_cellwise_classifier_thinning_mc',
        probability_method='monte_carlo',
        independent_nodes=True,
        independence_basis=independence_basis,
        upstream_probability_uncertainty='Monte Carlo; not included in downstream conditional numerical enclosure',
    )
