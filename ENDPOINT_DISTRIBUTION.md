# Complete endpoint perturbation distributions

The `endpoint_distribution` module adds two whole-state routes to the earlier observed-value bridge. It implements every endpoint listed below under the declared graph/geometry contracts. It does not infer a distribution by plugging an expected O/E into another statistic.

|Endpoint|Per-state definition and domain|
|---|---|
|O, E, R|Cross-class undirected count O; E=2M nA nB/[N(N−1)]; R=O/E only when E>0|
|z_node|Fixed-class-size node-permutation population mean/variance standardization; zero variance explicitly undefined|
|z_node_mc|Actual finite node-null counts supplied or generated; ddof=0 variance; finite, ±infinity and 0/0 distinct|
|giotto_enrichment|log2[(O+1)/(dA dB/(2M−1)+1)] under the verified Giotto endpoint-label-shuffle null; requires endpoint support|
|giotto_enrichment_mc|The same score using the saved finite-permutation mean; this is enrichment, not a z-score|
|co_occurrence|2M O/(dA dB); unsupported endpoint margins are undefined, not zero|
|G_a_to_b, G_b_to_a|Fractions of source-type nodes with an opposite-type neighbour within the declared radius; no edge correction; both types required|
|K_none, K_translation|Area×cross-weight sum/(nA nB), with weights 1 or area/rectangular overlap; both types and verified window required|
|L_none, L_translation|sqrt(K/pi), recomputed separately on every state|
|L_minus_r_none, L_minus_r_translation|L−r, signed; CV deliberately not reported|

The graph must be fixed, binary, undirected, loop-free and confined to one region. All scales share the same node IDs and order. A declared saved radius graph supports G; K/L additionally require aligned um coordinates and an explicit rectangular window with provenance, not a window fabricated from point extrema. Analysis windows need not be acquisition boundaries. Other corrections, directional/weighted graph nulls, population-independence tests, continuous-composition interpretations and PANORAMIC cohort estimands are not implemented by this module. RCTD continuous fields remain on the separate validated field path; no implicit argmax conversion occurs.

## Exact finite states

`exact_endpoint_distribution(probabilities, scales=..., classes=..., pair=..., independent_nodes=True, provenance=..., max_states=100000, null_permutations=None)` enumerates all positive-mass independent categorical label states. Inputs must sum exactly to one as rationals. Decimal strings or `Fraction` values make the intended rational model explicit; binary floats retain their exact binary values. No clipping, renormalization, positive-mass pruning or silent empirical fallback occurs. Classifier posterior marginals do not establish node independence.

If T contains all endpoints and radii, the exact pushforward is

`P(T=t)=sum_y P(Y=y) 1[T(y)=t]`.

Optional finite-permutation propagation expands this to `sum_y,u P(y)P(u|y)1[T(y,u)=t]`, with conditionally independent uniform permutation streams across tools and scales. It describes the declared ideal iid permutation model, not all possible seeds of a particular pseudorandom generator. The node null uses exhaustive fixed-size assignments; the edge-endpoint null uses a recursive perfect-matching count law. Finite sample sums/squared sums are convolved before the score is evaluated. State caps stop computation without returning a truncated, renormalized distribution. This route is exponential and intended for eligible small states.

The optional nested exact route additionally requires at most 30 edge endpoints (15 undirected edges) per scale. Fixed-size label assignments, convolution states and joint branches are also bounded by `max_states`. These extra limits do not apply to the canonical outer-state endpoint enumeration or the empirical route. Exceeding any limit raises `StateBudgetExceeded`.

Probability masses and rational endpoints are exact for the supplied model. Square roots/logarithms have symbolic keys plus floating evaluations; reported numeric moments are not interval certified. NumPy integers are converted to Python arbitrary-precision integers before rational operations. The old direct-CV numerical certificate is not transferred to these new empirical endpoints.

## Complete empirical realizations

`empirical_endpoint_distribution(outputs=..., failed=..., not_run=..., planned_ids=..., scales=..., classes=..., pair=..., provenance=..., null_permutations=32, seed=...)` accepts one complete, ID-aligned hard-label vector per scheduled outer realization. Each record carries `node_ids`, `classes`, `values`, `semantics='hard_labels'` and `provenance`.

Outputs, failures and unstarted IDs must partition the schedule. Invalid supplied records retain their own status. Every planned record has mass `1/B`, including failed/unstarted states, so this is an execution-augmented empirical measure. The baseline is evaluated separately. No complete-case filtering across endpoints is applied. Whole-output rows preserve dependence induced by classifier refitting; they do not require independent nodes.

One set of finite-null counts is generated for each outer ID/tool/scale using an explicit seed derived from the group seed, outer ID and scale. Those counts and seeds are saved. This is a NumPy implementation of the named null laws, not a claim of bitwise native R/Python RNG identity. Thirty-two inner permutations are not 32 new outer observations. A single finite-permutation score per outer vector samples algorithm randomness; it does not estimate the entire conditional score distribution over seeds at that vector. Large-data population distributions remain empirical approximations.

## Outputs and interpretation

- Complete joint atoms or outer records, with exact rational mass strings and explicit status atoms.
- Marginal PMFs; finite sub-CDFs and finite-conditional CDFs; 0/2.5/50/97.5/100% empirical quantiles. Undefined states are unordered and infinite states are listed separately.
- Conditional finite-domain means, variances and SDs; CV only for nonsigned endpoints with a positive finite mean. Variance describes the declared measure (`ddof=0` in empirical mode), distinct from the historical sample-SD CV (`ddof=1`). Quantile ranges are not confidence intervals.
- Pairwise covariance on each pair's common finite domain, with its mass. Different domains mean the assembled covariance table need not form a positive-semidefinite matrix. Full joint rows permit explicitly chosen common-domain calculations if needed.
- No automatic confidence bands, biological replication, calibrated significance decision or rare-event accuracy claim.

For a fixed positive graph/window multiplier and the same finite domain, K_none=area×2M/[N(N−1)]×R. Translation correction, L, degree-normalized co-occurrence and permutation scores require their own per-state evaluation. Low availability and observed finite zeros cannot establish a biological null or exclude unobserved positive tails.

## Executed checks and current data scope

The 2026-10-26 record is `review/endpoint_distribution_2026-10-26`. Thirty-six relevant tests passed. An independent enumeration of 576 molecular states matched the complete joint pushforward of 81 label states. An exact nested two-permutation example retained 95 distinct joint atoms, including infinite/0-over-0 z states. Actual installed Squidpy 1.6.5, Giotto 4.2.3 and spatstat fixtures passed 228 comparisons on three states; this is engineering evidence.

Twenty real-input groups yielded 576 endpoint/scale/group laws and 104,064 perturbation endpoint records. Lung/kidney used the same archived 256 molecular draws per source, two QC policies, two pairs and three radii. Twelve native-label groups reused 30 archived outputs each. Six FOVs lacked original geometry; all 2,160 affected K/L records retained `GEOMETRY_NOT_SUPPLIED`. Current continuous RCTD outputs are not silently recoded. See the report for complete provenance, original failures, resource receipts, independent-context review and the limits of empirical tails.
