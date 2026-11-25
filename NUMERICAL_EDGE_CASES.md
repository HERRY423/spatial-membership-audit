# Exact constant ratios and near-zero CV

For distinct categorical types A and B, let D=n_A n_B and c=2M/[N(N-1)].
When every node pair capable of carrying A and B is an edge, O=D pointwise.
Thus R=O/(cD)=1/c conditional on D>0, even when D itself is random.
Computing its delta variance by subtracting three floating-point terms can
otherwise create a small spurious positive variance.

Both analytic implementations now detect this condition using integer counts
of strictly positive support, including both orientations for pairs that can
carry either class. No probability tolerance is used. When only two nodes have
positive A/B support, D=1 conditional on D>0; their conditional moments are
also computed directly. If those nodes are unconnected, O=0 and CV remains
undefined, as before.

The established output keys and status/warning contract are retained. General
nonconstant ratios still use the delta approximation. This change is not a
general relative-error guarantee for arbitrarily tiny nonzero variances.

Tests include the originally failing probabilities, a complete graph with
variable D, and rare nonconstant cases with probabilities down to 1e-12.
Original failures and source snapshots are retained in
`review/rctd_preflight_cvfix_2026-10-19/before/`; current verification and the
RCTD preflight protocol are in its parent directory.
