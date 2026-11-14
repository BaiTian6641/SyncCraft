# ADR 0002: separate diagnostic results from release and research acceptance

Status: accepted for starter release 0.1.0. Date: 2026-11-13.

## Decision

The starter audit reports only its implemented checks, gives every contact
exemption, and sets `complete_for_release=false`. Its `passed` result cannot be
used as the denominator label for Gate G2. Actual contact regions, hand/display
constraints, UV guarantees, serialization checks, and runtime evidence must exist
before the full static audit is declared implemented.

The collision oracle uses direct minimum overlap on valid SAT axes. The smooth
surrogate remains separate and is not an intersection-volume measure. Small
positive intersections must not become valid simply because the smooth loss is zero.

Experimental cost is a vector: accepted, discarded and repair tokens; projection
steps; compiler calls; initial audits; and re-audits. A derived scalar can be added
only with preregistered coefficients and measurements. Fixed operation counts alone
do not establish identical physical compute across model sizes or algorithms.

G2's calculator stays blocked without complete evidence fields, real counts, and
dataset/evaluation manifest hashes. It validates the decision arithmetic; it does
not independently prove that a supplied evidence flag or manifest is truthful.

## Consequences

Starter tests and outputs are useful engineering evidence but cannot be promoted
into paper results. Failed baselines, missing hardware runs, and deferred features
remain visible. G2 requires a complete implementation and a preregistered policy.
