# ADR 0001: implement a bounded static slice first

Status: accepted for starter release 0.1.0. Date: 2026-11-13.

## Context

The user requested a start pack for a local coding agent, based on a full research
and engineering plan. A useful handoff needs executable foundations and precise
remaining work. GPU training, multiple installed game runtimes, and missing parts
of the PDF are not available as completed evidence in this task.

## Decision

Implement static box DSL, named hierarchy, attachment/placement, mirror expansion,
NumPy geometry, SAT diagnostics, static exporters, a flat atlas and CPU preview.
Use a bounded recursive parser and dataclasses so the default core needs only
NumPy and Pillow. Keep a reference Lark grammar for parity and future migration.
Lark/Pydantic/XGrammar are production implementation options in the plan, not
required imports for this minimal bootstrap.

Reject extrusion, `@shade`, `@ergonomics`, and `@motion` until implemented. Reject
ambiguous unplaced second boxes, nonzero `anchor` without a `base_box`, and nested
mirrors. These are conservative starter restrictions, not final language claims.

Provide per-face integer UVs using flat base color. This is a diagnostic texture,
not the specified six-face Box UV layout or full palette/shading engine. Retain
palette intent in AST/export metadata and emit an explicit limitation notice.

## Consequences

The package runs immediately on CPU, supplies reusable numerical oracles, and
cannot silently claim unsupported features. The report's full example is a later
acceptance input and does not parse under the deliberately smaller runtime subset.

The local agent must close the status gaps before Phase 1 completion. This ADR
does not authorize weakening any production acceptance gate. Create a new ADR to
replace a provisional semantic decision once the full implementation is ready.
