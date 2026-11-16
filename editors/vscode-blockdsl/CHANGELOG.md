# Changelog

## 0.4.0 — 2026-11-15
- Full **SyncCraft Block-DSL v4.0** (`ADR 0008`) language support alongside 100% backward compatibility for v3.0–v3.2 `@rig` files:
  - Top-down `@layout` parameter DAG (`param`, `let`, `budget`, `envelope`), reusable `@component` definitions (`material_slots`, `uv_policy: unique | shared`, oriented 6-DOF `frame` sockets), and `@assembly` / `subassembly` composition (`instance ... mirror_x(...)`, `connect ... to ... { type: fixed | revolute | prismatic }`, `bind`, `slide` prismatic motion drives).
  - Pure-JS v4.0 symbol indexer expanding `@component` instances, `mirror_x` twin instances, and dotted `<instance>.<bone>` / `<instance>.<box>` references for completions, hover cards, and zero-false-positive fast diagnostics.
  - Added `scaffold-assembly` and `@component` snippets.

## 0.3.2 — 2026-11-15
- Initial release supporting **SyncCraft Block-DSL v3.2** (`ADR 0001`–`ADR 0007`).
- Complete TextMate grammar (`source.blockdsl`) + Markdown fenced-block injection (`blockdsl`, `dsl`).
- Pure-JS symbol indexer expanding `mirror_x`, `mirror`, `radial`, `def`/`use`, and `chain` identifiers for context-aware auto-completion, hover cards, and go-to-definition.
- Instant keystroke diagnostics plus optional `.venv` SyncCraft Python bridge for SE(3) + 15-axis SAT (`strict` & `sculpted`), 12-frame swept `@behavior` clip audits, and canonical `format_program` document formatting.
