# ADR 0009: Painted component UVs and experimental source capacity

Date: 2026-11-16

Status: Local experimental integration for the V7 Monster Hunter study corpus.
Base: `1f41187e71d2776fea75caf3d1765c62924b90cb`.

## Context

The user authorized a 16,000,000-byte cap for each UTF-8 Block-DSL file. This is
not a size limit on PNGs, exports, videos, archives, or other files. The upstream
V4 assembly implementation is now native and supersedes the earlier external
experimental assembly lowering. Its component box template did not carry V3
painted `texture`, `uv`, or `uv_faces` declarations. The corpus uses explicit
RGBA atlases to shape planes, so losing or rejecting these declarations prevents
source-based reconstruction.

The corpus painter previously used world-axis chart coordinates and repaired
orientation after export. This made the exported files correct but left source
UV declarations dependent on a helper convention. The V7 authoring boundary now
emits canonical native face UVs and the dataset no longer edits native exports.

## Decision

1. Keep upstream `assembly_ir.py`, frame-mating equations, runtime-bone synthesis,
   source maps, and assembly syntax. Do not add the old `assembly.py` frontend.
2. Raise only the lexer source and lexical-token limits to 16,000,000 UTF-8 bytes
   and 2,000,000 tokens. The synthetic EOF token is excluded. Other structural
   guards remain, including 4,096 declarations and 64-level native rig depth.
3. Carry typed `texture`, `uv`, and `uv_faces` through component parsing, native
   root and child-bone lowering, mirroring, and canonical serialization. Reuse
   the direct-rig UV reflection convention; retain existing validation.
4. Treat native per-face UVs as the Blockbench outward-face convention. For
   Bedrock/Gecko JSON, reverse both rectangle axes on `up` and `down`. This is
   the inverse of Blockbench's Bedrock importer and matches its exporter.
5. Canonicalize painter coordinates in the external authoring emitter before
   parsing and before passing atlas metadata to exporters. `render_scene.json`
   explicitly keeps its existing painter convention for previews. This is not
   an export mutation or a geometry substitution.

## Verification

Focused tests cover the exact UTF-8 byte boundary, token/EOF boundary, root and
nested component UV fields, reflected UVs, rejection, source round trips, and
an asymmetric two-sided cutout under reflection. The cutout oracle reconstructs
image points into geometry independently from Blockbench's `CubeFace.UVToLocal`
and Bedrock `parseCube` conventions at commit
`ec0b2827104d0799c153bfb5f867579c0c44652f`.

Existing native V4 assembly and selected export/SE(3) tests are also run. A full
CPU-suite pass, GPU result, Blockbench GUI import, and Minecraft runtime import
are not claimed. Exact command logs and environment are recorded separately.

## Consequences

Painted components can be reproduced through native V4 source. Existing baked
atlases retain their appearance because orientation is stated at the authoring
boundary. The Bedrock top/bottom conversion changes previously incorrect raw
per-face orientation; it does not change geometry or animation transforms.
The 16 MB allowance does not guarantee that arbitrary input is inexpensive or
that all geometry guards are lifted. Alpha-plane support boxes remain different
from visible polygons for collision auditing.
