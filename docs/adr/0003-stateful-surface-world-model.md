# ADR 0003: stateful surface-material world model and multi-target pixel-PBR compiler

Status: accepted for P1-03a..e implementation. Date: 2026-11-14.

## Context

SyncCraft's geometry path uses typed `Block-DSL` ASTs, shared right-handed `SE(3)`
forward kinematics, and 15-axis `SAT` + `P1-02` physical contact verification rather
than neural vertex hallucination. Extending texture and animation under the same
mechanistic world-model principle requires stateful material programs whose outputs
respond causally to physical interventions (repacking UVs, reposing joints, changing
illumination, applying contact rubbing, altering environmental exposure, and scaling
output texel resolution) rather than 2D pixel-diffusion memorization.

Single-image internet assets cannot uniquely identify intrinsic roughness, lighting,
weathering history, or reaction-diffusion parameters; treating fitted candidates as
exact ground truth or hard-coding unverified Gate G2 prerequisites would violate
`ADR 0002` and `AGENTS.md`.

## Decision

### 1. Four separated surface representations (`P1-03a`)

Separate surface authoring, state, optics, and target export outside `uv_packer.py`
(whose sole responsibility remains integer UV island allocation):

- **`MaterialProgram`**: substrate family, coating layers, 3D solid grain/strata axes,
  reaction-diffusion / colonization parameters, stamps/overrides, and deterministic seeds.
- **`PersistentSurfaceState`**: rest-space fields (`moisture`, `oxidation`,
  `coating_damage`, `wear`, `biofilm`, `rd_u`, `rd_v`) evolved over time and explicitly
  remapped or reset when geometry changes.
- **`OpticalFields`**: linear-space `base_color`, micro-height `height`, tangent-space
  `normal`, `perceptual_roughness` ($\alpha_p \in [0, 1]$, microfacet $\alpha = \alpha_p^2$),
  `metalness` ($m \in [0, 1]$), distinct dielectric `f0` ($((n - 1)/(n + 1))^2$),
  `porosity`, `subsurface`, `opacity`, `emissive`, posed visibility `cavity_ao`, and
  intrinsic thin-film parameters (`iridescence_thickness_nm`, `iridescence_ior`).
- **`StyleConfig`**: OKLCH palette ramp budgets, pixel-stamp overrides, neutral studio
  vs. world lighting settings, reference view angle for static thin-film bakes, and
  target profile (`vanilla`, `labpbr_1_3`, `gltf_2_0`).

Expose compact optional `@materials` and `@style` sections in `Block-DSL` while keeping
spatial persistent fields in `WorldState`.

### 2. Canonical rest-space `SurfaceDomain` and three distinct graphs (`P1-03a`)

Discretize each cuboid face in rest coordinates at solver resolution
`subdivisions_per_unit` (default `4`, $\Delta x = 0.25$ model units), decoupled from
output `texel_density`. Maintain three strictly separate topological relations:

1. **Surface continuity (`manifold_seams`)**: connects cell-centered samples across a
   cuboid's 12 folded edges ($\text{geodesic step} = \Delta x$) and across flush
   cross-box patches *only* when both boxes share the same material and explicit
   `weld_group`. Distinct materials in contact (e.g., metal harness on skin) never
   share manifold edges.
2. **Posed physical contact (`posed_contacts`)**: evaluated in posed world coordinates
   from `SE(3)` transforms to measure OBB proximity, posed `cavity_ao` visibility, and
   sliding contact regions.
3. **Mirror correspondence (`symmetry_pairs`)**: links `mirror_x` twin primitives for
   `symmetry_mode="mirrored"` coordinate reflection while keeping left/right subtrees
   disconnected in `manifold_seams` (`symmetry_mode="independent_seed"` by default for
   organic reaction-diffusion).

### 3. Honest fidelity labels across the material-process catalog (`P1-03b`)

Support a composable catalog covering `painted_metal`, `brushed_metal`,
`polymer_dielectric`, `woven_fabric`, `solid_wood`, `weathered_stone`,
`keratin_scales`, `patterned_skin` (anisotropic Gray-Scott reaction-diffusion),
`lichen_biofilm`, and `thin_film_iridescence`:

- Label simplified moisture, Archard sliding wear ($\Delta w = k \cdot L_n \cdot \|\Delta \mathbf{s}_\parallel\|$),
  and oxidation rules explicitly as `reduced_order_model`.
- Label Gray-Scott reaction-diffusion as `supported_pattern_mechanism`, never as the
  proven biological cause of arbitrary animal markings.
- Enforce explicit CFL stability bounds, constant-field preservation, and zero-flux /
  closed-manifold mass conservation on diffusion operators.

### 4. Linear-space optics and separated target exporters (`P1-03c`)

Author palettes in perceptual OKLCH, execute all BRDF / optical math in linear color
space, preserve sharp 90-degree geometric cuboid normals while computing seam-consistent
tangent micro-normals from `height`, and maintain separate exporters for:

1. **Vanilla stylized bake** (`.bbmodel` + Java `1.21.11+`, evaluating thin-film at a
   documented reference angle $\theta_{\text{ref}}$ because a single static RGBA PNG
   cannot encode view-dependent angle shifts).
2. **Minecraft LabPBR 1.3** (`atlas.png`, `atlas_n.png`, `atlas_s.png` following LabPBR
   channel/metal-code/porosity-SSS/emissive conventions; generated PNGs are serialization
   outputs until verified in a pinned Minecraft shader runtime).
3. **glTF 2.0 PBR bundle** (baseColor, OpenGL normal, `ORM` metallic-roughness-occlusion,
   emissive, and `KHR_materials_iridescence` metadata).

### 5. Shared state transitions and inverse inference with uncertainty (`P1-03d`, `P1-03e`)

- Model world evolution as $S_{t+1} = F(S_t, a_t, E_t, \Delta t)$ over typed pose,
  trajectory, 2-bone IK, contact-rubbing, material-edit, UV-repack, and resolution
  interventions, using sub-stepped swept SAT checks between poses. Keep `@motion` text
  DSL syntax deferred until its clipped source grammar is resolved in a dedicated ADR.
- Treat internet asset ingestion as **inverse material inference** returning top-$K$
  candidate `MaterialProgram` hypotheses with OKLab/spectral reconstruction residuals,
  identifiability scores, and extracted pixel-stamp overrides—separate from Synthetic
  Gold trajectories.
- Compute Gate G2 prerequisite flags dynamically from actual implemented subsystem
  capabilities rather than static `True` literals.

## Consequences

`uv_packer.py` stays focused on integer island packing; repacking UVs, reposing limbs,
changing lighting, or changing output resolution preserves underlying physical material
fields within measured discretization bounds; and world-model training can compare
program-only SFT against `Infer` / `Predict` / `Plan-Repair` intervention supervision
under matched solver budgets.
