"""Core DAG Engine & AI-Workflow-Friendly Declarative Recipe Format for `synccraft.texgraph`.

Supports:
- Fluent Python DAG authoring (`TexGraph`)
- Lossless declarative JSON `.texgraph.json` recipe import/export (`to_recipe` / `from_recipe`)
- Topological evaluation with cross-frame static-subgraph memoization for fast flipbook rendering
- Direct 3D `.bbmodel` mesh baking (`MeshBakeContext`) & embedded PNG application (`apply_to_bbmodel`)
"""
from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from .engine_ops import (
    auto_levels,
    curvature_from_height,
    directional_selout,
    displace_uv,
    edge_wear_mask,
    flood_fill,
    histogram_scan,
    histogram_select,
    normal_light_2d,
    sdf_boolean,
    sdf_shape,
    slope_blur,
    tile_sampler,
)
from .generators_bio_materials import (
    aquatic_scales_skin,
    bark_and_timber,
    caustic_water_surface,
    chitin_carapace,
    cobblestone_masonry,
    crystalline_ore_vein,
    elemental_golem_surface,
    layered_plumage,
    organic_fur_pelt,
)
from .generators_character_materials import (
    forged_battle_metal,
    keratin_horn_bone,
    orc_flesh_skin,
    paint_entity_eyes_and_parts,
    stitched_leather,
    woven_battle_cloth,
)
from .pixel_composer_ops import (
    bake_emissive_bounce,
    chromatic_aberration,
    color_replace,
    de_corner,
    de_stray,
    heightmap_shadow,
    level_selector,
    palette_posterize,
    pixel_bevel_directional,
    pixel_clusters,
    pixel_dither,
    pixel_mirror,
    pixel_outline,
    pixel_sort,
    vignette,
)
from .seam_and_pbr_ops import (
    block_edge_transition,
    precompute_shadow_pbr,
    triplanar_project,
    uv_seam_heal,
)
from .region_ops import (
    region_base_fill,
    region_bevel_and_ao,
    region_composite,
    region_restrained_noise,
    uv_island_coords,
)
from .generators_gradients import (
    body_flow_gradient,
    cosine_gradient,
    gradient_field,
    multi_stop_gradient,
    per_island_gradient,
    spline_path_gradient,
)
from .generators_noise import (
    bayer_matrix,
    domain_warp,
    fbm_noise,
    gabor_noise,
    noise_3d_field,
    perlin_noise,
    uv_grid,
    value_noise,
    voronoi_noise,
)
from .generators_noise_ext import (
    blue_noise,
    bubble_noise,
    caustic_noise,
    crystal_noise,
    curl_flow_noise,
    fold_marble_noise,
    grunge_blotch,
    plasma_wave,
    scratch_noise,
    simplex_noise,
    starfield_sparkle,
    strand_noise,
    white_noise,
)
from .generators_patterns_ext import (
    checker_tartan,
    circuit_runes,
    clockwork_gears,
    crosshatch_engrave,
    crystal_facets,
    maze_labyrinth,
    panel_frame_trim,
    polka_rivets,
    polygon_star,
    scallop_shingles,
    stripes_pattern,
    weave_chainmail,
)
from .mesh_mask_ops import (
    mask_3d_distance,
    mask_combine,
    mask_smart_crevice_dirt,
    mask_smart_edge_wear,
    mask_uv_box,
    mask_uv_part,
    mask_world_direction,
    mask_world_position,
)
from .pbr_materials_library import (
    anim_flicker_pulse,
    anim_flow_warp,
    anim_shimmer,
    mat_brass,
    mat_bronze_copper,
    mat_ceramic_porcelain,
    mat_chrome_titanium,
    mat_crystal_gem,
    mat_dragon_scales_pbr,
    mat_glass,
    mat_glowing_runes_pbr,
    mat_gold_precious,
    mat_ice_frost,
    mat_marble_veined,
    mat_obsidian_volcanic,
    mat_painted_enamel,
    mat_plastic,
    mat_rubber_grip,
    mat_steel_iron,
    mat_velvet_silk,
    material_region_apply,
    material_split,
    material_tweak,
    material_weathering,
    normal_bevel,
    normal_from_height_multi,
    normal_strength_op,
    normal_to_curvature,
)
from .generators_painterly import (
    anatomical_muscle_scales,
    colossal_dragon_membrane,
    directional_brushstrokes,
    gradient_eye_paint,
    layered_scale_plates,
    organic_domain_warp_3d,
    painterly_brush_field,
    sculpted_pinion_wing,
    smooth_wing_membrane,
    stylized_foliage_clusters,
    stylized_wood_planks,
    temple_carved_stone,
)
from .generators_patterns import (
    brick_pattern,
    diamond_chevron_pattern,
    energy_veins_pattern,
    feather_barbs_pattern,
    fur_tufts_pattern,
    grid_pattern,
    hex_grid_pattern,
    imbricated_scales_pattern,
    mosaic_pattern,
    radial_rings_pattern,
    tri_grid_pattern,
    truchet_pattern,
)
from .mesh_baker import MeshBakeContext, bake_bbmodel_maps
from .ops_fx_color import (
    ao_from_height,
    bloom_glow,
    color_balance,
    curves_remap,
    directional_blur,
    gaussian_blur,
    morph_op,
    normal_blend_rnm,
    oklch_adjust,
    polar_warp,
    posterize_soft,
    sdf_from_mask,
    sobel_edges,
    tile_transform,
    twirl_distort,
    unsharp_mask,
    wave_distort,
)
from .pbr import (
    bake_pbr_to_albedo,
    export_baked_pbr_package,
    export_pbr_bundle,
    height_to_normal_map,
    pack_labpbr_1_3,
    render_pbr_material_sheet,
    render_pbr_preview_2d,
)
from .sim_animate import (
    eval_anim_driver,
    feedback_step_sequence,
    fluid_smoke_sim_field,
    iterate_inline,
    particle_sim_field,
)
from .pipeline import (
    MATERIAL_BUS_PORTS,
    SUBGRAPH_REGISTRY,
    MultiPassResult,
    PassSnapshot,
    PassSpec,
    is_material_bus,
    material_blend,
    material_grade,
    material_height_blend,
    material_make,
    material_pad_islands,
    material_set_alpha,
    render_multipass_sheet,
)
from .pixel_art_ops import (
    blend_layers,
    directional_shade,
    emissive_extract,
    encode_png,
    export_flipbook,
    hex_to_srgb,
    hue_shifted_ramp,
    levels_remap,
    paint_creature_eye,
    palette_quantize,
    pixel_bevel_emboss,
    rgba_float_to_u8,
    selout_outline,
    vibrance_grade,
    wing_scallop_alpha,
)
from .simulation_filters import (
    anisotropic_kuwahara,
    bilateral_denoise,
    cellular_automata,
    gray_scott_turing,
    iterate_process,
    morphological_despeckle,
    non_local_means_denoise,
    pixel_cluster_mode,
    total_variation_denoise,
    weathering_drip,
)
from .math_utility_ops import (
    checker_basic,
    clamp_value,
    color_constant,
    float_constant,
    fresnel_weight,
    invert_field,
    map_range,
    math_scalar,
    mix_blend,
    paint_mask,
    rgb_combine,
    rgb_separate,
    tex_coord,
    uv_mapping,
    vector_math,
)
from .v16_pro_ops import ScriptSecurityError, paint_layer_stack_op, script_expr_op

# Nodes whose output depends on animation `phase` (re-evaluated per flipbook frame; others cached!)
_TIME_AWARE_OPS: frozenset[str] = frozenset(
    {
        "perlin", "fbm", "voronoi", "domain_warp", "noise_3d",
        "radial_rings", "energy_veins", "scroll", "pulse", "sprite_mask",
        "cosine_gradient", "wave_distort", "caustic_water",
        "iterate_inline", "feedback_loop", "particle_sim", "fluid_smoke_sim", "anim_driver",
        "simplex_noise", "bubble_noise", "caustic_noise", "curl_flow_noise",
        "starfield_sparkle", "plasma_wave", "clockwork_gears",
        "mat_crystal_gem", "mat_obsidian_volcanic", "mat_glowing_runes_pbr",
        "anim_shimmer", "anim_flicker_pulse", "anim_flow_warp",
        "subgraph_group", "script_expr",
    }
)


# Ops whose primary input expects a full MATERIAL bus dict instead of a stripped albedo array
_BUS_INPUT_KEYS: dict[str, frozenset[str]] = {
    "material_blend": frozenset({"base", "top"}),
    "material_height_blend": frozenset({"base", "top"}),
    "material_grade": frozenset({"mat"}),
    "material_set_alpha": frozenset({"mat"}),
    "material_pad_islands": frozenset({"mat"}),
    "material_split": frozenset({"mat"}),
    "material_tweak": frozenset({"mat"}),
    "material_region_apply": frozenset({"base", "top"}),
    "material_weathering": frozenset({"mat"}),
    "anim_shimmer": frozenset({"src"}),
    "anim_flicker_pulse": frozenset({"src"}),
    "anim_flow_warp": frozenset({"src"}),
    "reroute": frozenset({"in"}),
    "subgraph_group": frozenset({"in_0", "in_1", "in_2", "in_3", "in_4", "in_5"}),
    "paint_layer_stack": frozenset({"base"}),
    "paint_entity_eyes_and_parts": frozenset({"base"}),
    "bake_emissive_bounce": frozenset({"mat"}),
    "triplanar_project": frozenset({"src"}),
    "uv_seam_heal": frozenset({"src"}),
    "block_edge_transition": frozenset({"base"}),
    "precompute_shadow_pbr": frozenset({"mat"}),
    "region_restrained_noise": frozenset({"mat"}),
    "region_bevel_and_ao": frozenset({"mat"}),
    "region_composite": frozenset({"base", "layer"}),
}


def _parse_ref(ref: str) -> tuple[str, str | None]:
    """Split a typed multi-port socket reference ``'node_id.port'`` into ``(node_id, port)``."""
    if "." in ref:
        base, port = ref.split(".", 1)
        return base, port
    return ref, None


def _resolve_ref(vals: dict[str, Any], ref: str, *, allow_bus: bool = False) -> Any:
    """Resolve a node or ``'node_id.port'`` socket reference from evaluated DAG values."""
    base, port = _parse_ref(ref)
    v = vals[base]
    if port is not None:
        if isinstance(v, dict):
            if port in v:
                return v[port]
            if port == "rgba" and "albedo" in v:
                return v["albedo"]
            if port == "albedo" and "rgba" in v:
                return v["rgba"]
            avail = [k for k in v if not k.startswith("__")]
            raise KeyError(f"Node {base!r} has no output port {port!r} (available: {avail})")
        raise TypeError(f"Node {base!r} returned a single array and cannot resolve port {port!r}")
    if isinstance(v, dict) and not allow_bus:
        for primary in ("albedo", "rgba", "height"):
            if primary in v:
                return v[primary]
        for k, arr in v.items():
            if not k.startswith("__"):
                return arr
    return v


@dataclass
class NodeSpec:
    id: str
    op: str
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # GUI canvas metadata (TexGraph v10 — serialized cleanly only when non-default!)
    pos: tuple[float, float] | None = None
    label: str = ""
    collapsed: bool = False
    bypassed: bool = False
    preview_pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "op": self.op,
            "inputs": dict(self.inputs),
            "params": copy.deepcopy(self.params),
        }
        if self.pos is not None:
            d["pos"] = [round(float(self.pos[0]), 1), round(float(self.pos[1]), 1)]
        if self.label:
            d["label"] = self.label
        if self.collapsed:
            d["collapsed"] = True
        if self.bypassed:
            d["bypassed"] = True
        if self.preview_pinned:
            d["preview_pinned"] = True
        return d


class GraphCycleError(ValueError):
    """Raised when a TexGraph DAG contains a directed cycle."""


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    topo_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "node_count": len(self.topo_order),
        }


@dataclass
class EvalResult:
    output: np.ndarray
    emissive: np.ndarray | None
    intermediates: dict[str, np.ndarray]
    timings_ms: dict[str, float]
    pbr_channels: dict[str, np.ndarray] = field(default_factory=dict)
    pbr_maps: dict[str, np.ndarray] | None = None
    pbr_preview: np.ndarray | None = None
    cache_hits: int = 0
    cache_misses: int = 0


class TexGraph:
    """Deterministic, AI-friendly Parametric Pixel Composer & Full PBR DAG."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        tileable: bool = False,
        seed: int = 42,
        name: str = "untitled_texgraph",
        mesh_context: MeshBakeContext | None = None,
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.tileable = bool(tileable)
        self.seed = int(seed)
        self.name = str(name)
        self.mesh_context = mesh_context
        self.nodes: dict[str, NodeSpec] = {}
        self.node_order: list[str] = []
        self.output_node: str | None = None
        self.emissive_node: str | None = None
        self.output_material_node: str | None = None
        # First-class PBR channel outputs (height, normal, roughness, metallic, ao, sss, porosity)
        self.pbr_nodes: dict[str, str] = {}
        self.pbr_config: dict[str, Any] = {"normal_strength": 2.4, "dielectric_f0": 0.04}
        # Staged Multi-Pass Pipeline (Pass 1 Bake -> Pass 2 Macro -> Pass 3 Relief -> Pass 4 Grade -> Pass 5 Final)
        self.passes: list[PassSpec] = []
        self._active_pass: PassSpec | None = None
        # Future-UI-ready graph metadata: exposed parameters, procedural drivers & visual node groups/frames
        self.exposed_params: dict[str, dict[str, Any]] = {}
        self.param_drivers: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self._external_inputs: dict[str, Any] = {}
        self.domain: str = "block"
        self.animation_config: dict[str, Any] = {"frames": 1, "frametime": 2, "interpolate": False, "frames_schedule": None}
        self._counter = 0
        self._static_cache: dict[str, Any] = {}
        self._content_cache: dict[str, tuple[str, Any]] = {}
        self._dynamic_ids: set[str] = set()

    def topological_order(self) -> list[str]:
        """Compute deterministic Kahn topological execution order; raises ``GraphCycleError`` on cycles."""
        if not self.nodes:
            return []
        indeg: dict[str, int] = {nid: 0 for nid in self.node_order}
        children: dict[str, list[str]] = {nid: [] for nid in self.node_order}
        order_rank = {nid: i for i, nid in enumerate(self.node_order)}
        for nid in self.node_order:
            sp = self.nodes[nid]
            seen_deps: set[str] = set()
            for ref in sp.inputs.values():
                dep, _ = _parse_ref(ref)
                if dep in self.nodes and dep not in seen_deps:
                    seen_deps.add(dep)
                    indeg[nid] += 1
                    children[dep].append(nid)
        # Preserve authoring order among ready nodes for deterministic execution
        ready = deque(sorted([nid for nid, d in indeg.items() if d == 0], key=lambda x: order_rank[x]))
        topo: list[str] = []
        while ready:
            u = ready.popleft()
            topo.append(u)
            newly_ready: list[str] = []
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    newly_ready.append(v)
            if newly_ready:
                newly_ready.sort(key=lambda x: order_rank[x])
                ready.extend(newly_ready)
        if len(topo) != len(self.nodes):
            cycled = [nid for nid, d in indeg.items() if d > 0]
            raise GraphCycleError(f"Cycle detected in TexGraph {self.name!r} involving nodes: {cycled}")
        return topo

    def validate_graph(self) -> ValidationReport:
        """Industry-grade static DAG validator: checks ops, sockets, cycles, outputs & orphan nodes."""
        from .registry import NODE_CATALOG

        errors: list[str] = []
        warnings: list[str] = []
        if not self.nodes:
            return ValidationReport(ok=False, errors=["Graph contains zero nodes."])

        for nid in self.node_order:
            sp = self.nodes[nid]
            meta = NODE_CATALOG.get(sp.op)
            if meta is None:
                warnings.append(f"Node {nid!r} uses unregistered op {sp.op!r}")
            for in_name, ref in sp.inputs.items():
                dep, port = _parse_ref(ref)
                if dep not in self.nodes:
                    errors.append(f"Node {nid!r} input {in_name!r} references missing node {dep!r}")
                    continue
                if port is not None:
                    dep_meta = NODE_CATALOG.get(self.nodes[dep].op)
                    if dep_meta and dep_meta.outputs:
                        valid_ports = {p.name for p in dep_meta.outputs} | {"rgba", "albedo"}
                        if dep_meta.output_type == "MATERIAL":
                            valid_ports |= set(MATERIAL_BUS_PORTS)
                        if port not in valid_ports:
                            errors.append(f"Node {nid!r} input {in_name!r} references unknown port {dep}.{port} (valid: {sorted(valid_ports)})")

        for label, ref in (("output_node", self.output_node), ("emissive_node", self.emissive_node), ("output_material_node", self.output_material_node)):
            if ref is not None:
                dep, _ = _parse_ref(ref)
                if dep not in self.nodes:
                    errors.append(f"{label} references missing node {ref!r}")
        for ch, ref in self.pbr_nodes.items():
            dep, _ = _parse_ref(ref)
            if dep not in self.nodes:
                errors.append(f"PBR channel {ch!r} references missing node {ref!r}")

        topo: list[str] = []
        try:
            topo = self.topological_order()
        except GraphCycleError as exc:
            errors.append(str(exc))

        # Warn on dead/unreachable nodes (does not fail validation)
        if not errors and topo:
            roots: set[str] = set()
            for ref in [self.output_node, self.emissive_node, self.output_material_node, *self.pbr_nodes.values()]:
                if ref:
                    roots.add(_parse_ref(ref)[0])
            if not roots:
                roots.add(topo[-1])
            reachable: set[str] = set()
            stack = list(roots)
            while stack:
                cur = stack.pop()
                if cur in reachable or cur not in self.nodes:
                    continue
                reachable.add(cur)
                for ref in self.nodes[cur].inputs.values():
                    stack.append(_parse_ref(ref)[0])
            orphans = [nid for nid in topo if nid not in reachable]
            if orphans:
                warnings.append(f"{len(orphans)} unused node(s) not feeding output: {orphans[:6]}")

        return ValidationReport(ok=len(errors) == 0, errors=errors, warnings=warnings, topo_order=topo)

    def _node_content_hash(self, sp: NodeSpec, eff_params: dict[str, Any], dep_hashes: dict[str, str], phase_tag: str) -> str:
        mesh_sig = ""
        if self.mesh_context is not None:
            mesh_sig = str(getattr(self.mesh_context, "content_fingerprint", "") or "")
            if not mesh_sig:
                mesh_sig = f"{self.mesh_context.width}x{self.mesh_context.height}:{len(self.mesh_context.islands)}:{id(self.mesh_context)}"
        payload = {
            "op": sp.op,
            "w": self.width,
            "h": self.height,
            "tile": self.tileable,
            "seed": self.seed,
            "mesh": mesh_sig,
            "bypassed": bool(sp.bypassed),
            "params": eff_params,
            "deps": dep_hashes,
            "phase": phase_tag,
        }
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def auto_layout(self, *, col_spacing: float = 290.0, row_spacing: float = 195.0) -> dict[str, tuple[float, float]]:
        """Assign clean vertically centered Sugiyama-layered GUI `(x, y)` coordinates for every node."""
        topo = self.topological_order()
        depth: dict[str, int] = {}
        for nid in topo:
            sp = self.nodes[nid]
            d = 0
            for ref in sp.inputs.values():
                dep, _ = _parse_ref(ref)
                if dep in depth:
                    d = max(d, depth[dep] + 1)
            depth[nid] = d
        cols: dict[int, list[str]] = {}
        for nid in topo:
            cols.setdefault(depth[nid], []).append(nid)
        max_rows = max((len(v) for v in cols.values()), default=1)
        total_h = float(max(0, max_rows - 1)) * float(row_spacing)
        positions: dict[str, tuple[float, float]] = {}
        for c_idx, nids in sorted(cols.items()):
            col_h = float(max(0, len(nids) - 1)) * float(row_spacing)
            y_offset = 0.5 * (total_h - col_h)
            for r_idx, nid in enumerate(nids):
                pos = (float(c_idx) * float(col_spacing), round(y_offset + float(r_idx) * float(row_spacing), 1))
                self.nodes[nid].pos = pos
                positions[nid] = pos
        return positions

    def profile(self, *, phase: float = 0.0, top_n: int = 8) -> dict[str, Any]:
        """Evaluate and return per-node runtime breakdown & incremental cache efficiency."""
        ev = self.evaluate(phase=phase, build_pbr=True)
        ranked = sorted(ev.timings_ms.items(), key=lambda kv: kv[1], reverse=True)
        total = round(sum(ev.timings_ms.values()), 2)
        return {
            "graph": self.name,
            "resolution": [self.width, self.height],
            "total_nodes": len(self.nodes),
            "total_ms": total,
            "cache_hits": ev.cache_hits,
            "cache_misses": ev.cache_misses,
            "slowest_nodes": [{"id": k, "op": self.nodes[k].op, "ms": v} for k, v in ranked[: max(1, int(top_n))]],
        }

    def bind_driver(
        self,
        node_id: str,
        param_key: str,
        *,
        kind: str = "sine",
        min_val: float = 0.0,
        max_val: float = 1.0,
        freq: float = 1.0,
        phase_offset: float = 0.0,
        duty: float = 0.5,
        octaves: int = 3,
        seed: int | None = None,
        keyframes: list[Any] | None = None,
    ) -> "TexGraph":
        """Bind a Pixel-Composer-style Animation Driver (`sine|triangle|sawtooth|square|wiggler|keyframes`) to any node parameter!"""
        base, _ = _parse_ref(node_id)
        if base not in self.nodes:
            raise KeyError(f"Unknown node for animation driver: {node_id!r}")
        drv: dict[str, Any] = {
            "node_id": base,
            "param_key": str(param_key),
            "kind": str(kind),
            "min_val": float(min_val),
            "max_val": float(max_val),
            "freq": float(freq),
            "phase_offset": float(phase_offset),
            "duty": float(duty),
            "octaves": int(octaves),
            "seed": int(self.seed if seed is None else seed),
        }
        if keyframes:
            drv["keyframes"] = [list(k) for k in keyframes]
        self.param_drivers.append(drv)
        self._static_cache.clear()
        return self

    def expose_param(
        self,
        public_name: str,
        node_id: str,
        param_key: str,
        *,
        label: str | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        step: float | None = None,
    ) -> "TexGraph":
        """Promote a node parameter to the top-level graph API (Pixel Composer Exposed Parameter parity)."""
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node for exposed param: {node_id!r}")
        self.exposed_params[public_name] = {
            "node_id": node_id,
            "param_key": param_key,
            "label": label or public_name,
            "min": min_val,
            "max": max_val,
            "step": step,
        }
        return self

    def set_exposed(self, public_name: str, value: Any) -> "TexGraph":
        """Override an exposed graph parameter cleanly; content-addressed cache automatically preserves untouched upstream nodes!"""
        info = self.exposed_params[public_name]
        self.nodes[info["node_id"]].params[info["param_key"]] = value
        self._static_cache.clear()
        return self

    def add_group(self, name: str, node_ids: list[str], *, color: str = "#1E293B") -> "TexGraph":
        """Tag a set of nodes into a named visual frame/group for future UI layout."""
        self.groups.append({"name": str(name), "nodes": list(node_ids), "color": str(color)})
        return self

    @contextmanager
    def pass_(
        self,
        name: str,
        *,
        title: str | None = None,
        description: str = "",
        color: str = "#1E293B",
    ) -> Iterator[PassSpec]:
        """Declare a named sequential stage in a Multi-Pass TexGraph pipeline.

        Every node created inside ``with g.pass_(...) as p:`` is recorded in that pass;
        set ``p.preview_ref = node_id`` (or call ``g.set_pass_preview(ref)``) to pick the
        stage checkpoint image rendered in the Multi-Pass Progression Contact Sheet.
        """
        spec = PassSpec(
            name=str(name),
            title=str(title or name),
            description=str(description),
            color=str(color),
        )
        self.passes.append(spec)
        prev = self._active_pass
        self._active_pass = spec
        try:
            yield spec
        finally:
            self._active_pass = prev

    def set_pass_preview(self, ref: str, *, pass_name: str | None = None) -> "TexGraph":
        """Set the checkpoint preview node/socket for the active (or named) pipeline pass."""
        base, _ = _parse_ref(ref)
        if base not in self.nodes:
            raise KeyError(f"Unknown pass preview node: {ref!r}")
        if pass_name is not None:
            for ps in self.passes:
                if ps.name == pass_name:
                    ps.preview_ref = ref
                    return self
            raise KeyError(f"Unknown pass name: {pass_name!r}")
        if self._active_pass is not None:
            self._active_pass.preview_ref = ref
        elif self.passes:
            self.passes[-1].preview_ref = ref
        return self

    def use_subgraph(
        self,
        subgraph_name: str,
        *,
        prefix: str | None = None,
        inputs: dict[str, str] | None = None,
        color: str = "#172554",
        **params: Any,
    ) -> str:
        """Instantiate a reusable parameterized Material Subgraph (Substance .sbsar / Group parity)."""
        if subgraph_name not in SUBGRAPH_REGISTRY:
            raise KeyError(f"Unknown TexGraph subgraph {subgraph_name!r}. Available: {sorted(SUBGRAPH_REGISTRY)}")
        info = SUBGRAPH_REGISTRY[subgraph_name]
        pref = prefix or f"{subgraph_name}_{self._counter + 1:02d}"
        before_n = len(self.node_order)
        merged = dict(info["defaults"])
        merged.update(params)
        out_id = info["builder"](self, prefix=pref, inputs=dict(inputs or {}), **merged)
        created = self.node_order[before_n:]
        if created:
            self.add_group(f"Subgraph: {subgraph_name} ({pref})", created, color=color)
        return out_id

    def _next_id(self, op: str, node_id: str | None) -> str:
        if node_id:
            if node_id in self.nodes:
                raise ValueError(f"Duplicate TexGraph node id: {node_id!r}")
            return node_id
        self._counter += 1
        return f"{op}_{self._counter:02d}"

    def add(
        self,
        op: str,
        *,
        id: str | None = None,
        inputs: dict[str, str] | None = None,
        validate_refs: bool = True,
        **params: Any,
    ) -> str:
        nid = self._next_id(op, id)
        clean_inputs = {k: v for k, v in (inputs or {}).items() if v is not None}
        if validate_refs:
            for ref in clean_inputs.values():
                base_ref, _ = _parse_ref(ref)
                if base_ref not in self.nodes:
                    raise KeyError(f"Node {nid!r} references unknown input node {ref!r}")
        spec = NodeSpec(id=nid, op=op, inputs=clean_inputs, params=params)
        self.nodes[nid] = spec
        self.node_order.append(nid)
        if self._active_pass is not None:
            self._active_pass.node_ids.append(nid)
        self._static_cache.clear()
        return nid

    def set_output(
        self,
        node_id: str,
        *,
        emissive_node: str | None = None,
        height_node: str | None = None,
        normal_node: str | None = None,
        roughness_node: str | None = None,
        metallic_node: str | None = None,
        ao_node: str | None = None,
        shadow_node: str | None = None,
        bent_normal_node: str | None = None,
        thickness_node: str | None = None,
        sss_node: str | None = None,
        porosity_node: str | None = None,
        normal_strength: float | None = None,
        dielectric_f0: float | None = None,
    ) -> "TexGraph":
        base_out, _ = _parse_ref(node_id)
        if base_out not in self.nodes:
            raise KeyError(f"Unknown output node: {node_id!r}")
        self.output_node = node_id
        if emissive_node is not None:
            base_em, _ = _parse_ref(emissive_node)
            if base_em not in self.nodes:
                raise KeyError(f"Unknown emissive node: {emissive_node!r}")
            self.emissive_node = emissive_node
        for ch_name, ch_id in (
            ("height", height_node),
            ("normal", normal_node),
            ("roughness", roughness_node),
            ("metallic", metallic_node),
            ("ao", ao_node),
            ("shadow", shadow_node),
            ("bent_normal", bent_normal_node),
            ("thickness", thickness_node),
            ("sss", sss_node),
            ("porosity", porosity_node),
        ):
            if ch_id is not None:
                base_ch, _ = _parse_ref(ch_id)
                if base_ch not in self.nodes:
                    raise KeyError(f"Unknown PBR {ch_name} node: {ch_id!r}")
                self.pbr_nodes[ch_name] = ch_id
        if normal_strength is not None:
            self.pbr_config["normal_strength"] = float(normal_strength)
        if dielectric_f0 is not None:
            self.pbr_config["dielectric_f0"] = float(dielectric_f0)
        return self

    def set_output_material(
        self,
        mat_node_id: str,
        *,
        normal_strength: float | None = None,
        dielectric_f0: float | None = None,
    ) -> "TexGraph":
        """Bind a unified 12-channel `MATERIAL` bus node to ALL graph outputs (Albedo, Emissive, Shadow, BentNormal, Thickness & LabPBR/MER/ORM)."""
        base, _ = _parse_ref(mat_node_id)
        if base not in self.nodes:
            raise KeyError(f"Unknown MATERIAL bus output node: {mat_node_id!r}")
        self.output_material_node = base
        return self.set_output(
            f"{base}.albedo",
            emissive_node=f"{base}.emissive",
            height_node=f"{base}.height",
            normal_node=f"{base}.normal",
            roughness_node=f"{base}.roughness",
            metallic_node=f"{base}.metallic",
            ao_node=f"{base}.ao",
            shadow_node=f"{base}.shadow",
            bent_normal_node=f"{base}.bent_normal",
            thickness_node=f"{base}.thickness",
            sss_node=f"{base}.sss",
            porosity_node=f"{base}.porosity",
            normal_strength=normal_strength,
            dielectric_f0=dielectric_f0,
        )

    def attach_mesh(self, mesh_context: MeshBakeContext) -> "TexGraph":
        prev_fp = getattr(self.mesh_context, "content_fingerprint", None) if self.mesh_context is not None else None
        new_fp = getattr(mesh_context, "content_fingerprint", None)
        self.mesh_context = mesh_context
        self.width = mesh_context.width
        self.height = mesh_context.height
        self._static_cache.clear()
        if prev_fp != new_fp or not new_fp:
            self._content_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Fluent Node Builders (Concise & AI-Readable)
    # ------------------------------------------------------------------
    def constant(self, value: float | str, *, id: str | None = None) -> str:
        return self.add("constant", id=id, value=value)

    def gradient(self, direction: str = "vertical", *, id: str | None = None) -> str:
        return self.add("gradient", id=id, direction=direction)

    def value_noise(self, scale: float = 8.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("value_noise", id=id, scale=scale, **kw)

    def perlin(self, scale: float = 8.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("perlin", id=id, scale=scale, **kw)

    def fbm(self, scale: float = 6.0, octaves: int = 4, mode: str = "fbm", *, id: str | None = None, **kw: Any) -> str:
        return self.add("fbm", id=id, scale=scale, octaves=octaves, mode=mode, **kw)

    def voronoi(self, scale: float = 6.0, output: str = "edge", *, id: str | None = None, **kw: Any) -> str:
        return self.add("voronoi", id=id, scale=scale, output=output, **kw)

    def gabor(self, scale: float = 12.0, angle_deg: float = 0.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("gabor", id=id, scale=scale, angle_deg=angle_deg, **kw)

    def domain_warp(self, src: str, strength: float = 0.08, scale: float = 4.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("domain_warp", id=id, inputs={"src": src}, strength=strength, scale=scale, **kw)

    def noise_3d(self, scale: float = 0.22, octaves: int = 3, mode: str = "fbm", *, id: str | None = None, **kw: Any) -> str:
        return self.add("noise_3d", id=id, scale=scale, octaves=octaves, mode=mode, **kw)

    def grid(self, cols: int = 4, rows: int = 4, *, id: str | None = None, **kw: Any) -> str:
        return self.add("grid", id=id, cols=cols, rows=rows, **kw)

    def brick(self, cols: int = 4, rows: int = 8, style: str = "running", *, id: str | None = None, **kw: Any) -> str:
        return self.add("brick", id=id, cols=cols, rows=rows, style=style, **kw)

    def hex_grid(self, cols: int = 4, *, id: str | None = None, **kw: Any) -> str:
        return self.add("hex_grid", id=id, cols=cols, **kw)

    def tri_grid(self, cols: int = 6, *, id: str | None = None, **kw: Any) -> str:
        return self.add("tri_grid", id=id, cols=cols, **kw)

    def diamond(self, cols: int = 6, rows: int = 6, style: str = "diamond", *, id: str | None = None, **kw: Any) -> str:
        return self.add("diamond", id=id, cols=cols, rows=rows, style=style, **kw)

    def truchet(self, cols: int = 6, style: str = "arcs", *, id: str | None = None, **kw: Any) -> str:
        return self.add("truchet", id=id, cols=cols, style=style, **kw)

    def mosaic(self, scale: float = 7.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("mosaic", id=id, scale=scale, **kw)

    def scales(self, cols: int = 8, rows: int = 10, shape: str = "dragon", *, id: str | None = None, **kw: Any) -> str:
        return self.add("imbricated_scales", id=id, cols=cols, rows=rows, shape=shape, **kw)

    def feathers(self, cols: int = 4, *, id: str | None = None, **kw: Any) -> str:
        return self.add("feather_barbs", id=id, cols=cols, **kw)

    def fur(self, cols: int = 10, rows: int = 8, *, id: str | None = None, **kw: Any) -> str:
        return self.add("fur_tufts", id=id, cols=cols, rows=rows, **kw)

    def radial_rings(self, freq: float = 6.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("radial_rings", id=id, freq=freq, **kw)

    def energy_veins(self, scale: float = 5.0, style: str = "lightning", *, id: str | None = None, **kw: Any) -> str:
        return self.add("energy_veins", id=id, scale=scale, style=style, **kw)

    def gray_scott(self, preset: str = "labyrinth", steps: int = 320, *, seed_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        return self.add("gray_scott", id=id, inputs={"seed_mask": seed_mask} if seed_mask else {}, preset=preset, steps=steps, **kw)

    def cellular_automata(self, preset: str = "cave", steps: int = 6, *, initial_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        return self.add("cellular_automata", id=id, inputs={"initial_mask": initial_mask} if initial_mask else {}, preset=preset, steps=steps, **kw)

    def iterate(self, src: str, operation: str = "erode_weather", steps: int = 6, *, id: str | None = None, **kw: Any) -> str:
        return self.add("iterate", id=id, inputs={"src": src}, operation=operation, steps=steps, **kw)

    def weathering_drip(self, src: str, *, cavity: str | None = None, id: str | None = None, **kw: Any) -> str:
        ins = {"src": src}
        if cavity:
            ins["cavity"] = cavity
        return self.add("weathering_drip", id=id, inputs=ins, **kw)

    def kuwahara(self, src: str, radius: int = 2, sharpness: float = 8.0, *, id: str | None = None) -> str:
        return self.add("kuwahara", id=id, inputs={"src": src}, radius=radius, sharpness=sharpness)

    def bilateral(self, src: str, radius: int = 2, sigma_s: float = 1.5, sigma_r: float = 0.14, *, id: str | None = None) -> str:
        return self.add("bilateral", id=id, inputs={"src": src}, radius=radius, sigma_s=sigma_s, sigma_r=sigma_r)

    def tv_denoise(self, src: str, weight: float = 0.08, iterations: int = 12, *, id: str | None = None) -> str:
        return self.add("tv_denoise", id=id, inputs={"src": src}, weight=weight, iterations=iterations)

    def nlm_denoise(self, src: str, patch_radius: int = 1, search_radius: int = 3, h_param: float = 0.08, *, id: str | None = None) -> str:
        return self.add("nlm_denoise", id=id, inputs={"src": src}, patch_radius=patch_radius, search_radius=search_radius, h_param=h_param)

    def despeckle(self, src: str, threshold: float = 0.14, *, id: str | None = None, **kw: Any) -> str:
        return self.add("despeckle", id=id, inputs={"src": src}, threshold=threshold, **kw)

    def cluster_mode(self, src: str, *, id: str | None = None) -> str:
        return self.add("cluster_mode", id=id, inputs={"src": src})

    def mesh_channel(self, channel: str, *, pattern: str | None = None, zone: str | None = None, faces: str | None = None, id: str | None = None) -> str:
        """Read a baked 3D channel: 'baked_ao', 'edge_bevel', 'joint_cavity', 'top_grad', 'front_grad', 'normal_y', 'side_sign', 'zone', 'element', 'outward_eye', 'face', 'island_mask', 'local_u', 'local_v'."""
        return self.add("mesh_channel", id=id, channel=channel, pattern=pattern, zone=zone, faces=faces)

    def height_to_normal(self, height: str, *, strength: float = 2.4, convention: str = "opengl", id: str | None = None) -> str:
        return self.add("height_to_normal", id=id, inputs={"height": height}, strength=strength, convention=convention)

    def vibrance(self, src: str, *, vibrance: float = 0.22, contrast: float = 1.06, lift: float = 0.015, shadow_cool: float = 0.05, id: str | None = None) -> str:
        return self.add("vibrance_grade", id=id, inputs={"src": src}, vibrance=vibrance, contrast=contrast, lift=lift, shadow_cool=shadow_cool)

    def eye_paint(self, local_u: str, local_v: str, *, side_sign: str | None = None, id: str | None = None, **kw: Any) -> tuple[str, str]:
        """Adds paired 7-layer procedural creature eye RGBA + selective iris emissive mask nodes."""
        ins = {"local_u": local_u, "local_v": local_v}
        if side_sign:
            ins["side_sign"] = side_sign
        rgb_id = self.add("eye_paint_rgba", id=id, inputs=ins, **kw)
        em_id = self.add("eye_paint_emissive", id=f"{rgb_id}_em", inputs=ins, **kw)
        return rgb_id, em_id

    def wing_scallop(self, local_u: str, local_v: str, *, arches: int = 2, depth: float = 0.28, rim_width: float = 0.085, side_taper: float = 0.06, id: str | None = None) -> tuple[str, str]:
        """Adds paired wing scallop (hard_alpha_mask, trailing_rim_mask) nodes."""
        ins = {"local_u": local_u, "local_v": local_v}
        kw = dict(arches=arches, depth=depth, rim_width=rim_width, side_taper=side_taper)
        a_id = self.add("wing_scallop_alpha", id=id, inputs=ins, **kw)
        r_id = self.add("wing_scallop_rim", id=f"{a_id}_rim", inputs=ins, **kw)
        return a_id, r_id

    def set_alpha(self, src: str, alpha: str, *, mask: str | None = None, id: str | None = None) -> str:
        """Replace RGBA alpha channel (globally or selectively inside ``mask``)."""
        ins = {"src": src, "alpha": alpha}
        if mask:
            ins["mask"] = mask
        return self.add("set_alpha", id=id, inputs=ins)

    def levels(self, src: str, *, id: str | None = None, **kw: Any) -> str:
        return self.add("levels", id=id, inputs={"src": src}, **kw)

    def math(self, a: str, b: str, op: str = "mul", *, id: str | None = None, **kw: Any) -> str:
        return self.add("math", id=id, inputs={"a": a, "b": b}, math_op=op, **kw)

    def shade(self, height: str, *, use_mesh: bool = True, id: str | None = None, **kw: Any) -> str:
        return self.add("directional_shade", id=id, inputs={"height": height}, use_mesh=use_mesh, **kw)

    def bevel_emboss(self, height: str, strength: float = 0.35, *, id: str | None = None, **kw: Any) -> str:
        return self.add("bevel_emboss", id=id, inputs={"height": height}, strength=strength, **kw)

    def hue_ramp(self, src: str, *, alpha_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        ins = {"src": src}
        if alpha_mask:
            ins["alpha_mask"] = alpha_mask
        return self.add("hue_ramp", id=id, inputs=ins, **kw)

    def quantize(self, src: str, palette_hex: list[str], *, id: str | None = None, **kw: Any) -> str:
        return self.add("palette_quantize", id=id, inputs={"src": src}, palette_hex=palette_hex, **kw)

    def selout(self, src: str, *, id: str | None = None, **kw: Any) -> str:
        return self.add("selout", id=id, inputs={"src": src}, **kw)

    def blend(self, base: str, top: str, mode: str = "normal", opacity: float = 1.0, *, mask: str | None = None, id: str | None = None) -> str:
        ins = {"base": base, "top": top}
        if mask:
            ins["mask"] = mask
        return self.add("blend", id=id, inputs=ins, mode=mode, opacity=opacity)

    def scroll(self, src: str, speed_x: float = 0.0, speed_y: float = 1.0, *, id: str | None = None) -> str:
        return self.add("scroll", id=id, inputs={"src": src}, speed_x=speed_x, speed_y=speed_y)

    def pulse(self, src: str, min_val: float = 0.45, max_val: float = 1.0, freq: float = 1.0, phase_offset: float = 0.0, *, id: str | None = None) -> str:
        return self.add("pulse", id=id, inputs={"src": src}, min_val=min_val, max_val=max_val, freq=freq, phase_offset=phase_offset)

    def emissive(self, albedo: str, mask: str, *, color: str | None = None, glow_bleed_px: int = 2, glow_strength: float = 0.35, id: str | None = None) -> tuple[str, str]:
        """Adds paired albedo-with-glow and emissive-only nodes, returning (albedo_id, emissive_id)."""
        ins = {"albedo": albedo, "mask": mask}
        if color:
            ins["color"] = color
        alb_id = self.add("emissive_albedo", id=id, inputs=ins, glow_bleed_px=glow_bleed_px, glow_strength=glow_strength)
        em_id = self.add("emissive_map", id=f"{alb_id}_emissive", inputs=ins, glow_bleed_px=glow_bleed_px, glow_strength=glow_strength)
        return alb_id, em_id

    def pad_islands(self, src: str, *, pad: int | None = None, id: str | None = None) -> str:
        return self.add("pad_islands", id=id, inputs={"src": src}, pad=pad)

    def sprite_mask(self, shape_expr: str, *, id: str | None = None, **kw: Any) -> str:
        """Procedural pixel-art sprite silhouette generator (for items/icons/weapons)."""
        return self.add("sprite_mask", id=id, shape=shape_expr, **kw)

    # --- Smooth Gradient Path, Painterly Studio, Transform, Filter & Color Builders (v3) ---
    def gradient_field(self, kind: str = "linear", *, id: str | None = None, **kw: Any) -> str:
        return self.add("gradient_field", id=id, kind=kind, **kw)

    def multi_stop_gradient(self, src: str, stops: list[Any], *, alpha_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        ins = {"src": src}
        if alpha_mask:
            ins["alpha_mask"] = alpha_mask
        return self.add("multi_stop_gradient", id=id, inputs=ins, stops=stops, **kw)

    def cosine_gradient(self, src: str, *, alpha_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        ins = {"src": src}
        if alpha_mask:
            ins["alpha_mask"] = alpha_mask
        return self.add("cosine_gradient", id=id, inputs=ins, **kw)

    def spline_path(self, points: list[Any], *, output: str = "ribbon", id: str | None = None, **kw: Any) -> str:
        return self.add("spline_path", id=id, points=points, output=output, **kw)

    def per_island_gradient(self, direction: str = "top_to_bottom", *, curve: float = 1.0, id: str | None = None) -> str:
        return self.add("per_island_gradient", id=id, direction=direction, curve=curve)

    def body_flow_gradient(self, axis: str = "dorsal_ventral", *, bias: float = 0.0, gamma: float = 1.0, id: str | None = None) -> str:
        return self.add("body_flow_gradient", id=id, axis=axis, bias=bias, gamma=gamma)

    def brushstrokes(self, angle_deg: float = 0.0, *, id: str | None = None, **kw: Any) -> str:
        return self.add("brushstrokes", id=id, angle_deg=angle_deg, **kw)

    def layered_plates(self, cols: int = 8, rows: int = 12, *, output: str = "height", id: str | None = None, **kw: Any) -> str:
        return self.add("layered_plates", id=id, cols=cols, rows=rows, output=output, **kw)

    def smooth_wing(self, local_u: str, local_v: str, *, id: str | None = None, **kw: Any) -> dict[str, str]:
        """Adds Master Dragon Wing Membrane nodes returning dict of IDs: rgba, alpha, rim, glow, height."""
        ins = {"local_u": local_u, "local_v": local_v}
        base = id or f"smooth_wing_{self._counter + 1:02d}"
        return {
            "rgba": self.add("smooth_wing_rgba", id=base, inputs=ins, **kw),
            "alpha": self.add("smooth_wing_alpha", id=f"{base}_alpha", inputs=ins, **kw),
            "rim": self.add("smooth_wing_rim", id=f"{base}_rim", inputs=ins, **kw),
            "glow": self.add("smooth_wing_glow", id=f"{base}_glow", inputs=ins, **kw),
            "height": self.add("smooth_wing_height", id=f"{base}_height", inputs=ins, **kw),
        }

    def gradient_eye(self, local_u: str, local_v: str, *, side_sign: str | None = None, id: str | None = None, **kw: Any) -> tuple[str, str]:
        """Adds Non-Standard Luminous Gradient Eye RGBA + emissive mask nodes (mirror-symmetric L/R)."""
        ins = {"local_u": local_u, "local_v": local_v}
        if side_sign:
            ins["side_sign"] = side_sign
        rgb_id = self.add("gradient_eye_rgba", id=id, inputs=ins, **kw)
        em_id = self.add("gradient_eye_emissive", id=f"{rgb_id}_em", inputs=ins, **kw)
        return rgb_id, em_id

    def temple_stone(self, cols: int = 2, rows: int = 4, *, output: str = "height", id: str | None = None, **kw: Any) -> str:
        return self.add("temple_stone", id=id, cols=cols, rows=rows, output=output, **kw)

    def wood_planks(self, planks: int = 4, *, output: str = "height", id: str | None = None, **kw: Any) -> str:
        return self.add("wood_planks", id=id, planks=planks, output=output, **kw)

    def foliage_clusters(self, cols: int = 5, rows: int = 5, *, output: str = "height", id: str | None = None, **kw: Any) -> str:
        return self.add("foliage_clusters", id=id, cols=cols, rows=rows, output=output, **kw)

    def curves(self, src: str, points: list[Any], *, id: str | None = None) -> str:
        return self.add("curves", id=id, inputs={"src": src}, points=points)

    def color_balance(self, src: str, *, shadows=(0.0, 0.0, 0.0), midtones=(0.0, 0.0, 0.0), highlights=(0.0, 0.0, 0.0), id: str | None = None) -> str:
        return self.add("color_balance", id=id, inputs={"src": src}, shadows=list(shadows), midtones=list(midtones), highlights=list(highlights))

    def oklch_adjust(self, src: str, *, hue_shift_deg: float = 0.0, chroma_scale: float = 1.0, lightness_offset: float = 0.0, id: str | None = None) -> str:
        return self.add("oklch_adjust", id=id, inputs={"src": src}, hue_shift_deg=hue_shift_deg, chroma_scale=chroma_scale, lightness_offset=lightness_offset)

    def posterize_soft(self, src: str, *, steps: int = 6, blend: float = 0.5, id: str | None = None) -> str:
        return self.add("posterize_soft", id=id, inputs={"src": src}, steps=steps, blend=blend)

    def tile_transform(self, src: str, *, id: str | None = None, **kw: Any) -> str:
        return self.add("tile_transform", id=id, inputs={"src": src}, **kw)

    def polar_warp(self, src: str, *, to_polar: bool = True, id: str | None = None) -> str:
        return self.add("polar_warp", id=id, inputs={"src": src}, to_polar=to_polar)

    def twirl(self, src: str, *, strength_deg: float = 90.0, radius: float = 0.5, id: str | None = None, **kw: Any) -> str:
        return self.add("twirl_distort", id=id, inputs={"src": src}, strength_deg=strength_deg, radius=radius, **kw)

    def wave(self, src: str, *, id: str | None = None, **kw: Any) -> str:
        return self.add("wave_distort", id=id, inputs={"src": src}, **kw)

    def blur(self, src: str, *, radius: int = 2, sigma: float = 0.0, id: str | None = None) -> str:
        return self.add("gaussian_blur", id=id, inputs={"src": src}, radius=radius, sigma=sigma)

    def directional_blur(self, src: str, *, angle_deg: float = 0.0, length: int = 4, id: str | None = None) -> str:
        return self.add("directional_blur", id=id, inputs={"src": src}, angle_deg=angle_deg, length=length)

    def sharpen(self, src: str, *, radius: int = 1, amount: float = 0.65, id: str | None = None) -> str:
        return self.add("unsharp_mask", id=id, inputs={"src": src}, radius=radius, amount=amount)

    def sobel_edges(self, src: str, *, strength: float = 1.5, id: str | None = None) -> str:
        return self.add("sobel_edges", id=id, inputs={"src": src}, strength=strength)

    def sdf(self, mask: str, *, max_dist_px: float = 12.0, invert: bool = False, id: str | None = None) -> str:
        return self.add("sdf_from_mask", id=id, inputs={"mask": mask}, max_dist_px=max_dist_px, invert=invert)

    def morph(self, src: str, *, op: str = "dilate", radius: int = 1, id: str | None = None) -> str:
        return self.add("morph_op", id=id, inputs={"src": src}, morph_op=op, radius=radius)

    def normal_blend(self, base: str, detail: str, *, strength: float = 1.0, id: str | None = None) -> str:
        return self.add("normal_blend", id=id, inputs={"base": base, "detail": detail}, strength=strength)

    def ao_from_height(self, height: str, *, radius: int = 3, strength: float = 1.0, id: str | None = None) -> str:
        return self.add("ao_from_height", id=id, inputs={"height": height}, radius=radius, strength=strength)

    def bloom(self, src: str, *, threshold: float = 0.72, radius: int = 3, intensity: float = 0.55, id: str | None = None) -> str:
        return self.add("bloom_glow", id=id, inputs={"src": src}, threshold=threshold, radius=radius, intensity=intensity)

    # --- Industry-Standard Multi-Port & First-Class MATERIAL Bus Builders (v4) ---
    def material_make(
        self,
        albedo: str,
        *,
        height: str | float | None = None,
        normal: str | None = None,
        roughness: str | float | None = 0.55,
        metallic: str | float | None = 0.0,
        ao: str | float | None = 1.0,
        emissive: str | None = None,
        sss: str | float | None = 0.0,
        porosity: str | float | None = 0.0,
        normal_strength: float = 2.4,
        id: str | None = None,
    ) -> str:
        ins: dict[str, str] = {"albedo": albedo}
        params: dict[str, Any] = {"normal_strength": float(normal_strength)}
        for k, v in (
            ("height", height),
            ("normal", normal),
            ("roughness", roughness),
            ("metallic", metallic),
            ("ao", ao),
            ("emissive", emissive),
            ("sss", sss),
            ("porosity", porosity),
        ):
            if isinstance(v, str):
                ins[k] = v
            elif v is not None:
                params[k] = float(v)
        return self.add("material_make", id=id, inputs=ins, **params)

    def material_blend(
        self,
        base: str,
        top: str,
        *,
        mask: str | None = None,
        opacity: float = 1.0,
        albedo_mode: str = "normal",
        normal_mode: str = "lerp",
        preserve_base_alpha: bool = False,
        id: str | None = None,
    ) -> str:
        ins: dict[str, str] = {"base": base, "top": top}
        if mask:
            ins["mask"] = mask
        return self.add(
            "material_blend",
            id=id,
            inputs=ins,
            opacity=opacity,
            albedo_mode=albedo_mode,
            normal_mode=normal_mode,
            preserve_base_alpha=preserve_base_alpha,
        )

    def material_height_blend(
        self,
        base: str,
        top: str,
        *,
        mask: str | None = None,
        height_offset: float = 0.0,
        contrast: float = 6.0,
        seam_ao_strength: float = 0.25,
        normal_mode: str = "lerp",
        id: str | None = None,
    ) -> str:
        ins: dict[str, str] = {"base": base, "top": top}
        if mask:
            ins["mask"] = mask
        return self.add(
            "material_height_blend",
            id=id,
            inputs=ins,
            height_offset=height_offset,
            contrast=contrast,
            seam_ao_strength=seam_ao_strength,
            normal_mode=normal_mode,
        )

    def material_grade(
        self,
        mat: str,
        *,
        ao_bake: str | None = None,
        detail_normal: str | None = None,
        id: str | None = None,
        **kw: Any,
    ) -> str:
        ins: dict[str, str] = {"mat": mat}
        if ao_bake:
            ins["ao_bake"] = ao_bake
        if detail_normal:
            ins["detail_normal"] = detail_normal
        return self.add("material_grade", id=id, inputs=ins, **kw)

    def material_set_alpha(self, mat: str, alpha: str, *, mask: str | None = None, id: str | None = None) -> str:
        ins: dict[str, str] = {"mat": mat, "alpha": alpha}
        if mask:
            ins["mask"] = mask
        return self.add("material_set_alpha", id=id, inputs=ins)

    def material_pad_islands(self, mat: str, *, pad: int | None = None, id: str | None = None) -> str:
        return self.add("material_pad_islands", id=id, inputs={"mat": mat}, pad=pad)

    def organic_domain_warp_3d(self, scale: float = 3.5, warp_strength: float = 0.42, octaves: int = 3, *, seed: int | None = None, id: str | None = None) -> str:
        """3D world-space organic domain-warped pigment & flow generator (ports: mottle, warp_x, warp_y, flow)."""
        return self.add("organic_domain_warp_3d", id=id, scale=scale, warp_strength=warp_strength, octaves=octaves, seed=seed)

    def painterly_brush(self, angle_deg: float = -16.0, stroke_scale: float = 12.0, organic_distortion: float = 0.58, mottle_weight: float = 0.48, *, seed: int | None = None, id: str | None = None) -> str:
        """Hand-painted anisotropic organic brushstroke & pigment-mottle generator (ports: painterly, brush, mottle, hue_drift)."""
        return self.add("painterly_brush_field", id=id, angle_deg=angle_deg, stroke_scale=stroke_scale, organic_distortion=organic_distortion, mottle_weight=mottle_weight, seed=seed)

    painterly_brush_field = painterly_brush

    # --- Industry-Standard Core Operators & Full Bio/Block Material Builders (TexGraph v7) ---
    def sdf_shape(self, shape: str = "rounded_box", *, id: str | None = None, **kw: Any) -> str:
        """Analytical 2D SDF shape generator (ports: sdf, mask, bevel, outline)."""
        return self.add("sdf_shape", id=id, shape=shape, **kw)

    def sdf_boolean(self, a: str, b: str, *, op: str = "smooth_union", smoothness: float = 0.04, id: str | None = None) -> str:
        """Boolean combine two 2D SDF fields (ports: sdf, mask, bevel)."""
        return self.add("sdf_boolean", id=id, inputs={"a": a, "b": b}, bool_op=op, smoothness=smoothness)

    def tile_sampler(
        self,
        stamp: str | None = None,
        *,
        density_map: str | None = None,
        scale_map: str | None = None,
        rot_map: str | None = None,
        id: str | None = None,
        **kw: Any,
    ) -> str:
        """Substance Tile Sampler / Pixel Composer Scatter & Splatter (ports: height, mask, id_map, rgba)."""
        ins: dict[str, str] = {}
        if stamp:
            ins["stamp"] = stamp
        if density_map:
            ins["density_map"] = density_map
        if scale_map:
            ins["scale_map"] = scale_map
        if rot_map:
            ins["rot_map"] = rot_map
        return self.add("tile_sampler", id=id, inputs=ins, **kw)

    def flood_fill(self, mask: str, *, threshold: float = 0.25, id: str | None = None, **kw: Any) -> str:
        """Substance Flood Fill island analyzer (ports: island_id, random_gray, gradient_u, gradient_v, random_slope, edge_dist)."""
        return self.add("flood_fill", id=id, inputs={"mask": mask}, threshold=threshold, **kw)

    def slope_blur(self, src: str, slope_map: str, *, intensity: float = 1.6, samples: int = 6, mode: str = "min", id: str | None = None) -> str:
        """Substance Slope Blur directional erosion/dilation/impasto."""
        return self.add("slope_blur", id=id, inputs={"src": src, "slope_map": slope_map}, intensity=intensity, samples=samples, mode=mode)

    def histogram_scan(self, src: str, *, position: float = 0.5, contrast: float = 0.6, invert: bool = False, id: str | None = None) -> str:
        return self.add("histogram_scan", id=id, inputs={"src": src}, position=position, contrast=contrast, invert=invert)

    def histogram_select(self, src: str, *, position: float = 0.5, range_width: float = 0.25, contrast: float = 0.5, id: str | None = None) -> str:
        return self.add("histogram_select", id=id, inputs={"src": src}, position=position, range_width=range_width, contrast=contrast)

    def auto_levels(self, src: str, *, low_pct: float = 1.0, high_pct: float = 99.0, id: str | None = None) -> str:
        return self.add("auto_levels", id=id, inputs={"src": src}, low_pct=low_pct, high_pct=high_pct)

    def curvature(self, height: str, *, radius: int = 2, strength: float = 2.2, id: str | None = None) -> str:
        """Multi-scale curvature detector (ports: curvature, convex, concave)."""
        return self.add("curvature", id=id, inputs={"height": height}, radius=radius, strength=strength)

    def edge_wear(self, height: str, *, wear_amount: float = 0.45, chip_scale: float = 8.0, grime_amount: float = 0.55, id: str | None = None, **kw: Any) -> str:
        """Procedural chipped edge wear & crevice grime masks (ports: wear, grime, curvature)."""
        return self.add("edge_wear", id=id, inputs={"height": height}, wear_amount=wear_amount, chip_scale=chip_scale, grime_amount=grime_amount, **kw)

    def displace_uv(self, src: str, disp_map: str, *, disp_y_map: str | None = None, strength_x: float = 2.0, strength_y: float = 2.0, mid_value: float = 0.5, pixel_snap: bool = False, id: str | None = None) -> str:
        ins: dict[str, str] = {"src": src, "disp_map": disp_map}
        if disp_y_map:
            ins["disp_y_map"] = disp_y_map
        return self.add("displace_uv", id=id, inputs=ins, strength_x=strength_x, strength_y=strength_y, mid_value=mid_value, pixel_snap=pixel_snap)

    def normal_light_2d(self, albedo: str, normal: str, *, id: str | None = None, **kw: Any) -> str:
        return self.add("normal_light_2d", id=id, inputs={"albedo": albedo, "normal": normal}, **kw)

    def directional_selout(self, src: str, *, light_angle_deg: float = -135.0, lit_darken: float = 0.64, shadow_darken: float = 0.30, id: str | None = None, **kw: Any) -> str:
        return self.add("directional_selout", id=id, inputs={"src": src}, light_angle_deg=light_angle_deg, lit_darken=lit_darken, shadow_darken=shadow_darken, **kw)

    def cobblestone_masonry(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("cobblestone_masonry", id=id, **kw)

    def bark_and_timber(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("bark_and_timber", id=id, **kw)

    def crystalline_ore_vein(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("crystalline_ore_vein", id=id, **kw)

    def caustic_water(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("caustic_water", id=id, **kw)

    def organic_fur_pelt(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("organic_fur_pelt", id=id, **kw)

    def layered_plumage(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("layered_plumage", id=id, **kw)

    def chitin_carapace(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("chitin_carapace", id=id, **kw)

    def aquatic_scales_skin(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("aquatic_scales_skin", id=id, **kw)

    def elemental_golem_surface(self, *, id: str | None = None, **kw: Any) -> str:
        return self.add("elemental_golem_surface", id=id, **kw)

    # --- TexGraph v8 Character/Orc Multi-Material Kit & Part-Aware Entity Compositor ---
    def orc_flesh_skin(self, *, id: str | None = None, **kw: Any) -> str:
        """Hue-shifted olive/emerald orc/goblin skin MaterialBus with scars, SSS blush & optional war paint."""
        return self.add("orc_flesh_skin", id=id, **kw)

    def stitched_leather(self, *, id: str | None = None, **kw: Any) -> str:
        """Warm stitched leather MaterialBus with concentric rim bevels & perimeter stitch dots."""
        return self.add("stitched_leather", id=id, **kw)

    def forged_battle_metal(self, *, id: str | None = None, **kw: Any) -> str:
        """Cool charcoal-indigo shadowed forged steel MaterialBus with 1 px bevels, scratches, rivets & rust."""
        return self.add("forged_battle_metal", id=id, **kw)

    def woven_battle_cloth(self, *, id: str | None = None, **kw: Any) -> str:
        """Muted draped battle cloth / twisted rope MaterialBus with soft folds & optional hem trim."""
        return self.add("woven_battle_cloth", id=id, **kw)

    def keratin_horn_bone(self, *, id: str | None = None, **kw: Any) -> str:
        """Dark-rooted keratin-to-ivory MaterialBus for tusks, horns, teeth, beaks, claws & hooves."""
        return self.add("keratin_horn_bone", id=id, **kw)

    def paint_entity_eyes_and_parts(self, base: str, *, id: str | None = None, **kw: Any) -> str:
        """Part-aware semantic anatomical landmark compositor (protects eyes, noses, beaks, mouths, teeth, claws, fins, runes)."""
        return self.add("paint_entity_eyes_and_parts", id=id, inputs={"base": base}, **kw)

    # --- TexGraph v9 Pixel Composer Atomic Operators & 3D Baked Emissive Bounce ---
    def pixel_dither(self, src: str, palette: list[Any], *, pattern: str = "bayer_2x2", contrast: float = 1.0, mode: str = "color", id: str | None = None, **kw: Any) -> str:
        return self.add("pixel_dither", id=id, inputs={"src": src}, palette=palette, pattern=pattern, contrast=contrast, mode=mode, **kw)

    def palette_posterize(self, src: str, palette: list[Any], *, space: str = "oklab", posterize_alpha: bool = True, id: str | None = None, **kw: Any) -> str:
        return self.add("palette_posterize", id=id, inputs={"src": src}, palette=palette, space=space, posterize_alpha=posterize_alpha, **kw)

    def color_replace(self, src: str, src_palette: list[Any], dst_palette: list[Any], *, tolerance: float = 0.25, id: str | None = None) -> str:
        return self.add("color_replace", id=id, inputs={"src": src}, src_palette=src_palette, dst_palette=dst_palette, tolerance=tolerance)

    def level_selector(self, src: str, *, center: float = 0.75, width: float = 0.20, softness: float = 0.05, id: str | None = None) -> str:
        return self.add("level_selector", id=id, inputs={"src": src}, center=center, width=width, softness=softness)

    def de_stray(self, src: str, *, strictness: str = "medium", tolerance: float = 0.06, iterations: int = 1, id: str | None = None) -> str:
        return self.add("de_stray", id=id, inputs={"src": src}, strictness=strictness, tolerance=tolerance, iterations=iterations)

    def de_corner(self, src: str, *, strictness: str = "medium", tolerance: float = 0.06, iterations: int = 1, id: str | None = None) -> str:
        return self.add("de_corner", id=id, inputs={"src": src}, strictness=strictness, tolerance=tolerance, iterations=iterations)

    def pixel_clusters(self, src: str, palette: list[Any], *, direction: str = "isotropic", cluster_size: int = 2, dither_strength: float = 0.0, id: str | None = None, **kw: Any) -> str:
        return self.add("pixel_clusters", id=id, inputs={"src": src}, palette=palette, direction=direction, cluster_size=cluster_size, dither_strength=dither_strength, **kw)

    def pixel_bevel_directional(self, *, light_angle_deg: float = -135.0, bevel_width: float = 0.11, chip_amount: float = 0.42, id: str | None = None, **kw: Any) -> str:
        return self.add("pixel_bevel_directional", id=id, light_angle_deg=light_angle_deg, bevel_width=bevel_width, chip_amount=chip_amount, **kw)

    def pixel_outline(self, src: str, *, width: int = 1, position: str = "outside", color: str = "#161224", corner_mode: str = "4way", id: str | None = None, **kw: Any) -> str:
        return self.add("pixel_outline", id=id, inputs={"src": src}, width=width, position=position, color=color, corner_mode=corner_mode, **kw)

    def heightmap_shadow(self, height: str, *, light_angle_deg: float = -135.0, elevation_deg: float = 34.0, max_steps: int = 8, strength: float = 0.72, id: str | None = None, **kw: Any) -> str:
        return self.add("heightmap_shadow", id=id, inputs={"height": height}, light_angle_deg=light_angle_deg, elevation_deg=elevation_deg, max_steps=max_steps, strength=strength, **kw)

    def pixel_mirror(self, src: str, *, axis: str = "x", id: str | None = None) -> str:
        return self.add("pixel_mirror", id=id, inputs={"src": src}, axis=axis)

    def pixel_sort(self, src: str, *, axis: str = "vertical", low_thresh: float = 0.25, high_thresh: float = 0.85, reverse: bool = False, id: str | None = None) -> str:
        return self.add("pixel_sort", id=id, inputs={"src": src}, axis=axis, low_thresh=low_thresh, high_thresh=high_thresh, reverse=reverse)

    def chromatic_aberration(self, src: str, *, shift_x: int = 1, shift_y: int = 0, strength: float = 0.65, id: str | None = None) -> str:
        return self.add("chromatic_aberration", id=id, inputs={"src": src}, shift_x=shift_x, shift_y=shift_y, strength=strength)

    def vignette(self, src: str, *, strength: float = 0.35, radius: float = 0.72, color: str = "#120E1D", id: str | None = None) -> str:
        return self.add("vignette", id=id, inputs={"src": src}, strength=strength, radius=radius, color=color)

    def bake_emissive_bounce(self, mat: str, *, radius_units: float = 7.5, intensity: float = 0.52, posterize_steps: int = 0, id: str | None = None, **kw: Any) -> str:
        return self.add("bake_emissive_bounce", id=id, inputs={"mat": mat}, radius_units=radius_units, intensity=intensity, posterize_steps=posterize_steps, **kw)

    # --- TexGraph v10 Seamless UV-Geometric Edge Solving & Precomputed Shadow Special PBR ---
    def triplanar_project(self, src: str, *, scale: float = 0.08, sharpness: float = 6.0, space: str = "world", dither_blend: bool = False, pixel_snap: bool = True, id: str | None = None, **kw: Any) -> str:
        return self.add("triplanar_project", id=id, inputs={"src": src}, scale=scale, sharpness=sharpness, space=space, dither_blend=dither_blend, pixel_snap=pixel_snap, **kw)

    def uv_seam_heal(self, src: str, *, strength: float = 0.72, radius_px: int = 2, heal_normals: bool = True, id: str | None = None) -> str:
        return self.add("uv_seam_heal", id=id, inputs={"src": src}, strength=strength, radius_px=radius_px, heal_normals=heal_normals)

    def block_edge_transition(self, base: str | None = None, *, neighbors: str = "exposed_all", style: str = "bevel_trim", border_width: float = 0.14, id: str | None = None, **kw: Any) -> str:
        ins = {"base": base} if base else {}
        return self.add("block_edge_transition", id=id, inputs=ins, neighbors=neighbors, style=style, border_width=border_width, **kw)

    def precompute_shadow_pbr(self, mat: str, *, preset: str = "studio_hero", shadow_strength: float = 0.62, bent_normal_strength: float = 0.48, thickness_sss_strength: float = 0.42, bake_into_albedo: float = 0.45, id: str | None = None, **kw: Any) -> str:
        return self.add("precompute_shadow_pbr", id=id, inputs={"mat": mat}, preset=preset, shadow_strength=shadow_strength, bent_normal_strength=bent_normal_strength, thickness_sss_strength=thickness_sss_strength, bake_into_albedo=bake_into_albedo, **kw)

    def sculpted_pinion_wing(
        self,
        span_u: str,
        trail_v: str,
        *,
        face_is_down: str | None = None,
        wave: str | None = None,
        id: str | None = None,
        **kw: Any,
    ) -> str:
        """Single-eval Sculpted Pinion-Blade Wing emitting a complete multi-port `MATERIAL` bus."""
        ins: dict[str, str] = {"span_u": span_u, "trail_v": trail_v}
        if face_is_down:
            ins["face_is_down"] = face_is_down
        if wave:
            ins["wave"] = wave
        return self.add("sculpted_pinion_wing", id=id, inputs=ins, **kw)

    def smooth_wing_multi(self, local_u: str, local_v: str, *, id: str | None = None, **kw: Any) -> str:
        """Single-eval multi-port Backlit Dragon Wing Membrane (ports: albedo, alpha, rim_mask, glow_mask, height)."""
        return self.add("smooth_wing_multi", id=id, inputs={"local_u": local_u, "local_v": local_v}, **kw)

    def gradient_eye_multi(self, local_u: str, local_v: str, *, side_sign: str | None = None, id: str | None = None, **kw: Any) -> str:
        """Single-eval multi-port Non-Standard Gradient Eye (ports: albedo, emissive_mask)."""
        ins: dict[str, str] = {"local_u": local_u, "local_v": local_v}
        if side_sign:
            ins["side_sign"] = side_sign
        return self.add("gradient_eye_multi", id=id, inputs=ins, **kw)

    def emissive_composite(
        self,
        albedo: str,
        mask: str,
        *,
        color: str | None = None,
        glow_bleed_px: int = 2,
        glow_strength: float = 0.35,
        id: str | None = None,
    ) -> str:
        """Single-eval multi-port Emissive Composite (ports: albedo, emissive)."""
        ins: dict[str, str] = {"albedo": albedo, "mask": mask}
        if color:
            ins["color"] = color
        return self.add("emissive_composite", id=id, inputs=ins, glow_bleed_px=glow_bleed_px, glow_strength=glow_strength)

    # --- Pixel-Composer-Parity Generate / Iterate / Simulate / Animate Builders (v5) ---
    def anim_driver(self, kind: str = "sine", *, id: str | None = None, **kw: Any) -> str:
        """Procedural timeline driver node emitting a scalar field driven by wavetable, wiggler, or keyframes."""
        return self.add("anim_driver", id=id, kind=kind, **kw)

    def iterate_inline(self, src: str, mode: str = "warp_cascade", *, steps: int = 5, id: str | None = None, **kw: Any) -> str:
        """Pixel-Composer-style Iterate Inline recursive multi-step accumulation loop."""
        return self.add("iterate_inline", id=id, inputs={"src": src}, mode=mode, steps=steps, **kw)

    def feedback_loop(self, src: str, *, decay: float = 0.80, drift_x_px: float = 0.0, drift_y_px: float = -1.0, blur_radius: int = 1, pre_roll: int = 6, loop_frames: int = 16, blend_mode: str = "screen", id: str | None = None) -> str:
        """Pixel-Composer-style Feedback Inline temporal state buffer with warm-up pre-roll for seamless looping trails."""
        return self.add("feedback_loop", id=id, inputs={"src": src}, decay=decay, drift_x_px=drift_x_px, drift_y_px=drift_y_px, blur_radius=blur_radius, pre_roll=pre_roll, loop_frames=loop_frames, blend_mode=blend_mode)

    def particle_sim(self, *, count: int = 48, emitter: str = "bottom", emitter_mask: str | None = None, id: str | None = None, **kw: Any) -> str:
        """Deterministic looping Pixel Particle System (ports: rgba, density, heat)."""
        ins = {"emitter_mask": emitter_mask} if emitter_mask else {}
        return self.add("particle_sim", id=id, inputs=ins, count=count, emitter=emitter, **kw)

    def fluid_smoke_sim(self, *, source_mask: str | None = None, steps: int = 16, id: str | None = None, **kw: Any) -> str:
        """2D Semi-Lagrangian Smoke / Flame / Plasma Fluid Simulation (ports: rgba, density, temperature)."""
        ins = {"source_mask": source_mask} if source_mask else {}
        return self.add("fluid_smoke_sim", id=id, inputs=ins, steps=steps, **kw)

    def anatomical_muscle_scales(self, *, id: str | None = None, **kw: Any) -> str:
        """3D-Anatomical Scale-to-Muscle Texture Fitting MaterialBus generator."""
        return self.add("anatomical_muscle_scales", id=id, **kw)

    def colossal_dragon_membrane(self, span_u: str, trail_v: str, *, face_is_down: str | None = None, id: str | None = None, **kw: Any) -> str:
        """Monumental Normal Dragon Wing Membrane MaterialBus (curved tendon ribs + pleats + tattered arch + backlit SSS)."""
        ins: dict[str, str] = {"span_u": span_u, "trail_v": trail_v}
        if face_is_down:
            ins["face_is_down"] = face_is_down
        return self.add("colossal_dragon_membrane", id=id, inputs=ins, **kw)

    # ------------------------------------------------------------------
    # Declarative JSON Recipe Import / Export (v3.1 Multi-Pass, Multi-Port & Drivers)
    # ------------------------------------------------------------------
    def to_recipe(self) -> dict[str, Any]:
        is_v31 = bool(self.param_drivers)
        is_v3 = bool(is_v31 or self.passes or self.output_material_node or any("." in r for sp in self.nodes.values() for r in sp.inputs.values()))
        rec: dict[str, Any] = {
            "version": "3.1" if is_v31 else ("3.0" if is_v3 else "2.1"),
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "tileable": self.tileable,
            "seed": self.seed,
            "output": self.output_node or (self.node_order[-1] if self.node_order else None),
            "emissive": self.emissive_node,
            "nodes": [self.nodes[nid].to_dict() for nid in self.node_order],
        }
        if self.output_material_node:
            rec["output_material"] = self.output_material_node
        if self.pbr_nodes:
            rec["pbr"] = dict(self.pbr_nodes)
            rec["pbr_config"] = dict(self.pbr_config)
        if self.passes:
            rec["passes"] = [ps.to_dict() for ps in self.passes]
        if self.param_drivers:
            rec["drivers"] = copy.deepcopy(self.param_drivers)
        if self.exposed_params:
            rec["exposed_params"] = copy.deepcopy(self.exposed_params)
        if self.groups:
            rec["groups"] = copy.deepcopy(self.groups)
        if self.frames:
            rec["frames"] = copy.deepcopy(self.frames)
        return rec

    def save_recipe(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_recipe(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_recipe(cls, recipe: dict[str, Any] | Path | str, *, mesh_context: MeshBakeContext | None = None) -> "TexGraph":
        if isinstance(recipe, (str, Path)):
            data = json.loads(Path(recipe).read_text(encoding="utf-8"))
        else:
            data = recipe
        g = cls(
            width=int(data["width"]),
            height=int(data["height"]),
            tileable=bool(data.get("tileable", False)),
            seed=int(data.get("seed", 42)),
            name=str(data.get("name", "recipe")),
            mesh_context=mesh_context,
        )
        for nd in data.get("nodes", []):
            nid = g.add(nd["op"], id=nd["id"], inputs=nd.get("inputs") or {}, validate_refs=False, **(nd.get("params") or {}))
            sp = g.nodes[nid]
            if nd.get("pos") and len(nd["pos"]) >= 2:
                sp.pos = (float(nd["pos"][0]), float(nd["pos"][1]))
            if nd.get("label"):
                sp.label = str(nd["label"])
            sp.collapsed = bool(nd.get("collapsed", False))
            sp.bypassed = bool(nd.get("bypassed", False))
            sp.preview_pinned = bool(nd.get("preview_pinned", False))
        pbr_d = data.get("pbr") or {}
        pbr_cfg = data.get("pbr_config") or {}
        if data.get("output_material"):
            g.set_output_material(
                data["output_material"],
                normal_strength=pbr_cfg.get("normal_strength"),
                dielectric_f0=pbr_cfg.get("dielectric_f0"),
            )
        elif data.get("output"):
            g.set_output(
                data["output"],
                emissive_node=data.get("emissive"),
                height_node=pbr_d.get("height"),
                normal_node=pbr_d.get("normal"),
                roughness_node=pbr_d.get("roughness"),
                metallic_node=pbr_d.get("metallic"),
                ao_node=pbr_d.get("ao"),
                sss_node=pbr_d.get("sss"),
                porosity_node=pbr_d.get("porosity"),
                normal_strength=pbr_cfg.get("normal_strength"),
                dielectric_f0=pbr_cfg.get("dielectric_f0"),
            )
        for pd in data.get("passes") or []:
            g.passes.append(
                PassSpec(
                    name=str(pd["name"]),
                    title=str(pd.get("title", pd["name"])),
                    description=str(pd.get("description", "")),
                    node_ids=list(pd.get("nodes") or []),
                    preview_ref=pd.get("preview"),
                    color=str(pd.get("color", "#1E293B")),
                )
            )
        if data.get("drivers"):
            g.param_drivers = copy.deepcopy(data["drivers"])
        if data.get("exposed_params"):
            g.exposed_params = copy.deepcopy(data["exposed_params"])
        if data.get("groups"):
            g.groups = copy.deepcopy(data["groups"])
        if data.get("frames"):
            g.frames = copy.deepcopy(data["frames"])
        return g

    def to_texdsl(self) -> str:
        """Decompile this TexGraph DAG into clean declarative `.texdsl` source code."""
        from .texdsl import export_texdsl
        return export_texdsl(self)

    def save_texdsl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_texdsl(), encoding="utf-8")

    @classmethod
    def from_texdsl(cls, source_or_path: Path | str, *, mesh_context: MeshBakeContext | None = None) -> "TexGraph":
        """Compile a `.texdsl` source string or file path directly into a runnable TexGraph DAG."""
        from .texdsl import load_texdsl, parse_texdsl

        if isinstance(source_or_path, Path) or (
            isinstance(source_or_path, str)
            and "\n" not in source_or_path
            and source_or_path.endswith(".texdsl")
            and Path(source_or_path).exists()
        ):
            return load_texdsl(source_or_path, mesh_context=mesh_context)
        return parse_texdsl(str(source_or_path), mesh_context=mesh_context)

    @classmethod
    def load(cls, path: Path | str, *, mesh_context: MeshBakeContext | None = None) -> "TexGraph":
        """Load either a `.texdsl` texture program or a `.texgraph.json` recipe seamlessly."""
        p = Path(path)
        if p.suffix.lower() == ".texdsl":
            return cls.from_texdsl(p, mesh_context=mesh_context)
        return cls.from_recipe(p, mesh_context=mesh_context)

    # ------------------------------------------------------------------
    # DAG Evaluation Engine (Multi-Port + Drivers + Static Caching Across Frames)
    # ------------------------------------------------------------------
    def _compute_dynamic_set(self) -> set[str]:
        driven_ids = {d["node_id"] for d in self.param_drivers}
        dyn: set[str] = set()
        for nid in self.node_order:
            sp = self.nodes[nid]
            is_dyn = nid in driven_ids
            if sp.op in (
                "scroll", "pulse", "anim_driver", "feedback_loop", "particle_sim",
                "fluid_smoke_sim", "iterate_inline", "caustic_water", "caustic_noise",
                "curl_flow_noise", "starfield_sparkle", "plasma_wave", "clockwork_gears",
                "mat_crystal_gem", "mat_obsidian_volcanic", "mat_glowing_runes_pbr",
                "anim_shimmer", "anim_flicker_pulse", "anim_flow_warp",
            ):
                is_dyn = True
            elif sp.op in _TIME_AWARE_OPS:
                p = sp.params
                if (
                    abs(float(p.get("loop_speed", 0.0))) > 1e-6
                    or abs(float(p.get("flow_speed", 0.0))) > 1e-6
                    or abs(float(p.get("speed", 0.0))) > 1e-6
                    or abs(float(p.get("rotate_speed", 0.0))) > 1e-6
                    or abs(float(p.get("twinkle_speed", 0.0))) > 1e-6
                    or abs(float(p.get("pulse_speed", 0.0))) > 1e-6
                ):
                    is_dyn = True
                if sp.op == "energy_veins" and abs(float(p.get("flow_speed", 1.0))) > 1e-6:
                    is_dyn = True
                if sp.op == "noise_3d" and any(abs(float(c)) > 1e-6 for c in p.get("flow_vec", (0, 0, 0))):
                    is_dyn = True
                if sp.op == "script_expr":
                    ex_all = f"{p.get('expr', '')} {p.get('expr_g', '')} {p.get('expr_b', '')}"
                    if "t" in ex_all or "phase" in ex_all:
                        is_dyn = True
            if not is_dyn and any(_parse_ref(ref)[0] in dyn for ref in sp.inputs.values()):
                is_dyn = True
            if is_dyn:
                dyn.add(nid)
        return dyn

    def _effective_params(self, sp: NodeSpec, phase: float) -> dict[str, Any]:
        if not self.param_drivers:
            return sp.params
        out: dict[str, Any] | None = None
        for d in self.param_drivers:
            if d["node_id"] == sp.id:
                if out is None:
                    out = dict(sp.params)
                out[d["param_key"]] = eval_anim_driver(
                    phase,
                    kind=d.get("kind", "sine"),
                    min_val=d.get("min_val", 0.0),
                    max_val=d.get("max_val", 1.0),
                    freq=d.get("freq", 1.0),
                    phase_offset=d.get("phase_offset", 0.0),
                    duty=d.get("duty", 0.5),
                    octaves=d.get("octaves", 3),
                    seed=d.get("seed", self.seed),
                    keyframes=d.get("keyframes"),
                )
        return sp.params if out is None else out

    def _eval_node(self, sp: NodeSpec, vals: dict[str, Any], phase: float) -> Any:
        H, W = self.height, self.width
        tile = self.tileable
        p = self._effective_params(sp, phase)
        op = sp.op
        bus_keys = _BUS_INPUT_KEYS.get(op, frozenset())
        ins = {k: _resolve_ref(vals, v, allow_bus=(k in bus_keys)) for k, v in sp.inputs.items()}
        seed = int(self.seed if p.get("seed") is None else p["seed"])

        if op == "reroute":
            if "in" in ins:
                return ins["in"]
            return np.zeros((H, W, 4), dtype=np.float32)

        if op == "constant":
            v = p["value"]
            if isinstance(v, str) and v.startswith("#"):
                r, g, b = hex_to_srgb(v)
                out = np.zeros((H, W, 4), dtype=np.float32)
                out[..., 0], out[..., 1], out[..., 2], out[..., 3] = r, g, b, 1.0
                return out
            return np.full((H, W), float(v), dtype=np.float32)

        if op == "gradient":
            u, v = uv_grid(H, W)
            d = p.get("direction", "vertical")
            if d == "horizontal":
                return u
            if d == "radial":
                return np.clip(1.0 - np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2.0, 0.0, 1.0).astype(np.float32)
            return v

        if op == "value_noise":
            return value_noise(H, W, scale=p.get("scale", 8.0), scale_y=p.get("scale_y"), seed=seed, tileable=p.get("tileable", tile), offset_x=p.get("offset_x", 0.0), offset_y=p.get("offset_y", 0.0))
        if op == "perlin":
            return perlin_noise(H, W, scale=p.get("scale", 8.0), scale_y=p.get("scale_y"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0), offset_x=p.get("offset_x", 0.0), offset_y=p.get("offset_y", 0.0))
        if op == "fbm":
            return fbm_noise(H, W, scale=p.get("scale", 6.0), scale_y=p.get("scale_y"), octaves=p.get("octaves", 4), persistence=p.get("persistence", 0.5), lacunarity=p.get("lacunarity", 2.0), mode=p.get("mode", "fbm"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0), offset_x=p.get("offset_x", 0.0), offset_y=p.get("offset_y", 0.0))
        if op == "voronoi":
            return voronoi_noise(H, W, scale=p.get("scale", 6.0), scale_y=p.get("scale_y"), jitter=p.get("jitter", 0.85), metric=p.get("metric", "euclidean"), output=p.get("output", "edge"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0))
        if op == "gabor":
            return gabor_noise(H, W, scale=p.get("scale", 12.0), angle_deg=p.get("angle_deg", 0.0), anisotropy=p.get("anisotropy", 3.5), octaves=p.get("octaves", 3), seed=seed, tileable=p.get("tileable", tile))
        if op == "domain_warp":
            return domain_warp(ins["src"], strength=p.get("strength", 0.08), scale=p.get("scale", 4.0), octaves=p.get("octaves", 3), mode=p.get("mode", "fbm"), seed=seed, tileable=p.get("tileable", tile), phase=phase, flow_speed=p.get("flow_speed", 0.0))
        if op == "noise_3d":
            if self.mesh_context is None:
                return fbm_noise(H, W, scale=max(2.0, float(p.get("scale", 0.2)) * 24.0), octaves=p.get("octaves", 3), mode="fbm", seed=seed, tileable=tile)
            return noise_3d_field(self.mesh_context.world_pos, scale=p.get("scale", 0.22), octaves=p.get("octaves", 3), mode=p.get("mode", "fbm"), seed=seed, phase=phase, flow_vec=tuple(p.get("flow_vec", (0.0, 0.0, 0.0))))

        if op == "grid":
            return grid_pattern(H, W, cols=p.get("cols", 4), rows=p.get("rows", 4), mortar=p.get("mortar", 0.08), bevel=p.get("bevel", 0.14), seed=seed, output=p.get("output", "height"))
        if op == "brick":
            return brick_pattern(H, W, cols=p.get("cols", 4), rows=p.get("rows", 8), style=p.get("style", "running"), offset=p.get("offset", 0.5), mortar=p.get("mortar", 0.08), bevel=p.get("bevel", 0.16), chip_strength=p.get("chip_strength", 0.12), seed=seed, output=p.get("output", "height"))
        if op == "hex_grid":
            return hex_grid_pattern(H, W, cols=p.get("cols", 4), mortar=p.get("mortar", 0.08), bevel=p.get("bevel", 0.14), seed=seed, output=p.get("output", "height"))
        if op == "tri_grid":
            return tri_grid_pattern(H, W, cols=p.get("cols", 6), mortar=p.get("mortar", 0.06), bevel=p.get("bevel", 0.14), seed=seed, output=p.get("output", "height"))
        if op == "diamond":
            return diamond_chevron_pattern(H, W, cols=p.get("cols", 6), rows=p.get("rows", 6), style=p.get("style", "diamond"), mortar=p.get("mortar", 0.08), bevel=p.get("bevel", 0.14), seed=seed, output=p.get("output", "height"))
        if op == "truchet":
            return truchet_pattern(H, W, cols=p.get("cols", 6), style=p.get("style", "arcs"), thickness=p.get("thickness", 0.22), seed=seed)
        if op == "mosaic":
            return mosaic_pattern(H, W, scale=p.get("scale", 7.0), jitter=p.get("jitter", 0.85), grout=p.get("grout", 0.12), bevel=p.get("bevel", 0.16), seed=seed, tileable=p.get("tileable", tile), output=p.get("output", "height"))
        if op == "imbricated_scales":
            return imbricated_scales_pattern(H, W, cols=p.get("cols", 8), rows=p.get("rows", 10), shape=p.get("shape", "dragon"), keel=p.get("keel", 0.35), seed=seed, output=p.get("output", "height"))
        if op == "feather_barbs":
            return feather_barbs_pattern(H, W, cols=p.get("cols", 4), barb_freq=p.get("barb_freq", 28.0), barb_angle_deg=p.get("barb_angle_deg", 32.0), rachis_width=p.get("rachis_width", 0.10), seed=seed)
        if op == "fur_tufts":
            return fur_tufts_pattern(H, W, cols=p.get("cols", 10), rows=p.get("rows", 8), droop=p.get("droop", 0.35), seed=seed)
        if op == "radial_rings":
            return radial_rings_pattern(H, W, cx=p.get("cx", 0.5), cy=p.get("cy", 0.5), freq=p.get("freq", 6.0), spokes=p.get("spokes", 0), phase=phase, loop_speed=p.get("loop_speed", 0.0))
        if op == "energy_veins":
            return energy_veins_pattern(H, W, scale=p.get("scale", 5.0), thickness=p.get("thickness", 0.16), warp_strength=p.get("warp_strength", 0.10), style=p.get("style", "lightning"), seed=seed, tileable=p.get("tileable", tile), phase=phase, flow_speed=p.get("flow_speed", 1.0))

        if op == "gray_scott":
            return gray_scott_turing(H, W, preset=p.get("preset", "labyrinth"), feed=p.get("feed"), kill=p.get("kill"), du=p.get("du", 0.16), dv=p.get("dv", 0.08), steps=p.get("steps", 320), sim_scale=p.get("sim_scale", 1), aniso_x=p.get("aniso_x", 1.0), aniso_y=p.get("aniso_y", 1.0), seed=seed, tileable=p.get("tileable", tile), seed_mask=ins.get("seed_mask"))
        if op == "cellular_automata":
            return cellular_automata(H, W, preset=p.get("preset", "cave"), born=p.get("born"), survive=p.get("survive"), density=p.get("density", 0.45), steps=p.get("steps", 6), seed=seed, tileable=p.get("tileable", tile), initial_mask=ins.get("initial_mask"))
        if op == "iterate":
            return iterate_process(ins["src"], operation=p.get("operation", "erode_weather"), steps=p.get("steps", 6), strength=p.get("strength", 0.35), seed=seed, tileable=p.get("tileable", tile))
        if op == "weathering_drip":
            return weathering_drip(ins["src"], cavity_mask=ins.get("cavity"), streak_length=p.get("streak_length", 8), strength=p.get("strength", 0.65), decay=p.get("decay", 0.84), seed=seed, tileable=p.get("tileable", tile))

        if op == "kuwahara":
            return anisotropic_kuwahara(ins["src"], radius=p.get("radius", 2), sharpness=p.get("sharpness", 8.0), tileable=p.get("tileable", tile))
        if op == "bilateral":
            return bilateral_denoise(ins["src"], radius=p.get("radius", 2), sigma_s=p.get("sigma_s", 1.5), sigma_r=p.get("sigma_r", 0.14), tileable=p.get("tileable", tile))
        if op == "tv_denoise":
            return total_variation_denoise(ins["src"], weight=p.get("weight", 0.08), iterations=p.get("iterations", 12))
        if op == "nlm_denoise":
            return non_local_means_denoise(ins["src"], patch_radius=p.get("patch_radius", 1), search_radius=p.get("search_radius", 3), h_param=p.get("h_param", 0.08), tileable=p.get("tileable", tile))
        if op == "despeckle":
            return morphological_despeckle(ins["src"], threshold=p.get("threshold", 0.14), min_dissent=p.get("min_dissent", 7), tileable=p.get("tileable", tile))
        if op == "cluster_mode":
            return pixel_cluster_mode(ins["src"], tileable=p.get("tileable", tile))

        if op == "mesh_channel":
            mc = self.mesh_context
            ch = str(p.get("channel", "baked_ao"))
            if mc is None:
                return np.ones((H, W), dtype=np.float32)
            if ch == "baked_ao":
                return mc.baked_ao
            if ch == "edge_bevel":
                return mc.edge_bevel
            if ch == "joint_cavity":
                return mc.joint_cavity
            if ch == "island_mask":
                return mc.island_mask.astype(np.float32)
            if ch == "top_grad":
                return mc.world_pos_norm[..., 1]
            if ch == "front_grad":
                return mc.world_pos_norm[..., 2]
            if ch == "side_sym":
                return np.abs(mc.world_pos_norm[..., 0] * 2.0 - 1.0)
            if ch == "normal_y":
                return np.clip(0.5 + 0.5 * mc.world_normal[..., 1], 0.0, 1.0)
            if ch == "local_u":
                return mc.local_uv[..., 0]
            if ch == "local_v":
                return mc.local_uv[..., 1]
            if ch == "zone":
                return mc.zone_mask(str(p.get("zone", "torso")))
            if ch == "element":
                return mc.match_elements_mask(str(p.get("pattern", ".*")))
            if ch == "outward_eye":
                return mc.outward_lateral_face_mask(str(p.get("pattern", r"(^|/|__)eye")))
            if ch == "face":
                return mc.face_filter_mask(str(p.get("faces", "up,down")))
            if ch == "side_sign":
                return np.sign(mc.elem_center_x_map).astype(np.float32)
            if ch == "wing_span_u":
                return mc.wing_span_u
            if ch == "wing_trail_v":
                return mc.wing_trail_v
            if ch == "face_is_down":
                return mc.face_is_down
            if ch == "wing_bay_u":
                return mc.wing_bay_u
            if ch == "wing_bay_v":
                return mc.wing_bay_v
            if ch in ("baked_shadow", "shadow"):
                return mc.baked_shadow
            if ch == "bent_normal":
                return np.clip(mc.bent_normal * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
            if ch == "thickness":
                return mc.thickness
            if ch == "curvature_3d":
                return mc.curvature_3d
            if ch == "uv_seam":
                return mc.uv_seam_mask
            if ch == "wing_bay_mask":
                return mc.wing_bay_mask
            if ch == "island_edge_px":
                return np.clip(mc.island_edge_px / 8.0, 0.0, 1.0).astype(np.float32)
            if ch == "island_inner_area":
                return mc.island_inner_area
            if ch == "island_size_min":
                return np.clip(mc.island_size_min / 32.0, 0.0, 1.0).astype(np.float32)
            if ch == "face_is_up":
                return mc.face_is_up
            if ch == "face_is_side":
                return mc.face_is_side
            raise ValueError(f"Unknown mesh_channel: {ch!r}")

        if op == "height_to_normal":
            return height_to_normal_map(
                ins["height"],
                strength=p.get("strength", 2.4),
                tileable=p.get("tileable", tile),
                convention=p.get("convention", "opengl"),
            )

        if op == "vibrance_grade":
            return vibrance_grade(
                ins["src"],
                vibrance=p.get("vibrance", 0.22),
                contrast=p.get("contrast", 1.06),
                lift=p.get("lift", 0.015),
                shadow_cool=p.get("shadow_cool", 0.05),
            )

        if op in ("eye_paint_rgba", "eye_paint_emissive"):
            ergba, eem = paint_creature_eye(
                ins["local_u"],
                ins["local_v"],
                side_sign=ins.get("side_sign"),
                iris_inner=p.get("iris_inner", "#FEF08A"),
                iris_mid=p.get("iris_mid", "#22D3EE"),
                iris_outer=p.get("iris_outer", "#0284C7"),
                limbal=p.get("limbal", "#1E1B4B"),
                sclera=p.get("sclera", "#0F172A"),
                pupil=p.get("pupil", "#070514"),
                catchlight=p.get("catchlight", "#FFFFFF"),
                pupil_shape=p.get("pupil_shape", "slit"),
            )
            return ergba if op == "eye_paint_rgba" else eem

        if op in ("wing_scallop_alpha", "wing_scallop_rim"):
            wa, wr = wing_scallop_alpha(
                ins["local_u"],
                ins["local_v"],
                arches=p.get("arches", 2),
                depth=p.get("depth", 0.28),
                rim_width=p.get("rim_width", 0.085),
                side_taper=p.get("side_taper", 0.06),
            )
            return wa if op == "wing_scallop_alpha" else wr

        if op == "set_alpha":
            src = np.asarray(ins["src"], dtype=np.float32).copy()
            if src.ndim == 2:
                src = np.concatenate((np.repeat(src[..., None], 3, axis=-1), np.ones((*src.shape, 1), dtype=np.float32)), axis=-1)
            new_a = np.clip(np.asarray(ins["alpha"], dtype=np.float32), 0.0, 1.0)
            if "mask" in ins:
                m = np.clip(np.asarray(ins["mask"], dtype=np.float32), 0.0, 1.0)
                src[..., 3] = src[..., 3] * (1.0 - m) + new_a * m
            else:
                src[..., 3] = new_a
            return src.astype(np.float32)

        if op == "levels":
            return levels_remap(ins["src"], in_min=p.get("in_min", 0.0), in_max=p.get("in_max", 1.0), gamma=p.get("gamma", 1.0), out_min=p.get("out_min", 0.0), out_max=p.get("out_max", 1.0), invert=p.get("invert", False))

        if op == "math":
            a, b = ins["a"], ins["b"]
            mo = p.get("math_op", "mul")
            if mo == "add":
                return np.clip(a + b, 0.0, 1.0).astype(np.float32)
            if mo == "sub":
                return np.clip(a - b, 0.0, 1.0).astype(np.float32)
            if mo == "max":
                return np.maximum(a, b).astype(np.float32)
            if mo == "min":
                return np.minimum(a, b).astype(np.float32)
            if mo == "lerp":
                t = float(p.get("t", 0.5))
                return (a * (1.0 - t) + b * t).astype(np.float32)
            return np.clip(a * b, 0.0, 1.0).astype(np.float32)

        if op == "directional_shade":
            mc = self.mesh_context if p.get("use_mesh", True) else None
            return directional_shade(
                ins["height"],
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                edge_bevel=mc.edge_bevel if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                light_dir=tuple(p.get("light_dir", (0.38, 0.84, 0.38))),
                bump_strength=p.get("bump_strength", 1.25),
                ao_strength=p.get("ao_strength", 0.55),
                bevel_strength=p.get("bevel_strength", 0.22),
                cavity_strength=p.get("cavity_strength", 0.45),
                rim_strength=p.get("rim_strength", 0.14),
                spec_strength=p.get("spec_strength", 0.18),
                spec_power=p.get("spec_power", 14.0),
                tileable=p.get("tileable", tile),
            )

        if op == "bevel_emboss":
            return pixel_bevel_emboss(ins["height"], strength=p.get("strength", 0.35), light_angle_deg=p.get("light_angle_deg", 135.0), tileable=p.get("tileable", tile))

        if op == "hue_ramp":
            return hue_shifted_ramp(
                ins["src"],
                base_lch=tuple(p.get("base_lch", (0.45, 0.08, 250.0))),
                steps=p.get("steps", 6),
                l_dark_delta=p.get("l_dark_delta", -0.26),
                l_light_delta=p.get("l_light_delta", 0.24),
                hue_shift=tuple(p.get("hue_shift", (-22.0, 18.0))),
                colors=p.get("colors"),
                quantize=p.get("quantize", True),
                dither=p.get("dither", "none"),
                dither_strength=p.get("dither_strength", 0.16),
                alpha_mask=ins.get("alpha_mask"),
            )

        if op == "palette_quantize":
            return palette_quantize(ins["src"], palette_hex=p.get("palette_hex"), dither=p.get("dither", "none"), dither_strength=p.get("dither_strength", 0.14))

        if op == "selout":
            return selout_outline(ins["src"], darken=p.get("darken", 0.52), cool_tint=tuple(p.get("cool_tint", (0.04, 0.02, 0.12))), edge_threshold=p.get("edge_threshold", 0.22), silhouette=p.get("silhouette", True), interior=p.get("interior", True), tileable=p.get("tileable", tile))

        if op == "blend":
            return blend_layers(ins["base"], ins["top"], mode=p.get("mode", "normal"), opacity=p.get("opacity", 1.0), mask=ins.get("mask"))

        if op == "scroll":
            sx = int(round(float(p.get("speed_x", 0.0)) * float(phase) * W))
            sy = int(round(float(p.get("speed_y", 1.0)) * float(phase) * H))
            return np.roll(np.roll(ins["src"], sy, axis=0), sx, axis=1)

        if op == "pulse":
            import math as _m
            wave = 0.5 + 0.5 * _m.sin(2.0 * _m.pi * (float(phase) * float(p.get("freq", 1.0)) + float(p.get("phase_offset", 0.0))))
            fac = float(p.get("min_val", 0.45)) + wave * (float(p.get("max_val", 1.0)) - float(p.get("min_val", 0.45)))
            return np.clip(ins["src"] * np.float32(fac), 0.0, 1.0).astype(np.float32)

        if op in ("emissive_albedo", "emissive_map"):
            alb, em = emissive_extract(
                ins["albedo"],
                ins["mask"],
                emissive_color_rgba=ins.get("color"),
                glow_bleed_px=p.get("glow_bleed_px", 2),
                glow_strength=p.get("glow_strength", 0.35),
                tileable=p.get("tileable", tile),
            )
            return alb if op == "emissive_albedo" else em

        if op == "pad_islands":
            if self.mesh_context is None:
                return ins["src"]
            return self.mesh_context.pad_rgba(ins["src"], pad=p.get("pad"))

        if op == "sprite_mask":
            return _eval_sprite_shape(H, W, str(p.get("shape", "crystal_blade")), phase=phase)

        # --- v3 Smooth Gradient Path, Painterly Studio, Transform, Filter & Color Ops ---
        if op == "gradient_field":
            return gradient_field(H, W, kind=p.get("kind", "linear"), angle_deg=p.get("angle_deg", 90.0), cx=p.get("cx", 0.5), cy=p.get("cy", 0.5), radius=p.get("radius", 0.55), aspect=p.get("aspect", 1.0), repeat=p.get("repeat", 1.0), ping_pong=p.get("ping_pong", False), ease=p.get("ease", "linear"))
        if op == "multi_stop_gradient":
            return multi_stop_gradient(ins["src"], p.get("stops", [(0.0, "#101626"), (1.0, "#F0F6FF")]), space=p.get("space", "oklab"), smooth_blend=p.get("smooth_blend", 1.0), quantize_steps=p.get("quantize_steps", 0), dither=p.get("dither", "none"), dither_strength=p.get("dither_strength", 0.12), alpha_mask=ins.get("alpha_mask"))
        if op == "cosine_gradient":
            return cosine_gradient(ins["src"], a=tuple(p.get("a", (0.5, 0.5, 0.5))), b=tuple(p.get("b", (0.5, 0.5, 0.5))), c=tuple(p.get("c", (1.0, 1.0, 1.0))), d=tuple(p.get("d", (0.0, 0.33, 0.67))), phase=float(p.get("phase_offset", 0.0)) + float(p.get("loop_speed", 0.0)) * phase, alpha_mask=ins.get("alpha_mask"))
        if op == "spline_path":
            return spline_path_gradient(H, W, p.get("points", [(0.1, 0.5), (0.9, 0.5)]), stroke_width=p.get("stroke_width", 0.12), samples_per_seg=p.get("samples_per_seg", 24), closed=p.get("closed", False), output=p.get("output", "ribbon"), falloff=p.get("falloff", "smoothstep"))
        if op == "per_island_gradient":
            if self.mesh_context is None:
                u, v = uv_grid(H, W)
                return per_island_gradient(u, v, direction=p.get("direction", "top_to_bottom"), curve=p.get("curve", 1.0))
            return per_island_gradient(self.mesh_context.local_uv[..., 0], self.mesh_context.local_uv[..., 1], direction=p.get("direction", "top_to_bottom"), curve=p.get("curve", 1.0))
        if op == "body_flow_gradient":
            if self.mesh_context is None:
                return gradient_field(H, W, kind="linear", angle_deg=-90.0)
            return body_flow_gradient(self.mesh_context.world_pos_norm, self.mesh_context.world_normal, axis=p.get("axis", "dorsal_ventral"), bias=p.get("bias", 0.0), gamma=p.get("gamma", 1.0))

        if op == "brushstrokes":
            return directional_brushstrokes(H, W, angle_deg=p.get("angle_deg", 0.0), scale_along=p.get("scale_along", 3.5), scale_across=p.get("scale_across", 16.0), contrast=p.get("contrast", 0.35), seed=seed, tileable=p.get("tileable", tile))
        if op == "layered_plates":
            d = layered_scale_plates(H, W, cols=p.get("cols", 8), rows=p.get("rows", 12), stagger=p.get("stagger", 0.5), gap=p.get("gap", 0.08), top_rim=p.get("top_rim", 0.22), variance=p.get("variance", 0.20), seed=seed)
            return d.get(str(p.get("output", "height")), d["height"])
        if op.startswith("smooth_wing_"):
            wd = smooth_wing_membrane(
                ins["local_u"],
                ins["local_v"],
                palette_stops=p.get("palette_stops", [(0.0, "#190E33"), (0.28, "#3B1D6E"), (0.58, "#7C3AED"), (0.84, "#C084FC"), (1.0, "#A5F3FC")]),
                rim_color=p.get("rim_color", "#67E8F9"),
                arch_depth=p.get("arch_depth", 0.24),
                arches=p.get("arches", 1),
                tatter_strength=p.get("tatter_strength", 0.06),
                fold_count=p.get("fold_count", 5.0),
                fold_strength=p.get("fold_strength", 0.14),
                glow_center_strength=p.get("glow_center_strength", 0.85),
                bone_shadow_width=p.get("bone_shadow_width", 0.24),
                rim_width=p.get("rim_width", 0.055),
                smooth_blend=p.get("smooth_blend", 0.72),
                quantize_steps=p.get("quantize_steps", 8),
                seed=seed,
            )
            sub_k = op.replace("smooth_wing_", "")
            if sub_k == "multi":
                return {
                    "__bus__": np.array([1.0], dtype=np.float32),
                    "albedo": wd["rgba"],
                    "rgba": wd["rgba"],
                    "alpha": wd["alpha"],
                    "rim_mask": wd["rim_mask"],
                    "glow_mask": wd["glow_mask"],
                    "height": wd["height"],
                }
            if sub_k == "rim":
                return wd["rim_mask"]
            if sub_k == "glow":
                return wd["glow_mask"]
            return wd[sub_k]
        if op in ("gradient_eye_rgba", "gradient_eye_emissive"):
            ergba, eem = gradient_eye_paint(
                ins["local_u"],
                ins["local_v"],
                side_sign=ins.get("side_sign"),
                stops=p.get("stops", [(0.0, "#0E4C92"), (0.28, "#06B6D4"), (0.58, "#67E8F9"), (0.82, "#FEF08A"), (1.0, "#FFFFFF")]),
                brow_shadow=p.get("brow_shadow", 0.42),
                angle_deg=p.get("angle_deg", -22.0),
                style=p.get("style", "gradient_glow"),
                rim_color=p.get("rim_color", "#0B1026"),
            )
            return ergba if op == "gradient_eye_rgba" else eem
        if op == "temple_stone":
            d = temple_carved_stone(H, W, cols=p.get("cols", 2), rows=p.get("rows", 4), carve_motif=p.get("carve_motif", True), seed=seed)
            return d.get(str(p.get("output", "height")), d["height"])
        if op == "wood_planks":
            d = stylized_wood_planks(H, W, planks=p.get("planks", 4), seed=seed)
            return d.get(str(p.get("output", "height")), d["height"])
        if op == "foliage_clusters":
            d = stylized_foliage_clusters(H, W, cols=p.get("cols", 5), rows=p.get("rows", 5), seed=seed)
            return d.get(str(p.get("output", "height")), d["height"])

        if op == "curves":
            return curves_remap(ins["src"], p.get("points", [(0.0, 0.0), (1.0, 1.0)]))
        if op == "color_balance":
            return color_balance(ins["src"], shadows=tuple(p.get("shadows", (0.0, 0.0, 0.0))), midtones=tuple(p.get("midtones", (0.0, 0.0, 0.0))), highlights=tuple(p.get("highlights", (0.0, 0.0, 0.0))))
        if op == "oklch_adjust":
            return oklch_adjust(ins["src"], hue_shift_deg=p.get("hue_shift_deg", 0.0), chroma_scale=p.get("chroma_scale", 1.0), lightness_offset=p.get("lightness_offset", 0.0))
        if op == "posterize_soft":
            return posterize_soft(ins["src"], steps=p.get("steps", 6), blend=p.get("blend", 0.5))
        if op == "tile_transform":
            return tile_transform(ins["src"], scale_x=p.get("scale_x", 1.0), scale_y=p.get("scale_y", 1.0), offset_x=p.get("offset_x", 0.0), offset_y=p.get("offset_y", 0.0), rotate_deg=p.get("rotate_deg", 0.0), tileable=p.get("tileable", tile))
        if op == "polar_warp":
            return polar_warp(ins["src"], to_polar=p.get("to_polar", True), tileable=p.get("tileable", tile))
        if op == "twirl_distort":
            return twirl_distort(ins["src"], strength_deg=p.get("strength_deg", 90.0), radius=p.get("radius", 0.5), cx=p.get("cx", 0.5), cy=p.get("cy", 0.5), tileable=p.get("tileable", tile))
        if op == "wave_distort":
            return wave_distort(ins["src"], amp_x=p.get("amp_x", 0.04), amp_y=p.get("amp_y", 0.0), freq_x=p.get("freq_x", 3.0), freq_y=p.get("freq_y", 3.0), phase=phase * float(p.get("loop_speed", 1.0)), tileable=p.get("tileable", tile))
        if op == "gaussian_blur":
            return gaussian_blur(ins["src"], radius=p.get("radius", 2), sigma=p.get("sigma", 0.0), tileable=p.get("tileable", tile))
        if op == "directional_blur":
            return directional_blur(ins["src"], angle_deg=p.get("angle_deg", 0.0), length=p.get("length", 4), tileable=p.get("tileable", tile))
        if op == "unsharp_mask":
            return unsharp_mask(ins["src"], radius=p.get("radius", 1), amount=p.get("amount", 0.65), tileable=p.get("tileable", tile))
        if op == "sobel_edges":
            return sobel_edges(ins["src"], strength=p.get("strength", 1.5), tileable=p.get("tileable", tile))
        if op == "sdf_from_mask":
            return sdf_from_mask(ins["mask"], max_dist_px=p.get("max_dist_px", 12.0), invert=p.get("invert", False))
        if op == "morph_op":
            return morph_op(ins["src"], op=p.get("morph_op", "dilate"), radius=p.get("radius", 1), tileable=p.get("tileable", tile))
        if op == "normal_blend":
            return normal_blend_rnm(ins["base"], ins["detail"], strength=p.get("strength", 1.0))
        if op == "ao_from_height":
            return ao_from_height(ins["height"], radius=p.get("radius", 3), strength=p.get("strength", 1.0), tileable=p.get("tileable", tile))
        if op == "bloom_glow":
            return bloom_glow(ins["src"], threshold=p.get("threshold", 0.72), radius=p.get("radius", 3), intensity=p.get("intensity", 0.55), tileable=p.get("tileable", tile))

        # --- Industry-Standard Multi-Port & First-Class MATERIAL Bus Ops (v4) ---
        if op == "material_make":
            return material_make(
                ins["albedo"],
                height=ins.get("height", p.get("height")),
                normal=ins.get("normal"),
                roughness=ins.get("roughness", p.get("roughness", 0.55)),
                metallic=ins.get("metallic", p.get("metallic", 0.0)),
                ao=ins.get("ao", p.get("ao", 1.0)),
                emissive=ins.get("emissive"),
                sss=ins.get("sss", p.get("sss", 0.0)),
                porosity=ins.get("porosity", p.get("porosity", 0.0)),
                normal_strength=float(p.get("normal_strength", 2.4)),
                tileable=p.get("tileable", tile),
            )
        if op == "material_blend":
            return material_blend(
                ins["base"],
                ins["top"],
                mask=ins.get("mask"),
                opacity=float(p.get("opacity", 1.0)),
                albedo_mode=str(p.get("albedo_mode", "normal")),
                normal_mode=str(p.get("normal_mode", "lerp")),
                preserve_base_alpha=bool(p.get("preserve_base_alpha", False)),
                tileable=p.get("tileable", tile),
            )
        if op == "material_height_blend":
            return material_height_blend(
                ins["base"],
                ins["top"],
                mask=ins.get("mask"),
                height_offset=float(p.get("height_offset", 0.0)),
                contrast=float(p.get("contrast", 6.0)),
                seam_ao_strength=float(p.get("seam_ao_strength", 0.25)),
                normal_mode=str(p.get("normal_mode", "lerp")),
                tileable=p.get("tileable", tile),
            )
        if op == "material_grade":
            mc = self.mesh_context
            return material_grade(
                ins["mat"],
                ao_bake=ins.get("ao_bake"),
                ao_darken=float(p.get("ao_darken", 0.0)),
                shadows=tuple(p.get("shadows", (0.0, 0.0, 0.0))),
                midtones=tuple(p.get("midtones", (0.0, 0.0, 0.0))),
                highlights=tuple(p.get("highlights", (0.0, 0.0, 0.0))),
                hue_shift_deg=float(p.get("hue_shift_deg", 0.0)),
                chroma_scale=float(p.get("chroma_scale", 1.0)),
                lightness_offset=float(p.get("lightness_offset", 0.0)),
                vibrance=float(p.get("vibrance", 0.0)),
                contrast=float(p.get("contrast", 1.0)),
                lift=float(p.get("lift", 0.0)),
                shadow_cool=float(p.get("shadow_cool", 0.0)),
                despeckle_threshold=float(p.get("despeckle_threshold", 0.0)),
                detail_normal=ins.get("detail_normal"),
                detail_normal_strength=float(p.get("detail_normal_strength", 0.5)),
                world_normal=(mc.world_normal if mc is not None else None),
                world_pos_norm=(mc.world_pos_norm if mc is not None else None),
                atmosphere_strength=float(p.get("atmosphere_strength", 0.0)),
                sky_tint=tuple(p.get("sky_tint", (0.14, 0.24, 0.42))),
                bounce_tint=tuple(p.get("bounce_tint", (0.12, 0.08, 0.24))),
                organic_wash_strength=float(p.get("organic_wash_strength", 0.0)),
                seed=seed,
                tileable=p.get("tileable", tile),
            )
        if op == "material_set_alpha":
            return material_set_alpha(ins["mat"], ins["alpha"], mask=ins.get("mask"), tileable=p.get("tileable", tile))
        if op == "material_pad_islands":
            return material_pad_islands(ins["mat"], self.mesh_context, pad=p.get("pad"), tileable=p.get("tileable", tile))
        if op == "sculpted_pinion_wing":
            return sculpted_pinion_wing(
                ins["span_u"],
                ins["trail_v"],
                face_is_down=ins.get("face_is_down"),
                wave=ins.get("wave"),
                vanes=int(p.get("vanes", 5)),
                sweep=float(p.get("sweep", 0.84)),
                envelope_style=str(p.get("envelope_style", "primary")),
                blade_stops=p.get("blade_stops", ((0.00, "#0A091E"), (0.24, "#171A45"), (0.52, "#283D7E"), (0.78, "#4D7EC7"), (0.93, "#99CEFA"), (1.00, "#E3F6FF"))),
                gold_stops=p.get("gold_stops", ((0.00, "#753E14"), (0.45, "#D99527"), (0.82, "#F9DA70"), (1.00, "#FFF9D9"))),
                neon_stops=p.get("neon_stops", ((0.00, "#0284C7"), (0.45, "#22D3EE"), (0.82, "#A5F3FC"), (1.00, "#FFFFFF"))),
                gold_arc=bool(p.get("gold_arc", True)),
                quill_glow_strength=float(p.get("quill_glow_strength", 0.92)),
                smooth_blend=float(p.get("smooth_blend", 0.70)),
                quantize_steps=int(p.get("quantize_steps", 8)),
                seed=seed,
            )
        if op == "smooth_wing_multi":
            wd = smooth_wing_membrane(
                ins["local_u"],
                ins["local_v"],
                palette_stops=p.get("palette_stops", [(0.0, "#190E33"), (0.28, "#3B1D6E"), (0.58, "#7C3AED"), (0.84, "#C084FC"), (1.0, "#A5F3FC")]),
                rim_color=p.get("rim_color", "#67E8F9"),
                arch_depth=p.get("arch_depth", 0.24),
                arches=p.get("arches", 1),
                tatter_strength=p.get("tatter_strength", 0.06),
                fold_count=p.get("fold_count", 5.0),
                fold_strength=p.get("fold_strength", 0.14),
                glow_center_strength=p.get("glow_center_strength", 0.85),
                bone_shadow_width=p.get("bone_shadow_width", 0.24),
                rim_width=p.get("rim_width", 0.055),
                smooth_blend=p.get("smooth_blend", 0.72),
                quantize_steps=p.get("quantize_steps", 8),
                seed=seed,
            )
            return {
                "__bus__": np.array([1.0], dtype=np.float32),
                "albedo": wd["rgba"],
                "rgba": wd["rgba"],
                "alpha": wd["alpha"],
                "rim_mask": wd["rim_mask"],
                "glow_mask": wd["glow_mask"],
                "height": wd["height"],
            }
        if op == "gradient_eye_multi":
            ergba, eem = gradient_eye_paint(
                ins["local_u"],
                ins["local_v"],
                side_sign=ins.get("side_sign"),
                stops=p.get("stops", [(0.0, "#0E4C92"), (0.28, "#06B6D4"), (0.58, "#67E8F9"), (0.82, "#FEF08A"), (1.0, "#FFFFFF")]),
                brow_shadow=p.get("brow_shadow", 0.42),
                angle_deg=p.get("angle_deg", -22.0),
                style=p.get("style", "gradient_glow"),
                rim_color=p.get("rim_color", "#0B1026"),
            )
            return {"albedo": ergba, "rgba": ergba, "emissive_mask": eem}
        if op == "emissive_composite":
            alb, em = emissive_extract(
                ins["albedo"],
                ins["mask"],
                emissive_color_rgba=ins.get("color"),
                glow_bleed_px=p.get("glow_bleed_px", 2),
                glow_strength=p.get("glow_strength", 0.35),
                tileable=p.get("tileable", tile),
            )
            return {"albedo": alb, "rgba": alb, "emissive": em}

        # --- Pixel-Composer-Parity Generate / Iterate / Simulate / Animate Ops (v5) ---
        if op == "anim_driver":
            v = eval_anim_driver(
                phase,
                kind=str(p.get("kind", "sine")),
                min_val=float(p.get("min_val", 0.0)),
                max_val=float(p.get("max_val", 1.0)),
                freq=float(p.get("freq", 1.0)),
                phase_offset=float(p.get("phase_offset", 0.0)),
                duty=float(p.get("duty", 0.5)),
                octaves=int(p.get("octaves", 3)),
                seed=seed,
                keyframes=p.get("keyframes"),
            )
            return np.full((H, W), float(v), dtype=np.float32)

        if op == "iterate_inline":
            return iterate_inline(
                ins["src"],
                mode=str(p.get("mode", "warp_cascade")),
                steps=int(p.get("steps", 5)),
                strength=float(p.get("strength", 0.35)),
                decay=float(p.get("decay", 0.82)),
                step_dx=float(p.get("step_dx", 0.0)),
                step_dy=float(p.get("step_dy", 0.0)),
                step_scale=float(p.get("step_scale", 1.0)),
                step_rotate_deg=float(p.get("step_rotate_deg", 0.0)),
                blend_mode=str(p.get("blend_mode", "screen")),
                seed=seed,
                tileable=p.get("tileable", tile),
                phase=phase,
            )

        if op == "feedback_loop":
            # Evaluate pre-roll history ending deterministically at `phase` so single-frame & flipbook match 100%!
            src_ref = sp.inputs["src"]
            src_base, _ = _parse_ref(src_ref)
            n_loop = max(4, int(p.get("loop_frames", 16)))
            pre = max(1, int(p.get("pre_roll", 6)))
            dt_ph = 1.0 / float(n_loop)
            # Build minimal ancestor topological slice for fast temporal sub-stepping
            anc: set[str] = set()
            stack = [src_base]
            while stack:
                cur_n = stack.pop()
                if cur_n in anc:
                    continue
                anc.add(cur_n)
                for r_in in self.nodes[cur_n].inputs.values():
                    stack.append(_parse_ref(r_in)[0])
            sub_order = [nid for nid in self.node_order if nid in anc]
            dyn_all = self._compute_dynamic_set()

            def _eval_src_at(t_ph: float) -> np.ndarray:
                if abs(((t_ph - phase + 0.5) % 1.0) - 0.5) < 1e-6:
                    return np.asarray(ins["src"], dtype=np.float32)
                sub_vals: dict[str, Any] = {}
                for nid_s in sub_order:
                    if nid_s not in dyn_all and nid_s in vals:
                        sub_vals[nid_s] = vals[nid_s]
                    else:
                        sub_vals[nid_s] = self._eval_node(self.nodes[nid_s], sub_vals, float(t_ph % 1.0))
                return np.asarray(_resolve_ref(sub_vals, src_ref), dtype=np.float32)

            state: np.ndarray | None = None
            dec = float(p.get("decay", 0.80))
            dx_px = int(round(float(p.get("drift_x_px", 0.0))))
            dy_px = int(round(float(p.get("drift_y_px", -1.0))))
            b_rad = int(p.get("blur_radius", 1))
            b_mode = str(p.get("blend_mode", "screen"))
            t_tile = bool(p.get("tileable", tile))
            for k in range(pre, -1, -1):
                t_ph = (phase - k * dt_ph) % 1.0
                cur_f = _eval_src_at(t_ph)
                if state is None:
                    state = cur_f.copy()
                else:
                    prev = state
                    if dx_px != 0 or dy_px != 0:
                        prev = np.roll(np.roll(prev, dy_px, axis=0), dx_px, axis=1)
                    if b_rad > 0:
                        prev = gaussian_blur(prev, radius=b_rad, tileable=t_tile)
                    prev = prev * dec
                    state = blend_layers(prev, cur_f, mode=b_mode, opacity=1.0)
            return np.clip(state, 0.0, 1.0).astype(np.float32)

        if op == "particle_sim":
            return particle_sim_field(
                H, W,
                count=int(p.get("count", 48)),
                emitter=str(p.get("emitter", "bottom")),
                emitter_mask=ins.get("emitter_mask"),
                vel_x=float(p.get("vel_x", 0.0)),
                vel_y=float(p.get("vel_y", -0.55)),
                spread=float(p.get("spread", 0.25)),
                gravity_y=float(p.get("gravity_y", -0.18)),
                turbulence=float(p.get("turbulence", 0.22)),
                vortex=float(p.get("vortex", 0.0)),
                size_px=float(p.get("size_px", 1.6)),
                trail_steps=int(p.get("trail_steps", 3)),
                phase=phase,
                seed=seed,
                tileable=p.get("tileable", tile),
                color_stops=p.get("color_stops"),
            )

        if op == "fluid_smoke_sim":
            return fluid_smoke_sim_field(
                H, W,
                source_mask=ins.get("source_mask"),
                steps=int(p.get("steps", 16)),
                buoyancy=float(p.get("buoyancy", 1.65)),
                curl_strength=float(p.get("curl_strength", 1.85)),
                dissipation=float(p.get("dissipation", 0.93)),
                viscosity=float(p.get("viscosity", 0.15)),
                phase=phase,
                seed=seed,
                tileable=p.get("tileable", tile),
                color_stops=p.get("color_stops"),
            )

        if op == "organic_domain_warp_3d":
            mc = self.mesh_context
            if mc is None:
                u, v = uv_grid(H, W)
                wpn = np.stack((u, 1.0 - v, np.full_like(u, 0.5)), axis=-1)
            else:
                wpn = mc.world_pos_norm
            return organic_domain_warp_3d(
                wpn,
                scale=float(p.get("scale", 3.5)),
                warp_strength=float(p.get("warp_strength", 0.42)),
                octaves=int(p.get("octaves", 3)),
                seed=seed,
            )

        if op == "painterly_brush_field":
            mc = self.mesh_context
            return painterly_brush_field(
                H, W,
                world_pos_norm=(mc.world_pos_norm if mc is not None else None),
                angle_deg=float(p.get("angle_deg", -16.0)),
                stroke_scale=float(p.get("stroke_scale", 12.0)),
                organic_distortion=float(p.get("organic_distortion", 0.58)),
                mottle_weight=float(p.get("mottle_weight", 0.48)),
                seed=seed,
                tileable=p.get("tileable", tile),
            )

        if op == "anatomical_muscle_scales":
            mc = self.mesh_context
            if mc is None:
                u, v = uv_grid(H, W)
                wpn = np.stack((u, 1.0 - v, np.full_like(u, 0.5)), axis=-1)
                wn = np.stack((np.zeros_like(u), np.ones_like(u), np.zeros_like(u)), axis=-1)
                ao_m = np.ones((H, W), dtype=np.float32)
                bev_m = np.zeros((H, W), dtype=np.float32)
                cav_m = np.zeros((H, W), dtype=np.float32)
                luv = np.stack((u, v), axis=-1)
            else:
                wpn, wn, ao_m, bev_m, cav_m, luv = mc.world_pos_norm, mc.world_normal, mc.baked_ao, mc.edge_bevel, mc.joint_cavity, mc.local_uv
            return anatomical_muscle_scales(
                wpn, wn, ao_m, bev_m, cav_m, luv,
                scale_stops=p.get("scale_stops", ((0.00, "#0A0C1D"), (0.24, "#1C2445"), (0.50, "#354C7E"), (0.76, "#5C82B8"), (0.92, "#97C2EB"), (1.00, "#E0F3FF"))),
                belly_stops=p.get("belly_stops", ((0.00, "#2E2733"), (0.42, "#6E646C"), (0.80, "#B8AE9E"), (1.00, "#ECE4D3"))),
                plate_cols=int(p.get("plate_cols", 84)),
                plate_rows=int(p.get("plate_rows", 84)),
                muscle_groove_depth=float(p.get("muscle_groove_depth", 0.42)),
                scale_relief=float(p.get("scale_relief", 0.44)),
                organic_distortion=float(p.get("organic_distortion", 0.64)),
                softness=float(p.get("softness", 0.58)),
                hue_variation=float(p.get("hue_variation", 0.32)),
                roughness_base=float(p.get("roughness_base", 0.38)),
                metallic_crown=float(p.get("metallic_crown", 0.18)),
                smooth_blend=float(p.get("smooth_blend", 0.64)),
                quantize_steps=int(p.get("quantize_steps", 8)),
                seed=seed,
            )

        if op == "colossal_dragon_membrane":
            return colossal_dragon_membrane(
                ins["span_u"],
                ins["trail_v"],
                face_is_down=ins.get("face_is_down"),
                palette_stops=p.get("palette_stops", ((0.00, "#1A0D1C"), (0.24, "#3B182E"), (0.52, "#6E2B3E"), (0.78, "#B45646"), (0.94, "#E9A26E"), (1.00, "#FFF0CC"))),
                dorsal_palette_stops=p.get("dorsal_palette_stops"),
                rim_color=str(p.get("rim_color", "#2A1224")),
                tendon_color=str(p.get("tendon_color", "#251120")),
                dorsal_tendon_color=p.get("dorsal_tendon_color"),
                arch_depth=float(p.get("arch_depth", 0.26)),
                arches=int(p.get("arches", 1)),
                tatter_strength=float(p.get("tatter_strength", 0.09)),
                tendon_ribs=int(p.get("tendon_ribs", 3)),
                pleat_count=float(p.get("pleat_count", 9.0)),
                pleat_strength=float(p.get("pleat_strength", 0.16)),
                organic_distortion=float(p.get("organic_distortion", 0.55)),
                backlit_sss=float(p.get("backlit_sss", 0.85)),
                smooth_blend=float(p.get("smooth_blend", 0.78)),
                quantize_steps=int(p.get("quantize_steps", 8)),
                seed=seed,
            )

        # --- TexGraph v7 Industry Operators & Multi-Domain Bio/Block Generators ---
        if op == "sdf_shape":
            return sdf_shape(
                H, W,
                shape=str(p.get("shape", "rounded_box")),
                center=tuple(p.get("center", (0.5, 0.5))),
                size=tuple(p.get("size", (0.34, 0.34))),
                corner_radius=float(p.get("corner_radius", 0.06)),
                angle_deg=float(p.get("angle_deg", 0.0)),
                sides=int(p.get("sides", 6)),
                inner_ratio=float(p.get("inner_ratio", 0.48)),
                bevel_width=float(p.get("bevel_width", 0.14)),
                outline_width=float(p.get("outline_width", 0.04)),
            )

        if op == "sdf_boolean":
            return sdf_boolean(ins["a"], ins["b"], op=str(p.get("bool_op", p.get("op", "smooth_union"))), smoothness=float(p.get("smoothness", 0.04)))

        if op == "tile_sampler":
            return tile_sampler(
                H, W,
                stamp=ins.get("stamp"),
                count_x=int(p.get("count_x", 5)),
                count_y=int(p.get("count_y", 5)),
                distribution=str(p.get("distribution", "jitter_grid")),
                pos_jitter=float(p.get("pos_jitter", 0.38)),
                scale_base=float(p.get("scale_base", 0.92)),
                scale_jitter=float(p.get("scale_jitter", 0.28)),
                rot_base_deg=float(p.get("rot_base_deg", 0.0)),
                rot_jitter_deg=float(p.get("rot_jitter_deg", 25.0)),
                lum_jitter=float(p.get("lum_jitter", 0.30)),
                density=float(p.get("density", 1.0)),
                density_map=ins.get("density_map"),
                scale_map=ins.get("scale_map"),
                rot_map=ins.get("rot_map"),
                blend_mode=str(p.get("blend_mode", "max")),
                tileable=bool(p.get("tileable", tile)),
                seed=seed,
            )

        if op == "flood_fill":
            return flood_fill(ins["mask"], threshold=float(p.get("threshold", 0.25)), tileable=bool(p.get("tileable", tile)), seed=seed)

        if op == "slope_blur":
            return slope_blur(ins["src"], ins["slope_map"], intensity=float(p.get("intensity", 1.6)), samples=int(p.get("samples", 6)), mode=str(p.get("mode", "min")), tileable=bool(p.get("tileable", tile)))

        if op == "histogram_scan":
            return histogram_scan(ins["src"], position=float(p.get("position", 0.5)), contrast=float(p.get("contrast", 0.6)), invert=bool(p.get("invert", False)))

        if op == "histogram_select":
            return histogram_select(ins["src"], position=float(p.get("position", 0.5)), range_width=float(p.get("range_width", 0.25)), contrast=float(p.get("contrast", 0.5)))

        if op == "auto_levels":
            return auto_levels(ins["src"], low_pct=float(p.get("low_pct", 1.0)), high_pct=float(p.get("high_pct", 99.0)))

        if op == "curvature":
            return curvature_from_height(ins["height"], radius=int(p.get("radius", 2)), strength=float(p.get("strength", 2.2)), tileable=bool(p.get("tileable", tile)))

        if op == "edge_wear":
            return edge_wear_mask(ins["height"], wear_amount=float(p.get("wear_amount", 0.45)), chip_scale=float(p.get("chip_scale", 8.0)), grime_amount=float(p.get("grime_amount", 0.55)), tileable=bool(p.get("tileable", tile)), seed=seed)

        if op == "displace_uv":
            return displace_uv(ins["src"], ins["disp_map"], disp_y_map=ins.get("disp_y_map"), strength_x=float(p.get("strength_x", 2.0)), strength_y=float(p.get("strength_y", 2.0)), mid_value=float(p.get("mid_value", 0.5)), pixel_snap=bool(p.get("pixel_snap", False)), tileable=bool(p.get("tileable", tile)))

        if op == "normal_light_2d":
            return normal_light_2d(
                ins["albedo"], ins["normal"],
                light_az_deg=float(p.get("light_az_deg", -45.0)),
                light_el_deg=float(p.get("light_el_deg", 48.0)),
                light_color=str(p.get("light_color", "#FFF7E6")),
                ambient_color=str(p.get("ambient_color", "#2A3352")),
                ambient_strength=float(p.get("ambient_strength", 0.34)),
                spec_strength=float(p.get("spec_strength", 0.22)),
                shininess=float(p.get("shininess", 18.0)),
                quantize_steps=int(p.get("quantize_steps", 0)),
            )

        if op == "directional_selout":
            return directional_selout(
                ins["src"],
                light_angle_deg=float(p.get("light_angle_deg", -135.0)),
                lit_darken=float(p.get("lit_darken", 0.64)),
                shadow_darken=float(p.get("shadow_darken", 0.30)),
                warm_tint=tuple(p.get("warm_tint", (0.08, 0.04, -0.02))),
                cool_tint=tuple(p.get("cool_tint", (-0.02, 0.01, 0.14))),
                interior_outline=bool(p.get("interior_outline", False)),
                interior_threshold=float(p.get("interior_threshold", 0.32)),
            )

        if op == "cobblestone_masonry":
            kw = {k: v for k, v in p.items() if k != "seed"}
            kw.setdefault("tileable", tile)
            return cobblestone_masonry(H, W, seed=seed, **kw)

        if op == "bark_and_timber":
            kw = {k: v for k, v in p.items() if k != "seed"}
            kw.setdefault("tileable", tile)
            return bark_and_timber(H, W, seed=seed, **kw)

        if op == "crystalline_ore_vein":
            kw = {k: v for k, v in p.items() if k != "seed"}
            kw.setdefault("tileable", tile)
            return crystalline_ore_vein(H, W, seed=seed, **kw)

        if op == "caustic_water":
            kw = {k: v for k, v in p.items() if k not in ("seed", "phase")}
            kw.setdefault("tileable", tile)
            return caustic_water_surface(H, W, phase=phase, seed=seed, **kw)

        if op == "organic_fur_pelt":
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            return organic_fur_pelt(
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                local_uv=mc.local_uv if mc else None,
                seed=seed,
                **kw,
            )

        if op == "layered_plumage":
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            return layered_plumage(
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                seed=seed,
                **kw,
            )

        if op == "chitin_carapace":
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            return chitin_carapace(
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                seed=seed,
                **kw,
            )

        if op == "aquatic_scales_skin":
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            return aquatic_scales_skin(
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                seed=seed,
                **kw,
            )

        if op == "elemental_golem_surface":
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            return elemental_golem_surface(
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                seed=seed,
                **kw,
            )

        # --- TexGraph v8 Character/Orc Materials & Part-Aware Entity Landmark Compositor ---
        if op in ("orc_flesh_skin", "stitched_leather", "forged_battle_metal", "woven_battle_cloth", "keratin_horn_bone"):
            mc = self.mesh_context
            kw = {k: v for k, v in p.items() if k != "seed"}
            fn_map = {
                "orc_flesh_skin": orc_flesh_skin,
                "stitched_leather": stitched_leather,
                "forged_battle_metal": forged_battle_metal,
                "woven_battle_cloth": woven_battle_cloth,
                "keratin_horn_bone": keratin_horn_bone,
            }
            return fn_map[op](
                H, W,
                world_pos_norm=mc.world_pos_norm if mc else None,
                world_normal=mc.world_normal if mc else None,
                baked_ao=mc.baked_ao if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                edge_bevel=mc.edge_bevel if mc else None,
                local_uv=mc.local_uv if mc else None,
                seed=seed,
                **kw,
            )

        if op == "paint_entity_eyes_and_parts":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return paint_entity_eyes_and_parts(ins["base"], self.mesh_context, seed=seed, **kw)

        # --- TexGraph v9 Pixel Composer Atomic Operators & 3D Baked Emissive Bounce ---
        if op == "pixel_dither":
            return pixel_dither(
                ins["src"],
                p.get("palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]),
                pattern=str(p.get("pattern", "bayer_2x2")),
                contrast=float(p.get("contrast", 1.0)),
                mode=str(p.get("mode", "color")),
                invert=bool(p.get("invert", False)),
                seed=seed,
            )

        if op == "palette_posterize":
            return palette_posterize(
                ins["src"],
                p.get("palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]),
                space=str(p.get("space", "oklab")),
                posterize_alpha=bool(p.get("posterize_alpha", True)),
                alpha_threshold=float(p.get("alpha_threshold", 0.5)),
                blend=float(p.get("blend", 1.0)),
            )

        if op == "color_replace":
            return color_replace(
                ins["src"],
                p["src_palette"],
                p["dst_palette"],
                tolerance=float(p.get("tolerance", 0.25)),
            )

        if op == "level_selector":
            return level_selector(
                ins["src"],
                center=float(p.get("center", 0.75)),
                width=float(p.get("width", 0.20)),
                softness=float(p.get("softness", 0.05)),
            )

        if op == "de_stray":
            return de_stray(
                ins["src"],
                strictness=str(p.get("strictness", "medium")),
                tolerance=float(p.get("tolerance", 0.06)),
                iterations=int(p.get("iterations", 1)),
                tileable=bool(p.get("tileable", tile)),
            )

        if op == "de_corner":
            return de_corner(
                ins["src"],
                strictness=str(p.get("strictness", "medium")),
                tolerance=float(p.get("tolerance", 0.06)),
                iterations=int(p.get("iterations", 1)),
                tileable=bool(p.get("tileable", tile)),
            )

        if op == "pixel_clusters":
            return pixel_clusters(
                ins["src"],
                p.get("palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]),
                direction=str(p.get("direction", "isotropic")),
                cluster_size=int(p.get("cluster_size", 2)),
                dither_strength=float(p.get("dither_strength", 0.0)),
                clean_strays=bool(p.get("clean_strays", True)),
                clean_corners=bool(p.get("clean_corners", True)),
                tileable=bool(p.get("tileable", tile)),
            )

        if op == "pixel_bevel_directional":
            mc = self.mesh_context
            return pixel_bevel_directional(
                H, W,
                local_uv=mc.local_uv if mc else None,
                world_normal=mc.world_normal if mc else None,
                joint_cavity=mc.joint_cavity if mc else None,
                light_angle_deg=float(p.get("light_angle_deg", -135.0)),
                bevel_width=float(p.get("bevel_width", 0.11)),
                chip_amount=float(p.get("chip_amount", 0.42)),
                breakup_scale=float(p.get("breakup_scale", 10.0)),
                seed=seed,
            )

        if op == "pixel_outline":
            return pixel_outline(
                ins["src"],
                width=int(p.get("width", 1)),
                position=str(p.get("position", "outside")),
                color=str(p.get("color", "#161224")),
                corner_mode=str(p.get("corner_mode", "4way")),
                alpha_threshold=float(p.get("alpha_threshold", 0.5)),
            )

        if op == "heightmap_shadow":
            return heightmap_shadow(
                ins["height"],
                light_angle_deg=float(p.get("light_angle_deg", -135.0)),
                elevation_deg=float(p.get("elevation_deg", 34.0)),
                max_steps=int(p.get("max_steps", 8)),
                strength=float(p.get("strength", 0.72)),
                tileable=bool(p.get("tileable", tile)),
            )

        if op == "pixel_mirror":
            return pixel_mirror(ins["src"], axis=str(p.get("axis", "x")))

        if op == "pixel_sort":
            return pixel_sort(
                ins["src"],
                axis=str(p.get("axis", "vertical")),
                low_thresh=float(p.get("low_thresh", 0.25)),
                high_thresh=float(p.get("high_thresh", 0.85)),
                reverse=bool(p.get("reverse", False)),
            )

        if op == "chromatic_aberration":
            return chromatic_aberration(
                ins["src"],
                shift_x=int(p.get("shift_x", 1)),
                shift_y=int(p.get("shift_y", 0)),
                strength=float(p.get("strength", 0.65)),
            )

        if op == "vignette":
            return vignette(
                ins["src"],
                strength=float(p.get("strength", 0.35)),
                radius=float(p.get("radius", 0.72)),
                color=str(p.get("color", "#120E1D")),
            )

        if op == "bake_emissive_bounce":
            return bake_emissive_bounce(
                ins["mat"],
                self.mesh_context,
                radius_units=float(p.get("radius_units", 7.5)),
                intensity=float(p.get("intensity", 0.52)),
                max_emitters=int(p.get("max_emitters", 96)),
                posterize_steps=int(p.get("posterize_steps", 0)),
            )

        # --- TexGraph v10 Seamless UV-Geometric Edge Solving & Precomputed Shadow Special PBR ---
        if op == "triplanar_project":
            return triplanar_project(
                ins["src"],
                self.mesh_context,
                scale=float(p.get("scale", 0.08)),
                sharpness=float(p.get("sharpness", 6.0)),
                offset=tuple(p.get("offset", (0.0, 0.0, 0.0))),
                space=str(p.get("space", "world")),
                dither_blend=bool(p.get("dither_blend", False)),
                pixel_snap=bool(p.get("pixel_snap", True)),
                seed=seed,
            )

        if op == "uv_seam_heal":
            return uv_seam_heal(
                ins["src"],
                self.mesh_context,
                strength=float(p.get("strength", 0.72)),
                radius_px=int(p.get("radius_px", 2)),
                heal_normals=bool(p.get("heal_normals", True)),
            )

        if op == "block_edge_transition":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return block_edge_transition(H, W, base=ins.get("base"), seed=seed, **kw)

        if op == "precompute_shadow_pbr":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return precompute_shadow_pbr(ins["mat"], self.mesh_context, **kw)

        # --- TexGraph v11 Pixel Composer Per-UV / Per-Body-Part Region Nodes ---
        if op == "uv_island_coords":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return uv_island_coords(H, W, self.mesh_context, **kw)

        if op == "region_base_fill":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return region_base_fill(H, W, self.mesh_context, seed=seed, **kw)

        if op == "region_restrained_noise":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return region_restrained_noise(ins["mat"], self.mesh_context, seed=seed, **kw)

        if op == "region_bevel_and_ao":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return region_bevel_and_ao(ins["mat"], self.mesh_context, seed=seed, **kw)

        if op == "region_composite":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return region_composite(ins["base"], ins["layer"], self.mesh_context, mask=ins.get("mask"), **kw)

        # --- TexGraph v13 Extended Noise Generators (13 ops) ---
        if op == "simplex_noise":
            return simplex_noise(H, W, scale=p.get("scale", 8.0), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0), offset_x=p.get("offset_x", 0.0), offset_y=p.get("offset_y", 0.0))
        if op == "white_noise":
            return white_noise(H, W, density=p.get("density", 1.0), contrast=p.get("contrast", 1.0), monochrome=p.get("monochrome", True), seed=seed)
        if op == "blue_noise":
            return blue_noise(H, W, scale=p.get("scale", 1.0), seed=seed)
        if op == "bubble_noise":
            return bubble_noise(H, W, scale=p.get("scale", 6.0), jitter=p.get("jitter", 0.85), rim_width=p.get("rim_width", 0.28), style=p.get("style", "dome"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0))
        if op == "crystal_noise":
            return crystal_noise(H, W, scale=p.get("scale", 6.0), sharpness=p.get("sharpness", 2.4), facet_contrast=p.get("facet_contrast", 0.72), seed=seed, tileable=p.get("tileable", tile))
        if op == "scratch_noise":
            return scratch_noise(H, W, count=p.get("count", 36), length=p.get("length", 0.28), width_px=p.get("width_px", 1.0), angle_deg=p.get("angle_deg", -28.0), angle_jitter=p.get("angle_jitter", 35.0), seed=seed, tileable=p.get("tileable", tile))
        if op == "strand_noise":
            return strand_noise(H, W, scale=p.get("scale", 12.0), flow_strength=p.get("flow_strength", 0.14), angle_deg=p.get("angle_deg", 0.0), seed=seed, tileable=p.get("tileable", tile))
        if op == "caustic_noise":
            return caustic_noise(H, W, scale=p.get("scale", 5.0), sharpness=p.get("sharpness", 2.8), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 1.0))
        if op == "fold_marble_noise":
            return fold_marble_noise(H, W, scale=p.get("scale", 4.0), veins=p.get("veins", 5.0), turbulence=p.get("turbulence", 2.6), angle_deg=p.get("angle_deg", 32.0), seed=seed, tileable=p.get("tileable", tile))
        if op == "curl_flow_noise":
            return curl_flow_noise(H, W, scale=p.get("scale", 5.0), octaves=p.get("octaves", 3), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 1.0))
        if op == "grunge_blotch":
            return grunge_blotch(H, W, scale=p.get("scale", 5.0), contrast=p.get("contrast", 1.45), speckle=p.get("speckle", 0.28), seed=seed, tileable=p.get("tileable", tile))
        if op == "starfield_sparkle":
            return starfield_sparkle(H, W, density=p.get("density", 0.08), twinkle_speed=p.get("twinkle_speed", 1.0), cross_glint=p.get("cross_glint", True), seed=seed, phase=phase)
        if op == "plasma_wave":
            return plasma_wave(H, W, freq=p.get("freq", 3.0), distortion=p.get("distortion", 1.4), speed=p.get("speed", 1.0), seed=seed, tileable=p.get("tileable", tile), phase=phase)

        # --- TexGraph v13 Extended Geometric Patterns (12 ops) ---
        if op == "stripes_pattern":
            return stripes_pattern(H, W, count=p.get("count", 8), angle_deg=p.get("angle_deg", 0.0), ratio=p.get("ratio", 0.5), wave_amp=p.get("wave_amp", 0.0), softness=p.get("softness", 0.08), seed=seed, tileable=p.get("tileable", tile))
        if op == "checker_tartan":
            return checker_tartan(H, W, cols=p.get("cols", 8), rows=p.get("rows", 8), style=p.get("style", "checker"), seed=seed)
        if op == "polka_rivets":
            return polka_rivets(H, W, cols=p.get("cols", 6), rows=p.get("rows", 6), radius=p.get("radius", 0.28), stagger=p.get("stagger", True), style=p.get("style", "dome"), seed=seed)
        if op == "weave_chainmail":
            return weave_chainmail(H, W, cols=p.get("cols", 8), rows=p.get("rows", 8), style=p.get("style", "weave"), thickness=p.get("thickness", 0.34), seed=seed)
        if op == "clockwork_gears":
            return clockwork_gears(H, W, teeth=p.get("teeth", 12), rings=p.get("rings", 2), style=p.get("style", "single"), rotate_speed=p.get("rotate_speed", 1.0), seed=seed, phase=phase)
        if op == "circuit_runes":
            return circuit_runes(H, W, scale=p.get("scale", 8), density=p.get("density", 0.55), style=p.get("style", "circuit"), seed=seed)
        if op == "panel_frame_trim":
            return panel_frame_trim(H, W, border_width=p.get("border_width", 0.14), corner_size=p.get("corner_size", 0.24), style=p.get("style", "bevel_frame"), rivets=p.get("rivets", True), seed=seed)
        if op == "polygon_star":
            return polygon_star(H, W, sides=p.get("sides", 6), inner_radius=p.get("inner_radius", 0.5), radius=p.get("radius", 0.38), rotation_deg=p.get("rotation_deg", 0.0), repeat=p.get("repeat", 1), bevel=p.get("bevel", 0.22), seed=seed)
        if op == "crosshatch_engrave":
            return crosshatch_engrave(H, W, freq=p.get("freq", 12.0), angle_deg=p.get("angle_deg", 45.0), cross=p.get("cross", True), weight=p.get("weight", 0.26), seed=seed, tileable=p.get("tileable", tile))
        if op == "crystal_facets":
            return crystal_facets(H, W, scale=p.get("scale", 5.0), facet_contrast=p.get("facet_contrast", 0.78), rim_width=p.get("rim_width", 0.14), seed=seed, tileable=p.get("tileable", tile))
        if op == "maze_labyrinth":
            return maze_labyrinth(H, W, cols=p.get("cols", 8), rows=p.get("rows", 8), wall_width=p.get("wall_width", 0.28), style=p.get("style", "orthogonal"), seed=seed)
        if op == "scallop_shingles":
            return scallop_shingles(H, W, cols=p.get("cols", 6), rows=p.get("rows", 8), overlap=p.get("overlap", 0.45), bevel=p.get("bevel", 0.18), seed=seed)

        # --- TexGraph v13 3D Model, Spatial, Direction & Smart Masks (8 ops) ---
        if op == "mask_uv_part":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_uv_part(H, W, self.mesh_context, **kw)
        if op == "mask_world_position":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_world_position(H, W, self.mesh_context, seed=seed, **kw)
        if op == "mask_world_direction":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_world_direction(H, W, self.mesh_context, seed=seed, **kw)
        if op == "mask_uv_box":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_uv_box(H, W, self.mesh_context, **kw)
        if op == "mask_3d_distance":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_3d_distance(H, W, self.mesh_context, **kw)
        if op == "mask_smart_edge_wear":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_smart_edge_wear(H, W, self.mesh_context, seed=seed, **kw)
        if op == "mask_smart_crevice_dirt":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_smart_crevice_dirt(H, W, self.mesh_context, seed=seed, **kw)
        if op == "mask_combine":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mask_combine(ins["mask_a"], ins.get("mask_b"), **kw)

        # --- TexGraph v13 Industry PBR Material Library (17 ops) ---
        if op == "mat_brass":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_brass(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_bronze_copper":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_bronze_copper(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_gold_precious":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_gold_precious(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_steel_iron":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_steel_iron(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_chrome_titanium":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_chrome_titanium(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_plastic":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_plastic(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_painted_enamel":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_painted_enamel(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_ceramic_porcelain":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_ceramic_porcelain(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_rubber_grip":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_rubber_grip(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_glass":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_glass(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_crystal_gem":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_crystal_gem(H, W, seed=seed, tileable=p.get("tileable", tile), phase=phase, **kw)
        if op == "mat_ice_frost":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_ice_frost(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_dragon_scales_pbr":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_dragon_scales_pbr(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_marble_veined":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_marble_veined(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_obsidian_volcanic":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_obsidian_volcanic(H, W, seed=seed, tileable=p.get("tileable", tile), phase=phase, **kw)
        if op == "mat_velvet_silk":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_velvet_silk(H, W, seed=seed, tileable=p.get("tileable", tile), **kw)
        if op == "mat_glowing_runes_pbr":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mat_glowing_runes_pbr(H, W, seed=seed, tileable=p.get("tileable", tile), phase=phase, **kw)

        # --- TexGraph v13 Material Bus Sequencer & PBR Normal Utilities (8 ops) ---
        if op == "material_split":
            return material_split(ins["mat"], tileable=tile)
        if op == "material_tweak":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return material_tweak(ins["mat"], tileable=tile, **kw)
        if op == "material_region_apply":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return material_region_apply(ins["base"], ins["top"], mask=ins.get("mask"), mesh_ctx=self.mesh_context, tileable=tile, **kw)
        if op == "material_weathering":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return material_weathering(ins["mat"], mesh_ctx=self.mesh_context, seed=seed, tileable=tile, **kw)
        if op == "normal_from_height_multi":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return normal_from_height_multi(ins["height"], tileable=p.get("tileable", tile), **kw)
        if op == "normal_bevel":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return normal_bevel(ins["src"], tileable=p.get("tileable", tile), **kw)
        if op == "normal_to_curvature":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return normal_to_curvature(ins["normal"], tileable=p.get("tileable", tile), **kw)
        if op == "normal_strength":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return normal_strength_op(ins["normal"], **kw)

        # --- TexGraph v13 Dynamic Animation FX (3 ops) ---
        if op == "anim_shimmer":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return anim_shimmer(ins["src"], phase=phase, tileable=tile, **kw)
        if op == "anim_flicker_pulse":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return anim_flicker_pulse(ins["src"], seed=seed, phase=phase, tileable=tile, **kw)
        if op == "anim_flow_warp":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return anim_flow_warp(ins["src"], seed=seed, phase=phase, tileable=tile, **kw)

        # --- TexGraph v15 Blender-Style Math, Vector, Mapping, Utility & 3D/2D Brush Paint Mask (15 ops) ---
        if op == "math_scalar":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return math_scalar(H, W, ins.get("a"), ins.get("b"), **kw)
        if op == "map_range":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return map_range(H, W, ins.get("value"), **kw)
        if op == "clamp_value":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return clamp_value(H, W, ins.get("value"), **kw)
        if op == "mix_blend":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return mix_blend(H, W, ins.get("a"), ins.get("b"), ins.get("factor"), **kw)
        if op == "rgb_combine":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return rgb_combine(H, W, ins.get("r"), ins.get("g"), ins.get("b"), ins.get("a"), **kw)
        if op == "rgb_separate":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return rgb_separate(H, W, ins.get("color"), **kw)
        if op == "vector_math":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return vector_math(H, W, ins.get("vec_a"), ins.get("vec_b"), **kw)
        if op == "uv_mapping":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return uv_mapping(H, W, ins.get("src"), tileable=p.get("tileable", tile), **kw)
        if op == "tex_coord":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return tex_coord(H, W, self.mesh_context, **kw)
        if op == "float_constant":
            return float_constant(H, W, value=p.get("value", 0.5))
        if op == "color_constant":
            return color_constant(H, W, color=p.get("color", "#3B82F6"), alpha=p.get("alpha", 1.0))
        if op == "invert_field":
            return invert_field(H, W, ins.get("src"), fac=p.get("fac", 1.0))
        if op == "fresnel_weight":
            return fresnel_weight(H, W, ins.get("normal"), self.mesh_context, ior=p.get("ior", 1.45), power=p.get("power", 3.0))
        if op == "checker_basic":
            return checker_basic(H, W, scale_x=p.get("scale_x", 8), scale_y=p.get("scale_y", 8), color_a=p.get("color_a", "#E2E8F0"), color_b=p.get("color_b", "#1E293B"))
        if op == "paint_mask":
            kw = {k: v for k, v in p.items() if k != "seed"}
            return paint_mask(H, W, **kw)

        # --- TexGraph v16 Multi-Channel PBR Paint Stack & AST-Sandboxed Script Expression ---
        if op == "paint_layer_stack":
            return paint_layer_stack_op(
                H,
                W,
                base=ins.get("base"),
                layers_json=p.get("layers_json", ""),
                height_normal_strength=float(p.get("height_normal_strength", 2.6)),
                tileable=tile,
            )
        if op == "script_expr":
            return script_expr_op(
                H,
                W,
                a=ins.get("a"),
                b=ins.get("b"),
                expr=str(p.get("expr", "0.5 + 0.5 * sin(r_dist * 18.0 - t * tau)")),
                expr_g=str(p.get("expr_g", "")),
                expr_b=str(p.get("expr_b", "")),
                mode=str(p.get("mode", "scalar")),
                param_k=float(p.get("param_k", 1.0)),
                phase=phase,
                seed=seed,
            )

        # --- TexGraph v15 Nested Compound Subgraph Group (`group_input` / `subgraph_group`) ---
        if op == "group_input":
            pname = str(p.get("port_name", "in_0"))
            if pname in self._external_inputs:
                return self._external_inputs[pname]
            return np.full((H, W, 4), 0.5, dtype=np.float32)

        if op == "subgraph_group":
            sub_rec = p.get("sub_recipe")
            if not isinstance(sub_rec, dict) or not sub_rec.get("nodes"):
                return next(iter(ins.values())) if ins else np.full((H, W, 4), 0.35, dtype=np.float32)
            rec_copy = copy.deepcopy(sub_rec)
            rec_copy["width"] = W
            rec_copy["height"] = H
            rec_copy["tileable"] = tile
            sub_g = TexGraph.from_recipe(rec_copy, mesh_context=self.mesh_context)
            sub_g._external_inputs = dict(ins)
            sub_ev = sub_g.evaluate(phase=phase, capture_intermediates=True, build_pbr=False, allow_orphan_errors=True)
            out_id = sub_g.output_material_node or (
                _parse_ref(sub_g.output_node)[0] if sub_g.output_node else (sub_g.node_order[-1] if sub_g.node_order else None)
            )
            if out_id and out_id in sub_ev.intermediates:
                return sub_ev.intermediates[out_id]
            return sub_ev.output

        raise ValueError(f"Unsupported TexGraph op: {op!r}")

    def group_nodes_into_subgraph(
        self,
        node_ids: list[str],
        *,
        title: str = "Node Group",
        color: str = "#7E22CE",
    ) -> str:
        """Collapse selected nodes into a single nested `subgraph_group` compound node (Ctrl+G)."""
        sel_ordered = [nid for nid in self.node_order if nid in set(node_ids) and nid in self.nodes]
        if len(sel_ordered) < 1:
            raise ValueError("Select at least one node to group into a compound subgraph")
        sel_set = set(sel_ordered)

        # Detect external inputs entering sel_set and primary output leaving sel_set
        ext_in_refs: dict[str, str] = {}  # ext_ref -> port_name (in_0, in_1, ...)
        parent_inputs: dict[str, str] = {}
        sub_nodes: list[dict[str, Any]] = []

        # Compute centroid position of grouped nodes
        xs = [float(self.nodes[nid].pos[0]) for nid in sel_ordered if self.nodes[nid].pos is not None]
        ys = [float(self.nodes[nid].pos[1]) for nid in sel_ordered if self.nodes[nid].pos is not None]
        cx = sum(xs) / len(xs) if xs else 260.0
        cy = sum(ys) / len(ys) if ys else 160.0

        # Find output node inside sel_set (referenced by outside node or graph output, else last selected)
        out_candidates: list[str] = []
        for nid, sp in self.nodes.items():
            if nid in sel_set:
                continue
            for ref in sp.inputs.values():
                base, _ = _parse_ref(ref)
                if base in sel_set and base not in out_candidates:
                    out_candidates.append(base)
        for g_ref in (self.output_material_node, self.output_node):
            if g_ref and _parse_ref(g_ref)[0] in sel_set:
                b = _parse_ref(g_ref)[0]
                if b not in out_candidates:
                    out_candidates.append(b)
        sub_out_id = out_candidates[-1] if out_candidates else sel_ordered[-1]

        # Build internal sub_recipe nodes, replacing incoming external edges with `group_input` nodes
        for nid in sel_ordered:
            sp = self.nodes[nid]
            nd = copy.deepcopy(sp.to_dict())
            new_ins: dict[str, str] = {}
            for pk, raw_ref in (sp.inputs or {}).items():
                base, _ = _parse_ref(raw_ref)
                if base in sel_set:
                    new_ins[pk] = raw_ref
                else:
                    if raw_ref not in ext_in_refs:
                        p_slot = f"in_{len(ext_in_refs)}"
                        ext_in_refs[raw_ref] = p_slot
                        parent_inputs[p_slot] = raw_ref
                        gin_id = f"group_{p_slot}"
                        sub_nodes.append({
                            "id": gin_id,
                            "op": "group_input",
                            "inputs": {},
                            "params": {"port_name": p_slot},
                            "pos": [40.0, 80.0 + len(ext_in_refs) * 140.0],
                            "label": f"In: {raw_ref}",
                        })
                    new_ins[pk] = f"group_{ext_in_refs[raw_ref]}"
            nd["inputs"] = new_ins
            sub_nodes.append(nd)

        from .registry import NODE_CATALOG as _NC

        out_meta = _NC.get(self.nodes[sub_out_id].op)
        is_mat_out = (
            self.output_material_node == sub_out_id
            or (out_meta is not None and out_meta.output_type == "MATERIAL")
        )
        sub_recipe: dict[str, Any] = {
            "name": title,
            "width": self.width,
            "height": self.height,
            "tileable": self.tileable,
            "seed": self.seed,
            "nodes": sub_nodes,
            "output": sub_out_id,
            "output_material": sub_out_id if is_mat_out else None,
        }

        base_gid = "group"
        gid = base_gid
        idx = 1
        while gid in self.nodes:
            gid = f"{base_gid}_{idx}"
            idx += 1

        grp_spec = NodeSpec(
            id=gid,
            op="subgraph_group",
            inputs=parent_inputs,
            params={"title": title, "color": color, "sub_recipe": sub_recipe},
            pos=(round(cx, 1), round(cy, 1)),
            label=title,
        )

        # Replace grouped nodes in parent graph and rewire outgoing references to gid
        first_idx = min(self.node_order.index(nid) for nid in sel_ordered)
        for nid in sel_ordered:
            self.nodes.pop(nid, None)
        self.node_order = [nid for nid in self.node_order if nid not in sel_set]
        self.nodes[gid] = grp_spec
        self.node_order.insert(min(first_idx, len(self.node_order)), gid)

        for sp in self.nodes.values():
            if sp.id == gid:
                continue
            for pk, raw_ref in list(sp.inputs.items()):
                base, subp = _parse_ref(raw_ref)
                if base in sel_set:
                    sp.inputs[pk] = f"{gid}.{subp}" if subp else gid

        if self.output_node and _parse_ref(self.output_node)[0] in sel_set:
            subp = _parse_ref(self.output_node)[1]
            self.output_node = f"{gid}.{subp}" if subp else gid
        if self.output_material_node and self.output_material_node in sel_set:
            self.output_material_node = gid
            self.pbr_nodes = {k: f"{gid}.{k}" for k in ("height", "normal", "roughness", "metallic", "ao", "shadow", "bent_normal", "thickness", "sss", "porosity")}
            self.emissive_node = f"{gid}.emissive"
        for fr in self.frames:
            fr["node_ids"] = [nid for nid in (fr.get("node_ids") or []) if nid not in sel_set]
        self.frames = [fr for fr in self.frames if fr.get("node_ids")]
        self._static_cache.clear()
        self._content_cache.clear()
        return gid

    def ungroup_subgraph_node(self, group_node_id: str) -> list[str]:
        """Expand a `subgraph_group` compound node cleanly back into its parent graph (Ctrl+Alt+G)."""
        if group_node_id not in self.nodes or self.nodes[group_node_id].op != "subgraph_group":
            raise ValueError(f"Node {group_node_id!r} is not a subgraph_group")
        grp = self.nodes[group_node_id]
        sub_rec = grp.params.get("sub_recipe") or {}
        raw_nodes = [n for n in (sub_rec.get("nodes") or []) if n.get("op") != "group_input"]
        if not raw_nodes:
            return []

        # Map internal `group_in_k` node IDs back to parent wire refs
        gin_map: dict[str, str] = {}
        for n in (sub_rec.get("nodes") or []):
            if n.get("op") == "group_input":
                p_slot = str((n.get("params") or {}).get("port_name", "in_0"))
                if p_slot in grp.inputs:
                    gin_map[str(n["id"])] = grp.inputs[p_slot]

        id_map: dict[str, str] = {}
        for nd in raw_nodes:
            orig = str(nd["id"])
            nid = orig
            idx = 1
            while nid in self.nodes or nid in id_map.values():
                nid = f"{orig}_{idx}"
                idx += 1
            id_map[orig] = nid

        insert_idx = self.node_order.index(group_node_id)
        self.nodes.pop(group_node_id, None)
        self.node_order.remove(group_node_id)

        restored: list[str] = []
        for offset, nd in enumerate(raw_nodes):
            nid = id_map[str(nd["id"])]
            mapped_in: dict[str, str] = {}
            for pk, raw_ref in (nd.get("inputs") or {}).items():
                base, subp = _parse_ref(str(raw_ref))
                if base in gin_map:
                    mapped_in[pk] = gin_map[base]
                elif base in id_map:
                    mapped_in[pk] = f"{id_map[base]}.{subp}" if subp else id_map[base]
            pos = nd.get("pos") or [(grp.pos[0] if grp.pos else 200.0) + offset * 240.0, grp.pos[1] if grp.pos else 140.0]
            sp = NodeSpec(
                id=nid,
                op=str(nd["op"]),
                inputs=mapped_in,
                params=copy.deepcopy(nd.get("params") or {}),
                pos=(float(pos[0]), float(pos[1])),
                label=str(nd.get("label") or ""),
            )
            self.nodes[nid] = sp
            self.node_order.insert(insert_idx + offset, nid)
            restored.append(nid)

        sub_out_orig = str(
            sub_rec.get("output_material")
            or sub_rec.get("output")
            or sub_rec.get("output_material_node")
            or sub_rec.get("output_node")
            or raw_nodes[-1]["id"]
        )
        sub_out_new = id_map.get(_parse_ref(sub_out_orig)[0], restored[-1])

        for sp in self.nodes.values():
            for pk, raw_ref in list(sp.inputs.items()):
                base, subp = _parse_ref(raw_ref)
                if base == group_node_id:
                    sp.inputs[pk] = f"{sub_out_new}.{subp}" if subp else sub_out_new

        if self.output_node and _parse_ref(self.output_node)[0] == group_node_id:
            subp = _parse_ref(self.output_node)[1]
            self.output_node = f"{sub_out_new}.{subp}" if subp else sub_out_new
        if self.output_material_node == group_node_id:
            self.output_material_node = sub_out_new
            self.pbr_nodes = {k: f"{sub_out_new}.{k}" for k in ("height", "normal", "roughness", "metallic", "ao", "shadow", "bent_normal", "thickness", "sss", "porosity")}
            self.emissive_node = f"{sub_out_new}.emissive"

        self._static_cache.clear()
        self._content_cache.clear()
        return restored

    def evaluate(
        self,
        *,
        phase: float = 0.0,
        target: str | None = None,
        capture_intermediates: bool = False,
        build_pbr: bool | None = None,
        light_az_deg: float = -38.0,
        light_el_deg: float = 52.0,
        allow_orphan_errors: bool = False,
    ) -> EvalResult:
        if not self.node_order:
            raise ValueError("TexGraph has no nodes")
        exec_order = self.topological_order()
        reachable: set[str] | None = None
        if allow_orphan_errors:
            reachable = set()
            roots: list[str] = []
            for r_ref in (target, self.output_node, self.output_material_node, self.emissive_node, *self.pbr_nodes.values()):
                if r_ref:
                    roots.append(_parse_ref(r_ref)[0])
            if not roots:
                roots.append(exec_order[-1])
            stack = list(roots)
            while stack:
                cur = stack.pop()
                if cur in reachable or cur not in self.nodes:
                    continue
                reachable.add(cur)
                for ref in self.nodes[cur].inputs.values():
                    stack.append(_parse_ref(ref)[0])

        dyn_ids = self._compute_dynamic_set()
        vals: dict[str, Any] = {}
        node_hashes: dict[str, str] = {}
        timings: dict[str, float] = {}
        cache_hits = 0
        cache_misses = 0

        for nid in exec_order:
            sp = self.nodes[nid]
            eff_p = self._effective_params(sp, float(phase))
            dep_h = {k: f"{node_hashes.get(_parse_ref(ref)[0], '')}:{_parse_ref(ref)[1] or ''}" for k, ref in sp.inputs.items()}
            p_tag = f"{float(phase):.6f}" if nid in dyn_ids else "static"
            chash = self._node_content_hash(sp, eff_p, dep_h, p_tag)
            node_hashes[nid] = chash

            cached = self._content_cache.get(nid)
            if cached is not None and cached[0] == chash:
                vals[nid] = cached[1]
                timings[nid] = 0.0
                cache_hits += 1
                continue

            t0 = time.perf_counter()
            try:
                if sp.bypassed and sp.inputs:
                    # GUI non-destructive bypass toggle: pass primary input directly through at 0.0ms!
                    first_key = next(iter(sp.inputs))
                    bus_keys = _BUS_INPUT_KEYS.get(sp.op, frozenset())
                    res = _resolve_ref(vals, sp.inputs[first_key], allow_bus=(first_key in bus_keys))
                else:
                    res = self._eval_node(sp, vals, float(phase))
            except ScriptSecurityError:
                raise
            except (KeyError, ValueError, TypeError):
                if allow_orphan_errors and reachable is not None and nid not in reachable:
                    # Unconnected freshly spawned node in interactive GUI: keep output live with neutral placeholder!
                    ph = np.zeros((self.height, self.width, 4), dtype=np.float32)
                    ph[..., :3] = 0.16
                    ph[..., 3] = 1.0
                    vals[nid] = ph
                    timings[nid] = 0.0
                    continue
                raise
            timings[nid] = round((time.perf_counter() - t0) * 1000.0, 2)
            vals[nid] = res
            self._content_cache[nid] = (chash, res)
            if nid not in dyn_ids:
                self._static_cache[nid] = res
            cache_misses += 1

        out_ref = target or self.output_node or exec_order[-1]
        out_arr = _resolve_ref(vals, out_ref)
        em_arr = (
            _resolve_ref(vals, self.emissive_node)
            if (self.emissive_node and _parse_ref(self.emissive_node)[0] in vals)
            else None
        )
        pbr_ch: dict[str, np.ndarray] = {}
        for k, ref in self.pbr_nodes.items():
            base_r, port_r = _parse_ref(ref)
            if base_r not in vals:
                continue
            v_r = vals[base_r]
            if port_r is not None and isinstance(v_r, dict) and port_r not in v_r and k in ("shadow", "bent_normal", "thickness"):
                continue
            pbr_ch[k] = _resolve_ref(vals, ref)

        want_pbr = bool(self.pbr_nodes) if build_pbr is None else bool(build_pbr)
        pbr_maps = None
        pbr_preview = None
        if want_pbr and out_arr.ndim == 3 and out_arr.shape[2] >= 3:
            def _pad_opt(a: np.ndarray | None) -> np.ndarray | None:
                if a is None or self.mesh_context is None:
                    return a
                return self.mesh_context.pad_rgba(a)

            h_ch = _pad_opt(pbr_ch.get("height"))
            n_ch = _pad_opt(pbr_ch.get("normal"))
            r_ch = _pad_opt(pbr_ch.get("roughness"))
            m_ch = _pad_opt(pbr_ch.get("metallic"))
            ao_ch = _pad_opt(pbr_ch.get("ao", self.mesh_context.baked_ao if self.mesh_context else None))
            sh_ch = _pad_opt(pbr_ch.get("shadow", self.mesh_context.baked_shadow if self.mesh_context else None))
            bn_ch = _pad_opt(pbr_ch.get("bent_normal", self.mesh_context.bent_normal if self.mesh_context else None))
            th_ch = _pad_opt(pbr_ch.get("thickness", self.mesh_context.thickness if self.mesh_context else None))
            sss_ch = _pad_opt(pbr_ch.get("sss"))
            por_ch = _pad_opt(pbr_ch.get("porosity"))
            em_ch = _pad_opt(em_arr)

            n_str = float(self.pbr_config.get("normal_strength", 2.4))
            d_f0 = float(self.pbr_config.get("dielectric_f0", 0.04))
            pbr_maps = pack_labpbr_1_3(
                out_arr,
                height=h_ch,
                normal_rgb=n_ch,
                roughness=r_ch,
                metallic=m_ch,
                ao=ao_ch,
                shadow=sh_ch,
                bent_normal=bn_ch,
                thickness=th_ch,
                porosity=por_ch,
                sss=sss_ch,
                emissive=em_ch,
                dielectric_f0=d_f0,
                normal_strength=n_str,
                tileable=self.tileable,
            )
            pbr_preview = render_pbr_preview_2d(
                out_arr,
                normal_rgb=pbr_maps["normal_rgb"],
                height=h_ch,
                roughness=r_ch,
                metallic=m_ch,
                ao=ao_ch,
                sss=sss_ch,
                emissive_rgba=em_ch,
                light_az_deg=light_az_deg,
                light_el_deg=light_el_deg,
                normal_strength=n_str,
                tileable=self.tileable,
            )

        return EvalResult(
            output=out_arr,
            emissive=em_arr,
            intermediates=dict(vals) if capture_intermediates else {},
            timings_ms=timings,
            pbr_channels=pbr_ch,
            pbr_maps=pbr_maps,
            pbr_preview=pbr_preview,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def evaluate_multipass(
        self,
        *,
        phase: float = 0.0,
        build_pbr: bool = True,
        light_az_deg: float = -38.0,
        light_el_deg: float = 52.0,
    ) -> MultiPassResult:
        """Evaluate the full DAG and capture checkpoint previews & timings for every named pipeline Pass."""
        ev = self.evaluate(
            phase=phase,
            capture_intermediates=True,
            build_pbr=build_pbr,
            light_az_deg=light_az_deg,
            light_el_deg=light_el_deg,
        )
        snapshots: list[PassSnapshot] = []
        if self.passes:
            for ps in self.passes:
                ref = ps.preview_ref or (ps.node_ids[-1] if ps.node_ids else None)
                prev_arr = None
                if ref is not None and _parse_ref(ref)[0] in ev.intermediates:
                    prev_arr = _resolve_ref(ev.intermediates, ref)
                t_ms = sum(ev.timings_ms.get(nid, 0.0) for nid in ps.node_ids)
                snapshots.append(
                    PassSnapshot(
                        name=ps.name,
                        title=ps.title,
                        description=ps.description,
                        preview_ref=ref,
                        preview_rgba=prev_arr,
                        node_count=len(ps.node_ids),
                        time_ms=round(t_ms, 2),
                    )
                )
        else:
            snapshots.append(
                PassSnapshot(
                    name="main",
                    title="Main Graph Output",
                    description="",
                    preview_ref=self.output_node,
                    preview_rgba=ev.output,
                    node_count=len(self.node_order),
                    time_ms=round(sum(ev.timings_ms.values()), 2),
                )
            )
        return MultiPassResult(
            eval_result=ev,
            passes=snapshots,
            total_ms=round(sum(ev.timings_ms.values()), 2),
        )

    def render_multipass_sheet(
        self,
        out_path: Path | str | None = None,
        *,
        phase: float = 0.0,
        title: str | None = None,
        tile_size: int = 256,
    ) -> tuple[MultiPassResult, Any]:
        """Evaluate all stages and render a dark-studio Multi-Pass Progression Contact Sheet PNG."""
        mp = self.evaluate_multipass(phase=phase, build_pbr=True)
        img = render_multipass_sheet(
            mp,
            title=title or f"{self.name} | Multi-Pass Pipeline ({self.width}x{self.height})",
            tile_size=tile_size,
        )
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, format="PNG", optimize=True)
        return mp, img

    def export_pbr_bundle(
        self,
        out_dir: Path | str,
        stem: str | None = None,
        *,
        phase: float = 0.0,
        include_standalone: bool = True,
        save_material_sheet: bool = True,
    ) -> dict[str, Any]:
        """Evaluate graph with full PBR channels and write LabPBR 1.3 + Bedrock MER + standard PBR bundle."""
        ev = self.evaluate(phase=phase, build_pbr=True)
        assert ev.pbr_maps is not None and ev.pbr_preview is not None
        st = stem or self.name
        written = export_pbr_bundle(
            ev.pbr_maps,
            out_dir,
            st,
            emissive_rgba=ev.emissive,
            pbr_preview_rgba=ev.pbr_preview,
            include_standalone=include_standalone,
        )
        if save_material_sheet:
            sheet = render_pbr_material_sheet(
                ev.pbr_maps,
                ev.pbr_preview,
                title=f"{st} | Full PBR Breakdown ({self.width}x{self.height})",
            )
            sp = Path(out_dir) / f"{st}_pbr_sheet.png"
            sheet.save(sp, format="PNG", optimize=True)
            written["pbr_sheet"] = str(sp)
        return {"files": written, "eval": ev}

    def export_baked_pbr(
        self,
        out_dir: Path | str,
        stem: str | None = None,
        *,
        phase: float = 0.0,
        preset: str = "studio_hero",
        bake_params: dict[str, Any] | None = None,
        bbmodel_doc: dict[str, Any] | None = None,
        recipe_path: str | None = None,
        embed_maps_in_manifest: bool = False,
    ) -> dict[str, Any]:
        """Evaluate full PBR channels and export Baked Studio Albedo (`_baked.png`), `.synccraft.pbr.json` manifest & optional `_baked.bbmodel`!"""
        ev = self.evaluate(phase=phase, build_pbr=True)
        assert ev.pbr_maps is not None
        st = stem or self.name
        res = export_baked_pbr_package(
            ev.pbr_maps,
            out_dir,
            st,
            emissive_rgba=ev.emissive,
            preset=preset,
            bake_params=bake_params,
            bbmodel_doc=bbmodel_doc,
            recipe_path=recipe_path,
            embed_maps_in_manifest=embed_maps_in_manifest,
        )
        res["eval"] = ev
        return res

    def render_pbr_flipbook(
        self,
        n_frames: int = 16,
        *,
        out_dir: Path | str,
        stem: str | None = None,
        frametime: int = 2,
        interpolate: bool = False,
        frames_schedule: list[int | dict[str, int]] | None = None,
        gif_scale: int = 4,
        rotate_light_in_preview: bool = False,
    ) -> dict[str, Any]:
        """Render synchronized multi-channel animated PBR `.mcmeta` flipbooks (Albedo, Emissive _e, LabPBR _n, _s, Bedrock _mer + PBR Lit GIF)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        st = stem or self.name
        n = max(1, int(n_frames))
        sched = frames_schedule if frames_schedule is not None else self.animation_config.get("frames_schedule")
        alb_fr: list[np.ndarray] = []
        em_fr: list[np.ndarray] = []
        has_em = False
        n_fr: list[np.ndarray] = []
        s_fr: list[np.ndarray] = []
        mer_fr: list[np.ndarray] = []
        lit_fr: list[np.ndarray] = []
        for k in range(n):
            ph = float(k) / float(n)
            az = (-38.0 + 360.0 * ph) if rotate_light_in_preview else -38.0
            ev = self.evaluate(phase=ph, build_pbr=True, light_az_deg=az)
            assert ev.pbr_maps is not None and ev.pbr_preview is not None
            alb_fr.append(ev.pbr_maps["albedo"].astype(np.float32) / 255.0)
            if ev.emissive is not None and float(np.max(ev.emissive[..., 3])) > 0.01:
                has_em = True
                em_fr.append(ev.emissive)
            else:
                em_fr.append(np.zeros((self.height, self.width, 4), dtype=np.float32))
            n_fr.append(ev.pbr_maps["labpbr_n"].astype(np.float32) / 255.0)
            s_fr.append(ev.pbr_maps["labpbr_s"].astype(np.float32) / 255.0)
            mer_fr.append(ev.pbr_maps["bedrock_mer"].astype(np.float32) / 255.0)
            lit_fr.append(ev.pbr_preview)

        fb_alb = export_flipbook(alb_fr, out_png_path=d / f"{st}_flipbook.png", frametime=frametime, interpolate=interpolate, frames_schedule=sched, out_gif_path=d / f"{st}.gif", gif_scale=gif_scale)
        fb_n = export_flipbook(n_fr, out_png_path=d / f"{st}_n_flipbook.png", frametime=frametime, interpolate=interpolate, frames_schedule=sched)
        fb_s = export_flipbook(s_fr, out_png_path=d / f"{st}_s_flipbook.png", frametime=frametime, interpolate=interpolate, frames_schedule=sched)
        fb_mer = export_flipbook(mer_fr, out_png_path=d / f"{st}_mer_flipbook.png", frametime=frametime, interpolate=interpolate, frames_schedule=sched)
        fb_lit = export_flipbook(lit_fr, out_gif_path=d / f"{st}_pbr_lit.gif", frametime=frametime, frames_schedule=sched, gif_scale=gif_scale)
        fb_em = None
        if has_em:
            fb_em = export_flipbook(em_fr, out_png_path=d / f"{st}_e_flipbook.png", frametime=frametime, interpolate=interpolate, frames_schedule=sched)
        res: dict[str, Any] = {
            "num_frames": n,
            "albedo": fb_alb,
            "labpbr_n": fb_n,
            "labpbr_s": fb_s,
            "bedrock_mer": fb_mer,
            "pbr_lit_gif": str(d / f"{st}_pbr_lit.gif"),
            "albedo_frames": alb_fr,
            "pbr_lit_frames": lit_fr,
        }
        if fb_em is not None:
            res["emissive"] = fb_em
            res["emissive_frames"] = em_fr
        return res

    def render_flipbook(
        self,
        n_frames: int = 16,
        *,
        frametime: int = 2,
        interpolate: bool = False,
        frames_schedule: list[int | dict[str, int]] | None = None,
        out_png_path: Path | str | None = None,
        out_gif_path: Path | str | None = None,
        gif_scale: int = 4,
        target: str | None = None,
    ) -> dict[str, Any]:
        frames: list[np.ndarray] = []
        n = max(1, int(n_frames))
        sched = frames_schedule if frames_schedule is not None else self.animation_config.get("frames_schedule")
        for k in range(n):
            ph = float(k) / float(n)
            ev = self.evaluate(phase=ph, target=target)
            frames.append(ev.output)
        return export_flipbook(
            frames,
            out_png_path=out_png_path,
            frametime=frametime,
            interpolate=interpolate,
            frames_schedule=sched,
            out_gif_path=out_gif_path,
            gif_scale=gif_scale,
        )

    def apply_to_bbmodel(
        self,
        bbmodel_doc: dict[str, Any],
        *,
        phase: float = 0.0,
        pad: int | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
        """Bake mesh if needed, evaluate graph, pad UV islands, and embed RGBA PNG into a .bbmodel copy."""
        doc = copy.deepcopy(bbmodel_doc)
        if self.mesh_context is None:
            self.attach_mesh(bake_bbmodel_maps(doc))
        ev = self.evaluate(phase=phase)
        out_rgba = self.mesh_context.pad_rgba(ev.output, pad=pad) if self.mesh_context else ev.output
        em_rgba = (
            self.mesh_context.pad_rgba(ev.emissive, pad=pad)
            if (ev.emissive is not None and self.mesh_context is not None)
            else ev.emissive
        )
        png_bytes = encode_png(out_rgba)
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        if doc.get("textures"):
            t0 = doc["textures"][0]
            t0["width"] = self.width
            t0["height"] = self.height
            t0["uv_width"] = self.width
            t0["uv_height"] = self.height
            t0["source"] = data_url
            t0["internal"] = True
        return doc, rgba_float_to_u8(out_rgba), (rgba_float_to_u8(em_rgba) if em_rgba is not None else None)


def _eval_sprite_shape(H: int, W: int, shape: str, *, phase: float = 0.0) -> np.ndarray:
    """Built-in procedural item/icon silhouette masks (crystal_blade, astrolabe_orb, dragon_crest)."""
    u, v = uv_grid(H, W)
    if shape == "crystal_blade":
        # Diagonal pixel-art greatsword from bottom-left (0.14, 0.86) to top-right (0.88, 0.12)
        dx = u - 0.5
        dy = (1.0 - v) - 0.5
        along = (dx + dy) * 0.7071  # [-0.6, +0.6]
        perp = np.abs(dx - dy) * 0.7071
        # Blade (along in [-0.08, 0.48]), widens mid-blade and tapers to razor tip
        blade_w = np.where(along < 0.34, 0.095 + 0.03 * np.sin((along + 0.08) * 6.0), np.clip((0.50 - along) * 0.65, 0.0, 0.12))
        blade = (along >= -0.06) & (along <= 0.50) & (perp <= blade_w)
        # Crossguard (perpendicular diamond wings at along ~ -0.09)
        guard = (np.abs(along + 0.09) <= 0.042) & (perp <= 0.22)
        # Grip + pommel
        grip = (along >= -0.34) & (along <= -0.09) & (perp <= 0.038)
        pommel = ((along + 0.37) ** 2 + perp ** 2) <= 0.055 ** 2
        return (blade | guard | grip | pommel).astype(np.float32)

    if shape == "crystal_blade_core":
        # Inner neon energy fuller channel running down center of blade + crossguard gem
        dx = u - 0.5
        dy = (1.0 - v) - 0.5
        along = (dx + dy) * 0.7071
        perp = np.abs(dx - dy) * 0.7071
        fuller = (along >= 0.0) & (along <= 0.38) & (perp <= 0.032)
        gem = ((along + 0.09) ** 2 + perp ** 2) <= 0.038 ** 2
        return (fuller | gem).astype(np.float32)

    if shape == "crystal_blade_guard":
        dx = u - 0.5
        dy = (1.0 - v) - 0.5
        along = (dx + dy) * 0.7071
        perp = np.abs(dx - dy) * 0.7071
        guard = (np.abs(along + 0.09) <= 0.045) & (perp <= 0.22)
        pommel = ((along + 0.37) ** 2 + perp ** 2) <= 0.058 ** 2
        return (guard | pommel).astype(np.float32)

    # Default circular gem/orb
    r = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
    return (r <= 0.42).astype(np.float32)
