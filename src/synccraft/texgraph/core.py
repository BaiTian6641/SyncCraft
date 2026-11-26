"""Core DAG Engine & AI-Workflow-Friendly Declarative Recipe Format for `synccraft.texgraph`.

Supports:
- Fluent Python DAG authoring (`TexGraph`)
- Lossless declarative JSON `.texgraph.json` recipe import/export (`to_recipe` / `from_recipe`)
- Topological evaluation with cross-frame static-subgraph memoization for fast flipbook rendering
- Direct 3D `.bbmodel` mesh baking (`MeshBakeContext`) & embedded PNG application (`apply_to_bbmodel`)
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

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
from .pbr import (
    export_pbr_bundle,
    height_to_normal_map,
    pack_labpbr_1_3,
    render_pbr_material_sheet,
    render_pbr_preview_2d,
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

# Nodes whose output depends on animation `phase` (re-evaluated per flipbook frame; others cached!)
_TIME_AWARE_OPS: frozenset[str] = frozenset(
    {
        "perlin", "fbm", "voronoi", "domain_warp", "noise_3d",
        "radial_rings", "energy_veins", "scroll", "pulse", "sprite_mask",
    }
)


@dataclass
class NodeSpec:
    id: str
    op: str
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": dict(self.inputs),
            "params": copy.deepcopy(self.params),
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
        # First-class PBR channel outputs (height, normal, roughness, metallic, ao, sss, porosity)
        self.pbr_nodes: dict[str, str] = {}
        self.pbr_config: dict[str, Any] = {"normal_strength": 2.4, "dielectric_f0": 0.04}
        self._counter = 0
        self._static_cache: dict[str, np.ndarray] = {}
        self._dynamic_ids: set[str] = set()

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
        **params: Any,
    ) -> str:
        nid = self._next_id(op, id)
        clean_inputs = {k: v for k, v in (inputs or {}).items() if v is not None}
        for ref in clean_inputs.values():
            if ref not in self.nodes:
                raise KeyError(f"Node {nid!r} references unknown input node {ref!r}")
        spec = NodeSpec(id=nid, op=op, inputs=clean_inputs, params=params)
        self.nodes[nid] = spec
        self.node_order.append(nid)
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
        sss_node: str | None = None,
        porosity_node: str | None = None,
        normal_strength: float | None = None,
        dielectric_f0: float | None = None,
    ) -> "TexGraph":
        if node_id not in self.nodes:
            raise KeyError(f"Unknown output node: {node_id!r}")
        self.output_node = node_id
        if emissive_node is not None:
            if emissive_node not in self.nodes:
                raise KeyError(f"Unknown emissive node: {emissive_node!r}")
            self.emissive_node = emissive_node
        for ch_name, ch_id in (
            ("height", height_node),
            ("normal", normal_node),
            ("roughness", roughness_node),
            ("metallic", metallic_node),
            ("ao", ao_node),
            ("sss", sss_node),
            ("porosity", porosity_node),
        ):
            if ch_id is not None:
                if ch_id not in self.nodes:
                    raise KeyError(f"Unknown PBR {ch_name} node: {ch_id!r}")
                self.pbr_nodes[ch_name] = ch_id
        if normal_strength is not None:
            self.pbr_config["normal_strength"] = float(normal_strength)
        if dielectric_f0 is not None:
            self.pbr_config["dielectric_f0"] = float(dielectric_f0)
        return self

    def attach_mesh(self, mesh_context: MeshBakeContext) -> "TexGraph":
        self.mesh_context = mesh_context
        self.width = mesh_context.width
        self.height = mesh_context.height
        self._static_cache.clear()
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

    # ------------------------------------------------------------------
    # Declarative JSON Recipe Import / Export
    # ------------------------------------------------------------------
    def to_recipe(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "version": "2.0",
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "tileable": self.tileable,
            "seed": self.seed,
            "output": self.output_node or (self.node_order[-1] if self.node_order else None),
            "emissive": self.emissive_node,
            "nodes": [self.nodes[nid].to_dict() for nid in self.node_order],
        }
        if self.pbr_nodes:
            rec["pbr"] = dict(self.pbr_nodes)
            rec["pbr_config"] = dict(self.pbr_config)
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
            g.add(nd["op"], id=nd["id"], inputs=nd.get("inputs") or {}, **(nd.get("params") or {}))
        pbr_d = data.get("pbr") or {}
        pbr_cfg = data.get("pbr_config") or {}
        if data.get("output"):
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
        return g

    # ------------------------------------------------------------------
    # DAG Evaluation Engine (With Static Caching Across Animation Frames)
    # ------------------------------------------------------------------
    def _compute_dynamic_set(self) -> set[str]:
        dyn: set[str] = set()
        for nid in self.node_order:
            sp = self.nodes[nid]
            is_dyn = False
            if sp.op in ("scroll", "pulse"):
                is_dyn = True
            elif sp.op in _TIME_AWARE_OPS:
                p = sp.params
                if abs(float(p.get("loop_speed", 0.0))) > 1e-6 or abs(float(p.get("flow_speed", 0.0))) > 1e-6:
                    is_dyn = True
                if sp.op == "energy_veins" and abs(float(p.get("flow_speed", 1.0))) > 1e-6:
                    is_dyn = True
                if sp.op == "noise_3d" and any(abs(float(c)) > 1e-6 for c in p.get("flow_vec", (0, 0, 0))):
                    is_dyn = True
            if not is_dyn and any(ref in dyn for ref in sp.inputs.values()):
                is_dyn = True
            if is_dyn:
                dyn.add(nid)
        return dyn

    def _eval_node(self, sp: NodeSpec, vals: dict[str, np.ndarray], phase: float) -> np.ndarray:
        H, W = self.height, self.width
        tile = self.tileable
        p = sp.params
        ins = {k: vals[v] for k, v in sp.inputs.items()}
        op = sp.op
        seed = int(p.get("seed", self.seed))

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
            return value_noise(H, W, scale=p.get("scale", 8.0), scale_y=p.get("scale_y"), seed=seed, tileable=p.get("tileable", tile))
        if op == "perlin":
            return perlin_noise(H, W, scale=p.get("scale", 8.0), scale_y=p.get("scale_y"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0))
        if op == "fbm":
            return fbm_noise(H, W, scale=p.get("scale", 6.0), scale_y=p.get("scale_y"), octaves=p.get("octaves", 4), persistence=p.get("persistence", 0.5), lacunarity=p.get("lacunarity", 2.0), mode=p.get("mode", "fbm"), seed=seed, tileable=p.get("tileable", tile), phase=phase, loop_speed=p.get("loop_speed", 0.0))
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

        raise ValueError(f"Unsupported TexGraph op: {op!r}")

    def evaluate(
        self,
        *,
        phase: float = 0.0,
        target: str | None = None,
        capture_intermediates: bool = False,
        build_pbr: bool | None = None,
        light_az_deg: float = -38.0,
        light_el_deg: float = 52.0,
    ) -> EvalResult:
        if not self.node_order:
            raise ValueError("TexGraph has no nodes")
        dyn_ids = self._compute_dynamic_set()
        vals: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}
        for nid in self.node_order:
            if nid not in dyn_ids and nid in self._static_cache:
                vals[nid] = self._static_cache[nid]
                timings[nid] = 0.0
                continue
            t0 = time.perf_counter()
            res = self._eval_node(self.nodes[nid], vals, float(phase))
            timings[nid] = round((time.perf_counter() - t0) * 1000.0, 2)
            vals[nid] = res
            if nid not in dyn_ids:
                self._static_cache[nid] = res

        out_id = target or self.output_node or self.node_order[-1]
        out_arr = vals[out_id]
        em_arr = vals[self.emissive_node] if (self.emissive_node and self.emissive_node in vals) else None
        pbr_ch: dict[str, np.ndarray] = {
            k: vals[nid] for k, nid in self.pbr_nodes.items() if nid in vals
        }

        want_pbr = bool(self.pbr_nodes) if build_pbr is None else bool(build_pbr)
        pbr_maps = None
        pbr_preview = None
        if want_pbr and out_arr.ndim == 3 and out_arr.shape[2] >= 3:
            # Pad island borders on PBR channels too when mesh_context is attached!
            def _pad_opt(a: np.ndarray | None) -> np.ndarray | None:
                if a is None or self.mesh_context is None:
                    return a
                return self.mesh_context.pad_rgba(a)

            h_ch = _pad_opt(pbr_ch.get("height"))
            n_ch = _pad_opt(pbr_ch.get("normal"))
            r_ch = _pad_opt(pbr_ch.get("roughness"))
            m_ch = _pad_opt(pbr_ch.get("metallic"))
            ao_ch = _pad_opt(pbr_ch.get("ao", self.mesh_context.baked_ao if self.mesh_context else None))
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
        )

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

    def render_pbr_flipbook(
        self,
        n_frames: int = 16,
        *,
        out_dir: Path | str,
        stem: str | None = None,
        frametime: int = 2,
        interpolate: bool = False,
        gif_scale: int = 4,
        rotate_light_in_preview: bool = False,
    ) -> dict[str, Any]:
        """Render synchronized multi-channel animated PBR `.mcmeta` flipbooks (Albedo, LabPBR _n, _s, Bedrock _mer + PBR Lit GIF)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        st = stem or self.name
        n = max(1, int(n_frames))
        alb_fr: list[np.ndarray] = []
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
            n_fr.append(ev.pbr_maps["labpbr_n"].astype(np.float32) / 255.0)
            s_fr.append(ev.pbr_maps["labpbr_s"].astype(np.float32) / 255.0)
            mer_fr.append(ev.pbr_maps["bedrock_mer"].astype(np.float32) / 255.0)
            lit_fr.append(ev.pbr_preview)

        fb_alb = export_flipbook(alb_fr, out_png_path=d / f"{st}_flipbook.png", frametime=frametime, interpolate=interpolate, out_gif_path=d / f"{st}.gif", gif_scale=gif_scale)
        fb_n = export_flipbook(n_fr, out_png_path=d / f"{st}_n_flipbook.png", frametime=frametime, interpolate=interpolate)
        fb_s = export_flipbook(s_fr, out_png_path=d / f"{st}_s_flipbook.png", frametime=frametime, interpolate=interpolate)
        fb_mer = export_flipbook(mer_fr, out_png_path=d / f"{st}_mer_flipbook.png", frametime=frametime, interpolate=interpolate)
        fb_lit = export_flipbook(lit_fr, out_gif_path=d / f"{st}_pbr_lit.gif", frametime=frametime, gif_scale=gif_scale)
        return {
            "num_frames": n,
            "albedo": fb_alb,
            "labpbr_n": fb_n,
            "labpbr_s": fb_s,
            "bedrock_mer": fb_mer,
            "pbr_lit_gif": str(d / f"{st}_pbr_lit.gif"),
            "albedo_frames": alb_fr,
            "pbr_lit_frames": lit_fr,
        }

    def render_flipbook(
        self,
        n_frames: int = 16,
        *,
        frametime: int = 2,
        interpolate: bool = False,
        out_png_path: Path | str | None = None,
        out_gif_path: Path | str | None = None,
        gif_scale: int = 4,
        target: str | None = None,
    ) -> dict[str, Any]:
        frames: list[np.ndarray] = []
        n = max(1, int(n_frames))
        for k in range(n):
            ph = float(k) / float(n)
            ev = self.evaluate(phase=ph, target=target)
            frames.append(ev.output)
        return export_flipbook(
            frames,
            out_png_path=out_png_path,
            frametime=frametime,
            interpolate=interpolate,
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
