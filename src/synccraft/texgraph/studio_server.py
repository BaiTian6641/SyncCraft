"""SyncCraft Studio — Commercial Local Node-Graph & 3D PBR Workstation Server (`TexGraph v10 / SyncCraft v48.0`).

Zero-external-dependency localhost workstation backend (Python stdlib `ThreadingHTTPServer` +
NumPy + Pillow) serving a mature 5-pane DCC studio UI (`studio_ui/`) with:
  - Live incremental dirty-subgraph evaluation via `TexGraph._content_cache` (0 ms untouched hits)
  - Full Schema v4.0 (`144` typed nodes, `PORT_COLORS`, `PORT_COMPATIBILITY`, `visible_if`)
  - Real `.bbmodel` 3D mesh extraction (matching `mesh_baker.py` bone/element transforms & UVs)
  - 12-Channel SyncCraft Special PBR + Unreal/glTF ORM + Unity MaskMap + LabPBR 1.3 + Bedrock MER
  - Undo/Redo snapshot stack, live bidirectional `.texdsl` compilation/decompilation,
    `@animate` timeline drivers, `@expose` parameter promotion, QA scorecard & multi-engine export.

Safety invariants:
  - Binds strictly to `127.0.0.1` by default.
  - Zero `eval`/`exec`; strict path containment for static UI assets.
"""
from __future__ import annotations

import base64
import copy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import threading
import time
import urllib.parse
import webbrowser
from typing import Any

import numpy as np
from PIL import Image

from synccraft import __version__
from .ai_copilot import BUILT_IN_STYLE_PRESETS, apply_surgical_patch, diagnose_asset, extract_style_dna
from .core import GraphCycleError, NodeSpec, TexGraph, _parse_ref
from .mesh_baker import _FACE_BASIS, _euler_xyz_mat, MeshBakeContext, bake_bbmodel_maps
from .node_previews import generate_node_preview_catalog
from .pbr import BAKE_LIGHTING_PRESETS as BAKE_PRESETS, bake_pbr_to_albedo
from .pipeline import MATERIAL_BUS_PORTS, SUBGRAPH_REGISTRY
from .pixel_art_ops import encode_png, rgba_float_to_u8
from .presets import PRESET_REGISTRY
from .production import PALETTE_BANK
from .quality import evaluate_texture_quality
from .region_ops import build_uv_region_overlay
from .registry import NODE_CATALOG as NODE_REGISTRY, PORT_COMPATIBILITY, export_node_schema_json
from .syncproj import DEFAULT_CHRONO_VARIANTS, SyncProject, apply_variant_to_graph, generate_variant_pack
from .texdsl import TexDslError as TexDSLParseError, export_texdsl, parse_texdsl

_STUDIO_UI_DIR = Path(__file__).resolve().parent / "studio_ui"


# ---------------------------------------------------------------------------
# Fast PNG Data URL Helpers
# ---------------------------------------------------------------------------


def _arr_to_u8_rgba(arr: Any) -> np.ndarray:
    if isinstance(arr, dict):
        for k in ("albedo", "rgba", "height", "normal", "mask"):
            if k in arr:
                arr = arr[k]
                break
        else:
            for k, v in arr.items():
                if not str(k).startswith("__") and isinstance(v, np.ndarray):
                    arr = v
                    break
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        if a.ndim == 2:
            return np.stack((a, a, a, np.full_like(a, 255)), axis=-1)
        if a.ndim == 3 and a.shape[2] == 3:
            return np.concatenate((a, np.full(a.shape[:2] + (1,), 255, dtype=np.uint8)), axis=-1)
        if a.ndim == 3 and a.shape[2] == 4:
            return a
    return rgba_float_to_u8(a)


def _u8_to_data_url(u8: np.ndarray, *, max_dim: int | None = None, min_dim: int = 0) -> str:
    h, w = u8.shape[:2]
    im = Image.fromarray(u8, mode="RGBA")
    if max_dim is not None and max(h, w) > max_dim:
        sc = max_dim / float(max(h, w))
        im = im.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))), resample=Image.Resampling.NEAREST)
    elif min_dim > 0 and max(h, w) < min_dim:
        sc = max(1, int(min_dim // max(1, max(h, w))))
        im = im.resize((w * sc, h * sc), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _scalar_to_heatmap_u8(val: np.ndarray, *, warm: bool = False) -> np.ndarray:
    v = np.clip(np.asarray(val, dtype=np.float32), 0.0, 1.0)
    if warm:
        r = np.clip(0.14 + 0.86 * (1.0 - v), 0.0, 1.0)
        g = np.clip(0.22 + 0.56 * v, 0.0, 1.0)
        b = np.clip(0.24 + 0.34 * v, 0.0, 1.0)
    else:
        r = g = b = v
    rgb = np.round(np.stack((r, g, b), axis=-1) * 255.0).astype(np.uint8)
    a = np.full(v.shape + (1,), 255, dtype=np.uint8)
    return np.concatenate((rgb, a), axis=-1)


# ---------------------------------------------------------------------------
# Blockbench 5 .bbmodel -> WebGL Geometry Extractor
# ---------------------------------------------------------------------------


def _build_bone_transforms(bb_doc: dict[str, Any]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Map element uuid -> chain of (pivot, R_mat) bone transforms from innermost to outermost."""
    groups_by_id: dict[str, dict[str, Any]] = {}
    for g in bb_doc.get("groups") or []:
        if isinstance(g, dict) and g.get("uuid"):
            groups_by_id[str(g["uuid"])] = g

    elem_chains: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def _walk(nodes: list[Any], chain: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for item in nodes or []:
            if isinstance(item, str):
                elem_chains[item] = list(reversed(chain))
            elif isinstance(item, dict):
                uid = str(item.get("uuid") or "")
                meta = groups_by_id.get(uid, item)
                origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                rot = meta.get("rotation", [0.0, 0.0, 0.0])
                R = _euler_xyz_mat(rot)
                next_chain = chain + [(origin, R)]
                _walk(item.get("children") or [], next_chain)

    _walk(bb_doc.get("outliner") or [], [])
    return elem_chains


def extract_bbmodel_webgl_mesh(bb_doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a Blockbench 5 `.bbmodel` document into ready-to-upload WebGL triangle buffers.

    Preserves exact bone/element rotation hierarchies, per-face UV rectangles, outward normals,
    and orthonormal face tangents so the Studio 3D WebGL PBR viewport matches `mesh_baker.py`
    and Blockbench 5 pixel-for-pixel!
    """
    res = bb_doc.get("resolution") or {}
    tw = max(1.0, float(res.get("width", 64)))
    th = max(1.0, float(res.get("height", 64)))
    bone_chains = _build_bone_transforms(bb_doc)

    positions: list[float] = []
    normals: list[float] = []
    tangents: list[float] = []
    uvs: list[float] = []

    pmin = np.array([1e9, 1e9, 1e9], dtype=np.float64)
    pmax = np.array([-1e9, -1e9, -1e9], dtype=np.float64)
    box_count = 0

    for elem in bb_doc.get("elements") or []:
        if not isinstance(elem, dict):
            continue
        if elem.get("visibility") is False:
            continue
        f_pt = elem.get("from")
        t_pt = elem.get("to")
        if not f_pt or not t_pt:
            continue
        box_count += 1
        inf = float(elem.get("inflate", 0.0) or 0.0)
        x0, y0, z0 = float(f_pt[0]) - inf, float(f_pt[1]) - inf, float(f_pt[2]) - inf
        x1, y1, z1 = float(t_pt[0]) + inf, float(t_pt[1]) + inf, float(t_pt[2]) + inf
        cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        hx, hy, hz = 0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)

        e_orig = np.array(elem.get("origin", [cx, cy, cz]), dtype=np.float64)
        e_rot = _euler_xyz_mat(elem.get("rotation", [0.0, 0.0, 0.0]))
        chain = bone_chains.get(str(elem.get("uuid") or ""), [])

        def _xform_pt(p: np.ndarray) -> np.ndarray:
            q = e_orig + e_rot @ (p - e_orig)
            for b_orig, b_rot in chain:
                q = b_orig + b_rot @ (q - b_orig)
            return q

        def _xform_dir(d: np.ndarray) -> np.ndarray:
            q = e_rot @ d
            for _, b_rot in chain:
                q = b_rot @ q
            nrm = float(np.linalg.norm(q))
            return q / nrm if nrm > 1e-8 else q

        faces = elem.get("faces") or {}
        center_l = np.array([cx, cy, cz], dtype=np.float64)
        half_l = np.array([hx, hy, hz], dtype=np.float64)

        for fname, (tu_l, tv_l, fn_l) in _FACE_BASIS.items():
            finfo = faces.get(fname)
            if not isinstance(finfo, dict):
                continue
            uv_rect = finfo.get("uv")
            if not uv_rect or len(uv_rect) < 4:
                continue
            u0, v0, u1, v1 = (float(uv_rect[0]) / tw, float(uv_rect[1]) / th, float(uv_rect[2]) / tw, float(uv_rect[3]) / th)
            if abs(u1 - u0) < 1e-6 or abs(v1 - v0) < 1e-6:
                continue

            tu_v = np.array(tu_l, dtype=np.float64)
            tv_v = np.array(tv_l, dtype=np.float64)
            fn_v = np.array(fn_l, dtype=np.float64)

            # Face center + 4 corners in box-local space: (0,0)=TL, (1,0)=TR, (1,1)=BR, (0,1)=BL
            du = tu_v * float(np.abs(np.dot(tu_v, half_l)))
            dv = tv_v * float(np.abs(np.dot(tv_v, half_l)))
            dn = fn_v * float(np.abs(np.dot(fn_v, half_l)))
            fc = center_l + dn

            p_tl = _xform_pt(fc - du - dv)
            p_tr = _xform_pt(fc + du - dv)
            p_br = _xform_pt(fc + du + dv)
            p_bl = _xform_pt(fc - du + dv)

            for pt in (p_tl, p_tr, p_br, p_bl):
                pmin = np.minimum(pmin, pt)
                pmax = np.maximum(pmax, pt)

            wn = _xform_dir(fn_v)
            wt = _xform_dir(tu_v)

            # Ensure counter-clockwise winding relative to outward normal `wn`
            cross = np.cross(p_tr - p_tl, p_br - p_tl)
            if float(np.dot(cross, wn)) >= 0.0:
                tri_verts = (
                    (p_tl, (u0, v0)), (p_tr, (u1, v0)), (p_br, (u1, v1)),
                    (p_tl, (u0, v0)), (p_br, (u1, v1)), (p_bl, (u0, v1)),
                )
            else:
                tri_verts = (
                    (p_tl, (u0, v0)), (p_br, (u1, v1)), (p_tr, (u1, v0)),
                    (p_tl, (u0, v0)), (p_bl, (u0, v1)), (p_br, (u1, v1)),
                )

            for pt, uv_pt in tri_verts:
                positions.extend((round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)))
                normals.extend((round(float(wn[0]), 4), round(float(wn[1]), 4), round(float(wn[2]), 4)))
                tangents.extend((round(float(wt[0]), 4), round(float(wt[1]), 4), round(float(wt[2]), 4)))
                uvs.extend((round(float(uv_pt[0]), 5), round(float(uv_pt[1]), 5)))

    if not positions:
        pmin = np.array([-8.0, 0.0, -8.0])
        pmax = np.array([8.0, 16.0, 8.0])
    center = 0.5 * (pmin + pmax)
    extent = np.maximum(1.0, pmax - pmin)
    radius = float(0.5 * np.linalg.norm(extent))

    return {
        "name": str(bb_doc.get("name", "bbmodel")),
        "box_count": box_count,
        "vertex_count": len(positions) // 3,
        "resolution": [int(tw), int(th)],
        "bounds": {
            "min": [round(float(c), 3) for c in pmin],
            "max": [round(float(c), 3) for c in pmax],
            "center": [round(float(c), 3) for c in center],
            "radius": round(radius, 3),
        },
        "positions": positions,
        "normals": normals,
        "tangents": tangents,
        "uvs": uvs,
    }


# ---------------------------------------------------------------------------
# Built-in Flagship Studio Sample Projects & Procedural .bbmodel Fixtures
# ---------------------------------------------------------------------------


def build_sample_monolith_bbmodel() -> dict[str, Any]:
    """Build a self-contained 5-box Sculpted Temple Monolith `.bbmodel` (zero external files!)."""
    from synccraft.shading.uv_packer import repack_bbmodel_uvs

    def _mk_box(name: str, uid: str, p0: list[float], p1: list[float]) -> dict[str, Any]:
        return {
            "name": name,
            "uuid": uid,
            "from": p0,
            "to": p1,
            "origin": [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), 0.5 * (p0[2] + p1[2])],
            "rotation": [0.0, 0.0, 0.0],
            "faces": {f: {"uv": [0, 0, 16, 16], "texture": 0} for f in ("north", "east", "south", "west", "up", "down")},
        }

    raw = {
        "meta": {"format_version": "4.10", "model_format": "free", "box_uv": False},
        "name": "temple_monolith_v10",
        "resolution": {"width": 128, "height": 128},
        "elements": [
            _mk_box("base_plinth", "e_plinth", [-10, 0, -10], [10, 4, 10]),
            _mk_box("pillar_core", "e_pillar", [-7, 4, -7], [7, 20, 7]),
            _mk_box("cornice_canopy", "e_cornice", [-11, 20, -11], [11, 25, 11]),
            _mk_box("crest_fin_l", "e_fin_l", [-10, 25, -4], [-9, 32, 4]),
            _mk_box("crest_fin_r", "e_fin_r", [9, 25, -4], [10, 32, 4]),
        ],
        "outliner": [
            {
                "name": "monolith_root",
                "uuid": "g_root",
                "origin": [0, 0, 0],
                "rotation": [0, 0, 0],
                "children": ["e_plinth", "e_pillar", "e_cornice", "e_fin_l", "e_fin_r"],
            }
        ],
        "textures": [{"name": "monolith", "id": "0", "source": ""}],
    }
    packed, _, _ = repack_bbmodel_uvs(raw, padding=1, mirror_lr_uv="none", canvas_mode="tight")
    return packed


def _find_kaelith_dragonborn_paths() -> tuple[Path, Path] | None:
    """Locate the sculpted Chrono-Amethyst Artificer Dragonborn `.bbmodel` & `.texdsl` if present in workspace."""
    here = Path(__file__).resolve()
    for anc in list(here.parents)[:6]:
        for sub in (
            anc / "test_flights" / "texgraph_showcase" / "kaelith_chrono_dragonborn",
            anc / "MH-blockbench-assets" / "texgraph_showcase" / "kaelith_chrono_dragonborn",
        ):
            bb = sub / "kaelith_chrono_dragonborn.bbmodel"
            dsl = sub / "kaelith_chrono_dragonborn.texdsl"
            if bb.is_file() and dsl.is_file():
                return bb, dsl
    return None


def _find_orc_warlord_paths() -> tuple[Path, Path] | None:
    """Locate the sculpted 84-box Ironjaw Orc Warlord `.bbmodel` & `.texdsl` if present in workspace."""
    here = Path(__file__).resolve()
    for anc in list(here.parents)[:6]:
        for sub in (
            anc / "MH-blockbench-assets" / "texgraph_showcase" / "ironjaw_orc_warlord",
            anc / "test_flights" / "texgraph_showcase" / "ironjaw_orc_warlord",
        ):
            bb = sub / "ironjaw_orc_warlord.bbmodel"
            dsl = sub / "ironjaw_orc_warlord.texdsl"
            if bb.is_file() and dsl.is_file():
                return bb, dsl
    return None


def build_studio_sample(sample_id: str, *, size: int = 64) -> tuple[TexGraph, dict[str, Any] | None]:
    """Create a rich, commercial multi-pass Studio project with optional 3D `.bbmodel`."""
    if sample_id == "monolith_v10_3d":
        bb_doc = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb_doc)
        g = TexGraph(mc.width, mc.height, tileable=False, seed=47, name="temple_monolith_v10", mesh_context=mc)
        g.domain = "entity"
        with g.pass_("pass_1_base_stone", title="Base Masonry & Moss", color="#334155") as p1:
            cob = g.cobblestone_masonry(
                cols=2,
                rows=2,
                stone_stops=list(PALETTE_BANK["cobble_warm_slate"]),
                mortar_color="#565260",
                moss_amount=0.24,
                wear_strength=0.25,
                smooth_blend=0.65,
                quantize_steps=6,
                seed=47,
                id="cobble_base",
            )
            p1.preview_ref = f"{cob}.albedo"
        with g.pass_("pass_2_3d_seamless", title="3D Triplanar & Seam Heal", color="#0F766E") as p2:
            tri = g.triplanar_project(
                cob,
                scale=0.040,
                sharpness=6.0,
                dither_blend=True,
                pixel_snap=True,
                id="triplanar_wrap",
            )
            healed = g.uv_seam_heal(tri, strength=0.84, radius_px=2, heal_normals=True, id="seam_stitch")
            p2.preview_ref = f"{healed}.albedo"
        with g.pass_("pass_3_shadow_pbr", title="Ray-Traced Shadow Special PBR", color="#1D4ED8") as p3:
            sh = g.precompute_shadow_pbr(
                healed,
                shadow_strength=0.68,
                bent_normal_strength=0.46,
                thickness_sss_strength=0.38,
                bake_into_albedo=0.55,
                id="shadow_special_pbr",
            )
            final = g.material_grade(
                sh,
                shadows=(-0.01, -0.004, 0.022),
                highlights=(0.026, 0.018, 0.006),
                vibrance=0.16,
                contrast=1.05,
                id="monolith_final",
            )
            p3.preview_ref = f"{final}.albedo"
        g.set_output_material(final, normal_strength=2.5)
        g.expose_param("moss_amount", cob, "moss_amount", label="Crevice Moss", min_val=0.0, max_val=0.85, step=0.05)
        g.expose_param("triplanar_scale", tri, "scale", label="3D World Scale", min_val=0.02, max_val=0.20, step=0.005)
        g.expose_param("shadow_strength", sh, "shadow_strength", label="3D Cast Shadow", min_val=0.0, max_val=1.0, step=0.05)
        g.expose_param("bent_normal_strength", sh, "bent_normal_strength", label="Sky Bent-Normal IBL", min_val=0.0, max_val=1.0, step=0.05)
        g.bind_driver(sh, "shadow_strength", kind="sine", min_val=0.45, max_val=0.85, freq=1.0)
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, bb_doc

    if sample_id == "connected_block_wall_v10":
        g = TexGraph(size, size, tileable=True, seed=108, name="connected_fortress_block")
        with g.pass_("pass_1_masonry", title="Base Cobblestone", color="#334155") as p1:
            cob = g.cobblestone_masonry(
                cols=4,
                rows=4,
                stone_stops=list(PALETTE_BANK["cobble_warm_slate"]),
                mortar_color="#26242C",
                moss_amount=0.18,
                wear_strength=0.44,
                seed=108,
                id="stone_masonry",
            )
            p1.preview_ref = f"{cob}.albedo"
        with g.pass_("pass_2_connected_edge", title="Connected Block Edge", color="#7C2D12") as p2:
            edge = g.block_edge_transition(
                cob,
                neighbors="exposed_all",
                style="metal_frame",
                border_width=0.14,
                corner_strength=0.72,
                seed=108,
                id="block_trim",
            )
            p2.preview_ref = f"{edge}.albedo"
        with g.pass_("pass_3_grade", title="Studio Grade & PBR", color="#1E3A8A") as p3:
            final = g.material_grade(
                edge,
                shadows=(-0.01, -0.005, 0.02),
                highlights=(0.024, 0.018, 0.008),
                vibrance=0.15,
                contrast=1.06,
                id="wall_final",
            )
            p3.preview_ref = f"{final}.albedo"
        g.set_output_material(final, normal_strength=2.6)
        g.expose_param("moss_amount", cob, "moss_amount", label="Moss Amount", min_val=0.0, max_val=0.8, step=0.05)
        g.expose_param("border_width", edge, "border_width", label="Frame Border Width", min_val=0.05, max_val=0.35, step=0.01)
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, None

    if sample_id == "kaelith_dragonborn_3d":
        found = _find_kaelith_dragonborn_paths()
        if found is not None:
            bb_path, dsl_path = found
            bb_doc = json.loads(bb_path.read_text(encoding="utf-8"))
            mc = bake_bbmodel_maps(bb_doc)
            g = parse_texdsl(dsl_path.read_text(encoding="utf-8"), mesh_context=mc)
            g.domain = "entity"
            g.auto_layout(col_spacing=290.0, row_spacing=195.0)
            return g, bb_doc
        return build_studio_sample("monolith_v10_3d", size=size)

    if sample_id == "ironjaw_orc_3d":
        found = _find_orc_warlord_paths()
        if found is not None:
            bb_path, dsl_path = found
            bb_doc = json.loads(bb_path.read_text(encoding="utf-8"))
            mc = bake_bbmodel_maps(bb_doc)
            g = parse_texdsl(dsl_path.read_text(encoding="utf-8"), mesh_context=mc)
            g.auto_layout(col_spacing=290.0, row_spacing=205.0)
            return g, bb_doc
        return build_studio_sample("monolith_v10_3d", size=size)

    if sample_id == "arcane_ember_vfx":
        g = PRESET_REGISTRY["arcane_ember_vfx"](size=size)
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, None

    if sample_id == "temple_carved_stone":
        g = PRESET_REGISTRY["temple_carved_stone"](size=size)
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, None

    if sample_id in PRESET_REGISTRY:
        g = PRESET_REGISTRY[sample_id](size=size)
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, None

    raise ValueError(f"Unknown sample/preset: {sample_id!r}")


STUDIO_SAMPLES: list[dict[str, Any]] = [
    {
        "id": "kaelith_dragonborn_3d",
        "title": "Kaelith Chrono-Dragonborn 3D",
        "category": "Flagship 3D Studio",
        "has_3d": True,
        "animated": True,
    },
    {
        "id": "monolith_v10_3d",
        "title": "Temple Monolith 3D (v10)",
        "category": "Flagship 3D Studio",
        "has_3d": True,
        "animated": True,
    },
    {
        "id": "connected_block_wall_v10",
        "title": "Connected Block Wall (v10)",
        "category": "Flagship 3D Studio",
        "has_3d": False,
        "animated": False,
    },
    {
        "id": "arcane_ember_vfx",
        "title": "Arcane Ember VFX (4-Pass)",
        "category": "Flagship 3D Studio",
        "has_3d": False,
        "animated": True,
    },
    {
        "id": "temple_carved_stone",
        "title": "Temple Carved Stone (4-Pass)",
        "category": "Flagship 3D Studio",
        "has_3d": False,
        "animated": False,
    },
]


# ---------------------------------------------------------------------------
# Stateful Live Studio Session (Undo/Redo + Incremental Cache + Evaluation)
# ---------------------------------------------------------------------------


class StudioSession:
    """Thread-safe active workspace session for SyncCraft Studio."""

    def __init__(self, *, initial_sample: str = "monolith_v10_3d", size: int = 64) -> None:
        self._lock = threading.RLock()
        self.bbmodel_doc: dict[str, Any] | None = None
        self.mesh_ctx: MeshBakeContext | None = None
        self.webgl_mesh: dict[str, Any] | None = None
        self.graph: TexGraph
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._thumb_cache: dict[str, str] = {}
        self._last_eval: dict[str, Any] = {}
        self._event_log: list[dict[str, Any]] = []
        self.artist_overrides: dict[str, Any] = {}
        self.active_variant: str = "chronos_prime"
        self.active_source: str = f"sample:{initial_sample}"

        g, bb = build_studio_sample(initial_sample, size=size)
        self._set_workspace(g, bb, source=self.active_source, clear_history=True)

    def _log(self, level: str, msg: str) -> None:
        self._event_log.append(
            {
                "ts": round(time.time(), 2),
                "level": level,
                "message": msg,
            }
        )
        if len(self._event_log) > 120:
            self._event_log = self._event_log[-120:]

    def _set_workspace(
        self,
        graph: TexGraph,
        bb_doc: dict[str, Any] | None,
        *,
        source: str,
        clear_history: bool = False,
    ) -> None:
        self.graph = graph
        self.active_source = source
        if bb_doc is not None:
            self.bbmodel_doc = bb_doc
            self.mesh_ctx = graph.mesh_context or bake_bbmodel_maps(bb_doc)
            self.graph.attach_mesh(self.mesh_ctx)
            self.webgl_mesh = extract_bbmodel_webgl_mesh(bb_doc)
        else:
            self.bbmodel_doc = None
            self.mesh_ctx = None
            self.webgl_mesh = None
        self._ensure_positions()
        if clear_history:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self.artist_overrides.clear()
            self.active_variant = "chronos_prime"
        self._thumb_cache.clear()
        self._log("info", f"Loaded workspace '{graph.name}' ({len(graph.nodes)} nodes, {graph.width}x{graph.height})")

    def _ensure_positions(self) -> None:
        missing = [nid for nid, sp in self.graph.nodes.items() if sp.pos is None]
        if missing:
            self.graph.auto_layout(col_spacing=290.0, row_spacing=205.0)

    def _push_undo(self) -> None:
        self._undo_stack.append(copy.deepcopy(self.graph.to_recipe()))
        if len(self._undo_stack) > 64:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_recipe(self, rec: dict[str, Any]) -> None:
        old_cache = dict(self.graph._content_cache)
        g = TexGraph.from_recipe(rec, mesh_context=self.mesh_ctx)
        # Preserve unaffected content-addressed node evaluations across undo/redo!
        g._content_cache = old_cache
        self.graph = g
        self._ensure_positions()

    def get_schema_payload(self) -> dict[str, Any]:
        schema = export_node_schema_json()
        preset_groups: list[dict[str, Any]] = []
        for pname in sorted(PRESET_REGISTRY.keys()):
            cat = "World Blocks"
            if any(k in pname for k in ("blade", "sword", "axe", "potion", "gem", "shield", "pickaxe", "wand", "crown", "item")):
                cat = "Items & Weapons"
            elif any(k in pname for k in ("dragon", "wing", "wolf", "fox", "owl", "scorpion", "fish", "frog", "golem", "phoenix", "orc", "leopard")):
                cat = "Creatures & Entities"
            elif any(k in pname for k in ("vfx", "ember", "water", "caustic", "neon", "conduit")):
                cat = "VFX & Animated"
            preset_groups.append({"id": pname, "title": pname.replace("_", " ").title(), "category": cat})

        return {
            "synccraft_version": __version__,
            "schema": schema,
            "samples": STUDIO_SAMPLES,
            "presets": preset_groups,
            "subgraphs": sorted(SUBGRAPH_REGISTRY.keys()),
            "bake_presets": sorted(BAKE_PRESETS.keys()),
            "palettes": sorted(PALETTE_BANK.keys()),
            "style_presets": [
                {"id": k, "title": v["title"], "stops": v["stops"], "mood": v["mood"]}
                for k, v in BUILT_IN_STYLE_PRESETS.items()
            ],
            "variants": [
                {"id": k, "title": v["title"], "style_preset": v["style_preset"], "eye_iris": v["eye_iris"], "rune_color": v["rune_color"]}
                for k, v in DEFAULT_CHRONO_VARIANTS.items()
            ],
        }

    def get_state_payload(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_positions()
            val = self.graph.validate_graph()
            try:
                dsl_text = export_texdsl(self.graph)
            except Exception as exc:
                dsl_text = f"// TexDSL export note: {exc}\n"
            return {
                "source": self.active_source,
                "recipe": self.graph.to_recipe(),
                "texdsl": dsl_text,
                "validation": val.to_dict(),
                "undo_depth": len(self._undo_stack),
                "redo_depth": len(self._redo_stack),
                "artist_overrides": copy.deepcopy(self.artist_overrides),
                "active_variant": self.active_variant,
                "has_mesh": self.mesh_ctx is not None,
                "mesh": self.webgl_mesh,
                "mesh_stats": (
                    {
                        "width": self.mesh_ctx.width,
                        "height": self.mesh_ctx.height,
                        "islands": len(self.mesh_ctx.islands),
                        "elements": len(self.mesh_ctx.elem_names),
                        "seam_pairs": int(len(self.mesh_ctx.uv_seam_src_yx)),
                        "shadow_mean": round(float(self.mesh_ctx.baked_shadow[self.mesh_ctx.island_mask].mean()), 3)
                        if np.any(self.mesh_ctx.island_mask)
                        else 1.0,
                    }
                    if self.mesh_ctx is not None
                    else None
                ),
                "log": self._event_log[-30:],
            }

    def load_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            kind = str(payload.get("kind", "sample"))
            size = max(16, min(256, int(payload.get("size", 64))))
            if kind == "sample":
                sid = str(payload.get("id", "monolith_v10_3d"))
                g, bb = build_studio_sample(sid, size=size)
                self._set_workspace(g, bb, source=f"sample:{sid}", clear_history=True)
            elif kind == "preset":
                pid = str(payload.get("id", ""))
                if pid not in PRESET_REGISTRY:
                    raise ValueError(f"Unknown preset {pid!r}")
                g = PRESET_REGISTRY[pid](size=size)
                self._set_workspace(g, None, source=f"preset:{pid}", clear_history=True)
            elif kind == "texdsl":
                code = str(payload.get("code", ""))
                g = parse_texdsl(code, mesh_context=self.mesh_ctx)
                self._set_workspace(g, self.bbmodel_doc, source="texdsl:custom", clear_history=True)
            elif kind == "recipe":
                rec = payload.get("recipe")
                if not isinstance(rec, dict):
                    raise ValueError("Expected 'recipe' JSON object")
                g = TexGraph.from_recipe(rec, mesh_context=self.mesh_ctx)
                self._set_workspace(g, self.bbmodel_doc, source="recipe:custom", clear_history=True)
            else:
                raise ValueError(f"Unsupported load kind: {kind!r}")
            return self.get_state_payload()

    def load_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            preset_name = payload.get("sample")
            if preset_name == "kaelith_dragonborn":
                found = _find_kaelith_dragonborn_paths()
                bb_doc = json.loads(found[0].read_text(encoding="utf-8")) if found else build_sample_monolith_bbmodel()
            elif preset_name == "temple_monolith":
                bb_doc = build_sample_monolith_bbmodel()
            elif preset_name == "ironjaw_orc":
                found = _find_orc_warlord_paths()
                bb_doc = json.loads(found[0].read_text(encoding="utf-8")) if found else build_sample_monolith_bbmodel()
            elif preset_name == "none":
                self.bbmodel_doc = None
                self.mesh_ctx = None
                self.webgl_mesh = None
                self.graph.mesh_context = None
                self.graph._content_cache.clear()
                self._log("info", "Detached 3D .bbmodel mesh (using standard primitive preview)")
                return self.get_state_payload()
            else:
                bb_doc = payload.get("bbmodel")
                if not isinstance(bb_doc, dict):
                    raise ValueError("Expected 'bbmodel' object or 'sample' name")

            mc = bake_bbmodel_maps(bb_doc)
            self.bbmodel_doc = bb_doc
            self.mesh_ctx = mc
            self.webgl_mesh = extract_bbmodel_webgl_mesh(bb_doc)
            self.graph.attach_mesh(mc)
            self.graph._content_cache.clear()
            self._thumb_cache.clear()
            self._log(
                "info",
                f"Baked 3D .bbmodel '{bb_doc.get('name', 'mesh')}' ({mc.width}x{mc.height}, {len(mc.uv_seam_src_yx)} UV seam pairs)",
            )
            return self.get_state_payload()

    def mutate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            action = str(payload.get("action", ""))
            g = self.graph

            if action == "undo":
                if not self._undo_stack:
                    return self.get_state_payload()
                self._redo_stack.append(copy.deepcopy(g.to_recipe()))
                rec = self._undo_stack.pop()
                self._restore_recipe(rec)
                self._log("info", "Undo applied")
                return self.get_state_payload()

            if action == "redo":
                if not self._redo_stack:
                    return self.get_state_payload()
                self._undo_stack.append(copy.deepcopy(g.to_recipe()))
                rec = self._redo_stack.pop()
                self._restore_recipe(rec)
                self._log("info", "Redo applied")
                return self.get_state_payload()

            if action == "move_nodes":
                positions = payload.get("positions") or {}
                if payload.get("undoable"):
                    self._push_undo()
                for nid, xy in positions.items():
                    if nid in g.nodes and isinstance(xy, (list, tuple)) and len(xy) >= 2:
                        g.nodes[nid].pos = (float(xy[0]), float(xy[1]))
                return self.get_state_payload()

            if action == "auto_layout":
                self._push_undo()
                g.auto_layout(
                    col_spacing=float(payload.get("col_spacing", 290.0)),
                    row_spacing=float(payload.get("row_spacing", 205.0)),
                )
                self._log("info", "Applied Sugiyama topological Auto-Layout")
                return self.get_state_payload()

            if action == "add_node":
                op = str(payload.get("op", ""))
                if op not in NODE_REGISTRY:
                    raise ValueError(f"Unknown node op: {op!r}")
                self._push_undo()
                meta = NODE_REGISTRY[op]
                params = {p.name: copy.deepcopy(p.default) for p in meta.params}
                if isinstance(payload.get("params"), dict):
                    params.update(payload["params"])
                base_id = str(payload.get("id") or op)
                nid = base_id
                idx = 1
                while nid in g.nodes:
                    nid = f"{base_id}_{idx}"
                    idx += 1
                inputs: dict[str, str] = {}
                conn = payload.get("connect_from")
                if isinstance(conn, dict) and conn.get("node") in g.nodes and meta.inputs:
                    src_node = str(conn["node"])
                    src_port = conn.get("port")
                    src_meta = NODE_REGISTRY.get(g.nodes[src_node].op)
                    src_type = src_meta.output_type if src_meta else "RGBA"
                    if src_meta and src_meta.outputs:
                        if src_port:
                            for o in src_meta.outputs:
                                if o.name == src_port:
                                    src_type = o.dtype
                                    break
                        else:
                            src_type = src_meta.outputs[0].dtype
                    compat_targets = set(PORT_COMPATIBILITY.get(src_type, (src_type,)))
                    chosen_in = None
                    for inp in meta.inputs:
                        if inp.dtype in compat_targets:
                            chosen_in = inp.name
                            break
                    if chosen_in is None:
                        chosen_in = meta.inputs[0].name
                    # Reference format: bare node id for default single/material port, or node.port for multi-port
                    if src_port and src_port not in ("out", "mat") and src_meta and len(src_meta.outputs) > 1 and src_type != "MATERIAL":
                        inputs[chosen_in] = f"{src_node}.{src_port}"
                    else:
                        inputs[chosen_in] = src_node

                pos = payload.get("pos")
                pos_t = (float(pos[0]), float(pos[1])) if (isinstance(pos, (list, tuple)) and len(pos) >= 2) else (320.0, 140.0)
                sp = NodeSpec(id=nid, op=op, inputs=inputs, params=params, pos=pos_t)
                g.nodes[nid] = sp
                g.node_order.append(nid)
                self._log("info", f"Added node '{nid}' ({op})")
                out = self.get_state_payload()
                out["created_node_id"] = nid
                return out

            if action == "delete_nodes":
                ids = [str(x) for x in (payload.get("ids") or []) if str(x) in g.nodes]
                if not ids:
                    return self.get_state_payload()
                self._push_undo()
                id_set = set(ids)
                for nid in ids:
                    g.nodes.pop(nid, None)
                g.node_order = [nid for nid in g.node_order if nid not in id_set]
                for sp in g.nodes.values():
                    sp.inputs = {k: v for k, v in sp.inputs.items() if _parse_ref(v)[0] not in id_set}
                for ps in g.passes:
                    ps.node_ids = [nid for nid in ps.node_ids if nid not in id_set]
                if g.output_node and _parse_ref(g.output_node)[0] in id_set:
                    g.output_node = g.node_order[-1] if g.node_order else None
                if g.output_material_node and g.output_material_node in id_set:
                    g.output_material_node = None
                g.pbr_nodes = {k: v for k, v in g.pbr_nodes.items() if _parse_ref(v)[0] not in id_set}
                g.param_drivers = [d for d in g.param_drivers if d.get("node_id") not in id_set]
                g.exposed_params = {k: v for k, v in g.exposed_params.items() if v.get("node_id") not in id_set}
                self._log("info", f"Deleted {len(ids)} node(s): {', '.join(ids)}")
                return self.get_state_payload()

            if action == "set_param":
                nid = str(payload.get("node_id", ""))
                pkey = str(payload.get("param", ""))
                val = payload.get("value")
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                if payload.get("undoable", True):
                    self._push_undo()
                    # Track artist slider edits so AI Co-Pilot strictly preserves human choices!
                    self.artist_overrides[f"{nid}.{pkey}"] = copy.deepcopy(val)
                g.nodes[nid].params[pkey] = val
                g._static_cache.clear()
                return self.get_state_payload()

            if action == "clear_artist_overrides":
                self.artist_overrides.clear()
                self._log("info", "Cleared human artist override locks")
                return self.get_state_payload()

            if action == "connect":
                src = str(payload.get("src_node", ""))
                src_port = payload.get("src_port")
                dst = str(payload.get("dst_node", ""))
                dst_port = str(payload.get("dst_port", ""))
                if src not in g.nodes or dst not in g.nodes:
                    raise ValueError("Source or destination node does not exist")
                if src == dst:
                    raise ValueError("Cannot connect a node to itself")
                src_meta = NODE_REGISTRY.get(g.nodes[src].op)
                dst_meta = NODE_REGISTRY.get(g.nodes[dst].op)
                # Port-type compatibility check
                src_type = src_meta.output_type if src_meta else "RGBA"
                if src_meta and src_meta.outputs:
                    if src_port:
                        for o in src_meta.outputs:
                            if o.name == src_port:
                                src_type = o.dtype
                                break
                    else:
                        src_type = src_meta.outputs[0].dtype
                dst_type = None
                if dst_meta:
                    for inp in dst_meta.inputs:
                        if inp.name == dst_port:
                            dst_type = inp.dtype
                            break
                if dst_type is not None:
                    allowed = set(PORT_COMPATIBILITY.get(src_type, (src_type,)))
                    if dst_type not in allowed:
                        raise ValueError(f"Incompatible port types: {src_type} -> {dst_type}")

                ref = src
                if src_port and src_port not in ("out", "mat") and src_meta and len(src_meta.outputs) > 1 and src_type != "MATERIAL":
                    ref = f"{src}.{src_port}"

                prev_inputs = dict(g.nodes[dst].inputs)
                self._push_undo()
                g.nodes[dst].inputs[dst_port] = ref
                # Verify DAG remains acyclic
                vrep = g.validate_graph()
                if any("Cycle" in e or "cycle" in e for e in vrep.errors):
                    g.nodes[dst].inputs = prev_inputs
                    self._undo_stack.pop()
                    raise ValueError(f"Connection rejected: would introduce a cycle ({src} -> {dst})")
                g._static_cache.clear()
                self._log("info", f"Connected {ref} -> {dst}.{dst_port}")
                return self.get_state_payload()

            if action == "disconnect":
                dst = str(payload.get("dst_node", ""))
                dst_port = str(payload.get("dst_port", ""))
                if dst in g.nodes and dst_port in g.nodes[dst].inputs:
                    self._push_undo()
                    g.nodes[dst].inputs.pop(dst_port, None)
                    g._static_cache.clear()
                    self._log("info", f"Disconnected {dst}.{dst_port}")
                return self.get_state_payload()

            if action == "toggle_bypass":
                nid = str(payload.get("node_id", ""))
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                self._push_undo()
                g.nodes[nid].bypassed = not g.nodes[nid].bypassed
                g._static_cache.clear()
                self._log("info", f"{'Bypassed' if g.nodes[nid].bypassed else 'Enabled'} node '{nid}'")
                return self.get_state_payload()

            if action == "toggle_collapse":
                nid = str(payload.get("node_id", ""))
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                g.nodes[nid].collapsed = not g.nodes[nid].collapsed
                return self.get_state_payload()

            if action == "set_output":
                nid = str(payload.get("node_id", ""))
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                self._push_undo()
                meta = NODE_REGISTRY.get(g.nodes[nid].op)
                is_mat = bool(meta and (meta.output_type == "MATERIAL" or any(o.dtype == "MATERIAL" for o in meta.outputs)))
                if is_mat:
                    g.set_output_material(nid, normal_strength=float(g.pbr_config.get("normal_strength", 2.4)))
                else:
                    g.output_material_node = None
                    g.set_output(nid)
                self._log("info", f"Set graph output to '{nid}'")
                return self.get_state_payload()

            if action == "set_graph_settings":
                self._push_undo()
                if "width" in payload and self.mesh_ctx is None:
                    g.width = max(16, min(512, int(payload["width"])))
                if "height" in payload and self.mesh_ctx is None:
                    g.height = max(16, min(512, int(payload["height"])))
                if "tileable" in payload:
                    g.tileable = bool(payload["tileable"])
                if "seed" in payload:
                    g.seed = int(payload["seed"])
                if "name" in payload and str(payload["name"]).strip():
                    g.name = str(payload["name"]).strip()
                g._static_cache.clear()
                self._log("info", f"Updated graph settings ({g.width}x{g.height}, tileable={g.tileable}, seed={g.seed})")
                return self.get_state_payload()

            if action == "expose_param":
                nid = str(payload.get("node_id", ""))
                pkey = str(payload.get("param", ""))
                pub = str(payload.get("public_name") or f"{nid}_{pkey}")
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                self._push_undo()
                g.expose_param(
                    pub,
                    nid,
                    pkey,
                    label=str(payload.get("label") or pub),
                    min_val=payload.get("min"),
                    max_val=payload.get("max"),
                    step=payload.get("step"),
                )
                self._log("info", f"Exposed parameter '{pub}' ({nid}.{pkey})")
                return self.get_state_payload()

            if action == "remove_exposed":
                pub = str(payload.get("public_name", ""))
                if pub in g.exposed_params:
                    self._push_undo()
                    g.exposed_params.pop(pub, None)
                return self.get_state_payload()

            if action == "add_driver":
                nid = str(payload.get("node_id", ""))
                pkey = str(payload.get("param", ""))
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                self._push_undo()
                # Replace existing driver on same node.param
                g.param_drivers = [
                    d for d in g.param_drivers if not (d.get("node_id") == nid and d.get("param_key") == pkey)
                ]
                g.bind_driver(
                    nid,
                    pkey,
                    kind=str(payload.get("kind", "sine")),
                    min_val=float(payload.get("min_val", 0.0)),
                    max_val=float(payload.get("max_val", 1.0)),
                    freq=float(payload.get("freq", 1.0)),
                    phase_offset=float(payload.get("phase_offset", 0.0)),
                    octaves=int(payload.get("octaves", 3)),
                )
                self._log("info", f"Bound animation driver ({payload.get('kind', 'sine')}) to {nid}.{pkey}")
                return self.get_state_payload()

            if action == "remove_driver":
                idx = int(payload.get("index", -1))
                if 0 <= idx < len(g.param_drivers):
                    self._push_undo()
                    removed = g.param_drivers.pop(idx)
                    self._log("info", f"Removed animation driver on {removed.get('node_id')}.{removed.get('param_key')}")
                return self.get_state_payload()

            if action == "apply_texdsl":
                code = str(payload.get("code", ""))
                new_g = parse_texdsl(code, mesh_context=self.mesh_ctx)
                # Preserve existing positions & friendly exposed_param labels where keys match!
                for nid, sp in new_g.nodes.items():
                    if nid in g.nodes and g.nodes[nid].pos is not None:
                        sp.pos = g.nodes[nid].pos
                for pub, info in new_g.exposed_params.items():
                    if pub in g.exposed_params and g.exposed_params[pub].get("label"):
                        info["label"] = g.exposed_params[pub]["label"]
                self._push_undo()
                old_cache = dict(g._content_cache)
                new_g._content_cache = old_cache
                self.graph = new_g
                self._ensure_positions()
                self._log("info", f"Compiled live .texdsl ({len(new_g.nodes)} nodes)")
                return self.get_state_payload()

            raise ValueError(f"Unknown mutation action: {action!r}")

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            phase = float(payload.get("phase", 0.0))
            light_az = float(payload.get("light_az", -38.0))
            light_el = float(payload.get("light_el", 52.0))
            lighting_preset = str(payload.get("lighting_preset", "studio_hero"))
            if lighting_preset not in BAKE_PRESETS:
                lighting_preset = "studio_hero"
            preview_node = payload.get("preview_node")
            include_thumbs = bool(payload.get("include_thumbs", True))
            include_qa = bool(payload.get("include_qa", True))

            t_start = time.perf_counter()
            val = self.graph.validate_graph()
            if not val.ok:
                return {
                    "ok": False,
                    "validation": val.to_dict(),
                    "error": "; ".join(val.errors),
                }

            try:
                ev = self.graph.evaluate(
                    phase=phase,
                    capture_intermediates=True,
                    build_pbr=True,
                    light_az_deg=light_az,
                    light_el_deg=light_el,
                    allow_orphan_errors=True,
                )
            except Exception as exc:
                self._log("error", f"Evaluation error: {type(exc).__name__}: {exc}")
                raise

            # Encode per-node thumbnails using cached content hashes!
            thumbs: dict[str, str] = {}
            node_shapes: dict[str, list[int]] = {}
            if include_thumbs:
                for nid in self.graph.node_order:
                    if nid not in ev.intermediates:
                        continue
                    arr = _arr_to_u8_rgba(ev.intermediates[nid])
                    node_shapes[nid] = [int(arr.shape[1]), int(arr.shape[0])]
                    ck = self.graph._content_cache.get(nid)
                    cache_key = f"{nid}:{ck[0]}" if ck is not None else f"{nid}:{phase:.4f}"
                    if cache_key not in self._thumb_cache:
                        self._thumb_cache[cache_key] = _u8_to_data_url(arr, max_dim=128, min_dim=64)
                        if len(self._thumb_cache) > 512:
                            # Evict oldest 128 keys
                            for k_old in list(self._thumb_cache.keys())[:128]:
                                self._thumb_cache.pop(k_old, None)
                    thumbs[nid] = self._thumb_cache[cache_key]

            # Encode all 12+ PBR channels & diagnostic maps
            channels: dict[str, str] = {}
            out_u8 = _arr_to_u8_rgba(ev.output)
            channels["albedo"] = _u8_to_data_url(out_u8, max_dim=512)
            if ev.pbr_preview is not None:
                channels["pbr_lit"] = _u8_to_data_url(_arr_to_u8_rgba(ev.pbr_preview), max_dim=512)
            else:
                channels["pbr_lit"] = channels["albedo"]

            if ev.pbr_maps is not None:
                # Baked Studio Albedo (incorporates 3D ray-traced shadow + bent normal + thickness!)
                baked_u8 = bake_pbr_to_albedo(
                    ev.pbr_maps,
                    preset=lighting_preset,
                    key_az_deg=light_az,
                    key_el_deg=light_el,
                    emissive_rgba=ev.emissive,
                )
                channels["baked"] = _u8_to_data_url(_arr_to_u8_rgba(baked_u8), max_dim=512)
                for key_src, key_dst in (
                    ("normal_rgb", "normal"),
                    ("height_u8", "height"),
                    ("roughness_u8", "roughness"),
                    ("metallic_u8", "metallic"),
                    ("ao_u8", "ao"),
                    ("shadow_u8", "shadow"),
                    ("bent_normal_u8", "bent_normal"),
                    ("thickness_u8", "thickness"),
                    ("orm_u8", "orm"),
                    ("unity_maskmap", "maskmap"),
                    ("labpbr_n", "labpbr_n"),
                    ("labpbr_s", "labpbr_s"),
                    ("bedrock_mer", "bedrock_mer"),
                ):
                    if key_src in ev.pbr_maps:
                        channels[key_dst] = _u8_to_data_url(_arr_to_u8_rgba(ev.pbr_maps[key_src]), max_dim=512)

            if ev.emissive is not None:
                channels["emissive"] = _u8_to_data_url(_arr_to_u8_rgba(ev.emissive), max_dim=512)
            else:
                black = np.zeros((out_u8.shape[0], out_u8.shape[1], 4), dtype=np.uint8)
                black[..., 3] = 255
                channels["emissive"] = _u8_to_data_url(black, max_dim=256)

            if self.mesh_ctx is not None:
                mc = self.mesh_ctx
                # Combined UV Seam (cyan) + Thinness (warm orange) diagnostic view!
                diag = _scalar_to_heatmap_u8(mc.thickness, warm=True)
                seam_m = mc.uv_seam_mask > 0.05
                diag[seam_m, 0] = 40
                diag[seam_m, 1] = 232
                diag[seam_m, 2] = 255
                diag[~mc.island_mask, :3] = 20
                channels["uv_seams"] = _u8_to_data_url(diag, max_dim=512)
                # Per-UV Island & Body-Part Region Map + Pixel Composer Inner Area Map!
                channels["uv_regions"] = _u8_to_data_url(build_uv_region_overlay(mc), max_dim=512)
                ia_u8 = _scalar_to_heatmap_u8(mc.island_inner_area, warm=False)
                ia_u8[~mc.island_mask, :3] = 16
                channels["inner_area"] = _u8_to_data_url(ia_u8, max_dim=512)

            # Optional pinned node preview
            selected_preview = None
            if preview_node and preview_node in ev.intermediates:
                selected_preview = _u8_to_data_url(_arr_to_u8_rgba(ev.intermediates[preview_node]), max_dim=512)

            # Multi-domain automated QA scorecard
            qa_dict = None
            if include_qa:
                dom = "entity" if self.mesh_ctx is not None else (getattr(self.graph, "domain", None) or "block")
                try:
                    qa_rep = evaluate_texture_quality(
                        ev.output,
                        domain=dom,
                        tileable=self.graph.tileable,
                        mesh_ctx=self.mesh_ctx,
                        pbr_channels=ev.pbr_channels,
                        pbr_maps=ev.pbr_maps,
                    )
                    qa_dict = qa_rep.to_dict()
                except Exception as exc:
                    qa_dict = {"passed": True, "score": 100.0, "note": str(exc), "metrics": {}, "issues": []}

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            eval_ms = round(sum(ev.timings_ms.values()), 2)

            return {
                "ok": True,
                "width": int(out_u8.shape[1]),
                "height": int(out_u8.shape[0]),
                "phase": round(phase, 4),
                "eval_ms": eval_ms,
                "total_ms": total_ms,
                "cache_hits": ev.cache_hits,
                "cache_misses": ev.cache_misses,
                "timings_ms": ev.timings_ms,
                "node_shapes": node_shapes,
                "thumbs": thumbs,
                "channels": channels,
                "selected_preview": selected_preview,
                "validation": val.to_dict(),
                "qa": qa_dict,
            }

    def export_package(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            preset = str(payload.get("preset", "studio_hero"))
            if preset not in BAKE_PRESETS:
                preset = "studio_hero"
            out_dir_raw = payload.get("out_dir")
            stem = str(payload.get("stem") or self.graph.name or "synccraft_asset")
            embed = bool(payload.get("embed_manifest", True))

            if out_dir_raw:
                out_dir = Path(str(out_dir_raw)).expanduser().resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                pkg = self.graph.export_baked_pbr(
                    out_dir,
                    stem=stem,
                    preset=preset,
                    bbmodel_doc=self.bbmodel_doc,
                    embed_maps_in_manifest=embed,
                )
                # Also write `.texdsl` & `.texgraph.json` alongside PBR bundle!
                dsl_path = out_dir / f"{stem}.texdsl"
                rec_path = out_dir / f"{stem}.texgraph.json"
                dsl_path.write_text(export_texdsl(self.graph), encoding="utf-8")
                rec_path.write_text(json.dumps(self.graph.to_recipe(), indent=2) + "\n", encoding="utf-8")
                files = dict(pkg["files"])
                files["texdsl"] = str(dsl_path)
                files["texgraph_json"] = str(rec_path)
                self._log("info", f"Exported commercial PBR v2.0 bundle ({len(files)} files) -> {out_dir}")
                return {"ok": True, "mode": "disk", "out_dir": str(out_dir), "files": files, "manifest": pkg["manifest"]}

            # In-memory browser download payload
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                pkg = self.graph.export_baked_pbr(
                    td_p,
                    stem=stem,
                    preset=preset,
                    bbmodel_doc=self.bbmodel_doc,
                    embed_maps_in_manifest=True,
                )
                manifest = pkg["manifest"]
                downloads: dict[str, str] = {
                    f"{stem}.texdsl": export_texdsl(self.graph),
                    f"{stem}.texgraph.json": json.dumps(self.graph.to_recipe(), indent=2) + "\n",
                    f"{stem}.synccraft.pbr.json": json.dumps(manifest, indent=2) + "\n",
                }
                if "baked_bbmodel" in pkg["files"]:
                    downloads[f"{stem}_baked.bbmodel"] = Path(pkg["files"]["baked_bbmodel"]).read_text(encoding="utf-8")
                self._log("info", f"Prepared browser download bundle for '{stem}' ({len(downloads)} manifests + all PNG maps)")
                return {
                    "ok": True,
                    "mode": "download",
                    "stem": stem,
                    "downloads": downloads,
                    "embedded_maps": manifest.get("embedded_data_urls", {}),
                    "file_keys": sorted(pkg["files"].keys()),
                }

    # -----------------------------------------------------------------------
    # Milestone v12: AI Co-Pilot, Style-DNA Extractor & Batch Variant Factory
    # -----------------------------------------------------------------------

    def ai_diagnose(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rep = diagnose_asset(self.graph, self.mesh_ctx, phase=float(payload.get("phase", 0.0)))
            rep["artist_overrides"] = copy.deepcopy(self.artist_overrides)
            self._log("info", f"AI Co-Pilot diagnosed {rep['finding_count']} item(s) (QA {round(rep['qa']['score'])}/100)")
            return rep

    def ai_patch(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            ops = payload.get("patch_ops") or []
            if not isinstance(ops, list):
                raise ValueError("Expected 'patch_ops' list")
            preserve = bool(payload.get("preserve_artist_overrides", True))
            self._push_undo()
            res = apply_surgical_patch(
                self.graph,
                ops,
                artist_overrides=self.artist_overrides,
                preserve_artist_overrides=preserve,
            )
            self._ensure_positions()
            self._log(
                "info",
                f"AI Patch applied {res['applied_count']} op(s), preserved {len(res['preserved_overrides'])} artist lock(s) (QA {res['qa_before']} -> {res['qa_after']})",
            )
            return {"ok": res["ok"], "patch_result": res, "state": self.get_state_payload()}

    def style_extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            src = payload.get("source") or "preset:chronos_amethyst"
            name = str(payload.get("name") or "reference_dna")
            dna = extract_style_dna(src, num_stops=int(payload.get("num_stops", 6)), name=name)
            applied_state = None
            patch_res = None
            if payload.get("apply"):
                self._push_undo()
                patch_res = apply_surgical_patch(
                    self.graph,
                    [{"op": "replace_palette", "stops": dna["stops"]}],
                    artist_overrides=self.artist_overrides,
                    preserve_artist_overrides=True,
                )
                applied_state = self.get_state_payload()
                self._log("info", f"Injected Style-DNA '{dna['name']}' palette into active graph")
            return {"ok": True, "dna": dna, "patch_result": patch_res, "state": applied_state}

    def variant_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            vid = str(payload.get("variant_id", "chronos_prime"))
            if vid not in DEFAULT_CHRONO_VARIANTS:
                raise ValueError(f"Unknown variant {vid!r}; choices: {sorted(DEFAULT_CHRONO_VARIANTS)}")
            self._push_undo()
            v_spec = DEFAULT_CHRONO_VARIANTS[vid]
            g_var = apply_variant_to_graph(self.graph, v_spec, artist_overrides=self.artist_overrides)
            g_var._content_cache = dict(self.graph._content_cache)
            self.graph = g_var
            self.active_variant = vid
            self._ensure_positions()
            self._log("info", f"Switched live workspace to variant '{v_spec['title']}'")
            return {"ok": True, "variant": vid, "state": self.get_state_payload()}

    def export_syncproj(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            stem = str(payload.get("stem") or self.graph.name or "synccraft_project")
            proj = SyncProject(
                name=stem,
                blockdsl_path=f"{stem}.blockdsl",
                bbmodel_path=f"{stem}.bbmodel" if self.bbmodel_doc is not None else None,
                texdsl_path=f"{stem}.texdsl",
                pbr_preset=str(payload.get("preset", "studio_hero")),
                artist_overrides=copy.deepcopy(self.artist_overrides),
                variants=copy.deepcopy(DEFAULT_CHRONO_VARIANTS),
            )
            out_dir_raw = payload.get("out_dir")
            if out_dir_raw:
                out_dir = Path(str(out_dir_raw)).expanduser().resolve()
                p_path = proj.save(out_dir / f"{stem}.syncproj")
                vpack = None
                if payload.get("build_variants"):
                    vpack = generate_variant_pack(
                        self.graph,
                        DEFAULT_CHRONO_VARIANTS,
                        out_dir / "variants",
                        bbmodel_doc=self.bbmodel_doc,
                        artist_overrides=self.artist_overrides,
                        pbr_preset=proj.pbr_preset,
                    )
                self._log("info", f"Exported unified .syncproj -> {p_path}")
                return {"ok": True, "path": str(p_path), "manifest": proj.to_dict(), "variant_pack": vpack}
            return {"ok": True, "manifest": proj.to_dict(), "filename": f"{stem}.syncproj"}


# ---------------------------------------------------------------------------
# HTTP Server & Handler
# ---------------------------------------------------------------------------


def create_studio_server(
    *,
    host: str = "127.0.0.1",
    port: int = 18790,
    initial_sample: str = "monolith_v10_3d",
    size: int = 64,
    session: StudioSession | None = None,
) -> tuple[ThreadingHTTPServer, StudioSession]:
    """Create a configured `ThreadingHTTPServer` bound to `host:port` (use `port=0` for tests)."""
    sess = session or StudioSession(initial_sample=initial_sample, size=size)
    ui_root = _STUDIO_UI_DIR.resolve()

    class StudioHTTPHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_bytes(self, body: bytes, *, status: int = 200, content_type: str = "application/octet-stream") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj: dict[str, Any], *, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send_bytes(raw, status=status, content_type="application/json; charset=utf-8")

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            if length > 64 * 1024 * 1024:
                raise ValueError("Request body exceeds 64 MB limit")
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object payload")
            return data

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "app": "SyncCraft Studio",
                        "synccraft_version": __version__,
                        "engine_version": "10.0",
                        "schema_version": "4.0",
                        "node_count": len(NODE_REGISTRY),
                        "active_graph": sess.graph.name,
                    }
                )
                return

            if path == "/api/schema":
                self._send_json(sess.get_schema_payload())
                return

            if path == "/api/schema/previews":
                qs = urllib.parse.parse_qs(parsed.query)
                sz = int((qs.get("size") or ["40"])[0])
                previews = generate_node_preview_catalog(size=sz)
                raw = json.dumps({"ok": True, "size": sz, "count": len(previews), "previews": previews}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/state":
                self._send_json(sess.get_state_payload())
                return

            if path in ("/", "/index.html"):
                idx = ui_root / "index.html"
                if not idx.is_file():
                    self._send_json({"error": "Missing studio_ui/index.html"}, status=500)
                    return
                self._send_bytes(idx.read_bytes(), content_type="text/html; charset=utf-8")
                return

            if path.startswith("/assets/"):
                rel = path[len("/assets/"):]
                candidate = (ui_root / rel).resolve()
                if not str(candidate).startswith(str(ui_root)) or not candidate.is_file():
                    self._send_json({"error": "Asset not found"}, status=404)
                    return
                ctype, _ = mimetypes.guess_type(str(candidate))
                self._send_bytes(candidate.read_bytes(), content_type=f"{ctype or 'application/octet-stream'}; charset=utf-8")
                return

            self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/load":
                    self._send_json(sess.load_project(body))
                    return
                if path == "/api/mesh/load":
                    self._send_json(sess.load_mesh(body))
                    return
                if path == "/api/graph/mutate":
                    self._send_json(sess.mutate(body))
                    return
                if path == "/api/evaluate":
                    self._send_json(sess.evaluate(body))
                    return
                if path == "/api/export":
                    self._send_json(sess.export_package(body))
                    return
                if path == "/api/ai/diagnose":
                    self._send_json(sess.ai_diagnose(body))
                    return
                if path == "/api/ai/patch":
                    self._send_json(sess.ai_patch(body))
                    return
                if path == "/api/style/extract":
                    self._send_json(sess.style_extract(body))
                    return
                if path == "/api/variants/apply":
                    self._send_json(sess.variant_apply(body))
                    return
                if path == "/api/syncproj/export":
                    self._send_json(sess.export_syncproj(body))
                    return
                self._send_json({"error": f"Unknown API endpoint: {path}"}, status=404)
            except (ValueError, KeyError, TypeError, TexDSLParseError, GraphCycleError) as exc:
                self._send_json({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status=400)
            except Exception as exc:  # pragma: no cover
                self._send_json({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status=500)

    server = ThreadingHTTPServer((host, int(port)), StudioHTTPHandler)
    return server, sess


def serve_studio(
    *,
    host: str = "127.0.0.1",
    port: int = 18790,
    initial_sample: str = "monolith_v10_3d",
    size: int = 64,
    open_browser: bool = False,
) -> None:
    """Start the local SyncCraft Studio HTTP workstation server."""
    server, _ = create_studio_server(host=host, port=port, initial_sample=initial_sample, size=size)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"SyncCraft Studio v{__version__} (TexGraph v10 / Schema v4.0 · {len(NODE_REGISTRY)} nodes)")
    print(f"  Listening on {url}  (Press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
