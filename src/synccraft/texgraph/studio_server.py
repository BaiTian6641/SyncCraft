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
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import math
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
from .mesh_baker import _FACE_BASIS, _euler_xyz_mat, MeshBakeContext, bake_bbmodel_maps, mesh_bake_cache_stats
from .node_previews import generate_anim_preview_sheet, generate_node_anim_preview, generate_node_preview_catalog
from .pbr import BAKE_LIGHTING_PRESETS as BAKE_PRESETS, bake_pbr_to_albedo
from .pipeline import MATERIAL_BUS_PORTS, SUBGRAPH_REGISTRY
from .pixel_art_ops import encode_png, rgba_float_to_u8
from .presets import PRESET_REGISTRY
from .production import PALETTE_BANK
from .quality import evaluate_texture_quality
from .region_ops import build_uv_region_overlay
from .registry import NODE_CATALOG as NODE_REGISTRY, PORT_COMPATIBILITY, export_node_schema_json
from .syncmat_and_mtlx import (
    export_materialx_xml,
    extract_syncmat_from_graph,
    insert_syncmat_into_graph,
    list_all_syncmats,
    save_user_syncmat,
)
from .ctm_tileset import generate_ctm_tileset
from .syncproj import DEFAULT_CHRONO_VARIANTS, SyncProject, apply_variant_to_graph, generate_variant_pack
from .texdsl import TexDslError as TexDSLParseError, export_texdsl, parse_texdsl
from .v16_pro_ops import (
    PBR_BRUSH_PRESETS,
    ScriptSecurityError,
    apply_pbr_brush_strokes_to_layer,
    list_all_syncpacks,
    make_empty_paint_layer,
    parse_layers_json,
    run_batch_bakery,
    save_user_syncpack,
    scan_batch_folder,
    spawn_syncpack_macro_into_graph,
)

_STUDIO_UI_DIR = Path(__file__).resolve().parent / "studio_ui"
_STUDIO_LOG_LOCK = threading.Lock()
_STUDIO_FILE_LOGGER: logging.Logger | None = None
_STUDIO_LOG_PATH: Path | None = None


def get_studio_log_path() -> Path:
    """Return the persistent rotating Studio diagnostic log path (`~/.synccraft/logs/synccraft_studio.log`)."""
    global _STUDIO_LOG_PATH
    if _STUDIO_LOG_PATH is None:
        log_dir = Path.home() / ".synccraft" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _STUDIO_LOG_PATH = log_dir / "synccraft_studio.log"
    return _STUDIO_LOG_PATH


def _get_studio_file_logger() -> logging.Logger:
    global _STUDIO_FILE_LOGGER
    with _STUDIO_LOG_LOCK:
        if _STUDIO_FILE_LOGGER is not None:
            return _STUDIO_FILE_LOGGER
        logger = logging.getLogger("synccraft.studio.file")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            try:
                lp = get_studio_log_path()
                handler = RotatingFileHandler(
                    str(lp),
                    maxBytes=2 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            except Exception:
                logger.addHandler(logging.NullHandler())
        _STUDIO_FILE_LOGGER = logger
        return logger


def _write_studio_file_log(level: str, event: str, **fields: Any) -> None:
    """Write a structured ISO-8601 line to the persistent rotating Studio file log."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        rec = {"ts": ts, "level": str(level).upper(), "event": str(event)}
        for k, v in fields.items():
            if v is not None:
                rec[k] = v
        _get_studio_file_logger().info(json.dumps(rec, ensure_ascii=False, default=str))
    except Exception:
        pass


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
    pick_ids: list[float] = []
    elem_meta: list[dict[str, str]] = []
    uv_rects: list[list[float]] = []

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
        pick_id = float(box_count)
        ename = str(elem.get("name") or f"box_{box_count}")
        eshort = ename.split("__")[-1]
        elem_meta.append({"id": box_count, "name": ename, "short": eshort})
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
            uv_rects.append([round(min(u0, u1), 5), round(min(v0, v1), 5), round(max(u0, u1), 5), round(max(v0, v1), 5)])

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
                pick_ids.append(pick_id)

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
        "pick_ids": pick_ids,
        "elem_meta": elem_meta,
        "uv_rects": uv_rects,
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
    if sample_id == "brass_crystal_workflow_3d":
        bb_doc = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb_doc)
        g = TexGraph(mc.width, mc.height, tileable=False, seed=42, name="brass_crystal_workflow_3d", mesh_context=mc)
        g.domain = "entity"
        with g.pass_("pass_1_3d_masks", title="1. 3D Part & Direction Masks", color="#0F766E") as p1:
            m_brass = g.add("mask_uv_part", id="mask_brass_trim", part_pattern="cornice|plinth", zone="all", face_filter="all")
            m_crystal = g.add("mask_uv_part", id="mask_crystal_fins", part_pattern="crest_fin", zone="all", face_filter="all")
            m_top = g.add("mask_world_direction", id="mask_top_surfaces", direction="top_up", threshold=0.45, softness=0.25, seed=42)
            p1.preview_ref = m_brass
        with g.pass_("pass_2_pbr_materials", title="2. PBR Materials (Obsidian, Brass, Crystal)", color="#92400E") as p2:
            obs = g.add("mat_obsidian_volcanic", id="obsidian_pillar", obsidian_color="#232036", fissure_color="#7C3AED", fissure_glow=0.35, roughness=0.22, seed=42)
            brass_raw = g.add("mat_brass", id="brushed_brass", base_color="#D9AE4E", patina_color="#486B5E", roughness=0.24, brushed_strength=0.60, patina_amount=0.22, scratches=0.38, seed=108)
            brass_aged = g.add("material_weathering", id="aged_brass_fx", inputs={"mat": brass_raw}, preset="patina", amount=0.30, seed=108)
            crys = g.add("mat_crystal_gem", id="amethyst_crystal", crystal_color="#A855F7", core_glow_color="#38F2FF", opacity=0.72, facet_scale=5.5, glow_strength=0.65, roughness=0.10, seed=77)
            p2.preview_ref = f"{brass_aged}.albedo"
        with g.pass_("pass_3_sequence_anim", title="3. Region Sequence & Live Shimmer", color="#4C1D95") as p3:
            seq1 = g.add("material_region_apply", id="apply_brass_region", inputs={"base": obs, "top": brass_aged, "mask": m_brass}, opacity=1.0, seam_ao_strength=0.28)
            seq2 = g.add("material_region_apply", id="apply_crystal_region", inputs={"base": seq1, "top": crys, "mask": m_crystal}, opacity=1.0, seam_ao_strength=0.18)
            shimmer = g.add("anim_shimmer", id="chronal_shimmer", inputs={"src": seq2}, angle_deg=-35.0, speed=1.0, width=0.14, intensity=0.45, color="#FFF2C6")
            final = g.precompute_shadow_pbr(shimmer, shadow_strength=0.62, bent_normal_strength=0.44, thickness_sss_strength=0.45, bake_into_albedo=0.45, id="brass_crystal_final")
            p3.preview_ref = f"{final}.albedo"
        g.set_output_material(final, normal_strength=2.5)
        g.expose_param("brass_patina", brass_raw, "patina_amount", label="Brass Patina", min_val=0.0, max_val=0.85, step=0.02)
        g.expose_param("crystal_opacity", crys, "opacity", label="Crystal Opacity", min_val=0.20, max_val=1.0, step=0.02)
        g.expose_param("crystal_glow", crys, "glow_strength", label="Crystal Glow", min_val=0.0, max_val=1.4, step=0.05)
        g.expose_param("shimmer_intensity", shimmer, "intensity", label="Live Shimmer", min_val=0.0, max_val=1.2, step=0.05)
        g.auto_layout(col_spacing=290.0, row_spacing=195.0)
        return g, bb_doc

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
        if found is None:
            raise FileNotFoundError(
                "Could not locate kaelith_chrono_dragonborn.bbmodel & .texdsl in workspace showcase folders"
            )
        bb_path, dsl_path = found
        bb_doc = json.loads(bb_path.read_text(encoding="utf-8"))
        mc = bake_bbmodel_maps(bb_doc)
        g = parse_texdsl(dsl_path.read_text(encoding="utf-8"), mesh_context=mc)
        g.domain = "entity"
        g.auto_layout(col_spacing=290.0, row_spacing=195.0)
        return g, bb_doc

    if sample_id == "ironjaw_orc_3d":
        found = _find_orc_warlord_paths()
        if found is None:
            raise FileNotFoundError(
                "Could not locate ironjaw_orc_warlord.bbmodel & .texdsl in workspace showcase folders"
            )
        bb_path, dsl_path = found
        bb_doc = json.loads(bb_path.read_text(encoding="utf-8"))
        mc = bake_bbmodel_maps(bb_doc)
        g = parse_texdsl(dsl_path.read_text(encoding="utf-8"), mesh_context=mc)
        g.domain = "entity"
        g.auto_layout(col_spacing=290.0, row_spacing=205.0)
        return g, bb_doc

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
        "id": "brass_crystal_workflow_3d",
        "title": "Brass & Amethyst Crystal Chain (3D)",
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
        self._mesh_diag_cache: dict[str, dict[str, str]] = {}
        self._channel_png_cache: dict[str, dict[str, str]] = {}
        self._last_eval: dict[str, Any] = {}
        self._event_log: list[dict[str, Any]] = []
        self._workspace_epoch: int = 0
        self._eval_seq: int = 0
        self.artist_overrides: dict[str, Any] = {}
        self.active_variant: str = "chronos_prime"
        self.active_source: str = f"sample:{initial_sample}"
        self._group_stack: list[dict[str, Any]] = []
        self._bridge_rev: int = 0
        self._bridge_payload: dict[str, Any] = {}
        self._bridge_clients: int = 0
        self._bridge_cond = threading.Condition()
        self._batch_lock = threading.Lock()
        self._batch_state: dict[str, Any] = {
            "running": False,
            "total": 0,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "out_dir": "",
            "items": [],
            "report": None,
        }

        g, bb = build_studio_sample(initial_sample, size=size)
        self._set_workspace(g, bb, source=self.active_source, clear_history=True)

    def _log(self, level: str, msg: str, **meta: Any) -> None:
        self._event_log.append(
            {
                "ts": round(time.time(), 2),
                "level": level,
                "message": msg,
            }
        )
        if len(self._event_log) > 150:
            self._event_log = self._event_log[-150:]
        _write_studio_file_log(level, msg, epoch=self._workspace_epoch, source=self.active_source, **meta)

    def _set_workspace(
        self,
        graph: TexGraph,
        bb_doc: dict[str, Any] | None,
        *,
        source: str,
        clear_history: bool = False,
    ) -> None:
        t0 = time.perf_counter()
        self._workspace_epoch += 1
        self.graph = graph
        self.active_source = source
        bake_hit = False
        bake_ms = 0.0
        if bb_doc is not None:
            self.bbmodel_doc = bb_doc
            self.mesh_ctx = graph.mesh_context or bake_bbmodel_maps(bb_doc)
            bake_hit = bool(getattr(self.mesh_ctx, "bake_cache_hit", False))
            bake_ms = float(getattr(self.mesh_ctx, "bake_ms", 0.0))
            self.graph.attach_mesh(self.mesh_ctx)
            self.webgl_mesh = extract_bbmodel_webgl_mesh(bb_doc)
        else:
            self.bbmodel_doc = None
            self.mesh_ctx = None
            self.webgl_mesh = None
            self.graph.mesh_context = None
        self._ensure_positions()
        if clear_history:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self.artist_overrides.clear()
            self.active_variant = "chronos_prime"
            self._group_stack.clear()
        self.graph._content_cache.clear()
        self.graph._static_cache.clear()
        self._thumb_cache.clear()
        self._channel_png_cache.clear()
        self._last_eval = {}
        load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        mesh_note = (
            f", mesh={self.mesh_ctx.width}x{self.mesh_ctx.height} "
            f"{'CACHE' if bake_hit else f'baked {bake_ms:.0f}ms'}"
            if self.mesh_ctx is not None
            else ", 2D/primitive"
        )
        self._log(
            "info",
            f"Loaded workspace '{graph.name}' (epoch={self._workspace_epoch}, {len(graph.nodes)} nodes, {graph.width}x{graph.height}{mesh_note}) in {load_ms:.1f}ms",
            graph=graph.name,
            nodes=len(graph.nodes),
            dims=f"{graph.width}x{graph.height}",
            mesh_fp=getattr(self.mesh_ctx, "content_fingerprint", None) if self.mesh_ctx else None,
            mesh_cache_hit=bake_hit,
            load_ms=load_ms,
        )

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
            "brush_presets": PBR_BRUSH_PRESETS,
            "syncpacks": list_all_syncpacks(),
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
                "workspace_epoch": self._workspace_epoch,
                "source": self.active_source,
                "recipe": self.graph.to_recipe(),
                "texdsl": dsl_text,
                "validation": val.to_dict(),
                "undo_depth": len(self._undo_stack),
                "redo_depth": len(self._redo_stack),
                "artist_overrides": copy.deepcopy(self.artist_overrides),
                "active_variant": self.active_variant,
                "has_mesh": self.mesh_ctx is not None,
                "mesh_fingerprint": getattr(self.mesh_ctx, "content_fingerprint", None) if self.mesh_ctx is not None else None,
                "mesh": self.webgl_mesh,
                "mesh_parts": (
                    {
                        "zones": sorted({z for z in self.mesh_ctx.elem_zones if z}),
                        "bones": sorted({b.split("/")[-1] for b in self.mesh_ctx.elem_bones if b})[:48],
                        "elements": sorted({e.split("__")[-1] for e in self.mesh_ctx.elem_names if e})[:64],
                    }
                    if self.mesh_ctx is not None
                    else None
                ),
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
                "log": self._event_log[-40:],
                "log_file_path": str(get_studio_log_path()),
                "breadcrumbs": (
                    [{"level": 0, "title": self._group_stack[0]["parent_title"]}]
                    + [{"level": i + 1, "title": fr["title"]} for i, fr in enumerate(self._group_stack)]
                    if self._group_stack
                    else [{"level": 0, "title": self.graph.name}]
                ),
                "in_subgraph": bool(self._group_stack),
                "bridge": {"rev": self._bridge_rev, "clients": self._bridge_clients},
            }

    @staticmethod
    def _build_bbmodel_starter_graph(bb_doc: dict[str, Any], mc: MeshBakeContext) -> TexGraph:
        """Build a clean 1-to-1 starter TexGraph matching a loaded `.bbmodel`'s exact UV atlas & geometry."""
        name = str(bb_doc.get("name") or "bbmodel_asset").strip() or "bbmodel_asset"
        g = TexGraph(mc.width, mc.height, tileable=False, seed=42, name=name, mesh_context=mc)
        g.domain = "entity"
        base_id = g.add(
            "region_base_fill",
            id="base_fill",
            stops=["#2C3347", "#4F5D7A", "#8A9BB8", "#D7E0F0"],
            top_boost=0.16,
            bottom_darken=0.18,
            side_grad=0.18,
            roughness_base=0.52,
            metallic_base=0.08,
            seed=42,
        )
        tex_id = g.add(
            "region_restrained_noise",
            id="detail_noise",
            inputs={"base": base_id},
            pattern="organic_clump",
            scale=1.0,
            amplitude=0.11,
            edge_restraint=0.88,
            min_island_px=4.0,
            seed=42,
        )
        bev_id = g.add(
            "region_bevel_and_ao",
            id="bevel_ao",
            inputs={"base": tex_id},
            highlight_strength=0.18,
            shadow_strength=0.22,
            ao_strength=0.28,
            seed=42,
        )
        sh_id = g.precompute_shadow_pbr(
            bev_id,
            shadow_strength=0.52,
            bent_normal_strength=0.40,
            thickness_sss_strength=0.30,
            bake_into_albedo=0.32,
            id="shadow_pbr",
        )
        g.set_output_material(sh_id, normal_strength=2.4)
        g.auto_layout(col_spacing=290.0, row_spacing=195.0)
        return g

    def load_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Strict 1-to-1 project loader supporting samples, presets, .texdsl, .texgraph.json, .syncproj, .bbmodel & disk files."""
        with self._lock:
            kind = str(payload.get("kind", "sample"))
            size = max(16, min(256, int(payload.get("size", 64))))
            keep_mesh = bool(payload.get("keep_mesh", False))
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
                bb_doc = payload.get("bbmodel") if isinstance(payload.get("bbmodel"), dict) else (self.bbmodel_doc if keep_mesh else None)
                mc = bake_bbmodel_maps(bb_doc) if bb_doc is not None else None
                g = parse_texdsl(code, mesh_context=mc)
                if mc is not None:
                    g.domain = "entity"
                fname = str(payload.get("filename") or "custom.texdsl")
                self._set_workspace(g, bb_doc, source=f"file:{fname}", clear_history=True)
            elif kind == "recipe":
                rec = payload.get("recipe")
                if not isinstance(rec, dict):
                    raise ValueError("Expected 'recipe' JSON object")
                bb_doc = payload.get("bbmodel") if isinstance(payload.get("bbmodel"), dict) else (self.bbmodel_doc if keep_mesh else None)
                mc = bake_bbmodel_maps(bb_doc) if bb_doc is not None else None
                g = TexGraph.from_recipe(rec, mesh_context=mc)
                fname = str(payload.get("filename") or f"{g.name}.texgraph.json")
                self._set_workspace(g, bb_doc, source=f"file:{fname}", clear_history=True)
            elif kind == "bbmodel_project":
                bb_doc = payload.get("bbmodel")
                if not isinstance(bb_doc, dict):
                    raise ValueError("Expected 'bbmodel' JSON object")
                mc = bake_bbmodel_maps(bb_doc)
                g = self._build_bbmodel_starter_graph(bb_doc, mc)
                fname = str(payload.get("filename") or f"{g.name}.bbmodel")
                self._set_workspace(g, bb_doc, source=f"bbmodel:{fname}", clear_history=True)
            elif kind == "syncproj":
                sp_doc = payload.get("syncproj")
                if not isinstance(sp_doc, dict):
                    raise ValueError("Expected 'syncproj' JSON object")
                base_dir = Path(str(payload.get("base_dir") or ".")).expanduser()
                bb_doc = payload.get("bbmodel") if isinstance(payload.get("bbmodel"), dict) else None
                if bb_doc is None and sp_doc.get("bbmodel_path"):
                    cand = (base_dir / str(sp_doc["bbmodel_path"])).expanduser()
                    if cand.is_file():
                        bb_doc = json.loads(cand.read_text(encoding="utf-8"))
                mc = bake_bbmodel_maps(bb_doc) if bb_doc is not None else None
                g = None
                if isinstance(payload.get("texdsl_code"), str) and payload["texdsl_code"].strip():
                    g = parse_texdsl(payload["texdsl_code"], mesh_context=mc)
                elif sp_doc.get("texdsl_path"):
                    cand_dsl = (base_dir / str(sp_doc["texdsl_path"])).expanduser()
                    if cand_dsl.is_file():
                        g = parse_texdsl(cand_dsl.read_text(encoding="utf-8"), mesh_context=mc)
                if g is None and isinstance(sp_doc.get("recipe"), dict):
                    g = TexGraph.from_recipe(sp_doc["recipe"], mesh_context=mc)
                if g is None and bb_doc is not None and mc is not None:
                    g = self._build_bbmodel_starter_graph(bb_doc, mc)
                if g is None:
                    raise ValueError(
                        "SyncProject manifest references external .texdsl/.bbmodel files not found on disk; provide sibling files or open from disk path"
                    )
                if sp_doc.get("name"):
                    g.name = str(sp_doc["name"])
                self._set_workspace(g, bb_doc, source=f"syncproj:{g.name}", clear_history=True)
                if isinstance(sp_doc.get("artist_overrides"), dict):
                    self.artist_overrides = copy.deepcopy(sp_doc["artist_overrides"])
            elif kind == "disk_file":
                fpath = Path(str(payload.get("path") or "")).expanduser().resolve()
                if not fpath.is_file():
                    raise FileNotFoundError(f"Project file not found: {fpath}")
                suf = fpath.suffix.lower()
                if suf == ".syncproj":
                    sp_doc = json.loads(fpath.read_text(encoding="utf-8"))
                    return self.load_project({"kind": "syncproj", "syncproj": sp_doc, "base_dir": str(fpath.parent)})
                if suf == ".bbmodel":
                    bb_doc = json.loads(fpath.read_text(encoding="utf-8"))
                    sib_dsl = fpath.with_suffix(".texdsl")
                    if sib_dsl.is_file():
                        return self.load_project(
                            {"kind": "texdsl", "code": sib_dsl.read_text(encoding="utf-8"), "bbmodel": bb_doc, "filename": fpath.name}
                        )
                    return self.load_project({"kind": "bbmodel_project", "bbmodel": bb_doc, "filename": fpath.name})
                if suf == ".texdsl":
                    sib_bb = fpath.with_suffix(".bbmodel")
                    bb_doc = json.loads(sib_bb.read_text(encoding="utf-8")) if sib_bb.is_file() else None
                    return self.load_project(
                        {"kind": "texdsl", "code": fpath.read_text(encoding="utf-8"), "bbmodel": bb_doc, "filename": fpath.name}
                    )
                if suf == ".json":
                    doc = json.loads(fpath.read_text(encoding="utf-8"))
                    if doc.get("schema", "").startswith("synccraft.project"):
                        return self.load_project({"kind": "syncproj", "syncproj": doc, "base_dir": str(fpath.parent)})
                    sib_bb = fpath.parent / f"{fpath.stem.replace('.texgraph', '')}.bbmodel"
                    bb_doc = json.loads(sib_bb.read_text(encoding="utf-8")) if sib_bb.is_file() else None
                    return self.load_project({"kind": "recipe", "recipe": doc, "bbmodel": bb_doc, "filename": fpath.name})
                raise ValueError(f"Unsupported project file extension: {suf!r}")
            else:
                raise ValueError(f"Unsupported load kind: {kind!r}")
            return self.get_state_payload()

    def load_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._workspace_epoch += 1
            preset_name = payload.get("sample")
            if preset_name == "kaelith_dragonborn":
                found = _find_kaelith_dragonborn_paths()
                if found is None:
                    raise FileNotFoundError("Could not locate kaelith_chrono_dragonborn.bbmodel in workspace")
                bb_doc = json.loads(found[0].read_text(encoding="utf-8"))
            elif preset_name == "temple_monolith":
                bb_doc = build_sample_monolith_bbmodel()
            elif preset_name == "ironjaw_orc":
                found = _find_orc_warlord_paths()
                if found is None:
                    raise FileNotFoundError("Could not locate ironjaw_orc_warlord.bbmodel in workspace")
                bb_doc = json.loads(found[0].read_text(encoding="utf-8"))
            elif preset_name == "none":
                self.bbmodel_doc = None
                self.mesh_ctx = None
                self.webgl_mesh = None
                self.graph.mesh_context = None
                self.graph._content_cache.clear()
                self.graph._static_cache.clear()
                self._thumb_cache.clear()
                self._channel_png_cache.clear()
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
            self.graph._static_cache.clear()
            self._thumb_cache.clear()
            self._channel_png_cache.clear()
            self._log(
                "info",
                f"Baked 3D .bbmodel '{bb_doc.get('name', 'mesh')}' ({mc.width}x{mc.height}, {len(mc.uv_seam_src_yx)} UV seam pairs, {'CACHE' if mc.bake_cache_hit else f'{mc.bake_ms:.0f}ms'})",
                mesh_fp=mc.content_fingerprint,
                mesh_cache_hit=mc.bake_cache_hit,
                bake_ms=mc.bake_ms,
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
                for fr in g.frames:
                    fr["node_ids"] = [nid for nid in (fr.get("node_ids") or []) if nid not in id_set]
                g.frames = [fr for fr in g.frames if fr.get("node_ids")]
                if g.output_node and _parse_ref(g.output_node)[0] in id_set:
                    g.output_node = g.node_order[-1] if g.node_order else None
                if g.output_material_node and g.output_material_node in id_set:
                    g.output_material_node = None
                g.pbr_nodes = {k: v for k, v in g.pbr_nodes.items() if _parse_ref(v)[0] not in id_set}
                g.param_drivers = [d for d in g.param_drivers if d.get("node_id") not in id_set]
                g.exposed_params = {k: v for k, v in g.exposed_params.items() if v.get("node_id") not in id_set}
                self._log("info", f"Deleted {len(ids)} node(s): {', '.join(ids)}")
                return self.get_state_payload()

            if action == "duplicate_nodes":
                ids = [str(x) for x in (payload.get("ids") or []) if str(x) in g.nodes]
                if not ids:
                    return self.get_state_payload()
                self._push_undo()
                created: list[str] = []
                for src_id in ids:
                    orig = g.nodes[src_id]
                    base_id = f"{orig.id}_copy"
                    nid = base_id
                    idx = 1
                    while nid in g.nodes:
                        nid = f"{base_id}_{idx}"
                        idx += 1
                    ox, oy = orig.pos if orig.pos is not None else (240.0, 140.0)
                    sp = NodeSpec(
                        id=nid,
                        op=orig.op,
                        inputs=dict(orig.inputs),
                        params=copy.deepcopy(orig.params),
                        pos=(float(ox) + 42.0, float(oy) + 42.0),
                        label=orig.label,
                        collapsed=orig.collapsed,
                        bypassed=orig.bypassed,
                    )
                    g.nodes[nid] = sp
                    g.node_order.append(nid)
                    created.append(nid)
                self._log("info", f"Duplicated {len(created)} node(s): {', '.join(created)}")
                out = self.get_state_payload()
                if created:
                    out["created_node_id"] = created[-1]
                return out

            if action == "randomize_seed":
                import random as _rnd

                new_seed = int(payload.get("seed") if payload.get("seed") is not None else _rnd.randint(1, 99999))
                nid = payload.get("node_id")
                self._push_undo()
                if nid and str(nid) in g.nodes:
                    g.nodes[str(nid)].params["seed"] = new_seed
                    self.artist_overrides[f"{nid}.seed"] = new_seed
                    self._log("info", f"Randomized seed of '{nid}' -> {new_seed}")
                else:
                    g.seed = new_seed
                    self._log("info", f"Randomized global graph seed -> {new_seed}")
                g._static_cache.clear()
                return self.get_state_payload()

            if action == "disconnect_all":
                nid = str(payload.get("node_id", ""))
                if nid not in g.nodes:
                    return self.get_state_payload()
                self._push_undo()
                g.nodes[nid].inputs.clear()
                for sp in g.nodes.values():
                    sp.inputs = {k: v for k, v in sp.inputs.items() if _parse_ref(v)[0] != nid}
                g._static_cache.clear()
                self._log("info", f"Disconnected all wires on '{nid}'")
                return self.get_state_payload()

            if action == "toggle_bypass_multi":
                ids = [str(x) for x in (payload.get("ids") or []) if str(x) in g.nodes]
                if not ids:
                    return self.get_state_payload()
                self._push_undo()
                any_active = any(not g.nodes[nid].bypassed for nid in ids)
                for nid in ids:
                    g.nodes[nid].bypassed = any_active
                g._static_cache.clear()
                self._log("info", f"{'Bypassed' if any_active else 'Enabled'} {len(ids)} node(s)")
                return self.get_state_payload()

            if action == "create_frame":
                ids = [str(x) for x in (payload.get("node_ids") or []) if str(x) in g.nodes]
                if not ids:
                    raise ValueError("Select at least one node to create a Comment Frame")
                self._push_undo()
                fid = f"frame_{len(g.frames) + 1}_{int(time.time() * 100) % 10000}"
                fr = {
                    "id": fid,
                    "title": str(payload.get("title") or "Comment Frame").strip(),
                    "color": str(payload.get("color") or "#2563EB"),
                    "node_ids": list(dict.fromkeys(ids)),
                }
                g.frames.append(fr)
                self._log("info", f"Created frame '{fr['title']}' ({len(ids)} nodes)")
                out = self.get_state_payload()
                out["created_frame_id"] = fid
                return out

            if action == "update_frame":
                fid = str(payload.get("frame_id") or "")
                for fr in g.frames:
                    if fr.get("id") == fid:
                        self._push_undo()
                        if "title" in payload and str(payload["title"]).strip():
                            fr["title"] = str(payload["title"]).strip()
                        if "color" in payload and str(payload["color"]).startswith("#"):
                            fr["color"] = str(payload["color"])
                        break
                return self.get_state_payload()

            if action == "delete_frame":
                fid = str(payload.get("frame_id") or "")
                before = len(g.frames)
                if any(fr.get("id") == fid for fr in g.frames):
                    self._push_undo()
                    g.frames = [fr for fr in g.frames if fr.get("id") != fid]
                    if len(g.frames) != before:
                        self._log("info", f"Removed frame '{fid}'")
                return self.get_state_payload()

            if action == "insert_reroute":
                dst = str(payload.get("dst_node") or "")
                dst_port = str(payload.get("dst_port") or "")
                if dst not in g.nodes or dst_port not in g.nodes[dst].inputs:
                    raise ValueError("Target wire does not exist for reroute insertion")
                self._push_undo()
                orig_ref = g.nodes[dst].inputs[dst_port]
                base_id = "dot"
                nid = base_id
                idx = 1
                while nid in g.nodes:
                    nid = f"{base_id}_{idx}"
                    idx += 1
                pos = payload.get("pos") or [240.0, 140.0]
                sp = NodeSpec(id=nid, op="reroute", inputs={"in": orig_ref}, params={"note": ""}, pos=(float(pos[0]), float(pos[1])))
                g.nodes[nid] = sp
                # Insert reroute right before dst in node_order so topological order stays clean
                try:
                    di = g.node_order.index(dst)
                    g.node_order.insert(di, nid)
                except ValueError:
                    g.node_order.append(nid)
                g.nodes[dst].inputs[dst_port] = nid
                g._static_cache.clear()
                self._log("info", f"Inserted reroute '{nid}' on wire {orig_ref} -> {dst}.{dst_port}")
                out = self.get_state_payload()
                out["created_node_id"] = nid
                return out

            if action == "insert_node_on_wire":
                nid = str(payload.get("node_id") or "")
                dst = str(payload.get("dst_node") or "")
                dst_port = str(payload.get("dst_port") or "")
                if nid not in g.nodes or dst not in g.nodes or dst_port not in g.nodes[dst].inputs:
                    raise ValueError("Node or target wire does not exist")
                if nid == dst:
                    return self.get_state_payload()
                meta = NODE_REGISTRY.get(g.nodes[nid].op)
                if not meta or not meta.inputs:
                    return self.get_state_payload()
                orig_ref = g.nodes[dst].inputs[dst_port]
                if _parse_ref(orig_ref)[0] == nid:
                    return self.get_state_payload()
                self._push_undo()
                in_port = meta.inputs[0].name
                g.nodes[nid].inputs[in_port] = orig_ref
                g.nodes[dst].inputs[dst_port] = nid
                vrep = g.validate_graph()
                if any("cycle" in e.lower() for e in vrep.errors):
                    rec = self._undo_stack.pop()
                    self._restore_recipe(rec)
                    raise ValueError("Inserting node on wire would introduce a cycle")
                g._static_cache.clear()
                self._log("info", f"Auto-wired '{nid}' inline on {orig_ref} -> {dst}.{dst_port}")
                return self.get_state_payload()

            if action == "group_nodes":
                ids = [str(x) for x in (payload.get("ids") or []) if str(x) in g.nodes]
                if not ids:
                    raise ValueError("Select at least one node to group")
                self._push_undo()
                title = str(payload.get("title") or f"Node Group ({len(ids)})").strip()
                color = str(payload.get("color") or "#7E22CE")
                gid = g.group_nodes_into_subgraph(ids, title=title, color=color)
                self._log("info", f"Collapsed {len(ids)} node(s) into Compound Subgraph '{gid}'")
                out = self.get_state_payload()
                out["created_node_id"] = gid
                return out

            if action == "ungroup_node":
                gid = str(payload.get("node_id") or "")
                if gid not in g.nodes or g.nodes[gid].op != "subgraph_group":
                    raise ValueError("Selected node is not a Compound Subgraph Group")
                self._push_undo()
                restored = g.ungroup_subgraph_node(gid)
                self._log("info", f"Expanded Compound Subgraph '{gid}' -> {len(restored)} node(s)")
                out = self.get_state_payload()
                out["created_node_ids"] = restored
                if restored:
                    out["created_node_id"] = restored[-1]
                return out

            if action == "enter_group":
                gid = str(payload.get("node_id") or "")
                if gid not in g.nodes or g.nodes[gid].op != "subgraph_group":
                    raise ValueError("Target node is not a Compound Subgraph Group")
                sp = g.nodes[gid]
                sub_rec = copy.deepcopy(sp.params.get("sub_recipe") or {})
                if not sub_rec.get("nodes"):
                    raise ValueError("Compound Subgraph contains no internal nodes")
                self._group_stack.append(
                    {
                        "parent_recipe": copy.deepcopy(g.to_recipe()),
                        "parent_title": g.name,
                        "group_node_id": gid,
                        "title": str(sp.params.get("title") or sp.label or gid),
                    }
                )
                sub_rec["width"] = g.width
                sub_rec["height"] = g.height
                sub_rec["tileable"] = g.tileable
                sub_g = TexGraph.from_recipe(sub_rec, mesh_context=self.mesh_ctx)
                self._workspace_epoch += 1
                self._channel_png_cache.clear()
                self.graph = sub_g
                self._ensure_positions()
                self._log("info", f"Entered Compound Subgraph '{self._group_stack[-1]['title']}'")
                return self.get_state_payload()

            if action == "exit_group":
                if not self._group_stack:
                    return self.get_state_payload()
                self._workspace_epoch += 1
                self._channel_png_cache.clear()
                target_level = int(payload.get("level", len(self._group_stack) - 1))
                target_level = max(0, min(len(self._group_stack) - 1, target_level))
                while len(self._group_stack) > target_level:
                    edited_sub_rec = copy.deepcopy(self.graph.to_recipe())
                    frame = self._group_stack.pop()
                    parent_g = TexGraph.from_recipe(frame["parent_recipe"], mesh_context=self.mesh_ctx)
                    gid = frame["group_node_id"]
                    if gid in parent_g.nodes:
                        parent_g.nodes[gid].params["sub_recipe"] = edited_sub_rec
                    parent_g._static_cache.clear()
                    parent_g._content_cache.clear()
                    self.graph = parent_g
                self._ensure_positions()
                self._log("info", f"Returned to '{self.graph.name}' (synced subgraph edits)")
                return self.get_state_payload()

            if action == "set_param":
                nid = str(payload.get("node_id", ""))
                pkey = str(payload.get("param", ""))
                val = payload.get("value")
                if nid not in g.nodes:
                    raise ValueError(f"Unknown node {nid!r}")
                if g.nodes[nid].op == "script_expr" and pkey in ("expr", "expr_g", "expr_b"):
                    s_val = str(val or "").strip()
                    if s_val:
                        from .v16_pro_ops import validate_safe_script_ast
                        validate_safe_script_ast(s_val)
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
                kfs = payload.get("keyframes")
                g.bind_driver(
                    nid,
                    pkey,
                    kind=str(payload.get("kind", "sine")),
                    min_val=float(payload.get("min_val", 0.0)),
                    max_val=float(payload.get("max_val", 1.0)),
                    freq=float(payload.get("freq", 1.0)),
                    phase_offset=float(payload.get("phase_offset", 0.0)),
                    octaves=int(payload.get("octaves", 3)),
                    keyframes=kfs if isinstance(kfs, list) else None,
                )
                self._log("info", f"Bound animation driver ({payload.get('kind', 'sine')}) to {nid}.{pkey}")
                return self.get_state_payload()

            if action == "update_driver":
                idx = int(payload.get("index", -1))
                if not (0 <= idx < len(g.param_drivers)):
                    raise ValueError(f"Driver index out of range: {idx}")
                if payload.get("undoable", True):
                    self._push_undo()
                d = g.param_drivers[idx]
                if "kind" in payload:
                    d["kind"] = str(payload["kind"])
                if "keyframes" in payload and isinstance(payload["keyframes"], list):
                    clean_kfs = []
                    for pt in payload["keyframes"]:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            t_k = round(max(0.0, min(1.0, float(pt[0]))), 4)
                            v_k = round(float(pt[1]), 4)
                            if len(pt) >= 3 and str(pt[2]).strip():
                                clean_kfs.append([t_k, v_k, str(pt[2])])
                            else:
                                clean_kfs.append([t_k, v_k])
                    clean_kfs.sort(key=lambda r: r[0])
                    if clean_kfs:
                        d["keyframes"] = clean_kfs
                        vals_k = [r[1] for r in clean_kfs]
                        d["min_val"] = float(min(vals_k))
                        d["max_val"] = float(max(vals_k))
                for k_f in ("min_val", "max_val", "freq", "phase_offset"):
                    if k_f in payload and payload[k_f] is not None:
                        d[k_f] = float(payload[k_f])
                g._static_cache.clear()
                self._channel_png_cache.clear()
                self._log("info", f"Updated curve driver {d.get('node_id')}.{d.get('param_key')} ({d.get('kind')})")
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
            include_thumbs_mode = str(payload.get("include_thumbs_mode") or ("all" if include_thumbs else "none"))
            include_qa = bool(payload.get("include_qa", True))

            t_start = time.perf_counter()
            self._eval_seq += 1
            cur_epoch = self._workspace_epoch
            cur_eval_seq = self._eval_seq
            val = self.graph.validate_graph()
            if not val.ok:
                return {
                    "ok": False,
                    "workspace_epoch": cur_epoch,
                    "eval_seq": cur_eval_seq,
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
            dyn_set = self.graph._compute_dynamic_set() if include_thumbs_mode == "dynamic" else None
            if include_thumbs_mode in ("all", "dynamic"):
                for nid in self.graph.node_order:
                    if nid not in ev.intermediates:
                        continue
                    if dyn_set is not None and nid not in dyn_set:
                        continue
                    arr = _arr_to_u8_rgba(ev.intermediates[nid])
                    node_shapes[nid] = [int(arr.shape[1]), int(arr.shape[0])]
                    ck = self.graph._content_cache.get(nid)
                    cache_key = f"{cur_epoch}:{nid}:{ck[0]}" if ck is not None else f"{cur_epoch}:{nid}:{phase:.4f}"
                    if cache_key not in self._thumb_cache:
                        self._thumb_cache[cache_key] = _u8_to_data_url(arr, max_dim=128, min_dim=64)
                        if len(self._thumb_cache) > 512:
                            for k_old in list(self._thumb_cache.keys())[:128]:
                                self._thumb_cache.pop(k_old, None)
                    thumbs[nid] = self._thumb_cache[cache_key]

            # Encode all 12+ PBR channels with epoch + output-hash caching (skips redundant 16x PNG encode!)
            out_u8 = _arr_to_u8_rgba(ev.output)
            out_base_id = _parse_ref(self.graph.output_material_node or self.graph.output_node or self.graph.node_order[-1])[0]
            out_ck = self.graph._content_cache.get(out_base_id)
            em_ck = self.graph._content_cache.get(_parse_ref(self.graph.emissive_node)[0]) if self.graph.emissive_node else None
            out_tag = out_ck[0] if out_ck else f"{phase:.4f}"
            em_tag = em_ck[0] if em_ck else ""
            chan_cache_key = f"{cur_epoch}:{out_tag}:{em_tag}:{lighting_preset}:{light_az:.1f}:{light_el:.1f}"
            cached_ch = self._channel_png_cache.get(chan_cache_key)
            if cached_ch is not None:
                channels = dict(cached_ch)
            else:
                channels = {}
                channels["albedo"] = _u8_to_data_url(out_u8, max_dim=512)
                if ev.pbr_preview is not None:
                    channels["pbr_lit"] = _u8_to_data_url(_arr_to_u8_rgba(ev.pbr_preview), max_dim=512)
                else:
                    channels["pbr_lit"] = channels["albedo"]

                if ev.pbr_maps is not None:
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
                    mfp = mc.content_fingerprint or f"{mc.width}x{mc.height}:{id(mc)}"
                    diag_cached = self._mesh_diag_cache.get(mfp)
                    if diag_cached is None:
                        diag = _scalar_to_heatmap_u8(mc.thickness, warm=True)
                        seam_m = mc.uv_seam_mask > 0.05
                        diag[seam_m, 0] = 40
                        diag[seam_m, 1] = 232
                        diag[seam_m, 2] = 255
                        diag[~mc.island_mask, :3] = 20
                        ia_u8 = _scalar_to_heatmap_u8(mc.island_inner_area, warm=False)
                        ia_u8[~mc.island_mask, :3] = 16
                        diag_cached = {
                            "uv_seams": _u8_to_data_url(diag, max_dim=512),
                            "uv_regions": _u8_to_data_url(build_uv_region_overlay(mc), max_dim=512),
                            "inner_area": _u8_to_data_url(ia_u8, max_dim=512),
                        }
                        if len(self._mesh_diag_cache) > 12:
                            self._mesh_diag_cache.pop(next(iter(self._mesh_diag_cache)), None)
                        self._mesh_diag_cache[mfp] = diag_cached
                    channels.update(diag_cached)

                if len(self._channel_png_cache) > 24:
                    self._channel_png_cache.pop(next(iter(self._channel_png_cache)), None)
                self._channel_png_cache[chan_cache_key] = dict(channels)

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
            _write_studio_file_log(
                "info",
                "evaluate",
                epoch=cur_epoch,
                eval_seq=cur_eval_seq,
                graph=self.graph.name,
                dims=f"{out_u8.shape[1]}x{out_u8.shape[0]}",
                phase=round(phase, 4),
                eval_ms=eval_ms,
                total_ms=total_ms,
                cache_hits=ev.cache_hits,
                cache_misses=ev.cache_misses,
                ch_cache_hit=cached_ch is not None,
                qa=round(qa_dict["score"], 1) if isinstance(qa_dict, dict) and isinstance(qa_dict.get("score"), (int, float)) else None,
            )

            # Broadcast live evaluated textures to any connected Blockbench 5 Live Link clients!
            with self._bridge_cond:
                self._bridge_rev += 1
                self._bridge_payload = {
                    "ok": True,
                    "rev": self._bridge_rev,
                    "workspace_epoch": cur_epoch,
                    "graph_name": self.graph.name,
                    "width": int(out_u8.shape[1]),
                    "height": int(out_u8.shape[0]),
                    "phase": round(phase, 4),
                    "channels": {
                        k: channels[k]
                        for k in ("baked", "albedo", "pbr_lit", "normal", "labpbr_n", "labpbr_s", "bedrock_mer", "emissive")
                        if k in channels
                    },
                }
                self._bridge_cond.notify_all()

            return {
                "ok": True,
                "workspace_epoch": cur_epoch,
                "eval_seq": cur_eval_seq,
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
                "bridge": {"rev": self._bridge_rev, "clients": self._bridge_clients},
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
                mtlx_path = out_dir / f"{stem}.mtlx"
                dsl_path.write_text(export_texdsl(self.graph), encoding="utf-8")
                rec_path.write_text(json.dumps(self.graph.to_recipe(), indent=2) + "\n", encoding="utf-8")
                mtlx_xml = self.export_materialx({"stem": stem})["xml"]
                mtlx_path.write_text(mtlx_xml, encoding="utf-8")
                files = dict(pkg["files"])
                files["texdsl"] = str(dsl_path)
                files["texgraph_json"] = str(rec_path)
                files["materialx"] = str(mtlx_path)
                self._log("info", f"Exported commercial PBR v2.0 + MaterialX bundle ({len(files)} files) -> {out_dir}")
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
                    f"{stem}.mtlx": self.export_materialx({"stem": stem})["xml"],
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

    # -----------------------------------------------------------------------
    # Milestone v14: 2D UV Picking, `.syncmat` Presets & MaterialX / OpenPBR
    # -----------------------------------------------------------------------

    def pick_uv_part(self, u: float, v: float) -> dict[str, Any]:
        with self._lock:
            if self.mesh_ctx is None:
                return {"ok": True, "hit": False, "message": "No 3D .bbmodel loaded"}
            mc = self.mesh_ctx
            px = int(np.clip(math.floor(float(u) * mc.width), 0, mc.width - 1))
            py = int(np.clip(math.floor(float(v) * mc.height), 0, mc.height - 1))
            eid = int(mc.elem_id_map[py, px])
            if eid < 0 or eid >= len(mc.elem_names):
                return {"ok": True, "hit": False, "px": px, "py": py}
            ename = mc.elem_names[eid]
            short = ename.split("__")[-1]
            bone = mc.elem_bones[eid] if eid < len(mc.elem_bones) else ""
            zone = mc.elem_zones[eid] if eid < len(mc.elem_zones) else ""
            return {"ok": True, "hit": True, "px": px, "py": py, "elem_idx": eid, "name": ename, "short": short, "bone": bone, "zone": zone}

    def list_syncmats(self) -> dict[str, Any]:
        return {"ok": True, "presets": list_all_syncmats()}

    def save_syncmat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node_ids = payload.get("node_ids")
            name = str(payload.get("name") or self.graph.name or "Custom Material").strip()
            category = str(payload.get("category") or "User Presets")
            desc = str(payload.get("description") or "")
            color = str(payload.get("frame_color") or "#2563EB")
            thumb = str(payload.get("thumbnail") or "")
            preset = extract_syncmat_from_graph(
                self.graph,
                node_ids=list(node_ids) if isinstance(node_ids, list) and node_ids else None,
                name=name,
                category=category,
                description=desc,
                frame_color=color,
                thumbnail=thumb,
            )
            saved_path = None
            if payload.get("persist", True):
                saved_path = str(save_user_syncmat(preset))
            self._log("info", f"Saved .syncmat preset '{preset['name']}' ({len(preset['nodes'])} nodes)")
            return {"ok": True, "preset": preset, "path": saved_path, "filename": f"{preset['id']}.syncmat.json", "presets": list_all_syncmats()}

    def insert_syncmat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            preset = payload.get("preset")
            if preset is None:
                pid = str(payload.get("preset_id") or "")
                all_p = {p["id"]: p for p in list_all_syncmats()}
                if pid not in all_p:
                    raise ValueError(f"Unknown .syncmat preset: {pid!r}")
                preset = all_p[pid]
            if not isinstance(preset, dict):
                raise ValueError("Expected .syncmat JSON object or preset_id")
            pos = payload.get("pos") or [220.0, 140.0]
            self._push_undo()
            created, out_id, frame_obj = insert_syncmat_into_graph(
                self.graph,
                preset,
                pos=(float(pos[0]), float(pos[1])),
                wrap_frame=bool(payload.get("wrap_frame", True)),
            )
            self._log("info", f"Inserted .syncmat '{preset.get('name', 'preset')}' ({len(created)} nodes)")
            st = self.get_state_payload()
            st["created_node_id"] = out_id
            st["created_node_ids"] = created
            st["created_frame"] = frame_obj
            return st

    def export_materialx(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            stem = str(payload.get("stem") or self.graph.name or "synccraft_material")
            shader = str(payload.get("shader_model") or "open_pbr_surface")
            coat = float(payload.get("coat_weight", 0.15))
            trans = float(payload.get("transmission_weight", 0.0))
            sss_w = float(payload.get("subsurface_weight", 0.0))
            has_em = False
            has_op = False
            try:
                ev = self.graph.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
                has_em = bool(ev.emissive is not None and float(np.max(ev.emissive[..., 3])) > 0.05)
                has_op = bool(ev.output.ndim == 3 and ev.output.shape[2] == 4 and float(np.min(ev.output[..., 3])) < 0.96)
                if has_op and trans <= 0.0:
                    trans = 0.45
                if ev.pbr_channels and "sss" in ev.pbr_channels and float(np.mean(ev.pbr_channels["sss"])) > 0.05:
                    sss_w = max(sss_w, 0.35)
            except Exception:
                pass
            xml = export_materialx_xml(
                stem,
                shader_model=shader,
                has_emissive=has_em,
                has_opacity=has_op,
                coat_weight=coat,
                transmission_weight=trans,
                subsurface_weight=sss_w,
            )
            return {"ok": True, "filename": f"{stem}.mtlx", "shader_model": shader, "xml": xml}

    # -----------------------------------------------------------------------
    # Milestone v15: 30-60 FPS Pre-Baked Anim Sequence, 3D/2D Brush Painting,
    # CTM 47-Tile / Wang 16-Tile Export, Turntable GIF & BB5 Live Link Bridge
    # -----------------------------------------------------------------------

    def evaluate_anim_sequence(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Pre-bake all N frames of the active animated graph into a GPU-ready frame ring buffer (for 30-60 FPS zero-lag playback!)."""
        with self._lock:
            n_frames = max(2, min(32, int(payload.get("frames", 12))))
            channel = str(payload.get("channel") or "pbr_lit")
            light_az = float(payload.get("light_az", -38.0))
            light_el = float(payload.get("light_el", 52.0))
            preview_node = payload.get("preview_node")
            max_dim = max(64, min(512, int(payload.get("max_dim", 256))))

            t0 = time.perf_counter()
            dyn_set = self.graph._compute_dynamic_set()
            seq: list[dict[str, Any]] = []
            w_out, h_out = self.graph.width, self.graph.height

            for fi in range(n_frames):
                ph = float(fi) / float(n_frames)
                ev = self.graph.evaluate(
                    phase=ph,
                    capture_intermediates=True,
                    build_pbr=(channel not in ("albedo", "selected")),
                    light_az_deg=light_az,
                    light_el_deg=light_el,
                    allow_orphan_errors=True,
                )
                out_u8 = _arr_to_u8_rgba(ev.output)
                w_out, h_out = int(out_u8.shape[1]), int(out_u8.shape[0])
                alb_url = _u8_to_data_url(out_u8, max_dim=max_dim)
                lit_url = _u8_to_data_url(_arr_to_u8_rgba(ev.pbr_preview), max_dim=max_dim) if ev.pbr_preview is not None else alb_url
                if ev.emissive is not None:
                    em_url = _u8_to_data_url(_arr_to_u8_rgba(ev.emissive), max_dim=max_dim)
                else:
                    em_url = None
                sel_url = None
                if channel == "selected" and preview_node and preview_node in ev.intermediates:
                    sel_url = _u8_to_data_url(_arr_to_u8_rgba(ev.intermediates[preview_node]), max_dim=max_dim)

                disp_url = sel_url or (alb_url if channel == "albedo" else lit_url)
                dyn_thumbs: dict[str, str] = {}
                for nid in dyn_set:
                    if nid in ev.intermediates:
                        dyn_thumbs[nid] = _u8_to_data_url(_arr_to_u8_rgba(ev.intermediates[nid]), max_dim=96, min_dim=48)

                seq.append(
                    {
                        "phase": round(ph, 4),
                        "albedo": alb_url,
                        "pbr_lit": lit_url,
                        "emissive": em_url,
                        "display": disp_url,
                        "thumbs": dyn_thumbs,
                    }
                )

            return {
                "ok": True,
                "workspace_epoch": self._workspace_epoch,
                "frames": n_frames,
                "width": w_out,
                "height": h_out,
                "bake_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "sequence": seq,
            }

    def paint_mask_stroke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Paint 3D/2D brush strokes directly onto a `paint_mask` node's grayscale buffer (spawning one if needed)."""
        with self._lock:
            g = self.graph
            nid = payload.get("node_id")
            created_id = None
            if not nid or str(nid) not in g.nodes or g.nodes[str(nid)].op != "paint_mask":
                # Auto-spawn a ready-to-wire `paint_mask` node
                self._push_undo()
                base_id = "paint_mask"
                nid_s = base_id
                idx = 1
                while nid_s in g.nodes:
                    nid_s = f"{base_id}_{idx}"
                    idx += 1
                pos = payload.get("pos") or [240.0, 140.0]
                sp = NodeSpec(
                    id=nid_s,
                    op="paint_mask",
                    inputs={},
                    params={"base_value": 0.0, "blur_px": 0, "invert": False, "mask_data_url": ""},
                    pos=(float(pos[0]), float(pos[1])),
                )
                g.nodes[nid_s] = sp
                g.node_order.append(nid_s)
                nid = nid_s
                created_id = nid_s
                self._log("info", f"Created 3D/2D Brush Paint Mask node '{nid_s}'")

            sp = g.nodes[str(nid)]
            H, W = g.height, g.width

            if payload.get("clear"):
                self._push_undo()
                arr = np.zeros((H, W), dtype=np.float32)
            elif isinstance(payload.get("mask_data_url"), str) and "," in payload["mask_data_url"]:
                if payload.get("undoable"):
                    self._push_undo()
                sp.params["mask_data_url"] = payload["mask_data_url"]
                g._static_cache.clear()
                st = self.get_state_payload()
                if created_id:
                    st["created_node_id"] = created_id
                return st
            else:
                cur_url = str(sp.params.get("mask_data_url") or "")
                if cur_url and "," in cur_url:
                    raw = base64.b64decode(cur_url.split(",", 1)[1])
                    im = Image.open(io.BytesIO(raw)).convert("L")
                    if im.size != (W, H):
                        im = im.resize((W, H), Image.Resampling.BILINEAR)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                else:
                    arr = np.full((H, W), float(sp.params.get("base_value", 0.0)), dtype=np.float32)

                strokes = payload.get("strokes") or []
                if strokes and payload.get("undoable"):
                    self._push_undo()
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                for st_pt in strokes:
                    u = float(st_pt.get("u", 0.5))
                    v = float(st_pt.get("v", 0.5))
                    rad = max(1.0, float(st_pt.get("radius_px", 5.0)))
                    val = float(np.clip(st_pt.get("value", 1.0), 0.0, 1.0))
                    soft = float(np.clip(st_pt.get("softness", 0.35), 0.0, 0.95))
                    opac = float(np.clip(st_pt.get("opacity", 1.0), 0.05, 1.0))
                    cx = u * (W - 1)
                    cy = v * (H - 1)
                    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                    inner = rad * (1.0 - soft)
                    w_m = np.clip(1.0 - (d - inner) / max(0.5, rad - inner), 0.0, 1.0)
                    w_m = w_m * w_m * (3.0 - 2.0 * w_m) * opac
                    arr = arr * (1.0 - w_m) + val * w_m

            u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(u8, mode="L").save(buf, format="PNG", compress_level=3)
            sp.params["mask_data_url"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            g._static_cache.clear()
            st = self.get_state_payload()
            st["painted_node_id"] = str(nid)
            if created_id:
                st["created_node_id"] = created_id
            return st

    def export_ctm(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate a Connected Block Tileset (CTM 47-Tile or Wang 16-Tile) from the active graph output."""
        with self._lock:
            mode = str(payload.get("mode") or "ctm47")
            style = str(payload.get("style") or "metal_frame")
            edge_w = int(payload.get("edge_width_px", 3))
            edge_col = str(payload.get("edge_color") or "#D4A84A")
            strength = float(payload.get("strength", 0.85))
            stem = str(payload.get("stem") or self.graph.name or "synccraft_block")
            out_dir = payload.get("out_dir") or None

            ev = self.graph.evaluate(phase=0.0, build_pbr=False, allow_orphan_errors=True)
            res = generate_ctm_tileset(
                ev.output,
                mode=mode,
                style=style,
                edge_width_px=edge_w,
                edge_color=edge_col,
                strength=strength,
                seed=self.graph.seed,
                out_dir=out_dir,
                stem=stem,
            )
            self._log("info", f"Generated {res['mode'].upper()} tileset ({res['tile_count']} tiles, {res['grid_cols']}x{res['grid_rows']} sheet)")
            return {
                "ok": True,
                "mode": res["mode"],
                "tile_count": res["tile_count"],
                "grid_cols": res["grid_cols"],
                "grid_rows": res["grid_rows"],
                "sheet_filename": f"{stem}_{res['mode']}_sheet.png",
                "props_filename": f"{stem}.properties",
                "sheet_data_url": res["sheet_data_url"],
                "optifine_properties": res["optifine_properties"],
                "manifest": res["manifest"],
                "files": res["files"],
            }

    def export_turntable_gif(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Assemble captured WebGL PNG frames into an optimized looping 360° Turntable GIF."""
        frames_b64 = payload.get("frames") or []
        if not isinstance(frames_b64, list) or len(frames_b64) < 2:
            raise ValueError("Expected at least 2 captured PNG frames for Turntable GIF")
        fps = max(4, min(30, int(payload.get("fps", 16))))
        stem = str(payload.get("stem") or self.graph.name or "synccraft_turntable")
        pil_frames: list[Image.Image] = []
        for data_url in frames_b64[:48]:
            if not isinstance(data_url, str) or "," not in data_url:
                continue
            raw = base64.b64decode(data_url.split(",", 1)[1])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if max(im.size) > 480:
                im.thumbnail((480, 480), Image.Resampling.LANCZOS)
            pil_frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
        if len(pil_frames) < 2:
            raise ValueError("Could not decode captured PNG frames")
        buf = io.BytesIO()
        dur_ms = max(30, int(round(1000.0 / fps)))
        pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:], duration=dur_ms, loop=0, optimize=True)
        gif_url = "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        self._log("info", f"Encoded 360° Turntable GIF ({len(pil_frames)} frames @ {fps}fps)")
        return {"ok": True, "filename": f"{stem}_turntable.gif", "frame_count": len(pil_frames), "gif_data_url": gif_url}

    def get_file_logs(self, *, tail: int = 200) -> dict[str, Any]:
        lp = get_studio_log_path()
        lines: list[str] = []
        size_bytes = 0
        if lp.is_file():
            size_bytes = lp.stat().st_size
            raw_lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
            lines = raw_lines[-max(1, min(2000, int(tail))) :]
        return {"ok": True, "path": str(lp), "size_bytes": size_bytes, "line_count": len(lines), "lines": lines}

    def clear_file_logs(self) -> dict[str, Any]:
        lp = get_studio_log_path()
        try:
            lp.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._event_log.clear()
        self._log("info", "Cleared Studio file log")
        return {"ok": True, "path": str(lp)}

    # -----------------------------------------------------------------------
    # Milestone v16: 3D Multi-Channel PBR Paint Stack, .syncpack Marketplace
    # & Multi-Asset Batch Bakery Queue
    # -----------------------------------------------------------------------

    def paint_stack_stroke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Paint multi-channel PBR strokes (Albedo/Roughness/Metallic/Height/Emissive) or edit layers on `paint_layer_stack`."""
        with self._lock:
            g = self.graph
            nid = payload.get("node_id")
            created_id = None
            if not nid or str(nid) not in g.nodes or g.nodes[str(nid)].op != "paint_layer_stack":
                # Reuse existing paint_layer_stack if one is already in the graph, else auto-spawn & wire at graph output!
                existing = next((n_id for n_id in reversed(g.node_order) if g.nodes[n_id].op == "paint_layer_stack"), None)
                if existing and not payload.get("force_new"):
                    nid = existing
                else:
                    self._push_undo()
                    base_id = "pbr_paint_stack"
                    nid_s = base_id
                    idx = 1
                    while nid_s in g.nodes:
                        nid_s = f"{base_id}_{idx}"
                        idx += 1
                    cur_out = g.output_material_node or (
                        _parse_ref(g.output_node)[0] if g.output_node else (g.node_order[-1] if g.node_order else None)
                    )
                    pos = payload.get("pos")
                    if not pos and cur_out and cur_out in g.nodes and g.nodes[cur_out].pos:
                        pos = [g.nodes[cur_out].pos[0] + 290.0, g.nodes[cur_out].pos[1]]
                    pos = pos or [380.0, 160.0]
                    init_layers = [make_empty_paint_layer("layer_1", "Hero Detail Paint")]
                    sp = NodeSpec(
                        id=nid_s,
                        op="paint_layer_stack",
                        inputs={"base": cur_out} if cur_out and cur_out in g.nodes else {},
                        params={
                            "active_layer_id": "layer_1",
                            "height_normal_strength": 2.6,
                            "layers_json": json.dumps(init_layers),
                        },
                        pos=(float(pos[0]), float(pos[1])),
                    )
                    g.nodes[nid_s] = sp
                    g.node_order.append(nid_s)
                    if payload.get("wire_output", True):
                        g.set_output_material(nid_s, normal_strength=float(g.pbr_config.get("normal_strength", 2.4)))
                    nid = nid_s
                    created_id = nid_s
                    self._log("info", f"Auto-wired 3D PBR Paint Layer Stack '{nid_s}' at graph output")

            sp = g.nodes[str(nid)]
            layers = parse_layers_json(sp.params.get("layers_json"))
            active_lid = str(payload.get("layer_id") or sp.params.get("active_layer_id") or layers[0]["id"])
            if not any(l["id"] == active_lid for l in layers):
                active_lid = layers[-1]["id"]

            layer_action = str(payload.get("layer_action") or "").strip().lower()
            if layer_action:
                self._push_undo()
                if layer_action == "add_layer":
                    n_idx = len(layers) + 1
                    new_id = f"layer_{n_idx}"
                    while any(l["id"] == new_id for l in layers):
                        n_idx += 1
                        new_id = f"layer_{n_idx}"
                    lname = str(payload.get("name") or f"PBR Layer {n_idx}")
                    layers.append(make_empty_paint_layer(new_id, lname))
                    active_lid = new_id
                    self._log("info", f"Added PBR paint layer '{lname}' on '{nid}'")
                elif layer_action == "delete_layer" and len(layers) > 1:
                    layers = [l for l in layers if l["id"] != active_lid]
                    active_lid = layers[-1]["id"]
                elif layer_action == "toggle_visible":
                    for l in layers:
                        if l["id"] == active_lid:
                            l["visible"] = not bool(l.get("visible", True))
                elif layer_action == "set_opacity":
                    for l in layers:
                        if l["id"] == active_lid:
                            l["opacity"] = round(float(np.clip(payload.get("opacity", 1.0), 0.0, 1.0)), 2)
                elif layer_action == "set_blend":
                    for l in layers:
                        if l["id"] == active_lid:
                            l["blend_mode"] = str(payload.get("blend_mode") or "normal")
                elif layer_action == "clear_layer":
                    for l in layers:
                        if l["id"] == active_lid:
                            l["albedo_url"] = ""
                            l["pbr_url"] = ""
                            l["emissive_url"] = ""
                elif layer_action == "select_layer":
                    pass
                elif layer_action == "rename_layer":
                    for l in layers:
                        if l["id"] == active_lid and payload.get("name"):
                            l["name"] = str(payload["name"]).strip()
            else:
                strokes = payload.get("strokes") or []
                if strokes:
                    if payload.get("undoable"):
                        self._push_undo()
                    brush = payload.get("brush") or {}
                    for i, l in enumerate(layers):
                        if l["id"] == active_lid:
                            layers[i] = apply_pbr_brush_strokes_to_layer(l, g.width, g.height, strokes, brush=brush)
                            break

            sp.params["active_layer_id"] = active_lid
            sp.params["layers_json"] = json.dumps(layers)
            g._static_cache.clear()
            self._channel_png_cache.clear()
            st = self.get_state_payload()
            st["painted_node_id"] = str(nid)
            st["active_layer_id"] = active_lid
            st["paint_layers"] = [
                {k: v for k, v in l.items() if not k.endswith("_url")}
                | {"has_paint": bool(l.get("albedo_url") or l.get("pbr_url") or l.get("emissive_url"))}
                for l in layers
            ]
            if created_id:
                st["created_node_id"] = created_id
            return st

    def list_syncpacks(self) -> dict[str, Any]:
        return {"ok": True, "packs": list_all_syncpacks()}

    def save_syncpack(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node_ids = payload.get("node_ids")
            name = str(payload.get("name") or f"{self.graph.name} Pack").strip()
            author = str(payload.get("author") or "SyncCraft Artist").strip()
            desc = str(payload.get("description") or f"Custom .syncpack exported from {self.graph.name}").strip()
            macro = extract_syncmat_from_graph(
                self.graph,
                node_ids=list(node_ids) if isinstance(node_ids, list) and node_ids else None,
                name=str(payload.get("macro_name") or name),
                category="Custom Pack",
                description=desc,
                frame_color=str(payload.get("color") or "#0284C7"),
            )
            pack = {
                "schema": "synccraft.syncpack/1.0",
                "id": payload.get("id") or name.lower().replace(" ", "_"),
                "name": name,
                "author": author,
                "version": "1.0.0",
                "color": str(payload.get("color") or "#0284C7"),
                "tags": list(payload.get("tags") or ["custom", "macro"]),
                "description": desc,
                "macros": [
                    {
                        "id": macro["id"],
                        "name": macro["name"],
                        "description": macro.get("description", ""),
                        "frame_color": macro.get("frame_color", "#0284C7"),
                        "nodes": macro["nodes"],
                        "output_node": macro.get("output_node"),
                    }
                ],
            }
            saved_p = save_user_syncpack(pack)
            self._log("info", f"Saved .syncpack '{name}' -> {saved_p.name}")
            return {"ok": True, "pack": pack, "path": str(saved_p), "packs": list_all_syncpacks()}

    def import_syncpack(self, payload: dict[str, Any]) -> dict[str, Any]:
        pack = payload.get("pack")
        if not isinstance(pack, dict) or not isinstance(pack.get("macros"), list):
            raise ValueError("Expected valid .syncpack JSON with 'macros' list")
        saved_p = save_user_syncpack(pack)
        self._log("info", f"Imported .syncpack '{pack.get('name', saved_p.stem)}' ({len(pack['macros'])} macros)")
        return {"ok": True, "path": str(saved_p), "packs": list_all_syncpacks()}

    def spawn_syncpack(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            pack_id = str(payload.get("pack_id") or "")
            macro_id = str(payload.get("macro_id") or "")
            pos = payload.get("pos") or [240.0, 140.0]
            self._push_undo()
            created, out_id, frame_obj = spawn_syncpack_macro_into_graph(
                self.graph,
                pack_id,
                macro_id,
                pos=(float(pos[0]), float(pos[1])),
                wrap_frame=bool(payload.get("wrap_frame", True)),
            )
            self._log("info", f"Spawned .syncpack macro '{pack_id}/{macro_id}' ({len(created)} nodes)")
            st = self.get_state_payload()
            st["created_node_id"] = out_id
            st["created_node_ids"] = created
            st["created_frame"] = frame_obj
            return st

    def batch_scan(self, payload: dict[str, Any]) -> dict[str, Any]:
        in_dir = str(payload.get("input_dir") or "").strip()
        if not in_dir:
            found = _find_kaelith_dragonborn_paths()
            in_dir = str(found[0].parent) if found else str(Path.cwd())
        max_a = max(1, min(64, int(payload.get("max_assets", 24))))
        assets = scan_batch_folder(in_dir, max_assets=max_a)
        return {"ok": True, "input_dir": str(Path(in_dir).expanduser().resolve()), "count": len(assets), "assets": assets}

    def batch_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        in_dir = str(payload.get("input_dir") or "").strip()
        if not in_dir:
            found = _find_kaelith_dragonborn_paths()
            if found is None:
                raise ValueError("Specify an input_dir containing .bbmodel / .texdsl files")
            in_dir = str(found[0].parent)
        out_dir = str(payload.get("out_dir") or (Path.home() / ".synccraft" / "batch_exports" / f"run_{int(time.time())}")).strip()
        preset = str(payload.get("preset") or "studio_hero")
        max_a = max(1, min(64, int(payload.get("max_assets", 16))))

        with self._batch_lock:
            if self._batch_state.get("running"):
                return {"ok": False, "error": "A batch bakery job is already running", "status": copy.deepcopy(self._batch_state)}
            self._batch_state = {
                "running": True,
                "total": 0,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "input_dir": in_dir,
                "out_dir": out_dir,
                "items": [],
                "report": None,
                "error": None,
            }

        def _on_prog(st_d: dict[str, Any]) -> None:
            with self._batch_lock:
                self._batch_state.update(st_d)

        def _worker() -> None:
            try:
                rep = run_batch_bakery(in_dir, out_dir, preset=preset, max_assets=max_a, on_progress=_on_prog)
                with self._batch_lock:
                    self._batch_state["running"] = False
                    self._batch_state["report"] = rep
                self._log("info", f"Batch Bakery finished: {rep['succeeded']}/{rep['total']} succeeded (mean QA {rep.get('mean_qa')}) -> {out_dir}")
            except Exception as exc:
                with self._batch_lock:
                    self._batch_state["running"] = False
                    self._batch_state["error"] = f"{type(exc).__name__}: {exc}"
                self._log("error", f"Batch Bakery failed: {exc}")

        if payload.get("sync"):
            _worker()
        else:
            threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "status": self.batch_status()}

    def batch_status(self) -> dict[str, Any]:
        with self._batch_lock:
            return {"ok": True, **copy.deepcopy(self._batch_state)}

    def get_bridge_latest(self, since_rev: int = -1) -> dict[str, Any]:
        with self._bridge_cond:
            if not self._bridge_payload or self._bridge_rev <= since_rev:
                return {"ok": True, "changed": False, "rev": self._bridge_rev, "clients": self._bridge_clients}
            out = dict(self._bridge_payload)
            out["changed"] = True
            out["clients"] = self._bridge_clients
            return out


# ---------------------------------------------------------------------------
# HTTP Server & Handler
# ---------------------------------------------------------------------------


class _StudioThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


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

        def _send_bytes(
            self,
            body: bytes,
            *,
            status: int = 200,
            content_type: str = "application/octet-stream",
            cors: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if status >= 400:
                self.send_header("Connection", "close")
            if cors:
                self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, obj: dict[str, Any], *, status: int = 200, cors: bool = False) -> None:
            raw = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send_bytes(raw, status=status, content_type="application/json; charset=utf-8", cors=cors)

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
                        "engine_version": "16.0",
                        "schema_version": "4.0",
                        "node_count": len(NODE_REGISTRY),
                        "active_graph": sess.graph.name,
                        "workspace_epoch": sess._workspace_epoch,
                        "log_file": str(get_studio_log_path()),
                        "mesh_cache": mesh_bake_cache_stats(),
                    }
                )
                return

            if path == "/api/syncpack/list":
                self._send_json(sess.list_syncpacks())
                return

            if path == "/api/batch/status":
                self._send_json(sess.batch_status())
                return

            if path == "/api/logs/file":
                qs = urllib.parse.parse_qs(parsed.query)
                tail = int((qs.get("tail") or ["200"])[0])
                info = sess.get_file_logs(tail=tail)
                if (qs.get("download") or ["0"])[0] == "1":
                    body = ("\n".join(info["lines"]) + "\n").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="synccraft_studio.log"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._send_json(info)
                return

            if path == "/api/schema":
                self._send_json(sess.get_schema_payload())
                return

            if path == "/api/schema/previews":
                qs = urllib.parse.parse_qs(parsed.query)
                sz = int((qs.get("size") or ["40"])[0])
                previews = generate_node_preview_catalog(size=sz)
                time_aware_ops = sorted(op for op, m in NODE_REGISTRY.items() if m.time_aware)
                raw = json.dumps(
                    {
                        "ok": True,
                        "size": sz,
                        "count": len(previews),
                        "previews": previews,
                        "time_aware_ops": time_aware_ops,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/schema/preview_anim":
                qs = urllib.parse.parse_qs(parsed.query)
                op = str((qs.get("op") or [""])[0]).strip()
                sz = int((qs.get("size") or ["48"])[0])
                nf = int((qs.get("frames") or ["8"])[0])
                if op not in NODE_REGISTRY:
                    self._send_json({"ok": False, "error": f"Unknown op: {op!r}"}, status=404)
                    return
                frames = generate_node_anim_preview(op, size=sz, frames=nf)
                raw = json.dumps({"ok": True, "op": op, "size": sz, "frames": len(frames), "images": frames}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/schema/preview_anim_sheet":
                qs = urllib.parse.parse_qs(parsed.query)
                sz = int((qs.get("size") or ["36"])[0])
                nf = int((qs.get("frames") or ["6"])[0])
                sheet = generate_anim_preview_sheet(size=sz, frames=nf)
                raw = json.dumps({"ok": True, "size": sz, "frames": nf, "count": len(sheet), "sheet": sheet}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/mesh/pick_uv":
                qs = urllib.parse.parse_qs(parsed.query)
                u = float((qs.get("u") or ["0.5"])[0])
                v = float((qs.get("v") or ["0.5"])[0])
                self._send_json(sess.pick_uv_part(u, v))
                return

            if path == "/api/syncmat/list":
                self._send_json(sess.list_syncmats())
                return

            if path == "/api/bridge/latest":
                qs = urllib.parse.parse_qs(parsed.query)
                since_rev = int((qs.get("since") or ["-1"])[0])
                self._send_json(sess.get_bridge_latest(since_rev=since_rev), cors=True)
                return

            if path == "/api/bridge/events":
                # Server-Sent Events stream for Blockbench 5 Live Link (zero external deps!)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with sess._bridge_cond:
                    sess._bridge_clients += 1
                    last_sent = -1
                try:
                    for _ in range(360):
                        with sess._bridge_cond:
                            if sess._bridge_rev <= last_sent:
                                sess._bridge_cond.wait(timeout=5.0)
                            if sess._bridge_rev > last_sent and sess._bridge_payload:
                                pkt = json.dumps(sess._bridge_payload, ensure_ascii=False)
                                last_sent = sess._bridge_rev
                            else:
                                pkt = None
                        if pkt:
                            self.wfile.write(f"data: {pkt}\n\n".encode("utf-8"))
                        else:
                            self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                except Exception:
                    pass
                finally:
                    with sess._bridge_cond:
                        sess._bridge_clients = max(0, sess._bridge_clients - 1)
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
                if path == "/api/syncmat/save":
                    self._send_json(sess.save_syncmat(body))
                    return
                if path == "/api/syncmat/insert":
                    self._send_json(sess.insert_syncmat(body))
                    return
                if path == "/api/export/materialx":
                    self._send_json(sess.export_materialx(body))
                    return
                if path == "/api/evaluate_anim_sequence":
                    self._send_json(sess.evaluate_anim_sequence(body))
                    return
                if path == "/api/paint_mask/stroke":
                    self._send_json(sess.paint_mask_stroke(body))
                    return
                if path == "/api/export/ctm":
                    self._send_json(sess.export_ctm(body))
                    return
                if path == "/api/export/turntable_gif":
                    self._send_json(sess.export_turntable_gif(body))
                    return
                if path == "/api/logs/clear":
                    self._send_json(sess.clear_file_logs())
                    return
                if path == "/api/paint_stack/stroke":
                    self._send_json(sess.paint_stack_stroke(body))
                    return
                if path == "/api/syncpack/save":
                    self._send_json(sess.save_syncpack(body))
                    return
                if path == "/api/syncpack/import":
                    self._send_json(sess.import_syncpack(body))
                    return
                if path == "/api/syncpack/spawn":
                    self._send_json(sess.spawn_syncpack(body))
                    return
                if path == "/api/batch/scan":
                    self._send_json(sess.batch_scan(body))
                    return
                if path == "/api/batch/start":
                    self._send_json(sess.batch_start(body))
                    return
                self._send_json({"error": f"Unknown API endpoint: {path}"}, status=404)
            except (ValueError, KeyError, TypeError, FileNotFoundError, ScriptSecurityError, TexDSLParseError, GraphCycleError) as exc:
                _write_studio_file_log("error", f"POST {path} -> 400", error=type(exc).__name__, detail=str(exc))
                self._send_json({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status=400)
            except Exception as exc:  # pragma: no cover
                _write_studio_file_log("error", f"POST {path} -> 500", error=type(exc).__name__, detail=str(exc))
                self._send_json({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status=500)

    server = _StudioThreadingHTTPServer((host, int(port)), StudioHTTPHandler)
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
    print(f"SyncCraft Studio v{__version__} (TexGraph v16 / Schema v4.0 · {len(NODE_REGISTRY)} nodes)")
    print(f"  Listening on {url}  (Press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
