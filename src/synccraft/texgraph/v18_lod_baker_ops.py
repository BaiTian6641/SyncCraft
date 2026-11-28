"""SyncCraft TexGraph v18 (`SyncCraft Studio v54.0`):
- Multi-Texture `.bbmodel` Slot Extraction & Per-Slot UV Masking (Blockbench 5.2.2 spec-accurate)
- Game-Ready Cuboid Auto-LOD Mesh Decimation (`LOD0 -> LOD1 -> LOD2`, bone & silhouette preserving)
- Substance Painter / Marmoset Smart Mesh Baker Layers (`mesh_bake_layer` & `vertex_color_bake`)
"""
from __future__ import annotations

import base64
import copy
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pbr import height_to_normal_map, pack_labpbr_1_3, rgba_float_to_u8
from .pixel_art_ops import hex_to_srgb


# ===========================================================================
# 1. Blockbench 5.2.2 Multi-Texture `.bbmodel` Slot Resolution & Masking
# ===========================================================================

_BB_FACES = ("north", "east", "south", "west", "up", "down")


def resolve_face_texture_slot(
    face_tex: Any,
    textures: list[dict[str, Any]],
) -> int | None:
    """Resolve a `.bbmodel` `faces.<dir>.texture` value to a 0-based index into `textures[]`.

    Follows exact Blockbench 5 `bbmodel.js` rules:
    - `None` -> culled/invisible face (returns `None`).
    - `False` -> unassigned blank face (maps to slot `0` if textures exist, else `0`).
    - `int` -> direct 0-based index into `textures[]`.
    - `str` -> match `uuid` first, then numeric index, then string `id`.
    """
    if face_tex is None:
        return None
    n = len(textures)
    if n == 0:
        return 0
    if isinstance(face_tex, bool):
        return 0
    if isinstance(face_tex, int):
        return face_tex if 0 <= face_tex < n else 0
    if isinstance(face_tex, str):
        s = face_tex.strip()
        if not s:
            return 0
        for idx, t in enumerate(textures):
            if str(t.get("uuid") or "") == s:
                return idx
        if s.isdigit():
            iv = int(s)
            if 0 <= iv < n:
                return iv
        for idx, t in enumerate(textures):
            if str(t.get("id") or "") == s:
                return idx
    return 0


def extract_bbmodel_texture_slots(bb_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Inspect all `textures[]` slots in a `.bbmodel` and count faces referencing each slot."""
    res = bb_doc.get("resolution") or {}
    def_w = max(1, int(res.get("width", 64) or 64))
    def_h = max(1, int(res.get("height", 64) or 64))
    raw_texs = [t for t in (bb_doc.get("textures") or []) if isinstance(t, dict)]

    slots: list[dict[str, Any]] = []
    if not raw_texs:
        slots.append(
            {
                "index": 0,
                "id": "0",
                "uuid": "slot_0",
                "name": f"{bb_doc.get('name') or 'texture'}.png",
                "width": def_w,
                "height": def_h,
                "uv_width": def_w,
                "uv_height": def_h,
                "face_count": 0,
                "has_source": False,
                "source": "",
            }
        )
    else:
        for idx, t in enumerate(raw_texs):
            w = max(1, int(t.get("width") or t.get("uv_width") or def_w))
            h = max(1, int(t.get("height") or t.get("uv_height") or def_h))
            uw = max(1, int(t.get("uv_width") or def_w))
            uh = max(1, int(t.get("uv_height") or def_h))
            src = str(t.get("source") or "")
            slots.append(
                {
                    "index": idx,
                    "id": str(t.get("id") if t.get("id") is not None else idx),
                    "uuid": str(t.get("uuid") or f"slot_{idx}"),
                    "name": str(t.get("name") or f"texture_{idx}.png"),
                    "width": w,
                    "height": h,
                    "uv_width": uw,
                    "uv_height": uh,
                    "face_count": 0,
                    "has_source": bool(src.startswith("data:image/")),
                    "source": src if src.startswith("data:image/") else "",
                }
            )

    for elem in bb_doc.get("elements") or []:
        if not isinstance(elem, dict) or elem.get("visibility") is False:
            continue
        faces = elem.get("faces") or {}
        for fname in _BB_FACES:
            finfo = faces.get(fname)
            if not isinstance(finfo, dict) or not finfo.get("uv"):
                continue
            s_idx = resolve_face_texture_slot(finfo.get("texture", 0), raw_texs)
            if s_idx is not None and 0 <= s_idx < len(slots):
                slots[s_idx]["face_count"] += 1

    return slots


def build_slot_uv_mask(
    bb_doc: dict[str, Any],
    slot_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a `(height, width)` float32 binary mask covering all UV rects owned by `slot_idx`."""
    H, W = max(1, int(height)), max(1, int(width))
    mask = np.zeros((H, W), dtype=np.float32)
    res = bb_doc.get("resolution") or {}
    raw_texs = [t for t in (bb_doc.get("textures") or []) if isinstance(t, dict)]
    if 0 <= slot_idx < len(raw_texs):
        tw = max(1.0, float(raw_texs[slot_idx].get("uv_width") or res.get("width") or W))
        th = max(1.0, float(raw_texs[slot_idx].get("uv_height") or res.get("height") or H))
    else:
        tw = max(1.0, float(res.get("width") or W))
        th = max(1.0, float(res.get("height") or H))

    for elem in bb_doc.get("elements") or []:
        if not isinstance(elem, dict) or elem.get("visibility") is False:
            continue
        faces = elem.get("faces") or {}
        for fname in _BB_FACES:
            finfo = faces.get(fname)
            if not isinstance(finfo, dict):
                continue
            uv = finfo.get("uv")
            if not uv or len(uv) < 4:
                continue
            s_idx = resolve_face_texture_slot(finfo.get("texture", 0), raw_texs)
            if s_idx != slot_idx:
                continue
            u0, v0, u1, v1 = float(uv[0]) / tw, float(uv[1]) / th, float(uv[2]) / tw, float(uv[3]) / th
            x0 = max(0, min(W, int(math.floor(min(u0, u1) * W))))
            x1 = max(0, min(W, int(math.ceil(max(u0, u1) * W))))
            y0 = max(0, min(H, int(math.floor(min(v0, v1) * H))))
            y1 = max(0, min(H, int(math.ceil(max(v0, v1) * H))))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
    return mask


# ===========================================================================
# 2. Game-Ready Cuboid Auto-LOD Mesh Decimation (`LOD0 -> LOD1 -> LOD2`)
# ===========================================================================

_CORE_KEYWORDS = (
    "torso", "chest", "pelvis", "hip", "waist", "spine", "body", "abdomen",
    "head", "skull", "cranium", "snout", "jaw", "neck",
    "thigh", "shin", "calf", "foot", "leg", "arm", "shoulder", "forearm", "hand",
    "wing", "spar", "tail", "coat", "cloak", "core", "blade", "staff", "pillar", "base",
)
_MICRO_KEYWORDS = (
    "rivet", "stud", "button", "buckle", "nail", "tooth", "teeth", "claw", "fang",
    "brow", "lash", "nostril", "spike_tiny", "trim", "piping", "bead", "screw",
    "gear", "cog", "bolt", "spark", "fringe", "tassel", "tuft", "scale_accent",
)


def _count_bbmodel_faces(bb_doc: dict[str, Any]) -> int:
    cnt = 0
    for el in bb_doc.get("elements") or []:
        if not isinstance(el, dict) or el.get("visibility") is False:
            continue
        faces = el.get("faces") or {}
        for fn in _BB_FACES:
            fi = faces.get(fn)
            if isinstance(fi, dict) and fi.get("uv") and fi.get("texture", 0) is not None:
                uv = fi["uv"]
                if len(uv) >= 4 and abs(float(uv[2]) - float(uv[0])) > 1e-6 and abs(float(uv[3]) - float(uv[1])) > 1e-6:
                    cnt += 1
    return cnt


def _count_outliner_bones(nodes: list[Any]) -> int:
    total = 0
    for item in nodes or []:
        if isinstance(item, dict):
            total += 1 + _count_outliner_bones(item.get("children") or [])
    return total


def _elem_geometry_metrics(el: dict[str, Any]) -> tuple[float, float, np.ndarray]:
    f = el.get("from") or [0.0, 0.0, 0.0]
    t = el.get("to") or [1.0, 1.0, 1.0]
    inf = float(el.get("inflate", 0.0) or 0.0)
    dx = max(0.05, abs(float(t[0]) - float(f[0])) + 2.0 * inf)
    dy = max(0.05, abs(float(t[1]) - float(f[1])) + 2.0 * inf)
    dz = max(0.05, abs(float(t[2]) - float(f[2])) + 2.0 * inf)
    vol = dx * dy * dz
    area = 2.0 * (dx * dy + dy * dz + dz * dx)
    ctr = np.array([0.5 * (float(f[0]) + float(t[0])), 0.5 * (float(f[1]) + float(t[1])), 0.5 * (float(f[2]) + float(t[2]))], dtype=np.float64)
    return vol, area, ctr


def _prune_outliner(
    nodes: list[Any],
    kept_elem_uids: set[str],
    *,
    max_bone_depth: int | None = None,
    depth: int = 0,
) -> list[Any]:
    """Prune culled elements from outliner; collapse empty leaf bones & optionally flatten deep micro-bones."""
    out: list[Any] = []
    for item in nodes or []:
        if isinstance(item, str):
            if item in kept_elem_uids:
                out.append(item)
        elif isinstance(item, dict):
            pruned_children = _prune_outliner(
                item.get("children") or [],
                kept_elem_uids,
                max_bone_depth=max_bone_depth,
                depth=depth + 1,
            )
            if not pruned_children:
                continue
            # At aggressive LOD2, flatten leaf micro-bones deeper than max_bone_depth up onto parent bone
            if max_bone_depth is not None and depth >= max_bone_depth:
                has_sub_group = any(isinstance(c, dict) for c in pruned_children)
                if not has_sub_group:
                    out.extend(pruned_children)
                    continue
            cloned = copy.deepcopy(item)
            cloned["children"] = pruned_children
            out.append(cloned)
    return out


def decimate_bbmodel_lod(
    bb_doc: dict[str, Any],
    *,
    target_ratio: float = 0.58,
    level_name: str = "lod1",
    strip_leaf_bones_depth: int | None = None,
) -> dict[str, Any]:
    """Generate a silhouette- & skeleton-preserving cuboid LOD `.bbmodel` at `target_ratio` in `(0, 1]`."""
    ratio = float(np.clip(target_ratio, 0.15, 1.0))
    doc = copy.deepcopy(bb_doc)
    elems = [e for e in (doc.get("elements") or []) if isinstance(e, dict) and e.get("visibility") is not False]
    orig_n = len(elems)
    orig_faces = _count_bbmodel_faces(doc)
    orig_bones = _count_outliner_bones(doc.get("outliner") or [])

    if ratio >= 0.999 or orig_n <= 3:
        return {
            "level": level_name,
            "target_ratio": round(ratio, 3),
            "box_count": orig_n,
            "face_count": orig_faces,
            "tri_count": orig_faces * 2,
            "bone_count": orig_bones,
            "reduction_pct": 0.0,
            "bbmodel": doc,
        }

    # Map each element UUID -> owning bone path & depth
    elem_bone_uid: dict[str, str] = {}
    elem_bone_name: dict[str, str] = {}
    elem_depth: dict[str, int] = {}

    def _index_outliner(nodes: list[Any], b_uid: str, b_name: str, d: int) -> None:
        for item in nodes or []:
            if isinstance(item, str):
                elem_bone_uid[item] = b_uid
                elem_bone_name[item] = b_name
                elem_depth[item] = d
            elif isinstance(item, dict):
                uid = str(item.get("uuid") or f"b_{d}")
                nm = str(item.get("name") or uid)
                _index_outliner(item.get("children") or [], uid, nm, d + 1)

    _index_outliner(doc.get("outliner") or [], "__root__", "root", 0)

    # Compute mesh centroid & per-element silhouette / structural importance scores
    vols: list[float] = []
    areas: list[float] = []
    ctrs: list[np.ndarray] = []
    for el in elems:
        v, a, c = _elem_geometry_metrics(el)
        vols.append(v)
        areas.append(a)
        ctrs.append(c)

    ctr_arr = np.stack(ctrs, axis=0)
    mesh_center = ctr_arr.mean(axis=0)
    mesh_rad = max(1.0, float(np.linalg.norm(ctr_arr - mesh_center[None, :], axis=1).max()))
    vol_arr = np.asarray(vols, dtype=np.float64)
    area_arr = np.asarray(areas, dtype=np.float64)
    vol_q25 = float(np.percentile(vol_arr, 25)) if orig_n >= 4 else 0.0

    # Identify largest cuboid in each bone so animated limbs never vanish at LOD1
    bone_largest_uid: dict[str, tuple[float, str]] = {}
    scores: list[tuple[float, str]] = []
    for idx, el in enumerate(elems):
        uid = str(el.get("uuid") or f"el_{idx}")
        el["uuid"] = uid
        ename = str(el.get("name") or "").lower()
        b_uid = elem_bone_uid.get(uid, "__root__")
        b_name = elem_bone_name.get(uid, "").lower()
        depth = elem_depth.get(uid, 0)

        # Base geometric importance: volume^(0.55) * area^(0.45) balances chunky cores + thin silhouette plates/wings
        geom_score = (vol_arr[idx] ** 0.55) * (area_arr[idx] ** 0.45)
        sil_dist = float(np.linalg.norm(ctrs[idx] - mesh_center)) / mesh_rad
        sil_w = 1.0 + 0.45 * sil_dist

        core_w = 1.0
        if any(k in ename or k in b_name for k in _CORE_KEYWORDS):
            core_w *= 1.9
        if any(k in ename for k in _MICRO_KEYWORDS) or vol_arr[idx] <= vol_q25:
            core_w *= 0.42
        # Shallow structural bones weighted higher than deep decorative leaf bones
        depth_w = 1.0 / (1.0 + 0.14 * max(0, depth - 1))

        total_score = geom_score * sil_w * core_w * depth_w
        scores.append((total_score, uid))
        prev = bone_largest_uid.get(b_uid)
        if prev is None or total_score > prev[0]:
            bone_largest_uid[b_uid] = (total_score, uid)

    target_n = max(3, min(orig_n, int(round(orig_n * ratio))))
    # At LOD1 (ratio >= 0.5), prioritize keeping the hero cuboid of every top/mid bone first
    protected_uids: set[str] = set()
    if ratio >= 0.50:
        bone_heroes = sorted(bone_largest_uid.values(), key=lambda x: x[0], reverse=True)
        for _, u_h in bone_heroes[: int(target_n * 0.72)]:
            protected_uids.add(u_h)

    ranked = sorted(scores, key=lambda x: x[0], reverse=True)
    kept_uids: set[str] = set(protected_uids)
    for _, uid in ranked:
        if len(kept_uids) >= target_n:
            break
        kept_uids.add(uid)

    doc["elements"] = [e for e in elems if str(e.get("uuid")) in kept_uids]
    if isinstance(doc.get("outliner"), list):
        doc["outliner"] = _prune_outliner(
            doc["outliner"],
            kept_uids,
            max_bone_depth=strip_leaf_bones_depth,
        )
    doc["name"] = f"{bb_doc.get('name') or 'model'}_{level_name}"

    # Prune `groups` metadata table to match kept bones if present
    kept_group_uids: set[str] = set()

    def _collect_group_uids(nodes: list[Any]) -> None:
        for item in nodes or []:
            if isinstance(item, dict):
                if item.get("uuid"):
                    kept_group_uids.add(str(item["uuid"]))
                _collect_group_uids(item.get("children") or [])

    _collect_group_uids(doc.get("outliner") or [])
    if isinstance(doc.get("groups"), list) and kept_group_uids:
        doc["groups"] = [g for g in doc["groups"] if isinstance(g, dict) and str(g.get("uuid")) in kept_group_uids]

    new_faces = _count_bbmodel_faces(doc)
    new_bones = _count_outliner_bones(doc.get("outliner") or [])
    red_pct = round(100.0 * (1.0 - (new_faces / max(1.0, float(orig_faces)))), 1)
    return {
        "level": level_name,
        "target_ratio": round(ratio, 3),
        "box_count": len(doc["elements"]),
        "face_count": new_faces,
        "tri_count": new_faces * 2,
        "bone_count": new_bones,
        "reduction_pct": max(0.0, red_pct),
        "bbmodel": doc,
    }


def generate_bbmodel_lod_chain(
    bb_doc: dict[str, Any],
    *,
    lod1_ratio: float = 0.58,
    lod2_ratio: float = 0.35,
) -> dict[str, dict[str, Any]]:
    """Build a complete 3-tier `LOD0 / LOD1 / LOD2` chain preserving skeleton & silhouette."""
    return {
        "lod0": decimate_bbmodel_lod(bb_doc, target_ratio=1.0, level_name="lod0", strip_leaf_bones_depth=None),
        "lod1": decimate_bbmodel_lod(bb_doc, target_ratio=lod1_ratio, level_name="lod1", strip_leaf_bones_depth=None),
        "lod2": decimate_bbmodel_lod(bb_doc, target_ratio=lod2_ratio, level_name="lod2", strip_leaf_bones_depth=3),
    }


def export_lod_bundle(
    bb_doc: dict[str, Any],
    out_dir: str | Path,
    *,
    stem: str | None = None,
    graph: Any = None,
    lod1_ratio: float = 0.58,
    lod2_ratio: float = 0.35,
    bleed_px: int = 4,
) -> dict[str, Any]:
    """Generate and write `LOD0 / LOD1 / LOD2` `.bbmodel` files + mip-dilated LOD textures + `lod_manifest.json`."""
    from .v17_decal_atlas_ops import mip_safe_dilate_rgba

    t0 = time.perf_counter()
    st = str(stem or bb_doc.get("name") or "synccraft_asset").strip() or "synccraft_asset"
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    chain = generate_bbmodel_lod_chain(bb_doc, lod1_ratio=lod1_ratio, lod2_ratio=lod2_ratio)

    # Optionally render mip-safe downscaled textures for LOD0 (100%), LOD1 (50%), LOD2 (25%)
    base_alb_u8: np.ndarray | None = None
    if graph is not None:
        ev = graph.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
        pm = ev.pbr_maps or pack_labpbr_1_3(ev.output)
        alb = pm["albedo"]
        mc = getattr(graph, "mesh_context", None)
        seed_m = mc.island_mask if (mc is not None and hasattr(mc, "island_mask")) else (alb[..., 3] > 5)
        if bleed_px > 0 and np.any(seed_m) and not np.all(seed_m):
            alb = mip_safe_dilate_rgba(alb, valid_mask=seed_m, bleed_px=bleed_px, preserve_alpha=True)
        base_alb_u8 = alb

    scale_map = {"lod0": 1.0, "lod1": 0.5, "lod2": 0.25}
    written_files: dict[str, str] = {}
    summary_levels: list[dict[str, Any]] = []

    for lvl in ("lod0", "lod1", "lod2"):
        info = chain[lvl]
        bb_out = copy.deepcopy(info["bbmodel"])
        tex_scale = scale_map[lvl]
        tex_w, tex_h = None, None
        if base_alb_u8 is not None:
            bh, bw = base_alb_u8.shape[:2]
            tw = max(16, int(round(bw * tex_scale)))
            th = max(16, int(round(bh * tex_scale)))
            tex_w, tex_h = tw, th
            im = Image.fromarray(base_alb_u8, mode="RGBA")
            if (tw, th) != (bw, bh):
                im = im.resize((tw, th), resample=Image.Resampling.BOX)
            tex_fp = out_root / f"{st}_{lvl}.png"
            im.save(tex_fp, format="PNG", optimize=True)
            written_files[f"{lvl}_texture"] = str(tex_fp)

            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            bb_out["textures"] = [
                {
                    "name": f"{st}_{lvl}.png",
                    "id": "0",
                    "width": tw,
                    "height": th,
                    "uv_width": bw,
                    "uv_height": bh,
                    "internal": True,
                    "source": b64,
                }
            ]

        bb_fp = out_root / f"{st}_{lvl}.bbmodel"
        bb_fp.write_text(json.dumps(bb_out, separators=(",", ":")), encoding="utf-8")
        written_files[f"{lvl}_bbmodel"] = str(bb_fp)
        summary_levels.append(
            {
                "level": lvl,
                "box_count": info["box_count"],
                "face_count": info["face_count"],
                "tri_count": info["tri_count"],
                "bone_count": info["bone_count"],
                "reduction_pct": info["reduction_pct"],
                "texture_size": [tex_w, tex_h] if tex_w and tex_h else None,
                "bbmodel_path": str(bb_fp),
            }
        )

    manifest = {
        "schema": "synccraft.lod_bundle/1.0",
        "ok": True,
        "stem": st,
        "out_dir": str(out_root),
        "bleed_px": int(bleed_px),
        "levels": summary_levels,
        "files": written_files,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    mf_path = out_root / f"{st}_lod_manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(mf_path)
    return manifest


# ===========================================================================
# 3. Substance Painter / Marmoset Smart Mesh Baker Layers
#    (`mesh_bake_layer` & `vertex_color_bake` -> 234 Total Typed Nodes!)
# ===========================================================================

SMART_BAKER_PRESETS: dict[str, dict[str, Any]] = {
    "edge_wear_metal": {
        "title": "Edge Wear & Scuffed Metal",
        "edge_color": "#E6ECF5",
        "edge_strength": 0.72,
        "edge_roughness": 0.22,
        "edge_metallic": 0.92,
        "cavity_color": "#1E1916",
        "cavity_strength": 0.42,
        "cavity_roughness": 0.88,
        "ao_multiply": 0.35,
        "top_color": "#C9D0DC",
        "top_strength": 0.0,
        "top_roughness": 0.78,
    },
    "crevice_grime": {
        "title": "Crevice Dirt & Contact Grime",
        "edge_color": "#D4C9B8",
        "edge_strength": 0.14,
        "edge_roughness": 0.55,
        "edge_metallic": 0.10,
        "cavity_color": "#261E18",
        "cavity_strength": 0.78,
        "cavity_roughness": 0.94,
        "ao_multiply": 0.55,
        "top_color": "#7B6E60",
        "top_strength": 0.18,
        "top_roughness": 0.90,
    },
    "alpine_snow_dust": {
        "title": "World-Normal Top-Down Snow / Frost",
        "edge_color": "#E8F4FF",
        "edge_strength": 0.25,
        "edge_roughness": 0.36,
        "edge_metallic": 0.0,
        "cavity_color": "#384B66",
        "cavity_strength": 0.28,
        "cavity_roughness": 0.65,
        "ao_multiply": 0.25,
        "top_color": "#EEF5FF",
        "top_strength": 0.82,
        "top_roughness": 0.48,
    },
    "moss_overgrowth": {
        "title": "Stylized Top-Down Moss & Crevice Lichen",
        "edge_color": "#90A868",
        "edge_strength": 0.12,
        "edge_roughness": 0.78,
        "edge_metallic": 0.0,
        "cavity_color": "#283D22",
        "cavity_strength": 0.62,
        "cavity_roughness": 0.92,
        "ao_multiply": 0.40,
        "top_color": "#4F7D36",
        "top_strength": 0.76,
        "top_roughness": 0.86,
    },
    "painted_chipped": {
        "title": "Chipped Paint Revealing Raw Steel",
        "edge_color": "#BCC7D6",
        "edge_strength": 0.85,
        "edge_roughness": 0.26,
        "edge_metallic": 0.96,
        "cavity_color": "#2A201A",
        "cavity_strength": 0.45,
        "cavity_roughness": 0.88,
        "ao_multiply": 0.32,
        "top_color": "#D4DCE6",
        "top_strength": 0.0,
        "top_roughness": 0.60,
    },
    "studio_ao_curvature": {
        "title": "Stylized Curvature & Contact AO Polish",
        "edge_color": "#FFF6E6",
        "edge_strength": 0.32,
        "edge_roughness": 0.40,
        "edge_metallic": 0.12,
        "cavity_color": "#1A1C28",
        "cavity_strength": 0.38,
        "cavity_roughness": 0.82,
        "ao_multiply": 0.45,
        "top_color": "#FFFFFF",
        "top_strength": 0.08,
        "top_roughness": 0.52,
    },
}


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-5, edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _hash_grain_2d(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = float((int(seed) % 9973) + 17)
    n1 = np.sin(xx * 0.47 + yy * 0.31 + s * 0.13) * np.cos(xx * 0.29 - yy * 0.53 + s * 0.07)
    n2 = np.sin(xx * 1.13 - yy * 0.89 + s * 0.29)
    return np.clip(0.5 + 0.35 * n1 + 0.15 * n2, 0.0, 1.0).astype(np.float32)


def _derive_height_curvature(height: np.ndarray) -> np.ndarray:
    """Compute Marmoset/Substance-style diverging curvature (`0=concave, 0.5=flat, 1=convex`) from a heightmap."""
    h = np.asarray(height, dtype=np.float32)
    up = np.roll(h, -1, axis=0)
    dn = np.roll(h, 1, axis=0)
    lf = np.roll(h, 1, axis=1)
    rt = np.roll(h, -1, axis=1)
    lap = h - 0.25 * (up + dn + lf + rt)
    return np.clip(0.5 + lap * 4.2, 0.0, 1.0).astype(np.float32)


def mesh_bake_layer_op(
    height: int,
    width: int,
    *,
    base: dict[str, np.ndarray] | np.ndarray | None = None,
    mask: np.ndarray | None = None,
    mesh_context: Any = None,
    preset: str = "edge_wear_metal",
    edge_strength: float = -1.0,
    cavity_strength: float = -1.0,
    ao_multiply: float = -1.0,
    top_strength: float = -1.0,
    edge_color: str = "",
    cavity_color: str = "",
    top_color: str = "",
    breakup: float = 0.35,
    contrast: float = 1.4,
    seed: int = 42,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Composite Substance/Marmoset 3D Curvature Edge Wear, Cavity Dirt, Contact AO & World-Normal Dust onto a MATERIAL bus."""
    H, W = int(height), int(width)
    if isinstance(base, dict) and "albedo" in base:
        alb = np.asarray(base["albedo"], dtype=np.float32).copy()
        h_ch = np.asarray(base.get("height", np.full((H, W), 0.5, dtype=np.float32)), dtype=np.float32).copy()
        r_ch = np.asarray(base.get("roughness", np.full((H, W), 0.55, dtype=np.float32)), dtype=np.float32).copy()
        m_ch = np.asarray(base.get("metallic", np.zeros((H, W), dtype=np.float32)), dtype=np.float32).copy()
        ao_ch = np.asarray(base.get("ao", np.ones((H, W), dtype=np.float32)), dtype=np.float32).copy()
        em_ch = base.get("emissive")
        sh_ch = base.get("shadow")
        bn_ch = base.get("bent_normal")
        th_ch = base.get("thickness")
        sss_ch = base.get("sss")
        por_ch = base.get("porosity")
        nrm_in = base.get("normal")
    elif isinstance(base, np.ndarray):
        arr = np.asarray(base, dtype=np.float32)
        if arr.ndim == 2:
            alb = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
        elif arr.shape[2] == 3:
            alb = np.concatenate((arr, np.ones((H, W, 1), dtype=np.float32)), axis=-1)
        else:
            alb = arr.copy()
        h_ch = np.mean(alb[..., :3], axis=-1).astype(np.float32)
        r_ch = np.full((H, W), 0.55, dtype=np.float32)
        m_ch = np.zeros((H, W), dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None
    else:
        alb = np.full((H, W, 4), [0.34, 0.39, 0.48, 1.0], dtype=np.float32)
        h_ch = np.full((H, W), 0.5, dtype=np.float32)
        r_ch = np.full((H, W), 0.55, dtype=np.float32)
        m_ch = np.full((H, W), 0.15, dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None

    spec = SMART_BAKER_PRESETS.get(str(preset), SMART_BAKER_PRESETS["edge_wear_metal"])
    e_str = float(spec["edge_strength"]) if edge_strength < 0.0 else float(np.clip(edge_strength, 0.0, 1.0))
    c_str = float(spec["cavity_strength"]) if cavity_strength < 0.0 else float(np.clip(cavity_strength, 0.0, 1.0))
    ao_m = float(spec["ao_multiply"]) if ao_multiply < 0.0 else float(np.clip(ao_multiply, 0.0, 1.0))
    t_str = float(spec["top_strength"]) if top_strength < 0.0 else float(np.clip(top_strength, 0.0, 1.0))

    e_rgb = np.array(hex_to_srgb(edge_color or spec["edge_color"]), dtype=np.float32)
    c_rgb = np.array(hex_to_srgb(cavity_color or spec["cavity_color"]), dtype=np.float32)
    t_rgb = np.array(hex_to_srgb(top_color or spec["top_color"]), dtype=np.float32)

    # Extract 3D baked channels if MeshBakeContext matches resolution, else synthesize from heightmap
    has_mc = (
        mesh_context is not None
        and hasattr(mesh_context, "curvature_3d")
        and mesh_context.curvature_3d.shape == (H, W)
    )
    if has_mc:
        curv = np.asarray(mesh_context.curvature_3d, dtype=np.float32)
        bevel = np.asarray(mesh_context.edge_bevel, dtype=np.float32)
        ao_map = np.asarray(mesh_context.baked_ao, dtype=np.float32)
        cav_map = np.asarray(mesh_context.joint_cavity, dtype=np.float32)
        wnrm = np.asarray(mesh_context.world_normal, dtype=np.float32)
        imask = np.asarray(mesh_context.island_mask, dtype=np.float32)
        # Combine 3D curvature with island edge bevel for crisp stylized convex highlights
        curv = np.clip(0.65 * curv + 0.35 * (0.5 + 0.5 * bevel), 0.0, 1.0)
        up_align = np.clip(wnrm[..., 1], 0.0, 1.0)
    else:
        curv = _derive_height_curvature(h_ch)
        ao_map = np.clip(ao_ch, 0.0, 1.0)
        cav_map = _smoothstep(0.50, 0.25, curv)
        yy = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]
        up_align = np.broadcast_to(yy, (H, W)).copy()
        imask = np.ones((H, W), dtype=np.float32)

    grain = _hash_grain_2d(H, W, seed)
    brk = float(np.clip(breakup, 0.0, 1.0))
    ctr = max(0.3, min(4.0, float(contrast)))

    # 1. Convex Edge Wear Mask (curvature > 0.5)
    raw_conv = np.clip((curv - 0.5) * 2.0 * ctr, 0.0, 1.0)
    conv_m = np.clip(raw_conv * (1.0 - brk + brk * (0.35 + 0.95 * grain)), 0.0, 1.0) * imask

    # 2. Concave Crevice / Cavity Dirt Mask (curvature < 0.5 + joint_cavity)
    raw_conc = np.clip((0.5 - curv) * 2.0 * ctr, 0.0, 1.0)
    cav_m = np.clip(np.maximum(raw_conc, cav_map) * (1.0 - 0.5 * brk + 0.5 * brk * grain), 0.0, 1.0) * imask

    # 3. World-Normal Top-Down Dust / Snow Mask
    top_m = _smoothstep(0.32, 0.88, up_align) * (1.0 - 0.35 * (1.0 - ao_map))
    top_m = np.clip(top_m * (1.0 - 0.65 * brk + 0.65 * brk * grain), 0.0, 1.0) * imask

    # Optional external mask constraint
    if mask is not None:
        m_ext = np.asarray(mask, dtype=np.float32)
        if m_ext.ndim == 3:
            m_ext = m_ext[..., 0]
        m_ext = np.clip(m_ext, 0.0, 1.0)
        conv_m *= m_ext
        cav_m *= m_ext
        top_m *= m_ext
        ao_eff = 1.0 - (1.0 - ao_map) * ao_m * m_ext
    else:
        ao_eff = 1.0 - (1.0 - ao_map) * ao_m * imask

    # Apply Contact AO Multiply
    if ao_m > 1e-4:
        alb[..., :3] = np.clip(alb[..., :3] * ao_eff[..., None], 0.0, 1.0)
        ao_ch = np.clip(ao_ch * ao_eff, 0.0, 1.0)

    # Apply Concave Crevice / Cavity Dirt
    if c_str > 1e-4:
        w_c = np.clip(cav_m * c_str, 0.0, 1.0)
        alb[..., :3] = alb[..., :3] * (1.0 - w_c[..., None]) + c_rgb[None, None, :] * w_c[..., None]
        r_ch = r_ch * (1.0 - w_c) + float(spec["cavity_roughness"]) * w_c
        m_ch = m_ch * (1.0 - w_c * 0.75)
        h_ch = np.clip(h_ch - w_c * 0.12, 0.0, 1.0)

    # Apply Convex Edge Wear
    if e_str > 1e-4:
        w_e = np.clip(conv_m * e_str, 0.0, 1.0)
        alb[..., :3] = alb[..., :3] * (1.0 - w_e[..., None]) + e_rgb[None, None, :] * w_e[..., None]
        r_ch = r_ch * (1.0 - w_e) + float(spec["edge_roughness"]) * w_e
        m_ch = m_ch * (1.0 - w_e) + float(spec["edge_metallic"]) * w_e
        h_ch = np.clip(h_ch + w_e * 0.10, 0.0, 1.0)

    # Apply World-Normal Top-Down Dust / Snow / Moss
    if t_str > 1e-4:
        w_t = np.clip(top_m * t_str, 0.0, 1.0)
        alb[..., :3] = alb[..., :3] * (1.0 - w_t[..., None]) + t_rgb[None, None, :] * w_t[..., None]
        r_ch = r_ch * (1.0 - w_t) + float(spec["top_roughness"]) * w_t
        m_ch = m_ch * (1.0 - w_t * 0.85)
        h_ch = np.clip(h_ch + w_t * 0.08, 0.0, 1.0)

    out_bus: dict[str, np.ndarray] = {
        "albedo": np.clip(alb, 0.0, 1.0).astype(np.float32),
        "height": np.clip(h_ch, 0.0, 1.0).astype(np.float32),
        "normal": nrm_in if nrm_in is not None else height_to_normal_map(h_ch, strength=2.2, tileable=tileable),
        "roughness": np.clip(r_ch, 0.04, 0.98).astype(np.float32),
        "metallic": np.clip(m_ch, 0.0, 1.0).astype(np.float32),
        "ao": np.clip(ao_ch, 0.0, 1.0).astype(np.float32),
    }
    if em_ch is not None:
        out_bus["emissive"] = em_ch
    if sh_ch is not None:
        out_bus["shadow"] = sh_ch
    if bn_ch is not None:
        out_bus["bent_normal"] = bn_ch
    if th_ch is not None:
        out_bus["thickness"] = th_ch
    if sss_ch is not None:
        out_bus["sss"] = sss_ch
    if por_ch is not None:
        out_bus["porosity"] = por_ch
    return out_bus


def _golden_id_palette(max_id: int, seed: int = 42) -> np.ndarray:
    """Generate vivid golden-ratio-spaced sRGB colors for Per-Part / Per-Bone / Per-Zone Color-ID maps."""
    n = max(2, int(max_id) + 2)
    lut = np.zeros((n, 3), dtype=np.float32)
    phi = 0.6180339887498949
    h0 = ((int(seed) * 0.381966) % 1.0)
    for i in range(1, n):
        hue = (h0 + i * phi) % 1.0
        sat = 0.68 + 0.22 * ((i * 7) % 3) / 2.0
        val = 0.72 + 0.22 * ((i * 5) % 3) / 2.0
        # HSV -> sRGB
        k = hue * 6.0
        sec = int(k) % 6
        f = k - math.floor(k)
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * f)
        t = val * (1.0 - sat * (1.0 - f))
        if sec == 0:
            rgb = (val, t, p)
        elif sec == 1:
            rgb = (q, val, p)
        elif sec == 2:
            rgb = (p, val, t)
        elif sec == 3:
            rgb = (p, q, val)
        elif sec == 4:
            rgb = (t, p, val)
        else:
            rgb = (val, p, q)
        lut[i] = rgb
    return lut


def vertex_color_bake_op(
    height: int,
    width: int,
    *,
    base: np.ndarray | dict[str, np.ndarray] | None = None,
    mesh_context: Any = None,
    mode: str = "part_id",
    tint_strength: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Bake Marmoset/Substance Color-ID (`part_id`, `bone_id`, `zone_id`), Curvature, World-Normal, World-Pos, or AO/Curv/Cavity RGB map."""
    H, W = int(height), int(width)
    m = str(mode or "part_id").strip().lower()
    has_mc = (
        mesh_context is not None
        and hasattr(mesh_context, "elem_id_map")
        and mesh_context.elem_id_map.shape == (H, W)
    )

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.ones((H, W), dtype=np.float32)

    if has_mc:
        imask = np.asarray(mesh_context.island_mask, dtype=bool)
        eids = np.asarray(mesh_context.elem_id_map, dtype=np.int32)
        if m == "part_id":
            max_e = max(1, int(eids.max()) + 1)
            lut = _golden_id_palette(max_e, seed=seed)
            rgb = lut[np.clip(eids + 1, 0, lut.shape[0] - 1)]
            rgb[~imask] = 0.08
        elif m == "bone_id":
            bones = list(getattr(mesh_context, "elem_bones", []) or [])
            uniq_b = {b: i + 1 for i, b in enumerate(sorted(set(bones)))}
            lut = _golden_id_palette(len(uniq_b) + 2, seed=seed + 13)
            b_idx_table = np.zeros((max(1, len(bones) + 1),), dtype=np.int32)
            for ei, bn in enumerate(bones):
                b_idx_table[ei] = uniq_b.get(bn, 1)
            mapped = np.where(eids >= 0, b_idx_table[np.clip(eids, 0, len(b_idx_table) - 1)], 0)
            rgb = lut[np.clip(mapped, 0, lut.shape[0] - 1)]
            rgb[~imask] = 0.08
        elif m == "zone_id":
            zones = list(getattr(mesh_context, "elem_zones", []) or [])
            uniq_z = {z: i + 1 for i, z in enumerate(sorted(set(zones)))}
            lut = _golden_id_palette(len(uniq_z) + 2, seed=seed + 29)
            z_idx_table = np.zeros((max(1, len(zones) + 1),), dtype=np.int32)
            for ei, zn in enumerate(zones):
                z_idx_table[ei] = uniq_z.get(zn, 1)
            mapped = np.where(eids >= 0, z_idx_table[np.clip(eids, 0, len(z_idx_table) - 1)], 0)
            rgb = lut[np.clip(mapped, 0, lut.shape[0] - 1)]
            rgb[~imask] = 0.08
        elif m == "world_normal_rgb":
            rgb = np.clip(np.asarray(mesh_context.world_normal, dtype=np.float32) * 0.5 + 0.5, 0.0, 1.0)
            rgb[~imask] = 0.5
        elif m == "world_pos_rgb":
            rgb = np.clip(np.asarray(mesh_context.world_pos_norm, dtype=np.float32), 0.0, 1.0)
            rgb[~imask] = 0.08
        elif m == "curvature_diverging":
            c = np.clip(np.asarray(mesh_context.curvature_3d, dtype=np.float32), 0.0, 1.0)
            # Concave = cool indigo, Flat = neutral gray, Convex = bright warm gold
            low = np.clip((0.5 - c) * 2.0, 0.0, 1.0)
            hi = np.clip((c - 0.5) * 2.0, 0.0, 1.0)
            rgb[..., 0] = np.clip(0.5 - 0.42 * low + 0.50 * hi, 0.0, 1.0)
            rgb[..., 1] = np.clip(0.5 - 0.36 * low + 0.42 * hi, 0.0, 1.0)
            rgb[..., 2] = np.clip(0.5 + 0.25 * low + 0.18 * hi, 0.0, 1.0)
            rgb[~imask] = 0.5
        else:  # ao_cavity_rgb (R=AO, G=Curvature, B=1-Cavity)
            rgb[..., 0] = np.clip(np.asarray(mesh_context.baked_ao, dtype=np.float32), 0.0, 1.0)
            rgb[..., 1] = np.clip(np.asarray(mesh_context.curvature_3d, dtype=np.float32), 0.0, 1.0)
            rgb[..., 2] = np.clip(1.0 - np.asarray(mesh_context.joint_cavity, dtype=np.float32), 0.0, 1.0)
            rgb[~imask] = 0.5
    else:
        # Clean procedural 2D fallback (works cleanly in node swatches & 2D graphs)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = xx / max(1.0, W - 1.0)
        v = yy / max(1.0, H - 1.0)
        if m in ("part_id", "bone_id", "zone_id"):
            gx = np.floor(u * 4.0).astype(np.int32)
            gy = np.floor(v * 4.0).astype(np.int32)
            cid = (gy * 4 + gx + 1).astype(np.int32)
            lut = _golden_id_palette(20, seed=seed)
            rgb = lut[np.clip(cid, 0, lut.shape[0] - 1)]
        elif m == "world_normal_rgb":
            rgb[..., 0] = 0.5 + 0.45 * np.sin((u - 0.5) * math.pi)
            rgb[..., 1] = 0.5 + 0.45 * np.cos((v - 0.5) * math.pi)
            rgb[..., 2] = 0.85
        elif m == "world_pos_rgb":
            rgb[..., 0] = u
            rgb[..., 1] = 1.0 - v
            rgb[..., 2] = 0.5 * (u + v)
        elif m == "curvature_diverging":
            r = np.hypot(u - 0.5, v - 0.5)
            c = np.clip(0.5 + 0.45 * np.cos(r * 14.0), 0.0, 1.0)
            rgb[..., 0] = np.clip(0.5 + (c - 0.5) * 0.95, 0.0, 1.0)
            rgb[..., 1] = np.clip(0.5 + (c - 0.5) * 0.78, 0.0, 1.0)
            rgb[..., 2] = np.clip(0.5 - (c - 0.5) * 0.45, 0.0, 1.0)
        else:
            r = np.hypot(u - 0.5, v - 0.5)
            rgb[..., 0] = np.clip(1.0 - r * 1.1, 0.15, 1.0)
            rgb[..., 1] = np.clip(0.5 + 0.4 * np.sin(u * 12.0) * np.cos(v * 12.0), 0.0, 1.0)
            rgb[..., 2] = np.clip(0.8 - 0.5 * r, 0.1, 1.0)

    ts = float(np.clip(tint_strength, 0.0, 1.0))
    if base is not None and ts < 0.999:
        if isinstance(base, dict) and "albedo" in base:
            b_arr = np.asarray(base["albedo"], dtype=np.float32)
        else:
            b_arr = np.asarray(base, dtype=np.float32)
        if b_arr.ndim == 2:
            b_rgb = np.stack((b_arr, b_arr, b_arr), axis=-1)
        else:
            b_rgb = b_arr[..., :3]
            if b_arr.shape[-1] == 4:
                alpha = b_arr[..., 3].astype(np.float32)
        # Soft-tint blend when mixing Color-ID / Vertex-Color onto base albedo
        tinted = np.clip(b_rgb * (0.55 + 0.90 * rgb), 0.0, 1.0)
        rgb = b_rgb * (1.0 - ts) + tinted * ts

    return np.concatenate((np.clip(rgb, 0.0, 1.0), alpha[..., None]), axis=-1).astype(np.float32)
