"""P1-03e Permissive Internet Asset Ingestion & Inverse Material Posterior Inference.

Supports decompiling external cuboid assets from:
1. Blockbench 5 ``.bbmodel`` (outliner/groups/elements + embedded PNG data URLs)
2. Minecraft Java ``.json`` (elements + texture_size)
3. Minecraft Bedrock ``.geo.json`` (``minecraft:geometry`` bones/cubes)

Enforces a strict permissive-license & provenance gate before ingestion, recovers
validated ``@rig`` ``Program`` ASTs with verified 3D vertex reconstruction, infers
top-K candidate ``MaterialProgram`` hypotheses with Oklab residuals and epistemic
identifiability scores (elevated when LabPBR ``_n``/``_s`` maps are present), and
extracts localized ``StampOverride`` layers for artist-authored markings (eyes, trim).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import io
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from synccraft.dsl.ast_nodes import (
    FACES,
    LOCATOR_ROLES,
    Attach,
    BehaviorDecl,
    Bone,
    Box,
    ClipDecl,
    Locator,
    Meta,
    Palette,
    Place,
    Program,
    TrackDecl,
    validate_program,
)
from synccraft.kinematics.se3_chain import compile_geometry, euler_xyz_matrix
from synccraft.surface.contracts import (
    MATERIAL_FAMILIES,
    MaterialProgram,
    MaterialSpec,
    StampOverride,
    linear_srgb_to_oklab,
    linear_srgb_to_oklch,
    srgb_u8_to_linear,
)
from synccraft.surface.domain import face_uv_extents_units

APPROVED_PERMISSIVE_LICENSES: frozenset[str] = frozenset({
    "CC0-1.0",
    "CC-BY-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
})

APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES: frozenset[str] = frozenset({
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
})

LICENSE_POLICIES: frozenset[str] = frozenset({
    "clean_room_permissive",
    "research_nc",
})

BANNED_PROVENANCE_TOKENS: tuple[str, ...] = (
    "mojang",
    "minecraft_jar",
    "official_vanilla_jar",
    "all_rights_reserved",
    "proprietary",
    "unlicensed",
)


class LicenseGateError(PermissionError):
    """Raised when an external asset lacks an approved license for the active policy or clean provenance."""


@dataclass(frozen=True)
class AssetProvenance:
    """Mandatory provenance & license record for internet-gathered 3D cuboid assets."""

    asset_id: str
    source_url: str
    author: str
    license_spdx: str
    attribution_notice: str = ""


def verify_asset_provenance(
    provenance: AssetProvenance,
    *,
    license_policy: str = "clean_room_permissive",
) -> str:
    """Enforce AGENTS.md / ADR 0003 / ADR 0006 license quarantine and anti-Mojang-asset gate.

    Returns the resolved license track: ``'clean_room_permissive'`` or ``'research_nc'``.
    Default ``license_policy='clean_room_permissive'`` strictly rejects ``CC-BY-NC`` assets
    unless ``license_policy='research_nc'`` is explicitly requested for non-commercial
    research evaluation/training quarantine (e.g. Cobblemon ``CC-BY-NC-4.0`` assets).
    """
    if license_policy not in LICENSE_POLICIES:
        raise ValueError(f"Unknown license_policy {license_policy!r}; expected one of {sorted(LICENSE_POLICIES)}")
    if not isinstance(provenance, AssetProvenance):
        raise LicenseGateError("AssetProvenance record is required for external ingestion")
    if not provenance.asset_id.strip() or not provenance.source_url.strip() or not provenance.author.strip():
        raise LicenseGateError("asset_id, source_url, and author must all be non-empty")

    combined = f"{provenance.asset_id} {provenance.source_url} {provenance.author} {provenance.attribution_notice}".lower()
    for banned in BANNED_PROVENANCE_TOKENS:
        if banned in combined:
            raise LicenseGateError(f"Asset provenance contains prohibited source marker {banned!r}")

    if provenance.license_spdx in APPROVED_PERMISSIVE_LICENSES:
        return "clean_room_permissive"
    if license_policy == "research_nc" and provenance.license_spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES:
        if not provenance.attribution_notice.strip():
            raise LicenseGateError(
                f"{provenance.license_spdx} requires a non-empty attribution_notice under CC-BY-NC quarantine"
            )
        return "research_nc"

    allowed = (
        sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES)
        if license_policy == "research_nc"
        else sorted(APPROVED_PERMISSIVE_LICENSES)
    )
    hint = (
        " (pass license_policy='research_nc' for quarantined non-commercial CC-BY-NC research splits)"
        if provenance.license_spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES
        else ""
    )
    raise LicenseGateError(
        f"Rejected license {provenance.license_spdx!r} under policy {license_policy!r}; "
        f"allowed licenses: {allowed}{hint}"
    )


def _sanitize_ident(raw: str, fallback: str, used: set[str]) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(raw).strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    cand = cleaned
    idx = 2
    while cand in used:
        cand = f"{cleaned}_{idx}"
        idx += 1
    used.add(cand)
    return cand


def _decode_png_rgba(png_bytes_or_url: bytes | str) -> np.ndarray:
    if isinstance(png_bytes_or_url, str):
        prefix = "data:image/png;base64,"
        if png_bytes_or_url.startswith(prefix):
            raw = base64.b64decode(png_bytes_or_url[len(prefix) :])
        else:
            raw = base64.b64decode(png_bytes_or_url)
    else:
        raw = png_bytes_or_url
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Multi-Format Geometry Decompilers (.bbmodel, Java .json, Bedrock .geo.json)
# ---------------------------------------------------------------------------


def _decompile_bbmodel(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Decompile Blockbench .bbmodel JSON -> validated Program + face UVs + max vertex error."""
    used_ids: set[str] = set()
    res = doc.get("resolution", {"width": 16, "height": 16})
    tex_w = max(1, int(res.get("width", 16)))
    tex_h = max(1, int(res.get("height", 16)))

    embedded_rgba: np.ndarray | None = None
    textures = doc.get("textures") or []
    if textures and isinstance(textures[0], dict) and str(textures[0].get("source", "")).startswith("data:image/png;base64,"):
        embedded_rgba = _decode_png_rgba(textures[0]["source"])
        tex_h, tex_w = embedded_rgba.shape[:2]

    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("bbmodel contains no cube elements")

    elem_by_uuid = {str(e.get("uuid", idx)): e for idx, e in enumerate(elements)}
    groups = doc.get("groups") or []
    group_by_uuid = {str(g.get("uuid", "")): g for g in groups if isinstance(g, dict)}
    outliner = doc.get("outliner") or []

    uv_map: dict[str, dict[str, list[float]]] = {}
    palettes = [Palette("mat_0", (0.62, 0.06, 220.0), 6)]
    orig_centers_world: dict[str, np.ndarray] = {}

    def _box_from_elem(elem: dict[str, Any], bone_origin: np.ndarray, first_box_name: str | None, first_center_loc: np.ndarray | None) -> tuple[Box, np.ndarray]:
        bname = _sanitize_ident(elem.get("name", "cube"), "box", used_ids)
        lo = np.asarray(elem["from"], dtype=np.float64)
        hi = np.asarray(elem["to"], dtype=np.float64)
        span = np.maximum(1.0, np.rint(hi - lo))
        size_i = (int(span[0]), int(span[1]), int(span[2]))
        ctr_unrot = 0.5 * (lo + hi)
        loc_ctr = ctr_unrot - bone_origin
        inflate = max(0.0, float(elem.get("inflate", 0.0)))

        # Store face UVs in pixel coordinates [u0, v0, u1, v1]
        f_dict = elem.get("faces") or {}
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
                uv_map[bname][f] = [float(x) for x in f_dict[f]["uv"]]
            else:
                wu, hv = face_uv_extents_units(size_i, f)
                uv_map[bname][f] = [0.0, 0.0, float(min(tex_w, wu)), float(min(tex_h, hv))]

        if first_box_name is None:
            place = Place(base_box=None, anchor=(0.0, 0.0, 0.0), align=(0.0, 0.0, 0.0), offset=(float(loc_ctr[0]), float(loc_ctr[1]), float(loc_ctr[2])))
        else:
            assert first_center_loc is not None
            diff = loc_ctr - first_center_loc
            place = Place(base_box=first_box_name, anchor=(0.0, 0.0, 0.0), align=(0.0, 0.0, 0.0), offset=(float(diff[0]), float(diff[1]), float(diff[2])))
        return Box(bname, size_i, "mat_0", place=place, inflate=inflate), loc_ctr

    # Reconstruct hierarchy from outliner if groups exist; otherwise single root bone
    def _build_bone_from_node(
        node: dict[str, Any],
        parent_origin: np.ndarray,
        parent_first_box: str | None,
        parent_first_box_loc: np.ndarray | None,
        is_root: bool,
    ) -> Bone | None:
        grp = group_by_uuid.get(str(node.get("uuid", "")))
        if grp is None:
            return None
        g_origin = np.asarray(grp.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        g_rot = tuple(float(x) for x in grp.get("rotation", [0.0, 0.0, 0.0]))
        bone_name = _sanitize_ident(grp.get("name", "bone"), "bone", used_ids)

        child_elems = [elem_by_uuid[c] for c in node.get("children", []) if isinstance(c, str) and c in elem_by_uuid]
        child_nodes = [c for c in node.get("children", []) if isinstance(c, dict)]

        items: list[Box | Bone] = []
        first_bname: str | None = None
        first_bloc: np.ndarray | None = None

        if not child_elems:
            # Synthesize a tiny 1x1x1 hub box if a group only holds child groups
            hub_name = _sanitize_ident(f"{bone_name}_hub", "hub", used_ids)
            items.append(Box(hub_name, (1, 1, 1), "mat_0"))
            uv_map[hub_name] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
            first_bname = hub_name
            first_bloc = np.zeros(3, dtype=np.float64)
        else:
            for idx_e, el in enumerate(child_elems):
                bx, loc_c = _box_from_elem(el, g_origin, first_bname, first_bloc)
                if idx_e == 0:
                    first_bname = bx.name
                    first_bloc = loc_c
                items.append(bx)

        for cn in child_nodes:
            cb = _build_bone_from_node(cn, g_origin, first_bname, first_bloc, is_root=False)
            if cb is not None:
                items.append(cb)

        if is_root:
            if any(abs(a) > 1e-8 for a in g_rot):
                # Root bone in Block-DSL has zero rest_euler; wrap rotated root under synthetic world_root
                wrap_root_name = _sanitize_ident("world_root", "root", used_ids)
                hub_root = _sanitize_ident("root_anchor", "box", used_ids)
                uv_map[hub_root] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
                att = Attach(parent_box=hub_root, anchor=(0.0, 0.0, 0.0), align=(0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0), detached=True)
                inner_bone = Bone(name=bone_name, rest_euler=g_rot, attach=att, items=items)
                return Bone(name=wrap_root_name, pivot_local=(float(g_origin[0]), float(g_origin[1]), float(g_origin[2])), items=[Box(hub_root, (1, 1, 1), "mat_0"), inner_bone])
            return Bone(name=bone_name, pivot_local=(float(g_origin[0]), float(g_origin[1]), float(g_origin[2])), items=items)

        assert parent_first_box is not None and parent_first_box_loc is not None
        rel_trans = g_origin - parent_origin - parent_first_box_loc
        # Use detached=True when joint pivot offset exceeds 0.5u from parent box center
        needs_detached = any(abs(float(x)) > 0.5 for x in rel_trans)
        att = Attach(
            parent_box=parent_first_box,
            anchor=(0.0, 0.0, 0.0),
            align=(0.0, 0.0, 0.0),
            offset=(float(rel_trans[0]), float(rel_trans[1]), float(rel_trans[2])),
            detached=needs_detached,
        )
        return Bone(name=bone_name, rest_euler=g_rot, attach=att, items=items)

    dict_roots = [n for n in outliner if isinstance(n, dict) and str(n.get("uuid", "")) in group_by_uuid]
    if len(dict_roots) == 1:
        root_bone = _build_bone_from_node(dict_roots[0], np.zeros(3), None, None, is_root=True)
        assert root_bone is not None
    else:
        # Flat element list fallback
        root_name = _sanitize_ident("imported_root", "bone", used_ids)
        b_items: list[Box | Bone] = []
        fb_name: str | None = None
        fb_loc: np.ndarray | None = None
        for idx_e, el in enumerate(elements):
            bx, loc_c = _box_from_elem(el, np.zeros(3), fb_name, fb_loc)
            orig_centers_world[bx.name] = loc_c
            if idx_e == 0:
                fb_name, fb_loc = bx.name, loc_c
            b_items.append(bx)
        root_bone = Bone(name=root_name, pivot_local=(0.0, 0.0, 0.0), items=b_items)

    prog = validate_program(Program(Meta("block_multipart", "bbmodel_5", 16), palettes, root_bone))
    compiled = compile_geometry(prog)
    max_err = 0.0
    if orig_centers_world:
        for b in compiled.boxes:
            if b.name in orig_centers_world:
                max_err = max(max_err, float(np.linalg.norm(b.center - orig_centers_world[b.name])))
    return prog, uv_map, embedded_rgba, max_err


def _decompile_java_json(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], float]:
    """Decompile Minecraft Java block/item .json -> validated Program + face UVs + max vertex error."""
    used_ids: set[str] = set()
    tex_size = doc.get("texture_size") or [16, 16]
    tex_w, tex_h = max(1, int(tex_size[0])), max(1, int(tex_size[1]))
    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("Java model JSON contains no elements")

    uv_map: dict[str, dict[str, list[float]]] = {}
    root_name = _sanitize_ident("java_root", "bone", used_ids)
    items: list[Box | Bone] = []
    first_bname: str | None = None
    first_ctr: np.ndarray | None = None
    expected_centers: dict[str, np.ndarray] = {}

    for idx, el in enumerate(elements):
        lo = np.asarray(el["from"], dtype=np.float64)
        hi = np.asarray(el["to"], dtype=np.float64)
        span = np.maximum(1.0, np.rint(hi - lo))
        size_i = (int(span[0]), int(span[1]), int(span[2]))
        ctr = 0.5 * (lo + hi)
        bname = _sanitize_ident(el.get("name", f"elem_{idx}"), "box", used_ids)

        f_dict = el.get("faces") or {}
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
                # Java UVs are normalized to [0..16]; scale to pixel [0..tex_w, 0..tex_h]
                u0, v0, u1, v1 = (float(x) for x in f_dict[f]["uv"])
                uv_map[bname][f] = [u0 * tex_w / 16.0, v0 * tex_h / 16.0, u1 * tex_w / 16.0, v1 * tex_h / 16.0]
            else:
                uv_map[bname][f] = [0.0, 0.0, float(size_i[0]), float(size_i[1])]

        rot_data = el.get("rotation")
        has_rot = False
        rot_euler = (0.0, 0.0, 0.0)
        rot_origin = ctr
        if isinstance(rot_data, dict):
            rot_origin = np.asarray(rot_data.get("origin", ctr.tolist()), dtype=np.float64)
            if "axis" in rot_data:
                ax = str(rot_data["axis"])
                ang = float(rot_data.get("angle", 0.0))
                rot_euler = (ang if ax == "x" else 0.0, ang if ax == "y" else 0.0, ang if ax == "z" else 0.0)
            else:
                rot_euler = (float(rot_data.get("x", 0.0)), float(rot_data.get("y", 0.0)), float(rot_data.get("z", 0.0)))
            has_rot = any(abs(a) > 1e-8 for a in rot_euler)

        if not has_rot or first_bname is None:
            if first_bname is None:
                place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(ctr[0]), float(ctr[1]), float(ctr[2])))
                first_bname = bname
                first_ctr = ctr
            else:
                assert first_ctr is not None
                diff = ctr - first_ctr
                place = Place(first_bname, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(diff[0]), float(diff[1]), float(diff[2])))
            items.append(Box(bname, size_i, "mat_0", place=place))
            expected_centers[bname] = ctr
        else:
            assert first_ctr is not None
            bone_name = _sanitize_ident(f"rot_{bname}", "bone", used_ids)
            rel_piv = rot_origin - first_ctr
            box_loc = ctr - rot_origin
            att = Attach(first_bname, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(rel_piv[0]), float(rel_piv[1]), float(rel_piv[2])), detached=True)
            b_place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(box_loc[0]), float(box_loc[1]), float(box_loc[2])))
            items.append(Bone(bone_name, rest_euler=rot_euler, attach=att, items=[Box(bname, size_i, "mat_0", place=b_place)]))

    prog = validate_program(
        Program(Meta("weapon_3d", "java_modern", 16), [Palette("mat_0", (0.62, 0.06, 220.0), 6)], Bone(root_name, pivot_local=(0.0, 0.0, 0.0), items=items))
    )
    compiled = compile_geometry(prog)
    max_err = max(
        (float(np.linalg.norm(b.center - expected_centers[b.name])) for b in compiled.boxes if b.name in expected_centers),
        default=0.0,
    )
    return prog, uv_map, max_err


def _infer_locator_role(raw_loc_name: str) -> str:
    low = raw_loc_name.lower()
    if "seat" in low:
        return "rider_seat"
    if any(tok in low for tok in ("eye", "head", "camera", "look")):
        return "camera_focus"
    if any(tok in low for tok in ("target", "hitbox", "hurtbox")):
        return "hitbox"
    if any(tok in low for tok in ("foot", "paw", "talon", "hand", "claw", "ik")):
        return "ik_effector"
    if any(tok in low for tok in ("effect", "special", "breath", "muzzle", "flame", "beam", "vfx")):
        return "vfx_emitter"
    return "socket"


def _decompile_bedrock_geojson(
    doc: dict[str, Any],
) -> tuple[Program, dict[str, dict[str, list[float]]], float, dict[str, str]]:
    """Decompile Minecraft Bedrock / Cobblemon .geo.json -> validated Program + UVs + error + bone_map."""
    used_ids: set[str] = set()
    geos = doc.get("minecraft:geometry") or []
    if not geos or not isinstance(geos[0], dict):
        raise ValueError("Bedrock .geo.json missing 'minecraft:geometry' list")
    g0 = geos[0]
    desc = g0.get("description") or {}
    tex_w_desc = max(16, int(desc.get("texture_width", 64)))
    density = 16 if tex_w_desc <= 128 else 16
    bones_raw = g0.get("bones") or []
    if not bones_raw:
        raise ValueError("Bedrock geometry contains no bones")

    uv_map: dict[str, dict[str, list[float]]] = {}
    bone_records: dict[str, dict[str, Any]] = {}
    children_map: dict[str | None, list[str]] = {None: []}
    raw_to_safe_bone: dict[str, str] = {}
    expected_box_centers: dict[str, np.ndarray] = {}

    total_cubes = 0
    for b in bones_raw:
        bname = str(b["name"])
        bone_records[bname] = b
        children_map.setdefault(bname, [])
        total_cubes += len(b.get("cubes") or [])
    for b in bones_raw:
        parent = b.get("parent")
        parent_key = str(parent) if parent in bone_records else None
        children_map[parent_key].append(str(b["name"]))

    need_fallback_root_box = total_cubes == 0

    def _emit_bone_locators(
        rec: dict[str, Any],
        piv: np.ndarray,
    ) -> list[Locator]:
        locs_raw = rec.get("locators")
        if not isinstance(locs_raw, dict):
            return []
        out_locs: list[Locator] = []
        for raw_lname, lspec in locs_raw.items():
            safe_lname = _sanitize_ident(f"loc_{raw_lname}", "locator", used_ids)
            role = _infer_locator_role(str(raw_lname))
            if isinstance(lspec, dict):
                pos_arr = np.asarray(lspec.get("offset", lspec.get("position", piv.tolist())), dtype=np.float64)
            else:
                pos_arr = np.asarray(lspec, dtype=np.float64)
            loc_local = pos_arr - piv
            out_locs.append(
                Locator(
                    name=safe_lname,
                    offset=(float(loc_local[0]), float(loc_local[1]), float(loc_local[2])),
                    role=role,
                )
            )
        return out_locs

    def _build_bedrock_bone(
        raw_name: str,
        parent_pivot: np.ndarray,
        parent_world_rot: np.ndarray,
        parent_first_anchor: str | None,
        parent_first_loc: np.ndarray | None,
        is_root: bool,
    ) -> Bone:
        rec = bone_records[raw_name]
        safe_bname = _sanitize_ident(raw_name, "bone", used_ids)
        raw_to_safe_bone[raw_name] = safe_bname
        piv = np.asarray(rec.get("pivot", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot = tuple(float(x) for x in rec.get("rotation", [0.0, 0.0, 0.0]))
        local_rot_mat = np.eye(3, dtype=np.float64) if is_root else euler_xyz_matrix(rot)
        world_rot = parent_world_rot @ local_rot_mat
        cubes = rec.get("cubes") or []

        items: list[Box | Locator | Bone] = []
        fb_name: str | None = None
        fb_loc: np.ndarray | None = None

        if cubes:
            for c_idx, cube in enumerate(cubes):
                origin = np.asarray(cube["origin"], dtype=np.float64)
                sz_raw = np.maximum(1.0, np.rint(np.asarray(cube["size"], dtype=np.float64)))
                sz_i = (int(sz_raw[0]), int(sz_raw[1]), int(sz_raw[2]))
                ctr = origin + 0.5 * sz_raw
                loc_c = ctr - piv
                inflate = max(0.0, float(cube.get("inflate", 0.0)))
                cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                expected_box_centers[cname] = piv + world_rot @ loc_c

                uv_raw = cube.get("uv", [0, 0])
                uv_map[cname] = {}
                if isinstance(uv_raw, dict):
                    for f in FACES:
                        if f in uv_raw and "uv" in uv_raw[f]:
                            u0, v0 = (float(x) for x in uv_raw[f]["uv"])
                            su, sv = (float(x) for x in uv_raw[f].get("uv_size", [sz_i[0], sz_i[1]]))
                            uv_map[cname][f] = [u0, v0, u0 + su, v0 + sv]
                        else:
                            wu, hv = face_uv_extents_units(sz_i, f)
                            uv_map[cname][f] = [0.0, 0.0, float(wu), float(hv)]
                else:
                    ub, vb = float(uv_raw[0]), float(uv_raw[1])
                    sx, sy, sz = sz_i
                    uv_map[cname] = {
                        "up": [ub + sz, vb, ub + sz + sx, vb + sz],
                        "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
                        "east": [ub, vb + sz, ub + sz, vb + sz + sy],
                        "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
                        "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
                        "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
                    }

                if fb_name is None:
                    pl = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(loc_c[0]), float(loc_c[1]), float(loc_c[2])))
                    fb_name, fb_loc = cname, loc_c
                else:
                    assert fb_loc is not None
                    diff = loc_c - fb_loc
                    pl = Place(fb_name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(diff[0]), float(diff[1]), float(diff[2])))
                items.append(Box(cname, sz_i, "mat_0", place=pl, inflate=inflate))

            items.extend(_emit_bone_locators(rec, piv))
        else:
            # Zero-cube structural/socket bone: emit Cobblemon locators or a dimensionless socket Locator!
            explicit_locs = _emit_bone_locators(rec, piv)
            if is_root and need_fallback_root_box:
                hub = _sanitize_ident(f"{safe_bname}_hub", "hub", used_ids)
                items.append(Box(hub, (1, 1, 1), "mat_0"))
                uv_map[hub] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
                fb_name, fb_loc = hub, np.zeros(3, dtype=np.float64)
                items.extend(explicit_locs)
            elif explicit_locs:
                items.extend(explicit_locs)
                first_loc = explicit_locs[0]
                fb_name = first_loc.name
                fb_loc = np.asarray(first_loc.offset, dtype=np.float64)
            else:
                piv_loc_name = _sanitize_ident(f"{safe_bname}_pivot", "locator", used_ids)
                items.append(Locator(name=piv_loc_name, offset=(0.0, 0.0, 0.0), role="socket"))
                fb_name, fb_loc = piv_loc_name, np.zeros(3, dtype=np.float64)

        for child_raw in children_map.get(raw_name, []):
            items.append(_build_bedrock_bone(child_raw, piv, world_rot, fb_name, fb_loc, is_root=False))

        if is_root:
            return Bone(safe_bname, pivot_local=(float(piv[0]), float(piv[1]), float(piv[2])), items=items)

        assert parent_first_anchor is not None and parent_first_loc is not None
        # Exact inverse parent rotation so world_pivot == piv even under rotated ancestor bones
        rel = parent_world_rot.T @ (piv - parent_pivot) - parent_first_loc
        # Attaching to a dimensionless Locator or beyond 0.5u requires detached=True
        parent_is_locator = parent_first_anchor not in uv_map
        att = Attach(
            parent_first_anchor,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (float(rel[0]), float(rel[1]), float(rel[2])),
            detached=(parent_is_locator or any(abs(float(x)) > 0.5 for x in rel)),
        )
        return Bone(safe_bname, rest_euler=rot, attach=att, items=items)

    top_bones = children_map[None]
    if len(top_bones) == 1 and not any(abs(float(x)) > 1e-8 for x in bone_records[top_bones[0]].get("rotation", [0, 0, 0])):
        root_bone = _build_bedrock_bone(top_bones[0], np.zeros(3), np.eye(3), None, None, is_root=True)
    else:
        syn_root = _sanitize_ident("bedrock_root", "bone", used_ids)
        if need_fallback_root_box:
            syn_anchor = _sanitize_ident("bedrock_hub", "box", used_ids)
            uv_map[syn_anchor] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
            r_items: list[Box | Locator | Bone] = [Box(syn_anchor, (1, 1, 1), "mat_0")]
        else:
            syn_anchor = _sanitize_ident("bedrock_origin", "locator", used_ids)
            r_items = [Locator(name=syn_anchor, offset=(0.0, 0.0, 0.0), role="socket")]
        for tb in top_bones:
            r_items.append(_build_bedrock_bone(tb, np.zeros(3), np.eye(3), syn_anchor, np.zeros(3), is_root=False))
        root_bone = Bone(syn_root, pivot_local=(0.0, 0.0, 0.0), items=r_items)

    prog = validate_program(Program(Meta("entity", "bedrock_entity", density), [Palette("mat_0", (0.62, 0.06, 220.0), 6)], root_bone))
    compiled = compile_geometry(prog)
    max_err = max(
        (float(np.linalg.norm(b.center - expected_box_centers[b.name])) for b in compiled.boxes if b.name in expected_box_centers),
        default=0.0,
    )
    return prog, uv_map, max_err, raw_to_safe_bone


_MOLANG_CALL_RE = re.compile(
    r"([+-]?\s*(?:\d+(?:\.\d+)?\s*\*\s*)?)math\.(sin|cos)\(([^()]{1,160})\)(?:\s*\*\s*([+-]?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_TRAILING_SHIFT_RE = re.compile(r"([+-]\s*\d+(?:\.\d+)?)\s*$")
_PLAIN_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _parse_molang_channel_zero_eval(
    expr: Any,
    bone_name: str,
) -> tuple[str, float, float, float] | None:
    """Zero-eval bounded parser translating Bedrock/Cobblemon Molang rotation channel to TrackDecl parameters."""
    if isinstance(expr, bool) or expr is None:
        return None
    if isinstance(expr, (int, float)):
        val = float(expr)
        if not math.isfinite(val) or abs(val) < 0.25:
            return None
        return ("hinge", 0.0, 0.0, float(np.clip(val, -180.0, 180.0)))
    if not isinstance(expr, str) or len(expr) > 256:
        return None
    text = expr.strip()
    if not text:
        return None
    try:
        const_val = float(text)
        if not math.isfinite(const_val) or abs(const_val) < 0.25:
            return None
        return ("hinge", 0.0, 0.0, float(np.clip(const_val, -180.0, 180.0)))
    except ValueError:
        pass

    m = _MOLANG_CALL_RE.search(text)
    if m is None:
        return None

    lead_str, trig_fn, inside_arg, trail_scale_str = m.groups()
    lead_clean = lead_str.replace(" ", "")
    if lead_clean in ("", "+"):
        scale_pre = 1.0
    elif lead_clean == "-":
        scale_pre = -1.0
    else:
        scale_pre = float(lead_clean.rstrip("*"))
    scale_post = float(trail_scale_str) if trail_scale_str is not None else 1.0
    amp = float(np.clip(scale_pre * scale_post, -180.0, 180.0))

    # Extract additive constant outside the matched trig term
    outside = (text[: m.start()] + " " + text[m.end() :]).strip()
    bias = 0.0
    for nm in _PLAIN_NUM_RE.findall(outside):
        bias += float(nm)
    bias = float(np.clip(bias, -180.0, 180.0))

    # Phase shift from inside-parens trailing degree offset + cos->sin +90 deg shift
    phase_deg = 90.0 if trig_fn.lower() == "cos" else 0.0
    sh_m = _TRAILING_SHIFT_RE.search(inside_arg)
    if sh_m is not None:
        # Ensure it's an additive phase term rather than the sole multiplier after '*'
        prefix_inside = inside_arg[: sh_m.start()].rstrip()
        if not prefix_inside.endswith("*"):
            phase_deg += float(sh_m.group(1).replace(" ", ""))
    phase_offset = (phase_deg / 360.0) % 1.0
    if phase_offset >= 0.999999:
        phase_offset = 0.0

    wf = "flap" if "wing" in bone_name.lower() else "wave"
    if abs(amp) < 0.1 and abs(bias) < 0.25:
        return None
    return (wf, round(amp, 4), round(phase_offset, 4), round(bias, 4))


def decompile_bedrock_animation_to_behavior(
    anim_doc: dict[str, Any],
    program: Program,
    *,
    raw_to_safe_bone: dict[str, str] | None = None,
) -> BehaviorDecl:
    """Convert Bedrock / Cobblemon .animation.json into a validated @behavior BehaviorDecl (zero eval)."""
    anims = anim_doc.get("animations")
    if not isinstance(anims, dict) or not anims:
        raise ValueError("Bedrock .animation.json missing non-empty 'animations' dictionary")

    compiled = compile_geometry(program)
    valid_bones = set(compiled.bone_by_name)
    bone_lookup: dict[str, str] = {b: b for b in valid_bones}
    bone_lookup.update({b.lower(): b for b in valid_bones})
    if raw_to_safe_bone:
        for raw_k, safe_v in raw_to_safe_bone.items():
            if safe_v in valid_bones:
                bone_lookup[raw_k] = safe_v
                bone_lookup[raw_k.lower()] = safe_v

    used_clip_ids: set[str] = set()
    clips: list[ClipDecl] = []

    for raw_clip_name, spec in anims.items():
        if not isinstance(spec, dict):
            continue
        short_name = str(raw_clip_name).split(".")[-1]
        clip_name = _sanitize_ident(short_name, "clip", used_clip_ids)
        loop_val = bool(spec.get("loop", True))
        dur = float(spec.get("animation_length", 1.5))
        dur = float(np.clip(dur if dur > 0 else 1.5, 0.1, 60.0))
        base_freq_hz = round(float(np.clip(1.0 / dur, 0.05, 20.0)), 4)

        bones_anim = spec.get("bones") or {}
        tracks: list[TrackDecl] = []
        if isinstance(bones_anim, dict):
            for raw_b, channels in bones_anim.items():
                target_b = bone_lookup.get(str(raw_b)) or bone_lookup.get(str(raw_b).lower())
                if target_b is None or target_b == program.root.name or not isinstance(channels, dict):
                    continue
                rot_ch = channels.get("rotation")
                if isinstance(rot_ch, (list, tuple)) and len(rot_ch) == 3:
                    for ax_idx, ax_name in enumerate(("x", "y", "z")):
                        parsed = _parse_molang_channel_zero_eval(rot_ch[ax_idx], target_b)
                        if parsed is not None:
                            wf, amp, ph, bs = parsed
                            tracks.append(
                                TrackDecl(
                                    bone=target_b,
                                    law=wf,
                                    axis=ax_name,
                                    amp_deg=float(np.clip(amp, -120.0, 120.0)),
                                    freq_hz=base_freq_hz,
                                    phase_deg=round(ph * 360.0, 2),
                                    bias_deg=float(np.clip(bs, -90.0, 90.0)),
                                )
                            )
                elif isinstance(rot_ch, dict) and rot_ch:
                    kf_rows: list[tuple[float, list[float]]] = []
                    for t_str, k_val in rot_ch.items():
                        try:
                            t_num = float(t_str)
                        except (TypeError, ValueError):
                            continue
                        vec_raw = k_val.get("post", k_val.get("lerp_mode")) if isinstance(k_val, dict) else k_val
                        if isinstance(vec_raw, (list, tuple)) and len(vec_raw) == 3:
                            try:
                                kf_rows.append((t_num, [float(vec_raw[0]), float(vec_raw[1]), float(vec_raw[2])]))
                            except (TypeError, ValueError):
                                continue
                    kf_rows.sort(key=lambda item: item[0])
                    if len(kf_rows) >= 2:
                        for ax_idx, ax_name in enumerate(("x", "y", "z")):
                            vals = [row[1][ax_idx] for row in kf_rows]
                            vmin, vmax = min(vals), max(vals)
                            span = vmax - vmin
                            if span < 0.25:
                                continue
                            v_start, v_end = vals[0], vals[-1]
                            if abs(v_start - vmin) < 0.2 * span and abs(v_end - vmin) < 0.2 * span:
                                wf = "hinge"
                                bs = float(np.clip(vmin, -90.0, 90.0))
                                amp = float(np.clip(span, -120.0, 120.0))
                            else:
                                wf = "flap" if "wing" in target_b.lower() else "wave"
                                bs = float(np.clip(0.5 * (vmax + vmin), -90.0, 90.0))
                                amp = float(np.clip(0.5 * span, -120.0, 120.0))
                            tracks.append(
                                TrackDecl(
                                    bone=target_b,
                                    law=wf,
                                    axis=ax_name,
                                    amp_deg=round(amp, 4),
                                    freq_hz=base_freq_hz,
                                    phase_deg=0.0,
                                    bias_deg=round(bs, 4),
                                )
                            )

        if not tracks:
            non_root_bones = sorted(valid_bones - {program.root.name})
            if not non_root_bones:
                continue
            tracks.append(
                TrackDecl(
                    bone=non_root_bones[0],
                    law="wave",
                    axis="y",
                    amp_deg=0.0,
                    freq_hz=base_freq_hz,
                )
            )

        clips.append(ClipDecl(name=clip_name, duration_s=round(dur, 4), loop=loop_val, tracks=tuple(tracks)))

    if not clips:
        raise ValueError("No valid animation clips found in .animation.json")
    return BehaviorDecl(clips=tuple(clips))


# ---------------------------------------------------------------------------
# Inverse Material Posterior & Stamp Override Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialHypothesis:
    """Ranked candidate MaterialSpec inferred from observed surface maps."""

    spec: MaterialSpec
    posterior_weight: float
    oklab_residual_rmse: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "posterior_weight": round(self.posterior_weight, 5),
            "oklab_residual_rmse": round(self.oklab_residual_rmse, 5),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IngestedAssetRecord:
    """Complete verified output of P1-03e / ADR 0006 real-asset inverse inference."""

    provenance: AssetProvenance
    format_kind: str
    program: Program
    vertex_reconstruction_error: float
    material_hypotheses: tuple[MaterialHypothesis, ...]
    best_material_program: MaterialProgram
    extracted_stamps: tuple[StampOverride, ...]
    identifiability_score: float
    has_labpbr_auxiliary_maps: bool
    license_track: str = "clean_room_permissive"
    commercial_weights_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        from synccraft.dsl.parser import format_program

        return {
            "asset_id": self.provenance.asset_id,
            "license_spdx": self.provenance.license_spdx,
            "license_track": self.license_track,
            "commercial_weights_eligible": self.commercial_weights_eligible,
            "format_kind": self.format_kind,
            "program_dsl": format_program(self.program),
            "vertex_reconstruction_error": self.vertex_reconstruction_error,
            "identifiability_score": round(self.identifiability_score, 4),
            "has_labpbr_auxiliary_maps": self.has_labpbr_auxiliary_maps,
            "extracted_stamps": [s.to_dict() for s in self.extracted_stamps],
            "material_hypotheses": [h.to_dict() for h in self.material_hypotheses],
        }


def infer_material_posterior(
    program: Program,
    uv_map: dict[str, dict[str, list[float]]],
    atlas_rgba: np.ndarray,
    *,
    labpbr_n_rgba: np.ndarray | None = None,
    labpbr_s_rgba: np.ndarray | None = None,
    top_k: int = 3,
    seed: int = 404,
) -> tuple[tuple[MaterialHypothesis, ...], tuple[StampOverride, ...], float]:
    """Infer top-K MaterialSpec hypotheses + localized StampOverrides from texture maps."""
    h_img, w_img = atlas_rgba.shape[:2]
    scene = compile_geometry(program)

    all_lab_pixels: list[np.ndarray] = []
    grad_u_vals: list[float] = []
    grad_v_vals: list[float] = []
    high_freq_vals: list[float] = []
    extracted_stamps: list[StampOverride] = []

    spec_s_g: list[np.ndarray] = []
    spec_s_r: list[np.ndarray] = []
    spec_s_b: list[np.ndarray] = []

    for box in scene.boxes:
        if box.name not in uv_map:
            continue
        for face in FACES:
            uv_rect = uv_map[box.name].get(face)
            if not uv_rect:
                continue
            x0 = int(np.clip(min(uv_rect[0], uv_rect[2]), 0, w_img - 1))
            x1 = int(np.clip(math.ceil(max(uv_rect[0], uv_rect[2])), x0 + 1, w_img))
            y0 = int(np.clip(min(uv_rect[1], uv_rect[3]), 0, h_img - 1))
            y1 = int(np.clip(math.ceil(max(uv_rect[1], uv_rect[3])), y0 + 1, h_img))

            crop_rgba = atlas_rgba[y0:y1, x0:x1]
            if crop_rgba.size == 0 or np.all(crop_rgba[..., 3] == 0):
                continue

            lin = srgb_u8_to_linear(crop_rgba[..., :3])
            lab = linear_srgb_to_oklab(lin)
            all_lab_pixels.append(lab.reshape(-1, 3))

            if lab.shape[1] >= 2:
                grad_u_vals.append(float(np.mean(np.abs(np.diff(lab[..., 0], axis=1)))))
            if lab.shape[0] >= 2:
                grad_v_vals.append(float(np.mean(np.abs(np.diff(lab[..., 0], axis=0)))))
            if lab.shape[0] >= 3 and lab.shape[1] >= 3:
                center_l = lab[1:-1, 1:-1, 0]
                lap_l = (
                    lab[:-2, 1:-1, 0] + lab[2:, 1:-1, 0] + lab[1:-1, :-2, 0] + lab[1:-1, 2:, 0] - 4.0 * center_l
                )
                high_freq_vals.append(float(np.mean(np.abs(lap_l))))

            # Detect localized high-contrast artist StampOverrides (eyes, emblems, warning stripes)
            if lab.shape[0] >= 3 and lab.shape[1] >= 3:
                med_lab = np.median(lab.reshape(-1, 3), axis=0)
                delta_e = np.linalg.norm(lab - med_lab[None, None, :], axis=-1)
                outlier = delta_e > 0.20
                outlier_frac = float(np.mean(outlier))
                if 0.02 <= outlier_frac <= 0.35:
                    rs, cs = np.where(outlier)
                    wu, hv = face_uv_extents_units(box.size, face)
                    u_min = float(cs.min()) / lab.shape[1] * wu
                    u_max = float(cs.max() + 1) / lab.shape[1] * wu
                    v_min = float(rs.min()) / lab.shape[0] * hv
                    v_max = float(rs.max() + 1) / lab.shape[0] * hv
                    mean_out_lin = lin[outlier].mean(axis=0)
                    out_lch = linear_srgb_to_oklch(mean_out_lin)
                    aspect = (u_max - u_min) / max(1e-6, v_max - v_min)
                    shape_guess = "stripe_u" if aspect > 2.2 else ("stripe_v" if aspect < 0.45 else "rect")
                    extracted_stamps.append(
                        StampOverride(
                            box_name=box.name,
                            face=face,
                            u_min=round(u_min, 4),
                            v_min=round(v_min, 4),
                            u_max=round(max(u_min + 0.1, u_max), 4),
                            v_max=round(max(v_min + 0.1, v_max), 4),
                            lch=(float(out_lch[0]), float(out_lch[1]), float(out_lch[2])),
                            shape=shape_guess,
                        )
                    )

            if labpbr_s_rgba is not None and labpbr_s_rgba.shape[:2] == (h_img, w_img):
                scrop = labpbr_s_rgba[y0:y1, x0:x1]
                spec_s_r.append(scrop[..., 0].ravel())
                spec_s_g.append(scrop[..., 1].ravel())
                spec_s_b.append(scrop[..., 2].ravel())

    if not all_lab_pixels:
        default_spec = MaterialSpec(name="mat_0", family="polymer_dielectric", seed=seed)
        return (MaterialHypothesis(default_spec, 1.0, 0.0, "empty_or_transparent_atlas_fallback"),), (), 0.15

    cat_lab = np.concatenate(all_lab_pixels, axis=0)
    q25 = np.percentile(cat_lab, 25, axis=0)
    q75 = np.percentile(cat_lab, 75, axis=0)
    med = np.median(cat_lab, axis=0)

    def _oklab_to_lch_tuple(lab_vec: np.ndarray) -> tuple[float, float, float]:
        L = float(np.clip(lab_vec[0], 0.04, 0.96))
        C = float(max(0.0, math.hypot(float(lab_vec[1]), float(lab_vec[2]))))
        H = float(math.degrees(math.atan2(float(lab_vec[2]), float(lab_vec[1]))) % 360.0)
        return (round(L, 4), round(C, 4), round(H, 2))

    dom_lch = _oklab_to_lch_tuple(q75)
    sec_lch = _oklab_to_lch_tuple(q25)
    med_lch = _oklab_to_lch_tuple(med)

    l_std = float(np.std(cat_lab[:, 0]))
    chroma_med = med_lch[1]
    hue_med = med_lch[2]
    gu = float(np.mean(grad_u_vals)) if grad_u_vals else 0.01
    gv = float(np.mean(grad_v_vals)) if grad_v_vals else 0.01
    aniso_ratio = max(gu, gv) / max(1e-4, min(gu, gv))
    hf_energy = float(np.mean(high_freq_vals)) if high_freq_vals else 0.01

    has_pbr = labpbr_s_rgba is not None or labpbr_n_rgba is not None
    obs_metal_frac = 0.0
    obs_rough = 0.45
    obs_sss_frac = 0.0
    obs_poros = 0.20

    if spec_s_g:
        g_all = np.concatenate(spec_s_g)
        r_all = np.concatenate(spec_s_r)
        b_all = np.concatenate(spec_s_b)
        obs_metal_frac = float(np.mean(g_all >= 230))
        obs_rough = float(np.clip(1.0 - np.mean(r_all) / 255.0, 0.04, 0.98))
        obs_sss_frac = float(np.mean(b_all >= 65))
        obs_poros = float(np.mean(np.where(b_all <= 64, b_all / 64.0, 0.1)))

    # Score all 10 physical material families against observed spectral/Oklab/LabPBR features
    scores: dict[str, tuple[float, str]] = {}
    for fam in sorted(MATERIAL_FAMILIES):
        log_score = 0.0
        notes: list[str] = []

        if fam == "brushed_metal":
            log_score += 2.2 * obs_metal_frac if spec_s_g else (0.9 if chroma_med < 0.04 else -0.8)
            if aniso_ratio > 1.35:
                log_score += 0.8
                notes.append("directional_groove_gradients")
        elif fam == "painted_metal":
            if spec_s_g:
                partial_metal = 1.0 if 0.05 < obs_metal_frac < 0.85 else (0.4 if obs_metal_frac >= 0.85 else -0.3)
                log_score += 1.8 * partial_metal
                notes.append("bimodal_dielectric_coat_conductor_f0")
            else:
                log_score += 0.75 if (chroma_med > 0.07 and l_std > 0.04) else 0.1
                notes.append("chromatic_coat_with_edge_contrast")
        elif fam == "solid_wood":
            warm_hue = 1.0 if 35.0 <= hue_med <= 90.0 and chroma_med >= 0.04 else -0.5
            log_score += 0.9 * warm_hue + (0.7 if aniso_ratio > 1.25 else 0.0)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("warm_orthotropic_ring_statistics")
        elif fam == "weathered_stone":
            low_chroma = 1.0 if chroma_med < 0.055 else -0.4
            log_score += 0.85 * low_chroma + (0.6 if obs_poros > 0.3 else 0.2)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("low_chroma_clastic_variance")
        elif fam == "patterned_skin":
            log_score += (1.1 if l_std > 0.08 else 0.1) + (1.4 * obs_sss_frac if spec_s_g else 0.3)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("bimodal_spot_stripe_contrast")
        elif fam == "keratin_scales":
            log_score += (0.95 if hf_energy > 0.06 else 0.0) + (0.6 * obs_sss_frac if spec_s_g else 0.2)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.2
            notes.append("periodic_cellular_laplacian")
        elif fam == "woven_fabric":
            iso_weave = 1.0 if (hf_energy > 0.05 and aniso_ratio < 1.35) else 0.0
            log_score += 0.85 * iso_weave + (0.5 if obs_poros > 0.4 else 0.1)
            if spec_s_g and obs_metal_frac > 0.1:
                log_score -= 2.5
            notes.append("bidirectional_yarn_frequency")
        elif fam == "polymer_dielectric":
            smooth_uniform = 1.0 if l_std < 0.06 else 0.1
            log_score += 0.8 * smooth_uniform
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.2
            notes.append("uniform_dielectric_reflectance")
        elif fam == "lichen_biofilm":
            green_hue = 1.0 if 100.0 <= hue_med <= 165.0 else -0.4
            log_score += 0.85 * green_hue + (0.4 if l_std > 0.06 else 0.0)
            if spec_s_g and obs_metal_frac > 0.1:
                log_score -= 2.5
            notes.append("chlorophyll_hue_cluster")
        elif fam == "thin_film_iridescence":
            multi_hue = float(np.std(cat_lab[:, 1]) + np.std(cat_lab[:, 2]))
            log_score += 1.3 if multi_hue > 0.10 else -0.3
            notes.append("high_chromatic_dispersion")

        scores[fam] = (log_score, ",".join(notes) if notes else "spectral_oklab_fit")

    ranked_fams = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)[: max(1, top_k)]
    temp = 0.65 if has_pbr else 1.15
    logits = np.array([sc / temp for _, (sc, _) in ranked_fams], dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    # Identifiability reflects entropy gap + availability of intrinsic normal/specular channels
    full_logits = np.array([sc / temp for sc, _ in scores.values()], dtype=np.float64)
    full_logits -= full_logits.max()
    full_p = np.exp(full_logits) / np.sum(np.exp(full_logits))
    norm_entropy = -float(np.sum(full_p * np.log(np.maximum(1e-12, full_p)))) / math.log(len(MATERIAL_FAMILIES))
    base_ident = 1.0 - norm_entropy
    identifiability = float(np.clip((0.55 + 0.42 * base_ident) if has_pbr else (0.28 + 0.40 * base_ident), 0.05, 0.98))

    mat_name = program.palettes[0].name
    hypotheses: list[MaterialHypothesis] = []
    for idx_h, (fam, (_, note)) in enumerate(ranked_fams):
        is_metal = fam in ("brushed_metal",)
        is_coated = fam == "painted_metal"
        cand_spec = MaterialSpec(
            name=mat_name,
            family=fam,
            substrate_lch=(0.74, 0.02, 240.0) if is_coated else dom_lch,
            secondary_lch=sec_lch,
            coating_lch=dom_lch if is_coated else med_lch,
            coating_thickness=0.55 if is_coated else 0.0,
            perceptual_roughness=round(obs_rough, 3),
            metalness=1.0 if is_metal else 0.0,
            substrate_metalness=1.0 if (is_metal or is_coated) else 0.0,
            porosity=round(obs_poros, 3),
            seed=seed + idx_h * 29,
        )
        # Residual RMSE proxy from unexplained Oklab spread
        resid = float(max(0.012, l_std * (0.45 + 0.18 * idx_h)))
        hypotheses.append(
            MaterialHypothesis(
                spec=cand_spec,
                posterior_weight=float(weights[idx_h]),
                oklab_residual_rmse=resid,
                rationale=note,
            )
        )

    return tuple(hypotheses), tuple(extracted_stamps), identifiability


def ingest_external_asset(
    payload: dict[str, Any],
    format_kind: str,
    provenance: AssetProvenance,
    *,
    atlas_png: bytes | None = None,
    labpbr_n_png: bytes | None = None,
    labpbr_s_png: bytes | None = None,
    animation_payload: dict[str, Any] | None = None,
    license_policy: str = "clean_room_permissive",
    top_k: int = 3,
    seed: int = 404,
) -> IngestedAssetRecord:
    """Gate provenance/license, decompile geometry (+ optional Bedrock/Cobblemon animation), and infer MaterialProgram."""
    resolved_track = verify_asset_provenance(provenance, license_policy=license_policy)

    embedded_rgba: np.ndarray | None = None
    raw_to_safe_bone: dict[str, str] | None = None
    if format_kind == "bbmodel":
        program, uv_map, embedded_rgba, vert_err = _decompile_bbmodel(payload)
    elif format_kind == "java_json":
        program, uv_map, vert_err = _decompile_java_json(payload)
    elif format_kind == "bedrock_geojson":
        program, uv_map, vert_err, raw_to_safe_bone = _decompile_bedrock_geojson(payload)
    else:
        raise ValueError(f"Unsupported external format_kind {format_kind!r}; expected 'bbmodel', 'java_json', or 'bedrock_geojson'")

    if animation_payload is not None:
        beh = decompile_bedrock_animation_to_behavior(
            animation_payload,
            program,
            raw_to_safe_bone=raw_to_safe_bone,
        )
        program = validate_program(replace(program, behavior=beh))

    if atlas_png is not None:
        atlas_rgba = _decode_png_rgba(atlas_png)
    elif embedded_rgba is not None:
        atlas_rgba = embedded_rgba
    else:
        atlas_rgba = np.full((16, 16, 4), (160, 160, 165, 255), dtype=np.uint8)

    n_rgba = _decode_png_rgba(labpbr_n_png) if labpbr_n_png is not None else None
    s_rgba = _decode_png_rgba(labpbr_s_png) if labpbr_s_png is not None else None

    hyps, stamps, ident_score = infer_material_posterior(
        program,
        uv_map,
        atlas_rgba,
        labpbr_n_rgba=n_rgba,
        labpbr_s_rgba=s_rgba,
        top_k=top_k,
        seed=seed,
    )
    best_mp = MaterialProgram(materials={hyps[0].spec.name: hyps[0].spec}, stamps=stamps)
    return IngestedAssetRecord(
        provenance=provenance,
        format_kind=format_kind,
        program=program,
        vertex_reconstruction_error=vert_err,
        material_hypotheses=hyps,
        best_material_program=best_mp,
        extracted_stamps=stamps,
        identifiability_score=ident_score,
        has_labpbr_auxiliary_maps=(n_rgba is not None or s_rgba is not None),
        license_track=resolved_track,
        commercial_weights_eligible=(resolved_track == "clean_room_permissive"),
    )
