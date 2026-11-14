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

_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)


def _obb_corners_world(center_w: np.ndarray, rot_w: np.ndarray, raw_size: np.ndarray) -> np.ndarray:
    """Compute canonical 8-corner world coordinates matching WorldBox.corners(include_inflate=False)."""
    return (_CORNER_SIGNS * (raw_size * 0.5)) @ rot_w.T + center_w


def _max_corner_reconstruction_error(
    compiled: Any,
    expected_corners: dict[str, np.ndarray],
) -> float:
    """Measure max L2 8-corner world vertex error across all source cubes."""
    if not expected_corners:
        return 0.0
    compiled_by_name = compiled.box_by_name
    max_err = 0.0
    for bname, exp_c in expected_corners.items():
        if bname not in compiled_by_name:
            return float("inf")
        comp_c = compiled_by_name[bname].corners(include_inflate=False)
        err = float(np.max(np.linalg.norm(comp_c - exp_c, axis=-1)))
        max_err = max(max_err, err)
    return max_err


def _decompile_bbmodel(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Decompile Blockbench .bbmodel JSON -> validated Program + face UVs + max 8-corner vertex error."""
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

    elem_uuid_order = [(str(e.get("uuid", idx)), e) for idx, e in enumerate(elements)]
    elem_by_uuid = dict(elem_uuid_order)
    groups = doc.get("groups") or []
    group_by_uuid = {str(g.get("uuid", "")): g for g in groups if isinstance(g, dict)}
    outliner = doc.get("outliner") or []

    referenced_elem_uuids: set[str] = set()

    def _collect_outliner_refs(items_list: list[Any]) -> None:
        for entry in items_list:
            if isinstance(entry, str) and entry in elem_by_uuid:
                referenced_elem_uuids.add(entry)
            elif isinstance(entry, dict):
                _collect_outliner_refs(entry.get("children") or [])

    _collect_outliner_refs(outliner)

    uv_map: dict[str, dict[str, list[float]]] = {}
    palettes = [Palette("mat_0", (0.62, 0.06, 220.0), 6)]
    expected_corners: dict[str, np.ndarray] = {}

    def _register_elem_uv(bname: str, elem: dict[str, Any], size_i: tuple[int, int, int]) -> None:
        f_dict = elem.get("faces") or {}
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
                uv_map[bname][f] = [float(x) for x in f_dict[f]["uv"]]
            else:
                wu, hv = face_uv_extents_units(size_i, f)
                uv_map[bname][f] = [0.0, 0.0, float(min(tex_w, wu)), float(min(tex_h, hv))]

    def _emit_elems_for_bone(
        bone_label: str,
        elem_list: list[dict[str, Any]],
        bone_origin: np.ndarray,
        bone_world_piv: np.ndarray,
        bone_world_rot: np.ndarray,
    ) -> tuple[list[Box | Locator | Bone], str, np.ndarray, bool]:
        direct_items: list[Box | Locator | Bone] = []
        rot_sub_bones: list[Bone] = []
        anchor_name: str | None = None
        anchor_loc: np.ndarray | None = None
        anchor_is_locator = False

        unrot_elems: list[dict[str, Any]] = []
        rot_elems: list[tuple[dict[str, Any], tuple[float, float, float]]] = []
        for el in elem_list:
            r_raw = el.get("rotation") or [0.0, 0.0, 0.0]
            el_rot = (float(r_raw[0]), float(r_raw[1]), float(r_raw[2]))
            if any(abs(a) > 1e-8 for a in el_rot):
                rot_elems.append((el, el_rot))
            else:
                unrot_elems.append(el)

        for el in unrot_elems:
            bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
            lo = np.asarray(el["from"], dtype=np.float64)
            hi = np.asarray(el["to"], dtype=np.float64)
            raw_span = np.maximum(1e-6, hi - lo)
            span = np.maximum(1.0, np.rint(hi - lo))
            size_i = (int(span[0]), int(span[1]), int(span[2]))
            ctr_unrot = 0.5 * (lo + hi)
            loc_ctr = ctr_unrot - bone_origin
            inflate = max(0.0, float(el.get("inflate", 0.0)))
            _register_elem_uv(bname, el, size_i)

            w_ctr = bone_world_piv + bone_world_rot @ loc_ctr
            expected_corners[bname] = _obb_corners_world(w_ctr, bone_world_rot, raw_span)

            if anchor_name is None:
                pl = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(loc_ctr[0]), float(loc_ctr[1]), float(loc_ctr[2])))
                anchor_name = bname
                anchor_loc = loc_ctr
            else:
                assert anchor_loc is not None
                diff = loc_ctr - anchor_loc
                pl = Place(anchor_name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(diff[0]), float(diff[1]), float(diff[2])))
            direct_items.append(Box(bname, size_i, "mat_0", place=pl, inflate=inflate))

        if anchor_name is None:
            piv_loc = _sanitize_ident(f"{bone_label}_pivot", "locator", used_ids)
            direct_items.append(Locator(name=piv_loc, offset=(0.0, 0.0, 0.0), role="socket"))
            anchor_name = piv_loc
            anchor_loc = np.zeros(3, dtype=np.float64)
            anchor_is_locator = True

        assert anchor_loc is not None
        for el, el_rot in rot_elems:
            bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
            lo = np.asarray(el["from"], dtype=np.float64)
            hi = np.asarray(el["to"], dtype=np.float64)
            raw_span = np.maximum(1e-6, hi - lo)
            span = np.maximum(1.0, np.rint(hi - lo))
            size_i = (int(span[0]), int(span[1]), int(span[2]))
            ctr_unrot = 0.5 * (lo + hi)
            el_origin = np.asarray(el.get("origin", ctr_unrot.tolist()), dtype=np.float64)
            inflate = max(0.0, float(el.get("inflate", 0.0)))
            _register_elem_uv(bname, el, size_i)

            el_piv_loc = el_origin - bone_origin
            box_rel = ctr_unrot - el_origin
            r_el = euler_xyz_matrix(el_rot)
            w_rot = bone_world_rot @ r_el
            w_ctr = bone_world_piv + bone_world_rot @ el_piv_loc + w_rot @ box_rel
            expected_corners[bname] = _obb_corners_world(w_ctr, w_rot, raw_span)

            rel_piv = el_piv_loc - anchor_loc
            sub_bname = _sanitize_ident(f"rot_{bname}", "bone", used_ids)
            att = Attach(
                parent_box=anchor_name,
                anchor=(0.0, 0.0, 0.0),
                align=(0.0, 0.0, 0.0),
                offset=(float(rel_piv[0]), float(rel_piv[1]), float(rel_piv[2])),
                detached=(anchor_is_locator or any(abs(float(x)) > 0.5 for x in rel_piv)),
            )
            b_place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(box_rel[0]), float(box_rel[1]), float(box_rel[2])))
            rot_sub_bones.append(
                Bone(sub_bname, rest_euler=el_rot, attach=att, items=[Box(bname, size_i, "mat_0", place=b_place, inflate=inflate)])
            )

        return [*direct_items, *rot_sub_bones], anchor_name, anchor_loc, anchor_is_locator

    def _build_bone_from_node(
        node: dict[str, Any],
        parent_origin: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
        parent_anchor_name: str | None,
        parent_anchor_loc: np.ndarray | None,
        parent_anchor_is_locator: bool,
        is_root: bool,
    ) -> Bone | None:
        grp = group_by_uuid.get(str(node.get("uuid", "")))
        if grp is None:
            return None
        g_origin = np.asarray(grp.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        g_rot = tuple(float(x) for x in grp.get("rotation", [0.0, 0.0, 0.0]))
        bone_name = _sanitize_ident(grp.get("name", "bone"), "bone", used_ids)

        local_trans = g_origin - parent_origin
        if is_root:
            world_piv = g_origin.copy()
            world_rot = np.eye(3, dtype=np.float64)
        else:
            world_piv = parent_world_piv + parent_world_rot @ local_trans
            world_rot = parent_world_rot @ euler_xyz_matrix(g_rot)

        child_elems = [elem_by_uuid[c] for c in node.get("children", []) if isinstance(c, str) and c in elem_by_uuid]
        child_nodes = [c for c in node.get("children", []) if isinstance(c, dict)]

        items, first_bname, first_bloc, first_is_loc = _emit_elems_for_bone(
            bone_name, child_elems, g_origin, world_piv, world_rot
        )

        for cn in child_nodes:
            cb = _build_bone_from_node(
                cn, g_origin, world_piv, world_rot, first_bname, first_bloc, first_is_loc, is_root=False
            )
            if cb is not None:
                items.append(cb)

        if is_root:
            return Bone(name=bone_name, pivot_local=(float(g_origin[0]), float(g_origin[1]), float(g_origin[2])), items=items)

        assert parent_anchor_name is not None and parent_anchor_loc is not None
        rel_trans = local_trans - parent_anchor_loc
        needs_detached = parent_anchor_is_locator or any(abs(float(x)) > 0.5 for x in rel_trans)
        att = Attach(
            parent_box=parent_anchor_name,
            anchor=(0.0, 0.0, 0.0),
            align=(0.0, 0.0, 0.0),
            offset=(float(rel_trans[0]), float(rel_trans[1]), float(rel_trans[2])),
            detached=needs_detached,
        )
        return Bone(name=bone_name, rest_euler=g_rot, attach=att, items=items)

    dict_roots = [n for n in outliner if isinstance(n, dict) and str(n.get("uuid", "")) in group_by_uuid]
    loose_elems = [elem_by_uuid[u] for u in outliner if isinstance(u, str) and u in elem_by_uuid]
    for u_str, el_obj in elem_uuid_order:
        if u_str not in referenced_elem_uuids:
            loose_elems.append(el_obj)

    if len(dict_roots) == 1 and not loose_elems:
        g0 = group_by_uuid[str(dict_roots[0]["uuid"])]
        g0_rot = tuple(float(x) for x in g0.get("rotation", [0.0, 0.0, 0.0]))
        if not any(abs(a) > 1e-8 for a in g0_rot):
            root_bone = _build_bone_from_node(
                dict_roots[0], np.zeros(3), np.zeros(3), np.eye(3), None, None, False, is_root=True
            )
            assert root_bone is not None
        else:
            wrap_name = _sanitize_ident("world_root", "root", used_ids)
            r_items, anc_name, anc_loc, anc_is_loc = _emit_elems_for_bone(
                wrap_name, [], np.zeros(3), np.zeros(3), np.eye(3)
            )
            cb0 = _build_bone_from_node(
                dict_roots[0], np.zeros(3), np.zeros(3), np.eye(3), anc_name, anc_loc, anc_is_loc, is_root=False
            )
            if cb0 is not None:
                r_items.append(cb0)
            root_bone = Bone(name=wrap_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)
    elif len(dict_roots) >= 1:
        wrap_name = _sanitize_ident("world_root", "root", used_ids)
        r_items, anc_name, anc_loc, anc_is_loc = _emit_elems_for_bone(
            wrap_name, loose_elems, np.zeros(3), np.zeros(3), np.eye(3)
        )
        for rn in dict_roots:
            cb = _build_bone_from_node(
                rn, np.zeros(3), np.zeros(3), np.eye(3), anc_name, anc_loc, anc_is_loc, is_root=False
            )
            if cb is not None:
                r_items.append(cb)
        root_bone = Bone(name=wrap_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)
    else:
        root_name = _sanitize_ident("imported_root", "bone", used_ids)
        r_items, _, _, _ = _emit_elems_for_bone(
            root_name, loose_elems, np.zeros(3), np.zeros(3), np.eye(3)
        )
        root_bone = Bone(name=root_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)

    prog = validate_program(Program(Meta("block_multipart", "bbmodel_5", 16), palettes, root_bone))
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
    return prog, uv_map, embedded_rgba, max_err


def _decompile_java_json(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], float]:
    """Decompile Minecraft Java block/item .json -> validated Program + face UVs + max 8-corner vertex error."""
    used_ids: set[str] = set()
    tex_size = doc.get("texture_size") or [16, 16]
    tex_w, tex_h = max(1, int(tex_size[0])), max(1, int(tex_size[1]))
    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("Java model JSON contains no elements")

    uv_map: dict[str, dict[str, list[float]]] = {}
    root_name = _sanitize_ident("java_root", "bone", used_ids)
    items: list[Box | Locator | Bone] = []
    first_bname: str | None = None
    first_ctr: np.ndarray | None = None
    first_is_locator = False
    expected_corners: dict[str, np.ndarray] = {}

    for idx, el in enumerate(elements):
        lo = np.asarray(el["from"], dtype=np.float64)
        hi = np.asarray(el["to"], dtype=np.float64)
        raw_span = np.maximum(1e-6, hi - lo)
        span = np.maximum(1.0, np.rint(hi - lo))
        size_i = (int(span[0]), int(span[1]), int(span[2]))
        ctr = 0.5 * (lo + hi)
        bname = _sanitize_ident(el.get("name", f"elem_{idx}"), "box", used_ids)

        f_dict = el.get("faces") or {}
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
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

        if not has_rot:
            if first_bname is None or first_is_locator:
                place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(ctr[0]), float(ctr[1]), float(ctr[2])))
                if first_bname is None:
                    first_bname = bname
                    first_ctr = ctr
                    first_is_locator = False
            else:
                assert first_ctr is not None
                diff = ctr - first_ctr
                place = Place(first_bname, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(diff[0]), float(diff[1]), float(diff[2])))
            # Place direct boxes before any rotated child bones for canonical AST ordering
            box_node = Box(bname, size_i, "mat_0", place=place)
            bone_split = [x for x in items if isinstance(x, Bone)]
            non_bone_split = [x for x in items if not isinstance(x, Bone)]
            items = [*non_bone_split, box_node, *bone_split]
            expected_corners[bname] = _obb_corners_world(ctr, np.eye(3, dtype=np.float64), raw_span)
        else:
            if first_bname is None:
                loc_anchor = _sanitize_ident("java_origin", "locator", used_ids)
                items.append(Locator(name=loc_anchor, offset=(0.0, 0.0, 0.0), role="socket"))
                first_bname = loc_anchor
                first_ctr = np.zeros(3, dtype=np.float64)
                first_is_locator = True
            assert first_ctr is not None
            r_el = euler_xyz_matrix(rot_euler)
            box_loc = ctr - rot_origin
            w_ctr = rot_origin + r_el @ box_loc
            expected_corners[bname] = _obb_corners_world(w_ctr, r_el, raw_span)

            bone_name = _sanitize_ident(f"rot_{bname}", "bone", used_ids)
            rel_piv = rot_origin - first_ctr
            att = Attach(
                first_bname,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (float(rel_piv[0]), float(rel_piv[1]), float(rel_piv[2])),
                detached=True,
            )
            b_place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(box_loc[0]), float(box_loc[1]), float(box_loc[2])))
            items.append(Bone(bone_name, rest_euler=rot_euler, attach=att, items=[Box(bname, size_i, "mat_0", place=b_place)]))

    prog = validate_program(
        Program(Meta("weapon_3d", "java_modern", 16), [Palette("mat_0", (0.62, 0.06, 220.0), 6)], Bone(root_name, pivot_local=(0.0, 0.0, 0.0), items=items))
    )
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
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
    """Decompile Minecraft Bedrock / Cobblemon .geo.json -> validated Program + UVs + 8-corner error + bone_map."""
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
    expected_corners: dict[str, np.ndarray] = {}

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

    def _register_bedrock_cube_uv(cname: str, cube: dict[str, Any], sz_i: tuple[int, int, int]) -> None:
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
        parent_world_piv: np.ndarray,
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
        local_trans = piv - parent_pivot
        if is_root:
            world_piv = piv.copy()
            world_rot = np.eye(3, dtype=np.float64)
        else:
            world_piv = parent_world_piv + parent_world_rot @ local_trans
            world_rot = parent_world_rot @ euler_xyz_matrix(rot)
        cubes = rec.get("cubes") or []

        direct_items: list[Box | Locator] = []
        rot_cube_bones: list[Bone] = []
        fb_name: str | None = None
        fb_loc: np.ndarray | None = None

        if cubes:
            unrot_cubes: list[tuple[int, dict[str, Any]]] = []
            rot_cubes: list[tuple[int, dict[str, Any], tuple[float, float, float]]] = []
            for c_idx, cube in enumerate(cubes):
                c_r = cube.get("rotation") or [0.0, 0.0, 0.0]
                c_rot = (float(c_r[0]), float(c_r[1]), float(c_r[2]))
                if any(abs(a) > 1e-8 for a in c_rot):
                    rot_cubes.append((c_idx, cube, c_rot))
                else:
                    unrot_cubes.append((c_idx, cube))

            for c_idx, cube in unrot_cubes:
                origin = np.asarray(cube["origin"], dtype=np.float64)
                raw_sz = np.maximum(1e-6, np.asarray(cube["size"], dtype=np.float64))
                sz_rnd = np.maximum(1.0, np.rint(raw_sz))
                sz_i = (int(sz_rnd[0]), int(sz_rnd[1]), int(sz_rnd[2]))
                ctr = origin + 0.5 * raw_sz
                loc_c = ctr - piv
                inflate = max(0.0, float(cube.get("inflate", 0.0)))
                cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                _register_bedrock_cube_uv(cname, cube, sz_i)

                w_ctr = world_piv + world_rot @ loc_c
                expected_corners[cname] = _obb_corners_world(w_ctr, world_rot, raw_sz)

                if fb_name is None:
                    pl = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(loc_c[0]), float(loc_c[1]), float(loc_c[2])))
                    fb_name, fb_loc = cname, loc_c
                else:
                    assert fb_loc is not None
                    diff = loc_c - fb_loc
                    pl = Place(fb_name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(diff[0]), float(diff[1]), float(diff[2])))
                direct_items.append(Box(cname, sz_i, "mat_0", place=pl, inflate=inflate))

            explicit_locs = _emit_bone_locators(rec, piv)
            direct_items.extend(explicit_locs)

            if fb_name is None:
                if explicit_locs:
                    fb_name = explicit_locs[0].name
                    fb_loc = np.asarray(explicit_locs[0].offset, dtype=np.float64)
                else:
                    piv_loc_name = _sanitize_ident(f"{safe_bname}_pivot", "locator", used_ids)
                    direct_items.append(Locator(name=piv_loc_name, offset=(0.0, 0.0, 0.0), role="socket"))
                    fb_name, fb_loc = piv_loc_name, np.zeros(3, dtype=np.float64)

            assert fb_loc is not None
            fb_is_locator = fb_name not in uv_map
            for c_idx, cube, c_rot in rot_cubes:
                origin = np.asarray(cube["origin"], dtype=np.float64)
                raw_sz = np.maximum(1e-6, np.asarray(cube["size"], dtype=np.float64))
                sz_rnd = np.maximum(1.0, np.rint(raw_sz))
                sz_i = (int(sz_rnd[0]), int(sz_rnd[1]), int(sz_rnd[2]))
                ctr = origin + 0.5 * raw_sz
                c_piv = np.asarray(cube.get("pivot", ctr.tolist()), dtype=np.float64)
                inflate = max(0.0, float(cube.get("inflate", 0.0)))
                cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                _register_bedrock_cube_uv(cname, cube, sz_i)

                c_piv_loc = c_piv - piv
                c_ctr_rel = ctr - c_piv
                r_cube = euler_xyz_matrix(c_rot)
                w_rot_c = world_rot @ r_cube
                w_ctr_c = world_piv + world_rot @ c_piv_loc + w_rot_c @ c_ctr_rel
                expected_corners[cname] = _obb_corners_world(w_ctr_c, w_rot_c, raw_sz)

                rel_c = c_piv_loc - fb_loc
                cbone_name = _sanitize_ident(f"{safe_bname}_crot_{c_idx}", "bone", used_ids)
                att_c = Attach(
                    fb_name,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (float(rel_c[0]), float(rel_c[1]), float(rel_c[2])),
                    detached=(fb_is_locator or any(abs(float(x)) > 0.5 for x in rel_c)),
                )
                pl_c = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(c_ctr_rel[0]), float(c_ctr_rel[1]), float(c_ctr_rel[2])))
                rot_cube_bones.append(
                    Bone(cbone_name, rest_euler=c_rot, attach=att_c, items=[Box(cname, sz_i, "mat_0", place=pl_c, inflate=inflate)])
                )
        else:
            explicit_locs = _emit_bone_locators(rec, piv)
            if is_root and need_fallback_root_box:
                hub = _sanitize_ident(f"{safe_bname}_hub", "hub", used_ids)
                direct_items.append(Box(hub, (1, 1, 1), "mat_0"))
                uv_map[hub] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
                fb_name, fb_loc = hub, np.zeros(3, dtype=np.float64)
                direct_items.extend(explicit_locs)
            elif explicit_locs:
                direct_items.extend(explicit_locs)
                first_loc = explicit_locs[0]
                fb_name = first_loc.name
                fb_loc = np.asarray(first_loc.offset, dtype=np.float64)
            else:
                piv_loc_name = _sanitize_ident(f"{safe_bname}_pivot", "locator", used_ids)
                direct_items.append(Locator(name=piv_loc_name, offset=(0.0, 0.0, 0.0), role="socket"))
                fb_name, fb_loc = piv_loc_name, np.zeros(3, dtype=np.float64)

        items: list[Box | Locator | Bone] = [*direct_items, *rot_cube_bones]
        for child_raw in children_map.get(raw_name, []):
            items.append(
                _build_bedrock_bone(child_raw, piv, world_piv, world_rot, fb_name, fb_loc, is_root=False)
            )

        if is_root:
            return Bone(safe_bname, pivot_local=(float(piv[0]), float(piv[1]), float(piv[2])), items=items)

        assert parent_first_anchor is not None and parent_first_loc is not None
        rel = local_trans - parent_first_loc
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
        root_bone = _build_bedrock_bone(
            top_bones[0], np.zeros(3), np.zeros(3), np.eye(3), None, None, is_root=True
        )
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
            r_items.append(
                _build_bedrock_bone(
                    tb, np.zeros(3), np.zeros(3), np.eye(3), syn_anchor, np.zeros(3), is_root=False
                )
            )
        root_bone = Bone(syn_root, pivot_local=(0.0, 0.0, 0.0), items=r_items)

    prog = validate_program(Program(Meta("entity", "bedrock_entity", density), [Palette("mat_0", (0.62, 0.06, 220.0), 6)], root_bone))
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
    return prog, uv_map, max_err, raw_to_safe_bone


_MOLANG_TERM_RE = re.compile(
    r"([+-]?\s*(?:\d+(?:\.\d+)?\s*\*\s*)*)math\.(sin|cos)\(([^()]{1,160})\)((?:\s*[*/]\s*[+-]?\d+(?:\.\d+)?)*)",
    re.IGNORECASE,
)
_MUL_DIV_FACTOR_RE = re.compile(r"([*/])\s*([+-]?\d+(?:\.\d+)?)")
_ANIM_TIME_RE = re.compile(r"\b(?:q|query|v|variable)\.anim_time\b", re.IGNORECASE)


def _unwrap_molang_clamp(text: str) -> tuple[str, tuple[float, float] | None]:
    """Unwrap up to two nested math.clamp(expr, lo, hi) calls without eval."""
    cur = text.strip()
    bounds: tuple[float, float] | None = None
    for _ in range(2):
        if not (cur.lower().startswith("math.clamp(") and cur.endswith(")")):
            break
        inner = cur[11:-1]
        depth = 0
        commas: list[int] = []
        valid_parens = True
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    valid_parens = False
                    break
            elif ch == "," and depth == 0:
                commas.append(i)
        if not valid_parens or depth != 0 or len(commas) != 2:
            break
        arg0 = inner[: commas[0]].strip()
        arg1 = inner[commas[0] + 1 : commas[1]].strip().replace(" ", "")
        arg2 = inner[commas[1] + 1 :].strip().replace(" ", "")
        try:
            lo = float(arg1)
            hi = float(arg2)
        except ValueError:
            break
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            break
        cur = arg0
        bounds = (lo, hi) if bounds is None else (max(bounds[0], lo), min(bounds[1], hi))
    return cur, bounds


def _split_top_level_additive_terms(expr_str: str) -> list[str]:
    """Split expression at top-level '+' or '-' (not after '*' or '/' or inside parens)."""
    terms: list[str] = []
    buf: list[str] = []
    depth = 0
    prev_nonspace = ""
    for ch in expr_str:
        if ch == "(":
            depth += 1
            buf.append(ch)
            prev_nonspace = ch
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            prev_nonspace = ch
        elif depth == 0 and ch in "+-" and buf and prev_nonspace not in ("", "*", "/", "("):
            term = "".join(buf).strip()
            if term:
                terms.append(term)
            buf = [ch]
            prev_nonspace = ch
        else:
            buf.append(ch)
            if not ch.isspace():
                prev_nonspace = ch
    tail = "".join(buf).strip()
    if tail:
        terms.append(tail)
    return terms


def _sum_standalone_constant_terms(rem: str) -> float:
    """Sum only standalone top-level numeric literal terms, ignoring variable/harmonic subterms."""
    total = 0.0
    for term in _split_top_level_additive_terms(rem):
        compact = term.replace(" ", "")
        try:
            val = float(compact)
            if math.isfinite(val):
                total += val
        except ValueError:
            continue
    return total


def _parse_molang_inside_arg(inside_arg: str) -> tuple[float, float | None, float]:
    """Return (phase_shift_deg, freq_hz_override_or_None, sign_flip_for_sin) from trig argument."""
    phase_deg = 0.0
    freq_hz: float | None = None
    sin_sign = 1.0
    for term in _split_top_level_additive_terms(inside_arg):
        compact = term.replace(" ", "")
        try:
            val = float(compact)
            if math.isfinite(val):
                phase_deg += val
            continue
        except ValueError:
            pass
        if _ANIM_TIME_RE.search(term) and freq_hz is None:
            # Multiplicative angular velocity in deg/sec (Molang trig operates in degrees)
            cleaned = _ANIM_TIME_RE.sub("1.0", term)
            cleaned_low = cleaned.lower().replace("math.pi", "180.0")
            parts_mul = cleaned_low.split("*")
            speed = 1.0
            ok = True
            for part in parts_mul:
                sub_div = part.split("/")
                try:
                    num = float(sub_div[0].replace(" ", ""))
                    for denom_s in sub_div[1:]:
                        d_val = float(denom_s.replace(" ", ""))
                        num = num / d_val if abs(d_val) > 1e-9 else 0.0
                    speed *= num
                except ValueError:
                    ok = False
                    break
            if ok and math.isfinite(speed) and abs(speed) > 1e-6:
                if speed < 0.0:
                    sin_sign = -1.0
                freq_hz = round(float(np.clip(abs(speed) / 360.0, 0.05, 20.0)), 4)
    return phase_deg, freq_hz, sin_sign


def _parse_molang_channel_zero_eval(
    expr: Any,
    bone_name: str,
) -> tuple[str, float, float, float, float | None] | None:
    """Zero-eval bounded parser translating Bedrock/Cobblemon Molang rotation channel to TrackDecl parameters.

    Returns ``(law, amp_deg, phase_offset_unit, bias_deg, freq_hz_override)`` or ``None``.
    """
    if isinstance(expr, bool) or expr is None:
        return None
    if isinstance(expr, (int, float)):
        val = float(expr)
        if not math.isfinite(val) or abs(val) < 0.25:
            return None
        return ("hinge", 0.0, 0.0, float(np.clip(val, -180.0, 180.0)), None)
    if not isinstance(expr, str) or len(expr) > 256:
        return None
    raw_text = expr.strip()
    if not raw_text:
        return None

    text, clamp_bounds = _unwrap_molang_clamp(raw_text)
    try:
        const_val = float(text.replace(" ", ""))
        if not math.isfinite(const_val):
            return None
        if clamp_bounds is not None:
            const_val = float(np.clip(const_val, clamp_bounds[0], clamp_bounds[1]))
        if abs(const_val) < 0.25:
            return None
        return ("hinge", 0.0, 0.0, float(np.clip(const_val, -180.0, 180.0)), None)
    except ValueError:
        pass

    matches = list(_MOLANG_TERM_RE.finditer(text))
    if not matches:
        return None

    m0 = matches[0]
    lead_str, trig_fn, inside_arg, trail_str = m0.groups()
    lead_clean = lead_str.replace(" ", "")
    if lead_clean in ("", "+"):
        scale_pre = 1.0
    elif lead_clean == "-":
        scale_pre = -1.0
    else:
        scale_pre = 1.0
        for tok in lead_clean.rstrip("*").split("*"):
            if tok in ("", "+"):
                continue
            if tok == "-":
                scale_pre *= -1.0
            else:
                scale_pre *= float(tok)

    scale_post = 1.0
    if trail_str:
        for op_ch, num_s in _MUL_DIV_FACTOR_RE.findall(trail_str):
            f_val = float(num_s)
            if op_ch == "*":
                scale_post *= f_val
            elif abs(f_val) > 1e-9:
                scale_post /= f_val

    arg_phase_deg, freq_override, sin_sign = _parse_molang_inside_arg(inside_arg)
    is_cos = trig_fn.lower() == "cos"
    amp_raw = scale_pre * scale_post * (1.0 if is_cos else sin_sign)
    amp = float(np.clip(amp_raw, -180.0, 180.0))

    # Strip ALL matched trig terms before summing standalone constant bias terms
    rem_chunks: list[str] = []
    cursor = 0
    for mt in matches:
        rem_chunks.append(text[cursor : mt.start()])
        cursor = mt.end()
    rem_chunks.append(text[cursor:])
    remainder = " ".join(rem_chunks).strip()
    bias = float(np.clip(_sum_standalone_constant_terms(remainder), -180.0, 180.0))

    if clamp_bounds is not None:
        c_lo, c_hi = clamp_bounds
        lo_eff = float(np.clip(bias - abs(amp), c_lo, c_hi))
        hi_eff = float(np.clip(bias + abs(amp), c_lo, c_hi))
        bias = 0.5 * (lo_eff + hi_eff)
        half_span = 0.5 * (hi_eff - lo_eff)
        amp = math.copysign(half_span, amp if abs(amp) > 1e-9 else 1.0)

    phase_deg = (90.0 if is_cos else 0.0) + arg_phase_deg
    phase_offset = (phase_deg / 360.0) % 1.0
    if phase_offset >= 0.999999:
        phase_offset = 0.0

    wf = "flap" if "wing" in bone_name.lower() else "wave"
    if abs(amp) < 0.1 and abs(bias) < 0.25:
        return None
    return (wf, round(amp, 4), round(phase_offset, 4), round(bias, 4), freq_override)


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

    # Seed used_clip_ids with all existing Program/Scene identifiers to prevent global AST namespace collisions
    used_clip_ids: set[str] = (
        set(compiled.bone_by_name)
        | set(compiled.box_by_name)
        | set(compiled.locator_by_name)
        | {p.name for p in program.palettes}
        | {m.name for m in program.materials}
        | {m.name for m in program.muscles}
    )
    if program.kinematics is not None:
        used_clip_ids |= {c.name for c in program.kinematics.ik_chains}
        used_clip_ids |= {c.name for c in program.kinematics.look_chains}
    if program.behavior is not None:
        used_clip_ids |= {c.name for c in program.behavior.clips}
        used_clip_ids |= {s.name for s in program.behavior.states}

    clips: list[ClipDecl] = []

    for raw_clip_name, spec in anims.items():
        if not isinstance(spec, dict):
            continue
        short_name = str(raw_clip_name).split(".")[-1]
        cand_short = f"{short_name}_clip" if short_name in used_clip_ids else short_name
        clip_name = _sanitize_ident(cand_short, "clip", used_clip_ids)
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
                            wf, amp, ph, bs, freq_ov = parsed
                            tracks.append(
                                TrackDecl(
                                    bone=target_b,
                                    law=wf,
                                    axis=ax_name,
                                    amp_deg=float(np.clip(amp, -120.0, 120.0)),
                                    freq_hz=freq_ov if freq_ov is not None else base_freq_hz,
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
