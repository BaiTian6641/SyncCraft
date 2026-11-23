#!/usr/bin/env python3
"""
Engine-agnostic deterministic geometry, proportion & Vigorous Carved Blockbench style-grammar
audit (G1-G12, v20). Zero third-party dependencies (pure Python stdlib).

Natively supports:
  - Blockbench projects (.bbmodel, v4 & v5 outliner/groups schemas)
  - Minecraft Bedrock / GeckoLib geometry (.geo.json / .json)
  - SyncCraft Block-DSL (.blockdsl, with optional <SyncCraft_repo_path> compiler or standalone)

Usage:
  python3 construct_check.py [--tier auto|small|standard|hero|epic_boss] <model.{bbmodel|geo.json|blockdsl}>
  python3 construct_check.py [--tier auto|small|standard|hero|epic_boss] <SyncCraft_repo_path> <model.blockdsl>
"""
from __future__ import annotations
import collections
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]


@dataclass
class BoneNode:
    name: str
    parent_name: Optional[str]
    local_pos: Vec3
    local_rot: Vec3
    world_pos: Vec3
    world_rot: Mat3
    mirrored: bool


@dataclass
class BoxElem:
    name: str
    bone_name: str
    size: Vec3
    center: Vec3
    local_center: Vec3
    local_rot: Vec3
    material: str
    is_plane: bool = False
    plane_normal: Optional[str] = None


EYE3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _vec3(seq: Sequence[float]) -> Vec3:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _mirror_x_vec(v: Vec3) -> Vec3:
    return (-v[0], v[1], v[2])


def _mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def _mirror_x_mat(m: Mat3) -> Mat3:
    s = (-1.0, 1.0, 1.0)
    return tuple(
        tuple(s[i] * m[i][j] * s[j] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def rot_zyx(rx: float, ry: float, rz: float) -> Mat3:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx_m: Mat3 = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry_m: Mat3 = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz_m: Mat3 = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _mat_mul(_mat_mul(rz_m, ry_m), rx_m)


def _apply_bind_chain(pt: Vec3, chain: List[Tuple[Vec3, Mat3]]) -> Vec3:
    """Transform bind-space point through leaf-to-root (pivot, R_local) chain."""
    cur = pt
    for piv, rmat in reversed(chain):
        cur = _add(piv, _mat_vec(rmat, _sub(cur, piv)))
    return cur


def _min_singular_value_3d(pts: List[Vec3]) -> float:
    """Exact smallest singular value of centered N x 3 point matrix via 3x3 symmetric Jacobi."""
    n = len(pts)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    mz = sum(p[2] for p in pts) / n
    a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for x, y, z in pts:
        v = (x - mx, y - my, z - mz)
        for i in range(3):
            for j in range(3):
                a[i][j] += v[i] * v[j]
    for _ in range(30):
        p, q = max(
            ((0, 1), (0, 2), (1, 2)),
            key=lambda ij: abs(a[ij[0]][ij[1]]),
        )
        if abs(a[p][q]) < 1e-12:
            break
        diff = a[q][q] - a[p][p]
        tau = diff / (2.0 * a[p][q])
        t = (1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))) * (1.0 if tau >= 0.0 else -1.0)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for r in range(3):
            if r != p and r != q:
                arp, arq = a[r][p], a[r][q]
                a[r][p] = a[p][r] = c * arp - s * arq
                a[r][q] = a[q][r] = s * arp + c * arq
    min_eig = min(a[0][0], a[1][1], a[2][2])
    return math.sqrt(max(0.0, min_eig))


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def mirror_name(name: str) -> str:
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if "_l_" in name:
        return name.replace("_l_", "_r_")
    if name.startswith("left_"):
        return "right_" + name[5:]
    return name + "_r"


def _extract_rot(obj: object) -> Vec3:
    for attr in ("local_euler", "rotation_deg", "local_rot", "rot", "euler_deg", "rest_euler"):
        val = getattr(obj, attr, None)
        if val is not None and len(val) == 3:
            return _vec3(val)
    return (0.0, 0.0, 0.0)


def parse_bbmodel(path: str) -> Tuple[List[BoneNode], List[BoxElem], List[BoxElem], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    elem_by_uuid: Dict[str, Dict[str, Any]] = {}
    mesh_count = 0
    for el in data.get("elements", []):
        if el.get("type", "cube") == "mesh":
            mesh_count += 1
            continue
        uid = el.get("uuid")
        if uid:
            elem_by_uuid[uid] = el

    group_by_uuid: Dict[str, Dict[str, Any]] = {}
    for g in data.get("groups", []):
        if isinstance(g, dict) and g.get("uuid"):
            group_by_uuid[g["uuid"]] = g

    bones: List[BoneNode] = []
    boxes: List[BoxElem] = []
    planes: List[BoxElem] = []
    assigned_uuids: set[str] = set()

    def walk_node(
        node: Any,
        parent_name: Optional[str],
        parent_bind_piv: Vec3,
        parent_wrot: Mat3,
        anc_chain: List[Tuple[Vec3, Mat3]],
    ) -> None:
        if isinstance(node, str):
            el = elem_by_uuid.get(node)
            if el is not None:
                assigned_uuids.add(node)
                _emit_bb_elem(el, parent_name or "root", parent_bind_piv, anc_chain)
            return
        if not isinstance(node, dict):
            return
        uid = node.get("uuid")
        gmeta = group_by_uuid.get(uid, {}) if uid else {}
        bname = str(node.get("name") or gmeta.get("name") or f"group_{len(bones)}")
        bind_piv = _vec3(node.get("origin") or gmeta.get("origin") or (0.0, 0.0, 0.0))
        lrot = _vec3(node.get("rotation") or gmeta.get("rotation") or (0.0, 0.0, 0.0))
        rmat = rot_zyx(*lrot)
        wpos = _apply_bind_chain(bind_piv, anc_chain)
        wrot = _mat_mul(parent_wrot, rmat)
        lpos = _sub(bind_piv, parent_bind_piv) if parent_name else bind_piv
        is_mir = ("__" in bname) or bname.endswith("_r") or ("_r_" in bname) or bname.startswith("right_")
        bones.append(BoneNode(bname, parent_name, lpos, lrot, wpos, wrot, is_mir))
        next_chain = anc_chain + [(bind_piv, rmat)]
        for ch in node.get("children", []):
            walk_node(ch, bname, bind_piv, wrot, next_chain)

    def _emit_bb_elem(
        el: Dict[str, Any],
        bname: str,
        bone_bind_piv: Vec3,
        chain: List[Tuple[Vec3, Mat3]],
    ) -> None:
        frm = _vec3(el.get("from", (0.0, 0.0, 0.0)))
        to = _vec3(el.get("to", (0.0, 0.0, 0.0)))
        sz = (abs(to[0] - frm[0]), abs(to[1] - frm[1]), abs(to[2] - frm[2]))
        bind_ctr = (0.5 * (frm[0] + to[0]), 0.5 * (frm[1] + to[1]), 0.5 * (frm[2] + to[2]))
        e_orig = _vec3(el.get("origin") or bind_ctr)
        e_rot = _vec3(el.get("rotation") or (0.0, 0.0, 0.0))
        r_el = rot_zyx(*e_rot)
        posed_bind = _add(e_orig, _mat_vec(r_el, _sub(bind_ctr, e_orig)))
        wctr = _apply_bind_chain(posed_bind, chain)
        lctr = _sub(bind_ctr, bone_bind_piv)
        is_p = min(sz) <= 0.05
        elem = BoxElem(
            str(el.get("name", "cube")),
            bname,
            sz,
            wctr,
            lctr,
            e_rot,
            "bbmodel",
            is_plane=is_p,
        )
        (planes if is_p else boxes).append(elem)

    for top in data.get("outliner", []):
        walk_node(top, None, (0.0, 0.0, 0.0), EYE3, [])

    # Handle any top-level ungrouped elements
    if len(assigned_uuids) < len(elem_by_uuid):
        if not any(b.name == "root" for b in bones):
            bones.append(BoneNode("root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), EYE3, False))
        for uid, el in elem_by_uuid.items():
            if uid not in assigned_uuids:
                _emit_bb_elem(el, "root", (0.0, 0.0, 0.0), [])

    return bones, boxes, planes, mesh_count


def parse_bedrock_geo(path: str) -> Tuple[List[BoneNode], List[BoxElem], List[BoxElem], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geos = data.get("minecraft:geometry")
    if isinstance(geos, list) and geos:
        raw_bones = geos[0].get("bones", [])
    else:
        # Legacy Bedrock 1.8/1.10 schema
        geom_key = next((k for k in data if k.startswith("geometry.")), None)
        raw_bones = data[geom_key].get("bones", []) if geom_key else []

    by_name: Dict[str, Dict[str, Any]] = {str(b["name"]): b for b in raw_bones if "name" in b}
    children_map: Dict[Optional[str], List[str]] = collections.defaultdict(list)
    for b in raw_bones:
        bname = str(b["name"])
        pname = b.get("parent")
        children_map[str(pname) if pname in by_name else None].append(bname)

    bones: List[BoneNode] = []
    boxes: List[BoxElem] = []
    planes: List[BoxElem] = []

    def walk_geo(
        bname: str,
        parent_name: Optional[str],
        parent_bind_piv: Vec3,
        parent_wrot: Mat3,
        anc_chain: List[Tuple[Vec3, Mat3]],
    ) -> None:
        rb = by_name[bname]
        raw_piv = _vec3(rb.get("pivot") or (0.0, 0.0, 0.0))
        raw_rot = _vec3(rb.get("rotation") or (0.0, 0.0, 0.0))
        # Convert Bedrock (+X left) to canonical Blockbench (-X left)
        bind_piv = (-raw_piv[0], raw_piv[1], raw_piv[2])
        lrot = (-raw_rot[0], -raw_rot[1], raw_rot[2])
        rmat = rot_zyx(*lrot)
        wpos = _apply_bind_chain(bind_piv, anc_chain)
        wrot = _mat_mul(parent_wrot, rmat)
        lpos = _sub(bind_piv, parent_bind_piv) if parent_name else bind_piv
        is_mir = ("__" in bname) or bname.endswith("_r") or ("_r_" in bname) or bname.startswith("right_")
        bones.append(BoneNode(bname, parent_name, lpos, lrot, wpos, wrot, is_mir))
        next_chain = anc_chain + [(bind_piv, rmat)]

        for idx, cb in enumerate(rb.get("cubes", [])):
            orig = _vec3(cb.get("origin") or (0.0, 0.0, 0.0))
            sz = _vec3(cb.get("size") or (0.0, 0.0, 0.0))
            frm = (-(orig[0] + sz[0]), orig[1], orig[2])
            to = (-orig[0], orig[1] + sz[1], orig[2] + sz[2])
            bind_ctr = (0.5 * (frm[0] + to[0]), 0.5 * (frm[1] + to[1]), 0.5 * (frm[2] + to[2]))
            c_rot_raw = cb.get("rotation")
            if c_rot_raw:
                cr = (-float(c_rot_raw[0]), -float(c_rot_raw[1]), float(c_rot_raw[2]))
                cp_raw = _vec3(cb.get("pivot") or (-bind_ctr[0], bind_ctr[1], bind_ctr[2]))
                cp = (-cp_raw[0], cp_raw[1], cp_raw[2])
                posed_bind = _add(cp, _mat_vec(rot_zyx(*cr), _sub(bind_ctr, cp)))
            else:
                cr = (0.0, 0.0, 0.0)
                posed_bind = bind_ctr
            wctr = _apply_bind_chain(posed_bind, next_chain)
            lctr = _sub(bind_ctr, bind_piv)
            is_p = min(sz) <= 0.05
            elem = BoxElem(f"{bname}_cube_{idx}", bname, sz, wctr, lctr, cr, "geo", is_plane=is_p)
            (planes if is_p else boxes).append(elem)

        for ch in children_map.get(bname, []):
            walk_geo(ch, bname, bind_piv, wrot, next_chain)

    for root_bn in children_map.get(None, []):
        walk_geo(root_bn, None, (0.0, 0.0, 0.0), EYE3, [])

    return bones, boxes, planes, 0


def parse_standalone_blockdsl(path: str) -> Tuple[List[BoneNode], List[BoxElem], List[BoxElem], int]:
    text = open(path, encoding="utf-8").read()
    stack: List[Tuple[Optional[str], Vec3, Mat3, bool]] = [
        (None, (0.0, 0.0, 0.0), EYE3, False)
    ]
    bones: List[BoneNode] = []
    boxes: List[BoxElem] = []
    planes: List[BoxElem] = []

    for raw in text.splitlines():
        s = raw.split("#", 1)[0].split("//", 1)[0].strip()
        if not s:
            continue

        m_bone = re.match(
            r"(mirror_x\s+)?bone\s+(\w+)\s+pos=\[([^\]]+)\]\s+rot=\[([^\]]+)\]\s*\{", s
        )
        if m_bone:
            mir = bool(m_bone.group(1))
            bname = m_bone.group(2)
            pos = _vec3([float(x) for x in m_bone.group(3).split(",")])
            rot = _vec3([float(x) for x in m_bone.group(4).split(",")])
            pname, ppos, prot, pmir = stack[-1]
            is_mir = mir or pmir
            r_loc = rot_zyx(*rot)
            wpos = _add(ppos, _mat_vec(prot, pos))
            wrot = _mat_mul(prot, r_loc)
            bones.append(BoneNode(bname, pname, pos, rot, wpos, wrot, is_mir))
            if is_mir:
                m_pos = _mirror_x_vec(pos)
                m_rot = (rot[0], -rot[1], -rot[2])
                p_mir_name = mirror_name(pname) if (pname and pmir) else pname
                mwpos = _mirror_x_vec(wpos)
                mwrot = _mirror_x_mat(wrot)
                bones.append(
                    BoneNode(mirror_name(bname), p_mir_name, m_pos, m_rot, mwpos, mwrot, True)
                )
            stack.append((bname, wpos, wrot, is_mir))
            continue

        m_box = re.match(
            r"(mirror_x\s+)?box\s+(\w+)\s+from=\[([^\]]+)\]\s+to=\[([^\]]+)\]\s+size=\[([^\]]+)\]\s+origin=\[([^\]]+)\]\s+rot=\[([^\]]+)\](?:\s+mat=(\w+))?",
            s,
        )
        if m_box:
            mir = bool(m_box.group(1))
            name = m_box.group(2)
            frm = _vec3([float(x) for x in m_box.group(3).split(",")])
            to = _vec3([float(x) for x in m_box.group(4).split(",")])
            sz = _vec3([float(x) for x in m_box.group(5).split(",")])
            orig = _vec3([float(x) for x in m_box.group(6).split(",")])
            rot = _vec3([float(x) for x in m_box.group(7).split(",")])
            mat = m_box.group(8) or "default"
            bname, bpos, brot, bmir = stack[-1]
            assert bname is not None
            lctr = (0.5 * (frm[0] + to[0]), 0.5 * (frm[1] + to[1]), 0.5 * (frm[2] + to[2]))
            r_box = rot_zyx(*rot)
            posed_lctr = _add(orig, _mat_vec(r_box, _sub(lctr, orig)))
            wctr = _add(bpos, _mat_vec(brot, posed_lctr))
            target = planes if min(sz) <= 0.05 else boxes
            target.append(BoxElem(name, bname, sz, wctr, lctr, rot, mat, is_plane=(min(sz) <= 0.05)))
            if mir or bmir:
                target_bone = mirror_name(bname) if bmir else bname
                m_wctr = _mirror_x_vec(wctr)
                m_lctr = _mirror_x_vec(lctr)
                m_rot = (rot[0], -rot[1], -rot[2])
                target.append(
                    BoxElem(mirror_name(name), target_bone, sz, m_wctr, m_lctr, m_rot, mat, is_plane=(min(sz) <= 0.05))
                )
            continue

        m_plane = re.match(r"(mirror_x\s+)?plane\s+(\w+)", s)
        if m_plane:
            bname, bpos, _, bmir = stack[-1]
            is_mir = bool(m_plane.group(1)) or bmir
            nm = m_plane.group(2)
            planes.append(BoxElem(nm, bname or "root", (0.0, 8.0, 8.0), bpos, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "plane", is_plane=True))
            if is_mir:
                planes.append(BoxElem(mirror_name(nm), mirror_name(bname or "root"), (0.0, 8.0, 8.0), _mirror_x_vec(bpos), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "plane", is_plane=True))
            continue

        if s == "}" and len(stack) > 1:
            stack.pop()

    return bones, boxes, planes, 0


def load_model(args: List[str]) -> Tuple[str, List[BoneNode], List[BoxElem], List[BoxElem], int]:
    model_path = args[-1]
    low_path = model_path.lower()
    if low_path.endswith(".bbmodel"):
        bones, boxes, planes, meshes = parse_bbmodel(model_path)
        return "bbmodel", bones, boxes, planes, meshes
    if low_path.endswith(".json"):
        bones, boxes, planes, meshes = parse_bedrock_geo(model_path)
        return "bedrock_geo", bones, boxes, planes, meshes

    if len(args) == 2:
        repo_path, dsl_path = args[0], args[1]
        try:
            sys.path.insert(0, repo_path + "/src")
            from synccraft.dsl.parser import parse_program  # type: ignore
            from synccraft.kinematics.se3_chain import compile_geometry  # type: ignore

            prog = parse_program(open(dsl_path, encoding="utf-8").read())
            s = compile_geometry(prog)
            bones = [
                BoneNode(
                    b.name,
                    b.parent_name,
                    _vec3(getattr(b, "local_translation", getattr(b, "local_pos", (0.0, 0.0, 0.0)))),
                    _extract_rot(b),
                    _vec3(getattr(b, "pivot_world", getattr(b, "world_pos", (0.0, 0.0, 0.0)))),
                    tuple(_vec3(row) for row in getattr(b, "rotation", getattr(b, "rotation_world", EYE3))),  # type: ignore[arg-type]
                    bool(getattr(b, "mirrored", False)) or ("__" in b.name),
                )
                for b in s.bones
            ]
            boxes = []
            planes = []
            for b in s.boxes:
                pnorm = getattr(b, "plane_normal", None)
                is_p = (min(b.size) <= 0.05) or (pnorm is not None)
                elem = BoxElem(
                    b.name,
                    b.bone_name,
                    _vec3(b.size),
                    _vec3(b.center),
                    _vec3(getattr(b, "local_center", b.center)),
                    _extract_rot(b),
                    getattr(b, "material", "default"),
                    is_plane=is_p,
                    plane_normal=pnorm,
                )
                if is_p:
                    planes.append(elem)
                else:
                    boxes.append(elem)
            return "blockdsl(synccraft)", bones, boxes, planes, 0
        except Exception:
            bones, boxes, planes, meshes = parse_standalone_blockdsl(dsl_path)
            return "blockdsl(standalone)", bones, boxes, planes, meshes

    bones, boxes, planes, meshes = parse_standalone_blockdsl(model_path)
    return "blockdsl(standalone)", bones, boxes, planes, meshes


def _infer_tier(total_elems: int, explicit_tier: str) -> str:
    if explicit_tier != "auto":
        return explicit_tier
    if total_elems < 45:
        return "small"
    if total_elems < 140:
        return "standard"
    if total_elems < 240:
        return "hero"
    return "epic_boss"


def run_audit(raw_args: List[str]) -> int:
    explicit_tier = "auto"
    pos_args: List[str] = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--tier" and i + 1 < len(raw_args):
            explicit_tier = raw_args[i + 1].strip().lower()
            i += 2
        elif raw_args[i].startswith("--tier="):
            explicit_tier = raw_args[i].split("=", 1)[1].strip().lower()
            i += 1
        else:
            pos_args.append(raw_args[i])
            i += 1

    fmt, bones, boxes, planes, mesh_count = load_model(pos_args)
    plane_count = len(planes)
    total_elems = len(boxes) + plane_count
    tier = _infer_tier(total_elems, explicit_tier)

    par = {b.name: b.parent_name for b in bones}
    bone_map = {b.name: b for b in bones}
    children_map: Dict[str, List[str]] = collections.defaultdict(list)
    for b in bones:
        if b.parent_name:
            children_map[b.parent_name].append(b.name)

    def in_wing(bn: Optional[str]) -> bool:
        while bn:
            low = bn.lower()
            if (
                ("wing" in low and "swing" not in low)
                or "elytr" in low
                or "tegmin" in low
            ):
                return True
            bn = par.get(bn)
        return False

    def subtree_bones(root_bn: str) -> List[str]:
        out = [root_bn]
        q = [root_bn]
        while q:
            cur = q.pop(0)
            for ch in children_map.get(cur, []):
                out.append(ch)
                q.append(ch)
        return out

    non_wing_boxes = [b for b in boxes if not in_wing(b.bone_name)]
    vol_all = sorted((b.size[0] * b.size[1] * b.size[2] for b in boxes), reverse=True)
    vol_nw = sorted((b.size[0] * b.size[1] * b.size[2] for b in non_wing_boxes), reverse=True)

    per_bone = collections.Counter(b.bone_name for b in boxes)
    non_head_bones = [cnt for bn, cnt in per_bone.items() if "head" not in bn.lower()]
    avg_non_head = (sum(non_head_bones) / len(non_head_bones)) if non_head_bones else 0.0

    key = lambda b, sx: (
        round(sx * b.center[0], 1),
        round(b.center[1], 1),
        round(b.center[2], 1),
    )
    left = sorted(key(b, -1.0) for b in boxes if b.center[0] < -0.05)
    right = sorted(key(b, 1.0) for b in boxes if b.center[0] > 0.05)
    left_p = sorted(key(p, -1.0) for p in planes if p.center[0] < -0.30)
    right_p = sorted(key(p, 1.0) for p in planes if p.center[0] > 0.30)
    mirror_ok = (left == right) and (left_p == right_p)

    top10_nw = (sum(vol_nw[:10]) / sum(vol_nw)) if vol_nw else 0.0
    top10_all = (sum(vol_all[:10]) / sum(vol_all)) if vol_all else 0.0

    # Compute 3D Proportional, Coiled-Stance & Trait Telemetry (using rotated OBB corner bounds!)
    all_elems = boxes + planes
    if all_elems:
        xs_all: List[float] = []
        ys_all: List[float] = []
        zs_all: List[float] = []
        for e in all_elems:
            hx_e, hy_e, hz_e = 0.5 * e.size[0], 0.5 * e.size[1], 0.5 * e.size[2]
            bnode = bone_map.get(e.bone_name)
            R_e = bnode.world_rot if bnode is not None else EYE3
            if any(abs(r) > 1e-6 for r in e.local_rot):
                R_e = _mat_mul(R_e, rot_zyx(*e.local_rot))
            ex = abs(R_e[0][0]) * hx_e + abs(R_e[0][1]) * hy_e + abs(R_e[0][2]) * hz_e
            ey = abs(R_e[1][0]) * hx_e + abs(R_e[1][1]) * hy_e + abs(R_e[1][2]) * hz_e
            ez = abs(R_e[2][0]) * hx_e + abs(R_e[2][1]) * hy_e + abs(R_e[2][2]) * hz_e
            xs_all.extend((e.center[0] - ex, e.center[0] + ex))
            ys_all.extend((e.center[1] - ey, e.center[1] + ey))
            zs_all.extend((e.center[2] - ez, e.center[2] + ez))
        y_ground = min(ys_all)
        span = (max(xs_all) - min(xs_all), max(ys_all) - y_ground, max(zs_all) - min(zs_all))
    else:
        y_ground = 0.0
        span = (0.0, 0.0, 0.0)

    _head_excl = ("horn", "ear", "collar", "frill", "crest", "fin", "sail", "plume", "antler")
    head_core_boxes = [
        b for b in non_wing_boxes
        if any(k in b.bone_name.lower() for k in ("head", "jaw", "maxilla", "mandible", "snout", "muzzle", "cheek", "brow"))
        and not any(k in b.name.lower() or k in b.bone_name.lower() for k in _head_excl)
    ]
    if head_core_boxes:
        hx = max(b.center[0] + 0.5 * b.size[0] for b in head_core_boxes) - min(b.center[0] - 0.5 * b.size[0] for b in head_core_boxes)
        hy = max(b.center[1] + 0.5 * b.size[1] for b in head_core_boxes) - min(b.center[1] - 0.5 * b.size[1] for b in head_core_boxes)
        hz = max(b.center[2] + 0.5 * b.size[2] for b in head_core_boxes) - min(b.center[2] - 0.5 * b.size[2] for b in head_core_boxes)
        head_lh = hz / max(1e-6, hy)
        head_wh = hx / max(1e-6, hy)
    else:
        hx = hy = hz = head_lh = head_wh = 0.0

    chest_w = max((b.size[0] for b in non_wing_boxes if any(k in b.bone_name.lower() for k in ("chest", "thorax", "cephalothorax", "prosoma"))), default=0.0)
    waist_w = max((b.size[0] for b in non_wing_boxes if any(k in b.bone_name.lower() for k in ("waist", "abdomen"))), default=0.0)
    v_taper = (chest_w / waist_w) if waist_w > 0 else 0.0

    torso_boxes = [
        b for b in non_wing_boxes
        if any(
            k in b.bone_name.lower()
            for k in ("chest", "torso", "body", "waist", "belly", "pelvis", "hips", "thorax", "pronotum", "abdomen", "carapace", "cephalothorax", "prosoma")
        )
    ]
    if torso_boxes:
        y_belly_min = min(b.center[1] - 0.5 * b.size[1] for b in torso_boxes)
        y_torso_top = max(b.center[1] + 0.5 * b.size[1] for b in torso_boxes)
        belly_clearance_ratio = (y_belly_min - y_ground) / max(1e-6, y_torso_top - y_ground)
    else:
        belly_clearance_ratio = 0.0

    print("=== VIGOROUS CARVED BLOCKBENCH V20 CONSTRUCTION, PROPORTION & STYLE-GRAMMAR AUDIT ===")
    print(
        f"format={fmt}  tier={tier}  total_elements={total_elems} (boxes={len(boxes)}, planes={plane_count}, meshes={mesh_count})  "
        f"bones={len(bones)}  boxes/bone(non-head)={avg_non_head:.2f}"
    )
    print(
        f"top-10 volume (non-wing)={top10_nw:.1%}  top-10 volume (all)={top10_all:.1%}  "
        f"mirror_parity={'OK' if mirror_ok else f'BROKEN ({len(left)}L vs {len(right)}R)'}"
    )
    print(
        f"telemetry: span={span[0]:.1f}x{span[1]:.1f}x{span[2]:.1f}  "
        f"wingspan/length={(span[0]/max(1e-6, span[2])):.2f}  "
        f"head_L/H={head_lh:.2f}  head_W/H={head_wh:.2f}  chest:waist={v_taper:.2f}  "
        f"belly_clearance={belly_clearance_ratio:.2f}"
    )
    print("busiest bones:", per_bone.most_common(5))

    warnings: List[str] = []

    top10_floor = {"small": 0.55, "standard": 0.48, "hero": 0.36, "epic_boss": 0.26}.get(tier, 0.45)
    avg_box_cap = 4.0 if tier == "epic_boss" else 3.5

    # Gate 1: Core hierarchy & symmetry guards
    if top10_nw < top10_floor:
        warnings.append(
            f"[FAIL G1: Volume Hierarchy] Top-10 non-wing boxes hold {top10_nw:.1%} (< {top10_floor:.0%} for tier='{tier}'). Primary anatomical cores must anchor secondary armor/scutes."
        )
    if avg_non_head > avg_box_cap:
        warnings.append(
            f"[FAIL G1: Shard Density] Non-head average boxes/bone is {avg_non_head:.2f} (> {avg_box_cap:.1f}). Distribute complex plates, digits, and horns onto articulated child bones."
        )
    for bn, cnt in per_bone.items():
        limit = 14 if "head" in bn.lower() else (10 if tier == "epic_boss" else 8)
        if cnt > limit:
            warnings.append(
                f"[WARN G1: Bone Bloat] Bone '{bn}' carries {cnt} boxes (> {limit}). Move multi-segment horns, frills, or digits onto child bones."
            )
    if not mirror_ok:
        warnings.append("[FAIL G1: Symmetry] Left/right mirror parity is broken in rest pose.")

    # Gate 2: Wedding-cake axis-aligned snout detector (v38: check bone-level yaw too for Bedrock .geo.json parity!)
    snout_boxes = [
        b
        for b in boxes
        if any(k in b.name.lower() for k in ("snout", "muzzle", "maxilla"))
        and abs(b.local_center[0]) < 0.05
    ]
    if len(snout_boxes) >= 2 and all(abs(b.local_rot[1]) < 1.0 for b in snout_boxes):
        def _eff_yaw(b: BoxElem) -> float:
            bnode = bone_map.get(b.bone_name)
            by = abs(bnode.local_rot[1]) if bnode is not None else 0.0
            return max(abs(b.local_rot[1]), by)

        has_v_rail = any(
            (
                any(k in b.name.lower() or k in b.bone_name.lower() for k in ("maxilla", "snout_wall", "lip_rail", "snout_side", "snout", "muzzle"))
                and _eff_yaw(b) >= 10.0
            )
            for b in boxes
        )
        if not has_v_rail:
            warnings.append(
                "[WARN G2: Wedding-Cake Snout] Multiple centred yaw=0 snout boxes stacked without yawed V-rail maxilla walls (12-22.5 deg yaw). Top view risks rectangular stair-steps."
            )

    # Gate 3: Crown negative-space / french-fry horn congestion
    def _leaf_g3(s: str) -> str:
        return s.lower().split("__")[-1]

    head_horns = [
        b
        for b in boxes
        if "head" in b.bone_name.lower()
        and any(k in _leaf_g3(b.name) for k in ("horn", "crest", "crystal"))
        and "collar" not in _leaf_g3(b.name)
        and "jaw" not in _leaf_g3(b.name)
        and not (
            min(b.size) <= 0.55
            and any(k in _leaf_g3(b.name) for k in ("sail", "membrane", "fin", "web", "fan"))
        )
    ]
    if len(head_horns) > 8:
        pitches = [b.local_rot[0] for b in head_horns if abs(b.local_rot[0]) > 5.0]
        pitch_std = _std(pitches)
        warnings.append(
            f"[WARN G3: Crown Congestion] {len(head_horns)} horn/crest/crystal boxes on head (pitch std={pitch_std:.1f} deg). Risk of 'organ-pipe / french-fry' silhouette; keep 1 splayed primary pair + <=1 secondary pair with open negative space."
        )

    # Gate 4: Wing 3D vault camber & wrist knuckle spacing
    left_wing_bones = [b for b in bones if in_wing(b.name) and b.world_pos[0] < -0.05 and "__" not in b.name]
    if len(left_wing_bones) >= 4:
        pts = [b.world_pos for b in left_wing_bones]
        xs = [p[0] for p in pts]
        span_w = max(xs) - min(xs)
        if span_w > 10.0:
            thickness_3d = _min_singular_value_3d(pts)
            camber_ratio = thickness_3d / span_w
            if camber_ratio < 0.05:
                warnings.append(
                    f"[WARN G4: 2D Plank Wing] Left wing bone chain out-of-plane camber ratio is {camber_ratio:.3f} (< 0.050). Add 3D arch/cup across forearm and 2-segment finger knuckles so top view is not a flat plank."
                )

        by_parent: Dict[str, List[BoneNode]] = collections.defaultdict(list)
        for b in left_wing_bones:
            if b.parent_name:
                by_parent[b.parent_name].append(b)
        for p_name, children in by_parent.items():
            f_kids = [
                c
                for c in children
                if any(k in c.name.lower() for k in ("f1", "f2", "f3", "f4", "finger", "wing_segment"))
            ]
            if len(f_kids) >= 3:
                dists = [
                    _norm(_sub(f_kids[i].local_pos, f_kids[i + 1].local_pos))
                    for i in range(len(f_kids) - 1)
                ]
                if min(dists) < 1.5:
                    warnings.append(
                        f"[WARN G4: Co-located Sibling Wing Knuckles] Sibling wing fingers under '{p_name}' have min root spacing {min(dists):.2f}u (< 1.5u). Either space knuckles along a palm block OR nest fingers sequentially as a shared-wrist fan chain."
                    )

    # Gate 5: Crate-train unchamfered cube spine & chimney neck check
    torso_neck_boxes = [
        b
        for b in non_wing_boxes
        if any(k in b.bone_name.lower() for k in ("chest", "waist", "hips", "neck"))
    ]
    core_cubes = [
        b
        for b in torso_neck_boxes
        if "core" in b.name.lower()
        and max(b.size) / max(min(b.size), 1e-6) < 1.35
        and all(abs(r) < 1.0 for r in b.local_rot)
    ]
    if len(core_cubes) >= 4:
        has_torso_gussets = any(
            any(k in b.name.lower() for k in ("gusset", "trapezius", "clavicle", "rib_slope", "lat_wedge"))
            for b in torso_neck_boxes
        )
        if not has_torso_gussets:
            warnings.append(
                f"[WARN G5: Crate-Train / Chimney Spine] {len(core_cubes)} near-cubic axis-aligned torso/neck cores without trapezius/clavicle gussets or lateral rib chamfers."
            )

    # Gate 6: Gundam hip-skirt slab & snowshoe plank-foot check
    for b in non_wing_boxes:
        low = b.name.lower()
        if any(k in low for k in ("haunch", "hip_plate", "hip_skirt")) and b.center[0] < -0.05:
            if abs(b.local_rot[0]) < 8.0 and abs(b.local_rot[2]) < 10.0:
                warnings.append(
                    f"[WARN G6: Pasted Hip-Skirt Slab] '{b.name}' sits nearly upright (pitch={b.local_rot[0]:.1f}, roll={b.local_rot[2]:.1f}). Replace flat outer slabs with pitched/rolled proximal thigh & hamstring ('meat') taper wedges."
                )
        if any(k in low for k in ("palm", "foot_base", "sole")) and b.center[0] < -0.05:
            if b.size[1] / max(b.size[0], b.size[2], 1e-6) < 0.38:
                warnings.append(
                    f"[WARN G6: Snowshoe Plank Foot] '{b.name}' height/length ratio is {b.size[1]/max(b.size[0], b.size[2]):.2f} (< 0.38). Thicken the paw/foot block so it continues the limb instead of reading as a flat board."
                )

    # Gate 7: Texel-clean grid sizing & Bedrock/Block-DSL naming hygiene
    off_grid = [
        b.name
        for b in boxes
        if any(abs((d * 2.0) - round(d * 2.0)) > 1e-3 for d in b.size)
    ]
    if off_grid:
        sample = ", ".join(off_grid[:5])
        warnings.append(
            f"[WARN G7: Off-Grid Texel Sizes] {len(off_grid)} box(es) use non-0.5u dimensions ({sample}...). Keep clean integer/0.5u sizes and resolve Z-fighting with layer/micro-inflate instead."
        )
    bad_bone_names = [
        b.name
        for b in bones
        if any(c.isupper() or c.isspace() for c in b.name)
    ]
    if bad_bone_names:
        sample = ", ".join(sorted(set(bad_bone_names))[:5])
        warnings.append(
            f"[WARN G7: Bone Naming Hygiene] Uppercase or spaced bone names detected ({sample}). Use lowercase snake_case for clean Bedrock/GeckoLib/Block-DSL parsing."
        )

    # Gate 8: Epic Complexity Floor & Anti-Stick-Limb Shaft Gate
    if tier == "epic_boss" and total_elems < 240:
        warnings.append(
            f"[FAIL G8: Under-Complex Epic Boss] Total sculpted elements={total_elems} (< 240 epic_boss floor). Add sunken compound musculature, stepped cuboid chamfers, articulated multi-joint digits/dewclaws, and coherent echeloned carapace rows (target 250-550 elements)."
        )

    if tier in ("hero", "epic_boss"):
        stick_bones: List[str] = []
        shaft_keywords = ("arm_l_upper", "arm_l_fore", "leg_l_thigh", "leg_l_shin", "leg_l_ankle", "wing_l_upper", "wing_l_fore")
        for b in bones:
            low = b.name.lower()
            if "__" in low or b.world_pos[0] > -0.05:
                continue
            if any(k in low for k in shaft_keywords):
                direct_boxes = [bx for bx in boxes if bx.bone_name == b.name]
                orn_kids = [
                    ch for ch in children_map.get(b.name, [])
                    if not any(sk in ch.lower() for sk in ("fore", "shin", "ankle", "paw", "hand", "foot", "palm", "f1", "f2", "f3", "f4", "thumb"))
                ]
                orn_boxes = [bx for bx in boxes if bx.bone_name in orn_kids]
                seg_boxes = direct_boxes + orn_boxes
                max_len = max((max(bx.size) for bx in direct_boxes), default=0.0)
                if len(seg_boxes) == 1:
                    s0 = seg_boxes[0].size
                    aspect = max(s0) / max(1e-6, sorted(s0)[1])
                    if aspect >= 2.6:
                        stick_bones.append(f"{b.name}({max(s0):.0f}u single stick)")
                elif tier == "epic_boss" and len(seg_boxes) == 2 and max_len >= 18.0:
                    stick_bones.append(f"{b.name}({max_len:.0f}u shaft has only 2 boxes)")
        if stick_bones:
            sev = "FAIL" if tier == "epic_boss" else "WARN"
            warnings.append(
                f"[{sev} G8: Bare / Sparse Stick-Limb Shafts] {len(stick_bones)} major limb/wing segment(s) lack compound mass-packing: {', '.join(stick_bones)}. Combine core shaft + 40-80% sunken muscle/chamfer block + overlapping vambrace/greave scutes."
            )

    # Gate 9: Articulated Extremities, Digit Hierarchy & Splay Gate
    if tier in ("hero", "epic_boss"):
        extremity_roots = [
            b for b in bones
            if "__" not in b.name
            and b.world_pos[0] < -0.05
            and any(b.name.lower().startswith(k) or b.name.lower().endswith(k) for k in ("paw_l", "hand_l", "foot_l", "front_foot_l", "hind_foot_l"))
        ]
        for ext in extremity_roots:
            sub_bns = subtree_bones(ext.name)
            digit_bones = [
                bone_map[bn] for bn in sub_bns
                if bn != ext.name and any(k in bn.lower() for k in ("toe", "finger", "digit", "thumb", "dewclaw", "hallux", "pollex"))
            ]
            root_digits = [d for d in digit_bones if d.parent_name == ext.name]
            min_digits = 4 if tier == "epic_boss" else 3
            sev = "FAIL" if tier == "epic_boss" else "WARN"
            if len(root_digits) < min_digits:
                warnings.append(
                    f"[{sev} G9: Under-Articulated Extremity] '{ext.name}' has {len(root_digits)} root digits (< {min_digits} for {tier}; include 3 splayed forward digits + opposable thumb or rear dewclaw/hallux)."
                )
            if len(root_digits) >= 3:
                yaws = [d.local_rot[1] + d.local_rot[2] for d in root_digits]
                fan_spread = max(yaws) - min(yaws)
                if fan_spread < 20.0:
                    warnings.append(
                        f"[{sev} G9: Parallel Peg Digits] Digits on '{ext.name}' have only {fan_spread:.1f} deg angular spread (< 20 deg). Splay outer digits laterally and angle thumb/dewclaw distinctly."
                    )
                d_boxes = [bx for bx in boxes if bx.bone_name in {d.name for d in root_digits}]
                if tier == "epic_boss" and d_boxes:
                    max_dl = max(max(bx.size) for bx in d_boxes)
                    min_dl = min(max(bx.size) for bx in d_boxes)
                    jointed_digits = sum(1 for d in root_digits if len(children_map.get(d.name, [])) >= 1)
                    if jointed_digits < min(3, len(root_digits)):
                        warnings.append(
                            f"[FAIL G9: Unjointed Stub Digits] '{ext.name}' has only {jointed_digits}/{len(root_digits)} multi-bone digits. On epic_boss tier, build articulated 2-to-3 bone phalanx chains (knuckle -> distal phalanx -> hooked talon) with middle digit 1.2-1.35x outer digits."
                        )
                    elif max_dl / max(1e-6, min_dl) < 1.15:
                        warnings.append(
                            f"[WARN G9: Uniform Clone Digits] '{ext.name}' digits have near-identical lengths (ratio={max_dl/max(1e-6, min_dl):.2f} < 1.15). Grade middle digit noticeably longer than inner/outer digits."
                        )

    # Gate 10: Wing Membrane Chord Coverage Gate (v38: supports both 0.0u planes AND 0.5u zero-blink dual-sided membrane slabs!)
    _mem_kw = ("bay", "sail", "membrane", "web", "patag", "wing_leaf", "wing_lobe", "wing_fan")
    left_wing_membranes = [
        p for p in planes if in_wing(p.bone_name) and p.center[0] < -0.05 and "__" not in p.name
    ] + [
        b for b in boxes
        if in_wing(b.bone_name) and b.center[0] < -0.05 and "__" not in b.name
        and min(b.size) <= 0.55 and sorted(b.size, reverse=True)[1] >= 6.0
        and any(k in b.name.lower() or k in b.bone_name.lower() for k in _mem_kw)
    ]
    if len(left_wing_bones) >= 4 and left_wing_membranes:
        finger_tips: List[Tuple[str, Vec3]] = []
        for idx_tag in ("f1", "f2", "f3", "f4", "wing_segment_1", "wing_segment_2", "wing_segment_3", "wing_segment_4"):
            f_boxes = [
                bx for bx in boxes
                if in_wing(bx.bone_name) and bx.center[0] < -0.05 and "__" not in bx.name
                and idx_tag in bx.bone_name.lower()
                and "armor" not in bx.name.lower()
                and "claw" not in bx.name.lower()
                and not (min(bx.size) <= 0.55 and any(k in bx.name.lower() for k in _mem_kw))
            ]
            if f_boxes:
                distal = min(f_boxes, key=lambda bx: bx.center[0])
                bnode = bone_map.get(distal.bone_name)
                half_len = 0.5 * max(distal.size)
                if bnode is not None:
                    ax_x = (bnode.world_rot[0][0], bnode.world_rot[1][0], bnode.world_rot[2][0])
                    sign = -1.0 if ax_x[0] > 0 else 1.0
                    tip_pt = _add(distal.center, (sign * ax_x[0] * half_len, sign * ax_x[1] * half_len, sign * ax_x[2] * half_len))
                    tip_cand_neg = _add(distal.center, (-bnode.world_rot[0][0] * half_len, -bnode.world_rot[1][0] * half_len, -bnode.world_rot[2][0] * half_len))
                    tip_pt = tip_cand_neg if tip_cand_neg[0] < distal.center[0] else tip_pt
                else:
                    tip_pt = distal.center
                finger_tips.append((idx_tag, tip_pt))

        if len(finger_tips) >= 3:
            tip_gaps = [
                _norm(_sub(finger_tips[i][1], finger_tips[i + 1][1]))
                for i in range(len(finger_tips) - 1)
            ]
            max_tip_gap = max(tip_gaps)
            plane_chords = [sorted(p.size, reverse=True)[1] for p in left_wing_membranes]
            max_chord = max(plane_chords) if plane_chords else 0.0
            coverage = max_chord / max(1e-6, max_tip_gap)
            print(f"wing_membrane_telemetry: finger_tips={len(finger_tips)}  max_adjacent_tip_gap={max_tip_gap:.1f}u  max_plane_chord={max_chord:.1f}u  chord_coverage={coverage:.2f}x")
            if coverage < 0.88:
                sev = "FAIL" if tier in ("hero", "epic_boss") else "WARN"
                warnings.append(
                    f"[{sev} G10: Gapped Wing Membrane Bays] Max wing membrane chord is {max_chord:.1f}u vs adjacent fingertip gap {max_tip_gap:.1f}u (coverage={coverage:.2f}x < 0.88x). Narrow rectangular strips open ugly V-shaped sky gaps between splayed fingers! Widen bay membranes to >= {max_tip_gap*1.08:.0f}u (nested fan overlap) or 3 echeloned alpha feather rows."
                )

    # Gate 11 (v20/v33/v38): Anti-Blender Fake-Curve, Micro-Slice & Polygon Mesh Abuse Gate
    if mesh_count > 0:
        warnings.append(
            f"[FAIL G11: Fake-Blender Polygon Mesh] Detected {mesh_count} non-cuboid 'mesh' element(s). Blockbench is Blockbench, NEVER a substitute for Blender — use pure 16x/32x cuboids (with optional 2x focal uv_scale on eyes/lenses) and 100% hard-alpha planes / 0.5u dual-sided membrane slabs!"
        )
    # v37/v38: Exempt purposeful 0.5u V-arch volumetric teeth/fangs, V-sloped cranial/snout/jaw facets,
    # AND zero-blink 0.5u dual-sided wing/tail/crest membrane slabs while keeping strict anti-wafer enforcement on body/limb/tail shafts!
    _cranial_or_mem_kw = (
        "dental", "teeth", "tooth", "fang", "canine", "incisor", "molar", "premolar",
        "lip", "serr", "snout", "muzzle", "nostril", "rhin", "jaw", "chin", "brow",
        "eyelid", "zygoma", "cheek", "horn", "head",
        "bay", "sail", "membrane", "web", "patag", "tail_fan", "tail_petal", "crest",
    )
    wafer_by_bone: Dict[str, int] = collections.Counter(
        b.bone_name
        for b in boxes
        if 0.05 < min(b.size) < 1.0
        and not any(k in b.bone_name.lower() or k in b.name.lower() for k in _cranial_or_mem_kw)
    )
    wafer_abusers = [(bn, cnt) for bn, cnt in wafer_by_bone.items() if cnt >= 6 and "__" not in bn]
    if wafer_abusers:
        desc = ", ".join(f"{bn}({cnt} slices)" for bn, cnt in wafer_abusers[:4])
        warnings.append(
            f"[FAIL G11: Fake-Blender Micro-Slice Curve] Bone(s) stack >= 6 sub-1.0u wafer slices ({desc}) to fake smooth curvature/spheres. Keep proud planar cuboids and carve sub-grid contours in 16x/32x pixel relief!"
        )

    # Gate 12 (v20/v38): Coiled Predatory Weight & Anti-Stilt Toy Posture Gate (skip free-arm bipeds whose arms do not touch ground!)
    forelimb_boxes = [
        bx for bx in boxes
        if any(k in bx.bone_name.lower() for k in ("arm_l", "foreleg_l", "front_leg_l", "paw_l"))
    ]
    forelimbs_grounded = bool(forelimb_boxes) and (
        min(bx.center[1] - 0.5 * bx.size[1] for bx in forelimb_boxes) <= y_ground + max(2.5, 0.18 * span[1])
    )
    has_hindlimbs = any(any(k in b.name.lower() for k in ("leg_l", "hindleg_l", "thigh_l")) for b in bones)
    if forelimbs_grounded and has_hindlimbs and torso_boxes and belly_clearance_ratio > 0.58:
        warnings.append(
            f"[WARN G12: Stilt-Leg Toy Posture] Grounded multi-legged belly clearance is {belly_clearance_ratio:.2f} (> 0.58 of torso top height). Heavy quadrupeds and Elder Dragons should hunker with coiled predatory weight (belly_clearance <= 0.35-0.45) and thick tucked limbs rather than tall skinny stilts."
        )

    # Gate 13 (v33/v38): Anti-Toy Tail / Plate Taper & Facet Carving Telemetry Gate
    _tail_leaf_excl = ("petal", "fin", "sail", "fan", "membrane", "plume")
    tail_bones = [
        b for b in bones
        if "tail" in b.name.lower() and "__" not in b.name
        and not any(k in b.name.lower() for k in _tail_leaf_excl)
    ]
    if len(tail_bones) >= 3:
        tail_box_counts = [
            sum(1 for bx in boxes if bx.bone_name == tb.name)
            for tb in tail_bones
        ]
        avg_tail_elems = sum(tail_box_counts) / max(1, len(tail_bones))
        tail_planes = sum(
            1 for p in planes if "tail" in p.bone_name.lower()
        ) + sum(
            1 for b in boxes if "tail" in b.bone_name.lower() and min(b.size) <= 0.55 and any(k in b.name.lower() for k in _tail_leaf_excl)
        )
        print(
            f"tail_carving_telemetry: tail_bones={len(tail_bones)}  avg_boxes_per_tail_bone={avg_tail_elems:.2f}  tail_membranes={tail_planes}"
        )
        if tier in ("hero", "epic_boss") and avg_tail_elems < 1.8:
            warnings.append(
                f"[WARN G13: Toy-Like Uncarved Tail] Multi-segment tail has only {avg_tail_elems:.2f} cuboids/bone (< 1.8). Avoid stacking plain uniform rectangular bricks — carve multi-plane chamfered flanks, staggered overlapping dorsal/ventral shingle scutes, or tapered multi-step tip blades!"
            )

    # Gate 14 (v33/v38.2): Focal Eye / Optical Sensor Periorbital Framing, Ratio & Anti-Extrusion Audit
    def _leaf(s: str) -> str:
        return s.lower().split("__")[-1]

    eye_boxes = [
        bx for bx in boxes
        if (
            any(k in _leaf(bx.name) for k in ("eye_l", "eye_r", "eye_main", "eye_mid", "eyeball", "optic", "visor_lens"))
            or any(k in _leaf(bx.bone_name) for k in ("eye_l", "eye_r", "eye_socket", "eye_main", "optic"))
        )
        and not any(x in _leaf(bx.name) for x in ("eyelid", "brow", "socket", "rim", "ring", "stalk", "back", "cradle", "frame", "bezel", "post", "bar", "hood", "ledge", "boss"))
    ]
    if eye_boxes:
        lid_or_brow = [
            bx for bx in boxes
            if any(k in _leaf(bx.name) or k in _leaf(bx.bone_name) for k in ("eyelid", "brow", "orbital", "zygoma", "ocular", "lens_bezel", "eye_ring", "eye_socket"))
            and bx not in eye_boxes
        ]
        # Compute eye_h / head_h ratio and lateral extrusion past surrounding head/cheek/brow frame!
        avg_eye_h = sum(e.size[1] for e in eye_boxes) / max(1, len(eye_boxes))
        head_frame_boxes = [
            bx for bx in boxes
            if bx not in eye_boxes
            and (
                "head" in bx.bone_name.lower()
                or "skull" in bx.bone_name.lower()
                or "cranium" in bx.bone_name.lower()
                or "brow" in bx.bone_name.lower()
                or "cheek" in bx.bone_name.lower()
                or "face" in bx.bone_name.lower()
                or "mask" in bx.bone_name.lower()
                or "orbital" in bx.bone_name.lower()
                or bx in lid_or_brow
            )
            and not any(k in _leaf(bx.name) or k in _leaf(bx.bone_name) for k in ("horn", "ear", "collar", "frill", "crest", "fin", "sail", "plume", "antler", "antenna", "tuft", "beak", "snout", "jaw"))
        ]
        # Detect ONLY genuinely dorsal-turret / stalked / compound-eye taxa (amphibians, insects, arachnids, gastropods, crustaceans).
        # All other taxa (mammals, birds, reptiles, fish, eels, characters) must obey eye_h/head_h, eye_vs_cranium & eye_above_crown!
        _fname = Path(pos_args[-1]).name.lower() if pos_args else ""
        all_names_low = " ".join(b.name.lower() for b in bones[:25]) + " " + _fname
        _tokens = set(re.findall(r"[a-z]+", all_names_low))
        _bulging_exact = {
            "toad", "toads", "frog", "frogs", "wasp", "wasps", "bug", "bugs", "bee", "bees",
            "ant", "ants", "spider", "spiders", "arachnid", "beetle", "beetles", "hopper",
            "fly", "flies", "moth", "moths", "snail", "snails", "slug", "slugs", "crab", "crabs",
            "prawn", "shrimp", "hornet", "mantis", "isopod", "pillbug", "cricket", "locust",
            "elytron", "elytra", "chelicera", "eyestalk", "eyestalks", "pedipalp", "ommatidia",
        }
        _bulging_suffixes = (
            "toad", "frog", "wasp", "bug", "beetle", "spider", "snail", "slug",
            "crab", "hopper", "fly", "flies", "moth", "hornet", "prawn", "shrimp",
            "mantis", "antula", "eyestalk", "chelicera",
        )
        is_bulging_taxon = bool(
            (_tokens & _bulging_exact)
            or any(t.endswith(_bulging_suffixes) for t in _tokens)
        )
        eye_h_ratio_str = ""
        eye_extr_str = ""
        eye_crown_str = ""
        if head_frame_boxes:
            hf_ys = [c for bx in head_frame_boxes for c in (bx.center[1] - 0.5 * bx.size[1], bx.center[1] + 0.5 * bx.size[1])]
            head_h_core = max(1e-4, max(hf_ys) - min(hf_ys))
            eye_h_ratio = avg_eye_h / head_h_core
            eye_h_ratio_str = f"  eye_h/head_h={eye_h_ratio:.2f}"
            # Measure lateral eye extrusion AND crown overflow directly against the midline cranial envelope (`cranium_core`/`head_core`/`skull`),
            # NOT against outer eye_rings or rear cheek ruffs that hide pasted-on bug-eyes!
            cranium_cores = [
                bx for bx in head_frame_boxes
                if abs(bx.center[0]) <= 0.25
                and (
                    any(k in _leaf(bx.name) for k in ("cranium", "head_core", "skull", "head_base", "head_main", "head_box", "head_puff", "head_body"))
                    or (bx.bone_name.lower() in ("head", "cranium", "skull") and bx.size[0] >= 2.0)
                )
                and not any(k in _leaf(bx.name) for k in ("eye", "eyelid", "brow", "orbital", "forehead", "crown", "mouth", "throat", "palate"))
            ]
            ref_frame = cranium_cores if cranium_cores else head_frame_boxes
            max_cranium_abs_x = max(abs(bx.center[0]) + 0.5 * bx.size[0] for bx in ref_frame)
            max_cranium_top_y = max(bx.center[1] + 0.5 * bx.size[1] for bx in ref_frame)
            max_eye_abs_x = max(abs(bx.center[0]) + 0.5 * bx.size[0] for bx in eye_boxes)
            max_eye_top_y = max(bx.center[1] + 0.5 * bx.size[1] for bx in eye_boxes)
            eye_extr = max_eye_abs_x - max_cranium_abs_x
            eye_above_crown = max_eye_top_y - max_cranium_top_y
            eye_extr_str = f"  eye_vs_cranium={eye_extr:+.2f}u"
            eye_crown_str = f"  eye_above_crown={eye_above_crown:+.2f}u"
            if not is_bulging_taxon and head_h_core >= 3.0:
                if eye_h_ratio > 0.32:
                    warnings.append(
                        f"[WARN G14: Oversized Eye Ratio] eye_h/head_h={eye_h_ratio:.2f} (> 0.32). Shrink 3D eye cuboids to match true reference ratio (0.12-0.30); use native uv_scale=1 on small creatures (clean 2x2-4x4 px eyes) or uv_scale=2 (64x max) on hero/epic bosses!"
                    )
                if eye_extr > 0.08:
                    warnings.append(
                        f"[WARN G14: Extruded Outward Eye] Eye protrudes {eye_extr:+.2f}u laterally beyond the cranium wall (> +0.08u). Tuck eye_l/r flush-recessed inside cranium_core & orbital bezel (max_eye_x <= cranium_half_w)!"
                    )
                if eye_above_crown > 0.10:
                    warnings.append(
                        f"[WARN G14: Eye Exceeds Head Top] Eye top rises {eye_above_crown:+.2f}u above cranium top (> +0.10u). Lower and compact eye_l/r cleanly inside the skull profile!"
                    )
            # Check for eye viewing-face occlusion/clipping by head_core, snout/throat_mask, or maxilla!
            eye_blockers: List[str] = []
            _frame_ok_toks = ("eyelid", "brow", "orbital", "zygoma", "ocular", "eye_ring", "eye_socket", "lens_bezel")
            for eb in eye_boxes:
                if eb.center[0] > 0.10:
                    continue
                ex0, ex1 = eb.center[0] - 0.5 * eb.size[0], eb.center[0] + 0.5 * eb.size[0]
                ey0, ey1 = eb.center[1] - 0.5 * eb.size[1], eb.center[1] + 0.5 * eb.size[1]
                ez0, ez1 = eb.center[2] - 0.5 * eb.size[2], eb.center[2] + 0.5 * eb.size[2]
                front_facing = eb.size[0] > eb.size[2] + 0.25
                for ob in boxes:
                    if ob is eb or ob in eye_boxes or ob.center[0] > 0.50:
                        continue
                    onm = _leaf(ob.name)
                    if any(k in onm for k in _frame_ok_toks):
                        continue
                    bx0, bx1 = ob.center[0] - 0.5 * ob.size[0], ob.center[0] + 0.5 * ob.size[0]
                    by0, by1 = ob.center[1] - 0.5 * ob.size[1], ob.center[1] + 0.5 * ob.size[1]
                    bz0, bz1 = ob.center[2] - 0.5 * ob.size[2], ob.center[2] + 0.5 * ob.size[2]
                    ox = min(ex1, bx1) - max(ex0, bx0)
                    oy = min(ey1, by1) - max(ey0, by0)
                    oz = min(ez1, bz1) - max(ez0, bz0)
                    if front_facing:
                        if bz1 >= ez1 - 0.02 and ox / max(1e-4, eb.size[0]) >= 0.40 and oy / max(1e-4, eb.size[1]) >= 0.40 and oz > 0.05:
                            eye_blockers.append(ob.name)
                    else:
                        if bx0 <= ex0 + 0.02 and oy / max(1e-4, eb.size[1]) >= 0.40 and oz / max(1e-4, eb.size[2]) >= 0.40 and ox > 0.05:
                            eye_blockers.append(ob.name)
            if eye_blockers:
                uniq_b = sorted(set(eye_blockers))[:4]
                warnings.append(
                    f"[WARN G14: Eye Blocked/Overlapped] Solid head/snout/mask box(es) {uniq_b} clip or cover >=40% of the eye's viewing face! Reposition eye_l/r cleanly proud of head_core/snout_mask (while staying <= cranium_half_w) with zero overlap!"
                )
        print(
            f"eye_carving_telemetry: focal_eye_boxes={len(eye_boxes)}  periorbital_framing_elems={len(lid_or_brow)}{eye_h_ratio_str}{eye_extr_str}{eye_crown_str}"
        )
        if tier in ("hero", "epic_boss") and len(lid_or_brow) < 2:
            warnings.append(
                f"[WARN G14: Unframed Flat Eye] Detected {len(eye_boxes)} eye/optic box(es) with only {len(lid_or_brow)} periorbital framing elements (< 2). Carve recessed orbital framing (upper lid hood, lower lid cradle, brow/bezel) and use 2x focal uv_scale for multi-layer iris/pupil/catchlight realism!"
            )
    if head_lh == 0.0 and belly_clearance_ratio == 0.0 and all_elems:
        radial_chains = sum(1 for b in bones if any(k in b.name.lower() for k in ("tentacle", "oral_arm", "bell_lobe", "ribbon")))
        print(f"radial_pelagic_telemetry: floating_min_y={y_ground:.2f}u  radial_chains={radial_chains}  plane_ratio={plane_count/max(1,total_elems):.2f}")

    if warnings:
        print("\n--- DIAGNOSTIC FINDINGS ---")
        for w in warnings:
            print(" ", w)
    else:
        print("\nALL GEOMETRIC, PROPORTION & VIGOROUS BLOCKBENCH GATES (G1-G14) PASSED.")

    return 1 if any(w.startswith("[FAIL") for w in warnings) else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 construct_check.py [--tier auto|small|standard|hero|epic_boss] [<SyncCraft_repo>] <model.{bbmodel|geo.json|blockdsl}>")
        sys.exit(2)
    sys.exit(run_audit(sys.argv[1:]))
