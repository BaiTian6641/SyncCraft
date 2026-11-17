"""Deterministic Morphology & Anatomical Role Analyzer (`@morphology`, ADR 0011).

Derives limb chains, anatomical roles (`fore_l`, `fore_r`, `mid_l`, `mid_r`, `hind_l`, `hind_r`),
posture (`erect` | `crouched` | `sprawled`), hip height, leg length, reach bounds, local sole
contact points, `spine` / `neck` / `tail` / `wing` chains, box-volume Center of Mass (CoM),
2D ground support polygon, and bilateral symmetry pairs directly from any compiled Block-DSL
`Program` / `Scene` without hand-annotated tags.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import IKChainDecl, Program, Vec3
from synccraft.kinematics.se3_chain import Scene, WorldBone, WorldBox, compile_geometry


@dataclass(frozen=True)
class LimbMorphology:
    """Derived geometric and kinematic descriptor for a single ground-contacting limb."""

    role: str
    chain: tuple[str, ...]
    posture: str  # "erect" | "crouched" | "sprawled"
    hip_height: float
    length: float
    reach: tuple[float, float]
    contact_local: Vec3
    contact_world: Vec3
    hip_world: Vec3
    side: str  # "l" | "r" | "c"
    mirror_of: str | None = None
    ik_chain_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "chain": list(self.chain),
            "posture": self.posture,
            "hip_height": round(self.hip_height, 4),
            "length": round(self.length, 4),
            "reach": [round(self.reach[0], 4), round(self.reach[1], 4)],
            "contact": [round(c, 4) for c in self.contact_local],
            "contact_world": [round(c, 4) for c in self.contact_world],
            "hip_world": [round(c, 4) for c in self.hip_world],
            "side": self.side,
        }
        if self.mirror_of is not None:
            payload["mirror_of"] = self.mirror_of
        if self.ik_chain_name is not None:
            payload["ik_chain_name"] = self.ik_chain_name
        return payload


@dataclass(frozen=True)
class WingMorphology:
    """Derived descriptor for a non-ground lateral wing appendage."""

    role: str
    chain: tuple[str, ...]
    span: float
    mirror_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "role": self.role,
            "chain": list(self.chain),
            "span": round(self.span, 4),
        }
        if self.mirror_of is not None:
            out["mirror_of"] = self.mirror_of
        return out


@dataclass(frozen=True)
class MorphologyReport:
    """Complete derived `@morphology` anatomical & inertial summary of an asset."""

    limbs: tuple[LimbMorphology, ...]
    spine: tuple[str, ...]
    neck: tuple[str, ...]
    tail: tuple[str, ...]
    wings: tuple[WingMorphology, ...]
    total_mass: float
    com: Vec3
    ground_y: float
    forward_sign_z: float
    body_span: Vec3
    support_polygon_xz: tuple[tuple[float, float], ...]
    symmetry_pairs: tuple[tuple[str, str], ...]

    @property
    def limb_by_role(self) -> dict[str, LimbMorphology]:
        return {lb.role: lb for lb in self.limbs}

    def to_dict(self) -> dict[str, Any]:
        return {
            "limbs": [lb.to_dict() for lb in self.limbs],
            "spine": list(self.spine),
            "neck": list(self.neck),
            "tail": list(self.tail),
            "wings": [wg.to_dict() for wg in self.wings],
            "mass": {
                "total": round(self.total_mass, 4),
                "com": [round(c, 4) for c in self.com],
                "ground_y": round(self.ground_y, 4),
                "support_polygon": [[round(p[0], 4), round(p[1], 4)] for p in self.support_polygon_xz],
            },
            "forward_sign_z": self.forward_sign_z,
            "body_span": [round(x, 4) for x in self.body_span],
            "symmetry_pairs": [list(p) for p in self.symmetry_pairs],
        }


def _convex_hull_2d(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Monotone-chain 2D convex hull in XZ ground plane."""
    pts = sorted({(round(float(x), 6), round(float(z), 6)) for x, z in points})
    if len(pts) <= 2:
        return tuple(pts)

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def point_in_convex_polygon_margin(pt_xz: tuple[float, float], hull_xz: tuple[tuple[float, float], ...]) -> float:
    """Signed inward distance from `pt_xz` to `hull_xz` (>= 0 inside or on boundary, < 0 outside)."""
    n = len(hull_xz)
    if n == 0:
        return 0.0
    if n == 1:
        return -math.hypot(pt_xz[0] - hull_xz[0][0], pt_xz[1] - hull_xz[0][1])
    if n == 2:
        # Distance to line segment for biped stance line
        ax, az = hull_xz[0]
        bx, bz = hull_xz[1]
        dx, dz = bx - ax, bz - az
        seg_sq = dx * dx + dz * dz
        if seg_sq < 1e-9:
            return -math.hypot(pt_xz[0] - ax, pt_xz[1] - az)
        t = max(0.0, min(1.0, ((pt_xz[0] - ax) * dx + (pt_xz[1] - az) * dz) / seg_sq))
        px, pz = ax + t * dx, az + t * dz
        return -math.hypot(pt_xz[0] - px, pt_xz[1] - pz)

    min_signed = float("inf")
    for i in range(n):
        ax, az = hull_xz[i]
        bx, bz = hull_xz[(i + 1) % n]
        ex, ez = bx - ax, bz - az
        edge_len = math.hypot(ex, ez)
        if edge_len < 1e-9:
            continue
        # _convex_hull_2d produces counter-clockwise vertices in (x, z)
        signed_d = (ex * (pt_xz[1] - az) - ez * (pt_xz[0] - ax)) / edge_len
        min_signed = min(min_signed, signed_d)
    return min_signed if math.isfinite(min_signed) else 0.0


def _infer_side_label(bone_name: str, x_coord: float, left_sign_x: float) -> str:
    low = bone_name.lower()
    tokens = low.replace("__", "_").split("_")
    if "l" in tokens or "left" in tokens or any(t.endswith("fl") or t.endswith("hl") or t.endswith("ml") for t in tokens):
        return "l"
    if "r" in tokens or "right" in tokens or any(t.endswith("fr") or t.endswith("hr") or t.endswith("mr") for t in tokens):
        return "r"
    if abs(x_coord) <= 0.25:
        return "c"
    return "l" if (x_coord * left_sign_x >= 0.0) else "r"


def analyze_morphology(program: Program, scene: Scene | None = None) -> MorphologyReport:
    """Deterministically extract `@morphology` roles, chains, posture, reach, CoM, and support polygon."""
    sc = scene if scene is not None else compile_geometry(program)
    if not sc.boxes or not sc.bones:
        return MorphologyReport(
            limbs=(),
            spine=(),
            neck=(),
            tail=(),
            wings=(),
            total_mass=0.0,
            com=(0.0, 0.0, 0.0),
            ground_y=0.0,
            forward_sign_z=1.0,
            body_span=(0.0, 0.0, 0.0),
            support_polygon_xz=(),
            symmetry_pairs=(),
        )

    bone_by_name: dict[str, WorldBone] = sc.bone_by_name
    children_by_bone: dict[str, list[str]] = {b.name: [] for b in sc.bones}
    for b in sc.bones:
        if b.parent_name is not None and b.parent_name in children_by_bone:
            children_by_bone[b.parent_name].append(b.name)

    boxes_by_bone: dict[str, list[WorldBox]] = {b.name: [] for b in sc.bones}
    all_mins: list[np.ndarray] = []
    all_maxs: list[np.ndarray] = []
    mass_sum = 0.0
    weighted_com = np.zeros(3, dtype=np.float64)
    bone_mass: dict[str, float] = {b.name: 0.0 for b in sc.bones}

    for bx in sc.boxes:
        boxes_by_bone.setdefault(bx.bone_name, []).append(bx)
        corners = bx.corners(include_inflate=True)
        all_mins.append(np.min(corners, axis=0))
        all_maxs.append(np.max(corners, axis=0))
        pos_dims = bx.size[bx.size > 1e-6]
        if getattr(bx, "plane_normal", None) is not None:
            area = float(np.prod(pos_dims)) if pos_dims.size >= 2 else 1.0
            vol = 0.1 * area
        else:
            vol = float(bx.size[0] * bx.size[1] * bx.size[2])
        mass_sum += vol
        weighted_com += vol * bx.center
        bone_mass[bx.bone_name] = bone_mass.get(bx.bone_name, 0.0) + vol

    g_min = np.min(np.vstack(all_mins), axis=0)
    g_max = np.max(np.vstack(all_maxs), axis=0)
    span = g_max - g_min
    ground_y = float(g_min[1])
    total_h = max(1.0, float(span[1]))
    com_vec = (weighted_com / max(1e-9, mass_sum)) if mass_sum > 0.0 else np.zeros(3, dtype=np.float64)

    # Detect which X sign corresponds to `_l` in this asset (defaults to -1.0 when `_l` is at X < 0, or +1.0 if X > 0)
    l_xs: list[float] = []
    for b in sc.bones:
        toks = b.name.lower().replace("__", "_").split("_")
        if ("l" in toks or "left" in toks or any(t in ("fl", "hl", "ml") for t in toks)) and abs(float(b.pivot_world[0])) > 0.25:
            l_xs.append(float(b.pivot_world[0]))
    left_sign_x = (1.0 if float(np.median(l_xs)) > 0.0 else -1.0) if l_xs else -1.0

    # Detect bilateral X-symmetry bone pairs
    symmetry_pairs: list[tuple[str, str]] = []
    paired_bones: set[str] = set()
    bones_list = list(sc.bones)
    for i, bi in enumerate(bones_list):
        if bi.name in paired_bones or abs(float(bi.pivot_world[0])) <= 0.15:
            continue
        best_j: WorldBone | None = None
        best_d = 0.5
        for j in range(i + 1, len(bones_list)):
            bj = bones_list[j]
            if bj.name in paired_bones:
                continue
            mir_p = np.array([-bi.pivot_world[0], bi.pivot_world[1], bi.pivot_world[2]], dtype=np.float64)
            d = float(np.linalg.norm(bj.pivot_world - mir_p))
            if d < best_d and abs(bone_mass.get(bi.name, 0.0) - bone_mass.get(bj.name, 0.0)) <= max(1.0, 0.25 * bone_mass.get(bi.name, 0.0)):
                best_d = d
                best_j = bj
        if best_j is not None:
            si = _infer_side_label(bi.name, float(bi.pivot_world[0]), left_sign_x)
            pair_tup = (bi.name, best_j.name) if si == "l" else (best_j.name, bi.name)
            symmetry_pairs.append(pair_tup)
            paired_bones.add(bi.name)
            paired_bones.add(best_j.name)
    mirror_partner_of: dict[str, str] = {}
    for bl, br in symmetry_pairs:
        mirror_partner_of[br] = bl
        mirror_partner_of[bl] = br

    # Infer forward Z direction (+1.0 if head/snout/neck is at +Z relative to CoM/tail, -1.0 if at -Z)
    fore_z_votes: list[float] = []
    for b in sc.bones:
        low = b.name.lower()
        dz = float(b.pivot_world[2] - com_vec[2])
        if any(k in low for k in ("head", "skull", "snout", "jaw", "neck", "cerv", "beak", "antler", "fore")):
            fore_z_votes.append(dz)
        elif any(k in low for k in ("tail", "caudal", "pelvis", "hind", "aft")):
            fore_z_votes.append(-dz)
    fwd_sign = 1.0
    if fore_z_votes:
        med_v = float(np.median(fore_z_votes))
        if abs(med_v) > 0.25:
            fwd_sign = 1.0 if med_v > 0.0 else -1.0

    def _subtree_bones(start: str) -> list[str]:
        acc = [start]
        for ch in children_by_bone.get(start, ()):
            acc.extend(_subtree_bones(ch))
        return acc

    def _subtree_bounds(start: str) -> tuple[np.ndarray, np.ndarray, bool]:
        sub_b = _subtree_bones(start)
        bxs = [bx for bn in sub_b for bx in boxes_by_bone.get(bn, ())]
        if not bxs:
            p = bone_by_name[start].pivot_world
            return p.copy(), p.copy(), False
        c_all = np.vstack([bx.corners(include_inflate=True) for bx in bxs])
        has_planes = any(getattr(bx, "plane_normal", None) is not None for bx in bxs)
        return np.min(c_all, axis=0), np.max(c_all, axis=0), has_planes

    def _longest_primary_chain(start: str) -> tuple[str, ...]:
        best: tuple[str, ...] = (start,)
        for ch in children_by_bone.get(start, ()):
            # Skip tiny accessory twigs (spurs/claws/feathers) if a deeper structural limb segment exists
            cand = (start, *_longest_primary_chain(ch))
            if len(cand) > len(best):
                best = cand
            elif len(cand) == len(best):
                # Prefer child whose pivot drops lower toward the ground (for legs) or extends farther
                if bone_by_name[cand[-1]].pivot_world[1] < bone_by_name[best[-1]].pivot_world[1]:
                    best = cand
        return best

    # Map declared IK chains by root bone for seamless reuse when present
    ik_by_root: dict[str, IKChainDecl] = {}
    if program.kinematics is not None:
        for ikc in program.kinematics.ik_chains:
            if ikc.bones:
                ik_by_root[ikc.bones[0]] = ikc

    # 1. Identify axial trunk (`spine`), `neck`, and `tail` along the central sagittal plane
    root_name = sc.bones[0].name
    center_tol = max(1.0, 0.16 * float(span[0]))
    axial_bones: set[str] = {root_name}
    q_ax = [root_name]
    while q_ax:
        cur = q_ax.pop(0)
        for ch in children_by_bone.get(cur, ()):
            cb = bone_by_name[ch]
            low = ch.lower()
            is_wing_or_limb_kw = any(
                k in low for k in ("leg", "arm", "wing", "femur", "humerus", "thigh", "shin", "paw", "foot", "claw", "horn", "ear", "gear", "wheel")
            )
            if abs(float(cb.pivot_world[0])) <= center_tol and ch not in paired_bones and not is_wing_or_limb_kw:
                axial_bones.add(ch)
                q_ax.append(ch)

    neck_list: list[str] = []
    tail_list: list[str] = []
    spine_list: list[str] = [root_name]

    for bname in sc.bones:
        if bname.name not in axial_bones or bname.name == root_name:
            continue
        low = bname.name.lower()
        dz_f = fwd_sign * float(bname.pivot_world[2] - bone_by_name[root_name].pivot_world[2])
        if any(k in low for k in ("tail", "caudal")):
            tail_list.append(bname.name)
        elif any(k in low for k in ("neck", "cerv", "head", "skull")):
            neck_list.append(bname.name)
        elif any(k in low for k in ("jaw", "chin", "crest", "fin", "turret", "gun", "barrel")):
            continue
        elif dz_f < -0.22 * float(span[2]) and not children_by_bone.get(bname.name):
            tail_list.append(bname.name)
        elif dz_f > 0.25 * float(span[2]) and float(bname.pivot_world[1]) > float(com_vec[1]):
            neck_list.append(bname.name)
        else:
            # Check if ancestor is already in neck or tail
            anc = bname.parent_name
            in_neck = False
            in_tail = False
            while anc is not None:
                if anc in neck_list:
                    in_neck = True
                    break
                if anc in tail_list:
                    in_tail = True
                    break
                anc = bone_by_name[anc].parent_name if anc in bone_by_name else None
            if in_neck:
                neck_list.append(bname.name)
            elif in_tail:
                tail_list.append(bname.name)
            elif any(
                abs(float(bone_by_name[c].pivot_world[0])) > center_tol for c in children_by_bone.get(bname.name, ())
            ) or any(k in low for k in ("torso", "chest", "waist", "lumbar", "pelvis", "thorax", "abdomen", "hull", "chassis")):
                spine_list.append(bname.name)

    # Sort spine from rear to front along forward_sign_z
    spine_sorted = tuple(sorted(spine_list, key=lambda n: fwd_sign * float(bone_by_name[n].pivot_world[2])))

    # 2. Identify lateral appendages branching from axial trunk / shoulder / hip mounts
    ground_tol = max(2.2, 0.14 * total_h)
    raw_limb_candidates: list[dict[str, Any]] = []
    raw_wings: list[WingMorphology] = []
    visited_subtrees: set[str] = set()

    for b in sc.bones:
        if b.name in axial_bones or b.name in visited_subtrees:
            continue
        if b.parent_name not in axial_bones:
            continue
        s_min, s_max, has_planes = _subtree_bounds(b.name)
        sub_all = _subtree_bones(b.name)
        low = b.name.lower()

        # Skip decorative horns/ears/exhausts/pistons/gears that don't touch ground
        reaches_ground = (float(s_min[1]) - ground_y) <= ground_tol
        declared_ik = ik_by_root.get(b.name)
        if declared_ik is None:
            # Check if an IK chain starts one bone below a static shoulder/hip girdle
            for ch_b in children_by_bone.get(b.name, ()):
                if ch_b in ik_by_root:
                    declared_ik = ik_by_root[ch_b]
                    break

        is_wing_kw = "wing" in low
        lat_span = float(s_max[0] - s_min[0])
        if (is_wing_kw and not reaches_ground) or (
            not reaches_ground and declared_ik is None and (has_planes or lat_span > 0.35 * float(span[0])) and float(b.pivot_world[1] - ground_y) > 0.45 * total_h
        ):
            chain_w = _longest_primary_chain(b.name)
            if len(chain_w) >= 2 or is_wing_kw:
                side_w = _infer_side_label(b.name, float(b.pivot_world[0]), left_sign_x)
                role_w = f"wing_{side_w}" if side_w in ("l", "r") else f"wing_{b.name}"
                mir_b = mirror_partner_of.get(b.name)
                mir_role = "wing_l" if (side_w == "r" and mir_b is not None) else None
                raw_wings.append(WingMorphology(role=role_w, chain=chain_w, span=lat_span, mirror_of=mir_role))
                visited_subtrees.update(sub_all)
                continue

        if not reaches_ground and declared_ik is None:
            continue
        if any(k in low for k in ("tail", "jaw", "horn", "gear", "flywheel", "turret", "piston")):
            continue

        if declared_ik is not None:
            chain = tuple(declared_ik.bones)
            tip_b = bone_by_name[chain[-1]]
            eff_loc = np.asarray(declared_ik.effector, dtype=np.float64)
            eff_w = tip_b.pivot_world + tip_b.rotation @ eff_loc
            ik_name: str | None = declared_ik.name
        else:
            chain = _longest_primary_chain(b.name)
            if len(chain) < 2:
                continue
            tip_b = bone_by_name[chain[-1]]
            tip_boxes = boxes_by_bone.get(tip_b.name, [])
            if tip_boxes:
                tip_corners = np.vstack([bx.corners(include_inflate=True) for bx in tip_boxes])
                min_y_tip = float(np.min(tip_corners[:, 1]))
                bottom_pts = tip_corners[tip_corners[:, 1] <= min_y_tip + 0.5]
                eff_w = np.array(
                    [float(np.mean(bottom_pts[:, 0])), min_y_tip, float(np.mean(bottom_pts[:, 2]))],
                    dtype=np.float64,
                )
            else:
                eff_w = np.array([float(tip_b.pivot_world[0]), float(s_min[1]), float(tip_b.pivot_world[2])], dtype=np.float64)
            eff_loc = tip_b.rotation.T @ (eff_w - tip_b.pivot_world)
            ik_name = None

        hip_b = bone_by_name[chain[0]]
        hip_w = hip_b.pivot_world.copy()
        hip_h = max(0.5, float(hip_w[1] - ground_y))

        seg_lens: list[float] = []
        for idx_c in range(1, len(chain)):
            p_prev = bone_by_name[chain[idx_c - 1]].pivot_world
            p_cur = bone_by_name[chain[idx_c]].pivot_world
            seg_lens.append(float(np.linalg.norm(p_cur - p_prev)))
        seg_lens.append(float(np.linalg.norm(eff_loc)))
        tot_len = max(0.5, sum(seg_lens))
        max_s = max(seg_lens) if seg_lens else tot_len
        min_reach = max(0.20 * hip_h, abs(2.0 * max_s - tot_len) + 0.05)
        max_reach = max(min_reach + 0.25, 0.98 * tot_len)

        lat_offset = abs(float(eff_w[0] - hip_w[0]))
        max_rest_bend = max(
            (float(np.max(np.abs(bone_by_name[cb].local_euler))) for cb in chain),
            default=0.0,
        )
        if lat_offset / max(1.0, hip_h) > 0.42:
            posture = "sprawled"
        elif (tot_len / max(1.0, hip_h)) > 1.12 or max_rest_bend >= 20.0:
            posture = "crouched"
        else:
            posture = "erect"

        side = _infer_side_label(chain[0], float(hip_w[0]), left_sign_x)
        raw_limb_candidates.append(
            {
                "chain": chain,
                "posture": posture,
                "hip_h": hip_h,
                "length": tot_len,
                "reach": (min_reach, max_reach),
                "contact_local": (float(eff_loc[0]), float(eff_loc[1]), float(eff_loc[2])),
                "contact_world": (float(eff_w[0]), float(eff_w[1]), float(eff_w[2])),
                "hip_world": (float(hip_w[0]), float(hip_w[1]), float(hip_w[2])),
                "side": side,
                "fore_z": fwd_sign * float(hip_w[2]),
                "ik_name": ik_name,
            }
        )
        visited_subtrees.update(sub_all)

    # 3. Group limbs into fore-to-aft bilateral pairs and assign canonical anatomical roles
    left_limbs = sorted([c for c in raw_limb_candidates if c["side"] == "l"], key=lambda d: -d["fore_z"])
    right_limbs = sorted([c for c in raw_limb_candidates if c["side"] == "r"], key=lambda d: -d["fore_z"])
    center_limbs = sorted([c for c in raw_limb_candidates if c["side"] == "c"], key=lambda d: -d["fore_z"])

    n_pairs = max(len(left_limbs), len(right_limbs))
    if n_pairs == 1:
        tier_names = ["hind"]
    elif n_pairs == 2:
        tier_names = ["fore", "hind"]
    elif n_pairs == 3:
        tier_names = ["fore", "mid", "hind"]
    else:
        tier_names = [
            "fore" if i == 0 else ("hind" if i == n_pairs - 1 else (f"mid" if i == 1 else f"mid{i}"))
            for i in range(n_pairs)
        ]

    limbs_out: list[LimbMorphology] = []
    for idx_p in range(n_pairs):
        t_prefix = tier_names[idx_p]
        has_l = idx_p < len(left_limbs)
        has_r = idx_p < len(right_limbs)
        if has_l:
            cl = left_limbs[idx_p]
            limbs_out.append(
                LimbMorphology(
                    role=f"{t_prefix}_l",
                    chain=cl["chain"],
                    posture=cl["posture"],
                    hip_height=cl["hip_h"],
                    length=cl["length"],
                    reach=cl["reach"],
                    contact_local=cl["contact_local"],
                    contact_world=cl["contact_world"],
                    hip_world=cl["hip_world"],
                    side="l",
                    mirror_of=None,
                    ik_chain_name=cl["ik_name"],
                )
            )
        if has_r:
            cr = right_limbs[idx_p]
            limbs_out.append(
                LimbMorphology(
                    role=f"{t_prefix}_r",
                    chain=cr["chain"],
                    posture=cr["posture"],
                    hip_height=cr["hip_h"],
                    length=cr["length"],
                    reach=cr["reach"],
                    contact_local=cr["contact_local"],
                    contact_world=cr["contact_world"],
                    hip_world=cr["hip_world"],
                    side="r",
                    mirror_of=f"{t_prefix}_l" if has_l else None,
                    ik_chain_name=cr["ik_name"],
                )
            )
    for idx_c, cc in enumerate(center_limbs):
        limbs_out.append(
            LimbMorphology(
                role=f"center_{idx_c + 1}",
                chain=cc["chain"],
                posture=cc["posture"],
                hip_height=cc["hip_h"],
                length=cc["length"],
                reach=cc["reach"],
                contact_local=cc["contact_local"],
                contact_world=cc["contact_world"],
                hip_world=cc["hip_world"],
                side="c",
                mirror_of=None,
                ik_chain_name=cc["ik_name"],
            )
        )

    support_pts = [(lb.contact_world[0], lb.contact_world[2]) for lb in limbs_out]
    hull_xz = _convex_hull_2d(support_pts)

    return MorphologyReport(
        limbs=tuple(limbs_out),
        spine=spine_sorted,
        neck=tuple(neck_list),
        tail=tuple(tail_list),
        wings=tuple(raw_wings),
        total_mass=float(mass_sum),
        com=(float(com_vec[0]), float(com_vec[1]), float(com_vec[2])),
        ground_y=ground_y,
        forward_sign_z=fwd_sign,
        body_span=(float(span[0]), float(span[1]), float(span[2])),
        support_polygon_xz=hull_xz,
        symmetry_pairs=tuple(symmetry_pairs),
    )


def format_morphology_block(report: MorphologyReport) -> str:
    """Serialize a derived `MorphologyReport` into the canonical read-only `@morphology { ... }` block."""
    lines = ["# Derived — never hand-edited", "@morphology {"]
    for lb in report.limbs:
        if lb.mirror_of is not None:
            lines.append(f"  limb {lb.role} {{ mirror_of: {lb.mirror_of} }}")
        else:
            ch_s = "[" + ", ".join(lb.chain) + "]"
            rc_s = f"[{lb.reach[0]:.2f}, {lb.reach[1]:.2f}]"
            ct_s = f"[{lb.contact_local[0]:.2f}, {lb.contact_local[1]:.2f}, {lb.contact_local[2]:.2f}]"
            lines.append(
                f"  limb {lb.role} {{ chain: {ch_s}, posture: {lb.posture}, "
                f"hip_height: {lb.hip_height:.2f}, length: {lb.length:.2f}, reach: {rc_s}, contact: {ct_s} }}"
            )
    if report.spine:
        lines.append("  spine [" + ", ".join(report.spine) + "]")
    if report.neck:
        lines.append("  neck [" + ", ".join(report.neck) + "]")
    if report.tail:
        lines.append("  tail [" + ", ".join(report.tail) + "]")
    for wg in report.wings:
        if wg.mirror_of is not None:
            lines.append(f"  wing {wg.role} {{ mirror_of: {wg.mirror_of} }}")
        else:
            lines.append(f"  wing {wg.role} {{ chain: [{', '.join(wg.chain)}] }}")
    poly_s = "[" + ", ".join(f"[{p[0]:.2f}, {p[1]:.2f}]" for p in report.support_polygon_xz) + "]"
    lines.append(
        f"  mass {{ total: {report.total_mass:.1f}, "
        f"com: [{report.com[0]:.2f}, {report.com[1]:.2f}, {report.com[2]:.2f}], "
        f"support_polygon: {poly_s} }}"
    )
    lines.append("}")
    return "\n".join(lines) + "\n"
