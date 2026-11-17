"""src/synccraft/data_gen/mh_dragons_v8/dsl_primitives.py

Pure Geometric & Kinematic Authoring Primitives for Monumental Sculpted Blockbench Dragons.

Contains ZERO generic creature templates (no shared animal head/limb/torso stencils).
Every species in `mh_dragons_v8/species/` uses only low-level SE(3) spatial primitives
(`bone`, `box`, `span_box`, `tapered_arch`, `shingle_arc`, `cutout_panel`) so each
Monster Hunter creature has its own deliberate, unmistakable silhouette, muzzle ratio,
armor hierarchy, limb posture, and signature anatomy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def v3(p: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64)


def lerp3(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, t: float) -> np.ndarray:
    pa, pb = v3(a), v3(b)
    return pa + (pb - pa) * float(t)


def unit(vec: Sequence[float] | np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = v3(vec)
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return v3(fallback)
    return arr / n


def qhalf(val: float, minimum: float = 0.5) -> float:
    return max(minimum, round(float(val) * 2.0) / 2.0)


def basis_from_dir(
    direction: Sequence[float] | np.ndarray,
    up_hint: Sequence[float] | np.ndarray = (0.0, 1.0, 0.0),
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Right-handed orthonormal frame whose local +Z aligns with `direction`."""
    z_ax = unit(direction, (0.0, 0.0, 1.0))
    up = unit(up_hint, (0.0, 1.0, 0.0))
    if abs(float(np.dot(z_ax, up))) > 0.965:
        up = np.array([0.0, 0.0, 1.0]) if abs(z_ax[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    x_ax = unit(np.cross(up, z_ax), (1.0, 0.0, 0.0))
    y_ax = unit(np.cross(z_ax, x_ax), (0.0, 1.0, 0.0))
    if abs(roll_deg) > 1e-6:
        r = math.radians(roll_deg)
        cr, sr = math.cos(r), math.sin(r)
        x_ax, y_ax = cr * x_ax + sr * y_ax, -sr * x_ax + cr * y_ax
    return np.column_stack((x_ax, y_ax, z_ax))


def matrix_to_xyz_deg(rot: np.ndarray) -> list[float]:
    """Extract SyncCraft right-handed `Rz @ Ry @ Rx` Euler degrees `[rx, ry, rz]`."""
    sy = float(np.clip(-rot[2, 0], -1.0, 1.0))
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-6:
        rx = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        ry = math.asin(sy)
        rz = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        rx = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        ry = math.asin(sy)
        rz = 0.0
    return [round(math.degrees(rx), 3), round(math.degrees(ry), 3), round(math.degrees(rz), 3)]


class RigCanvas:
    """Low-level spatial canvas for composing unique sculpted Blockbench dragons in stages."""

    def __init__(
        self,
        monster_id: str,
        title: str,
        title_epithet: str,
        species_class: str,
        *,
        variant_kind: str = "canonical_flagship",
        parent_species_id: str | None = None,
        wiki_url: str = "",
        materials: dict[str, dict[str, Any]],
        notes: Sequence[str] = (),
        texels_per_unit: int = 2,
    ) -> None:
        self.monster_id = monster_id
        self._used_names: set[str] = set()
        self._bones_by_name: dict[str, dict[str, Any]] = {}
        self.doc: dict[str, Any] = {
            "id": monster_id,
            "title": title,
            "title_epithet": title_epithet,
            "species_class": species_class,
            "variant_kind": variant_kind,
            "parent_species_id": parent_species_id,
            "wiki_url": wiki_url,
            "texels_per_unit": int(texels_per_unit),
            "materials": materials,
            "bones": [],
            "elements": [],
            "stage1_elements": [],
            "wing_articulation": [],
            "notes": list(notes),
        }
        self.bone("root", None, (0.0, 0.0, 0.0))

    def _uname(self, base: str) -> str:
        clean = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in base).strip("_") or "part"
        if clean not in self._used_names:
            self._used_names.add(clean)
            return clean
        idx = 2
        while f"{clean}_{idx}" in self._used_names:
            idx += 1
        out = f"{clean}_{idx}"
        self._used_names.add(out)
        return out

    def bone(self, name: str, parent: str | None, pivot: Sequence[float]) -> str:
        p = [round(float(x), 4) for x in pivot]
        if name in self._bones_by_name:
            self._bones_by_name[name]["pivot"] = p
            return name
        entry = {"name": name, "parent": parent, "pivot": p}
        self._bones_by_name[name] = entry
        self.doc["bones"].append(entry)
        return name

    def box(
        self,
        name: str,
        bone: str,
        center: Sequence[float],
        size: Sequence[float],
        material: str,
        *,
        rotation: Sequence[float] = (0.0, 0.0, 0.0),
        face_materials: dict[str, str] | None = None,
        pattern: str = "",
        component: str = "surface",
        stage1: bool = False,
        tip_color: Sequence[int] | None = None,
        tip_start: float = 0.65,
        seam_color: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        uname = self._uname(name)
        sz = [qhalf(s, 0.5) for s in size]
        elem: dict[str, Any] = {
            "name": uname,
            "bone": bone,
            "center": [round(float(x), 4) for x in center],
            "size": sz,
            "material": material,
            "rotation": [round(float(r), 3) for r in rotation],
            "assembly_component": component,
            "inflate": 0.0,
        }
        if face_materials:
            elem["face_materials"] = dict(face_materials)
        if pattern:
            elem["pattern"] = pattern
        if tip_color is not None:
            elem["tip_color"] = [int(x) for x in tip_color]
            elem["tip_start"] = float(tip_start)
        if seam_color is not None:
            elem["seam_color"] = [int(x) for x in seam_color]
        self.doc["elements"].append(elem)
        if stage1:
            self.doc["stage1_elements"].append(dict(elem))
        return elem

    def span_box(
        self,
        name: str,
        bone: str,
        p_start: Sequence[float],
        p_end: Sequence[float],
        width: float,
        height: float,
        material: str,
        *,
        up_hint: Sequence[float] = (0.0, 1.0, 0.0),
        roll_deg: float = 0.0,
        len_pad: float = 0.8,
        face_materials: dict[str, str] | None = None,
        pattern: str = "",
        component: str = "surface",
        stage1: bool = False,
        tip_color: Sequence[int] | None = None,
        tip_start: float = 0.65,
        seam_color: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        a, b = v3(p_start), v3(p_end)
        delta = b - a
        seg_len = max(0.5, float(np.linalg.norm(delta)))
        mid = 0.5 * (a + b)
        rot_m = basis_from_dir(delta, up_hint=up_hint, roll_deg=roll_deg)
        return self.box(
            name,
            bone,
            mid,
            (width, height, seg_len + len_pad),
            material,
            rotation=matrix_to_xyz_deg(rot_m),
            face_materials=face_materials,
            pattern=pattern,
            component=component,
            stage1=stage1,
            tip_color=tip_color,
            tip_start=tip_start,
            seam_color=seam_color,
        )

    def tapered_arch(
        self,
        prefix: str,
        bone: str,
        pts: Sequence[Sequence[float]],
        wh_list: Sequence[tuple[float, float]],
        material: str,
        *,
        len_pad: float = 0.9,
        roll_deg: float = 0.0,
        ridge_mat: str | None = None,
        pattern: str = "horn_keratin",
        component: str = "horns",
        stage1: bool = False,
        tip_color: Sequence[int] | None = None,
        tip_start: float = 0.62,
        seam_color: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Chain arbitrary point & `(w, h)` stations cleanly into chisel-tapered Blockbench cuboids."""
        out: list[dict[str, Any]] = []
        n = len(pts) - 1
        for i in range(n):
            w = 0.5 * (wh_list[i][0] + wh_list[i + 1][0])
            h = 0.5 * (wh_list[i][1] + wh_list[i + 1][1])
            fm = {"up": ridge_mat} if ridge_mat else None
            out.append(
                self.span_box(
                    f"{prefix}_{i}",
                    bone,
                    pts[i],
                    pts[i + 1],
                    w,
                    h,
                    material,
                    roll_deg=roll_deg,
                    len_pad=len_pad if i < n - 1 else min(0.5, len_pad * 0.5),
                    face_materials=fm,
                    pattern=pattern,
                    component=component,
                    stage1=(stage1 and i == 0),
                    tip_color=tip_color if i == n - 1 else None,
                    tip_start=tip_start,
                    seam_color=seam_color,
                )
            )
        return out

    def shingle_arc(
        self,
        prefix: str,
        bone: str,
        root_start: Sequence[float],
        root_end: Sequence[float],
        dir_start: Sequence[float],
        dir_end: Sequence[float],
        *,
        count: int,
        length: float,
        width: float,
        thickness: float,
        material: str,
        alt_material: str | None = None,
        tip_mat: str | None = None,
        tip_color: Sequence[int] | None = None,
        seam_color: Sequence[int] | None = None,
        arch_curve: float = 0.30,
        roll_base: float = 12.0,
        bend_drop: float = -0.15,
        two_step: bool = True,
        pattern: str = "armor_shingle",
        component: str = "spines",
    ) -> list[dict[str, Any]]:
        """Lay deliberately irregular, overlapping 1- or 2-step Blockbench shingle planks along any contour."""
        out: list[dict[str, Any]] = []
        n = max(2, int(count))
        for i in range(n):
            u = i / max(1, n - 1)
            r0 = lerp3(root_start, root_end, u)
            d0 = unit(lerp3(dir_start, dir_end, u))
            asym = (1.0 - arch_curve * 0.4) + arch_curve * math.sin(math.pi * min(0.95, max(0.08, u)))
            asym *= 0.94 + 0.12 * ((i * 7 + 3) % 3) / 2.0  # organic irregular height!
            tot_l = length * asym
            wobble = 6.0 if (i % 2 == 0) else -5.0
            mat_use = alt_material if (alt_material and i % 2 == 1) else material
            w_cur = max(1.0, width * (0.95 + 0.10 * (i % 2)))
            if two_step and tot_l >= 7.0:
                mid = r0 + d0 * (tot_l * 0.56)
                d1 = unit(d0 + np.array([0.0, bend_drop, -0.14]))
                tip = mid + d1 * (tot_l * 0.50)
                out.append(
                    self.span_box(
                        f"{prefix}_{i}_a",
                        bone,
                        r0,
                        mid,
                        w_cur,
                        thickness,
                        mat_use,
                        roll_deg=roll_base + wobble,
                        len_pad=0.9,
                        pattern=pattern,
                        component=component,
                    )
                )
                out.append(
                    self.span_box(
                        f"{prefix}_{i}_b",
                        bone,
                        mid - d0 * 0.6,
                        tip,
                        max(1.0, w_cur * 0.78),
                        max(1.0, thickness * 0.80),
                        tip_mat or mat_use,
                        roll_deg=roll_base + wobble,
                        len_pad=0.6,
                        pattern=pattern,
                        component=component,
                        tip_color=tip_color,
                    )
                )
            else:
                tip = r0 + d0 * tot_l
                out.append(
                    self.span_box(
                        f"{prefix}_{i}",
                        bone,
                        r0,
                        tip,
                        w_cur,
                        thickness,
                        mat_use,
                        roll_deg=roll_base + wobble,
                        len_pad=0.8,
                        pattern=pattern,
                        component=component,
                        tip_color=tip_color,
                        seam_color=seam_color,
                    )
                )
        return out

    def cutout_panel(
        self,
        name: str,
        bone: str,
        points_world: Sequence[Sequence[float]],
        material: str,
        *,
        back_material: str | None = None,
        pattern: str = "membrane",
        component: str = "membrane",
        paint_uvs: Sequence[Sequence[float]] | None = None,
        edge_bleed: float = 0.55,
        serrated_hem: bool = False,
        tattered_holes: bool = False,
        tip_color: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        pts = np.asarray(points_world, dtype=np.float64)
        if len(pts) < 3:
            return None
        normal = None
        for i in range(1, len(pts) - 1):
            cand = np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
            if float(np.linalg.norm(cand)) > 1e-6:
                normal = unit(cand)
                break
        if normal is None:
            return None
        u_ax = unit(pts[1] - pts[0])
        v_ax = unit(np.cross(normal, u_ax))
        origin = pts.mean(axis=0)
        uv_2d = np.column_stack(((pts - origin) @ u_ax, (pts - origin) @ v_ax))
        lo = uv_2d.min(axis=0)
        hi = uv_2d.max(axis=0)
        span = np.maximum(1.0, np.ceil((hi - lo) * 2.0) * 0.5)
        mid_2d = 0.5 * (lo + hi)
        center_3d = origin + u_ax * mid_2d[0] + v_ax * mid_2d[1]
        norm_uv = np.clip((uv_2d - (mid_2d - 0.5 * span)) / span, 0.0, 1.0)
        rot_m = np.column_stack((u_ax, v_ax, normal))

        uname = self._uname(name)
        elem: dict[str, Any] = {
            "name": uname,
            "bone": bone,
            "center": [round(float(x), 4) for x in center_3d],
            "size": [round(float(span[0]), 4), round(float(span[1]), 4), 0.0],
            "material": material,
            "back_material": back_material or material,
            "rotation": matrix_to_xyz_deg(rot_m),
            "pattern": pattern,
            "assembly_component": component,
            "alpha": {
                "polygon": norm_uv.tolist(),
                "edge_bleed": float(edge_bleed),
                "serrated_hem": bool(serrated_hem),
                "tattered_holes": bool(tattered_holes),
            },
            "surface_vertices": pts.tolist(),
            "surface_uv": norm_uv.tolist(),
        }
        if paint_uvs is not None and len(paint_uvs) == len(pts):
            elem["paint_uvs"] = [list(map(float, uv)) for uv in paint_uvs]
        if tip_color is not None:
            elem["tip_color"] = [int(x) for x in tip_color]
        self.doc["elements"].append(elem)
        return elem

    def winged_sail_rig(
        self,
        side_sign: float,
        parent_bone: str,
        root_xyz: Sequence[float],
        elbow_xyz: Sequence[float],
        wrist_xyz: Sequence[float],
        finger_tips: Sequence[Sequence[float]],
        flank_anchor_xyz: Sequence[float],
        *,
        spar_mat: str,
        hub_mat: str,
        claw_mat: str,
        mem_mat: str,
        mem_back_mat: str,
        mem_pattern: str = "membrane",
        spar_w: float = 3.6,
        arm_w: float = 4.6,
        leading_scales: int = 5,
        tattered_outer: bool = True,
    ) -> dict[str, str]:
        """Build 3-hinge-bank (`wing_wrist`, `wing_mid`, `wing_distal`) non-uni-sheet wing bones & panels."""
        q = "l" if side_sign > 0 else "r"
        sh, el, wr, flank = v3(root_xyz), v3(elbow_xyz), v3(wrist_xyz), v3(flank_anchor_xyz)
        tips = [v3(t) for t in finger_tips]

        b_up = self.bone(f"wing_{q}", parent_bone, sh)
        b_fo = self.bone(f"wing_fore_{q}", b_up, el)
        b_wr = self.bone(f"wing_wrist_{q}", b_fo, wr)

        self.box(
            f"wing_shoulder_bulge_{q}",
            b_up,
            sh + np.array([-side_sign * 1.2, 0.8, 0.0]),
            (arm_w * 1.4, arm_w * 1.35, arm_w * 1.55),
            spar_mat,
            face_materials={"up": hub_mat},
            rotation=(11.25, side_sign * 11.25, 0.0),
            pattern="calm_flank",
            component="wing_arm",
            stage1=True,
        )
        self.span_box(f"wing_humerus_{q}", b_up, sh, el, arm_w * 1.15, arm_w * 1.10, spar_mat, face_materials={"up": hub_mat}, len_pad=2.0, pattern="calm_flank", component="wing_arm", stage1=True)
        self.span_box(f"wing_forearm_{q}", b_fo, el, wr, arm_w * 0.98, arm_w * 0.92, spar_mat, face_materials={"up": hub_mat}, len_pad=2.0, pattern="calm_flank", component="wing_arm", stage1=True)

        # Octagonal rotated elbow & wrist joint hubs
        for hub_name, hub_bone, hp_pt, r_off in (
            ("elbow_hub", b_fo, el, 0.0),
            ("wrist_hub", b_wr, wr, 15.0),
        ):
            for ang in (0.0, 45.0):
                self.box(
                    f"wing_{hub_name}_{q}_{int(ang)}",
                    hub_bone,
                    hp_pt,
                    (arm_w * 1.18, arm_w * 1.15, arm_w * 1.18),
                    hub_mat,
                    rotation=(r_off, side_sign * ang, 10.0),
                    pattern="armor_shingle",
                    component="wing_arm",
                )

        # Leading-edge overlapping scale shingles on forearm & humerus
        for li in range(leading_scales):
            u = (li + 0.5) / max(1, leading_scales)
            lp = lerp3(el, wr, u) + np.array([0.0, arm_w * 0.45, 0.9])
            self.box(
                f"wing_lead_scale_{q}_{li}",
                b_fo,
                lp,
                (arm_w * 0.72, 2.0, 4.5),
                hub_mat,
                rotation=(16.0, side_sign * 22.0, 0.0),
                pattern="armor_shingle",
                component="spines",
            )

        # Wrist alula thumb claw
        self.tapered_arch(
            f"wing_thumb_claw_{q}",
            b_wr,
            [
                wr + np.array([0.0, 0.8, 1.2]),
                wr + np.array([side_sign * 2.0, 2.2, 5.2]),
                wr + np.array([side_sign * 2.5, 0.2, 8.2]),
            ],
            [(2.8, 2.8), (2.0, 2.0), (1.0, 1.0)],
            claw_mat,
            component="claws",
        )

        l1_a = lerp3(wr, tips[0], 0.36)
        l1_b = lerp3(el, tips[-1], 0.38)
        l2_a = lerp3(wr, tips[0], 0.70)
        l2_b = lerp3(el, tips[-1], 0.72)

        b_mid = self.bone(f"wing_mid_{q}", b_wr, 0.5 * (l1_a + l1_b))
        b_dst = self.bone(f"wing_distal_{q}", b_mid, 0.5 * (l2_a + l2_b))

        grid: list[list[np.ndarray]] = []
        n_rays = len(tips)
        for rj, tip in enumerate(tips):
            u = rj / max(1, n_rays - 1)
            r0 = lerp3(wr, el, min(0.55, u * 0.46))
            r1 = lerp3(l1_a, l1_b, u)
            r2 = lerp3(l2_a, l2_b, u)
            r3 = tip
            grid.append([r0, r1, r2, r3])
            sw = max(1.8, spar_w * (1.0 - 0.10 * rj))
            self.span_box(f"wing_spar_p_{q}_{rj}", b_wr, r0, r1, sw, sw * 0.86, spar_mat, len_pad=1.4, pattern="armor_shingle", component="wing_spars", stage1=(rj in (0, n_rays - 1)))
            self.span_box(f"wing_spar_m_{q}_{rj}", b_mid, r1, r2, max(1.5, sw * 0.84), max(1.5, sw * 0.76), spar_mat, len_pad=1.2, pattern="armor_shingle", component="wing_spars", stage1=(rj == 0))
            self.span_box(f"wing_spar_d_{q}_{rj}", b_dst, r2, r3, max(1.2, sw * 0.70), max(1.2, sw * 0.65), claw_mat if rj == 0 else spar_mat, len_pad=0.8, pattern="horn_keratin", component="wing_spars")
            self.box(f"wing_knuckle_{q}_{rj}", b_mid, r1, (sw * 1.05, sw * 0.95, sw * 1.05), hub_mat, rotation=(15.0, side_sign * 30.0, 0.0), pattern="armor_shingle", component="wing_spars")

        # Continuous inner plagiopatagium bay from inner finger to body flank
        inner_mid = lerp3(grid[-1][0], grid[-1][3], 0.92)
        plagio = [wr, el, sh + np.array([0.0, -0.8, -1.5]), flank, inner_mid]
        if float(np.cross(plagio[1] - plagio[0], plagio[2] - plagio[0])[1] + np.cross(plagio[1] - plagio[0], plagio[2] - plagio[0])[2]) < 0.0:
            plagio = plagio[::-1]
        self.cutout_panel(
            f"wing_plagio_{q}",
            b_fo,
            plagio,
            mem_mat,
            back_material=mem_back_mat,
            pattern=mem_pattern,
            component="membrane",
            edge_bleed=1.35,
            serrated_hem=True,
        )

        # Height-stepped fan-overlapping membrane bays across all 3 articulated banks
        banks = (b_wr, b_mid, b_dst)
        patches: dict[str, list[str]] = {b_wr: [], b_mid: [], b_dst: []}
        for rj in range(n_rays - 1):
            ra, rb = grid[rj], grid[rj + 1]
            y_step = np.array([0.0, 0.35 * (rj - 1.0), 0.0])
            chord_dir = unit(rb[2] - ra[2])
            spar_dir = unit(ra[3] - ra[0])
            for bi, b_bank in enumerate(banks):
                ov_c = chord_dir * 1.85
                ov_s = spar_dir * (1.25 if bi < 2 else 0.0)
                p00 = ra[bi] - chord_dir * 0.45 - spar_dir * (0.85 if bi > 0 else 0.0) + y_step
                p01 = ra[bi + 1] - chord_dir * 0.45 + ov_s + y_step
                p11 = rb[bi + 1] + ov_c + ov_s + y_step
                p10 = rb[bi] + ov_c - spar_dir * (0.85 if bi > 0 else 0.0) + y_step
                quad = [p00, p01, p11, p10]
                uvs = [(bi / 3.0, 1.0), ((bi + 1) / 3.0, 1.0), ((bi + 1) / 3.0, 0.0), (bi / 3.0, 0.0)]
                nc = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                if float(nc[1] + nc[2]) < 0.0:
                    quad = [p01, p00, p10, p11]
                    uvs = [((bi + 1) / 3.0, 1.0), (bi / 3.0, 1.0), (bi / 3.0, 0.0), ((bi + 1) / 3.0, 0.0)]
                card = self.cutout_panel(
                    f"wing_sail_{q}_r{rj}_b{bi}",
                    b_bank,
                    quad,
                    mem_mat,
                    back_material=mem_back_mat,
                    pattern=mem_pattern,
                    component="membrane",
                    paint_uvs=uvs,
                    edge_bleed=1.35,
                    serrated_hem=(bi == 2 or rj == n_rays - 2),
                    tattered_holes=(tattered_outer and bi == 2 and rj <= 1),
                )
                if card is not None:
                    patches[b_bank].append(card["name"])

        self.doc["wing_articulation"].extend(
            [
                {"bone": b_wr, "parent": b_fo, "role": "proximal", "side": q, "patches": patches[b_wr]},
                {"bone": b_mid, "parent": b_wr, "role": "middle", "side": q, "patches": patches[b_mid]},
                {"bone": b_dst, "parent": b_mid, "role": "distal", "side": q, "patches": patches[b_dst]},
            ]
        )
        return {"upper": b_up, "fore": b_fo, "wrist": b_wr, "mid": b_mid, "distal": b_dst}
