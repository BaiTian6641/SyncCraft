#!/usr/bin/env python3
"""Universal Blockbench 5 (.bbmodel) Rest-Pose & Animation Ground/Wing/Floating Verifier (v38).

Parses any exported `.bbmodel` directly (groups, outliner hierarchy, group origins/rest
rotations, element from/to/origin/rotation, and every animation clip's keyframed
position/rotation/scale channels) and evaluates exact world-space Y-min across:
  - Quadrupeds (`FL`, `FR`, `HL`, `HR`)
  - Bipeds & 2-legged Flying Wyverns (`HL`, `HR` + optional `FREE_ARM_L`/`FREE_ARM_R`)
  - 6-legged Hexapods & 8-legged Arachnids (token-boundary N-leg auto-discovery, zero `/leg_r` collisions!)
  - Winged flyers (`WING_L`, `WING_R` clearance)
  - Limbless floating / pelagic / hovering entities (`ALL` global solid Y-min + floor-plunge guard)

Usage:
    python3 scripts/verify_bbmodel_ground.py [--stance auto|quadruped|biped|arthropod|floating|hover] path/to/model.bbmodel
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _eval_channel(kfs: list[dict[str, Any]], t: float) -> np.ndarray:
    if not kfs:
        return np.zeros(3, dtype=np.float64)
    pts = sorted(kfs, key=lambda k: float(k["time"]))
    if t <= float(pts[0]["time"]):
        dp = pts[0]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    if t >= float(pts[-1]["time"]):
        dp = pts[-1]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    for i in range(len(pts) - 1):
        t0, t1 = float(pts[i]["time"]), float(pts[i + 1]["time"])
        if t0 <= t <= t1:
            alpha = (t - t0) / max(1e-9, t1 - t0)
            d0, d1 = pts[i]["data_points"][0], pts[i + 1]["data_points"][0]
            v0 = np.array([float(d0["x"]), float(d0["y"]), float(d0["z"])], dtype=np.float64)
            v1 = np.array([float(d1["x"]), float(d1["y"]), float(d1["z"])], dtype=np.float64)
            return v0 * (1.0 - alpha) + v1 * alpha
    return np.zeros(3, dtype=np.float64)


def _element_corners_local(el: dict[str, Any]) -> np.ndarray:
    f0 = np.array(el["from"], dtype=np.float64)
    f1 = np.array(el["to"], dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (f0[0], f1[0]) for y in (f0[1], f1[1]) for z in (f0[2], f1[2])],
        dtype=np.float64,
    )
    rot = el.get("rotation")
    if rot and any(abs(float(r)) > 1e-6 for r in rot):
        org = np.array(el.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        R = _euler_xyz_deg(float(rot[0]), float(rot[1]), float(rot[2]))
        corners = (corners - org) @ R.T + org
    return corners


_RE_MULTI_LEG = re.compile(r"^(?:leg|limb)_([fmr]\d*|\d+)_(l|r)(?:_|$)")
_RE_ARM_L = re.compile(r"^(?:arm|foreleg|front_leg)_l(?:_|$)")
_RE_ARM_R = re.compile(r"^(?:arm|foreleg|front_leg)_r(?:_|$)")
_RE_LEG_L = re.compile(r"^(?:leg|hindleg|back_leg)_l(?:_|$)")
_RE_LEG_R = re.compile(r"^(?:leg|hindleg|back_leg)_r(?:_|$)")
_RE_WING_L = re.compile(r"^(?:wing|hindwing|forewing|elytron)_l(?:_|$)")
_RE_WING_R = re.compile(r"^(?:wing|hindwing|forewing|elytron)_r(?:_|$)")


def _classify_limb(bone_path: list[str]) -> str | None:
    """Token-boundary limb/wing/tail classifier supporting 2-, 4-, 6-, and 8-legged creatures cleanly."""
    for seg in bone_path:
        s = seg.lower().split("__")[0]
        if "oral_arm" in s:
            continue
        m = _RE_MULTI_LEG.match(s)
        if m:
            slot, side = m.group(1).upper(), m.group(2).upper()
            return f"L_{slot}_{side}"
        if _RE_ARM_R.match(s):
            return "FR"
        if _RE_ARM_L.match(s):
            return "FL"
        if _RE_LEG_R.match(s):
            return "HR"
        if _RE_LEG_L.match(s):
            return "HL"
        if _RE_WING_R.match(s):
            return "WING_R"
        if _RE_WING_L.match(s):
            return "WING_L"
        if s == "tail" or s.startswith("tail_"):
            return "TAIL"
    return None


def verify_bbmodel_ground(
    bb_path: Path,
    *,
    rest_tol: float = 0.40,
    stance: str = "auto",
) -> dict[str, Any]:
    bb = json.loads(bb_path.read_text(encoding="utf-8"))
    el_by_uuid = {e["uuid"]: e for e in bb["elements"]}
    grp_by_uuid = {g["uuid"]: g for g in bb.get("groups", [])}

    def walk(
        nodes: list[Any],
        parent_R: np.ndarray,
        parent_t: np.ndarray,
        parent_s: np.ndarray,
        anim_pos: dict[str, np.ndarray],
        anim_rot: dict[str, np.ndarray],
        anim_scl: dict[str, np.ndarray],
        limb_ymins: dict[str, list[float]],
        all_ys: list[float],
        bone_path: list[str],
        bone_origin: np.ndarray,
    ) -> None:
        for nd in nodes:
            if isinstance(nd, str):
                el = el_by_uuid.get(nd)
                if el is None:
                    continue
                loc_c = (_element_corners_local(el) - bone_origin) * parent_s + bone_origin
                wc = loc_c @ parent_R.T + parent_t
                ymin_e = float(wc[:, 1].min())
                ymax_e = float(wc[:, 1].max())
                all_ys.append(ymin_e)
                all_ys.append(ymax_e)
                bucket = _classify_limb(bone_path)
                if bucket is not None:
                    limb_ymins.setdefault(bucket, []).append(ymin_e)
            elif isinstance(nd, dict):
                guid = nd.get("uuid", "")
                g = grp_by_uuid.get(guid, nd)
                gname = g.get("name", "")
                org = np.array(g.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                rest_r = g.get("rotation", [0.0, 0.0, 0.0])
                dpos = anim_pos.get(guid, np.zeros(3, dtype=np.float64))
                drot = anim_rot.get(guid, np.zeros(3, dtype=np.float64))
                dscl = anim_scl.get(guid, np.ones(3, dtype=np.float64))
                R_bone = _euler_xyz_deg(
                    float(rest_r[0]) + float(drot[0]),
                    float(rest_r[1]) + float(drot[1]),
                    float(rest_r[2]) + float(drot[2]),
                )
                cur_s = parent_s * dscl
                new_R = parent_R @ R_bone
                scaled_rel = bone_origin + (org + dpos - bone_origin) * parent_s
                new_t = parent_R @ (scaled_rel - R_bone @ org) + parent_t
                walk(
                    nd.get("children", []),
                    new_R,
                    new_t,
                    cur_s,
                    anim_pos,
                    anim_rot,
                    anim_scl,
                    limb_ymins,
                    all_ys,
                    bone_path + [gname],
                    org,
                )

    def eval_pose(
        anim_pos: dict[str, np.ndarray],
        anim_rot: dict[str, np.ndarray],
        anim_scl: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, float], tuple[float, float]]:
        ly: dict[str, list[float]] = {}
        all_ys: list[float] = []
        walk(bb["outliner"], np.eye(3), np.zeros(3), np.ones(3), anim_pos, anim_rot, anim_scl or {}, ly, all_ys, [], np.zeros(3))
        out = {k: round(min(v), 2) for k, v in ly.items() if v}
        bounds = (round(min(all_ys), 2), round(max(all_ys), 2)) if all_ys else (0.0, 0.0)
        return out, bounds

    rest_buckets, rest_all = eval_pose({}, {}, {})
    H_total = max(1e-3, rest_all[1] - rest_all[0])

    # Identify leg buckets vs wings/tail
    non_foot_keys = {"TAIL", "WING_L", "WING_R"}
    leg_keys = sorted(k for k in rest_buckets if k not in non_foot_keys)
    grounded_leg_keys = [k for k in leg_keys if abs(rest_buckets[k]) <= rest_tol]

    # Auto-detect skeletal stance class cleanly
    eff_stance = stance.lower()
    free_arm_keys: list[str] = []
    if eff_stance == "auto":
        if not leg_keys:
            eff_stance = "floating"
        elif not grounded_leg_keys and abs(rest_all[0]) <= rest_tol and all(rest_buckets[k] > max(1.8, 0.08 * H_total) for k in leg_keys):
            eff_stance = "hover"
        elif (
            "HL" in grounded_leg_keys
            and "HR" in grounded_leg_keys
            and "FL" in rest_buckets
            and "FR" in rest_buckets
            and min(rest_buckets["FL"], rest_buckets["FR"]) > max(2.0, 0.20 * H_total)
        ):
            eff_stance = "biped_with_free_arms"
            free_arm_keys = ["FL", "FR"]
        elif len(leg_keys) >= 6 and len(grounded_leg_keys) >= 4:
            eff_stance = f"arthropod_{len(grounded_leg_keys)}leg"
            free_arm_keys = [k for k in leg_keys if k not in grounded_leg_keys and rest_buckets[k] > max(2.0, 0.18 * H_total)]
        elif set(leg_keys) == {"HL", "HR"}:
            eff_stance = "biped_2foot"
        else:
            eff_stance = "quadruped_4foot"
    elif eff_stance == "biped":
        free_arm_keys = [k for k in ("FL", "FR") if k in leg_keys]

    check_foot_keys = [k for k in leg_keys if k not in free_arm_keys]
    if eff_stance in ("floating", "hover"):
        rest_ok = rest_all[0] >= -rest_tol
    elif check_foot_keys:
        rest_ok = all(abs(rest_buckets[k]) <= rest_tol for k in check_foot_keys)
    else:
        rest_ok = rest_all[0] >= -rest_tol

    display_rest = dict(rest_buckets)
    display_rest["ALL_min"] = rest_all[0]
    label = eff_stance.upper()
    print(f"=== BLOCKBENCH 5 REST-POSE {label} GROUND CHECK ({bb_path.name}) ===")
    print(f"  feet={display_rest}  PASS={rest_ok} (tol=±{rest_tol}u)")

    clip_summary: dict[str, Any] = {}
    anim_plunge_fail = False
    if bb.get("animations"):
        print("\n=== BLOCKBENCH 5 ANIMATION FK GROUND CHECK ===")
    for anim in bb.get("animations", []):
        cname = anim["name"]
        dur = float(anim.get("length", 1.0))
        animators = anim.get("animators", {})
        res: dict[str, list[float]] = {k: [] for k in list(rest_buckets.keys()) + ["ALL_min", "ALL_max"]}
        for i in range(13):
            t = dur * (i / 12.0)
            apos: dict[str, np.ndarray] = {}
            arot: dict[str, np.ndarray] = {}
            ascl: dict[str, np.ndarray] = {}
            for uid, adata in animators.items():
                kfs = adata.get("keyframes", [])
                pk = [k for k in kfs if k.get("channel") == "position"]
                rk = [k for k in kfs if k.get("channel") == "rotation"]
                sk = [k for k in kfs if k.get("channel") == "scale"]
                if pk:
                    apos[uid] = _eval_channel(pk, t)
                if rk:
                    arot[uid] = _eval_channel(rk, t)
                if sk:
                    ascl[uid] = _eval_channel(sk, t)
            ym, (amin, amax) = eval_pose(apos, arot, ascl)
            for k, val in ym.items():
                res.setdefault(k, []).append(val)
            res["ALL_min"].append(amin)
            res["ALL_max"].append(amax)
        clip_summary[cname] = {k: [round(min(v), 1), round(max(v), 1)] for k, v in res.items() if v}
        parts = [f"{k}=[{min(res[k]):5.1f}..{max(res[k]):5.1f}]" for k in check_foot_keys if res.get(k)]
        if not parts:
            parts.append(f"ALL=[{min(res['ALL_min']):5.1f}..{max(res['ALL_max']):5.1f}]")
        if res.get("WING_L") or res.get("WING_R"):
            wmin = min(res.get("WING_L", [999.0]) + res.get("WING_R", [999.0]))
            parts.append(f"WING_min={wmin:5.1f}")
        if res.get("TAIL"):
            parts.append(f"TAIL_min={min(res['TAIL']):5.1f}")
        if eff_stance in ("floating", "hover") and min(res["ALL_min"]) < -0.5:
            anim_plunge_fail = True
            parts.append("[FLOOR_PLUNGE!]")
        print(f"  {cname:30s} " + " ".join(parts))
    overall_ok = rest_ok and not anim_plunge_fail
    return {"stance": eff_stance, "rest": display_rest, "rest_ok": rest_ok, "overall_ok": overall_ok, "clips": clip_summary}


if __name__ == "__main__":
    args = sys.argv[1:]
    stance_arg = "auto"
    filtered: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--stance" and i + 1 < len(args):
            stance_arg = args[i + 1]
            i += 2
        elif args[i].startswith("--stance="):
            stance_arg = args[i].split("=", 1)[1]
            i += 1
        else:
            filtered.append(args[i])
            i += 1
    if not filtered:
        print("Usage: python3 verify_bbmodel_ground.py [--stance auto|quadruped|biped|arthropod|floating|hover] <model.bbmodel>")
        sys.exit(2)
    rep = verify_bbmodel_ground(Path(filtered[0]), stance=stance_arg)
    sys.exit(0 if rep["overall_ok"] else 1)
