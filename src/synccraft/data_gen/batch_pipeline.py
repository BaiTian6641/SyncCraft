"""Batch-by-Batch Dataset Generation, Statistical Process Control (SPC) QC & 3D Visual Contact-Sheet Renderer (ADR 0005).

Implements SyncCraft's scalable batch engineering workflow:
1. Stratified multi-batch generation across all 55 categorized families (40 Seen + 15 Unseen),
   4 progressive complexity tiers (simple -> masterwork), and ~5% controlled defect-repair mix.
2. Per-batch statistical quality gates (SAT/P1-02 pass rate, G2 >=16-box unseen floor,
   QD behavioral entropy, SHA-256 uniqueness, controlled-defect calibration, animation swept-SAT).
3. Exact per-pixel Z-buffered 3D orthographic textured OBB renderer + 2D Pixel-PBR/Turing atlas
   contact-sheet generator for random-sample human & VLM subagent visual inspection.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from synccraft.dsl.ast_nodes import Program
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from synccraft.shading.compiler import compile_vanilla_atlas
from synccraft.surface.contracts import (
    EnvironmentState,
    MaterialProgram,
    StyleConfig,
    WorldState,
)
from synccraft.surface.domain import (
    FACE_AXES_BOX,
    FACES,
    build_surface_domain,
    initialize_surface_state,
)
from synccraft.surface.processes import integrate_gray_scott_on_state
from .curriculum_and_assistant import apply_animation_keyframe
from .diversity_engine import compute_program_tree_shape
from .taxonomy_catalog import HOLDOUT_EVALUATION_SPLITS


def extract_animation_spec_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract structured animation specification from row metadata or typed `@behavior` AST."""
    prov = row.get("prompt_and_provenance", {})
    if isinstance(prov.get("animation_plan"), dict) and prov["animation_plan"].get("channels"):
        return prov["animation_plan"]
    if isinstance(prov.get("animation_plan_json"), str) and prov["animation_plan_json"].strip():
        try:
            parsed_plan = json.loads(prov["animation_plan_json"])
            if isinstance(parsed_plan, dict) and parsed_plan.get("channels"):
                return parsed_plan
        except Exception:
            pass
    dsl_text = row.get("canonical_dsl") or prov.get("assistant_completion") or ""
    if "@behavior" in dsl_text:
        try:
            prog = parse_program(dsl_text)
            if prog.behavior and prog.behavior.clips:
                clip = prog.behavior.clips[0]
                return {
                    "clip_name": clip.name,
                    "duration_s": clip.duration_s,
                    "loop": clip.loop,
                    "channels": [
                        {
                            "bone": tr.bone,
                            "wave": tr.law,
                            "axis": tr.axis,
                            "amp_deg": tr.amp_deg,
                            "freq_hz": tr.freq_hz,
                            "phase_deg": tr.phase_deg,
                            "bias_deg": tr.bias_deg,
                        }
                        for tr in clip.tracks
                    ],
                    "keyframe_tracks": [
                        {
                            "bone": kf.bone,
                            "channel": kf.channel,
                            "interpolation": kf.interpolation,
                            "keys": [[float(t), list(v)] for t, v in kf.keys],
                        }
                        for kf in clip.keyframes
                    ],
                }
        except Exception:
            return None
    return None


def _build_camera_basis(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-handed camera basis (cam_right, cam_up, cam_forward_toward_viewer)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    # Direction from origin toward viewer
    fwd = np.array(
        [
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ],
        dtype=np.float64,
    )
    fwd /= max(1e-12, float(np.linalg.norm(fwd)))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, fwd)
    r_norm = float(np.linalg.norm(right))
    if r_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= r_norm
    up = np.cross(fwd, right)
    up /= max(1e-12, float(np.linalg.norm(up)))
    return right, up, fwd


def _compile_program_atlas_rgba(
    program: Program,
    *,
    seed: int = 101,
    subdivisions_per_unit: int = 4,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compile stateful GGX + Cavity-AO + Gray-Scott Turing RGBA atlas for ``program``."""
    from dataclasses import replace as _dc_replace

    mat_prog = MaterialProgram.from_program(program, seed=seed)
    # Use concise rd_initial_time=10.0 for crisp Turing spot/stripe bloom at fast batch QC speed
    for m_name, spec in list(mat_prog.materials.items()):
        if spec.family == "patterned_skin":
            mat_prog = mat_prog.with_material(_dc_replace(spec, rd_initial_time=10.0))

    style_cfg = StyleConfig.from_program(program)
    domain = build_surface_domain(program, mat_prog, subdivisions_per_unit=subdivisions_per_unit)
    surf_state = initialize_surface_state(domain)

    world = WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=surf_state,
        style_config=style_cfg,
        environment=EnvironmentState(),
    )
    atlas_desc, png_bytes = compile_vanilla_atlas(world)
    atlas_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return atlas_desc, np.asarray(atlas_img, dtype=np.uint8)


_OPPOSITE_FACE: dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "up": "down",
    "down": "up",
}


def _rasterize_scene_with_atlas(
    scene: Scene,
    atlas_desc: dict[str, Any],
    atlas_rgba: np.ndarray,
    *,
    canvas_size: tuple[int, int] = (220, 220),
    azimuth_deg: float = 38.0,
    elevation_deg: float = 25.0,
    bg_rgb: tuple[int, int, int] = (24, 28, 38),
    camera_bounds: tuple[float, float, float, float] | None = None,
    draw_contours: bool = True,
) -> Image.Image:
    """Per-pixel Z-buffered orthographic 3D OBB rasterizer with alpha cutout, translucency, signed UVs & UV rotation."""
    w_img, h_img = canvas_size
    cam_right, cam_up, cam_fwd = _build_camera_basis(azimuth_deg, elevation_deg)

    if camera_bounds is not None:
        min_x, max_x, min_y, max_y = camera_bounds
    else:
        all_pts_cam: list[np.ndarray] = []
        unit_signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        for wb in scene.boxes:
            local_corners = unit_signs * wb.half_extents[None, :]
            world_corners = wb.center[None, :] + local_corners @ wb.rotation.T
            xs = world_corners @ cam_right
            ys = world_corners @ cam_up
            zs = world_corners @ cam_fwd
            all_pts_cam.append(np.stack([xs, ys, zs], axis=-1))

        if not all_pts_cam:
            return Image.new("RGB", canvas_size, bg_rgb)

        pts = np.concatenate(all_pts_cam, axis=0)
        min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
        min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)
    margin = 0.84
    scale = min((w_img * margin) / span_x, (h_img * margin) / span_y)
    cx_cam = 0.5 * (min_x + max_x)
    cy_cam = 0.5 * (min_y + max_y)

    color_buf = np.empty((h_img, w_img, 3), dtype=np.uint8)
    color_buf[:, :, 0] = bg_rgb[0]
    color_buf[:, :, 1] = bg_rgb[1]
    color_buf[:, :, 2] = bg_rgb[2]

    # Subtle studio grid lines in background for CAD/Blockbench readability
    for gy in range(0, h_img, 22):
        color_buf[gy, :] = (31, 36, 48)
    for gx in range(0, w_img, 22):
        color_buf[:, gx] = (31, 36, 48)

    z_buf = np.full((h_img, w_img), -1e18, dtype=np.float64)
    faces_uv = atlas_desc["faces"]
    at_h, at_w = atlas_rgba.shape[:2]

    translucent_queue: list[tuple[float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for wb in scene.boxes:
        box_uv = faces_uv.get(wb.name, {})
        ast_uv_faces = getattr(wb, "uv_faces", None) or {}
        he = wb.half_extents
        rot = wb.rotation

        for face_name in FACES:
            t_right_b, t_down_b, n_out_b = FACE_AXES_BOX[face_name]
            t_r = np.asarray(t_right_b, dtype=np.float64)
            t_d = np.asarray(t_down_b, dtype=np.float64)
            n_o = np.asarray(n_out_b, dtype=np.float64)

            d_n = float(np.abs(n_o) @ he)
            d_u = float(np.abs(t_r) @ he)
            d_v = float(np.abs(t_d) @ he)
            if d_u <= 1e-9 or d_v <= 1e-9:
                continue

            # Honor explicitly omitted/disabled faces on authored models
            if ast_uv_faces and face_name not in ast_uv_faces:
                continue
            if (wb.name in faces_uv) and (face_name not in box_uv):
                continue

            n_world = rot @ n_o
            ndotv = float(np.dot(n_world, cam_fwd))
            if abs(ndotv) <= 1e-6:
                continue
            if ndotv < 0.0:
                # Only render backfaces when the opposite face is omitted or 100% alpha-transparent
                # (single-face-painted cutouts like glTF foliage), preserving strict single-sided
                # culling on two-sided Blockbench/Minecraft planes (e.g. Charizard wing top vs bottom).
                opp_map = {
                    "north": "south",
                    "south": "north",
                    "east": "west",
                    "west": "east",
                    "up": "down",
                    "down": "up",
                }
                opp_fn = opp_map[face_name]
                opp_has_opaque_art = False
                if opp_fn in box_uv:
                    opp_uv = box_uv[opp_fn]
                    ou0 = int(max(0, min(at_w - 1, math.floor(min(float(opp_uv[0]), float(opp_uv[2]))))))
                    ou1 = int(max(0, min(at_w, math.ceil(max(float(opp_uv[0]), float(opp_uv[2]))))))
                    ov0 = int(max(0, min(at_h - 1, math.floor(min(float(opp_uv[1]), float(opp_uv[3]))))))
                    ov1 = int(max(0, min(at_h, math.ceil(max(float(opp_uv[1]), float(opp_uv[3]))))))
                    if ou1 > ou0 and ov1 > ov0 and int(atlas_rgba[ov0:ov1, ou0:ou1, 3].max()) > 8:
                        opp_has_opaque_art = True
                elif wb.name not in faces_uv and not ast_uv_faces:
                    opp_has_opaque_art = True
                if opp_has_opaque_art:
                    continue
                ndotv = -ndotv

            # Top-left UV origin corner (u=0, v=0) + spanning vectors E_u (u->1), E_v (v->1)
            p00_local = n_o * d_n - t_r * d_u - t_d * d_v
            eu_local = 2.0 * d_u * t_r
            ev_local = 2.0 * d_v * t_d

            p00_w = wb.center + rot @ p00_local
            eu_w = rot @ eu_local
            ev_w = rot @ ev_local

            # Project to screen pixels (x right, y down) and depth z toward viewer
            x0 = (float(np.dot(p00_w, cam_right)) - cx_cam) * scale + 0.5 * w_img
            y0 = -(float(np.dot(p00_w, cam_up)) - cy_cam) * scale + 0.5 * h_img
            z0 = float(np.dot(p00_w, cam_fwd))

            eux = float(np.dot(eu_w, cam_right)) * scale
            euy = -float(np.dot(eu_w, cam_up)) * scale
            euz = float(np.dot(eu_w, cam_fwd))

            evx = float(np.dot(ev_w, cam_right)) * scale
            evy = -float(np.dot(ev_w, cam_up)) * scale
            evz = float(np.dot(ev_w, cam_fwd))

            det = eux * evy - euy * evx
            if abs(det) < 1e-5:
                continue
            inv_det = 1.0 / det

            xs_quad = [x0, x0 + eux, x0 + evx, x0 + eux + evx]
            ys_quad = [y0, y0 + euy, y0 + evy, y0 + euy + evy]
            px_min = max(0, int(math.floor(min(xs_quad))))
            px_max = min(w_img - 1, int(math.ceil(max(xs_quad))))
            py_min = max(0, int(math.floor(min(ys_quad))))
            py_max = min(h_img - 1, int(math.ceil(max(ys_quad))))
            if px_min > px_max or py_min > py_max:
                continue

            gx, gy = np.meshgrid(
                np.arange(px_min, px_max + 1, dtype=np.float64) + 0.5,
                np.arange(py_min, py_max + 1, dtype=np.float64) + 0.5,
            )
            dx = gx - x0
            dy = gy - y0
            u_frac = (dx * evy - dy * evx) * inv_det
            v_frac = (eux * dy - euy * dx) * inv_det

            inside = (u_frac >= -1e-4) & (u_frac <= 1.0001) & (v_frac >= -1e-4) & (v_frac <= 1.0001)
            if not np.any(inside):
                continue

            u_c = np.clip(u_frac, 0.0, 1.0)
            v_c = np.clip(v_frac, 0.0, 1.0)

            raw_uv = box_uv.get(face_name, [0.0, 0.0, 1.0, 1.0])
            u0_uv, v0_uv, u1_uv, v1_uv = float(raw_uv[0]), float(raw_uv[1]), float(raw_uv[2]), float(raw_uv[3])
            uv_rot = int(raw_uv[4]) % 360 if len(raw_uv) >= 5 else 0
            if uv_rot == 0 and ast_uv_faces and face_name in ast_uv_faces and len(ast_uv_faces[face_name]) >= 5:
                uv_rot = int(ast_uv_faces[face_name][4]) % 360

            if uv_rot == 90:
                s_u, s_v = v_c, 1.0 - u_c
            elif uv_rot == 180:
                s_u, s_v = 1.0 - u_c, 1.0 - v_c
            elif uv_rot == 270:
                s_u, s_v = 1.0 - v_c, u_c
            else:
                s_u, s_v = u_c, v_c

            s_u_safe = np.clip(s_u, 1e-5, 1.0 - 1e-5)
            s_v_safe = np.clip(s_v, 1e-5, 1.0 - 1e-5)
            uu = np.clip(np.floor(u0_uv + s_u_safe * (u1_uv - u0_uv)).astype(np.int32), 0, at_w - 1)
            vv = np.clip(np.floor(v0_uv + s_v_safe * (v1_uv - v0_uv)).astype(np.int32), 0, at_h - 1)

            texel_rgba = atlas_rgba[vv, uu]
            alpha = texel_rgba[:, :, 3]
            sampled = texel_rgba[:, :, :3].astype(np.float64)
            shade = 0.78 + 0.26 * ndotv
            shaded = np.clip(sampled * shade, 0.0, 255.0).astype(np.uint8)

            z_pix = z0 + u_frac * euz + v_frac * evz
            sub_z = z_buf[py_min : py_max + 1, px_min : px_max + 1]

            opaque_mask = inside & (alpha >= 200) & (z_pix > sub_z)
            if np.any(opaque_mask):
                if draw_contours:
                    edge_px = 0.95
                    u_edge = np.minimum(u_frac, 1.0 - u_frac) * math.hypot(eux, euy)
                    v_edge = np.minimum(v_frac, 1.0 - v_frac) * math.hypot(evx, evy)
                    on_border = (np.minimum(u_edge, v_edge) < edge_px) & opaque_mask
                    shaded_op = shaded.copy()
                    shaded_op[on_border] = (shaded_op[on_border].astype(np.uint16) * 55 // 100).astype(np.uint8)
                else:
                    shaded_op = shaded
                sub_z[opaque_mask] = z_pix[opaque_mask]
                sub_col = color_buf[py_min : py_max + 1, px_min : px_max + 1]
                sub_col[opaque_mask] = shaded_op[opaque_mask]

            trans_cand = inside & (alpha >= 20) & (alpha < 200)
            if np.any(trans_cand):
                mean_z = float(np.mean(z_pix[trans_cand]))
                translucent_queue.append(
                    (mean_z, py_min, py_max, px_min, px_max, trans_cand, z_pix, shaded, alpha)
                )

    # Back-to-front translucent alpha-blending pass (e.g. display cases, fossil glass, slime, crystals)
    if translucent_queue:
        translucent_queue.sort(key=lambda item: item[0])
        for _, py_min, py_max, px_min, px_max, trans_cand, z_pix, shaded, alpha in translucent_queue:
            sub_z = z_buf[py_min : py_max + 1, px_min : px_max + 1]
            vis_t = trans_cand & (z_pix > sub_z)
            if not np.any(vis_t):
                continue
            sub_col = color_buf[py_min : py_max + 1, px_min : px_max + 1]
            a_f = (alpha[vis_t].astype(np.float64) / 255.0)[:, None]
            blended = np.clip(
                shaded[vis_t].astype(np.float64) * a_f + sub_col[vis_t].astype(np.float64) * (1.0 - a_f),
                0.0,
                255.0,
            ).astype(np.uint8)
            sub_col[vis_t] = blended

    return Image.fromarray(color_buf, mode="RGB")


def render_row_visual_triptych(
    row: dict[str, Any],
    *,
    view_size: tuple[int, int] = (210, 210),
    atlas_panel_size: tuple[int, int] = (176, 210),
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render (rest_3d_img, animated_pose_3d_img, atlas_2d_img) for a verified dataset row."""
    from .curriculum_and_assistant import ensure_concise_texture_program

    raw_prog = parse_program(row["canonical_dsl"])
    prog, _ = ensure_concise_texture_program(
        raw_prog,
        family_id=row["family_id"],
        rng=random.Random(int(row.get("seed", 101)) ^ 0x4411),
    )
    rest_scene = compile_geometry(prog)
    atlas_desc, atlas_rgba = _compile_program_atlas_rgba(
        prog,
        seed=int(row.get("seed", 101)) % 100_000,
        subdivisions_per_unit=4,
    )

    rest_3d = _rasterize_scene_with_atlas(
        rest_scene,
        atlas_desc,
        atlas_rgba,
        canvas_size=view_size,
        azimuth_deg=36.0,
        elevation_deg=24.0,
    )

    posed_scene: Scene | None = None
    if prog.behavior and prog.behavior.clips:
        try:
            from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip

            # Pick the clip with the most animated channels/keyframes and sample the most expressive pose frame
            best_clip = max(
                prog.behavior.clips,
                key=lambda c: len(c.tracks) + len(c.keyframes) + len(c.muscle_drives) + len(c.ik_drives),
            )
            frames = evaluate_behavior_clip(prog, best_clip.name, num_frames=5)
            best_disp = -1.0
            for fr in frames[1:]:
                disp = float(
                    sum(
                        np.linalg.norm(fr.scene.box_by_name[b.name].center - b.center)
                        + 4.0 * np.linalg.norm(fr.scene.box_by_name[b.name].rotation - b.rotation)
                        for b in rest_scene.boxes
                        if b.name in fr.scene.box_by_name
                    )
                )
                if disp > best_disp:
                    best_disp = disp
                    posed_scene = fr.scene
        except Exception:
            posed_scene = None

    if posed_scene is None:
        anim_spec = extract_animation_spec_from_row(row)
        if anim_spec and anim_spec.get("channels"):
            posed_prog = apply_animation_keyframe(prog, anim_spec, 0.25)
            posed_scene = compile_geometry(posed_prog)
        else:
            posed_scene = rest_scene

    posed_3d = _rasterize_scene_with_atlas(
        posed_scene,
        atlas_desc,
        atlas_rgba,
        canvas_size=view_size,
        azimuth_deg=-30.0,
        elevation_deg=22.0,
    )

    # Nearest-neighbor fit of the 2D RGBA UV atlas over a dark checkerboard canvas
    aw, ah = atlas_panel_size
    checker = np.zeros((ah, aw, 3), dtype=np.uint8)
    yy, xx = np.indices((ah, aw))
    even = ((xx // 8) + (yy // 8)) % 2 == 0
    checker[even] = (30, 34, 44)
    checker[~even] = (20, 23, 31)
    atlas_bg = Image.fromarray(checker, mode="RGB")

    raw_atlas_pil = Image.fromarray(atlas_rgba, mode="RGBA")
    # Crop tight around occupied UV islands so small models fill the preview cleanly
    alpha_mask = atlas_rgba[:, :, 3] > 0
    if np.any(alpha_mask):
        ys_occ, xs_occ = np.where(alpha_mask)
        crop_box = (
            max(0, int(xs_occ.min()) - 2),
            max(0, int(ys_occ.min()) - 2),
            min(atlas_rgba.shape[1], int(xs_occ.max()) + 3),
            min(atlas_rgba.shape[0], int(ys_occ.max()) + 3),
        )
        raw_atlas_pil = raw_atlas_pil.crop(crop_box)

    src_w, src_h = raw_atlas_pil.size
    fit_scale = min((aw - 12) / max(1, src_w), (ah - 12) / max(1, src_h))
    dst_w = max(1, int(round(src_w * fit_scale)))
    dst_h = max(1, int(round(src_h * fit_scale)))
    resized_atlas = raw_atlas_pil.resize((dst_w, dst_h), resample=Image.Resampling.NEAREST)
    ox = (aw - dst_w) // 2
    oy = (ah - dst_h) // 2
    atlas_bg.paste(resized_atlas, (ox, oy), resized_atlas)

    return rest_3d, posed_3d, atlas_bg


def _ascii_safe_label(text: str) -> str:
    return text.replace("\u2014", "--").replace("\u2013", "-").replace("\u2192", "->")


def render_batch_inspection_sheet(
    sampled_rows: list[dict[str, Any]],
    out_png_path: Path | str,
    *,
    title: str = "SyncCraft v3.0 Batch Visual QC -- Rest 3D | Animated Pose (t=0.25T) | 2D PBR/Turing Atlas",
    columns: int = 2,
) -> Path:
    """Render a multi-card visual inspection contact sheet PNG for random batch QC."""
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cards = max(1, len(sampled_rows))
    cols = max(1, min(columns, n_cards))
    rows_grid = int(math.ceil(n_cards / cols))

    card_w = 626
    card_h = 286
    pad = 16
    header_h = 54
    sheet_w = pad + cols * (card_w + pad)
    sheet_h = header_h + pad + rows_grid * (card_h + pad)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (15, 18, 25))
    draw = ImageDraw.Draw(sheet)

    # Top title banner
    draw.rectangle((0, 0, sheet_w, header_h), fill=(22, 27, 38))
    draw.text((18, 12), _ascii_safe_label(title), fill=(232, 240, 252))
    draw.text(
        (18, 31),
        f"Sampled assets: {len(sampled_rows)} | Shared SE(3) + 15-Axis SAT + P1-02 Contact + Stateful OKLCH/GGX/Turing Compiler",
        fill=(148, 165, 192),
    )

    for idx, row in enumerate(sampled_rows):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = pad + c_idx * (card_w + pad)
        y0 = header_h + pad + r_idx * (card_h + pad)

        gp = row.get("generator_parameters", {})
        aud = row.get("audit_version_and_results", {})
        quality_label = gp.get("quality_label", "verified_clean")
        border_col = (218, 146, 54) if quality_label != "verified_clean" else (58, 76, 108)

        draw.rectangle((x0, y0, x0 + card_w, y0 + card_h), fill=(20, 24, 34), outline=border_col, width=2)

        rest_3d, posed_3d, atlas_2d = render_row_visual_triptych(row)
        vy = y0 + 42
        sheet.paste(rest_3d, (x0 + 10, vy))
        sheet.paste(posed_3d, (x0 + 228, vy))
        sheet.paste(atlas_2d, (x0 + 444, vy))

        # Panel captions
        draw.text((x0 + 14, vy + 4), "Rest Pose (36 deg)", fill=(185, 205, 232))
        draw.text((x0 + 232, vy + 4), "Anim t=0.25T (-30 deg)", fill=(168, 230, 195))
        draw.text((x0 + 448, vy + 4), "2D UV PBR/RD Atlas", fill=(235, 208, 158))

        fam = row["family_id"]
        dom = gp.get("domain", "artificial")
        cat = gp.get("category", "general")
        tier = gp.get("complexity_tier", "tier_3_complex")
        motion = gp.get("motion_archetype", "hinge")
        n_bx = aud.get("boxes", 0)
        n_bn = aud.get("bones", 0)
        mode = gp.get("assistant_task_mode", "create_full_asset")

        draw.text(
            (x0 + 10, y0 + 8),
            _ascii_safe_label(f"[{dom.upper()}/{cat}] {fam}  ({row['split']})"),
            fill=(240, 244, 255),
        )
        draw.text(
            (x0 + 10, y0 + 23),
            _ascii_safe_label(f"tier={tier} | {n_bx} boxes, {n_bn} bones | motion={motion} | mode={mode}"),
            fill=(156, 176, 206),
        )
        badge_text = (
            f"REPAIR_PAIR [{quality_label}]"
            if quality_label != "verified_clean"
            else f"SAT+P1-02 CLEAN [{quality_label}]"
        )
        badge_col = (242, 174, 78) if quality_label != "verified_clean" else (98, 218, 154)
        draw.text(
            (x0 + 10, y0 + card_h - 24),
            _ascii_safe_label(f"QC: {badge_text} | {gp.get('material_theme', '')}"),
            fill=badge_col,
        )

    sheet.save(out_path, format="PNG")
    return out_path


def compute_batch_statistical_qc(
    batch_rows: list[dict[str, Any]],
    *,
    batch_index: int = 0,
    max_bad_data_ratio: float = 0.12,
) -> dict[str, Any]:
    """Evaluate statistical process control (SPC) metrics and acceptance gates for a batch."""
    if not batch_rows:
        raise ValueError("Cannot compute statistical QC on an empty batch")

    total = len(batch_rows)
    splits: dict[str, int] = {}
    domains: dict[str, int] = {}
    categories: dict[str, int] = {}
    tiers: dict[str, int] = {}
    modes: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    qd_counts: dict[str, int] = {}
    seen_hashes: set[str] = set()

    boxes = np.array([int(r["audit_version_and_results"]["boxes"]) for r in batch_rows], dtype=np.float64)
    bones = np.array([int(r["audit_version_and_results"]["bones"]) for r in batch_rows], dtype=np.float64)

    sat_passed = 0
    unseen_g2_violations = 0
    anim_verified_count = 0
    defect_rows = 0
    defect_repaired_ok = 0
    comment_free_count = 0
    tree_shapes_by_family: dict[str, set[str]] = {}

    for r in batch_rows:
        sp = r["split"]
        splits[sp] = splits.get(sp, 0) + 1
        gp = r["generator_parameters"]
        aud = r["audit_version_and_results"]
        prov = r.get("prompt_and_provenance", {})

        dom = str(gp.get("domain", "artificial"))
        cat = str(gp.get("category", "general"))
        tier = str(gp.get("complexity_tier", "tier_3_complex"))
        mode = str(gp.get("assistant_task_mode", "create_full_asset"))
        q_lbl = str(gp.get("quality_label", "verified_clean"))
        qd = str(gp.get("qd_cell", "unknown"))

        domains[dom] = domains.get(dom, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        tiers[tier] = tiers.get(tier, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
        quality_counts[q_lbl] = quality_counts.get(q_lbl, 0) + 1
        qd_counts[qd] = qd_counts.get(qd, 0) + 1
        seen_hashes.add(r["canonical_dsl_sha256"])

        t_sig = gp.get("tree_shape_signature")
        if not isinstance(t_sig, str) or not t_sig:
            t_sig = compute_program_tree_shape(parse_program(r["canonical_dsl"]))
        tree_shapes_by_family.setdefault(r["family_id"], set()).add(t_sig)

        comp_txt = str(prov.get("assistant_completion") or "")
        canon_txt = str(r.get("canonical_dsl") or "")
        if "//" not in comp_txt and "//" not in canon_txt:
            comment_free_count += 1

        if aud.get("passed") is True and aud.get("collision_pairs", 1) == 0 and aud.get("contact_violations", 1) == 0:
            sat_passed += 1
        if sp == "test_unseen" and (int(aud.get("boxes", 0)) < 16 or q_lbl != "verified_clean"):
            unseen_g2_violations += 1
        if aud.get("animation_swept_sat_passed", True):
            anim_verified_count += 1
        if q_lbl == "controlled_defect_with_repair":
            defect_rows += 1
            ap_txt = str(prov.get("assistant_prompt", ""))
            if ("DIAGNOSE AND REPAIR" in ap_txt or "DIAGNOSE_AND_REPAIR" in ap_txt) and aud.get("passed") is True:
                defect_repaired_ok += 1

    # Shannon entropy over QD cells
    qd_entropy = 0.0
    for cnt in qd_counts.values():
        p = cnt / total
        if p > 0.0:
            qd_entropy -= p * math.log2(p)

    sat_pass_rate = sat_passed / total
    unique_hash_ratio = len(seen_hashes) / total
    anim_pass_rate = anim_verified_count / total
    controlled_bad_ratio = defect_rows / total
    defect_repair_rate = (defect_repaired_ok / defect_rows) if defect_rows > 0 else 1.0
    comment_free_rate = comment_free_count / total

    rejections: list[str] = []
    if sat_pass_rate < 1.0:
        rejections.append(f"sat_pass_rate={sat_pass_rate:.4f} < 1.0")
    if unseen_g2_violations > 0:
        rejections.append(f"unseen_g2_violations={unseen_g2_violations} > 0")
    if unique_hash_ratio < 0.995:
        rejections.append(f"unique_hash_ratio={unique_hash_ratio:.4f} < 0.995")
    if anim_pass_rate < 1.0:
        rejections.append(f"anim_pass_rate={anim_pass_rate:.4f} < 1.0")
    if controlled_bad_ratio > max_bad_data_ratio:
        rejections.append(f"controlled_bad_ratio={controlled_bad_ratio:.4f} > {max_bad_data_ratio}")
    if defect_repair_rate < 1.0:
        rejections.append(f"defect_repair_rate={defect_repair_rate:.4f} < 1.0")
    if comment_free_rate < 1.0:
        rejections.append(f"comment_free_rate={comment_free_rate:.4f} < 1.0")

    distinct_shapes_by_fam = {fam: len(sigs) for fam, sigs in sorted(tree_shapes_by_family.items())}
    all_distinct_shapes = set().union(*tree_shapes_by_family.values()) if tree_shapes_by_family else set()

    return {
        "batch_index": int(batch_index),
        "total_rows": total,
        "batch_accepted": len(rejections) == 0,
        "rejection_reasons": rejections,
        "split_counts": dict(sorted(splits.items())),
        "domain_counts": dict(sorted(domains.items())),
        "category_counts": dict(sorted(categories.items())),
        "complexity_tier_counts": dict(sorted(tiers.items())),
        "assistant_task_mode_counts": dict(sorted(modes.items())),
        "quality_label_counts": dict(sorted(quality_counts.items())),
        "distinct_tree_shapes_by_family": distinct_shapes_by_fam,
        "distinct_tree_shapes_total": len(all_distinct_shapes),
        "box_distribution": {
            "min": int(boxes.min()),
            "p10": round(float(np.percentile(boxes, 10)), 2),
            "p50": round(float(np.percentile(boxes, 50)), 2),
            "mean": round(float(boxes.mean()), 2),
            "p90": round(float(np.percentile(boxes, 90)), 2),
            "max": int(boxes.max()),
            "std": round(float(boxes.std()), 2),
        },
        "bone_distribution": {
            "min": int(bones.min()),
            "p50": round(float(np.percentile(bones, 50)), 2),
            "mean": round(float(bones.mean()), 2),
            "max": int(bones.max()),
            "std": round(float(bones.std()), 2),
        },
        "clean_sat_pass_rate": round(sat_pass_rate, 6),
        "animation_swept_sat_pass_rate": round(anim_pass_rate, 6),
        "comment_free_targets_rate": round(comment_free_rate, 6),
        "controlled_defect_ratio": round(controlled_bad_ratio, 4),
        "defect_detection_and_repair_rate": round(defect_repair_rate, 6),
        "unique_sha256_ratio": round(unique_hash_ratio, 6),
        "qd_shannon_entropy_bits": round(qd_entropy, 3),
        "unique_qd_cells": len(qd_counts),
    }


def generate_dataset_batches(
    out_dir: Path | str,
    *,
    num_batches: int = 4,
    train_per_seen_per_batch: int = 4,
    val_per_seen_per_batch: int = 1,
    test_per_unseen_per_batch: int = 2,
    base_seed: int = 2026,
    samples_to_render_per_batch: int = 6,
    write_parquet: bool = True,
) -> dict[str, Any]:
    """Generate dataset batch-by-batch with statistical QC gates and random visual inspection PNGs."""
    from .static_families import (
        GENERATOR_REVISION,
        SEEN_STATIC_FAMILIES,
        SYNCCRAFT_ATTRIBUTION,
        UNSEEN_STATIC_FAMILIES,
        _write_parquet_splits_if_available,
        verify_and_package_candidate,
        verify_and_package_compositional_candidate,
    )
    from .taxonomy_catalog import COMPOSITIONAL_SEEN_PAIRINGS

    root_dir = Path(out_dir)
    batches_dir = root_dir / "batches"
    sheets_dir = root_dir / "inspection_sheets"
    batches_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": root_dir / "train.jsonl",
        "val": root_dir / "val.jsonl",
        "test_unseen": root_dir / "test_unseen.jsonl",
    }
    aux_holdout_paths = {
        "test_unseen_params": root_dir / "test_unseen_params.jsonl",
        "test_compositional": root_dir / "test_compositional.jsonl",
    }
    split_handles = {sp: open(p, "w", encoding="utf-8") for sp, p in split_paths.items()}
    aux_handles = {sp: open(p, "w", encoding="utf-8") for sp, p in aux_holdout_paths.items()}
    split_hashers = {sp: hashlib.sha256() for sp in split_paths}
    aux_hashers = {sp: hashlib.sha256() for sp in aux_holdout_paths}
    seen_global_hashes: set[str] = set()

    batch_reports: list[dict[str, Any]] = []
    sheet_paths: list[str] = []
    split_totals = {"train": 0, "val": 0, "test_unseen": 0}
    aux_holdout_totals = {"test_unseen_params": 0, "test_compositional": 0}
    family_totals: dict[str, int] = {}
    global_tree_shapes: dict[str, set[str]] = {}

    try:
        for b_idx in range(num_batches):
            batch_rows: list[dict[str, Any]] = []
            t_start = b_idx * train_per_seen_per_batch
            v_start = 100_000 + b_idx * val_per_seen_per_batch
            u_start = b_idx * test_per_unseen_per_batch

            for fam in SEEN_STATIC_FAMILIES:
                for off in range(train_per_seen_per_batch):
                    v_idx = t_start + off
                    row = verify_and_package_candidate(
                        fam,
                        split="train",
                        variant_index=v_idx,
                        base_seed=base_seed,
                        run_exporter_check=(b_idx == 0 and off == 0),
                    )
                    salt = 1
                    while row["canonical_dsl_sha256"] in seen_global_hashes:
                        row = verify_and_package_candidate(
                            fam,
                            split="train",
                            variant_index=v_idx + 1_000_000 * salt,
                            base_seed=base_seed + 7919 * salt,
                        )
                        salt += 1
                    seen_global_hashes.add(row["canonical_dsl_sha256"])
                    batch_rows.append(row)

                for off in range(val_per_seen_per_batch):
                    v_idx = v_start + off
                    row = verify_and_package_candidate(
                        fam,
                        split="val",
                        variant_index=v_idx,
                        base_seed=base_seed + 17,
                    )
                    salt = 1
                    while row["canonical_dsl_sha256"] in seen_global_hashes:
                        row = verify_and_package_candidate(
                            fam,
                            split="val",
                            variant_index=v_idx + 1_000_000 * salt,
                            base_seed=base_seed + 17 + 7919 * salt,
                        )
                        salt += 1
                    seen_global_hashes.add(row["canonical_dsl_sha256"])
                    batch_rows.append(row)

            for fam in UNSEEN_STATIC_FAMILIES:
                for off in range(test_per_unseen_per_batch):
                    v_idx = u_start + off
                    row = verify_and_package_candidate(
                        fam,
                        split="test_unseen",
                        variant_index=v_idx,
                        base_seed=base_seed + 31,
                        run_exporter_check=(b_idx == 0 and off == 0),
                    )
                    salt = 1
                    while row["canonical_dsl_sha256"] in seen_global_hashes:
                        row = verify_and_package_candidate(
                            fam,
                            split="test_unseen",
                            variant_index=v_idx + 1_000_000 * salt,
                            base_seed=base_seed + 31 + 7919 * salt,
                        )
                        salt += 1
                    seen_global_hashes.add(row["canonical_dsl_sha256"])
                    batch_rows.append(row)

            qc = compute_batch_statistical_qc(batch_rows, batch_index=b_idx)
            if not qc["batch_accepted"]:
                raise RuntimeError(f"Batch {b_idx} failed statistical QC: {qc['rejection_reasons']}")

            batch_jsonl = batches_dir / f"batch_{b_idx:04d}.jsonl"
            with open(batch_jsonl, "w", encoding="utf-8") as bh:
                for r in batch_rows:
                    line = json.dumps(r, sort_keys=True, ensure_ascii=False)
                    bh.write(line + "\n")
                    sp = r["split"]
                    split_handles[sp].write(line + "\n")
                    split_hashers[sp].update((line + "\n").encode("utf-8"))
                    split_totals[sp] += 1
                    fam = r["family_id"]
                    family_totals[fam] = family_totals.get(fam, 0) + 1
                    t_sig = r["generator_parameters"].get("tree_shape_signature") or compute_program_tree_shape(
                        parse_program(r["canonical_dsl"])
                    )
                    global_tree_shapes.setdefault(fam, set()).add(t_sig)

            # Emit complementary holdout shards (`test_unseen_params` + `test_compositional`) per batch
            if test_per_unseen_per_batch > 0:
                p_slice_len = max(1, len(SEEN_STATIC_FAMILIES) // max(1, num_batches))
                p_fams = SEEN_STATIC_FAMILIES[b_idx * p_slice_len : (b_idx + 1) * p_slice_len] or SEEN_STATIC_FAMILIES[:4]
                for fam in p_fams:
                    prow = verify_and_package_candidate(
                        fam,
                        split="test_unseen_params",
                        variant_index=300_000 + b_idx * 17,
                        base_seed=base_seed + 43,
                    )
                    salt = 1
                    while prow["canonical_dsl_sha256"] in seen_global_hashes:
                        prow = verify_and_package_candidate(
                            fam,
                            split="test_unseen_params",
                            variant_index=300_000 + b_idx * 17 + 1_000_000 * salt,
                            base_seed=base_seed + 43 + 7919 * salt,
                        )
                        salt += 1
                    seen_global_hashes.add(prow["canonical_dsl_sha256"])
                    pline = json.dumps(prow, sort_keys=True, ensure_ascii=False)
                    aux_handles["test_unseen_params"].write(pline + "\n")
                    aux_hashers["test_unseen_params"].update((pline + "\n").encode("utf-8"))
                    aux_holdout_totals["test_unseen_params"] += 1

                c_slice_len = max(1, len(COMPOSITIONAL_SEEN_PAIRINGS) // max(1, num_batches))
                c_start = (b_idx * c_slice_len) % len(COMPOSITIONAL_SEEN_PAIRINGS)
                for c_off in range(c_slice_len):
                    p_idx = (c_start + c_off) % len(COMPOSITIONAL_SEEN_PAIRINGS)
                    crow = verify_and_package_compositional_candidate(
                        pairing_index=p_idx,
                        variant_index=b_idx,
                        base_seed=base_seed + 59,
                    )
                    salt = 1
                    while crow["canonical_dsl_sha256"] in seen_global_hashes:
                        crow = verify_and_package_compositional_candidate(
                            pairing_index=p_idx,
                            variant_index=b_idx + 100_000 * salt,
                            base_seed=base_seed + 59 + 7919 * salt,
                        )
                        salt += 1
                    seen_global_hashes.add(crow["canonical_dsl_sha256"])
                    cline = json.dumps(crow, sort_keys=True, ensure_ascii=False)
                    aux_handles["test_compositional"].write(cline + "\n")
                    aux_hashers["test_compositional"].update((cline + "\n").encode("utf-8"))
                    aux_holdout_totals["test_compositional"] += 1

            qc_path = batches_dir / f"batch_{b_idx:04d}_qc.json"
            qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
            batch_reports.append(qc)

            if samples_to_render_per_batch > 0:
                sample_rng = random.Random(base_seed + 9001 * (b_idx + 1))
                by_dom: dict[str, list[dict[str, Any]]] = {}
                for r in batch_rows:
                    by_dom.setdefault(r["generator_parameters"]["domain"], []).append(r)
                chosen: list[dict[str, Any]] = []
                dom_keys = sorted(by_dom.keys())
                sample_rng.shuffle(dom_keys)
                for dk in dom_keys:
                    if len(chosen) >= samples_to_render_per_batch:
                        break
                    chosen.append(sample_rng.choice(by_dom[dk]))
                while len(chosen) < min(samples_to_render_per_batch, len(batch_rows)):
                    cand = sample_rng.choice(batch_rows)
                    if cand not in chosen:
                        chosen.append(cand)

                sheet_file = sheets_dir / f"batch_{b_idx:04d}_inspection.png"
                render_batch_inspection_sheet(
                    chosen,
                    sheet_file,
                    title=f"SyncCraft Batch {b_idx:04d} Random QC ({len(batch_rows)} rows, QD H={qc['qd_shannon_entropy_bits']} bits)",
                )
                qc["inspection_sheet"] = f"inspection_sheets/batch_{b_idx:04d}_inspection.png"
                qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
                sheet_paths.append(str(sheet_file))
    finally:
        for h in split_handles.values():
            h.close()
        for h in aux_handles.values():
            h.close()

    all_parquet_paths = {**split_paths, **aux_holdout_paths}
    parquet_files = _write_parquet_splits_if_available(root_dir, all_parquet_paths) if write_parquet else {}
    split_sha256 = {sp: hasher.hexdigest() for sp, hasher in split_hashers.items()}
    holdout_split_sha256 = {
        "test_unseen_params": aux_hashers["test_unseen_params"].hexdigest(),
        "test_unseen": split_sha256["test_unseen"],
        "test_compositional": aux_hashers["test_compositional"].hexdigest(),
    }
    distinct_shapes_summary = {fam: len(sigs) for fam, sigs in sorted(global_tree_shapes.items())}
    all_distinct_shapes = set().union(*global_tree_shapes.values()) if global_tree_shapes else set()

    summary = {
        "schema": "synccraft.batch-dataset-summary/1.0",
        "generator_revision": GENERATOR_REVISION,
        "num_batches": num_batches,
        "all_batches_accepted": all(b["batch_accepted"] for b in batch_reports),
        "total_records": sum(split_totals.values()),
        "split_counts": split_totals,
        "split_sha256": split_sha256,
        "holdout_splits": list(HOLDOUT_EVALUATION_SPLITS),
        "holdout_split_counts": {
            "test_unseen_params": aux_holdout_totals["test_unseen_params"],
            "test_unseen": split_totals["test_unseen"],
            "test_compositional": aux_holdout_totals["test_compositional"],
        },
        "holdout_split_sha256": holdout_split_sha256,
        "distinct_tree_shapes_by_family": distinct_shapes_summary,
        "distinct_tree_shapes_total": len(all_distinct_shapes),
        "family_counts": dict(sorted(family_totals.items())),
        "parquet_shards": parquet_files,
        "inspection_sheets": sheet_paths,
        "batches": batch_reports,
        "batch_qc_reports": batch_reports,
        "attribution": SYNCCRAFT_ATTRIBUTION,
    }
    (root_dir / "batch_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
