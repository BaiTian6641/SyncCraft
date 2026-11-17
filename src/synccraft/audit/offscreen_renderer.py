"""CPU orthographic diagnostic preview. This is not the planned ModernGL renderer.

Painter's algorithm is adequate for these starter examples, but intersecting faces
can sort incorrectly. Never use these previews as engine-compatibility evidence.
"""
from __future__ import annotations

import io
import itertools
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synccraft.shading.uv_packer import oklch_to_srgb


def render_four_views(program, scene, output_path):
    colors = {p.name: oklch_to_srgb(p.base_lch) for p in program.palettes}
    panels = [
        ("FRONT / -Z", (1, 0, 0), (0, 1, 0)),
        ("RIGHT / +X", (0, 0, 1), (0, 1, 0)),
        ("TOP / +Y", (1, 0, 0), (0, 0, 1)),
        ("ISOMETRIC", (0.7071, 0, -0.7071), (-0.4082, 0.8165, -0.4082)),
    ]
    width, height = 1200, 920
    image = Image.new("RGB", (width, height), "#101b2b")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 23)
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = font = small = ImageFont.load_default()
    draw.text((28, 20), "SyncCraft / deterministic starter preview", font=title_font, fill="#e9f3ff")
    draw.text((28, 56), f"{len(scene.boxes)} boxes   |   {len(scene.bones)} bones   |   CPU orthographic projection",
              font=font, fill="#94b5d8")
    signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    face_indices = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                    (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    corner_sets = []
    for box in scene.boxes:
        corners = (signs * (np.asarray(box.size) / 2 + box.inflate)) @ np.asarray(box.rotation).T
        corners += np.asarray(box.center)
        corner_sets.append((box, corners))
    all_points = np.concatenate([c for _, c in corner_sets])
    for index, (label, right_raw, up_raw) in enumerate(panels):
        col, row = index % 2, index // 2
        x0, y0 = 24 + col * 590, 100 + row * 390
        draw.rounded_rectangle((x0, y0, x0+566, y0+368), radius=10, fill="#19283b", outline="#294560")
        draw.text((x0+18, y0+13), label, font=font, fill="#a5d9e9")
        right, up = np.array(right_raw, dtype=float), np.array(up_raw, dtype=float)
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        depth = np.cross(right, up)
        projected = np.column_stack((all_points @ right, all_points @ up))
        lo, hi = projected.min(0), projected.max(0)
        center = (lo+hi)/2
        scale = min(470/max(hi[0]-lo[0], 1), 290/max(hi[1]-lo[1], 1))
        primitives = []
        for box, corners in corner_sets:
            for face in face_indices:
                vertices = corners[list(face)]
                pts = np.column_stack((vertices @ right, vertices @ up))
                pts = (pts-center)*scale
                points = [(float(x0+283+p[0]), float(y0+205-p[1])) for p in pts]
                primitives.append((float((vertices @ depth).mean()), points, colors[box.material]))
        for _, points, color in sorted(primitives, key=lambda p:p[0], reverse=True):
            draw.polygon(points, fill=color, outline="#07131f", width=2)
    draw.text((28, 897), "Diagnostic preview only. Flat material colors; no engine import, AO, or animation validation.",
              font=small, fill="#94b5d8")
    image.save(output_path)
    return image


CAMERA_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),   # cam_out = -Z, looking from -Z toward +Z (north face visible)
    "south": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),    # cam_out = +Z, looking from +Z toward -Z (south face visible)
    "right": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),   # cam_out = +X, looking from +X toward -X (east face visible)
    "top": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),     # cam_out = +Y, looking from +Y toward -Y (up face visible)
    "iso": ((0.70710678, 0.0, -0.70710678), (-0.40824829, 0.81649658, -0.40824829)),
}


def render_world_observation(
    world_state: Any,
    *,
    atlas_override: tuple[dict[str, Any], bytes] | None = None,
    uv_layout: dict[str, Any] | None = None,
    panel_size: tuple[int, int] = (192, 192),
    views: tuple[str, ...] = ("south", "iso"),
    fixed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    output_path: Any | None = None,
) -> dict[str, Any]:
    """Z-buffered orthographic ray-OBB 3D world renderer sampling the compiled UV atlas.

    Unlike ``render_four_views`` (which uses Painter's algorithm and flat palette colors),
    ``render_world_observation`` intersects exact 3D OBB slabs per screen pixel and
    samples the compiled ``atlas.png`` through ``atlas_desc['faces'][box][face]``.
    """
    from synccraft.kinematics.se3_chain import compile_posed_geometry
    from synccraft.shading.compiler import compile_vanilla_atlas
    from synccraft.surface.domain import FACE_AXES_BOX, face_uv_extents_units

    posed_scene = compile_posed_geometry(world_state.program, world_state.pose_euler_deltas)
    if atlas_override is not None:
        atlas_desc, atlas_png = atlas_override
    else:
        atlas_desc, atlas_png = compile_vanilla_atlas(world_state, uv_layout=uv_layout)

    atlas_rgba = np.asarray(Image.open(io.BytesIO(atlas_png)).convert("RGBA"), dtype=np.uint8)
    pw, ph = int(panel_size[0]), int(panel_size[1])

    all_corners = np.concatenate([b.corners(include_inflate=True) for b in posed_scene.boxes], axis=0)
    if fixed_bounds is not None:
        w_min, w_max = np.asarray(fixed_bounds[0], dtype=np.float64), np.asarray(fixed_bounds[1], dtype=np.float64)
    else:
        w_min, w_max = all_corners.min(axis=0), all_corners.max(axis=0)
    w_center = 0.5 * (w_min + w_max)
    max_radius = max(1.0, float(np.linalg.norm(w_max - w_min)) * 0.55)

    axis_faces = (
        ("west", "east"),
        ("down", "up"),
        ("north", "south"),
    )

    view_arrays: dict[str, np.ndarray] = {}
    rgba_view_arrays: dict[str, np.ndarray] = {}
    hit_box_maps: dict[str, np.ndarray] = {}

    for view_name in views:
        right_raw, up_raw = CAMERA_PRESETS[view_name]
        right = np.asarray(right_raw, dtype=np.float64)
        right /= np.linalg.norm(right)
        up = np.asarray(up_raw, dtype=np.float64)
        up = up - np.dot(up, right) * right
        up /= np.linalg.norm(up)
        cam_out = np.cross(right, up)  # Points from scene toward camera
        ray_dir = -cam_out

        # Project scene corners to frame tightly unless fixed_bounds is given
        if fixed_bounds is None:
            proj_x = (all_corners - w_center) @ right
            proj_y = (all_corners - w_center) @ up
            half_span = max(
                1.0,
                float(np.max(np.abs(proj_x))) * 1.12,
                float(np.max(np.abs(proj_y))) * 1.12,
            )
        else:
            half_span = max_radius

        xs = np.linspace(-half_span, half_span, pw, dtype=np.float64)
        ys = np.linspace(half_span, -half_span, ph, dtype=np.float64)  # top-to-bottom
        xg, yg = np.meshgrid(xs, ys)

        z_start = max_radius * 4.0
        ray_orig = (
            w_center[None, None, :]
            + xg[..., None] * right
            + yg[..., None] * up
            + z_start * cam_out
        )

        rgb_buf = np.full((ph, pw, 3), (18, 28, 42), dtype=np.uint8)
        z_buf = np.full((ph, pw), np.inf, dtype=np.float64)
        box_idx_buf = np.full((ph, pw), -1, dtype=np.int32)

        for b_idx, box in enumerate(posed_scene.boxes):
            rot = np.asarray(box.rotation, dtype=np.float64)
            ctr = np.asarray(box.center, dtype=np.float64)
            half = np.asarray(box.size, dtype=np.float64) / 2.0 + float(box.inflate)

            o_loc = (ray_orig - ctr) @ rot
            d_loc = ray_dir @ rot

            t_near = np.full((ph, pw), -np.inf, dtype=np.float64)
            t_far = np.full((ph, pw), np.inf, dtype=np.float64)
            enter_axis = np.zeros((ph, pw), dtype=np.int32)
            enter_sign = np.ones((ph, pw), dtype=np.int32)

            valid = np.ones((ph, pw), dtype=bool)
            for k in range(3):
                dk = float(d_loc[k])
                ok = o_loc[..., k]
                hk = float(half[k])
                if abs(dk) < 1e-10:
                    valid &= np.abs(ok) <= hk
                else:
                    t1 = (-hk - ok) / dk
                    t2 = (hk - ok) / dk
                    tn_k = np.minimum(t1, t2)
                    tf_k = np.maximum(t1, t2)
                    # Entering from +hk happens when dk < 0; entering from -hk when dk > 0
                    sgn = 1 if dk < 0.0 else -1
                    promote = tn_k > t_near
                    t_near = np.where(promote, tn_k, t_near)
                    enter_axis = np.where(promote, k, enter_axis)
                    enter_sign = np.where(promote, sgn, enter_sign)
                    t_far = np.minimum(t_far, tf_k)

            hit_mask = valid & (t_far >= np.maximum(0.0, t_near)) & (t_near < z_buf)
            if not np.any(hit_mask):
                continue

            hit_loc = o_loc + t_near[..., None] * d_loc

            for k in range(3):
                for sgn_idx, sgn_val in enumerate((-1, 1)):
                    f_mask = hit_mask & (enter_axis == k) & (enter_sign == sgn_val)
                    if not np.any(f_mask):
                        continue
                    face_name = axis_faces[k][sgn_idx]
                    if face_name in box.cullface:
                        continue
                    t_r, t_d, _ = (np.asarray(v, dtype=np.float64) for v in FACE_AXES_BOX[face_name])
                    wu, hv = face_uv_extents_units(box.size, face_name)
                    span_u = wu + 2.0 * float(box.inflate)
                    span_v = hv + 2.0 * float(box.inflate)

                    pts_f = hit_loc[f_mask]
                    u_frac = np.clip((pts_f @ t_r) / max(1e-8, span_u) + 0.5, 0.0, 1.0 - 1e-9)
                    v_frac = np.clip((pts_f @ t_d) / max(1e-8, span_v) + 0.5, 0.0, 1.0 - 1e-9)

                    u0, v0, u1, v1 = atlas_desc["faces"][box.name][face_name]
                    ax = (int(u0) + np.floor(u_frac * max(1, u1 - u0)).astype(np.intp))
                    ay = (int(v0) + np.floor(v_frac * max(1, v1 - v0)).astype(np.intp))
                    sampled_rgba = atlas_rgba[ay, ax]
                    opaque_sub = sampled_rgba[:, 3] >= 128
                    if not np.any(opaque_sub):
                        continue
                    fy, fx = np.nonzero(f_mask)
                    oy = fy[opaque_sub]
                    ox = fx[opaque_sub]
                    z_buf[oy, ox] = t_near[oy, ox]
                    box_idx_buf[oy, ox] = b_idx
                    rgb_buf[oy, ox] = sampled_rgba[opaque_sub, :3]

        alpha_silhouette = np.where(box_idx_buf >= 0, 255, 0).astype(np.uint8)
        rgba_view_arrays[view_name] = np.concatenate((rgb_buf, alpha_silhouette[..., None]), axis=-1)
        view_arrays[view_name] = rgb_buf
        hit_box_maps[view_name] = box_idx_buf

    composite = np.concatenate([view_arrays[v] for v in views], axis=1)
    if output_path is not None:
        Image.fromarray(composite, mode="RGB").save(output_path)

    return {
        "views": view_arrays,
        "rgba_views": rgba_view_arrays,
        "hit_box_maps": hit_box_maps,
        "composite_rgb": composite,
        "atlas_descriptor": atlas_desc,
    }
