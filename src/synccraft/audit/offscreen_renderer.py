"""CPU orthographic diagnostic preview. This is not the planned ModernGL renderer.

Painter's algorithm is adequate for these starter examples, but intersecting faces
can sort incorrectly. Never use these previews as engine-compatibility evidence.
"""
from __future__ import annotations

import itertools
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
