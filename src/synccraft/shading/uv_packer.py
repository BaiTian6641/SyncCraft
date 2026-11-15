"""Integer per-face UV atlas for the starter subset, not the planned Box UV shader.

Original solid material colors only. No baked lighting, cavity AO, emissive engine
configuration, palette-ramp synthesis, or seam painting is claimed here.
"""
from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageDraw


def oklch_to_srgb(lch):
    """Diagnostic OKLCH base color, with explicit channel clipping (no gamut mapper)."""
    lightness, chroma, hue = map(float, lch)
    angle = math.radians(hue)
    a, b = chroma * math.cos(angle), chroma * math.sin(angle)
    try:
        l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    except OverflowError as exc:
        raise ValueError("OKLCH value exceeds the diagnostic converter's numeric range") from exc
    linear = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    if not all(math.isfinite(v) for v in linear):
        raise ValueError("OKLCH conversion produced nonfinite color channels")
    def encode(v):
        v = min(1.0, max(0.0, v))
        return round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))
    return tuple(encode(v) for v in linear)


def allocate_uv_layout(
    program,
    scene,
    *,
    max_width=512,
    padding=1,
    box_order=None,
    face_order=None,
):
    """Allocate integer padded per-face UV rectangles without baking color.

    Keeping layout allocation separate from material evaluation allows repacking
    UVs (changing island order, max_width, or padding) without altering the
    underlying rest-space material pattern.
    """
    if type(max_width) is not int or max_width < 8:
        raise ValueError("max_width must be an integer >= 8")
    if type(padding) is not int or padding < 1:
        raise ValueError("padding must be an integer >= 1")
    try:
        density = program.meta.texel_density / 16
    except OverflowError as exc:
        raise ValueError("texel_density exceeds the atlas numeric range") from exc

    boxes_by_name = {b.name: b for b in scene.boxes}
    if box_order is None:
        ordered_boxes = list(scene.boxes)
    else:
        if set(box_order) != set(boxes_by_name) or len(box_order) != len(boxes_by_name):
            raise ValueError("box_order must be an exact permutation of scene box names")
        ordered_boxes = [boxes_by_name[name] for name in box_order]

    default_faces = ("north", "south", "east", "west", "up", "down")
    if face_order is None:
        ordered_faces = default_faces
    else:
        if set(face_order) != set(default_faces) or len(face_order) != 6:
            raise ValueError("face_order must be an exact permutation of the six cuboid faces")
        ordered_faces = tuple(face_order)

    x = y = padding
    row_height = 0
    occupied_width = 1
    placements = []
    faces = {b.name: {} for b in scene.boxes}
    for box in ordered_boxes:
        sx, sy, sz = map(float, box.size)
        sizes = {"north": (sx, sy), "south": (sx, sy), "east": (sz, sy),
                 "west": (sz, sy), "up": (sx, sz), "down": (sx, sz)}
        for face in ordered_faces:
            u, v = sizes[face]
            u_is_pos_int = math.isclose(u, round(u), abs_tol=1e-9) and u > 0.0
            v_is_pos_int = math.isclose(v, round(v), abs_tol=1e-9) and v > 0.0
            fw, fh = u * density, v * density
            if (u_is_pos_int and not math.isclose(fw, round(fw), abs_tol=1e-7)) or (
                v_is_pos_int and not math.isclose(fh, round(fh), abs_tol=1e-7)
            ):
                raise ValueError("texel_density produces fractional texels; integer atlas required")
            w, h = max(1, int(round(fw))), max(1, int(round(fh)))
            if min(w, h) < 1 or w + 2 * padding > max_width:
                raise ValueError("face cannot fit atlas width at the requested density")
            if x + w + padding > max_width:
                x = padding
                y += row_height + 2 * padding
                row_height = 0
            faces[box.name][face] = [x, y, x + w, y + h]
            placements.append((box.name, face, x, y, w, h, box.material))
            row_height = max(row_height, h)
            occupied_width = max(occupied_width, x + w + padding)
            x += w + 2 * padding
    if not placements:
        raise ValueError("cannot pack an empty scene")

    def pow2(v):
        return 1 << (int(v) - 1).bit_length()

    width, height = pow2(occupied_width), pow2(y + row_height + padding)
    if max(width, height) > 8192:
        raise ValueError("diagnostic atlas exceeds 8192-pixel limit; use production packer")
    return {
        "width": width,
        "height": height,
        "padding": padding,
        "faces": faces,
        "placements": placements,
    }


def build_atlas(program, scene, *, max_width=512, padding=1):
    """Return (descriptor, PNG bytes). Every box face gets a unique padded island.

    texel_density means pixels per 16 model units. Fractional per-face dimensions
    fail explicitly rather than silently changing texture scale.
    """
    layout = allocate_uv_layout(program, scene, max_width=max_width, padding=padding)
    colors = {p.name: oklch_to_srgb(p.base_lch) for p in program.palettes}
    width, height = layout["width"], layout["height"]
    faces = layout["faces"]
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for _, _, u, v, w, h, mat_name in layout["placements"]:
        color = colors[mat_name]
        draw.rectangle((u-padding, v-padding, u+w+padding-1, v+h+padding-1), fill=(*color, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False)
    payload = stream.getvalue()
    return {"width": width, "height": height,
            "data_url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
            "faces": faces, "mode": "diagnostic_per_face_flat_base_color"}, payload
