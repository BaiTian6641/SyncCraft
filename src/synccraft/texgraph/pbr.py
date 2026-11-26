"""Full Physically-Based Rendering (PBR) Pipeline for `synccraft.texgraph`.

Supports:
- Crisp Scharr 3x3 Height -> Tangent-Space Normal Map synthesis (OpenGL green-up & DirectX)
- Spec-compliant Minecraft LabPBR 1.3 multi-texture packing:
    * `<stem>.png`   : sRGB Albedo + Alpha
    * `<stem>_n.png` : R=Nx, G=Ny, B=Baked AO, A=POM Height [1..255]
    * `<stem>_s.png` : R=Smoothness (1-Rough), G=F0/MetalID [0..229 | 230..255],
                       B=Porosity [0..64] / SSS [65..255], A=Emissive [0..254, 255=none]
- Minecraft Bedrock RTX PBR packing (`<stem>_mer.png` + `<stem>.texture_set.json`)
- Standard Metallic-Roughness PBR export (`_normal`, `_roughness`, `_metallic`, `_height`, `_ao`, `_emissive`)
- Pure-NumPy Cook-Torrance GGX Microfacet PBR Preview Renderer + Studio 6-Panel Material Sheet
- Synchronized multi-channel animated PBR `.mcmeta` flipbooks + animated PBR preview GIFs
"""
from __future__ import annotations

import base64
import copy
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .pixel_art_ops import encode_png, export_flipbook, linear_to_srgb, rgba_float_to_u8, srgb_to_linear

# LabPBR 1.3 Hardcoded Conductor Metal Codes (G channel in [230..255])
LABPBR_METAL_CODES: dict[str, int] = {
    "iron": 230,
    "gold": 231,
    "aluminum": 232,
    "chrome": 233,
    "copper": 234,
    "lead": 235,
    "platinum": 236,
    "silver": 237,
}


def height_to_normal_map(
    height_field: np.ndarray,
    *,
    strength: float = 2.4,
    tileable: bool = False,
    convention: str = "opengl",
) -> np.ndarray:
    """Convert scalar heightfield [0, 1] -> tangent-space RGB normal map in [0, 1]^3.

    Uses an isotropic 3x3 Scharr gradient kernel (much cleaner diagonal edges on pixel art
    than naive finite differences). OpenGL convention sets +Y pointing up (green-up),
    matching LabPBR 1.3 and Minecraft shader packs.
    """
    h = np.asarray(height_field, dtype=np.float32)
    if h.ndim == 3:
        h = 0.299 * h[..., 0] + 0.587 * h[..., 1] + 0.114 * h[..., 2]
    mode = "wrap" if tileable else "edge"
    p = np.pad(h, 1, mode=mode)

    # Isotropic Scharr 3x3 derivative (normalized by 32)
    gx = (
        3.0 * p[:-2, 2:] + 10.0 * p[1:-1, 2:] + 3.0 * p[2:, 2:]
        - 3.0 * p[:-2, :-2] - 10.0 * p[1:-1, :-2] - 3.0 * p[2:, :-2]
    ) / 32.0
    gy = (
        3.0 * p[2:, :-2] + 10.0 * p[2:, 1:-1] + 3.0 * p[2:, 2:]
        - 3.0 * p[:-2, :-2] - 10.0 * p[:-2, 1:-1] - 3.0 * p[:-2, 2:]
    ) / 32.0

    s = float(strength)
    nx = -gx * s
    # Image Y increases downward; OpenGL +Y tangent points upward -> ny = +gy * s
    ny = (gy * s) if convention == "opengl" else (-gy * s)
    nz = np.ones_like(h, dtype=np.float32)
    inv_l = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
    nx, ny, nz = nx * inv_l, ny * inv_l, nz * inv_l

    rgb = np.stack((nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5), axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _scalar_map(arr: np.ndarray | float | None, shape: tuple[int, int], default: float) -> np.ndarray:
    H, W = shape
    if arr is None:
        return np.full((H, W), float(default), dtype=np.float32)
    if isinstance(arr, (int, float)):
        return np.full((H, W), float(arr), dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[2] == 4:
            # For emissive RGBA, use max(RGB)*A so intensity matches perceived glow
            return np.clip(np.max(a[..., :3], axis=-1) * a[..., 3], 0.0, 1.0).astype(np.float32)
        return np.clip(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], 0.0, 1.0).astype(np.float32)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def pack_labpbr_1_3(
    albedo_rgba: np.ndarray,
    *,
    height: np.ndarray | None = None,
    normal_rgb: np.ndarray | None = None,
    roughness: np.ndarray | float | None = None,
    metallic: np.ndarray | float | None = None,
    metal_id: np.ndarray | int | None = None,
    ao: np.ndarray | float | None = None,
    porosity: np.ndarray | float | None = None,
    sss: np.ndarray | float | None = None,
    emissive: np.ndarray | float | None = None,
    dielectric_f0: float = 0.04,
    normal_strength: float = 2.4,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Pack TexGraph channels into spec-compliant Minecraft LabPBR 1.3 + Bedrock MER uint8 arrays.

    Returns dict with uint8 RGBA arrays:
      - ``albedo``      : sRGB Albedo + Alpha
      - ``labpbr_n``    : LabPBR 1.3 `_n.png` (R=Nx, G=Ny, B=AO, A=Height[1..255])
      - ``labpbr_s``    : LabPBR 1.3 `_s.png` (R=Smooth, G=F0/MetalID, B=Porosity/SSS, A=Emissive[0..254|255])
      - ``bedrock_mer`` : Bedrock RTX `_mer.png` (R=Metallic, G=Emissive, B=Roughness, A=255)
      - ``normal_rgb``  : Standard tangent-space normal RGB + A=255
      - ``height_u8``   : Grayscale height RGBA
      - ``roughness_u8``: Grayscale roughness RGBA
      - ``metallic_u8`` : Grayscale metallic RGBA
      - ``ao_u8``       : Grayscale AO RGBA
    """
    alb_u8 = rgba_float_to_u8(albedo_rgba)
    H, W = alb_u8.shape[:2]

    h_map = _scalar_map(height, (H, W), 0.5)
    if normal_rgb is None:
        nrm = height_to_normal_map(h_map, strength=normal_strength, tileable=tileable, convention="opengl")
    else:
        nrm = np.clip(np.asarray(normal_rgb, dtype=np.float32)[..., :3], 0.0, 1.0)

    r_map = _scalar_map(roughness, (H, W), 0.55)
    m_map = _scalar_map(metallic, (H, W), 0.0)
    ao_map = _scalar_map(ao, (H, W), 1.0)
    por_map = _scalar_map(porosity, (H, W), 0.0)
    sss_map = _scalar_map(sss, (H, W), 0.0)
    em_map = _scalar_map(emissive, (H, W), 0.0)

    # 1. LabPBR 1.3 _n.png: R=Nx, G=Ny, B=AO, A=Height in [1..255] (never 0!)
    nx_u8 = np.clip(np.rint(nrm[..., 0] * 255.0), 0, 255).astype(np.uint8)
    ny_u8 = np.clip(np.rint(nrm[..., 1] * 255.0), 0, 255).astype(np.uint8)
    nz_u8 = np.clip(np.rint(nrm[..., 2] * 255.0), 0, 255).astype(np.uint8)
    ao_u8 = np.clip(np.rint(ao_map * 255.0), 0, 255).astype(np.uint8)
    h_pom_u8 = np.clip(np.rint(1.0 + h_map * 254.0), 1, 255).astype(np.uint8)
    labpbr_n = np.stack((nx_u8, ny_u8, ao_u8, h_pom_u8), axis=-1)

    # 2. LabPBR 1.3 _s.png: R=Smoothness, G=F0/MetalID, B=Porosity/SSS, A=Emissive (255=none)
    smooth_u8 = np.clip(np.rint((1.0 - r_map) * 255.0), 0, 255).astype(np.uint8)

    diel_f0_u8 = np.clip(np.rint(float(dielectric_f0) * 255.0), 0, 229).astype(np.uint8)
    if metal_id is None:
        # Auto-infer gold vs silver/iron conductor ID from warm vs cool albedo hue where metallic >= 0.5
        alb_f = alb_u8[..., :3].astype(np.float32) / 255.0
        is_warm_gold = (alb_f[..., 0] > alb_f[..., 2] + 0.14) & (alb_f[..., 1] > alb_f[..., 2] + 0.08)
        cond_code = np.where(is_warm_gold, np.uint8(231), np.uint8(230))  # 231=Gold, 230=Iron/Steel
    elif isinstance(metal_id, int):
        cond_code = np.full((H, W), int(np.clip(metal_id, 230, 255)), dtype=np.uint8)
    else:
        cond_code = np.clip(np.rint(np.asarray(metal_id, dtype=np.float32)), 230, 255).astype(np.uint8)

    # Dielectric F0 increases subtly with partial metalness (<0.5); >=0.5 uses hardcoded LabPBR conductor ID
    diel_var = np.clip(np.rint((float(dielectric_f0) + m_map * 0.25) * 255.0), 0, 229).astype(np.uint8)
    f0_metal_u8 = np.where(m_map >= 0.5, cond_code, np.maximum(diel_f0_u8, diel_var))

    use_sss = (sss_map > 0.04) & (sss_map >= por_map)
    sss_u8 = np.clip(np.rint(65.0 + sss_map * 190.0), 65, 255).astype(np.uint8)
    por_u8 = np.clip(np.rint(por_map * 64.0), 0, 64).astype(np.uint8)
    por_sss_u8 = np.where(use_sss, sss_u8, por_u8)

    is_em = em_map > 0.03
    em_lab_val = np.clip(np.rint(em_map * 254.0), 1, 254).astype(np.uint8)
    em_lab_u8 = np.where(is_em, em_lab_val, np.uint8(255))
    labpbr_s = np.stack((smooth_u8, f0_metal_u8, por_sss_u8, em_lab_u8), axis=-1)

    # 3. Bedrock RTX MER: R=Metallic, G=Emissive, B=Roughness, A=255
    m_u8 = np.clip(np.rint(m_map * 255.0), 0, 255).astype(np.uint8)
    em_u8 = np.clip(np.rint(em_map * 255.0), 0, 255).astype(np.uint8)
    r_u8 = np.clip(np.rint(r_map * 255.0), 0, 255).astype(np.uint8)
    ones_u8 = np.full((H, W), 255, dtype=np.uint8)
    bedrock_mer = np.stack((m_u8, em_u8, r_u8, ones_u8), axis=-1)

    # 4. Standard standalone RGBA visualizations
    normal_std = np.stack((nx_u8, ny_u8, nz_u8, ones_u8), axis=-1)
    h_vis = np.clip(np.rint(h_map * 255.0), 0, 255).astype(np.uint8)
    height_u8 = np.stack((h_vis, h_vis, h_vis, ones_u8), axis=-1)
    roughness_u8 = np.stack((r_u8, r_u8, r_u8, ones_u8), axis=-1)
    metallic_u8 = np.stack((m_u8, m_u8, m_u8, ones_u8), axis=-1)
    ao_vis_u8 = np.stack((ao_u8, ao_u8, ao_u8, ones_u8), axis=-1)

    return {
        "albedo": alb_u8,
        "labpbr_n": labpbr_n,
        "labpbr_s": labpbr_s,
        "bedrock_mer": bedrock_mer,
        "normal_rgb": normal_std,
        "height_u8": height_u8,
        "roughness_u8": roughness_u8,
        "metallic_u8": metallic_u8,
        "ao_u8": ao_vis_u8,
    }


def render_pbr_preview_2d(
    albedo_rgba: np.ndarray,
    *,
    normal_rgb: np.ndarray | None = None,
    height: np.ndarray | None = None,
    roughness: np.ndarray | float | None = None,
    metallic: np.ndarray | float | None = None,
    ao: np.ndarray | float | None = None,
    sss: np.ndarray | float | None = None,
    emissive_rgba: np.ndarray | None = None,
    light_az_deg: float = -38.0,
    light_el_deg: float = 52.0,
    normal_strength: float = 2.4,
    tileable: bool = False,
) -> np.ndarray:
    """Physically-Based Cook-Torrance GGX Microfacet PBR 2D Studio Preview Renderer (pure NumPy).

    Evaluates:
      - Trowbridge-Reitz GGX NDF + Smith-Schlick Geometry + Schlick Fresnel F0
      - Energy-conserving diffuse + specular IBL sky/ground hemisphere bounce
      - Subsurface Scattering (SSS) wrap light
      - Additive emissive bloom + ACES Filmic tonemapping
    Returns float32 RGBA in [0, 1].
    """
    alb_f = np.asarray(albedo_rgba, dtype=np.float32)
    if alb_f.dtype == np.uint8 or alb_f.max() > 1.5:
        alb_f = alb_f / 255.0
    H, W = alb_f.shape[:2]
    alpha = alb_f[..., 3:4] if alb_f.shape[2] == 4 else np.ones((H, W, 1), dtype=np.float32)
    base_lin = srgb_to_linear(alb_f[..., :3])

    if normal_rgb is None:
        h_map = _scalar_map(height, (H, W), 0.5)
        nrm_01 = height_to_normal_map(h_map, strength=normal_strength, tileable=tileable)
    else:
        nrm_01 = np.asarray(normal_rgb, dtype=np.float32)[..., :3]
        if nrm_01.max() > 1.5:
            nrm_01 = nrm_01 / 255.0

    # Tangent-space normal in [-1, 1] (+Y up)
    N = nrm_01 * 2.0 - 1.0
    n_len = np.sqrt(np.sum(N * N, axis=-1, keepdims=True) + 1e-12)
    N = N / n_len

    rough = np.clip(_scalar_map(roughness, (H, W), 0.48), 0.06, 0.98)[..., None]
    metal = np.clip(_scalar_map(metallic, (H, W), 0.0), 0.0, 1.0)[..., None]
    ao_m = np.clip(_scalar_map(ao, (H, W), 1.0), 0.15, 1.0)[..., None]
    sss_m = np.clip(_scalar_map(sss, (H, W), 0.0), 0.0, 1.0)[..., None]

    # Key studio light + cool rim fill light in tangent space
    az, el = math.radians(float(light_az_deg)), math.radians(float(light_el_deg))
    L1 = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)], dtype=np.float32)
    L1 /= max(1e-6, float(np.linalg.norm(L1)))
    L2 = np.array([-L1[0] * 0.8, -L1[1] * 0.6, 0.45], dtype=np.float32)
    L2 /= max(1e-6, float(np.linalg.norm(L2)))
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    F0 = 0.04 * (1.0 - metal) + base_lin * metal
    a_ggx = rough * rough
    a2 = a_ggx * a_ggx
    k_g = ((rough + 1.0) ** 2) / 8.0
    ndotv = np.clip(N[..., 2:3], 0.001, 1.0)

    def _eval_light(L_vec: np.ndarray, radiance: np.ndarray) -> np.ndarray:
        H_vec = L_vec + V
        H_vec = H_vec / max(1e-6, float(np.linalg.norm(H_vec)))
        ndotl = np.clip(np.sum(N * L_vec[None, None, :], axis=-1, keepdims=True), 0.0, 1.0)
        ndoth = np.clip(np.sum(N * H_vec[None, None, :], axis=-1, keepdims=True), 0.0, 1.0)
        hdotv = np.clip(float(np.dot(H_vec, V)), 0.0, 1.0)

        # Trowbridge-Reitz GGX NDF
        denom = ndoth * ndoth * (a2 - 1.0) + 1.0
        D = a2 / (math.pi * denom * denom + 1e-6)
        # Schlick-GGX Geometry
        G = (ndotv / (ndotv * (1.0 - k_g) + k_g)) * (ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-6))
        # Schlick Fresnel
        F = F0 + (1.0 - F0) * ((1.0 - hdotv) ** 5)

        spec = (D * G * F) / (4.0 * ndotv * ndotl + 1e-4)
        kD = (1.0 - F) * (1.0 - metal)
        diff = kD * base_lin / math.pi

        # Subsurface wrap diffuse
        wrap = np.clip((np.sum(N * L_vec[None, None, :], axis=-1, keepdims=True) + 0.45) / 1.45, 0.0, 1.0)
        sss_contrib = sss_m * (1.0 - metal) * base_lin * (wrap * 0.42)

        return (diff + spec) * radiance[None, None, :] * ndotl + sss_contrib * radiance[None, None, :]

    col = (
        _eval_light(L1, np.array([2.65, 2.52, 2.38], dtype=np.float32))
        + _eval_light(L2, np.array([0.75, 0.92, 1.25], dtype=np.float32))
    )

    # Hemisphere IBL ambient + specular reflection (sky zenith blue-ivory vs warm ground bounce)
    sky_w = np.clip(0.5 + 0.5 * N[..., 1:2], 0.0, 1.0)
    sky_col = np.array([0.42, 0.48, 0.62], dtype=np.float32)[None, None, :]
    gnd_col = np.array([0.24, 0.21, 0.28], dtype=np.float32)[None, None, :]
    env_diff = sky_w * sky_col + (1.0 - sky_w) * gnd_col
    # Reflected env streak for glossy/metallic sheen
    R_y = np.clip(2.0 * N[..., 2:3] * N[..., 1:2], -1.0, 1.0)
    env_spec = np.where(R_y > 0.0, sky_col * (1.2 + 0.9 * R_y), gnd_col * 0.95)
    F_amb = F0 + (np.maximum(1.0 - rough, F0) - F0) * ((1.0 - ndotv) ** 5)
    col = col + ((1.0 - F_amb) * (1.0 - metal) * base_lin * env_diff + F_amb * env_spec * (1.0 - 0.55 * rough)) * ao_m

    # Emissive glow
    if emissive_rgba is not None:
        ef = np.asarray(emissive_rgba, dtype=np.float32)
        if ef.max() > 1.5:
            ef = ef / 255.0
        if ef.ndim == 3 and ef.shape[2] >= 3:
            ea = ef[..., 3:4] if ef.shape[2] == 4 else np.ones((H, W, 1), dtype=np.float32)
            col = col + srgb_to_linear(ef[..., :3]) * ea * 1.85

    # ACES Filmic tonemapping
    x = np.maximum(0.0, col * 1.08)
    mapped = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    srgb = linear_to_srgb(mapped)
    return np.concatenate((srgb, alpha), axis=-1).astype(np.float32)


def render_pbr_material_sheet(
    pbr_maps: dict[str, np.ndarray],
    pbr_preview_rgba: np.ndarray,
    *,
    title: str = "SyncCraft PBR Material Breakdown",
    panel_px: int = 224,
) -> Image.Image:
    """Render a crisp 6-panel studio PBR breakdown strip:
    Albedo | Height | Normal (_n) | LabPBR Spec (_s) | Roughness | GGX PBR Lit.
    """
    panels = [
        ("Albedo", pbr_maps["albedo"]),
        ("Height (POM)", pbr_maps["height_u8"]),
        ("Normal (_n RGB)", pbr_maps["normal_rgb"]),
        ("LabPBR Spec (_s)", pbr_maps["labpbr_s"]),
        ("Roughness", pbr_maps["roughness_u8"]),
        ("GGX PBR Lit", rgba_float_to_u8(pbr_preview_rgba)),
    ]
    n = len(panels)
    hdr = 34
    lbl_h = 22
    pad = 8
    w_tot = n * panel_px + (n + 1) * pad
    h_tot = hdr + panel_px + lbl_h + pad * 2

    canvas = Image.new("RGBA", (w_tot, h_tot), (15, 19, 27, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 4, 10), title, fill=(235, 242, 255, 255))

    # Checkerboard backdrop for transparent sprites
    chk = Image.new("RGBA", (panel_px, panel_px), (28, 33, 46, 255))
    cd = ImageDraw.Draw(chk)
    cs = 14
    for y in range(0, panel_px, cs):
        for x in range(0, panel_px, cs):
            if ((x // cs) + (y // cs)) % 2 == 0:
                cd.rectangle((x, y, x + cs - 1, y + cs - 1), fill=(38, 45, 62, 255))

    for i, (lbl, arr_u8) in enumerate(panels):
        x0 = pad + i * (panel_px + pad)
        y0 = hdr + pad
        im = Image.fromarray(arr_u8, mode="RGBA")
        # Preserve aspect ratio inside panel_px box using nearest-neighbor for crisp pixel art
        ih, iw = arr_u8.shape[:2]
        sc = min(panel_px / max(1, iw), panel_px / max(1, ih))
        rw, rh = max(1, int(round(iw * sc))), max(1, int(round(ih * sc)))
        im_r = im.resize((rw, rh), resample=Image.Resampling.NEAREST)
        tile = chk.copy()
        tile.alpha_composite(im_r, ((panel_px - rw) // 2, (panel_px - rh) // 2))
        canvas.paste(tile, (x0, y0))
        draw.rectangle((x0, y0, x0 + panel_px - 1, y0 + panel_px - 1), outline=(52, 65, 88, 255), width=1)
        draw.text((x0 + 4, y0 + panel_px + 5), lbl, fill=(176, 198, 228, 255))

    return canvas.convert("RGB")


def export_pbr_bundle(
    pbr_maps: dict[str, np.ndarray],
    out_dir: Path | str,
    stem: str,
    *,
    emissive_rgba: np.ndarray | None = None,
    pbr_preview_rgba: np.ndarray | None = None,
    include_standalone: bool = True,
) -> dict[str, str]:
    """Write complete LabPBR 1.3 + Bedrock RTX MER + optional standard PBR PNG maps to disk."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _save(name: str, arr_u8: np.ndarray) -> str:
        p = d / name
        Image.fromarray(arr_u8, mode="RGBA").save(p, format="PNG", optimize=True)
        return str(p)

    # Core LabPBR 1.3 trio
    written["albedo"] = _save(f"{stem}.png", pbr_maps["albedo"])
    written["labpbr_n"] = _save(f"{stem}_n.png", pbr_maps["labpbr_n"])
    written["labpbr_s"] = _save(f"{stem}_s.png", pbr_maps["labpbr_s"])

    # Bedrock RTX MER + texture_set.json
    written["bedrock_mer"] = _save(f"{stem}_mer.png", pbr_maps["bedrock_mer"])
    tex_set = {
        "format_version": "1.16.100",
        "minecraft:texture_set": {
            "color": stem,
            "metalness_emissive_roughness": f"{stem}_mer",
            "normal": f"{stem}_normal",
        },
    }
    ts_path = d / f"{stem}.texture_set.json"
    ts_path.write_text(json.dumps(tex_set, indent=2) + "\n", encoding="utf-8")
    written["bedrock_texture_set"] = str(ts_path)

    if include_standalone:
        written["normal"] = _save(f"{stem}_normal.png", pbr_maps["normal_rgb"])
        written["height"] = _save(f"{stem}_height.png", pbr_maps["height_u8"])
        written["roughness"] = _save(f"{stem}_roughness.png", pbr_maps["roughness_u8"])
        written["metallic"] = _save(f"{stem}_metallic.png", pbr_maps["metallic_u8"])
        written["ao"] = _save(f"{stem}_ao.png", pbr_maps["ao_u8"])
        if emissive_rgba is not None:
            written["emissive"] = _save(f"{stem}_emissive.png", rgba_float_to_u8(emissive_rgba))

    if pbr_preview_rgba is not None:
        written["pbr_preview"] = _save(f"{stem}_pbr_preview.png", rgba_float_to_u8(pbr_preview_rgba))

    return written


# ---------------------------------------------------------------------------
# Baked Studio PBR Texture Format + Canonical `.synccraft.pbr.json` Manifest
# ---------------------------------------------------------------------------

BAKE_LIGHTING_PRESETS: dict[str, dict[str, Any]] = {
    "studio_hero": {
        "label": "Studio Hero (Warm Key / Cool Rim)",
        "key_az_deg": -36.0,
        "key_el_deg": 54.0,
        "key_rgb": [2.25, 2.16, 2.02],
        "fill_az_deg": 138.0,
        "fill_el_deg": 28.0,
        "fill_rgb": [0.78, 0.92, 1.18],
        "sky_rgb": [0.46, 0.52, 0.64],
        "ground_rgb": [0.26, 0.23, 0.30],
        "ao_strength": 0.62,
        "spec_strength": 1.0,
        "sss_strength": 0.85,
        "emissive_strength": 1.55,
        "blend_with_albedo": 0.22,
        "exposure": 1.08,
    },
    "dramatic_rim": {
        "label": "Dramatic Obsidian Rim",
        "key_az_deg": -52.0,
        "key_el_deg": 42.0,
        "key_rgb": [2.55, 2.38, 2.15],
        "fill_az_deg": 152.0,
        "fill_el_deg": 24.0,
        "fill_rgb": [0.65, 1.05, 1.48],
        "sky_rgb": [0.34, 0.40, 0.56],
        "ground_rgb": [0.18, 0.16, 0.24],
        "ao_strength": 0.78,
        "spec_strength": 1.25,
        "sss_strength": 1.05,
        "emissive_strength": 1.85,
        "blend_with_albedo": 0.14,
        "exposure": 1.12,
    },
    "high_noon": {
        "label": "High Noon Daylight",
        "key_az_deg": -18.0,
        "key_el_deg": 72.0,
        "key_rgb": [2.35, 2.32, 2.24],
        "fill_az_deg": 160.0,
        "fill_el_deg": 35.0,
        "fill_rgb": [0.85, 0.96, 1.12],
        "sky_rgb": [0.56, 0.64, 0.78],
        "ground_rgb": [0.34, 0.32, 0.28],
        "ao_strength": 0.50,
        "spec_strength": 0.88,
        "sss_strength": 0.75,
        "emissive_strength": 1.25,
        "blend_with_albedo": 0.26,
        "exposure": 1.05,
    },
    "moonlit_obsidian": {
        "label": "Moonlit Obsidian",
        "key_az_deg": 40.0,
        "key_el_deg": 48.0,
        "key_rgb": [1.55, 1.88, 2.45],
        "fill_az_deg": -135.0,
        "fill_el_deg": 24.0,
        "fill_rgb": [0.72, 0.48, 0.85],
        "sky_rgb": [0.28, 0.36, 0.54],
        "ground_rgb": [0.16, 0.14, 0.24],
        "ao_strength": 0.72,
        "spec_strength": 1.20,
        "sss_strength": 0.90,
        "emissive_strength": 1.95,
        "blend_with_albedo": 0.16,
        "exposure": 1.10,
    },
}


def bake_pbr_to_albedo(
    pbr_maps: dict[str, np.ndarray],
    *,
    preset: str = "studio_hero",
    key_az_deg: float | None = None,
    key_el_deg: float | None = None,
    normal_strength: float = 1.0,
    spec_strength: float | None = None,
    ao_strength: float | None = None,
    sss_strength: float | None = None,
    emissive_strength: float | None = None,
    blend_with_albedo: float | None = None,
    exposure: float | None = None,
    emissive_rgba: np.ndarray | None = None,
) -> np.ndarray:
    """Physically bake LabPBR 1.3 maps (`albedo`, `_n`, `_s`) into a crisp game-ready sRGB Baked Albedo RGBA [0, 1].

    Designed so flat/vanilla Blockbench & Minecraft renderers without shader packs still exhibit
    metallic gold reflections, normal-mapped scale bevels, cool contact AO, wing SSS & emissive glow!
    """
    cfg = dict(BAKE_LIGHTING_PRESETS.get(preset, BAKE_LIGHTING_PRESETS["studio_hero"]))
    az1 = float(cfg["key_az_deg"] if key_az_deg is None else key_az_deg)
    el1 = float(cfg["key_el_deg"] if key_el_deg is None else key_el_deg)
    s_spec = float(cfg["spec_strength"] if spec_strength is None else spec_strength)
    s_ao = float(cfg["ao_strength"] if ao_strength is None else ao_strength)
    s_sss = float(cfg["sss_strength"] if sss_strength is None else sss_strength)
    s_em = float(cfg["emissive_strength"] if emissive_strength is None else emissive_strength)
    b_alb = float(np.clip(cfg["blend_with_albedo"] if blend_with_albedo is None else blend_with_albedo, 0.0, 1.0))
    exp = float(cfg["exposure"] if exposure is None else exposure)

    alb_u8 = pbr_maps["albedo"]
    H, W = alb_u8.shape[:2]
    alb = alb_u8.astype(np.float32) / 255.0
    base_lin = srgb_to_linear(alb[..., :3])
    alpha = alb[..., 3:4]

    # Unpack LabPBR 1.3 _n (R=Nx, G=Ny, B=AO, A=POM Height) & _s (R=Smooth, G=F0/Metal, B=Por/SSS, A=Emissive)
    n_u8 = pbr_maps["labpbr_n"].astype(np.float32)
    s_u8 = pbr_maps["labpbr_s"].astype(np.float32)

    nx = (n_u8[..., 0] / 127.5 - 1.0) * float(normal_strength)
    ny = (n_u8[..., 1] / 127.5 - 1.0) * float(normal_strength)
    nz = np.sqrt(np.maximum(0.04, 1.0 - np.clip(nx * nx + ny * ny, 0.0, 0.96)))
    N = np.stack((nx, ny, nz), axis=-1)
    N = N / np.maximum(1e-6, np.linalg.norm(N, axis=-1, keepdims=True))

    ao_raw = np.clip(n_u8[..., 2:3] / 255.0, 0.08, 1.0)
    ao_m = 1.0 - (1.0 - ao_raw) * np.clip(s_ao, 0.0, 1.5)
    ao_m = np.clip(ao_m, 0.12, 1.0)

    smooth = np.clip(s_u8[..., 0:1] / 255.0, 0.0, 1.0)
    rough = np.clip(1.0 - smooth, 0.06, 1.0)
    g_raw = s_u8[..., 1:2]
    metal = (g_raw >= 229.5).astype(np.float32)
    diel_f0 = np.clip(g_raw / 255.0, 0.02, 0.25)
    b_raw = s_u8[..., 2:3]
    sss_m = np.where(b_raw >= 64.5, (b_raw - 64.0) / 191.0, 0.0).astype(np.float32)
    em_scalar = np.where(s_u8[..., 3:4] < 254.5, s_u8[..., 3:4] / 254.0, 0.0).astype(np.float32)

    F0 = diel_f0 * (1.0 - metal) + base_lin * metal
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ndotv = np.maximum(0.05, N[..., 2:3])

    def _dir(az: float, el: float) -> np.ndarray:
        a, e = math.radians(az), math.radians(el)
        v = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)], dtype=np.float32)
        return v / max(1e-6, float(np.linalg.norm(v)))

    L1 = _dir(az1, el1)
    L2 = _dir(float(cfg["fill_az_deg"]), float(cfg["fill_el_deg"]))

    def _shade(L_vec: np.ndarray, rad: np.ndarray) -> np.ndarray:
        H_v = L_vec + V
        H_v = H_v / max(1e-6, float(np.linalg.norm(H_v)))
        ndotl = np.maximum(0.0, np.sum(N * L_vec[None, None, :], axis=-1, keepdims=True))
        ndoth = np.maximum(0.0, np.sum(N * H_v[None, None, :], axis=-1, keepdims=True))
        hdotv = float(max(0.0, np.dot(H_v, V)))
        a2 = (rough * rough) ** 2
        denom = (ndoth * ndoth) * (a2 - 1.0) + 1.0
        D = a2 / (math.pi * denom * denom + 1e-6)
        k_g = ((rough + 1.0) ** 2) / 8.0
        G = (ndotv / (ndotv * (1.0 - k_g) + k_g)) * (ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-6))
        F = F0 + (1.0 - F0) * ((1.0 - hdotv) ** 5)
        spec = ((D * G * F) / (4.0 * ndotv * ndotl + 1e-4)) * s_spec
        kD = (1.0 - F) * (1.0 - metal)
        diff = kD * base_lin / math.pi
        wrap = np.clip((np.sum(N * L_vec[None, None, :], axis=-1, keepdims=True) + 0.48) / 1.48, 0.0, 1.0)
        sss_c = sss_m * s_sss * (1.0 - metal) * base_lin * (wrap * 0.45)
        return (diff + spec) * rad[None, None, :] * ndotl + sss_c * rad[None, None, :]

    col = (
        _shade(L1, np.array(cfg["key_rgb"], dtype=np.float32))
        + _shade(L2, np.array(cfg["fill_rgb"], dtype=np.float32))
    )

    # Hemisphere IBL + crisp metallic sky reflection
    sky_w = np.clip(0.5 + 0.5 * N[..., 1:2], 0.0, 1.0)
    sky_c = np.array(cfg["sky_rgb"], dtype=np.float32)[None, None, :]
    gnd_c = np.array(cfg["ground_rgb"], dtype=np.float32)[None, None, :]
    env_diff = sky_w * sky_c + (1.0 - sky_w) * gnd_c
    R_y = np.clip(2.0 * N[..., 2:3] * N[..., 1:2], -1.0, 1.0)
    env_spec = np.where(R_y > 0.0, sky_c * (1.25 + 0.95 * R_y), gnd_c * 0.95)
    F_amb = F0 + (np.maximum(1.0 - rough, F0) - F0) * ((1.0 - ndotv) ** 5)
    col = col + ((1.0 - F_amb) * (1.0 - metal) * base_lin * env_diff + F_amb * env_spec * s_spec * (1.0 - 0.52 * rough)) * ao_m

    # Blend subtle fraction of original unlit albedo so flat Blockbench models stay rich on every face
    if b_alb > 1e-4:
        col = col * (1.0 - b_alb) + base_lin * ao_m * b_alb

    # Emissive glow (either explicit RGBA or unpacked from LabPBR _s alpha)
    if emissive_rgba is not None:
        ef = np.asarray(emissive_rgba, dtype=np.float32)
        if ef.max() > 1.5:
            ef = ef / 255.0
        if ef.ndim == 3 and ef.shape[2] >= 3:
            ea = ef[..., 3:4] if ef.shape[2] == 4 else np.ones((H, W, 1), dtype=np.float32)
            col = col + srgb_to_linear(ef[..., :3]) * ea * s_em
    elif float(em_scalar.max()) > 1e-4:
        col = col + base_lin * em_scalar * s_em

    # ACES Filmic tonemap -> crisp sRGB
    x = np.maximum(0.0, col * exp)
    mapped = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    srgb = linear_to_srgb(mapped)
    return np.concatenate((srgb, alpha), axis=-1).astype(np.float32)


def _u8_to_data_url(arr_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode="RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def export_baked_pbr_package(
    pbr_maps: dict[str, np.ndarray],
    out_dir: Path | str,
    stem: str,
    *,
    emissive_rgba: np.ndarray | None = None,
    preset: str = "studio_hero",
    bake_params: dict[str, Any] | None = None,
    bbmodel_doc: dict[str, Any] | None = None,
    recipe_path: str | None = None,
    embed_maps_in_manifest: bool = False,
) -> dict[str, Any]:
    """Export a complete Baked Studio Texture + `.synccraft.pbr.json` Manifest + optional `_baked.bbmodel`!

    Produces:
      - `<stem>_baked.png`           : Physically baked studio Albedo (ready for vanilla Blockbench/Minecraft!)
      - `<stem>.synccraft.pbr.json`  : Canonical PBR rebuild manifest loadable by SyncCraft PBR Rebuilder plugin
      - `<stem>_baked.bbmodel`       : Optional `.bbmodel` copy with the baked studio texture embedded
      - All standard LabPBR 1.3 (`_n`, `_s`), Bedrock RTX (`_mer`, `.texture_set.json`) & standalone PBR maps
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    bp = dict(bake_params or {})
    baked_f = bake_pbr_to_albedo(pbr_maps, preset=preset, emissive_rgba=emissive_rgba, **bp)
    baked_u8 = rgba_float_to_u8(baked_f)

    written = export_pbr_bundle(
        pbr_maps,
        d,
        stem,
        emissive_rgba=emissive_rgba,
        pbr_preview_rgba=baked_f,
        include_standalone=True,
    )
    baked_path = d / f"{stem}_baked.png"
    Image.fromarray(baked_u8, mode="RGBA").save(baked_path, format="PNG", optimize=True)
    written["baked_albedo"] = str(baked_path)

    H, W = baked_u8.shape[:2]
    manifest: dict[str, Any] = {
        "schema": "synccraft.pbr.bundle/1.0",
        "stem": stem,
        "width": W,
        "height": H,
        "pbr_standard": "labpbr_1_3+bedrock_mer",
        "bake_preset": preset,
        "bake_params": {**BAKE_LIGHTING_PRESETS.get(preset, BAKE_LIGHTING_PRESETS["studio_hero"]), **bp},
        "maps": {
            "albedo": f"{stem}.png",
            "baked": f"{stem}_baked.png",
            "labpbr_n": f"{stem}_n.png",
            "labpbr_s": f"{stem}_s.png",
            "bedrock_mer": f"{stem}_mer.png",
            "bedrock_texture_set": f"{stem}.texture_set.json",
            "normal": f"{stem}_normal.png",
            "height": f"{stem}_height.png",
            "roughness": f"{stem}_roughness.png",
            "metallic": f"{stem}_metallic.png",
            "ao": f"{stem}_ao.png",
        },
    }
    if emissive_rgba is not None:
        manifest["maps"]["emissive"] = f"{stem}_emissive.png"
    if recipe_path:
        manifest["texgraph_recipe"] = str(recipe_path)
    if embed_maps_in_manifest:
        manifest["embedded_data_urls"] = {
            "albedo": _u8_to_data_url(pbr_maps["albedo"]),
            "baked": _u8_to_data_url(baked_u8),
            "labpbr_n": _u8_to_data_url(pbr_maps["labpbr_n"]),
            "labpbr_s": _u8_to_data_url(pbr_maps["labpbr_s"]),
            "bedrock_mer": _u8_to_data_url(pbr_maps["bedrock_mer"]),
            "normal": _u8_to_data_url(pbr_maps["normal_rgb"]),
        }
        if emissive_rgba is not None:
            manifest["embedded_data_urls"]["emissive"] = _u8_to_data_url(rgba_float_to_u8(emissive_rgba))

    man_path = d / f"{stem}.synccraft.pbr.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["pbr_manifest"] = str(man_path)

    if bbmodel_doc is not None:
        doc_copy = copy.deepcopy(bbmodel_doc)
        data_url = _u8_to_data_url(baked_u8)
        if doc_copy.get("textures"):
            t0 = doc_copy["textures"][0]
            t0["width"] = W
            t0["height"] = H
            t0["uv_width"] = W
            t0["uv_height"] = H
            t0["source"] = data_url
            t0["internal"] = True
            t0["name"] = f"{stem}_baked.png"
        baked_bb_path = d / f"{stem}_baked.bbmodel"
        baked_bb_path.write_text(json.dumps(doc_copy, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        written["baked_bbmodel"] = str(baked_bb_path)

    return {"files": written, "manifest": manifest, "baked_rgba": baked_f, "baked_u8": baked_u8}
