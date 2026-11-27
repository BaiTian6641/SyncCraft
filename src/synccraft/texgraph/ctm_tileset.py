"""Connected Block Tileset (CTM 47-Tile & Wang 16-Tile) Batch Generator (`TexGraph v15`).

Uses SyncCraft's `block_edge_transition` solver to synthesize full connected-texture
autotile sets ready for:
- **Minecraft OptiFine / Continuity CTM 47-Tile (`method=ctm`)**: 47 canonical 8-bit neighbor
  states + `.properties` manifest + packed 12x4 atlas preview + individual `0.png..46.png` tiles!
- **Wang 16-Tile / 4-Bit Autotile (Hytale / Godot / Unity Tilemap)**: 16 canonical 4-bit cardinal
  neighbor states (N/E/S/W) + 4x4 packed sprite sheet + JSON tileset manifest!
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pixel_art_ops import rgba_float_to_u8
from .seam_and_pbr_ops import block_edge_transition

# Canonical 4-bit cardinal neighbor masks: bit0=N(1), bit1=E(2), bit2=S(4), bit3=W(8)
# Plus 4 diagonal corner bits for OptiFine CTM 47-tile: bit4=NE(16), bit5=SE(32), bit6=SW(64), bit7=NW(128)
# A diagonal corner is only active if BOTH adjacent cardinal neighbors are connected.
WANG_16_MASKS: list[int] = list(range(16))


def _canonical_ctm_47_masks() -> list[int]:
    """Generate the exact 47 canonical 8-bit autotile neighbor bitmasks (OptiFine / Godot Blob-47)."""
    seen: list[int] = []
    seen_set: set[int] = set()
    for m in range(256):
        n = bool(m & 1)
        e = bool(m & 2)
        s = bool(m & 4)
        w = bool(m & 8)
        ne = bool(m & 16) and n and e
        se = bool(m & 32) and s and e
        sw = bool(m & 64) and s and w
        nw = bool(m & 128) and n and w
        canon = (
            (1 if n else 0)
            | (2 if e else 0)
            | (4 if s else 0)
            | (8 if w else 0)
            | (16 if ne else 0)
            | (32 if se else 0)
            | (64 if sw else 0)
            | (128 if nw else 0)
        )
        if canon not in seen_set:
            seen_set.add(canon)
            seen.append(canon)
    assert len(seen) == 47
    return seen


CTM_47_MASKS: list[int] = _canonical_ctm_47_masks()


def _apply_corner_notches(
    tile_rgba: np.ndarray,
    bitmask_8: int,
    *,
    edge_width_px: int = 3,
    edge_color: tuple[float, float, float] = (0.18, 0.22, 0.28),
) -> np.ndarray:
    """Refine inner concave corner notches for 8-bit CTM 47-tile diagonal states."""
    out = tile_rgba.copy()
    h, w = out.shape[:2]
    ew = max(1, min(int(edge_width_px), min(h, w) // 4))
    n = bool(bitmask_8 & 1)
    e = bool(bitmask_8 & 2)
    s = bool(bitmask_8 & 4)
    w_b = bool(bitmask_8 & 8)
    ne = bool(bitmask_8 & 16)
    se = bool(bitmask_8 & 32)
    sw = bool(bitmask_8 & 64)
    nw = bool(bitmask_8 & 128)

    col = np.array(edge_color, dtype=np.float32)
    # If both adjacent cardinals connect, but diagonal is missing -> draw crisp concave corner pin!
    if n and e and not ne:
        out[:ew, w - ew :, :3] = out[:ew, w - ew :, :3] * 0.35 + col * 0.65
    if s and e and not se:
        out[h - ew :, w - ew :, :3] = out[h - ew :, w - ew :, :3] * 0.35 + col * 0.65
    if s and w_b and not sw:
        out[h - ew :, :ew, :3] = out[h - ew :, :ew, :3] * 0.35 + col * 0.65
    if n and w_b and not nw:
        out[:ew, :ew, :3] = out[:ew, :ew, :3] * 0.35 + col * 0.65
    return np.clip(out, 0.0, 1.0)


def generate_ctm_tileset(
    base_rgba: np.ndarray,
    *,
    mode: str = "ctm47",
    style: str = "metal_frame",
    edge_width_px: int = 3,
    edge_color: str = "#D4A84A",
    strength: float = 0.85,
    seed: int = 42,
    out_dir: str | Path | None = None,
    stem: str = "synccraft_block",
) -> dict[str, Any]:
    """Generate a full Wang 16-Tile or OptiFine/Continuity CTM 47-Tile connected tileset.

    Returns dict with `tile_count`, `grid_cols`, `grid_rows`, `sheet_u8`, `sheet_data_url`,
    `optifine_properties`, `manifest`, and optional written `files`.
    """
    arr = np.asarray(base_rgba, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.concatenate((arr, np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)), axis=-1)
    if float(np.max(arr)) > 1.5:
        arr = arr / 255.0

    h, w = arr.shape[:2]
    is_47 = str(mode or "ctm47").strip().lower() in ("ctm47", "47", "blob47", "optifine")
    masks = CTM_47_MASKS if is_47 else WANG_16_MASKS
    cols = 12 if is_47 else 4
    rows = 4 if is_47 else 4

    from .pixel_art_ops import hex_to_srgb

    ec_rgb = hex_to_srgb(edge_color)
    tiles_u8: list[np.ndarray] = []
    sheet_u8 = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)

    bw_frac = float(np.clip(float(edge_width_px) / float(max(1, min(h, w))), 0.05, 0.35))
    for idx, bm8 in enumerate(masks):
        nesw_bits = (
            ("1" if (bm8 & 1) else "0")
            + ("1" if (bm8 & 2) else "0")
            + ("1" if (bm8 & 4) else "0")
            + ("1" if (bm8 & 8) else "0")
        )
        res = block_edge_transition(
            h,
            w,
            base=arr,
            neighbors=nesw_bits,
            style=style,
            border_width=bw_frac,
            corner_strength=float(strength),
            transition_color=edge_color,
            seed=int(seed) + idx * 7,
        )
        tile_f = res["albedo"] if isinstance(res, dict) else res
        if is_47:
            tile_f = _apply_corner_notches(tile_f, bm8, edge_width_px=int(edge_width_px), edge_color=ec_rgb)
        t_u8 = rgba_float_to_u8(tile_f)
        tiles_u8.append(t_u8)
        r = idx // cols
        c = idx % cols
        sheet_u8[r * h : (r + 1) * h, c * w : (c + 1) * w] = t_u8

    buf = io.BytesIO()
    Image.fromarray(sheet_u8, mode="RGBA").save(buf, format="PNG")
    sheet_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    optifine_props = (
        f"# Generated by SyncCraft Studio v15 CTM Batch Generator\n"
        f"matchBlocks={stem}\n"
        f"method={'ctm' if is_47 else 'ctm_compact'}\n"
        f"tiles=0-{len(masks) - 1}\n"
        f"connect=block\n"
    )
    manifest = {
        "schema": "synccraft.ctm.tileset/1.0",
        "stem": stem,
        "mode": "ctm47" if is_47 else "wang16",
        "style": style,
        "tile_size": [w, h],
        "tile_count": len(masks),
        "sheet_grid": [cols, rows],
        "bitmask_order": masks,
        "bit_legend": {"N": 1, "E": 2, "S": 4, "W": 8, "NE": 16, "SE": 32, "SW": 64, "NW": 128},
    }

    written_files: list[str] = []
    if out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        sheet_p = od / f"{stem}_{manifest['mode']}_sheet.png"
        Image.fromarray(sheet_u8, mode="RGBA").save(sheet_p)
        written_files.append(str(sheet_p))
        props_p = od / f"{stem}.properties"
        props_p.write_text(optifine_props, encoding="utf-8")
        written_files.append(str(props_p))
        man_p = od / f"{stem}_{manifest['mode']}.json"
        man_p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written_files.append(str(man_p))
        tiles_sub = od / "tiles"
        tiles_sub.mkdir(exist_ok=True)
        for idx, t_u8 in enumerate(tiles_u8):
            tp = tiles_sub / f"{idx}.png"
            Image.fromarray(t_u8, mode="RGBA").save(tp)
        written_files.append(str(tiles_sub))

    return {
        "ok": True,
        "mode": manifest["mode"],
        "tile_count": len(masks),
        "grid_cols": cols,
        "grid_rows": rows,
        "tile_size": [w, h],
        "sheet_u8": sheet_u8,
        "sheet_data_url": sheet_url,
        "optifine_properties": optifine_props,
        "manifest": manifest,
        "files": written_files,
    }
