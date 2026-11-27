"""Industry Production Batch Pipeline, Curated Hue-Shifted Palette Bank & Sprite Atlas Packer (`TexGraph v7`).

Built for large-scale asset production jobs (not single-toy demos):
- ``PALETTE_BANK``     : 18 curated hue-shifted 5-6 step studio OKLab ramps across Blocks, Items,
                         Real-Life Animals & Imaginary Creatures.
- ``BatchTextureJob``  : Declarative specification for rendering a texture asset with custom size,
                         parameter overrides, colorway remapping, Baked/LabPBR/MER export, and QA gate.
- ``run_batch(...)``   : Executes production batches deterministically with timing/cache telemetry,
                         automated multi-domain ``QualityReport`` auditing, and JSON manifest output.
- ``pack_sprite_atlas``: Packs multiple generated block/item textures into a Minecraft tilesheet PNG + UV JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from .pixel_art_ops import rgba_float_to_u8
from .quality import QualityReport, evaluate_texture_quality


# ---------------------------------------------------------------------------
# 1. Curated Hue-Shifted Studio Palette Bank (Cool Shadows -> Warm Highlights!)
# ---------------------------------------------------------------------------

PALETTE_BANK: dict[str, tuple[tuple[float, str], ...]] = {
    # World Masonry & Earth
    "cobble_warm_slate": ((0.00, "#34313A"), (0.26, "#5C575B"), (0.54, "#8A827C"), (0.80, "#B7A699"), (1.00, "#DFD4C8")),
    "deepslate_obsidian": ((0.00, "#161624"), (0.28, "#2C2E42"), (0.58, "#4C5069"), (0.84, "#777E99"), (1.00, "#ABB4CE")),
    "sandstone_ruin": ((0.00, "#3D4754"), (0.28, "#727570"), (0.56, "#AC9E84"), (0.82, "#D6C59E"), (1.00, "#F3E9C9")),
    "oak_bark": ((0.00, "#24171A"), (0.28, "#472C24"), (0.58, "#724A34"), (0.84, "#9E7250"), (1.00, "#C9A17A")),
    "lush_canopy": ((0.00, "#163B28"), (0.30, "#2E6F36"), (0.62, "#63AB3B"), (0.86, "#A6DB4C"), (1.00, "#E4F88D")),
    # Metals, Gems & Arcane Items
    "forged_steel": ((0.00, "#1B1E32"), (0.28, "#3B4666"), (0.58, "#6E80A4"), (0.84, "#B0C4DE"), (1.00, "#F2F8FF")),
    "champagne_gold": ((0.00, "#592D12"), (0.30, "#A8661C"), (0.62, "#E3A735"), (0.86, "#FAD96D"), (1.00, "#FFF9D9")),
    "ruby_crimson": ((0.00, "#360A1E"), (0.30, "#881337"), (0.60, "#E11D48"), (0.84, "#FB7185"), (1.00, "#FFE4E6")),
    "emerald_jade": ((0.00, "#0A3628"), (0.30, "#0E7A50"), (0.62, "#22C572"), (0.86, "#86EFAC"), (1.00, "#ECFDF5")),
    "starlight_sapphire": ((0.00, "#0E2459"), (0.30, "#0284C7"), (0.62, "#38BDF8"), (0.86, "#A5F3FC"), (1.00, "#F0FDFF")),
    "amethyst_void": ((0.00, "#1F0D3B"), (0.32, "#5B21B6"), (0.64, "#A855F7"), (0.88, "#E879F9"), (1.00, "#FDF4FF")),
    # Real-Life Animal Coats & Plumage
    "husky_espresso": ((0.00, "#241918"), (0.28, "#4A362F"), (0.58, "#7D6154"), (0.84, "#B59C8D"), (1.00, "#E3D6CC")),
    "red_fox_russet": ((0.00, "#421B18"), (0.28, "#8C381F"), (0.58, "#CC6229"), (0.84, "#EB9954"), (1.00, "#FCD49C")),
    "barn_owl_plumage": ((0.00, "#2B2834"), (0.28, "#6E656C"), (0.58, "#B4A89F"), (0.84, "#E2D9CE"), (1.00, "#FAF6F0")),
    "golden_eagle": ((0.00, "#211618"), (0.28, "#4D3226"), (0.58, "#855838"), (0.84, "#C28B53"), (1.00, "#F0CE8B")),
    "scorpion_chitin": ((0.00, "#1E1822"), (0.26, "#382D35"), (0.54, "#5E4C4B"), (0.80, "#8E7364"), (1.00, "#CBB299")),
    "reef_turquoise": ((0.00, "#0A284B"), (0.28, "#0E6491"), (0.58, "#1FB3C8"), (0.84, "#7CE8E2"), (1.00, "#E4FFF9")),
    "dart_frog_cobalt": ((0.00, "#0F1035"), (0.28, "#1D369E"), (0.60, "#2E7CF6"), (0.86, "#67E8F9"), (1.00, "#E0F7FF")),
}


# ---------------------------------------------------------------------------
# 2. Production Batch Runner
# ---------------------------------------------------------------------------


@dataclass
class BatchTextureJob:
    name: str
    builder: str | Callable[..., Any]
    domain: str = "block"  # "block" | "item" | "entity"
    size: int = 32
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    export_pbr: bool = True
    export_baked: bool = True
    lighting_preset: str = "studio_hero"


@dataclass
class BatchJobResult:
    name: str
    domain: str
    size: int
    time_ms: float
    cache_hits: int
    quality: QualityReport
    files: dict[str, str] = field(default_factory=dict)


def run_batch(
    jobs: list[BatchTextureJob],
    out_dir: Path | str,
    *,
    preset_registry: dict[str, Callable[..., Any]] | None = None,
    pack_atlas_name: str | None = None,
) -> dict[str, Any]:
    """Executes a production batch of TexGraph jobs with validation, PBR/Baked export & QA scoring."""
    from .presets import PRESET_REGISTRY

    reg = preset_registry if preset_registry is not None else PRESET_REGISTRY
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    results: list[BatchJobResult] = []
    atlas_sprites: dict[str, np.ndarray] = {}
    t_all = time.perf_counter()

    for job in jobs:
        t0 = time.perf_counter()
        if callable(job.builder):
            g = job.builder(size=job.size, seed=job.seed, **job.params)
        else:
            fn = reg.get(str(job.builder))
            if fn is None:
                raise KeyError(f"Unknown preset builder {job.builder!r} in batch job {job.name!r}")
            g = fn(size=job.size, seed=job.seed, **job.params)

        # Hard production gate: validate DAG before evaluation!
        vrep = g.validate_graph()
        if not vrep.ok:
            raise ValueError(f"Graph validation failed for batch job {job.name!r}: {vrep.errors}")

        ev = g.evaluate(build_pbr=job.export_pbr or job.export_baked)
        qa = evaluate_texture_quality(
            ev.output,
            domain=job.domain,
            tileable=g.tileable,
            mesh_ctx=g.mesh_context,
            pbr_channels=ev.pbr_channels,
            pbr_maps=ev.pbr_maps,
        )

        files: dict[str, str] = {}
        job_dir = od / job.name
        job_dir.mkdir(parents=True, exist_ok=True)
        if job.export_baked:
            pkg = g.export_baked_pbr(job_dir, stem=job.name, preset=job.lighting_preset)
            files = {k: str(v) for k, v in pkg["files"].items()}
        else:
            alb_p = job_dir / f"{job.name}.png"
            Image.fromarray(rgba_float_to_u8(ev.output), mode="RGBA").save(alb_p, format="PNG")
            files["albedo"] = str(alb_p)

        g.save_recipe(job_dir / f"{job.name}.texgraph.json")
        files["recipe"] = str(job_dir / f"{job.name}.texgraph.json")

        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        results.append(
            BatchJobResult(
                name=job.name,
                domain=job.domain,
                size=job.size,
                time_ms=elapsed,
                cache_hits=int(getattr(ev, "cache_hits", 0)),
                quality=qa,
                files=files,
            )
        )
        atlas_sprites[job.name] = rgba_float_to_u8(ev.output)

    atlas_info = None
    if pack_atlas_name and atlas_sprites:
        atlas_info = pack_sprite_atlas(atlas_sprites, od / f"{pack_atlas_name}.png", pad=2)

    manifest = {
        "schema": "synccraft.texgraph.batch/1.0",
        "total_jobs": len(results),
        "passed_qa": sum(1 for r in results if r.quality.passed),
        "avg_qa_score": round(float(np.mean([r.quality.score for r in results])) if results else 0.0, 2),
        "total_time_ms": round((time.perf_counter() - t_all) * 1000.0, 2),
        "atlas": atlas_info,
        "jobs": [
            {
                "name": r.name,
                "domain": r.domain,
                "size": r.size,
                "time_ms": r.time_ms,
                "cache_hits": r.cache_hits,
                "quality": r.quality.to_dict(),
                "files": r.files,
            }
            for r in results
        ],
    }
    manifest_path = od / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def pack_sprite_atlas(
    sprites: dict[str, np.ndarray],
    out_png: Path | str,
    *,
    pad: int = 2,
    cols: int | None = None,
) -> dict[str, Any]:
    """Packs multiple generated block/item RGBA textures into a clean Minecraft tilesheet + UV JSON."""
    names = list(sprites.keys())
    if not names:
        raise ValueError("Cannot pack empty sprite dict")
    max_w = max(int(sprites[n].shape[1]) for n in names)
    max_h = max(int(sprites[n].shape[0]) for n in names)
    cell_w = max_w + 2 * int(pad)
    cell_h = max_h + 2 * int(pad)
    n_cols = int(cols) if cols else max(1, int(np.ceil(np.sqrt(len(names)))))
    n_rows = int(np.ceil(len(names) / float(n_cols)))

    atlas = np.zeros((n_rows * cell_h, n_cols * cell_w, 4), dtype=np.uint8)
    uvs: dict[str, dict[str, Any]] = {}
    for idx, nm in enumerate(names):
        r, c = divmod(idx, n_cols)
        img = sprites[nm]
        if img.dtype != np.uint8:
            img = rgba_float_to_u8(img)
        h, w = img.shape[:2]
        x0 = c * cell_w + int(pad)
        y0 = r * cell_h + int(pad)
        atlas[y0 : y0 + h, x0 : x0 + w] = img
        uvs[nm] = {
            "x": x0,
            "y": y0,
            "w": w,
            "h": h,
            "uv_norm": [round(x0 / atlas.shape[1], 6), round(y0 / atlas.shape[0], 6), round((x0 + w) / atlas.shape[1], 6), round((y0 + h) / atlas.shape[0], 6)],
        }

    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas, mode="RGBA").save(out_p, format="PNG")
    meta_p = out_p.with_suffix(".atlas.json")
    info = {"png": str(out_p), "width": int(atlas.shape[1]), "height": int(atlas.shape[0]), "count": len(names), "sprites": uvs}
    meta_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
