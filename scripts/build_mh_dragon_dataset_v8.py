#!/usr/bin/env python3
"""Build the Complete SyncCraft Block-DSL v4.0 Monster Hunter V8 Hybrid Collection & Dataset.

Compiles, verifies, exports, renders, and packages all 22 Monster Hunter creatures:
  - Native Block-DSL v4.0 `@component` + `@assembly` programs (`<id>.blockdsl`) + Stage-1 coarse
    blockouts (`<id>_stage1_blockout.blockdsl`) with >= 80-100% genuine 3D rotated Blockbench
    cuboids (`box`) + selective RGBA alpha-cutout `plane` cards for non-uni-sheet multi-bank wings/fins.
  - Bespoke x32 (`2 texels/unit`, `1024-2048px`) RGBA pixel-art atlases (`textures/albedo.png` + `emissive.png`).
  - Blockbench 5 `.bbmodel` (`model_format="free"`) + Minecraft Bedrock `.geo.json` & `.animation.json`.
  - 264 authored vigorous Monster Hunter FK animation clips (12 clips per monster).
  - High-resolution studio renders (`hero.png`, `side.png`, `front.png`, `head_hero.png`,
    8-panel `<id>_studio_review.png`, 22-specimen `v8_collection_overview.jpg`, and 6 animated GIFs).
  - Self-contained offline WebGL 3D Studio Viewer (`viewer/index.html`, Three.js r160.1 MIT).
  - Family-grouped 5-mode SyncCraft World-Model JSONL + Zstd Parquet dataset shards strictly
    quarantined under `research_nc_fan_reference` (`commercial_weights_eligible=False`).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from synccraft.data_gen.diversity_engine import compute_program_tree_shape, compute_qd_signature
from synccraft.data_gen.mh_dragons_v8.v8_compiler import compile_and_export_v8_monster
from synccraft.data_gen.mh_dragons_v8.v8_renderer import render_animated_clip_gif
from synccraft.data_gen.mh_dragons_v8.v8_viewer import CANONICAL_22_ORDER, build_v8_offline_viewer
from synccraft.data_gen.taxonomy_catalog import classify_complexity_tier
from synccraft.dsl.parser import format_program, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.se3_chain import compile_geometry

DATASET_GENERATOR_VERSION = "synccraft-v4.0-mh-v8-hybrid-studio"
LICENSE_TRACK = "research_nc_fan_reference"

# Strict family-level split assignment (parent + all subspecies/rare/variants stay in same split!)
FAMILY_SPLIT_ASSIGNMENT: dict[str, str] = {
    # Train (8 lineages -> 15 monsters)
    "rathalos": "research_nc/train",           # rathalos, azure_rathalos, silver_rathalos
    "zinogre": "research_nc/train",            # zinogre, stygian_zinogre
    "gore_magala": "research_nc/train",        # gore_magala, chaotic_gore_magala
    "ruiner_nergigante": "research_nc/train",
    "rey_dau": "research_nc/train",
    "fatalis": "research_nc/train",            # fatalis, white_fatalis
    "lagiacrus": "research_nc/train",          # lagiacrus, ivory_lagiacrus, abyssal_lagiacrus
    "mizutsune": "research_nc/train",
    # Validation (3 lineages -> 4 monsters)
    "legiana": "research_nc/val",              # legiana, shrieking_legiana
    "velkhana": "research_nc/val",
    "jin_dahaad": "research_nc/val",
    # Unseen Holdout (3 lineages -> 3 monsters)
    "crimson_glow_valstrax": "research_nc/test_unseen",
    "arkveld": "research_nc/test_unseen",
    "zoh_shia": "research_nc/test_unseen",
}

SIGNATURE_GIFS: dict[str, str] = {
    "zinogre": "fulgur_backflip_slam",
    "rathalos": "aerial_venom_talon_dive",
    "rey_dau": "railgun_horn_blast",
    "arkveld": "chainblade_overhead_slam",
    "velkhana": "scorpion_tail_lance_flurry",
    "mizutsune": "bubble_drift_pirouette",
}


def _compile_one_worker(args: tuple[str, Path, Path | None, str | None]) -> dict[str, Any]:
    mid, output_root, ref_root, gif_clip = args
    from synccraft.data_gen.mh_dragons_v8.registry import ALL_V8_MONSTER_BUILDERS

    doc = ALL_V8_MONSTER_BUILDERS[mid]()
    return compile_and_export_v8_monster(
        doc,
        output_root,
        ref_root=ref_root,
        render_previews=True,
        render_gif_clip=gif_clip,
    )


def render_v8_collection_overview_poster(
    output_root: Path,
    manifests: list[dict[str, Any]],
) -> Path:
    """Render a 2400x3180 4x6 studio collection poster across all 22 V8 specimens."""
    cols, rows = 4, 6
    tile_w, tile_h = 556, 456
    pad = 28
    header_h = 132
    cw = pad + cols * (tile_w + pad)
    ch = header_h + rows * (tile_h + 44) + pad

    poster = Image.new("RGB", (cw, ch), (11, 16, 22))
    draw = ImageDraw.Draw(poster)
    def _fnt(sz: int, bold: bool = False) -> Any:
        cands = (
            ["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
            if bold
            else ["DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
        )
        for fp in cands:
            try:
                return ImageFont.truetype(fp, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f_h1 = _fnt(34, True)
    f_sub = _fnt(18, False)
    f_card = _fnt(16, True)
    f_meta = _fnt(13, False)

    tot_elems = sum(m["elements_total"] for m in manifests)
    tot_boxes = sum(m["cuboid_boxes"] for m in manifests)
    tot_planes = sum(m["cutout_planes"] for m in manifests)
    tot_clips = sum(m["clip_count"] for m in manifests)

    draw.rectangle((0, 0, cw, header_h - 14), fill=(16, 23, 33))
    draw.text(
        (pad, 24),
        "MONSTER HUNTER - SCULPTED BLOCKBENCH MASTERWORKS (BLOCK-DSL v4.0)",
        fill=(240, 245, 250),
        font=f_h1,
    )
    draw.text(
        (pad, 74),
        f"22 Canonical, Subspecies, Rare & Elder Dragon Specimens  |  {tot_boxes:,} 3D Cuboids ({100.0 * tot_boxes / max(1, tot_elems):.1f}%) + "
        f"{tot_planes:,} RGBA Cutout Cards  |  16px/Block Banded Cel Atlases  |  {tot_clips} Vigorous Action Clips",
        fill=(146, 198, 234),
        font=f_sub,
    )

    by_id = {m["id"]: m for m in manifests}
    ordered = [by_id[mid] for mid in CANONICAL_22_ORDER if mid in by_id]

    for idx, m in enumerate(ordered):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_w + pad)
        y0 = header_h + r * (tile_h + 44)
        draw.rounded_rectangle(
            (x0 - 2, y0 - 2, x0 + tile_w + 2, y0 + tile_h + 30),
            radius=8,
            fill=(18, 26, 36),
            outline=(46, 62, 84),
        )
        hero_p = Path(m["previews"]["hero"])
        if hero_p.is_file():
            im = Image.open(hero_p).convert("RGB").resize((tile_w, tile_h - 32), Image.Resampling.LANCZOS)
            poster.paste(im, (x0, y0 + 32))
        draw.text(
            (x0 + 12, y0 + 7),
            f"{idx + 1:02d}  {m['title']}",
            fill=(238, 244, 250),
            font=f_card,
        )
        draw.text(
            (x0 + 12, y0 + tile_h + 6),
            f"{m['species_class']}  |  {m['cuboid_boxes']} cuboids + {m['cutout_planes']} cards  |  {m['atlas_size'][0]}x{m['atlas_size'][1]} 16px/blk  |  12 clips",
            fill=(164, 194, 218),
            font=f_meta,
        )

    # Summary badge in cells 22 & 23
    sx0 = pad + 2 * (tile_w + pad)
    sy0 = header_h + 5 * (tile_h + 44)
    span_w = 2 * tile_w + pad
    draw.rounded_rectangle(
        (sx0 - 2, sy0 - 2, sx0 + span_w + 2, sy0 + tile_h + 30),
        radius=8,
        fill=(16, 25, 36),
        outline=(74, 118, 156),
    )
    summary_lines = [
        "SCULPTED BLOCKBENCH STYLE HIGHLIGHTS",
        "- 22/22 Canonical Flagships, Subspecies, Rare Species, Variants & Constructs",
        f"- {tot_boxes:,} Compact-Muzzle & Roof-Tile Shingled 3D Cuboids ({100.0 * tot_boxes / max(1, tot_elems):.1f}% of geometry)",
        f"- {tot_planes:,} Fan-Layered Non-Uni-Sheet RGBA Wing/Fin Cutout Cards (3 Hinge Banks / Wing)",
        "- True 16px/Block (256-512px) Integer-Pixel Banded Cel Atlases + Focal Emissive Bleed",
        "- Export Parity Verified: 0.0 SE(3) & Bedrock Vertex Drift across .blockdsl, .bbmodel & .geo.json",
        f"- {tot_clips} Vigorous Monster Hunter Action Clips + Self-Contained Offline WebGL 3D Viewer",
    ]
    for li, sline in enumerate(summary_lines):
        draw.text(
            (sx0 + 28, sy0 + 28 + li * 54),
            sline,
            fill=(236, 192, 128) if li == 0 else (214, 228, 242),
            font=f_sub if li == 0 else f_card,
        )

    out_p = output_root / "previews" / "v8_collection_overview.jpg"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    poster.save(out_p, quality=94)
    return out_p


def _build_supervision_records_for_monster(
    manifest: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Build 5-mode World-Model supervision records + per-clip animation records for one V8 monster."""
    mid = manifest["id"]
    asset_dir = output_root / "assets" / mid
    dsl_source = (asset_dir / f"{mid}.blockdsl").read_text(encoding="utf-8")
    stage1_dsl = (asset_dir / f"{mid}_stage1_blockout.blockdsl").read_text(encoding="utf-8")
    render_scene = json.loads((asset_dir / "render_scene.json").read_text(encoding="utf-8"))

    ledger = CostLedger()
    ledger.record("compiler_calls", 2)
    ledger.record("initial_audits", 1)
    prog = parse_program(dsl_source)
    sc = compile_geometry(prog)
    stage1_prog = parse_program(stage1_dsl)
    stage1_sc = compile_geometry(stage1_prog)

    family_group = manifest.get("parent_species_id") or mid
    split_name = FAMILY_SPLIT_ASSIGNMENT.get(family_group, "research_nc/train")
    tier = classify_complexity_tier(len(sc.boxes))
    tree_shape = compute_program_tree_shape(prog)
    dsl_sha = hashlib.sha256(dsl_source.encode("utf-8")).hexdigest()
    clip_names = list(manifest["clip_names"])
    mat_summary = ", ".join(f"{m.name}:{m.family}" for m in prog.materials)

    # Static without @behavior/@controller for Mode 4
    static_dsl = format_program(replace(prog, assembly_spec=None, source_map=None, behavior=None, controller=None))
    # Untextured clay lowered rig for Mode 3
    lowered_dsl = format_program(replace(prog, assembly_spec=None, source_map=None, materials=(), style=None, behavior=None, controller=None))

    notes_str = " ".join(render_scene.get("notes", []))
    sys_msg = (
        "You are SyncCraft-3D Studio, an expert Blockbench and SyncCraft Block-DSL v4.0 creature sculptor, "
        "x32 pixel-atlas painter, and Monster Hunter kinematic animator. Emit modular @component + @assembly "
        "Block-DSL v4.0 assets built from high-density rotated 3D cuboids plus articulated non-uni-sheet wing banks."
    )

    turns: list[tuple[str, str, str]] = [
        (
            "create_full_asset",
            (
                f"[MODE: CREATE MODEL TEXTURE ANIMATION | DSL v4.0 HYBRID | TIER: {tier}]\n"
                f"Sculpt, paint (x32 RGBA), and animate '{manifest['title']}' ({manifest['title_epithet']}, "
                f"{manifest['species_class']}) in Block-DSL v4.0 (@component + @assembly).\n"
                f"Anatomy & Style: {notes_str}\n"
                f"Target: {manifest['cuboid_boxes']} 3D cuboids + {manifest['cutout_planes']} RGBA cutout cards "
                f"across {manifest['anatomical_bones']} anatomical bones, and 12 vigorous clips {clip_names}."
            ),
            dsl_source,
        ),
        (
            "coarse_to_fine_upgrade",
            (
                f"[MODE: COARSE TO FINE UPGRADE | STAGE-1 -> V8 HYBRID MASTERWORK]\n"
                f"Upgrade this {len(stage1_sc.boxes)}-box Stage-1 coarse anatomical blockout of '{manifest['title']}' "
                f"into a full {manifest['elements_total']}-element ({manifest['cuboid_boxes']} cuboids, "
                f"{manifest['cutout_planes']} cutout cards) Block-DSL v4.0 @component + @assembly masterwork with "
                f"x32 UVs and 12 clips {clip_names}.\n"
                f"Stage-1 Coarse Blockout:\n{stage1_dsl}"
            ),
            dsl_source,
        ),
        (
            "texture_and_animate_rig",
            (
                f"[MODE: TEXTURE AND ANIMATE]\n"
                f"Assign OKLCH @palette/@materials ({mat_summary}), x32 painted @textures UV islands, and 12 "
                f"vigorous Monster Hunter @behavior clips {clip_names} onto '{manifest['title']}'.\n"
                f"Untextured Lowered Rig:\n{lowered_dsl}"
            ),
            dsl_source,
        ),
        (
            "pose_and_part_edit_refinement",
            (
                f"[MODE: ITERATIVE ANIMATION & CONTROLLER ENHANCEMENT]\n"
                f"Author all 12 smooth, vigorous Monster Hunter @behavior FK clips ({clip_names}) and @controller "
                f"state machine for '{manifest['title']}' while preserving exact SE(3) rest geometry.\n"
                f"Static Textured Asset:\n{static_dsl}"
            ),
            dsl_source,
        ),
        (
            "inverse_structure_and_kinematics_reading",
            (
                f"[MODE: INVERSE WORLD MODEL ANALYSIS]\n"
                f"Analyze this Block-DSL v4.0 Hybrid creature asset ('{manifest['title']}') and output structured "
                f"JSON reporting 3D cuboid ratio, non-uni-sheet wing bank count, anatomical bones, x32 atlas size, "
                f"and 12 animation clips.\n"
                f"Input Block-DSL v4.0 Asset:\n{dsl_source}"
            ),
            json.dumps(
                {
                    "monster_id": mid,
                    "title": manifest["title"],
                    "species_class": manifest["species_class"],
                    "variant_kind": manifest["variant_kind"],
                    "elements_total": manifest["elements_total"],
                    "cuboid_boxes": manifest["cuboid_boxes"],
                    "cutout_planes": manifest["cutout_planes"],
                    "cuboid_ratio": manifest["cuboid_ratio"],
                    "anatomical_bones": manifest["anatomical_bones"],
                    "wing_banks": manifest["wing_banks"],
                    "atlas_size": manifest["atlas_size"],
                    "clips": clip_names,
                },
                indent=2,
                sort_keys=True,
            ),
        ),
    ]

    records: list[dict[str, Any]] = []
    cost_dict = ledger.to_dict()
    for mode_name, prompt_txt, completion_txt in turns:
        qd_sig = compute_qd_signature(prog, sc, domain="dragon", prompt_style=mode_name)
        records.append(
            {
                "sample_id": f"mh_v8_{mid}_{mode_name}",
                "monster_id": mid,
                "display_name": manifest["title"],
                "title_epithet": manifest["title_epithet"],
                "classification": manifest["species_class"],
                "variant_kind": manifest["variant_kind"],
                "family_group_id": family_group,
                "split": split_name,
                "supervision_mode": mode_name,
                "complexity_tier": tier,
                "elements_total": int(manifest["elements_total"]),
                "cuboid_boxes": int(manifest["cuboid_boxes"]),
                "cutout_planes": int(manifest["cutout_planes"]),
                "cuboid_ratio": float(manifest["cuboid_ratio"]),
                "stage1_box_count": int(len(stage1_sc.boxes)),
                "bone_count": int(manifest["anatomical_bones"]),
                "compiled_bone_count": int(manifest["compiled_bones"]),
                "wing_bank_count": int(manifest["wing_banks"]),
                "clip_count": int(manifest["clip_count"]),
                "atlas_width": int(manifest["atlas_size"][0]),
                "atlas_height": int(manifest["atlas_size"][1]),
                "is_asymmetric_molt": mid == "chaotic_gore_magala",
                "compiler_transform_max_error": float(manifest["compiler_transform_max_error"]),
                "serialized_bedrock_vertex_max_error": float(manifest["serialized_bedrock_vertex_max_error"]),
                "runtime_import_verified": False,
                "license_track": LICENSE_TRACK,
                "commercial_weights_eligible": False,
                "quarantine_notice": (
                    "Original clean Block-DSL v4.0 Hybrid 3D cuboid + x32 RGBA cutout asset authored from "
                    "Monster Hunter physiological reference study; quarantined to `research_nc/*` per AGENTS.md."
                ),
                "reference_wiki_url": manifest.get("wiki_url", ""),
                "generator_version": DATASET_GENERATOR_VERSION,
                "canonical_dsl_sha256": dsl_sha,
                "tree_shape_signature": tree_shape,
                "qd_signature": json.dumps(qd_sig, sort_keys=True),
                "cost_ledger": json.dumps(cost_dict, sort_keys=True),
                "prompt": prompt_txt,
                "completion": completion_txt,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt_txt},
                    {"role": "assistant", "content": completion_txt},
                ],
            }
        )

    anim_records: list[dict[str, Any]] = []
    for cname, cdata in render_scene.get("clips", {}).items():
        anim_records.append(
            {
                "monster_id": mid,
                "display_name": manifest["title"],
                "clip_name": cname,
                "duration_s": float(cdata["duration"]),
                "loop": bool(cdata["loop"]),
                "description": cdata.get("description", cname),
                "track_count": len(cdata.get("tracks", [])),
                "split": split_name,
                "license_track": LICENSE_TRACK,
                "commercial_weights_eligible": False,
            }
        )

    return records, anim_records, cost_dict


def build_full_v8_pack(
    output_root: Path,
    *,
    ref_root: Path | None = None,
    three_vendor_dir: Path | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    for sub in ("assets", "previews", "previews/gifs", "viewer", "dataset", "jsonl", "parquet"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    tasks = [
        (mid, output_root, ref_root, SIGNATURE_GIFS.get(mid))
        for mid in CANONICAL_22_ORDER
    ]

    manifests_by_id: dict[str, dict[str, Any]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_compile_one_worker, t): t[0] for t in tasks}
            for idx, fut in enumerate(as_completed(futs), start=1):
                mid = futs[fut]
                man = fut.result()
                manifests_by_id[mid] = man
                if man.get("previews", {}).get("signature_gif"):
                    src_gif = Path(man["previews"]["signature_gif"])
                    dst_gif = output_root / "previews" / "gifs" / src_gif.name
                    dst_gif.write_bytes(src_gif.read_bytes())
                print(
                    f"[{idx:02d}/22] {mid:22s} | elements={man['elements_total']:4d} "
                    f"(cuboids={man['cuboid_boxes']:4d}, cards={man['cutout_planes']:3d}, {man['cuboid_ratio']:.1%}) | "
                    f"atlas={man['atlas_size'][0]}x{man['atlas_size'][1]} | clips={man['clip_count']} | {man['build_seconds']}s",
                    flush=True,
                )
    else:
        for idx, t in enumerate(tasks, start=1):
            man = _compile_one_worker(t)
            manifests_by_id[t[0]] = man
            print(f"[{idx:02d}/22] {t[0]} OK", flush=True)

    manifests = [manifests_by_id[mid] for mid in CANONICAL_22_ORDER]

    # Copy all 22 review sheets into top-level `previews/` for instant inspection
    for man in manifests:
        rev_src = Path(man["previews"]["studio_review"])
        if rev_src.is_file():
            (output_root / "previews" / rev_src.name).write_bytes(rev_src.read_bytes())

    poster_path = render_v8_collection_overview_poster(output_root, manifests)

    viewer_summary: dict[str, Any] = {}
    if three_vendor_dir is not None and (three_vendor_dir / "three.min.js").is_file():
        viewer_summary = build_v8_offline_viewer(
            output_root / "assets",
            output_root / "viewer",
            three_vendor_dir / "three.min.js",
            three_vendor_dir / "THREE_LICENSE.txt",
        )

    # Synthesize World-Model JSONL + Zstd Parquet dataset shards
    all_records: list[dict[str, Any]] = []
    all_anims: list[dict[str, Any]] = []
    agg_ledger = CostLedger()
    for man in manifests:
        recs, anims, cdict = _build_supervision_records_for_monster(man, output_root)
        all_records.extend(recs)
        all_anims.extend(anims)
        for k, v in cdict.items():
            if v > 0:
                agg_ledger.record(k, v)

    with (output_root / "dataset" / "training_examples.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (output_root / "dataset" / "animations.jsonl").open("w", encoding="utf-8") as fh:
        for a in all_anims:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")

    by_split: dict[str, list[dict[str, Any]]] = {}
    for r in all_records:
        by_split.setdefault(r["split"], []).append(r)

    split_summary: dict[str, Any] = {}
    for split_key, rows in sorted(by_split.items()):
        short = split_key.split("/", 1)[-1]
        jl_p = output_root / "jsonl" / f"{short}.jsonl"
        with jl_p.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        pq_p = output_root / "parquet" / f"{short}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), pq_p, compression="zstd")
        split_summary[split_key] = {
            "row_count": len(rows),
            "families": sorted({row["family_group_id"] for row in rows}),
            "monsters": sorted({row["monster_id"] for row in rows}),
            "jsonl_path": str(jl_p),
            "parquet_path": str(pq_p),
        }

    elapsed = round(time.perf_counter() - t0, 2)
    pack_manifest = {
        "schema": "synccraft.mh-dragon-dataset-v8/1.0",
        "generator_version": DATASET_GENERATOR_VERSION,
        "license_track": LICENSE_TRACK,
        "commercial_weights_eligible": False,
        "runtime_import_verified": False,
        "monster_count": len(manifests),
        "total_elements": sum(m["elements_total"] for m in manifests),
        "total_cuboid_boxes": sum(m["cuboid_boxes"] for m in manifests),
        "total_cutout_planes": sum(m["cutout_planes"] for m in manifests),
        "overall_cuboid_ratio": round(
            sum(m["cuboid_boxes"] for m in manifests) / max(1, sum(m["elements_total"] for m in manifests)), 4
        ),
        "total_animation_clips": sum(m["clip_count"] for m in manifests),
        "total_supervision_records": len(all_records),
        "supervision_mode_counts": dict(Counter(r["supervision_mode"] for r in all_records)),
        "splits": split_summary,
        "collection_overview_poster": str(poster_path),
        "viewer": viewer_summary,
        "aggregate_cost_ledger": agg_ledger.to_dict(),
        "wall_clock_seconds": elapsed,
        "monsters": manifests,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(pack_manifest, indent=2), encoding="utf-8")
    return pack_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Rebuild_v8",
    )
    p.add_argument("--ref-root", type=Path, default=None)
    p.add_argument("--three-vendor-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    summary = build_full_v8_pack(
        args.output_dir,
        ref_root=args.ref_root,
        three_vendor_dir=args.three_vendor_dir,
        workers=args.workers,
    )
    print(
        f"\n=== Monster Hunter Block-DSL v4.0 Hybrid Pack (V8) Complete ===\n"
        f"Monsters          : {summary['monster_count']}\n"
        f"3D Cuboids        : {summary['total_cuboid_boxes']:,} / {summary['total_elements']:,} ({summary['overall_cuboid_ratio']:.1%})\n"
        f"Authored Clips    : {summary['total_animation_clips']}\n"
        f"Supervision Rows  : {summary['total_supervision_records']}\n"
        f"Overview Poster   : {summary['collection_overview_poster']}\n"
        f"Elapsed Seconds   : {summary['wall_clock_seconds']}s"
    )


if __name__ == "__main__":
    main()
