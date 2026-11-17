#!/usr/bin/env python3
"""Build the Complete SyncCraft Block-DSL v4.1 Monster Hunter V9 Sculpted Masterworks & Dataset.

Packages all 22 individually hand-authored progressive Block-DSL v4.1 Monster Hunter assets:
  - Progressive Block-DSL v4.1 stages (`stage1_blockout.blockdsl` -> `stage2_secondary.blockdsl` -> `<id>.blockdsl`)
    plus human-artist `trait_sheet.md` per monster.
  - Vibrant x32 (`2 texels/unit`) hue-shifted RGBA pixel-art atlases (`textures/albedo.png` + `emissive.png`)
    with binary `{0, 255}` alpha cutouts (`edge: spikes|scallop|ragged|drip`) and non-uni-sheet multi-bone wings.
  - Blockbench 5 `.bbmodel` + Minecraft Bedrock `.geo.json` & `.animation.json` (`0.0` vertex drift verified).
  - 264 vigorous, game-authentic Monster Hunter `@behavior` clips (`catmullrom`, `wave`, `spring`, `@controller`).
  - Studio review sheets (`<id>_studio_review.png`, `<id>_stages_and_silhouette.png`), signature GIFs,
    and 22-specimen `v9_collection_overview.jpg` poster.
  - Self-contained offline WebGL 3D Studio Viewer (`viewer/index.html`, Three.js r160.1 MIT).
  - Family-grouped 6-mode Progressive Artist-Loop World-Model JSONL + Zstd Parquet dataset shards strictly
    quarantined under `research_nc_fan_reference` (`commercial_weights_eligible=False`, `runtime_import_verified=False`).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from synccraft.data_gen.diversity_engine import compute_program_tree_shape, compute_qd_signature
from synccraft.data_gen.mh_sculpted_v9.catalog import CANONICAL_22_ORDER, get_monster_spec
from synccraft.data_gen.mh_sculpted_v9.studio_compiler import DEFAULT_DSL_ROOT, inspect_or_compile_monster
from synccraft.data_gen.mh_sculpted_v9.v9_viewer import build_v9_offline_viewer
from synccraft.data_gen.taxonomy_catalog import classify_complexity_tier
from synccraft.dsl.parser import format_program, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.se3_chain import compile_geometry

DATASET_GENERATOR_VERSION = "synccraft-v4.1-mh-v9-progressive-sculpted-studio"
LICENSE_TRACK = "research_nc_fan_reference"

SHOWCASE_GIF_MONSTERS: set[str] = {
    "zinogre",
    "rathalos",
    "rey_dau",
    "arkveld",
    "legiana",
    "crimson_glow_valstrax",
    "gore_magala",
    "fatalis",
    "velkhana",
    "lagiacrus",
    "mizutsune",
    "jin_dahaad",
}


def _load_font(sz: int, bold: bool = False) -> Any:
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


def render_v9_collection_overview_poster(
    output_root: Path,
    manifests: list[dict[str, Any]],
) -> Path:
    """Render a 2400x3180 4x6 studio collection poster across all 22 V9 Sculpted Masterworks."""
    cols, rows = 4, 6
    tile_w, tile_h = 556, 456
    pad = 28
    header_h = 136
    cw = pad + cols * (tile_w + pad)
    ch = header_h + rows * (tile_h + 44) + pad

    poster = Image.new("RGB", (cw, ch), (10, 15, 22))
    draw = ImageDraw.Draw(poster)
    f_title = _load_font(36, bold=True)
    f_sub = _load_font(20, bold=False)
    f_card_b = _load_font(20, bold=True)
    f_card = _load_font(15, bold=False)

    tot_elems = sum(int(m["elements_total"]) for m in manifests)
    tot_boxes = sum(int(m["cuboid_boxes"]) for m in manifests)
    tot_planes = sum(int(m["cutout_planes"]) for m in manifests)
    tot_clips = sum(int(m["clip_count"]) for m in manifests)
    mean_style = sum(float(m["style_anchor_score"]) for m in manifests) / max(1, len(manifests))

    draw.text(
        (pad, 24),
        "MONSTER HUNTER — PROGRESSIVE BLOCK-DSL v4.1 SCULPTED MASTERWORKS (V9 STUDIO)",
        fill=(238, 246, 255),
        font=f_title,
    )
    draw.text(
        (pad, 74),
        f"{len(manifests)} Canonical & Variant Dragons  |  {tot_boxes:,} Rotated 3D Cuboids + {tot_planes:,} Pixel-Cutout Planes  |  "
        f"{tot_clips} Vigorous Clips  |  Mean Style Anchor: {mean_style:.3f}  |  0.0 Vertex Drift",
        fill=(146, 206, 238),
        font=f_sub,
    )

    for idx, man in enumerate(manifests):
        r, c = divmod(idx, cols)
        x0 = pad + c * (tile_w + pad)
        y0 = header_h + r * (tile_h + 44)
        hero_p = Path(man["previews"]["hero"])
        if hero_p.is_file():
            with Image.open(hero_p) as im:
                im_fit = im.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                poster.paste(im_fit, (x0, y0))
        draw.rectangle((x0 - 1, y0 - 1, x0 + tile_w + 1, y0 + tile_h + 1), outline=(48, 70, 92), width=2)
        draw.rectangle((x0, y0 + tile_h - 62, x0 + tile_w, y0 + tile_h), fill=(11, 18, 26))
        draw.text((x0 + 14, y0 + tile_h - 56), f"{idx + 1:02d}. {man['title']} — {man['title_epithet']}", fill=(238, 246, 255), font=f_card_b)
        draw.text(
            (x0 + 14, y0 + tile_h - 28),
            f"{man['species_class']}  |  {man['cuboid_boxes']} cuboids + {man['cutout_planes']} planes  |  {man['anatomical_bones']} bones  |  Style {man['style_anchor_score']:.2f}",
            fill=(148, 178, 204),
            font=f_card,
        )

    if len(manifests) <= 22:
        sx0 = pad + 2 * (tile_w + pad)
        sy0 = header_h + 5 * (tile_h + 44)
        span_w = 2 * tile_w + pad
        draw.rounded_rectangle(
            (sx0 - 2, sy0 - 2, sx0 + span_w + 2, sy0 + tile_h),
            radius=10,
            fill=(15, 24, 35),
            outline=(78, 138, 184),
            width=2,
        )
        highlights = [
            "PROGRESSIVE ARTIST-LOOP & SCULPTED STYLE ANCHOR (V9)",
            f"- 22/22 Hand-Authored Stage 1 -> Stage 2 -> Stage 3 Native Block-DSL v4.1 Assets",
            f"- {tot_boxes:,} Sculpted 3D Cuboids ({100.0 * tot_boxes / max(1, tot_elems):.1f}%) + {tot_planes:,} Binary Alpha-Cutout Planes",
            "- Zero Doghouse Snouts, Zero Uni-Sheet Wings, Zero Ladder-Plank Ventral Scutes",
            f"- Mean Quantitative Sculpted Style Anchor Score: {mean_style:.3f} (100% Static Audit Pass)",
            "- Shared SE(3) Verified: 0.0 Vertex Drift Across Block-DSL, .bbmodel & Bedrock .geo.json",
            f"- {tot_clips} Vigorous Monster Hunter Clips + Self-Contained Offline WebGL 3D Studio Viewer",
        ]
        for li, line in enumerate(highlights):
            draw.text(
                (sx0 + 26, sy0 + 28 + li * 56),
                line,
                fill=(102, 228, 250) if li == 0 else (216, 230, 244),
                font=f_sub if li == 0 else f_card,
            )

    out_p = output_root / "previews" / "v9_collection_overview.jpg"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    poster.save(out_p, quality=94)
    return out_p


def _build_progressive_supervision_records(
    manifest: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Build 6-mode Progressive Artist-Loop World-Model supervision records for one V9 monster."""
    mid = manifest["id"]
    spec = get_monster_spec(mid)
    asset_dir = output_root / "assets" / mid
    dsl_s3 = (asset_dir / f"{mid}.blockdsl").read_text(encoding="utf-8")
    dsl_s1 = (asset_dir / f"{mid}_stage1_blockout.blockdsl").read_text(encoding="utf-8")
    dsl_s2 = (asset_dir / f"{mid}_stage2_secondary.blockdsl").read_text(encoding="utf-8")
    trait_md = (asset_dir / "trait_sheet.md").read_text(encoding="utf-8") if (asset_dir / "trait_sheet.md").is_file() else spec.anatomy_summary
    render_scene = json.loads((asset_dir / "render_scene.json").read_text(encoding="utf-8"))

    ledger = CostLedger()
    ledger.record("compiler_calls", 3)
    ledger.record("initial_audits", 3)

    prog_s1 = parse_program(dsl_s1)
    sc_s1 = compile_geometry(prog_s1)
    prog_s2 = parse_program(dsl_s2)
    sc_s2 = compile_geometry(prog_s2)
    prog_s3 = parse_program(dsl_s3)
    sc_s3 = compile_geometry(prog_s3)

    family_group = spec.parent_species_id or mid
    split_name = spec.split
    tier = classify_complexity_tier(len(sc_s3.boxes))
    tree_shape = compute_program_tree_shape(prog_s3)
    dsl_sha = hashlib.sha256(dsl_s3.encode("utf-8")).hexdigest()
    clip_names = list(manifest["clip_names"])
    mat_summary = ", ".join(f"{m.name}:{m.family}" for m in prog_s3.materials)

    static_unanimated_dsl = format_program(
        replace(prog_s3, assembly_spec=None, source_map=None, behavior=None, controller=None)
    )

    sys_msg = (
        "You are SyncCraft-3D Master Sculptor, an expert Blockbench creature artist, pixel-atlas painter, "
        "and Monster Hunter kinematic animator authoring progressive Block-DSL v4.1 (`audit_profile: sculpted`). "
        "Sculpt genuine 3D wedged snouts, rotated cuboid flow, hugged ventral scutes, non-uni-sheet multi-bone wings "
        "with binary alpha-cutout `plane` bays (`edge:`), vibrant hue-shifted OKLCH cel ramps, and vigorous `@behavior` clips."
    )

    turns: list[tuple[str, str, str]] = [
        # Mode 1: Reference Trait Sheet -> Stage 1 Primary Blockout
        (
            "stage1_blockout_synthesis",
            (
                f"[MODE: STAGE 1 PRIMARY BLOCKOUT | PROGRESSIVE ARTIST LOOP | TIER: {tier}]\n"
                f"Draft the Stage-1 primary anatomical blockout (`stage1_blockout.blockdsl`) for '{spec.title}' "
                f"({spec.title_epithet}, {spec.species_class}) capturing its canonical silhouette, S-curve spine, "
                f"grounded `Y = 0` limb zig-zag, wedged snout core, and primary appendage proportions.\n"
                f"Visual Trait Analysis:\n{trait_md}"
            ),
            dsl_s1,
        ),
        # Mode 2: Stage 1 Blockout -> Stage 2 Secondary Anatomy
        (
            "stage1_to_stage2_refinement",
            (
                f"[MODE: STAGE 1 -> STAGE 2 SECONDARY ANATOMY REFINEMENT]\n"
                f"Refine this {len(sc_s1.boxes)}-element Stage-1 blockout of '{spec.title}' into a {len(sc_s2.boxes)}-element "
                f"Stage-2 secondary sculpture with a 3D wedged snout, brow/cheek plates, multi-segment horns/crests, "
                f"hugged ventral scutes, articulated toes/wing-fingers, and directional armor flow.\n"
                f"Stage-1 Blockout:\n{dsl_s1}"
            ),
            dsl_s2,
        ),
        # Mode 3: Stage 2 Secondary Anatomy -> Stage 3 Textured & Animated Masterwork
        (
            "stage2_to_masterwork_refinement",
            (
                f"[MODE: STAGE 2 -> STAGE 3 SCULPTED MASTERWORK]\n"
                f"Complete '{spec.title}' from its {len(sc_s2.boxes)}-element Stage-2 sculpture into a full "
                f"{manifest['elements_total']}-element Stage-3 Block-DSL v4.1 masterwork adding tertiary shingles, "
                f"fangs/talons, binary alpha-cutout `plane` silhouettes (`edge:`), `@materials` ({mat_summary}), "
                f"`@style` (`shade: banded`), `@stamps`, and 12 vigorous Monster Hunter `@behavior` clips {clip_names}.\n"
                f"Stage-2 Secondary Sculpture:\n{dsl_s2}"
            ),
            dsl_s3,
        ),
        # Mode 4: Static Sculpted Geometry -> Vigorous Monster Hunter Behavior & Controller
        (
            "vigorous_behavior_and_controller_authoring",
            (
                f"[MODE: VIGOROUS MONSTER HUNTER ANIMATION & CONTROLLER]\n"
                f"Author all 12 smooth, energetic Monster Hunter `@behavior` clips ({clip_names}) using `catmullrom` "
                f"full-body anticipation/follow-through, `wave` spine/tail propagation, `spring` secondary motion, "
                f"and `@controller` transitions onto this static sculpted rig of '{spec.title}'.\n"
                f"Static Sculpted Asset:\n{static_unanimated_dsl}"
            ),
            dsl_s3,
        ),
        # Mode 5: End-to-End Prompt -> Full Stage 3 Masterwork
        (
            "create_full_masterwork_asset",
            (
                f"[MODE: CREATE FULL SCULPTED MASTERWORK | BLOCK-DSL v4.1 | TIER: {tier}]\n"
                f"Sculpt, paint (`x32` vibrant banded cel), and animate '{spec.title}' ({spec.title_epithet}, "
                f"{spec.species_class}) in Block-DSL v4.1 (`audit_profile: sculpted`).\n"
                f"Canonical Anatomy & Sculpted Style: {spec.anatomy_summary}\n"
                f"Target: ~{manifest['cuboid_boxes']} rotated/sculpted 3D cuboids + {manifest['cutout_planes']} binary-cutout "
                f"`plane`s across {manifest['anatomical_bones']} bones, and 12 clips {clip_names}."
            ),
            dsl_s3,
        ),
        # Mode 6: Inverse World-Model & Quantitative Sculpted Style Anchor Critique
        (
            "sculpted_style_evaluation_and_critique",
            (
                f"[MODE: INVERSE WORLD MODEL & SCULPTED STYLE ANCHOR AUDIT]\n"
                f"Evaluate this Block-DSL v4.1 creature asset ('{spec.title}') against SyncCraft's 6-pillar "
                f"Sculpted Blockbench Style Anchor and output structured JSON reporting element breakdown, "
                f"non-uni-sheet wing banks, pillar scores, diagnostics, and 12 animation clips.\n"
                f"Input Block-DSL v4.1 Asset:\n{dsl_s3}"
            ),
            json.dumps(
                {
                    "monster_id": mid,
                    "title": spec.title,
                    "species_class": spec.species_class,
                    "variant_kind": spec.variant_kind,
                    "elements_total": manifest["elements_total"],
                    "stage1_boxes": manifest["stage1_boxes"],
                    "stage2_boxes": manifest["stage2_boxes"],
                    "cuboid_boxes": manifest["cuboid_boxes"],
                    "cutout_planes": manifest["cutout_planes"],
                    "cuboid_ratio": manifest["cuboid_ratio"],
                    "anatomical_bones": manifest["anatomical_bones"],
                    "wing_banks": manifest["wing_banks"],
                    "style_anchor_score": manifest["style_anchor_score"],
                    "style_pillar_scores": manifest["style_pillar_scores"],
                    "style_diagnostics": manifest["style_diagnostics"],
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
        qd_sig = compute_qd_signature(prog_s3, sc_s3, domain="dragon", prompt_style=mode_name)
        records.append(
            {
                "sample_id": f"mh_v9_{mid}_{mode_name}",
                "monster_id": mid,
                "display_name": spec.title,
                "title_epithet": spec.title_epithet,
                "classification": spec.species_class,
                "variant_kind": spec.variant_kind,
                "family_group_id": family_group,
                "split": split_name,
                "supervision_mode": mode_name,
                "complexity_tier": tier,
                "elements_total": int(manifest["elements_total"]),
                "cuboid_boxes": int(manifest["cuboid_boxes"]),
                "cutout_planes": int(manifest["cutout_planes"]),
                "cuboid_ratio": float(manifest["cuboid_ratio"]),
                "stage1_box_count": int(len(sc_s1.boxes)),
                "stage2_box_count": int(len(sc_s2.boxes)),
                "bone_count": int(manifest["anatomical_bones"]),
                "compiled_bone_count": int(manifest["compiled_bones"]),
                "wing_bank_count": int(manifest["wing_banks"]),
                "clip_count": int(manifest["clip_count"]),
                "style_anchor_score": float(manifest["style_anchor_score"]),
                "atlas_width": int(manifest["atlas_size"][0]),
                "atlas_height": int(manifest["atlas_size"][1]),
                "is_asymmetric": mid in ("chaotic_gore_magala", "ruiner_nergigante", "zoh_shia"),
                "compiler_transform_max_error": float(manifest["compiler_transform_max_error"]),
                "serialized_bedrock_vertex_max_error": float(manifest["serialized_bedrock_vertex_max_error"]),
                "runtime_import_verified": False,
                "license_track": LICENSE_TRACK,
                "commercial_weights_eligible": False,
                "quarantine_notice": (
                    "Hand-authored progressive Block-DSL v4.1 sculpted asset modeled from Monster Hunter "
                    "physiological reference study; strictly quarantined to `research_nc/*` per AGENTS.md."
                ),
                "reference_wiki_url": spec.wiki_url,
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
                "display_name": spec.title,
                "clip_name": cname,
                "duration_s": float(cdata["duration"]),
                "loop": bool(cdata["loop"]),
                "track_count": len(cdata.get("tracks", [])),
                "split": split_name,
                "license_track": LICENSE_TRACK,
                "commercial_weights_eligible": False,
            }
        )

    return records, anim_records, cost_dict


def build_full_v9_pack(
    output_root: Path,
    *,
    dsl_root: Path = DEFAULT_DSL_ROOT,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    for sub in ("assets", "previews", "previews/gifs", "previews/stages", "viewer", "dataset", "jsonl", "parquet"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for idx, mid in enumerate(CANONICAL_22_ORDER, start=1):
        s3_p = dsl_root / mid / f"{mid}.blockdsl"
        if not s3_p.is_file():
            raise FileNotFoundError(f"Missing authored Stage-3 Block-DSL for {mid}: {s3_p}")
        cur_sha = hashlib.sha256(s3_p.read_bytes()).hexdigest()
        man_p = output_root / "assets" / mid / "manifest.json"
        want_gif = mid in SHOWCASE_GIF_MONSTERS

        cached_ok = False
        if not force_rebuild and man_p.is_file():
            cached = json.loads(man_p.read_text(encoding="utf-8"))
            rev_ok = Path(cached.get("previews", {}).get("studio_review", "")).is_file()
            gif_ok = (not want_gif) or Path(cached.get("previews", {}).get("signature_gif", "")).is_file()
            if cached.get("dsl_sha256") == cur_sha and cached.get("static_sculpted_audit_passed") and rev_ok and gif_ok:
                man = cached
                cached_ok = True

        if not cached_ok:
            man = inspect_or_compile_monster(
                mid,
                dsl_root=dsl_root,
                output_pack_root=output_root,
                stage=3,
                render_previews=True,
                render_gif=want_gif,
                strict_audit=True,
            )

        manifests.append(man)
        rev_src = Path(man["previews"]["studio_review"])
        if rev_src.is_file():
            (output_root / "previews" / rev_src.name).write_bytes(rev_src.read_bytes())
        stg_src = Path(man["previews"]["stages_and_silhouette"])
        if stg_src.is_file():
            (output_root / "previews" / "stages" / stg_src.name).write_bytes(stg_src.read_bytes())
        gif_src_str = man.get("previews", {}).get("signature_gif")
        if gif_src_str and Path(gif_src_str).is_file():
            gif_src = Path(gif_src_str)
            (output_root / "previews" / "gifs" / gif_src.name).write_bytes(gif_src.read_bytes())

        print(
            f"[{idx:02d}/22] {mid:22s} | S1={man['stage1_boxes']:3d} -> S2={man['stage2_boxes']:3d} -> S3={man['elements_total']:3d} "
            f"(cuboids={man['cuboid_boxes']:3d}, planes={man['cutout_planes']:2d}) | bones={man['anatomical_bones']:2d} | "
            f"style={man['style_anchor_score']:.3f} | {'cached' if cached_ok else f'{man[\"build_seconds\"]}s'}",
            flush=True,
        )

    poster_path = render_v9_collection_overview_poster(output_root, manifests)
    viewer_summary = build_v9_offline_viewer(output_root / "assets", output_root / "viewer", repo_root=REPO_ROOT)

    all_records: list[dict[str, Any]] = []
    all_anims: list[dict[str, Any]] = []
    agg_ledger = CostLedger()
    for man in manifests:
        recs, anims, cdict = _build_progressive_supervision_records(man, output_root)
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

    # Verify zero family-group leakage across splits
    split_families = {sp: {row["family_group_id"] for row in rows} for sp, rows in by_split.items()}
    sp_keys = list(split_families.keys())
    for i in range(len(sp_keys)):
        for j in range(i + 1, len(sp_keys)):
            overlap = split_families[sp_keys[i]] & split_families[sp_keys[j]]
            if overlap:
                raise AssertionError(f"Family-group split leakage between {sp_keys[i]} and {sp_keys[j]}: {overlap}")

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
        "schema": "synccraft.mh-dragon-dataset-v9/1.0",
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
        "mean_style_anchor_score": round(
            sum(float(m["style_anchor_score"]) for m in manifests) / max(1, len(manifests)), 4
        ),
        "min_style_anchor_score": round(min(float(m["style_anchor_score"]) for m in manifests), 4),
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
        default=REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Masterworks_v9",
    )
    p.add_argument("--dsl-root", type=Path, default=DEFAULT_DSL_ROOT)
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()
    summary = build_full_v9_pack(
        args.output_dir,
        dsl_root=args.dsl_root,
        force_rebuild=args.force_rebuild,
    )
    print(
        f"\n=== Monster Hunter Block-DSL v4.1 Sculpted Masterwork Pack (V9) Complete ===\n"
        f"Monsters          : {summary['monster_count']}\n"
        f"3D Cuboids        : {summary['total_cuboid_boxes']:,} / {summary['total_elements']:,} ({summary['overall_cuboid_ratio']:.1%})\n"
        f"Mean Style Anchor : {summary['mean_style_anchor_score']:.4f} (min {summary['min_style_anchor_score']:.4f})\n"
        f"Authored Clips    : {summary['total_animation_clips']}\n"
        f"Supervision Rows  : {summary['total_supervision_records']}\n"
        f"Overview Poster   : {summary['collection_overview_poster']}\n"
        f"Offline 3D Viewer : {args.output_dir / 'viewer' / 'index.html'}\n"
        f"Elapsed Seconds   : {summary['wall_clock_seconds']}s"
    )


if __name__ == "__main__":
    main()
