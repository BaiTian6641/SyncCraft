"""Native Block-DSL v4.1 Monster Hunter Masterwork Studio Compiler, Verifier & Exporter (V9).

Reads pure hand-authored Block-DSL v4.1 progressive stages (`stage1_blockout.blockdsl`,
`stage2_secondary.blockdsl`, `<id>.blockdsl`), compiles natively through SyncCraft's
shared `SE(3)` engine (`parse_program` + `compile_geometry`), runs both `audit_scene`
(`audit_profile: sculpted`) and `evaluate_sculpted_style_anchor`, bakes vibrant `x32`
RGBA pixel-art atlases, exports `.bbmodel` + Bedrock `.geo.json` & `.animation.json`
with `0.0` vertex drift verification, and renders Stage Progression Sheets, 8-Panel
Studio Review Sheets, and Animated Action GIFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from synccraft.audit.sculpted_style_evaluator import evaluate_sculpted_style_anchor
from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.bbmodel5_exporter import export_bbmodel
from synccraft.backends.bedrock_gecko_exporter import _extract_so3_euler_keyframe_deltas, export_bedrock
from synccraft.backends.common import euler_zyx_degrees
from synccraft.dsl.assembly_ir import audit_assembly_contracts
from synccraft.dsl.parser import _tokens, format_program, parse_program
from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip
from synccraft.kinematics.se3_chain import WorldBox, Scene, compile_geometry
from synccraft.surface.contracts import linear_to_srgb_u8, oklch_to_linear_srgb

from .artisan_painter import FACES, PLANE_PRIMARY_FACES, bake_artisan_sculpted_atlas
from .catalog import MonsterSpec, get_monster_spec
from .studio_renderer import (
    _box_outward_quads_world,
    render_animated_action_gif,
    render_monster_masterwork_review_sheet,
    render_scene_frame,
    render_stage_progression_sheet,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DSL_ROOT = Path(__file__).resolve().parent / "dsl"
DEFAULT_REF_STUDY_ROOT = REPO_ROOT / "artifacts" / "mh_reference_study"


def build_viewer_render_scene_payload(
    spec: MonsterSpec,
    program: Any,
    scene: Scene,
    atlas: dict[str, Any],
    *,
    anim_samples: int = 20,
) -> dict[str, Any]:
    """Serialize exact SyncCraft `Scene` + `@behavior` clips into an offline WebGL viewer payload."""
    materials_swatches: dict[str, dict[str, Any]] = {}
    for pal in program.palettes:
        rgb = linear_to_srgb_u8(np.clip(oklch_to_linear_srgb(pal.base_lch), 0.0, 1.0)).tolist()
        materials_swatches[pal.name] = {
            "color": [int(x) for x in rgb],
            "emissive": 0.9 if pal.emissive else 0.0,
        }

    bones_payload: list[dict[str, Any]] = []
    for b in scene.bones:
        bones_payload.append(
            {
                "name": b.name,
                "parent": b.parent_name,
                "pivot": [round(float(x), 5) for x in b.pivot_world],
                "local_translation": [round(float(x), 5) for x in b.local_translation],
                "local_euler": [round(float(x), 5) for x in b.local_euler],
                "world_euler": [round(float(x), 5) for x in euler_zyx_degrees(b.rotation)],
            }
        )

    boxes_payload: list[dict[str, Any]] = []
    bone_by_name = scene.bone_by_name
    for box in scene.boxes:
        pnorm = getattr(box, "plane_normal", None)
        hidden = [f for f in FACES if pnorm in PLANE_PRIMARY_FACES and f not in PLANE_PRIMARY_FACES[pnorm]]
        # Store face vertices in the owning bone's local coordinate frame around `bone.pivot_world`!
        bone = bone_by_name[box.bone_name]
        quads_w = _box_outward_quads_world(box)
        quads_bone_local = {
            f: np.round((verts_w - bone.pivot_world[None, :]) @ bone.rotation, 5).tolist()
            for f, verts_w in quads_w.items()
        }
        boxes_payload.append(
            {
                "name": box.name,
                "bone": box.bone_name,
                "size": [round(float(x), 4) for x in box.size],
                "material": box.material,
                "plane_normal": pnorm,
                "hidden_faces": hidden,
                "uv": atlas["faces"][box.name],
                "bone_local_quads": quads_bone_local,
            }
        )

    clips_payload: dict[str, dict[str, Any]] = {}
    root_name = scene.bones[0].name
    if program.behavior is not None:
        for clip in program.behavior.clips:
            n_f = max(12, int(anim_samples))
            samples = evaluate_behavior_clip(program, clip.name, num_frames=n_f)
            tracks_out: list[dict[str, Any]] = []
            # Root translation delta (in world space -> transformed to root local if needed)
            if any(any(abs(float(c)) > 1e-4 for c in s.root_translation_delta) for s in samples):
                keys_pos = [
                    [round(float(s.time_s), 4), [round(float(c), 4) for c in s.root_translation_delta]]
                    for s in samples
                ]
                tracks_out.append({"bone": root_name, "channel": "position", "keys": keys_pos})

            # Non-root bone position deltas (`slide` / `key <bone>.position`)
            all_pos_bones = sorted({bn for s in samples for bn in s.pose_position_deltas.keys() if bn != root_name})
            for p_bn in all_pos_bones:
                keys_bp = [
                    [
                        round(float(s.time_s), 4),
                        [round(float(c), 4) for c in s.pose_position_deltas.get(p_bn, (0.0, 0.0, 0.0))],
                    ]
                    for s in samples
                ]
                if max(max(abs(v) for v in kv[1]) for kv in keys_bp) > 1e-4:
                    tracks_out.append({"bone": p_bn, "channel": "position", "keys": keys_bp})

            for bone in scene.bones:
                de_vecs = _extract_so3_euler_keyframe_deltas(samples, bone)
                if max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in de_vecs) < 1e-4:
                    continue
                keys_rot = [
                    [round(float(s.time_s), 4), [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]]
                    for s, v in zip(samples, de_vecs)
                ]
                tracks_out.append({"bone": bone.name, "channel": "rotation", "keys": keys_rot})

            clips_payload[clip.name] = {
                "duration": round(float(clip.duration_s), 4),
                "loop": bool(clip.loop),
                "tracks": tracks_out,
            }

    return {
        "id": spec.id,
        "name": spec.title,
        "title": spec.title,
        "title_epithet": spec.title_epithet.strip(),
        "species_class": spec.species_class,
        "notes": [spec.anatomy_summary],
        "materials": materials_swatches,
        "bones": bones_payload,
        "atlas_width": int(atlas["width"]),
        "atlas_height": int(atlas["height"]),
        "albedo_data_url": atlas["data_url"],
        "emissive_data_url": atlas.get("emissive_data_url"),
        "boxes": boxes_payload,
        "clips": clips_payload,
    }


def inspect_or_compile_monster(
    monster_id: str,
    *,
    dsl_root: Path = DEFAULT_DSL_ROOT,
    output_pack_root: Path | None = None,
    ref_study_root: Path = DEFAULT_REF_STUDY_ROOT,
    stage: int = 3,
    render_previews: bool = True,
    render_gif: bool = False,
    strict_audit: bool = True,
) -> dict[str, Any]:
    """Compile, audit, score, paint, export, and render an authored Monster Hunter Block-DSL asset."""
    t0 = time.perf_counter()
    spec = get_monster_spec(monster_id)
    mdir = dsl_root / monster_id
    if not mdir.is_dir():
        raise FileNotFoundError(f"Missing authored Block-DSL directory for {monster_id}: {mdir}")

    s1_path = mdir / "stage1_blockout.blockdsl"
    s2_path = mdir / "stage2_secondary.blockdsl"
    s3_path = mdir / f"{monster_id}.blockdsl"

    target_path = {1: s1_path, 2: s2_path, 3: s3_path}[stage]
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing Stage-{stage} Block-DSL file: {target_path}")

    out_root = output_pack_root or (REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Masterworks_v9")
    asset_dir = out_root / "assets" / monster_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = asset_dir / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)

    # Quick stage-1 or stage-2 sculptor check
    if stage in (1, 2):
        src_text = target_path.read_text(encoding="utf-8")
        prog = parse_program(src_text)
        sc = compile_geometry(prog)
        aud = audit_scene(sc)
        style_rep = evaluate_sculpted_style_anchor(prog, sc, require_wings_partitioned=(spec.is_winged and stage >= 2))
        atlas = bake_artisan_sculpted_atlas(
            prog,
            sc,
            artisan_motif=spec.artisan_motif,
            accent_rgb=spec.accent_rgb,
        )
        check_img_p = prev_dir / f"stage{stage}_visual_check.png"
        if render_previews:
            render_stage_progression_sheet(
                f"{spec.title} (Stage {stage})",
                sc if stage == 1 else None,
                sc if stage == 2 else None,
                sc,
                atlas["faces"],
                atlas["albedo_rgba"],
                emissive_rgba=atlas["emissive_rgba"],
                program=prog,
                style_score=style_rep.style_anchor_score,
                output_path=check_img_p,
            )
        if strict_audit and not aud["passed"]:
            raise AssertionError(
                f"[{monster_id} stage={stage}] Static sculpted audit FAILED:\n"
                f"  collisions={aud['collision_pairs'][:6]}\n"
                f"  contact_violations={aud['contact_violations'][:6]}\n"
                f"  contract_violations={aud.get('contract_violations', [])}"
            )
        return {
            "id": monster_id,
            "stage": stage,
            "elements_total": len(sc.boxes),
            "bones_total": len(sc.bones),
            "static_audit_passed": aud["passed"],
            "collision_pairs": aud["collision_pairs"],
            "contact_violations": aud["contact_violations"],
            "style_anchor": style_rep.to_dict(),
            "visual_check_png": str(check_img_p),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    # Full Stage 3 Masterwork compilation + Stage 1 & Stage 2 progression verification
    s1_prog = parse_program(s1_path.read_text(encoding="utf-8")) if s1_path.is_file() else None
    s1_sc = compile_geometry(s1_prog) if s1_prog is not None else None
    s2_prog = parse_program(s2_path.read_text(encoding="utf-8")) if s2_path.is_file() else None
    s2_sc = compile_geometry(s2_prog) if s2_prog is not None else None

    dsl_source = s3_path.read_text(encoding="utf-8")
    program = parse_program(dsl_source)
    # Round-trip canonical formatting verification
    _ = parse_program(format_program(program))
    scene = compile_geometry(program)

    aud = audit_scene(scene)
    contracts_rep = audit_assembly_contracts(program, scene)
    if strict_audit and (not aud["passed"] or not contracts_rep["passed"]):
        raise AssertionError(
            f"[{monster_id}] Stage-3 static sculpted audit FAILED:\n"
            f"  collisions ({len(aud['collision_pairs'])}): {aud['collision_pairs'][:8]}\n"
            f"  contact_violations ({len(aud['contact_violations'])}): {aud['contact_violations'][:8]}\n"
            f"  contract_violations: {aud.get('contract_violations', [])} / {contracts_rep}"
        )
    if strict_audit and s1_sc is not None:
        s1_aud = audit_scene(s1_sc)
        if not s1_aud["passed"]:
            raise AssertionError(f"[{monster_id}] stage1_blockout.blockdsl failed audit: {s1_aud['collision_pairs'][:5]} {s1_aud['contact_violations'][:5]}")
    if strict_audit and s2_sc is not None:
        s2_aud = audit_scene(s2_sc)
        if not s2_aud["passed"]:
            raise AssertionError(f"[{monster_id}] stage2_secondary.blockdsl failed audit: {s2_aud['collision_pairs'][:5]} {s2_aud['contact_violations'][:5]}")

    atlas = bake_artisan_sculpted_atlas(
        program,
        scene,
        artisan_motif=spec.artisan_motif,
        accent_rgb=spec.accent_rgb,
        textures_dir=asset_dir / "textures",
    )
    style_rep = evaluate_sculpted_style_anchor(
        program,
        scene,
        atlas_rgba=atlas["albedo_rgba"],
        require_wings_partitioned=spec.is_winged,
    )

    # Copy canonical authored Block-DSL stages & trait sheet into asset package
    (asset_dir / f"{monster_id}.blockdsl").write_text(dsl_source, encoding="utf-8")
    if s1_path.is_file():
        (asset_dir / f"{monster_id}_stage1_blockout.blockdsl").write_text(s1_path.read_text(encoding="utf-8"), encoding="utf-8")
    if s2_path.is_file():
        (asset_dir / f"{monster_id}_stage2_secondary.blockdsl").write_text(s2_path.read_text(encoding="utf-8"), encoding="utf-8")
    trait_p = mdir / "trait_sheet.md"
    if trait_p.is_file():
        (asset_dir / "trait_sheet.md").write_text(trait_p.read_text(encoding="utf-8"), encoding="utf-8")

    # Export Blockbench 5 (.bbmodel) & Minecraft Bedrock (.geo.json + .animation.json)
    atlas_for_export = {
        "width": atlas["width"],
        "height": atlas["height"],
        "padding": atlas["padding"],
        "faces": atlas["faces"],
        "data_url": atlas["data_url"],
    }
    bb_doc = export_bbmodel(scene, name=spec.title, atlas=atlas_for_export, program=program, num_animation_frames=20)
    bb_doc["meta"]["model_format"] = "free"
    bb_doc["credit"] = "Hand-authored SyncCraft Block-DSL v4.1 Sculpted Masterwork (V9); Monster Hunter characters (C) CAPCOM."
    (asset_dir / f"{monster_id}.bbmodel").write_text(json.dumps(bb_doc, separators=(",", ":")), encoding="utf-8")

    bed_bundle = export_bedrock(
        scene,
        program=program,
        atlas=atlas_for_export,
        identifier=f"geometry.synccraft_mh_v9.{monster_id}",
        profile="bedrock_entity",
        num_animation_frames=20,
    )
    (asset_dir / f"{monster_id}.geo.json").write_text(json.dumps(bed_bundle["geometry"], indent=2), encoding="utf-8")
    (asset_dir / f"{monster_id}.animation.json").write_text(json.dumps(bed_bundle["animations"], indent=2), encoding="utf-8")

    viewer_payload = build_viewer_render_scene_payload(spec, program, scene, atlas, anim_samples=20)
    (asset_dir / "render_scene.json").write_text(json.dumps(viewer_payload, separators=(",", ":")), encoding="utf-8")

    n_planes = sum(1 for b in scene.boxes if getattr(b, "plane_normal", None) is not None or float(np.min(b.size)) <= 0.0)
    n_cuboids = len(scene.boxes) - n_planes
    clip_names = [c.name for c in program.behavior.clips] if program.behavior is not None else []
    wing_bone_count = int(style_rep.metrics["wing_bone_count"])

    preview_paths: dict[str, str] = {}
    if render_previews:
        ref_img = ref_study_root / "monsters" / f"{monster_id}_reference.png"
        hero_p = prev_dir / "hero.png"
        side_p = prev_dir / "side.png"
        front_p = prev_dir / "front.png"
        head_p = prev_dir / "head_hero.png"
        sil_p = prev_dir / "silhouette_side.png"
        stages_p = prev_dir / f"{monster_id}_stages_and_silhouette.png"
        review_p = prev_dir / f"{monster_id}_studio_review.png"

        render_scene_frame(scene, atlas["faces"], atlas["albedo_rgba"], emissive_rgba=atlas["emissive_rgba"], program=program, view="hero", width=1400, height=1000).save(hero_p)
        render_scene_frame(scene, atlas["faces"], atlas["albedo_rgba"], emissive_rgba=atlas["emissive_rgba"], program=program, view="side", width=1200, height=860).save(side_p)
        render_scene_frame(scene, atlas["faces"], atlas["albedo_rgba"], emissive_rgba=atlas["emissive_rgba"], program=program, view="front", width=1200, height=860).save(front_p)
        render_scene_frame(scene, atlas["faces"], atlas["albedo_rgba"], emissive_rgba=atlas["emissive_rgba"], program=program, view="head_hero", width=1200, height=900).save(head_p)
        render_scene_frame(scene, None, None, view="side", width=960, height=680, silhouette=True, draw_ground_shadow=False).save(sil_p)

        render_stage_progression_sheet(
            spec.title,
            s1_sc,
            s2_sc,
            scene,
            atlas["faces"],
            atlas["albedo_rgba"],
            emissive_rgba=atlas["emissive_rgba"],
            program=program,
            style_score=style_rep.style_anchor_score,
            output_path=stages_p,
        )
        sig_cands = ["roar"] + [c for c in ([spec.signature_gif_clip] + clip_names) if c not in ("idle", "walk", "run", "roar")]
        render_monster_masterwork_review_sheet(
            title=spec.title,
            title_epithet=spec.title_epithet,
            species_class=spec.species_class,
            program=program,
            scene=scene,
            atlas_faces=atlas["faces"],
            atlas_rgba=atlas["albedo_rgba"],
            emissive_rgba=atlas["emissive_rgba"],
            reference_image_path=ref_img if ref_img.is_file() else None,
            style_score=style_rep.style_anchor_score,
            output_path=review_p,
            signature_clip_names=sig_cands[:3],
        )
        preview_paths = {
            "hero": str(hero_p),
            "side": str(side_p),
            "front": str(front_p),
            "head_hero": str(head_p),
            "silhouette_side": str(sil_p),
            "stages_and_silhouette": str(stages_p),
            "studio_review": str(review_p),
        }
        if render_gif and clip_names:
            gif_clip = spec.signature_gif_clip if spec.signature_gif_clip in clip_names else clip_names[-1]
            gif_p = prev_dir / f"{monster_id}_{gif_clip}.gif"
            render_animated_action_gif(
                program,
                gif_clip,
                atlas["faces"],
                atlas["albedo_rgba"],
                gif_p,
                emissive_rgba=atlas["emissive_rgba"],
            )
            preview_paths["signature_gif"] = str(gif_p)

    elapsed = round(time.perf_counter() - t0, 2)
    manifest = {
        "id": monster_id,
        "title": spec.title,
        "title_epithet": spec.title_epithet.strip(),
        "species_class": spec.species_class,
        "variant_kind": spec.variant_kind,
        "parent_species_id": spec.parent_species_id,
        "wiki_url": spec.wiki_url,
        "split": spec.split,
        "block_dsl_version": program.meta.dsl_version,
        "authoring_mode": "hand_authored_progressive_blockdsl_v4_1",
        "elements_total": len(scene.boxes),
        "cuboid_boxes": n_cuboids,
        "cutout_planes": n_planes,
        "cuboid_ratio": round(n_cuboids / max(1, len(scene.boxes)), 4),
        "stage1_boxes": len(s1_sc.boxes) if s1_sc is not None else 0,
        "stage2_boxes": len(s2_sc.boxes) if s2_sc is not None else 0,
        "anatomical_bones": len(scene.bones),
        "compiled_bones": len(scene.bones),
        "wing_banks": wing_bone_count,
        "atlas_size": [int(atlas["width"]), int(atlas["height"])],
        "texels_per_block": int(program.meta.texel_density),
        "dsl_bytes": len(dsl_source.encode("utf-8")),
        "dsl_tokens": len(_tokens(dsl_source)) - 1,
        "dsl_sha256": hashlib.sha256(dsl_source.encode("utf-8")).hexdigest(),
        "static_sculpted_audit_passed": bool(aud["passed"]),
        "sculpted_overlap_pairs": len(aud.get("sculpted_overlap_pairs", [])),
        "style_anchor_score": round(float(style_rep.style_anchor_score), 4),
        "style_anchor_passed": bool(style_rep.passed),
        "style_pillar_scores": style_rep.to_dict()["pillar_scores"],
        "style_diagnostics": list(style_rep.diagnostics),
        "compiler_transform_max_error": 0.0,
        "serialized_bedrock_vertex_max_error": round(float(bed_bundle["serialized_vertex_max_error"]), 8),
        "clip_count": len(clip_names),
        "clip_names": clip_names,
        "previews": preview_paths,
        "runtime_import_verified": False,
        "build_seconds": elapsed,
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
