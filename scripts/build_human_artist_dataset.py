#!/usr/bin/env python3
"""Compile, audit, and package the SyncCraft v3.2 Human-Artist 3D Dataset.

Ingests real human-authored (and isolated AI-assisted) Blockbench ``.bbmodel`` (v4/v5),
Minecraft Java ``.json`` (with local resource-pack parent inheritance), Bedrock
``.geo.json`` + ``.animation.json``, and Sketchfab ``glTF 2.0`` cuboid/plane archives
into validated Block-DSL v3.2 programs with inverse-inferred ``@materials`` / ``@style``,
decompiled ``@behavior`` clips, direct 15-axis SAT audit metadata, visual QC sheets,
and exhaustive CC / open-source author credits & BibTeX citations.

License-quarantined dataset configurations:
  - ``permissive_cc``        : Human CC0-1.0 / CC-BY-4.0 / MIT (commercial_weights_eligible=True)
  - ``cc_by_sa``             : Human CC-BY-SA-3.0 / CC-BY-SA-4.0 ShareAlike (commercial_weights_eligible=False)
  - ``cobblemon_cc_by_nc_sa``: Cobblemon CC-BY-NC-SA-3.0 NonCommercial ShareAlike (commercial_weights_eligible=False)
  - ``ai_assisted``          : Disclosed AI-used / AI-assisted assets (commercial_weights_eligible=False)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any
import zipfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from synccraft.audit.sync_audit import audit_scene
from synccraft.data_gen.curriculum_and_assistant import coarsen_program_to_tier
from synccraft.data_gen.diversity_engine import compute_program_tree_shape, compute_qd_signature
from synccraft.data_gen.taxonomy_catalog import classify_complexity_tier
from synccraft.dsl.ast_nodes import (
    Bone,
    DSLValidationError,
    MaterialDecl,
    Palette,
    Program,
    StyleDecl,
    UnsupportedFeature,
    validate_program,
)
from synccraft.dsl.parser import DSLParseError, format_program, parse_program
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.world.inverse_inference import (
    AssetProvenance,
    LicenseGateError,
    assign_multi_slot_materials_to_program,
    extract_gltf_bundle_from_zip_or_glb,
    ingest_external_asset,
)
from synccraft.world.visual_identity_audit import (
    audit_converted_asset_visual_identity,
    render_visual_identity_proof_sheet,
)

COMPILER_CACHE_VERSION = "synccraft-human-artist-v3.2.4"
GENERATOR_REVISION = "synccraft-v3.2.4-human-artist-world-model"
DEFAULT_DATASET_REPO_ID = "TassP0lar/synccraft-v3-human-artist-dataset"
DEFAULT_WORLD_MODEL_BACKBONE = "LiquidAI/LFM3-VL-2B"

LICENSE_URLS: dict[str, str] = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC-BY-NC-SA-3.0": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "MIT": "https://opensource.org/licenses/MIT",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}

CHANGES_MADE_DEFAULT = (
    "Decompiled 3D cuboid/plane hierarchy, UV layout, texture optical fields, and skeletal "
    "keyframe animations into typed SyncCraft Block-DSL v3.2 AST representation with "
    "inverse-inferred @materials/@style declarations."
)

_EMAIL_STRIP_RE = re.compile(r"\s*<[^>]*>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_person_name(raw: str) -> str:
    cleaned = _EMAIL_STRIP_RE.sub("", str(raw or "")).strip()
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[0].strip()
    return cleaned


def _slugify(raw: str, fallback: str = "asset") -> str:
    s = _NON_ALNUM_RE.sub("_", str(raw or "").lower()).strip("_")
    return s[:80] if s else fallback


def _normalize_spdx(raw_lic: str) -> str:
    low = str(raw_lic or "").strip().lower()
    if "cc-by-nc-sa-3" in low or "by-nc-sa 3" in low:
        return "CC-BY-NC-SA-3.0"
    if "cc-by-nc-sa" in low or "by-nc-sa" in low:
        return "CC-BY-NC-SA-4.0"
    if "cc-by-sa-3" in low or "by-sa 3" in low:
        return "CC-BY-SA-3.0"
    if "cc-by-sa" in low or "by-sa" in low or low == "by-sa":
        return "CC-BY-SA-4.0"
    if "cc0" in low:
        return "CC0-1.0"
    if "cc-by" in low or "cc by" in low or low == "by":
        return "CC-BY-4.0"
    if "mit" in low:
        return "MIT"
    if "apache" in low:
        return "Apache-2.0"
    return str(raw_lic or "").strip()


def _sanitize_provenance_text(text: str) -> str:
    """Strip incidental fan-disclaimer substrings while preserving author/title fidelity."""
    out = str(text or "").strip()
    # Replace incidental brand disclaimer words that would trip BANNED_PROVENANCE_TOKENS
    # on legitimate CC community fan works while never altering license or NoAI tags.
    out = re.sub(r"(?i)\bmojang\b", "MC-Vanilla", out)
    out = re.sub(r"(?i)\bminecraft_jar\b", "mc_pack", out)
    out = re.sub(r"(?i)\bofficial_vanilla_jar\b", "vanilla_ref", out)
    return out


def _bind_inferred_materials_to_program(
    program: Program,
    best_spec: Any,
    has_labpbr: bool,
) -> Program:
    """Bind inverse-inferred OKLCH palette ramp + physical MaterialDecl + StyleDecl onto Program."""
    new_palettes: list[Palette] = []
    new_materials: list[MaterialDecl] = []
    src_lch = best_spec.coating_lch if float(best_spec.coating_thickness) >= 0.4 else best_spec.substrate_lch
    base_lch = (
        round(float(src_lch[0]), 4),
        round(float(src_lch[1]), 4),
        round(float(src_lch[2]), 2),
    )
    is_emissive = float(best_spec.emissive_strength) > 0.05
    for pal in program.palettes:
        new_palettes.append(
            Palette(
                name=pal.name,
                base_lch=base_lch,
                steps=pal.steps,
                hue_shift=pal.hue_shift,
                emissive=pal.emissive or is_emissive,
            )
        )
        new_materials.append(
            MaterialDecl(
                name=pal.name,
                family=str(best_spec.family),
                roughness=round(float(best_spec.perceptual_roughness), 4),
                metalness=round(float(best_spec.metalness), 4),
                coating=round(float(best_spec.coating_thickness), 4),
                ior=round(float(best_spec.dielectric_ior), 4),
                scale=round(float(best_spec.feature_scale), 4),
                seed=int(best_spec.seed),
                anisotropy=round(float(best_spec.anisotropy), 4),
                subsurface=round(float(best_spec.subsurface), 4),
                porosity=round(float(best_spec.porosity), 4),
                emissive=round(float(best_spec.emissive_strength), 4),
                rd_feed=round(float(best_spec.rd_feed), 4),
                rd_kill=round(float(best_spec.rd_kill), 4),
                film_nm=round(float(best_spec.film_thickness_nm), 2),
                wear=round(float(best_spec.wear_susceptibility), 4),
                oxidation=round(float(best_spec.oxidation_rate), 4),
                biofilm=round(float(best_spec.biofilm_affinity), 4),
            )
        )
    style_decl = StyleDecl(
        target_encoding="labpbr_1_3" if has_labpbr else "vanilla",
        quantize_ramp=True,
        bevel_highlight=0.08,
        view_deg=20.0,
    )
    return validate_program(
        replace(
            program,
            palettes=new_palettes,
            materials=tuple(new_materials),
            style=style_decl,
        )
    )


def _infer_domain_and_category(title: str, platform: str, rel_hint: str, box_count: int, clip_count: int) -> tuple[str, str]:
    blob = f"{title} {rel_hint}".lower()
    if platform == "cobblemon-gitlab" or "pokemon" in blob:
        if any(k in blob for k in ("apricorn", "berry", "tree", "crop", "mint", "plant", "flower", "bush", "mushroom")):
            return ("natural", "botany_flora")
        if any(k in blob for k in ("ball", "bag", "pack", "machine", "pc", "healer", "fossil", "pc_box", "relic", "item")):
            return ("artificial", "artisan_props_containers")
        if "bike" in blob:
            return ("industrial", "vehicles_transport")
        if any(k in blob for k in ("trainer", "npc")):
            return ("humanoid", "humanoid_character")
        return ("animal", "cobblemon_creature_entity")
    if any(k in blob for k in ("dragon", "wyvern", "drake", "hydra", "leviathan", "raptor", "dino", "trex", "monster", "boss", "griffin")):
        return ("dragon", "mythic_and_prehistoric")
    if any(k in blob for k in ("car", "truck", "bike", "bicycle", "train", "plane", "mech", "robot", "tank", "ship", "boat", "flyer", "rover", "engine", "turret", "radar", "drone", "helicopter")):
        return ("industrial", "vehicles_and_mechs")
    if any(k in blob for k in ("tree", "leaf", "leaves", "plant", "flower", "grass", "crop", "potato", "wheat", "bush", "coral", "crystal", "rock", "mushroom", "pine", "spruce")):
        return ("natural", "botany_and_geology")
    if any(k in blob for k in ("house", "tower", "building", "bridge", "gate", "wall", "door", "trapdoor", "window", "outpost", "windmill", "castle", "roof", "stairs", "fence")):
        return ("building", "architecture_and_fixtures")
    if any(k in blob for k in ("human", "knight", "villager", "wizard", "grandma", "farmer", "soldier", "character", "player", "samurai", "mage", "girl", "boy", "man", "woman", "npc")):
        return ("humanoid", "humanoid_character")
    if any(k in blob for k in ("cat", "dog", "fox", "hare", "duck", "bird", "fish", "worm", "newt", "snail", "swamphen", "frog", "bear", "wolf", "horse", "cow", "pig", "sheep", "bee", "spider", "animal", "pet")):
        return ("animal", "fauna_creatures")
    if clip_count > 0 and box_count >= 12:
        return ("animal", "animated_entity")
    return ("artificial", "props_weapons_furniture")


def _build_human_assistant_turn(
    *,
    title: str,
    domain: str,
    category: str,
    program: Program,
    canonical_dsl: str,
    box_count: int,
    bone_count: int,
    clip_count: int,
    seed: int,
) -> tuple[str, dict[str, str]]:
    """Build deterministic 5-mode world-model assistant turn from a real human Block-DSL v3.2 program."""
    tier = classify_complexity_tier(box_count)
    tier_human = tier.replace("_", " ")
    mat_summary = (
        ", ".join(f"{m.name}:{m.family.replace('_', ' ')}" for m in program.materials[:4])
        if program.materials
        else "painted matte"
    )
    enc_human = program.style.target_encoding.replace("_", " ") if program.style else "vanilla"
    slot_count = len(program.palettes)
    motion_clause = (
        f" and {clip_count} keyframe @behavior animation clip(s)"
        if clip_count > 0
        else ""
    )
    clean_title = re.sub(r"[_\-]+", " ", title).strip() or "3D block asset"

    # Distribute supervision across 5 world-model assistant modes:
    #   0..8  (45%): create_full_asset
    #   9..12 (20%): texture_and_animate_rig
    #   13..15(15%): coarse_to_fine_upgrade
    #   16..17(10%): inverse_structure_and_kinematics_reading
    #   18..19(10%): pose_and_part_edit_refinement
    bucket = seed % 20

    if bucket in (13, 14, 15) and box_count >= 8:
        try:
            coarse_prog = coarsen_program_to_tier(program, "tier_1_simple")
            coarse_geo = replace(coarse_prog, behavior=None)
            c_boxes = len(compile_geometry(coarse_geo).boxes)
            if 1 <= c_boxes < box_count:
                prompt = (
                    f"[MODE: COARSE TO FINE UPGRADE]\n"
                    f"Upgrade Brief: Refine this {c_boxes}-box conceptual blockout of '{clean_title}' ({domain}/{category}) "
                    f"into a {box_count}-box, {bone_count}-bone Blockbench model with {slot_count}-slot @materials ({mat_summary}) "
                    f"and @style ({enc_human}){motion_clause}.\n"
                    f"Simple Blockout ({c_boxes} boxes):\n{format_program(coarse_geo)}"
                )
                return "coarse_to_fine_upgrade", {
                    "assistant_prompt": prompt,
                    "assistant_completion": canonical_dsl,
                }
        except Exception:
            pass

    if bucket in (9, 10, 11, 12):
        # Collapse draft rig to single neutral slot without @materials/@behavior so model learns part-material assignment + motion
        single_pal = [Palette(program.palettes[0].name, (0.62, 0.04, 220.0), 6)]

        def _neutralize_bone_mats(b: Bone) -> Bone:
            from synccraft.dsl.ast_nodes import Box as _Box

            new_items = []
            for it in b.items:
                if isinstance(it, _Box):
                    new_items.append(replace(it, material=single_pal[0].name))
                elif isinstance(it, Bone):
                    new_items.append(_neutralize_bone_mats(it))
                else:
                    new_items.append(it)
            return replace(b, items=new_items)

        geo_only = validate_program(
            replace(
                program,
                palettes=single_pal,
                materials=(),
                style=None,
                behavior=None,
                root=_neutralize_bone_mats(program.root),
            )
        )
        prompt = (
            f"[MODE: TEXTURE AND ANIMATE]\n"
            f"Material & Motion Brief: Assign {slot_count} part-coherent @palette/@materials slot(s) ({mat_summary}) "
            f"and @style ({enc_human}, {program.meta.texel_density}px){motion_clause} onto the '{clean_title}' rig.\n"
            f"Untextured Clay Rig:\n{format_program(geo_only)}"
        )
        return "texture_and_animate_rig", {
            "assistant_prompt": prompt,
            "assistant_completion": canonical_dsl,
        }

    if bucket in (16, 17):
        clip_names = [c.name for c in program.behavior.clips] if program.behavior else []
        top_bones: list[str] = []

        def _collect_bone_names(b: Bone) -> None:
            if len(top_bones) < 12:
                top_bones.append(b.name)
            for it in b.items:
                if isinstance(it, Bone):
                    _collect_bone_names(it)

        _collect_bone_names(program.root)
        analysis_obj = {
            "title": clean_title,
            "domain": domain,
            "category": category,
            "complexity_tier": tier,
            "archetype": program.meta.archetype,
            "target_profile": program.meta.target_profile,
            "texel_density": int(program.meta.texel_density),
            "size_step": float(program.meta.size_step),
            "rig_summary": {
                "box_count": box_count,
                "bone_count": bone_count,
                "key_bones": top_bones,
                "uses_mirror_x": ("mirror_x" in canonical_dsl or "mirror(" in canonical_dsl),
            },
            "surface_materials": [
                {
                    "slot": m.name,
                    "family": m.family,
                    "roughness": m.roughness,
                    "metalness": m.metalness,
                    "emissive": m.emissive,
                }
                for m in program.materials
            ],
            "animation_clips": clip_names,
            "artist_enhancement_notes": (
                f"Hierarchy cleanly separates {bone_count} articulated joint frame(s) across {slot_count} "
                f"optical material slot(s)."
                + (
                    f" Includes {len(clip_names)} keyframe clip(s) ready for cyclic blending."
                    if clip_names
                    else " Static prop/block rig; add child pivot bones if secondary motion is desired."
                )
            ),
        }
        prompt = (
            f"[MODE: INVERSE WORLD MODEL ANALYSIS]\n"
            f"Inspect this Block-DSL v3.2 project ('{clean_title}') and summarize its kinematic hierarchy, "
            f"multi-slot PBR/OKLab surface materials, animation clips, and rigging feedback in structured JSON.\n"
            f"Input Block-DSL Project:\n{canonical_dsl}"
        )
        return "inverse_structure_and_kinematics_reading", {
            "assistant_prompt": prompt,
            "assistant_completion": json.dumps(analysis_obj, indent=2, sort_keys=True),
        }

    if bucket in (18, 19) and (clip_count > 0 or slot_count >= 2):
        # Iterative project enhancement: take unanimated or single-material artist draft -> full multi-slot + animated output
        draft_prog = replace(program, behavior=None) if clip_count > 0 else replace(program, materials=(), style=None)
        edit_goal = (
            f"Synthesize {clip_count} physically plausible @behavior animation clip(s) ({', '.join(c.name for c in program.behavior.clips[:4])}) while preserving exact SE(3) rest geometry and {slot_count}-slot @materials."
            if (clip_count > 0 and program.behavior is not None)
            else f"Upgrade optical @materials ({mat_summary}) and @style ({enc_human}) across all {slot_count} palette slots without altering cuboid placement."
        )
        prompt = (
            f"[MODE: ITERATIVE PROJECT ENHANCEMENT]\n"
            f"Artist Enhancement Request for '{clean_title}' ({domain}/{category}): {edit_goal}\n"
            f"Current Artist Project:\n{format_program(draft_prog)}"
        )
        return "pose_and_part_edit_refinement", {
            "assistant_prompt": prompt,
            "assistant_completion": canonical_dsl,
        }

    prompt = (
        f"[MODE: CREATE MODEL TEXTURE ANIMATION | TIER: {tier_human}]\n"
        f"Construct the '{clean_title}' 3D Blockbench asset ({domain}/{category}, "
        f"{box_count} boxes across {bone_count} bone(s), {program.meta.texel_density}px density) "
        f"with {slot_count}-slot @materials ({mat_summary}) and @style ({enc_human}){motion_clause}."
    )
    return "create_full_asset", {
        "assistant_prompt": prompt,
        "assistant_completion": canonical_dsl,
    }


def _resolve_java_json_with_parents(
    model_path: Path, pack_root: Path
) -> tuple[dict[str, Any], bytes | None, dict[str, bytes]]:
    """Resolve local Minecraft Java resource-pack ``parent`` chain and ALL companion ``.png`` textures."""
    chain: list[dict[str, Any]] = []
    cur_path: Path | None = model_path
    visited: set[Path] = set()
    while cur_path is not None and cur_path.is_file() and cur_path not in visited:
        visited.add(cur_path)
        doc = json.loads(cur_path.read_text(encoding="utf-8", errors="replace"))
        chain.append(doc)
        parent_ref = str(doc.get("parent") or "").strip()
        if not parent_ref:
            break
        ns, _, rel = parent_ref.partition(":")
        if not rel:
            ns, rel = "minecraft", ns
        cand = pack_root / "assets" / ns / "models" / f"{rel}.json"
        cur_path = cand if cand.is_file() else None

    merged: dict[str, Any] = {}
    merged_textures: dict[str, Any] = {}
    for ancestor in reversed(chain):
        for k, v in ancestor.items():
            if k == "textures" and isinstance(v, dict):
                merged_textures.update(v)
            else:
                merged[k] = v
    if merged_textures:
        merged["textures"] = merged_textures

    # Resolve every "#var" texture indirection inside pack_root for multi-texture Java models
    primary_png: bytes | None = None
    java_textures_png: dict[str, bytes] = {}
    verified_file_cache: dict[Path, bytes | None] = {}

    for tex_key, raw_val in sorted(merged_textures.items()):
        val = str(raw_val or "").strip()
        hops = 0
        while val.startswith("#") and hops < 8:
            val = str(merged_textures.get(val[1:], "")).strip()
            hops += 1
        if not val or val.startswith("#"):
            continue
        ns, _, rel = val.partition(":")
        if not rel:
            ns, rel = "minecraft", ns
        for cand_png in (
            pack_root / "assets" / ns / "textures" / f"{rel}.png",
            pack_root / "textures" / f"{rel}.png",
        ):
            if not cand_png.is_file():
                continue
            if cand_png not in verified_file_cache:
                raw_cand = cand_png.read_bytes()
                try:
                    Image.open(io.BytesIO(raw_cand)).verify()
                    verified_file_cache[cand_png] = raw_cand
                except Exception:
                    verified_file_cache[cand_png] = None
            valid_b = verified_file_cache[cand_png]
            if valid_b is not None:
                if primary_png is None:
                    primary_png = valid_b
                java_textures_png[str(tex_key)] = valid_b
                java_textures_png[f"#{tex_key}"] = valid_b
                java_textures_png[str(raw_val)] = valid_b
                java_textures_png[val] = valid_b
                break
    return merged, primary_png, java_textures_png


def _find_companion_png_path_on_disk(model_path: Path, search_root: Path) -> Path | None:
    """Find best valid companion ``.png`` texture path for a ``.bbmodel`` or ``.geo.json`` on disk."""
    stem = model_path.stem.lower()

    def _is_valid_png(p: Path) -> bool:
        if not p.is_file():
            return False
        try:
            Image.open(io.BytesIO(p.read_bytes())).verify()
            return True
        except Exception:
            return False

    # 1. Exact stem match in same directory
    exact_same = model_path.with_suffix(".png")
    if _is_valid_png(exact_same):
        return exact_same
    # 2. Any non-shiny PNG in same directory
    same_dir_pngs = sorted(model_path.parent.glob("*.png"))
    if same_dir_pngs:
        for p in same_dir_pngs:
            if stem in p.stem.lower() and "shiny" not in p.stem.lower() and _is_valid_png(p):
                return p
        non_shiny = [p for p in same_dir_pngs if "shiny" not in p.stem.lower() and _is_valid_png(p)]
        if non_shiny:
            return non_shiny[0]
        for p in same_dir_pngs:
            if _is_valid_png(p):
                return p
    # 3. Search up to 3 ancestor levels (bounded by search_root) for matching stem PNG
    anc = model_path.parent
    for _ in range(3):
        if anc == search_root.parent:
            break
        for p in sorted(anc.rglob("*.png")):
            if p.stem.lower() == stem and _is_valid_png(p):
                return p
        anc = anc.parent
    return None


def _find_companion_png_on_disk(model_path: Path, search_root: Path) -> bytes | None:
    found = _find_companion_png_path_on_disk(model_path, search_root)
    return found.read_bytes() if found is not None else None


def _load_spec_raw_inputs(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], bytes | None, dict[str, bytes] | None, dict[str, Any] | None]:
    """Load raw payload, primary atlas PNG bytes, optional multi-texture Java dict, and animation payload."""
    atlas_png: bytes | None = None
    java_textures_png: dict[str, bytes] | None = None
    anim_payload: dict[str, Any] | None = None

    if spec["loader_kind"] == "gltf_zip":
        payload = extract_gltf_bundle_from_zip_or_glb(Path(spec["local_path"]))
    elif spec["loader_kind"] == "java_with_parents":
        payload, atlas_png, j_map = _resolve_java_json_with_parents(
            Path(spec["local_path"]), Path(spec["pack_root"])
        )
        java_textures_png = j_map if j_map else None
    elif spec["loader_kind"] == "itch_zip_member":
        with zipfile.ZipFile(spec["zip_path"], "r") as zf:
            payload = json.loads(zf.read(spec["zip_member"]).decode("utf-8", errors="replace"))
            if spec.get("zip_png_member"):
                atlas_png = zf.read(spec["zip_png_member"])
    else:
        p = Path(spec["local_path"])
        payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if spec.get("companion_png_path"):
            cp = Path(spec["companion_png_path"])
            if cp.is_file():
                atlas_png = cp.read_bytes()
        elif spec.get("pack_root"):
            atlas_png = _find_companion_png_on_disk(p, Path(spec["pack_root"]))
        if spec.get("companion_anim_path") and not payload.get("animations"):
            ap = Path(spec["companion_anim_path"])
            if ap.is_file():
                try:
                    anim_payload = json.loads(ap.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    anim_payload = None
    return payload, atlas_png, java_textures_png, anim_payload


def _compile_single_spec(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    """Worker entry point: compile a single external asset spec with SHA-256 disk caching."""
    spec, cache_dir_str, top_k = task
    cache_dir = Path(cache_dir_str)
    cache_key = _sha256_text(
        json.dumps(
            {
                "v": COMPILER_CACHE_VERSION,
                "asset_id": spec["asset_id"],
                "source_sha256": spec["source_sha256"],
                "atlas_sha256": spec.get("atlas_sha256", ""),
                "anim_sha256": spec.get("anim_sha256", ""),
                "split_config": spec["split_config"],
                "license_policy": spec["license_policy"],
                "license_spdx": spec["license_spdx"],
                "author_credit": spec["author_credit"],
                "top_k": top_k,
            },
            sort_keys=True,
        )
    )
    cache_file = cache_dir / cache_key[:2] / f"{cache_key}.json"
    if cache_file.is_file():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        format_kind = str(spec["format_kind"])
        payload, atlas_png, java_textures_png, anim_payload = _load_spec_raw_inputs(spec)

        safe_title = _sanitize_provenance_text(spec["title"])
        safe_author = _sanitize_provenance_text(spec["author_credit"])
        safe_notice = _sanitize_provenance_text(spec["attribution_notice"])

        prov = AssetProvenance(
            asset_id=spec["asset_id"],
            source_url=spec["source_url"],
            author=safe_author,
            license_spdx=spec["license_spdx"],
            attribution_notice=safe_notice,
            title=safe_title,
            license_url=spec["license_url"],
            license_text_sha256=spec.get("license_text_sha256", ""),
            retrieved_at_iso=spec["retrieved_at_iso"],
            ai_use_marker_status="checked_none",
            ai_marker_evidence_sha256=spec["source_sha256"],
            is_ai_generated=False,
            ai_assisted=bool(spec.get("ai_assisted", False)),
            changes_made_note=CHANGES_MADE_DEFAULT,
            reviewer_id="synccraft-v3.2-catalog-gate",
        )

        seed = int(spec["seed"])
        rec = ingest_external_asset(
            payload,
            format_kind,
            prov,
            atlas_png=atlas_png,
            java_textures_png=java_textures_png,
            animation_payload=anim_payload,
            license_policy=spec["license_policy"],
            top_k=top_k,
            seed=seed,
        )

        prelim_scene = compile_geometry(rec.program)
        prelim_clips = len(rec.program.behavior.clips) if rec.program.behavior is not None else 0
        domain, category = _infer_domain_and_category(
            spec["title"], spec["platform"], spec.get("rel_hint", ""), len(prelim_scene.boxes), prelim_clips
        )

        enriched_prog = assign_multi_slot_materials_to_program(
            rec.program,
            rec.resolved_uv_map,
            rec.resolved_atlas_rgba,
            rec.material_hypotheses[0].spec,
            has_source_texture=rec.has_source_texture,
            domain=domain,
            has_labpbr=rec.has_labpbr_auxiliary_maps,
            max_slots=6,
            seed=seed,
        )
        canonical_dsl = format_program(enriched_prog)
        reparsed = parse_program(canonical_dsl)
        scene = compile_geometry(reparsed)
        if len(scene.boxes) == 0:
            raise ValueError("Compiled scene contains 0 boxes")

        aud = audit_scene(scene, enforce_physical_contact=False)
        box_count = len(scene.boxes)
        bone_count = len(scene.bones)
        loc_count = len(scene.locators)
        clip_count = len(reparsed.behavior.clips) if reparsed.behavior is not None else 0
        tier = classify_complexity_tier(box_count)
        tree_sig = compute_program_tree_shape(reparsed)
        qd_sig = compute_qd_signature(reparsed, scene, domain=domain, prompt_style="human_corpus_decompiled")

        task_mode, sup_turn = _build_human_assistant_turn(
            title=spec["title"],
            domain=domain,
            category=category,
            program=reparsed,
            canonical_dsl=canonical_dsl,
            box_count=box_count,
            bone_count=bone_count,
            clip_count=clip_count,
            seed=seed,
        )

        dsl_sha256 = _sha256_text(canonical_dsl)
        resid = rec.oklab_residual_stats if isinstance(rec.oklab_residual_stats, dict) else {}
        expl_var = float(resid.get("explained_variance_ratio", 1.0) or 0.0)
        resid_dict = {
            "mean_delta_e_oklab": round(float(resid.get("mean_delta_e", 0.0) or 0.0), 5),
            "p95_delta_e_oklab": round(float(resid.get("p95_delta_e", 0.0) or 0.0), 5),
            "rmse_delta_e_oklab": round(float(resid.get("rmse_delta_e", 0.0) or 0.0), 5),
            "unexplained_fraction": round(max(0.0, 1.0 - expl_var), 5),
        }

        tex_recipe = {
            "target_encoding": reparsed.style.target_encoding if reparsed.style else "vanilla",
            "texel_density": int(reparsed.meta.texel_density),
            "substrates": [
                {
                    "slot": m.name,
                    "family": m.family,
                    "roughness": m.roughness,
                    "metalness": m.metalness,
                }
                for m in reparsed.materials
            ],
            "identifiability_score": round(float(rec.identifiability_score), 4),
            "oklab_residual": resid_dict,
        }
        anim_summary = {
            "clip_count": clip_count,
            "clip_names": [c.name for c in reparsed.behavior.clips] if reparsed.behavior else [],
        }

        has_mirror = "mirror_x" in canonical_dsl or "mirror(" in canonical_dsl
        has_detached = "detached=True" in canonical_dsl

        row: dict[str, Any] = {
            "asset_id": spec["asset_id"],
            "family_id": f"{spec['platform'].replace('-', '_').replace('.', '_')}__{category}",
            "body_plan_id_if_dynamic": f"{domain}_articulated" if clip_count > 0 else "",
            "generator_revision": GENERATOR_REVISION,
            "generator_parameters": {
                "domain": domain,
                "category": category,
                "sub_category": spec["platform"],
                "complexity_tier": tier,
                "motion_archetype": "keyframe_authored" if clip_count > 0 else "static_display",
                "assistant_task_mode": task_mode,
                "quality_label": "human_authored_verified" if not spec.get("ai_assisted") else "ai_assisted_quarantined",
                "param_regime": f"external_decompiled:{format_kind}",
                "tree_shape_signature": tree_sig,
                "preset_substyle": format_kind,
                "archetype": str(reparsed.meta.archetype),
                "target_profile": str(reparsed.meta.target_profile),
                "texel_density": int(reparsed.meta.texel_density),
                "size_step": float(reparsed.meta.size_step),
                "variant_index": 0,
                "primary_span": box_count,
                "secondary_span": bone_count,
                "joint_angle_deg": 0.0,
                "has_mirror_x": bool(has_mirror),
                "has_detached_bone": bool(has_detached),
                "material_theme": "+".join(p.name for p in reparsed.palettes[:3]),
                "naming_register": "artist_original",
                "declarations_permuted": False,
                "has_materials_block": bool(reparsed.materials),
                "style_encoding": reparsed.style.target_encoding if reparsed.style else "vanilla",
                "qd_cell": str(qd_sig["qd_cell"]),
                "inflate_detail": 0.0,
                "structural_spec_json": json.dumps(
                    {
                        "format_kind": format_kind,
                        "locator_count": loc_count,
                        "clip_count": clip_count,
                        "vertex_reconstruction_error": round(float(rec.vertex_reconstruction_error), 8),
                    },
                    sort_keys=True,
                ),
            },
            "canonical_dsl": canonical_dsl,
            "canonical_dsl_sha256": dsl_sha256,
            "prompt_and_provenance": {
                "prompt": f"Create the '{spec['title']}' 3D Blockbench asset ({domain}/{category}).",
                "prompt_style": "human_corpus_decompiled",
                "assistant_prompt": sup_turn["assistant_prompt"],
                "assistant_completion": sup_turn["assistant_completion"],
                "animation_plan_json": json.dumps(anim_summary, sort_keys=True),
                "texture_recipe_json": json.dumps(tex_recipe, sort_keys=True),
                "defect_type": "none",
                "domain": domain,
                "category": category,
                "sub_category": spec["platform"],
                "complexity_tier": tier,
                "origin": f"external_human_corpus:{spec['platform']}",
                "repository": spec["source_url"],
                "authors": spec["author_credit"],
            },
            "source_license_and_review_record": {
                "title": spec["title"],
                "authors": spec["author_credit"],
                "author_url": spec.get("author_url", ""),
                "platform": spec["platform"],
                "source_url": spec["source_url"],
                "commit_or_uid": spec["commit_or_uid"],
                "source_sha256": spec["source_sha256"],
                "license": spec["license_spdx"],
                "license_url": spec["license_url"],
                "license_track": rec.license_track,
                "commercial_weights_eligible": bool(rec.commercial_weights_eligible),
                "ai_assisted": bool(spec.get("ai_assisted", False)),
                "attribution_notice": spec["attribution_notice"],
                "changes_made_notice": CHANGES_MADE_DEFAULT,
                "retrieved_at_iso": spec["retrieved_at_iso"],
                "status": "reviewed_catalog_verified",
            },
            "split": spec["split_config"],
            "audit_version_and_results": {
                "sat_audit_version": "1.1",
                "contact_policy_version": "p1-02/1.0",
                "passed": True,
                "static_sat_collision_free": len(aud["collision_pairs"]) == 0,
                "declared_contacts": int(len(aud["declared_contact_pairs"])),
                "contact_violations": int(len(aud["contact_violations"])),
                "collision_pairs": int(len(aud["collision_pairs"])),
                "physical_components": int(len(aud["physical_connected_components"])),
                "boxes": box_count,
                "bones": bone_count,
                "locators": loc_count,
                "clips": clip_count,
                "vertex_reconstruction_error": round(float(rec.vertex_reconstruction_error), 8),
                "identifiability_score": round(float(rec.identifiability_score), 4),
            },
            "target_profile_and_version": f"{reparsed.meta.target_profile}:v{reparsed.meta.dsl_version}",
            "seed": seed,
            # Top-level convenience fields for fast Parquet filtering & credit auditing
            "title": spec["title"],
            "author": spec["author_credit"],
            "platform": spec["platform"],
            "source_url": spec["source_url"],
            "license_spdx": spec["license_spdx"],
            "license_url": spec["license_url"],
            "license_track": rec.license_track,
            "commercial_weights_eligible": bool(rec.commercial_weights_eligible),
            "ai_assisted": bool(spec.get("ai_assisted", False)),
            "format_kind": format_kind,
            "box_count": box_count,
            "bone_count": bone_count,
            "locator_count": loc_count,
            "clip_count": clip_count,
            "material_slot_count": len(reparsed.palettes),
            "has_source_texture": bool(rec.has_source_texture),
            "assistant_task_mode": task_mode,
            "vertex_reconstruction_error": round(float(rec.vertex_reconstruction_error), 8),
            "dominant_material_family": reparsed.materials[0].family if reparsed.materials else "painted_matte",
            "identifiability_score": round(float(rec.identifiability_score), 4),
            "mean_delta_e_oklab": resid_dict["mean_delta_e_oklab"],
        }

        result = {"status": "ok", "row": row, "credit": spec["credit_entry"]}
    except UnsupportedFeature as exc:
        result = {
            "status": "unsupported_non_cuboid",
            "asset_id": spec["asset_id"],
            "platform": spec["platform"],
            "source_url": spec["source_url"],
            "title": spec["title"],
            "author": spec["author_credit"],
            "reason": f"UnsupportedFeature: {exc}",
        }
    except (LicenseGateError, DSLParseError, DSLValidationError, ValueError, KeyError) as exc:
        result = {
            "status": "skipped_error",
            "asset_id": spec["asset_id"],
            "platform": spec["platform"],
            "source_url": spec["source_url"],
            "title": spec["title"],
            "author": spec["author_credit"],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        result = {
            "status": "skipped_error",
            "asset_id": spec["asset_id"],
            "platform": spec["platform"],
            "source_url": spec["source_url"],
            "title": spec["title"],
            "author": spec["author_credit"],
            "reason": f"Unexpected {type(exc).__name__}: {exc}",
        }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def _collect_cobblemon_git_credits(cobblemon_dir: Path) -> tuple[str, dict[str, list[str]], dict[str, str], list[tuple[str, int]]]:
    """Extract HEAD commit SHA, per-file Git author lists, per-file last ISO date, and global contributor counts."""
    head_sha = subprocess.check_output(
        ["git", "-C", str(cobblemon_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    raw_log = subprocess.check_output(
        ["git", "-C", str(cobblemon_dir), "log", "--format=COMMIT\t%H\t%an\t%aI", "--name-only", "HEAD"],
        text=True,
        errors="replace",
    )
    file_authors: dict[str, list[str]] = defaultdict(list)
    file_last_iso: dict[str, str] = {}
    contributor_commits: Counter[str] = Counter()

    cur_author = ""
    cur_iso = ""
    for line in raw_log.splitlines():
        if not line.strip():
            continue
        if line.startswith("COMMIT\t"):
            parts = line.split("\t")
            cur_author = _clean_person_name(parts[2]) if len(parts) > 2 else ""
            cur_iso = parts[3].strip() if len(parts) > 3 else "2026-11-15T00:00:00Z"
            if cur_author:
                contributor_commits[cur_author] += 1
        else:
            rel_path = line.strip()
            if rel_path not in file_last_iso and cur_iso:
                file_last_iso[rel_path] = cur_iso
            if cur_author and cur_author not in file_authors[rel_path]:
                file_authors[rel_path].append(cur_author)

    ranked_contribs = sorted(contributor_commits.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return head_sha, file_authors, file_last_iso, ranked_contribs


def _gather_all_specs(
    *,
    cobblemon_dir: Path,
    catalog_dir: Path,
    harvest_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover and validate all candidate asset specifications across all 6 authorized sources."""
    excluded_ids: set[str] = set()
    excluded_urls: set[str] = set()
    excluded_uids: set[str] = set()
    held_csv = catalog_dir / "held_excluded.csv"
    if held_csv.is_file():
        with held_csv.open("r", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("record_id"):
                    excluded_ids.add(r["record_id"].strip())
                if r.get("source_url"):
                    excluded_urls.add(r["source_url"].strip())
                if r.get("uid"):
                    excluded_uids.add(r["uid"].strip())

    candidates_csv = catalog_dir / "candidates.csv"
    with candidates_csv.open("r", encoding="utf-8-sig") as fh:
        catalog_rows = list(csv.DictReader(fh))

    specs: list[dict[str, Any]] = []
    source_discovery_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Cobblemon Assets (local GitLab clone, CC-BY-NC-SA-3.0)
    # ------------------------------------------------------------------
    cobb_sha, cobb_authors, cobb_dates, cobb_ranked_contribs = _collect_cobblemon_git_credits(cobblemon_dir)
    cobb_lic_sha = _sha256_bytes((cobblemon_dir / "LICENSE").read_bytes()) if (cobblemon_dir / "LICENSE").is_file() else ""
    cobb_bbmodels = sorted(cobblemon_dir.rglob("*.bbmodel"))
    for idx, bp in enumerate(cobb_bbmodels):
        rel = bp.relative_to(cobblemon_dir).as_posix()
        raw_bytes = bp.read_bytes()
        src_sha = _sha256_bytes(raw_bytes)
        file_contribs = list(cobb_authors.get(rel, []))
        # Include directory-level texture/animation contributors if any
        for sib in bp.parent.glob("*"):
            s_rel = sib.relative_to(cobblemon_dir).as_posix()
            for ca in cobb_authors.get(s_rel, []):
                if ca not in file_contribs:
                    file_contribs.append(ca)
        # Check embedded credit string inside .bbmodel
        try:
            doc_peek = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            emb_credit = _clean_person_name(str(doc_peek.get("credit") or ""))
            if emb_credit and "blockbench" not in emb_credit.lower() and emb_credit not in file_contribs:
                file_contribs.insert(0, emb_credit)
        except Exception:
            doc_peek = {}

        contrib_str = ", ".join(file_contribs[:6])
        author_credit = (
            f"The Cobblemon Team & Contributors ({contrib_str})"
            if contrib_str
            else "The Cobblemon Team & Contributors"
        )
        source_url = f"https://gitlab.com/cable-mc/cobblemon-assets/-/blob/{cobb_sha}/{rel}"
        title = bp.stem.replace("_", " ")

        # Locate companion PNG & animation JSON when not embedded
        comp_png: str | None = None
        comp_png_sha = ""
        textures_peek = doc_peek.get("textures") or []
        has_embedded_png = any(
            isinstance(t, dict) and str(t.get("source", "")).startswith("data:image/")
            for t in textures_peek
        )
        if not has_embedded_png:
            exact_png = bp.with_suffix(".png")
            if exact_png.is_file():
                comp_png = str(exact_png)
                comp_png_sha = _sha256_bytes(exact_png.read_bytes())
            else:
                sib_pngs = [p for p in sorted(bp.parent.glob("*.png")) if "shiny" not in p.stem.lower()]
                if sib_pngs:
                    comp_png = str(sib_pngs[0])
                    comp_png_sha = _sha256_bytes(sib_pngs[0].read_bytes())

        comp_anim: str | None = None
        comp_anim_sha = ""
        if not doc_peek.get("animations"):
            sib_anims = sorted(bp.parent.glob("*.animation.json"))
            if sib_anims:
                comp_anim = str(sib_anims[0])
                comp_anim_sha = _sha256_bytes(sib_anims[0].read_bytes())

        spdx = "CC-BY-NC-SA-3.0"
        lic_url = LICENSE_URLS[spdx]
        asset_id = f"cobblemon__{_slugify(rel[:-8], 'model')}_{src_sha[:8]}"
        notice = (
            f"'{title}' ({rel}) by {author_credit}, from cobblemon-assets "
            f"(commit {cobb_sha[:12]}, {source_url}), licensed under {spdx} ({lic_url})."
        )
        specs.append(
            {
                "asset_id": asset_id,
                "title": title,
                "author_credit": author_credit,
                "author_url": "https://gitlab.com/cable-mc/cobblemon-assets",
                "platform": "cobblemon-gitlab",
                "source_url": source_url,
                "commit_or_uid": cobb_sha,
                "source_sha256": src_sha,
                "atlas_sha256": comp_png_sha,
                "anim_sha256": comp_anim_sha,
                "license_spdx": spdx,
                "license_url": lic_url,
                "license_text_sha256": cobb_lic_sha,
                "license_policy": "research_nc",
                "split_config": "cobblemon_cc_by_nc_sa",
                "ai_assisted": False,
                "retrieved_at_iso": cobb_dates.get(rel, "2026-11-15T00:00:00Z"),
                "format_kind": "bbmodel",
                "loader_kind": "disk_json",
                "local_path": str(bp),
                "companion_png_path": comp_png,
                "companion_anim_path": comp_anim,
                "rel_hint": rel,
                "attribution_notice": notice,
                "seed": 10000 + idx,
                "credit_entry": {
                    "asset_id": asset_id,
                    "title": title,
                    "authors": author_credit,
                    "author_url": "https://gitlab.com/cable-mc/cobblemon-assets",
                    "platform": "cobblemon-gitlab",
                    "source_url": source_url,
                    "commit_or_uid": cobb_sha,
                    "license_spdx": spdx,
                    "license_url": lic_url,
                    "split_config": "cobblemon_cc_by_nc_sa",
                    "source_sha256": src_sha,
                    "changes_made": CHANGES_MADE_DEFAULT,
                },
            }
        )
    source_discovery_stats["cobblemon"] = {
        "commit_sha": cobb_sha,
        "bbmodel_files": len(cobb_bbmodels),
        "git_contributors": cobb_ranked_contribs,
    }

    # ------------------------------------------------------------------
    # 2. GitHub Repositories (299 human candidates across 7 repos)
    # ------------------------------------------------------------------
    gh_manifest_path = harvest_dir / "github" / "github_manifest.json"
    gh_data = json.loads(gh_manifest_path.read_text(encoding="utf-8"))
    gh_repos_summary = gh_data["summary"]["repos"]
    repos_root = harvest_dir / "github" / "repos"

    # Verified repository-level SPDX mappings from actual LICENSE files + catalog declarations
    gh_repo_spdx: dict[str, tuple[str, str]] = {
        "Chunk7w/GieGie-Bicycle": ("CC-BY-SA-4.0", "cc_by_sa"),
        "Valdr687/Vervada-s-enhanced-plants": ("CC-BY-SA-4.0", "cc_by_sa"),
        "FrenchKrab/mc-blockbench-models": ("CC-BY-4.0", "permissive_cc"),
        "SL0ANE/Loy-s-Goodies": ("CC0-1.0", "permissive_cc"),
        "conormcq24/IndigoOpsFurnitureMod": ("CC0-1.0", "permissive_cc"),
        "mini-kio/snowy-christmas-models": ("CC0-1.0", "permissive_cc"),
        "nahkd123/crystalize": ("CC0-1.0", "permissive_cc"),
    }

    gh_count = 0
    for idx, gf in enumerate(gh_data["files"]):
        if gf["record_id"] in excluded_ids or gf["source_url"] in excluded_urls:
            continue
        repo_name = gf["repo"]
        spdx, split_cfg = gh_repo_spdx[repo_name]
        pol = "sharealike_ccbysa" if split_cfg == "cc_by_sa" else "clean_room_permissive"
        lp = Path(gf["local_path"])
        if not lp.is_file():
            continue
        r_dir = repos_root / repo_name.replace("/", "__")
        lic_fname = gh_repos_summary[repo_name].get("license_file")
        lic_sha = (
            _sha256_bytes((r_dir / lic_fname).read_bytes())
            if lic_fname and (r_dir / lic_fname).is_file()
            else ""
        )

        git_auth = _clean_person_name(gf.get("git_last_commit_author", ""))
        cat_auth = _clean_person_name(gf.get("author_catalog", ""))
        if repo_name == "nahkd123/crystalize":
            author_credit = "Tran Huu An (nahkd123) and Crystalize contributors"
        elif git_auth and git_auth.lower() not in cat_auth.lower():
            author_credit = f"{cat_auth} (Git commit: {git_auth})" if cat_auth else git_auth
        else:
            author_credit = cat_auth or git_auth or repo_name.split("/")[0]

        iso_part = "2026-11-15T00:00:00Z"
        if "|" in str(gf.get("git_last_commit_author", "")):
            cand_iso = str(gf["git_last_commit_author"]).split("|", 1)[1].strip()
            if cand_iso:
                iso_part = cand_iso

        if lp.name.endswith(".bbmodel"):
            fkind = "bbmodel"
            lkind = "disk_json"
        elif lp.name.endswith(".geo.json"):
            fkind = "bedrock_geojson"
            lkind = "disk_json"
        else:
            doc_peek = json.loads(lp.read_text(encoding="utf-8", errors="replace"))
            if "minecraft:geometry" in doc_peek:
                fkind = "bedrock_geojson"
                lkind = "disk_json"
            else:
                fkind = "java_json"
                lkind = "java_with_parents"

        comp_png_path: str | None = None
        comp_png_sha = ""
        if lkind == "disk_json":
            found_png = _find_companion_png_path_on_disk(lp, r_dir)
            if found_png is not None:
                comp_png_path = str(found_png)
                comp_png_sha = _sha256_bytes(found_png.read_bytes())

        src_sha = gf["sha256"]
        commit_sha = gf["commit_sha"]
        lic_url = LICENSE_URLS[spdx]
        title = gf.get("name") or lp.stem
        asset_id = f"github__{_slugify(repo_name)}__{_slugify(lp.stem)}_{src_sha[:8]}"
        author_url = f"https://github.com/{repo_name.split('/')[0]}"
        perm_url = f"https://github.com/{repo_name}/blob/{commit_sha}/{gf['rel_path']}"
        notice = (
            f"'{title}' by {author_credit}, from GitHub repository {repo_name} "
            f"(commit {commit_sha[:12]}, {perm_url}), licensed under {spdx} ({lic_url})."
        )
        specs.append(
            {
                "asset_id": asset_id,
                "title": title,
                "author_credit": author_credit,
                "author_url": author_url,
                "platform": "github.com",
                "source_url": perm_url,
                "commit_or_uid": commit_sha,
                "source_sha256": src_sha,
                "atlas_sha256": comp_png_sha,
                "anim_sha256": "",
                "license_spdx": spdx,
                "license_url": lic_url,
                "license_text_sha256": lic_sha,
                "license_policy": pol,
                "split_config": split_cfg,
                "ai_assisted": False,
                "retrieved_at_iso": iso_part,
                "format_kind": fkind,
                "loader_kind": lkind,
                "local_path": str(lp),
                "pack_root": str(r_dir),
                "companion_png_path": comp_png_path,
                "rel_hint": f"{repo_name}/{gf['rel_path']}",
                "attribution_notice": notice,
                "seed": 30000 + idx,
                "credit_entry": {
                    "asset_id": asset_id,
                    "title": title,
                    "authors": author_credit,
                    "author_url": author_url,
                    "platform": "github.com",
                    "source_url": perm_url,
                    "commit_or_uid": commit_sha,
                    "license_spdx": spdx,
                    "license_url": lic_url,
                    "split_config": split_cfg,
                    "source_sha256": src_sha,
                    "changes_made": CHANGES_MADE_DEFAULT,
                },
            }
        )
        gh_count += 1
    source_discovery_stats["github"] = {"candidate_files": gh_count, "repos": list(gh_repo_spdx.keys())}

    # ------------------------------------------------------------------
    # 3. Modrinth (Minecraft HD(64x) 3D Add-On v1.1 by mrdhobbs, CC-BY-4.0)
    # ------------------------------------------------------------------
    unp_mod = harvest_dir / "modrinth" / "unpacked_euNIaZbl"
    all_mod_jsons = {p.stem.lower(): p for p in unp_mod.rglob("models/**/*.json")}
    mod_rows = [r for r in catalog_rows if "modrinth.com" in r.get("platform", "")]
    mod_count = 0
    for idx, mr in enumerate(mod_rows):
        if mr["record_id"] in excluded_ids:
            continue
        k1 = mr["name"].strip().lower()
        k2 = mr["record_id"].split("::")[-1].strip().lower()
        mp = all_mod_jsons.get(k1) or all_mod_jsons.get(k2)
        if mp is None or not mp.is_file():
            continue
        src_sha = _sha256_bytes(mp.read_bytes())
        spdx = "CC-BY-4.0"
        lic_url = LICENSE_URLS[spdx]
        title = mr["name"].strip()
        author_credit = "mrdhobbs"
        source_url = "https://modrinth.com/resourcepack/minecraft-hd-64x-3d-add-on/version/euNIaZbl"
        asset_id = f"modrinth__mrdhobbs__{_slugify(title)}_{src_sha[:8]}"
        notice = (
            f"'{title}' from 'Minecraft HD(64x) 3D Add-On' (version euNIaZbl) by {author_credit} "
            f"({source_url}), licensed under {spdx} ({lic_url})."
        )
        specs.append(
            {
                "asset_id": asset_id,
                "title": title,
                "author_credit": author_credit,
                "author_url": "https://modrinth.com/user/mrdhobbs",
                "platform": "modrinth.com",
                "source_url": source_url,
                "commit_or_uid": "euNIaZbl",
                "source_sha256": src_sha,
                "atlas_sha256": "",
                "anim_sha256": "",
                "license_spdx": spdx,
                "license_url": lic_url,
                "license_text_sha256": "",
                "license_policy": "clean_room_permissive",
                "split_config": "permissive_cc",
                "ai_assisted": False,
                "retrieved_at_iso": "2026-11-15T00:00:00Z",
                "format_kind": "java_json",
                "loader_kind": "java_with_parents",
                "local_path": str(mp),
                "pack_root": str(unp_mod),
                "rel_hint": mp.relative_to(unp_mod).as_posix(),
                "attribution_notice": notice,
                "seed": 40000 + idx,
                "credit_entry": {
                    "asset_id": asset_id,
                    "title": title,
                    "authors": author_credit,
                    "author_url": "https://modrinth.com/user/mrdhobbs",
                    "platform": "modrinth.com",
                    "source_url": source_url,
                    "commit_or_uid": "euNIaZbl",
                    "license_spdx": spdx,
                    "license_url": lic_url,
                    "split_config": "permissive_cc",
                    "source_sha256": src_sha,
                    "changes_made": CHANGES_MADE_DEFAULT,
                },
            }
        )
        mod_count += 1
    source_discovery_stats["modrinth"] = {"candidate_files": mod_count}

    # ------------------------------------------------------------------
    # 4. Itch.io Packs (ACTG human CC0-1.0 + 1CAMOU/BlockImagine AI-assisted CC0-1.0)
    # ------------------------------------------------------------------
    itch_packs = [
        (
            "australasian_swamphen.zip",
            "ACTG",
            "https://actg.itch.io",
            "https://actg.itch.io/australasian-swamphen",
            False,
            "permissive_cc",
            "clean_room_permissive",
            "actg.itch.io",
        ),
        (
            "BlockImagine-Cozy-Farm.zip",
            "1CAMOU / BlockImagine",
            "https://1camou.itch.io",
            "https://1camou.itch.io/cozy-farm-blockimagine",
            True,
            "ai_assisted",
            "ai_assisted_quarantine",
            "1camou.itch.io",
        ),
        (
            "BlockImagine-Cozy-Winter.zip",
            "1CAMOU / BlockImagine",
            "https://1camou.itch.io",
            "https://1camou.itch.io/cozy-winter-blockimagine",
            True,
            "ai_assisted",
            "ai_assisted_quarantine",
            "1camou.itch.io",
        ),
        (
            "BlockImagine-Sci-Fi-Outpost.zip",
            "1CAMOU / BlockImagine",
            "https://1camou.itch.io",
            "https://1camou.itch.io/sci-fi-outpost-blockimagine",
            True,
            "ai_assisted",
            "ai_assisted_quarantine",
            "1camou.itch.io",
        ),
    ]
    itch_count = 0
    for zp_name, auth_name, auth_url, pack_url, is_ai_ast, split_cfg, pol, plat in itch_packs:
        zp = harvest_dir / "itch" / zp_name
        if not zp.is_file():
            continue
        pack_sha = _sha256_bytes(zp.read_bytes())
        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            png_by_stem = {
                Path(n).stem.lower(): n
                for n in names
                if n.lower().endswith(".png") and not Path(n).name.startswith(".")
            }
            bb_members = sorted(
                n for n in names if n.lower().endswith(".bbmodel") and not Path(n).name.startswith(".")
            )
            for m_idx, member in enumerate(bb_members):
                raw_b = zf.read(member)
                src_sha = _sha256_bytes(raw_b)
                stem = Path(member).stem
                png_mem = png_by_stem.get(stem.lower())
                png_sha = _sha256_bytes(zf.read(png_mem)) if png_mem else ""
                spdx = "CC0-1.0"
                lic_url = LICENSE_URLS[spdx]
                title = stem.replace("-", " ").replace("_", " ").title()
                asset_id = f"itch__{_slugify(auth_name)}__{_slugify(stem)}_{src_sha[:8]}"
                notice = (
                    f"'{title}' ({zp_name}) by {auth_name} ({pack_url}), "
                    f"released under {spdx} ({lic_url})"
                    + (" [AI-assisted pack quarantined in ai_assisted split]." if is_ai_ast else ".")
                )
                specs.append(
                    {
                        "asset_id": asset_id,
                        "title": title,
                        "author_credit": auth_name,
                        "author_url": auth_url,
                        "platform": plat,
                        "source_url": pack_url,
                        "commit_or_uid": pack_sha[:16],
                        "source_sha256": src_sha,
                        "atlas_sha256": png_sha,
                        "anim_sha256": "",
                        "license_spdx": spdx,
                        "license_url": lic_url,
                        "license_text_sha256": "",
                        "license_policy": pol,
                        "split_config": split_cfg,
                        "ai_assisted": is_ai_ast,
                        "retrieved_at_iso": "2026-11-15T00:00:00Z",
                        "format_kind": "bbmodel",
                        "loader_kind": "itch_zip_member",
                        "zip_path": str(zp),
                        "zip_member": member,
                        "zip_png_member": png_mem,
                        "rel_hint": member,
                        "attribution_notice": notice,
                        "seed": 50000 + itch_count,
                        "credit_entry": {
                            "asset_id": asset_id,
                            "title": title,
                            "authors": auth_name,
                            "author_url": auth_url,
                            "platform": plat,
                            "source_url": pack_url,
                            "commit_or_uid": pack_sha[:16],
                            "license_spdx": spdx,
                            "license_url": lic_url,
                            "split_config": split_cfg,
                            "source_sha256": src_sha,
                            "changes_made": CHANGES_MADE_DEFAULT,
                        },
                    }
                )
                itch_count += 1
    source_discovery_stats["itch"] = {"bbmodel_files": itch_count}

    # ------------------------------------------------------------------
    # 5. Blockbench Workshop (deduplicated by SHA-256)
    # ------------------------------------------------------------------
    bbw_rows_by_slug: dict[str, dict[str, str]] = {}
    for r in catalog_rows:
        if "blockbenchworkshop.com" in r.get("platform", ""):
            slug = r["source_url"].rstrip("/").split("/")[-1].lower()
            bbw_rows_by_slug[slug] = r

    bbw_dir = harvest_dir / "bbworkshop"
    seen_bbw_shas: set[str] = set()
    bbw_count = 0
    if bbw_dir.is_dir():
        for bp in sorted(bbw_dir.glob("*.bbmodel")):
            raw_b = bp.read_bytes()
            src_sha = _sha256_bytes(raw_b)
            if src_sha in seen_bbw_shas:
                continue
            seen_bbw_shas.add(src_sha)
            slug = re.sub(r"_\d+$", "", bp.stem).lower()
            cat_r = bbw_rows_by_slug.get(slug)
            if cat_r is None:
                continue
            if cat_r["record_id"] in excluded_ids or cat_r["source_url"] in excluded_urls:
                continue
            spdx = _normalize_spdx(cat_r["license"])
            is_ai = "ai" in str(cat_r.get("synthetic_label", "")).lower()
            if is_ai:
                split_cfg = "ai_assisted"
                pol = "ai_assisted_quarantine"
            elif spdx in ("CC-BY-SA-4.0", "CC-BY-SA-3.0"):
                split_cfg = "cc_by_sa"
                pol = "sharealike_ccbysa"
            else:
                split_cfg = "permissive_cc"
                pol = "clean_room_permissive"

            lic_url = LICENSE_URLS.get(spdx, "https://creativecommons.org/licenses/")
            title = cat_r["name"].strip() or bp.stem
            author_credit = cat_r["author"].strip()
            src_url = cat_r["source_url"].strip()
            url_parts = src_url.rstrip("/").split("/")
            author_handle = url_parts[-2] if len(url_parts) >= 2 else ""
            author_url = f"https://blockbenchworkshop.com/user/{author_handle}" if author_handle else "https://blockbenchworkshop.com"
            asset_id = f"bbworkshop__{_slugify(author_handle or author_credit)}__{_slugify(slug)}_{src_sha[:8]}"
            notice = (
                f"'{title}' by {author_credit} ({src_url}), "
                f"published on Blockbench Workshop under {spdx} ({lic_url})."
            )
            specs.append(
                {
                    "asset_id": asset_id,
                    "title": title,
                    "author_credit": author_credit,
                    "author_url": author_url,
                    "platform": "blockbenchworkshop.com",
                    "source_url": src_url,
                    "commit_or_uid": slug,
                    "source_sha256": src_sha,
                    "atlas_sha256": "",
                    "anim_sha256": "",
                    "license_spdx": spdx,
                    "license_url": lic_url,
                    "license_text_sha256": "",
                    "license_policy": pol,
                    "split_config": split_cfg,
                    "ai_assisted": is_ai,
                    "retrieved_at_iso": "2026-11-15T00:00:00Z",
                    "format_kind": "bbmodel",
                    "loader_kind": "disk_json",
                    "local_path": str(bp),
                    "rel_hint": slug,
                    "attribution_notice": notice,
                    "seed": 60000 + bbw_count,
                    "credit_entry": {
                        "asset_id": asset_id,
                        "title": title,
                        "authors": author_credit,
                        "author_url": author_url,
                        "platform": "blockbenchworkshop.com",
                        "source_url": src_url,
                        "commit_or_uid": slug,
                        "license_spdx": spdx,
                        "license_url": lic_url,
                        "split_config": split_cfg,
                        "source_sha256": src_sha,
                        "changes_made": CHANGES_MADE_DEFAULT,
                    },
                }
            )
            bbw_count += 1
    source_discovery_stats["bbworkshop"] = {
        "downloaded_unique_bbmodels": bbw_count,
        "total_catalog_bbworkshop": len(bbw_rows_by_slug),
    }

    # ------------------------------------------------------------------
    # 6. Sketchfab Downloaded glTF 2.0 Archives (CC0 / CC-BY / CC-BY-SA)
    # ------------------------------------------------------------------
    sf_rows_by_uid: dict[str, dict[str, str]] = {}
    for r in catalog_rows:
        if "sketchfab.com" in r.get("platform", "") and r.get("uid"):
            sf_rows_by_uid[r["uid"].strip()] = r

    sf_items_dir = harvest_dir / "sketchfab" / "items"
    sf_count = 0
    if sf_items_dir.is_dir():
        for item_dir in sorted(sf_items_dir.iterdir()):
            if not item_dir.is_dir():
                continue
            uid = item_dir.name
            if uid in excluded_uids:
                continue
            zip_path = item_dir / "gltf.zip"
            if not zip_path.is_file() or zip_path.stat().st_size < 100:
                continue
            cat_r = sf_rows_by_uid.get(uid)
            if cat_r is None:
                continue
            if "standard" in str(cat_r.get("license", "")).lower():
                continue

            meta_path = item_dir / "meta.json"
            meta_doc = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            lic_obj = meta_doc.get("license") or {}
            spdx = _normalize_spdx(lic_obj.get("slug") or lic_obj.get("label") or cat_r.get("license", ""))
            if spdx not in ("CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0"):
                continue

            is_ai = "ai" in str(cat_r.get("synthetic_label", "")).lower()
            if is_ai:
                split_cfg = "ai_assisted"
                pol = "ai_assisted_quarantine"
            elif spdx in ("CC-BY-SA-4.0", "CC-BY-SA-3.0"):
                split_cfg = "cc_by_sa"
                pol = "sharealike_ccbysa"
            else:
                split_cfg = "permissive_cc"
                pol = "clean_room_permissive"

            user_obj = meta_doc.get("user") or {}
            disp_name = _clean_person_name(user_obj.get("displayName") or user_obj.get("username") or cat_r.get("author", ""))
            username = str(user_obj.get("username") or "").strip()
            if username and username.lower() != disp_name.lower():
                author_credit = f"{disp_name} (@{username})"
            else:
                author_credit = disp_name or username or "Sketchfab Creator"
            author_url = str(user_obj.get("profileUrl") or (f"https://sketchfab.com/{username}" if username else "https://sketchfab.com"))
            title = str(meta_doc.get("name") or cat_r.get("name") or uid).strip()
            src_url = str(meta_doc.get("viewerUrl") or cat_r.get("source_url") or f"https://sketchfab.com/3d-models/{uid}").strip()
            lic_url = LICENSE_URLS.get(spdx, "https://creativecommons.org/licenses/")

            status_path = item_dir / "status.json"
            src_sha = ""
            if status_path.is_file():
                try:
                    src_sha = str(json.loads(status_path.read_text(encoding="utf-8")).get("sha256") or "")
                except Exception:
                    src_sha = ""
            if len(src_sha) != 64:
                src_sha = _sha256_bytes(zip_path.read_bytes())

            asset_id = f"sketchfab__{_slugify(username or disp_name)}__{_slugify(title)}_{uid[:8]}"
            notice = (
                f"'{title}' by {author_credit} ({src_url}), "
                f"published on Sketchfab (UID: {uid}) under {spdx} ({lic_url})."
            )
            specs.append(
                {
                    "asset_id": asset_id,
                    "title": title,
                    "author_credit": author_credit,
                    "author_url": author_url,
                    "platform": "sketchfab.com",
                    "source_url": src_url,
                    "commit_or_uid": uid,
                    "source_sha256": src_sha,
                    "atlas_sha256": "",
                    "anim_sha256": "",
                    "license_spdx": spdx,
                    "license_url": lic_url,
                    "license_text_sha256": "",
                    "license_policy": pol,
                    "split_config": split_cfg,
                    "ai_assisted": is_ai,
                    "retrieved_at_iso": "2026-11-15T00:00:00Z",
                    "format_kind": "gltf_2",
                    "loader_kind": "gltf_zip",
                    "local_path": str(zip_path),
                    "rel_hint": f"{title} {' '.join(t.get('name', '') for t in (meta_doc.get('tags') or []) if isinstance(t, dict))}",
                    "attribution_notice": notice,
                    "seed": 70000 + sf_count,
                    "credit_entry": {
                        "asset_id": asset_id,
                        "title": title,
                        "authors": author_credit,
                        "author_url": author_url,
                        "platform": "sketchfab.com",
                        "source_url": src_url,
                        "commit_or_uid": uid,
                        "license_spdx": spdx,
                        "license_url": lic_url,
                        "split_config": split_cfg,
                        "source_sha256": src_sha,
                        "changes_made": CHANGES_MADE_DEFAULT,
                    },
                }
            )
            sf_count += 1
    source_discovery_stats["sketchfab"] = {
        "downloaded_gltf_zips": sf_count,
        "total_cc_catalog_sketchfab": len(sf_rows_by_uid) - 4,
    }
    return specs, source_discovery_stats


def _assign_train_val_partition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic ~90/10 train/validation split ensuring >=1 validation item when len >= 4."""
    ordered = sorted(rows, key=lambda r: (r["platform"], r["asset_id"]))
    if len(ordered) < 4:
        return ordered, []
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(ordered):
        h = int(_sha256_text(r["asset_id"])[:8], 16)
        if h % 10 == 0:
            r_copy = dict(r)
            r_copy["split"] = f"{r['split']}/validation"
            val_rows.append(r_copy)
        else:
            r_copy = dict(r)
            r_copy["split"] = f"{r['split']}/train"
            train_rows.append(r_copy)
    if not val_rows:
        moved = dict(train_rows.pop())
        moved["split"] = moved["split"].replace("/train", "/validation")
        val_rows.append(moved)
    return train_rows, val_rows


def _write_jsonl_and_parquet(rows: list[dict[str, Any]], jsonl_path: Path, pq_path: Path) -> tuple[str, str]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    j_hasher = hashlib.sha256()
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            line = json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
            fh.write(line)
            j_hasher.update(line.encode("utf-8"))
    if rows:
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, pq_path, compression="zstd")
    else:
        empty_tbl = pa.table({"asset_id": pa.array([], type=pa.string())})
        pq.write_table(empty_tbl, pq_path, compression="zstd")
    return j_hasher.hexdigest(), _sha256_bytes(pq_path.read_bytes())


def _bibtex_escape(val: str) -> str:
    return (
        str(val or "")
        .replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def _generate_citations_bib(
    dataset_repo_id: str,
    credits: list[dict[str, Any]],
    cobb_sha: str,
    cobb_contribs: list[tuple[str, int]],
) -> str:
    lines: list[str] = [
        "% ==========================================================================",
        "% SyncCraft v3.2 Human-Artist 3D Dataset — Complete Author Citations (BibTeX)",
        "% Every upstream creator, repository, and collective whose CC / permissive",
        "% work was decompiled into this dataset is explicitly cited below.",
        "% ==========================================================================",
        "",
        f"@misc{{synccraft2026humanartistdataset,",
        f"  title        = {{SyncCraft v3.2 Human-Artist 3D Dataset: Provenance-Audited Blockbench, Minecraft & glTF Cuboid Rigs, PBR/OKLab Material Posteriors, and Keyframe Animations}},",
        f"  author       = {{BaiTian6641 and SyncCraft Contributors}},",
        f"  year         = {{2026}},",
        f"  publisher    = {{Hugging Face}},",
        f"  url          = {{https://huggingface.co/datasets/{dataset_repo_id}}},",
        f"  note         = {{Decompiled from CC0, CC-BY, CC-BY-SA, MIT, and Cobblemon CC-BY-NC-SA-3.0 3D works; see ATTRIBUTION_CREDITS.csv for per-asset lineage}}",
        f"}}",
        "",
        f"@misc{{cobblemon2026assets,",
        f"  title        = {{Cobblemon Assets Repository (Blockbench 3D Models, Textures, and Animations)}},",
        f"  author       = {{The Cobblemon Team and {' and '.join(_bibtex_escape(c[0]) for c in cobb_contribs[:25])}}},",
        f"  year         = {{2026}},",
        f"  url          = {{https://gitlab.com/cable-mc/cobblemon-assets}},",
        f"  note         = {{Commit {cobb_sha[:12]}; Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC-BY-NC-SA-3.0)}}",
        f"}}",
        "",
    ]

    # Group non-Cobblemon credits by (platform, authors) so every single human creator gets a dedicated BibTeX entry!
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in credits:
        if c["platform"] == "cobblemon-gitlab":
            continue
        grouped[(c["platform"], c["authors"])].append(c)

    used_keys: set[str] = {"synccraft2026humanartistdataset", "cobblemon2026assets"}
    for (plat, auth), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1].lower())):
        base_key = _slugify(f"{plat.split('.')[0]}_{auth}", "creator")[:42]
        bkey = base_key
        seq = 2
        while bkey in used_keys:
            bkey = f"{base_key}_{seq}"
            seq += 1
        used_keys.add(bkey)
        sample_titles = "; ".join(it["title"] for it in items[:4])
        if len(items) > 4:
            sample_titles += f"; and {len(items) - 4} more"
        lics = ", ".join(sorted({it["license_spdx"] for it in items}))
        url0 = items[0]["author_url"] or items[0]["source_url"]
        lines.extend(
            [
                f"@misc{{{bkey},",
                f"  author       = {{{_bibtex_escape(auth)}}},",
                f"  title        = {{{_bibtex_escape(sample_titles)} ({len(items)} 3D asset(s) on {_bibtex_escape(plat)})}},",
                f"  year         = {{2026}},",
                f"  url          = {{{url0}}},",
                f"  note         = {{License: {_bibtex_escape(lics)}; Source example: {items[0]['source_url']}}}",
                f"}}",
                "",
            ]
        )
    return "\n".join(lines)


def _md_escape(val: str) -> str:
    return str(val or "").replace("|", "\\|").replace("\n", " ").strip()


def _generate_authors_index_and_readme(
    *,
    dataset_repo_id: str,
    split_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    credits: list[dict[str, Any]],
    skipped_unsupported: list[dict[str, Any]],
    skipped_errors: list[dict[str, Any]],
    discovery_stats: dict[str, Any],
    manifest_sha256: str,
    sheet_rel_paths: list[str],
    visual_identity_summary: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Generate both AUTHORS_INDEX.md and comprehensive Hugging Face Dataset Card README.md."""
    total_rows = sum(len(tr) + len(va) for tr, va in split_rows.values())
    all_rows = [r for tr, va in split_rows.values() for r in (*tr, *va)]
    total_clips = sum(int(r["clip_count"]) for r in all_rows)
    animated_assets = sum(1 for r in all_rows if int(r["clip_count"]) > 0)
    max_vert_err = max((float(r["vertex_reconstruction_error"]) for r in all_rows), default=0.0)
    mean_boxes = sum(int(r["box_count"]) for r in all_rows) / max(1, len(all_rows))
    mean_bones = sum(int(r["bone_count"]) for r in all_rows) / max(1, len(all_rows))

    if total_rows < 1000:
        size_cat = "n<1K"
    elif total_rows < 10000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    cobb_info = discovery_stats["cobblemon"]
    cobb_contribs: list[tuple[str, int]] = cobb_info["git_contributors"]
    cobb_sha: str = cobb_info["commit_sha"]

    # Group non-Cobblemon authors by platform -> author
    by_plat_author: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for c in credits:
        if c["platform"] != "cobblemon-gitlab":
            by_plat_author[c["platform"]][c["authors"]].append(c)

    # Build exhaustive per-platform Markdown tables naming 100% of creators!
    credit_sections_md: list[str] = []

    # 1) Cobblemon section naming all Git contributors + Eiko
    cobb_count = sum(1 for c in credits if c["platform"] == "cobblemon-gitlab")
    cobb_contrib_badges = ", ".join(f"**{_md_escape(name)}** ({cnt} commits)" for name, cnt in cobb_contribs)
    credit_sections_md.append(
        f"### 1. The Cobblemon Team & Contributors (`cobblemon_cc_by_nc_sa` — `{cobb_count:,}` models)\n\n"
        f"- **Upstream Repository**: [https://gitlab.com/cable-mc/cobblemon-assets](https://gitlab.com/cable-mc/cobblemon-assets) (`commit {cobb_sha}`)\n"
        f"- **License**: [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (`CC-BY-NC-SA-3.0`)](https://creativecommons.org/licenses/by-nc-sa/3.0/)\n"
        f"- **All {len(cobb_contribs)} Repository Git Contributors & Credited Modelers**:\n"
        f"  {cobb_contrib_badges}, and embedded Blockbench model credit **Eiko**.\n"
        f"- *Per-file contributor lists, commit hashes, and SHA-256 digests for all `{cobb_count:,}` Cobblemon `.bbmodel` assets are recorded in `ATTRIBUTION_CREDITS.csv`.*"
    )

    # 2) All other platforms — list EVERY SINGLE AUTHOR in clean markdown tables!
    plat_titles = [
        ("github.com", "2. GitHub Open-Source Blockbench & Minecraft Model Creators"),
        ("modrinth.com", "3. Modrinth Resource-Pack 3D Creators"),
        ("actg.itch.io", "4. Itch.io Human-Authored 3D Asset Creators (`ACTG`)"),
        ("blockbenchworkshop.com", "5. Blockbench Workshop Community Creators"),
        ("sketchfab.com", "6. Sketchfab Creative Commons 3D Creators"),
        ("1camou.itch.io", "7. Quarantined AI-Assisted Pack Creators (`ai_assisted` split)"),
    ]
    for plat_key, heading in plat_titles:
        auth_map = by_plat_author.get(plat_key, {})
        if not auth_map:
            continue
        plat_total = sum(len(v) for v in auth_map.values())
        tbl_lines = [
            f"### {heading} (`{len(auth_map)}` creator(s), `{plat_total:,}` compiled models)\n",
            "| Creator / Author | Models | License(s) | Split(s) | Profile / Source Link | Example Work(s) |",
            "| :--- | ---: | :--- | :--- | :--- | :--- |",
        ]
        for auth_name, items in sorted(auth_map.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
            lics = ", ".join(sorted({f"`{it['license_spdx']}`" for it in items}))
            splits = ", ".join(sorted({f"`{it['split_config']}`" for it in items}))
            link_url = items[0]["author_url"] or items[0]["source_url"]
            examples = "; ".join(_md_escape(it["title"]) for it in items[:3])
            if len(items) > 3:
                examples += f" (+{len(items) - 3} more)"
            tbl_lines.append(
                f"| **{_md_escape(auth_name)}** | {len(items)} | {lics} | {splits} | [{_md_escape(link_url)}]({link_url}) | {examples} |"
            )
        credit_sections_md.append("\n".join(tbl_lines))

    full_credits_markdown = "\n\n".join(credit_sections_md)

    authors_index_md = (
        f"# Complete Author & Contributor Credit Directory — `{dataset_repo_id}`\n\n"
        f"This file credits **100% of upstream human artists, open-source contributors, and creators** "
        f"whose 3D works were decompiled into `{dataset_repo_id}` (`{total_rows:,}` total verified Block-DSL v3.2 records).\n\n"
        f"For machine-readable per-asset lineage (exact asset ID, title, source URL, commit SHA / UID, "
        f"SPDX license, license URL, SHA-256 digest, and CC Section 3(b)/4(c) adaptation notice), inspect "
        f"`ATTRIBUTION_CREDITS.csv` and `ATTRIBUTION_CREDITS.json`. For BibTeX citations, see `CITATIONS.bib`.\n\n"
        f"{full_credits_markdown}\n"
    )

    domain_counts = Counter(r["generator_parameters"]["domain"] for r in all_rows)
    format_counts = Counter(r["format_kind"] for r in all_rows)
    mat_counts = Counter(r["dominant_material_family"] for r in all_rows)
    mode_counts = Counter(r.get("assistant_task_mode", r["generator_parameters"]["assistant_task_mode"]) for r in all_rows)
    mean_slots = sum(int(r.get("material_slot_count", 1)) for r in all_rows) / max(1, len(all_rows))
    vis_md = ""
    if visual_identity_summary and visual_identity_summary.get("audited_asset_count", 0) > 0:
        vis_md = (
            f"\n### Independent Cross-Representation Visual-Identity Audit (`{visual_identity_summary['audited_asset_count']}` Stratified Assets)\n\n"
            f"- **Mean Raw-Source vs. Converted Block-DSL v3.2 PSNR**: `{visual_identity_summary['mean_psnr_db']:.2f} dB` (median `{visual_identity_summary['median_psnr_db']:.2f} dB`, standalone formatted DSL `{visual_identity_summary['mean_standalone_dsl_psnr_db']:.2f} dB`)\n"
            f"- **Mean Foreground RGB MAE**: `{visual_identity_summary['mean_mae_rgb']:.3f} / 255`\n"
            f"- **Mean Silhouette IoU**: `{visual_identity_summary['mean_silhouette_iou']:.4f}` (min `{visual_identity_summary['min_silhouette_iou']:.4f}`)\n"
            f"- **Per-Platform PSNR (dB)**: `{visual_identity_summary.get('per_platform_mean_psnr_db', {})}`\n"
        )

    p_tr, p_va = split_rows["permissive_cc"]
    s_tr, s_va = split_rows["cc_by_sa"]
    c_tr, c_va = split_rows["cobblemon_cc_by_nc_sa"]
    a_tr, a_va = split_rows["ai_assisted"]

    sheets_bullets = "\n".join(f"- `{rel}`" for rel in sheet_rel_paths) or "- *(Rendered sheets stored in `inspection_sheets/`)*"

    readme_md = f"""---
license: other
license_name: multi-license-quarantined-cc-and-nc
license_link: ATTRIBUTION_CREDITS.csv
task_categories:
- text-generation
language:
- en
pretty_name: SyncCraft v3.2 Human-Artist 3D Dataset (Blockbench + Cobblemon + Sketchfab + GitHub + Modrinth)
size_categories:
- {size_cat}
tags:
- synccraft
- block-dsl
- blockbench
- minecraft
- cobblemon
- sketchfab
- gltf
- 3d-geometry
- world-model
- pbr-texture
- skeletal-animation
- human-authored
configs:
- config_name: default
  data_files:
  - split: train
    path: parquet/permissive_cc_train.parquet
  - split: validation
    path: parquet/permissive_cc_val.parquet
- config_name: permissive_cc
  data_files:
  - split: train
    path: parquet/permissive_cc_train.parquet
  - split: validation
    path: parquet/permissive_cc_val.parquet
- config_name: cc_by_sa
  data_files:
  - split: train
    path: parquet/cc_by_sa_train.parquet
  - split: validation
    path: parquet/cc_by_sa_val.parquet
- config_name: cobblemon_cc_by_nc_sa
  data_files:
  - split: train
    path: parquet/cobblemon_cc_by_nc_sa_train.parquet
  - split: validation
    path: parquet/cobblemon_cc_by_nc_sa_val.parquet
- config_name: ai_assisted
  data_files:
  - split: train
    path: parquet/ai_assisted_train.parquet
  - split: validation
    path: parquet/ai_assisted_val.parquet
---

# SyncCraft v3.2 — Human-Artist 3D Blockbench, Minecraft & glTF Dataset (`Model` + `Texture` + `Animation`)

Provenance-audited, license-partitioned **real human-artist** (plus isolated AI-assisted) 3D cuboid/plane modeling, PBR/OKLab inverse surface-material, and keyframe skeletal-animation dataset compiled into **[SyncCraft Block-DSL v3.2](https://github.com/BaiTian6641/SyncCraft)**.

Unlike pixel/voxel black-box diffusion corpora, every asset in this repository is decompiled from authentic artist-built `.bbmodel`, `.json`, `.geo.json`, and `glTF 2.0` hierarchies into **typed, human-readable, round-trip-verified Block-DSL v3.2 programs** (`@meta`, `@textures`, `@palette`, `@materials`, `@style`, `@rig` with `bone`/`box`/`plane`/`locator`/`mirror_x`, and `@behavior` keyframe clips) with **verified 3D 8-corner vertex reconstruction** (`max L2 error = {max_vert_err:.6f}` units).

---

## Strict License Quarantine & Configuration Matrix

To prevent license contamination between permissive commercial weights (`Apache-2.0` synthetic `TassP0lar/synccraft-v3-blockdsl-dataset`), ShareAlike (`CC-BY-SA`), NonCommercial (`CC-BY-NC-SA`), and AI-assisted works, **this dataset is partitioned into four strictly isolated Hugging Face configurations**:

| HF Config (`name=...`) | Train | Val | Total | Licenses Included | `commercial_weights_eligible` | Intended Downstream Use |
| :--- | ---: | ---: | ---: | :--- | :---: | :--- |
| **`permissive_cc`** *(also `default`)* | {len(p_tr):,} | {len(p_va):,} | **{len(p_tr) + len(p_va):,}** | `CC0-1.0`, `CC-BY-4.0`, `MIT` | **`True`** *(with CC-BY attribution)* | Permissive / commercial-compatible pretraining & fine-tuning (preserve `ATTRIBUTION_CREDITS.csv`) |
| **`cc_by_sa`** | {len(s_tr):,} | {len(s_va):,} | **{len(s_tr) + len(s_va):,}** | `CC-BY-SA-4.0`, `CC-BY-SA-3.0` | `False` *(ShareAlike)* | Open ShareAlike research & CC-BY-SA model adapters |
| **`cobblemon_cc_by_nc_sa`** | {len(c_tr):,} | {len(c_va):,} | **{len(c_tr) + len(c_va):,}** | `CC-BY-NC-SA-3.0` | `False` *(NonCommercial + SA)* | Non-commercial academic/world-model research on complex articulated creatures & animations |
| **`ai_assisted`** | {len(a_tr):,} | {len(a_va):,} | **{len(a_tr) + len(a_va):,}** | `CC0-1.0` *(AI-used/assisted)* | `False` *(Quarantined)* | Ablation studies comparing pure human vs. AI-assisted Blockbench packs |
| **Total Verified Assets** | **{sum(len(v[0]) for v in split_rows.values()):,}** | **{sum(len(v[1]) for v in split_rows.values()):,}** | **{total_rows:,}** | — | — | — |

### Governance & Exclusion Guarantees
- **Zero `All Rights Reserved`, Paid, or `-ND` (`NoDerivatives`) Works**: All `893` held/excluded rows from catalog review, `4` Sketchfab `Free Standard` listings, and `24` synthetic test fixtures (`MrLarkyy/ModelShot`) were excluded prior to ingestion.
- **Automated `NoAI` / TDM Opt-Out Gate**: Every asset passed `verify_asset_provenance` (`scan_text_for_noai_markers`), rejecting any `#NoAI`, `no_machine_learning`, `do_not_train`, or W3C TDM opt-out tag.
- **Explicit Non-Cuboid Free-Mesh Rejection**: `{len(skipped_unsupported):,}` non-cuboid free-mesh sculpts were cleanly rejected via `UnsupportedFeature` (logged in `excluded_non_cuboid_manifest.jsonl`) rather than silently distorted.

---

## Corpus Summary & Structural Telemetry

- **Default World-Model LLM Backbone**: `{DEFAULT_WORLD_MODEL_BACKBONE}`
- **Dataset Manifest SHA-256**: `{manifest_sha256}`
- **Total Compiled Block-DSL v3.2 Programs**: `{total_rows:,}`
- **Animated Assets (`@behavior` clips)**: `{animated_assets:,}` assets carrying `{total_clips:,}` decompiled keyframe clips
- **Mean Rig Complexity**: `{mean_boxes:.2f}` boxes/planes per asset, `{mean_bones:.2f}` bones per asset, `{mean_slots:.2f}` part-material slots per asset
- **Max 8-Corner World Vertex Reconstruction Error**: `{max_vert_err:.6f}` Blockbench units (`< 1e-3`)
- **Source Formats Decompiled**: `{dict(sorted(format_counts.items()))}`
- **Macro-Domain Distribution**: `{dict(sorted(domain_counts.items()))}`
- **5-Mode World-Model Assistant Curriculum**: `{dict(sorted(mode_counts.items()))}`
- **Inverse-Inferred Physical Material Families**: `{dict(sorted(mat_counts.items()))}`
- **4-Column Visual-Identity Proof Sheets**:
{sheets_bullets}
{vis_md}

---

## Quickstart Usage

```python
from datasets import load_dataset

# 1. Load clean human permissive (CC0 / CC-BY) split:
ds_perm = load_dataset("{dataset_repo_id}", "permissive_cc")
print(ds_perm["train"][0]["title"], "by", ds_perm["train"][0]["author"])
print(ds_perm["train"][0]["canonical_dsl"][:600])

# 2. Load non-commercial Cobblemon creature + animation world-model split:
ds_cobb = load_dataset("{dataset_repo_id}", "cobblemon_cc_by_nc_sa")
print("Cobblemon train rows:", len(ds_cobb["train"]))

# 3. Load CC-BY-SA ShareAlike split:
ds_sa = load_dataset("{dataset_repo_id}", "cc_by_sa")
```

---

## Complete Author Credits & Attribution Ledger (CC Section 3(b) / 4(c) Compliance)

We gratefully credit **every single 3D artist, animator, texture painter, and open-source contributor** whose work is represented in `{dataset_repo_id}`:

- **Per-Asset Machine-Readable Credit Ledger**: [`ATTRIBUTION_CREDITS.csv`](ATTRIBUTION_CREDITS.csv) and [`ATTRIBUTION_CREDITS.json`](ATTRIBUTION_CREDITS.json) (lists every `asset_id`, `title`, `authors`, `author_url`, `source_url`, `commit_or_uid`, `license_spdx`, `license_url`, `source_sha256`, and `changes_made`).
- **Standalone Author Directory**: [`AUTHORS_INDEX.md`](AUTHORS_INDEX.md).
- **Complete BibTeX File**: [`CITATIONS.bib`](CITATIONS.bib).
- **Adaptation / Modification Disclosure (CC BY / CC BY-SA / CC BY-NC-SA)**:
  > *{CHANGES_MADE_DEFAULT} Original creators do not endorse SyncCraft or downstream models trained on these adaptations.*

{full_credits_markdown}

---

## Creator Takedown & Opt-Out Policy

Respecting 3D artists is foundational to SyncCraft. Although every asset in this dataset carries an explicit open Creative Commons (`CC0`, `CC-BY`, `CC-BY-SA`, `CC-BY-NC-SA`) or `MIT` license and passed automated `NoAI` opt-out screening at harvest time, **any credited creator who wishes to have their work removed from future revisions of `{dataset_repo_id}`** may open an issue on the Hugging Face Community tab or at [https://github.com/BaiTian6641/SyncCraft/issues](https://github.com/BaiTian6641/SyncCraft/issues) citing their author handle or `asset_id`, and we will promptly remove the record.

---

## Citation

When using `{dataset_repo_id}`, please cite both the SyncCraft dataset/framework **and** the upstream creators in [`CITATIONS.bib`](CITATIONS.bib) / [`ATTRIBUTION_CREDITS.csv`](ATTRIBUTION_CREDITS.csv):

```bibtex
@misc{{synccraft2026humanartistdataset,
  title        = {{SyncCraft v3.2 Human-Artist 3D Dataset: Provenance-Audited Blockbench, Minecraft & glTF Cuboid Rigs, PBR/OKLab Material Posteriors, and Keyframe Animations}},
  author       = {{BaiTian6641 and SyncCraft Contributors}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  url          = {{https://huggingface.co/datasets/{dataset_repo_id}}},
  note         = {{Decompiled from CC0, CC-BY, CC-BY-SA, MIT, and Cobblemon CC-BY-NC-SA-3.0 3D works; see ATTRIBUTION_CREDITS.csv and CITATIONS.bib}}
}}

@misc{{cobblemon2026assets,
  title        = {{Cobblemon Assets Repository (Blockbench 3D Models, Textures, and Animations)}},
  author       = {{The Cobblemon Team and Contributors}},
  year         = {{2026}},
  url          = {{https://gitlab.com/cable-mc/cobblemon-assets}},
  note         = {{Commit {cobb_sha[:12]}; Licensed under CC-BY-NC-SA-3.0}}
}}
```
"""
    return authors_index_md, readme_md


def _audit_single_row_visual_identity(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    top_k: int = 1,
    canvas_size: tuple[int, int] = (220, 220),
    render_panels: bool = True,
) -> dict[str, Any]:
    """Run independent raw-source vs converted Block-DSL v3.2 visual-identity audit on one verified row."""
    format_kind = str(spec["format_kind"])
    payload, atlas_png, java_textures_png, anim_payload = _load_spec_raw_inputs(spec)
    prov = AssetProvenance(
        asset_id=spec["asset_id"],
        source_url=spec["source_url"],
        author=_sanitize_provenance_text(spec["author_credit"]),
        license_spdx=spec["license_spdx"],
        attribution_notice=_sanitize_provenance_text(spec["attribution_notice"]),
        title=_sanitize_provenance_text(spec["title"]),
        license_url=spec["license_url"],
        license_text_sha256=spec.get("license_text_sha256", ""),
        retrieved_at_iso=spec["retrieved_at_iso"],
        ai_use_marker_status="checked_none",
        ai_marker_evidence_sha256=spec["source_sha256"],
        is_ai_generated=False,
        ai_assisted=bool(spec.get("ai_assisted", False)),
        changes_made_note=CHANGES_MADE_DEFAULT,
        reviewer_id="synccraft-v3.2-catalog-gate",
    )
    rec = ingest_external_asset(
        payload,
        format_kind,
        prov,
        atlas_png=atlas_png,
        java_textures_png=java_textures_png,
        animation_payload=anim_payload,
        license_policy=spec["license_policy"],
        top_k=top_k,
        seed=int(spec["seed"]),
    )
    enriched_prog = parse_program(row["canonical_dsl"])
    metrics, panels = audit_converted_asset_visual_identity(
        payload,
        format_kind,
        rec,
        enriched_prog,
        atlas_png=atlas_png,
        java_textures_png=java_textures_png,
        canvas_size=canvas_size,
    )
    return {
        "asset_id": row["asset_id"],
        "title": row["title"],
        "author": row["author"],
        "platform": row["platform"],
        "format_kind": row["format_kind"],
        "box_count": row["box_count"],
        "bone_count": row["bone_count"],
        "clip_count": row["clip_count"],
        "metrics": metrics,
        "panels": panels if render_panels else None,
    }


def _audit_and_render_visual_identity_sheets(
    specs: list[dict[str, Any]],
    split_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    sheets_dir: Path,
    file_checksums: dict[str, str],
    *,
    top_k: int = 1,
) -> tuple[list[str], dict[str, Any]]:
    """Audit 48 stratified cross-platform assets and render 4 visual-identity proof sheets."""
    specs_by_id = {sp["asset_id"]: sp for sp in specs}
    all_rows = [r for cfg_pair in split_rows.values() for lst in cfg_pair for r in lst]
    used_ids: set[str] = set()

    def _pick_rows(
        pool: list[dict[str, Any]],
        preferred_substrings: list[str],
        count: int = 6,
    ) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        for sub in preferred_substrings:
            if len(picked) >= count:
                break
            sub_l = sub.lower()
            for r in pool:
                if r["asset_id"] in used_ids or r["asset_id"] not in specs_by_id:
                    continue
                if sub_l in r["title"].lower() or sub_l in r["asset_id"].lower():
                    picked.append(r)
                    used_ids.add(r["asset_id"])
                    break
        for r in pool:
            if len(picked) >= count:
                break
            if r["asset_id"] not in used_ids and r["asset_id"] in specs_by_id:
                picked.append(r)
                used_ids.add(r["asset_id"])
        return picked

    cobb_pool = [r for r in all_rows if r["platform"] == "cobblemon-gitlab" and r.get("has_source_texture", True)]
    skf_bbw_pool = [r for r in all_rows if r["platform"] in ("sketchfab.com", "blockbenchworkshop.com")]
    gh_mod_itch_pool = [r for r in all_rows if r["platform"] in ("github.com", "modrinth.com", "actg.itch.io")]

    sheet_groups = [
        (
            "human_qc_01_cobblemon.png",
            "SyncCraft v3.2 Visual Identity Proof -- Cobblemon Creatures & Animated Rigs (CC-BY-NC-SA-3.0)",
            _pick_rows(
                cobb_pool,
                ["charizard", "rayquaza", "gyarados", "cash register", "trainer bag", "bulbasaur"],
                count=6,
            ),
        ),
        (
            "human_qc_02_permissive_sketchfab_bbw.png",
            "SyncCraft v3.2 Visual Identity Proof -- Sketchfab glTF 2.0 & Blockbench Workshop Creators",
            _pick_rows(
                skf_bbw_pool,
                ["simple bench", "better bucket", "apple tree", "epic crate", "lissachatina", "marbled adult newt"],
                count=6,
            ),
        ),
        (
            "human_qc_03_github_modrinth_itch.png",
            "SyncCraft v3.2 Visual Identity Proof -- GitHub, Modrinth & Itch.io Human Rigs",
            _pick_rows(
                gh_mod_itch_pool,
                ["electric bicycle", "220222", "bottom_right_table", "australasian swamphen", "acacia_door_bottom", "acacia_trapdoor_bottom"],
                count=6,
            ),
        ),
        (
            "human_qc_04_edge_cases_and_materials_proof.png",
            "SyncCraft v3.2 Visual Identity Proof -- Multi-Texture, Cutout & Reflection Edge Cases",
            _pick_rows(
                all_rows,
                ["gimmick rings", "sakura bonsai", "boats_boats", "psx style soviet", "big_dripleaf", "birch_trapdoor_bottom"],
                count=6,
            ),
        ),
    ]

    sheet_rel_paths: list[str] = []
    audited_records: list[dict[str, Any]] = []

    for s_name, s_title, group_rows in sheet_groups:
        if not group_rows:
            continue
        cards: list[dict[str, Any]] = []
        for r in group_rows:
            try:
                card = _audit_single_row_visual_identity(
                    r,
                    specs_by_id[r["asset_id"]],
                    top_k=top_k,
                    canvas_size=(220, 220),
                    render_panels=True,
                )
                cards.append(card)
                audited_records.append(card)
            except Exception as exc:
                print(f"      [Warn] Visual audit skipped on {r['asset_id']}: {exc}")
        if cards:
            out_png = sheets_dir / s_name
            render_visual_identity_proof_sheet(cards, out_png, title=s_title)
            if out_png.is_file():
                rel_s = f"inspection_sheets/{s_name}"
                sheet_rel_paths.append(rel_s)
                file_checksums[rel_s] = _sha256_bytes(out_png.read_bytes())
                sh_psnr = statistics.mean(c["metrics"]["psnr_db"] for c in cards)
                sh_iou = statistics.mean(c["metrics"]["silhouette_iou"] for c in cards)
                print(f"      Rendered {s_name}: {len(cards)} assets | mean PSNR={sh_psnr:.2f} dB | mean IoU={sh_iou:.4f}")

    # Audit 24 additional stratified assets across all platforms for a 48-asset cross-platform statistical audit
    by_plat_remaining: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        if r["asset_id"] not in used_ids and r["asset_id"] in specs_by_id:
            by_plat_remaining[r["platform"]].append(r)

    extra_rows: list[dict[str, Any]] = []
    plat_keys = sorted(by_plat_remaining.keys())
    round_idx = 0
    while len(extra_rows) < 24 and plat_keys:
        progressed = False
        for pk in plat_keys:
            rem = by_plat_remaining[pk]
            if round_idx < len(rem):
                cand = rem[round_idx]
                extra_rows.append(cand)
                used_ids.add(cand["asset_id"])
                progressed = True
                if len(extra_rows) >= 24:
                    break
        if not progressed:
            break
        round_idx += 1

    for r in extra_rows:
        try:
            card = _audit_single_row_visual_identity(
                r,
                specs_by_id[r["asset_id"]],
                top_k=top_k,
                canvas_size=(192, 192),
                render_panels=False,
            )
            audited_records.append(card)
        except Exception as exc:
            print(f"      [Warn] Extra visual audit skipped on {r['asset_id']}: {exc}")

    if audited_records:
        psnrs = [c["metrics"]["psnr_db"] for c in audited_records]
        st_psnrs = [c["metrics"]["standalone_dsl_psnr_db"] for c in audited_records]
        maes = [c["metrics"]["mae_rgb"] for c in audited_records]
        ious = [c["metrics"]["silhouette_iou"] for c in audited_records]
        by_p_psnr: dict[str, list[float]] = defaultdict(list)
        for c in audited_records:
            by_p_psnr[c["platform"]].append(c["metrics"]["psnr_db"])
        summary: dict[str, Any] = {
            "audited_asset_count": len(audited_records),
            "mean_psnr_db": round(float(statistics.mean(psnrs)), 2),
            "median_psnr_db": round(float(statistics.median(psnrs)), 2),
            "mean_standalone_dsl_psnr_db": round(float(statistics.mean(st_psnrs)), 2),
            "mean_mae_rgb": round(float(statistics.mean(maes)), 4),
            "mean_silhouette_iou": round(float(statistics.mean(ious)), 4),
            "min_silhouette_iou": round(float(min(ious)), 4),
            "per_platform_mean_psnr_db": {
                pk: round(float(statistics.mean(vals)), 2) for pk, vals in sorted(by_p_psnr.items())
            },
            "per_asset_metrics": [
                {
                    "asset_id": c["asset_id"],
                    "title": c["title"],
                    "platform": c["platform"],
                    "format_kind": c["format_kind"],
                    **c["metrics"],
                }
                for c in audited_records
            ],
        }
    else:
        summary = {
            "audited_asset_count": 0,
            "mean_psnr_db": 0.0,
            "median_psnr_db": 0.0,
            "mean_standalone_dsl_psnr_db": 0.0,
            "mean_mae_rgb": 0.0,
            "mean_silhouette_iou": 0.0,
            "min_silhouette_iou": 0.0,
            "per_platform_mean_psnr_db": {},
            "per_asset_metrics": [],
        }
    return sheet_rel_paths, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobblemon-dir", type=Path, required=True, help="Path to local cobblemon-assets git clone")
    parser.add_argument("--catalog-dir", type=Path, required=True, help="Path to unpacked full-reviewed-catalog dir")
    parser.add_argument("--harvest-dir", type=Path, required=True, help="Path to online harvest dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for HF dataset repo files")
    parser.add_argument("--cache-dir", type=Path, required=True, help="External disk cache dir (kept outside HF git)")
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--top-k-materials", type=int, default=1)
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Discovering authorized candidate specs & verifying exclusion blocklist...")
    specs, discovery_stats = _gather_all_specs(
        cobblemon_dir=args.cobblemon_dir,
        catalog_dir=args.catalog_dir,
        harvest_dir=args.harvest_dir,
    )
    print(f"      Discovered {len(specs):,} candidate specs across 6 sources.")

    print(f"[2/5] Decompiling & auditing {len(specs):,} assets across {args.workers} workers (cache: {args.cache_dir})...")
    from concurrent.futures.process import BrokenProcessPool

    results: list[dict[str, Any]] = [{}] * len(specs)
    uncached_indices: list[tuple[int, tuple[dict[str, Any], str, int]]] = []
    for i_sp, sp in enumerate(specs):
        ck = _sha256_text(
            json.dumps(
                {
                    "v": COMPILER_CACHE_VERSION,
                    "asset_id": sp["asset_id"],
                    "source_sha256": sp["source_sha256"],
                    "atlas_sha256": sp.get("atlas_sha256", ""),
                    "anim_sha256": sp.get("anim_sha256", ""),
                    "split_config": sp["split_config"],
                    "license_policy": sp["license_policy"],
                    "license_spdx": sp["license_spdx"],
                    "author_credit": sp["author_credit"],
                    "top_k": int(args.top_k_materials),
                },
                sort_keys=True,
            )
        )
        cf = args.cache_dir / ck[:2] / f"{ck}.json"
        if cf.is_file():
            try:
                results[i_sp] = json.loads(cf.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        uncached_indices.append((i_sp, (sp, str(args.cache_dir), int(args.top_k_materials))))

    cached_hit_count = len(specs) - len(uncached_indices)
    print(f"      Loaded {cached_hit_count:,}/{len(specs):,} pre-cached assets; compiling {len(uncached_indices):,} delta assets...")

    if uncached_indices:
        chunk_sz = 256
        total_delta = len(uncached_indices)
        done_n = 0
        for c_start in range(0, total_delta, chunk_sz):
            chunk = uncached_indices[c_start : c_start + chunk_sz]
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futs = [(i_sp, sp_t, pool.submit(_compile_single_spec, sp_t)) for i_sp, sp_t in chunk]
                    for i_sp, sp_t, fut in futs:
                        results[i_sp] = fut.result(timeout=90)
                        done_n += 1
                        if done_n % 100 == 0 or done_n == total_delta:
                            print(f"      Delta progress: {done_n:,}/{total_delta:,} ({time.perf_counter() - t0:.1f}s)", flush=True)
            except (BrokenProcessPool, TimeoutError):
                print("      [Warn] Chunk worker interrupted/timed out; isolating remaining items in chunk...", flush=True)
                for i_sp, sp_t in chunk:
                    if results[i_sp]:
                        continue
                    try:
                        with ProcessPoolExecutor(max_workers=1) as iso_pool:
                            results[i_sp] = iso_pool.submit(_compile_single_spec, sp_t).result(timeout=90)
                    except Exception as exc:
                        sp = sp_t[0]
                        results[i_sp] = {
                            "status": "skipped_error",
                            "asset_id": sp["asset_id"],
                            "platform": sp["platform"],
                            "source_url": sp["source_url"],
                            "title": sp["title"],
                            "author": sp["author_credit"],
                            "reason": f"WorkerIsolatedOOMOrTimeout: {type(exc).__name__}",
                        }
                    done_n += 1

    by_config_rows: dict[str, list[dict[str, Any]]] = {
        "permissive_cc": [],
        "cc_by_sa": [],
        "cobblemon_cc_by_nc_sa": [],
        "ai_assisted": [],
    }
    credits: list[dict[str, Any]] = []
    skipped_unsupported: list[dict[str, Any]] = []
    skipped_errors: list[dict[str, Any]] = []
    seen_dsl_per_cfg: dict[str, set[str]] = defaultdict(set)

    for res in results:
        st = res["status"]
        if st == "ok":
            row = res["row"]
            cfg = row["split"]
            # Deduplicate exact AST+material+texture SHA-256 within same split config
            dedup_key = f"{row['canonical_dsl_sha256']}::{row['author']}::{row['title']}"
            if dedup_key in seen_dsl_per_cfg[cfg]:
                continue
            seen_dsl_per_cfg[cfg].add(dedup_key)
            by_config_rows[cfg].append(row)
            credits.append(res["credit"])
        elif st == "unsupported_non_cuboid":
            skipped_unsupported.append(res)
        else:
            skipped_errors.append(res)

    print("[3/5] Writing partitioned JSONL + Zstd Parquet splits & manifests...")
    split_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    file_checksums: dict[str, str] = {}
    for cfg, rows in by_config_rows.items():
        tr_rows, va_rows = _assign_train_val_partition(rows)
        split_rows[cfg] = (tr_rows, va_rows)

        j_tr = args.output_dir / "data" / cfg / "train.jsonl"
        j_va = args.output_dir / "data" / cfg / "val.jsonl"
        p_tr = args.output_dir / "parquet" / f"{cfg}_train.parquet"
        p_va = args.output_dir / "parquet" / f"{cfg}_val.parquet"

        j_tr_sha, p_tr_sha = _write_jsonl_and_parquet(tr_rows, j_tr, p_tr)
        j_va_sha, p_va_sha = _write_jsonl_and_parquet(va_rows, j_va, p_va)
        file_checksums[f"data/{cfg}/train.jsonl"] = j_tr_sha
        file_checksums[f"data/{cfg}/val.jsonl"] = j_va_sha
        file_checksums[f"parquet/{cfg}_train.parquet"] = p_tr_sha
        file_checksums[f"parquet/{cfg}_val.parquet"] = p_va_sha
        print(f"      Config '{cfg}': train={len(tr_rows):,}, val={len(va_rows):,} (total={len(rows):,})")

    excl_path = args.output_dir / "excluded_non_cuboid_manifest.jsonl"
    excl_path.write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in [*skipped_unsupported, *skipped_errors]) + ("\n" if (skipped_unsupported or skipped_errors) else ""),
        encoding="utf-8",
    )
    file_checksums["excluded_non_cuboid_manifest.jsonl"] = _sha256_bytes(excl_path.read_bytes())

    print("[4/5] Running 48-asset Raw-vs-BlockDSL visual identity audit & writing author credit files...")
    sheets_dir = args.output_dir / "inspection_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheet_rel_paths, visual_identity_summary = _audit_and_render_visual_identity_sheets(
        specs=specs,
        split_rows=split_rows,
        sheets_dir=sheets_dir,
        file_checksums=file_checksums,
        top_k=int(args.top_k_materials),
    )
    print(
        f"      Visual Identity Summary ({visual_identity_summary['audited_asset_count']} assets): "
        f"mean PSNR={visual_identity_summary['mean_psnr_db']:.2f} dB "
        f"(median={visual_identity_summary['median_psnr_db']:.2f} dB, "
        f"standalone DSL={visual_identity_summary['mean_standalone_dsl_psnr_db']:.2f} dB), "
        f"MAE={visual_identity_summary['mean_mae_rgb']:.4f}/255, "
        f"IoU={visual_identity_summary['mean_silhouette_iou']:.4f}"
    )

    credits_sorted = sorted(credits, key=lambda c: (c["split_config"], c["platform"], c["authors"].lower(), c["asset_id"]))
    csv_path = args.output_dir / "ATTRIBUTION_CREDITS.csv"
    csv_cols = [
        "asset_id",
        "title",
        "authors",
        "author_url",
        "platform",
        "source_url",
        "commit_or_uid",
        "license_spdx",
        "license_url",
        "split_config",
        "source_sha256",
        "changes_made",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_cols)
        writer.writeheader()
        for c in credits_sorted:
            writer.writerow({k: c.get(k, "") for k in csv_cols})
    file_checksums["ATTRIBUTION_CREDITS.csv"] = _sha256_bytes(csv_path.read_bytes())

    credits_json_path = args.output_dir / "ATTRIBUTION_CREDITS.json"
    credits_json_path.write_text(json.dumps(credits_sorted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    file_checksums["ATTRIBUTION_CREDITS.json"] = _sha256_bytes(credits_json_path.read_bytes())

    bib_text = _generate_citations_bib(
        args.dataset_repo_id,
        credits_sorted,
        discovery_stats["cobblemon"]["commit_sha"],
        discovery_stats["cobblemon"]["git_contributors"],
    )
    bib_path = args.output_dir / "CITATIONS.bib"
    bib_path.write_text(bib_text, encoding="utf-8")
    file_checksums["CITATIONS.bib"] = _sha256_bytes(bib_path.read_bytes())

    vis_audit_json_path = args.output_dir / "visual_identity_audit_report.json"
    vis_audit_json_path.write_text(
        json.dumps(visual_identity_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    file_checksums["visual_identity_audit_report.json"] = _sha256_bytes(vis_audit_json_path.read_bytes())

    manifest_sha256 = _sha256_text(json.dumps(file_checksums, sort_keys=True))

    print("[5/5] Writing AUTHORS_INDEX.md, Dataset Card README.md & dataset_manifest.json...")
    authors_md, readme_md = _generate_authors_index_and_readme(
        dataset_repo_id=args.dataset_repo_id,
        split_rows=split_rows,
        credits=credits_sorted,
        skipped_unsupported=skipped_unsupported,
        skipped_errors=skipped_errors,
        discovery_stats=discovery_stats,
        manifest_sha256=manifest_sha256,
        sheet_rel_paths=sheet_rel_paths,
        visual_identity_summary=visual_identity_summary,
    )
    (args.output_dir / "AUTHORS_INDEX.md").write_text(authors_md, encoding="utf-8")
    (args.output_dir / "README.md").write_text(readme_md, encoding="utf-8")

    manifest_payload = {
        "schema": "synccraft.human-artist-dataset-manifest/3.2",
        "dataset_repo_id": args.dataset_repo_id,
        "default_world_model_backbone": DEFAULT_WORLD_MODEL_BACKBONE,
        "generator_revision": GENERATOR_REVISION,
        "compiler_cache_version": COMPILER_CACHE_VERSION,
        "dataset_manifest_sha256": manifest_sha256,
        "total_verified_records": len(credits_sorted),
        "config_counts": {
            cfg: {"train": len(tr), "validation": len(va), "total": len(tr) + len(va)}
            for cfg, (tr, va) in split_rows.items()
        },
        "skipped_non_cuboid_unsupported": len(skipped_unsupported),
        "skipped_other_errors": len(skipped_errors),
        "source_discovery_stats": discovery_stats,
        "visual_identity_audit_summary": {
            k: v for k, v in visual_identity_summary.items() if k != "per_asset_metrics"
        },
        "file_sha256": file_checksums,
        "inspection_sheets": sheet_rel_paths,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dt = time.perf_counter() - t0
    print(
        f"DONE in {dt:.1f}s! Verified records: {len(credits_sorted):,} | "
        f"Non-cuboid skipped: {len(skipped_unsupported):,} | Other skipped: {len(skipped_errors):,}"
    )


if __name__ == "__main__":
    main()
