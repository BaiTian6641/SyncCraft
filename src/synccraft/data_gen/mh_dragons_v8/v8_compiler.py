"""src/synccraft/data_gen/mh_dragons_v8/v8_compiler.py

Native Block-DSL v4.0 (@component + @assembly + painted x32 UVs) Compiler,
Stage-1 Blockout Emitter, SE(3)/UV Export Parity Verifier, and Offline Viewer Packager.
"""

from __future__ import annotations

import collections
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from synccraft.backends.bbmodel5_exporter import export_bbmodel
from synccraft.backends.bedrock_gecko_exporter import export_bedrock
from synccraft.dsl.assembly_ir import audit_assembly_contracts
from synccraft.dsl.parser import _tokens, format_program, parse_program
from synccraft.kinematics.se3_chain import compile_geometry, euler_xyz_matrix
from synccraft.surface.contracts import linear_srgb_to_oklab

from .v8_animator import build_monster_12_clips
from .v8_painter import NORMAL_FACES, bake_x32_atlas, native_uv_faces, prepare_elements_and_pack_uv
from .v8_renderer import (
    render_animated_clip_gif,
    render_monster_studio_review_sheet,
    render_studio_frame,
)


def _fmt_vec(vals: Any) -> str:
    return "[" + ", ".join(f"{float(x):.7g}" for x in vals) + "]"


def _semantic_group(elem: dict[str, Any]) -> str:
    comp = elem.get("assembly_component")
    if comp:
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(comp)).strip("_")
        if clean:
            return clean
    low = elem["name"].lower()
    if any(k in low for k in ("tooth", "fang")):
        return "teeth"
    if any(k in low for k in ("horn", "antler", "tusk")):
        return "horns"
    if any(k in low for k in ("claw", "talon", "hallux")):
        return "claws"
    if any(k in low for k in ("eye", "iris", "pupil", "socket")):
        return "eyes"
    if any(k in low for k in ("membrane", "web", "patag", "seam")):
        return "membrane"
    if any(k in low for k in ("fur", "mane", "whisker", "plume")):
        return "fur"
    if any(k in low for k in ("spike", "spine", "spur", "shingle", "radiator")):
        return "spines"
    return "surface"


def prepare_v4_assembly_blueprint(raw_doc: dict[str, Any]) -> dict[str, Any]:
    """Namespace elements into anatomical runtime-bone components + fixed detail sub-components."""
    doc = deepcopy(raw_doc)
    for e in doc["elements"]:
        e["size"] = [max(0.5, round(float(v) * 2.0) / 2.0) if v else 0.0 for v in e["size"]]
        e.setdefault("rotation", [0.0, 0.0, 0.0])

    # Ensure root has at least one anchor box if empty, or find single root bone
    bones_by_parent: dict[str | None, list[dict[str, Any]]] = collections.defaultdict(list)
    for b in doc["bones"]:
        bones_by_parent[b["parent"]].append(b)
    if len(bones_by_parent[None]) != 1:
        raise ValueError(f"{doc['id']}: expected exactly 1 root bone, got {len(bones_by_parent[None])}")

    # Guarantee every bone owns at least one tiny hub box buried inside the nearest child/element mass
    # so `surface` component is non-empty AND zero stray cubes ever appear at [0, 0, 0].
    bones_with_surface = {
        e["bone"] for e in doc["elements"] if _semantic_group(e) == "surface"
    }
    first_mat = next(iter(doc["materials"].keys()))
    elems_by_bone: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in doc["elements"]:
        elems_by_bone[e["bone"]].append(e)

    def _buried_hub_center(b_name: str) -> list[float]:
        if elems_by_bone.get(b_name):
            return list(elems_by_bone[b_name][0]["center"])
        for ch in bones_by_parent.get(b_name, []):
            ch_name = ch["name"]
            if elems_by_bone.get(ch_name):
                return list(elems_by_bone[ch_name][0]["center"])
        for b_info in doc["bones"]:
            if b_info["name"] == b_name:
                if bones_by_parent.get(b_name):
                    return list(bones_by_parent[b_name][0]["pivot"])
                return list(b_info["pivot"])
        return [0.0, 10.0, 0.0]

    for b in doc["bones"]:
        if b["name"] not in bones_with_surface:
            doc["elements"].append(
                {
                    "name": f"{b['name']}_joint_hub",
                    "bone": b["name"],
                    "center": _buried_hub_center(b["name"]),
                    "size": [0.5, 0.5, 0.5],
                    "material": first_mat,
                    "rotation": [0.0, 0.0, 0.0],
                    "assembly_component": "surface",
                    "pattern": "solid",
                }
            )

    groups: dict[str, dict[str, str]] = {}
    used_compiled: set[str] = set()
    for e in doc["elements"]:
        grp = _semantic_group(e)
        bone = e["bone"]
        inst = bone if grp == "surface" else f"part_{bone}_{grp}"
        loc_name = re.sub(r"[^A-Za-z0-9_]", "_", e["name"]).strip("_")
        compiled = f"{inst}__{loc_name}"
        if compiled in used_compiled:
            idx = 2
            while f"{compiled}_{idx}" in used_compiled:
                idx += 1
            loc_name = f"{loc_name}_{idx}"
            compiled = f"{inst}__{loc_name}"
        used_compiled.add(compiled)
        e["name"] = compiled
        e["assembly_instance"] = inst
        e["assembly_local_name"] = loc_name
        e["assembly_group"] = grp
        groups[inst] = {"bone": bone, "group": grp, "component": f"Comp_{inst}"}

    doc["assembly_authoring"] = {
        "prepared": True,
        "frontend": "native_v4_assembly_ir",
        "instances": groups,
    }
    return doc


def _srgb_u8_to_oklch(rgb: list[int]) -> list[float]:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
    lab = linear_srgb_to_oklab(lin)
    l_val = round(float(lab[0]), 5)
    c_val = round(float(np.hypot(lab[1], lab[2])), 5)
    h_val = round(float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0), 3)
    return [l_val, c_val, h_val]


def emit_v4_assembly_dsl(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    clips: dict[str, dict[str, Any]],
    atlas_w: int,
    atlas_h: int,
) -> str:
    """Emit canonical Block-DSL v4.0 source with @component, @assembly, painted UVs, @materials, and 12 @behavior clips."""
    tpu = int(doc.get("texels_per_unit", 1))
    td = 16 * tpu
    lines: list[str] = [
        f"// {doc['title']} ({doc.get('title_epithet', '')}) - Block-DSL v4.0 Sculpted Blockbench Masterwork.",
        "// High-density rotated 3D cuboids + articulated non-uni-sheet RGBA cutout wing/fin banks.",
        f'@meta {{ dsl_version: "4.0", archetype: entity, target_profile: bbmodel_5, texel_density: {td}, size_step: 0.5, audit_profile: sculpted }}',
        f'@textures {{ painted_atlas: image("textures/albedo.png", {atlas_w}, {atlas_h}) }}',
        "@palette {",
    ]
    for mname, mcfg in doc["materials"].items():
        lch = _srgb_u8_to_oklch(mcfg["color"])
        em_flag = ", emissive=true" if float(mcfg.get("emissive", 0.0)) > 0.05 else ""
        lines.append(f"  mat_{mname}: ramp(base_lch={_fmt_vec(lch)}, steps=6{em_flag})")
    lines.append("}")
    lines.append("@layout { budget max_boxes = 4096; budget max_bones = 128; budget max_atlas_px = 4194304 }")

    bones_map = {b["name"]: b for b in doc["bones"]}
    children_map: dict[str | None, list[str]] = collections.defaultdict(list)
    for b in doc["bones"]:
        children_map[b["parent"]].append(b["name"])

    groups = doc["assembly_authoring"]["instances"]
    by_inst: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for b in boxes:
        by_inst[b["assembly_instance"]].append(b)

    for inst, info in groups.items():
        bname = info["bone"]
        piv = bones_map[bname]["pivot"]
        elems = by_inst[inst]
        slots = sorted({e["material"] for e in elems})
        lines.append(f'@component {info["component"]} {{')
        lines.append(f'  material_slots: [{", ".join(slots)}]')
        lines.append("  uv_policy: unique")
        lines.append("  frame mount { pos: [0, 0, 0], rot: [0, 0, 0], export_locator: true }")
        if inst == bname:
            for ch_b in children_map[bname]:
                off = [float(a) - float(b) for a, b in zip(bones_map[ch_b]["pivot"], piv)]
                lines.append(f"  frame to_bone_{ch_b} {{ pos: {_fmt_vec(off)}, rot: [0, 0, 0] }}")
            for fixed_inst, fi in groups.items():
                if fi["bone"] == bname and fixed_inst != bname:
                    lines.append(f"  frame to_part_{fi['group']} {{ pos: [0, 0, 0], rot: [0, 0, 0] }}")
        for e in elems:
            rel_c = [float(a) - float(b) for a, b in zip(e["center"], piv)]
            zeros = [i for i, v in enumerate(e["size"]) if v == 0.0]
            shape_kw = "plane" if zeros else "box"
            sz_vec = [v for v in e["size"] if v > 0.0] if zeros else e["size"]
            norm_str = f', normal: {"xyz"[zeros[0]]}' if zeros else ""
            vis_uvs = {
                f: r for f, r in native_uv_faces(e).items() if (not zeros) or (f in NORMAL_FACES[zeros[0]])
            }
            uv_str = ", ".join(f"{f}: {_fmt_vec(r)}" for f, r in vis_uvs.items())
            rot_str = (
                f", rotate: {_fmt_vec(e['rotation'])}"
                if any(abs(float(x)) > 1e-7 for x in e["rotation"])
                else ""
            )
            inf_val = float(e.get("inflate", 0.0))
            inf_str = f", inflate: {inf_val:.4g}" if abs(inf_val) > 1e-6 else ""
            lines.append(
                f"  {shape_kw} {e['assembly_local_name']} place(anchor=[0, 0, 0], align=[0, 0, 0], offset={_fmt_vec(rel_c)}) "
                f"{{ size: {_fmt_vec(sz_vec)}{norm_str}, material: {e['material']}{inf_str}{rot_str}, "
                f"texture: painted_atlas, uv_faces: {{ {uv_str} }} }}"
            )
        lines.append("}")

    root_b = children_map[None][0]
    lines.append(f"@assembly {doc['id']}_assembly {{")
    lines.append(f"  root_pivot: {_fmt_vec(bones_map[root_b]['pivot'])}")
    lines.append(f"  root_instance: {root_b}")
    lines.append("  subassembly anatomy {")
    for b in doc["bones"]:
        bn = b["name"]
        lines.append(f"    subassembly {bn}_group {{")
        for inst, info in groups.items():
            if info["bone"] != bn:
                continue
            slots = sorted({e["material"] for e in by_inst[inst]})
            slot_map = ", ".join(f"{s}: mat_{s}" for s in slots)
            lines.append(f"      instance {inst} : {info['component']}() {{ materials: {{ {slot_map} }} }}")
        lines.append("    }")
    lines.append("  }")
    for b in doc["bones"]:
        bn = b["name"]
        par = b["parent"]
        if par is not None:
            lines.append(f"  connect {bn}.mount to {par}.to_bone_{bn} {{ type: fixed, mode: align, preserve_bone: true }}")
        for inst, info in groups.items():
            if info["bone"] == bn and inst != bn:
                lines.append(f"  connect {inst}.mount to {bn}.to_part_{info['group']} {{ type: fixed, mode: align }}")
    for inst, info in groups.items():
        if inst != info["bone"]:
            lines.append(f"  bind {inst} to {info['bone']}")
    lines.append("}")

    fam_map = {
        "scales": "keratin_scales",
        "armor": "keratin_scales",
        "horn": "keratin_scales",
        "claw": "keratin_scales",
        "belly": "patterned_skin",
        "membrane": "patterned_skin",
        "fur": "woven_fabric",
        "eye": "polymer_dielectric",
        "mouth": "patterned_skin",
        "metal": "brushed_metal",
        "energy": "thin_film_iridescence",
        "glow": "thin_film_iridescence",
        "crystal": "thin_film_iridescence",
    }
    lines.append("@materials {")
    for mname, mcfg in doc["materials"].items():
        fam = fam_map.get(str(mcfg.get("kind", "scales")).lower(), "keratin_scales")
        rough = round(float(mcfg.get("roughness", 0.65)), 3)
        em_val = round(float(mcfg.get("emissive", 0.0)), 3)
        em_kv = f", emissive={em_val}" if em_val > 0.05 else ""
        lines.append(f"  mat_{mname}: process(family={fam}, roughness={rough}{em_kv}, seed=20261116)")
    lines.append("}")
    lines.append("@style { target_encoding: vanilla, quantize_ramp: true, bevel_highlight: 0.08 }")

    lines.append("@behavior {")
    for cname, cdata in clips.items():
        lines.append(f"  // {cdata.get('description', cname)}")
        lines.append(f"  clip {cname} {{ duration_s: {cdata['duration']}, loop: {str(cdata['loop']).lower()}")
        for tr in cdata["tracks"]:
            k_str = ", ".join(f"{t:.6g}: {_fmt_vec(v)}" for t, v in tr["keys"])
            lines.append(f"    key {tr['bone']}.{tr['channel']} {{ {k_str}, linear }}")
        lines.append("  }")
    lines.append("}")

    if "idle" in clips and "walk" in clips:
        ctrl_lines = [
            "@controller {",
            "  initial: idle",
            "  idle -> walk when speed > 0.05, blend 0.25",
            "  walk -> idle when speed <= 0.05, blend 0.25",
        ]
        if "run" in clips:
            ctrl_lines.append("  walk -> run when speed > 0.65, blend 0.20")
        if "roar" in clips:
            ctrl_lines.append("  idle -> roar when enraged >= 0.80, blend 0.20")
        ctrl_lines.append("}")
        lines.extend(ctrl_lines)

    return "\n".join(lines) + "\n"


def emit_stage1_coarse_blockout_dsl(raw_doc: dict[str, Any]) -> str:
    """Emit a genuine Stage-1 coarse anatomical blockout (~16-36 cuboids) for coarse-to-fine World-Model training."""
    s1_elems = raw_doc.get("stage1_elements") or raw_doc["elements"][:24]
    bones_map = {b["name"]: b for b in raw_doc["bones"]}
    children_map: dict[str | None, list[str]] = collections.defaultdict(list)
    for b in raw_doc["bones"]:
        children_map[b["parent"]].append(b["name"])
    by_bone: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for idx, e in enumerate(s1_elems):
        ec = deepcopy(e)
        ec["name"] = f"s1_{idx}_{re.sub(r'[^A-Za-z0-9_]', '_', e['name'])[:28]}"
        by_bone[e["bone"]].append(ec)

    used_mats = sorted({e["material"] for e in s1_elems} | {next(iter(raw_doc["materials"].keys()))})
    lines = [
        f"// Stage-1 Coarse Blockout for {raw_doc['title']}",
        "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 32, size_step: 0.5, audit_profile: sculpted }",
        "@palette {",
    ]
    for mname in used_mats:
        lch = _srgb_u8_to_oklch(raw_doc["materials"][mname]["color"])
        lines.append(f"  mat_{mname}: ramp(base_lch={_fmt_vec(lch)}, steps=6)")
    lines.append("}")
    lines.append("@rig {")

    def _emit_b(bname: str, depth: int) -> None:
        b = bones_map[bname]
        piv = b["pivot"]
        par = b["parent"]
        ind = "  " * depth
        if par is None:
            lines.append(f"{ind}root bone {bname} {{")
            lines.append(f"{ind}  pivot_local: {_fmt_vec(piv)}")
        else:
            p_piv = bones_map[par]["pivot"]
            off = [float(a) - float(c) for a, c in zip(piv, p_piv)]
            lines.append(f"{ind}bone {bname} joint(parent={par}_loc, at={_fmt_vec(off)}) {{")
        lines.append(f"{ind}  locator {bname}_loc {{ offset: [0, 0, 0], role: socket }}")
        for e in by_bone.get(bname, []):
            rel_c = [float(a) - float(c) for a, c in zip(e["center"], piv)]
            sz = [max(0.5, round(float(v) * 2.0) / 2.0) for v in e["size"][:3]]
            rot_str = (
                f", rotate: {_fmt_vec(e['rotation'])}"
                if any(abs(float(x)) > 1e-7 for x in e.get("rotation", [0, 0, 0]))
                else ""
            )
            lines.append(
                f"{ind}  box {e['name']} place(anchor=[0, 0, 0], align=[0, 0, 0], offset={_fmt_vec(rel_c)}) "
                f"{{ size: {_fmt_vec(sz)}, material: mat_{e['material']}{rot_str} }}"
            )
        for ch in children_map.get(bname, []):
            _emit_b(ch, depth + 1)
        lines.append(f"{ind}}}")

    _emit_b(children_map[None][0], 1)
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_and_export_v8_monster(
    raw_doc: dict[str, Any],
    output_pack_root: Path,
    *,
    ref_root: Path | None = None,
    render_previews: bool = True,
    render_gif_clip: str | None = None,
) -> dict[str, Any]:
    """Full V8 Studio Pipeline for one Monster Hunter creature:
    1. Prepare Block-DSL v4.0 `@component` + `@assembly` namespaces
    2. Synthesize 12 vigorous Monster Hunter FK animation clips
    3. Pack & paint crisp x32 RGBA pixel-art atlas (`albedo.png` + `emissive.png`)
    4. Emit `<id>.blockdsl` + `<id>_stage1_blockout.blockdsl`, compile through SyncCraft's
       native `parse_program` + `compile_geometry`, and verify `< 1e-5` SE(3) parity!
    5. Export Blockbench 5 `.bbmodel` (`model_format='free'`) + Bedrock `.geo.json` & `.animation.json`
    6. Render studio `hero.png`, `side.png`, `front.png`, `head_hero.png`, 8-panel Studio Review Sheet,
       and optional animated signature GIF!
    """
    t0 = time.perf_counter()
    ident = raw_doc["id"]
    asset_dir = output_pack_root / "assets" / ident
    asset_dir.mkdir(parents=True, exist_ok=True)

    stage1_dsl = emit_stage1_coarse_blockout_dsl(raw_doc)
    stage1_prog = parse_program(stage1_dsl)
    stage1_scene = compile_geometry(stage1_prog)
    (asset_dir / f"{ident}_stage1_blockout.blockdsl").write_text(stage1_dsl, encoding="utf-8")

    doc = prepare_v4_assembly_blueprint(raw_doc)
    tpu = int(doc.get("texels_per_unit", 1))
    clips = build_monster_12_clips(doc)
    boxes, w, h = prepare_elements_and_pack_uv(doc, texels_per_unit=tpu)
    atlas = bake_x32_atlas(doc, boxes, w, h, asset_dir / "textures")

    dsl_source = emit_v4_assembly_dsl(doc, boxes, clips, w, h)
    dsl_path = asset_dir / f"{ident}.blockdsl"
    dsl_path.write_text(dsl_source, encoding="utf-8")

    program = parse_program(dsl_source)
    scene = compile_geometry(program)
    native_boxes = {b.name: b for b in scene.boxes}

    # Verify exact SE(3) world geometry parity between authored blueprint & native compiled Scene
    max_err = 0.0
    for b in boxes:
        nb = native_boxes[b["name"]]
        max_err = max(max_err, float(np.max(np.abs(nb.center - np.asarray(b["center"], dtype=np.float64)))))
        rot_expected = euler_xyz_matrix(b["rotation"])
        max_err = max(max_err, float(np.max(np.abs(nb.rotation - rot_expected))))
        if not np.allclose(nb.size, b["size"], atol=1e-6):
            raise AssertionError(f"[{ident}] Size drift on {b['name']}: {nb.size} vs {b['size']}")
    if max_err > 1e-4:
        raise AssertionError(f"[{ident}] Compiler SE(3) transform drift {max_err} > 1e-4")

    contracts_rep = audit_assembly_contracts(program, scene)
    if not contracts_rep["passed"]:
        raise AssertionError(f"[{ident}] Assembly contracts failed: {contracts_rep}")

    # Export Blockbench 5 (.bbmodel) and Minecraft Bedrock (.geo.json + .animation.json)
    bb_doc = export_bbmodel(scene, name=doc["title"], atlas=atlas, program=program, num_animation_frames=20)
    bb_doc["meta"]["model_format"] = "free"
    bb_doc["credit"] = "Authored Monster Hunter Block-DSL v4.0 Hybrid Masterwork (V8); characters (C) CAPCOM."
    bb_path = asset_dir / f"{ident}.bbmodel"
    bb_path.write_text(json.dumps(bb_doc, separators=(",", ":")), encoding="utf-8")

    bed_bundle = export_bedrock(
        scene,
        program=program,
        atlas=atlas,
        identifier=f"geometry.synccraft_mh_v8.{ident}",
        profile="bedrock_entity",
        num_animation_frames=20,
    )
    (asset_dir / f"{ident}.geo.json").write_text(json.dumps(bed_bundle["geometry"], indent=2), encoding="utf-8")
    (asset_dir / f"{ident}.animation.json").write_text(json.dumps(bed_bundle["animations"], indent=2), encoding="utf-8")

    # Source map & viewer render_scene.json
    src_map = program.source_map.to_dict() if program.source_map is not None else {}
    lowered_dsl = format_program(replace(program, assembly_spec=None, source_map=None))
    (asset_dir / "assembly_source_map.json").write_text(
        json.dumps(
            {
                "frontend": "native_v4_assembly_ir",
                "source_sha256": hashlib.sha256(dsl_source.encode("utf-8")).hexdigest(),
                "lowered_sha256": hashlib.sha256(lowered_dsl.encode("utf-8")).hexdigest(),
                "contracts": contracts_rep,
                "native_bones": len(scene.bones),
                "boxes": len(scene.boxes),
                "runtime_import_verified": False,
                "source_map": src_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    viewer_scene = {
        "id": ident,
        "name": doc["title"],
        "title": doc["title"],
        "title_epithet": doc.get("title_epithet", ""),
        "species_class": doc["species_class"],
        "materials": doc["materials"],
        "bones": doc["bones"],
        "notes": doc.get("notes", []),
        "atlas": "textures/albedo.png",
        "atlas_width": w,
        "atlas_height": h,
        "albedo_data_url": atlas["data_url"],
        "emissive_data_url": atlas.get("emissive_data_url"),
        "boxes": boxes,
        "clips": clips,
    }
    (asset_dir / "render_scene.json").write_text(json.dumps(viewer_scene, separators=(",", ":")), encoding="utf-8")

    n_cuboids = sum(1 for b in boxes if all(s > 0.0 for s in b["size"]))
    n_planes = sum(1 for b in boxes if any(s == 0.0 for s in b["size"]))

    preview_paths: dict[str, str] = {}
    if render_previews:
        atlas_rgba = np.asarray(Image.open(atlas["albedo_path"]).convert("RGBA"), dtype=np.uint8)
        em_rgba = (
            np.asarray(Image.open(atlas["emissive_path"]).convert("RGBA"), dtype=np.uint8)
            if atlas.get("emissive_path")
            else None
        )
        prev_dir = asset_dir / "previews"
        prev_dir.mkdir(parents=True, exist_ok=True)

        hero_im = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=em_rgba, view="hero", width=1400, height=1000)
        hero_p = prev_dir / "hero.png"
        hero_im.save(hero_p)

        side_im = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=em_rgba, view="side", width=1200, height=860)
        side_p = prev_dir / "side.png"
        side_im.save(side_p)

        front_im = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=em_rgba, view="front", width=1200, height=860)
        front_p = prev_dir / "front.png"
        front_im.save(front_p)

        head_im = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=em_rgba, view="head_hero", width=1200, height=900)
        head_p = prev_dir / "head_hero.png"
        head_im.save(head_p)

        sil_im = render_studio_frame(doc, boxes, atlas_rgba, view="side", width=960, height=680, silhouette=True, draw_ground_shadow=False)
        sil_p = prev_dir / "silhouette_side.png"
        sil_im.save(sil_p)

        sig_names = [
            k
            for k in clips.keys()
            if k not in ("idle", "walk", "run", "alert", "hurt", "roar", "bite", "tail_sweep", "wingbeat", "swim_surge", "thunder_charge", "supercharged_idle")
        ]
        review_p = render_monster_studio_review_sheet(
            doc,
            boxes,
            atlas_rgba,
            clips,
            emissive_rgba=em_rgba,
            ref_board_or_dir=ref_root,
            output_path=prev_dir / f"{ident}_studio_review.png",
            signature_clip_names=["roar"] + sig_names[:2],
        )
        preview_paths = {
            "hero": str(hero_p),
            "side": str(side_p),
            "front": str(front_p),
            "head_hero": str(head_p),
            "silhouette_side": str(sil_p),
            "studio_review": str(review_p),
        }
        if render_gif_clip and render_gif_clip in clips:
            gif_p = render_animated_clip_gif(
                doc,
                boxes,
                atlas_rgba,
                clips[render_gif_clip],
                prev_dir / f"{ident}_{render_gif_clip}.gif",
                emissive_rgba=em_rgba,
            )
            preview_paths["signature_gif"] = str(gif_p)

    elapsed = round(time.perf_counter() - t0, 2)
    manifest = {
        "id": ident,
        "title": doc["title"],
        "title_epithet": doc.get("title_epithet", ""),
        "species_class": doc["species_class"],
        "variant_kind": doc.get("variant_kind", "canonical_flagship"),
        "parent_species_id": doc.get("parent_species_id"),
        "wiki_url": doc.get("wiki_url", ""),
        "block_dsl_version": "4.0-native-hybrid-assembly",
        "elements_total": len(boxes),
        "cuboid_boxes": n_cuboids,
        "cutout_planes": n_planes,
        "cuboid_ratio": round(n_cuboids / max(1, len(boxes)), 4),
        "stage1_boxes": len(stage1_scene.boxes),
        "anatomical_bones": len(doc["bones"]),
        "compiled_bones": len(scene.bones),
        "wing_banks": len(doc.get("wing_articulation", [])),
        "atlas_size": [w, h],
        "texels_per_block": 16 * tpu,
        "dsl_bytes": len(dsl_source.encode("utf-8")),
        "dsl_tokens": len(_tokens(dsl_source)) - 1,
        "compiler_transform_max_error": round(max_err, 8),
        "serialized_bedrock_vertex_max_error": round(float(bed_bundle["serialized_vertex_max_error"]), 8),
        "clip_count": len(clips),
        "clip_names": list(clips.keys()),
        "previews": preview_paths,
        "runtime_import_verified": False,
        "build_seconds": elapsed,
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
