"""Localhost HTTP Bridge & Deterministic Action Dispatcher for the Blockbench 5 Plugin (ADR 0013).

Exposes SyncCraft's morphology analyzer, 15-axis OBB SAT + biomechanical motion auditor,
Froude gait synthesizer, cross-rig retargeter, secondary-motion layerer, and non-destructive
`preserve_and_layer` painted-texture pipeline to Blockbench (`editors/blockbench-synccraft/`)
over a `127.0.0.1`-only JSON HTTP service.

Safety & Trustworthiness Invariants:
- Binds strictly to `127.0.0.1` unless explicitly overridden by the local user.
- Zero `eval` / `exec`; all `.bbmodel` payloads are parsed into typed Block-DSL ASTs.
- Every generated animation is re-audited AFTER `.bbmodel` keyframe serialization; unverified
  animations return `verified: False` and omit `animation` unless `allow_unverified=True` is
  explicitly requested for diagnostic inspection.
- `runtime_import_verified` remains `False` until exercised inside a live Blockbench GUI client.
"""

from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from typing import Any

import numpy as np

from synccraft import __version__
from synccraft.audit.sculpted_style_evaluator import evaluate_sculpted_style_anchor
from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.bbmodel5_exporter import _serialize_bbmodel_animations
from synccraft.dsl.ast_nodes import (
    BehaviorDecl,
    BehaviorStateDecl,
    ClipDecl,
    ClipLayerDecl,
    EffectTriggerDecl,
    Program,
    TrackDecl,
    validate_program,
)
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.gait_inversion import (
    retarget_clip_across_models,
    search_or_repair_gait,
)
from synccraft.kinematics.morphology import analyze_morphology
from synccraft.kinematics.musculoskeletal import audit_behavior_clip, evaluate_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading.preserve_and_layer import (
    preserve_and_layer_painted_texture,
    rebake_selected_faces,
    rgba_to_data_url,
)
from synccraft.shading.uv_packer import build_atlas

SUPPORTED_BRIDGE_ACTIONS: tuple[str, ...] = (
    "health",
    "analyze",
    "check",
    "generate_locomotion",
    "retarget",
    "secondary_motion",
    "texture_pass",
)


def _group_uuid_map_from_bbmodel(doc: dict[str, Any]) -> dict[str, str]:
    """Map sanitized & raw bone names to original Blockbench `Group.uuid` strings."""
    from synccraft.world.inverse_inference import _sanitize_ident

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for grp in doc.get("groups") or []:
        if not isinstance(grp, dict):
            continue
        uid = str(grp.get("uuid") or "")
        raw_name = str(grp.get("name") or "")
        if not uid or not raw_name:
            continue
        mapping.setdefault(raw_name, uid)
        mapping.setdefault(raw_name.lower(), uid)
        safe = _sanitize_ident(raw_name, "bone", used)
        mapping.setdefault(safe, uid)
    return mapping


def _remap_animation_uuids_to_bbmodel(anim_dict: dict[str, Any], bbmodel_doc: dict[str, Any]) -> dict[str, Any]:
    """Re-key serialized `animators` to match the live `Group.uuid`s of `bbmodel_doc`."""
    name_to_uid = _group_uuid_map_from_bbmodel(bbmodel_doc)
    old_animators = anim_dict.get("animators") or {}
    new_animators: dict[str, Any] = {}
    for key, val in old_animators.items():
        if key == "effects" or not isinstance(val, dict):
            new_animators[key] = val
            continue
        bname = str(val.get("name") or "")
        mapped_uid = name_to_uid.get(bname) or name_to_uid.get(bname.lower()) or key
        new_animators[mapped_uid] = val
    out = dict(anim_dict)
    out["animators"] = new_animators
    return out


def _ingest_bbmodel_bundle(doc: dict[str, Any], *, source_name: str = "blockbench_live.bbmodel") -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Validate workspace restrictions and decompile `.bbmodel` with face UVs & embedded RGBA."""
    from synccraft.world.artist_loop import (
        LOCAL_WORKSPACE_LICENSE_POLICY,
        build_unlabeled_workspace_provenance,
        ingest_artist_project_record,
    )
    from synccraft.world.inverse_inference import _decompile_bbmodel

    if not isinstance(doc, dict):
        raise ValueError("expected a JSON object for 'bbmodel'")
    # Run restriction check (`NoAI` / `tdm-reservation` guard) via `ingest_artist_project_record`
    prov = build_unlabeled_workspace_provenance(asset_id=source_name)
    ingest_artist_project_record(
        doc,
        format_kind="bbmodel",
        provenance=prov,
        license_policy=LOCAL_WORKSPACE_LICENSE_POLICY,
    )
    prog, uv_map, embedded_rgba, max_err = _decompile_bbmodel(doc)
    return prog, uv_map, embedded_rgba, max_err


def _serialize_and_reaudit_clip(
    program: Program,
    clip_name: str,
    target_bbmodel_doc: dict[str, Any],
    *,
    num_frames: int = 16,
    enforce_gait_contracts: bool = False,
    expected_gait_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Serialize `clip_name` to `.bbmodel` keyframes, remap UUIDs, and re-audit the quantized output."""
    scene = compile_geometry(program)
    asset_name = str(target_bbmodel_doc.get("name") or "synccraft_asset")
    anims = _serialize_bbmodel_animations(
        program,
        scene,
        asset_name=asset_name,
        num_animation_frames=max(8, int(num_frames)),
    )
    matching = [a for a in anims if a.get("name") == clip_name]
    if not matching:
        raise ValueError(f"failed to serialize clip {clip_name!r}")
    remapped_anim = _remap_animation_uuids_to_bbmodel(matching[0], target_bbmodel_doc)

    # Re-import serialized keyframes onto a copy of `program` and re-audit (AGENTS.md serialization rule)
    from synccraft.world.inverse_inference import decompile_bbmodel_animations_to_behavior

    merged_doc = dict(target_bbmodel_doc)
    merged_doc["animations"] = [remapped_anim]
    uid_to_bone: dict[str, str] = {}
    for uid, animator in (remapped_anim.get("animators") or {}).items():
        if uid != "effects" and isinstance(animator, dict) and animator.get("name"):
            uid_to_bone[str(uid)] = str(animator["name"])

    re_beh = decompile_bbmodel_animations_to_behavior(
        merged_doc,
        program,
        group_uuid_to_safe_bone=uid_to_bone,
    )
    if re_beh is None or not re_beh.clips:
        return remapped_anim, {"passed": False, "reason": "serialized_animation_empty"}, False

    re_prog = validate_program(replace(program, behavior=re_beh))
    re_clip_name = re_beh.clips[0].name
    reaudit = audit_behavior_clip(re_prog, re_clip_name, num_frames=max(8, int(num_frames)))
    serialized_ok = bool(reaudit.get("passed", False))
    if enforce_gait_contracts and expected_gait_audit is not None:
        serialized_ok = serialized_ok and bool(expected_gait_audit.get("passed", False))
    return remapped_anim, reaudit, serialized_ok


class BlockbenchBridgeDispatcher:
    """Stateless JSON action dispatcher backing `synccraft serve-blockbench`."""

    def dispatch(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = payload or {}
        act = action.strip().lstrip("/")
        if act == "health":
            return self.handle_health()
        if act == "analyze":
            return self.handle_analyze(body)
        if act == "check":
            return self.handle_check(body)
        if act == "generate_locomotion":
            return self.handle_generate_locomotion(body)
        if act == "retarget":
            return self.handle_retarget(body)
        if act == "secondary_motion":
            return self.handle_secondary_motion(body)
        if act == "texture_pass":
            return self.handle_texture_pass(body)
        raise ValueError(f"unknown bridge action {act!r}; expected one of {list(SUPPORTED_BRIDGE_ACTIONS)}")

    def handle_health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "synccraft_blockbench_bridge",
            "version": __version__,
            "actions": list(SUPPORTED_BRIDGE_ACTIONS),
            "runtime_import_verified": False,
        }

    def handle_analyze(self, body: dict[str, Any]) -> dict[str, Any]:
        doc = body.get("bbmodel")
        prog, _, embedded_rgba, max_err = _ingest_bbmodel_bundle(
            doc,
            source_name=str(body.get("name") or "analyze.bbmodel"),
        )
        scene = compile_geometry(prog)
        morph = analyze_morphology(prog, scene)
        static_audit = audit_scene(scene)
        style_report = evaluate_sculpted_style_anchor(prog, scene, atlas_rgba=embedded_rgba)
        return {
            "ok": True,
            "bones": len(scene.bones),
            "boxes": len(scene.boxes),
            "locators": len(scene.locators),
            "max_corner_reconstruction_error": round(float(max_err), 6),
            "morphology": morph.to_dict(),
            "static_audit": static_audit,
            "sculpted_style": style_report.to_dict(),
            "has_embedded_texture": embedded_rgba is not None,
            "animation_clips": [c.name for c in prog.behavior.clips] if prog.behavior else [],
            "runtime_import_verified": False,
        }

    def handle_check(self, body: dict[str, Any]) -> dict[str, Any]:
        doc = body.get("bbmodel")
        prog, _, _, _ = _ingest_bbmodel_bundle(doc, source_name=str(body.get("name") or "check.bbmodel"))
        scene = compile_geometry(prog)
        static_audit = audit_scene(scene)
        n_frames = max(8, int(body.get("num_frames", 16)))
        target_clip = body.get("clip_name")

        clip_reports: dict[str, Any] = {}
        if prog.behavior is not None:
            for clip in prog.behavior.clips:
                if target_clip and clip.name != target_clip:
                    continue
                aud = audit_behavior_clip(prog, clip.name, num_frames=n_frames)
                samples = evaluate_behavior_clip(prog, clip.name, num_frames=n_frames)
                offending_frames: list[dict[str, Any]] = []
                for fr in samples:
                    fr_aud = audit_scene(fr.scene, enforce_physical_contact=False)
                    if fr_aud["collision_pairs"]:
                        offending_frames.append(
                            {
                                "frame_index": fr.frame_index,
                                "time_s": round(float(fr.time_s), 4),
                                "collision_pairs": fr_aud["collision_pairs"][:5],
                            }
                        )
                clip_reports[clip.name] = {
                    **aud,
                    "offending_frames": offending_frames,
                }

        all_clips_ok = all(r.get("passed", False) for r in clip_reports.values()) if clip_reports else True
        return {
            "ok": True,
            "passed": bool(static_audit["passed"] and all_clips_ok),
            "static_audit": static_audit,
            "clips": clip_reports,
            "runtime_import_verified": False,
        }

    def handle_generate_locomotion(self, body: dict[str, Any]) -> dict[str, Any]:
        doc = body.get("bbmodel")
        prog, _, _, _ = _ingest_bbmodel_bundle(doc, source_name=str(body.get("name") or "locomotion.bbmodel"))
        scene = compile_geometry(prog)
        morph = analyze_morphology(prog, scene)
        if not morph.limbs:
            raise ValueError("cannot generate locomotion: @morphology detected 0 ground-contacting limbs")

        phrase = str(body.get("style_phrase") or "steady walk")
        clip_name = str(body.get("clip_name") or "walk")
        dur_s = max(0.4, float(body.get("duration_s", 1.6)))
        n_frames = max(8, int(body.get("num_frames", 16)))
        allow_unverified = bool(body.get("allow_unverified", False))

        from synccraft.world.proposer_tasks import style_phrase_to_gait_preset

        ledger = CostLedger()
        seed_gait = style_phrase_to_gait_preset(phrase, morph, gait_name=f"{clip_name}_gait")
        if "export_root_motion" in body:
            seed_gait = replace(seed_gait, export_root_motion=bool(body["export_root_motion"]))

        search_res = search_or_repair_gait(
            prog,
            seed_gait,
            duration_s=dur_s,
            clip_name=clip_name,
            num_audit_frames=n_frames,
            max_evals=int(body.get("max_evals", 8)),
            ledger=ledger,
        )
        work_prog = search_res.program

        # Attach optional footstep touchdown effects
        sound_fx = body.get("footstep_sound")
        particle_fx = body.get("footstep_particle")
        if sound_fx or particle_fx:
            eff = EffectTriggerDecl(
                trigger="touchdown",
                sound=str(sound_fx) if sound_fx else None,
                particle=str(particle_fx) if particle_fx else None,
            )
            clips_updated = tuple(
                replace(c, effects=(eff,)) if c.name == clip_name else c
                for c in work_prog.behavior.clips  # type: ignore[union-attr]
            )
            work_prog = validate_program(replace(work_prog, behavior=replace(work_prog.behavior, clips=clips_updated)))  # type: ignore[arg-type]

        anim_json, reaudit, reaudit_ok = _serialize_and_reaudit_clip(
            work_prog,
            clip_name,
            doc,
            num_frames=n_frames,
            enforce_gait_contracts=True,
            expected_gait_audit=search_res.audit_report,
        )
        ledger.record("reaudit_calls", 1)
        verified = bool(search_res.passed and reaudit_ok)

        return {
            "ok": True,
            "verified": verified,
            "animation": anim_json if (verified or allow_unverified) else None,
            "gait": search_res.to_dict()["gait"],
            "repair_steps": list(search_res.repair_steps),
            "audit": search_res.audit_report,
            "serialized_reaudit": reaudit,
            "ledger": ledger.to_dict(),
            "runtime_import_verified": False,
        }

    def handle_retarget(self, body: dict[str, Any]) -> dict[str, Any]:
        src_doc = body.get("source_bbmodel")
        dst_doc = body.get("target_bbmodel")
        clip_name = str(body.get("clip_name") or "")
        if not clip_name:
            raise ValueError("'clip_name' is required for retarget")
        new_name = str(body.get("new_clip_name") or clip_name)
        n_frames = max(8, int(body.get("num_frames", 16)))
        allow_unverified = bool(body.get("allow_unverified", False))

        src_prog, _, _, _ = _ingest_bbmodel_bundle(src_doc, source_name="retarget_src.bbmodel")
        dst_prog, _, _, _ = _ingest_bbmodel_bundle(dst_doc, source_name="retarget_dst.bbmodel")
        ledger = CostLedger()

        res = retarget_clip_across_models(
            src_prog,
            clip_name,
            dst_prog,
            new_clip_name=new_name,
            max_repair_evals=int(body.get("max_evals", 8)),
            ledger=ledger,
        )
        anim_json, reaudit, reaudit_ok = _serialize_and_reaudit_clip(
            res.program,
            new_name,
            dst_doc,
            num_frames=n_frames,
            enforce_gait_contracts=True,
            expected_gait_audit=res.audit_report,
        )
        ledger.record("reaudit_calls", 1)
        verified = bool(res.passed and reaudit_ok)
        return {
            "ok": True,
            "verified": verified,
            "animation": anim_json if (verified or allow_unverified) else None,
            "gait": res.to_dict()["gait"],
            "repair_steps": list(res.repair_steps),
            "audit": res.audit_report,
            "serialized_reaudit": reaudit,
            "ledger": ledger.to_dict(),
            "runtime_import_verified": False,
        }

    def handle_secondary_motion(self, body: dict[str, Any]) -> dict[str, Any]:
        doc = body.get("bbmodel")
        prog, _, _, _ = _ingest_bbmodel_bundle(doc, source_name="secondary_motion.bbmodel")
        scene = compile_geometry(prog)
        morph = analyze_morphology(prog, scene)
        kind = str(body.get("kind") or "breathe").lower()
        dur_s = max(0.5, float(body.get("duration_s", 2.0)))
        freq_hz = round(1.0 / dur_s, 6)
        sec_name = str(body.get("clip_name") or kind)
        base_clip_name = body.get("base_clip_name")
        layer_weight = float(body.get("layer_weight", 0.75))
        n_frames = max(8, int(body.get("num_frames", 16)))
        allow_unverified = bool(body.get("allow_unverified", False))

        root_name = prog.root.name
        tracks: list[TrackDecl] = []
        if kind == "tail_sway" and morph.tail:
            for idx_t, bname in enumerate(morph.tail):
                if bname == root_name:
                    continue
                tracks.append(
                    TrackDecl(
                        bone=bname,
                        law="wave",
                        axis="y",
                        amp_deg=round(4.0 + 1.2 * idx_t, 3),
                        freq_hz=freq_hz,
                        phase_deg=round(-32.0 * idx_t, 3),
                    )
                )
        elif kind == "wing_settle" and morph.wings:
            for wg in morph.wings:
                sign_w = 1.0 if wg.role.endswith("_l") else -1.0
                for idx_w, bname in enumerate(wg.chain[:3]):
                    if bname == root_name:
                        continue
                    tracks.append(
                        TrackDecl(
                            bone=bname,
                            law="wave",
                            axis="z",
                            amp_deg=round(sign_w * (3.0 + 0.8 * idx_w), 3),
                            freq_hz=freq_hz,
                            phase_deg=round(-25.0 * idx_w, 3),
                        )
                    )
        else:
            # Default `breathe` across spine / neck / non-limb children (or gentle root pitch)
            targets = [b for b in (*morph.spine, *morph.neck) if b != root_name]
            if targets:
                for idx_s, bname in enumerate(targets[:4]):
                    tracks.append(
                        TrackDecl(
                            bone=bname,
                            law="wave",
                            axis="x",
                            amp_deg=round(2.2 / max(1, len(targets[:4])), 3),
                            freq_hz=freq_hz,
                            phase_deg=round(-20.0 * idx_s, 3),
                        )
                    )
            else:
                tracks.append(
                    TrackDecl(
                        bone=root_name,
                        law="wave",
                        axis="x",
                        amp_deg=1.5,
                        freq_hz=freq_hz,
                        phase_deg=0.0,
                    )
                )

        sec_clip = ClipDecl(name=sec_name, duration_s=dur_s, loop=True, tracks=tuple(tracks))
        existing_clips = list(prog.behavior.clips) if prog.behavior else []
        existing_clips = [c for c in existing_clips if c.name != sec_name]
        existing_clips.append(sec_clip)

        export_clip_name = sec_name
        if base_clip_name:
            updated_list: list[ClipDecl] = []
            found_base = False
            for c in existing_clips:
                if c.name == str(base_clip_name):
                    found_base = True
                    new_layers = tuple([*getattr(c, "layers", ()), ClipLayerDecl(source_clip=sec_name, weight=layer_weight)])
                    updated_list.append(replace(c, layers=new_layers))
                else:
                    updated_list.append(c)
            if not found_base:
                raise ValueError(f"base_clip_name {base_clip_name!r} not found in model animations")
            existing_clips = updated_list
            export_clip_name = str(base_clip_name)

        states = tuple(prog.behavior.states) if prog.behavior else ()
        springs = tuple(prog.behavior.springs) if prog.behavior else ()
        beh = BehaviorDecl(clips=tuple(existing_clips), states=states, springs=springs)
        work_prog = validate_program(replace(prog, behavior=beh))

        aud = audit_behavior_clip(work_prog, export_clip_name, num_frames=n_frames)
        anim_json, reaudit, reaudit_ok = _serialize_and_reaudit_clip(
            work_prog,
            export_clip_name,
            doc,
            num_frames=n_frames,
        )
        verified = bool(aud.get("passed", False) and reaudit_ok)
        return {
            "ok": True,
            "verified": verified,
            "clip_name": export_clip_name,
            "animation": anim_json if (verified or allow_unverified) else None,
            "audit": aud,
            "serialized_reaudit": reaudit,
            "runtime_import_verified": False,
        }

    def handle_texture_pass(self, body: dict[str, Any]) -> dict[str, Any]:
        doc = body.get("bbmodel")
        prog, uv_map, embedded_rgba, _ = _ingest_bbmodel_bundle(doc, source_name="texture_pass.bbmodel")
        scene = compile_geometry(prog)
        mode = str(body.get("mode") or "preserve_and_layer").lower()

        if embedded_rgba is not None and mode in ("preserve_and_layer", "variant"):
            h_px, w_px = int(embedded_rgba.shape[0]), int(embedded_rgba.shape[1])
            atlas_meta = {"width": w_px, "height": h_px, "faces": uv_map}
            hue_rot = body.get("hue_rotate_deg")
            res = preserve_and_layer_painted_texture(
                prog,
                scene,
                atlas_meta,
                embedded_rgba,
                wear_strength=float(body.get("wear_amount", 0.35)),
                glow_strength=float(body.get("glow_amount", 0.25)),
                variant_hue_shift_deg=float(hue_rot) if hue_rot is not None else (0.0 if mode == "variant" else None),
                variant_chroma_scale=float(body.get("chroma_scale", 1.0)),
                variant_lightness_shift=float(body.get("lightness_offset", 0.0)),
            )
            layers_out = [
                {
                    "name": "synccraft_wear_overlay",
                    "role": "wear_layer",
                    "width": w_px,
                    "height": h_px,
                    "source": rgba_to_data_url(res["wear_layer_rgba"]),
                },
                {
                    "name": "synccraft_glow_overlay",
                    "role": "glow_layer",
                    "width": w_px,
                    "height": h_px,
                    "source": rgba_to_data_url(res["glow_layer_rgba"]),
                },
            ]
            blank_px = int(np.sum(res["blank_fill_rgba"][..., 3] > 0))
            if blank_px > 0:
                layers_out.append(
                    {
                        "name": "synccraft_blank_face_fill",
                        "role": "blank_fill_layer",
                        "width": w_px,
                        "height": h_px,
                        "source": rgba_to_data_url(res["blank_fill_rgba"]),
                    }
                )
            if res["variant_rgba"] is not None:
                layers_out.append(
                    {
                        "name": "synccraft_oklch_variant",
                        "role": "oklch_variant_copy",
                        "width": w_px,
                        "height": h_px,
                        "source": rgba_to_data_url(res["variant_rgba"]),
                    }
                )
            return {
                "ok": True,
                "mode": mode,
                "original_pixels_changed": res["original_pixels_changed"],
                "blank_pixels_filled": blank_px,
                "palette_profile": res["palette_profile"],
                "layers": layers_out,
                "runtime_import_verified": False,
            }

        if embedded_rgba is not None and mode == "rebake_faces":
            h_px, w_px = int(embedded_rgba.shape[0]), int(embedded_rgba.shape[1])
            atlas_meta = {"width": w_px, "height": h_px, "faces": uv_map}
            sel_boxes = [str(x) for x in (body.get("selected_boxes") or [])]
            sel_faces = body.get("selected_faces")
            if sel_faces:
                targets: list[str | tuple[str, str]] = [(b, str(f)) for b in sel_boxes for f in sel_faces]
            else:
                targets = list(sel_boxes)
            rebaked = rebake_selected_faces(prog, atlas_meta, embedded_rgba, targets)
            return {
                "ok": True,
                "mode": "rebake_faces",
                "updated_face_count": rebaked["updated_face_count"],
                "non_target_pixels_changed": rebaked["non_target_pixels_changed"],
                "layers": [
                    {
                        "name": "synccraft_selective_rebake",
                        "role": "selective_rebake_copy",
                        "width": w_px,
                        "height": h_px,
                        "source": rebaked["data_url"],
                    }
                ],
                "runtime_import_verified": False,
            }

        # Unpainted fallback: bake fresh sculpted RGBA atlas
        atlas, png_bytes = build_atlas(prog, scene)
        return {
            "ok": True,
            "mode": "fresh_sculpted_bake",
            "original_pixels_changed": 0,
            "layers": [
                {
                    "name": "synccraft_baked_atlas",
                    "role": "base_atlas",
                    "width": int(atlas["width"]),
                    "height": int(atlas["height"]),
                    "source": atlas["data_url"],
                }
            ],
            "atlas_faces": atlas["faces"],
            "png_byte_length": len(png_bytes),
            "runtime_import_verified": False,
        }


def make_blockbench_http_server(
    host: str = "127.0.0.1",
    port: int = 18765,
    dispatcher: BlockbenchBridgeDispatcher | None = None,
) -> ThreadingHTTPServer:
    """Create a `ThreadingHTTPServer` bound to `(host, port)` for the Blockbench 5 plugin."""
    disp = dispatcher or BlockbenchBridgeDispatcher()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].strip("/") or "health"
            if path != "health":
                self._send_json(404, {"ok": False, "error": f"GET only supports /health (got /{path})"})
                return
            self._send_json(200, disp.handle_health())

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].strip("/")
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                result = disp.dispatch(path, payload)
                self._send_json(200, result)
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )

    return ThreadingHTTPServer((host, int(port)), _Handler)


def serve_blockbench_bridge(host: str = "127.0.0.1", port: int = 18765) -> None:
    """Run the localhost Blockbench bridge until interrupted."""
    server = make_blockbench_http_server(host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    print(
        json.dumps(
            {
                "service": "synccraft_blockbench_bridge",
                "listening": f"http://{bound_host}:{bound_port}",
                "version": __version__,
                "actions": list(SUPPORTED_BRIDGE_ACTIONS),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
