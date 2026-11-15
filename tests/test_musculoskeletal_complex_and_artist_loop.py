"""Comprehensive verification for ADR 0006: Block-DSL v3.1, Nested Mirrors, Musculoskeletal Tendon/IK Physics,
Cobblemon CC-BY-NC Quarantine + Molang Decompiler, Closed-Loop Artist Assistant, and Apex Complex Entity Pack.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.common import reflected_cullfaces
from synccraft.data_gen import (
    ALL_APEX_FAMILIES,
    APEX_SEEN_FAMILIES,
    APEX_UNSEEN_FAMILIES,
    SEEN_STATIC_FAMILIES,
    UNSEEN_STATIC_FAMILIES,
    build_apex_complex_program,
    generate_apex_complex_dataset,
)
from synccraft.dsl import (
    Attach,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    DSLValidationError,
    IKChainDecl,
    IKDriveDecl,
    KinematicsDecl,
    Locator,
    LookChainDecl,
    MaterialDecl,
    MaterialModDecl,
    Meta,
    MuscleDecl,
    MuscleDriveDecl,
    Palette,
    Place,
    Program,
    StyleDecl,
    TendonCoupling,
    TrackDecl,
    UnsupportedFeature,
    format_program,
    parse_program,
    validate_program,
)
from synccraft.kinematics import compile_geometry, compile_posed_geometry
from synccraft.kinematics.musculoskeletal import (
    audit_behavior_clip,
    evaluate_behavior_clip,
    evaluate_muscles_at_activations,
    solve_look_chain,
)
from synccraft.surface.contracts import MaterialProgram, StyleConfig, WorldState
from synccraft.surface.domain import build_surface_domain, initialize_surface_state
from synccraft.world import (
    ActivateBehaviorStateAction,
    AssetProvenance,
    ExecuteBehaviorClipAction,
    LicenseGateError,
    action_from_dict,
    analyze_artist_project,
    build_progressive_from_scratch_trajectory,
    build_quarantined_research_nc_dataset,
    decompile_bedrock_animation_to_behavior,
    export_artist_project_bundle,
    ingest_external_asset,
    run_looped_artist_enhancement,
    step_world,
    verify_asset_provenance,
)


class TestBlockDSLv31AndNestedMirrors(unittest.TestCase):
    """Verify Block-DSL v3.1 grammar, validation bounds, and arbitrary-depth nested mirror_x O(3) parity."""

    def test_v31_full_dsl_parse_format_roundtrip(self) -> None:
        dsl_text = """
@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16 }
@palette {
  hide: ramp(base_lch=[0.45, 0.14, 28.0], steps=6, hue_shift=[-8.0, 10.0], emissive=false)
  glow: ramp(base_lch=[0.8, 0.2, 180.0], steps=4, hue_shift=[0.0, 0.0], emissive=true)
}
@rig {
  root bone root {
    pivot_local: [0.0, 12.0, 0.0]
    box torso { size: [8, 6, 10], material: hide, inflate: 0.0 }
    locator seat { offset: [0.0, 4.0, 0.0], role: rider_seat }
    bone arm_l attach(parent_box=torso, anchor=[1.0, 0.2, 0.0], align=[-1.0, 0.0, 0.0], offset=[0.0, 0.0, 0.0], detached=false) mirror_x(arm_r) {
      rest_euler: [0.0, 0.0, 10.0]
      joint_limits: [[-30.0, 30.0], [-25.0, 25.0], [-60.0, 60.0]]
      box humerus { size: [6, 2, 3], material: hide, inflate: 0.0, cullface: [west] }
      bone digit_in attach(parent_box=humerus, anchor=[0.5, 0.0, -1.0], align=[0.0, 0.0, 1.0], offset=[0.0, 0.0, 0.0], detached=false) mirror_x(digit_out) {
        rest_euler: [-4.0, 0.0, 0.0]
        joint_limits: [[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]]
        box phalanx { size: [1, 1, 5], material: glow, inflate: 0.0, cullface: [east] }
        locator tip_vfx { offset: [0.0, 0.0, -2.5], role: vfx_emitter }
      }
    }
    bone thigh_l attach(parent_box=torso, anchor=[1.0, -0.3, 0.0], align=[-1.0, 0.0, 0.0], offset=[0.0, 0.0, 0.0], detached=false) mirror_x(thigh_r) {
      rest_euler: [6.0, 0.0, 0.0]
      joint_limits: [[-65.0, 65.0], [-20.0, 20.0], [-20.0, 20.0]]
      box femur { size: [3, 5, 3], material: hide, inflate: 0.0 }
      bone shank_l attach(parent_box=femur, anchor=[0.0, -1.0, 0.0], align=[0.0, 1.0, 0.0], offset=[0.0, 0.0, 0.0], detached=false) {
        rest_euler: [-12.0, 0.0, 0.0]
        joint_limits: [[-75.0, 55.0], [-15.0, 15.0], [-15.0, 15.0]]
        box tibia { size: [2, 5, 2], material: hide, inflate: 0.0 }
        locator sole_l { offset: [0.0, -5.0, 0.0], role: ik_effector }
      }
    }
  }
}
@materials {
  hide: process(family=keratin_scales, roughness=0.34, metalness=0.0, coating=0.0, ior=1.5, scale=1.0, seed=101, anisotropy=1.6, subsurface=0.25, wear=0.4, weld_group=body_skin)
  glow: process(family=thin_film_iridescence, roughness=0.18, metalness=0.0, coating=0.0, ior=1.5, scale=1.0, seed=102, emissive=1.2, film_nm=440.0)
}
@style { target_encoding: labpbr_1_3, quantize_ramp: false, bevel_highlight: 0.12, view_deg: 22.0 }
@muscles {
  muscle flap_l { origin_bone: root, insertion_bone: arm_l, action_axis: z, moment_arm_deg: -18.0, tendon_coupling: [digit_in, x, -0.4], belly_box: humerus, bulge_inflate: 0.08 }
  muscle flap_r { origin_bone: root, insertion_bone: arm_r, action_axis: z, moment_arm_deg: 18.0, tendon_coupling: [arm_r__digit_in, x, -0.4], belly_box: arm_r__humerus, bulge_inflate: 0.08 }
}
@kinematics {
  ik_chain leg_l { upper: thigh_l, lower: shank_l, effector: [0.0, -5.0, 0.0], hinge_axis: x }
  ik_chain leg_r { upper: thigh_r, lower: thigh_r__shank_l, effector: [0.0, -5.0, 0.0], hinge_axis: x }
  look_chain gaze { bones: [arm_l], weight_decay: 0.8 }
}
@behavior {
  clip patrol {
    duration_s: 1.6, loop: true
    track arm_l: wave(axis=y, amp_deg=3.0, freq_hz=0.625, phase_deg=0.0, bias_deg=0.0)
    muscle_drive flap_l: activation(peak=0.6, freq_hz=0.625, phase_deg=0.0, baseline=0.05)
    muscle_drive flap_r: activation(peak=0.6, freq_hz=0.625, phase_deg=0.0, baseline=0.05)
    ik_drive leg_l: foot_cycle(stride=1.4, lift=0.7, phase_deg=0.0)
    ik_drive leg_r: foot_cycle(stride=1.4, lift=0.7, phase_deg=180.0)
  }
  state enraged {
    pose_bias arm_l: [-3.0, 0.0, 0.0]
    muscle_tone flap_l: 0.5
    material_mod glow: { emissive: 1.9, moisture: 0.2, wear_delta: 0.03 }
  }
}
"""
        prog1 = parse_program(dsl_text)
        formatted1 = format_program(prog1)
        prog2 = parse_program(formatted1)
        formatted2 = format_program(prog2)
        self.assertEqual(formatted1, formatted2)
        self.assertEqual(prog1, prog2)

        scene = compile_geometry(prog1)
        # Verify 2-level nested mirror O(3) parity: det(R) == +1 on all bones, cullface parity toggles twice!
        for bone in scene.bones:
            self.assertAlmostEqual(float(np.linalg.det(bone.rotation)), 1.0, places=9)

        box_by_name = {b.name: b for b in scene.boxes}
        self.assertFalse(box_by_name["phalanx"].mirrored)
        self.assertTrue(box_by_name["digit_out__phalanx"].mirrored)
        self.assertTrue(box_by_name["arm_r__phalanx"].mirrored)
        # Doubly mirrored node toggles back to mirrored=False and restores original east cullface via reflected_cullfaces!
        self.assertFalse(box_by_name["arm_r__digit_out__phalanx"].mirrored)
        self.assertEqual(reflected_cullfaces(box_by_name["phalanx"]), ("east",))
        self.assertEqual(reflected_cullfaces(box_by_name["digit_out__phalanx"]), ("west",))
        self.assertEqual(reflected_cullfaces(box_by_name["arm_r__phalanx"]), ("west",))
        self.assertEqual(reflected_cullfaces(box_by_name["arm_r__digit_out__phalanx"]), ("east",))

        # Verify SurfaceDomain symmetry_pairs captures nested mirror pairs
        mp = MaterialProgram.from_program(prog1)
        dom = build_surface_domain(prog1, mp, subdivisions_per_unit=1)
        self.assertGreater(len(dom.symmetry_pairs), 0)

    def test_three_level_nested_mirror_and_exponential_bomb_guard(self) -> None:
        # 3-level nested mirror compiles cleanly (2^3 = 8 leaf branches)
        leaf = Bone("lvl3_a", attach=Attach("b2", anchor=(0.5, 0.0, 1.0), align=(0.0, 0.0, -1.0)), mirror_name="lvl3_b", items=[Box("b3", (1, 1, 2), "m")])
        mid = Bone("lvl2_a", attach=Attach("b1", anchor=(0.5, 1.0, 0.0), align=(0.0, -1.0, 0.0)), mirror_name="lvl2_b", items=[Box("b2", (4, 2, 4), "m"), leaf])
        outer = Bone("lvl1_a", attach=Attach("b0", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)), mirror_name="lvl1_b", items=[Box("b1", (6, 2, 6), "m"), mid])
        root = Bone("root", items=[Box("b0", (6, 4, 6), "m"), outer])
        prog_3 = validate_program(Program(Meta("entity", "bbmodel_5", 16), [Palette("m", (0.5, 0.1, 40.0), steps=4)], root))
        sc_3 = compile_geometry(prog_3)
        triple_box = next(b for b in sc_3.boxes if b.name == "lvl1_b__lvl2_b__lvl3_b__b3")
        self.assertTrue(triple_box.mirrored)  # 3 reflections: False -> True -> False -> True
        self.assertAlmostEqual(float(np.linalg.det(triple_box.rotation)), 1.0, places=9)

        # 15-level nested mirror bomb (2^15 = 32,768 > 8192) must be rejected before expansion
        cur = Bone("n_14", attach=Attach("bx_13", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)), mirror_name="m_14", items=[Box("bx_14", (1, 1, 1), "m")])
        for d in range(13, -1, -1):
            p_bx = "root_bx" if d == 0 else f"bx_{d - 1}"
            cur = Bone(
                f"n_{d}",
                attach=Attach(p_bx, anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)),
                mirror_name=f"m_{d}",
                items=[Box(f"bx_{d}", (1, 1, 1), "m"), cur],
            )
        bomb_prog = Program(Meta("entity", "bbmodel_5", 16), [Palette("m", (0.5, 0.1, 40.0), steps=4)], Bone("root", items=[Box("root_bx", (2, 2, 2), "m"), cur]))
        with self.assertRaises(DSLValidationError):
            validate_program(bomb_prog)

    def test_validation_rejects_invalid_v31_contracts_and_preserves_unsupported_guards(self) -> None:
        # Invalid locator role
        with self.assertRaises(DSLValidationError):
            validate_program(
                Program(
                    Meta("entity", "bbmodel_5", 16),
                    [Palette("m", (0.5, 0.1, 40.0), steps=4)],
                    Bone("root", items=[Box("b", (2, 2, 2), "m"), Locator("bad", role="teleport_beacon")]),
                )
            )
        # Inverted joint_limits (lo > hi)
        with self.assertRaises(DSLValidationError):
            validate_program(
                Program(
                    Meta("entity", "bbmodel_5", 16),
                    [Palette("m", (0.5, 0.1, 40.0), steps=4)],
                    Bone("root", joint_limits=((20.0, -10.0), (-10.0, 10.0), (-10.0, 10.0)), items=[Box("b", (2, 2, 2), "m")]),
                )
            )
        # Protected unsupported features (@shade, @motion, @ergonomics, extrude) still fail explicitly
        for forbidden_snippet in (
            "@shade { ambient: 0.5 }",
            "@motion { walk: 1 }",
            "@ergonomics { grip: 1 }",
            "@rig { root bone r { pivot_local: [0, 0, 0] extrude blade { size: [2, 2, 1] } } }",
        ):
            with self.assertRaises(UnsupportedFeature):
                if forbidden_snippet.startswith("@rig"):
                    parse_program(
                        "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16 }\n"
                        "@palette { m: ramp(base_lch=[0.5, 0.1, 40.0], steps=4, hue_shift=[0.0, 0.0], emissive=false) }\n"
                        f"{forbidden_snippet}\n"
                    )
                else:
                    parse_program(
                        "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16 }\n"
                        "@palette { m: ramp(base_lch=[0.5, 0.1, 40.0], steps=4, hue_shift=[0.0, 0.0], emissive=false) }\n"
                        "@rig { root bone r { pivot_local: [0, 0, 0] box b { size: [2, 2, 2], material: m, inflate: 0.0 } } }\n"
                        f"{forbidden_snippet}\n"
                    )


class TestMusculoskeletalTendonAndClosedLoopIK(unittest.TestCase):
    """Verify Hill-type agonist-antagonist co-contraction, tendon pantographs, belly bulge, and <0.02u stance IK."""

    def _build_antagonist_arm_rig(self) -> Program:
        wrist = Bone(
            "wrist",
            attach=Attach("forearm_box", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
            joint_limits=((-60.0, 60.0), (-15.0, 15.0), (-15.0, 15.0)),
            items=[Box("hand_box", (2, 3, 2), "skin")],
        )
        elbow = Bone(
            "elbow",
            attach=Attach("upper_box", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
            joint_limits=((-75.0, 75.0), (-15.0, 15.0), (-15.0, 15.0)),
            items=[Box("forearm_box", (2, 5, 2), "skin"), wrist],
        )
        root = Bone(
            "shoulder",
            pivot_local=(0.0, 14.0, 0.0),
            items=[
                Box("upper_box", (3, 6, 3), "skin"),
                Box("bicep_belly", (2, 4, 1), "muscle_mat", place=Place("upper_box", anchor=(0.0, 0.0, 1.0), align=(0.0, 0.0, -1.0))),
                Box("tricep_belly", (2, 4, 1), "muscle_mat", place=Place("upper_box", anchor=(0.0, 0.0, -1.0), align=(0.0, 0.0, 1.0))),
                elbow,
            ],
        )
        muscles = (
            MuscleDecl("bicep_flexor", "shoulder", "elbow", "x", 32.0, TendonCoupling("wrist", "x", 0.50), "bicep_belly", 0.14),
            MuscleDecl("tricep_extensor", "shoulder", "elbow", "x", -32.0, None, "tricep_belly", 0.14),
        )
        return validate_program(
            Program(
                Meta("entity", "bbmodel_5", 16),
                [Palette("skin", (0.62, 0.09, 48.0), steps=5), Palette("muscle_mat", (0.48, 0.16, 22.0), steps=5)],
                root,
                muscles=muscles,
                kinematics=KinematicsDecl(look_chains=(LookChainDecl("arm_aim", ("elbow", "wrist"), 0.75),)),
            )
        )

    def test_agonist_antagonist_cocontraction_stiffness_and_isochoric_bulge(self) -> None:
        prog = self._build_antagonist_arm_rig()

        # 1. Pure flexion: elbow flexes > 0 deg, coupled wrist flexes by 0.5x pantograph, bicep bulges
        flex_res = evaluate_muscles_at_activations(prog, {"bicep_flexor": 0.85, "tricep_extensor": 0.0})
        elbow_rx = flex_res.pose_euler_deltas["elbow"][0]
        wrist_rx = flex_res.pose_euler_deltas["wrist"][0]
        self.assertGreater(elbow_rx, 15.0)
        self.assertAlmostEqual(wrist_rx, 0.50 * elbow_rx, places=4)
        self.assertGreater(flex_res.box_inflate_deltas["bicep_belly"], 0.05)
        self.assertEqual(flex_res.box_inflate_deltas.get("tricep_belly", 0.0), 0.0)

        # Integer box sizes remain strictly immutable (Box-UV safe!)
        bicep_box = next(b for b in flex_res.scene.boxes if b.name == "bicep_belly")
        np.testing.assert_array_equal(bicep_box.size, np.array([2.0, 4.0, 1.0]))
        self.assertGreater(bicep_box.inflate, 0.05)

        # 2. Symmetric isometric co-contraction: net angle cancels to 0, stiffness increases, BOTH bellies bulge!
        coco_res = evaluate_muscles_at_activations(prog, {"bicep_flexor": 0.85, "tricep_extensor": 0.85})
        self.assertAlmostEqual(coco_res.pose_euler_deltas.get("elbow", (0.0, 0.0, 0.0))[0], 0.0, places=6)
        self.assertGreater(coco_res.joint_stiffness["elbow"], flex_res.joint_stiffness["elbow"] * 1.4)
        self.assertGreater(coco_res.box_inflate_deltas["bicep_belly"], 0.05)
        self.assertGreater(coco_res.box_inflate_deltas["tricep_belly"], 0.05)

        # 3. Look chain converges monotonically toward target (both downward -y arm aim and forward -z head gaze)
        rest_sc = compile_geometry(prog)
        rest_tip = next(b for b in rest_sc.boxes if b.name == "hand_box").center
        target = np.array([2.0, 5.0, 6.0], dtype=np.float64)
        look_deltas = solve_look_chain(prog, "arm_aim", target, forward_axis="-y")
        posed_sc = compile_posed_geometry(prog, look_deltas)
        posed_tip = next(b for b in posed_sc.boxes if b.name == "hand_box").center
        elbow_piv = next(b for b in rest_sc.bones if b.name == "elbow").pivot_world
        v_rest = (rest_tip - elbow_piv) / np.linalg.norm(rest_tip - elbow_piv)
        v_posed = (posed_tip - elbow_piv) / np.linalg.norm(posed_tip - elbow_piv)
        v_targ = (target - elbow_piv) / np.linalg.norm(target - elbow_piv)
        self.assertGreater(float(np.dot(v_posed, v_targ)), float(np.dot(v_rest, v_targ)))

        # Also check +z forward 3-bone cervical head/neck gaze on an Apex wyvern
        wyvern = build_apex_complex_program("apex_flying_wyvern", seed=101)
        w_rest = compile_geometry(wyvern)
        gaze_target = np.array([10.0, 28.0, 35.0], dtype=np.float64)
        w_look = solve_look_chain(wyvern, "cervical_gaze", gaze_target, forward_axis="+z")
        w_posed = compile_posed_geometry(wyvern, w_look)
        neck_piv = next(b for b in w_rest.bones if b.name == "neck_1").pivot_world
        head_rest_dir = next(b for b in w_rest.bones if b.name == "wyvern_head").rotation[:, 2]
        head_posed_dir = next(b for b in w_posed.bones if b.name == "wyvern_head").rotation[:, 2]
        gaze_dir = (gaze_target - neck_piv) / np.linalg.norm(gaze_target - neck_piv)
        self.assertGreater(float(np.dot(head_posed_dir, gaze_dir)), float(np.dot(head_rest_dir, gaze_dir)))

    def test_world_behavior_actions_and_serialization_roundtrip(self) -> None:
        prog = build_apex_complex_program("fanged_thunder_wyvern", seed=101)
        mp = MaterialProgram.from_program(prog, seed=101)
        dom = build_surface_domain(prog, mp, subdivisions_per_unit=1)
        ws = WorldState(
            program=prog,
            material_program=mp,
            surface_state=initialize_surface_state(dom),
            style_config=StyleConfig.from_program(prog),
        )

        act_clip = ExecuteBehaviorClipAction(clip_name="quadruped_prowl_trot", cycles=2, num_frames_per_cycle=8)
        self.assertEqual(action_from_dict(act_clip.to_dict()), act_clip)
        res_clip = step_world(ws, act_clip, dt=1.4)
        self.assertTrue(res_clip.swept_audit_passed)
        self.assertGreater(res_clip.total_sliding_wear_added, 0.0)
        self.assertLess(res_clip.ik_metadata["clip_audit"]["max_stance_slip_units"], 0.02)

        act_state = ActivateBehaviorStateAction(state_name="fulgur_overcharge_state", substeps=6)
        self.assertEqual(action_from_dict(act_state.to_dict()), act_state)
        res_state = step_world(res_clip.next_state, act_state, dt=0.5)
        self.assertTrue(res_state.swept_audit_passed)
        self.assertGreaterEqual(
            res_state.next_state.material_program.materials["fulgur_core"].emissive_strength,
            1.8,
        )


class TestCobblemonCCBYNCQuarantineAndDecompilers(unittest.TestCase):
    """Verify CC-BY-NC quarantine policy, Bedrock/Cobblemon .geo.json locators, and zero-eval Molang decompilation."""

    def _sample_cobblemon_geojson_and_anim(self) -> tuple[dict, dict]:
        geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {
                        "identifier": "geometry.synccraft.cleanroom_sproutling",
                        "texture_width": 64,
                        "texture_height": 64,
                    },
                    "bones": [
                        {
                            "name": "body_root",
                            "pivot": [0, 6, 0],
                            "cubes": [{"origin": [-4, 2, -4], "size": [8, 8, 8], "uv": [0, 0]}],
                            "locators": {
                                "seat_1": [0, 11, 0],
                                "target_hitbox": {"offset": [0, 6, 0], "rotation": [0, 0, 0]},
                            },
                        },
                        {
                            "name": "head_group",
                            "parent": "body_root",
                            "pivot": [0, 10, 0],
                            "rotation": [5, 0, 0],
                            "cubes": [{"origin": [-3, 10, -3], "size": [6, 5, 6], "uv": [0, 16]}],
                            "locators": {
                                "head_camera_focus": [0, 13, -3],
                                "solar_beam_emitter": [0, 12, -4],
                            },
                        },
                        {
                            "name": "leaf_ear_l",
                            "parent": "head_group",
                            "pivot": [3, 15, 0],
                            "rotation": [0, 0, 15],
                            "cubes": [{"origin": [2, 15, -1], "size": [2, 5, 2], "uv": [24, 16]}],
                        },
                    ],
                }
            ],
        }
        anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.sproutling.ground_idle": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "head_group": {
                            "rotation": [
                                "math.sin(q.anim_time * 180) * 6 + 2",
                                "math.cos(query.anim_time * 180 - 45) * 8",
                                0,
                            ]
                        },
                        "leaf_ear_l": {
                            "rotation": {
                                "0.0": [0, 0, 0],
                                "1.0": [0, 0, 12],
                                "2.0": [0, 0, 0],
                            }
                        },
                    },
                }
            },
        }
        return geo, anim

    def test_cc_by_nc_quarantine_and_mojang_ban(self) -> None:
        geo, anim = self._sample_cobblemon_geojson_and_anim()
        nc_prov = AssetProvenance(
            asset_id="community_creature_01",
            source_url="https://gitlab.com/cable-mc/cobblemon-assets-example/sproutling",
            author="CommunityArtist",
            license_spdx="CC-BY-NC-4.0",
            attribution_notice="CC-BY-NC-4.0 CommunityArtist (non-commercial research split)",
        )

        # 1. Default clean_room_permissive policy MUST reject CC-BY-NC-4.0
        with self.assertRaises(LicenseGateError):
            ingest_external_asset(geo, "bedrock_geojson", nc_prov, license_policy="clean_room_permissive")

        # 2. Missing attribution_notice under research_nc MUST be rejected
        no_attr = AssetProvenance(
            asset_id="community_creature_01",
            source_url="https://example.org/sprout",
            author="CommunityArtist",
            license_spdx="CC-BY-NC-4.0",
            attribution_notice="",
        )
        with self.assertRaises(LicenseGateError):
            verify_asset_provenance(no_attr, license_policy="research_nc")

        # 3. Mojang / proprietary tokens MUST be rejected even under research_nc
        mojang_prov = AssetProvenance(
            asset_id="warden_rip",
            source_url="https://example.org/mojang_jar_extract",
            author="Extractor",
            license_spdx="CC0-1.0",
        )
        with self.assertRaises(LicenseGateError):
            verify_asset_provenance(mojang_prov, license_policy="research_nc")

        # 4. Quarantined research_nc ingestion succeeds, marks commercial_weights_eligible=False,
        #    preserves 3D vertices + semantic locators, and decompiles Molang animations without eval!
        rec = ingest_external_asset(
            geo,
            "bedrock_geojson",
            nc_prov,
            animation_payload=anim,
            license_policy="research_nc",
        )
        self.assertEqual(rec.license_track, "research_nc")
        self.assertFalse(rec.commercial_weights_eligible)
        self.assertLess(rec.vertex_reconstruction_error, 1e-5)

        sc = compile_geometry(rec.program)
        roles = {loc.role for loc in sc.locators}
        self.assertIn("rider_seat", roles)
        self.assertIn("hitbox", roles)
        self.assertIn("camera_focus", roles)
        self.assertIn("vfx_emitter", roles)

        self.assertIsNotNone(rec.program.behavior)
        assert rec.program.behavior is not None
        self.assertEqual(len(rec.program.behavior.clips), 1)
        self.assertGreaterEqual(len(rec.program.behavior.clips[0].tracks), 2)

    def test_molang_decompiler_ignores_adversarial_code_strings_without_eval(self) -> None:
        geo, _ = self._sample_cobblemon_geojson_and_anim()
        adversarial_anim = {
            "animations": {
                "animation.adversarial.test": {
                    "loop": True,
                    "animation_length": 1.0,
                    "bones": {
                        "head_group": {
                            "rotation": [
                                "__import__('os').system('touch /tmp/synccraft_should_never_exist')",
                                "eval('1+1')",
                                "math.sin(q.anim_time * 360) * 5",
                            ]
                        }
                    },
                }
            }
        }
        prov = AssetProvenance(
            asset_id="safe_test",
            source_url="https://example.org/safe",
            author="CleanAuthor",
            license_spdx="CC0-1.0",
        )
        rec = ingest_external_asset(geo, "bedrock_geojson", prov, animation_payload=adversarial_anim)
        self.assertFalse(Path("/tmp/synccraft_should_never_exist").exists())
        assert rec.program.behavior is not None
        tracks = rec.program.behavior.clips[0].tracks
        # Only the valid math.sin Z-axis expression compiles into a TrackDecl
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].axis, "z")
        self.assertAlmostEqual(tracks[0].amp_deg, 5.0, places=4)


class TestArtistAssistantLoopAndProgressiveRefinement(unittest.TestCase):
    """Verify closed-loop artist critique/enhancement and 5-stage from-scratch trajectory construction."""

    def test_looped_artist_enhancement_upgrades_raw_draft_to_verified_masterwork(self) -> None:
        # Take Stage 2 (pure clay geometry, no @materials, no @muscles, no @behavior) of a complex entity
        master = build_apex_complex_program("brute_theropod_wyvern", seed=101)
        stages = build_progressive_from_scratch_trajectory(master)
        raw_clay_draft = replace(
            stages[2].program,
            root=replace(stages[2].program.root, joint_limits=None),
        )

        initial_crit = analyze_artist_project(raw_clay_draft)
        self.assertGreater(len(initial_crit.suggestions), 0)
        self.assertLess(initial_crit.readiness_score, 0.80)

        loop_out = run_looped_artist_enhancement(raw_clay_draft, max_iterations=5)
        self.assertGreater(loop_out["final_readiness_score"], loop_out["initial_readiness_score"] + 0.15)
        upgraded: Program = loop_out["final_program"]
        self.assertGreater(len(upgraded.materials), 0)
        self.assertGreater(len(upgraded.muscles), 0)
        self.assertIsNotNone(upgraded.behavior)
        assert upgraded.behavior is not None
        self.assertTrue(audit_scene(compile_geometry(upgraded), enforce_physical_contact=True)["passed"])
        clip_rep = audit_behavior_clip(upgraded, upgraded.behavior.clips[0].name, num_frames=12)
        self.assertTrue(clip_rep["passed"])

    def test_five_stage_progressive_from_scratch_trajectory_invariants(self) -> None:
        master = build_apex_complex_program("apex_flying_wyvern", seed=101)
        stages = build_progressive_from_scratch_trajectory(master)
        self.assertEqual(len(stages), 5)

        prev_boxes = 0
        prev_score = -1.0
        for st in stages:
            # Every intermediate stage must round-trip parse and pass strict SAT + P1-02 physical contact audit!
            parsed = parse_program(st.program_dsl)
            sc = compile_geometry(parsed)
            aud = audit_scene(sc, enforce_physical_contact=True)
            self.assertTrue(aud["passed"], f"Stage {st.stage_name} failed SAT/P1-02: {aud}")
            self.assertGreaterEqual(st.box_count, prev_boxes)
            self.assertGreaterEqual(st.readiness_score + 1e-6, prev_score)
            prev_boxes = st.box_count
            prev_score = st.readiness_score

        # Stage 0 is a concise silhouette blockout; Stage 4 is the full 56-box musculoskeletal wyvern
        self.assertLess(stages[0].box_count, stages[4].box_count)
        self.assertEqual(stages[4].box_count, 56)
        self.assertGreater(len(stages[4].program.muscles), 0)
        self.assertIsNotNone(stages[4].program.behavior)


class TestApexComplexEntityAndAssetPack(unittest.TestCase):
    """Verify all 10 clean-room Apex families (8 seen + 2 held-out >=40 boxes) and multi-turn dataset shard."""

    def test_benchmark_static_family_counts_preserved_and_all_ten_apex_families_pass(self) -> None:
        self.assertEqual(len(SEEN_STATIC_FAMILIES), 40)
        self.assertEqual(len(UNSEEN_STATIC_FAMILIES), 15)
        self.assertEqual(len(APEX_SEEN_FAMILIES), 8)
        self.assertEqual(len(APEX_UNSEEN_FAMILIES), 2)
        self.assertEqual(len(ALL_APEX_FAMILIES), 10)

        for fam in ALL_APEX_FAMILIES:
            for seed in (101, 202):
                prog = build_apex_complex_program(fam, seed=seed)
                dsl_txt = format_program(prog)
                reparsed = parse_program(dsl_txt)
                self.assertEqual(format_program(reparsed), dsl_txt)
                sc = compile_geometry(reparsed)
                aud = audit_scene(sc, enforce_physical_contact=True)
                self.assertTrue(aud["passed"], f"{fam} seed={seed} failed static audit: {aud}")
                if fam in APEX_UNSEEN_FAMILIES:
                    self.assertGreaterEqual(len(sc.boxes), 40, f"Unseen {fam} has {len(sc.boxes)} < 40 boxes")
                self.assertIsNotNone(reparsed.behavior)
                assert reparsed.behavior is not None
                clip_aud = audit_behavior_clip(reparsed, reparsed.behavior.clips[0].name, num_frames=12)
                self.assertTrue(clip_aud["passed"], f"{fam} seed={seed} failed dynamic clip audit: {clip_aud}")
                self.assertLess(clip_aud["max_stance_slip_units"], 0.02)

    def test_generate_apex_complex_dataset_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = generate_apex_complex_dataset(
                tmp,
                samples_per_seen_family=1,
                samples_per_unseen_family=1,
                seed=202611,
            )
            summary = bundle["summary"]
            rows = bundle["rows"]
            self.assertEqual(summary["total_rows"], 30)  # 10 families * 3 supervision turns
            self.assertEqual(summary["static_sat_pass_rate"], 1.0)
            self.assertEqual(summary["dynamic_clip_pass_rate"], 1.0)
            self.assertLess(summary["max_stance_slip_observed"], 0.02)
            self.assertEqual(summary["default_target_llm"], "LiquidAI/LFM3-VL-2B")

            manifest = json.loads((Path(tmp) / "apex_complex_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_rows"], 30)
            try:
                import pyarrow.parquet as pq

                full_tbl = pq.read_table(Path(tmp) / "apex_complex_entities_v3_1.parquet")
                train_tbl = pq.read_table(Path(tmp) / "train.parquet")
                unseen_tbl = pq.read_table(Path(tmp) / "test_unseen.parquet")
                self.assertEqual(full_tbl.num_rows, 30)
                self.assertEqual(train_tbl.num_rows, 24)
                self.assertEqual(unseen_tbl.num_rows, 6)
                for col in ("source_stage", "target_stage", "initial_readiness_score", "critique_json"):
                    self.assertIn(col, full_tbl.column_names)
            except ImportError:
                pass
            for row in rows:
                self.assertEqual(row["license_spdx"], "Apache-2.0")
                self.assertTrue(row["commercial_weights_eligible"])
                parsed = parse_program(row["assistant_dsl"])
                compiled_row = compile_geometry(parsed)
                self.assertTrue(audit_scene(compiled_row, enforce_physical_contact=True)["passed"])
                # Row metadata must reflect each row's actual assistant_dsl target AST/Scene (not Stage 4 splatting)
                self.assertEqual(row["box_count"], len(compiled_row.boxes))
                self.assertEqual(row["bone_count"], len(compiled_row.bones))
                self.assertEqual(row["locator_count"], len(compiled_row.locators))
                self.assertEqual(row["muscle_count"], len(parsed.muscles))
                self.assertEqual(parsed.root.pivot_local, (0.0, 0.0, 0.0))
                self.assertNotIn("apricorn", row["user_prompt"].lower())
                self.assertNotIn("apricorn", row["assistant_dsl"].lower())
                self.assertNotIn("//", row["assistant_dsl"])

            self.assertIn("distinct_tree_shapes_by_family", manifest)
            self.assertEqual(
                set(manifest["holdout_splits"]),
                {"test_unseen_params", "test_unseen", "test_compositional"},
            )
            for h_split in ("test_unseen_params", "test_unseen", "test_compositional"):
                self.assertGreater(manifest["holdout_split_counts"][h_split], 0)
                self.assertTrue((Path(tmp) / f"{h_split}.jsonl").is_file())

    def test_clean_room_apex_renames_legacy_aliases_and_multi_seed_topological_breadth(self) -> None:
        from synccraft.data_gen import compute_program_tree_shape

        self.assertIn("elemental_companion_critter", APEX_SEEN_FAMILIES)
        self.assertIn("modular_botanical_healing_station", APEX_SEEN_FAMILIES)
        self.assertNotIn("elemental_pocket_companion", ALL_APEX_FAMILIES)
        self.assertNotIn("modular_apricorn_healing_station", ALL_APEX_FAMILIES)

        # Silent input alias lookup preserves compatibility for legacy callers while emitting clean-room ASTs
        critter_alias = build_apex_complex_program("elemental_pocket_companion", seed=101)
        critter_canon = build_apex_complex_program("elemental_companion_critter", seed=101)
        self.assertEqual(format_program(critter_alias), format_program(critter_canon))

        station_alias = build_apex_complex_program("modular_apricorn_healing_station", seed=101)
        station_canon = build_apex_complex_program("modular_botanical_healing_station", seed=101)
        station_dsl = format_program(station_canon)
        self.assertEqual(format_program(station_alias), station_dsl)
        self.assertNotIn("apricorn", station_dsl.lower())
        self.assertIn("botanical_hopper_pod", station_dsl)

        # Multi-seed generation produces rich distinct tree shapes across Apex families
        for fam in ("apex_flying_wyvern", "elemental_companion_critter", "modular_botanical_healing_station"):
            shapes = {
                compute_program_tree_shape(build_apex_complex_program(fam, seed=1000 + k * 37))
                for k in range(12)
            }
            self.assertGreaterEqual(len(shapes), 5, f"{fam} produced only {len(shapes)} distinct tree shapes across 12 seeds")


class TestAdversarialRegressionGuardsV31(unittest.TestCase):
    """Regression suite locking in fixes for cross-package import order, P1-02 locator connectivity,
    per-axis Hill muscle/tendon joint stops, multi-root/rotated decompilers, and flat-rig articulation.
    """

    def test_clean_import_order_world_and_data_gen_subprocesses(self) -> None:
        repo_src = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=repo_src)
        for mod in ("synccraft.world", "synccraft.data_gen"):
            proc = subprocess.run(
                [sys.executable, "-c", f"import {mod}"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"Standalone import {mod} failed:\n{proc.stderr}")

    def test_p1_02_connectivity_rejects_floating_box_with_locator_only_intermediate_bone(self) -> None:
        prog = validate_program(
            Program(
                Meta("entity", "bbmodel_5", 16),
                [Palette("m", (0.5, 0.1, 40.0), 4)],
                Bone(
                    "root",
                    items=[
                        Box("torso", (6, 6, 6), "m"),
                        Box(
                            "floating_illegal",
                            (2, 2, 2),
                            "m",
                            place=Place(None, anchor=(0.0, 0.0, 0.0), align=(0.0, 0.0, 0.0), offset=(0.0, 20.0, 0.0)),
                        ),
                        Bone(
                            "socket_hub",
                            attach=Attach("torso", anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0), detached=False),
                            items=[
                                Locator("hub_loc", offset=(0.0, 0.0, 0.0), role="socket"),
                                Bone(
                                    "spire",
                                    attach=Attach("hub_loc", anchor=(0.0, 0.0, 0.0), align=(0.0, -1.0, 0.0), detached=False),
                                    items=[Box("spire_bx", (2, 4, 2), "m")],
                                ),
                            ],
                        ),
                    ],
                ),
            )
        )
        aud = audit_scene(compile_geometry(prog), enforce_physical_contact=True)
        self.assertFalse(aud["passed"])
        self.assertTrue(any(v["violation"] == "disconnected_physical_components" for v in aud["contact_violations"]))

    def test_per_axis_muscle_independence_and_joint_limit_clamping_before_tendon_coupling(self) -> None:
        prog = validate_program(
            Program(
                Meta("entity", "bbmodel_5", 16),
                [Palette("m", (0.5, 0.1, 40.0), 4)],
                Bone(
                    "root",
                    items=[
                        Box("b0", (4, 4, 4), "m"),
                        Bone(
                            "elbow",
                            attach=Attach("b0", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
                            joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)),
                            items=[
                                Box("b1", (2, 4, 2), "m"),
                                Bone(
                                    "wrist",
                                    attach=Attach("b1", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
                                    joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
                                    items=[Box("b2", (2, 3, 2), "m")],
                                ),
                            ],
                        ),
                    ],
                ),
                muscles=(
                    MuscleDecl("pitch_m", "root", "elbow", "x", 45.0, TendonCoupling("wrist", "x", 0.5)),
                    MuscleDecl("yaw_m", "root", "elbow", "y", 18.0),
                ),
            )
        )
        res_x = evaluate_muscles_at_activations(prog, {"pitch_m": 1.0, "yaw_m": 0.0})
        res_xy = evaluate_muscles_at_activations(prog, {"pitch_m": 1.0, "yaw_m": 1.0})
        # Primary elbow X excursion clamps at 20 deg stop, so 0.5x passive wrist tendon receives 10 deg (not 22.5 deg)
        self.assertAlmostEqual(res_x.pose_euler_deltas["elbow"][0], 20.0, places=6)
        self.assertAlmostEqual(res_x.pose_euler_deltas["wrist"][0], 10.0, places=6)
        # Orthogonal Y muscle does not dilute X excursion
        self.assertAlmostEqual(res_xy.pose_euler_deltas["elbow"][0], 20.0, places=6)
        self.assertAlmostEqual(res_xy.pose_euler_deltas["elbow"][1], 18.0, places=6)

    def test_decompiler_8_corner_fidelity_multiroot_bbmodel_rotated_cubes_and_molang_clamp(self) -> None:
        # 1. Bedrock/Cobblemon rotated cube + bone/clip name collision + multi-harmonic & math.clamp Molang
        geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.test", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {"name": "body", "pivot": [0, 8, 0], "cubes": [{"origin": [-3, 5, -3], "size": [6, 6, 6], "uv": [0, 0]}]},
                        {
                            "name": "tail",
                            "parent": "body",
                            "pivot": [0, 8, -3],
                            "rotation": [10, 0, 0],
                            "cubes": [
                                {"origin": [-1, 7, -9], "size": [2, 2, 6], "uv": [0, 16]},
                                {"origin": [-1, 9, -8], "size": [2, 3, 2], "pivot": [0, 9, -7], "rotation": [35, 0, 25], "uv": [16, 16]},
                            ],
                        },
                    ],
                }
            ],
        }
        anim = {
            "animations": {
                "animation.creature.tail": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "tail": {
                            "rotation": [
                                "math.sin(q.anim_time * 90) * 10 + 0.5 * math.sin(q.anim_time * 180 + 30) * 5",
                                "math.clamp(math.sin(q.anim_time * 720) * 12, -5, 15)",
                                0,
                            ]
                        }
                    },
                }
            }
        }
        prov_nc = AssetProvenance("cobble_tail", "https://example.org/c", "Artist", "CC-BY-NC-4.0", "CC-BY-NC-4.0 Artist")
        rec_bed = ingest_external_asset(geo, "bedrock_geojson", prov_nc, animation_payload=anim, license_policy="research_nc")
        self.assertLess(rec_bed.vertex_reconstruction_error, 1e-9)
        assert rec_bed.program.behavior is not None
        clip = rec_bed.program.behavior.clips[0]
        self.assertNotEqual(clip.name, "tail")
        tr_x, tr_y = clip.tracks
        self.assertAlmostEqual(tr_x.bias_deg, 0.0, places=5)
        self.assertAlmostEqual(tr_x.freq_hz, 0.25, places=4)
        self.assertAlmostEqual(tr_y.freq_hz, 2.0, places=4)
        self.assertAlmostEqual(tr_y.bias_deg - abs(tr_y.amp_deg), -5.0, places=4)
        self.assertAlmostEqual(tr_y.bias_deg + abs(tr_y.amp_deg), 12.0, places=4)

        # 2. Multi-root .bbmodel with rotated group & rotated element
        bb_multi = {
            "resolution": {"width": 32, "height": 32},
            "elements": [
                {"name": "torso", "uuid": "e1", "from": [-3, 6, -2], "to": [3, 12, 2], "origin": [0, 9, 0], "rotation": [0, 0, 0]},
                {"name": "arm_l", "uuid": "e2", "from": [3, 6, -1], "to": [5, 12, 1], "origin": [4, 12, 0], "rotation": [0, 0, 0]},
                {"name": "crest", "uuid": "e3", "from": [-1, 12, -1], "to": [1, 16, 1], "origin": [0, 12, 0], "rotation": [25, 15, 0]},
            ],
            "groups": [
                {"name": "body_grp", "uuid": "g1", "origin": [0, 9, 0], "rotation": [0, 0, 0]},
                {"name": "arm_l_grp", "uuid": "g2", "origin": [4, 12, 0], "rotation": [45, 0, 20]},
            ],
            "outliner": [
                {"uuid": "g1", "children": ["e1", "e3"]},
                {"uuid": "g2", "children": ["e2"]},
            ],
        }
        rec_bb = ingest_external_asset(bb_multi, "bbmodel", AssetProvenance("bb_m", "https://example.org/b", "A", "MIT"))
        sc_bb = compile_geometry(rec_bb.program)
        self.assertLess(rec_bb.vertex_reconstruction_error, 1e-9)
        self.assertFalse(np.allclose(sc_bb.box_by_name["arm_l"].rotation, np.eye(3)))
        self.assertFalse(np.allclose(sc_bb.box_by_name["crest"].rotation, np.eye(3)))

        # 3. Java .json with rotated first element (elements[0])
        java_rot0 = {
            "texture_size": [16, 16],
            "elements": [
                {"name": "blade_0", "from": [7, 2, 7], "to": [9, 14, 9], "rotation": {"origin": [8, 8, 8], "axis": "y", "angle": 45}},
                {"name": "guard_1", "from": [5, 6, 7], "to": [11, 8, 9]},
            ],
        }
        rec_jv = ingest_external_asset(java_rot0, "java_json", AssetProvenance("jv_r", "https://example.org/j", "A", "MIT"))
        sc_jv = compile_geometry(rec_jv.program)
        self.assertLess(rec_jv.vertex_reconstruction_error, 1e-9)
        self.assertFalse(np.allclose(sc_jv.box_by_name["blade_0"].rotation, np.eye(3)))

    def test_flat_ten_box_entity_looped_enhancement_articulates_and_animates_with_zero_drift(self) -> None:
        flat_draft = validate_program(
            Program(
                Meta("entity", "bbmodel_5", 16),
                [Palette("hide", (0.52, 0.12, 38.0), 6), Palette("horn", (0.78, 0.06, 80.0), 4)],
                Bone(
                    "root",
                    pivot_local=(0.0, 12.0, 0.0),
                    items=[
                        Box("torso", (8, 6, 12), "hide"),
                        Box("neck", (4, 4, 4), "hide", place=Place("torso", anchor=(0.0, 0.3, 1.0), align=(0.0, 0.0, -1.0))),
                        Box("head", (5, 4, 5), "horn", place=Place("neck", anchor=(0.0, 0.2, 1.0), align=(0.0, 0.0, -1.0))),
                        Box("tail_1", (4, 3, 6), "hide", place=Place("torso", anchor=(0.0, 0.1, -1.0), align=(0.0, 0.0, 1.0))),
                        Box("tail_2", (2, 2, 6), "horn", place=Place("tail_1", anchor=(0.0, 0.0, -1.0), align=(0.0, 0.0, 1.0))),
                        Box("thigh_l", (3, 5, 3), "hide", place=Place("torso", anchor=(0.75, -1.0, 0.6), align=(0.0, 1.0, 0.0))),
                        Box("shank_l", (2, 5, 2), "hide", place=Place("thigh_l", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0))),
                        Box("thigh_r", (3, 5, 3), "hide", place=Place("torso", anchor=(-0.75, -1.0, 0.6), align=(0.0, 1.0, 0.0))),
                        Box("shank_r", (2, 5, 2), "hide", place=Place("thigh_r", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0))),
                        Box("wing_l", (8, 1, 6), "hide", place=Place("torso", anchor=(1.0, 0.8, 0.2), align=(-1.0, 0.0, 0.0))),
                    ],
                ),
            )
        )
        orig_sc = compile_geometry(flat_draft)
        loop_res = run_looped_artist_enhancement(flat_draft, max_iterations=6)
        upgraded: Program = loop_res["final_program"]
        up_sc = compile_geometry(upgraded)

        for bx in orig_sc.boxes:
            err = float(np.max(np.linalg.norm(bx.corners(include_inflate=False) - up_sc.box_by_name[bx.name].corners(include_inflate=False), axis=-1)))
            self.assertLess(err, 1e-9, f"Articulation shifted rest vertices of {bx.name} by {err}")

        self.assertGreaterEqual(len(up_sc.bones), 8)
        self.assertEqual(len(upgraded.materials), 2)
        self.assertGreater(len(upgraded.muscles), 0)
        self.assertIsNotNone(upgraded.kinematics)
        assert upgraded.kinematics is not None
        self.assertGreaterEqual(len(upgraded.kinematics.ik_chains), 2)
        self.assertIsNotNone(upgraded.behavior)
        assert upgraded.behavior is not None
        self.assertTrue(audit_scene(up_sc, enforce_physical_contact=True)["passed"])
        self.assertTrue(audit_behavior_clip(upgraded, upgraded.behavior.clips[0].name, num_frames=12)["passed"])


class TestCobblemonRealWorldIdiomsAndClosedLoopRoundtrip(unittest.TestCase):
    """Verify real-world Cobblemon (.geo.json / .animation.json) idioms, CC-BY-NC quarantine builder, and roundtrip export."""

    def test_real_world_cobblemon_nested_parens_molang_catmullrom_and_timestamped_molang_keyframes(self) -> None:
        skel_prog = validate_program(
            Program(
                Meta("entity", "bedrock_entity", 16),
                [Palette("mat", (0.6, 0.1, 40.0), 4)],
                Bone(
                    "body_root",
                    items=[
                        Box("body_bx", (6, 6, 8), "mat"),
                        Bone("tail_1", attach=Attach("body_bx", (0, 0, -1), (0, 0, 1)), items=[Box("tail_bx", (2, 2, 6), "mat")]),
                        Bone("wing_l", attach=Attach("body_bx", (1, 0, 0), (-1, 0, 0)), items=[Box("wing_bx", (6, 1, 4), "mat")]),
                        Bone("ear_r", attach=Attach("body_bx", (-1, 1, 0), (1, -1, 0)), items=[Box("ear_bx", (1, 3, 2), "mat")]),
                        Bone("crest", attach=Attach("body_bx", (0, 1, 0), (0, -1, 0)), items=[Box("crest_bx", (1, 3, 3), "mat")]),
                        Bone("jaw", attach=Attach("body_bx", (0, -1, 1), (0, 1, -1)), items=[Box("jaw_bx", (3, 1, 3), "mat")]),
                    ],
                ),
            )
        )
        anim_payload = {
            "format_version": "1.8.0",
            "animations": {
                "animation.cobble_creature.air_idle": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "tail_1": {
                            "rotation": [
                                "math.sin((q.anim_time - 0.25) * 90 * 4) * 10",
                                "0 - math.sin((query.anim_time + 0.125) * 360) * 8",
                                "-(4 + math.cos(q.anim_time * 180) * -6)",
                            ]
                        },
                        "wing_l": {
                            "rotation": {
                                "0.0": {"pre": [0, 0, 0], "post": [[18, 0, -6]], "lerp_mode": "catmullrom"},
                                "1.0": {"pre": [[18, 0, -6]], "post": [-14, 0, 8], "lerp_mode": "catmullrom"},
                                "2.0": {"pre": [0, 0, 0], "lerp_mode": "catmullrom"},
                            }
                        },
                        "ear_r": {
                            "rotation": {
                                "0.0": ["math.sin((q.anim_time - 0.2) * 360) * 12", 0, "math.sin(q.anim_time * 180) * -5"],
                            }
                        },
                        "crest": {
                            "rotation": {
                                "0.0": {"vector": [0, 0, 7.5]},
                            }
                        },
                        "jaw": {
                            "rotation": [12.5, 0, 0],
                        },
                    },
                }
            },
        }
        beh = decompile_bedrock_animation_to_behavior(anim_payload, skel_prog)
        self.assertIsNotNone(beh)
        assert beh is not None
        tracks = {(t.bone, t.axis): t for t in beh.clips[0].tracks}

        # 1. Nested-paren Molang: math.sin((t - 0.25) * 360) * 10 -> freq=1.0Hz, phase=270 deg (-90 mod 360), amp=10
        t_tail_x = tracks[("tail_1", "x")]
        self.assertAlmostEqual(t_tail_x.freq_hz, 1.0, places=5)
        self.assertAlmostEqual(t_tail_x.phase_deg, 270.0, places=4)
        self.assertAlmostEqual(t_tail_x.amp_deg, 10.0, places=5)

        # 2. Leading 0 - math.sin((t + 0.125) * 360) * 8 -> negated amplitude (-8.0), freq=1.0Hz, phase=45 deg
        t_tail_y = tracks[("tail_1", "y")]
        self.assertAlmostEqual(t_tail_y.freq_hz, 1.0, places=5)
        self.assertAlmostEqual(t_tail_y.phase_deg, 45.0, places=4)
        self.assertAlmostEqual(t_tail_y.amp_deg, -8.0, places=5)

        # 3. Signed outer parens + negative trailing multiplier: -(4 + cos(180*t)*-6) = -4 + 6*cos(180*t)
        t_tail_z = tracks[("tail_1", "z")]
        self.assertAlmostEqual(t_tail_z.bias_deg, -4.0, places=5)
        self.assertAlmostEqual(t_tail_z.freq_hz, 0.5, places=5)
        self.assertAlmostEqual(t_tail_z.amp_deg, 6.0, places=5)

        # 4. Catmull-Rom wrapped-post [[18, 0, -6]] + pre-only final keyframe
        self.assertIn(("wing_l", "x"), tracks)
        self.assertIn(("wing_l", "z"), tracks)
        self.assertGreater(abs(tracks[("wing_l", "x")].amp_deg), 10.0)

        # 5. Timestamped Molang keyframe dict {"0.0": ["math.sin((q.anim_time - 0.2) * 360) * 12", ...]}
        t_ear_x = tracks[("ear_r", "x")]
        self.assertAlmostEqual(t_ear_x.freq_hz, 1.0, places=5)
        self.assertAlmostEqual(t_ear_x.amp_deg, 12.0, places=5)
        t_ear_z = tracks[("ear_r", "z")]
        self.assertAlmostEqual(t_ear_z.amp_deg, -5.0, places=5)

        # 6. GeckoLib {"vector": ...} single keyframe & constant pose vector
        self.assertAlmostEqual(tracks[("crest", "z")].bias_deg, 7.5, places=5)
        self.assertAlmostEqual(tracks[("jaw", "x")].bias_deg, 12.5, places=5)

    def test_planar_zero_thickness_cobblemon_cubes_preserve_box_uv_strips_and_vertex_fidelity(self) -> None:
        import io
        from PIL import Image

        atlas_rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        # Distinct teal pigment on the exact planar fin strip [u=0..12, v=0..8] for size=[0, 8, 12] at uv=[0, 0]
        atlas_rgba[0:8, 0:24] = np.array([30, 195, 175, 255], dtype=np.uint8)
        atlas_rgba[8:32, :] = np.array([180, 70, 50, 255], dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(atlas_rgba, mode="RGBA").save(buf, format="PNG")
        atlas_png = buf.getvalue()

        geo_planar = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.cobble.finling", "texture_width": 32, "texture_height": 32},
                    "bones": [
                        {
                            "name": "body",
                            "pivot": [0, 6, 0],
                            "cubes": [
                                {"origin": [-3, 3, -4], "size": [6, 6, 8], "uv": [0, 12]},
                                {"origin": [0, 9, -6], "size": [0, 8, 12], "uv": [0, 0]},
                                {"origin": [-4, 9, 4], "size": [8, 6, 0], "uv": [8, 12]},
                            ],
                        }
                    ],
                }
            ],
        }
        prov = AssetProvenance(
            asset_id="cobble_finling",
            source_url="https://gitlab.com/cable-mc/cobblemon-assets-example/finling",
            author="CobbleContributor",
            license_spdx="CC-BY-NC-4.0",
            attribution_notice="CC-BY-NC-4.0 CobbleContributor",
        )
        rec = ingest_external_asset(
            geo_planar,
            "bedrock_geojson",
            prov,
            atlas_png=atlas_png,
            license_policy="research_nc",
        )
        # Valid planar 0-thickness quads achieve < 1e-9 corner reconstruction error (not 0.5u false positive!)
        self.assertLess(rec.vertex_reconstruction_error, 1e-9)
        sc = compile_geometry(rec.program)
        # Midplane center of origin=[0, 9, -6], size=[0, 8, 12] is strictly [0.0, 13.0, 0.0]
        fin_box = sc.boxes[1]
        np.testing.assert_allclose(fin_box.center, np.array([0.0, 13.0, 0.0]), atol=1e-9)

        # Conversely, non-zero fractional cube size [2.4, 6, 8] MUST honestly report ~0.2u quantization error
        geo_frac = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.cobble.frac", "texture_width": 32, "texture_height": 32},
                    "bones": [{"name": "body", "pivot": [0, 6, 0], "cubes": [{"origin": [-1.2, 3, -4], "size": [2.4, 6, 8], "uv": [0, 0]}]}],
                }
            ],
        }
        rec_frac = ingest_external_asset(geo_frac, "bedrock_geojson", prov, license_policy="research_nc")
        self.assertAlmostEqual(rec_frac.vertex_reconstruction_error, 0.2, places=5)

    def test_build_quarantined_research_nc_dataset_on_cobblemon_entity_and_block_assets(self) -> None:
        creature_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.cobblemon.mossback", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {
                            "name": "torso",
                            "pivot": [0, 7, 0],
                            "cubes": [{"origin": [-4, 3, -5], "size": [8, 7, 10], "uv": [0, 0]}],
                            "locators": {"seat_1": [0, 10, 0], "target": [0, 6, -5]},
                        },
                        {
                            "name": "head",
                            "parent": "torso",
                            "pivot": [0, 9, -5],
                            "cubes": [
                                {"origin": [-3, 7, -10], "size": [6, 5, 5], "uv": [0, 18]},
                                {"origin": [0, 12, -9], "size": [0, 4, 5], "uv": [24, 18]},
                            ],
                        },
                    ],
                }
            ],
        }
        creature_anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.mossback.ground_idle": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "head": {"rotation": ["math.sin((q.anim_time - 0.25) * 90 * 2) * 6", 0, 0]}
                    },
                }
            },
        }
        block_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.cobblemon.healing_machine", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {
                            "name": "pedestal",
                            "pivot": [0, 0, 0],
                            "cubes": [
                                {"origin": [-7, 0, -7], "size": [14, 10, 14], "uv": [0, 0]},
                                {"origin": [-5, 10, -5], "size": [10, 3, 10], "uv": [0, 24]},
                            ],
                            "locators": {"ball_slot_1": [-2, 13, -2], "ball_slot_2": [2, 13, -2]},
                        }
                    ],
                }
            ],
        }
        items = [
            {
                "payload": creature_geo,
                "format_kind": "bedrock_geojson",
                "animation_payload": creature_anim,
                "provenance": AssetProvenance(
                    "cobblemon_mossback",
                    "https://gitlab.com/cable-mc/cobblemon/-/blob/main/common/src/main/resources/assets/cobblemon/bedrock/pokemon/models/mossback.geo.json",
                    "Cobblemon Team",
                    "CC-BY-NC-4.0",
                    "Cobblemon Assets (CC-BY-NC-4.0) - Non-Commercial Research Quarantine",
                ),
            },
            {
                "payload": block_geo,
                "format_kind": "bedrock_geojson",
                "provenance": AssetProvenance(
                    "cobblemon_healing_machine",
                    "https://gitlab.com/cable-mc/cobblemon/-/blob/main/common/src/main/resources/assets/cobblemon/bedrock/block/models/healing_machine.geo.json",
                    "Cobblemon Team",
                    "CC-BY-NC-4.0",
                    "Cobblemon Assets (CC-BY-NC-4.0) - Non-Commercial Research Quarantine",
                ),
            },
        ]

        # Must reject Mojang or unattributed CC-BY-NC records in quarantined builder
        with self.assertRaises(LicenseGateError):
            build_quarantined_research_nc_dataset(
                [
                    {
                        "payload": creature_geo,
                        "format_kind": "bedrock_geojson",
                        "provenance": AssetProvenance(
                            "bad_no_attr",
                            "https://example.org/model",
                            "Author",
                            "CC-BY-NC-4.0",
                            "",
                        ),
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_quarantined_research_nc_dataset(items, output_dir=tmp)
            self.assertEqual(bundle["total_rows"], 2)
            self.assertFalse(bundle["commercial_weights_eligible"])
            self.assertTrue((Path(tmp) / "research_nc" / "quarantined_assets_v3_1.jsonl").is_file())
            self.assertTrue((Path(tmp) / "research_nc" / "quarantine_manifest.json").is_file())
            for r in bundle["rows"]:
                self.assertTrue(r["split"].startswith("research_nc/"))
                self.assertFalse(r["commercial_weights_eligible"])
                self.assertLess(r["vertex_reconstruction_error"], 1e-9)

    def test_closed_loop_artist_roundtrip_export_bedrock_geckolib_and_animated_bbmodel(self) -> None:
        # Import a Cobblemon-style Bedrock entity -> run closed-loop artist enhancement -> export & roundtrip all 3 formats
        raw_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.artist.drakelet", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {
                            "name": "torso",
                            "pivot": [0, 9, 0],
                            "cubes": [{"origin": [-4, 6, -5], "size": [8, 6, 10], "uv": [0, 0]}],
                            "locators": {"rider_seat": [0, 12, 0]},
                        },
                        {
                            "name": "head",
                            "parent": "torso",
                            "pivot": [0, 10, -5],
                            "cubes": [{"origin": [-3, 8, -10], "size": [6, 5, 5], "uv": [0, 16]}],
                        },
                        {
                            "name": "leg_l",
                            "parent": "torso",
                            "pivot": [3, 6, -2],
                            "cubes": [{"origin": [2, 0, -3], "size": [2, 6, 2], "uv": [24, 16]}],
                        },
                        {
                            "name": "leg_r",
                            "parent": "torso",
                            "pivot": [-3, 6, -2],
                            "cubes": [{"origin": [-4, 0, -3], "size": [2, 6, 2], "uv": [32, 16]}],
                        },
                    ],
                }
            ],
        }
        raw_anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.drakelet.idle": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "head": {"rotation": ["math.sin((q.anim_time - 0.1) * 180) * 6", 0, 0]},
                    },
                }
            },
        }
        prov = AssetProvenance(
            "artist_drakelet",
            "https://example.org/artist_drakelet",
            "StudioArtist",
            "CC-BY-NC-4.0",
            "CC-BY-NC-4.0 StudioArtist",
        )
        loop_out = run_looped_artist_enhancement(
            raw_geo,
            format_kind="bedrock_geojson",
            animation_payload=raw_anim,
            provenance=prov,
            license_policy="research_nc",
            max_iterations=4,
        )
        enhanced_prog: Program = loop_out["final_program"]
        self.assertGreater(loop_out["final_readiness_score"], loop_out["initial_readiness_score"])

        for target_fmt in ("bedrock_entity", "geckolib", "bbmodel_5"):
            exp = export_artist_project_bundle(
                enhanced_prog,
                export_format=target_fmt,
                name="drakelet_enhanced",
            )
            self.assertFalse(exp["runtime_import_verified"])
            self.assertTrue(exp["static_audit_passed"])
            self.assertLess(exp["roundtrip_vertex_max_error"], 1e-8)
            self.assertGreaterEqual(exp["roundtrip_clip_count"], 1)

    def test_cobblemon_empty_grouping_hierarchy_progressive_trajectory_affine_molang_and_antiphase_keyframes(self) -> None:
        from synccraft.world.inverse_inference import (
            _parse_molang_channel_zero_eval,
            decompile_bedrock_animation_to_behavior,
        )

        # 1. Authentic Cobblemon empty-grouping-bone + rotated planar wing hierarchy -> 5-stage progressive trajectory
        cobble_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.cobblemon.aerodragon", "texture_width": 128, "texture_height": 128},
                    "bones": [
                        {"name": "aerodragon", "pivot": [0, 0, 0]},
                        {"name": "body", "parent": "aerodragon", "pivot": [0, 14, 0]},
                        {
                            "name": "torso",
                            "parent": "body",
                            "pivot": [0, 14, 0],
                            "cubes": [
                                {"origin": [-5, 10, -8], "size": [10, 8, 16], "uv": [0, 0]},
                                {"origin": [0, 18, -6], "size": [0, 6, 12], "uv": [20, 10]},
                            ],
                            "locators": {"seat_1": [0, 18, 0]},
                        },
                        {"name": "wings", "parent": "torso", "pivot": [0, 17, -2]},
                        {
                            "name": "wing_left",
                            "parent": "wings",
                            "pivot": [5, 17, -2],
                            "rotation": [0, 0, 12],
                            "cubes": [
                                {
                                    "origin": [5, 17, -8],
                                    "size": [16, 0, 12],
                                    "pivot": [5, 17, -2],
                                    "rotation": [5, -10, 0],
                                    "uv": [0, 32],
                                }
                            ],
                        },
                        {
                            "name": "wing_right",
                            "parent": "wings",
                            "pivot": [-5, 17, -2],
                            "rotation": [0, 0, -12],
                            "cubes": [
                                {
                                    "origin": [-21, 17, -8],
                                    "size": [16, 0, 12],
                                    "pivot": [-5, 17, -2],
                                    "rotation": [5, 10, 0],
                                    "uv": [0, 32],
                                    "mirror": True,
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        cobble_anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.aerodragon.ground_idle": {
                    "loop": True,
                    "animation_length": 2.0,
                    "bones": {
                        "wing_left": {
                            "rotation": [
                                "(1 - math.cos(q.anim_time * 180)) * 12",
                                0,
                                "math.sin((q.anim_time - 0.25) * 180) * 24",
                            ]
                        },
                        "wing_right": {
                            "rotation": [
                                "(1 - math.cos(q.anim_time * 180)) * 12",
                                0,
                                "-math.sin((q.anim_time - 0.25) * 180) * 24",
                            ]
                        },
                    },
                }
            },
        }
        prov = AssetProvenance(
            "cobblemon_aerodragon",
            "https://gitlab.com/cable-mc/cobblemon",
            "Cobblemon Contributors",
            "CC-BY-NC-4.0",
            "Cobblemon Assets (CC-BY-NC-4.0)",
        )
        rec = ingest_external_asset(
            cobble_geo,
            "bedrock_geojson",
            prov,
            license_policy="research_nc",
            animation_payload=cobble_anim,
        )
        self.assertLess(rec.vertex_reconstruction_error, 1e-8)
        assert rec.program.behavior is not None
        self.assertEqual(len(rec.program.behavior.clips[0].tracks), 4)
        stages = build_progressive_from_scratch_trajectory(rec.program)
        self.assertEqual(len(stages), 5)
        for st in stages:
            self.assertTrue(st.static_audit_passed)

        # 2. Affine-on-trig, q.life_time, ternary guards, math.abs, and leading-minus bias sign preservation
        self.assertEqual(
            _parse_molang_channel_zero_eval("(1 - math.cos(q.anim_time * 360)) * 15", "tail"),
            ("wave", -15.0, 0.25, 15.0, 1.0),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("(math.sin(q.anim_time * 360) + 1) * 10", "tail"),
            ("wave", 10.0, 0.0, 10.0, 1.0),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("0.5 * (math.sin((q.anim_time - 0.25) * 360) * 12 + 8)", "tail"),
            ("wave", 6.0, 0.75, 4.0, 1.0),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("math.sin(q.life_time * 180) * 6", "tail"),
            ("wave", 6.0, 0.0, 0.0, 0.5),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("q.is_moving ? math.sin(q.anim_time * 360) * 14 : 0", "tail"),
            ("wave", 14.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("-math.abs(math.sin(q.anim_time * 180)) * 20 + 5", "tail"),
            ("hinge", -20.0, 0.0, 5.0, 1.0),
        )
        self.assertEqual(
            _parse_molang_channel_zero_eval("-math.sin(q.anim_time * 360) * 10 + 5", "tail"),
            ("wave", -10.0, 0.0, 5.0, 1.0),
        )

        # 3. Anti-phase keyframe walk timelines preserve 180 deg phase opposition
        walk_anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.aerodragon.walk": {
                    "loop": True,
                    "animation_length": 1.0,
                    "bones": {
                        "wing_left": {
                            "rotation": {
                                "0.0": [0, 0, 0],
                                "0.25": [25, 0, 0],
                                "0.5": [0, 0, 0],
                                "0.75": [-25, 0, 0],
                                "1.0": [0, 0, 0],
                            }
                        },
                        "wing_right": {
                            "rotation": {
                                "0.0": [0, 0, 0],
                                "0.25": [-25, 0, 0],
                                "0.5": [0, 0, 0],
                                "0.75": [25, 0, 0],
                                "1.0": [0, 0, 0],
                            }
                        },
                    },
                }
            },
        }
        beh_walk = decompile_bedrock_animation_to_behavior(walk_anim, rec.program)
        tr_l = [t for t in beh_walk.clips[0].tracks if t.bone == "wing_left"][0]
        tr_r = [t for t in beh_walk.clips[0].tracks if t.bone == "wing_right"][0]
        self.assertAlmostEqual(tr_l.amp_deg, 25.0, places=2)
        self.assertAlmostEqual(tr_r.amp_deg, 25.0, places=2)
        self.assertAlmostEqual(abs(tr_r.phase_deg - tr_l.phase_deg), 180.0, places=2)


if __name__ == "__main__":
    unittest.main()
