"""ADR 0007 regression suite: sculpted partial-overlap contract, continuous spin, colossal UVs, and flagship exemplars."""

from __future__ import annotations

import unittest

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.backends import export_bbmodel, export_bedrock, export_geckolib, export_gltf_pbr, export_labpbr_1_3
from synccraft.dsl.ast_nodes import DSLValidationError
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.musculoskeletal import audit_behavior_clip, evaluate_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.modeling.pgd_projector import project_program, verify_torch_se3_parity
from synccraft.shading.compiler import compile_vanilla_atlas
from synccraft.shading.uv_packer import allocate_uv_layout
from synccraft.world.artist_loop import analyze_artist_project
from synccraft.world.exemplars import (
    build_colossal_walking_citadel_world,
    build_sculpted_elder_wyvern_world,
)


class SculptedAndColossalEntityTests(unittest.TestCase):
    def test_sculpted_profile_and_blend_with_ast_roundtrip_and_clone_expansion(self) -> None:
        dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
        @palette { armor: ramp(base_lch=[0.5, 0.08, 240], steps=6) }
        @rig {
          root bone hull {
            pivot_local: [0, 8, 0]
            box base_hull { size: [10, 4, 12], material: armor }
            box base_drum place(base_box=base_hull, anchor=[0, 1, -0.2], align=[0, -1, 0]) {
              size: [6, 4, 6], material: armor
            }
            bone prow_l attach(parent_box=base_hull, anchor=[-0.4, 0, 1], align=[0, 0, -0.5]) mirror_x(prow_r) {
              rest_euler: [0, 24, 0]
              box wedge_l { size: [5, 4, 4], material: armor, blend_with: [prow_r__wedge_l] }
            }
            bone rib joint(parent=base_drum, at=[0, 2, 2.5], align=[0, -0.8, 0.5]) radial(count=4, axis=y) {
              rest_euler: [-20, 0, 0]
              box arch { size: [3, 2, 2], material: armor, blend_with: [rib_r1__arch] }
            }
          }
        }
        """
        program = parse_program(dsl)
        self.assertEqual(program.meta.audit_profile, "sculpted")
        formatted = format_program(program)
        self.assertIn("audit_profile: sculpted", formatted)
        self.assertIn("blend_with: [prow_r__wedge_l]", formatted)
        self.assertEqual(parse_program(formatted), program)

        scene = compile_geometry(program)
        # Mirror cross-centerline pair + all 4 cyclic radial dome pairs (0-1, 1-2, 2-3, 3-0)
        expected_pairs = {
            ("prow_r__wedge_l", "wedge_l"),
            ("arch", "rib_r1__arch"),
            ("rib_r1__arch", "rib_r2__arch"),
            ("rib_r2__arch", "rib_r3__arch"),
            ("arch", "rib_r3__arch"),
        }
        self.assertTrue(expected_pairs.issubset(scene.blend_pairs))
        report = audit_scene(scene, enforce_physical_contact=True)
        self.assertTrue(report["passed"], msg=str(report))
        self.assertGreaterEqual(len(report["sculpted_overlap_pairs"]), 5)

    def test_blend_with_rejects_self_unknown_and_negative_inflate_in_sculpted(self) -> None:
        with self.assertRaisesRegex(DSLValidationError, "blend_with"):
            parse_program(
                "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }\n"
                "@palette { m: ramp(base_lch=[0.5,0.05,120], steps=4) }\n"
                "@rig { root bone r { pivot_local: [0,2,0] box a { size: [4,4,4], material: m, blend_with: [a] } } }"
            )
        with self.assertRaisesRegex(DSLValidationError, "blend_with"):
            parse_program(
                "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }\n"
                "@palette { m: ramp(base_lch=[0.5,0.05,120], steps=4) }\n"
                "@rig { root bone r { pivot_local: [0,2,0] box a { size: [4,4,4], material: m, blend_with: [missing_box] } } }"
            )
        with self.assertRaisesRegex(DSLValidationError, "inflate"):
            parse_program(
                "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }\n"
                "@palette { m: ramp(base_lch=[0.5,0.05,120], steps=4) }\n"
                "@rig { root bone r { pivot_local: [0,2,0] box a { size: [4,4,4], material: m, inflate: -0.1 } } }"
            )

    def test_sloped_beak_and_72deg_horn_fail_strict_and_pass_sculpted(self) -> None:
        dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
        @palette { hide: ramp(base_lch=[0.5, 0.08, 45], steps=6) }
        @rig {
          root bone skull {
            pivot_local: [0, 6, 0]
            box cranium { size: [8, 6, 8], material: hide }
            box sloped_snout place(base_box=cranium, anchor=[0, 0.35, 1], align=[0, 0.2, -0.6]) {
              size: [6, 3, 8], material: hide, rotate: [24, 0, 0]
            }
            bone swept_horn attach(parent_box=cranium, anchor=[0, 1, -0.2], align=[0, -0.85, 0.2]) {
              rest_euler: [-72, 0, 0]
              box horn_core { size: [3, 8, 4], material: hide }
            }
          }
        }
        """
        scene = compile_geometry(parse_program(dsl))
        strict_rep = audit_scene(scene, audit_profile="strict", enforce_physical_contact=True)
        sculpt_rep = audit_scene(scene, audit_profile="sculpted", enforce_physical_contact=True)
        self.assertFalse(strict_rep["passed"])
        self.assertGreaterEqual(len(strict_rep["contact_violations"]), 2)
        self.assertTrue(sculpt_rep["passed"], msg=str(sculpt_rep))
        self.assertEqual(len(sculpt_rep["sculpted_overlap_pairs"]), 2)
        for entry in sculpt_rep["sculpted_overlap_pairs"]:
            self.assertGreater(entry["min_protrusion_units"], entry["min_required_protrusion"])
            self.assertLessEqual(entry["raw_min_overlap"], entry["max_allowed_overlap"])

    def test_sculpted_rejects_swallowed_box_z_fighting_sibling_collision_and_floating_gap(self) -> None:
        # 1. Swallowed redundant box inside larger host OBB
        swallowed_prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box outer_shell { size: [8, 8, 8], material: m }
                box buried_inner place(base_box=outer_shell, anchor=[0, 0, 0], align=[0, 0, 0]) {
                  size: [2, 2, 2], material: m
                }
              }
            }
            """
        )
        sw_rep = audit_scene(compile_geometry(swallowed_prog), enforce_physical_contact=True)
        self.assertFalse(sw_rep["passed"])
        self.assertEqual(sw_rep["contact_violations"][0]["violation"], "swallowed_redundant_box")
        crit = analyze_artist_project(swallowed_prog)
        self.assertIn("swallowed redundant box", crit.suggestions[0].rationale)

        # 2. Coplanar Z-fighting (flat zero-thickness plane on unculled box face) rejected in sculpted mode
        zfight_prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box a { size: [6, 4, 6], material: m }
                plane decal place(base_box=a, anchor=[0, 1, 0], align=[0, 0, 0]) {
                  size: [4, 4], normal: y, material: m
                }
              }
            }
            """
        )
        zf_rep = audit_scene(compile_geometry(zfight_prog), enforce_physical_contact=True)
        self.assertFalse(zf_rep["passed"])
        self.assertGreaterEqual(len(zf_rep["z_fighting_pairs"]), 1)

        # 3. Undeclared sibling-bone collision (left thigh vs right thigh) has ZERO implicit immunity
        sibling_prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone pelvis {
                pivot_local: [0, 8, 0]
                box core { size: [4, 4, 4], material: m }
                bone thigh_l attach(parent_box=core, anchor=[-0.2, -1, 0], align=[0, 1, 0]) mirror_x(thigh_r) {
                  box femur_l { size: [3, 6, 3], material: m }
                }
              }
            }
            """
        )
        sib_rep = audit_scene(compile_geometry(sibling_prog), enforce_physical_contact=True)
        self.assertFalse(sib_rep["passed"])
        self.assertTrue(
            any(set(c["pair"]) == {"femur_l", "thigh_r__femur_l"} for c in sib_rep["collision_pairs"]),
            msg=str(sib_rep),
        )

        # 4. Separated declared contact fails in sculpted rest pose and is repaired by PGD projector
        gap_prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box base { size: [4, 4, 4], material: m }
                box cap place(base_box=base, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0, 0.25, 0]) {
                  size: [3, 2, 3], material: m
                }
              }
            }
            """
        )
        gap_rep = audit_scene(compile_geometry(gap_prog), enforce_physical_contact=True)
        self.assertFalse(gap_rep["passed"])
        proj = project_program(gap_prog, max_steps=12)
        self.assertTrue(proj.passed)

    def test_continuous_spin_track_gimbal_safe_keyframes_and_molang_export(self) -> None:
        dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }
        @palette { brass: ramp(base_lch=[0.7, 0.12, 80], steps=6) }
        @rig {
          root bone frame {
            pivot_local: [0, 6, 0]
            box block { size: [6, 4, 6], material: brass }
            bone turret_y attach(parent_box=block, anchor=[0, 1, 0], align=[0, -1, 0]) {
              box dome { size: [3, 3, 3], material: brass }
            }
            bone wheel_x attach(parent_box=block, anchor=[-1, 0, -0.5], align=[1, 0, 0]) {
              box rim { size: [1, 3, 3], material: brass }
            }
            bone pinion_x attach(parent_box=block, anchor=[1, 0, 0.5], align=[-1, 0, 0]) {
              box gear { size: [1, 2, 2], material: brass }
            }
          }
        }
        @behavior {
          clip drive {
            duration_s: 2.0
            track turret_y: spin(axis=y, amp_deg=360, freq_hz=0.5)
            track wheel_x: spin(axis=x, amp_deg=360, freq_hz=0.5)
            track pinion_x: spin(axis=x, amp_deg=-720, freq_hz=0.5)
          }
        }
        """
        program = parse_program(dsl)
        self.assertEqual(parse_program(format_program(program)), program)
        frames = evaluate_behavior_clip(program, "drive", num_frames=4)
        self.assertAlmostEqual(frames[2].pose_euler_deltas["turret_y"][1], 180.0, places=5)
        self.assertAlmostEqual(frames[2].pose_euler_deltas["pinion_x"][0], -360.0, places=5)

        scene = compile_geometry(program)
        # Baked keyframe mode (even with coarse num_frames=4 on -720 deg gear & 360 deg Y turret)
        baked = export_bedrock(scene, program=program, animation_mode="keyframes", num_animation_frames=4)
        anim_bones = baked["animations"]["animations"]["animation.entity.drive"]["bones"]
        turret_keys = anim_bones["turret_y"]["rotation"]
        wheel_keys = anim_bones["wheel_x"]["rotation"]
        pinion_keys = anim_bones["pinion_x"]["rotation"]
        # Bedrock X-reflection maps (rx, ry, rz) -> (-rx, -ry, rz): turret goes 0 -> -360 without X/Z gimbal flip;
        # wheel (+360 SyncCraft) -> -360 Bedrock, meshed pinion (-720 SyncCraft) -> +720 Bedrock (exact -2:1 ratio)
        self.assertAlmostEqual(turret_keys["0.0"][1], 0.0, places=4)
        self.assertAlmostEqual(turret_keys["2.0"][1], -360.0, places=4)
        self.assertAlmostEqual(turret_keys["1.0"][0], 0.0, places=4)
        self.assertAlmostEqual(turret_keys["1.0"][2], 0.0, places=4)
        self.assertAlmostEqual(wheel_keys["2.0"][0], -360.0, places=4)
        self.assertAlmostEqual(pinion_keys["2.0"][0], 720.0, places=4)

        # Native Molang mode emits linear query.anim_time expressions
        molang_out = export_bedrock(scene, program=program, animation_mode="molang")
        mol_bones = molang_out["animations"]["animations"]["animation.entity.drive"]["bones"]
        self.assertIn("query.anim_time", str(mol_bones["turret_y"]["rotation"][1]))
        self.assertIn("query.anim_time", str(mol_bones["pinion_x"]["rotation"][0]))

    def test_colossal_uv_atlas_auto_expansion_and_explicit_max_width_cap(self) -> None:
        small_prog = parse_program(
            "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }\n"
            "@palette { p: ramp(base_lch=[0.6, 0.05, 90], steps=4) }\n"
            "@rig { root bone r { pivot_local: [0, 4, 0] box b { size: [8, 8, 8], material: p } } }"
        )
        small_uv = allocate_uv_layout(small_prog, compile_geometry(small_prog))
        self.assertLessEqual(small_uv["width"], 512)
        self.assertEqual(small_uv["width"], 64)

        colossal_prog = parse_program(
            """
            @meta { archetype: entity, target_profile: geckolib, texel_density: 64, audit_profile: sculpted }
            @palette { stone: ramp(base_lch=[0.55, 0.04, 80], steps=6) }
            @rig {
              root bone bastion {
                pivot_local: [0, 24, 0]
                box keep_a { size: [48, 36, 48], material: stone }
                box keep_b place(base_box=keep_a, anchor=[0, 1, 0], align=[0, -1, 0]) {
                  size: [40, 32, 40], material: stone
                }
                box keep_c place(base_box=keep_b, anchor=[0, 1, 0], align=[0, -1, 0]) {
                  size: [32, 28, 32], material: stone
                }
              }
            }
            """
        )
        colossal_scene = compile_geometry(colossal_prog)
        auto_uv = allocate_uv_layout(colossal_prog, colossal_scene)
        self.assertGreaterEqual(auto_uv["width"], 1024)
        self.assertEqual(auto_uv["width"] & (auto_uv["width"] - 1), 0)

        with self.assertRaisesRegex(ValueError, "cannot fit atlas width"):
            allocate_uv_layout(colossal_prog, colossal_scene, max_width=128)

    def test_flagship_sculpted_elder_wyvern_and_colossal_walking_citadel_end_to_end(self) -> None:
        parity = verify_torch_se3_parity()
        if parity["torch_available"]:
            self.assertTrue(parity["passed"])

        # 1. Flagship Sculpted Elder Wyvern (62 boxes, 30 bones)
        wyvern_ws = build_sculpted_elder_wyvern_world()
        wyvern_scene = compile_geometry(wyvern_ws.program)
        self.assertGreaterEqual(len(wyvern_scene.boxes), 60)
        self.assertGreaterEqual(len(wyvern_scene.bones), 28)

        w_sculpt = audit_scene(wyvern_scene, enforce_physical_contact=True)
        w_strict = audit_scene(wyvern_scene, audit_profile="strict", enforce_physical_contact=True)
        self.assertTrue(w_sculpt["passed"], msg=str(w_sculpt))
        self.assertFalse(w_strict["passed"])
        self.assertGreaterEqual(len(w_sculpt["sculpted_overlap_pairs"]), 40)

        for clip in wyvern_ws.program.behavior.clips:
            rep = audit_behavior_clip(wyvern_ws.program, clip.name, num_frames=12)
            self.assertTrue(rep["passed"], msg=str(rep))
            self.assertLess(rep["max_stance_slip_units"], 0.02)

        w_crit = analyze_artist_project(wyvern_ws.program)
        self.assertAlmostEqual(w_crit.readiness_score, 1.0, places=4)

        w_atlas, w_png = compile_vanilla_atlas(wyvern_ws)
        self.assertGreater(len(w_png), 1024)
        w_bed = export_bedrock(wyvern_scene, atlas=w_atlas, program=wyvern_ws.program)
        self.assertLessEqual(w_bed["serialized_vertex_max_error"], 1e-6)
        export_bbmodel(wyvern_scene, atlas=w_atlas, program=wyvern_ws.program)
        export_gltf_pbr(wyvern_ws)
        export_labpbr_1_3(wyvern_ws)

        # 2. Flagship Colossal Walking Citadel Vehicle (74 boxes, 38 bones)
        citadel_ws = build_colossal_walking_citadel_world()
        citadel_scene = compile_geometry(citadel_ws.program)
        self.assertGreaterEqual(len(citadel_scene.boxes), 70)
        self.assertGreaterEqual(len(citadel_scene.bones), 35)

        c_sculpt = audit_scene(citadel_scene, enforce_physical_contact=True)
        c_strict = audit_scene(citadel_scene, audit_profile="strict", enforce_physical_contact=True)
        self.assertTrue(c_sculpt["passed"], msg=str(c_sculpt))
        self.assertFalse(c_strict["passed"])
        self.assertGreaterEqual(len(c_sculpt["sculpted_overlap_pairs"]), 35)

        for clip in citadel_ws.program.behavior.clips:
            rep = audit_behavior_clip(citadel_ws.program, clip.name, num_frames=12)
            self.assertTrue(rep["passed"], msg=str(rep))
            self.assertLess(rep["max_stance_slip_units"], 0.02)

        c_crit = analyze_artist_project(citadel_ws.program)
        self.assertAlmostEqual(c_crit.readiness_score, 1.0, places=4)

        c_atlas, c_png = compile_vanilla_atlas(citadel_ws)
        self.assertGreater(len(c_png), 1024)
        c_gecko = export_geckolib(citadel_scene, atlas=c_atlas, program=citadel_ws.program)
        self.assertLessEqual(c_gecko["serialized_vertex_max_error"], 1e-6)
        export_bbmodel(citadel_scene, atlas=c_atlas, program=citadel_ws.program)
        export_gltf_pbr(citadel_ws)
        export_labpbr_1_3(citadel_ws)


if __name__ == "__main__":
    unittest.main()
