"""Universal Unified SO(3) / SE(3) / Sim(3) Biomechanical Solver Verification Suite.

Validates across 6 distinct real & imaginary creature body plans:
1. Humanoid (2 legs + 2 free arms + finger_flex anti-hyperextension + auto arm counter-swing)
2. Quadruped (in_place gait + Sim(3) muscle scale_bulge + head_stabilize + spring + .bbmodel/.animation.json export)
3. 8-legged Imaginary Arachnid with generic non-English bone names (geometry-first morphology + metachronal_wave + auto_bio_limits)
4. 0-legged Serpentine Sea-Wyrm (limbless undulatory serpentine_wave + gaze lock + in_place root_motion)
5. 4-winged / 2-headed Imaginary Seraph-Hydra (multi-pair aerodynamic figure-8 hover + multi-neck gaze lock + bilateral symmetry)
6. Exact Lark grammar <-> parser.py <-> format_program roundtrip parity on all new DSL constructs
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

try:
    from lark import Lark
except ImportError:
    Lark = None  # type: ignore[assignment,misc]
import numpy as np

from synccraft.backends import export_bbmodel, export_bedrock
from synccraft.dsl import DSLValidationError, format_program, parse_program
from synccraft.kinematics import (
    analyze_morphology,
    analyze_motion_capabilities,
    audit_behavior_clip,
    compile_geometry,
    evaluate_behavior_clip,
    infer_universal_bio_constraints,
)
from synccraft.shading.uv_packer import build_atlas

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = REPO_ROOT / "src" / "synccraft" / "dsl" / "grammar.lark"


class TestUnifiedBiomechanicalSolver(unittest.TestCase):
    def test_humanoid_finger_bio_joint_and_biped_in_place_arm_swing(self) -> None:
        dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  body: ramp(base_lch=[0.55, 0.08, 220], steps=4)
  limb: ramp(base_lch=[0.42, 0.06, 220], steps=4)
}
@rig {
  root bone pelvis {
    pivot_local: [0, 14, 0]
    box hip_core { size: [6, 3, 4], material: body }
    bone chest attach(parent_box=hip_core, anchor=[0, 1, 0], align=[0, -1, 0]) {
      bio_joint: spine_vertebra
      box torso_box { size: [7, 8, 4], material: body }
      bone neck attach(parent_box=torso_box, anchor=[0, 1, 0], align=[0, -1, 0]) {
        bio_joint: neck_vertebra
        box neck_box { size: [3, 2, 3], material: body }
        bone head attach(parent_box=neck_box, anchor=[0, 1, 0], align=[0, -1, 0]) {
          bio_joint: head_atlas
          box skull { size: [5, 5, 5], material: body }
        }
      }
      bone arm_l attach(parent_box=torso_box, anchor=[-1, 0.7, 0], align=[1, 1, 0]) mirror_x(arm_r) {
        bio_joint: ball_shoulder
        box upper_arm { size: [2.5, 5, 2.5], material: limb }
        bone forearm attach(parent_box=upper_arm, anchor=[0, -1, 0], align=[0, 1, 0]) {
          bio_joint: hinge_elbow
          box lower_arm { size: [2, 5, 2], material: limb }
          bone finger_index attach(parent_box=lower_arm, anchor=[0, -1, 0.4], align=[0, 1, 0]) {
            bio_joint: finger_flex
            box phalanx { size: [1, 2, 1], material: limb }
          }
        }
      }
    }
    bone leg_l attach(parent_box=hip_core, anchor=[-0.55, -1, 0], align=[0, 1, 0]) mirror_x(leg_r) {
      bio_joint: ball_hip
      box thigh { size: [2.5, 6.5, 2.5], material: limb }
      bone shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [8, 0, 0]
        bio_joint: hinge_knee
        box calf { size: [2, 6.5, 2], material: limb }
      }
    }
  }
}
@behavior {
  clip walk_human {
    duration_s: 1.2, in_place: true, head_stabilize: 0.85, auto_bio_limits: true
    gait human_stride {
      pattern: biped_alternate,
      froude: 0.22,
      duty: 0.62,
      lift_ratio: 0.14,
      in_place: true,
      head_stabilize: 0.85
    }
  }
}
"""
        if Lark is not None:
            lark_parser = Lark(GRAMMAR_PATH.read_text(encoding="utf-8"), start="start")
            lark_parser.parse(dsl)
        prog = parse_program(dsl)
        formatted = format_program(prog)
        if Lark is not None:
            lark_parser.parse(formatted)
        self.assertEqual(parse_program(formatted), prog)

        # Capabilities & geometry-first morphology must find 2 grounded legs + 2 free arms + terminal digits!
        morph = analyze_morphology(prog)
        self.assertEqual(len(morph.limbs), 2)
        self.assertEqual(len(morph.free_limbs), 2)
        self.assertGreaterEqual(len(morph.free_limbs[0].digits), 1)
        caps = analyze_motion_capabilities(prog)
        self.assertEqual(caps["locomotion_class"], "biped")
        self.assertTrue(caps["can_arm_counter_swing"])

        # Biological anti-hyperextension check: bending finger_index backward -40 deg must fail DSL validation!
        bad_finger_dsl = dsl.replace(
            "@behavior {",
            "@behavior {\n  clip bad_finger { duration_s: 1.0 track finger_index: hinge(axis=x, amp_deg=-40, freq_hz=1) }\n",
        )
        with self.assertRaises(DSLValidationError):
            parse_program(bad_finger_dsl)

        # Evaluate biped walk: verify in_place (0 Z-drift), foot lock (< 0.02u), and automatic arm counter-swing!
        frames = evaluate_behavior_clip(prog, "walk_human", num_frames=16)
        max_z_drift = max(abs(fr.root_translation_delta[2]) for fr in frames)
        self.assertLessEqual(max_z_drift, 1e-4)
        self.assertGreater(abs(frames[0].root_motion_velocity[2]), 1.0)
        arm_l_excursion = max(abs(fr.pose_euler_deltas.get("arm_l", (0, 0, 0))[0]) for fr in frames)
        self.assertGreater(arm_l_excursion, 8.0, "Free humanoid arms must automatically counter-swing!")
        aud = audit_behavior_clip(prog, "walk_human", num_frames=16)
        self.assertTrue(aud["passed"], msg=f"Humanoid in-place gait failed: {aud}")

    def test_quadruped_in_place_sim3_muscle_bulge_and_exports(self) -> None:
        dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  coat: ramp(base_lch=[0.56, 0.12, 50], steps=4)
  dark: ramp(base_lch=[0.28, 0.04, 240], steps=4)
}
@rig {
  root bone pelvis {
    pivot_local: [0, 8.35, 0]
    box chest { size: [4, 4, 6], material: coat }
    bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {
      bio_joint: spine_vertebra
      box hips { size: [3.5, 3.5, 5], material: coat }
      bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.65, -0.5], align=[1, 1, 0]) mirror_x(hind_leg_r) {
        bio_joint: ball_hip
        box femur { size: [1.5, 3.5, 2], material: coat }
        bone shin attach(parent_box=femur, anchor=[0, -1, 0], align=[0, 1, 0]) {
          rest_euler: [10, 0, 0]
          bio_joint: hinge_knee
          box tibia { size: [1.5, 3.5, 1.5], material: dark }
        }
      }
      bone tail_1 attach(parent_box=hips, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {
        bio_joint: tail_vertebra
        box caudal_1 { size: [1.5, 1.5, 4.5], material: coat }
      }
    }
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.65, 0.6], align=[1, 1, 0]) mirror_x(fore_leg_r) {
      bio_joint: ball_shoulder
      box humerus { size: [1.5, 3.5, 2], material: coat }
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [-10, 0, 0]
        bio_joint: hinge_elbow
        box radius { size: [1.5, 3.5, 1.5], material: dark }
      }
    }
    bone neck attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, 0, -1], seat=0.25) {
      rest_euler: [-15, 0, 0]
      bio_joint: neck_vertebra
      box neck_seg { size: [2, 2, 3.5], material: coat }
      bone skull attach(parent_box=neck_seg, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {
        rest_euler: [12, 0, 0]
        bio_joint: head_atlas
        box snout { size: [2.5, 2, 3.5], material: coat }
      }
    }
  }
}
@muscles {
  muscle spine_breathe { origin_bone: pelvis, insertion_bone: lumbar, action_axis: x, moment_arm_deg: 6, belly_box: hips, bulge_inflate: 0.05, scale_bulge: 0.06 }
}
@kinematics {
  look_chain gaze { bones: [neck, skull], weight_decay: 0.72 }
  spring tail_1 { stiffness: 36, damping: 5.5 }
}
@behavior {
  clip prowl_inplace {
    duration_s: 1.6, in_place: true, head_stabilize: 0.85
    gait prowl_g {
      pattern: lateral_sequence,
      froude: 0.16,
      duty: 0.68,
      lift_ratio: 0.11,
      in_place: true,
      head_stabilize: 0.85
    }
    muscle_drive spine_breathe: activation(peak=0.85, freq_hz=0.625, baseline=0.1)
  }
}
"""
        prog = parse_program(dsl)
        aud = audit_behavior_clip(prog, "prowl_inplace", num_frames=16)
        self.assertTrue(aud["passed"], msg=f"Quadruped in-place gait failed: {aud}")
        self.assertLessEqual(aud["max_stance_slip_units"], 0.02)
        self.assertTrue(aud["root_motion"]["in_place"])

        # Verify Sim(3) scale channel + in_place root_motion exported in BOTH Bedrock AND .bbmodel!
        scene = compile_geometry(prog)
        atlas, _ = build_atlas(prog, scene)
        bed = export_bedrock(scene, program=prog, identifier="geometry.synccraft.quad", atlas=atlas)
        clip_bed = bed["animations"]["animations"]["animation.quad.prowl_inplace"]
        self.assertIn("scale", clip_bed["bones"]["lumbar"], "Bedrock export missing Sim(3) muscle scale channel!")
        self.assertTrue(clip_bed["synccraft_root_motion"]["in_place"])

        bb_prog = replace(prog, meta=replace(prog.meta, target_profile="bbmodel_5"))
        bb_scene = compile_geometry(bb_prog)
        bb = export_bbmodel(bb_scene, name="quad", atlas=atlas, program=bb_prog)
        clip_bb = bb["animations"][0]
        self.assertTrue(clip_bb["synccraft_root_motion"]["in_place"])
        lumbar_anim = next(a for a in clip_bb["animators"].values() if a.get("name") == "lumbar")
        self.assertTrue(any(kf["channel"] == "scale" for kf in lumbar_anim["keyframes"]))

    def test_universal_8_legged_alien_generic_names_metachronal_wave(self) -> None:
        # Zero English anatomy keywords ("leg", "wing", "tail") — pure 3D geometry & topology!
        dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  chitin: ramp(base_lch=[0.36, 0.08, 300], steps=4)
}
@rig {
  root bone core_0 {
    pivot_local: [0, 6.5, 0]
    box hull_0 { size: [5, 3, 14], material: chitin }
    bone stalk_a_l attach(parent_box=hull_0, anchor=[-1, -0.3, 0.72], align=[1, 0, 0]) mirror_x(stalk_a_r) {
      box seg_a1 { size: [4, 1.5, 1.5], material: chitin }
      bone seg_a2 attach(parent_box=seg_a1, anchor=[-1, -0.5, 0], align=[1, 1, 0]) {
        box seg_a2_b { size: [1.5, 5.5, 1.5], material: chitin }
      }
    }
    bone stalk_b_l attach(parent_box=hull_0, anchor=[-1, -0.3, 0.24], align=[1, 0, 0]) mirror_x(stalk_b_r) {
      box seg_b1 { size: [4, 1.5, 1.5], material: chitin }
      bone seg_b2 attach(parent_box=seg_b1, anchor=[-1, -0.5, 0], align=[1, 1, 0]) {
        box seg_b2_b { size: [1.5, 5.5, 1.5], material: chitin }
      }
    }
    bone stalk_c_l attach(parent_box=hull_0, anchor=[-1, -0.3, -0.24], align=[1, 0, 0]) mirror_x(stalk_c_r) {
      box seg_c1 { size: [4, 1.5, 1.5], material: chitin }
      bone seg_c2 attach(parent_box=seg_c1, anchor=[-1, -0.5, 0], align=[1, 1, 0]) {
        box seg_c2_b { size: [1.5, 5.5, 1.5], material: chitin }
      }
    }
    bone stalk_d_l attach(parent_box=hull_0, anchor=[-1, -0.3, -0.72], align=[1, 0, 0]) mirror_x(stalk_d_r) {
      box seg_d1 { size: [4, 1.5, 1.5], material: chitin }
      bone seg_d2 attach(parent_box=seg_d1, anchor=[-1, -0.5, 0], align=[1, 1, 0]) {
        box seg_d2_b { size: [1.5, 5.5, 1.5], material: chitin }
      }
    }
  }
}
@behavior {
  clip scuttle {
    duration_s: 1.2, in_place: true, auto_bio_limits: true
    gait alien_walk {
      pattern: metachronal_wave,
      froude: 0.14,
      duty: 0.70,
      lift_ratio: 0.12,
      in_place: true,
      auto_bio_limits: true
    }
  }
}
"""
        prog = parse_program(dsl)
        morph = analyze_morphology(prog)
        self.assertEqual(len(morph.limbs), 8, f"Expected 8 auto-detected limbs, got {len(morph.limbs)}")
        caps = analyze_motion_capabilities(prog)
        self.assertEqual(caps["recommended_gait_pattern"], "metachronal_wave")
        inferred = infer_universal_bio_constraints(prog)
        self.assertGreaterEqual(len(inferred), 16)
        aud = audit_behavior_clip(prog, "scuttle", num_frames=12)
        self.assertTrue(aud["passed"], msg=f"8-legged alien gait failed: {aud}")

    def test_legless_serpent_undulatory_wave(self) -> None:
        dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  scales: ramp(base_lch=[0.48, 0.13, 145], steps=4)
}
@rig {
  root bone body_0 {
    pivot_local: [0, 2, 0]
    box seg_0 { size: [4, 3, 5], material: scales }
    bone head attach(parent_box=seg_0, anchor=[0, 0.2, 1], align=[0, 0, -1], seat=0.1) {
      box skull { size: [4.5, 3, 5], material: scales }
    }
    chain body_aft attach(parent_box=seg_0, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.1) segments=5 taper=0.86 {
      axis: -z
      box s_b { size: [3.5, 2.5, 4.5], material: scales }
    }
  }
}
@behavior {
  clip slither {
    duration_s: 1.4, in_place: true, head_stabilize: 0.85, auto_bio_limits: true
    gait wyrm_slither {
      pattern: serpentine_wave,
      froude: 0.20,
      spine_wave_deg: 9.0,
      in_place: true,
      head_stabilize: 0.85,
      auto_bio_limits: true
    }
  }
}
"""
        prog = parse_program(dsl)
        caps = analyze_motion_capabilities(prog)
        self.assertEqual(caps["locomotion_class"], "limbless_undulatory")
        self.assertEqual(caps["recommended_gait_pattern"], "serpentine_wave")
        frames = evaluate_behavior_clip(prog, "slither", num_frames=16)
        self.assertGreater(abs(frames[0].root_motion_velocity[2]), 2.0)
        # Verify traveling lateral S-wave across body_aft segments
        yaw_excursion = max(abs(fr.pose_euler_deltas.get("body_aft_2", (0, 0, 0))[1]) for fr in frames)
        self.assertGreater(yaw_excursion, 8.0)
        aud = audit_behavior_clip(prog, "slither", num_frames=16)
        self.assertTrue(aud["passed"], msg=f"Serpentine undulatory clip failed: {aud}")

    def test_four_winged_two_headed_imaginary_seraph_hydra_hover(self) -> None:
        dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  gold: ramp(base_lch=[0.74, 0.14, 85], steps=4)
  plume: ramp(base_lch=[0.88, 0.04, 240], steps=4)
}
@rig {
  root bone torso {
    pivot_local: [0, 14, 0]
    box core { size: [6, 6, 16], material: gold }
    bone neck_l attach(parent_box=core, anchor=[-0.85, 0.6, 1], align=[0, 0, -1], seat=0.1) mirror_x(neck_r) {
      rest_euler: [0, -16, 0]
      box n_seg { size: [2, 2, 5], material: gold }
      bone head_seg attach(parent_box=n_seg, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.1) {
        box skull_b { size: [2.5, 2.5, 3.5], material: gold }
      }
    }
    bone wing_fore_l attach(parent_box=core, anchor=[-1, 0.7, 0.72], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(wing_fore_r) {
      rest_euler: [0, 0, -18]
      box wf_arm { size: [8, 1.5, 2], material: gold }
      plane wf_sail place(base_box=wf_arm, anchor=[0, 0, -1], align=[0, 0, 1]) { size: [7, 4], normal: y, material: plume }
      bone wf_outer attach(parent_box=wf_arm, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        box wf_tip { size: [7, 1, 1.5], material: gold }
        plane wf_tip_sail place(base_box=wf_tip, anchor=[0, 0, -1], align=[0, 0, 1]) { size: [6, 3.5], normal: y, material: plume }
      }
    }
    bone wing_aft_l attach(parent_box=core, anchor=[-1, 0.45, -0.55], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(wing_aft_r) {
      rest_euler: [0, 0, -12]
      box wa_arm { size: [7, 1.5, 2], material: gold }
      plane wa_sail place(base_box=wa_arm, anchor=[0, 0, -1], align=[0, 0, 1]) { size: [6, 3.5], normal: y, material: plume }
      bone wa_outer attach(parent_box=wa_arm, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        box wa_tip { size: [6, 1, 1.5], material: gold }
      }
    }
    bone tail_1 attach(parent_box=core, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.1) {
      box t1 { size: [2.5, 2.5, 6], material: gold }
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.1) {
        box t2 { size: [2, 2, 6], material: gold }
      }
    }
  }
}
@behavior {
  clip seraph_hover {
    duration_s: 1.6, loop: true
    flight multi_wing_hover {
      mode: hover,
      flap_hz: 0.625,
      stroke_deg: 28,
      sweep_deg: 10,
      pronation_deg: 12,
      elbow_fold_deg: 14,
      lift_recoil_u: 2.6,
      body_pitch_deg: 12,
      head_stabilize: 0.85,
      pair_phase_deg: 45,
      in_place: true
    }
  }
}
"""
        if Lark is not None:
            lark_parser = Lark(GRAMMAR_PATH.read_text(encoding="utf-8"), start="start")
            lark_parser.parse(dsl)
        prog = parse_program(dsl)
        formatted = format_program(prog)
        if Lark is not None:
            lark_parser.parse(formatted)
        self.assertEqual(parse_program(formatted), prog)

        morph = analyze_morphology(prog)
        self.assertEqual(len(morph.wings), 4, f"Expected 4 wings (2 pairs), got {len(morph.wings)}")
        self.assertEqual(len(morph.neck_chains), 2, f"Expected 2 hydra necks, got {len(morph.neck_chains)}")

        aud = audit_behavior_clip(prog, "seraph_hover", num_frames=16)
        self.assertTrue(aud["passed"], msg=f"4-winged seraph-hydra hover failed: {aud}")
        fl_aud = aud["flight_audit"]
        self.assertTrue(fl_aud["in_place"])
        self.assertLessEqual(fl_aud["max_bilateral_asymmetry_units"], 0.1)
        self.assertLessEqual(fl_aud["max_root_y_units"], 3.5)  # strictly in-place COM bob, NOT +35u!


if __name__ == "__main__":
    unittest.main()
