"""Unit tests for SyncCraft v20 native underwater buoyancy & musculoskeletal swim positioning (`swim_solver.py`)."""
from __future__ import annotations

import unittest

from synccraft.backends.bedrock_gecko_exporter import export_bedrock
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.musculoskeletal import audit_behavior_clip, evaluate_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry

SWIM_TEST_DSL = """
@meta {
  archetype: entity,
  target_profile: bedrock_entity,
  dsl_version: "4.1",
  audit_profile: sculpted,
  units_per_block: 16,
  texel_density: 16
}

@layout {
  budget max_boxes = 40
  budget max_bones = 24
  envelope grounded = true
}

@palette {
  hide: ramp(base_lch=[0.46, 0.11, 228], steps=5, hue_shift=[-14, 16])
}

@rig {
  root bone pelvis_root {
    box pelvis place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[0, 8, 0]) {
      size: [8, 6, 10],
      material: hide
    }
    bone chest attach(parent_box=pelvis, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.15) {
      box chest_core place(anchor=[0, 0, 0], align=[0, 0, -1]) {
        size: [9, 7, 18],
        material: hide
      }
      bone neck_1 attach(parent_box=chest_core, anchor=[0, 0.1, 1], align=[0, 0, -1], seat=0.15) {
        box neck_core place(anchor=[0, 0, 0], align=[0, 0, -1]) {
          size: [5, 5, 8],
          material: hide
        }
        bone head attach(parent_box=neck_core, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.15) {
          box head_core place(anchor=[0, 0, 0], align=[0, 0, -1]) {
            size: [6, 5, 9],
            material: hide
          }
        }
      }
      bone arm_l_upper attach(parent_box=chest_core, anchor=[-1, -0.2, 0.3], align=[1, 0, 0], seat=0.1) mirror_x(arm_r_upper) {
        rest_euler: [0, 0, 20]
        box arm_u place(anchor=[0, 0, 0], align=[1, 0, 0]) {
          size: [6, 3, 3],
          material: hide
        }
        bone arm_l_fore attach(parent_box=arm_u, anchor=[-1, 0, 0], align=[0, 1, 0], seat=0.1) {
          box arm_f place(anchor=[0, 0, 0], align=[0, 1, 0]) {
            size: [3, 6, 3],
            material: hide
          }
          bone paw_l attach(parent_box=arm_f, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.05) {
            box paw_b place(anchor=[0, 0, 0], align=[0, 1, 0]) {
              size: [4, 2, 5],
              material: hide
            }
          }
        }
      }
    }
    bone leg_l_thigh attach(parent_box=pelvis, anchor=[-1, -0.2, -0.2], align=[1, 0, 0], seat=0.1) mirror_x(leg_r_thigh) {
      rest_euler: [0, 0, 20]
      box leg_u place(anchor=[0, 0, 0], align=[1, 0, 0]) {
        size: [6, 3, 3],
        material: hide
      }
      bone leg_l_shin attach(parent_box=leg_u, anchor=[-1, 0, 0], align=[0, 1, 0], seat=0.1) {
        box leg_f place(anchor=[0, 0, 0], align=[0, 1, 0]) {
          size: [3, 6, 3],
          material: hide
        }
        bone foot_l attach(parent_box=leg_f, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.05) {
          box foot_b place(anchor=[0, 0, 0], align=[0, 1, 0]) {
            size: [4, 2, 5],
            material: hide
          }
        }
      }
    }
    bone tail_1 attach(parent_box=pelvis, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.15) {
      box t1 place(anchor=[0, 0, 0], align=[0, 0, 1]) {
        size: [6, 5, 10],
        material: hide
      }
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.15) {
        box t2 place(anchor=[0, 0, 0], align=[0, 0, 1]) {
          size: [4, 4, 10],
          material: hide
        }
        bone tail_3 attach(parent_box=t2, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.15) {
          box t3 place(anchor=[0, 0, 0], align=[0, 0, 1]) {
            size: [3, 4, 10],
            material: hide
          }
        }
      }
    }
  }
}

@muscles {
  muscle thorax_diaphragm {
    origin_bone: pelvis_root,
    insertion_bone: chest,
    action_axis: x,
    moment_arm_deg: 2.0,
    scale_bulge: 0.03
  }
}

@behavior {
  clip swim_cruise {
    duration_s: 2.0
    loop: true
    in_place: true
    auto_rig: true
    auto_bio_limits: true

    swim test_cruise {
      mode: leviathan_cruise
      freq_hz: 0.5
      wave_amp_deg: 14.0
      buoyancy_y: 12.0
      heave_amp_u: 1.5
      body_pitch_deg: -4.0
      head_stabilize: 0.80
      limb_mode: hydro_tuck
      limb_tuck: 0.85
      myomere_bulge: 0.80
      fin_lag_deg: 12.0
      in_place: true
      auto_bio_limits: true
      ground_clearance: true
    }
    muscle_drive thorax_diaphragm: activation(peak=0.70, freq_hz=0.5, baseline=0.15)
  }
}
"""


class TestSwimSolver(unittest.TestCase):
    def test_underwater_buoyancy_bilateral_symmetry_myomere_and_export(self) -> None:
        prog = parse_program(SWIM_TEST_DSL)
        rest_scene = compile_geometry(prog, validate=False)

        frames = evaluate_behavior_clip(prog, "swim_cruise", num_frames=16)
        self.assertEqual(len(frames), 16)

        # 1. Neutral buoyancy + in-place Z=0
        for fr in frames:
            self.assertAlmostEqual(float(fr.root_translation_delta[2]), 0.0, places=5)
            self.assertGreater(float(fr.root_translation_delta[1]), 8.0)

        # 2. Exact bilateral limb symmetry across full swim cycle (mean left Y == mean right Y!)
        fl_ys, fr_ys, hl_ys, hr_ys = [], [], [], []
        for fr in frames:
            fl_ys.append(float(fr.scene.bone_by_name["paw_l"].pivot_world[1]))
            fr_ys.append(float(fr.scene.bone_by_name["arm_r_upper__paw_l"].pivot_world[1]))
            hl_ys.append(float(fr.scene.bone_by_name["foot_l"].pivot_world[1]))
            hr_ys.append(float(fr.scene.bone_by_name["leg_r_thigh__foot_l"].pivot_world[1]))
        self.assertAlmostEqual(sum(fl_ys) / len(fl_ys), sum(fr_ys) / len(fr_ys), delta=0.5)
        self.assertAlmostEqual(sum(hl_ys) / len(hl_ys), sum(hr_ys) / len(hr_ys), delta=0.5)

        # 3. Full swim audit contract pass (buoyancy, myomere Sim(3), in-place, zero collisions)
        rep = audit_behavior_clip(prog, "swim_cruise", max_frames=96, audit_profile="sculpted")
        self.assertTrue(rep["passed"], f"swim_cruise failed: {rep}")
        sa = rep["swim_audit"]
        self.assertTrue(sa["passed"])
        self.assertTrue(all(sa["checks"].values()))

        # 4. Bedrock export speed_blocks_per_s matches swim_audit (no double /16 division!)
        bed = export_bedrock(rest_scene, program=prog, num_animation_frames=12, animation_mode="keyframes")
        anim_key = next(iter(bed["animations"]["animations"]))
        rm = bed["animations"]["animations"][anim_key]["synccraft_root_motion"]
        self.assertTrue(rm["in_place"])
        self.assertAlmostEqual(
            float(rm["speed_blocks_per_s"]),
            float(sa["root_motion"]["speed_blocks_per_s"]),
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
