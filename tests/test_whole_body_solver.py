"""Comprehensive tests for SyncCraft v27 Industry-Grade Full-Body Biomechanical Solver.

Proves full anatomical coverage (torso/spine, COM/angular-momentum balance, serpentine
S-neck Spline-IK, full pectoral-to-phalanx wings, limbs -> foot-roll -> multi-phalanx
hands/feet digits, and whole-body Sim(3) antagonist muscle bulge) across:
1. Humanoid with multi-phalanx fingers + opposable thumb + foot-roll + limb Sim(3) bulge.
2. Long-necked Winged Elder Dragon with Spline-IK aiming, pectoral downstroke bulge,
   aerial talon curl, COM/tail counterweight, rigid wing_anchor_locked, and DSL roundtrip.
3. Brand-new Imaginary Non-English 6-Legged 2-Winged Chimera assembled purely from geometry.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from synccraft.backends.bbmodel5_exporter import export_bbmodel
from synccraft.backends.bedrock_gecko_exporter import export_bedrock
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.musculoskeletal import audit_behavior_clip, evaluate_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.kinematics.whole_body_solver import (
    auto_assemble_biomechanical_rig,
    solve_com_balance_and_counterweight,
    solve_spine_neck_spline_ik,
)
from synccraft.shading.uv_packer import build_atlas


HUMANOID_FULL_BODY_DSL = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  body: ramp(base_lch=[0.55, 0.08, 220], steps=4)
  limb: ramp(base_lch=[0.45, 0.09, 230], steps=4)
  hand: ramp(base_lch=[0.70, 0.07, 65], steps=4)
  accent: ramp(base_lch=[0.75, 0.12, 85], steps=4)
}
@rig {
  root bone pelvis {
    pivot_local: [0, 18, 0]
    box pelvis_box { size: [8, 4, 5], material: body }
    bone torso attach(parent_box=pelvis_box, anchor=[0, 1, 0], align=[0, -1, 0]) {
      bio_joint: spine_vertebra
      box chest_box { size: [9, 10, 5], material: body }
      bone neck attach(parent_box=chest_box, anchor=[0, 1, 0], align=[0, -1, 0]) {
        bio_joint: neck_vertebra
        box neck_box { size: [3, 2.5, 3], material: hand }
        bone head attach(parent_box=neck_box, anchor=[0, 1, 0], align=[0, -1, 0]) {
          bio_joint: head_atlas
          box head_box { size: [6, 6, 6], material: hand }
        }
      }
      bone arm_l attach(parent_box=chest_box, anchor=[-1, 0.7, 0], align=[1, 1, 0], offset=[-0.5, 0, 0]) mirror_x(arm_r) {
        bio_joint: ball_shoulder
        box upper_arm { size: [3, 6, 3], material: limb }
        bone forearm_l attach(parent_box=upper_arm, anchor=[0, -1, 0], align=[0, 1, 0]) {
          bio_joint: hinge_elbow
          box lower_arm { size: [2.5, 5.5, 2.5], material: limb }
          bone hand_l attach(parent_box=lower_arm, anchor=[0, -1, 0], align=[0, 1, 0]) {
            bio_joint: wrist_ankle
            box palm_l { size: [2.5, 2, 2.5], material: hand }
            bone finger_index_p attach(parent_box=palm_l, anchor=[0, -1, 0.6], align=[0, 1, 0]) {
              bio_joint: finger_proximal
              box f_idx_1 { size: [0.5, 1.5, 0.5], material: hand }
              bone finger_index_d attach(parent_box=f_idx_1, anchor=[0, -1, 0], align=[0, 1, 0]) {
                bio_joint: finger_flex
                box f_idx_2 { size: [0.5, 1.0, 0.5], material: hand }
              }
            }
            bone finger_mid_p attach(parent_box=palm_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
              bio_joint: finger_proximal
              box f_mid_1 { size: [0.5, 1.5, 0.5], material: hand }
              bone finger_mid_d attach(parent_box=f_mid_1, anchor=[0, -1, 0], align=[0, 1, 0]) {
                bio_joint: finger_flex
                box f_mid_2 { size: [0.5, 1.0, 0.5], material: hand }
              }
            }
            bone thumb_p attach(parent_box=palm_l, anchor=[1, -0.4, 0.8], align=[-1, 1, 0]) {
              bio_joint: thumb_saddle
              box thumb_1 { size: [0.5, 1.0, 0.5], material: accent }
              bone thumb_d attach(parent_box=thumb_1, anchor=[0, -1, 0], align=[0, 1, 0]) {
                bio_joint: finger_flex
                box thumb_2 { size: [0.5, 1.0, 0.5], material: accent }
              }
            }
          }
        }
      }
    }
    bone thigh_l attach(parent_box=pelvis_box, anchor=[-0.55, -1, 0], align=[0, 1, 0], offset=[-0.5, 0, 0]) mirror_x(thigh_r) {
      bio_joint: ball_hip
      box thigh_box { size: [3.5, 8, 3.5], material: limb }
      bone shin_l attach(parent_box=thigh_box, anchor=[0, -1, 0], align=[0, 1, 0]) {
        bio_joint: hinge_knee
        box shin_box { size: [3, 6.5, 3], material: limb }
        bone foot_l attach(parent_box=shin_box, anchor=[0, -1, 0], align=[0, 1, -0.3]) {
          bio_joint: wrist_ankle
          box foot_box { size: [3, 1.5, 4.5], material: accent }
          bone toe_l attach(parent_box=foot_box, anchor=[0, -0.4, 1], align=[0, -0.4, -1]) {
            bio_joint: finger_flex
            box toe_box { size: [2.5, 1.0, 1.5], material: accent }
          }
        }
      }
    }
  }
}
@kinematics {
  ik_chain leg_l { bones: [thigh_l, shin_l], effector: [0, -8.0, 0], pole: [-6, 8, 12] }
  ik_chain leg_r { bones: [thigh_r, thigh_r__shin_l], effector: [0, -8.0, 0], pole: [6, 8, 12] }
}
@behavior {
  clip walk_auto {
    duration_s: 1.2, in_place: true, auto_rig: true
    gait biped_walk {
      pattern: biped_alternate, froude: 0.20, duty: 0.64,
      lift_ratio: 0.14, heave_ratio: 0.03, export_root_motion: true,
      in_place: true, foot_roll: 0.75, digit_articulation: 0.85,
      com_balance: 0.45, antagonist_bulge: 0.70
    }
  }
  clip fist_grasp {
    duration_s: 0.8, loop: true, in_place: true, auto_rig: true, digit_grasp: 0.90, antagonist_bulge: 0.75
    track arm_l: wave(axis=x, amp_deg=24, freq_hz=1.25)
    track forearm_l: wave(axis=x, amp_deg=35, freq_hz=1.25, bias_deg=-35)
  }
}
"""


SERPENTINE_DRAGON_FULL_BODY_DSL = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  ivory: ramp(base_lch=[0.85, 0.04, 85], steps=4)
  gold: ramp(base_lch=[0.72, 0.11, 80], steps=4)
  crimson: ramp(base_lch=[0.42, 0.16, 20], steps=4)
  claw: ramp(base_lch=[0.28, 0.04, 300], steps=4)
}
@rig {
  root bone pelvis_root {
    pivot_local: [0, 16, 0]
    box pelvis_core { size: [10, 7, 10], material: ivory }
    bone chest attach(parent_box=pelvis_core, anchor=[0, 0.2, 1], align=[0, 0.2, -1]) {
      bio_joint: spine_vertebra
      box chest_core { size: [12, 9, 12], material: ivory }
      bone neck_1 attach(parent_box=chest_core, anchor=[0, 0.4, 1], align=[0, 0, -1]) {
        rest_euler: [-18, 0, 0]
        bio_joint: neck_vertebra
        box neck_1_box { size: [5, 5, 7], material: ivory }
        bone neck_2 attach(parent_box=neck_1_box, anchor=[0, 0, 1], align=[0, 0, -1]) {
          rest_euler: [-10, 0, 0]
          bio_joint: neck_vertebra
          box neck_2_box { size: [4.5, 4.5, 7], material: ivory }
          bone neck_3 attach(parent_box=neck_2_box, anchor=[0, 0, 1], align=[0, 0, -1]) {
            rest_euler: [12, 0, 0]
            bio_joint: neck_vertebra
            box neck_3_box { size: [4, 4, 6], material: ivory }
            bone head attach(parent_box=neck_3_box, anchor=[0, 0.1, 1], align=[0, 0.1, -1]) {
              rest_euler: [16, 0, 0]
              bio_joint: head_atlas
              box cranium { size: [5.5, 4.5, 8], material: ivory }
              bone jaw attach(parent_box=cranium, anchor=[0, -0.8, 0.2], align=[0, 1, -0.5]) {
                bio_joint: jaw_hinge
                box jaw_box { size: [4.5, 2, 6.5], material: gold }
                bone tongue_base attach(parent_box=jaw_box, anchor=[0, 0.8, -0.2], align=[0, -1, -1]) {
                  box tongue_box { size: [2, 1, 4], material: crimson }
                }
              }
            }
          }
        }
      }
      bone wing_l_upper attach(parent_box=chest_core, anchor=[-1, 0.7, 0.1], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(wing_r_upper) {
        rest_euler: [0, 0, -24]
        bio_joint: wing_shoulder
        box wing_hum { size: [12, 2.5, 3], material: ivory }
        bone wing_l_fore attach(parent_box=wing_hum, anchor=[-1, 0, 0], align=[1, 0, 0]) {
          rest_euler: [0, 10, 8]
          bio_joint: wing_elbow
          box wing_rad { size: [14, 2, 2.5], material: ivory }
          bone wing_l_f4 attach(parent_box=wing_rad, anchor=[-0.5, 0, -1], align=[0, 0, 1]) {
            bio_joint: wing_anchor_locked
            box f4_spar { size: [1.5, 1.5, 10], material: ivory }
            plane wing_bay_4 place(base_box=f4_spar, anchor=[0, 0, 0], align=[0, 0, 0]) { size: [12, 10], normal: y, material: crimson }
          }
          bone wing_l_wrist attach(parent_box=wing_rad, anchor=[-1, 0, 0], align=[1, 0, 0]) {
            bio_joint: wing_wrist
            box wing_carp { size: [3, 2, 3], material: gold }
            bone wing_l_f1 attach(parent_box=wing_carp, anchor=[-1, 0, 0.3], align=[1, 0, 0]) {
              bio_joint: wing_finger
              box f1_spar { size: [18, 1.5, 1.5], material: ivory }
              plane wing_bay_1 place(base_box=f1_spar, anchor=[0, 0, -1], align=[0, 0, 1]) { size: [18, 12], normal: y, material: crimson }
            }
            bone wing_l_f2 attach(parent_box=wing_carp, anchor=[-0.6, 0, -0.5], align=[0.7, 0, 0.7]) {
              bio_joint: wing_finger
              box f2_spar { size: [14, 1.5, 1.5], material: ivory }
            }
          }
        }
      }
      bone arm_l_upper attach(parent_box=chest_core, anchor=[-1, -0.6, 0.5], align=[0.5, 1, 0], offset=[-0.5, 0, 0]) mirror_x(arm_r_upper) {
        rest_euler: [12, 0, 0]
        bio_joint: ball_shoulder
        box hum_l { size: [4, 7.5, 4], material: ivory }
        bone arm_l_fore attach(parent_box=hum_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
          rest_euler: [-24, 0, 0]
          bio_joint: hinge_elbow
          box fore_l { size: [3.5, 6.5, 3.5], material: ivory }
          bone paw_l attach(parent_box=fore_l, anchor=[0, -1, 0], align=[0, 1, -0.3]) {
            rest_euler: [12, 0, 0]
            bio_joint: wrist_ankle
            box paw_pad { size: [4.5, 2, 4.5], material: gold }
            bone claw_l_1 attach(parent_box=paw_pad, anchor=[-0.6, -0.3, 1], align=[0, 0, -1]) {
              bio_joint: finger_proximal
              box claw_1_p { size: [1, 1, 2], material: claw }
              bone claw_l_1_tip attach(parent_box=claw_1_p, anchor=[0, 0, 1], align=[0, 0, -1]) {
                bio_joint: finger_flex
                box claw_1_d { size: [1, 1, 1.5], material: claw }
              }
            }
            bone claw_l_2 attach(parent_box=paw_pad, anchor=[0.6, -0.3, 1], align=[0, 0, -1]) {
              bio_joint: finger_proximal
              box claw_2_p { size: [1, 1, 2], material: claw }
            }
          }
        }
      }
    }
    bone leg_l_thigh attach(parent_box=pelvis_core, anchor=[-1, -0.5, -0.3], align=[0.5, 1, 0], offset=[-0.5, 0, 0]) mirror_x(leg_r_thigh) {
      rest_euler: [-15, 0, 0]
      bio_joint: ball_hip
      box thigh_box { size: [4.5, 8, 5], material: ivory }
      bone leg_l_shin attach(parent_box=thigh_box, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [30, 0, 0]
        bio_joint: hinge_knee
        box shin_box { size: [3.5, 7, 3.5], material: ivory }
        bone foot_l attach(parent_box=shin_box, anchor=[0, -1, 0], align=[0, 1, -0.3]) {
          rest_euler: [-15, 0, 0]
          bio_joint: wrist_ankle
          box foot_pad { size: [4.5, 2, 5], material: gold }
          bone talon_l_1 attach(parent_box=foot_pad, anchor=[0, -0.3, 1], align=[0, 0, -1]) {
            bio_joint: finger_flex
            box talon_1_b { size: [1.5, 1, 2], material: claw }
          }
        }
      }
    }
    bone tail_1 attach(parent_box=pelvis_core, anchor=[0, 0, -1], align=[0, 0, 1]) {
      bio_joint: tail_vertebra
      box t1 { size: [5, 5, 10], material: ivory }
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1]) {
        bio_joint: tail_vertebra
        box t2 { size: [4, 4, 10], material: ivory }
        bone tail_3 attach(parent_box=t2, anchor=[0, 0, -1], align=[0, 0, 1]) {
          bio_joint: tail_vertebra
          box t3 { size: [3, 3, 10], material: gold }
        }
      }
    }
  }
}
@behavior {
  clip hover_full {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true
    flight aero_hover {
      mode: hover, flap_hz: 0.625, stroke_deg: 24, sweep_deg: 10,
      elbow_fold_deg: 16, lift_recoil_u: 2.5, body_pitch_deg: 10,
      limb_tuck: 0.45, in_place: true, digit_articulation: 0.85,
      pectoral_bulge: 0.80, antagonist_bulge: 0.70, com_balance: 0.40
    }
  }
  clip serpent_aim_roar {
    duration_s: 1.2, loop: true, in_place: true, auto_rig: true,
    neck_aim: [18, 28, 36], spine_coil: 18, digit_grasp: 0.75, com_balance: 0.75, antagonist_bulge: 0.80
    track jaw: hinge(axis=x, amp_deg=36, freq_hz=0.8333)
    track wing_l_upper: wave(axis=y, amp_deg=20, freq_hz=0.8333, bias_deg=15)
  }
}
"""


NON_ENGLISH_IMAGINARY_CHIMERA_DSL = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  panzer: ramp(base_lch=[0.45, 0.08, 240], steps=4)
  schwinge: ramp(base_lch=[0.52, 0.14, 12], steps=4)
  spitze: ramp(base_lch=[0.78, 0.10, 88], steps=4)
}
@rig {
  root bone zentral_0 {
    pivot_local: [0, 10, 0]
    box rumpf_a { size: [8, 5, 16], material: panzer }
    bone hals_1 attach(parent_box=rumpf_a, anchor=[0, 0.3, 1], align=[0, 0, -1]) {
      box h1 { size: [4, 4, 5], material: panzer }
      bone hals_2 attach(parent_box=h1, anchor=[0, 0, 1], align=[0, 0, -1]) {
        box h2 { size: [4, 4, 5], material: spitze }
      }
    }
    bone schweif_1 attach(parent_box=rumpf_a, anchor=[0, 0, -1], align=[0, 0, 1]) {
      box s1 { size: [3.5, 3.5, 7], material: panzer }
      bone schweif_2 attach(parent_box=s1, anchor=[0, 0, -1], align=[0, 0, 1]) {
        box s2 { size: [2.5, 2.5, 7], material: spitze }
      }
    }
    bone fluegel_l attach(parent_box=rumpf_a, anchor=[-1, 0.8, 0.2], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(fluegel_r) {
      rest_euler: [0, 0, -18]
      box fl_a { size: [12, 1.5, 4], material: schwinge }
      bone fluegel_l_tip attach(parent_box=fl_a, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        box fl_b { size: [10, 1.5, 3.5], material: schwinge }
      }
    }
    bone stelze_a attach(parent_box=rumpf_a, anchor=[-1, -0.4, 0.7], align=[1, 0.2, 0], offset=[-0.5, 0, 0]) mirror_x(stelze_a_r) {
      rest_euler: [0, 15, 20]
      box st_a1 { size: [6, 2, 2], material: panzer }
      bone stelze_a_2 attach(parent_box=st_a1, anchor=[-1, 0, 0], align=[0, 1, 0]) {
        rest_euler: [0, 0, -20]
        box st_a2 { size: [2, 8.5, 2], material: spitze }
      }
    }
    bone stelze_b attach(parent_box=rumpf_a, anchor=[-1, -0.4, 0.0], align=[1, 0.2, 0], offset=[-0.5, 0, 0]) mirror_x(stelze_b_r) {
      rest_euler: [0, 0, 20]
      box st_b1 { size: [6, 2, 2], material: panzer }
      bone stelze_b_2 attach(parent_box=st_b1, anchor=[-1, 0, 0], align=[0, 1, 0]) {
        rest_euler: [0, 0, -20]
        box st_b2 { size: [2, 8.5, 2], material: spitze }
      }
    }
    bone stelze_c attach(parent_box=rumpf_a, anchor=[-1, -0.4, -0.7], align=[1, 0.2, 0], offset=[-0.5, 0, 0]) mirror_x(stelze_c_r) {
      rest_euler: [0, -15, 20]
      box st_c1 { size: [6, 2, 2], material: panzer }
      bone stelze_c_2 attach(parent_box=st_c1, anchor=[-1, 0, 0], align=[0, 1, 0]) {
        rest_euler: [0, 0, -20]
        box st_c2 { size: [2, 8.5, 2], material: spitze }
      }
    }
  }
}
@behavior {
  clip chimera_scuttle {
    duration_s: 1.2, loop: true, in_place: true, auto_rig: true
    gait hex_walk { pattern: hexapod_tripod, froude: 0.14, duty: 0.62, lift_ratio: 0.14, in_place: true, auto_bio_limits: true }
  }
  clip chimera_hover {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true
    flight hex_fly { mode: hover, flap_hz: 0.625, stroke_deg: 22, sweep_deg: 10, lift_recoil_u: 1.8, body_pitch_deg: 6, limb_tuck: 0.30, in_place: true }
  }
}
"""


class TestWholeBodyBiomechanicalSolver(unittest.TestCase):
    def test_auto_rig_humanoid_foot_roll_multi_phalanx_digits_grasp_and_limb_sim3_bulge(self) -> None:
        prog = parse_program(HUMANOID_FULL_BODY_DSL)
        sc = compile_geometry(prog)
        asm = auto_assemble_biomechanical_rig(prog, sc)

        # 1. Verify universal skeleton-to-rig assembly discovers legs, free arms, hands, feet, multi-phalanx digits & thumbs!
        self.assertEqual(len(asm.limb_assemblies), 4)
        free_arms = [la for la in asm.limb_assemblies if la.is_free]
        grounded_legs = [la for la in asm.limb_assemblies if not la.is_free]
        self.assertEqual(len(free_arms), 2)
        self.assertEqual(len(grounded_legs), 2)
        left_arm = next(la for la in free_arms if la.side == "l")
        self.assertEqual(left_arm.foot_hand_bone, "hand_l")
        self.assertGreaterEqual(len(left_arm.digit_chains), 2)
        self.assertEqual(len(left_arm.thumb_chains), 1)
        # Verify multi-phalanx chain length (proximal -> distal)
        self.assertTrue(any(len(dc) == 2 for dc in left_arm.digit_chains))
        left_leg = next(la for la in grounded_legs if la.side == "l")
        self.assertEqual(left_leg.foot_hand_bone, "foot_l")
        self.assertGreaterEqual(len(left_leg.digit_chains), 1)
        # Whole-body auto muscle groups span torso, neck, upper limbs & lower limbs!
        regions = {mg.region for mg in asm.auto_muscle_groups}
        self.assertIn("limb_upper", regions)
        self.assertIn("limb_lower", regions)
        self.assertIn("torso_core", regions)

        # 2. Verify walk_auto: foot-roll rocker, toe articulation, <0.02u slip, in_place, AND multi-bone Sim(3) limb bulge!
        frames = evaluate_behavior_clip(prog, "walk_auto", num_frames=16)
        foot_pitches = [abs(fr.pose_euler_deltas.get("foot_l", (0.0, 0.0, 0.0))[0]) for fr in frames]
        toe_flexes = [fr.pose_euler_deltas.get("toe_l", (0.0, 0.0, 0.0))[0] for fr in frames]
        self.assertGreater(max(foot_pitches), 2.5, "Expected active reverse-foot rocker pitch during walk")
        self.assertGreater(max(toe_flexes), 2.0, "Expected toe articulation during swing phase")
        self.assertGreaterEqual(min(toe_flexes), -5.5, "Toes must NEVER hyperextend backward!")

        # Check Sim(3) muscle bulge appears on thighs/arms and exports directly to .bbmodel & Bedrock!
        bulged_bones = {b for fr in frames for b, s in fr.pose_scale_factors.items() if any(abs(v - 1.0) > 1e-4 for v in s)}
        self.assertTrue(any("thigh" in b or "arm" in b for b in bulged_bones), f"Expected limb Sim(3) bulge, got {bulged_bones}")

        aud = audit_behavior_clip(prog, "walk_auto", max_frames=240)
        self.assertTrue(aud["passed"], f"Humanoid auto_rig walk failed: {aud}")
        self.assertEqual(aud["total_collision_events"], 0)
        self.assertLess(aud["max_stance_slip_units"], 0.02)

        # 3. Verify fist_grasp: coordinated proximal -> distal curl + thumb opposition, ZERO backward hyperextension!
        grasp_frames = evaluate_behavior_clip(prog, "fist_grasp", num_frames=8)
        f0 = grasp_frames[0]
        p_idx = f0.pose_euler_deltas["finger_index_p"][0]
        d_idx = f0.pose_euler_deltas["finger_index_d"][0]
        th_p = f0.pose_euler_deltas["thumb_p"]
        self.assertGreater(p_idx, 18.0)
        self.assertGreater(d_idx, 24.0)
        self.assertGreater(abs(th_p[1]), 8.0, "Opposable thumb should oppose laterally during grasp")
        self.assertGreater(f0.pose_scale_factors["arm_l"][0], 1.005, "Biceps on arm_l should bulge radially in Sim(3)")

        # Export verification: .bbmodel & Bedrock carry multi-bone Sim(3) scale channels!
        atlas, _ = build_atlas(prog, sc)
        bb = export_bbmodel(sc, name="humanoid_v27", atlas=atlas, program=prog, num_animation_frames=12)
        br = export_bedrock(sc, program=prog, atlas=atlas, num_animation_frames=12)
        walk_bb = next(a for a in bb["animations"] if a["name"] == "walk_auto")
        scale_bones_bb = {
            ani["name"]
            for ani in walk_bb["animators"].values()
            if any(kf["channel"] == "scale" for kf in ani["keyframes"])
        }
        self.assertTrue(any("thigh" in b or "arm" in b for b in scale_bones_bb))
        walk_br_key = next(k for k in br["animations"]["animations"] if k.endswith(".walk_auto"))
        walk_br = br["animations"]["animations"][walk_br_key]["bones"]
        self.assertTrue(any("scale" in bdata for bname, bdata in walk_br.items() if "thigh" in bname or "arm" in bname))

    def test_serpentine_winged_elder_dragon_spline_ik_neck_pectoral_wing_talons_and_com_balance(self) -> None:
        prog = parse_program(SERPENTINE_DRAGON_FULL_BODY_DSL)
        # Verify exact DSL format -> parse roundtrip of all new v27 attributes!
        reparsed = parse_program(format_program(prog))
        self.assertEqual(format_program(prog), format_program(reparsed))

        sc = compile_geometry(prog)
        asm = auto_assemble_biomechanical_rig(prog, sc)

        # Verify full anatomical assembly: 4-bone neck + jaw + tongue, 2 wings + locked inner spar, 4 limbs + claws
        self.assertEqual(len(asm.neck_assemblies), 1)
        self.assertEqual(len(asm.neck_assemblies[0].chain), 4)
        self.assertIn("jaw", asm.neck_assemblies[0].jaw_bones)
        self.assertIn("tongue_base", asm.neck_assemblies[0].tongue_bones)
        self.assertEqual(len(asm.wing_assemblies), 2)
        self.assertEqual(asm.wing_assemblies[0].pectoral_host_bone, "chest")
        self.assertIn("wing_l_f4", asm.wing_assemblies[0].locked_spar_bones)

        # 1. Verify Hover: pectoral + wing_arm Sim(3) bulge, aerial claw curl, locked wing_l_f4 == [0,0,0], 0 SAT collisions!
        h_frames = evaluate_behavior_clip(prog, "hover_full", num_frames=16)
        self.assertTrue(all(fr.pose_euler_deltas.get("wing_l_f4", (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0) for fr in h_frames))
        self.assertGreater(max(fr.pose_euler_deltas.get("claw_l_1", (0.0, 0.0, 0.0))[0] for fr in h_frames), 12.0)
        self.assertGreater(max(fr.pose_scale_factors.get("chest", (1.0, 1.0, 1.0))[0] for fr in h_frames), 1.015)
        self.assertGreater(max(fr.pose_scale_factors.get("wing_l_upper", (1.0, 1.0, 1.0))[1] for fr in h_frames), 1.01)

        # 2. Verify S-Neck Spline-IK 3D Target Aiming + Throat-Sac Sim(3) Bulge + COM Tail Counterweight!
        r_frames = evaluate_behavior_clip(prog, "serpent_aim_roar", num_frames=12)
        peak_fr = max(r_frames, key=lambda f: f.pose_euler_deltas.get("jaw", (0.0, 0.0, 0.0))[0])
        head_wb = peak_fr.scene.bone_by_name["head"]
        fwd_w = head_wb.rotation @ np.array([0.0, 0.0, 1.0])
        to_tgt = np.array([18.0, 28.0, 36.0]) - head_wb.pivot_world
        aim_err_deg = math.degrees(math.acos(float(np.clip(np.dot(fwd_w, to_tgt / np.linalg.norm(to_tgt)), -1.0, 1.0))))
        self.assertLess(aim_err_deg, 18.0, f"Spline-IK neck aim should align head toward target, got err={aim_err_deg:.1f} deg")
        # Throat sac should expand during jaw roar + neck arch!
        throat_b = asm.neck_assemblies[0].throat_bone
        assert throat_b is not None
        self.assertGreater(peak_fr.pose_scale_factors[throat_b][0], 1.005)
        # Asymmetric wing/neck lunge to +X should automatically trigger COM tail counterbalance!
        tail_yaws = [abs(fr.pose_euler_deltas.get("tail_2", (0.0, 0.0, 0.0))[1]) for fr in r_frames]
        self.assertGreater(max(tail_yaws), 1.5, "COM balance should swing tail to counter asymmetric neck/wing moment")

    def test_imaginary_non_english_6_legged_2_winged_chimera_full_auto_rig(self) -> None:
        prog = parse_program(NON_ENGLISH_IMAGINARY_CHIMERA_DSL)
        sc = compile_geometry(prog)
        asm = auto_assemble_biomechanical_rig(prog, sc)

        # Pure geometry-first assembly with ZERO English bone names!
        self.assertEqual(len(asm.limb_assemblies), 6)
        self.assertEqual(len(asm.wing_assemblies), 2)
        self.assertEqual(len(asm.neck_assemblies), 1)
        self.assertEqual(len(asm.tail_chains), 1)
        self.assertGreaterEqual(len(asm.auto_muscle_groups), 8)

        # Grounded hexapod walk + aerial hover both pass with auto_rig=True, in_place=True, 0 SAT collisions!
        aud_walk = audit_behavior_clip(prog, "chimera_scuttle", max_frames=180)
        self.assertTrue(aud_walk["passed"], f"Imaginary chimera auto_rig walk failed: {aud_walk}")
        self.assertEqual(aud_walk["total_collision_events"], 0)
        self.assertLess(aud_walk["max_stance_slip_units"], 0.02)

        aud_fly = audit_behavior_clip(prog, "chimera_hover", max_frames=180)
        self.assertTrue(aud_fly["passed"], f"Imaginary chimera auto_rig hover failed: {aud_fly}")
        self.assertEqual(aud_fly["total_collision_events"], 0)

    def test_standing_ik_pinning_eliminates_fk_floor_dip_and_selective_bipedal_rear_up(self) -> None:
        # Extend the dragon rig with a heavy chest-heave standing clip (`ground_pin_limbs: true`)
        # AND a bipedal rear-up roar (`ground_pin_limbs: [hind_l, hind_r]`) + spine spring!
        pin_dsl = SERPENTINE_DRAGON_FULL_BODY_DSL.replace(
            "@behavior {",
            """@behavior {
  spring chest { stiffness: 65, damping: 12 }
  clip stand_unpinned {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true
    track chest: hinge(axis=x, amp_deg=12, freq_hz=0.625)
    track chest: wave(axis=z, amp_deg=5, freq_hz=0.625, phase_deg=90)
  }
  clip stand_pinned {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true, ground_pin_limbs: true, antagonist_bulge: 0.70
    track chest: hinge(axis=x, amp_deg=12, freq_hz=0.625)
    track chest: wave(axis=z, amp_deg=5, freq_hz=0.625, phase_deg=90)
  }
  clip roar_rear_up {
    duration_s: 1.6, loop: false, in_place: true, auto_rig: true, ground_pin_limbs: [hind_l, hind_r], digit_grasp: 0.85
    key pelvis_root.rotation { 0.0: [0, 0, 0], 0.8: [-14, 0, 0], 1.6: [0, 0, 0], catmullrom }
    track chest: hinge(axis=x, amp_deg=-22, freq_hz=0.625)
    track jaw: hinge(axis=x, amp_deg=38, freq_hz=0.625)
  }""",
        )
        prog = parse_program(pin_dsl)
        # Verify DSL format -> parse roundtrip of `ground_pin_limbs: true` AND `ground_pin_limbs: [hind_l, hind_r]`!
        reparsed = parse_program(format_program(prog))
        self.assertEqual(format_program(prog), format_program(reparsed))

        rest_sc = compile_geometry(prog)
        rest_paw_min_y = min(
            float(bx.corners(include_inflate=False)[:, 1].min())
            for bx in rest_sc.boxes
            if bx.bone_name in ("paw_l", "claw_l_1", "claw_l_1_tip", "claw_l_2")
        )
        rest_foot_min_y = min(
            float(bx.corners(include_inflate=False)[:, 1].min())
            for bx in rest_sc.boxes
            if bx.bone_name in ("foot_l", "talon_l_1")
        )

        # Unpinned FK swings front paws deep below rest_paw_min_y during +14 deg chest pitch!
        unpinned_frames = evaluate_behavior_clip(prog, "stand_unpinned", num_frames=16)
        unpinned_paw_min_y = min(
            float(bx.corners(include_inflate=False)[:, 1].min())
            for fr in unpinned_frames
            for bx in fr.scene.boxes
            if bx.bone_name in ("paw_l", "claw_l_1", "claw_l_1_tip", "claw_l_2")
        )
        self.assertLess(unpinned_paw_min_y, rest_paw_min_y - 1.5, "Expected raw FK chest pitch to dip unpinned paws below ground")

        # Pinned standing IK (`ground_pin_limbs: true`) locks BOTH front paws AND hind feet within < 0.05u of rest ground!
        pinned_frames = evaluate_behavior_clip(prog, "stand_pinned", num_frames=16)
        for fr in pinned_frames:
            paw_y = min(
                float(bx.corners(include_inflate=False)[:, 1].min())
                for bx in fr.scene.boxes
                if bx.bone_name in ("paw_l", "claw_l_1", "claw_l_1_tip", "claw_l_2")
            )
            foot_y = min(
                float(bx.corners(include_inflate=False)[:, 1].min())
                for bx in fr.scene.boxes
                if bx.bone_name in ("foot_l", "talon_l_1")
            )
            self.assertAlmostEqual(paw_y, rest_paw_min_y, delta=0.08)
            self.assertAlmostEqual(foot_y, rest_foot_min_y, delta=0.08)

        # Selective bipedal rear-up (`ground_pin_limbs: [hind_l, hind_r]`): hind feet stay pinned while forepaws lift into the air!
        roar_frames = evaluate_behavior_clip(prog, "roar_rear_up", num_frames=13)
        max_fore_lift = max(
            min(
                float(bx.corners(include_inflate=False)[:, 1].min())
                for bx in fr.scene.boxes
                if bx.bone_name in ("paw_l", "claw_l_1")
            )
            for fr in roar_frames
        )
        self.assertGreater(max_fore_lift, rest_paw_min_y + 2.5, "Forelimbs should rear high into the air when only hind limbs are pinned")
        for fr in roar_frames:
            foot_y = min(
                float(bx.corners(include_inflate=False)[:, 1].min())
                for bx in fr.scene.boxes
                if bx.bone_name in ("foot_l", "talon_l_1")
            )
            self.assertAlmostEqual(foot_y, rest_foot_min_y, delta=0.08)

    def test_mechanical_vehicle_and_mecha_joint_presets_and_rigid_exemption(self) -> None:
        from synccraft.kinematics.bio_constraints import analyze_motion_capabilities

        vehicle_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  hull: ramp(base_lch=[0.42, 0.06, 220], steps=4)
  rubber: ramp(base_lch=[0.20, 0.02, 240], steps=4)
  accent: ramp(base_lch=[0.65, 0.16, 45], steps=4)
}
@rig {
  root bone chassis_root {
    pivot_local: [0, 6, 0]
    box hull_core { size: [14, 6, 24], material: hull }
    bone steer_fl attach(parent_box=hull_core, anchor=[-1, -0.7, 0.7], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(steer_fr) {
      bio_joint: steered_wheel_knuckle
      box knuckle_box { size: [1.5, 2, 2], material: hull }
      bone wheel_fl attach(parent_box=knuckle_box, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        bio_joint: wheel_axle
        box tire_f { size: [3, 6, 6], material: rubber }
      }
    }
    bone wheel_rl attach(parent_box=hull_core, anchor=[-1, -0.7, -0.7], align=[1, 0, 0], offset=[-0.5, 0, 0]) mirror_x(wheel_rr) {
      bio_joint: wheel_axle
      box tire_r { size: [3, 6, 6], material: rubber }
    }
    bone turret_base attach(parent_box=hull_core, anchor=[0, 1, -0.1], align=[0, -1, 0]) {
      bio_joint: turret_yaw
      box turret_pod { size: [8, 4, 9], material: accent }
      bone cannon_main attach(parent_box=turret_pod, anchor=[0, 0.1, 1], align=[0, 0, -1]) {
        bio_joint: cannon_elevation
        box barrel_box { size: [2, 2, 14], material: hull }
      }
      bone radar_rotor attach(parent_box=turret_pod, anchor=[0, 1, -0.4], align=[0, -1, 0]) {
        bio_joint: rotor_propeller
        box dish_box { size: [5, 1, 1.5], material: accent }
      }
    }
    bone cargo_hatch attach(parent_box=hull_core, anchor=[0, 0.5, -1], align=[0, -1, 1]) {
      bio_joint: hatch_door
      box ramp_box { size: [10, 5, 1], material: accent }
    }
  }
}
@behavior {
  clip convoy_patrol {
    duration_s: 2.0, loop: true, in_place: true, auto_rig: true, antagonist_bulge: 0.85
    track wheel_fl: spin(axis=x, amp_deg=720, freq_hz=0.5)
    track steer_fr__wheel_fl: spin(axis=x, amp_deg=720, freq_hz=0.5)
    track wheel_rl: spin(axis=x, amp_deg=720, freq_hz=0.5)
    track wheel_rr: spin(axis=x, amp_deg=720, freq_hz=0.5)
    track steer_fl: wave(axis=y, amp_deg=22, freq_hz=0.5)
    track steer_fr: wave(axis=y, amp_deg=22, freq_hz=0.5)
    track turret_base: wave(axis=y, amp_deg=45, freq_hz=0.5)
    track cannon_main: hinge(axis=x, amp_deg=28, freq_hz=0.5)
    track radar_rotor: spin(axis=y, amp_deg=360, freq_hz=0.5)
    track cargo_hatch: hinge(axis=x, amp_deg=65, freq_hz=0.5)
  }
}
"""
        prog = parse_program(vehicle_dsl)
        sc = compile_geometry(prog)
        caps = analyze_motion_capabilities(prog, sc)
        self.assertEqual(caps["locomotion_class"], "wheeled_vehicle")
        self.assertEqual(caps["wheel_count"], 4)
        self.assertEqual(caps["turret_count"], 1)
        self.assertEqual(caps["rotor_count"], 1)
        self.assertTrue(caps["is_mechanical_or_vehicle"])

        asm = auto_assemble_biomechanical_rig(prog, sc)
        self.assertEqual(len(asm.mechanical.wheels), 4)
        self.assertEqual(len(asm.mechanical.steered_knuckles), 2)
        self.assertEqual(len(asm.mechanical.turrets), 1)
        self.assertEqual(len(asm.mechanical.cannons), 1)
        self.assertEqual(len(asm.mechanical.hatches), 1)
        # Rigid mechanical hardware must NOT receive organic soft-tissue Sim(3) muscle groups!
        self.assertEqual(len(asm.auto_muscle_groups), 0)

        frames = evaluate_behavior_clip(prog, "convoy_patrol", num_frames=12)
        self.assertGreater(max(abs(fr.pose_euler_deltas["wheel_fl"][0]) for fr in frames), 300.0)
        self.assertTrue(all(fr.pose_scale_factors == {} for fr in frames), "Rigid mechanical parts must not squish in Sim(3)")

    def test_deep_zfold_digitigrade_so3_limits_sloped_bridges_and_3of4_pinning(self) -> None:
        """v31 Universal Test: deep Z-fold (+132 deg knee / -104 deg hock) SO(3) auto-limits,
        sloped joint bridges under dynamic flexion, 3-of-4 selective standing IK, AND strict
        rejection of genuine non-joint inter-limb collisions."""
        from synccraft.audit.sync_audit import audit_scene
        from synccraft.kinematics.bio_constraints import infer_universal_bio_constraints
        from synccraft.kinematics.musculoskeletal import audit_behavior_clip

        zfold_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  hide: ramp(base_lch=[0.52, 0.10, 200], steps=4)
  gold: ramp(base_lch=[0.74, 0.11, 86], steps=4)
}
@rig {
  root bone pelvis_root {
    pivot_local: [0, 17.5, 0]
    box pelvis_core { size: [12, 8, 18], material: hide, sculpt_group: "pelvis_sc|hip_flex" }
    bone chest attach(parent_box=pelvis_core, anchor=[0, 0.1, 1], align=[0, 0.0, -1], seat=0.1) {
      rest_euler: [-12, 0, 0]
      box chest_core { size: [14, 10, 16], material: hide, blend_with: [pelvis_core], sculpt_group: "chest_sc|fore_thorax" }
      bone arm_l_upper attach(parent_box=chest_core, anchor=[-1, -0.5, 0.65], align=[0.8, 0.9, 0], offset=[-0.5, 0, 0]) mirror_x(arm_r_upper) {
        rest_euler: [40, 0, -16]
        box hum { size: [4, 10, 4], material: hide, blend_with: [chest_core], sculpt_group: "arm_u|elbow_flex|fore_thorax" }
        bone arm_l_fore attach(parent_box=hum, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.2) {
          rest_euler: [-34, 0, 14]
          box fore { size: [3.5, 11, 3.5], material: gold, blend_with: [hum], sculpt_group: "arm_f|elbow_flex|wrist_flex" }
          bone paw_l attach(parent_box=fore, anchor=[0, -1, 0.3], align=[0, 1, -0.3], seat=0.1) {
            rest_euler: [6, 0, 0]
            box paw_pad { size: [4.5, 2.5, 4.5], material: gold, blend_with: [fore], sculpt_group: "paw_sc|wrist_flex" }
            box paw_wrist_bridge place(base_box=paw_pad, anchor=[0, 0.2, -0.95], align=[0, -0.15, 0.85]) {
              size: [3.5, 2.0, 2.5], material: hide, rotate: [-24, 0, 0], blend_with: [paw_pad, fore], sculpt_group: "paw_sc|arm_f|wrist_flex"
            }
          }
        }
      }
    }
    bone leg_l_thigh attach(parent_box=pelvis_core, anchor=[-1, 0.0, -0.70], align=[0.8, 0.9, 0], offset=[-0.5, 0, 0]) mirror_x(leg_r_thigh) {
      rest_euler: [-64, 0, -18]
      box thigh { size: [5, 13, 5.5], material: hide, blend_with: [pelvis_core], sculpt_group: "th_sc|hip_flex|knee_flex" }
      bone leg_l_shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.15) {
        rest_euler: [132, 0, 12]
        box shin { size: [4, 13, 4.5], material: hide, blend_with: [thigh, pelvis_core], sculpt_group: "sh_sc|hip_flex|knee_flex" }
        bone leg_l_ankle attach(parent_box=shin, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.2) {
          rest_euler: [-104, 0, 6]
          box ankle { size: [3.5, 10.5, 3.5], material: gold, blend_with: [shin], sculpt_group: "an_sc|sh_sc|knee_flex|ankle_flex" }
          box hock_bridge place(base_box=ankle, anchor=[0, 0.75, -0.4], align=[0, -0.3, 0.2]) {
            size: [3.5, 3.5, 3.5], material: gold, rotate: [42, 0, 0], blend_with: [ankle, shin], sculpt_group: "an_sc|sh_sc|knee_flex"
          }
          bone foot_l attach(parent_box=ankle, anchor=[0, -1, 0.25], align=[0, 1, -0.4], seat=0.1) {
            rest_euler: [36, 0, 0]
            box foot_pad { size: [4.5, 2.5, 4.5], material: gold, blend_with: [ankle], sculpt_group: "ft_sc|ankle_flex" }
          }
        }
      }
    }
  }
}
@behavior {
  clip zfold_walk {
    duration_s: 2.0, loop: true, in_place: true, auto_rig: true
    gait prowl {
      pattern: lateral_sequence, froude: 0.048, duty: 0.66,
      lift_ratio: 0.09, heave_ratio: 0.018, pitch_deg: 2.0, spine_wave_deg: 4.0,
      in_place: true, auto_bio_limits: true, foot_roll: 0.80, antagonist_bulge: 0.70
    }
  }
  clip slam_3pin {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true, ground_pin_limbs: [fore_r, hind_l, hind_r],
    antagonist_bulge: 0.75
    key pelvis_root.position { 0.0: [0, 0, 0], 0.5: [0, -0.8, 0], 1.0: [0, -1.2, 0], 1.6: [0, 0, 0], catmullrom }
    key chest.rotation { 0.0: [0, 0, 0], 0.5: [-4, -3, 2], 1.0: [4, 3, -2], 1.6: [0, 0, 0], catmullrom }
    key arm_l_upper.rotation { 0.0: [0, 0, 0], 0.5: [-55, 14, -22], 1.0: [0, 0, 0], 1.6: [0, 0, 0], catmullrom }
    key arm_l_fore.rotation { 0.0: [0, 0, 0], 0.5: [16, 0, 0], 1.0: [0, 0, 0], 1.6: [0, 0, 0], catmullrom }
  }
  clip deliberate_arm_self_hit {
    duration_s: 1.0, loop: true, in_place: true, auto_rig: false
    key arm_l_upper.rotation { 0.0: [0, 0, 0], 0.5: [0, 0, 75], 1.0: [0, 0, 0], catmullrom }
  }
}
"""
        prog = parse_program(zfold_dsl)
        sc = compile_geometry(prog)
        st_aud = audit_scene(sc, audit_profile="sculpted")
        self.assertTrue(st_aud["passed"], f"Static Z-fold audit failed: {st_aud['collision_pairs']}")

        # 1) Auto-inferred SO(3) limits must allow deep Z-fold knee (+132) & hock (-104) to straighten back to 0 deg!
        lims = infer_universal_bio_constraints(prog, sc)
        shin_x_lo, shin_x_hi = lims["leg_l_shin"][0][0]
        ank_x_lo, ank_x_hi = lims["leg_l_ankle"][0][0]
        self.assertLessEqual(shin_x_lo, -132.0, "Deep Z-fold knee (+132) must allow straightening to 0 deg")
        self.assertLessEqual(shin_x_hi, 35.0, "Deep Z-fold knee (+132) must not over-fold backward through femur")
        self.assertGreaterEqual(ank_x_hi, 104.0, "Deep Z-fold hock (-104) must allow straightening to 0 deg")

        # 2) Dynamic walk & 3-of-4 selective limb pinning pass swept OBB SAT cleanly (< 0.02u slip)
        #    AND preserve exact vertical-sagittal Z-fold plane (ZERO frog-leg sideways knee/hock roll!)
        from synccraft.kinematics.morphology import analyze_morphology
        morph_z = analyze_morphology(prog, sc)
        hind_morphs = [lb for lb in morph_z.limbs if lb.role.startswith("hind")]
        self.assertTrue(hind_morphs and all(lb.posture != "sprawled" for lb in hind_morphs),
                        f"Vertical Z-fold hindlimbs must NOT be misclassified as sprawled: {[lb.posture for lb in hind_morphs]}")

        aud_walk = audit_behavior_clip(prog, "zfold_walk", num_frames=16, max_frames=144)
        self.assertTrue(aud_walk["passed"], f"zfold_walk failed: {aud_walk['colliding_frames'][:2]} slip={aud_walk['max_stance_slip_units']} gait={aud_walk.get('gait_audit')}")
        self.assertLess(aud_walk["max_stance_slip_units"], 0.02)

        frames_w = evaluate_behavior_clip(prog, "zfold_walk", num_frames=12)
        for bn in ("leg_l_shin", "leg_l_ankle"):
            max_off = max(max(abs(fr.pose_euler_deltas.get(bn, (0, 0, 0))[1]), abs(fr.pose_euler_deltas.get(bn, (0, 0, 0))[2])) for fr in frames_w)
            self.assertLessEqual(max_off, 8.0, f"Frog-leg regression! {bn} twisted {max_off:.1f} deg off sagittal plane during walk!")
        max_hip_roll = max(abs(fr.pose_euler_deltas.get("leg_l_thigh", (0, 0, 0))[2]) for fr in frames_w)
        self.assertLessEqual(max_hip_roll, 12.0, f"Frog-leg regression! leg_l_thigh rolled {max_hip_roll:.1f} deg outward during walk!")

        # 2b) Explicit @kinematics ik_chains ending above paw_l/foot_l MUST still trigger SE(3) wrist/ankle split (never slamming feet underground!)
        fe_l = sc.bone_by_name["paw_l"].local_translation
        fe_r = sc.bone_by_name["arm_r_upper__paw_l"].local_translation
        he_l = sc.bone_by_name["foot_l"].local_translation
        he_r = sc.bone_by_name["leg_r_thigh__foot_l"].local_translation
        zfold_explicit_ik = zfold_dsl.replace(
            "@behavior {",
            f"""@kinematics {{
  ik_chain fore_l {{ bones: [arm_l_upper, arm_l_fore], effector: [{fe_l[0]:.3f}, {fe_l[1]:.3f}, {fe_l[2]:.3f}], plane: sagittal }}
  ik_chain fore_r {{ bones: [arm_r_upper, arm_r_upper__arm_l_fore], effector: [{fe_r[0]:.3f}, {fe_r[1]:.3f}, {fe_r[2]:.3f}], plane: sagittal }}
  ik_chain hind_l {{ bones: [leg_l_thigh, leg_l_shin, leg_l_ankle], effector: [{he_l[0]:.3f}, {he_l[1]:.3f}, {he_l[2]:.3f}], plane: sagittal }}
  ik_chain hind_r {{ bones: [leg_r_thigh, leg_r_thigh__leg_l_shin, leg_r_thigh__leg_l_ankle], effector: [{he_r[0]:.3f}, {he_r[1]:.3f}, {he_r[2]:.3f}], plane: sagittal }}
}}
@behavior {{""",
        )
        prog_eik = parse_program(zfold_explicit_ik)
        rest_gy = min(float(sc.box_by_name["foot_pad"].corners(include_inflate=False)[:, 1].min()),
                      float(sc.box_by_name["paw_pad"].corners(include_inflate=False)[:, 1].min()))
        frames_eik = evaluate_behavior_clip(prog_eik, "zfold_walk", num_frames=12)
        min_foot_y_eik = min(
            float(fr.scene.box_by_name[bn].corners(include_inflate=False)[:, 1].min())
            for fr in frames_eik for bn in ("paw_pad", "foot_pad")
        )
        self.assertGreaterEqual(min_foot_y_eik, rest_gy - 1.5, f"Explicit ik_chain slammed feet underground! min_y={min_foot_y_eik:.2f} vs rest_gy={rest_gy:.2f}")
        for bn in ("leg_l_shin", "leg_l_ankle"):
            max_off = max(max(abs(fr.pose_euler_deltas.get(bn, (0, 0, 0))[1]), abs(fr.pose_euler_deltas.get(bn, (0, 0, 0))[2])) for fr in frames_eik)
            self.assertLessEqual(max_off, 8.0, f"Explicit sagittal ik_chain frog-leg regression on {bn}: {max_off:.1f} deg!")

        aud_slam = audit_behavior_clip(prog, "slam_3pin", num_frames=16, max_frames=144)
        self.assertTrue(aud_slam["passed"], f"slam_3pin failed: {aud_slam['colliding_frames'][:2]} slip={aud_slam['max_stance_slip_units']}")
        self.assertLess(aud_slam["max_stance_slip_units"], 0.02)

        # 3) Genuine non-joint collision (left forearm swinging +75 deg straight into right leg/chest) STILL fails SAT!
        aud_bad = audit_behavior_clip(prog, "deliberate_arm_self_hit", num_frames=8, max_frames=16)
        self.assertFalse(aud_bad["passed"], "Deliberate inter-limb/chest self-collision MUST still be rejected by SAT!")

    def test_pillar_quadruped_non_zfold_and_folding_mechanical_excavator_boom(self) -> None:
        """Verify our v31 solver & SAT updates remain 100% general across straight pillar quadrupeds
        (elephants/bears/golems) AND acute-folding mechanical hydraulic crane/excavator booms."""
        from synccraft.kinematics.musculoskeletal import audit_behavior_clip

        pillar_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette { hide: ramp(base_lch=[0.48, 0.04, 220], steps=4) }
@rig {
  root bone body_root {
    pivot_local: [0, 25.0, 0]
    box torso { size: [14, 12, 28], material: hide, sculpt_group: "torso_sc" }
    bone fore_l attach(parent_box=torso, anchor=[-0.8, -1, 0.78], align=[0, 1, 0]) mirror_x(fore_r) {
      rest_euler: [6, 0, 0]
      box f_up { size: [5, 10, 5], material: hide, blend_with: [torso], sculpt_group: "fl_u" }
      bone fore_l_low attach(parent_box=f_up, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.1) {
        rest_euler: [-12, 0, 0]
        box f_lo { size: [4.5, 10, 4.5], material: hide, blend_with: [f_up], sculpt_group: "fl_l" }
      }
    }
    bone hind_l attach(parent_box=torso, anchor=[-0.8, -1, -0.78], align=[0, 1, 0]) mirror_x(hind_r) {
      rest_euler: [-6, 0, 0]
      box h_up { size: [5, 10, 5], material: hide, blend_with: [torso], sculpt_group: "hl_u" }
      bone hind_l_low attach(parent_box=h_up, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.1) {
        rest_euler: [12, 0, 0]
        box h_lo { size: [4.5, 10, 4.5], material: hide, blend_with: [h_up], sculpt_group: "hl_l" }
      }
    }
  }
}
@behavior {
  clip heavy_march {
    duration_s: 1.5, loop: true, in_place: true, auto_rig: true
    gait march {
      pattern: lateral_sequence, froude: 0.036, duty: 0.72,
      lift_ratio: 0.06, heave_ratio: 0.015, pitch_deg: 1.5, spine_wave_deg: 2.0,
      in_place: true, auto_bio_limits: true, antagonist_bulge: 0.55
    }
  }
}
"""
        prog_p = parse_program(pillar_dsl)
        aud_p = audit_behavior_clip(prog_p, "heavy_march", num_frames=16, max_frames=144)
        self.assertTrue(aud_p["passed"], f"Pillar quadruped walk failed: {aud_p['colliding_frames'][:2]} slip={aud_p['max_stance_slip_units']} step={aud_p['step_bound_satisfied']} gait={aud_p.get('gait_audit')}")

        excavator_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  steel: ramp(base_lch=[0.70, 0.14, 82], steps=4)
  iron: ramp(base_lch=[0.30, 0.02, 230], steps=4)
}
@rig {
  root bone chassis_base {
    pivot_local: [0, 4.0, 0]
    box track_hull { size: [14, 6, 18], material: iron }
    bone turret_cab attach(parent_box=track_hull, anchor=[0, 1, 0], align=[0, -1, 0]) {
      bio_joint: turret_yaw
      box cab_body { size: [12, 8, 12], material: steel, blend_with: [track_hull] }
      bone hydraulic_boom attach(parent_box=cab_body, anchor=[0, 0.3, 1], align=[0, 0, -1], seat=0.1) {
        rest_euler: [-45, 0, 0]
        bio_joint: hydraulic_hinge
        box boom_arm { size: [3.5, 3.5, 18], material: steel, blend_with: [cab_body], sculpt_group: "boom_sc" }
        bone hydraulic_stick attach(parent_box=boom_arm, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.15) {
          rest_euler: [110, 0, 0]
          bio_joint: hydraulic_hinge
          box stick_arm { size: [3, 3, 14], material: iron, blend_with: [boom_arm], sculpt_group: "boom_sc" }
          bone hydraulic_bucket attach(parent_box=stick_arm, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.15) {
            rest_euler: [45, 0, 0]
            bio_joint: hydraulic_hinge
            box bucket_scoop { size: [6, 4, 5], material: iron, blend_with: [stick_arm], sculpt_group: "boom_sc" }
          }
        }
      }
    }
  }
}
@behavior {
  clip dig_cycle {
    duration_s: 2.0, loop: true, in_place: true, auto_rig: true, antagonist_bulge: 0.90
    key turret_cab.rotation { 0.0: [0, 0, 0], 1.0: [0, 35, 0], 2.0: [0, 0, 0], catmullrom }
    key hydraulic_boom.rotation { 0.0: [0, 0, 0], 0.8: [32, 0, 0], 1.5: [-18, 0, 0], 2.0: [0, 0, 0], catmullrom }
    key hydraulic_stick.rotation { 0.0: [0, 0, 0], 0.8: [-65, 0, 0], 1.5: [28, 0, 0], 2.0: [0, 0, 0], catmullrom }
    key hydraulic_bucket.rotation { 0.0: [0, 0, 0], 0.8: [40, 0, 0], 1.5: [-25, 0, 0], 2.0: [0, 0, 0], catmullrom }
  }
}
"""
        prog_m = parse_program(excavator_dsl)
        aud_m = audit_behavior_clip(prog_m, "dig_cycle", num_frames=12, max_frames=36)
        self.assertTrue(aud_m["passed"], f"Excavator 138-deg hydraulic fold failed SAT: {aud_m['colliding_frames'][:2]}")
        frames_m = evaluate_behavior_clip(prog_m, "dig_cycle", num_frames=12)
        self.assertTrue(all(fr.pose_scale_factors == {} for fr in frames_m), "Rigid hydraulic boom must stay 0% Sim(3) squished")

    def test_jet_propulsion_vtol_hover_cruise_boost_and_cladding_exclusion(self) -> None:
        """Verify universal VTOL/jet_hover, banked jet_cruise, jet_boost (ZERO bird flapping!) & zero-offset mirror_x cladding exclusion."""
        jet_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  alloy: ramp(base_lch=[0.74, 0.03, 240], steps=4)
  vent: ramp(base_lch=[0.55, 0.20, 28], steps=4)
}
@rig {
  root bone pelvis {
    pivot_local: [0, 12.0, 0]
    box core { size: [8, 6, 12], material: alloy }
    bone head_root attach(parent_box=core, anchor=[0, 0.3, 1], align=[0, 0, -1], seat=0.1) {
      box skull { size: [5, 4, 6], material: alloy, blend_with: [core] }
      bone skull_side_l joint(parent=skull, at=[0, 0, 0]) mirror_x(skull_side_r) {
        box cheek_l place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[-2.8, 0, 1.0]) { size: [1.5, 3, 4], material: alloy, blend_with: [skull, core] }
      }
    }
    bone tail_1 attach(parent_box=core, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.1) {
      box t1 { size: [4, 3, 8], material: alloy, blend_with: [core] }
      bone tail_2 attach(parent_box=t1, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.1) {
        box t2 { size: [3, 2, 8], material: alloy, blend_with: [t1] }
      }
    }
    bone leg_l_thigh joint(parent=core, at=[-4.5, -2.0, -2.0]) mirror_x(leg_r_thigh) {
      box thigh_l place(anchor=[0, 0, 0], align=[0, 1, 0]) { size: [2.5, 6, 3], material: alloy, blend_with: [core] }
      bone leg_l_shin attach(parent_box=thigh_l, anchor=[0, -1, 0], align=[0, 1, 0], seat=0.1) {
        box shin_l { size: [2, 5.5, 2.5], material: alloy, blend_with: [thigh_l] }
      }
    }
    bone wing_l_upper joint(parent=core, at=[-5.0, 2.5, 1.0]) mirror_x(wing_r_upper) {
      rest_euler: [0, 0, 20]
      joint_limits: [[-120, 120], [-95, 95], [-120, 120]]
      box pylon_l place(anchor=[0, 0, 0], align=[0, -1, 0]) { size: [2.5, 8, 3], material: alloy, blend_with: [core] }
      bone wing_l_fore joint(parent=pylon_l, at=[-2.5, 7.0, 0.0]) {
        joint_limits: [[-180, 180], [-115, 115], [-115, 115]]
        box nacelle_l place(anchor=[0, 0, 0], align=[0.5, 0, 0]) { size: [3, 4, 6], material: alloy, blend_with: [pylon_l] }
        bone wing_l_f1 joint(parent=nacelle_l, at=[0, 3.2, -3.5]) {
          bio_joint: gimbal_thruster
          box eng1 place(anchor=[0, 0, 0], align=[0, 0, 1]) { size: [1.5, 1.5, 9], material: vent, blend_with: [nacelle_l] }
        }
        bone wing_l_f2 joint(parent=nacelle_l, at=[0, 0.0, -3.8]) {
          bio_joint: gimbal_thruster
          box eng2 place(anchor=[0, 0, 0], align=[0, 0, 1]) { size: [1.5, 1.5, 9], material: vent, blend_with: [nacelle_l] }
        }
        bone wing_l_f3 joint(parent=nacelle_l, at=[0, -3.2, -3.5]) {
          bio_joint: gimbal_thruster
          box eng3 place(anchor=[0, 0, 0], align=[0, 0, 1]) { size: [1.5, 1.5, 9], material: vent, blend_with: [nacelle_l] }
        }
      }
    }
  }
}
@muscles {
  muscle wing_pectoralis_l { origin_bone: pelvis, insertion_bone: wing_l_upper, action_axis: z, moment_arm_deg: 24, scale_bulge: 0.0 }
  muscle wing_pectoralis_r { origin_bone: pelvis, insertion_bone: wing_r_upper, action_axis: z, moment_arm_deg: -24, scale_bulge: 0.0 }
}
@behavior {
  clip vtol_hover {
    duration_s: 1.6, loop: true, in_place: true, auto_rig: true
    flight jh { mode: jet_hover, flap_hz: 0.625, lift_recoil_u: 1.2, body_pitch_deg: 12, limb_tuck: 0.70, gimbal_deg: 3.0, in_place: true }
  }
  clip supersonic_bank {
    duration_s: 1.4, loop: true, in_place: true, auto_rig: true
    flight jc { mode: jet_cruise, flap_hz: 0.714, lift_recoil_u: 0.8, body_pitch_deg: 0, limb_tuck: 0.85, gimbal_deg: 2.5, bank_deg: 7.5, in_place: true }
  }
  clip midair_boost {
    duration_s: 1.2, loop: false, in_place: true, auto_rig: true
    flight jb { mode: jet_boost, flap_hz: 0.833, lift_recoil_u: 1.8, body_pitch_deg: 18, limb_tuck: 0.75, gimbal_deg: 4.0, in_place: true }
  }
}
"""
        prog = parse_program(jet_dsl)
        from synccraft.dsl.parser import format_program
        from synccraft.kinematics.morphology import analyze_morphology
        from synccraft.kinematics.se3_chain import compile_geometry
        # Canonical format + re-parse roundtrip must preserve gimbal_deg and bank_deg
        re_prog = parse_program(format_program(prog))
        self.assertEqual(len(re_prog.behavior.clips), 3)

        # 1. Zero-offset mirror_x cladding helper bones (`skull_side_l`/`r`) must NOT be sucked into axial spine/neck
        rest_sc = compile_geometry(prog, validate=False)
        morph = analyze_morphology(prog, rest_sc)
        self.assertNotIn("skull_side_l", morph.neck)
        self.assertNotIn("skull_side_r", morph.neck)
        self.assertNotIn("skull_side_l", morph.spine)
        self.assertNotIn("skull_side_r", morph.spine)
        # All 3 independent jet-thruster blades must be discovered as unlocked rays
        l_wings = [w for w in morph.wings if w.side == "l"]
        self.assertEqual(len(l_wings), 1)
        self.assertEqual(len(l_wings[0].finger_bones), 3)
        self.assertEqual(len(l_wings[0].locked_spar_bones), 0)

        # 2. VTOL hover: ZERO bird flapping on wing_l_upper (must NOT auto-fire 24-deg wing_pectoralis muscle!), 0.0000u Z-drift
        aud_h = audit_behavior_clip(prog, "vtol_hover", num_frames=16, max_frames=48)
        self.assertTrue(aud_h["passed"], f"jet_hover failed: {aud_h}")
        fa_h = aud_h["flight_audit"]
        self.assertTrue(fa_h["passed"], f"jet_hover flight_audit failed: {fa_h}")
        self.assertEqual(fa_h["max_root_z_units"], 0.0)
        self.assertLess(fa_h["max_bilateral_asymmetry_units"], 1e-3)
        fr_h = evaluate_behavior_clip(prog, "vtol_hover", num_frames=16)
        max_pylon_flap = max(max(abs(c) for c in f.pose_euler_deltas.get("wing_l_upper", (0, 0, 0))) for f in fr_h)
        self.assertLess(max_pylon_flap, 4.0, f"jet_hover flapped wing pylon by {max_pylon_flap} deg!")
        # Verify all 3 thruster nozzles actively gimbal independently
        for f_bn in ("wing_l_f1", "wing_l_f2", "wing_l_f3"):
            exc = max(max(abs(c) for c in f.pose_euler_deltas.get(f_bn, (0, 0, 0))) for f in fr_h)
            self.assertGreater(exc, 0.5, f"Thruster blade {f_bn} failed to gimbal!")

        # 3. Supersonic banked jet_cruise: rolls cleanly, supersonic root_motion speed, body-frame symmetry passes, 0 Z-drift
        aud_c = audit_behavior_clip(prog, "supersonic_bank", num_frames=16, max_frames=48)
        self.assertTrue(aud_c["passed"], f"jet_cruise failed: {aud_c}")
        fa_c = aud_c["flight_audit"]
        self.assertTrue(fa_c["passed"], f"jet_cruise flight_audit failed: {fa_c}")
        self.assertEqual(fa_c["max_root_z_units"], 0.0)
        self.assertGreater(fa_c["root_motion"]["speed_units_per_s"], 35.0)
        self.assertLess(fa_c["max_bilateral_asymmetry_units"], 1e-3)
        fr_c = evaluate_behavior_clip(prog, "supersonic_bank", num_frames=16)
        max_roll = max(abs(f.pose_euler_deltas.get("pelvis", (0, 0, 0))[2]) for f in fr_c)
        self.assertGreater(max_roll, 6.0, "jet_cruise bank_deg did not produce fighter-jet banking roll!")

        # 4. Impulsive VTOL jet_boost: passes all flight + SAT contracts with 0 Z-drift
        aud_b = audit_behavior_clip(prog, "midair_boost", num_frames=16, max_frames=48)
        self.assertTrue(aud_b["passed"], f"jet_boost failed: {aud_b}")
        self.assertTrue(aud_b["flight_audit"]["passed"], f"jet_boost flight_audit failed: {aud_b['flight_audit']}")
        self.assertEqual(aud_b["flight_audit"]["max_root_z_units"], 0.0)


if __name__ == "__main__":
    unittest.main()
