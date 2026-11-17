"""End-to-end regression suite for ADR 0011 (Gate A fixes 1-5, Gate B @morphology,
Gate C 10-body zero-shot Froude gait transfer, sculpted pixel-art alpha cut-outs /
banded cel shading / @stamps / octagon / seat / layer, and < 2.0s 120+ box speed benchmark).
"""

from __future__ import annotations

import io
import time
import unittest

import numpy as np
from PIL import Image

from synccraft.audit.offscreen_renderer import render_world_observation
from synccraft.audit.sync_audit import audit_scene
from synccraft.backends import export_bbmodel, export_bedrock, export_geckolib
from synccraft.dsl import (
    DSLValidationError,
    format_program,
    parse_program,
)
from synccraft.kinematics import (
    analyze_morphology,
    audit_behavior_clip,
    compile_geometry,
    evaluate_behavior_clip,
    format_morphology_block,
)
from synccraft.shading.compiler import compile_vanilla_atlas
from synccraft.surface.contracts import MaterialProgram, StyleConfig, WorldState
from synccraft.surface.domain import build_surface_domain, initialize_surface_state
from synccraft.world.assembly_world_tasks import (
    MODULAR_ARTILLERY_WALKER_V4_DSL,
    MODULAR_SCULPTED_DRAGON_V4_DSL,
)
from synccraft.world.exemplars import (
    COLOSSAL_WALKING_CITADEL_DSL,
    SCULPTED_ELDER_WYVERN_DSL,
)


# ---------------------------------------------------------------------------
# Clean-room 10-Creature Morphology Zoo for Gate B & Gate C Zero-Shot Gait
# ---------------------------------------------------------------------------

UNCHANGED_WALK_GAIT = """
  clip walk {
    duration_s: 1.6
    gait walk {
      pattern: lateral_sequence,
      froude: 0.18,
      duty: 0.68,
      lift_ratio: 0.12,
      heave_ratio: 0.03,
      pitch_deg: 1.5,
      spine_wave_deg: 3.5,
      tail_lag: 0.2,
      style: { weight: 0.5, snap: 0.3 }
    }
  }
""".strip()

UNCHANGED_TROT_GAIT = """
  clip trot {
    duration_s: 1.2
    gait trot {
      pattern: diagonal_pairs,
      froude: 0.45,
      duty: 0.52,
      lift_ratio: 0.16,
      heave_ratio: 0.04,
      pitch_deg: 2.0,
      spine_wave_deg: 4.0,
      tail_lag: 0.22,
      style: { weight: 0.45, snap: 0.55 }
    }
  }
""".strip()


def _quadruped_species_dsl(
    *,
    chest_sz: tuple[float, float, float],
    hip_sz: tuple[float, float, float],
    upper_h: float,
    lower_h: float,
    rest_knee_deg: float,
    sprawled: bool = False,
    has_wings: bool = False,
    three_necks: bool = False,
) -> str:
    """Construct a clean grounded creature rig for Gate B / Gate C evaluation."""
    if sprawled:
        # Lateral femur + vertical shin for crocodile
        hip_y = lower_h + 1.0
        leg_block = f"""
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.4, 0.65], align=[1, 0, 0]) mirror_x(fore_leg_r) {{
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box fore_femur {{ size: [{upper_h}, 1.5, 2], material: hide }}
      bone fore_shin attach(parent_box=fore_femur, anchor=[-1, 0, 0], align=[0, 1, 0]) {{
        joint_limits: [[-65, 65], [-55, 55], [-65, 65]]
        box fore_tibia {{ size: [1.5, {lower_h}, 2], material: hide }}
      }}
    }}
    bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.4, -0.45], align=[1, 0, 0]) mirror_x(hind_leg_r) {{
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box hind_femur {{ size: [{upper_h}, 1.5, 2], material: hide }}
      bone hind_shin attach(parent_box=hind_femur, anchor=[-1, 0, 0], align=[0, 1, 0]) {{
        joint_limits: [[-65, 65], [-55, 55], [-65, 65]]
        box hind_tibia {{ size: [1.5, {lower_h}, 2], material: hide }}
      }}
    }}"""
        root_y = lower_h + 0.4 * chest_sz[1]
    else:
        rad_k = np.deg2rad(rest_knee_deg)
        shin_drop = lower_h * np.cos(rad_k)
        root_y = round(float(upper_h + shin_drop + 0.35 * chest_sz[1]), 3)
        leg_block = f"""
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.7, 0.65], align=[1, 1, 0]) mirror_x(fore_leg_r) {{
      joint_limits: [[-65, 65], [-30, 30], [-30, 30]]
      box fore_femur {{ size: [2, {upper_h}, 2], material: hide }}
      bone fore_shin attach(parent_box=fore_femur, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [{rest_knee_deg}, 0, 0]
        joint_limits: [[-55, 85], [-20, 20], [-20, 20]]
        box fore_tibia {{ size: [1.5, {lower_h}, 2], material: hide }}
      }}
    }}
    bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.7, -0.5], align=[0.5, 1, 0]) mirror_x(hind_leg_r) {{
      joint_limits: [[-65, 65], [-30, 30], [-30, 30]]
      box hind_femur {{ size: [2, {upper_h}, 2.5], material: hide }}
      bone hind_shin attach(parent_box=hind_femur, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [{rest_knee_deg}, 0, 0]
        joint_limits: [[-55, 85], [-20, 20], [-20, 20]]
        box hind_tibia {{ size: [1.5, {lower_h}, 2], material: hide }}
      }}
    }}"""

    wing_block = ""
    if has_wings:
        wing_block = """
    bone wing_l attach(parent_box=chest, anchor=[-1, 0.85, 0.2], align=[1, 0, 0]) mirror_x(wing_r) {
      rest_euler: [0, 0, 18]
      joint_limits: [[-35, 35], [-35, 35], [-55, 55]]
      box wing_spar { size: [7, 1.5, 2], material: accent }
      plane wing_web place(base_box=wing_spar, anchor=[0, 0, -1], align=[0, 0, 1]) {
        size: [6, 5], normal: y, material: accent, edge: scallop(bays=3, depth=2, side=bottom), layer: 1
      }
      bone wing_fore attach(parent_box=wing_spar, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        joint_limits: [[-35, 35], [-35, 35], [-55, 55]]
        box wing_tip_spar { size: [7, 1, 1.5], material: accent }
      }
    }"""

    if three_necks:
        neck_block = """
    bone neck_c attach(parent_box=chest, anchor=[0, 0.5, 1], align=[0, 0, -1], seat=0.25) {
      rest_euler: [-18, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_c_box { size: [2, 2.5, 5], material: hide }
      bone head_c attach(parent_box=neck_c_box, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {
        rest_euler: [16, 0, 0]
        joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
        box skull_c { size: [2.5, 2.5, 4], material: hide }
      }
    }
    bone neck_side_l attach(parent_box=chest, anchor=[-0.85, 0.45, 1], align=[0, 0, -1], seat=0.08) mirror_x(neck_side_r) {
      rest_euler: [-14, -24, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_s_box { size: [2, 2, 4.5], material: hide }
    }"""
    else:
        neck_block = """
    bone neck_1 attach(parent_box=chest, anchor=[0, 0.4, 1], align=[0, 0, -1], seat=0.25) {
      rest_euler: [-16, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_box { size: [2.5, 2.5, 4], material: hide }
      bone head attach(parent_box=neck_box, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {
        rest_euler: [14, 0, 0]
        joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
        box skull { size: [3.5, 3, 4.5], material: hide }
      }
    }"""

    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  hide: ramp(base_lch=[0.48, 0.10, 42], steps=5)
  accent: ramp(base_lch=[0.68, 0.14, 75], steps=4)
}}
@rig {{
  root bone torso {{
    pivot_local: [0, {root_y}, 0]
    box chest {{ size: [{chest_sz[0]}, {chest_sz[1]}, {chest_sz[2]}], material: hide }}
    bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {{
      joint_limits: [[-18, 18], [-22, 22], [-12, 12]]
      box hips {{ size: [{hip_sz[0]}, {hip_sz[1]}, {hip_sz[2]}], material: hide }}
      {leg_block.split('bone hind_leg_l', 1)[1] if 'bone hind_leg_l' in leg_block else ''}
      bone tail_1 attach(parent_box=hips, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {{
        joint_limits: [[-20, 20], [-28, 28], [-15, 15]]
        box caudal_1 {{ size: [2, 2, 4.5], material: hide }}
        bone tail_2 attach(parent_box=caudal_1, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {{
          joint_limits: [[-20, 20], [-32, 32], [-15, 15]]
          box caudal_2 {{ size: [1.5, 1.5, 4.5], material: hide }}
        }}
      }}
    }}
    {'bone fore_leg_l' + leg_block.split('bone fore_leg_l', 1)[1].split('bone hind_leg_l', 1)[0]}
    {wing_block}
    {neck_block}
  }}
}}
@behavior {{
  {UNCHANGED_WALK_GAIT}
  {UNCHANGED_TROT_GAIT}
}}
""".replace(
        "bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {\n"
        "      joint_limits: [[-18, 18], [-22, 22], [-12, 12]]\n"
        f"      box hips {{ size: [{hip_sz[0]}, {hip_sz[1]}, {hip_sz[2]}], material: hide }}\n"
        "      ",
        "bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {\n"
        "      joint_limits: [[-18, 18], [-22, 22], [-12, 12]]\n"
        f"      box hips {{ size: [{hip_sz[0]}, {hip_sz[1]}, {hip_sz[2]}], material: hide }}\n"
        "      bone hind_leg_l",
    )


def _biped_species_dsl(*, chest_sz: tuple[float, float, float], upper_h: float, lower_h: float, rest_knee_deg: float, wing_span: float) -> str:
    rad_k = np.deg2rad(rest_knee_deg)
    root_y = round(float(upper_h + lower_h * np.cos(rad_k) + 0.35 * chest_sz[1]), 3)
    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  hide: ramp(base_lch=[0.50, 0.09, 210], steps=5)
  feather: ramp(base_lch=[0.72, 0.08, 70], steps=4)
}}
@rig {{
  root bone pelvis {{
    pivot_local: [0, {root_y}, 0]
    box body_core {{ size: [{chest_sz[0]}, {chest_sz[1]}, {chest_sz[2]}], material: hide }}
    bone hind_leg_l attach(parent_box=body_core, anchor=[-1, -0.65, 0], align=[1, 1, 0]) mirror_x(hind_leg_r) {{
      joint_limits: [[-65, 65], [-25, 25], [-25, 25]]
      box thigh {{ size: [2, {upper_h}, 2.5], material: hide }}
      bone shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [{rest_knee_deg}, 0, 0]
        joint_limits: [[-55, 65], [-20, 20], [-20, 20]]
        box tarsus {{ size: [1.5, {lower_h}, 2], material: feather }}
      }}
    }}
    bone wing_l attach(parent_box=body_core, anchor=[-1, 0.7, 0.3], align=[1, 0, 0]) mirror_x(wing_r) {{
      rest_euler: [0, 0, 15]
      joint_limits: [[-35, 35], [-35, 35], [-55, 55]]
      box humerus {{ size: [{wing_span}, 1.5, 2], material: feather }}
      bone forearm attach(parent_box=humerus, anchor=[-1, 0, 0], align=[1, 0, 0]) {{
        joint_limits: [[-35, 35], [-35, 35], [-55, 55]]
        box ulna {{ size: [{wing_span}, 1, 1.5], material: feather }}
      }}
    }}
    bone neck attach(parent_box=body_core, anchor=[0, 0.5, 1], align=[0, 0, -1], seat=0.25) {{
      rest_euler: [-20, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_seg {{ size: [2, 2.5, 4], material: hide }}
      bone skull attach(parent_box=neck_seg, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {{
        rest_euler: [18, 0, 0]
        joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
        box head_box {{ size: [2.5, 2.5, 4], material: feather }}
      }}
    }}
    bone tail_1 attach(parent_box=body_core, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {{
      joint_limits: [[-20, 20], [-30, 30], [-15, 15]]
      box tail_box {{ size: [2, 1.5, 6], material: hide }}
    }}
  }}
}}
@behavior {{
  {UNCHANGED_WALK_GAIT}
  {UNCHANGED_TROT_GAIT}
}}
"""


def _hexapod_beetle_dsl() -> str:
    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  carapace: ramp(base_lch=[0.32, 0.08, 165], steps=5)
  horn: ramp(base_lch=[0.65, 0.10, 75], steps=4)
}}
@rig {{
  root bone thorax {{
    pivot_local: [0, 4.2, 0]
    box shell {{ size: [6, 3.5, 9], material: carapace }}
    bone head attach(parent_box=shell, anchor=[0, -0.1, 1], align=[0, 0, -1], seat=0.25) {{
      joint_limits: [[-20, 20], [-20, 20], [-15, 15]]
      box head_cap {{ size: [4, 2.5, 3], material: carapace }}
    }}
    bone leg_fore_l attach(parent_box=shell, anchor=[-1, -0.4, 0.72], align=[1, 0, 0]) mirror_x(leg_fore_r) {{
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box femur_f {{ size: [3, 1, 1], material: horn }}
      bone tibia_fore_l attach(parent_box=femur_f, anchor=[-1, 0, 0], align=[0, 1, 0]) {{
        joint_limits: [[-65, 65], [-55, 55], [-65, 65]]
        box shin_f {{ size: [1, 3.5, 1], material: horn }}
      }}
    }}
    bone leg_mid_l attach(parent_box=shell, anchor=[-1, -0.4, 0.0], align=[1, 0, 0]) mirror_x(leg_mid_r) {{
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box femur_m {{ size: [3, 1, 1], material: horn }}
      bone tibia_mid_l attach(parent_box=femur_m, anchor=[-1, 0, 0], align=[0, 1, 0]) {{
        joint_limits: [[-65, 65], [-55, 55], [-65, 65]]
        box shin_m {{ size: [1, 3.5, 1], material: horn }}
      }}
    }}
    bone leg_hind_l attach(parent_box=shell, anchor=[-1, -0.4, -0.72], align=[1, 0, 0]) mirror_x(leg_hind_r) {{
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box femur_h {{ size: [3, 1, 1], material: horn }}
      bone tibia_hind_l attach(parent_box=femur_h, anchor=[-1, 0, 0], align=[0, 1, 0]) {{
        joint_limits: [[-65, 65], [-55, 55], [-65, 65]]
        box shin_h {{ size: [1, 3.5, 1], material: horn }}
      }}
    }}
  }}
}}
@behavior {{
  {UNCHANGED_WALK_GAIT}
  {UNCHANGED_TROT_GAIT}
}}
"""


def build_ten_body_zoo() -> dict[str, tuple[str, int, str]]:
    """Return 10 diverse creature bodies -> `(dsl_source, expected_leg_count, expected_posture)`."""
    return {
        "fox": (_quadruped_species_dsl(chest_sz=(4, 3.5, 5), hip_sz=(3.5, 3, 4.5), upper_h=3.0, lower_h=3.0, rest_knee_deg=24.0), 4, "crouched"),
        "stag": (_quadruped_species_dsl(chest_sz=(5, 5, 7), hip_sz=(4.5, 4.5, 6), upper_h=5.5, lower_h=5.5, rest_knee_deg=6.0), 4, "erect"),
        "bear": (_quadruped_species_dsl(chest_sz=(7, 6, 8), hip_sz=(6.5, 5.5, 7), upper_h=4.5, lower_h=4.5, rest_knee_deg=8.0), 4, "erect"),
        "sabre_drake": (_quadruped_species_dsl(chest_sz=(6, 5, 7), hip_sz=(5, 4.5, 6.5), upper_h=4.5, lower_h=4.5, rest_knee_deg=26.0), 4, "crouched"),
        "elder_dragon": (_quadruped_species_dsl(chest_sz=(9, 7, 10), hip_sz=(7.5, 6, 9), upper_h=7.0, lower_h=7.0, rest_knee_deg=28.0, has_wings=True), 4, "crouched"),
        "hydra": (_quadruped_species_dsl(chest_sz=(8, 6, 9), hip_sz=(7, 5.5, 8), upper_h=5.5, lower_h=5.5, rest_knee_deg=22.0, three_necks=True), 4, "crouched"),
        "wyvern": (_biped_species_dsl(chest_sz=(6, 5.5, 8), upper_h=5.0, lower_h=5.0, rest_knee_deg=28.0, wing_span=8.0), 2, "crouched"),
        "ground_eagle": (_biped_species_dsl(chest_sz=(4.5, 4, 6), upper_h=3.5, lower_h=4.0, rest_knee_deg=24.0, wing_span=5.5), 2, "crouched"),
        "crocodile": (_quadruped_species_dsl(chest_sz=(6, 3.5, 8), hip_sz=(5, 3, 7), upper_h=3.5, lower_h=3.0, rest_knee_deg=0.0, sprawled=True), 4, "sprawled"),
        "beetle": (_hexapod_beetle_dsl(), 6, "sprawled"),
    }


class GateABCAndSculptedStyleTests(unittest.TestCase):
    # -----------------------------------------------------------------------
    # Gate A — All 5 Reviewer Correctness Probes + Speed Benchmark
    # -----------------------------------------------------------------------

    def test_gate_a_fix1_joint_limits_hard_error_continuous_spin_and_export_parity(self) -> None:
        base = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.0" }
        @palette { m: ramp(base_lch=[0.6, 0.1, 80], steps=4) }
        @rig {
          root bone body {
            pivot_local: [0, 4, 0]
            box core { size: [4, 4, 4], material: m }
            bone rotor attach(parent_box=core, anchor=[1, 0, 0], align=[-1, 0, 0]) {
              joint_limits: [[-45, 45], [-10, 10], [-10, 10]]
              box blade { size: [1, 3, 3], material: m }
            }
          }
        }
        """
        # 1. Analytic spin / wave exceeding finite joint_limits MUST raise DSLValidationError
        with self.assertRaisesRegex(DSLValidationError, "joint_limits"):
            parse_program(base + '\n@behavior { clip c { duration_s: 1.0 track rotor: spin(axis=x, amp_deg=360, freq_hz=1.0) } }')

        # 2. Keyframe exceeding finite joint_limits MUST raise DSLValidationError (never exported unclamped!)
        with self.assertRaisesRegex(DSLValidationError, "joint_limits"):
            parse_program(base + '\n@behavior { clip c { duration_s: 1.0 key rotor.rotation { 0.0: [0, 0, 0], 1.0: [90, 0, 0] } } }')

        # 3. `joint_limits: continuous` allows multi-revolution spin & preserves unwrapped angles in BOTH Bedrock and .bbmodel
        cont_dsl = base.replace("[[-45, 45], [-10, 10], [-10, 10]]", "[continuous, [-10, 10], [-10, 10]]") + (
            '\n@behavior { clip c { duration_s: 2.0 track rotor: spin(axis=x, amp_deg=-720, freq_hz=0.5) } }'
        )
        cont_prog = parse_program(cont_dsl)
        self.assertEqual(parse_program(format_program(cont_prog)), cont_prog)
        cont_sc = compile_geometry(cont_prog)
        bed = export_bedrock(cont_sc, program=cont_prog, num_animation_frames=4)
        bed_rot = bed["animations"]["animations"]["animation.entity.c"]["bones"]["rotor"]["rotation"]
        self.assertAlmostEqual(bed_rot["2.0"][0], 720.0, places=3)

        bb = export_bbmodel(cont_sc, program=cont_prog, num_animation_frames=4)
        rotor_anim = next(a for a in bb["animations"][0]["animators"].values() if a["name"] == "rotor")
        self.assertAlmostEqual(rotor_anim["keyframes"][-1]["data_points"][0]["x"], -720.0, places=3)

        # 4. Valid keyframe clip has exact pose parity between audit evaluation and Bedrock export
        valid_kf_dsl = base + '\n@behavior { clip c { duration_s: 1.0, loop: false key rotor.rotation { 0.0: [0, 0, 0], 1.0: [30, 0, 0] } } }'
        kf_prog = parse_program(valid_kf_dsl)
        kf_samples = evaluate_behavior_clip(kf_prog, "c", sample_phases=[0.0, 1.0])
        kf_bed = export_bedrock(compile_geometry(kf_prog), program=kf_prog)
        exp_end = kf_bed["animations"]["animations"]["animation.entity.c"]["bones"]["rotor"]["rotation"]["1.0"]
        # Bedrock negates X angle relative to right-handed SyncCraft
        self.assertAlmostEqual(-exp_end[0], kf_samples[-1].pose_euler_deltas["rotor"][0], places=4)

    def test_gate_a_fix2_unified_swept_audit_catches_midpoint_key_collision_and_high_freq(self) -> None:
        # 1. v3.2 `key` clip sweeping arm 180 deg right through torso at midpoint t = 0.5s
        sweep_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "3.2" }
        @palette { m: ramp(base_lch=[0.5, 0.08, 200], steps=4) }
        @rig {
          root bone root_b {
            pivot_local: [0, 8, 0]
            box torso { size: [6, 8, 4], material: m }
            bone obstacle attach(parent_box=torso, anchor=[0, 0, 1], align=[0, 0, -1], detached=true, offset=[0, 0, 3]) {
              box wall { size: [6, 4, 2], material: m }
            }
            bone arm attach(parent_box=torso, anchor=[0, 1, 0], align=[0, -1, 0]) {
              joint_limits: [[-180, 180], [-10, 10], [-10, 10]]
              box club { size: [2, 8, 2], material: m }
            }
          }
        }
        @behavior {
          clip smash {
            duration_s: 1.0, loop: false
            key arm.rotation { 0.0: [0, 0, 0], 1.0: [180, 0, 0] }
          }
        }
        """
        sweep_prog = parse_program(sweep_dsl)
        rep = audit_behavior_clip(sweep_prog, "smash", num_frames=2)
        self.assertEqual(rep["motion_audit_method"], "adaptive_discrete_step")
        self.assertLessEqual(rep["measured_max_step_deg"], 10.0)
        self.assertFalse(rep["swept_collision_free"])
        self.assertFalse(rep["passed"])

        # 2. High-frequency clip exceeding 144-frame step budget MUST fail (`step_bound_satisfied == False`)
        hf_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }
        @palette { m: ramp(base_lch=[0.5, 0.08, 200], steps=4) }
        @rig {
          root bone root_b {
            pivot_local: [0, 6, 0]
            box torso { size: [4, 4, 4], material: m }
            bone arm attach(parent_box=torso, anchor=[1, 0, 0], align=[-1, 0, 0]) {
              box bar { size: [4, 1, 1], material: m }
            }
          }
        }
        @behavior {
          clip buzz {
            duration_s: 1.0
            track arm: wave(axis=z, amp_deg=90, freq_hz=20.0)
          }
        }
        """
        hf_rep = audit_behavior_clip(parse_program(hf_dsl), "buzz")
        self.assertEqual(hf_rep["num_frames"], 144)
        self.assertFalse(hf_rep["step_bound_satisfied"])
        self.assertFalse(hf_rep["passed"])

    def test_gate_a_fix3_sculpted_burial_rejection_cross_bone_blend_and_octagon(self) -> None:
        # 1. Two 8x8x8 cubes on one bone overlapping 6.5u deep (81.25% buried) -> REJECTED
        cubes_65 = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box a { size: [8, 8, 8], material: m }
                box b place(base_box=a, anchor=[1, 0, 0], align=[1, 0, 0], offset=[1.5, 0, 0]) {
                  size: [8, 8, 8], material: m
                }
              }
            }
            """
        )
        rep_65 = audit_scene(compile_geometry(cubes_65))
        self.assertFalse(rep_65["passed"])
        self.assertEqual(rep_65["contact_violations"][0]["violation"], "swallowed_redundant_box")

        # 2. An 8x1x8 plate buried 7.7/8 (96.25% buried) inside a 10x6x10 torso -> REJECTED
        plate_77 = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, size_step: 0.5, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.5, 0.05, 100], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box torso { size: [10, 6, 10], material: m }
                box plate place(base_box=torso, anchor=[1, 0, 0], align=[1, 0, 0], offset=[0.3, 0, 0]) {
                  size: [8, 1, 8], material: m
                }
              }
            }
            """
        )
        rep_77 = audit_scene(compile_geometry(plate_77))
        self.assertFalse(rep_77["passed"])
        self.assertEqual(rep_77["contact_violations"][0]["violation"], "swallowed_redundant_box")

        # 3. Joint-bridging crest plate across grandparent (`head` -> `neck_1` -> `torso`):
        # fails without `blend_with`, passes with cross-bone `blend_with: [torso_box]`!
        bridge_base = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
        @palette { m: ramp(base_lch=[0.5, 0.08, 180], steps=4) }
        @rig {
          root bone torso {
            pivot_local: [0, 6, 0]
            box torso_box { size: [6, 5, 6], material: m }
            bone neck_1 attach(parent_box=torso_box, anchor=[0, 0, 1], align=[0, 0, -1]) {
              box neck_box { size: [4, 4, 3], material: m }
              bone head attach(parent_box=neck_box, anchor=[0, 0, 1], align=[0, 0, -1]) {
                box skull_box { size: [5, 4, 4], material: m }
                box crest_plate place(base_box=skull_box, anchor=[0, 1, -1], align=[0, 0, 0.2]) {
                  size: [2, 2, 6], material: m, rotate: [-12, 0, 0]__BLEND__
                }
              }
            }
          }
        }
        """
        no_blend_aud = audit_scene(compile_geometry(parse_program(bridge_base.replace("__BLEND__", ""))))
        self.assertFalse(no_blend_aud["passed"])
        with_blend_prog = parse_program(bridge_base.replace("__BLEND__", ", blend_with: [torso_box]"))
        with_blend_aud = audit_scene(compile_geometry(with_blend_prog))
        self.assertTrue(with_blend_aud["passed"], msg=str(with_blend_aud))

        # 4. Raw concentric boxes rejected, whereas `octagon` primitive passes as `sculpted_prism_hub`!
        raw_oct = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.6, 0.1, 80], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                box hub_0 { size: [6, 6, 3], material: m }
                box hub_1 place(base_box=hub_0, anchor=[0, 0, 0], align=[0, 0, 0]) {
                  size: [6, 6, 3], material: m, rotate: [0, 0, 45]
                }
              }
            }
            """
        )
        self.assertFalse(audit_scene(compile_geometry(raw_oct))["passed"])

        prim_oct = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }
            @palette { m: ramp(base_lch=[0.6, 0.1, 80], steps=4) }
            @rig {
              root bone r {
                pivot_local: [0, 4, 0]
                octagon hub { size: [6, 6, 3], axis: z, material: m }
              }
            }
            """
        )
        oct_aud = audit_scene(compile_geometry(prim_oct))
        self.assertTrue(oct_aud["passed"], msg=str(oct_aud))
        self.assertEqual(oct_aud["sculpted_overlap_pairs"][0]["exemption_reason"], "sculpted_prism_hub")

    def test_gate_a_fix4_and_fix5_layout_contracts_in_audit_scene_grounded_and_v4_version_gate(self) -> None:
        # Fix #5: v4 `@layout`/`@component`/`@assembly` without `dsl_version: "4.0"` MUST fail
        v4_missing_ver = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16 }
        @layout { budget max_boxes = 1 }
        @palette { p: ramp(base_lch=[0.5, 0.05, 200], steps=4) }
        @component TwoBoxes {
          material_slots: [m]
          box a { size: [4, 4, 4], material: m }
          box b place(base_box=a, anchor=[0, 1, 0], align=[0, -1, 0]) { size: [2, 2, 2], material: m }
        }
        @assembly A {
          root_pivot: [0, 2, 0]
          instance x : TwoBoxes() { id: "x1", materials: { m: p } }
        }
        """
        with self.assertRaisesRegex(DSLValidationError, "dsl_version"):
            parse_program(v4_missing_ver)

        # Fix #4: With `dsl_version: "4.0"`, `audit_scene` itself fails on `budget max_boxes = 1` for a 2-box asset!
        v4_with_ver = v4_missing_ver.replace("texel_density: 16", 'texel_density: 16, dsl_version: "4.0"')
        prog_over = parse_program(v4_with_ver)
        aud_over = audit_scene(compile_geometry(prog_over))
        self.assertFalse(aud_over["passed"])
        self.assertEqual(aud_over["contract_violations"][0]["violation"], "layout_budget_max_boxes")

        # Floating check via `envelope grounded = true` + flagship groundedness check
        for label, src in (
            ("DragonV4", MODULAR_SCULPTED_DRAGON_V4_DSL),
            ("WalkerV4", MODULAR_ARTILLERY_WALKER_V4_DSL),
            ("Wyvern", SCULPTED_ELDER_WYVERN_DSL),
            ("Citadel", COLOSSAL_WALKING_CITADEL_DSL),
        ):
            sc = compile_geometry(parse_program(src))
            aud = audit_scene(sc)
            self.assertTrue(aud["passed"], msg=f"{label} failed audit: {aud}")
            min_y = min(float(np.min(b.corners()[:, 1])) for b in sc.boxes)
            self.assertAlmostEqual(min_y, 0.0, delta=0.06, msg=f"{label} floating/sunk: min_y={min_y}")

    # -----------------------------------------------------------------------
    # Gate B (`@morphology`) & Gate C (10-Body Zero-Shot Froude `gait` Transfer)
    # -----------------------------------------------------------------------

    def test_gate_b_morphology_and_gate_c_ten_body_zero_shot_walk_and_trot(self) -> None:
        zoo = build_ten_body_zoo()
        self.assertEqual(len(zoo), 10)

        walk_speeds: dict[str, float] = {}
        walk_strides: dict[str, float] = {}
        role_matches = 0

        for species, (dsl_src, exp_legs, exp_posture) in zoo.items():
            with self.subTest(species=species):
                prog = parse_program(dsl_src)
                # Round-trip format_program including `gait`!
                self.assertEqual(parse_program(format_program(prog)), prog)

                sc = compile_geometry(prog)
                static_aud = audit_scene(sc)
                self.assertTrue(static_aud["passed"], msg=f"{species} static audit failed: {static_aud}")

                morph = analyze_morphology(prog, sc)
                morph_txt = format_morphology_block(morph)
                self.assertIn("@morphology {", morph_txt)

                self.assertEqual(len(morph.limbs), exp_legs, msg=f"{species}: got {[l.role for l in morph.limbs]}")
                postures = {lb.posture for lb in morph.limbs}
                self.assertIn(exp_posture, postures, msg=f"{species}: postures={postures}")
                if species in ("elder_dragon", "wyvern", "ground_eagle"):
                    self.assertEqual(len(morph.wings), 2)
                role_matches += 1

                for clip_name in ("walk", "trot"):
                    rep = audit_behavior_clip(prog, clip_name)
                    self.assertTrue(
                        rep["passed"],
                        msg=f"{species}.{clip_name} failed gait audit: {rep}",
                    )
                    ga = rep["gait_audit"]
                    for chk_name, chk_ok in ga["checks"].items():
                        self.assertTrue(chk_ok, msg=f"{species}.{clip_name} failed {chk_name}: {ga}")
                    self.assertLessEqual(ga["max_stance_slip_units"], 0.02)

                    if clip_name == "walk":
                        walk_speeds[species] = rep["root_motion"]["speed_units_per_s"]
                        walk_strides[species] = ga["stride_length"]

        # Gate B >= 90% role accuracy (10/10 == 100%)
        self.assertGreaterEqual(role_matches / float(len(zoo)), 0.90)

        # Froude dynamic similarity: tall elder_dragon walks faster with longer strides than small fox & beetle
        self.assertGreater(walk_speeds["elder_dragon"], walk_speeds["fox"])
        self.assertGreater(walk_strides["elder_dragon"], walk_strides["beetle"] * 2.0)

        # Root motion exported into Bedrock & .bbmodel animations
        fox_prog = parse_program(zoo["fox"][0])
        fox_sc = compile_geometry(fox_prog)
        bed = export_bedrock(fox_sc, program=fox_prog)
        walk_anim = bed["animations"]["animations"]["animation.entity.walk"]
        self.assertIn("synccraft_root_motion", walk_anim)
        self.assertGreater(walk_anim["synccraft_root_motion"]["speed_blocks_per_s"], 0.2)

    # -----------------------------------------------------------------------
    # Sculpted Style: `edge` Alpha Cut-Outs, Ray Pass-Through, `banded`, `@stamps`, `< 2s` Speed
    # -----------------------------------------------------------------------

    def test_sculpted_edge_alpha_cutout_ray_passthrough_banded_shading_and_stamps(self) -> None:
        cutout_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
        @palette {
          gold_bg: ramp(base_lch=[0.82, 0.16, 85], steps=4)
          dark_screen: ramp(base_lch=[0.28, 0.06, 210], steps=4)
          eye_glow: ramp(base_lch=[0.78, 0.20, 345], steps=3, emissive=true)
        }
        @rig {
          root bone stage {
            pivot_local: [0, 5, 0]
            box backdrop { size: [8, 8, 2], material: gold_bg }
            plane curtain place(base_box=backdrop, anchor=[0, 0, 1], align=[0, 0, 0], offset=[0, 0, 2]) {
              size: [8, 8], normal: z, material: dark_screen,
              edge: spikes(width=2, length=3..6, side=bottom, seed=11)
            }
          }
        }
        @materials {
          gold_bg: process(family=patterned_skin, coat=plain)
          dark_screen: process(family=patterned_skin, coat=scale_rows)
        }
        @style {
          target_encoding: vanilla,
          shade: banded(steps=4, direction=down, cluster=diagonal),
          glow_bleed: 2
        }
        @stamps {
          stamp gem on backdrop.south { at: [0.3, 0.35], size: [2, 2], color: eye_glow, emissive: 1.0, mirror: true }
        }
        """
        prog = parse_program(cutout_dsl)
        self.assertEqual(parse_program(format_program(prog)), prog)

        mp = MaterialProgram.from_program(prog)
        # `mirror: true` on `backdrop.south` at u=0.3 expands to 2 bilateral stamps!
        self.assertEqual(len(mp.stamps), 2)
        st_cfg = StyleConfig.from_program(prog)
        self.assertEqual(st_cfg.shade_mode, "banded")
        self.assertEqual(st_cfg.glow_bleed, 2)

        dom = build_surface_domain(prog, mp, subdivisions_per_unit=1)
        ws = WorldState(
            program=prog,
            material_program=mp,
            surface_state=initialize_surface_state(dom),
            style_config=st_cfg,
        )
        atlas_desc, png_bytes = compile_vanilla_atlas(ws)
        rgba = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"), dtype=np.uint8)
        u0, v0, u1, v1 = atlas_desc["faces"]["curtain"]["south"]
        curtain_island = rgba[v0:v1, u0:u1]
        # Binary alpha cutout: contains both transparent (0) and opaque (255) pixels!
        unique_alphas = set(np.unique(curtain_island[..., 3]).tolist())
        self.assertEqual(unique_alphas, {0, 255})

        # Render from `south` (+Z camera looking through `curtain` onto `backdrop` behind it!)
        obs = render_world_observation(ws, atlas_override=(atlas_desc, png_bytes), views=("south",), panel_size=(96, 96))
        hit_map = obs["hit_box_maps"]["south"]
        # Both `backdrop` (idx 0, seen RIGHT THROUGH the cut-out notches of `curtain`!) and `curtain` (idx 1) are hit!
        self.assertIn(0, set(np.unique(hit_map).tolist()))
        self.assertIn(1, set(np.unique(hit_map).tolist()))

    def test_interactive_120_plus_box_sculpted_dragon_verifies_and_bakes_under_2_seconds(self) -> None:
        # Construct a 126-box sculpted glade dragon using chains, shingles, planes, octagons, banded shade & stamps
        shingle_boxes = "\n".join(
            f"    box dorsal_shingle_{i} place(base_box=chest, anchor=[0, 1, {round(0.85 - i * 0.14, 3)}], align=[0, -0.5, 0]) {{\n"
            f"      size: [2, 2.5, 2], material: accent, rotate: [-18, 0, 0]\n"
            f"    }}"
            for i in range(12)
        )
        rib_plates = "\n".join(
            f"    box flank_plate_l_{i} place(base_box=chest, anchor=[-1, {round(0.55 - (i % 3) * 0.4, 2)}, {round(0.28 - (i // 3) * 0.22, 3)}], align=[0.4, 0, 0]) {{\n"
            f"      size: [1.5, 2, 2], material: hide, rotate: [0, 8, 12]\n"
            f"    }}\n"
            f"    box flank_plate_r_{i} place(base_box=chest, anchor=[1, {round(0.55 - (i % 3) * 0.4, 2)}, {round(0.28 - (i // 3) * 0.22, 3)}], align=[-0.4, 0, 0]) {{\n"
            f"      size: [1.5, 2, 2], material: hide, rotate: [0, -8, -12]\n"
            f"    }}"
            for i in range(18)
        )
        belly_plates = "\n".join(
            f"    box belly_scute_{i} place(base_box=chest, anchor=[0, -1, {round(0.85 - i * 0.16, 3)}], align=[0, 0.5, 0]) {{\n"
            f"      size: [6, 1.5, 2.5], material: belly\n"
            f"    }}"
            for i in range(11)
        )
        dsl_126 = f"""
        @meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
        @palette {{
          hide: ramp(base_lch=[0.48, 0.11, 188], steps=4)
          belly: ramp(base_lch=[0.82, 0.05, 85], steps=4)
          accent: ramp(base_lch=[0.70, 0.16, 345], steps=4, emissive=true)
        }}
        @rig {{
          root bone torso {{
            pivot_local: [0, 11.5, 0]
            box chest {{ size: [9, 7, 16], material: hide }}
            octagon shoulder_hub_l place(base_box=chest, anchor=[-1, -0.3, 0.65], align=[0.5, 0, 0]) {{
              size: [3.5, 3.5, 2], axis: x, material: hide
            }}
            octagon shoulder_hub_r place(base_box=chest, anchor=[1, -0.3, 0.65], align=[-0.5, 0, 0]) {{
              size: [3.5, 3.5, 2], axis: x, material: hide
            }}
            {shingle_boxes}
            {rib_plates}
            {belly_plates}
            chain neck attach(parent_box=chest, anchor=[0, 0.45, 1], align=[0, 0, -1], seat=0.28) segments=5 taper=0.88 bend=[-16, -12, 4, 14, 10] {{
              box seg {{ size: [5, 4.5, 4], material: hide }}
              plane crest place(base_box=seg, anchor=[0, 1, 0], align=[0, -1, 0]) {{
                size: [2.5, 2.5], normal: x, material: accent, edge: spikes(width=2, length=1..3, side=top), layer: 1
              }}
            }}
            chain tail attach(parent_box=chest, anchor=[0, 0.15, -1], align=[0, 0, 1], seat=0.28) segments=8 taper=0.86 bend=[-4, -3, 2, 5, 6, 6, 4, 2] {{
              box seg {{ size: [5, 4, 4.5], material: hide }}
              plane fin place(base_box=seg, anchor=[0, 1, 0], align=[0, -1, 0]) {{
                size: [2, 4], normal: x, material: accent, edge: ragged(depth=2, side=top, seed=9), layer: 1
              }}
              box tail_scute_l place(base_box=seg, anchor=[-1, 0, -0.45], align=[0.4, 0, 0]) {{
                size: [1.5, 1.5, 2], material: hide, rotate: [0, 0, 14]
              }}
              box tail_scute_r place(base_box=seg, anchor=[1, 0, -0.45], align=[-0.4, 0, 0]) {{
                size: [1.5, 1.5, 2], material: hide, rotate: [0, 0, -14]
              }}
            }}
            bone wing_l attach(parent_box=chest, anchor=[-1, 0.75, 0.3], align=[1, 0, 0]) mirror_x(wing_r) {{
              rest_euler: [0, 0, 18]
              box spar_1 {{ size: [9, 2, 2.5], material: hide }}
              plane bay_1 place(base_box=spar_1, anchor=[0, 0, -1], align=[0, 0, 1]) {{
                size: [8, 7], normal: y, material: belly, edge: scallop(bays=3, depth=2, side=bottom), layer: 1
              }}
              bone forearm_l attach(parent_box=spar_1, anchor=[-1, 0, 0], align=[1, 0, 0]) {{
                box spar_2 {{ size: [10, 1.5, 2], material: hide }}
                plane bay_2 place(base_box=spar_2, anchor=[0, 0, -1], align=[0, 0, 1]) {{
                  size: [9, 8], normal: y, material: belly, edge: ragged(depth=3, holes=2, side=bottom, seed=5), layer: 2
                }}
              }}
            }}
            bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.7, 0.55], align=[1, 1, 0]) mirror_x(fore_leg_r) {{
              box f_thigh {{ size: [2.5, 5, 3], material: hide }}
              bone fore_shin_l attach(parent_box=f_thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
                rest_euler: [12, 0, 0]
                box f_shin {{ size: [2, 5, 2.5], material: hide }}
              }}
            }}
            bone hind_leg_l attach(parent_box=chest, anchor=[-1, -0.7, -0.65], align=[1, 1, 0]) mirror_x(hind_leg_r) {{
              box h_thigh {{ size: [3, 5, 3.5], material: hide }}
              bone hind_shin_l attach(parent_box=h_thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
                rest_euler: [18, 0, 0]
                box h_shin {{ size: [2, 5, 2.5], material: hide }}
              }}
            }}
          }}
        }}
        @materials {{
          hide: process(family=patterned_skin, coat=plain)
          belly: process(family=patterned_skin, coat=plain)
          accent: process(family=patterned_skin, coat=plain, emissive=0.8)
        }}
        @style {{
          target_encoding: vanilla,
          shade: banded(steps=4, direction=down, cluster=diagonal),
          glow_bleed: 1
        }}
        @stamps {{
          stamp flank_mark on chest.west {{ at: [0.25, 0.4], size: [2, 2], color: accent, emissive: 0.9, mirror: true }}
        }}
        """
        t0 = time.perf_counter()
        prog = parse_program(dsl_126)
        sc = compile_geometry(prog)
        self.assertGreaterEqual(len(sc.boxes), 120)
        aud = audit_scene(sc)
        self.assertTrue(aud["passed"], msg=str(aud))

        mp = MaterialProgram.from_program(prog)
        st = StyleConfig.from_program(prog)
        dom = build_surface_domain(prog, mp, subdivisions_per_unit=1)
        ws = WorldState(program=prog, material_program=mp, surface_state=initialize_surface_state(dom), style_config=st)
        atlas, png = compile_vanilla_atlas(ws)
        bed = export_bedrock(sc, atlas=atlas, program=prog)
        elapsed = time.perf_counter() - t0

        self.assertGreater(len(png), 512)
        self.assertIn("minecraft:geometry", bed["geometry"])
        self.assertLess(elapsed, 2.0, msg=f"120+ box verify+bake+export took {elapsed:.3f}s (> 2.0s budget)")


if __name__ == "__main__":
    unittest.main()
