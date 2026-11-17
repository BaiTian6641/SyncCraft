"""Modular Block-DSL v4.0 Exemplars, World-Model Supervision Tasks & Inferred-Subtree Decompiler (ADR 0008).

Bridges CAD-inspired Block-DSL v4.0 assemblies (`@layout`, `@component`, `@assembly`, `SourceMap`)
directly into SyncCraft's `LiquidAI/LFM3-VL-2B` constructive 3D world-model pipeline:

1. Two clean-room flagship Block-DSL v4.0 demonstrations:
   - ``MODULAR_SCULPTED_DRAGON_V4_DSL`` (Organic sculpted creature: top-down `@layout`, nested
     `head_group` subassembly, `fixed` horns welded onto `skull` bone with `sculpted_blend`,
     `revolute` jaw, 4 independent instances of a single `DragonLimb` `@component` with
     `foot_cycle` IK, Hill `@muscles`, and `@controller`).
   - ``MODULAR_ARTILLERY_WALKER_V4_DSL`` (Strict mechanical walker: top-down `@layout`,
     `revolute` 360° turret + elevation cradle, `prismatic` cannon recoil + reciprocating steam
     pistons exported to native `position` channels, `uv_policy: shared` across repeated gears
     and leg struts, and adaptive `<= 10°` swept motion validation).
2. Four compositional World-Model training & evaluation tasks:
   - ``plan_layout_and_interfaces``
   - ``synthesize_component_from_contract``
   - ``surgical_component_swap_and_repair`` (measuring exact non-target 8-corner vertex invariance)
   - ``propagate_layout_and_instance_controls``
3. Honest decompiled assembly inference (`infer_assembly_from_program`) tagging recovered
   subtrees from flat `.bbmodel` / Bedrock imports strictly as ``assembly_provenance="inferred_subtree"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.assembly_ir import (
    AssemblyDecl,
    AssemblySpec,
    AttachmentFrameDecl,
    ComponentBoxTemplate,
    ComponentDef,
    ComponentInstanceDecl,
    ConnectionDecl,
    SourceMap,
    SourceMapEntry,
    audit_assembly_contracts,
    evaluate_layout_dag,
    format_assembly_sections,
    override_layout_params,
    swap_component_definition,
)
from synccraft.dsl.ast_nodes import Bone, Box, Locator, Program
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.musculoskeletal import audit_behavior_clip
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from synccraft.surface.contracts import MaterialProgram, StyleConfig, WorldState
from synccraft.surface.domain import build_surface_domain, initialize_surface_state


# ---------------------------------------------------------------------------
# 1. Flagship Block-DSL v4.0 Organic & Mechanical Demonstrations
# ---------------------------------------------------------------------------

MODULAR_SCULPTED_DRAGON_V4_DSL: str = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.0", audit_profile: sculpted }

@layout {
  param body_length = 26 [18, 48]
  param chest_width = 10 [8, 18]
  param wing_span_seg = 8 [5, 14]
  let head_width = max(4, round(chest_width * 0.6))
  let pelvis_len = max(6, round(body_length * 0.35))
  let chest_len = max(8, body_length - pelvis_len - 6)
  budget max_boxes = 48
  budget max_bones = 28
  envelope max_span = [56, 38, 60]
  envelope min_ground_y = -0.5
  envelope grounded = true
}

@palette {
  obsidian_scale: ramp(base_lch=[0.34, 0.06, 262], steps=8, hue_shift=[-5, 4])
  pale_sternum: ramp(base_lch=[0.74, 0.05, 80], steps=6)
  horn_ivory: ramp(base_lch=[0.83, 0.06, 78], steps=6)
  ember_organ: ramp(base_lch=[0.68, 0.19, 36], steps=8, emissive=true)
}

@component DragonPelvis {
  param w = 8 [6, 18]
  param l = 9 [6, 20]
  material_slots: [hide, crest]
  box pelvis_core { size: [w, 6, l], material: hide }
  box sacral_ridge place(base_box=pelvis_core, anchor=[0, 1, 0], align=[0, -0.5, 0]) {
    size: [2, 2, max(3, l - 3)], material: crest, rotate: [-8, 0, 0]
  }
  frame chest_socket { on_box: pelvis_core, anchor: [0, 0.15, 1], rot: [0, 0, 0] }
  frame hip_l { on_box: pelvis_core, anchor: [-1, -0.35, 0], rot: [0, 0, 0] }
  frame hip_r { on_box: pelvis_core, anchor: [1, -0.35, 0], rot: [0, 0, 0] }
  frame tail_socket { on_box: pelvis_core, anchor: [0, 0.1, -1], rot: [-6, 0, 0] }
}

@component DragonTorso {
  param w = 10 [6, 20]
  param l = 11 [6, 24]
  material_slots: [hide, belly, crest]
  box ribcage { size: [w, 7, l], material: hide }
  box keel place(base_box=ribcage, anchor=[0, -1, 0.1], align=[0, 0.6, 0]) {
    size: [max(4, w - 4), 2, max(4, l - 2)], material: belly, rotate: [-6, 0, 0]
  }
  box dorsal_scute place(base_box=ribcage, anchor=[0, 1, 0], align=[0, -0.5, 0]) {
    size: [2, 3, max(4, l - 3)], material: crest, rotate: [10, 0, 0]
  }
  frame waist_mount { on_box: ribcage, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame neck_socket { on_box: ribcage, anchor: [0, 0.35, 1], rot: [-16, 0, 0] }
  frame wing_socket_l { on_box: ribcage, anchor: [-1, 0.75, 0.15], rot: [0, 0, 0] }
  frame wing_socket_r { on_box: ribcage, anchor: [1, 0.75, 0.15], rot: [0, 0, 0] }
  frame shoulder_l { on_box: ribcage, anchor: [-1, -0.45, 0.55], rot: [0, 0, 0] }
  frame shoulder_r { on_box: ribcage, anchor: [1, -0.45, 0.55], rot: [0, 0, 0] }
  frame saddle_point { on_box: ribcage, anchor: [0, 1, -0.2], offset: [0, 2, 0], role: rider_seat }
}

@component DragonNeck {
  param len = 6 [4, 14]
  material_slots: [hide]
  box neck_core { size: [4, 4, len], material: hide }
  frame base_mount { on_box: neck_core, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame head_socket { on_box: neck_core, anchor: [0, 0.15, 1], rot: [14, 0, 0] }
}

@component DragonSkull {
  param w = 6 [4, 14]
  material_slots: [hide, organ]
  box cranium { size: [w, 4, 6], material: hide }
  box snout place(base_box=cranium, anchor=[0, -0.1, 1], align=[0, 0, -0.8]) {
    size: [max(2, w - 2), 3, 5], material: hide, rotate: [10, 0, 0]
  }
  frame neck_mount { on_box: cranium, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame horn_socket_l { on_box: cranium, anchor: [-0.72, 1, -0.25], rot: [32, -14, 0] }
  frame horn_socket_r { on_box: cranium, anchor: [0.72, 1, -0.25], rot: [32, 14, 0] }
  frame jaw_hinge { on_box: cranium, anchor: [0, -1, 0.55], rot: [0, 0, 0] }
  frame flame_emitter { on_box: snout, anchor: [0, 0, 1], rot: [10, 0, 0], role: vfx_emitter }
}

@component SweptHorn {
  param len = 6 [3, 12]
  material_slots: [keratin]
  uv_policy: shared
  box horn_shaft { size: [1, 2, len], material: keratin }
  frame base_mount { on_box: horn_shaft, anchor: [0, -1, 1], rot: [0, 0, 0] }
}

@component DragonJaw {
  param w = 4 [2, 10]
  material_slots: [belly]
  box mandible { size: [w, 1, 5], material: belly }
  frame hinge_mount { on_box: mandible, anchor: [0, 1, -1], rot: [0, 0, 0] }
}

@component DragonWing {
  param span = 8 [5, 16]
  material_slots: [hide, membrane, spar]
  box humerus { size: [span, 2, 3], material: hide }
  plane inner_web place(base_box=humerus, anchor=[0, 0, -1], align=[0, 0, 1]) {
    size: [max(3, span - 2), 6], normal: y, material: membrane
  }
  bone forearm attach(parent_box=humerus, anchor=[-1, 0, 0], align=[1, 0, 0]) {
    joint_limits: [[-20, 20], [-35, 35], [-65, 45]]
    box radius_spar { size: [span + 1, 2, 2], material: spar }
    plane outer_web place(base_box=radius_spar, anchor=[0, 0, -1], align=[0, 0, 1]) {
      size: [span, 7], normal: y, material: membrane
    }
  }
  frame root_mount { on_box: humerus, anchor: [1, 0, 0], rot: [0, 0, 0] }
}

@component DragonLimb {
  param upper_h = 6 [4, 12]
  param lower_h = 6 [4, 12]
  material_slots: [hide, talon]
  box femur { size: [3, upper_h, 3], material: hide }
  bone shin attach(parent_box=femur, anchor=[0, -1, 0], align=[0, 1, 0]) {
    rest_euler: [32, 0, 0]
    joint_limits: [[-32, 45], [-15, 15], [-15, 15]]
    box tibia { size: [2, lower_h, 3], material: talon }
  }
  frame socket_mount { on_box: femur, anchor: [1, 0.75, 0], rot: [18, 0, 0] }
}

@component TailSegment {
  param w = 4 [2, 8]
  param l = 6 [3, 12]
  material_slots: [hide]
  box caudal { size: [w, max(2, w - 1), l], material: hide }
  frame fore_mount { on_box: caudal, anchor: [0, 0, 1], rot: [0, 0, 0] }
  frame aft_socket { on_box: caudal, anchor: [0, 0, -1], rot: [0, 0, 0] }
}

@assembly ModularElderDragon {
  root_pivot: [0, 12, 0]

  instance pelvis : DragonPelvis(w=max(6, chest_width - 2), l=pelvis_len) {
    id: "pelvis_01"
    materials: { hide: obsidian_scale, crest: horn_ivory }
  }
  instance torso : DragonTorso(w=chest_width, l=chest_len) {
    id: "torso_01"
    materials: { hide: obsidian_scale, belly: pale_sternum, crest: horn_ivory }
  }

  subassembly head_group {
    instance neck : DragonNeck(len=6) {
      id: "neck_01"
      materials: { hide: obsidian_scale }
    }
    instance skull : DragonSkull(w=head_width) {
      id: "skull_01"
      materials: { hide: obsidian_scale, organ: ember_organ }
    }
    instance horn_l : SweptHorn(len=6) mirror_x(horn_r) {
      id: "horn_l_01"
      materials: { keratin: horn_ivory }
    }
    instance jaw : DragonJaw(w=max(2, head_width - 2)) {
      id: "jaw_01"
      materials: { belly: pale_sternum }
    }

    connect neck.base_mount to torso.neck_socket { type: revolute, axis: x, limits: [-30, 30] }
    connect skull.neck_mount to neck.head_socket { type: revolute, axis: x, limits: [-28, 32] }
    connect horn_l.base_mount to skull.horn_socket_l { type: fixed, sculpted_blend: true }
    connect jaw.hinge_mount to skull.jaw_hinge { type: revolute, axis: x, limits: [-5, 42] }
  }

  subassembly flight_rig {
    instance wing_l : DragonWing(span=wing_span_seg) mirror_x(wing_r) {
      id: "wing_l_01"
      materials: { hide: obsidian_scale, membrane: ember_organ, spar: horn_ivory }
    }
    connect wing_l.root_mount to torso.wing_socket_l { type: revolute, axis: z, limits: [-55, 60] }
  }

  subassembly quadruped_limbs {
    instance foreleg_l : DragonLimb(upper_h=6, lower_h=6) mirror_x(foreleg_r) {
      id: "foreleg_l_01"
      materials: { hide: obsidian_scale, talon: horn_ivory }
    }
    instance hindleg_l : DragonLimb(upper_h=6, lower_h=6) mirror_x(hindleg_r) {
      id: "hindleg_l_01"
      materials: { hide: obsidian_scale, talon: horn_ivory }
    }
    connect foreleg_l.socket_mount to torso.shoulder_l { type: revolute, axis: x, limits: [-55, 55] }
    connect hindleg_l.socket_mount to pelvis.hip_l { type: revolute, axis: x, limits: [-55, 55] }
  }

  subassembly caudal_chain {
    instance tail_1 : TailSegment(w=4, l=6) {
      id: "tail_01"
      materials: { hide: obsidian_scale }
    }
    instance tail_2 : TailSegment(w=3, l=6) {
      id: "tail_02"
      materials: { hide: obsidian_scale }
    }
    connect tail_1.fore_mount to pelvis.tail_socket { type: revolute, axis: y, limits: [-28, 28] }
    connect tail_2.fore_mount to tail_1.aft_socket { type: revolute, axis: y, limits: [-32, 32] }
  }

  connect torso.waist_mount to pelvis.chest_socket { type: revolute, axis: x, limits: [-20, 20] }
}

@materials {
  obsidian_scale: process(family=keratin_scales, roughness=0.34, coating=0.38, rd_feed=0.038, rd_kill=0.061, wear=0.25, weld_group=dragon_hide)
  pale_sternum: process(family=keratin_scales, roughness=0.46, subsurface=0.2, wear=0.28)
  horn_ivory: process(family=weathered_stone, roughness=0.52, porosity=0.18, wear=0.38)
  ember_organ: process(family=thin_film_iridescence, roughness=0.18, ior=1.58, film_nm=470, emissive=0.55)
}

@style { target_encoding: vanilla, quantize_ramp: false, bevel_highlight: 0.1 }

@muscles {
  muscle pectoralis_l {
    origin_bone: torso, insertion_bone: wing_l, action_axis: z, moment_arm_deg: -40,
    tendon_coupling: [wing_l.forearm, z, -0.45], belly_box: torso.keel, bulge_inflate: 0.14
  }
  muscle pectoralis_r {
    origin_bone: torso, insertion_bone: wing_r, action_axis: z, moment_arm_deg: 40,
    tendon_coupling: [wing_r.forearm, z, 0.45], belly_box: torso.keel, bulge_inflate: 0.14
  }
}

@kinematics {
  ik_chain gait_fl { bones: [foreleg_l, foreleg_l.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain gait_fr { bones: [foreleg_r, foreleg_r.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain gait_hl { bones: [hindleg_l, hindleg_l.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain gait_hr { bones: [hindleg_r, hindleg_r.shin], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  look_chain predator_aim { bones: [torso, neck, skull], weight_decay: 0.7 }
  spring tail_2 { stiffness: 28, damping: 5.0 }
}

@behavior {
  clip stalk_march {
    duration_s: 1.6
    ik_drive gait_fl: foot_cycle(stride=3.2, lift=1.4, phase_deg=0)
    ik_drive gait_hr: foot_cycle(stride=3.2, lift=1.4, phase_deg=0)
    ik_drive gait_fr: foot_cycle(stride=3.2, lift=1.4, phase_deg=180)
    ik_drive gait_hl: foot_cycle(stride=3.2, lift=1.4, phase_deg=180)
    track tail_1: wave(axis=y, amp_deg=10, freq_hz=0.625)
    track jaw: hinge(axis=x, amp_deg=14, freq_hz=0.625)
  }

  clip sky_hunt {
    duration_s: 2.0
    muscle_drive pectoralis_l: activation(peak=0.85, freq_hz=0.5, phase_deg=0, baseline=0.08)
    muscle_drive pectoralis_r: activation(peak=0.85, freq_hz=0.5, phase_deg=0, baseline=0.08)
    track tail_1: wave(axis=y, amp_deg=12, freq_hz=0.5)
  }

  state ember_roar {
    pose_bias jaw: [28, 0, 0]
    muscle_tone pectoralis_l: 0.3
    muscle_tone pectoralis_r: 0.3
    material_mod ember_organ: { emissive: 1.5, moisture: 0.25, wear_delta: 0.05 }
  }
}

@controller {
  initial: stalk_march
  stalk_march -> sky_hunt when altitude > 2.5, blend 0.35
  sky_hunt -> ember_roar when threat >= 0.8, blend 0.2
}
""".strip()


MODULAR_ARTILLERY_WALKER_V4_DSL: str = """
@meta { archetype: entity, target_profile: geckolib, texel_density: 16, dsl_version: "4.0" }

@layout {
  param hull_width = 14 [10, 22]
  param hull_length = 18 [14, 28]
  param barrel_length = 9 [6, 14]
  budget max_boxes = 40
  budget max_bones = 24
  envelope max_span = [44, 38, 48]
  envelope min_ground_y = -0.5
  envelope grounded = true
}

@palette {
  riveted_hull: ramp(base_lch=[0.42, 0.08, 245], steps=8, hue_shift=[-4, 4])
  forged_brass: ramp(base_lch=[0.74, 0.12, 82], steps=6)
  cast_steel: ramp(base_lch=[0.52, 0.03, 250], steps=6)
  furnace_core: ramp(base_lch=[0.72, 0.20, 38], steps=6, emissive=true)
}

@component WalkerChassis {
  param w = 14 [10, 24]
  param l = 18 [12, 30]
  material_slots: [armor, firebox]
  box main_hull { size: [w, 6, l], material: armor }
  box boiler_belly place(base_box=main_hull, anchor=[0, -1, 0], align=[0, 1, 0]) {
    size: [w - 4, 2, l - 4], material: firebox
  }
  frame turret_ring_socket { on_box: main_hull, anchor: [0, 1, 0.1], rot: [0, 0, 0] }
  frame cyl_socket_l { on_box: main_hull, anchor: [-0.65, 1, -0.65], rot: [0, 0, 0] }
  frame cyl_socket_r { on_box: main_hull, anchor: [0.65, 1, -0.65], rot: [0, 0, 0] }
  frame gear_axle_l { on_box: main_hull, anchor: [-1, 0.2, 0], rot: [0, 0, 0] }
  frame gear_axle_r { on_box: main_hull, anchor: [1, 0.2, 0], rot: [0, 0, 0] }
  frame hip_fl { on_box: main_hull, anchor: [-1, -0.5, 0.75], rot: [0, 0, 0] }
  frame hip_fr { on_box: main_hull, anchor: [1, -0.5, 0.75], rot: [0, 0, 0] }
  frame hip_hl { on_box: main_hull, anchor: [-1, -0.5, -0.75], rot: [0, 0, 0] }
  frame hip_hr { on_box: main_hull, anchor: [1, -0.5, -0.75], rot: [0, 0, 0] }
}

@component TurretYawBase {
  material_slots: [armor, brass]
  box barbette { size: [8, 3, 8], material: brass }
  box cupola place(base_box=barbette, anchor=[0, 1, 0], align=[0, -1, 0]) {
    size: [8, 4, 8], material: armor
  }
  frame ring_mount { on_box: barbette, anchor: [0, -1, 0], rot: [0, 0, 0] }
  frame trunnion_socket { on_box: cupola, anchor: [0, 0.2, 1], rot: [0, 0, 0] }
}

@component GunCradle {
  material_slots: [steel]
  box cradle_housing { size: [4, 3, 4], material: steel }
  frame trunnion_mount { on_box: cradle_housing, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame slide_rail { on_box: cradle_housing, anchor: [0, 0, 1], rot: [0, 0, 0] }
}

@component RecoilBarrel {
  param len = 9 [5, 16]
  material_slots: [brass]
  box ordnance_tube { size: [2, 2, len], material: brass }
  frame breech_slider { on_box: ordnance_tube, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame muzzle_flash { on_box: ordnance_tube, anchor: [0, 0, 1], rot: [0, 0, 0], role: vfx_emitter }
}

@component SteamCylinder {
  material_slots: [steel]
  uv_policy: shared
  box casing { size: [3, 4, 3], material: steel }
  frame deck_mount { on_box: casing, anchor: [0, -1, 0], rot: [0, 0, 0] }
  frame bore_top { on_box: casing, anchor: [0, 1, 0], rot: [0, 0, 0] }
}

@component SteamPistonRod {
  material_slots: [brass]
  uv_policy: shared
  box plunger { size: [1, 4, 1], material: brass }
  frame rod_base { on_box: plunger, anchor: [0, -1, 0], rot: [0, 0, 0] }
}

@component SpurGearWheel {
  material_slots: [brass]
  uv_policy: shared
  box wheel_disc { size: [2, 5, 5], material: brass }
  frame hub_mount { on_box: wheel_disc, anchor: [1, 0, 0], rot: [0, 0, 0] }
}

@component WalkerLeg {
  material_slots: [armor, steel, brass]
  uv_policy: shared
  box upper_strut { size: [3, 7, 3], material: armor }
  bone lower_leg attach(parent_box=upper_strut, anchor=[-1, -0.75, 0], align=[1, 0.85, 0]) {
    rest_euler: [32, 0, 0]
    joint_limits: [[-32, 45], [-15, 15], [-15, 15]]
    box lower_strut { size: [2, 7, 3], material: steel }
    box foot_pad place(base_box=lower_strut, anchor=[0, -1, 0.2], align=[0, 1, 0]) {
      size: [4, 2, 5], material: brass
    }
  }
  frame hip_mount { on_box: upper_strut, anchor: [1, 0.75, 0], rot: [18, 0, 0] }
}

@assembly ModularArtilleryWalker {
  root_pivot: [0, 15.164, 0]

  instance chassis : WalkerChassis(w=hull_width, l=hull_length) {
    id: "chassis_01"
    materials: { armor: riveted_hull, firebox: furnace_core }
  }

  subassembly ordnance_turret {
    instance turret_yaw : TurretYawBase() {
      id: "turret_yaw_01"
      materials: { armor: riveted_hull, brass: forged_brass }
    }
    instance gun_cradle : GunCradle() {
      id: "gun_cradle_01"
      materials: { steel: cast_steel }
    }
    instance recoil_barrel : RecoilBarrel(len=barrel_length) {
      id: "barrel_01"
      materials: { brass: forged_brass }
    }

    connect turret_yaw.ring_mount to chassis.turret_ring_socket { type: revolute, axis: y, limits: [-180, 180] }
    connect gun_cradle.trunnion_mount to turret_yaw.trunnion_socket { type: revolute, axis: x, limits: [-35, 15] }
    connect recoil_barrel.breech_slider to gun_cradle.slide_rail { type: prismatic, axis: z, limits: [-2.0, 0.5] }
  }

  subassembly steam_engine {
    instance cyl_l : SteamCylinder() mirror_x(cyl_r) {
      id: "cyl_l_01"
      materials: { steel: cast_steel }
    }
    instance piston_l : SteamPistonRod() mirror_x(piston_r) {
      id: "piston_l_01"
      materials: { brass: forged_brass }
    }
    instance gear_l : SpurGearWheel() mirror_x(gear_r) {
      id: "gear_l_01"
      materials: { brass: forged_brass }
    }

    connect cyl_l.deck_mount to chassis.cyl_socket_l { type: fixed }
    connect piston_l.rod_base to cyl_l.bore_top { type: prismatic, axis: y, limits: [0, 2.5] }
    connect gear_l.hub_mount to chassis.gear_axle_l { type: revolute, axis: x, limits: continuous }
  }

  subassembly walker_gait_rig {
    instance leg_fl : WalkerLeg() mirror_x(leg_fr) {
      id: "leg_fl_01"
      materials: { armor: riveted_hull, steel: cast_steel, brass: forged_brass }
    }
    instance leg_hl : WalkerLeg() mirror_x(leg_hr) {
      id: "leg_hl_01"
      materials: { armor: riveted_hull, steel: cast_steel, brass: forged_brass }
    }
    connect leg_fl.hip_mount to chassis.hip_fl { type: revolute, axis: x, limits: [-55, 55] }
    connect leg_hl.hip_mount to chassis.hip_hl { type: revolute, axis: x, limits: [-55, 55] }
  }
}

@materials {
  riveted_hull: process(family=painted_metal, roughness=0.36, coating=0.5, wear=0.55, weld_group=walker_hull)
  forged_brass: process(family=brushed_metal, roughness=0.24, metalness=0.92, wear=0.22)
  cast_steel: process(family=brushed_metal, roughness=0.38, metalness=0.88, wear=0.35)
  furnace_core: process(family=polymer_dielectric, roughness=0.25, emissive=0.85)
}

@style { target_encoding: vanilla, quantize_ramp: false, bevel_highlight: 0.1 }

@kinematics {
  ik_chain walk_fl { bones: [leg_fl, leg_fl.lower_leg], effector: [0, -7.5, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain walk_fr { bones: [leg_fr, leg_fr.lower_leg], effector: [0, -7.5, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain walk_hl { bones: [leg_hl, leg_hl.lower_leg], effector: [0, -7.5, 0.5], hinge_axis: x, pole: [0, 0, 32] }
  ik_chain walk_hr { bones: [leg_hr, leg_hr.lower_leg], effector: [0, -7.5, 0.5], hinge_axis: x, pole: [0, 0, 32] }
}

@behavior {
  clip artillery_advance {
    duration_s: 2.0
    ik_drive walk_fl: foot_cycle(stride=3.2, lift=1.4, phase_deg=0)
    ik_drive walk_hr: foot_cycle(stride=3.2, lift=1.4, phase_deg=0)
    ik_drive walk_fr: foot_cycle(stride=3.2, lift=1.4, phase_deg=180)
    ik_drive walk_hl: foot_cycle(stride=3.2, lift=1.4, phase_deg=180)
    track gear_l: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track gear_r: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track turret_yaw: wave(axis=y, amp_deg=25, freq_hz=0.5)
    track gun_cradle: hinge(axis=x, amp_deg=-12, freq_hz=0.5)
    slide recoil_barrel: impulse(axis=z, amp_units=-1.5, freq_hz=0.5)
    slide piston_l: hinge(axis=y, amp_units=1.8, freq_hz=1.0, phase_deg=0)
    slide piston_r: hinge(axis=y, amp_units=1.8, freq_hz=1.0, phase_deg=180)
  }
}
""".strip()


def build_modular_sculpted_dragon_v4_world(seed: int = 801) -> WorldState:
    """Compile the Block-DSL v4.0 Modular Sculpted Dragon assembly into a full ``WorldState``."""
    program = parse_program(MODULAR_SCULPTED_DRAGON_V4_DSL)
    mat_prog = MaterialProgram.from_program(program, seed=seed)
    style_cfg = StyleConfig.from_program(program)
    domain = build_surface_domain(program, mat_prog)
    surf_state = initialize_surface_state(domain)
    return WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=surf_state,
        style_config=style_cfg,
    )


def build_modular_artillery_walker_v4_world(seed: int = 802) -> WorldState:
    """Compile the Block-DSL v4.0 Modular Artillery Walker assembly into a full ``WorldState``."""
    program = parse_program(MODULAR_ARTILLERY_WALKER_V4_DSL)
    mat_prog = MaterialProgram.from_program(program, seed=seed)
    style_cfg = StyleConfig.from_program(program)
    domain = build_surface_domain(program, mat_prog)
    surf_state = initialize_surface_state(domain)
    return WorldState(
        program=program,
        material_program=mat_prog,
        surface_state=surf_state,
        style_config=style_cfg,
    )


# ---------------------------------------------------------------------------
# 2. Honest Inferred-Subtree Decompiler for Flat Imported Human Rigs
# ---------------------------------------------------------------------------


def infer_assembly_from_program(
    program: Program,
    *,
    assembly_name: str = "InferredAssetAssembly",
) -> tuple[AssemblySpec, SourceMap]:
    """Recover candidate submodel components from a flat ``@rig`` ``Program`` with honest provenance.

    Imported Bedrock / Blockbench files contain evaluated bones and cubes, not the human
    author's original parametric design DAG. Per ADR 0008, recovered structures are ALWAYS
    marked ``assembly_provenance="inferred_subtree"`` so training/evaluation never conflates
    inferred part groupings with human-authored parametric assemblies.
    """
    scene = compile_geometry(program)
    comps: list[ComponentDef] = []
    insts: list[ComponentInstanceDecl] = []
    conns: list[ConnectionDecl] = []
    entries: list[SourceMapEntry] = []

    boxes_by_bone: dict[str, list[Any]] = {b.name: [] for b in scene.bones}
    locs_by_bone: dict[str, list[Any]] = {b.name: [] for b in scene.bones}
    for bx in scene.boxes:
        boxes_by_bone[bx.bone_name].append(bx)
    for lc in scene.locators:
        if lc.bone_name in locs_by_bone:
            locs_by_bone[lc.bone_name].append(lc)

    for b in scene.bones:
        comp_name = f"Inferred_{b.name}"
        b_boxes = boxes_by_bone[b.name]
        used_mats = sorted({bx.material for bx in b_boxes})
        mat_slots = tuple((f"slot_{i}", m) for i, m in enumerate(used_mats))
        mat_to_slot = {m: f"slot_{i}" for i, m in enumerate(used_mats)}

        tmpl_boxes: list[ComponentBoxTemplate] = []
        for idx_x, bx in enumerate(b_boxes):
            eul = tuple(str(round(float(v), 4)) for v in bx.local_euler)
            ctr = tuple(str(round(float(v), 4)) for v in bx.local_center)
            sz_s = tuple(str(int(round(float(v))) if abs(float(v) - round(float(v))) <= 1e-9 else round(float(v), 4)) for v in bx.size)
            tmpl_boxes.append(
                ComponentBoxTemplate(
                    name=bx.name,
                    is_plane=bx.plane_normal is not None,
                    size_exprs=sz_s,
                    material_slot=mat_to_slot[bx.material],
                    has_place=idx_x > 0,
                    place_anchor=("0", "0", "0"),
                    place_align=("0", "0", "0"),
                    place_offset=ctr,
                    inflate_expr=str(round(float(bx.inflate), 6)),
                    rotate_exprs=eul,  # type: ignore[arg-type]
                    cullface=bx.cullface,
                    plane_normal=bx.plane_normal,
                )
            )
        frames = (AttachmentFrameDecl(name="pivot_frame", pos=("0", "0", "0"), rot=("0", "0", "0")),)
        comps.append(
            ComponentDef(
                name=comp_name,
                material_slots=mat_slots,
                root_boxes=tuple(tmpl_boxes),
                frames=frames,
            )
        )
        insts.append(
            ComponentInstanceDecl(
                name=b.name,
                component_name=comp_name,
                stable_id=f"inferred_{b.name}",
                material_bindings=tuple((f"slot_{i}", m) for i, m in enumerate(used_mats)),
            )
        )
        if b.parent_name is not None:
            conns.append(
                ConnectionDecl(
                    child_instance=b.name,
                    child_frame="pivot_frame",
                    parent_instance=b.parent_name,
                    parent_frame="pivot_frame",
                    conn_type="revolute",
                    axis="x",
                )
            )
        entries.append(
            SourceMapEntry(
                stable_id=f"inferred_{b.name}",
                instance_name=b.name,
                readable_path=f"{assembly_name}.{b.name}",
                component_name=comp_name,
                mirrored=b.mirrored,
                bound_params=(),
                bound_materials=tuple((f"slot_{i}", m) for i, m in enumerate(used_mats)),
                uv_policy="unique",
                runtime_bone_name=b.name,
                exported_boxes=tuple(bx.name for bx in b_boxes),
                exported_locators=tuple(lc.name for lc in locs_by_bone[b.name]),
                exported_sub_bones=(),
            )
        )

    root_piv_s = tuple(str(round(float(v), 4)) for v in scene.bones[0].pivot_world)
    asm_decl = AssemblyDecl(
        name=assembly_name,
        root_pivot=root_piv_s,  # type: ignore[arg-type]
        root_instance=scene.bones[0].name,
        instances=tuple(insts),
        connections=tuple(conns),
    )
    spec = AssemblySpec(
        layout=None,
        components=tuple(comps),
        assembly=asm_decl,
        assembly_provenance="inferred_subtree",
    )
    smap = SourceMap(
        assembly_name=assembly_name,
        assembly_provenance="inferred_subtree",
        layout_values=(),
        entries=tuple(entries),
        dotted_bone_map=tuple((b.name, b.name) for b in scene.bones),
        dotted_box_map=tuple((bx.name, bx.name) for bx in scene.boxes),
    )
    return spec, smap


# ---------------------------------------------------------------------------
# 3. Four Modular World-Model Supervision & Evaluation Tasks (`LFM3-VL-2B`)
# ---------------------------------------------------------------------------


def _serialize_single_component_dsl(comp: ComponentDef) -> str:
    tmp_spec = AssemblySpec(layout=None, components=(comp,), assembly=None)
    return "\n".join(format_assembly_sections(tmp_spec))


def measure_non_target_vertex_drift(
    before_program: Program,
    after_program: Program,
    target_component_name: str,
) -> dict[str, Any]:
    """Measure exact 8-corner world vertex invariance across all non-target component instances."""
    if before_program.source_map is None or after_program.source_map is None:
        raise ValueError("Both programs must carry a v4.0 SourceMap")

    sc_before = compile_geometry(before_program)
    sc_after = compile_geometry(after_program)

    non_target_boxes: list[str] = []
    target_boxes_before: list[str] = []
    for entry in before_program.source_map.entries:
        if entry.component_name == target_component_name:
            target_boxes_before.extend(entry.exported_boxes)
        else:
            non_target_boxes.extend(entry.exported_boxes)

    max_drift = 0.0
    missing_boxes: list[str] = []
    for bname in non_target_boxes:
        if bname not in sc_before.box_by_name or bname not in sc_after.box_by_name:
            missing_boxes.append(bname)
            continue
        c0 = sc_before.box_by_name[bname].corners(include_inflate=True)
        c1 = sc_after.box_by_name[bname].corners(include_inflate=True)
        drift = float(np.max(np.abs(c1 - c0)))
        max_drift = max(max_drift, drift)

    return {
        "target_component": target_component_name,
        "non_target_box_count": len(non_target_boxes),
        "missing_non_target_boxes": missing_boxes,
        "max_non_target_vertex_drift": round(max_drift, 12),
        "edit_locality_exact": len(missing_boxes) == 0 and max_drift <= 1e-9,
    }


def build_modular_assembly_supervision_suite() -> list[dict[str, Any]]:
    """Construct verified prompt + oracle target records for all 4 v4.0 World-Model tasks."""
    dragon_prog = parse_program(MODULAR_SCULPTED_DRAGON_V4_DSL)
    walker_prog = parse_program(MODULAR_ARTILLERY_WALKER_V4_DSL)
    dragon_spec: AssemblySpec = dragon_prog.assembly_spec
    walker_spec: AssemblySpec = walker_prog.assembly_spec

    records: list[dict[str, Any]] = []

    # Task 1: Top-Down Layout & Interface Contract Planning
    dragon_layout_only = "\n".join(
        format_assembly_sections(AssemblySpec(layout=dragon_spec.layout, components=(), assembly=None))
    )
    records.append({
        "task_id": "v4_wm_01_layout_plan_dragon",
        "task_type": "plan_layout_and_interfaces",
        "assembly_provenance": "authored",
        "prompt": (
            "Design a top-down @layout block for a 4-legged winged sculpted dragon assembly "
            "with driving parameters body_length, chest_width, and wing_span_seg, derived head_width "
            "and chest_len, max_boxes <= 48, max_bones <= 28, and min_ground_y >= -0.5."
        ),
        "oracle_completion": dragon_layout_only,
        "required_params": ["body_length", "chest_width", "wing_span_seg"],
    })

    # Task 2: Scoped Single-@component Synthesis Against Interface Contract
    skull_comp = next(c for c in dragon_spec.components if c.name == "DragonSkull")
    skull_dsl = _serialize_single_component_dsl(skull_comp)
    records.append({
        "task_id": "v4_wm_02_component_synth_dragon_skull",
        "task_type": "synthesize_component_from_contract",
        "assembly_provenance": "authored",
        "parent_assembly_dsl": MODULAR_SCULPTED_DRAGON_V4_DSL,
        "target_component_name": "DragonSkull",
        "required_frames": [f.name for f in skull_comp.frames],
        "required_material_slots": [s for s, _ in skull_comp.material_slots],
        "prompt": (
            "Synthesize ONLY `@component DragonSkull` in its local namespace exposing parameter `w`, "
            "material_slots `[hide, organ]`, and public attachment frames `neck_mount`, "
            "`horn_socket_l`, `horn_socket_r`, `jaw_hinge`, and `flame_emitter` so it mounts cleanly "
            "into `ModularElderDragon` with 0 SAT collisions."
        ),
        "oracle_completion": skull_dsl,
    })

    # Task 3: Surgical Component Swap with Guaranteed Non-Target Edit Locality
    armored_barrel_dsl = """
@component RecoilBarrel {
  param len = 9 [5, 16]
  material_slots: [brass]
  box ordnance_tube { size: [2, 2, len], material: brass }
  box muzzle_brake place(base_box=ordnance_tube, anchor=[0, 0, 1], align=[0, 0, -1]) {
    size: [3, 3, 2], material: brass
  }
  frame breech_slider { on_box: ordnance_tube, anchor: [0, 0, -1], rot: [0, 0, 0] }
  frame muzzle_flash { on_box: muzzle_brake, anchor: [0, 0, 1], rot: [0, 0, 0], role: vfx_emitter }
}
""".strip()
    records.append({
        "task_id": "v4_wm_03_surgical_swap_walker_barrel",
        "task_type": "surgical_component_swap_and_repair",
        "assembly_provenance": "authored",
        "base_assembly_dsl": MODULAR_ARTILLERY_WALKER_V4_DSL,
        "target_component_name": "RecoilBarrel",
        "prompt": (
            "Upgrade `@component RecoilBarrel` in `ModularArtilleryWalker` by adding a flush `[3, 3, 2]` "
            "`muzzle_brake` at the front of `ordnance_tube`, advancing `muzzle_flash` to the front of "
            "`muzzle_brake`, and preserving `breech_slider` and all other walker components unchanged."
        ),
        "oracle_completion": armored_barrel_dsl,
    })

    # Task 4: Top-Down @layout Parameter Propagation Prediction
    override_dict = {"chest_width": 14}
    perturbed_dragon = override_layout_params(dragon_prog, override_dict)
    sc_pert = compile_geometry(perturbed_dragon)
    aud_pert = audit_assembly_contracts(perturbed_dragon, sc_pert)
    sat_pert = audit_scene(sc_pert, enforce_physical_contact=True)
    gt_payload = {
        "torso_ribcage_width": float(sc_pert.box_by_name["torso__ribcage"].size[0]),
        "skull_cranium_width": float(sc_pert.box_by_name["skull__cranium"].size[0]),
        "audit_passed": bool(sat_pert["passed"] and aud_pert["passed"]),
        "world_span_x": round(float(aud_pert["world_span"][0]), 4),
    }
    records.append({
        "task_id": "v4_wm_04_propagate_dragon_chest_width",
        "task_type": "propagate_layout_and_instance_controls",
        "assembly_provenance": "authored",
        "base_assembly_dsl": MODULAR_SCULPTED_DRAGON_V4_DSL,
        "layout_overrides": override_dict,
        "prompt": (
            "Predict JSON `{torso_ribcage_width, skull_cranium_width, audit_passed, world_span_x}` "
            "when `@layout` parameter `chest_width` in `ModularElderDragon` increases from `10` to `14`."
        ),
        "ground_truth": gt_payload,
        "oracle_completion": json.dumps(gt_payload, sort_keys=True),
    })

    return records


def _parse_isolated_component(comp_dsl_text: str) -> ComponentDef:
    """Parse a standalone `@component <Name> { ... }` snippet using the standard zero-eval parser."""
    from synccraft.dsl.parser import _Parser

    p = _Parser(comp_dsl_text)
    p.take("@")
    p.take("component")
    return p._parse_component_section()


def evaluate_modular_assembly_world_tasks(
    records: Sequence[Mapping[str, Any]],
    completions_by_task_id: Mapping[str, str],
) -> dict[str, Any]:
    """Deterministically score completions across all 4 v4.0 Modular Assembly World-Model tasks."""
    per_task: list[dict[str, Any]] = []
    passed_count = 0

    for rec in records:
        tid = str(rec["task_id"])
        ttype = str(rec["task_type"])
        raw_out = completions_by_task_id.get(tid, "").strip()
        if not raw_out:
            per_task.append({"task_id": tid, "task_type": ttype, "passed": False, "reason": "missing_completion"})
            continue

        try:
            if ttype == "plan_layout_and_interfaces":
                from synccraft.dsl.parser import _Parser

                p = _Parser(raw_out)
                p.take("@")
                p.take("layout")
                lay = p._parse_layout_section()
                vals = evaluate_layout_dag(lay)
                req = set(rec.get("required_params", ()))
                ok = req.issubset(set(vals.keys())) and lay.budget.max_boxes is not None
                per_task.append({"task_id": tid, "task_type": ttype, "passed": ok, "evaluated_layout": vals})
                if ok:
                    passed_count += 1

            elif ttype == "synthesize_component_from_contract":
                comp = _parse_isolated_component(raw_out)
                parent_prog = parse_program(str(rec["parent_assembly_dsl"]))
                swapped = swap_component_definition(parent_prog, comp)
                sc = compile_geometry(swapped)
                aud = audit_scene(sc, enforce_physical_contact=True)
                caud = audit_assembly_contracts(swapped, sc)
                frame_names = {f.name for f in comp.frames}
                frames_ok = set(rec.get("required_frames", ())).issubset(frame_names)
                ok = bool(aud["passed"] and caud["passed"] and frames_ok)
                per_task.append({
                    "task_id": tid,
                    "task_type": ttype,
                    "passed": ok,
                    "audit_passed": bool(aud["passed"]),
                    "contracts_passed": bool(caud["passed"]),
                    "frames_ok": frames_ok,
                })
                if ok:
                    passed_count += 1

            elif ttype == "surgical_component_swap_and_repair":
                comp = _parse_isolated_component(raw_out)
                base_prog = parse_program(str(rec["base_assembly_dsl"]))
                swapped = swap_component_definition(base_prog, comp)
                sc = compile_geometry(swapped)
                aud = audit_scene(sc, enforce_physical_contact=True)
                drift = measure_non_target_vertex_drift(base_prog, swapped, str(rec["target_component_name"]))
                clips_ok = True
                if swapped.behavior and swapped.behavior.clips:
                    for cl in swapped.behavior.clips:
                        if not audit_behavior_clip(swapped, cl.name)["passed"]:
                            clips_ok = False
                ok = bool(aud["passed"] and clips_ok and drift["edit_locality_exact"])
                per_task.append({
                    "task_id": tid,
                    "task_type": ttype,
                    "passed": ok,
                    "audit_passed": bool(aud["passed"]),
                    "clips_passed": clips_ok,
                    "max_non_target_vertex_drift": drift["max_non_target_vertex_drift"],
                    "edit_locality_exact": drift["edit_locality_exact"],
                })
                if ok:
                    passed_count += 1

            elif ttype == "propagate_layout_and_instance_controls":
                pred = json.loads(raw_out)
                gt = dict(rec["ground_truth"])
                w_err = abs(float(pred["torso_ribcage_width"]) - float(gt["torso_ribcage_width"]))
                s_err = abs(float(pred["skull_cranium_width"]) - float(gt["skull_cranium_width"]))
                span_err = abs(float(pred["world_span_x"]) - float(gt["world_span_x"]))
                aud_match = bool(pred["audit_passed"]) == bool(gt["audit_passed"])
                ok = w_err <= 1e-3 and s_err <= 1e-3 and span_err <= 0.25 and aud_match
                per_task.append({
                    "task_id": tid,
                    "task_type": ttype,
                    "passed": ok,
                    "span_mae": round(span_err, 4),
                    "audit_match": aud_match,
                })
                if ok:
                    passed_count += 1
            else:
                per_task.append({"task_id": tid, "task_type": ttype, "passed": False, "reason": f"unknown_task:{ttype}"})
        except Exception as exc:
            per_task.append({
                "task_id": tid,
                "task_type": ttype,
                "passed": False,
                "reason": f"{type(exc).__name__}:{exc}",
            })

    total = len(records)
    return {
        "total_tasks": total,
        "passed_tasks": passed_count,
        "pass_rate": round(passed_count / max(1, total), 4),
        "results": per_task,
    }
