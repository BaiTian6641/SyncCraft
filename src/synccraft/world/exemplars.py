"""Flagship stateful world exemplars for P1-03d/e causal verification."""

from __future__ import annotations

from dataclasses import replace

from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Place, Program, validate_program
from synccraft.dsl.parser import parse_program
from synccraft.surface.contracts import (
    EnvironmentState,
    MaterialProgram,
    MaterialSpec,
    StampOverride,
    StyleConfig,
    WorldState,
)
from synccraft.surface.domain import build_surface_domain, initialize_surface_state
from synccraft.surface.processes import integrate_gray_scott_on_state


def build_articulated_mechanism_world(
    *,
    seed: int = 101,
    subdivisions_per_unit: int = 4,
) -> WorldState:
    """Build Flagship Exemplar 1: Articulated painted-metal + wood + stone mechanism."""
    program = validate_program(
        Program(
            meta=Meta("block_multipart", "bbmodel_5", 16),
            palettes=[
                Palette("stone_mat", (0.52, 0.03, 75.0), 6),
                Palette("wood_mat", (0.56, 0.09, 62.0), 6),
                Palette("painted_iron", (0.48, 0.14, 245.0), 6),
                Palette("brushed_steel", (0.76, 0.02, 235.0), 6),
            ],
            root=Bone(
                name="base_frame",
                pivot_local=(0.0, 2.0, 0.0),
                items=[
                    Box("stone_pedestal", (10, 4, 10), "stone_mat"),
                    Box(
                        "timber_post",
                        (4, 8, 4),
                        "wood_mat",
                        place=Place("stone_pedestal", anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0)),
                    ),
                    Box(
                        "housing",
                        (6, 4, 6),
                        "painted_iron",
                        place=Place("timber_post", anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0)),
                    ),
                    Bone(
                        name="upper_arm",
                        attach=Attach("housing", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)),
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("upper_link", (6, 2, 2), "painted_iron"),
                            Bone(
                                name="fore_arm",
                                attach=Attach("upper_link", anchor=(1.0, 0.0, 0.0), align=(-1.0, 0.0, 0.0)),
                                rest_euler=(0.0, 0.0, 0.0),
                                items=[
                                    Box("fore_link", (5, 2, 2), "brushed_steel"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    )

    material_program = MaterialProgram(
        materials={
            "stone_mat": MaterialSpec(
                name="stone_mat",
                family="weathered_stone",
                substrate_lch=(0.54, 0.03, 78.0),
                secondary_lch=(0.40, 0.04, 68.0),
                perceptual_roughness=0.82,
                porosity=0.52,
                wear_susceptibility=0.08,
                biofilm_affinity=0.35,
                seed=seed,
            ),
            "wood_mat": MaterialSpec(
                name="wood_mat",
                family="solid_wood",
                substrate_lch=(0.62, 0.09, 66.0),
                secondary_lch=(0.40, 0.08, 52.0),
                grain_axis=(0.0, 1.0, 0.0),
                perceptual_roughness=0.64,
                porosity=0.42,
                wear_susceptibility=0.15,
                seed=seed + 11,
            ),
            "painted_iron": MaterialSpec(
                name="painted_iron",
                family="painted_metal",
                substrate_lch=(0.74, 0.02, 240.0),
                coating_lch=(0.48, 0.15, 248.0),
                oxide_lch=(0.44, 0.13, 40.0),
                coating_thickness=0.50,
                perceptual_roughness=0.32,
                substrate_roughness=0.22,
                metalness=0.0,
                substrate_metalness=1.0,
                wear_susceptibility=0.85,
                oxidation_rate=0.55,
                labpbr_metal_code=230,  # Iron
                seed=seed + 23,
            ),
            "brushed_steel": MaterialSpec(
                name="brushed_steel",
                family="brushed_metal",
                substrate_lch=(0.78, 0.02, 235.0),
                oxide_lch=(0.46, 0.10, 44.0),
                coating_thickness=0.0,
                perceptual_roughness=0.28,
                substrate_roughness=0.26,
                metalness=1.0,
                substrate_metalness=1.0,
                wear_susceptibility=0.25,
                oxidation_rate=0.12,
                labpbr_metal_code=230,
                seed=seed + 37,
            ),
        },
        stamps=(
            StampOverride(
                box_name="housing",
                face="south",
                u_min=1.0,
                v_min=1.25,
                u_max=5.0,
                v_max=2.75,
                lch=(0.82, 0.16, 86.0),
                shape="stripe_u",
                perceptual_roughness=0.38,
                metalness=0.0,
            ),
        ),
    )

    domain = build_surface_domain(program, material_program, subdivisions_per_unit=subdivisions_per_unit)
    surface_state = initialize_surface_state(domain)
    return WorldState(
        program=program,
        material_program=material_program,
        surface_state=surface_state,
        style_config=StyleConfig(quantize_to_palette_ramp=False, target_encoding="vanilla"),
        environment=EnvironmentState(),
    )


def build_biological_creature_world(
    *,
    seed: int = 202,
    subdivisions_per_unit: int = 8,
) -> WorldState:
    """Build Flagship Exemplar 2: Patterned RD + scaled + iridescent creature with mirrored limbs."""
    program = validate_program(
        Program(
            meta=Meta("entity", "bedrock_entity", 16),
            palettes=[
                Palette("pelt_skin", (0.68, 0.11, 72.0), 6),
                Palette("limb_scales", (0.44, 0.08, 135.0), 6),
                Palette("elytra_crest", (0.36, 0.06, 210.0), 6),
            ],
            root=Bone(
                name="body_root",
                pivot_local=(0.0, 12.0, 0.0),
                items=[
                    Box("torso", (6, 6, 10), "pelt_skin"),
                    Bone(
                        name="neck_head",
                        attach=Attach("torso", anchor=(0.0, 0.25, -1.0), align=(0.0, -0.25, 1.0)),
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("skull", (4, 4, 4), "pelt_skin"),
                        ],
                    ),
                    Bone(
                        name="left_thigh",
                        attach=Attach("torso", anchor=(-1.0, -0.5, -0.5), align=(1.0, 1.0, 0.0)),
                        mirror_name="right_thigh",
                        rest_euler=(0.0, 0.0, 0.0),
                        items=[
                            Box("thigh_seg", (2, 5, 2), "pelt_skin"),
                            Bone(
                                name="left_shin",
                                attach=Attach("thigh_seg", anchor=(0.0, -1.0, 0.0), align=(0.0, 1.0, 0.0)),
                                rest_euler=(0.0, 0.0, 0.0),
                                items=[
                                    Box("shin_seg", (2, 5, 2), "limb_scales"),
                                ],
                            ),
                        ],
                    ),
                    Bone(
                        name="left_crest",
                        attach=Attach("torso", anchor=(-0.6, 1.0, 0.0), align=(1.0, -1.0, 0.0)),
                        mirror_name="right_crest",
                        rest_euler=(0.0, 0.0, 15.0),
                        items=[
                            Box("crest_plate", (5, 1, 6), "elytra_crest"),
                        ],
                    ),
                ],
            ),
        )
    )

    skin_spec = MaterialSpec(
        name="pelt_skin",
        family="patterned_skin",
        substrate_lch=(0.74, 0.10, 76.0),
        secondary_lch=(0.26, 0.06, 48.0),
        perceptual_roughness=0.62,
        subsurface=0.35,
        porosity=0.22,
        anisotropy=1.35,
        rd_du=0.075,
        rd_dv=0.036,
        rd_feed=0.037,
        rd_kill=0.061,
        rd_initial_time=24.0,
        countershade_strength=0.24,
        extremity_darkening=0.14,
        symmetry_mode="independent_seed",
        seed=seed,
    )

    material_program = MaterialProgram(
        materials={
            "pelt_skin": skin_spec,
            "limb_scales": MaterialSpec(
                name="limb_scales",
                family="keratin_scales",
                substrate_lch=(0.48, 0.09, 138.0),
                secondary_lch=(0.28, 0.05, 125.0),
                perceptual_roughness=0.38,
                subsurface=0.18,
                feature_scale=1.2,
                countershade_strength=0.18,
                seed=seed + 17,
            ),
            "elytra_crest": MaterialSpec(
                name="elytra_crest",
                family="thin_film_iridescence",
                substrate_lch=(0.22, 0.04, 220.0),
                perceptual_roughness=0.14,
                metalness=0.15,
                film_thickness_nm=395.0,
                film_ior=1.52,
                wear_susceptibility=0.45,
                seed=seed + 31,
            ),
        },
        # Asymmetric left/right eye markings on skull north (-Z front) face
        stamps=(
            StampOverride(
                box_name="skull",
                face="north",
                u_min=0.5,
                v_min=1.2,
                u_max=1.5,
                v_max=2.2,
                lch=(0.82, 0.17, 82.0),  # Amber eye on one side
                shape="disc",
                emissive=0.65,
            ),
            StampOverride(
                box_name="skull",
                face="north",
                u_min=2.5,
                v_min=1.2,
                u_max=3.5,
                v_max=2.2,
                lch=(0.76, 0.14, 205.0),  # Asymmetric cyan ocular marking on opposite side
                shape="cross",
                emissive=0.85,
            ),
        ),
    )

    domain = build_surface_domain(program, material_program, subdivisions_per_unit=subdivisions_per_unit)
    surface_state = initialize_surface_state(domain)
    skin_faces = {fk for fk, g in domain.faces.items() if g.material_name == "pelt_skin"}
    integrate_gray_scott_on_state(domain, surface_state, skin_spec, skin_faces, duration=0.0)

    return WorldState(
        program=program,
        material_program=material_program,
        surface_state=surface_state,
        style_config=StyleConfig(quantize_to_palette_ramp=False, target_encoding="vanilla"),
        environment=EnvironmentState(),
    )


SCULPTED_ELDER_WYVERN_DSL: str = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: sculpted }

@palette {
  carapace_chitin: ramp(base_lch=[0.32, 0.08, 28], steps=8, hue_shift=[-6, 6])
  wyvern_hide: ramp(base_lch=[0.52, 0.09, 54], steps=8, hue_shift=[-4, 5])
  horn_keratin: ramp(base_lch=[0.78, 0.05, 82], steps=6)
  wing_patagium: ramp(base_lch=[0.46, 0.14, 215], steps=8)
  ember_organ: ramp(base_lch=[0.72, 0.20, 42], steps=6, emissive=true)
}

@rig {
  root bone pelvis_root {
    pivot_local: [0, 18, 0]
    box pelvis_core { size: [8, 7, 8], material: wyvern_hide }
    box sacral_plate place(base_box=pelvis_core, anchor=[0, 1, 0], align=[0, -0.4, 0]) {
      size: [6, 3, 7], material: carapace_chitin, rotate: [-18, 0, 0]
    }

    bone thorax attach(parent_box=pelvis_core, anchor=[0, 0.15, 1], align=[0, 0, -1]) {
      joint_limits: [[-20, 20], [-18, 18], [-14, 14]]
      box ribcage_core { size: [10, 8, 12], material: wyvern_hide }
      box sternum_keel place(base_box=ribcage_core, anchor=[0, -1, 0.1], align=[0, 0.45, 0]) {
        size: [6, 3, 10], material: carapace_chitin, rotate: [-16, 0, 0]
      }
      box dorsal_plate_fore place(base_box=ribcage_core, anchor=[0, 1, 0.35], align=[0, -0.35, 0]) {
        size: [6, 3, 6], material: carapace_chitin, rotate: [-24, 0, 0]
      }
      box dorsal_plate_aft place(base_box=ribcage_core, anchor=[0, 1, -0.35], align=[0, -0.35, 0]) {
        size: [6, 3, 6], material: carapace_chitin, rotate: [-24, 0, 0], blend_with: [dorsal_plate_fore]
      }
      box throat_furnace place(base_box=ribcage_core, anchor=[0, -0.2, 1], align=[0, 0, -0.4]) {
        size: [4, 4, 3], material: ember_organ, rotate: [18, 0, 0]
      }
      locator saddle_mount { offset: [0, 6, 4], role: rider_seat }

      bone wing_humerus_l attach(parent_box=ribcage_core, anchor=[-0.9, 1.0, -0.15], align=[1, -0.5, 0]) mirror_x(wing_humerus_r) {
        rest_euler: [0, 10, -24]
        joint_limits: [[-28, 28], [-32, 32], [-60, 30]]
        box humerus_spar_l { size: [8, 2, 3], material: carapace_chitin }
        plane propatagium_l place(base_box=humerus_spar_l, anchor=[0, 0, -1], align=[0, 0, 1]) {
          size: [7, 7], normal: y, material: wing_patagium
        }

        bone wing_ulna_l attach(parent_box=humerus_spar_l, anchor=[-1, 0, 0], align=[1, 0, 0]) {
          rest_euler: [0, -14, -8]
          joint_limits: [[-24, 24], [-36, 36], [-65, 48]]
          box ulna_spar_l { size: [10, 2, 2], material: carapace_chitin }
          box thumb_claw_l place(base_box=ulna_spar_l, anchor=[-0.9, 0.3, 1], align=[0.5, 0, -0.7]) {
            size: [1, 1, 3], material: horn_keratin, rotate: [10, 20, 0]
          }
          plane plagiopatagium_l place(base_box=ulna_spar_l, anchor=[0, 0, -1], align=[0, 0, 1]) {
            size: [9, 9], normal: y, material: wing_patagium
          }

          bone wing_digit_outer_l attach(parent_box=ulna_spar_l, anchor=[-1, 0, 0.2], align=[1, 0, 0]) {
            rest_euler: [0, -14, -8]
            joint_limits: [[-20, 20], [-30, 30], [-45, 45]]
            box phalanx_outer_l { size: [10, 1, 2], material: horn_keratin }
            plane dactylopatagium_l place(base_box=phalanx_outer_l, anchor=[0, 0, -1], align=[0, 0, 1]) {
              size: [9, 8], normal: y, material: wing_patagium
            }
          }
        }
      }

      bone fore_humerus_l attach(parent_box=ribcage_core, anchor=[-1, -0.4, 0.65], align=[1, 0.8, 0]) mirror_x(fore_humerus_r) {
        rest_euler: [10, 0, 0]
        joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
        box fore_upper_l { size: [3, 7, 4], material: wyvern_hide }
        box shoulder_shell_l place(base_box=fore_upper_l, anchor=[-1, 0.25, 0], align=[0.4, 0, 0]) {
          size: [2, 4, 4], material: carapace_chitin, rotate: [6, 0, 20]
        }
        box shoulder_spur_l place(base_box=shoulder_shell_l, anchor=[-1, 0.4, -0.2], align=[0.4, -0.3, 0]) {
          size: [2, 2, 3], material: horn_keratin, rotate: [-14, -10, 18]
        }

        bone fore_radius_l attach(parent_box=fore_upper_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
          rest_euler: [-10, 0, 0]
          joint_limits: [[-65, 45], [-15, 15], [-15, 15]]
          box fore_lower_l { size: [3, 8, 3], material: carapace_chitin }
          box fore_talon_pad_l place(base_box=fore_lower_l, anchor=[0, -1, 0.3], align=[0, 1, -0.2]) {
            size: [4, 2, 5], material: horn_keratin
          }
          locator foot_fl_tip { offset: [0, -10, 1], role: ik_effector }
        }
      }

      bone neck_1 attach(parent_box=ribcage_core, anchor=[0, 0.45, 1], align=[0, 0, -0.8]) {
        rest_euler: [-22, 0, 0]
        joint_limits: [[-30, 30], [-30, 30], [-18, 18]]
        box cerv_1 { size: [6, 5, 6], material: wyvern_hide }
        box cerv_1_scute place(base_box=cerv_1, anchor=[0, 1, 0], align=[0, -0.4, 0]) {
          size: [4, 2, 5], material: carapace_chitin, rotate: [-22, 0, 0]
        }

        bone neck_2 attach(parent_box=cerv_1, anchor=[0, 0.15, 1], align=[0, 0, -0.8]) {
          rest_euler: [12, 0, 0]
          joint_limits: [[-30, 30], [-32, 32], [-18, 18]]
          box cerv_2 { size: [5, 4, 6], material: wyvern_hide }
          box cerv_2_scute place(base_box=cerv_2, anchor=[0, 1, 0], align=[0, -0.4, 0]) {
            size: [3, 2, 5], material: carapace_chitin, rotate: [-22, 0, 0]
          }

          bone skull attach(parent_box=cerv_2, anchor=[0, 0.1, 1], align=[0, 0, -0.8]) {
            rest_euler: [14, 0, 0]
            joint_limits: [[-32, 35], [-42, 42], [-20, 20]]
            box cranium { size: [6, 5, 6], material: carapace_chitin }
            box snout_bridge place(base_box=cranium, anchor=[0, 0.35, 1], align=[0, 0.2, -0.6]) {
              size: [4, 3, 6], material: carapace_chitin, rotate: [20, 0, 0]
            }
            box snout_beak place(base_box=snout_bridge, anchor=[0, -0.2, 1], align=[0, 0.4, -0.5]) {
              size: [2, 3, 3], material: horn_keratin, rotate: [28, 0, 0]
            }
            locator brow_focus { offset: [0, 2, 5], role: camera_focus }
            locator flame_throat_vent { offset: [0, -1, 9], role: vfx_emitter }

            bone cheek_flange_l attach(parent_box=cranium, anchor=[-1, 0.05, 0.5], align=[0.5, 0, -0.4]) mirror_x(cheek_flange_r) {
              rest_euler: [6, 22, -16]
              joint_limits: [[-10, 10], [-12, 12], [-10, 10]]
              box zygoma_l { size: [2, 3, 5], material: carapace_chitin, blend_with: [snout_bridge] }
            }

            bone crown_horn_l attach(parent_box=cranium, anchor=[-0.85, 1.0, 0.1], align=[0.3, -1.0, 0]) mirror_x(crown_horn_r) {
              rest_euler: [-68, 16, -24]
              joint_limits: [[-10, 10], [-10, 10], [-10, 10]]
              box horn_base_l { size: [2, 6, 3], material: horn_keratin }

              bone crown_horn_tip_l attach(parent_box=horn_base_l, anchor=[0, 1, 0], align=[0, -0.9, 0]) {
                rest_euler: [-16, 6, -8]
                joint_limits: [[-8, 8], [-8, 8], [-8, 8]]
                box horn_spire_l { size: [1, 6, 2], material: horn_keratin }
              }
            }

            bone mandible attach(parent_box=cranium, anchor=[0, -1.0, 0.35], align=[0, 1.0, -0.85]) {
              rest_euler: [6, 0, 0]
              joint_limits: [[-6, 42], [-8, 8], [-6, 6]]
              box jaw_ramus { size: [4, 2, 6], material: wyvern_hide }
              box chin_keel place(base_box=jaw_ramus, anchor=[0, -1, 0.4], align=[0, 0.4, 0]) {
                size: [2, 2, 4], material: horn_keratin, rotate: [-20, 0, 0]
              }
            }
          }
        }
      }
    }

    bone hind_femur_l attach(parent_box=pelvis_core, anchor=[-1, -0.3, 0.3], align=[1, 0.8, 0]) mirror_x(hind_femur_r) {
      rest_euler: [14, 0, 0]
      joint_limits: [[-58, 58], [-20, 20], [-20, 20]]
      box thigh_mass_l { size: [4, 8, 4], material: wyvern_hide }
      box thigh_armor_l place(base_box=thigh_mass_l, anchor=[-1, 0.2, 0], align=[0.4, 0, 0]) {
        size: [2, 6, 4], material: carapace_chitin, rotate: [0, 0, 16]
      }

      bone hind_crus_l attach(parent_box=thigh_mass_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [-14, 0, 0]
        joint_limits: [[-68, 48], [-15, 15], [-15, 15]]
        box crus_shank_l { size: [3, 8, 4], material: carapace_chitin }
        box hind_talon_pad_l place(base_box=crus_shank_l, anchor=[0, -1, 0.3], align=[0, 1, -0.2]) {
          size: [4, 2, 6], material: horn_keratin
        }
        locator foot_hl_tip { offset: [0, -10, 1], role: ik_effector }
      }
    }

    chain tail_seg attach(parent_box=pelvis_core, anchor=[0, 0.25, -1], align=[0, 0, 1]) segments=4 taper=0.84 {
      axis: -z,
      joint_limits: [[-24, 24], [-30, 30], [-16, 16]]
      box caudal_core { size: [5, 4, 7], material: wyvern_hide }
      box caudal_scute place(base_box=caudal_core, anchor=[0, 1, 0], align=[0, -0.4, 0]) {
        size: [3, 2, 5], material: carapace_chitin, rotate: [-22, 0, 0]
      }
    }
  }
}

@materials {
  carapace_chitin: process(family=keratin_scales, roughness=0.34, coating=0.42, wear=0.28, weld_group=dorsal_carapace)
  wyvern_hide: process(family=patterned_skin, roughness=0.58, subsurface=0.32, rd_feed=0.037, rd_kill=0.061, weld_group=body_hide)
  horn_keratin: process(family=weathered_stone, roughness=0.48, porosity=0.20, wear=0.38)
  wing_patagium: process(family=thin_film_iridescence, roughness=0.18, ior=1.54, film_nm=410, subsurface=0.38)
  ember_organ: process(family=polymer_dielectric, roughness=0.28, emissive=0.75)
}

@style { target_encoding: vanilla, quantize_ramp: false, bevel_highlight: 0.08 }

@muscles {
  muscle pectoralis_l {
    origin_bone: thorax, insertion_bone: wing_humerus_l, action_axis: z, moment_arm_deg: 22,
    tendon_coupling: [wing_ulna_l, z, 0.28], belly_box: sternum_keel, bulge_inflate: 0.14
  }
  muscle supracoracoideus_l {
    origin_bone: thorax, insertion_bone: wing_humerus_l, action_axis: z, moment_arm_deg: -26,
    tendon_coupling: [wing_ulna_l, z, -0.24]
  }
  muscle pectoralis_r {
    origin_bone: thorax, insertion_bone: wing_humerus_r, action_axis: z, moment_arm_deg: -22,
    tendon_coupling: [wing_humerus_r__wing_ulna_l, z, -0.28], belly_box: sternum_keel, bulge_inflate: 0.14
  }
  muscle supracoracoideus_r {
    origin_bone: thorax, insertion_bone: wing_humerus_r, action_axis: z, moment_arm_deg: 26,
    tendon_coupling: [wing_humerus_r__wing_ulna_l, z, 0.24]
  }
}

@kinematics {
  ik_chain leg_fl_ik { bones: [fore_humerus_l, fore_radius_l], effector: [0, -9, 1], hinge_axis: x, pole: [0, 0, -1] }
  ik_chain leg_fr_ik { bones: [fore_humerus_r, fore_humerus_r__fore_radius_l], effector: [0, -9, 1], hinge_axis: x, pole: [0, 0, -1] }
  ik_chain leg_hl_ik { bones: [hind_femur_l, hind_crus_l], effector: [0, -9, 1], hinge_axis: x, pole: [0, 0, 1] }
  ik_chain leg_hr_ik { bones: [hind_femur_r, hind_femur_r__hind_crus_l], effector: [0, -9, 1], hinge_axis: x, pole: [0, 0, 1] }
  look_chain wyvern_gaze { bones: [thorax, neck_1, neck_2, skull], weight_decay: 0.72 }
  spring tail_seg_3 { stiffness: 30, damping: 5.5 }
  spring tail_seg_4 { stiffness: 22, damping: 4.5 }
}

@behavior {
  clip stalk_gait {
    duration_s: 1.6
    ik_drive leg_fl_ik: foot_cycle(stride=4.5, lift=1.8, phase_deg=0)
    ik_drive leg_hr_ik: foot_cycle(stride=4.5, lift=1.8, phase_deg=0)
    ik_drive leg_fr_ik: foot_cycle(stride=4.5, lift=1.8, phase_deg=180)
    ik_drive leg_hl_ik: foot_cycle(stride=4.5, lift=1.8, phase_deg=180)
    track tail_seg_1: wave(axis=y, amp_deg=10, freq_hz=0.625, phase_deg=0)
    track tail_seg_2: wave(axis=y, amp_deg=14, freq_hz=0.625, phase_deg=45)
    track neck_1: wave(axis=x, amp_deg=5, freq_hz=0.625, phase_deg=90)
  }

  clip aerial_flap {
    duration_s: 2.0
    muscle_drive pectoralis_l: activation(peak=0.85, freq_hz=0.5, phase_deg=0, baseline=0.06)
    muscle_drive pectoralis_r: activation(peak=0.85, freq_hz=0.5, phase_deg=0, baseline=0.06)
    muscle_drive supracoracoideus_l: activation(peak=0.72, freq_hz=0.5, phase_deg=180, baseline=0.05)
    muscle_drive supracoracoideus_r: activation(peak=0.72, freq_hz=0.5, phase_deg=180, baseline=0.05)
    track tail_seg_1: wave(axis=x, amp_deg=8, freq_hz=0.5, phase_deg=0)
    key mandible.rotation { 0: [0, 0, 0], 1.0: [18, 0, 0], 2.0: [0, 0, 0], catmullrom }
  }

  state ember_enraged {
    pose_bias mandible: [26, 0, 0]
    muscle_tone pectoralis_l: 0.30
    muscle_tone pectoralis_r: 0.30
    material_mod ember_organ: { emissive: 1.5, moisture: 0.2, wear_delta: 0.05 }
  }
}

@controller {
  initial: stalk_gait
  stalk_gait -> aerial_flap when altitude > 2.5, blend 0.35
  aerial_flap -> stalk_gait when altitude <= 0.5, blend 0.30
  stalk_gait -> ember_enraged when rage_meter >= 0.8, blend 0.25
}
"""


COLOSSAL_WALKING_CITADEL_DSL: str = """
@meta { archetype: entity, target_profile: geckolib, texel_density: 16, audit_profile: sculpted }

@palette {
  ashlar_masonry: ramp(base_lch=[0.56, 0.03, 78], steps=8, hue_shift=[-3, 3])
  riveted_armor: ramp(base_lch=[0.42, 0.08, 245], steps=8, hue_shift=[-4, 4])
  forged_brass: ramp(base_lch=[0.72, 0.12, 82], steps=6)
  fortress_timber: ramp(base_lch=[0.50, 0.09, 58], steps=6)
  boiler_firebox: ramp(base_lch=[0.74, 0.21, 38], steps=6, emissive=true)
}

@rig {
  root bone chassis_hull {
    pivot_local: [0, 20.73, 0]
    box main_hull { size: [18, 8, 24], material: riveted_armor }
    box underbelly_keel place(base_box=main_hull, anchor=[0, -1, 0], align=[0, 0.45, 0]) {
      size: [12, 4, 20], material: riveted_armor, rotate: [12, 0, 0]
    }
    box boiler_core place(base_box=main_hull, anchor=[0, 0.1, -1], align=[0, 0, 0.8]) {
      size: [12, 7, 6], material: boiler_firebox
    }
    box timber_deck place(base_box=main_hull, anchor=[0, 1, 0], align=[0, -1, 0]) {
      size: [16, 2, 22], material: fortress_timber
    }
    locator helm_bridge_seat { offset: [0, 18, 6], role: rider_seat }

    bone prow_wing_l attach(parent_box=main_hull, anchor=[-0.55, 0, 1], align=[0, 0, -0.5]) mirror_x(prow_wing_r) {
      rest_euler: [0, 26, 0]
      joint_limits: [[-5, 5], [-5, 5], [-5, 5]]
      box prow_plate_l { size: [8, 7, 6], material: riveted_armor, blend_with: [prow_wing_r__prow_plate_l] }
      box ram_spur_l place(base_box=prow_plate_l, anchor=[0, -0.6, 1], align=[0, 0.3, -0.4]) {
        size: [4, 3, 4], material: forged_brass, rotate: [20, 0, 0], blend_with: [prow_wing_r__ram_spur_l]
      }
    }

    bone citadel_keep attach(parent_box=timber_deck, anchor=[0, 1, -0.18], align=[0, -1, 0]) {
      joint_limits: [[-4, 4], [-6, 6], [-4, 4]]
      box keep_walls { size: [14, 10, 14], material: ashlar_masonry }
      box buttress_fore place(base_box=keep_walls, anchor=[0, -0.1, 1], align=[0, 0, -0.5]) {
        size: [6, 7, 3], material: ashlar_masonry, rotate: [18, 0, 0]
      }
      box buttress_aft place(base_box=keep_walls, anchor=[0, -0.1, -1], align=[0, 0, 0.5]) {
        size: [6, 7, 3], material: ashlar_masonry, rotate: [-18, 0, 0]
      }
      box parapet_fore place(base_box=keep_walls, anchor=[0, 1, 0.85], align=[0, -1, 0]) {
        size: [6, 2, 2], material: ashlar_masonry
      }
      box parapet_aft place(base_box=keep_walls, anchor=[0, 1, -0.85], align=[0, -1, 0]) {
        size: [6, 2, 2], material: ashlar_masonry
      }
      box parapet_side_l place(base_box=keep_walls, anchor=[-0.85, 1, 0], align=[0, -1, 0]) {
        size: [2, 2, 6], material: ashlar_masonry
      }
      box parapet_side_r place(base_box=keep_walls, anchor=[0.85, 1, 0], align=[0, -1, 0]) {
        size: [2, 2, 6], material: ashlar_masonry
      }
      locator periscope_focus { offset: [0, 14, 7], role: camera_focus }

      bone side_bastion_l attach(parent_box=keep_walls, anchor=[-1, 0.15, 0], align=[0.5, 0, 0]) mirror_x(side_bastion_r) {
        rest_euler: [0, 0, -18]
        joint_limits: [[-4, 4], [-4, 4], [-4, 4]]
        box sloped_casemate_l { size: [4, 7, 10], material: ashlar_masonry }
      }

      bone corner_turret_fl attach(parent_box=keep_walls, anchor=[-0.85, 1, 0.85], align=[0, -1, 0]) mirror_x(corner_turret_fr) {
        joint_limits: [[-5, 5], [-5, 5], [-5, 5]]
        box watchtower_fl { size: [4, 6, 4], material: ashlar_masonry }
        box spire_roof_fl place(base_box=watchtower_fl, anchor=[0, 1, 0], align=[0, -0.6, 0]) {
          size: [3, 4, 3], material: riveted_armor, rotate: [0, 45, 0]
        }
      }

      bone corner_turret_hl attach(parent_box=keep_walls, anchor=[-0.85, 1, -0.85], align=[0, -1, 0]) mirror_x(corner_turret_hr) {
        joint_limits: [[-5, 5], [-5, 5], [-5, 5]]
        box watchtower_hl { size: [4, 6, 4], material: ashlar_masonry }
        box spire_roof_hl place(base_box=watchtower_hl, anchor=[0, 1, 0], align=[0, -0.6, 0]) {
          size: [3, 4, 3], material: riveted_armor, rotate: [0, 45, 0]
        }
      }

      bone bastion_drum attach(parent_box=keep_walls, anchor=[0, 1, 0], align=[0, -1, 0]) {
        joint_limits: [[-4, 4], [-180, 180], [-4, 4]]
        box drum_core { size: [6, 4, 6], material: ashlar_masonry }
        box cupola_pedestal place(base_box=drum_core, anchor=[0, 1, 0], align=[0, -1, 0]) {
          size: [3, 3, 3], material: ashlar_masonry
        }
        box cupola_crown place(base_box=cupola_pedestal, anchor=[0, 1, 0], align=[0, -1, 0]) {
          size: [4, 2, 4], material: forged_brass
        }

        bone dome_rib joint(parent=drum_core, at=[0, 2, 2.5], align=[0, -0.8, 0.5]) radial(count=4, axis=y) {
          rest_euler: [-22, 0, 0]
          joint_limits: [[-4, 4], [-4, 4], [-4, 4]]
          box arch_slab { size: [3, 3, 2], material: ashlar_masonry, blend_with: [dome_rib_r1__arch_slab, cupola_pedestal, cupola_crown] }
        }

        bone beacon_rotor attach(parent_box=cupola_crown, anchor=[0, 1, 0], align=[0, -1, 0]) {
          joint_limits: [[-5, 5], continuous, [-5, 5]]
          box lantern_spindle { size: [2, 5, 2], material: forged_brass }
          box beacon_lens place(base_box=lantern_spindle, anchor=[0, 1, 0], align=[0, -1, 0]) {
            size: [4, 3, 4], material: boiler_firebox
          }
          box beacon_finial place(base_box=beacon_lens, anchor=[0, 1, 0], align=[0, -1, 0]) {
            size: [2, 3, 2], material: forged_brass
          }
        }
      }

      bone windmill_mast attach(parent_box=parapet_aft, anchor=[0, 1, 0], align=[0, -1, 0]) {
        joint_limits: [[-6, 6], [-6, 6], [-6, 6]]
        box mast_post { size: [2, 10, 2], material: fortress_timber }
        box mast_boom place(base_box=mast_post, anchor=[0, 0.85, -1], align=[0, 0, 1]) {
          size: [2, 2, 3], material: fortress_timber
        }

        bone windmill_hub attach(parent_box=mast_boom, anchor=[0, 0, -1], align=[0, 0, 1]) {
          joint_limits: [[-5, 5], [-5, 5], continuous]
          box hub_boss { size: [3, 3, 2], material: forged_brass }

          bone vane_arm joint(parent=hub_boss, at=[0, 1.5, -0.5], align=[0, -1, 0]) radial(count=4, axis=z) {
            joint_limits: [[-5, 5], [-5, 5], [-5, 5]]
            box sail_spar { size: [2, 5, 1], material: fortress_timber }
          }
        }
      }
    }

    bone smokestack_l attach(parent_box=boiler_core, anchor=[-0.65, 1, -0.45], align=[0, -1, 0]) mirror_x(smokestack_r) {
      rest_euler: [-8, 0, 0]
      joint_limits: [[-18, 18], [-12, 12], [-18, 18]]
      box flue_base_l { size: [4, 3, 4], material: riveted_armor }
      box flue_pipe_l place(base_box=flue_base_l, anchor=[0, 1, 0], align=[0, -1, 0]) {
        size: [3, 9, 3], material: riveted_armor
      }
      box flue_crown_l place(base_box=flue_pipe_l, anchor=[0, 1, 0], align=[0, -1, 0]) {
        size: [4, 2, 4], material: forged_brass
      }
      locator exhaust_vfx_l { offset: [0, 14, 0], role: vfx_emitter }
    }

    bone fore_cannon_turret attach(parent_box=main_hull, anchor=[0, 1, 0.92], align=[0, -1, 0]) {
      joint_limits: [[-5, 5], [-90, 90], [-5, 5]]
      box turret_barbette { size: [5, 3, 4], material: ashlar_masonry }
      box turret_cupola place(base_box=turret_barbette, anchor=[0, 1, 0], align=[0, -1, 0]) {
        size: [6, 3, 4], material: riveted_armor
      }

      bone cannon_barrel attach(parent_box=turret_cupola, anchor=[0, 0.2, 1], align=[0, 0, -1]) {
        rest_euler: [-6, 0, 0]
        joint_limits: [[-28, 12], [-5, 5], [-5, 5]]
        box ordnance_tube { size: [2, 2, 8], material: forged_brass }
        box muzzle_brake place(base_box=ordnance_tube, anchor=[0, 0, 1], align=[0, 0, -1]) {
          size: [3, 3, 2], material: riveted_armor
        }
        locator cannon_muzzle { offset: [0, 0, 10], role: vfx_emitter }
      }
    }

    bone flywheel_l attach(parent_box=main_hull, anchor=[-1, 0.35, -0.12], align=[1, 0, 0]) mirror_x(flywheel_r) {
      joint_limits: [continuous, [-5, 5], [-5, 5]]
      box drive_wheel_l { size: [2, 6, 6], material: forged_brass }
      box wheel_rim_cross_l place(base_box=drive_wheel_l, anchor=[-1, 0, 0], align=[1, 0, 0]) {
        size: [1, 5, 5], material: riveted_armor, rotate: [45, 0, 0]
      }
    }

    bone pinion_gear_l attach(parent_box=main_hull, anchor=[-1, 0.35, 0.45], align=[1, 0, 0]) mirror_x(pinion_gear_r) {
      joint_limits: [continuous, [-5, 5], [-5, 5]]
      box spur_gear_l { size: [2, 3, 3], material: forged_brass }
    }

    bone leg_fore_hip_l attach(parent_box=main_hull, anchor=[-1, -0.55, 0.78], align=[1, 0.7, 0]) mirror_x(leg_fore_hip_r) {
      rest_euler: [10, 0, 0]
      joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
      box fore_femur_girdle_l { size: [4, 9, 4], material: riveted_armor }
      box piston_cyl_fl place(base_box=fore_femur_girdle_l, anchor=[-1, 0.2, 0], align=[0.4, 0, 0]) {
        size: [2, 6, 3], material: forged_brass, rotate: [0, 0, 14]
      }

      bone leg_fore_shin_l attach(parent_box=fore_femur_girdle_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [-10, 0, 0]
        joint_limits: [[-65, 45], [-15, 15], [-15, 15]]
        box fore_strut_l { size: [4, 9, 4], material: riveted_armor }
        box fore_foot_pad_l place(base_box=fore_strut_l, anchor=[0, -1, 0.2], align=[0, 1, 0]) {
          size: [6, 2, 7], material: forged_brass
        }
        locator pad_fl_effector { offset: [0, -11, 0.5], role: ik_effector }
      }
    }

    bone leg_aft_hip_l attach(parent_box=main_hull, anchor=[-1, -0.55, -0.78], align=[1, 0.7, 0]) mirror_x(leg_aft_hip_r) {
      rest_euler: [-10, 0, 0]
      joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
      box aft_femur_girdle_l { size: [4, 9, 4], material: riveted_armor }
      box piston_cyl_hl place(base_box=aft_femur_girdle_l, anchor=[-1, 0.2, 0], align=[0.4, 0, 0]) {
        size: [2, 6, 3], material: forged_brass, rotate: [0, 0, 14]
      }

      bone leg_aft_shin_l attach(parent_box=aft_femur_girdle_l, anchor=[0, -1, 0], align=[0, 1, 0]) {
        rest_euler: [10, 0, 0]
        joint_limits: [[-65, 45], [-15, 15], [-15, 15]]
        box aft_strut_l { size: [4, 9, 4], material: riveted_armor }
        box aft_foot_pad_l place(base_box=aft_strut_l, anchor=[0, -1, 0.2], align=[0, 1, 0]) {
          size: [6, 2, 7], material: forged_brass
        }
        locator pad_hl_effector { offset: [0, -11, 0.5], role: ik_effector }
      }
    }
  }
}

@materials {
  ashlar_masonry: process(family=weathered_stone, roughness=0.82, porosity=0.48, wear=0.18, weld_group=citadel_stone)
  riveted_armor: process(family=painted_metal, roughness=0.36, coating=0.48, wear=0.65, weld_group=hull_plates)
  forged_brass: process(family=brushed_metal, roughness=0.26, wear=0.22)
  fortress_timber: process(family=solid_wood, roughness=0.64, porosity=0.38, wear=0.24)
  boiler_firebox: process(family=polymer_dielectric, roughness=0.25, emissive=0.85)
}

@style { target_encoding: vanilla, quantize_ramp: false, bevel_highlight: 0.10 }

@muscles {
  muscle hydraulic_ram_fl {
    origin_bone: chassis_hull, insertion_bone: leg_fore_hip_l, action_axis: x, moment_arm_deg: -28,
    tendon_coupling: [leg_fore_shin_l, x, 0.35], belly_box: piston_cyl_fl, bulge_inflate: 0.10
  }
  muscle hydraulic_ram_fr {
    origin_bone: chassis_hull, insertion_bone: leg_fore_hip_r, action_axis: x, moment_arm_deg: -28,
    tendon_coupling: [leg_fore_hip_r__leg_fore_shin_l, x, 0.35], belly_box: leg_fore_hip_r__piston_cyl_fl, bulge_inflate: 0.10
  }
}

@kinematics {
  ik_chain gait_fl { bones: [leg_fore_hip_l, leg_fore_shin_l], effector: [0, -10, 0.5], hinge_axis: x, pole: [0, 0, 1] }
  ik_chain gait_fr { bones: [leg_fore_hip_r, leg_fore_hip_r__leg_fore_shin_l], effector: [0, -10, 0.5], hinge_axis: x, pole: [0, 0, 1] }
  ik_chain gait_hl { bones: [leg_aft_hip_l, leg_aft_shin_l], effector: [0, -10, 0.5], hinge_axis: x, pole: [0, 0, -1] }
  ik_chain gait_hr { bones: [leg_aft_hip_r, leg_aft_hip_r__leg_aft_shin_l], effector: [0, -10, 0.5], hinge_axis: x, pole: [0, 0, -1] }
  look_chain turret_aim { bones: [fore_cannon_turret, cannon_barrel], weight_decay: 0.65 }
  spring smokestack_l { stiffness: 38, damping: 6.0 }
  spring smokestack_r { stiffness: 38, damping: 6.0 }
}

@behavior {
  clip fortress_march {
    duration_s: 2.0
    ik_drive gait_fl: foot_cycle(stride=5.0, lift=2.0, phase_deg=0)
    ik_drive gait_hr: foot_cycle(stride=5.0, lift=2.0, phase_deg=0)
    ik_drive gait_fr: foot_cycle(stride=5.0, lift=2.0, phase_deg=180)
    ik_drive gait_hl: foot_cycle(stride=5.0, lift=2.0, phase_deg=180)
    track flywheel_l: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track flywheel_r: spin(axis=x, amp_deg=360, freq_hz=0.5)
    track pinion_gear_l: spin(axis=x, amp_deg=-720, freq_hz=0.5)
    track pinion_gear_r: spin(axis=x, amp_deg=-720, freq_hz=0.5)
    track windmill_hub: spin(axis=z, amp_deg=360, freq_hz=0.5)
    track beacon_rotor: spin(axis=y, amp_deg=360, freq_hz=0.5)
    track fore_cannon_turret: wave(axis=y, amp_deg=22, freq_hz=0.5)
  }

  state boiler_overdrive {
    pose_bias cannon_barrel: [-10, 0, 0]
    muscle_tone hydraulic_ram_fl: 0.25
    muscle_tone hydraulic_ram_fr: 0.25
    material_mod boiler_firebox: { emissive: 1.6, moisture: 0.05, wear_delta: 0.08 }
  }
}

@controller {
  initial: fortress_march
  fortress_march -> boiler_overdrive when steam_pressure >= 0.85, blend 0.30
}
"""


def build_sculpted_elder_wyvern_world(
    *,
    seed: int = 303,
    subdivisions_per_unit: int = 1,
    texel_density: int = 16,
) -> WorldState:
    """Build Flagship Exemplar 3 (ADR 0007): 62-box biomechanical sculpted wyvern with sloped snout, 68-deg horns, patagium wings, Hill muscles, and 4-limb stance IK."""
    program = parse_program(SCULPTED_ELDER_WYVERN_DSL)
    if texel_density != program.meta.texel_density:
        program = validate_program(replace(program, meta=replace(program.meta, texel_density=texel_density)))

    base_mp = MaterialProgram.from_program(program, seed=seed)
    material_program = replace(
        base_mp,
        stamps=(
            StampOverride(
                box_name="cranium",
                face="west",
                u_min=1.0,
                v_min=1.5,
                u_max=2.5,
                v_max=2.8,
                lch=(0.80, 0.20, 52.0),
                shape="disc",
                emissive=0.85,
            ),
            StampOverride(
                box_name="cranium",
                face="east",
                u_min=3.5,
                v_min=1.5,
                u_max=5.0,
                v_max=2.8,
                lch=(0.80, 0.20, 52.0),
                shape="disc",
                emissive=0.85,
            ),
        ),
    )
    domain = build_surface_domain(program, material_program, subdivisions_per_unit=subdivisions_per_unit)
    surface_state = initialize_surface_state(domain)
    hide_spec = material_program.materials["wyvern_hide"]
    hide_faces = {fk for fk, g in domain.faces.items() if g.material_name == "wyvern_hide"}
    integrate_gray_scott_on_state(domain, surface_state, hide_spec, hide_faces, duration=0.0)
    return WorldState(
        program=program,
        material_program=material_program,
        surface_state=surface_state,
        style_config=StyleConfig.from_program(program),
        environment=EnvironmentState(),
    )


def build_colossal_walking_citadel_world(
    *,
    seed: int = 404,
    subdivisions_per_unit: int = 1,
    texel_density: int = 16,
) -> WorldState:
    """Build Flagship Exemplar 4 (ADR 0007): 74-box walking castle fortress vehicle with mitered prow, radial dome, continuous spin gears/windmill/beacon, and 4-leg hydraulic stance IK."""
    program = parse_program(COLOSSAL_WALKING_CITADEL_DSL)
    if texel_density != program.meta.texel_density:
        program = validate_program(replace(program, meta=replace(program.meta, texel_density=texel_density)))

    material_program = MaterialProgram.from_program(program, seed=seed)
    domain = build_surface_domain(program, material_program, subdivisions_per_unit=subdivisions_per_unit)
    surface_state = initialize_surface_state(domain)
    return WorldState(
        program=program,
        material_program=material_program,
        surface_state=surface_state,
        style_config=StyleConfig.from_program(program),
        environment=EnvironmentState(),
    )
