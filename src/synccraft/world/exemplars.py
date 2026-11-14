"""Flagship stateful world exemplars for P1-03d/e causal verification."""

from __future__ import annotations

from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Place, Program, validate_program
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
