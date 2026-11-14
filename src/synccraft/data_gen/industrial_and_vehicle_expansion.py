"""Industrial Mechanical Engineering, Vehicle & Craft Expansion Generators for SyncCraft v3.0.

Implements 12 procedural Block-DSL v3 AST generators (10 Seen + 2 Held-Out Unseen)
across mechanical engineering machines, land/rail/air/marine vehicles, and horological
or scientific craft apparatus:
  A. Industrial Mechanical Engineering (3 Seen):
     1. piston_crank_engine
     2. cnc_milling_press
     3. centrifugal_pump_valve_skid
  B. Vehicles — Land, Construction, Rail, Air & Marine (5 Seen + 2 Unseen):
     4. wheeled_haul_truck (Seen)
     5. tracked_excavator (Seen)
     6. steam_rail_locomotive (Seen)
     7. twin_rotor_helicopter_drone (Seen)
     8. cargo_tugboat_vessel (Seen)
     9. articulated_forklift_loader (Unseen Held-Out, >= 18 boxes)
    10. submersible_rov_craft (Unseen Held-Out, >= 18 boxes)
  C. Craft / Mechanical Props (2 Seen):
    11. clockwork_pendulum_clock
    12. alchemical_workbench_apparatus

All 12 generators target ``Meta("block_multipart", profile, density)`` with
``profile in ("bbmodel_5", "java_modern")``, keep world OBBs strictly within
Java Modern's ``[-16.0, 32.0]`` envelope, and pass 15-axis SAT + P1-02 physical
contact audits across parametric variants.
"""

from __future__ import annotations

import random
from typing import Any

from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Place, Program

# Domain mapping for the 12 expansion families (matches taxonomy_catalog.py)
INDUSTRIAL_AND_VEHICLE_DOMAINS: dict[str, str] = {
    "piston_crank_engine": "industrial",
    "cnc_milling_press": "industrial",
    "centrifugal_pump_valve_skid": "industrial",
    "wheeled_haul_truck": "industrial",
    "tracked_excavator": "industrial",
    "steam_rail_locomotive": "industrial",
    "twin_rotor_helicopter_drone": "industrial",
    "cargo_tugboat_vessel": "industrial",
    "articulated_forklift_loader": "industrial",
    "submersible_rov_craft": "industrial",
    "clockwork_pendulum_clock": "artificial",
    "alchemical_workbench_apparatus": "artificial",
}

SEEN_INDUSTRIAL_AND_VEHICLE_FAMILIES: tuple[str, ...] = (
    "piston_crank_engine",
    "cnc_milling_press",
    "centrifugal_pump_valve_skid",
    "wheeled_haul_truck",
    "tracked_excavator",
    "steam_rail_locomotive",
    "twin_rotor_helicopter_drone",
    "cargo_tugboat_vessel",
    "clockwork_pendulum_clock",
    "alchemical_workbench_apparatus",
)

UNSEEN_INDUSTRIAL_AND_VEHICLE_FAMILIES: tuple[str, ...] = (
    "articulated_forklift_loader",
    "submersible_rov_craft",
)

# Curated perceptual OKLCH palette banks
_HEAVY_ALLOYS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("cast_iron_block", (0.34, 0.018, 255.0)),
    ("machined_billet_steel", (0.72, 0.020, 242.0)),
    ("gunmetal_housing", (0.42, 0.024, 250.0)),
    ("anodized_aluminum", (0.78, 0.016, 236.0)),
    ("forged_crank_steel", (0.58, 0.022, 246.0)),
)

_INDUSTRIAL_COATINGS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("caterpillar_yellow", (0.78, 0.172, 84.0)),
    ("safety_orange_coat", (0.66, 0.185, 46.0)),
    ("process_cobalt_enamel", (0.50, 0.138, 248.0)),
    ("machine_vista_green", (0.56, 0.095, 162.0)),
    ("maritime_hull_red", (0.46, 0.165, 28.0)),
    ("rov_syntactic_yellow", (0.82, 0.176, 88.0)),
    ("locomotive_boiler_black", (0.25, 0.016, 258.0)),
    ("rescue_guard_white", (0.88, 0.014, 245.0)),
)

_DARK_RUBBER_CARBON: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("vulcanized_tire_rubber", (0.23, 0.014, 260.0)),
    ("crawler_track_iron", (0.29, 0.018, 252.0)),
    ("graphite_seal_trim", (0.26, 0.016, 265.0)),
)

_BRASS_COPPER_TRIM: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("horological_brass", (0.74, 0.148, 80.0)),
    ("alembic_copper", (0.64, 0.162, 46.0)),
    ("valve_bronze_trim", (0.68, 0.142, 66.0)),
)

_CABINET_TIMBERS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("mahogany_casework", (0.38, 0.098, 38.0)),
    ("seasoned_workbench_oak", (0.52, 0.105, 58.0)),
    ("walnut_horologe_wood", (0.36, 0.078, 48.0)),
)

_OPTIC_GLOW_ACCENTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("amber_beacon_lens", (0.82, 0.185, 76.0)),
    ("cyan_canopy_glaze", (0.76, 0.135, 212.0)),
    ("emerald_telemetry_led", (0.78, 0.188, 152.0)),
    ("alchemical_distillate", (0.74, 0.192, 298.0)),
    ("ivory_enamel_dial", (0.90, 0.032, 86.0)),
)


def _jitter_lch(
    rng: random.Random,
    lch: tuple[float, float, float],
) -> tuple[float, float, float]:
    l0, c0, h0 = lch
    l_val = round(min(0.95, max(0.16, l0 + rng.uniform(-0.04, 0.04))), 4)
    c_val = round(max(0.006, c0 + rng.uniform(-0.014, 0.020)), 4)
    h_val = round((h0 + rng.uniform(-16.0, 16.0)) % 360.0, 2)
    return (l_val, c_val, h_val)


def _pal(
    rng: random.Random,
    spec: tuple[str, tuple[float, float, float]],
    *,
    emissive: bool = False,
    suffix: str = "",
) -> Palette:
    raw_name, lch = spec
    steps = rng.choice((4, 5, 6, 7))
    hue_shift = (round(rng.uniform(-6.0, 0.0), 2), round(rng.uniform(0.0, 8.0), 2))
    return Palette(
        name=f"{raw_name}{suffix}",
        base_lch=_jitter_lch(rng, lch),
        steps=steps,
        hue_shift=hue_shift,
        emissive=emissive,
    )


def _jpiv(
    rng: random.Random,
    base_xyz: tuple[float, float, float],
    span: float = 0.45,
) -> tuple[float, float, float]:
    return (
        round(base_xyz[0] + rng.uniform(-span, span), 3),
        round(base_xyz[1] + rng.uniform(-0.25, 0.25), 3),
        round(base_xyz[2] + rng.uniform(-span, span), 3),
    )


def _static_profile_and_density(v_idx: int) -> tuple[str, int]:
    profile = "java_modern" if (v_idx // 3) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 6 == 0) else 16
    return profile, density


# ==============================================================================
# A. INDUSTRIAL MECHANICAL ENGINEERING (3 Seen, >= 16 boxes, >= 4 bones)
# ==============================================================================


def _gen_piston_crank_engine(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("inline_four_diesel", "v_twin_compressor", "marine_crank_block")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_block = _pal(rng, _HEAVY_ALLOYS[0])
    p_mach = _pal(rng, _HEAVY_ALLOYS[1], suffix="_mach")
    p_coat = _pal(rng, _INDUSTRIAL_COATINGS[v_idx % 4])
    p_brass = _pal(rng, rng.choice(_BRASS_COPPER_TRIM))

    block_len = rng.choice((10, 12))
    block_h = rng.choice((4, 5))
    flywheel_roll = rng.choice((0.0, 15.0, 22.5, 30.0, 45.0))
    pulley_roll = rng.choice((15.0, 30.0, 45.0))
    piston_tilt = rng.choice((0.0, 5.0, 7.5))
    hub_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)
    has_third_piston = (block_len == 12 and v_idx % 2 == 0)

    flywheel_bone = Bone(
        "flywheel_rotor_rig",
        rest_euler=(0.0, 0.0, flywheel_roll),
        attach=Attach("bellhousing_collar", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("flywheel_inertia_disc", (6, 6, 2), p_mach.name),
            Box("clutch_pilot_boss", (2, 2, 1), p_brass.name, Place("flywheel_inertia_disc", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), inflate=hub_inf),
        ],
    )

    pulley_fan_bone = Bone(
        "belt_pulley_fan_shaft",
        rest_euler=(0.0, 0.0, pulley_roll),
        attach=Attach("front_radiator_core", (0.0, -0.1, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("crank_pulley_sheave", (2, 2, 1), p_mach.name),
            Box("cooling_blade_up", (1, 2, 1), p_coat.name, Place("crank_pulley_sheave", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("cooling_blade_dn", (1, 2, 1), p_coat.name, Place("crank_pulley_sheave", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    piston_fore = Bone(
        "piston_conrod_fore",
        rest_euler=(0.0, 0.0, piston_tilt),
        attach=Attach("cylinder_head_bank", (0.0, 1.0, -0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("conrod_shank_fore", (2, 2, 2), p_mach.name),
            Box("piston_crown_fore", (2, 2, 2), p_brass.name, Place("conrod_shank_fore", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    piston_aft = Bone(
        "piston_conrod_aft",
        rest_euler=(0.0, 0.0, -piston_tilt),
        attach=Attach("cylinder_head_bank", (0.0, 1.0, 0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("conrod_shank_aft", (2, 2, 2), p_mach.name),
            Box("piston_crown_aft", (2, 2, 2), p_brass.name, Place("conrod_shank_aft", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("crankcase_main_block", (6, block_h, block_len), p_coat.name),
        Box("cast_iron_sump_pan", (4, 2, block_len - 2), p_block.name, Place("crankcase_main_block", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("bedplate_rail_e", (1, 2, block_len - 2), p_block.name, Place("crankcase_main_block", (1.0, -0.5, 0.0), (-1.0, 0.0, 0.0))),
        Box("bedplate_rail_w", (1, 2, block_len - 2), p_block.name, Place("crankcase_main_block", (-1.0, -0.5, 0.0), (1.0, 0.0, 0.0))),
        Box("cylinder_head_bank", (4, 3, block_len - 2), p_mach.name, Place("crankcase_main_block", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("intake_plenum_e", (2, 2, block_len - 4), p_mach.name, Place("cylinder_head_bank", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        Box("air_filter_canister", (2, 2, 2), p_block.name, Place("intake_plenum_e", (0.0, 1.0, -0.4), (0.0, -1.0, 0.0))),
        Box("exhaust_header_w", (2, 2, block_len - 4), p_brass.name, Place("cylinder_head_bank", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        Box("exhaust_stack_riser", (1, 4, 1), p_block.name, Place("exhaust_header_w", (-0.5, 1.0, 0.4), (0.0, -1.0, 0.0))),
        Box("timing_gear_cover", (4, block_h, 2), p_mach.name, Place("crankcase_main_block", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("front_radiator_core", (6, block_h + 1, 2), p_block.name, Place("timing_gear_cover", (0.0, 0.2, -1.0), (0.0, 0.0, 1.0))),
        Box("bellhousing_collar", (4, 4, 1), p_block.name, Place("crankcase_main_block", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        flywheel_bone,
        pulley_fan_bone,
        piston_fore,
        piston_aft,
    ]
    if has_third_piston:
        root_items.append(
            Bone(
                "piston_conrod_mid",
                rest_euler=(0.0, 0.0, 0.0),
                attach=Attach("cylinder_head_bank", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
                items=[
                    Box("conrod_shank_mid", (2, 2, 2), p_mach.name),
                    Box("piston_crown_mid", (2, 2, 2), p_brass.name, Place("conrod_shank_mid", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                ],
            )
        )

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_block, p_mach, p_coat, p_brass],
        root=Bone("engine_crankcase_datum", pivot_local=_jpiv(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": block_len + 8,
        "secondary_span": block_h + 10,
        "joint_angle_deg": flywheel_roll,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": hub_inf,
        "material_theme": f"{p_coat.name}+{p_mach.name}+{p_block.name}",
        "notes": f"{block_len}-long reciprocating engine ({substyle}) with cast sump pan, intake/exhaust manifolds, front radiator + pulley fan, rear {flywheel_roll:g} deg flywheel rotor, and articulated piston/conrod assemblies",
        "extra": {"block_len": block_len, "block_h": block_h, "has_third_piston": has_third_piston},
    }


def _gen_cnc_milling_press(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("three_axis_cnc_mill", "bridgeport_turret_mill", "heavy_stamping_press")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_cast = _pal(rng, _HEAVY_ALLOYS[2])
    p_steel = _pal(rng, _HEAVY_ALLOYS[3], suffix="_table")
    p_coat = _pal(rng, _INDUSTRIAL_COATINGS[(v_idx + 2) % 4])
    p_lamp = _pal(rng, _OPTIC_GLOW_ACCENTS[2], emissive=True)

    base_w = rng.choice((10, 12))
    base_d = rng.choice((12, 14))
    col_h = rng.choice((11, 13))
    turret_tilt = rng.choice((-7.5, 0.0, 7.5))
    cutter_spin = rng.choice((0.0, 22.5, 45.0))
    pendant_yaw = rng.choice((-22.5, -15.0, 15.0, 22.5))
    chuck_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    worktable_bone = Bone(
        "xy_worktable_stage",
        attach=Attach("cast_machine_base", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0)),
        items=[
            Box("xy_saddle_slide", (4, 1, 4), p_cast.name),
            Box("tslot_worktable", (6, 1, 4), p_steel.name, Place("xy_saddle_slide", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("vise_fixed_jaw", (3, 1, 1), p_cast.name, Place("tslot_worktable", (0.0, 1.0, -0.75), (0.0, -1.0, 0.0))),
            Box("vise_clamping_jaw", (3, 1, 1), p_cast.name, Place("tslot_worktable", (0.0, 1.0, 0.75), (0.0, -1.0, 0.0))),
            Box("billet_workpiece", (2, 2, 1), p_coat.name, Place("tslot_worktable", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    quill_chuck_bone = Bone(
        "quill_cutter_chuck",
        rest_euler=(0.0, cutter_spin, 0.0),
        attach=Attach("z_spindle_housing", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("collet_chuck_nose", (2, 2, 2), p_steel.name, inflate=chuck_inf),
            Box("carbide_endmill_bit", (1, 2, 1), p_steel.name, Place("collet_chuck_nose", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    spindle_head_bone = Bone(
        "z_axis_spindle_head",
        rest_euler=(0.0, 0.0, turret_tilt),
        attach=Attach("column_overarm_ram", (0.0, -1.0, -0.7), (0.0, 1.0, 0.0)),
        items=[
            Box("z_spindle_housing", (3, 2, 2), p_coat.name),
            Box("coolant_jet_nozzle", (1, 1, 1), p_steel.name, Place("z_spindle_housing", (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0))),
            quill_chuck_bone,
        ],
    )

    hmi_pendant_bone = Bone(
        "hmi_pendant_arm",
        rest_euler=(0.0, pendant_yaw, 0.0),
        attach=Attach("rear_vertical_column", (1.0, 0.35, 0.0), (-1.0, 0.0, 0.0)),
        items=[
            Box("pendant_swivel_boom", (3, 1, 1), p_steel.name),
            Box("hmi_operator_panel", (1, 3, 3), p_coat.name, Place("pendant_swivel_boom", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("cycle_status_beacon", (1, 2, 1), p_lamp.name, Place("hmi_operator_panel", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("cast_machine_base", (base_w, 3, base_d), p_cast.name),
        Box("isolation_pad_fore", (base_w - 2, 1, 2), p_cast.name, Place("cast_machine_base", (0.0, -1.0, -0.75), (0.0, 1.0, 0.0))),
        Box("isolation_pad_aft", (base_w - 2, 1, 2), p_cast.name, Place("cast_machine_base", (0.0, -1.0, 0.75), (0.0, 1.0, 0.0))),
        Box("chip_tray_curb_e", (1, 1, base_d - 6), p_steel.name, Place("cast_machine_base", (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))),
        Box("chip_tray_curb_w", (1, 1, base_d - 6), p_steel.name, Place("cast_machine_base", (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))),
        Box("rear_vertical_column", (6, col_h, 4), p_coat.name, Place("cast_machine_base", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("cnc_servo_cabinet", (4, 6, 2), p_cast.name, Place("rear_vertical_column", (0.0, -0.2, 1.0), (0.0, 0.0, -1.0))),
        Box("column_overarm_ram", (5, 3, 7), p_coat.name, Place("rear_vertical_column", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("spindle_drive_motor", (3, 3, 3), p_cast.name, Place("column_overarm_ram", (0.0, 1.0, -0.6), (0.0, -1.0, 0.0))),
        worktable_bone,
        spindle_head_bone,
        hmi_pendant_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_cast, p_steel, p_coat, p_lamp],
        root=Bone("cnc_mill_bed_datum", pivot_local=_jpiv(rng, (8.0, 3.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": col_h + 10,
        "secondary_span": base_d + 2,
        "joint_angle_deg": pendant_yaw,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": chuck_inf,
        "material_theme": f"{p_coat.name}+{p_steel.name}+{p_cast.name}",
        "notes": f"{col_h}-tall column CNC mill/press ({substyle}) with T-slot vise worktable, workpiece, articulated Z spindle + spinning endmill chuck, and {pendant_yaw:g} deg HMI pendant arm",
        "extra": {"base_w": base_w, "base_d": base_d, "col_h": col_h, "turret_tilt": turret_tilt},
    }


def _gen_centrifugal_pump_valve_skid(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("chemical_transfer_skid", "boiler_feed_pump", "coolant_manifold_unit")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_skid = _pal(rng, _HEAVY_ALLOYS[0])
    p_pipe = _pal(rng, _HEAVY_ALLOYS[3], suffix="_pipe")
    p_pump = _pal(rng, _INDUSTRIAL_COATINGS[2])
    p_valve = _pal(rng, _INDUSTRIAL_COATINGS[1], suffix="_valve")

    skid_len = rng.choice((14, 16))
    skid_w = rng.choice((8, 10))
    valve_yaw = rng.choice((0.0, 15.0, 30.0, 45.0))
    gauge_tilt = rng.choice((0.0, 11.25))
    lever_pitch = rng.choice((-15.0, 0.0, 15.0))
    flange_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    gate_valve_bone = Bone(
        "gate_valve_handwheel_rig",
        rest_euler=(0.0, valve_yaw, 0.0),
        attach=Attach("discharge_valve_bonnet", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("rising_valve_stem", (1, 2, 1), p_pipe.name),
            Box("handwheel_crossbar", (5, 1, 1), p_valve.name, Place("rising_valve_stem", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("handwheel_rim_fore", (1, 1, 2), p_valve.name, Place("handwheel_crossbar", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("handwheel_rim_aft", (1, 1, 2), p_valve.name, Place("handwheel_crossbar", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    gauge_bone = Bone(
        "pressure_gauge_pod",
        rest_euler=(0.0, 0.0, gauge_tilt),
        attach=Attach("discharge_valve_bonnet", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        items=[
            Box("gauge_siphon_stem", (2, 1, 1), p_pipe.name),
            Box("bourdon_gauge_dial", (1, 3, 3), p_valve.name, Place("gauge_siphon_stem", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    bypass_lever_bone = Bone(
        "bypass_shutoff_lever",
        rest_euler=(lever_pitch, 0.0, 0.0),
        attach=Attach("bypass_manifold_standpipe", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("quarter_turn_handle", (1, 1, 3), p_valve.name),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("skid_deck_plate", (skid_w, 1, skid_len), p_skid.name),
        Box("ibeam_runner_e", (2, 2, skid_len), p_skid.name, Place("skid_deck_plate", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))),
        Box("ibeam_runner_w", (2, 2, skid_len), p_skid.name, Place("skid_deck_plate", (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0))),
        Box("motor_mount_plinth", (4, 1, 4), p_skid.name, Place("skid_deck_plate", (0.0, 1.0, 0.55), (0.0, -1.0, 0.0))),
        Box("electric_drive_motor", (5, 5, 5), p_pump.name, Place("motor_mount_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("motor_cooling_fins", (3, 1, 4), p_pipe.name, Place("electric_drive_motor", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("conduit_terminal_box", (1, 2, 2), p_skid.name, Place("electric_drive_motor", (1.0, 0.2, 0.0), (-1.0, 0.0, 0.0))),
        Box("shaft_coupling_guard", (3, 3, 2), p_valve.name, Place("skid_deck_plate", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("volute_cradle_foot", (4, 1, 3), p_skid.name, Place("skid_deck_plate", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0))),
        Box("volute_pump_casing", (6, 6, 3), p_pump.name, Place("volute_cradle_foot", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("suction_inlet_spool", (3, 3, 2), p_pipe.name, Place("volute_pump_casing", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("suction_weld_flange", (4, 4, 1), p_pipe.name, Place("suction_inlet_spool", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), inflate=flange_inf),
        Box("discharge_riser_pipe", (2, 4, 2), p_pipe.name, Place("volute_pump_casing", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("discharge_valve_bonnet", (4, 3, 3), p_pump.name, Place("discharge_riser_pipe", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("bypass_manifold_standpipe", (1, 4, 1), p_pipe.name, Place("skid_deck_plate", (-0.75, 1.0, 0.0), (0.0, -1.0, 0.0))),
        gate_valve_bone,
        gauge_bone,
        bypass_lever_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_skid, p_pipe, p_pump, p_valve],
        root=Bone("pump_skid_datum", pivot_local=_jpiv(rng, (8.0, 4.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 20,
        "secondary_span": skid_len + 2,
        "joint_angle_deg": valve_yaw,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": flange_inf,
        "material_theme": f"{p_pump.name}+{p_pipe.name}+{p_valve.name}",
        "notes": f"{skid_len}-long I-beam process pump skid ({substyle}) with finned electric drive motor, coupling guard, volute casing, vertical riser, {valve_yaw:g} deg gate-valve handwheel, pressure gauge, and bypass line",
        "extra": {"skid_len": skid_len, "skid_w": skid_w, "valve_yaw": valve_yaw},
    }


# ==============================================================================
# B. VEHICLES — LAND, CONSTRUCTION, RAIL, AIR & MARINE (5 Seen + 2 Unseen)
# ==============================================================================


def _gen_wheeled_haul_truck(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("mining_dump_hauler", "six_wheel_cargo_tipper", "offroad_utility_truck")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_body = _pal(rng, _INDUSTRIAL_COATINGS[v_idx % 2])
    p_frame = _pal(rng, _HEAVY_ALLOYS[0])
    p_tire = _pal(rng, _DARK_RUBBER_CARBON[0])
    p_lamp = _pal(rng, _OPTIC_GLOW_ACCENTS[0], emissive=True)

    chassis_len = rng.choice((16, 18))
    bed_pitch = rng.choice((11.25, 15.0, 18.0))
    wheel_spin = rng.choice((0.0, 15.0, 30.0))
    hub_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    front_wheel = Bone(
        "front_wheel_l",
        rest_euler=(wheel_spin, 0.0, 0.0),
        attach=Attach("ladder_frame_spine", (1.0, -0.5, -0.65), (-1.0, 0.0, 0.0)),
        mirror_name="front_wheel_r",
        items=[
            Box("front_axle_spindle", (2, 1, 1), p_frame.name),
            Box("front_haul_tire", (2, 4, 4), p_tire.name, Place("front_axle_spindle", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    rear_wheel = Bone(
        "rear_wheel_l",
        rest_euler=(wheel_spin, 0.0, 0.0),
        attach=Attach("ladder_frame_spine", (1.0, -0.5, 0.65), (-1.0, 0.0, 0.0)),
        mirror_name="rear_wheel_r",
        items=[
            Box("rear_axle_tube", (2, 1, 1), p_frame.name),
            Box("rear_dual_tire", (3, 4, 4), p_tire.name, Place("rear_axle_tube", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("rear_planetary_hub", (1, 2, 2), p_body.name, Place("rear_dual_tire", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), inflate=hub_inf),
        ],
    )

    dump_bed_bone = Bone(
        "tilting_dump_bed",
        rest_euler=(bed_pitch, 0.0, 0.0),
        attach=Attach("rear_hinge_bolster", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0)),
        items=[
            Box("dump_bed_floor", (8, 1, 7), p_body.name),
            Box("bed_sideboard_e", (1, 3, 7), p_body.name, Place("dump_bed_floor", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("bed_sideboard_w", (1, 3, 7), p_body.name, Place("dump_bed_floor", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
            Box("bed_headboard_guard", (4, 3, 1), p_frame.name, Place("dump_bed_floor", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("ladder_frame_spine", (4, 2, chassis_len), p_frame.name),
        Box("front_heavy_bumper", (8, 2, 1), p_frame.name, Place("ladder_frame_spine", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("headlight_cluster_e", (2, 1, 1), p_lamp.name, Place("front_heavy_bumper", (0.75, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("headlight_cluster_w", (2, 1, 1), p_lamp.name, Place("front_heavy_bumper", (-0.75, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("engine_bonnet_hood", (6, 3, 4), p_body.name, Place("ladder_frame_spine", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        Box("radiator_grille_louvers", (4, 2, 1), p_frame.name, Place("engine_bonnet_hood", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("operator_cab_shell", (6, 5, 4), p_body.name, Place("ladder_frame_spine", (0.0, 1.0, -0.15), (0.0, -1.0, 0.0))),
        Box("roof_strobe_beacon", (2, 1, 2), p_lamp.name, Place("operator_cab_shell", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("saddle_fuel_tank", (2, 1, 3), p_frame.name, Place("ladder_frame_spine", (1.0, -0.5, -0.1), (-1.0, 0.0, 0.0))),
        Box("rear_hinge_bolster", (4, 1, 2), p_frame.name, Place("ladder_frame_spine", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        front_wheel,
        rear_wheel,
        dump_bed_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_body, p_frame, p_tire, p_lamp],
        root=Bone("haul_chassis_datum", pivot_local=_jpiv(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": chassis_len + 2,
        "secondary_span": 16,
        "joint_angle_deg": bed_pitch,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": hub_inf,
        "material_theme": f"{p_body.name}+{p_frame.name}+{p_tire.name}",
        "notes": f"{chassis_len}-long offroad haul truck ({substyle}) with front grille/headlights, operator cab + beacon, mirrored front/rear axle wheels, and {bed_pitch:g} deg rear-hinged dump bed",
        "extra": {"chassis_len": chassis_len, "bed_pitch": bed_pitch},
    }


def _gen_tracked_excavator(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("heavy_quarry_excavator", "trenching_backhoe_crawler", "demolition_shovel")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_paint = _pal(rng, _INDUSTRIAL_COATINGS[v_idx % 2])
    p_track = _pal(rng, _DARK_RUBBER_CARBON[1])
    p_steel = _pal(rng, _HEAVY_ALLOYS[1])
    p_glaze = _pal(rng, _OPTIC_GLOW_ACCENTS[1], emissive=(v_idx % 3 == 0))

    turret_yaw = rng.choice((-15.0, -11.25, 0.0, 11.25, 15.0))
    boom_pitch = rng.choice((15.0, 22.5))
    stick_pitch = rng.choice((-35.0, -30.0))
    bucket_curl = rng.choice((-22.5, -15.0))
    boom_len = rng.choice((6, 7))
    stick_len = rng.choice((5, 6))
    cw_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    crawler_track = Bone(
        "crawler_track_pod_l",
        attach=Attach("carbody_h_frame", (1.0, -0.2, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="crawler_track_pod_r",
        items=[
            Box("track_roller_frame", (3, 3, 10), p_track.name),
            Box("front_idler_wheel", (2, 2, 2), p_steel.name, Place("track_roller_frame", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("rear_drive_sprocket", (2, 2, 2), p_steel.name, Place("track_roller_frame", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    bucket_bone = Bone(
        "toothed_excavator_bucket",
        rest_euler=(bucket_curl, 0.0, 0.0),
        attach=Attach("dipper_stick_spar", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
        items=[
            Box("bucket_scoop_body", (4, 3, 3), p_track.name),
            Box("ripper_tooth_e", (1, 1, 2), p_steel.name, Place("bucket_scoop_body", (0.7, -1.0, -1.0), (0.0, -1.0, 1.0))),
            Box("ripper_tooth_w", (1, 1, 2), p_steel.name, Place("bucket_scoop_body", (-0.7, -1.0, -1.0), (0.0, -1.0, 1.0))),
        ],
    )

    stick_bone = Bone(
        "dipper_stick_stage",
        rest_euler=(stick_pitch, 0.0, 0.0),
        attach=Attach("boom_apex_knuckle", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
        items=[
            Box("dipper_stick_spar", (2, 2, stick_len), p_paint.name),
            Box("bucket_ram_cylinder", (1, 1, stick_len - 2), p_steel.name, Place("dipper_stick_spar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            bucket_bone,
        ],
    )

    boom_bone = Bone(
        "main_boom_stage",
        rest_euler=(boom_pitch, 0.0, 0.0),
        attach=Attach("boom_foot_pedestal", (0.0, 1.0, -1.0), (0.0, -1.0, 1.0)),
        items=[
            Box("main_boom_beam", (2, 2, boom_len), p_paint.name),
            Box("hoist_hydraulic_ram", (1, 1, boom_len - 2), p_steel.name, Place("main_boom_beam", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("boom_apex_knuckle", (2, 2, 2), p_steel.name, Place("main_boom_beam", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            stick_bone,
        ],
    )

    turret_bone = Bone(
        "slewing_house_turret",
        rest_euler=(0.0, turret_yaw, 0.0),
        attach=Attach("slew_ring_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("turret_deck_bed", (8, 2, 9), p_paint.name),
            Box("cast_counterweight", (8, 4, 3), p_track.name, Place("turret_deck_bed", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0)), inflate=cw_inf),
            Box("operator_cab_pod", (2, 5, 4), p_glaze.name, Place("turret_deck_bed", (-1.0, 1.0, -0.9), (-1.0, -1.0, -1.0))),
            Box("cab_work_lamp", (1, 1, 1), p_steel.name, Place("operator_cab_pod", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0))),
            Box("hydraulic_pump_bay", (2, 3, 4), p_paint.name, Place("turret_deck_bed", (1.0, 1.0, -0.9), (1.0, -1.0, -1.0))),
            Box("diesel_muffler_pipe", (1, 2, 1), p_track.name, Place("hydraulic_pump_bay", (0.0, 1.0, 0.4), (0.0, -1.0, 0.0))),
            Box("boom_foot_pedestal", (2, 2, 2), p_steel.name, Place("turret_deck_bed", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            boom_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("carbody_h_frame", (6, 2, 8), p_track.name),
        Box("slew_ring_collar", (4, 2, 4), p_steel.name, Place("carbody_h_frame", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        crawler_track,
        turret_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_paint, p_track, p_steel, p_glaze],
        root=Bone("crawler_carbody_datum", pivot_local=_jpiv(rng, (8.0, 4.0, 11.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 10 + boom_len + stick_len,
        "secondary_span": 12,
        "joint_angle_deg": turret_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": cw_inf,
        "material_theme": f"{p_paint.name}+{p_track.name}+{p_steel.name}",
        "notes": f"crawler excavator ({substyle}) with mirrored track pods, {turret_yaw:g} deg slewing turret, operator cab, counterweight, and 3-bone boom-stick-bucket arm",
        "extra": {"turret_yaw": turret_yaw, "boom_len": boom_len, "stick_len": stick_len},
    }


def _gen_steam_rail_locomotive(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("pacific_steam_engine", "geared_logging_locomotive", "armored_switch_engine")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_boiler = _pal(rng, _INDUSTRIAL_COATINGS[6])
    p_trim = _pal(rng, _INDUSTRIAL_COATINGS[4], suffix="_running")
    p_brass = _pal(rng, _BRASS_COPPER_TRIM[0])
    p_lamp = _pal(rng, _OPTIC_GLOW_ACCENTS[0], emissive=True)

    chassis_len = rng.choice((18, 20))
    pilot_pitch = rng.choice((-22.5, -18.0))
    whistle_pitch = rng.choice((-11.25, 0.0, 11.25))
    lamp_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    cowcatcher_bone = Bone(
        "cowcatcher_pilot_rig",
        rest_euler=(pilot_pitch, 0.0, 0.0),
        attach=Attach("front_buffer_beam", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
        items=[
            Box("pilot_plow_frame", (6, 2, 3), p_trim.name),
            Box("pilot_deflector_toe", (4, 1, 2), p_trim.name, Place("pilot_plow_frame", (0.0, -1.0, -1.0), (0.0, -1.0, 1.0))),
        ],
    )

    drive_wheel_bone = Bone(
        "drive_wheel_gear_l",
        attach=Attach("rail_chassis_bed", (1.0, -0.5, -0.2), (-1.0, 0.0, 0.0)),
        mirror_name="drive_wheel_gear_r",
        items=[
            Box("journal_equalizer_bar", (1, 1, 10), p_trim.name),
            Box("fore_driver_wheel", (1, 4, 4), p_trim.name, Place("journal_equalizer_bar", (1.0, 0.0, -0.65), (-1.0, 0.0, 0.0))),
            Box("aft_driver_wheel", (1, 4, 4), p_trim.name, Place("journal_equalizer_bar", (1.0, 0.0, 0.65), (-1.0, 0.0, 0.0))),
            Box("side_coupling_rod", (1, 1, 7), p_brass.name, Place("fore_driver_wheel", (1.0, -0.2, 1.0), (-1.0, 0.0, -0.4))),
        ],
    )

    whistle_bone = Bone(
        "steam_whistle_valve",
        rest_euler=(whistle_pitch, 0.0, 0.0),
        attach=Attach("cab_clerestory_roof", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0)),
        items=[
            Box("brass_chime_whistle", (1, 2, 1), p_brass.name),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("rail_chassis_bed", (4, 2, chassis_len), p_trim.name),
        Box("front_buffer_beam", (8, 2, 1), p_trim.name, Place("rail_chassis_bed", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("boiler_saddle_block", (4, 1, 7), p_boiler.name, Place("rail_chassis_bed", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0))),
        Box("cylindrical_boiler_barrel", (5, 5, 7), p_boiler.name, Place("boiler_saddle_block", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("smokebox_front_shell", (5, 5, 3), p_boiler.name, Place("cylindrical_boiler_barrel", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("brass_arc_headlamp", (2, 2, 1), p_lamp.name, Place("smokebox_front_shell", (0.0, 0.3, -1.0), (0.0, 0.0, 1.0)), inflate=lamp_inf),
        Box("smokestack_funnel", (2, 4, 2), p_boiler.name, Place("smokebox_front_shell", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("steam_regulator_dome", (3, 2, 2), p_brass.name, Place("cylindrical_boiler_barrel", (0.0, 1.0, -0.45), (0.0, -1.0, 0.0))),
        Box("sand_traction_dome", (3, 2, 2), p_brass.name, Place("cylindrical_boiler_barrel", (0.0, 1.0, 0.45), (0.0, -1.0, 0.0))),
        Box("engineer_cab_house", (5, 6, 4), p_boiler.name, Place("rail_chassis_bed", (0.0, 1.0, 0.42), (0.0, -1.0, 0.0))),
        Box("cab_clerestory_roof", (6, 1, 5), p_trim.name, Place("engineer_cab_house", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("coal_tender_bunker", (5, 4, 3), p_boiler.name, Place("rail_chassis_bed", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("rear_drawbar_coupler", (2, 1, 2), p_trim.name, Place("rail_chassis_bed", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        cowcatcher_bone,
        drive_wheel_bone,
        whistle_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_boiler, p_trim, p_brass, p_lamp],
        root=Bone("locomotive_frame_datum", pivot_local=_jpiv(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": chassis_len + 5,
        "secondary_span": 10,
        "joint_angle_deg": pilot_pitch,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": lamp_inf,
        "material_theme": f"{p_boiler.name}+{p_trim.name}+{p_brass.name}",
        "notes": f"{chassis_len}-long steam rail locomotive ({substyle}) with boiler, smokebox headlamp, smokestack, steam/sand domes, engineer cab, coal bunker, {pilot_pitch:g} deg cowcatcher, and mirrored drive wheels + coupling rods",
        "extra": {"chassis_len": chassis_len, "pilot_pitch": pilot_pitch},
    }


def _gen_twin_rotor_helicopter_drone(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("utility_rescue_helicopter", "coaxial_scout_rotorcraft", "heavy_skycrane_drone")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_airframe = _pal(rng, _INDUSTRIAL_COATINGS[(v_idx + 1) % 3])
    p_rotor = _pal(rng, _DARK_RUBBER_CARBON[2])
    p_alloy = _pal(rng, _HEAVY_ALLOYS[3])
    p_canopy = _pal(rng, _OPTIC_GLOW_ACCENTS[1], emissive=(v_idx % 2 == 0))

    blade_len = rng.choice((6, 7))
    rotor_yaw = rng.choice((15.0, 22.5, 30.0, 45.0))
    tail_spin = rng.choice((0.0, 22.5, 45.0))
    skid_splay = rng.choice((0.0, 7.5, 11.25))
    gimbal_yaw = rng.choice((-22.5, 0.0, 22.5))
    sensor_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    nose_gimbal_bone = Bone(
        "nose_flir_gimbal",
        rest_euler=(0.0, gimbal_yaw, 0.0),
        attach=Attach("cockpit_canopy_glaze", (0.0, -1.0, -0.4), (0.0, 1.0, 0.0)),
        items=[
            Box("electro_optic_turret", (2, 2, 2), p_alloy.name, inflate=sensor_inf),
        ],
    )

    landing_skid_bone = Bone(
        "landing_skid_l",
        rest_euler=(0.0, 0.0, skid_splay),
        attach=Attach("fuselage_cabin_pod", (0.8, -1.0, 0.0), (0.0, 1.0, 0.0)),
        mirror_name="landing_skid_r",
        items=[
            Box("skid_oleo_strut", (1, 3, 2), p_alloy.name),
            Box("tubular_skid_rail", (1, 1, 9), p_rotor.name, Place("skid_oleo_strut", (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0))),
            Box("skid_fore_horn", (1, 1, 2), p_rotor.name, Place("tubular_skid_rail", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        ],
    )

    main_rotor_bone = Bone(
        "main_rotor_hub_rig",
        rest_euler=(0.0, rotor_yaw, 0.0),
        attach=Attach("main_rotor_mast", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("swashplate_hub_cap", (2, 1, 2), p_alloy.name),
            Box("rotor_blade_fore", (1, 1, blade_len), p_rotor.name, Place("swashplate_hub_cap", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("rotor_blade_aft", (1, 1, blade_len), p_rotor.name, Place("swashplate_hub_cap", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("rotor_blade_port", (blade_len, 1, 1), p_rotor.name, Place("swashplate_hub_cap", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("rotor_blade_stbd", (blade_len, 1, 1), p_rotor.name, Place("swashplate_hub_cap", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        ],
    )

    tail_rotor_bone = Bone(
        "tail_anti_torque_rotor",
        rest_euler=(tail_spin, 0.0, 0.0),
        attach=Attach("vertical_stabilizer_fin", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
        items=[
            Box("tail_gearbox_hub", (2, 1, 1), p_alloy.name),
            Box("tail_blade_upper", (1, 2, 1), p_rotor.name, Place("tail_gearbox_hub", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("tail_blade_lower", (1, 2, 1), p_rotor.name, Place("tail_gearbox_hub", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("fuselage_cabin_pod", (6, 4, 8), p_airframe.name),
        Box("cockpit_canopy_glaze", (4, 3, 3), p_canopy.name, Place("fuselage_cabin_pod", (0.0, -0.1, -1.0), (0.0, 0.0, 1.0))),
        Box("turboshaft_engine_deck", (4, 2, 5), p_alloy.name, Place("fuselage_cabin_pod", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("main_rotor_mast", (2, 3, 2), p_alloy.name, Place("turboshaft_engine_deck", (0.0, 1.0, -0.2), (0.0, -1.0, 0.0))),
        Box("tail_boom_spar", (2, 2, 8), p_airframe.name, Place("fuselage_cabin_pod", (0.0, 0.5, 1.0), (0.0, 0.0, -1.0))),
        Box("horiz_stabilizer_e", (2, 1, 2), p_airframe.name, Place("tail_boom_spar", (1.0, 0.0, 0.4), (-1.0, 0.0, 0.0))),
        Box("horiz_stabilizer_w", (2, 1, 2), p_airframe.name, Place("tail_boom_spar", (-1.0, 0.0, 0.4), (1.0, 0.0, 0.0))),
        Box("vertical_stabilizer_fin", (1, 4, 3), p_airframe.name, Place("tail_boom_spar", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        nose_gimbal_bone,
        landing_skid_bone,
        main_rotor_bone,
        tail_rotor_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_airframe, p_rotor, p_alloy, p_canopy],
        root=Bone("rotorcraft_airframe_datum", pivot_local=_jpiv(rng, (8.0, 8.0, 6.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 19,
        "secondary_span": 2 + 2 * blade_len,
        "joint_angle_deg": rotor_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": sensor_inf,
        "material_theme": f"{p_airframe.name}+{p_rotor.name}+{p_canopy.name}",
        "notes": f"rotary-wing helicopter/drone ({substyle}) with glazed canopy, FLIR nose gimbal, mirrored landing skids, tail boom + spinning tail rotor, and {rotor_yaw:g} deg 4-blade main rotor",
        "extra": {"blade_len": blade_len, "rotor_yaw": rotor_yaw, "skid_splay": skid_splay},
    }


def _gen_cargo_tugboat_vessel(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("harbor_pusher_tug", "icebreaker_workboat", "offshore_supply_tug")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_hull = _pal(rng, _INDUSTRIAL_COATINGS[4])
    p_house = _pal(rng, _INDUSTRIAL_COATINGS[7])
    p_fender = _pal(rng, _DARK_RUBBER_CARBON[0])
    p_brass = _pal(rng, _BRASS_COPPER_TRIM[2])

    hull_len = rng.choice((16, 18))
    hull_w = rng.choice((8, 10))
    rudder_yaw = rng.choice((-22.5, -15.0, 15.0, 22.5))
    prop_roll = rng.choice((0.0, 22.5, 45.0))
    radar_yaw = rng.choice((-30.0, -15.0, 15.0, 30.0))
    fender_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    radar_bone = Bone(
        "mast_radar_scanner",
        rest_euler=(0.0, radar_yaw, 0.0),
        attach=Attach("navigation_mast_spar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("radar_waveguide_bar", (4, 1, 1), p_house.name),
        ],
    )

    funnel_bone = Bone(
        "exhaust_funnel_l",
        rest_euler=(11.25, 0.0, 0.0),
        attach=Attach("main_deckhouse_cabin", (0.6, 1.0, 0.6), (0.0, -1.0, 0.0)),
        mirror_name="exhaust_funnel_r",
        items=[
            Box("funnel_casing_stack", (1, 4, 2), p_hull.name),
            Box("funnel_spark_cowl", (1, 1, 2), p_fender.name, Place("funnel_casing_stack", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    propeller_bone = Bone(
        "stern_propeller_hub",
        rest_euler=(0.0, 0.0, prop_roll),
        attach=Attach("keel_skeg_timber", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tailshaft_stern_tube", (1, 1, 3), p_brass.name),
            Box("prop_boss_cone", (1, 1, 1), p_brass.name, Place("tailshaft_stern_tube", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("prop_blade_up", (1, 1, 1), p_brass.name, Place("prop_boss_cone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("prop_blade_dn", (1, 1, 1), p_brass.name, Place("prop_boss_cone", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("prop_blade_port", (1, 1, 1), p_brass.name, Place("prop_boss_cone", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("prop_blade_stbd", (1, 1, 1), p_brass.name, Place("prop_boss_cone", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        ],
    )

    rudder_bone = Bone(
        "stern_rudder_rig",
        rest_euler=(0.0, rudder_yaw, 0.0),
        attach=Attach("stern_transom_counter", (0.0, -1.0, 0.75), (0.0, 1.0, 0.0)),
        items=[
            Box("rudder_pintle_stock", (1, 1, 1), p_hull.name),
            Box("rudder_hydrofoil_blade", (1, 3, 1), p_hull.name, Place("rudder_pintle_stock", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("displacement_hull_mid", (hull_w, 3, hull_len - 4), p_hull.name),
        Box("keel_skeg_timber", (2, 2, hull_len - 6), p_fender.name, Place("displacement_hull_mid", (0.0, -1.0, -0.2), (0.0, 1.0, 0.0))),
        Box("bow_forefoot_hull", (hull_w - 2, 3, 3), p_hull.name, Place("displacement_hull_mid", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("raised_bow_bulwark", (hull_w - 2, 1, 3), p_hull.name, Place("bow_forefoot_hull", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("bow_push_fender", (hull_w - 2, 2, 2), p_fender.name, Place("bow_forefoot_hull", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), inflate=fender_inf),
        Box("stern_transom_counter", (hull_w - 2, 2, 3), p_hull.name, Place("displacement_hull_mid", (0.0, 1.0, 1.0), (0.0, 1.0, -1.0))),
        Box("aft_towing_bitt", (2, 2, 2), p_fender.name, Place("displacement_hull_mid", (0.0, 1.0, 0.75), (0.0, -1.0, 0.0))),
        Box("main_deckhouse_cabin", (hull_w - 2, 3, 7), p_house.name, Place("displacement_hull_mid", (0.0, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("pilothouse_bridge", (hull_w - 4, 3, 3), p_house.name, Place("main_deckhouse_cabin", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0))),
        Box("bridge_window_visor", (hull_w - 4, 1, 1), p_fender.name, Place("pilothouse_bridge", (0.0, 0.5, -1.0), (0.0, 0.0, 1.0))),
        Box("navigation_mast_spar", (1, 3, 1), p_brass.name, Place("pilothouse_bridge", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        radar_bone,
        funnel_bone,
        propeller_bone,
        rudder_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_hull, p_house, p_fender, p_brass],
        root=Bone("tugboat_keel_datum", pivot_local=_jpiv(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": hull_len + 4,
        "secondary_span": hull_w,
        "joint_angle_deg": rudder_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": fender_inf,
        "material_theme": f"{p_hull.name}+{p_house.name}+{p_fender.name}",
        "notes": f"{hull_len}-long harbor tugboat ({substyle}) with bow push-fender, deckhouse + glazed pilothouse, radar mast, mirrored raked exhaust funnels, 4-blade propeller, and {rudder_yaw:g} deg rudder",
        "extra": {"hull_len": hull_len, "hull_w": hull_w, "rudder_yaw": rudder_yaw},
    }


def _gen_articulated_forklift_loader(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Held-Out Vehicle (>= 18 boxes across ALL variants)."""
    substyles = ("counterbalance_warehouse_forklift", "heavy_yard_mast_loader", "tele_mast_pallet_lifter")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_body = _pal(rng, _INDUSTRIAL_COATINGS[0])
    p_mast = _pal(rng, _HEAVY_ALLOYS[0])
    p_tire = _pal(rng, _DARK_RUBBER_CARBON[0])
    p_strobe = _pal(rng, _OPTIC_GLOW_ACCENTS[0], emissive=True)

    mast_h = rng.choice((11, 13))
    mast_tilt = rng.choice((-5.0, 0.0, 5.0))
    steer_yaw = rng.choice((-15.0, 0.0, 15.0))
    lift_frac = rng.choice((-0.3, 0.0, 0.35))
    fork_len = rng.choice((5, 6))
    cw_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    rear_wheel_connected = Bone(
        "rear_steer_wheel_l",
        rest_euler=(0.0, steer_yaw, 0.0),
        attach=Attach("rear_steer_bolster", (1.0, -0.4, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="rear_steer_wheel_r",
        items=[
            Box("rear_steer_tire", (2, 3, 3), p_tire.name),
        ],
    )
    front_wheel_compact = Bone(
        "front_drive_wheel_l",
        attach=Attach("chassis_tub_frame", (1.0, -0.4, -0.55), (-1.0, 0.0, 0.0)),
        mirror_name="front_drive_wheel_r",
        items=[
            Box("front_drive_spindle", (1, 1, 1), p_mast.name),
            Box("front_load_tire", (2, 4, 4), p_tire.name, Place("front_drive_spindle", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )
    fork_carriage_compact = Bone(
        "fork_carriage_rig",
        attach=Attach("lift_hydraulic_cylinder", (0.0, lift_frac, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("carriage_backrest_plate", (6, 3, 1), p_mast.name),
            Box("fork_tine_blade_e", (1, 1, fork_len), p_mast.name, Place("carriage_backrest_plate", (0.7, -1.0, -1.0), (0.0, -1.0, 1.0))),
            Box("fork_tine_blade_w", (1, 1, fork_len), p_mast.name, Place("carriage_backrest_plate", (-0.7, -1.0, -1.0), (0.0, -1.0, 1.0))),
        ],
    )
    mast_bone_compact = Bone(
        "tilting_vertical_mast",
        rest_euler=(mast_tilt, 0.0, 0.0),
        attach=Attach("front_mast_trunnion", (0.0, 1.0, -1.0), (0.0, -1.0, 1.0)),
        items=[
            Box("mast_lower_crossbrace", (6, 2, 2), p_mast.name),
            Box("mast_channel_rail_e", (1, mast_h, 2), p_mast.name, Place("mast_lower_crossbrace", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("mast_channel_rail_w", (1, mast_h, 2), p_mast.name, Place("mast_lower_crossbrace", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
            Box("lift_hydraulic_cylinder", (2, mast_h - 1, 2), p_body.name, Place("mast_lower_crossbrace", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            fork_carriage_compact,
        ],
    )
    root_items: list[Box | Bone] = [
        Box("chassis_tub_frame", (6, 2, 7), p_body.name),
        Box("rear_steer_bolster", (4, 2, 4), p_mast.name, Place("chassis_tub_frame", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("counterweight_tail_block", (6, 4, 2), p_body.name, Place("rear_steer_bolster", (0.0, 0.5, 1.0), (0.0, -0.25, -1.0)), inflate=cw_inf),
        Box("operator_suspension_seat", (2, 2, 2), p_tire.name, Place("chassis_tub_frame", (0.0, 1.0, 0.35), (0.0, -1.0, 0.0))),
        Box("steering_dash_column", (2, 3, 1), p_mast.name, Place("chassis_tub_frame", (0.0, 1.0, -0.55), (0.0, -1.0, 0.0))),
        Box("rops_pillar_fl", (1, 6, 1), p_mast.name, Place("chassis_tub_frame", (1.0, 1.0, -0.8), (1.0, -1.0, -1.0))),
        Box("rops_pillar_fr", (1, 6, 1), p_mast.name, Place("chassis_tub_frame", (-1.0, 1.0, -0.8), (-1.0, -1.0, -1.0))),
        Box("rops_pillar_rl", (1, 6, 1), p_mast.name, Place("chassis_tub_frame", (1.0, 1.0, 0.9), (1.0, -1.0, 1.0))),
        Box("rops_pillar_rr", (1, 6, 1), p_mast.name, Place("chassis_tub_frame", (-1.0, 1.0, 0.9), (-1.0, -1.0, 1.0))),
        Box("rops_overhead_guard", (6, 1, 6), p_mast.name, Place("rops_pillar_fl", (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))),
        Box("rops_safety_strobe", (2, 1, 2), p_strobe.name, Place("rops_overhead_guard", (0.0, 1.0, 0.4), (0.0, -1.0, 0.0))),
        Box("front_mast_trunnion", (4, 2, 2), p_mast.name, Place("chassis_tub_frame", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        front_wheel_compact,
        rear_wheel_connected,
        mast_bone_compact,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_body, p_mast, p_tire, p_strobe],
        root=Bone("forklift_chassis_datum", pivot_local=_jpiv(rng, (8.0, 5.0, 9.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 13 + fork_len,
        "secondary_span": mast_h + 4,
        "joint_angle_deg": steer_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": cw_inf,
        "material_theme": f"{p_body.name}+{p_mast.name}+{p_tire.name}",
        "notes": f"counterbalance mast forklift ({substyle}) with 4-post ROPS cage, rear counterweight, steerable {steer_yaw:g} deg rear wheels, {mast_tilt:g} deg tilting {mast_h}-tall twin-rail mast, and fork carriage",
        "extra": {"mast_h": mast_h, "fork_len": fork_len, "lift_frac": lift_frac},
    }


def _gen_submersible_rov_craft(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Held-Out Vehicle (>= 18 boxes across ALL variants)."""
    substyles = ("abyssal_science_rov", "offshore_inspection_rov", "trench_coring_submersible")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_foam = _pal(rng, _INDUSTRIAL_COATINGS[5])
    p_frame = _pal(rng, _DARK_RUBBER_CARBON[2])
    p_titanium = _pal(rng, _HEAVY_ALLOYS[1])
    p_dome = _pal(rng, _OPTIC_GLOW_ACCENTS[1], emissive=True)

    rov_len = rng.choice((10, 12))
    rov_w = rng.choice((8, 10))
    thruster_yaw = rng.choice((-22.5, -15.0, 15.0, 22.5))
    arm_pitch = rng.choice((-11.25, 0.0, 7.5))
    arm_yaw = rng.choice((-15.0, -7.5))
    claw_pitch = rng.choice((7.5, 15.0))
    dome_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    thruster_bone = Bone(
        "vectored_thruster_l",
        rest_euler=(0.0, thruster_yaw, 0.0),
        attach=Attach("syntactic_buoyancy_pack", (1.0, -0.4, 0.4), (-1.0, 0.0, 0.0)),
        mirror_name="vectored_thruster_r",
        items=[
            Box("thruster_outboard_pylon", (2, 1, 1), p_titanium.name),
            Box("kort_thruster_duct", (2, 2, 3), p_frame.name, Place("thruster_outboard_pylon", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    claw_wrist_bone = Bone(
        "manipulator_claw_stage",
        rest_euler=(claw_pitch, 7.5, 0.0),
        attach=Attach("upper_manip_boom", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("forearm_servo_tube", (1, 1, 3), p_titanium.name),
            Box("claw_wrist_yoke", (3, 1, 1), p_frame.name, Place("forearm_servo_tube", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("claw_pincer_outer", (1, 1, 2), p_titanium.name, Place("claw_wrist_yoke", (1.0, 0.0, -1.0), (1.0, 0.0, 1.0))),
            Box("claw_pincer_inner", (1, 1, 2), p_titanium.name, Place("claw_wrist_yoke", (-1.0, 0.0, -1.0), (-1.0, 0.0, 1.0))),
        ],
    )

    manipulator_bone = Bone(
        "manipulator_arm_l",
        rest_euler=(arm_pitch, arm_yaw, 0.0),
        attach=Attach("manipulator_mount_bar", (0.85, 0.0, -1.0), (0.0, 0.0, 1.0)),
        mirror_name="manipulator_arm_r",
        items=[
            Box("upper_manip_boom", (1, 1, 4), p_titanium.name),
            claw_wrist_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("electronics_pressure_hull", (4, 3, rov_len - 2), p_titanium.name),
        Box("syntactic_buoyancy_pack", (rov_w, 3, rov_len), p_foam.name, Place("electronics_pressure_hull", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("doppler_sonar_turret", (2, 2, 2), p_titanium.name, Place("syntactic_buoyancy_pack", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0))),
        Box("umbilical_tether_bell", (2, 2, 2), p_frame.name, Place("syntactic_buoyancy_pack", (0.0, 1.0, 0.5), (0.0, -1.0, 0.0))),
        Box("subsea_led_lightbar", (rov_w - 2, 1, 1), p_dome.name, Place("syntactic_buoyancy_pack", (0.0, 0.5, -1.0), (0.0, 0.0, 1.0))),
        Box("acrylic_viewport_dome", (3, 2, 2), p_dome.name, Place("electronics_pressure_hull", (0.0, -0.15, -1.0), (0.0, 0.0, 1.0)), inflate=dome_inf),
        Box("tubular_crash_frame_e", (1, 3, rov_len), p_frame.name, Place("syntactic_buoyancy_pack", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))),
        Box("tubular_crash_frame_w", (1, 3, rov_len), p_frame.name, Place("syntactic_buoyancy_pack", (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0))),
        Box("manipulator_mount_bar", (5, 1, 2), p_frame.name, Place("electronics_pressure_hull", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0))),
        Box("benthic_sample_basket", (4, 1, 3), p_frame.name, Place("electronics_pressure_hull", (0.0, -1.0, 0.45), (0.0, 1.0, 0.0))),
        thruster_bone,
        manipulator_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_foam, p_frame, p_titanium, p_dome],
        root=Bone("rov_pressure_datum", pivot_local=_jpiv(rng, (8.0, 8.0, 10.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": rov_len + 9,
        "secondary_span": rov_w + 8,
        "joint_angle_deg": thruster_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": dome_inf,
        "material_theme": f"{p_foam.name}+{p_titanium.name}+{p_dome.name}",
        "notes": f"deep-sea ROV ({substyle}) with syntactic-foam buoyancy pack, acrylic viewport dome, LED light bar, mirrored {thruster_yaw:g} deg vectored thruster ducts, and twin 2-stage manipulator claws",
        "extra": {"rov_len": rov_len, "rov_w": rov_w, "thruster_yaw": thruster_yaw},
    }


# ==============================================================================
# C. CRAFT / MECHANICAL PROPS (2 Seen, >= 16 boxes, >= 4 bones)
# ==============================================================================


def _gen_clockwork_pendulum_clock(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("mahogany_grandfather_clock", "brass_regulator_horologe", "astronomical_longcase_clock")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_wood = _pal(rng, _CABINET_TIMBERS[v_idx % len(_CABINET_TIMBERS)])
    p_trim = _pal(rng, _CABINET_TIMBERS[(v_idx + 1) % len(_CABINET_TIMBERS)], suffix="_molding")
    p_brass = _pal(rng, _BRASS_COPPER_TRIM[0])
    p_dial = _pal(rng, _OPTIC_GLOW_ACCENTS[4])

    waist_h = rng.choice((11, 12))
    pend_deg = rng.choice((-8.0, -6.0, 6.0, 8.0))
    minute_deg = rng.choice((-45.0, -22.5, 30.0, 60.0))
    hour_deg = rng.choice((60.0, 90.0, 120.0))
    bob_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    pendulum_bone = Bone(
        "swinging_pendulum_rig",
        rest_euler=(0.0, 0.0, pend_deg),
        attach=Attach("hood_cornice_ledge", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("pendulum_suspension_rod", (1, waist_h - 4, 1), p_brass.name),
            Box("lenticular_brass_bob", (2, 2, 1), p_brass.name, Place("pendulum_suspension_rod", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)), inflate=bob_inf),
        ],
    )

    hour_hand_bone = Bone(
        "hour_hand_hub",
        rest_euler=(0.0, 0.0, hour_deg),
        attach=Attach("arbor_center_pin", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("hour_pipe_collar", (1, 1, 1), p_brass.name),
            Box("hour_spade_pointer", (1, 1, 1), p_trim.name, Place("hour_pipe_collar", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    minute_hand_bone = Bone(
        "minute_hand_arbor",
        rest_euler=(0.0, 0.0, minute_deg),
        attach=Attach("enamel_chapter_dial", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("arbor_center_pin", (1, 1, 1), p_brass.name),
            Box("minute_lance_pointer", (1, 2, 1), p_trim.name, Place("arbor_center_pin", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            hour_hand_bone,
        ],
    )

    winding_key_bone = Bone(
        "side_chime_winding_crank",
        rest_euler=(rng.choice((0.0, 22.5, 45.0)), 0.0, 0.0),
        attach=Attach("dial_hood_cabinet", (1.0, -0.2, 0.0), (-1.0, 0.0, 0.0)),
        items=[
            Box("winding_key_arbor", (1, 1, 2), p_brass.name),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("molded_base_plinth", (10, 3, 6), p_wood.name),
        Box("plinth_bracket_foot_e", (2, 1, 6), p_trim.name, Place("molded_base_plinth", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))),
        Box("plinth_bracket_foot_w", (2, 1, 6), p_trim.name, Place("molded_base_plinth", (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0))),
        Box("waist_lower_sill", (8, 1, 5), p_trim.name, Place("molded_base_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("waist_acoustic_backboard", (8, waist_h, 1), p_wood.name, Place("waist_lower_sill", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("open_waist_pilaster_e", (1, waist_h, 3), p_wood.name, Place("waist_lower_sill", (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))),
        Box("open_waist_pilaster_w", (1, waist_h, 3), p_wood.name, Place("waist_lower_sill", (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))),
        Box("hood_cornice_ledge", (10, 1, 6), p_trim.name, Place("waist_acoustic_backboard", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("dial_hood_cabinet", (8, 6, 5), p_wood.name, Place("hood_cornice_ledge", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("swan_neck_pediment", (10, 1, 6), p_trim.name, Place("dial_hood_cabinet", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("central_spire_finial", (2, 2, 2), p_brass.name, Place("swan_neck_pediment", (0.0, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("corner_urn_finial_e", (1, 2, 1), p_brass.name, Place("swan_neck_pediment", (0.85, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("corner_urn_finial_w", (1, 2, 1), p_brass.name, Place("swan_neck_pediment", (-0.85, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("enamel_chapter_dial", (6, 4, 1), p_dial.name, Place("dial_hood_cabinet", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        pendulum_bone,
        minute_hand_bone,
        winding_key_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_wood, p_trim, p_brass, p_dial],
        root=Bone("longcase_plinth_datum", pivot_local=_jpiv(rng, (8.0, 3.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": waist_h + 15,
        "secondary_span": 10,
        "joint_angle_deg": pend_deg,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": bob_inf,
        "material_theme": f"{p_wood.name}+{p_brass.name}+{p_dial.name}",
        "notes": f"tall grandfather clock ({substyle}) with bracketed base plinth, recessed open-pilaster waist housing a {pend_deg:g} deg swinging brass pendulum, finialed dial hood, and articulated hour/minute hands",
        "extra": {"waist_h": waist_h, "pend_deg": pend_deg, "minute_deg": minute_deg},
    }


def _gen_alchemical_workbench_apparatus(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("alembic_distillation_bench", "apothecary_assay_table", "metallurgist_retort_station")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_timber = _pal(rng, _CABINET_TIMBERS[1])
    p_copper = _pal(rng, _BRASS_COPPER_TRIM[1])
    p_iron = _pal(rng, _HEAVY_ALLOYS[0])
    p_glass = _pal(rng, _OPTIC_GLOW_ACCENTS[3], emissive=(v_idx % 2 == 0))

    table_w = rng.choice((16, 18))
    beam_tilt = rng.choice((-11.25, -7.5, 7.5, 11.25))
    condenser_drop = rng.choice((-22.5, -18.0))
    flask_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    condenser_bone = Bone(
        "overhead_condenser_bridge",
        rest_euler=(0.0, 0.0, condenser_drop),
        attach=Attach("alembic_swan_riser", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
        items=[
            Box("liebig_condenser_tube", (4, 1, 1), p_copper.name),
        ],
    )

    scale_pan_e = Bone(
        "hanging_scale_pan_e",
        rest_euler=(0.0, 0.0, -beam_tilt),
        attach=Attach("precision_balance_beam", (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("pan_stirrup_chain_e", (1, 3, 1), p_copper.name),
            Box("assay_weigh_pan_e", (2, 1, 2), p_copper.name, Place("pan_stirrup_chain_e", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    scale_pan_w = Bone(
        "hanging_scale_pan_w",
        rest_euler=(0.0, 0.0, -beam_tilt),
        attach=Attach("precision_balance_beam", (-1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("pan_stirrup_chain_w", (1, 3, 1), p_copper.name),
            Box("assay_weigh_pan_w", (2, 1, 2), p_copper.name, Place("pan_stirrup_chain_w", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    balance_beam_bone = Bone(
        "assay_balance_rig",
        rest_euler=(0.0, 0.0, beam_tilt),
        attach=Attach("balance_fulcrum_pillar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("precision_balance_beam", (6, 1, 1), p_copper.name),
            scale_pan_e,
            scale_pan_w,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("timber_workbench_slab", (table_w, 2, 10), p_timber.name),
        Box("trestle_leg_fl", (2, 6, 2), p_timber.name, Place("timber_workbench_slab", (0.85, -1.0, -0.8), (0.0, 1.0, 0.0))),
        Box("trestle_leg_fr", (2, 6, 2), p_timber.name, Place("timber_workbench_slab", (-0.85, -1.0, -0.8), (0.0, 1.0, 0.0))),
        Box("trestle_leg_rl", (2, 6, 2), p_timber.name, Place("timber_workbench_slab", (0.85, -1.0, 0.8), (0.0, 1.0, 0.0))),
        Box("trestle_leg_rr", (2, 6, 2), p_timber.name, Place("timber_workbench_slab", (-0.85, -1.0, 0.8), (0.0, 1.0, 0.0))),
        Box("apron_drawer_chest", (6, 2, 6), p_timber.name, Place("timber_workbench_slab", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("shelf_stanchion_e", (1, 5, 2), p_timber.name, Place("timber_workbench_slab", (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))),
        Box("shelf_stanchion_w", (1, 5, 2), p_timber.name, Place("timber_workbench_slab", (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0))),
        Box("reagent_gallery_shelf", (table_w, 1, 3), p_timber.name, Place("shelf_stanchion_e", (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))),
        Box("reagent_bottle_west", (2, 2, 2), p_glass.name, Place("reagent_gallery_shelf", (-0.4, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("reagent_tincture_east", (2, 3, 2), p_glass.name, Place("reagent_gallery_shelf", (0.4, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("brazier_furnace_tripod", (3, 2, 3), p_iron.name, Place("timber_workbench_slab", (-0.55, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("retort_boiling_vessel", (3, 3, 3), p_copper.name, Place("brazier_furnace_tripod", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("alembic_swan_riser", (1, 2, 1), p_copper.name, Place("retort_boiling_vessel", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("receiving_distillate_carafe", (2, 3, 2), p_glass.name, Place("timber_workbench_slab", (-0.12, 1.0, -0.3), (0.0, -1.0, 0.0)), inflate=flask_inf),
        Box("assay_pedestal_foot", (2, 1, 2), p_iron.name, Place("timber_workbench_slab", (0.55, 1.0, -0.3), (0.0, -1.0, 0.0))),
        Box("balance_fulcrum_pillar", (1, 6, 1), p_copper.name, Place("assay_pedestal_foot", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        condenser_bone,
        balance_beam_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_timber, p_copper, p_iron, p_glass],
        root=Bone("workbench_frame_datum", pivot_local=_jpiv(rng, (8.0, 7.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": table_w,
        "secondary_span": 17,
        "joint_angle_deg": beam_tilt,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": flask_inf,
        "material_theme": f"{p_timber.name}+{p_copper.name}+{p_glass.name}",
        "notes": f"{table_w}-wide artisan distillation & assay workbench ({substyle}) with reagent shelf, brazier furnace + copper retort, angled condenser bridge, distillate carafe, and {beam_tilt:g} deg precision balance with level hanging pans",
        "extra": {"table_w": table_w, "beam_tilt": beam_tilt, "condenser_drop": condenser_drop},
    }


INDUSTRIAL_AND_VEHICLE_DISPATCH: dict[str, Any] = {
    # A. Industrial Mechanical Engineering (3 Seen)
    "piston_crank_engine": _gen_piston_crank_engine,
    "cnc_milling_press": _gen_cnc_milling_press,
    "centrifugal_pump_valve_skid": _gen_centrifugal_pump_valve_skid,
    # B. Vehicles — Land, Construction, Rail, Air & Marine (5 Seen + 2 Unseen)
    "wheeled_haul_truck": _gen_wheeled_haul_truck,
    "tracked_excavator": _gen_tracked_excavator,
    "steam_rail_locomotive": _gen_steam_rail_locomotive,
    "twin_rotor_helicopter_drone": _gen_twin_rotor_helicopter_drone,
    "cargo_tugboat_vessel": _gen_cargo_tugboat_vessel,
    "articulated_forklift_loader": _gen_articulated_forklift_loader,
    "submersible_rov_craft": _gen_submersible_rov_craft,
    # C. Craft / Mechanical Props (2 Seen)
    "clockwork_pendulum_clock": _gen_clockwork_pendulum_clock,
    "alchemical_workbench_apparatus": _gen_alchemical_workbench_apparatus,
}
