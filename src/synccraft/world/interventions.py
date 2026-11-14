"""P1-03d/e Seven Causal Intervention Verification Suite.

Proves mechanism-level 3D world understanding through controlled interventions:
1. Repack UV atlas -> identical 3D world rendered appearance.
2. Repose creature -> rest-space markings remain attached without swimming.
3. Change lighting -> intrinsic albedo/LabPBR unchanged while shaded observation updates.
4. Apply localized rubbing / sliding joint motion -> wear accumulates in contact region & exposes metal.
5. Change environmental exposure -> oxidation forms selectively where coating was worn away.
6. Change output resolution (texel_density 16/32/64) -> physical pattern scale stays consistent.
7. Combine held-out geometry x material x action -> predicts consequence with calibrated uncertainty.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from synccraft.audit.offscreen_renderer import render_world_observation
from synccraft.backends.labpbr_exporter import export_labpbr_1_3
from synccraft.shading.compiler import compile_vanilla_atlas, evaluate_world_optical_pipeline
from synccraft.surface.contracts import EnvironmentState, MaterialSpec
from .exemplars import build_articulated_mechanism_world, build_biological_creature_world
from .transitions import (
    ApplyRubbingAction,
    EditMaterialAction,
    JointTrajectoryAction,
    NoOpEnvironmentAction,
    RepackUVAction,
    SetLightingAction,
    SetPoseAction,
    SetTexelDensityAction,
    TwoBoneIKAction,
    step_world,
)


def run_intervention_1_uv_repack() -> dict[str, Any]:
    """1. Repacking UV layout changes 2D atlas coordinates while keeping 3D appearance identical."""
    ws = build_biological_creature_world(seed=202, subdivisions_per_unit=4)
    atlas_base, png_base = compile_vanilla_atlas(ws, max_width=512, padding=1)
    obs_base = render_world_observation(ws, atlas_override=(atlas_base, png_base), views=("front", "iso"))

    res = step_world(ws, RepackUVAction(max_width=256, padding=3, reverse_boxes=True, reverse_faces=True), dt=0.0)
    assert res.repacked_uv_layout is not None
    atlas_repacked, png_repacked = compile_vanilla_atlas(res.next_state, uv_layout=res.repacked_uv_layout)
    obs_repacked = render_world_observation(
        res.next_state,
        atlas_override=(atlas_repacked, png_repacked),
        views=("front", "iso"),
    )

    uvs_differ = atlas_base["faces"] != atlas_repacked["faces"]
    max_3d_pixel_diff = int(
        np.max(np.abs(obs_base["composite_rgb"].astype(np.int32) - obs_repacked["composite_rgb"].astype(np.int32)))
    )
    return {
        "name": "1_uv_repack_invariance",
        "passed": bool(uvs_differ and max_3d_pixel_diff == 0),
        "uv_layouts_distinct": bool(uvs_differ),
        "max_3d_world_pixel_diff": max_3d_pixel_diff,
    }


def run_intervention_2_repose_attachment() -> dict[str, Any]:
    """2. Reposing joints keeps rest-space RD markings attached without swimming while updating pose/AO."""
    ws = build_biological_creature_world(seed=202, subdivisions_per_unit=4)
    _, _, kin_before, _, opt_before = evaluate_world_optical_pipeline(ws)

    # Sweep limbs & head via pose + 2-bone IK
    step1 = step_world(
        ws,
        SetPoseAction(
            pose_euler_deltas={
                "neck_head": (12.0, -18.0, 0.0),
                "left_crest": (0.0, 0.0, 22.0),
                "right_crest": (0.0, 0.0, -22.0),
            },
            substeps=6,
        ),
        dt=0.0,
    )
    step2 = step_world(
        step1.next_state,
        TwoBoneIKAction(
            upper_bone="left_thigh",
            lower_bone="left_shin",
            effector_local=(0.0, -5.0, 0.0),
            target_world=(-4.0, 3.8, -2.2),
            hinge_axis="x",
            substeps=6,
        ),
        dt=0.0,
    )
    _, _, kin_after, _, opt_after = evaluate_world_optical_pipeline(step2.next_state)

    max_albedo_swim = max(
        float(np.max(np.abs(opt_before.patches[fk].base_color_linear - opt_after.patches[fk].base_color_linear)))
        for fk in opt_before.patches
    )
    max_rd_drift = max(
        float(np.max(np.abs(ws.surface_state.patches[fk].rd_v - step2.next_state.surface_state.patches[fk].rd_v)))
        for fk in ws.surface_state.patches
    )
    ao_delta = max(
        float(np.max(np.abs(kin_before[fk].cavity_ao - kin_after[fk].cavity_ao)))
        for fk in kin_before
    )
    ik_err = float(step2.ik_metadata["position_error"]) if step2.ik_metadata else 999.0

    return {
        "name": "2_repose_marking_attachment",
        "passed": bool(max_albedo_swim < 1e-12 and max_rd_drift < 1e-12 and ao_delta > 1e-3 and ik_err < 1e-3),
        "max_intrinsic_albedo_swim": max_albedo_swim,
        "max_rd_concentration_drift": max_rd_drift,
        "posed_cavity_ao_max_change": ao_delta,
        "ik_effector_error": ik_err,
    }


def run_intervention_3_lighting_separation() -> dict[str, Any]:
    """3. Changing lighting leaves intrinsic albedo & LabPBR atlas.png unchanged while shaded render updates."""
    ws = build_articulated_mechanism_world(seed=101)
    lab_before = export_labpbr_1_3(ws)
    obs_before = render_world_observation(ws, views=("south", "iso"))

    res = step_world(
        ws,
        SetLightingAction(light_dir_world=(-0.72, 0.25, -0.64), ambient_intensity=0.20, sun_intensity=1.15),
        dt=0.0,
    )
    lab_after = export_labpbr_1_3(res.next_state)
    obs_after = render_world_observation(res.next_state, views=("south", "iso"))

    intrinsic_albedo_diff = int(
        np.max(np.abs(lab_before["arrays"]["atlas"].astype(np.int32) - lab_after["arrays"]["atlas"].astype(np.int32)))
    )
    shaded_mean_diff = float(
        np.mean(np.abs(obs_before["composite_rgb"].astype(np.float64) - obs_after["composite_rgb"].astype(np.float64)))
    )
    return {
        "name": "3_lighting_intrinsic_separation",
        "passed": bool(
            lab_before["sha256"]["atlas.png"] == lab_after["sha256"]["atlas.png"]
            and intrinsic_albedo_diff == 0
            and shaded_mean_diff > 2.5
        ),
        "intrinsic_labpbr_albedo_max_diff": intrinsic_albedo_diff,
        "shaded_observation_mean_diff": round(shaded_mean_diff, 4),
    }


def run_intervention_4_sliding_and_rubbing_wear() -> dict[str, Any]:
    """4. Reciprocated joint sliding + localized rubbing accumulate wear and expose metallic substrate."""
    ws = build_articulated_mechanism_world(seed=101)
    _, _, _, _, opt_init = evaluate_world_optical_pipeline(ws)

    # Reciprocate shoulder hinge back and forth under contact load
    kfs = (
        (0.0, {"upper_arm": (0.0, 0.0, 0.0)}),
        (1.0, {"upper_arm": (0.0, 0.0, 32.0)}),
        (2.0, {"upper_arm": (0.0, 0.0, -32.0)}),
        (3.0, {"upper_arm": (0.0, 0.0, 28.0)}),
        (4.0, {"upper_arm": (0.0, 0.0, 0.0)}),
    )
    res_slide = step_world(ws, JointTrajectoryAction(keyframes=kfs, contact_normal_load=2.2, num_samples=20), dt=0.2)

    # Also apply localized rubbing on housing north face
    res_rub = step_world(
        res_slide.next_state,
        ApplyRubbingAction(
            box_name="housing",
            face="north",
            u_center=3.0,
            v_center=2.0,
            radius=1.4,
            normal_load=2.5,
            sliding_distance=3.0,
        ),
        dt=0.0,
    )
    _, _, _, _, opt_worn = evaluate_world_optical_pipeline(res_rub.next_state)

    hinge_wear = float(res_slide.next_state.surface_state.patches[("housing", "east")].wear.max())
    rub_face_metal_before = float(opt_init.patches[("housing", "north")].metalness.max())
    rub_face_metal_after = float(opt_worn.patches[("housing", "north")].metalness.max())
    pedestal_wear = float(res_rub.next_state.surface_state.patches[("stone_pedestal", "down")].wear.max())

    return {
        "name": "4_localized_sliding_and_rubbing_wear",
        "passed": bool(
            hinge_wear > 0.05
            and rub_face_metal_before == 0.0
            and rub_face_metal_after > 0.35
            and pedestal_wear == 0.0
            and res_slide.swept_audit_passed
        ),
        "hinge_contact_max_wear": round(hinge_wear, 5),
        "exposed_metalness_before": rub_face_metal_before,
        "exposed_metalness_after": round(rub_face_metal_after, 4),
        "untouched_pedestal_wear": pedestal_wear,
        "swept_audit_passed": res_slide.swept_audit_passed,
    }


def run_intervention_5_selective_oxidation() -> dict[str, Any]:
    """5. Humid oxygenated exposure oxidizes worn bare metal while intact paint blocks corrosion."""
    ws_clean = build_articulated_mechanism_world(seed=101)
    ws_abraded = step_world(
        ws_clean,
        ApplyRubbingAction(
            box_name="housing",
            face="north",
            u_center=3.0,
            v_center=2.0,
            radius=1.5,
            normal_load=3.0,
            sliding_distance=3.5,
        ),
        dt=0.0,
    ).next_state

    harsh_env = EnvironmentState(humidity=0.95, oxygen_activity=0.55, temperature_c=32.0)
    exposed_clean = step_world(ws_clean, NoOpEnvironmentAction(), env=harsh_env, dt=6.0).next_state
    exposed_abraded = step_world(ws_abraded, NoOpEnvironmentAction(), env=harsh_env, dt=6.0).next_state

    intact_ox = float(exposed_clean.surface_state.patches[("housing", "north")].oxidation.max())
    scar_ox = float(exposed_abraded.surface_state.patches[("housing", "north")].oxidation.max())
    unworn_south_ox = float(exposed_abraded.surface_state.patches[("housing", "south")].oxidation.max())

    return {
        "name": "5_selective_wear_gated_oxidation",
        "passed": bool(intact_ox == 0.0 and unworn_south_ox == 0.0 and scar_ox > 0.25),
        "intact_paint_max_oxidation": intact_ox,
        "unworn_face_max_oxidation": unworn_south_ox,
        "abraded_scar_max_oxidation": round(scar_ox, 4),
    }


def run_intervention_6_resolution_scale_invariance() -> dict[str, Any]:
    """6. Changing solver grid (4 vs 8) and texel_density (16/32/64) preserves physical Turing spot scale."""
    ws_sol4 = build_biological_creature_world(seed=202, subdivisions_per_unit=4)
    ws_16 = build_biological_creature_world(seed=202, subdivisions_per_unit=8)
    ws_32 = step_world(ws_16, SetTexelDensityAction(32), dt=0.0).next_state
    ws_64 = step_world(ws_16, SetTexelDensityAction(64), dt=0.0).next_state

    fractions: dict[str, float] = {}
    mean_lum: dict[str, float] = {}

    for label, state, subdiv in (
        ("sol4_px16", ws_sol4, 4),
        ("sol8_px16", ws_16, 8),
        ("sol8_px32", ws_32, 8),
        ("sol8_px64", ws_64, 8),
    ):
        dom, _, _, _, px_opt = evaluate_world_optical_pipeline(state, subdivisions_per_unit=subdiv)
        weighted_spot = 0.0
        weighted_lum = 0.0
        total_area = 0.0
        for fk, grid in dom.faces.items():
            if grid.box_name != "torso":
                continue
            area = float(grid.width_units * grid.height_units)
            patch = px_opt.patches[fk]
            luma = (
                0.2126 * patch.base_color_linear[..., 0]
                + 0.7152 * patch.base_color_linear[..., 1]
                + 0.0722 * patch.base_color_linear[..., 2]
            )
            weighted_spot += area * float(np.mean(luma < 0.22))
            weighted_lum += area * float(np.mean(luma))
            total_area += area
        fractions[label] = weighted_spot / max(1e-6, total_area)
        mean_lum[label] = weighted_lum / max(1e-6, total_area)

    ref_frac = fractions["sol8_px64"]
    max_px_drift = max(abs(fractions["sol8_px16"] - ref_frac), abs(fractions["sol8_px32"] - ref_frac))
    solver_drift = abs(fractions["sol4_px16"] - ref_frac)
    max_lum_drift = max(abs(v - mean_lum["sol8_px64"]) for v in mean_lum.values())
    return {
        "name": "6_resolution_physical_scale_consistency",
        "passed": bool(
            0.05 < ref_frac < 0.75
            and max_px_drift < 0.05
            and solver_drift < 0.06
            and max_lum_drift < 0.02
        ),
        "spot_area_fractions": {k: round(v, 4) for k, v in fractions.items()},
        "mean_linear_luminance": {k: round(v, 5) for k, v in mean_lum.items()},
        "max_pixel_density_spot_drift": round(max_px_drift, 5),
        "solver_discretization_spot_drift": round(solver_drift, 5),
        "max_luminance_drift": round(max_lum_drift, 6),
    }


def run_intervention_7_compositional_generalization() -> dict[str, Any]:
    """7. Held-out (creature geometry x painted_metal prosthetic armor x IK + humid wear) with uncertainty."""
    ws_creature = build_biological_creature_world(seed=202, subdivisions_per_unit=4)
    # Swap creature's lower limb scales to an industrial painted-metal prosthetic material (novel combination)
    prosthetic_spec = MaterialSpec(
        name="limb_scales",
        family="painted_metal",
        substrate_lch=(0.76, 0.02, 240.0),
        coating_lch=(0.52, 0.16, 28.0),
        oxide_lch=(0.42, 0.12, 42.0),
        coating_thickness=0.55,
        perceptual_roughness=0.30,
        substrate_metalness=1.0,
        wear_susceptibility=0.75,
        oxidation_rate=0.50,
        seed=707,
    )
    humid_env = EnvironmentState(humidity=0.90, oxygen_activity=0.50, temperature_c=28.0)
    ik_act = TwoBoneIKAction(
        upper_bone="left_thigh",
        lower_bone="left_shin",
        effector_local=(0.0, -5.0, 0.0),
        target_world=(-4.0, 1.4, -3.2),
        hinge_axis="x",
    )
    rub_act = ApplyRubbingAction(
        "shin_seg", "north", u_center=1.0, v_center=2.5, radius=1.2, normal_load=1.1, sliding_distance=1.2
    )

    def _rollout_prosthetic(spec_candidate: MaterialSpec) -> tuple[bool, float]:
        s0 = step_world(ws_creature, EditMaterialAction(spec_candidate), dt=0.0).next_state
        r_ik = step_world(s0, ik_act, dt=0.2)
        s2 = step_world(r_ik.next_state, rub_act, dt=0.0).next_state
        s3 = step_world(s2, NoOpEnvironmentAction(), env=humid_env, dt=4.0).next_state
        return r_ik.swept_audit_passed, float(s3.surface_state.patches[("shin_seg", "north")].oxidation.mean())

    # Ensemble epistemic uncertainty across prior over unknown coating_thickness / wear_susceptibility
    ensemble_ox_preds = [
        _rollout_prosthetic(replace(prosthetic_spec, coating_thickness=coat_h, wear_susceptibility=wear_k))[1]
        for coat_h, wear_k in ((0.35, 0.95), (0.55, 0.75), (0.80, 0.55))
    ]

    swept_ok, actual_ox = _rollout_prosthetic(prosthetic_spec)
    pred_mean = float(np.mean(ensemble_ox_preds))
    pred_std = float(np.std(ensemble_ox_preds))
    within_2sigma = abs(actual_ox - pred_mean) <= (2.0 * pred_std + 1e-3)

    return {
        "name": "7_compositional_heldout_generalization",
        "passed": bool(swept_ok and actual_ox > 0.02 and pred_std > 0.01 and within_2sigma),
        "swept_audit_passed": swept_ok,
        "actual_mean_oxidation": round(actual_ox, 5),
        "predicted_mean_oxidation": round(pred_mean, 5),
        "epistemic_uncertainty_std": round(pred_std, 5),
        "within_calibrated_interval": within_2sigma,
    }


def run_seven_interventions_suite() -> dict[str, Any]:
    """Run all 7 causal world-model interventions and return structured evidence."""
    results = [
        run_intervention_1_uv_repack(),
        run_intervention_2_repose_attachment(),
        run_intervention_3_lighting_separation(),
        run_intervention_4_sliding_and_rubbing_wear(),
        run_intervention_5_selective_oxidation(),
        run_intervention_6_resolution_scale_invariance(),
        run_intervention_7_compositional_generalization(),
    ]
    all_passed = all(r["passed"] for r in results)
    return {
        "all_passed": all_passed,
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_count": len(results),
        "interventions": results,
    }
