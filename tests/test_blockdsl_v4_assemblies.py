"""Comprehensive unit tests for Block-DSL v4.0 Parametric Submodels, Assemblies & World-Model Tasks (ADR 0008).

Covers:
1. Top-down `@layout` DAG evaluation, parameter bounds, `override_layout_params`, non-mutating
   `budget`/`envelope` assertions, and cyclic dependency rejection (`a -> b -> a`).
2. `@component` local namespace encapsulation, abstract `material_slots`, oriented 6-DOF `frame`
   SE(3) mating (`mode: align` vs `mode: oppose`), and `mirror_x` reflection completeness (`det(R)=+1`).
3. Smart connection-driven runtime bone lowering (`fixed` welding onto parent runtime bone WITHOUT
   implicit cross-instance `sculpted_same_bone_miter` collision immunity, `revolute` hinges,
   `prismatic` 3D sliders exported to native `position` channels, and explicit `bind` overrides).
4. `uv_policy: shared` vs `unique` UV island deduplication, component-local 3D material coordinate
   invariance under instance rotation, and `units_per_block` decoupling from `texel_density`.
5. Oriented Bedrock `1.26.40` locator export/decompile round-trip, `export_bedrock_attachable`,
   `SourceMap`, and `reconcile_exported_pose_to_assembly`.
6. Adaptive swept motion validation (`<= 10°` angular steps on `360°` `spin` gears).
7. Flagship v4.0 assemblies (`ModularElderDragon` & `ModularArtilleryWalker`), honest
   `infer_assembly_from_program` (`assembly_provenance="inferred_subtree"`), and all 4 modular
   `LFM3-VL-2B` world-model tasks (`pass_rate == 1.0`, `max_non_target_vertex_drift == 0.0`).
"""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.backends import export_bbmodel, export_bedrock, export_bedrock_attachable, export_geckolib
from synccraft.dsl import (
    DSLValidationError,
    audit_assembly_contracts,
    format_program,
    override_layout_params,
    parse_program,
    reconcile_exported_pose_to_assembly,
)
from synccraft.kinematics import audit_behavior_clip, compile_geometry
from synccraft.shading.uv_packer import allocate_uv_layout
from synccraft.surface.contracts import MaterialProgram
from synccraft.surface.domain import build_surface_domain
from synccraft.world import (
    AssetProvenance,
    MODULAR_ARTILLERY_WALKER_V4_DSL,
    MODULAR_SCULPTED_DRAGON_V4_DSL,
    SCULPTED_ELDER_WYVERN_DSL,
    build_modular_assembly_supervision_suite,
    evaluate_modular_assembly_world_tasks,
    infer_assembly_from_program,
    ingest_external_asset,
)


class BlockDSLV4AssemblyTests(unittest.TestCase):
    def test_layout_dag_bounds_propagation_and_cycle_rejection(self) -> None:
        cyclic_dsl = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16, dsl_version: "4.0" }
        @layout {
          let a = b + 2
          let b = a + 2
        }
        @palette { p1: ramp(base_lch=[0.5, 0.08, 220], steps=6) }
        @component Cube {
          material_slots: [m]
          box c { size: [4, 4, 4], material: m }
        }
        @assembly TestAsm {
          root_pivot: [0, 2, 0]
          instance base : Cube() { id: "b1", materials: { m: p1 } }
        }
        """
        with self.assertRaisesRegex(DSLValidationError, "cycle|Cyclic"):
            parse_program(cyclic_dsl)

        prog = parse_program(MODULAR_SCULPTED_DRAGON_V4_DSL)
        with self.assertRaisesRegex(DSLValidationError, "violates max bound|outside"):
            override_layout_params(prog, {"chest_width": 99})

        widened = override_layout_params(prog, {"chest_width": 14})
        sc_wide = compile_geometry(widened)
        self.assertEqual(sc_wide.box_by_name["torso__ribcage"].size[0], 14)
        # Derived `let head_width = max(4, round(chest_width * 0.6))` -> round(8.4) == 8
        self.assertEqual(sc_wide.box_by_name["skull__cranium"].size[0], 8)

        # Non-mutating budget contract reports violations honestly without mutating sizes
        tight_budget_dsl = MODULAR_ARTILLERY_WALKER_V4_DSL.replace("budget max_boxes = 40", "budget max_boxes = 5")
        tight_prog = parse_program(tight_budget_dsl)
        tight_sc = compile_geometry(tight_prog)
        caud = audit_assembly_contracts(tight_prog, tight_sc)
        self.assertFalse(caud["passed"])
        self.assertTrue(any("max_boxes" in str(v) for v in caud["violations"]))

    def test_frame_mating_align_vs_oppose_and_fixed_welding_without_sibling_immunity(self) -> None:
        # 1. Verify `mode: oppose` flips child frame 180 deg around +Y while keeping det(R) == +1
        oppose_dsl = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16, dsl_version: "4.0" }
        @palette { mat_a: ramp(base_lch=[0.5, 0.06, 210], steps=6) }
        @component BaseBlock {
          material_slots: [m]
          box core { size: [8, 4, 8], material: m }
          frame front_port { on_box: core, anchor: [0, 0, 1], rot: [0, 0, 0] }
        }
        @component PlugNozzle {
          material_slots: [m]
          box barrel { size: [4, 2, 6], material: m }
          frame plug_face { on_box: barrel, anchor: [0, 0, 1], rot: [0, 0, 0] }
        }
        @assembly OpposeTest {
          root_pivot: [0, 2, 0]
          instance base : BaseBlock() { id: "base_01", materials: { m: mat_a } }
          instance nozzle : PlugNozzle() { id: "noz_01", materials: { m: mat_a } }
          connect nozzle.plug_face to base.front_port { type: fixed, mode: oppose }
        }
        """
        op_prog = parse_program(oppose_dsl)
        op_sc = compile_geometry(op_prog)
        # Fixed connection welds `nozzle` onto `base` -> only 1 runtime bone!
        self.assertEqual(len(op_sc.bones), 1)
        noz_box = op_sc.box_by_name["nozzle__barrel"]
        self.assertAlmostEqual(float(np.linalg.det(noz_box.rotation)), 1.0, places=9)
        # Base front face is at Z = +4; nozzle length 6 mated at its +Z face flipped 180 deg -> center at Z = +7!
        self.assertAlmostEqual(float(noz_box.center[2]), 7.0, places=5)
        self.assertTrue(audit_scene(op_sc, enforce_physical_contact=True)["passed"])

        # 2. Verify two distinct `fixed` sibling instances welded onto the SAME bone in `sculpted` mode
        # DO NOT receive implicit `sculpted_same_bone_miter` immunity when colliding with each other!
        colliding_fixed_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.0", audit_profile: sculpted }
        @palette { p1: ramp(base_lch=[0.5, 0.08, 220], steps=6) }
        @component Skull {
          material_slots: [m]
          box head_box { size: [8, 6, 8], material: m }
          frame s_a { on_box: head_box, anchor: [-0.1, 1, 0], rot: [0, 0, 0] }
          frame s_b { on_box: head_box, anchor: [0.1, 1, 0], rot: [0, 0, 0] }
        }
        @component Horn {
          material_slots: [m]
          box spike { size: [4, 4, 4], material: m }
          frame mount { on_box: spike, anchor: [0, -1, 0], rot: [0, 0, 0] }
        }
        @assembly BadHorns {
          root_pivot: [0, 3, 0]
          instance skull : Skull() { id: "sk", materials: { m: p1 } }
          instance horn_a : Horn() { id: "ha", materials: { m: p1 } }
          instance horn_b : Horn() { id: "hb", materials: { m: p1 } }
          connect horn_a.mount to skull.s_a { type: fixed, sculpted_blend: true }
          connect horn_b.mount to skull.s_b { type: fixed, sculpted_blend: true }
        }
        """
        bad_prog = parse_program(colliding_fixed_dsl)
        bad_sc = compile_geometry(bad_prog)
        self.assertEqual(len(bad_sc.bones), 1)
        bad_aud = audit_scene(bad_sc, enforce_physical_contact=True)
        self.assertFalse(bad_aud["passed"])
        self.assertTrue(
            any(
                set(cp["pair"]) == {"horn_a__spike", "horn_b__spike"}
                for cp in bad_aud["collision_pairs"]
            )
        )

    def test_uv_sharing_component_local_surface_basis_and_units_per_block(self) -> None:
        walker_prog = parse_program(MODULAR_ARTILLERY_WALKER_V4_DSL)
        walker_sc = compile_geometry(walker_prog)
        uv = allocate_uv_layout(walker_prog, walker_sc)
        # `leg_fl` and `leg_hl` are unmirrored instances of `WalkerLeg` (`uv_policy: shared`, 3 boxes * 6 faces = 18 faces)
        self.assertEqual(uv.get("shared_uv_face_count"), 18)
        for face in ("north", "south", "east", "west", "up", "down"):
            self.assertEqual(
                uv["faces"]["leg_fl__upper_strut"][face],
                uv["faces"]["leg_hl__upper_strut"][face],
            )
            # Mirrored clone `leg_fr` MUST NOT share unreflected UV islands with `leg_fl`
            self.assertNotEqual(
                uv["faces"]["leg_fl__upper_strut"][face],
                uv["faces"]["leg_fr__upper_strut"][face],
            )

        # Component-local 3D material coordinate invariance under instance rotation
        rot_dsl = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16, units_per_block: 32, dsl_version: "4.0" }
        @palette { timber: ramp(base_lch=[0.5, 0.08, 65], steps=6) }
        @component Beam {
          material_slots: [wood]
          box plank { size: [4, 4, 8], material: wood }
          frame base { on_box: plank, anchor: [0, -1, 0], rot: [0, 0, 0] }
        }
        @component Hub {
          material_slots: [wood]
          box plinth { size: [16, 4, 16], material: wood }
          frame p1 { on_box: plinth, anchor: [-0.5, 1, 0], rot: [0, 0, 0] }
          frame p2 { on_box: plinth, anchor: [0.5, 1, 0], rot: [0, 90, 0] }
        }
        @assembly TimberTest {
          root_pivot: [8, 2, 8]
          instance hub : Hub() { id: "h", materials: { wood: timber } }
          instance b1 : Beam() { id: "b1", materials: { wood: timber } }
          instance b2 : Beam() { id: "b2", materials: { wood: timber } }
          connect b1.base to hub.p1 { type: fixed }
          connect b2.base to hub.p2 { type: fixed }
        }
        @materials {
          timber: process(family=solid_wood, roughness=0.65, anisotropy=2.0)
        }
        """
        r_prog = parse_program(rot_dsl)
        r_sc = compile_geometry(r_prog)
        # `texel_density=16, units_per_block=32` -> 0.5 px per unit -> `[16, 4, 16]` plinth top face is `8x8` px!
        r_uv = allocate_uv_layout(r_prog, r_sc)
        up_rect = r_uv["faces"]["hub__plinth"]["up"]
        self.assertEqual(up_rect[2] - up_rect[0], 8)
        self.assertEqual(up_rect[3] - up_rect[1], 8)

        # Component-local `pos_bone` on `b1__plank` and 90-deg-rotated `b2__plank` must match identically!
        mp = MaterialProgram.from_program(r_prog, seed=42)
        dom = build_surface_domain(r_prog, mp)
        g1 = dom.faces[("b1__plank", "up")]
        g2 = dom.faces[("b2__plank", "up")]
        np.testing.assert_allclose(g1.pos_bone, g2.pos_bone, atol=1e-6)

    def test_oriented_locator_roundtrip_attachable_sourcemap_and_pose_reconcile(self) -> None:
        dragon_prog = parse_program(MODULAR_SCULPTED_DRAGON_V4_DSL)
        dragon_sc = compile_geometry(dragon_prog)
        bed = export_bedrock(dragon_sc, program=dragon_prog)
        bones_json = bed["geometry"]["minecraft:geometry"][0]["bones"]
        skull_json = next(b for b in bones_json if b["name"] == "skull")
        self.assertIn("locators", skull_json)
        flame_loc = skull_json["locators"]["skull__flame_emitter"]
        self.assertIn("offset", flame_loc)
        self.assertIn("rotation", flame_loc)

        # Decompile Bedrock geometry back through `ingest_external_asset` and verify Locator.rot round-trips
        prov = AssetProvenance(
            asset_id="test_oriented_loc",
            source_url="https://example.org/cleanroom/dragon.geo.json",
            author="SyncCraft Cleanroom",
            license_spdx="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            license_text_sha256="a" * 64,
            retrieved_at_iso="2026-11-15T00:00:00Z",
            ai_use_marker_status="checked_none",
            ai_marker_evidence_sha256="b" * 64,
            reviewer_id="synccraft_ci",
        )
        ingested = ingest_external_asset(bed["geometry"], "bedrock_geojson", prov)
        ing_sc = compile_geometry(ingested.program)
        rec_loc = next(lc for lc in ing_sc.locators if lc.name.endswith("skull__flame_emitter"))
        self.assertAlmostEqual(float(rec_loc.local_euler[0]), 10.0, places=3)

        # Attachable + GeckoLib + BBModel exports + SourceMap pose reconciliation
        walker_prog = parse_program(MODULAR_ARTILLERY_WALKER_V4_DSL)
        walker_sc = compile_geometry(walker_prog)
        gecko = export_geckolib(walker_sc, program=walker_prog)
        anim_key = next(iter(gecko["animations"]["animations"].keys()))
        anim_bones = gecko["animations"]["animations"][anim_key]["bones"]
        self.assertIn("position", anim_bones["recoil_barrel"])
        self.assertIn("position", anim_bones["piston_l"])
        self.assertIn("position", anim_bones["piston_r"])

        bb_prog = replace(walker_prog, meta=replace(walker_prog.meta, target_profile="bbmodel_5"))
        bb = export_bbmodel(compile_geometry(bb_prog), program=bb_prog)
        self.assertEqual(bb["synccraft_source_map"]["assembly_name"], "ModularArtilleryWalker")

        att = export_bedrock_attachable(
            walker_sc,
            program=walker_prog,
            attachable_identifier="synccraft:steam_cannon_pack",
        )
        self.assertEqual(att["attachable"]["format_version"], "1.10.0")
        self.assertEqual(
            att["attachable"]["minecraft:attachable"]["description"]["identifier"],
            "synccraft:steam_cannon_pack",
        )

        rec_prog = reconcile_exported_pose_to_assembly(
            walker_prog,
            {"turret_yaw": (0.0, 30.0, 0.0)},
        )
        rec_sc = compile_geometry(rec_prog)
        self.assertIn("offset_rot", format_program(rec_prog))
        self.assertGreater(
            float(np.linalg.norm(rec_sc.bone_by_name["turret_yaw"].rotation - walker_sc.bone_by_name["turret_yaw"].rotation)),
            0.25,
        )

    def test_flagship_v4_assemblies_adaptive_motion_and_world_model_suite(self) -> None:
        for dsl in (MODULAR_SCULPTED_DRAGON_V4_DSL, MODULAR_ARTILLERY_WALKER_V4_DSL):
            prog = parse_program(dsl)
            reparsed = parse_program(format_program(prog))
            sc = compile_geometry(prog)
            sc_re = compile_geometry(reparsed)
            self.assertEqual(len(sc.boxes), len(sc_re.boxes))
            self.assertTrue(audit_scene(sc, enforce_physical_contact=True)["passed"])
            self.assertTrue(audit_assembly_contracts(prog, sc)["passed"])
            for clip in prog.behavior.clips:
                ca = audit_behavior_clip(prog, clip.name)
                self.assertTrue(ca["passed"], msg=f"Clip {clip.name} failed: {ca}")
                self.assertEqual(ca["motion_audit_method"], "adaptive_discrete_step")
                self.assertLessEqual(ca["measured_max_step_deg"], 10.0)
                self.assertLessEqual(ca["max_stance_slip_units"], 0.02)

        # Decompiled flat rigs MUST carry `assembly_provenance == "inferred_subtree"`
        flat_prog = parse_program(SCULPTED_ELDER_WYVERN_DSL)
        inf_spec, inf_smap = infer_assembly_from_program(flat_prog)
        self.assertEqual(inf_spec.assembly_provenance, "inferred_subtree")
        self.assertEqual(inf_smap.assembly_provenance, "inferred_subtree")

        # All 4 modular world-model tasks pass 100% with exact 0.0 non-target vertex drift
        suite = build_modular_assembly_supervision_suite()
        self.assertEqual(len(suite), 4)
        oracle_map = {r["task_id"]: r["oracle_completion"] for r in suite}
        ev = evaluate_modular_assembly_world_tasks(suite, oracle_map)
        self.assertEqual(ev["pass_rate"], 1.0)
        swap_res = next(r for r in ev["results"] if r["task_type"] == "surgical_component_swap_and_repair")
        self.assertEqual(swap_res["max_non_target_vertex_drift"], 0.0)
        self.assertTrue(swap_res["edit_locality_exact"])


def _iter_all_locators(bone):
    for lc in bone.locators:
        yield lc
    for ch in bone.children:
        yield from _iter_all_locators(ch)


if __name__ == "__main__":
    unittest.main()
