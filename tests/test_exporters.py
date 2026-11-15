"""Exporter checks exercise transforms and failures, not just JSON presence."""

import copy
from dataclasses import replace
import itertools
import json
import unittest

import numpy as np

from synccraft.backends import ExportError, export_bbmodel, export_java
from synccraft.backends.common import euler_zyx_degrees
from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Program
from synccraft.kinematics.se3_chain import compile_geometry


def rotation(eulers):
    x, y, z = np.deg2rad(eulers)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def scene(angles=(0, 0, 0), second_angles=None, *, mirror=False, profile="bbmodel_5"):
    child = Bone("arm", rest_euler=angles,
                 attach=Attach("base", (1, 0, 0), (-1, 0, 0)),
                 mirror_name="other_arm" if mirror else None,
                 items=[Box("arm_box", (3, 2, 1), "mat")])
    if second_angles is not None:
        child.items.append(Bone("tip", rest_euler=second_angles,
                               attach=Attach("arm_box", (1, 0, 0), (-1, 0, 0)),
                               items=[Box("tip_box", (2, 1, 1), "mat")]))
    program = Program(Meta("weapon_3d", profile, 16), [Palette("mat", (0.6, 0.1, 30), 4)],
                      Bone("root", pivot_local=(8, 8, 8), items=[Box("base", (2, 4, 2), "mat"), child]))
    return compile_geometry(program)


class ExportTests(unittest.TestCase):
    def test_bbmodel_reconstructs_nested_rotated_world_geometry(self):
        compiled = scene((20, 30, -15), (10, -10, 35), mirror=True)
        model = export_bbmodel(compiled)
        group_map = {g["uuid"]: g for g in model["groups"]}
        element_map = {e["uuid"]: e for e in model["elements"]}
        expected = {b.name: b for b in compiled.boxes}
        observed = set()

        def visit(node, parent_pivot, parent_rot, parent_origin):
            group = group_map[node["uuid"]]
            origin = np.asarray(group["origin"])
            pivot = parent_pivot + parent_rot @ (origin - parent_origin)
            world_rot = parent_rot @ rotation(group["rotation"])
            for child in node["children"]:
                if isinstance(child, dict):
                    visit(child, pivot, world_rot, origin)
                else:
                    cube = element_map[child]
                    center = (np.array(cube["from"]) + np.array(cube["to"])) / 2
                    actual = pivot + world_rot @ (center - origin)
                    np.testing.assert_allclose(actual, expected[cube["name"]].center, atol=1e-8)
                    np.testing.assert_allclose(world_rot, expected[cube["name"]].rotation, atol=1e-8)
                    observed.add(cube["name"])
        for root in model["outliner"]:
            visit(root, np.zeros(3), np.eye(3), np.zeros(3))
        self.assertEqual(observed, set(expected))
        self.assertEqual(model["meta"]["format_version"], "5.0")
        self.assertTrue(all(set(node) == {"uuid", "isOpen", "children"} for node in model["outliner"]))
        self.assertEqual(json.dumps(model, sort_keys=True), json.dumps(export_bbmodel(compiled), sort_keys=True))

    def test_java_modern_reconstructs_nested_obb_vertices(self):
        compiled = scene((20, 30, -15), (10, -10, 35), mirror=True)
        model = export_java(compiled, profile="java_modern")
        for box, element in zip(compiled.boxes, model["elements"]):
            center = (np.array(element["from"]) + element["to"]) / 2
            half = (np.array(element["to"]) - element["from"]) / 2
            data = element.get("rotation", {})
            r = rotation([data.get(axis, 0) for axis in "xyz"])
            self.assertNotIn("axis", data)
            local = np.array(list(itertools.product((-1, 1), repeat=3))) * half
            actual = center + local @ r.T
            predicted = box.center + local @ box.rotation.T
            np.testing.assert_allclose(actual, predicted, atol=1e-8)

    def test_euler_roundtrip_including_gimbal_lock(self):
        for angles in ((12, 90, 37), (12, -90, 37), (179, -80, -179), (13, 27, 81)):
            expected = rotation(angles)
            np.testing.assert_allclose(rotation(euler_zyx_degrees(expected)), expected, atol=1e-8)

    def test_legacy_checks_composed_rotation(self):
        with self.assertRaises(ExportError):
            export_java(scene((22.5, 0, 0), (0, 22.5, 0)), profile="java_legacy")
        with self.assertRaises(ExportError):
            export_java(scene((0, 0, 45), (0, 0, 45)), profile="java_legacy")
        model = export_java(scene((0, 0, 22.5), (0, 0, 22.5)), profile="java_legacy")
        self.assertEqual(model["elements"][-1]["rotation"]["angle"], 45.0)

    def test_target_and_bounds_rejection(self):
        with self.assertRaises(ExportError):
            export_java(scene())
        program = Program(Meta("weapon_3d", "bbmodel_5", 16), [Palette("mat", (0.6, 0.1, 30), 4)],
                          Bone("root", pivot_local=(100, 8, 8), items=[Box("base", (2, 4, 2), "mat")]))
        compiled = compile_geometry(program)
        with self.assertRaisesRegex(ExportError, "bounds"):
            export_java(compiled, profile="java_modern")

    def test_mutated_scene_cannot_disagree_between_audit_and_export(self):
        compiled = scene()
        compiled.boxes[0].center[0] += 1
        for exporter in (export_bbmodel, lambda s: export_java(s, profile="java_modern")):
            with self.assertRaisesRegex(ExportError, "local/world center disagreement"):
                exporter(compiled)

    def test_mirrored_cullface_swaps_east_and_west(self):
        compiled = scene(mirror=True)
        changed = tuple(replace(b, cullface=("east",)) for b in compiled.boxes)
        compiled = replace(compiled, boxes=changed)
        bb = export_bbmodel(compiled)
        java = export_java(compiled, profile="java_modern")
        for box, element, java_element in zip(compiled.boxes, bb["elements"], java["elements"]):
            expected = "west" if box.mirrored else "east"
            for candidate in (element, java_element):
                cullfaces = {face for face, data in candidate["faces"].items() if "cullface" in data}
                self.assertEqual(cullfaces, {expected})

    def test_uv_pixels_are_normalized_only_for_java(self):
        compiled = scene()
        atlas = {
            "width": 64, "height": 32, "data_url": "data:image/png;base64,placeholder",
            "faces": {b.name: {f: [4, 2, 20, 10] for f in ("north", "south", "east", "west", "up", "down")} for b in compiled.boxes},
        }
        original = copy.deepcopy(atlas)
        bb = export_bbmodel(compiled, atlas=atlas)
        java = export_java(compiled, atlas=atlas, profile="java_modern")
        self.assertEqual(bb["elements"][0]["faces"]["north"]["uv"], [4, 2, 20, 10])
        self.assertEqual(java["elements"][0]["faces"]["north"]["uv"], [1, 1, 5, 5])
        self.assertEqual(atlas, original)
        del atlas["faces"][compiled.boxes[0].name]["up"]
        with self.assertRaises(ExportError):
            export_bbmodel(compiled, atlas=atlas)

    def test_bedrock_and_gecko_fail_explicitly(self):
        from synccraft.backends.bedrock_gecko_exporter import export_bedrock, export_geckolib
        for backend in (export_bedrock, export_geckolib):
            with self.assertRaises(ExportError):
                backend(scene())

    def test_bedrock_and_geckolib_reconstruct_nested_rotated_mirrored_geometry_locators_and_animations(self):
        from synccraft.backends import export_bedrock, export_geckolib
        from synccraft.dsl.ast_nodes import BehaviorDecl, ClipDecl, Locator, TrackDecl
        from synccraft.world.inverse_inference import AssetProvenance, ingest_external_asset

        for target_prof, exporter_fn in (("bedrock_entity", export_bedrock), ("geckolib", export_geckolib)):
            arm = Bone(
                "arm_l",
                rest_euler=(20.0, 30.0, -15.0),
                attach=Attach("torso_box", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
                mirror_name="arm_r",
                items=[
                    Box("arm_l_box", (3, 2, 2), "mat", inflate=0.125),
                    Locator("claw_tip_l", offset=(1.5, 0.0, 0.0), role="vfx_emitter"),
                    Bone(
                        "forearm_l",
                        rest_euler=(10.0, -15.0, 25.0),
                        attach=Attach("arm_l_box", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                        mirror_name="forearm_r",
                        items=[Box("forearm_l_box", (2, 1, 1), "mat")],
                    ),
                ],
            )
            prog = Program(
                Meta("entity", target_prof, 16),
                [Palette("mat", (0.62, 0.11, 35.0), 4)],
                Bone(
                    "body_root",
                    pivot_local=(0.0, 10.0, 0.0),
                    items=[
                        Box("torso_box", (6, 8, 4), "mat"),
                        Locator("rider_mount", offset=(0.0, 5.0, 0.0), role="rider_seat"),
                        arm,
                    ],
                ),
                behavior=BehaviorDecl(
                    clips=(
                        ClipDecl(
                            "idle_sway",
                            duration_s=1.6,
                            loop=True,
                            tracks=(TrackDecl("arm_l", "wave", "z", 12.0, 0.625, 0.0, 0.0),),
                        ),
                    )
                ),
            )
            compiled = compile_geometry(prog)
            bundle = exporter_fn(compiled, program=prog, animation_mode="keyframes", num_animation_frames=10)
            self.assertEqual(bundle["geometry"]["format_version"], "1.12.0")
            self.assertFalse(bundle["runtime_import_verified"])
            self.assertLessEqual(bundle["serialized_vertex_max_error"], 1e-8)
            self.assertIsNotNone(bundle["animations"])
            anim_clip = next(iter(bundle["animations"]["animations"].values()))
            self.assertTrue(anim_clip["loop"])
            self.assertIn("arm_l", anim_clip["bones"])

            # Independent roundtrip through Bedrock decompiler verifies 8-corner fidelity & locator survival
            prov = AssetProvenance("rt_entity", "https://github.com/BaiTian6641/SyncCraft", "SyncCraft", "Apache-2.0")
            rec = ingest_external_asset(
                bundle["geometry"],
                "bedrock_geojson",
                prov,
                animation_payload=bundle["animations"],
            )
            self.assertLess(rec.vertex_reconstruction_error, 1e-8)
            rt_scene = compile_geometry(rec.program)
            self.assertEqual(len(rt_scene.boxes), len(compiled.boxes))
            unmatched = list(rt_scene.boxes)
            for orig_bx in compiled.boxes:
                best_idx = int(np.argmin([np.linalg.norm(rb.center - orig_bx.center) for rb in unmatched]))
                rt_bx = unmatched.pop(best_idx)
                err = float(
                    np.max(
                        np.linalg.norm(
                            orig_bx.corners(include_inflate=False) - rt_bx.corners(include_inflate=False),
                            axis=-1,
                        )
                    )
                )
                self.assertLess(err, 1e-8)
            self.assertGreaterEqual(len(rt_scene.locators), 3)
            self.assertIsNotNone(rec.program.behavior)

            # Also verify Molang animation emission mode
            molang_bundle = exporter_fn(compiled, program=prog, animation_mode="molang")
            molang_clip = next(iter(molang_bundle["animations"]["animations"].values()))
            molang_rot = molang_clip["bones"]["arm_l"]["rotation"]
            self.assertTrue(any(isinstance(x, str) and "math.sin" in x for x in molang_rot))

            # Mutated scene must be rejected before serialization
            tampered = compile_geometry(prog)
            tampered.boxes[0].center[0] += 1.0
            with self.assertRaisesRegex(ExportError, "local/world center disagreement"):
                exporter_fn(tampered, program=prog)

    def test_bbmodel_animated_export_emits_keyframes_and_roundtrips_clips(self):
        from synccraft.data_gen.complex_entity_and_asset_pack import build_apex_complex_program
        from synccraft.world.inverse_inference import AssetProvenance, ingest_external_asset

        prog = build_apex_complex_program("elemental_pocket_companion", seed=101)
        compiled = compile_geometry(prog)
        static_bb = export_bbmodel(compiled, name="static_companion")
        self.assertNotIn("animations", static_bb)

        anim_bb = export_bbmodel(compiled, name="anim_companion", program=prog, num_animation_frames=10)
        self.assertIn("animations", anim_bb)
        self.assertGreaterEqual(len(anim_bb["animations"]), 1)
        first_anim = anim_bb["animations"][0]
        self.assertGreater(len(first_anim["animators"]), 0)

        prov = AssetProvenance("rt_bb", "https://github.com/BaiTian6641/SyncCraft", "SyncCraft", "Apache-2.0")
        rec = ingest_external_asset(anim_bb, "bbmodel", prov)
        self.assertLess(rec.vertex_reconstruction_error, 1e-8)
        self.assertIsNotNone(rec.program.behavior)
        assert rec.program.behavior is not None
        self.assertGreaterEqual(len(rec.program.behavior.clips), 1)

    def test_exported_keyframes_preserve_exact_so3_matrix_composition_and_joint_limits_on_rotated_bones(self):
        import math
        from synccraft.backends import bedrock_rotation_matrix, export_bedrock
        from synccraft.dsl.ast_nodes import BehaviorDecl, ClipDecl, TrackDecl
        from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip

        prog = Program(
            Meta("entity", "bedrock_entity", 16),
            [Palette("mat", (0.6, 0.1, 30.0), 4)],
            Bone(
                "root_b",
                pivot_local=(0.0, 10.0, 0.0),
                items=[
                    Box("torso", (6, 8, 4), "mat"),
                    Bone(
                        "arm_l",
                        rest_euler=(25.0, 35.0, -20.0),
                        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-8.0, 8.0)),
                        attach=Attach("torso", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
                        items=[Box("arm_b", (4, 2, 2), "mat", inflate=0.125)],
                    ),
                ],
            ),
            behavior=BehaviorDecl(
                clips=(ClipDecl("sway", 1.0, True, tracks=(TrackDecl("arm_l", "wave", "z", 20.0, 1.0),)),)
            ),
        )
        compiled = compile_geometry(prog)
        bundle = export_bedrock(compiled, program=prog, num_animation_frames=8, animation_mode="keyframes")
        samples = evaluate_behavior_clip(prog, "sway", num_frames=8)

        bed_bones = {b["name"]: b for b in bundle["geometry"]["minecraft:geometry"][0]["bones"]}
        rest_bed = np.asarray(bed_bones["arm_l"]["rotation"], dtype=np.float64)
        np.testing.assert_allclose(rest_bed, np.array([-25.0, -35.0, -20.0]), atol=1e-4)

        m_x = np.diag([-1.0, 1.0, 1.0])
        kfs = bundle["animations"]["animations"]["animation.entity.sway"]["bones"]["arm_l"]["rotation"]
        for s in samples:
            t_key = f"{round(float(s.time_s), 4):.4f}".rstrip("0").rstrip(".")
            if "." not in t_key:
                t_key = f"{t_key}.0"
            kf = np.asarray(kfs[t_key], dtype=np.float64)
            r_bed_posed = bedrock_rotation_matrix(tuple(rest_bed + kf))
            r_recon = m_x @ r_bed_posed @ m_x
            r_exact = s.scene.bone_by_name["arm_l"].rotation
            cos_t = float(np.clip((np.trace(r_recon.T @ r_exact) - 1.0) * 0.5, -1.0, 1.0))
            deg_err = math.degrees(math.acos(cos_t))
            self.assertLess(deg_err, 1e-3)

    def test_c5_m16_asymmetric_rotated_multibone_bedrock_and_geckolib_axis_conjugation_and_texel_density(self):
        import math
        from synccraft.backends import (
            bedrock_rotation_matrix,
            export_bedrock,
            export_geckolib,
        )
        from synccraft.dsl.ast_nodes import (
            BehaviorDecl,
            ClipDecl,
            IKChainDecl,
            IKDriveDecl,
            KinematicsDecl,
            Locator,
            Place,
            TrackDecl,
        )
        from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip
        from synccraft.world.inverse_inference import (
            AssetProvenance,
            _decompile_bedrock_geojson,
            ingest_external_asset,
        )

        m_x = np.diag([-1.0, 1.0, 1.0])
        for target_prof, exporter_fn in (("bedrock_entity", export_bedrock), ("geckolib", export_geckolib)):
            arm_l = Bone(
                "arm_l",
                rest_euler=(18.0, 27.0, -14.0),
                attach=Attach("torso_box", (1.0, 0.5, 0.25), (-1.0, 0.0, 0.0)),
                items=[
                    Box("arm_l_box", (3, 7, 2), "mat", place=Place(None, (0.0, 0.0, 0.0), (-1.0, 1.0, 0.0))),
                    Locator("emitter_l", offset=(1.5, -2.0, 0.5), role="vfx_emitter"),
                    Bone(
                        "hand_l",
                        rest_euler=(-12.0, 22.0, 19.0),
                        attach=Attach("arm_l_box", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                        items=[Box("hand_l_box", (2, 3, 2), "mat", place=Place(None, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))],
                    ),
                ],
            )
            arm_r = Bone(
                "arm_r",
                rest_euler=(-24.0, -19.0, 31.0),
                attach=Attach("torso_box", (-1.0, 0.25, -0.25), (1.0, 0.0, 0.0)),
                items=[
                    Box("arm_r_box", (2, 5, 4), "mat", place=Place(None, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
                ],
            )
            prog = Program(
                Meta("entity", target_prof, 16),
                [Palette("mat", (0.58, 0.12, 42.0), 4)],
                Bone(
                    "torso_root",
                    pivot_local=(2.0, 12.0, -1.0),
                    items=[
                        Box("torso_box", (6, 8, 4), "mat"),
                        Locator("beacon_socket", offset=(2.5, 4.0, -1.0), role="socket"),
                        arm_l,
                        arm_r,
                    ],
                ),
                kinematics=KinematicsDecl(
                    ik_chains=(IKChainDecl("arm_l_ik", "arm_l", "hand_l", (0.0, -3.0, 0.0), "x"),)
                ),
                behavior=BehaviorDecl(
                    clips=(
                        ClipDecl(
                            "asym_stride",
                            duration_s=1.2,
                            loop=True,
                            tracks=(
                                TrackDecl("arm_l", "wave", "x", 15.0, 1.0 / 1.2, 0.0, 4.0),
                                TrackDecl("arm_l", "wave", "y", -11.0, 1.0 / 1.2, 90.0, -3.0),
                                TrackDecl("arm_l", "wave", "z", 9.0, 1.0 / 1.2, 0.0, 2.0),
                                TrackDecl("arm_r", "wave", "x", -13.0, 1.0 / 1.2, 45.0, -5.0),
                                TrackDecl("arm_r", "wave", "y", 17.0, 1.0 / 1.2, 0.0, 6.0),
                                TrackDecl("arm_r", "wave", "z", 9.0, 1.0 / 1.2, 0.0, 2.0),
                            ),
                            ik_drives=(IKDriveDecl("arm_l_ik", stride=1.5, lift=0.6, phase_deg=0.0),),
                        ),
                    )
                ),
            )
            compiled = compile_geometry(prog)
            bundle = exporter_fn(compiled, program=prog, animation_mode="keyframes", num_animation_frames=6)
            self.assertLessEqual(bundle["serialized_vertex_max_error"], 1e-8)

            geo_bones = {b["name"]: b for b in bundle["geometry"]["minecraft:geometry"][0]["bones"]}
            # 1. Pivot X & Locator X are strictly negated in Bedrock 1.12.0 space
            np.testing.assert_allclose(geo_bones["torso_root"]["pivot"], [-2.0, 12.0, -1.0], atol=1e-6)
            sync_arm_l_piv = compiled.bone_by_name["arm_l"].pivot_world
            sync_arm_r_piv = compiled.bone_by_name["arm_r"].pivot_world
            np.testing.assert_allclose(
                geo_bones["arm_l"]["pivot"],
                [-sync_arm_l_piv[0], sync_arm_l_piv[1], sync_arm_l_piv[2]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                geo_bones["arm_r"]["pivot"],
                [-sync_arm_r_piv[0], sync_arm_r_piv[1], sync_arm_r_piv[2]],
                atol=1e-6,
            )
            loc_root_sync = compiled.locator_by_name["beacon_socket"]
            loc_root_world = compiled.bone_by_name["torso_root"].pivot_world + loc_root_sync.local_offset
            loc_bed = next(iter(geo_bones["torso_root"]["locators"].values()))
            np.testing.assert_allclose(loc_bed, [-loc_root_world[0], loc_root_world[1], loc_root_world[2]], atol=1e-6)

            # 2. Cube origin X = -(sync_from_x + size_x) and rest Euler X & Y negated, Z preserved
            torso_cb = geo_bones["torso_root"]["cubes"][0]
            # torso_box center=(2, 12, -1), size=(6, 8, 4) -> sync_from=(-1, 8, -3) -> bedrock_origin_x = -(-1 + 6) = -5
            np.testing.assert_allclose(torso_cb["origin"], [-5.0, 8.0, -3.0], atol=1e-6)
            np.testing.assert_allclose(geo_bones["arm_l"]["rotation"], [-18.0, -27.0, -14.0], atol=1e-5)
            np.testing.assert_allclose(geo_bones["arm_r"]["rotation"], [24.0, 19.0, 31.0], atol=1e-5)
            np.testing.assert_allclose(geo_bones["hand_l"]["rotation"], [12.0, -22.0, 19.0], atol=1e-5)

            # Verify exact O(3) conjugation M_x @ R_bed @ M_x == R_sync_local on every rotated bone
            for bname in ("arm_l", "arm_r", "hand_l"):
                wb = compiled.bone_by_name[bname]
                parent_r = compiled.bone_by_name[wb.parent_name].rotation
                r_local_sync = parent_r.T @ wb.rotation
                r_bed = bedrock_rotation_matrix(geo_bones[bname]["rotation"])
                np.testing.assert_allclose(m_x @ r_bed @ m_x, r_local_sync, atol=1e-6)

            # 3. Verify animated keyframe rotations & root position X negation
            samples = evaluate_behavior_clip(prog, "asym_stride", num_frames=6)
            anim_bones = next(iter(bundle["animations"]["animations"].values()))["bones"]
            for s in samples:
                t_key = f"{round(float(s.time_s), 4):.4f}".rstrip("0").rstrip(".")
                if "." not in t_key:
                    t_key = f"{t_key}.0"
                for bname in ("arm_l", "arm_r", "hand_l"):
                    wb_s = s.scene.bone_by_name[bname]
                    parent_r_s = s.scene.bone_by_name[wb_s.parent_name].rotation
                    r_loc_exact = parent_r_s.T @ wb_s.rotation
                    rest_b = np.asarray(geo_bones[bname]["rotation"], dtype=np.float64)
                    kf_b = np.asarray(anim_bones[bname]["rotation"][t_key], dtype=np.float64)
                    r_loc_recon = m_x @ bedrock_rotation_matrix(tuple(rest_b + kf_b)) @ m_x
                    cos_t = float(np.clip((np.trace(r_loc_recon.T @ r_loc_exact) - 1.0) * 0.5, -1.0, 1.0))
                    self.assertLess(math.degrees(math.acos(cos_t)), 1e-3)

            # 4. Verify Molang X & Y negation vs Z preservation (and roundtrip recovery)
            mol_bundle = exporter_fn(compiled, program=prog, animation_mode="molang")
            mol_arm_r = next(iter(mol_bundle["animations"]["animations"].values()))["bones"]["arm_r"]["rotation"]
            # SyncCraft arm_r x: bias=-5, amp=-13 -> Bedrock x negated: bias=+5, amp=+13
            self.assertIn("* 13.0 + 5.0", str(mol_arm_r[0]))
            # SyncCraft arm_r y: bias=+6, amp=+17 -> Bedrock y negated: bias=-6, amp=-17
            self.assertIn("* -17.0 - 6.0", str(mol_arm_r[1]))
            # SyncCraft arm_r z: bias=+2, amp=+9 -> Bedrock z preserved: bias=+2, amp=+9
            self.assertIn("* 9.0 + 2.0", str(mol_arm_r[2]))

            # 5. Full roundtrip through _decompile_bedrock_geojson + animation decompiler
            prov = AssetProvenance("asym_c5", "https://github.com/BaiTian6641/SyncCraft", "SyncCraft", "Apache-2.0")
            rec = ingest_external_asset(
                bundle["geometry"],
                "bedrock_geojson",
                prov,
                animation_payload=bundle["animations"],
            )
            self.assertLess(rec.vertex_reconstruction_error, 1e-7)
            rt_sc = compile_geometry(rec.program)
            self.assertEqual(len(rt_sc.boxes), 4)
            # Confirm distinct L/R sizes were NOT collapsed into a mirror
            rt_sizes = sorted(tuple(float(x) for x in bx.size) for bx in rt_sc.boxes)
            self.assertEqual(
                rt_sizes,
                sorted([(6.0, 8.0, 4.0), (3.0, 7.0, 2.0), (2.0, 5.0, 4.0), (2.0, 3.0, 2.0)]),
            )
            unmatched = list(rt_sc.boxes)
            for orig_bx in compiled.boxes:
                best_i = int(np.argmin([np.linalg.norm(rb.center - orig_bx.center) for rb in unmatched]))
                rt_bx = unmatched.pop(best_i)
                c_err = float(
                    np.max(
                        np.linalg.norm(
                            orig_bx.corners(include_inflate=False) - rt_bx.corners(include_inflate=False),
                            axis=-1,
                        )
                    )
                )
                self.assertLess(c_err, 1e-7)

        # 6. M16: Verify texel_density is inferred from cube UV ratios / texture_width (not dead-coded 16)
        geo_16 = {
            "format_version": "1.12.0",
            "minecraft:geometry": [{
                "description": {"identifier": "geometry.test.d16", "texture_width": 64, "texture_height": 64},
                "bones": [{
                    "name": "root",
                    "pivot": [0, 0, 0],
                    "cubes": [{
                        "origin": [-2, 0, -2],
                        "size": [4, 4, 4],
                        "uv": {"north": {"uv": [0, 0], "uv_size": [4, 4]}},
                    }],
                }],
            }],
        }
        geo_32 = copy.deepcopy(geo_16)
        geo_32["minecraft:geometry"][0]["description"]["texture_width"] = 128
        geo_32["minecraft:geometry"][0]["description"]["texture_height"] = 128
        geo_32["minecraft:geometry"][0]["bones"][0]["cubes"][0]["uv"]["north"]["uv_size"] = [8, 8]
        p16, _, _, _ = _decompile_bedrock_geojson(geo_16)
        p32, _, _, _ = _decompile_bedrock_geojson(geo_32)
        self.assertEqual(p16.meta.texel_density, 16)
        self.assertEqual(p32.meta.texel_density, 32)

    def test_java_legacy_and_modern_rotated_elements_export_and_decompile_roundtrip(self):
        from synccraft.dsl.ast_nodes import Place
        from synccraft.world.inverse_inference import AssetProvenance, ingest_external_asset

        prov = AssetProvenance("java_rot", "https://github.com/BaiTian6641/SyncCraft", "SyncCraft", "Apache-2.0")

        # 1. java_legacy single-axis 22.5 / -45.0 deg rotated elements with off-center pivots
        legacy_prog = Program(
            Meta("weapon_3d", "java_legacy", 16),
            [Palette("steel", (0.65, 0.04, 240.0), 4)],
            Bone(
                "hilt_root",
                pivot_local=(8.0, 8.0, 8.0),
                items=[
                    Box("grip", (2, 6, 2), "steel"),
                    Bone(
                        "guard_x",
                        rest_euler=(22.5, 0.0, 0.0),
                        attach=Attach("grip", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
                        items=[Box("guard_bar", (6, 1, 2), "steel")],
                    ),
                    Bone(
                        "pommel_z",
                        rest_euler=(0.0, 0.0, -45.0),
                        attach=Attach("grip", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                        items=[Box("pommel_gem", (2, 2, 2), "steel")],
                    ),
                ],
            ),
        )
        legacy_sc = compile_geometry(legacy_prog)
        legacy_json = export_java(legacy_sc, profile="java_legacy")
        self.assertEqual(legacy_json["elements"][1]["rotation"]["axis"], "x")
        self.assertAlmostEqual(legacy_json["elements"][1]["rotation"]["angle"], 22.5, places=5)
        self.assertEqual(legacy_json["elements"][2]["rotation"]["axis"], "z")
        self.assertAlmostEqual(legacy_json["elements"][2]["rotation"]["angle"], -45.0, places=5)

        rec_leg = ingest_external_asset(legacy_json, "java_json", prov)
        self.assertLess(rec_leg.vertex_reconstruction_error, 1e-6)
        rt_leg_sc = compile_geometry(rec_leg.program)
        self.assertEqual(len(rt_leg_sc.boxes), 3)

        # 2. java_modern multi-axis rotated elements roundtrip
        modern_sc = scene((20.0, 30.0, -15.0), (10.0, -10.0, 35.0), mirror=True, profile="java_modern")
        modern_json = export_java(modern_sc, profile="java_modern")
        rec_mod = ingest_external_asset(modern_json, "java_json", prov)
        self.assertLess(rec_mod.vertex_reconstruction_error, 1e-6)
        rt_mod_sc = compile_geometry(rec_mod.program)
        self.assertEqual(len(rt_mod_sc.boxes), len(modern_sc.boxes))

    def test_h7_provenance_cc_by_nc_3_0_allowlist_nd_ban_noai_gate_and_unlabeled_policy(self):
        import hashlib
        from synccraft.world.inverse_inference import (
            AssetProvenance,
            LicenseGateError,
            build_quarantined_research_nc_dataset,
            ingest_external_asset,
            scan_text_for_noai_markers,
            verify_asset_provenance,
        )

        minimal_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [{
                "description": {"identifier": "geometry.test.sprout", "texture_width": 32, "texture_height": 32},
                "bones": [{"name": "body", "pivot": [0, 4, 0], "cubes": [{"origin": [-2, 2, -2], "size": [4, 4, 4], "uv": [0, 0]}]}],
            }],
        }
        lic_sha = hashlib.sha256(b"Creative Commons Attribution-NonCommercial 3.0 Unported").hexdigest()
        ev_sha = hashlib.sha256(b"README checked: zero NoAI/TDM opt-out markers").hexdigest()

        # 1. Cobblemon's CC-BY-NC-3.0 and CC-BY-NC-SA-3.0 succeed under research_nc and fail under clean_room_permissive
        for nc_spdx in ("CC-BY-NC-3.0", "CC-BY-NC-SA-3.0"):
            cobble_prov = AssetProvenance(
                asset_id=f"critter_{nc_spdx.lower().replace('-', '_').replace('.', '_')}",
                source_url="https://gitlab.com/cable-mc/cobblemon",
                author="Cobblemon Contributors",
                license_spdx=nc_spdx,
                attribution_notice=f"Cobblemon Contributors ({nc_spdx})",
                title="Sprout Critter",
                license_url="https://creativecommons.org/licenses/by-nc/3.0/",
                license_text_sha256=lic_sha,
                retrieved_at_iso="2026-11-14T18:00:00Z",
                ai_use_marker_status="checked_none",
                ai_marker_evidence_sha256=ev_sha,
                is_ai_generated=False,
                changes_made_note="Decompiled to Block-DSL",
                reviewer_id="bedrock-surface",
            )
            with self.assertRaises(LicenseGateError):
                verify_asset_provenance(cobble_prov, license_policy="clean_room_permissive")

            rec_nc = ingest_external_asset(minimal_geo, "bedrock_geojson", cobble_prov, license_policy="research_nc")
            self.assertEqual(rec_nc.license_track, "research_nc")
            self.assertFalse(rec_nc.commercial_weights_eligible)
            self.assertEqual(rec_nc.to_dict()["provenance"]["license_text_sha256"], lic_sha)

            manifest = build_quarantined_research_nc_dataset([
                {"payload": minimal_geo, "format_kind": "bedrock_geojson", "provenance": cobble_prov.to_dict()}
            ])
            self.assertFalse(manifest["commercial_weights_eligible"])
            self.assertIn(nc_spdx, manifest["spdx_licenses"])

        # 2. All NoDerivatives (-ND) licenses MUST be strictly rejected under EVERY policy
        for nd_spdx in ("CC-BY-ND-4.0", "CC-BY-NC-ND-3.0", "CC-BY-NC-ND-4.0"):
            nd_prov = AssetProvenance(
                asset_id="nd_asset",
                source_url="https://example.org/nd",
                author="Artist",
                license_spdx=nd_spdx,
                attribution_notice="Artist",
            )
            for pol in ("clean_room_permissive", "research_nc", "local_workspace_unverified"):
                with self.assertRaisesRegex(LicenseGateError, "NoDerivatives"):
                    verify_asset_provenance(nd_prov, license_policy=pol)

        # 3. NoAI / TDM opt-out / ai-generated marker gate
        for phrase in (
            "Tagged #NoAI by creator",
            "Please respect no-ai policy",
            "Strictly no machine learning allowed",
            "Do Not Train on this model",
            "no generative ai permitted",
            "TDM-Reservation: 1",
            "Tagged ai-generated mesh",
        ):
            self.assertGreater(len(scan_text_for_noai_markers(phrase)), 0, msg=phrase)
            tagged_geo = copy.deepcopy(minimal_geo)
            tagged_geo["credit"] = phrase
            perm_prov = AssetProvenance("perm_id", "https://example.org/a", "Author", "CC-BY-4.0")
            with self.assertRaises(LicenseGateError):
                ingest_external_asset(tagged_geo, "bedrock_geojson", perm_prov, license_policy="clean_room_permissive")

        with self.assertRaises(LicenseGateError):
            verify_asset_provenance(
                AssetProvenance(
                    "ai_gen",
                    "https://example.org/a",
                    "Author",
                    "CC-BY-4.0",
                    is_ai_generated=True,
                ),
                license_policy="clean_room_permissive",
            )
        with self.assertRaises(LicenseGateError):
            verify_asset_provenance(
                AssetProvenance(
                    "unchecked_ai",
                    "https://example.org/a",
                    "Author",
                    "CC-BY-4.0",
                    ai_use_marker_status="unchecked",
                ),
                license_policy="clean_room_permissive",
            )

        # 4. Unlabeled assets ("NOASSERTION" / blank) are NEVER stamped CC-BY-4.0; rejected for training, allowed only in local_workspace_unverified
        unlabeled_prov = AssetProvenance(
            asset_id="local_draft",
            source_url="local://workspace",
            author="local_artist",
            license_spdx="NOASSERTION",
            attribution_notice="Unverified workspace file",
            ai_use_marker_status="unchecked",
        )
        for train_pol in ("clean_room_permissive", "research_nc"):
            with self.assertRaises(LicenseGateError):
                verify_asset_provenance(unlabeled_prov, license_policy=train_pol)
        local_rec = ingest_external_asset(
            minimal_geo,
            "bedrock_geojson",
            unlabeled_prov,
            license_policy="local_workspace_unverified",
        )
        self.assertEqual(local_rec.provenance.license_spdx, "NOASSERTION")
        self.assertEqual(local_rec.license_track, "local_workspace_unverified")
        self.assertFalse(local_rec.commercial_weights_eligible)

    def test_h6_real_format_7_bone_9_cube_creature_import_and_oklab_residual(self):
        import base64
        import io
        from PIL import Image
        from synccraft.audit.sync_audit import audit_scene
        from synccraft.backends.bedrock_gecko_exporter import (
            sync_cube_from_to_bedrock_origin,
            sync_euler_to_bedrock,
            sync_point_to_bedrock,
        )
        from synccraft.dsl.parser import format_program, parse_program
        from synccraft.surface.contracts import linear_srgb_to_oklab, srgb_u8_to_linear
        from synccraft.world.inverse_inference import (
            _HAS_DSL_V32,
            AssetProvenance,
            ingest_external_asset,
        )

        # Build a deterministic 64x64 RGBA texture with hand-painted eyes, stripes, and gradient details
        yy, xx = np.meshgrid(np.arange(64, dtype=np.float64), np.arange(64, dtype=np.float64), indexing="ij")
        atlas_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        atlas_rgba[..., 0] = np.clip(95.0 + 70.0 * np.sin(xx * 0.21) + 25.0 * np.cos(yy * 0.17), 15, 245).astype(np.uint8)
        atlas_rgba[..., 1] = np.clip(150.0 + 45.0 * np.cos(yy * 0.25), 20, 245).astype(np.uint8)
        atlas_rgba[..., 2] = np.clip(125.0 + 55.0 * np.sin((xx + yy) * 0.13), 20, 245).astype(np.uint8)
        atlas_rgba[..., 3] = 255
        # Hand-painted bright gold eye & crimson cheek markings
        atlas_rgba[4:8, 4:8, :3] = np.array([250, 215, 35], dtype=np.uint8)
        atlas_rgba[10:14, 12:20, :3] = np.array([225, 45, 65], dtype=np.uint8)
        png_buf = io.BytesIO()
        Image.fromarray(atlas_rgba, mode="RGBA").save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()
        png_data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

        # 7-bone, 9-cube creature specification in right-handed model space:
        # Fractional sizes: 5.5, 4.5, 1.5, 6.5, 2.5; two 0-thickness ear planes; 1 per-cube rotated horn; symmetric ears & legs
        bones_rh = [
            {
                "name": "torso",
                "parent": None,
                "pivot": [0.0, 6.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "cubes": [
                    {"name": "belly", "from": [-2.75, 3.75, -3.25], "size": [5.5, 4.5, 6.5], "uv": [0, 0]},
                    {"name": "dorsal_ridge", "from": [-0.75, 8.25, -2.25], "size": [1.5, 1.5, 4.5], "uv": [24, 0]},
                ],
            },
            {
                "name": "head",
                "parent": "torso",
                "pivot": [0.0, 8.25, -3.25],
                "rotation": [5.0, 0.0, 0.0],
                "cubes": [
                    {"name": "cranium", "from": [-2.25, 8.25, -7.75], "size": [4.5, 4.5, 4.5], "uv": [0, 14]},
                    {
                        "name": "snout_horn",
                        "from": [-0.75, 10.25, -9.75],
                        "size": [1.5, 2.5, 2.5],
                        "uv": [20, 14],
                        "rotation": [18.0, 0.0, 0.0],
                        "pivot": [0.0, 10.5, -8.5],
                    },
                ],
            },
            {
                "name": "ear_left",
                "parent": "head",
                "pivot": [2.25, 12.25, -5.5],
                "rotation": [0.0, 0.0, 15.0],
                "cubes": [
                    {"name": "ear_blade_l", "from": [2.25, 12.25, -6.75], "size": [0.0, 4.5, 2.5], "uv": [32, 14]},
                ],
            },
            {
                "name": "ear_right",
                "parent": "head",
                "pivot": [-2.25, 12.25, -5.5],
                "rotation": [0.0, 0.0, -15.0],
                "cubes": [
                    {"name": "ear_blade_r", "from": [-2.25, 12.25, -6.75], "size": [0.0, 4.5, 2.5], "uv": [40, 14]},
                ],
            },
            {
                "name": "leg_front_l",
                "parent": "torso",
                "pivot": [2.0, 3.75, -2.0],
                "rotation": [0.0, 0.0, 0.0],
                "cubes": [
                    {"name": "foreleg_l", "from": [1.25, 0.0, -2.75], "size": [1.5, 3.75, 1.5], "uv": [0, 28]},
                ],
            },
            {
                "name": "leg_front_r",
                "parent": "torso",
                "pivot": [-2.0, 3.75, -2.0],
                "rotation": [0.0, 0.0, 0.0],
                "cubes": [
                    {"name": "foreleg_r", "from": [-2.75, 0.0, -2.75], "size": [1.5, 3.75, 1.5], "uv": [8, 28]},
                ],
            },
            {
                "name": "tail",
                "parent": "torso",
                "pivot": [0.0, 6.5, 3.25],
                "rotation": [-10.0, 0.0, 0.0],
                "cubes": [
                    {"name": "tail_plume", "from": [-0.75, 5.75, 3.25], "size": [1.5, 1.5, 5.5], "uv": [18, 28]},
                ],
            },
        ]

        # 1. Build Bedrock 1.12.0 .geo.json + 1.8.0 .animation.json (with C5 left-handed axis conversion)
        bed_bones_json = []
        for b_spec in bones_rh:
            bed_cubes = []
            for c_spec in b_spec["cubes"]:
                c_entry = {
                    "origin": sync_cube_from_to_bedrock_origin(c_spec["from"], c_spec["size"]),
                    "size": list(c_spec["size"]),
                    "uv": list(c_spec["uv"]),
                }
                if "rotation" in c_spec:
                    c_entry["rotation"] = sync_euler_to_bedrock(c_spec["rotation"])
                    c_entry["pivot"] = sync_point_to_bedrock(c_spec["pivot"])
                bed_cubes.append(c_entry)
            b_entry = {
                "name": b_spec["name"],
                "pivot": sync_point_to_bedrock(b_spec["pivot"]),
                "rotation": sync_euler_to_bedrock(b_spec["rotation"]),
                "cubes": bed_cubes,
            }
            if b_spec["parent"] is not None:
                b_entry["parent"] = b_spec["parent"]
            bed_bones_json.append(b_entry)

        bedrock_geo = {
            "format_version": "1.12.0",
            "minecraft:geometry": [{
                "description": {
                    "identifier": "geometry.synccraft.moss_kit",
                    "texture_width": 64,
                    "texture_height": 64,
                },
                "bones": bed_bones_json,
            }],
        }
        bedrock_walk_anim = {
            "format_version": "1.8.0",
            "animations": {
                "animation.moss_kit.walk": {
                    "loop": True,
                    "animation_length": 1.0,
                    "bones": {
                        "torso": {
                            "position": {
                                "0.0": [-0.25, 0.0, 0.0],
                                "0.5": [0.25, 0.35, 0.0],
                                "1.0": [-0.25, 0.0, 0.0],
                            },
                        },
                        "head": {
                            "scale": {
                                "0.0": [1.0, 1.0, 1.0],
                                "0.5": [1.04, 1.05, 1.04],
                                "1.0": [1.0, 1.0, 1.0],
                            },
                        },
                        "tail": {
                            "rotation": {
                                "0.0": {"post": [0.0, -14.0, 0.0], "lerp_mode": "catmullrom"},
                                "0.5": {"post": [6.0, 14.0, 0.0], "lerp_mode": "catmullrom"},
                                "1.0": {"post": [0.0, -14.0, 0.0], "lerp_mode": "catmullrom"},
                            },
                        },
                        "leg_front_l": {
                            "rotation": {
                                "0.0": [-20.0, 0.0, 0.0],
                                "0.5": [20.0, 0.0, 0.0],
                                "1.0": [-20.0, 0.0, 0.0],
                            },
                        },
                        "leg_front_r": {
                            "rotation": {
                                "0.0": [20.0, 0.0, 0.0],
                                "0.5": [-20.0, 0.0, 0.0],
                                "1.0": [20.0, 0.0, 0.0],
                            },
                        },
                    },
                },
            },
        }

        # 2. Build equivalent Blockbench 5.0 .bbmodel (Cobblemon's native source format)
        bb_groups = []
        bb_elements = []
        children_by_parent = {b["name"]: [] for b in bones_rh}
        for b_spec in bones_rh:
            if b_spec["parent"] is not None:
                children_by_parent[b_spec["parent"]].append(b_spec["name"])

        def _make_bb_face_uvs(u0, v0, sz):
            dx = max(0, int(round(sz[0])))
            dy = max(0, int(round(sz[1])))
            dz = max(0, int(round(sz[2])))
            return {
                "north": {"uv": [u0 + dz, v0 + dz, u0 + dz + max(1, dx), v0 + dz + max(1, dy)], "texture": 0},
                "south": {"uv": [u0 + 2 * dz + dx, v0 + dz, u0 + 2 * dz + dx + max(1, dx), v0 + dz + max(1, dy)], "texture": 0},
                "east": {"uv": [u0 + dz + dx, v0 + dz, u0 + 2 * dz + dx, v0 + dz + max(1, dy)], "texture": 0},
                "west": {"uv": [u0, v0 + dz, u0 + dz, v0 + dz + max(1, dy)], "texture": 0},
                "up": {"uv": [u0 + dz, v0, u0 + dz + max(1, dx), v0 + dz], "texture": 0},
                "down": {"uv": [u0 + dz + dx, v0, u0 + dz + 2 * max(1, dx), v0 + dz], "texture": 0},
            }

        bone_spec_map = {b["name"]: b for b in bones_rh}

        def _build_bb_outliner(bname):
            b_sp = bone_spec_map[bname]
            g_uuid = f"grp-{bname}"
            bb_groups.append({
                "uuid": g_uuid,
                "name": bname,
                "origin": list(b_sp["pivot"]),
                "rotation": list(b_sp["rotation"]),
            })
            ch_nodes = []
            for c_sp in b_sp["cubes"]:
                e_uuid = f"elem-{c_sp['name']}"
                f_min = np.asarray(c_sp["from"], dtype=np.float64)
                f_to = f_min + np.asarray(c_sp["size"], dtype=np.float64)
                el = {
                    "uuid": e_uuid,
                    "type": "cube",
                    "name": c_sp["name"],
                    "from": f_min.tolist(),
                    "to": f_to.tolist(),
                    "origin": list(c_sp.get("pivot", (0.5 * (f_min + f_to)).tolist())),
                    "rotation": list(c_sp.get("rotation", [0.0, 0.0, 0.0])),
                    "inflate": 0.0,
                    "faces": _make_bb_face_uvs(c_sp["uv"][0], c_sp["uv"][1], c_sp["size"]),
                }
                bb_elements.append(el)
                ch_nodes.append(e_uuid)
            for sub_b in children_by_parent[bname]:
                ch_nodes.append(_build_bb_outliner(sub_b))
            return {"uuid": g_uuid, "isOpen": True, "children": ch_nodes}

        bbmodel_doc = {
            "meta": {"format_version": "5.0", "model_format": "free", "box_uv": False},
            "name": "moss_kit",
            "resolution": {"width": 64, "height": 64},
            "elements": bb_elements,
            "groups": bb_groups,
            "outliner": [_build_bb_outliner("torso")],
            "textures": [{"name": "moss_kit.png", "id": "0", "source": png_data_url}],
            "animations": [{
                "name": "walk",
                "loop": "loop",
                "length": 1.0,
                "animators": {
                    "grp-torso": {
                        "name": "torso",
                        "type": "bone",
                        "keyframes": [
                            {"channel": "position", "time": 0.0, "interpolation": "linear", "data_points": [{"x": 0.25, "y": 0.0, "z": 0.0}]},
                            {"channel": "position", "time": 0.5, "interpolation": "linear", "data_points": [{"x": -0.25, "y": 0.35, "z": 0.0}]},
                            {"channel": "position", "time": 1.0, "interpolation": "linear", "data_points": [{"x": 0.25, "y": 0.0, "z": 0.0}]},
                        ],
                    },
                    "grp-head": {
                        "name": "head",
                        "type": "bone",
                        "keyframes": [
                            {"channel": "scale", "time": 0.0, "interpolation": "linear", "data_points": [{"x": 1.0, "y": 1.0, "z": 1.0}]},
                            {"channel": "scale", "time": 0.5, "interpolation": "linear", "data_points": [{"x": 1.04, "y": 1.05, "z": 1.04}]},
                            {"channel": "scale", "time": 1.0, "interpolation": "linear", "data_points": [{"x": 1.0, "y": 1.0, "z": 1.0}]},
                        ],
                    },
                    "grp-tail": {
                        "name": "tail",
                        "type": "bone",
                        "keyframes": [
                            {"channel": "rotation", "time": 0.0, "interpolation": "catmullrom", "data_points": [{"x": 0.0, "y": 14.0, "z": 0.0}]},
                            {"channel": "rotation", "time": 0.5, "interpolation": "catmullrom", "data_points": [{"x": -6.0, "y": -14.0, "z": 0.0}]},
                            {"channel": "rotation", "time": 1.0, "interpolation": "catmullrom", "data_points": [{"x": 0.0, "y": 14.0, "z": 0.0}]},
                        ],
                    },
                },
            }],
        }

        prov = AssetProvenance(
            asset_id="moss_kit_creature",
            source_url="https://gitlab.com/cable-mc/cobblemon",
            author="Cobblemon Contributors",
            license_spdx="CC-BY-NC-3.0",
            attribution_notice="Cobblemon Contributors (CC-BY-NC-3.0)",
        )

        rec_bed = ingest_external_asset(
            bedrock_geo,
            "bedrock_geojson",
            prov,
            atlas_png=png_bytes,
            animation_payload=bedrock_walk_anim,
            license_policy="research_nc",
        )
        rec_bb = ingest_external_asset(
            bbmodel_doc,
            "bbmodel",
            prov,
            license_policy="research_nc",
        )

        for label, rec in (("bedrock_geojson", rec_bed), ("bbmodel", rec_bb)):
            sc = compile_geometry(rec.program)
            self.assertEqual(len(sc.boxes), 9, msg=label)
            # Symmetric L/R pairs (ear_left/ear_right and leg_front_l/leg_front_r) detected via mirror_name
            mirrored_bones = [b for b in sc.bones if b.mirrored]
            self.assertGreaterEqual(len(mirrored_bones), 2, msg=label)

            # Verify OKLab residual summary + exact residual texel reconstruction of artist-painted details
            rstats = rec.oklab_residual_stats
            self.assertIn("residual_oklab_texels", rstats)
            self.assertIn("procedural_oklab_texels", rstats)
            self.assertEqual(rstats["residual_canvas_shape"], (64, 64, 3))
            self.assertGreater(rstats["active_texel_count"], 0)
            self.assertGreater(rstats["max_delta_e"], 0.01)

            mask = rstats["active_uv_mask"]
            recon_lab = rstats["procedural_oklab_texels"][mask] + rstats["residual_oklab_texels"][mask]
            obs_lab_full = linear_srgb_to_oklab(srgb_u8_to_linear(atlas_rgba[..., :3]))
            np.testing.assert_allclose(recon_lab, obs_lab_full[mask], atol=1e-5)

            if _HAS_DSL_V32:
                self.assertLess(rec.vertex_reconstruction_error, 1e-4, msg=label)
                self.assertEqual(rec.program.meta.audit_profile, "authored")
                self.assertEqual(rec.program.meta.size_step, 0.25)
                # Per-cube rotation uses Box.rotate directly without synthesizing extra dummy bones
                self.assertEqual(len(sc.bones), 7, msg=label)
                # Two 0-thickness ear planes preserved with exact 0.0 normal thickness
                planar_boxes = [bx for bx in sc.boxes if any(abs(float(s)) < 1e-9 for s in bx.size)]
                self.assertEqual(len(planar_boxes), 2, msg=label)
                # @textures populated & DSL roundtrips cleanly
                self.assertGreaterEqual(len(rec.program.textures), 1, msg=label)
                rt_prog = parse_program(format_program(rec.program))
                self.assertEqual(len(compile_geometry(rt_prog).boxes), 9)
                # Walk animation preserves position, scale, and catmullrom rotation keyframes
                assert rec.program.behavior is not None
                kfs_clip = rec.program.behavior.clips[0].keyframes
                kf_channels = {(k.bone, k.channel, k.interpolation) for k in kfs_clip}
                self.assertIn(("torso", "position", "linear"), kf_channels)
                self.assertIn(("head", "scale", "linear"), kf_channels)
                self.assertIn(("tail", "rotation", "catmullrom"), kf_channels)
                # Audit PASSES under authored profile
                aud = audit_scene(sc)
                self.assertTrue(aud["passed"], msg=f"{label} audit failed: {aud}")


if __name__ == "__main__":
    unittest.main()
