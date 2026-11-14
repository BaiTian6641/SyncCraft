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


if __name__ == "__main__":
    unittest.main()
