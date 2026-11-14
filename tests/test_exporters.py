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


if __name__ == "__main__":
    unittest.main()
