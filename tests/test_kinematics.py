"""Reference geometry regressions; these are CPU tests, not gradient gates."""

import unittest
from dataclasses import replace

import numpy as np

from synccraft.dsl.parser import parse_program
from synccraft.kinematics.se3_chain import compile_geometry, euler_xyz_matrix


def program_with(rig: str):
    return parse_program(
        "@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }\n"
        "@palette { steel: ramp(base_lch=[0.7,0.02,250], steps=4) }\n"
        "@rig {" + rig + "}"
    )


class KinematicsTests(unittest.TestCase):
    def test_euler_rotation_order_and_proper_rotation(self):
        np.testing.assert_allclose(
            euler_xyz_matrix([90, 0, 90]) @ [0, 1, 0], [0, 0, 1], atol=1e-14
        )
        rotation = euler_xyz_matrix([17, 31, -28])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_root_and_same_bone_place(self):
        scene = compile_geometry(program_with("""
          root bone assembly {
            pivot_local: [8,8,8]
            box grip { size:[2,6,2], material:steel }
            box pommel place(base_box=grip, anchor=[0,-1,0], align=[0,1,0]) {
              size:[4,2,4], material:steel
            }
          }
        """))
        np.testing.assert_allclose(scene.box_by_name["grip"].center, [8, 8, 8])
        np.testing.assert_allclose(scene.box_by_name["pommel"].center, [8, 4, 8])
        self.assertEqual(scene.adjacency, frozenset({("grip", "pommel")}))

    def test_child_joint_and_rotated_primitive_alignment(self):
        scene = compile_geometry(program_with("""
          root bone assembly {
            pivot_local: [8,8,8]
            box base { size:[4,2,2], material:steel }
            bone arm attach(parent_box=base,anchor=[0,1,0],align=[0,-1,0]) {
              rest_euler:[0,0,90]
              box tip { size:[2,4,2], material:steel }
            }
          }
        """))
        bone = scene.bone_by_name["arm"]
        tip = scene.box_by_name["tip"]
        np.testing.assert_allclose(bone.pivot_world, [8, 9, 8])
        np.testing.assert_allclose(tip.local_center, [0, 2, 0])
        np.testing.assert_allclose(tip.center, [6, 9, 8], atol=1e-14)
        contact = tip.center + tip.rotation @ (tip.size / 2 * [0, -1, 0])
        np.testing.assert_allclose(contact, bone.pivot_world, atol=1e-14)

    def test_nested_parent_rotation_is_shared(self):
        scene = compile_geometry(program_with("""
          root bone assembly {
            pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
            bone arm attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0]) {
              rest_euler:[0,0,90]
              box first {size:[2,2,2],material:steel}
              bone end attach(parent_box=first,anchor=[1,0,0],align=[-1,0,0]) {
                rest_euler:[0,90,0]
                box last {size:[2,2,2],material:steel}
              }
            }
          }
        """))
        end = scene.bone_by_name["end"]
        np.testing.assert_allclose(end.pivot_world, [1, 2, 0], atol=1e-14)
        expected = euler_xyz_matrix([0, 0, 90]) @ euler_xyz_matrix([0, 90, 0])
        np.testing.assert_allclose(end.rotation, expected, atol=1e-14)

    def test_mirror_generates_proper_reflected_subtree(self):
        scene = compile_geometry(program_with("""
          root bone assembly {
            pivot_local:[8,8,8]
            box base {size:[4,2,2],material:steel}
            bone left attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0]) mirror_x(right) {
              rest_euler:[13,17,22]
              box wing {size:[4,2,2],material:steel}
              box tip place(base_box=wing,anchor=[1,0,0],align=[-1,0,0]) {
                size:[2,1,1],material:steel
              }
              bone feather attach(parent_box=wing,anchor=[0,1,0],align=[0,-1,0]) {
                rest_euler:[7,11,19]
                box plume {size:[2,3,1],material:steel}
              }
            }
          }
        """))
        s = np.diag([-1.0, 1.0, 1.0])
        reference = np.array([8, 8, 8])
        for original in ("wing", "tip", "plume"):
            left = scene.box_by_name[original]
            right = scene.box_by_name["right__" + original]
            np.testing.assert_allclose(right.center, reference + s @ (left.center - reference), atol=1e-13)
            np.testing.assert_allclose(right.rotation, s @ left.rotation @ s, atol=1e-13)
            self.assertAlmostEqual(np.linalg.det(right.rotation), 1)
            reflected_corners = (left.corners() - reference) @ s + reference
            left_set = {tuple(np.round(v, 10)) for v in reflected_corners}
            right_set = {tuple(np.round(v, 10)) for v in right.corners()}
            self.assertEqual(left_set, right_set)
        self.assertIn(("base", "right__wing"), scene.adjacency)
        self.assertNotIn(("right__wing", "wing"), scene.adjacency)

    def test_detached_attachment_not_exempt(self):
        scene = compile_geometry(program_with("""
          root bone assembly {
            pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
            bone detached_part attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0],offset=[2,0,0],detached=true) {
              box floating {size:[2,2,2],material:steel}
            }
          }
        """))
        self.assertNotIn(("base", "floating"), scene.adjacency)

    def test_direct_ast_calls_cannot_bypass_validation(self):
        program = program_with("""
          root bone assembly { pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
          }
        """)
        invalid_box = replace(program.root.items[0], size=(True, 2, 2))
        invalid_root = replace(program.root, items=[invalid_box])
        with self.assertRaises(ValueError):
            compile_geometry(replace(program, root=invalid_root))


if __name__ == "__main__":
    unittest.main()
