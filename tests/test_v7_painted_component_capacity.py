"""Focused native V4 regressions for the painted experimental creature corpus."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from synccraft.dsl import parse_program, format_program, DSLValidationError
from synccraft.dsl import parser
from synccraft.kinematics import compile_geometry
from synccraft.backends import export_bbmodel, export_bedrock

SOURCE = '''
@meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 32, dsl_version: "4.0", size_step: 0.5, audit_profile: sculpted }
@textures { painted: image("atlas.png", 64, 64) }
@palette { p: ramp(base_lch=[0.5, 0.08, 220], steps=6) }
@component Body {
 material_slots: [m]
 frame mount { pos: [0,0,0] }
 frame hand { pos: [5,3,1] }
 box shell { size: [4,4,4], material: m, texture: painted, uv: box(2,3), uv_faces: { north: [1,2,9,10,90], south:[10,2,18,10] } }
 bone jaw joint(parent=shell,at=[0,-2,2]) {
  plane lower { size:[3,2], normal:z, material:m, texture:painted, uv_faces: { north:[2,20,8,24], south:[8,20,2,24] } }
 }
}
@component Fin {
 material_slots:[m]
 frame mount { pos:[0,0,0] }
 plane blade { size:[4,2], normal:z, material:m, texture:painted, uv_faces:{ north:[16,20,24,24], south:[24,20,16,24] } }
}
@assembly Painted {
 root_pivot:[0,8,0]
 root_instance:root
 instance root : Body() { materials:{m:p} }
 instance fin : Fin() mirror_x(fin_r) { materials:{m:p} }
 connect fin.mount to root.hand { type:fixed, preserve_bone:true }
}
'''

class PaintedComponentTests(unittest.TestCase):
 def test_root_nested_and_mirrored_uv_roundtrip(self):
  program=parse_program(SOURCE);scene=compile_geometry(program)
  b=scene.box_by_name
  self.assertEqual(b['root__shell'].uv,(2.,3.))
  self.assertEqual(b['root__shell'].uv_faces['north'],(1.,2.,9.,10.,90))
  self.assertEqual(b['root__jaw__lower'].texture,'painted')
  self.assertEqual(b['root__jaw__lower'].uv_faces['south'],(8.,20.,2.,24.))
  self.assertEqual(b['fin__blade'].uv_faces['north'],(16.,20.,24.,24.))
  self.assertEqual(b['fin_r__blade'].uv_faces['north'],(24.,20.,16.,24.))
  again=compile_geometry(parse_program(format_program(program)))
  for name,box in b.items():
   self.assertEqual(box.uv_faces,again.box_by_name[name].uv_faces)
   self.assertEqual(box.texture,again.box_by_name[name].texture)
   np.testing.assert_allclose(box.corners(),again.box_by_name[name].corners(),atol=1e-9)
  bb=export_bbmodel(scene,name='Painted',program=program)
  self.assertTrue(all(e['faces']['north']['texture'] is not None for e in bb['elements']))
  bed=export_bedrock(scene,program=program,profile='bedrock_entity')
  self.assertLess(bed['serialized_vertex_max_error'],1e-6)

 def test_undeclared_texture_and_bad_face_rejected(self):
  with self.assertRaises(DSLValidationError):parse_program(SOURCE.replace('texture:painted','texture:missing'))
  with self.assertRaises(DSLValidationError):parse_program(SOURCE.replace('north:[16,20,24,24]','badface:[16,20,24,24]'))

class SourceCapacityTests(unittest.TestCase):
 def test_authorized_limits(self):
  self.assertEqual(parser.MAX_SOURCE_BYTES,16_000_000)
  self.assertEqual(parser.MAX_LEXICAL_TOKENS,2_000_000)

 def test_utf8_byte_boundary_including_comments(self):
  source='//'+'é'*7_999_999
  self.assertEqual(len(source.encode()),16_000_000)
  self.assertEqual(len(parser._tokens(source)),1)
  with self.assertRaisesRegex(parser.DSLParseError,'byte Block-DSL limit'):
   parser._tokens(source+' ')

 def test_token_boundary_excludes_comments_spaces_and_eof(self):
  with patch.object(parser,'MAX_LEXICAL_TOKENS',4):
   self.assertEqual(len(parser._tokens('a b c d // no lexical token\n')),5)
   with self.assertRaisesRegex(parser.DSLParseError,'token Block-DSL limit'):
    parser._tokens('a b c d e')

if __name__=='__main__':unittest.main()
