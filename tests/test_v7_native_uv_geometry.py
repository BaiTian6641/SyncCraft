"""Independent image-point to spatial-point checks for native painted exports.

Oracle follows Blockbench CubeFace.UVToLocal and Bedrock parseCube at
JannisX11/blockbench ec0b2827104d0799c153bfb5f867579c0c44652f.
"""
import base64
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
from synccraft.dsl import parse_program, format_program
from synccraft.kinematics import compile_geometry
from synccraft.backends import export_bbmodel, export_bedrock

FACES=('north','south','east','west','up','down')
SOURCE='''
@meta { archetype:block_multipart, target_profile:bbmodel_5, texel_density:32, dsl_version:"4.0", audit_profile:sculpted }
@textures { painted:image("alpha.png",64,64) }
@palette { p:ramp(base_lch=[0.5,0.08,220],steps=5) }
@component Base {
 frame socket_l { pos:[10,9,0] }
 frame socket_r { pos:[-10,9,0] }
 box cube { size:[2,2,2], material:p, texture:painted, uv_faces:{ north:[0,0,4,4], south:[4,0,8,4], east:[8,0,12,4], west:[12,0,16,4], up:[16,0,20,4], down:[20,0,24,4] } }
}
@component Fin {
 frame mount { pos:[0,0,0] }
 plane face { size:[8,6], normal:z, material:p, rotate:[20,31,7], texture:painted, uv_faces:{north:[32,8,16,20],south:[0,8,16,20]} }
}
@assembly Test {
 instance root:Base() {}
 instance fin:Fin() mirror_x(fin_r) {}
 connect fin.mount to root.socket_l { type:fixed, preserve_bone:true }
}
'''

def rotation_xyz(a):
 x,y,z=np.deg2rad(a);cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def image_to_world(e,face,points):
 f=np.array(e['from']);t=np.array(e['to']);uv=np.array(e['faces'][face]['uv'])
 xy=(np.asarray(points)-uv[:2])/(uv[2:]-uv[:2])
 for _ in range(int(e['faces'][face].get('rotation',0))//90):xy=np.column_stack((1-xy[:,1],xy[:,0]))
 x,y=xy.T;out=np.broadcast_to(f,(len(x),3)).copy();lerp=lambda a,b,k:a+(b-a)*k
 if face=='east':out[:,0]=t[0];out[:,1]=lerp(t[1],f[1],y);out[:,2]=lerp(t[2],f[2],x)
 if face=='west':out[:,1]=lerp(t[1],f[1],y);out[:,2]=lerp(f[2],t[2],x)
 if face=='up':out[:,1]=t[1];out[:,2]=lerp(f[2],t[2],y);out[:,0]=lerp(f[0],t[0],x)
 if face=='down':out[:,2]=lerp(t[2],f[2],y);out[:,0]=lerp(f[0],t[0],x)
 if face=='south':out[:,2]=t[2];out[:,1]=lerp(t[1],f[1],y);out[:,0]=lerp(f[0],t[0],x)
 if face=='north':out[:,1]=lerp(t[1],f[1],y);out[:,0]=lerp(t[0],f[0],x)
 o=np.array(e['origin']);return (out-o)@rotation_xyz(e['rotation']).T+o

def import_bed(c):
 size=np.array(c['size']);f=np.array(c['origin']);f[0]=-(f[0]+size[0]);p=np.array(c.get('pivot',[0,0,0]));p[0]*=-1
 faces={}
 for face,x in c['uv'].items():
  uv=np.r_[x['uv'],np.array(x['uv'])+x['uv_size']]
  if face in ('up','down'):uv=uv[[2,3,0,1]]
  faces[face]={'uv':uv,'rotation':x.get('uv_rotation',0)}
 return {'from':f,'to':f+size,'origin':p,'rotation':np.array(c.get('rotation',[0,0,0]))*[-1,-1,1],'faces':faces}

class NativePaintedUVGeometryTests(unittest.TestCase):
 def test_asymmetric_two_sided_mirrored_alpha_shape_in_raw_exports(self):
  p=parse_program(SOURCE);s=compile_geometry(p)
  # Deliberately asymmetric triangle, with separate front and back atlas charts.
  q=np.array([[.1,.2],[.9,.35],[.3,.85]])
  image=Image.new('RGBA',(64,64),(0,0,0,0));draw=ImageDraw.Draw(image)
  for x0 in (0,16):draw.polygon([(x0+u*16,8+(1-v)*12) for u,v in q],fill=(180,210,225,255))
  buffer=io.BytesIO();image.save(buffer,format='PNG')
  faces={}
  for b in s.boxes:
   uv=dict(b.uv_faces);fallback=next(iter(uv.values()));faces[b.name]={f:list(uv.get(f,fallback)) for f in FACES}
  atlas={'width':64,'height':64,'faces':faces,'data_url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}
  bb=export_bbmodel(s,name='Triangle',program=p,atlas=atlas)
  bed=export_bedrock(s,program=p,atlas=atlas,profile='bedrock_entity')
  bbe={e['name']:e for e in bb['elements']}
  bedb={b['name']:b for b in bed['geometry']['minecraft:geometry'][0]['bones']}
  local=np.column_stack(((q[:,0]-.5)*8,(q[:,1]-.5)*6,np.zeros(3)))
  expected=local@rotation_xyz([20,31,7]).T+[10,9,0]
  for name,sign in [('fin',1),('fin_r',-1)]:
   want=expected*np.array([sign,1,1]);bc=import_bed(bedb[name]['cubes'][0])
   for face,x0 in [('south',0),('north',16)]:
    # PNG rows increase down; the spatial authoring coordinates increase up.
    points=np.column_stack((x0+q[:,0]*16,8+(1-q[:,1])*12))
    np.testing.assert_allclose(image_to_world(bbe[name+'__face'],face,points),want,atol=1e-6)
    np.testing.assert_allclose(image_to_world(bc,face,points),want,atol=1e-6)
  for face in FACES:
   np.testing.assert_allclose(import_bed(bedb['root']['cubes'][0])['faces'][face]['uv'],bbe['root__cube']['faces'][face]['uv'])
  self.assertEqual(base64.b64decode(bb['textures'][0]['source'].split(',')[1]),buffer.getvalue())
  again=compile_geometry(parse_program(format_program(p)))
  self.assertEqual([b.uv_faces for b in s.boxes],[b.uv_faces for b in again.boxes])

 def test_bedrock_explicit_uv_without_atlas_keeps_top_bottom(self):
  p=parse_program(SOURCE);s=compile_geometry(p)
  bed=export_bedrock(s,program=p,profile='bedrock_entity')
  cube=bed['geometry']['minecraft:geometry'][0]['bones'][0]['cubes'][0]
  imported=import_bed(cube)
  for face in ('up','down'):
   np.testing.assert_allclose(imported['faces'][face]['uv'],s.box_by_name['root__cube'].uv_faces[face])

if __name__=='__main__':unittest.main()
