"""src/synccraft/data_gen/mh_dragons_v8/v8_viewer.py

Self-contained Offline WebGL 3D Studio Viewer Builder for the 22-Monster V8 Collection.

Embeds all 22 `render_scene.json` payloads, x32 RGBA `albedo.png` + `emissive.png`
data URLs, and MIT-licensed Three.js r160.1 into a single `viewer/index.html`
requiring zero server, network, or external font.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image

CANONICAL_22_ORDER: tuple[str, ...] = (
    "rathalos",
    "azure_rathalos",
    "silver_rathalos",
    "legiana",
    "shrieking_legiana",
    "rey_dau",
    "arkveld",
    "zinogre",
    "stygian_zinogre",
    "gore_magala",
    "chaotic_gore_magala",
    "ruiner_nergigante",
    "crimson_glow_valstrax",
    "fatalis",
    "white_fatalis",
    "velkhana",
    "zoh_shia",
    "lagiacrus",
    "ivory_lagiacrus",
    "abyssal_lagiacrus",
    "mizutsune",
    "jin_dahaad",
)

HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Monster Hunter — Block-DSL v4.0 Hybrid Masterworks (V8 · 22 Specimens)</title>
<style>
:root{--bg:#0c1117;--panel:#121a22;--line:#29333d;--text:#eaf0f4;--muted:#96a4b2;--accent:#e8a77d;--faint:#657688;--font:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--font);overflow-x:hidden}
button,select,input{font:inherit}button,select{color:var(--text);background:#1b2631;border:1px solid #35414d;border-radius:7px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent);background:#24323e}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
button{padding:8px 12px}button:disabled{opacity:.4;cursor:default}select{padding:9px 30px 9px 11px;max-width:100%}
header{height:89px;display:flex;align-items:center;justify-content:space-between;padding:0 29px;border-bottom:1px solid var(--line);gap:20px}
.wordmark{font-size:11px;letter-spacing:.21em;text-transform:uppercase;font-weight:750;color:var(--accent)}
.heading{font-size:24px;font-weight:600;letter-spacing:-.035em;line-height:1.25;margin-top:4px}
.header-right{display:flex;align-items:center;gap:15px}.caps{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:650}
.badge{display:inline-flex;align-items:center;gap:7px;padding:5px 9px;border:1px solid var(--line);border-radius:20px;font-size:10px;letter-spacing:.08em;white-space:nowrap;color:var(--muted)}
.dot{width:5px;height:5px;border-radius:50%;background:#94bea1;display:inline-block}
main{height:calc(100vh - 89px);min-height:570px;display:grid;grid-template-columns:minmax(0,1fr) 316px}
.stage{position:relative;min-width:0;overflow:hidden;background:radial-gradient(ellipse at 50% 45%,#1b2934 0%,#101923 52%,#0b121a 100%)}
#viewport{position:absolute;inset:0 0 106px}canvas{display:block;width:100%;height:100%;touch-action:none;outline:none}
.stage-label{position:absolute;left:29px;top:25px;pointer-events:none}.stage-label .number{color:var(--accent);font-size:11px;letter-spacing:.18em;font-weight:650}.stage-label h1{margin:5px 0 0;font-size:36px;letter-spacing:-.055em;font-weight:550;line-height:1.15}.stage-label p{margin:8px 0 0;font-size:12px;color:var(--muted);max-width:410px}
.view-controls{position:absolute;left:29px;bottom:123px;display:flex;gap:5px}.view-controls button{padding:6px 9px;font-size:10px;background:#14212cd9;backdrop-filter:blur(10px)}
.view-controls button.active{border-color:var(--accent);color:var(--accent)}
.canvas-note{position:absolute;right:22px;bottom:127px;font-size:10px;color:#778a9b;text-align:right;pointer-events:none}
.timeline{position:absolute;bottom:0;left:0;right:0;height:106px;padding:16px 29px;background:#0e161f;border-top:1px solid var(--line)}
.transport{display:flex;align-items:center;gap:13px}.play{width:37px;height:33px;display:flex;align-items:center;justify-content:center;padding:0;color:var(--accent)}
.clip-title{min-width:110px;font-size:12px;font-weight:600}.timecode{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;color:var(--muted)}
#scrub{flex:1;min-width:30px;accent-color:var(--accent);cursor:pointer;height:18px}.timeline-foot{margin:10px 0 0 50px;display:flex;justify-content:space-between;font-size:10px;color:var(--faint)}
aside{background:var(--panel);border-left:1px solid var(--line);padding:24px 22px 18px;overflow-y:auto}.section{margin-bottom:22px}.section+.section{padding-top:20px;border-top:1px solid var(--line)}
.section h2{margin:0 0 12px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:650}.section select{width:100%}.field{display:flex;align-items:center;justify-content:space-between;margin:11px 0;font-size:12px}.field label{cursor:pointer}.field input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}.field select{width:91px;padding:5px 7px;font-size:11px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:16px 10px}.stat strong{display:block;font-size:21px;letter-spacing:-.04em;font-weight:550;line-height:1.1}.stat span{font-size:10px;color:var(--muted)}
.palette{display:flex;gap:4px;margin-top:14px;flex-wrap:wrap}.swatch{width:17px;height:17px;border-radius:3px;border:1px solid #ffffff18}
.small{color:var(--muted);font-size:11px;line-height:1.7}.help{font-size:11px;color:var(--muted);line-height:1.85}.help kbd{padding:1px 4px;margin-right:3px;border:1px solid #39434e;border-radius:3px;font:10px var(--font);color:#c8d2dc}.reset{width:100%;margin-top:10px;font-size:11px}
.footer{margin-top:20px;font-size:9px;color:#7e8b97;line-height:1.8}.footer details summary{cursor:pointer;color:var(--muted)}.footer pre{font:9px/1.5 var(--font);white-space:pre-wrap;max-height:200px;overflow:auto}
#loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;color:var(--muted);background:#101923dd;font-size:12px;z-index:4}
#error{display:none;position:absolute;left:30px;right:30px;top:40%;background:#3d2427;border:1px solid #b27676;padding:20px;border-radius:10px;z-index:7;color:#ffe3e3}
.legend{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted);margin-top:12px}.legend i{display:inline-block;width:19px;height:1px;background:var(--accent)}
</style>
</head>
<body>
<header><div><div class="wordmark">Monster Hunter · SyncCraft Block-DSL v4.0</div><div class="heading">Hybrid 3D Cuboid Masterworks · V8 (22 Specimens)</div></div><div class="header-right"><span class="badge"><span class="dot"></span> OFFLINE STUDIO</span><label class="caps" for="modelSelect">Specimen</label><select id="modelSelect" aria-label="Choose creature"></select></div></header>
<main>
 <section class="stage" aria-label="Interactive creature preview">
  <div id="viewport"></div>
  <div class="stage-label"><div class="number" id="specimenNumber">SPECIMEN 01</div><h1 id="modelTitle">Loading</h1><p id="modelTraits"></p></div>
  <div class="view-controls" aria-label="Camera views"><button data-view="quarter" class="active">3/4</button><button data-view="front">Front</button><button data-view="side">Side</button><button data-view="back">Back</button><button data-view="top">Top</button></div>
  <div class="canvas-note">Drag to orbit · scroll to zoom<br>Shift + drag to pan</div>
  <div id="loading">Preparing V8 Hybrid 3D Cuboid + x32 Cutout geometry…</div><div id="error" role="alert"></div>
  <div class="timeline"><div class="transport"><button id="play" class="play" aria-label="Pause animation" title="Play / pause (Space)">Ⅱ</button><span class="clip-title" id="clipLabel">Idle</span><span class="timecode" id="currentTime">0.00</span><input id="scrub" type="range" min="0" max="1000" value="0" step="1" aria-label="Animation timeline"><span class="timecode" id="durationTime">/ 0.00 s</span></div><div class="timeline-foot"><span id="loopLabel">Looping clip</span><span>×32 · 32 texels / block · Native @component + @assembly</span></div></div>
 </section>
 <aside>
  <div class="section"><h2>Animation (12 Clips / Monster)</h2><select id="clipSelect" aria-label="Choose animation"></select><div class="field"><label for="speed">Playback speed</label><select id="speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></div><div class="legend"><i></i><span>Sampled from explicit Block-DSL v4.0 @behavior FK curves</span></div></div>
  <div class="section"><h2>Hybrid Blockbench Geometry</h2><div class="stats"><div class="stat"><strong id="cuboidCount">—</strong><span>3D cuboid boxes</span></div><div class="stat"><strong id="planeCount">—</strong><span>RGBA cutout cards</span></div><div class="stat"><strong id="boneCount">—</strong><span>anatomical bones</span></div><div class="stat"><strong id="atlasSize">—</strong><span>x32 pixel atlas</span></div></div><div class="palette" id="palette"></div></div>
  <div class="section"><h2>View options</h2><div class="field"><label for="clay">Neutral clay material</label><input id="clay" type="checkbox"></div><div class="field"><label for="wireframe">Cutout card outlines</label><input id="wireframe" type="checkbox"></div><div class="field"><label for="grid">Block grid</label><input id="grid" type="checkbox" checked></div><div class="field"><label for="floor">Ground plane</label><input id="floor" type="checkbox" checked></div><div class="field"><label for="autoOrbit">Turntable</label><input id="autoOrbit" type="checkbox"></div><button class="reset" id="reset">Reset camera <span style="color:var(--muted)">· R</span></button></div>
  <div class="section info-section"><h2>Architecture</h2><p class="small" style="margin:0">Stylized Minecraft Blockbench craft: clean single-cuboid anatomical masses, painterly hue-shifted x32 bevel shading, RGBA pixel-cutout jaw skirts &amp; fur fringes, and 3-bank non-uni-sheet tattered wings.</p><p class="help" style="margin:10px 0 0"><kbd>Space</kbd> Play / pause &nbsp; <kbd>R</kbd> Reset<br><kbd>←</kbd><kbd>→</kbd> Step frame · 1 grid = 16 units</p></div>
  <div class="footer">Authored Monster Hunter Block-DSL v4.0 studies (research_nc).<br>Monster Hunter characters belong to CAPCOM.<br><details><summary>Renderer credits &amp; offline details</summary><p>Three.js r160.1 · MIT license. 100% offline embedded file.</p><pre>__THREE_LICENSE__</pre></details></div>
 </aside>
</main>
<script id="assetData" type="application/json">__ASSETS__</script>
<script>__THREE_JS__</script>
<script>
'use strict';
(()=>{
 const assets=JSON.parse(document.getElementById('assetData').textContent);
 const $=id=>document.getElementById(id), viewport=$('viewport');
 const labels={
  rathalos:{accent:'#e87d67',traits:'King of the Skies · hooked beak · 3-bank flame-scroll wings'},
  azure_rathalos:{accent:'#6db4ea',traits:'Azure subspecies · sapphire carapace · tan flame-scroll wings'},
  silver_rathalos:{accent:'#d8e0ea',traits:'Rare Silver Sol · brushed chrome plates · incandescent blue throat'},
  legiana:{accent:'#8cd8eb',traits:'Apex of Coral Highlands · 4-lobe kite wings · striped rudder tail'},
  shrieking_legiana:{accent:'#b7e8f7',traits:'Hoarfrost variant · rime-encrusted wing spars · white-ice crest'},
  rey_dau:{accent:'#e8c468',traits:'Windward Plains apex · folding railgun horns · blue fulgurite wing-scythes'},
  arkveld:{accent:'#cfd6e6',traits:'The White Wraith · ram cheek-hooks · articulated 4-link wrist chainblades'},
  zinogre:{accent:'#6be0ec',traits:'Lord of Lightning · zyhfisme lupine snout · golden/shockfur crests & tail mace'},
  stygian_zinogre:{accent:'#ec5f78',traits:'Emperor of Hell · obsidian carapace · crimson Dracophage shockfur'},
  gore_magala:{accent:'#b97cf2',traits:'Black Eclipse · 6-limbed posture · unfurling Frenzy feelers & tattered cloak'},
  chaotic_gore_magala:{accent:'#e5c469',traits:'Split-molt aberration · left dark Gore vs right golden Shagaru seraph wing'},
  ruiner_nergigante:{accent:'#d4c6b0',traits:'Devourer of Elders · asymmetric bull horns · metallic iron shoulder spikes'},
  crimson_glow_valstrax:{accent:'#f46c74',traits:'Silver Comet · avian helm · rotating 3-nozzle trident jet manifolds'},
  fatalis:{accent:'#c98b6b',traits:'The Black Dragon · 3-bone S-neck · 4 crown horns & molten Hellfire chest'},
  white_fatalis:{accent:'#ece6df',traits:'Ancestral Elder · branching stag antlers · flowing white beard & crimson chest'},
  velkhana:{accent:'#9ad8f5',traits:'Iceborne Lord · 5-blade frost tiara · crystal wings & scorpion tail-lance'},
  zoh_shia:{accent:'#dfd8eb',traits:'Forbidden Construct · alabaster Wylk seraph wing-shrouds & dark corruption horns'},
  lagiacrus:{accent:'#68c9df',traits:'Lord of the Seas · flared cobra neck-hoods · 6 orange dorsal crystal shockers'},
  ivory_lagiacrus:{accent:'#e2decb',traits:'Terrestrial Conqueror · pearl-ivory hide · violet-blue shock organs'},
  abyssal_lagiacrus:{accent:'#58d5f7',traits:'Hadal Sovereign · trench-black armor · bioluminescent cyan shock crystals'},
  mizutsune:{accent:'#f28ab8',traits:'Bewitching Dancer · 5-petal sakura blossom crest · plush violet kitsune brush tail'},
  jin_dahaad:{accent:'#cfa482',traits:'Iceshard Cliffs Titan · bronze-and-frost cathedral heat-sink radiator lamellae'}
 };
 const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(35,1,.1,10000);
 let renderer;
 try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}catch(error){$('loading').style.display='none';$('error').style.display='block';$('error').textContent='WebGL could not start: '+error.message;return;}
 renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
 renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 viewport.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;
 scene.add(new THREE.HemisphereLight(0xc9e2f5,0x2b2530,1.45));
 const keyLight=new THREE.DirectionalLight(0xffe5cc,2.55);keyLight.position.set(-100,180,130);keyLight.castShadow=true;keyLight.shadow.mapSize.set(2048,2048);keyLight.shadow.bias=-.00025;keyLight.shadow.normalBias=.28;scene.add(keyLight);scene.add(keyLight.target);
 const fill=new THREE.DirectionalLight(0xa4c9ec,.72);fill.position.set(120,75,-100);scene.add(fill);
 let current=null,clip=null,clipName='idle',time=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,modelIndex=0;
 const cache=new Map();let floorMesh=null,gridHelper=null;
 let target=new THREE.Vector3(),radius=180,theta=.68,phi=1.18,defaultRadius=180,currentBounds=null;
 const pointers=new Map();let pinch=0;
 function faceVertices(box){
   if(box.face_vertices)return box.face_vertices;
   const [x0,y0,z0]=box.min,[x1,y1,z1]=box.max;
   return {north:[[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]],south:[[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],east:[[x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1]],west:[[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],up:[[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]],down:[[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]]};
 }
 function faceTexCoords(face,uv,a){
   let corners=[[0,1],[1,1],[1,0],[0,0]];
   if(face==='north'||face==='east')corners=corners.map(([u,v])=>[1-u,v]);
   else if(face==='up')corners=corners.map(([u,v])=>[u,1-v]);
   return corners.map(([u,v])=>[(uv[0]+u*(uv[2]-uv[0]))/a.atlas_width,1-(uv[1]+v*(uv[3]-uv[1]))/a.atlas_height]);
 }
 function textureFromData(url){const tx=new THREE.TextureLoader().load(url);tx.colorSpace=THREE.SRGBColorSpace;tx.magFilter=THREE.NearestFilter;tx.minFilter=THREE.NearestFilter;tx.generateMipmaps=false;return tx;}
 function makeAsset(a){
   const root=new THREE.Group(),boneMap=new Map(),meshes=[],edges=[];
   const material=new THREE.MeshStandardMaterial({map:textureFromData(a.albedo_data_url),roughness:.88,metalness:.04,alphaTest:.5,side:THREE.FrontSide});
   material.userData.clay={value:$('clay').checked?1:0};
   material.onBeforeCompile=shader=>{shader.uniforms.reviewClay=material.userData.clay;shader.fragmentShader='uniform float reviewClay;\n'+shader.fragmentShader;shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>','#include <map_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.46, 0.51, 0.57), reviewClay);');};
   material.customProgramCacheKey=()=> 'v8_hybrid_clay';
   if(a.emissive_data_url){material.emissive=new THREE.Color(0xffffff);material.emissiveMap=textureFromData(a.emissive_data_url);material.emissiveIntensity=.85;}
   for(const b of a.bones){const group=new THREE.Group();group.name=b.name;group.rotation.order='ZYX';group.userData.pivot=new THREE.Vector3(...b.pivot);group.userData.bone=b;boneMap.set(b.name,group);}
   for(const b of a.bones){const g=boneMap.get(b.name),p=boneMap.get(b.parent);g.position.copy(g.userData.pivot);if(p){g.position.sub(p.userData.pivot);p.add(g);}else root.add(g);g.userData.restPosition=g.position.clone();}
   const merged=new Map();
   for(const box of a.boxes){
     if(!boneMap.has(box.bone))throw Error('Missing animation bone '+box.bone);
     if(!merged.has(box.bone))merged.set(box.bone,{position:[],normal:[],uv:[]});
     const dst=merged.get(box.bone),pivot=boneMap.get(box.bone).userData.pivot.toArray(),hidden=new Set(box.hidden_faces||[]);
     for(const [face,points] of Object.entries(faceVertices(box))){
       if(hidden.has(face))continue;
       const uv=box.uv[face];if(!uv)continue;
       const p0=new THREE.Vector3(...points[0]),p1=new THREE.Vector3(...points[1]),p2=new THREE.Vector3(...points[2]);
       const n=new THREE.Vector3().crossVectors(p1.sub(p0),p2.sub(p0));
       if(n.lengthSq()<1e-16)continue;
       const vertices=points.map(p=>p.map((x,k)=>x-pivot[k]));
       const tex=faceTexCoords(face,uv,a),indices=[0,1,2,0,2,3],normal=n.normalize().toArray();
       for(const idx of indices){dst.position.push(...vertices[idx]);dst.normal.push(...normal);dst.uv.push(...tex[idx]);}
     }
   }
   for(const [bone,data] of merged){if(!data.position.length)continue;const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(data.position,3));geo.setAttribute('normal',new THREE.Float32BufferAttribute(data.normal,3));geo.setAttribute('uv',new THREE.Float32BufferAttribute(data.uv,2));geo.computeBoundingSphere();const mesh=new THREE.Mesh(geo,material);mesh.castShadow=true;mesh.receiveShadow=false;mesh.name=bone+'_geometry';boneMap.get(bone).add(mesh);meshes.push(mesh);}
   root.updateMatrixWorld(true);const bounds=new THREE.Box3().setFromObject(root);
   return {doc:a,root,boneMap,material,meshes,edges,bounds};
 }
 function makeEdges(){
   if(current.edges.length)return;
   const material=new THREE.LineBasicMaterial({color:0x9bb6c7,transparent:true,opacity:.45,depthWrite:false}),grouped=new Map();
   for(const box of current.doc.boxes){const p=box.surface_vertices;if(!p)continue;const pivot=current.boneMap.get(box.bone).userData.pivot.toArray();if(!grouped.has(box.bone))grouped.set(box.bone,[]);const a=grouped.get(box.bone);for(let i=0;i<p.length;i++){a.push(...p[i].map((v,k)=>v-pivot[k]),...p[(i+1)%p.length].map((v,k)=>v-pivot[k]));}}
   for(const [bone,positions] of grouped){const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));const edges=new THREE.LineSegments(geo,material);edges.renderOrder=2;current.boneMap.get(bone).add(edges);current.edges.push(edges);}
 }
 function fitCamera(){if(!current)return;currentBounds=current.bounds.clone();target.copy(currentBounds.getCenter(new THREE.Vector3()));const size=currentBounds.getSize(new THREE.Vector3());const aspect=Math.max(.3,viewport.clientWidth/Math.max(1,viewport.clientHeight));const sphereRadius=size.length()*.48;defaultRadius=sphereRadius/Math.sin(THREE.MathUtils.degToRad(camera.fov*.5))*1.02*Math.max(1,1/aspect);radius=defaultRadius;theta=.68;phi=1.18;updateCamera();setViewActive('quarter');}
 function updateCamera(){camera.position.set(target.x+radius*Math.sin(phi)*Math.sin(theta),target.y+radius*Math.cos(phi),target.z+radius*Math.sin(phi)*Math.cos(theta));camera.lookAt(target);camera.near=Math.max(.01,radius/5000);camera.far=Math.max(1000,radius*25);camera.updateProjectionMatrix();}
 function ground(){if(floorMesh){scene.remove(floorMesh);floorMesh.geometry.dispose();floorMesh.material.dispose();}if(gridHelper){scene.remove(gridHelper);gridHelper.geometry.dispose();gridHelper.material.dispose();}const sz=current.bounds.getSize(new THREE.Vector3()),c=current.bounds.getCenter(new THREE.Vector3()),span=Math.ceil(Math.max(sz.x,sz.z)*1.5/16)*16;floorMesh=new THREE.Mesh(new THREE.PlaneGeometry(span,span),new THREE.MeshStandardMaterial({color:0x15212c,roughness:1,metalness:0,transparent:true,opacity:.7}));floorMesh.rotation.x=-Math.PI/2;floorMesh.position.set(c.x,current.bounds.min.y-.14,c.z);floorMesh.receiveShadow=true;scene.add(floorMesh);gridHelper=new THREE.GridHelper(span,span/16,0x596f7f,0x344c5e);gridHelper.position.set(c.x,current.bounds.min.y-.10,c.z);gridHelper.material.transparent=true;gridHelper.material.opacity=.4;scene.add(gridHelper);floorMesh.visible=$('floor').checked;gridHelper.visible=$('grid').checked;const r=sz.length()*.7;keyLight.position.set(c.x-r*.7,c.y+r*1.4,c.z+r);keyLight.target.position.copy(c);keyLight.shadow.camera.left=-r;keyLight.shadow.camera.right=r;keyLight.shadow.camera.top=r;keyLight.shadow.camera.bottom=-r;keyLight.shadow.camera.near=.5;keyLight.shadow.camera.far=r*5;keyLight.shadow.camera.updateProjectionMatrix();}
 function setViewActive(name){document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));}
 function pickView(name){if(name==='quarter'){theta=.68;phi=1.18;}if(name==='front'){theta=0;phi=Math.PI/2;}if(name==='side'){theta=Math.PI/2;phi=Math.PI/2;}if(name==='back'){theta=Math.PI;phi=Math.PI/2;}if(name==='top'){theta=0;phi=.008;}target.copy(current.bounds.getCenter(new THREE.Vector3()));radius=defaultRadius;setViewActive(name);updateCamera();}
 function nice(s){return s.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());}
 function chooseAsset(index){
   modelIndex=index;const a=assets[index];if(current)scene.remove(current.root);if(!cache.has(a.id))cache.set(a.id,makeAsset(a));current=cache.get(a.id);scene.add(current.root);$('modelSelect').value=String(index);
   const label=labels[a.id]||{accent:'#e8a77d',traits:'Canonical Monster Hunter anatomy · x32 pixel atlas'};document.documentElement.style.setProperty('--accent',label.accent);
   $('modelTitle').textContent=(a.title||nice(a.id));$('modelTraits').textContent=label.traits;$('specimenNumber').textContent='SPECIMEN '+String(index+1).padStart(2,'0')+' / '+String(assets.length).padStart(2,'0');
   const nPlanes=a.boxes.filter(b=>b.size.some(s=>s===0)).length, nCuboids=a.boxes.length-nPlanes;
   $('cuboidCount').textContent=nCuboids.toLocaleString();$('planeCount').textContent=nPlanes.toLocaleString();$('boneCount').textContent=a.bones.length;
   $('atlasSize').textContent=a.atlas_width===a.atlas_height?(a.atlas_width+'²'):(a.atlas_width+'×'+a.atlas_height);
   $('palette').replaceChildren();for(const [name,m] of Object.entries(a.materials||{})){const sw=document.createElement('span');sw.className='swatch';sw.style.background='rgb('+m.color.join(',')+')';sw.title=nice(name);$('palette').appendChild(sw);}
   $('clipSelect').replaceChildren();const rest=document.createElement('option');rest.value='__rest';rest.textContent='Rest pose';$('clipSelect').appendChild(rest);for(const name of Object.keys(a.clips||{})){const opt=document.createElement('option');opt.value=name;opt.textContent=nice(name);$('clipSelect').appendChild(opt);}
   selectClip(a.clips&&a.clips.idle?'idle':Object.keys(a.clips||{})[0]||'__rest');$('floor').checked=a.id!=='abyssal_lagiacrus';$('grid').checked=a.id!=='abyssal_lagiacrus';ground();fitCamera();if($('wireframe').checked)makeEdges();for(const e of current.edges)e.visible=$('wireframe').checked;current.material.userData.clay.value=$('clay').checked?1:0;
   window.__viewerState={ready:true,asset:a.id,cuboids:nCuboids,planes:nPlanes,bones:a.bones.length,clip:clipName};$('loading').style.display='none';
 }
 function selectClip(name){clipName=name;clip=current.doc.clips&&current.doc.clips[name]||null;time=0;$('clipSelect').value=name;$('clipLabel').textContent=clip?nice(name):'Rest pose';$('durationTime').textContent='/ '+(clip?clip.duration.toFixed(2):'0.00')+' s';$('loopLabel').textContent=clip?(clip.loop?'Looping clip':'One-shot clip'):'Unposed construction view';$('play').disabled=!clip;$('scrub').disabled=!clip;updatePlayButton();applyPose();updateTimeline();}
 function sampleKeys(keys,t){if(!keys||!keys.length)return [0,0,0];if(t<=keys[0][0])return keys[0][1];if(t>=keys[keys.length-1][0])return keys[keys.length-1][1];let hi=1;while(keys[hi][0]<t)hi++;const lo=hi-1,span=keys[hi][0]-keys[lo][0],u=span>1e-9?(t-keys[lo][0])/span:0;return keys[lo][1].map((v,k)=>v+(keys[hi][1][k]-v)*u);}
 function applyPose(){if(!current)return;for(const b of current.boneMap.values()){b.position.copy(b.userData.restPosition);b.rotation.set(0,0,0,'ZYX');b.scale.set(1,1,1);}if(!clip)return;for(const tr of clip.tracks||[]){const b=current.boneMap.get(tr.bone);if(!b)continue;const v=sampleKeys(tr.keys,time);if(tr.channel==='rotation')b.rotation.set(...v.map(THREE.MathUtils.degToRad),'ZYX');else if(tr.channel==='position')b.position.copy(b.userData.restPosition).add(new THREE.Vector3(...v));else if(tr.channel==='scale')b.scale.set(...v);}current.root.updateMatrixWorld(true);}
 function updateTimeline(){$('scrub').value=clip?String(Math.round(time/clip.duration*1000)):'0';$('currentTime').textContent=time.toFixed(2);}
 function updatePlayButton(){$('play').textContent=playing?'Ⅱ':'▶';}
 function togglePlay(){if(!clip)return;if(!clip.loop&&time>=clip.duration)time=0;playing=!playing;updatePlayButton();}
 for(let i=0;i<assets.length;i++){const o=document.createElement('option');o.value=String(i);o.textContent=(i+1).toString().padStart(2,'0')+' · '+(assets[i].title||nice(assets[i].id));$('modelSelect').appendChild(o);}
 $('modelSelect').onchange=e=>chooseAsset(Number(e.target.value));$('clipSelect').onchange=e=>selectClip(e.target.value);$('play').onclick=togglePlay;
 $('scrub').oninput=e=>{if(!clip)return;playing=false;time=Number(e.target.value)/1000*clip.duration;updatePlayButton();applyPose();updateTimeline();};
 $('wireframe').onchange=()=>{if($('wireframe').checked)makeEdges();for(const e of current.edges)e.visible=$('wireframe').checked;};$('grid').onchange=()=>{gridHelper.visible=$('grid').checked;};$('floor').onchange=()=>{floorMesh.visible=$('floor').checked;};$('reset').onclick=fitCamera;
 $('clay').onchange=()=>{current.material.userData.clay.value=$('clay').checked?1:0;};
 document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>pickView(b.dataset.view));
 function pan(dx,dy){const scale=radius/Math.max(1,viewport.clientHeight)*.8;const right=new THREE.Vector3(),up=new THREE.Vector3();right.setFromMatrixColumn(camera.matrixWorld,0);up.setFromMatrixColumn(camera.matrixWorld,1);target.addScaledVector(right,-dx*scale).addScaledVector(up,dy*scale);}
 renderer.domElement.addEventListener('pointerdown',e=>{renderer.domElement.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pointers.size===2){const pp=[...pointers.values()];pinch=Math.hypot(pp[0].x-pp[1].x,pp[0].y-pp[1].y);}e.preventDefault();});
 renderer.domElement.addEventListener('pointermove',e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId),dx=e.clientX-old.x,dy=e.clientY-old.y;pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pointers.size===2){const pp=[...pointers.values()],dist=Math.hypot(pp[0].x-pp[1].x,pp[0].y-pp[1].y);if(pinch>0)radius=Math.min(defaultRadius*8,Math.max(defaultRadius*.12,radius*pinch/Math.max(1,dist)));pinch=dist;pan(dx*.5,dy*.5);}else if(e.shiftKey||e.button===2||(e.buttons&2)){pan(dx,dy);}else{theta-=dx*.006;phi=Math.min(Math.PI-.025,Math.max(.025,phi-dy*.006));setViewActive('');}updateCamera();});
 for(const ev of ['pointerup','pointercancel'])renderer.domElement.addEventListener(ev,e=>{pointers.delete(e.pointerId);pinch=0;});renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
 renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();radius=Math.min(defaultRadius*8,Math.max(defaultRadius*.12,radius*Math.exp(e.deltaY*.001)));updateCamera();},{passive:false});
 window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key.toLowerCase()==='r')fitCamera();if(clip&&(e.key==='ArrowLeft'||e.key==='ArrowRight')){e.preventDefault();playing=false;time=Math.max(0,Math.min(clip.duration,time+(e.key==='ArrowRight'?1:-1)/24));updatePlayButton();applyPose();updateTimeline();}});
 function resize(){const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/Math.max(1,h);camera.updateProjectionMatrix();}new ResizeObserver(resize).observe(viewport);resize();
 let previous=performance.now(),lastUi=0;
 function frame(now){requestAnimationFrame(frame);const dt=Math.min((now-previous)/1000,.1);previous=now;if(clip&&playing){time+=dt*Number($('speed').value);if(time>clip.duration){if(clip.loop)time%=clip.duration;else{time=clip.duration;playing=false;updatePlayButton();}}applyPose();}if($('autoOrbit').checked){theta+=dt*.16;updateCamera();setViewActive('');}if(now-lastUi>75){updateTimeline();lastUi=now;}renderer.render(scene,camera);}
 try{if(!assets.length)throw Error('No creature scenes were embedded');chooseAsset(0);requestAnimationFrame(frame);}catch(error){$('loading').style.display='none';$('error').style.display='block';$('error').textContent='Display error: '+error.message;console.error(error);}
})();
</script>
</body></html>'''


def build_v8_offline_viewer(
    assets_root: Path,
    viewer_out_dir: Path,
    three_js_path: Path,
    three_license_path: Path,
) -> dict[str, Any]:
    """Embed all exported `render_scene.json` files + Three.js r160.1 into `viewer/index.html`."""
    if not three_js_path.is_file() or not three_license_path.is_file():
        raise FileNotFoundError(f"Missing Three.js vendor files at {three_js_path}")

    order_map = {mid: idx for idx, mid in enumerate(CANONICAL_22_ORDER)}
    scene_paths = sorted(assets_root.glob("*/render_scene.json"))
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []

    for sp in scene_paths:
        doc = json.loads(sp.read_text(encoding="utf-8"))
        tex_p = sp.parent / "textures" / "albedo.png"
        if not tex_p.is_file():
            raise FileNotFoundError(f"Missing {tex_p}")
        if "albedo_data_url" not in doc:
            doc["albedo_data_url"] = "data:image/png;base64," + base64.b64encode(tex_p.read_bytes()).decode("ascii")
        if "atlas_width" not in doc or "atlas_height" not in doc:
            with Image.open(tex_p) as im:
                doc["atlas_width"], doc["atlas_height"] = im.size
        records.append(doc)
        provenance.append(
            {
                "id": doc["id"],
                "scene_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                "albedo_sha256": hashlib.sha256(tex_p.read_bytes()).hexdigest(),
            }
        )

    records.sort(key=lambda d: (order_map.get(d["id"], 999), d["id"]))
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = (
        HTML_TEMPLATE.replace("__ASSETS__", payload)
        .replace("__THREE_JS__", three_js_path.read_text(encoding="utf-8").replace("</script", "<\\/script"))
        .replace("__THREE_LICENSE__", html.escape(three_license_path.read_text(encoding="utf-8")))
    )
    viewer_out_dir.mkdir(parents=True, exist_ok=True)
    index_p = viewer_out_dir / "index.html"
    index_p.write_text(page, encoding="utf-8")
    (viewer_out_dir / "THREE_LICENSE.txt").write_bytes(three_license_path.read_bytes())

    summary = {
        "viewer": "index.html",
        "models_count": len(records),
        "models": [r["id"] for r in records],
        "total_clips": sum(len(r.get("clips", {})) for r in records),
        "self_contained": True,
        "requires_network": False,
        "renderer": "Three.js 0.160.1 (MIT)",
        "html_bytes": index_p.stat().st_size,
        "assets": provenance,
    }
    (viewer_out_dir / "viewer_build.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
