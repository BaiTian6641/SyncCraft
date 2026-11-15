"""glTF 2.0 Metallic-Roughness PBR Exporter with KHR_materials_iridescence (P1-03c).

Maintains strict separation from LabPBR 1.3:
- ``baseColorTexture``: sRGB baseColor (RGB) + linear opacity (A)
- ``normalTexture``: OpenGL tangent-space normal (+X right, +Y up, +Z out in B)
- ``occlusionTexture`` / ``metallicRoughnessTexture``: standard ORM packing
  (R = cavity AO, G = perceptual roughness, B = metalness)
- ``emissiveTexture``: linear emissive * baseColor encoded as sRGB
- Node hierarchy shares SyncCraft's exact right-handed SE(3) chain.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np

from synccraft.backends.common import FACES, stable_uuid, validate_scene_for_export, vector
from synccraft.kinematics.se3_chain import euler_xyz_matrix
from synccraft.shading.compiler import (
    encode_rgba_png,
    evaluate_world_optical_pipeline,
    write_padded_island,
)
from synccraft.shading.uv_packer import allocate_uv_layout
from synccraft.surface.contracts import WorldState, linear_to_srgb_u8
from synccraft.surface.domain import FACE_AXES_BOX, face_uv_extents_units


def rotation_matrix_to_quaternion_xyzw(rot: np.ndarray) -> list[float]:
    """Convert proper 3x3 orthonormal rotation matrix to glTF unit quaternion [x, y, z, w]."""
    r = np.asarray(rot, dtype=np.float64)
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(1e-12, float(np.linalg.norm(q)))
    return vector(q)


def export_gltf_pbr(
    world_state: WorldState,
    *,
    name: str = "SyncCraft_WorldAsset",
    uv_layout: dict[str, Any] | None = None,
    max_width: int = 512,
    padding: int = 1,
) -> dict[str, Any]:
    """Export ``WorldState`` to a complete glTF 2.0 PBR JSON document + texture bundle."""
    _, posed_scene, _, _, pixel_optics = evaluate_world_optical_pipeline(world_state)
    validate_scene_for_export(posed_scene)

    layout = uv_layout or allocate_uv_layout(
        world_state.program,
        posed_scene,
        max_width=max_width,
        padding=padding,
    )
    width, height = int(layout["width"]), int(layout["height"])
    pad = int(layout.get("padding", padding))

    base_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    normal_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    normal_canvas[..., 0] = 128
    normal_canvas[..., 1] = 128
    normal_canvas[..., 2] = 255
    normal_canvas[..., 3] = 255
    orm_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    orm_canvas[..., 0] = 255
    orm_canvas[..., 1] = 128
    orm_canvas[..., 3] = 255
    emissive_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    emissive_canvas[..., 3] = 255

    max_irid_nm = 0.0
    mean_irid_ior = 1.45
    has_emissive = False

    for fk, patch in pixel_optics.patches.items():
        box_name, face = fk
        u0, v0, u1, v1 = layout["faces"][box_name][face]

        # BaseColor (sRGB + opacity)
        bc_rgb = linear_to_srgb_u8(patch.base_color_linear)
        bc_a = np.rint(np.clip(patch.opacity, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None]
        write_padded_island(base_canvas, int(u0), int(v0), np.concatenate((bc_rgb, bc_a), axis=-1), pad)

        # OpenGL Tangent Normal (+X right, +Y up, +Z out)
        tn = patch.tangent_normal
        n_rgb = np.rint(np.clip(tn * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
        n_a = np.full(tn.shape[:2] + (1,), 255, dtype=np.uint8)
        write_padded_island(normal_canvas, int(u0), int(v0), np.concatenate((n_rgb, n_a), axis=-1), pad)

        # glTF ORM (R=Occlusion, G=PerceptualRoughness, B=Metalness)
        occ_u8 = np.rint(np.clip(patch.cavity_ao, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgh_u8 = np.rint(np.clip(patch.perceptual_roughness, 0.0, 1.0) * 255.0).astype(np.uint8)
        mtl_u8 = np.rint(np.clip(patch.metalness, 0.0, 1.0) * 255.0).astype(np.uint8)
        orm_a = np.full_like(occ_u8, 255)
        write_padded_island(orm_canvas, int(u0), int(v0), np.stack((occ_u8, rgh_u8, mtl_u8, orm_a), axis=-1), pad)

        # Emissive
        if np.any(patch.emissive > 1e-4):
            has_emissive = True
        em_rgb = linear_to_srgb_u8(np.clip(patch.emissive[..., None] * patch.base_color_linear, 0.0, 1.0))
        write_padded_island(emissive_canvas, int(u0), int(v0), np.concatenate((em_rgb, orm_a[..., None]), axis=-1), pad)

        peak_irid = float(patch.iridescence_thickness_nm.max())
        if peak_irid > max_irid_nm:
            max_irid_nm = peak_irid
            mean_irid_ior = float(patch.iridescence_ior.mean())

    base_png = encode_rgba_png(base_canvas)
    normal_png = encode_rgba_png(normal_canvas)
    orm_png = encode_rgba_png(orm_canvas)
    emissive_png = encode_rgba_png(emissive_canvas)

    # Build binary geometry buffer per bone (grouping all boxes of a bone into one mesh)
    boxes_by_bone: dict[str, list[Any]] = {b.name: [] for b in posed_scene.bones}
    for box in posed_scene.boxes:
        boxes_by_bone[box.bone_name].append(box)

    bin_chunks: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    bone_mesh_idx: dict[str, int] = {}

    def _append_accessor(arr: np.ndarray, comp_type: int, acc_type: str, target: int | None) -> int:
        raw = arr.tobytes()
        pad_len = (4 - (len(raw) % 4)) % 4
        offset = sum(len(c) for c in bin_chunks)
        bin_chunks.append(raw + (b"\x00" * pad_len))
        bv: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)

        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": comp_type,
            "count": int(arr.shape[0]),
            "type": acc_type,
        }
        if acc_type == "VEC3":
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    for bone in posed_scene.bones:
        b_boxes = boxes_by_bone[bone.name]
        if not b_boxes:
            continue
        pos_list: list[np.ndarray] = []
        nrm_list: list[np.ndarray] = []
        tan_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        v_base = 0

        for box in b_boxes:
            half_ext = np.maximum(0.0, np.asarray(box.size, dtype=np.float64) / 2.0 + float(box.inflate))
            loc_ctr = np.asarray(box.local_center, dtype=np.float64)
            r_box_in_bone = np.asarray(bone.rotation, dtype=np.float64).T @ np.asarray(box.rotation, dtype=np.float64)
            wu_map = {f: face_uv_extents_units(box.size, f) for f in FACES}

            for face in FACES:
                t_r_raw, t_d_raw, n_out_raw = (np.asarray(vec, dtype=np.float64) for vec in FACE_AXES_BOX[face])
                wu, hv = wu_map[face]
                hw = max(0.0, 0.5 * float(wu) + float(box.inflate))
                hh = max(0.0, 0.5 * float(hv) + float(box.inflate))
                n_dist = float(np.abs(n_out_raw) @ half_ext)
                t_r = r_box_in_bone @ t_r_raw
                t_d = r_box_in_bone @ t_d_raw
                n_out = r_box_in_bone @ n_out_raw

                # CCW quad corners along +n_out: top-left, bottom-left, bottom-right, top-right
                c0 = loc_ctr - hw * t_r - hh * t_d + n_dist * n_out
                c1 = loc_ctr - hw * t_r + hh * t_d + n_dist * n_out
                c2 = loc_ctr + hw * t_r + hh * t_d + n_dist * n_out
                c3 = loc_ctr + hw * t_r - hh * t_d + n_dist * n_out
                pos_list.append(np.stack((c0, c1, c2, c3), axis=0))
                nrm_list.append(np.repeat(n_out[None, :], 4, axis=0))
                tan_vec = np.array([t_r[0], t_r[1], t_r[2], 1.0], dtype=np.float64)
                tan_list.append(np.repeat(tan_vec[None, :], 4, axis=0))

                u0, v0, u1, v1 = layout["faces"][box.name][face]
                uv_quad = np.array(
                    [
                        [u0 / width, v0 / height],
                        [u0 / width, v1 / height],
                        [u1 / width, v1 / height],
                        [u1 / width, v0 / height],
                    ],
                    dtype=np.float64,
                )
                uv_list.append(uv_quad)
                idx_list.append(np.array([v_base, v_base + 1, v_base + 2, v_base, v_base + 2, v_base + 3], dtype=np.uint16))
                v_base += 4

        pos_arr = np.concatenate(pos_list, axis=0).astype(np.float32)
        nrm_arr = np.concatenate(nrm_list, axis=0).astype(np.float32)
        tan_arr = np.concatenate(tan_list, axis=0).astype(np.float32)
        uv_arr = np.concatenate(uv_list, axis=0).astype(np.float32)
        idx_arr = np.concatenate(idx_list, axis=0).astype(np.uint16)

        acc_pos = _append_accessor(pos_arr, 5126, "VEC3", 34962)
        acc_nrm = _append_accessor(nrm_arr, 5126, "VEC3", 34962)
        acc_tan = _append_accessor(tan_arr, 5126, "VEC4", 34962)
        acc_uv = _append_accessor(uv_arr, 5126, "VEC2", 34962)
        acc_idx = _append_accessor(idx_arr, 5123, "SCALAR", 34963)

        mesh_idx = len(meshes)
        meshes.append(
            {
                "name": f"{bone.name}_mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": acc_pos,
                            "NORMAL": acc_nrm,
                            "TANGENT": acc_tan,
                            "TEXCOORD_0": acc_uv,
                        },
                        "indices": acc_idx,
                        "material": 0,
                    }
                ],
            }
        )
        bone_mesh_idx[bone.name] = mesh_idx

    # Assemble SE(3) node hierarchy
    bone_index_map = {b.name: idx for idx, b in enumerate(posed_scene.bones)}
    children_by_parent: dict[str | None, list[int]] = {None: []}
    for b in posed_scene.bones:
        children_by_parent[b.name] = []
    for idx, b in enumerate(posed_scene.bones):
        children_by_parent[b.parent_name].append(idx)

    nodes: list[dict[str, Any]] = []
    for b in posed_scene.bones:
        loc_rot = euler_xyz_matrix(b.local_euler)
        node_obj: dict[str, Any] = {
            "name": b.name,
            "translation": vector(b.local_translation),
            "rotation": rotation_matrix_to_quaternion_xyzw(loc_rot),
            "extras": {"uuid": stable_uuid("gltf_node", b.name, name)},
        }
        if b.name in bone_mesh_idx:
            node_obj["mesh"] = bone_mesh_idx[b.name]
        if children_by_parent[b.name]:
            node_obj["children"] = children_by_parent[b.name]
        nodes.append(node_obj)

    full_bin = b"".join(bin_chunks)
    images = [
        {"name": "base_color", "mimeType": "image/png", "uri": "data:image/png;base64," + base64.b64encode(base_png).decode("ascii")},
        {"name": "normal_gl", "mimeType": "image/png", "uri": "data:image/png;base64," + base64.b64encode(normal_png).decode("ascii")},
        {"name": "orm", "mimeType": "image/png", "uri": "data:image/png;base64," + base64.b64encode(orm_png).decode("ascii")},
        {"name": "emissive", "mimeType": "image/png", "uri": "data:image/png;base64," + base64.b64encode(emissive_png).decode("ascii")},
    ]
    # Nearest-neighbor pixel-art magnification filter (9728 = NEAREST)
    samplers = [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}]
    textures = [{"sampler": 0, "source": i} for i in range(4)]

    extensions_used: list[str] = ["KHR_materials_ior"]
    mat_extensions: dict[str, Any] = {"KHR_materials_ior": {"ior": 1.5}}
    if max_irid_nm > 1.0:
        extensions_used.append("KHR_materials_iridescence")
        mat_extensions["KHR_materials_iridescence"] = {
            "iridescenceFactor": 1.0,
            "iridescenceIor": round(mean_irid_ior, 4),
            "iridescenceThicknessMinimum": 100.0,
            "iridescenceThicknessMaximum": round(max_irid_nm, 2),
        }

    material_def: dict[str, Any] = {
        "name": f"{name}_material",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 2},
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        },
        "normalTexture": {"index": 1, "scale": 1.0},
        "occlusionTexture": {"index": 2, "strength": 1.0},
        "emissiveTexture": {"index": 3},
        "emissiveFactor": [1.0, 1.0, 1.0] if has_emissive else [0.0, 0.0, 0.0],
        "extensions": mat_extensions,
    }

    gltf_doc = {
        "asset": {
            "version": "2.0",
            "generator": "SyncCraft v3.0 Stateful Surface Compiler (P1-03c)",
        },
        "extensionsUsed": extensions_used,
        "scene": 0,
        "scenes": [{"name": name, "nodes": children_by_parent[None]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [material_def],
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [
            {
                "byteLength": len(full_bin),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(full_bin).decode("ascii"),
            }
        ],
    }

    return {
        "standard": "glTF_2.0_PBR",
        "gltf": gltf_doc,
        "textures_png": {
            "base_color.png": base_png,
            "normal_gl.png": normal_png,
            "orm.png": orm_png,
            "emissive.png": emissive_png,
        },
        "arrays": {
            "base_color": base_canvas,
            "normal_gl": normal_canvas,
            "orm": orm_canvas,
            "emissive": emissive_canvas,
        },
        "binary_buffer": full_bin,
        "bone_index_map": bone_index_map,
    }
