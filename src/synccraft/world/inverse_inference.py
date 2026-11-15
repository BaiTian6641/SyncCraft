"""P1-03e Permissive Internet Asset Ingestion & Inverse Material Posterior Inference.

Supports decompiling external cuboid assets from:
1. Blockbench 5 ``.bbmodel`` (outliner/groups/elements + embedded PNG data URLs)
2. Minecraft Java ``.json`` (elements + texture_size)
3. Minecraft Bedrock ``.geo.json`` (``minecraft:geometry`` bones/cubes)

Enforces a strict permissive-license & provenance gate before ingestion, recovers
validated ``@rig`` ``Program`` ASTs with verified 3D vertex reconstruction, infers
top-K candidate ``MaterialProgram`` hypotheses with Oklab residuals and epistemic
identifiability scores (elevated when LabPBR ``_n``/``_s`` maps are present), and
extracts localized ``StampOverride`` layers for artist-authored markings (eyes, trim).
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import io
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from synccraft.backends.bedrock_gecko_exporter import (
    bedrock_euler_to_sync,
    bedrock_origin_to_sync_cube_from,
    bedrock_point_to_sync,
)
import synccraft.dsl.ast_nodes as _ast_nodes
from synccraft.dsl.ast_nodes import (
    FACES,
    LOCATOR_ROLES,
    Attach,
    BehaviorDecl,
    Bone,
    Box,
    ClipDecl,
    Locator,
    Meta,
    Palette,
    Place,
    Program,
    TrackDecl,
    validate_program,
)
from synccraft.kinematics.se3_chain import compile_geometry, euler_xyz_matrix
from synccraft.surface.contracts import (
    MATERIAL_FAMILIES,
    MaterialProgram,
    MaterialSpec,
    StampOverride,
    linear_srgb_to_oklab,
    linear_srgb_to_oklch,
    srgb_u8_to_linear,
)
from synccraft.surface.domain import (
    build_surface_domain,
    evaluate_posed_surface_kinematics,
    face_uv_extents_units,
    initialize_surface_state,
)
from synccraft.surface.processes import evaluate_optical_fields

_HAS_DSL_V32: bool = "size_step" in Meta.__dataclass_fields__ and "rotate" in Box.__dataclass_fields__
_TextureDecl: Any = getattr(_ast_nodes, "TextureDecl", None)
_KeyframeTrackDecl: Any = getattr(_ast_nodes, "KeyframeTrackDecl", None)

APPROVED_PERMISSIVE_LICENSES: frozenset[str] = frozenset({
    "CC0-1.0",
    "CC-BY-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
})

APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES: frozenset[str] = frozenset({
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-NC-3.0",
    "CC-BY-NC-SA-3.0",
})

LICENSE_POLICIES: frozenset[str] = frozenset({
    "clean_room_permissive",
    "research_nc",
    "local_workspace_unverified",
})

AI_USE_MARKER_STATUSES: frozenset[str] = frozenset({
    "checked_none",
    "opt_in_signed",
    "restricted_noai",
    "unchecked",
})

BANNED_PROVENANCE_TOKENS: tuple[str, ...] = (
    "mojang",
    "minecraft_jar",
    "official_vanilla_jar",
    "all_rights_reserved",
    "proprietary",
    "unlicensed",
)

_ND_LICENSE_RE = re.compile(
    r"(?:^|[-_\s])nd(?:$|[-_\s0-9.])|noderivatives|no[-_\s]+derivatives",
    re.IGNORECASE,
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_NOAI_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("noai", re.compile(r"(?<![a-z0-9])#?no[-_\s]?ai(?![a-z0-9])", re.IGNORECASE)),
    ("no_machine_learning", re.compile(r"\bno\s+machine\s+learning\b|\bno[-_\s]?ml[-_\s]?train(?:ing)?\b", re.IGNORECASE)),
    (
        "do_not_train",
        re.compile(r"\b(?:do\s+not|don'?t)\s+(?:use\s+for\s+)?train(?:ing)?\b|\bno[-_\s]+ai[-_\s]+train(?:ing)?\b", re.IGNORECASE),
    ),
    ("no_generative_ai", re.compile(r"\bno\s+generative\s+ai\b|\bno\s+gen[-_\s]?ai\b", re.IGNORECASE)),
    (
        "tdm_opt_out",
        re.compile(
            r"\btdm[-_\s]?(?:opt[-_\s]?out|reservation\s*[:=]\s*1|policy)\b"
            r"|\btext[-_\s]+and[-_\s]+data[-_\s]+mining\s+opt[-_\s]?out\b"
            r"|\bx-robots-tag\s*:\s*[^\n]*(?:noai|noimageai)\b"
            r"|\bopt[-_\s]?out\s+of\s+ai\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ai_generated",
        re.compile(
            r"(?<![a-z0-9])#?ai[-_\s]generated(?![a-z0-9])|\bgenerated\s+(?:by|with)\s+(?:generative\s+)?ai\b",
            re.IGNORECASE,
        ),
    ),
)


class LicenseGateError(PermissionError):
    """Raised when an external asset lacks an approved license, clean provenance, or AI consent."""


def scan_text_for_noai_markers(text: str) -> tuple[str, ...]:
    """Scan metadata, README, or tag strings for NoAI, TDM opt-out, or AI-generated markers (H7).

    Returns a deterministic tuple of canonical marker categories found (empty if clean).
    """
    if not isinstance(text, str) or not text.strip():
        return ()
    found: list[str] = []
    for label, pattern in _NOAI_MARKER_PATTERNS:
        if pattern.search(text) is not None:
            found.append(label)
    return tuple(found)


def _extract_payload_metadata_text(payload: dict[str, Any]) -> str:
    """Collect textual header/credit/description fields from external JSON payloads for NoAI scanning."""
    if not isinstance(payload, dict):
        return ""
    chunks: list[str] = []
    for k in ("credit", "description", "readme", "license", "tags", "comment", "notes", "author", "title"):
        val = payload.get(k)
        if isinstance(val, str) and val.strip():
            chunks.append(val)
        elif isinstance(val, (list, tuple)):
            chunks.extend(str(x) for x in val if isinstance(x, (str, int, float)))
    meta_sub = payload.get("meta")
    if isinstance(meta_sub, dict):
        for mk, mv in meta_sub.items():
            if isinstance(mv, str) and mk not in ("format_version", "model_format"):
                chunks.append(mv)
    geos = payload.get("minecraft:geometry")
    if isinstance(geos, list):
        for g in geos[:2]:
            if isinstance(g, dict) and isinstance(g.get("description"), dict):
                for dk, dv in g["description"].items():
                    if isinstance(dv, str):
                        chunks.append(dv)
    return " | ".join(chunks)


@dataclass(frozen=True)
class AssetProvenance:
    """Mandatory provenance, license, and AI-consent record for external 3D cuboid assets (H7)."""

    asset_id: str
    source_url: str
    author: str
    license_spdx: str
    attribution_notice: str = ""
    title: str = ""
    license_url: str = ""
    license_text_sha256: str = ""
    retrieved_at_iso: str = ""
    ai_use_marker_status: str = "checked_none"
    ai_marker_evidence_sha256: str = ""
    is_ai_generated: bool = False
    changes_made_note: str = ""
    reviewer_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "source_url": self.source_url,
            "author": self.author,
            "license_spdx": self.license_spdx,
            "attribution_notice": self.attribution_notice,
            "title": self.title,
            "license_url": self.license_url,
            "license_text_sha256": self.license_text_sha256,
            "retrieved_at_iso": self.retrieved_at_iso,
            "ai_use_marker_status": self.ai_use_marker_status,
            "ai_marker_evidence_sha256": self.ai_marker_evidence_sha256,
            "is_ai_generated": self.is_ai_generated,
            "changes_made_note": self.changes_made_note,
            "reviewer_id": self.reviewer_id,
        }


def verify_asset_provenance(
    provenance: AssetProvenance,
    *,
    license_policy: str = "clean_room_permissive",
    metadata_text: str = "",
) -> str:
    """Enforce AGENTS.md / ADR 0003 / ADR 0006 / H7 license, NoDerivatives, and NoAI marker gate.

    Returns the resolved license track: ``'clean_room_permissive'``, ``'research_nc'``, or
    ``'local_workspace_unverified'`` (when ``license_policy='local_workspace_unverified'`` is
    explicitly requested for local unverified artist files marked ``NOASSERTION``).
    Unlabeled assets are NEVER stamped permissive by default.
    """
    if license_policy not in LICENSE_POLICIES:
        raise ValueError(f"Unknown license_policy {license_policy!r}; expected one of {sorted(LICENSE_POLICIES)}")
    if not isinstance(provenance, AssetProvenance):
        raise LicenseGateError("AssetProvenance record is required for external ingestion")
    if not provenance.asset_id.strip() or not provenance.source_url.strip() or not provenance.author.strip():
        raise LicenseGateError("asset_id, source_url, and author must all be non-empty")

    spdx = provenance.license_spdx.strip()
    if not spdx:
        raise LicenseGateError("license_spdx must be non-empty; unlabeled assets cannot default to permissive")

    if type(provenance.is_ai_generated) is not bool:
        raise LicenseGateError("is_ai_generated must be a boolean")
    if provenance.is_ai_generated:
        raise LicenseGateError("AI-generated assets (is_ai_generated=True) are prohibited from ingestion")

    if provenance.ai_use_marker_status not in AI_USE_MARKER_STATUSES:
        raise LicenseGateError(
            f"Invalid ai_use_marker_status {provenance.ai_use_marker_status!r}; "
            f"expected one of {sorted(AI_USE_MARKER_STATUSES)}"
        )
    if provenance.ai_use_marker_status == "restricted_noai":
        raise LicenseGateError("Asset carries ai_use_marker_status='restricted_noai' and is excluded")
    if provenance.ai_use_marker_status == "unchecked" and license_policy != "local_workspace_unverified":
        raise LicenseGateError(
            f"ai_use_marker_status='unchecked' is rejected under policy {license_policy!r}; "
            "verify creator metadata for NoAI/TDM markers before ingestion"
        )

    if provenance.license_text_sha256.strip() and _SHA256_HEX_RE.fullmatch(provenance.license_text_sha256.strip()) is None:
        raise LicenseGateError("license_text_sha256 must be a 64-character hexadecimal SHA-256 digest")
    if (
        provenance.ai_marker_evidence_sha256.strip()
        and _SHA256_HEX_RE.fullmatch(provenance.ai_marker_evidence_sha256.strip()) is None
    ):
        raise LicenseGateError("ai_marker_evidence_sha256 must be a 64-character hexadecimal SHA-256 digest")
    if provenance.retrieved_at_iso.strip() and _ISO8601_RE.fullmatch(provenance.retrieved_at_iso.strip()) is None:
        raise LicenseGateError(f"retrieved_at_iso {provenance.retrieved_at_iso!r} is not a valid ISO-8601 timestamp")

    noai_scan_blob = " | ".join(
        part
        for part in (
            provenance.asset_id,
            provenance.title,
            provenance.source_url,
            provenance.author,
            provenance.attribution_notice,
            provenance.license_url,
            provenance.changes_made_note,
            metadata_text,
        )
        if part
    )
    noai_hits = scan_text_for_noai_markers(noai_scan_blob)
    if noai_hits:
        raise LicenseGateError(
            f"Asset rejected due to NoAI / TDM opt-out / AI-generated marker(s) {list(noai_hits)} in metadata"
        )

    # Strictly reject all NoDerivatives (-ND) licenses under EVERY policy (H7)
    if _ND_LICENSE_RE.search(spdx) is not None:
        raise LicenseGateError(
            f"NoDerivatives (-ND) license {spdx!r} is strictly prohibited under all SyncCraft policies"
        )

    combined = (
        f"{provenance.asset_id} {provenance.title} {provenance.source_url} "
        f"{provenance.author} {provenance.attribution_notice} {provenance.license_url}"
    ).lower()
    for banned in BANNED_PROVENANCE_TOKENS:
        if banned in combined:
            raise LicenseGateError(f"Asset provenance contains prohibited source marker {banned!r}")

    if spdx in APPROVED_PERMISSIVE_LICENSES:
        return "clean_room_permissive"
    if license_policy in ("research_nc", "local_workspace_unverified") and spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES:
        if not provenance.attribution_notice.strip():
            raise LicenseGateError(
                f"{spdx} requires a non-empty attribution_notice under CC-BY-NC quarantine"
            )
        return "research_nc"
    if license_policy == "local_workspace_unverified" and spdx == "NOASSERTION":
        return "local_workspace_unverified"

    allowed = (
        sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES)
        if license_policy == "research_nc"
        else (
            sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES | {"NOASSERTION"})
            if license_policy == "local_workspace_unverified"
            else sorted(APPROVED_PERMISSIVE_LICENSES)
        )
    )
    hint = (
        " (pass license_policy='research_nc' for quarantined non-commercial CC-BY-NC research splits)"
        if spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES
        else ""
    )
    raise LicenseGateError(
        f"Rejected license {spdx!r} under policy {license_policy!r}; "
        f"allowed licenses: {allowed}{hint}"
    )


def _sanitize_ident(raw: str, fallback: str, used: set[str]) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(raw).strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    cand = cleaned
    idx = 2
    while cand in used:
        cand = f"{cleaned}_{idx}"
        idx += 1
    used.add(cand)
    return cand


def _decode_png_rgba(png_bytes_or_url: bytes | str) -> np.ndarray:
    if isinstance(png_bytes_or_url, str):
        prefix = "data:image/png;base64,"
        if png_bytes_or_url.startswith(prefix):
            raw = base64.b64decode(png_bytes_or_url[len(prefix) :])
        else:
            raw = base64.b64decode(png_bytes_or_url)
    else:
        raw = png_bytes_or_url
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Multi-Format Geometry Decompilers (.bbmodel, Java .json, Bedrock .geo.json)
# ---------------------------------------------------------------------------

_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)


def _obb_corners_world(center_w: np.ndarray, rot_w: np.ndarray, raw_size: np.ndarray) -> np.ndarray:
    """Compute canonical 8-corner world coordinates matching WorldBox.corners(include_inflate=False)."""
    return (_CORNER_SIGNS * (raw_size * 0.5)) @ rot_w.T + center_w


def _max_corner_reconstruction_error(
    compiled: Any,
    expected_corners: dict[str, np.ndarray],
) -> float:
    """Measure max L2 8-corner world vertex error across all source cubes."""
    if not expected_corners:
        return 0.0
    compiled_by_name = compiled.box_by_name
    max_err = 0.0
    for bname, exp_c in expected_corners.items():
        if bname not in compiled_by_name:
            return float("inf")
        comp_c = compiled_by_name[bname].corners(include_inflate=False)
        err = float(np.max(np.linalg.norm(comp_c - exp_c, axis=-1)))
        max_err = max(max_err, err)
    return max_err


_SNAP_UNIT_CANDIDATES: tuple[float, ...] = (0.0, 1.0, -1.0, 0.5, -0.5)
_CANONICAL_TEXEL_DENSITIES: tuple[int, ...] = (8, 16, 32, 64, 128)


def _infer_dyadic_size_step(raw_sizes: Sequence[np.ndarray]) -> float:
    """Choose coarsest v3.2 dyadic size_step in (1.0, 0.5, 0.25, 0.125) dividing all positive dimensions."""
    if not _HAS_DSL_V32 or not raw_sizes:
        return 1.0
    pos_vals: list[float] = []
    for arr in raw_sizes:
        for v in np.asarray(arr, dtype=np.float64).ravel():
            if v > 1e-9:
                pos_vals.append(float(v))
    if not pos_vals:
        return 1.0
    for step in (1.0, 0.5, 0.25, 0.125):
        ok = True
        for v in pos_vals:
            q = round(v / step)
            if q < 1 or abs(q * step - v) > 1e-5:
                ok = False
                break
        if ok:
            return step
    return 1.0


def _quantize_box_dimensions(
    raw_sz: np.ndarray,
    size_step: float,
) -> tuple[tuple[int | float, int | float, int | float], str | None, np.ndarray]:
    """Quantize raw 3D cuboid dimensions onto (ast_size, plane_normal, expected_eff_size)."""
    if _HAS_DSL_V32:
        zero_axes = [i for i in range(3) if float(raw_sz[i]) <= 1e-9]
        plane_normal = ("x", "y", "z")[zero_axes[0]] if len(zero_axes) == 1 else None
        p_idx = zero_axes[0] if plane_normal is not None else -1
        comps: list[int | float] = []
        for i in range(3):
            if i == p_idx:
                comps.append(0 if size_step == 1.0 else 0.0)
            else:
                steps = max(1, int(round(float(raw_sz[i]) / size_step)))
                if size_step == 1.0:
                    comps.append(int(steps))
                else:
                    val = round(steps * size_step, 6)
                    comps.append(int(round(val)) if abs(val - round(val)) <= 1e-9 else val)
        eff = raw_sz.copy() if plane_normal is not None or not zero_axes else np.where(raw_sz <= 1e-9, size_step, raw_sz)
        return (comps[0], comps[1], comps[2]), plane_normal, eff

    sz_rnd = np.maximum(1.0, np.rint(raw_sz))
    size_i = (int(sz_rnd[0]), int(sz_rnd[1]), int(sz_rnd[2]))
    eff_v31 = np.where(raw_sz <= 1e-9, 1.0, raw_sz)
    return size_i, None, eff_v31


def _infer_texel_density(
    tex_w: int,
    tex_h: int,
    uv_ratios: Sequence[float],
    total_area_u: float,
    has_nonzero_box_uv: bool,
) -> int:
    """Derive sensible texel_density (px per 16u block) from explicit UV/size ratios and atlas dimensions (M16)."""
    finite_ratios = [float(r) for r in uv_ratios if math.isfinite(r) and 2.0 <= float(r) <= 256.0]
    if finite_ratios:
        med = float(np.median(np.asarray(finite_ratios, dtype=np.float64)))
        return min(_CANONICAL_TEXEL_DENSITIES, key=lambda c: abs(math.log2(med / float(c))))
    if not has_nonzero_box_uv and max(tex_w, tex_h) >= 256 and total_area_u > 1.0:
        est = math.sqrt((float(tex_w * tex_h) * 0.35) / total_area_u) * 16.0
        if est >= 24.0:
            return min((16, 32, 64, 128), key=lambda c: abs(math.log2(est / float(c))))
    return 16


def _make_v32_meta(archetype: str, target_profile: str, density: int, size_step: float) -> Meta:
    if _HAS_DSL_V32:
        return Meta(
            archetype,
            target_profile,
            density,
            dsl_version="3.2",
            size_step=size_step,
            audit_profile="authored",
        )
    return Meta(archetype, target_profile, density)


def _make_v32_box(
    name: str,
    ast_size: tuple[int | float, int | float, int | float],
    material: str,
    *,
    place: Place | None = None,
    inflate: float = 0.0,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pivot: tuple[float, float, float] | None = None,
    plane_normal: str | None = None,
    uv: tuple[float, float] | None = None,
    uv_faces: dict[str, list[float]] | None = None,
    texture: str | None = None,
) -> Box:
    if _HAS_DSL_V32:
        uv_f_dict = (
            {f: (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for f, r in uv_faces.items()}
            if uv_faces
            else ()
        )
        return Box(  # type: ignore[call-arg]
            name,
            ast_size,
            material,
            place=place,
            inflate=inflate,
            rotate=rotate,
            pivot=pivot,
            plane_normal=plane_normal,
            uv=uv,
            uv_faces=uv_f_dict,  # type: ignore[arg-type]
            texture=texture,
        )
    return Box(name, ast_size, material, place=place, inflate=inflate)  # type: ignore[arg-type]


def _make_v32_program(
    meta: Meta,
    palettes: list[Palette],
    root_bone: Bone,
    textures_tuple: tuple[Any, ...] = (),
) -> Program:
    if _HAS_DSL_V32 and _TextureDecl is not None and textures_tuple:
        return Program(meta, palettes, root_bone, textures=textures_tuple)  # type: ignore[call-arg]
    return Program(meta, palettes, root_bone)


def _snap_place_without_base(unrot_loc: np.ndarray, box_sz: Sequence[float]) -> Place:
    """Convert raw bone-local center into snapped align in {-1,-0.5,0,0.5,1} when residual <= 0.5u."""
    al_out: list[float] = []
    off_out: list[float] = []
    for i in range(3):
        t_val = float(unrot_loc[i])
        sz_i = float(box_sz[i])
        if sz_i > 1e-9:
            best_al = 0.0
            best_off = t_val
            best_key = (abs(t_val), 0.0)
            for al_c in _SNAP_UNIT_CANDIDATES:
                off_c = t_val + 0.5 * sz_i * al_c
                k = (round(abs(off_c), 9), abs(al_c))
                if k < best_key:
                    best_key = k
                    best_al = al_c
                    best_off = off_c
            if abs(best_off) <= 0.5 + 1e-9:
                al_out.append(best_al)
                off_out.append(0.0 if abs(best_off) < 1e-9 else round(float(np.clip(best_off, -0.5, 0.5)), 8))
                continue
        al_out.append(0.0)
        off_out.append(0.0 if abs(t_val) < 1e-9 else round(t_val, 8))
    return Place(
        None,
        (0.0, 0.0, 0.0),
        (al_out[0], al_out[1], al_out[2]),
        (off_out[0], off_out[1], off_out[2]),
    )


def _snap_place_with_base(
    unrot_loc: np.ndarray,
    box_sz: Sequence[float],
    base_name: str,
    base_loc: np.ndarray,
    base_sz: Sequence[float] | None,
    base_is_unrot_box: bool,
) -> Place:
    """Convert sibling box placement into clean anchor/align where residual offset lies in [-0.5, 0.5]."""
    diff = unrot_loc - base_loc
    if not base_is_unrot_box or base_sz is None:
        return Place(
            base_name,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (
                0.0 if abs(float(diff[0])) < 1e-9 else round(float(diff[0]), 8),
                0.0 if abs(float(diff[1])) < 1e-9 else round(float(diff[1]), 8),
                0.0 if abs(float(diff[2])) < 1e-9 else round(float(diff[2]), 8),
            ),
        )
    anc_out: list[float] = []
    al_out: list[float] = []
    off_out: list[float] = []
    for i in range(3):
        d_i = float(diff[i])
        bs_i = float(base_sz[i])
        cs_i = float(box_sz[i])
        best_a = 0.0
        best_al = 0.0
        best_off = d_i
        best_key = (round(abs(d_i), 9), 0.0)
        for a_c in _SNAP_UNIT_CANDIDATES:
            for al_c in _SNAP_UNIT_CANDIDATES:
                off_c = d_i - 0.5 * bs_i * a_c + 0.5 * cs_i * al_c
                k = (round(abs(off_c), 9), abs(a_c) + abs(al_c))
                if k < best_key:
                    best_key = k
                    best_a = a_c
                    best_al = al_c
                    best_off = off_c
        if abs(best_off) <= 0.5 + 1e-9:
            anc_out.append(best_a)
            al_out.append(best_al)
            off_out.append(0.0 if abs(best_off) < 1e-9 else round(float(np.clip(best_off, -0.5, 0.5)), 8))
        else:
            anc_out.append(0.0)
            al_out.append(0.0)
            off_out.append(0.0 if abs(d_i) < 1e-9 else round(d_i, 8))
    return Place(
        base_name,
        (anc_out[0], anc_out[1], anc_out[2]),
        (al_out[0], al_out[1], al_out[2]),
        (off_out[0], off_out[1], off_out[2]),
    )


def _snap_attach_to_parent(
    local_trans: np.ndarray,
    candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
    *,
    allow_connected: bool,
) -> Attach:
    """Snap child bone joint attachment onto the closest parent box anchor within 0.5u when available."""
    best_score: tuple[int, float, float] | None = None
    best_att: Attach | None = None
    for p_name, p_loc, p_sz, p_is_unrot_box in candidates:
        raw_diff = local_trans - p_loc
        if p_is_unrot_box and p_sz is not None:
            anc: list[float] = []
            off: list[float] = []
            for i in range(3):
                d_i = float(raw_diff[i])
                ps_i = float(p_sz[i])
                best_a = 0.0
                best_o = d_i
                best_k = (round(abs(d_i), 9), 0.0)
                if ps_i > 1e-9:
                    for a_c in _SNAP_UNIT_CANDIDATES:
                        o_c = d_i - 0.5 * ps_i * a_c
                        k = (round(abs(o_c), 9), abs(a_c))
                        if k < best_k:
                            best_k = k
                            best_a = a_c
                            best_o = o_c
                if abs(best_o) <= 0.5 + 1e-9:
                    anc.append(best_a)
                    off.append(0.0 if abs(best_o) < 1e-9 else round(float(np.clip(best_o, -0.5, 0.5)), 8))
                else:
                    anc.append(0.0)
                    off.append(0.0 if abs(d_i) < 1e-9 else round(d_i, 8))
            max_o = max(abs(x) for x in off)
            sum_o = sum(abs(x) for x in off)
            within_half = max_o <= 0.5
            raw_within_half = all(abs(float(x)) <= 0.5 for x in raw_diff)
            detached = not (within_half and (allow_connected or raw_within_half))
            tier = 0 if within_half else 1
            cand_score = (tier, round(max_o, 8), round(sum_o, 8))
            cand_att = Attach(
                parent_box=p_name,
                anchor=(anc[0], anc[1], anc[2]),
                align=(0.0, 0.0, 0.0),
                offset=(off[0], off[1], off[2]),
                detached=detached,
            )
        else:
            off_raw = (
                0.0 if abs(float(raw_diff[0])) < 1e-9 else round(float(raw_diff[0]), 8),
                0.0 if abs(float(raw_diff[1])) < 1e-9 else round(float(raw_diff[1]), 8),
                0.0 if abs(float(raw_diff[2])) < 1e-9 else round(float(raw_diff[2]), 8),
            )
            max_o = max(abs(x) for x in off_raw)
            sum_o = sum(abs(x) for x in off_raw)
            cand_score = (2, round(max_o, 8), round(sum_o, 8))
            cand_att = Attach(
                parent_box=p_name,
                anchor=(0.0, 0.0, 0.0),
                align=(0.0, 0.0, 0.0),
                offset=off_raw,
                detached=True,
            )
        if best_score is None or cand_score < best_score:
            best_score = cand_score
            best_att = cand_att
    assert best_att is not None
    return best_att


_LR_SUB_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|[_\-\s])left($|[_\-\s0-9])", re.IGNORECASE), r"\1right\2"),
    (re.compile(r"(^|[_\-\s])l($|[_\-\s0-9])", re.IGNORECASE), r"\1r\2"),
    (re.compile(r"Left($|[A-Z0-9_])"), r"Right\1"),
    (re.compile(r"^left([A-Z])"), r"right\1"),
)


def _match_lr_bone_name_pair(name_a: str, name_b: str) -> tuple[str, str] | None:
    """If (name_a, name_b) form a bilateral Left/Right identifier pair, return (left_name, right_name)."""
    if name_a.lower() == name_b.lower():
        return None
    for pat, repl in _LR_SUB_PATTERNS:
        swapped_a = pat.sub(repl, name_a, count=1)
        if swapped_a != name_a and swapped_a.lower() == name_b.lower():
            return (name_a, name_b)
        swapped_b = pat.sub(repl, name_b, count=1)
        if swapped_b != name_b and swapped_b.lower() == name_a.lower():
            return (name_b, name_a)
    return None


def _decompile_bbmodel(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Decompile Blockbench .bbmodel JSON -> validated Program + face UVs + max 8-corner vertex error."""
    used_ids: set[str] = set()
    res = doc.get("resolution", {"width": 16, "height": 16})
    tex_w = max(1, int(res.get("width", 16)))
    tex_h = max(1, int(res.get("height", 16)))

    embedded_rgba: np.ndarray | None = None
    textures = doc.get("textures") or []
    raw_tex_name = "atlas_0"
    raw_tex_path = "atlas_0.png"
    if textures and isinstance(textures[0], dict):
        t0_name = str(textures[0].get("name") or "atlas_0").strip()
        if t0_name:
            raw_tex_path = t0_name if t0_name.lower().endswith(".png") else f"{t0_name}.png"
            raw_tex_name = re.sub(r"\.[A-Za-z0-9]+$", "", t0_name) or "atlas_0"
        if str(textures[0].get("source", "")).startswith("data:image/png;base64,"):
            embedded_rgba = _decode_png_rgba(textures[0]["source"])
            tex_h, tex_w = embedded_rgba.shape[:2]

    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("bbmodel contains no cube elements")

    all_raw_spans: list[np.ndarray] = []
    uv_ratios: list[float] = []
    total_area_u = 0.0
    for el in elements:
        lo_e = np.asarray(el["from"], dtype=np.float64)
        hi_e = np.asarray(el["to"], dtype=np.float64)
        sp_e = np.maximum(0.0, hi_e - lo_e)
        all_raw_spans.append(sp_e)
        total_area_u += float(2.0 * (sp_e[0] * sp_e[1] + sp_e[1] * sp_e[2] + sp_e[2] * sp_e[0]))
        f_d = el.get("faces") or {}
        if isinstance(f_d, dict):
            for f in FACES:
                if f in f_d and isinstance(f_d[f], dict) and "uv" in f_d[f]:
                    uv_r = f_d[f]["uv"]
                    if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                        wu_e, hv_e = face_uv_extents_units((float(sp_e[0]), float(sp_e[1]), float(sp_e[2])), f)
                        du = abs(float(uv_r[2]) - float(uv_r[0]))
                        dv = abs(float(uv_r[3]) - float(uv_r[1]))
                        if wu_e > 0.25 and du > 0.25:
                            uv_ratios.append(16.0 * du / wu_e)
                        if hv_e > 0.25 and dv > 0.25:
                            uv_ratios.append(16.0 * dv / hv_e)

    size_step = _infer_dyadic_size_step(all_raw_spans)
    density = _infer_texel_density(tex_w, tex_h, uv_ratios, total_area_u, has_nonzero_box_uv=True)

    tex_decl_id: str | None = None
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
        tex_decl_id = _sanitize_ident(raw_tex_name, "tex", used_ids)
        textures_tuple = (_TextureDecl(tex_decl_id, raw_tex_path, tex_w, tex_h),)

    elem_uuid_order = [(str(e.get("uuid", idx)), e) for idx, e in enumerate(elements)]
    elem_by_uuid = dict(elem_uuid_order)
    groups = doc.get("groups") or []
    group_by_uuid = {str(g.get("uuid", "")): g for g in groups if isinstance(g, dict)}
    outliner = doc.get("outliner") or []

    referenced_elem_uuids: set[str] = set()

    def _collect_outliner_refs(items_list: list[Any]) -> None:
        for entry in items_list:
            if isinstance(entry, str) and entry in elem_by_uuid:
                referenced_elem_uuids.add(entry)
            elif isinstance(entry, dict):
                _collect_outliner_refs(entry.get("children") or [])

    _collect_outliner_refs(outliner)

    uv_map: dict[str, dict[str, list[float]]] = {}
    palettes = [Palette("mat_0", (0.62, 0.06, 220.0), 6)]
    expected_corners: dict[str, np.ndarray] = {}
    group_uuid_to_safe_bone: dict[str, str] = {}

    def _register_elem_uv(
        bname: str,
        elem: dict[str, Any],
        uv_dims: tuple[float, float, float],
    ) -> None:
        f_dict = elem.get("faces") or {}
        uv_box_raw = elem.get("uv_offset")
        if not f_dict and isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) >= 2:
            ub, vb = float(uv_box_raw[0]), float(uv_box_raw[1])
            sx = max(0, int(math.floor(uv_dims[0] + 1e-9)))
            sy = max(0, int(math.floor(uv_dims[1] + 1e-9)))
            sz = max(0, int(math.floor(uv_dims[2] + 1e-9)))
            uv_map[bname] = {
                "up": [ub + sz, vb, ub + sz + sx, vb + sz],
                "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
                "east": [ub, vb + sz, ub + sz, vb + sz + sy],
                "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
                "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
                "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
            }
            return
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
                uv_map[bname][f] = [float(x) for x in f_dict[f]["uv"]]
            else:
                wu, hv = face_uv_extents_units(uv_dims, f)
                uv_map[bname][f] = [0.0, 0.0, float(min(tex_w, wu)), float(min(tex_h, hv))]

    def _emit_elems_for_bone(
        bone_label: str,
        elem_list: list[dict[str, Any]],
        bone_origin: np.ndarray,
        bone_world_piv: np.ndarray,
        bone_world_rot: np.ndarray,
    ) -> tuple[list[Box | Locator | Bone], list[tuple[str, np.ndarray, np.ndarray | None, bool]]]:
        direct_items: list[Box | Locator | Bone] = []
        rot_sub_bones: list[Bone] = []
        anchor_candidates: list[tuple[str, np.ndarray, np.ndarray | None, bool]] = []
        anchor_name: str | None = None
        anchor_loc: np.ndarray | None = None
        anchor_sz: np.ndarray | None = None
        anchor_is_unrot_box = False

        unrot_elems: list[dict[str, Any]] = []
        rot_elems: list[tuple[dict[str, Any], tuple[float, float, float]]] = []
        for el in elem_list:
            r_raw = el.get("rotation") or [0.0, 0.0, 0.0]
            el_rot = (float(r_raw[0]), float(r_raw[1]), float(r_raw[2]))
            if any(abs(a) > 1e-8 for a in el_rot):
                rot_elems.append((el, el_rot))
            else:
                unrot_elems.append(el)

        for el in unrot_elems:
            bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
            lo = np.asarray(el["from"], dtype=np.float64)
            hi = np.asarray(el["to"], dtype=np.float64)
            raw_span = np.maximum(0.0, hi - lo)
            ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
            ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
            ctr_unrot = 0.5 * (lo + hi)
            loc_ctr = ctr_unrot - bone_origin
            inflate = max(0.0, float(el.get("inflate", 0.0)))
            _register_elem_uv(bname, el, (float(raw_span[0]), float(raw_span[1]), float(raw_span[2])))
            uv_box_raw = el.get("uv_offset")
            uv_pair = (
                (float(uv_box_raw[0]), float(uv_box_raw[1]))
                if isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) == 2
                else None
            )

            w_ctr = bone_world_piv + bone_world_rot @ loc_ctr
            expected_corners[bname] = _obb_corners_world(w_ctr, bone_world_rot, eff_span)

            if anchor_name is None:
                pl = _snap_place_without_base(loc_ctr, ast_sz_arr)
                anchor_name = bname
                anchor_loc = loc_ctr
                anchor_sz = ast_sz_arr
                anchor_is_unrot_box = True
            else:
                assert anchor_loc is not None
                pl = _snap_place_with_base(loc_ctr, ast_sz_arr, anchor_name, anchor_loc, anchor_sz, anchor_is_unrot_box)
            has_explicit_faces = bool(el.get("faces"))
            direct_items.append(
                _make_v32_box(
                    bname,
                    ast_size,
                    "mat_0",
                    place=pl,
                    inflate=inflate,
                    plane_normal=plane_normal,
                    uv=uv_pair,
                    uv_faces=None if (uv_pair is not None and not has_explicit_faces) else uv_map[bname],
                    texture=tex_decl_id,
                )
            )
            anchor_candidates.append((bname, loc_ctr, ast_sz_arr, True))

        if _HAS_DSL_V32:
            # Emit per-cube rotations directly on Box without synthesizing extra rot_* dummy bones
            for el, el_rot in rot_elems:
                bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
                lo = np.asarray(el["from"], dtype=np.float64)
                hi = np.asarray(el["to"], dtype=np.float64)
                raw_span = np.maximum(0.0, hi - lo)
                ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
                ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
                ctr_unrot = 0.5 * (lo + hi)
                el_origin = np.asarray(el.get("origin", ctr_unrot.tolist()), dtype=np.float64)
                inflate = max(0.0, float(el.get("inflate", 0.0)))
                _register_elem_uv(bname, el, (float(raw_span[0]), float(raw_span[1]), float(raw_span[2])))
                uv_box_raw = el.get("uv_offset")
                uv_pair = (
                    (float(uv_box_raw[0]), float(uv_box_raw[1]))
                    if isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) == 2
                    else None
                )

                unrot_loc = ctr_unrot - bone_origin
                el_piv_loc = el_origin - bone_origin
                r_el = euler_xyz_matrix(el_rot)
                actual_loc_ctr = el_piv_loc + r_el @ (unrot_loc - el_piv_loc)
                w_rot = bone_world_rot @ r_el
                w_ctr = bone_world_piv + bone_world_rot @ actual_loc_ctr
                expected_corners[bname] = _obb_corners_world(w_ctr, w_rot, eff_span)

                if anchor_name is None:
                    pl = _snap_place_without_base(unrot_loc, ast_sz_arr)
                    anchor_name = bname
                    anchor_loc = actual_loc_ctr
                    anchor_sz = ast_sz_arr
                    anchor_is_unrot_box = False
                else:
                    assert anchor_loc is not None
                    pl = _snap_place_with_base(unrot_loc, ast_sz_arr, anchor_name, anchor_loc, anchor_sz, anchor_is_unrot_box)
                piv_tuple = (
                    round(float(el_piv_loc[0]), 8),
                    round(float(el_piv_loc[1]), 8),
                    round(float(el_piv_loc[2]), 8),
                )
                has_explicit_faces = bool(el.get("faces"))
                direct_items.append(
                    _make_v32_box(
                        bname,
                        ast_size,
                        "mat_0",
                        place=pl,
                        inflate=inflate,
                        rotate=el_rot,
                        pivot=piv_tuple,
                        plane_normal=plane_normal,
                        uv=uv_pair,
                        uv_faces=None if (uv_pair is not None and not has_explicit_faces) else uv_map[bname],
                        texture=tex_decl_id,
                    )
                )
                anchor_candidates.append((bname, actual_loc_ctr, ast_sz_arr, False))

        if anchor_name is None:
            piv_loc = _sanitize_ident(f"{bone_label}_pivot", "locator", used_ids)
            direct_items.append(Locator(name=piv_loc, offset=(0.0, 0.0, 0.0), role="socket"))
            anchor_name = piv_loc
            anchor_loc = np.zeros(3, dtype=np.float64)
            anchor_candidates.append((piv_loc, anchor_loc, None, False))

        if not _HAS_DSL_V32:
            assert anchor_loc is not None
            anchor_is_locator = not anchor_is_unrot_box
            for el, el_rot in rot_elems:
                bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
                lo = np.asarray(el["from"], dtype=np.float64)
                hi = np.asarray(el["to"], dtype=np.float64)
                raw_span = np.maximum(0.0, hi - lo)
                ast_size, _, eff_span = _quantize_box_dimensions(raw_span, size_step)
                ctr_unrot = 0.5 * (lo + hi)
                el_origin = np.asarray(el.get("origin", ctr_unrot.tolist()), dtype=np.float64)
                inflate = max(0.0, float(el.get("inflate", 0.0)))
                _register_elem_uv(bname, el, (float(raw_span[0]), float(raw_span[1]), float(raw_span[2])))

                el_piv_loc = el_origin - bone_origin
                box_rel = ctr_unrot - el_origin
                r_el = euler_xyz_matrix(el_rot)
                w_rot = bone_world_rot @ r_el
                w_ctr = bone_world_piv + bone_world_rot @ el_piv_loc + w_rot @ box_rel
                expected_corners[bname] = _obb_corners_world(w_ctr, w_rot, eff_span)

                rel_piv = el_piv_loc - anchor_loc
                sub_bname = _sanitize_ident(f"rot_{bname}", "bone", used_ids)
                att = Attach(
                    parent_box=anchor_name,
                    anchor=(0.0, 0.0, 0.0),
                    align=(0.0, 0.0, 0.0),
                    offset=(float(rel_piv[0]), float(rel_piv[1]), float(rel_piv[2])),
                    detached=(anchor_is_locator or any(abs(float(x)) > 0.5 for x in rel_piv)),
                )
                b_place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(box_rel[0]), float(box_rel[1]), float(box_rel[2])))
                rot_sub_bones.append(
                    Bone(sub_bname, rest_euler=el_rot, attach=att, items=[Box(bname, ast_size, "mat_0", place=b_place, inflate=inflate)])  # type: ignore[arg-type]
                )

        return [*direct_items, *rot_sub_bones], anchor_candidates

    def _can_mirror_bbmodel_pair(
        node_l: dict[str, Any],
        node_r: dict[str, Any],
        parent_origin: np.ndarray,
        parent_candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
    ) -> bool:
        grp_l = group_by_uuid.get(str(node_l.get("uuid", "")))
        grp_r = group_by_uuid.get(str(node_r.get("uuid", "")))
        if grp_l is None or grp_r is None:
            return False
        if any(isinstance(c, dict) for c in (node_l.get("children") or [])) or any(
            isinstance(c, dict) for c in (node_r.get("children") or [])
        ):
            return False
        elems_l = [elem_by_uuid[c] for c in (node_l.get("children") or []) if isinstance(c, str) and c in elem_by_uuid]
        elems_r = [elem_by_uuid[c] for c in (node_r.get("children") or []) if isinstance(c, str) and c in elem_by_uuid]
        if not elems_l or len(elems_l) != len(elems_r):
            return False
        if not any(abs(float(c[1][0])) <= 1e-5 and (c[3] or c[2] is None) for c in parent_candidates):
            return False
        orig_l = np.asarray(grp_l.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        orig_r = np.asarray(grp_r.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        lt_l = orig_l - parent_origin
        lt_r = orig_r - parent_origin
        if not np.allclose(lt_r, lt_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
            return False
        rot_l = tuple(float(x) for x in grp_l.get("rotation", [0.0, 0.0, 0.0]))
        rot_r = tuple(float(x) for x in grp_r.get("rotation", [0.0, 0.0, 0.0]))
        if not np.allclose(euler_xyz_matrix(rot_r), euler_xyz_matrix((rot_l[0], -rot_l[1], -rot_l[2])), atol=1e-4):
            return False
        for el_l, el_r in zip(elems_l, elems_r):
            lo_l, hi_l = np.asarray(el_l["from"], dtype=np.float64), np.asarray(el_l["to"], dtype=np.float64)
            lo_r, hi_r = np.asarray(el_r["from"], dtype=np.float64), np.asarray(el_r["to"], dtype=np.float64)
            sz_l, sz_r = np.maximum(0.0, hi_l - lo_l), np.maximum(0.0, hi_r - lo_r)
            if not np.allclose(sz_l, sz_r, atol=1e-4):
                return False
            if abs(float(el_l.get("inflate", 0.0)) - float(el_r.get("inflate", 0.0))) > 1e-5:
                return False
            rl_raw = el_l.get("rotation") or [0.0, 0.0, 0.0]
            rr_raw = el_r.get("rotation") or [0.0, 0.0, 0.0]
            rl = (float(rl_raw[0]), float(rl_raw[1]), float(rl_raw[2]))
            rr = (float(rr_raw[0]), float(rr_raw[1]), float(rr_raw[2]))
            if not _HAS_DSL_V32 and (any(abs(a) > 1e-8 for a in rl) or any(abs(a) > 1e-8 for a in rr)):
                return False
            if not np.allclose(euler_xyz_matrix(rr), euler_xyz_matrix((rl[0], -rl[1], -rl[2])), atol=1e-4):
                return False
            ul_l = 0.5 * (lo_l + hi_l) - orig_l
            ul_r = 0.5 * (lo_r + hi_r) - orig_r
            if not np.allclose(ul_r, ul_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
                return False
            if any(abs(a) > 1e-8 for a in rl):
                pl_l = np.asarray(el_l.get("origin", (0.5 * (lo_l + hi_l)).tolist()), dtype=np.float64) - orig_l
                pl_r = np.asarray(el_r.get("origin", (0.5 * (lo_r + hi_r)).tolist()), dtype=np.float64) - orig_r
                if not np.allclose(pl_r, pl_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
                    return False
        return True

    def _register_mirrored_bbmodel_right_node(
        node_r: dict[str, Any],
        left_bone: Bone,
        safe_r: str,
        parent_origin: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
    ) -> None:
        uid_r = str(node_r.get("uuid", ""))
        grp_r = group_by_uuid[uid_r]
        if uid_r:
            group_uuid_to_safe_bone[uid_r] = safe_r
        orig_r = np.asarray(grp_r.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot_r = tuple(float(x) for x in grp_r.get("rotation", [0.0, 0.0, 0.0]))
        w_piv_r = parent_world_piv + parent_world_rot @ (orig_r - parent_origin)
        w_rot_r = parent_world_rot @ euler_xyz_matrix(rot_r)
        elems_r = [elem_by_uuid[c] for c in (node_r.get("children") or []) if isinstance(c, str) and c in elem_by_uuid]
        unrot_r = [e for e in elems_r if not any(abs(float(a)) > 1e-8 for a in (e.get("rotation") or [0.0, 0.0, 0.0]))]
        rot_r_list = [e for e in elems_r if any(abs(float(a)) > 1e-8 for a in (e.get("rotation") or [0.0, 0.0, 0.0]))]
        ordered_r = [*unrot_r, *rot_r_list]
        left_boxes = [it for it in left_bone.items if isinstance(it, Box)]
        for bx_l, el_r in zip(left_boxes, ordered_r):
            mir_bname = f"{safe_r}__{bx_l.name}"
            lo_r = np.asarray(el_r["from"], dtype=np.float64)
            hi_r = np.asarray(el_r["to"], dtype=np.float64)
            raw_sp_r = np.maximum(0.0, hi_r - lo_r)
            _, _, eff_sp_r = _quantize_box_dimensions(raw_sp_r, size_step)
            _register_elem_uv(mir_bname, el_r, (float(raw_sp_r[0]), float(raw_sp_r[1]), float(raw_sp_r[2])))
            ctr_r = 0.5 * (lo_r + hi_r)
            rr_raw = el_r.get("rotation") or [0.0, 0.0, 0.0]
            rr = (float(rr_raw[0]), float(rr_raw[1]), float(rr_raw[2]))
            if any(abs(a) > 1e-8 for a in rr):
                piv_r_el = np.asarray(el_r.get("origin", ctr_r.tolist()), dtype=np.float64)
                r_el_m = euler_xyz_matrix(rr)
                w_rot_m = w_rot_r @ r_el_m
                w_ctr_m = w_piv_r + w_rot_r @ (piv_r_el - orig_r) + w_rot_m @ (ctr_r - piv_r_el)
                expected_corners[mir_bname] = _obb_corners_world(w_ctr_m, w_rot_m, eff_sp_r)
            else:
                w_ctr_m = w_piv_r + w_rot_r @ (ctr_r - orig_r)
                expected_corners[mir_bname] = _obb_corners_world(w_ctr_m, w_rot_r, eff_sp_r)

    def _build_bone_from_node(
        node: dict[str, Any],
        parent_origin: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
        parent_candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
        is_root: bool,
        mirror_partner_node: dict[str, Any] | None = None,
    ) -> Bone | None:
        uid_str = str(node.get("uuid", ""))
        grp = group_by_uuid.get(uid_str)
        if grp is None:
            return None
        g_origin = np.asarray(grp.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        g_rot = tuple(float(x) for x in grp.get("rotation", [0.0, 0.0, 0.0]))
        bone_name = _sanitize_ident(grp.get("name", "bone"), "bone", used_ids)
        if uid_str:
            group_uuid_to_safe_bone[uid_str] = bone_name

        local_trans = g_origin - parent_origin
        if is_root:
            world_piv = g_origin.copy()
            world_rot = np.eye(3, dtype=np.float64)
        else:
            world_piv = parent_world_piv + parent_world_rot @ local_trans
            world_rot = parent_world_rot @ euler_xyz_matrix(g_rot)

        child_elems = [elem_by_uuid[c] for c in node.get("children", []) if isinstance(c, str) and c in elem_by_uuid]
        child_nodes = [c for c in node.get("children", []) if isinstance(c, dict)]

        items, own_candidates = _emit_elems_for_bone(
            bone_name, child_elems, g_origin, world_piv, world_rot
        )

        # Detect symmetric L/R child bone pairs when parent frame is X-symmetric
        x_sym_cands = [c for c in own_candidates if abs(float(c[1][0])) <= 1e-5 and (c[3] or c[2] is None)]
        parent_is_x_sym = abs(float(world_piv[0])) <= 1e-5 and abs(float(world_rot[0, 1])) <= 1e-5 and abs(float(world_rot[0, 2])) <= 1e-5
        paired_right_indices: dict[int, int] = {}
        right_skip: set[int] = set()
        if parent_is_x_sym and x_sym_cands:
            for i_idx, cn_i in enumerate(child_nodes):
                if i_idx in right_skip or i_idx in paired_right_indices:
                    continue
                gi = group_by_uuid.get(str(cn_i.get("uuid", "")))
                if gi is None:
                    continue
                ni_raw = str(gi.get("name", ""))
                for j_idx in range(i_idx + 1, len(child_nodes)):
                    if j_idx in right_skip or j_idx in paired_right_indices:
                        continue
                    cn_j = child_nodes[j_idx]
                    gj = group_by_uuid.get(str(cn_j.get("uuid", "")))
                    if gj is None:
                        continue
                    nj_raw = str(gj.get("name", ""))
                    lr_hit = _match_lr_bone_name_pair(ni_raw, nj_raw)
                    if lr_hit is None:
                        continue
                    if lr_hit[0] == ni_raw and _can_mirror_bbmodel_pair(cn_i, cn_j, g_origin, x_sym_cands):
                        paired_right_indices[i_idx] = j_idx
                        right_skip.add(j_idx)
                        break
                    if lr_hit[0] == nj_raw and _can_mirror_bbmodel_pair(cn_j, cn_i, g_origin, x_sym_cands):
                        paired_right_indices[j_idx] = i_idx
                        right_skip.add(i_idx)
                        break

        for idx_cn, cn in enumerate(child_nodes):
            if idx_cn in right_skip:
                continue
            partner_cn = child_nodes[paired_right_indices[idx_cn]] if idx_cn in paired_right_indices else None
            cands_for_child = x_sym_cands if partner_cn is not None else own_candidates
            cb = _build_bone_from_node(
                cn,
                g_origin,
                world_piv,
                world_rot,
                cands_for_child,
                is_root=False,
                mirror_partner_node=partner_cn,
            )
            if cb is not None:
                items.append(cb)

        if is_root:
            return Bone(name=bone_name, pivot_local=(float(g_origin[0]), float(g_origin[1]), float(g_origin[2])), items=items)

        att = _snap_attach_to_parent(local_trans, parent_candidates, allow_connected=_HAS_DSL_V32)
        mirror_safe_name: str | None = None
        if mirror_partner_node is not None:
            grp_r = group_by_uuid[str(mirror_partner_node.get("uuid", ""))]
            mirror_safe_name = _sanitize_ident(grp_r.get("name", "bone_r"), "bone", used_ids)
        built_bone = Bone(name=bone_name, rest_euler=g_rot, attach=att, mirror_name=mirror_safe_name, items=items)
        if mirror_partner_node is not None and mirror_safe_name is not None:
            _register_mirrored_bbmodel_right_node(
                mirror_partner_node, built_bone, mirror_safe_name, parent_origin, parent_world_piv, parent_world_rot
            )
        return built_bone

    dict_roots = [n for n in outliner if isinstance(n, dict) and str(n.get("uuid", "")) in group_by_uuid]
    loose_elems = [elem_by_uuid[u] for u in outliner if isinstance(u, str) and u in elem_by_uuid]
    for u_str, el_obj in elem_uuid_order:
        if u_str not in referenced_elem_uuids:
            loose_elems.append(el_obj)

    if len(dict_roots) == 1 and not loose_elems:
        g0 = group_by_uuid[str(dict_roots[0]["uuid"])]
        g0_rot = tuple(float(x) for x in g0.get("rotation", [0.0, 0.0, 0.0]))
        if not any(abs(a) > 1e-8 for a in g0_rot):
            root_bone = _build_bone_from_node(
                dict_roots[0], np.zeros(3), np.zeros(3), np.eye(3), (), is_root=True
            )
            assert root_bone is not None
        else:
            wrap_name = _sanitize_ident("world_root", "root", used_ids)
            r_items, r_cands = _emit_elems_for_bone(
                wrap_name, [], np.zeros(3), np.zeros(3), np.eye(3)
            )
            cb0 = _build_bone_from_node(
                dict_roots[0], np.zeros(3), np.zeros(3), np.eye(3), r_cands, is_root=False
            )
            if cb0 is not None:
                r_items.append(cb0)
            root_bone = Bone(name=wrap_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)
    elif len(dict_roots) >= 1:
        wrap_name = _sanitize_ident("world_root", "root", used_ids)
        r_items, r_cands = _emit_elems_for_bone(
            wrap_name, loose_elems, np.zeros(3), np.zeros(3), np.eye(3)
        )
        for rn in dict_roots:
            cb = _build_bone_from_node(
                rn, np.zeros(3), np.zeros(3), np.eye(3), r_cands, is_root=False
            )
            if cb is not None:
                r_items.append(cb)
        root_bone = Bone(name=wrap_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)
    else:
        root_name = _sanitize_ident("imported_root", "bone", used_ids)
        r_items, _ = _emit_elems_for_bone(
            root_name, loose_elems, np.zeros(3), np.zeros(3), np.eye(3)
        )
        root_bone = Bone(name=root_name, pivot_local=(0.0, 0.0, 0.0), items=r_items)

    anims_raw = doc.get("animations")
    has_anims = isinstance(anims_raw, list) and len(anims_raw) > 0
    archetype = "entity" if (has_anims or len(group_by_uuid) >= 2) else "block_multipart"
    meta_node = _make_v32_meta(archetype, "bbmodel_5", density, size_step)
    prog = validate_program(_make_v32_program(meta_node, palettes, root_bone, textures_tuple))
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
    if has_anims:
        beh = decompile_bbmodel_animations_to_behavior(doc, prog, group_uuid_to_safe_bone=group_uuid_to_safe_bone)
        if beh is not None:
            prog = validate_program(replace(prog, behavior=beh))
    return prog, uv_map, embedded_rgba, max_err


def _decompile_java_json(doc: dict[str, Any]) -> tuple[Program, dict[str, dict[str, list[float]]], float]:
    """Decompile Minecraft Java block/item .json -> validated Program + face UVs + max 8-corner vertex error."""
    used_ids: set[str] = set()
    tex_size = doc.get("texture_size") or [16, 16]
    tex_w, tex_h = max(1, int(tex_size[0])), max(1, int(tex_size[1]))
    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("Java model JSON contains no elements")

    all_raw_spans: list[np.ndarray] = []
    uv_ratios: list[float] = []
    total_area_u = 0.0
    for el in elements:
        lo_e = np.asarray(el["from"], dtype=np.float64)
        hi_e = np.asarray(el["to"], dtype=np.float64)
        sp_e = np.maximum(0.0, hi_e - lo_e)
        all_raw_spans.append(sp_e)
        total_area_u += float(2.0 * (sp_e[0] * sp_e[1] + sp_e[1] * sp_e[2] + sp_e[2] * sp_e[0]))
        f_d = el.get("faces") or {}
        if isinstance(f_d, dict):
            for f in FACES:
                if f in f_d and isinstance(f_d[f], dict) and "uv" in f_d[f]:
                    uv_r = f_d[f]["uv"]
                    if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                        wu_e, hv_e = face_uv_extents_units((float(sp_e[0]), float(sp_e[1]), float(sp_e[2])), f)
                        du = abs(float(uv_r[2]) - float(uv_r[0])) * tex_w / 16.0
                        dv = abs(float(uv_r[3]) - float(uv_r[1])) * tex_h / 16.0
                        if wu_e > 0.25 and du > 0.25:
                            uv_ratios.append(16.0 * du / wu_e)
                        if hv_e > 0.25 and dv > 0.25:
                            uv_ratios.append(16.0 * dv / hv_e)

    size_step = _infer_dyadic_size_step(all_raw_spans)
    density = _infer_texel_density(tex_w, tex_h, uv_ratios, total_area_u, has_nonzero_box_uv=True)

    tex_decl_id: str | None = None
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
        tex_dict = doc.get("textures") if isinstance(doc.get("textures"), dict) else {}
        tex_ref = str(tex_dict.get("atlas") or next((v for v in tex_dict.values() if isinstance(v, str) and not v.startswith("#")), "block/atlas"))
        tex_decl_id = _sanitize_ident("atlas_0", "tex", used_ids)
        textures_tuple = (_TextureDecl(tex_decl_id, tex_ref, tex_w, tex_h),)

    uv_map: dict[str, dict[str, list[float]]] = {}
    root_name = _sanitize_ident("java_root", "bone", used_ids)
    items: list[Box | Locator | Bone] = []
    first_bname: str | None = None
    first_ctr: np.ndarray | None = None
    first_sz: np.ndarray | None = None
    first_is_unrot_box = False
    first_is_locator = False
    expected_corners: dict[str, np.ndarray] = {}

    for idx, el in enumerate(elements):
        lo = np.asarray(el["from"], dtype=np.float64)
        hi = np.asarray(el["to"], dtype=np.float64)
        raw_span = np.maximum(0.0, hi - lo)
        ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
        ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
        ctr = 0.5 * (lo + hi)
        bname = _sanitize_ident(el.get("name", f"elem_{idx}"), "box", used_ids)

        f_dict = el.get("faces") or {}
        uv_map[bname] = {}
        for f in FACES:
            if f in f_dict and "uv" in f_dict[f]:
                u0, v0, u1, v1 = (float(x) for x in f_dict[f]["uv"])
                uv_map[bname][f] = [u0 * tex_w / 16.0, v0 * tex_h / 16.0, u1 * tex_w / 16.0, v1 * tex_h / 16.0]
            else:
                wu, hv = face_uv_extents_units((float(raw_span[0]), float(raw_span[1]), float(raw_span[2])), f)
                uv_map[bname][f] = [0.0, 0.0, float(wu), float(hv)]

        rot_data = el.get("rotation")
        has_rot = False
        rot_euler = (0.0, 0.0, 0.0)
        rot_origin = ctr
        if isinstance(rot_data, dict):
            rot_origin = np.asarray(rot_data.get("origin", ctr.tolist()), dtype=np.float64)
            if "axis" in rot_data:
                ax = str(rot_data["axis"])
                ang = float(rot_data.get("angle", 0.0))
                rot_euler = (ang if ax == "x" else 0.0, ang if ax == "y" else 0.0, ang if ax == "z" else 0.0)
            else:
                rot_euler = (float(rot_data.get("x", 0.0)), float(rot_data.get("y", 0.0)), float(rot_data.get("z", 0.0)))
            has_rot = any(abs(a) > 1e-8 for a in rot_euler)

        if not has_rot:
            if first_bname is None or first_is_locator:
                place = _snap_place_without_base(ctr, ast_sz_arr)
                if first_bname is None:
                    first_bname = bname
                    first_ctr = ctr
                    first_sz = ast_sz_arr
                    first_is_unrot_box = True
                    first_is_locator = False
            else:
                assert first_ctr is not None
                place = _snap_place_with_base(ctr, ast_sz_arr, first_bname, first_ctr, first_sz, first_is_unrot_box)
            box_node = _make_v32_box(
                bname,
                ast_size,
                "mat_0",
                place=place,
                plane_normal=plane_normal,
                uv_faces=uv_map[bname],
                texture=tex_decl_id,
            )
            bone_split = [x for x in items if isinstance(x, Bone)]
            non_bone_split = [x for x in items if not isinstance(x, Bone)]
            items = [*non_bone_split, box_node, *bone_split]
            expected_corners[bname] = _obb_corners_world(ctr, np.eye(3, dtype=np.float64), eff_span)
        elif _HAS_DSL_V32:
            r_el = euler_xyz_matrix(rot_euler)
            w_ctr = rot_origin + r_el @ (ctr - rot_origin)
            expected_corners[bname] = _obb_corners_world(w_ctr, r_el, eff_span)
            if first_bname is None:
                place = _snap_place_without_base(ctr, ast_sz_arr)
                first_bname = bname
                first_ctr = w_ctr
                first_sz = ast_sz_arr
                first_is_unrot_box = False
                first_is_locator = False
            else:
                assert first_ctr is not None
                place = _snap_place_with_base(ctr, ast_sz_arr, first_bname, first_ctr, first_sz, first_is_unrot_box)
            piv_t = (round(float(rot_origin[0]), 8), round(float(rot_origin[1]), 8), round(float(rot_origin[2]), 8))
            items.append(
                _make_v32_box(
                    bname,
                    ast_size,
                    "mat_0",
                    place=place,
                    rotate=rot_euler,
                    pivot=piv_t,
                    plane_normal=plane_normal,
                    uv_faces=uv_map[bname],
                    texture=tex_decl_id,
                )
            )
        else:
            if first_bname is None:
                loc_anchor = _sanitize_ident("java_origin", "locator", used_ids)
                items.append(Locator(name=loc_anchor, offset=(0.0, 0.0, 0.0), role="socket"))
                first_bname = loc_anchor
                first_ctr = np.zeros(3, dtype=np.float64)
                first_is_locator = True
            assert first_ctr is not None
            r_el = euler_xyz_matrix(rot_euler)
            box_loc = ctr - rot_origin
            w_ctr = rot_origin + r_el @ box_loc
            expected_corners[bname] = _obb_corners_world(w_ctr, r_el, eff_span)

            bone_name = _sanitize_ident(f"rot_{bname}", "bone", used_ids)
            rel_piv = rot_origin - first_ctr
            att = Attach(
                first_bname,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (float(rel_piv[0]), float(rel_piv[1]), float(rel_piv[2])),
                detached=True,
            )
            b_place = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(box_loc[0]), float(box_loc[1]), float(box_loc[2])))
            items.append(Bone(bone_name, rest_euler=rot_euler, attach=att, items=[Box(bname, ast_size, "mat_0", place=b_place)]))  # type: ignore[arg-type]

    meta_node = _make_v32_meta("weapon_3d", "java_modern", density, size_step)
    prog = validate_program(
        _make_v32_program(
            meta_node,
            [Palette("mat_0", (0.62, 0.06, 220.0), 6)],
            Bone(root_name, pivot_local=(0.0, 0.0, 0.0), items=items),
            textures_tuple,
        )
    )
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
    return prog, uv_map, max_err


def _infer_locator_role(raw_loc_name: str) -> str:
    low = raw_loc_name.lower()
    if any(tok in low for tok in ("seat", "mount", "rider", "saddle")):
        return "rider_seat"
    if any(tok in low for tok in ("eye", "head", "camera", "look", "focus", "gaze")):
        return "camera_focus"
    if any(tok in low for tok in ("target", "hitbox", "hurtbox")):
        return "hitbox"
    if any(tok in low for tok in ("foot", "paw", "talon", "hand", "claw", "ik", "effector", "toe", "hoof")):
        return "ik_effector"
    if any(tok in low for tok in ("effect", "special", "breath", "muzzle", "flame", "beam", "vfx", "emitter")):
        return "vfx_emitter"
    return "socket"


def _decompile_bedrock_geojson(
    doc: dict[str, Any],
) -> tuple[Program, dict[str, dict[str, list[float]]], float, dict[str, str]]:
    """Decompile Minecraft Bedrock / Cobblemon .geo.json -> validated Program + UVs + 8-corner error + bone_map (C5, M16, H6)."""
    used_ids: set[str] = set()
    geos = doc.get("minecraft:geometry") or []
    if not geos or not isinstance(geos[0], dict):
        raise ValueError("Bedrock .geo.json missing 'minecraft:geometry' list")
    g0 = geos[0]
    desc = g0.get("description") or {}
    tex_w_desc = max(16, int(desc.get("texture_width", 64)))
    tex_h_desc = max(16, int(desc.get("texture_height", tex_w_desc)))
    bones_raw = g0.get("bones") or []
    if not bones_raw:
        raise ValueError("Bedrock geometry contains no bones")

    uv_map: dict[str, dict[str, list[float]]] = {}
    bone_records: dict[str, dict[str, Any]] = {}
    children_map: dict[str | None, list[str]] = {None: []}
    raw_to_safe_bone: dict[str, str] = {}
    expected_corners: dict[str, np.ndarray] = {}

    all_raw_sizes: list[np.ndarray] = []
    uv_ratios: list[float] = []
    total_area_u = 0.0
    has_nonzero_box_uv = False
    total_cubes = 0
    for b in bones_raw:
        bname = str(b["name"])
        bone_records[bname] = b
        children_map.setdefault(bname, [])
        b_cubes = b.get("cubes") or []
        total_cubes += len(b_cubes)
        for cb in b_cubes:
            rsz = np.maximum(0.0, np.asarray(cb.get("size", [1.0, 1.0, 1.0]), dtype=np.float64))
            all_raw_sizes.append(rsz)
            total_area_u += float(2.0 * (rsz[0] * rsz[1] + rsz[1] * rsz[2] + rsz[2] * rsz[0]))
            uv_c = cb.get("uv", [0, 0])
            if isinstance(uv_c, dict):
                for f in FACES:
                    if f in uv_c and isinstance(uv_c[f], dict) and "uv_size" in uv_c[f]:
                        wu_c, hv_c = face_uv_extents_units((float(rsz[0]), float(rsz[1]), float(rsz[2])), f)
                        su_c, sv_c = (abs(float(x)) for x in uv_c[f]["uv_size"][:2])
                        if wu_c > 0.25 and su_c > 0.25:
                            uv_ratios.append(16.0 * su_c / wu_c)
                        if hv_c > 0.25 and sv_c > 0.25:
                            uv_ratios.append(16.0 * sv_c / hv_c)
            elif isinstance(uv_c, (list, tuple)) and len(uv_c) >= 2:
                if abs(float(uv_c[0])) > 1e-6 or abs(float(uv_c[1])) > 1e-6:
                    has_nonzero_box_uv = True

    for b in bones_raw:
        parent = b.get("parent")
        parent_key = str(parent) if parent in bone_records else None
        children_map[parent_key].append(str(b["name"]))

    size_step = _infer_dyadic_size_step(all_raw_sizes)
    density = _infer_texel_density(tex_w_desc, tex_h_desc, uv_ratios, total_area_u, has_nonzero_box_uv)

    tex_decl_id: str | None = None
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
        short_geo = str(desc.get("identifier", "entity")).split(".")[-1] or "entity"
        tex_decl_id = _sanitize_ident("atlas_0", "tex", used_ids)
        textures_tuple = (_TextureDecl(tex_decl_id, f"{short_geo}.png", tex_w_desc, tex_h_desc),)

    need_fallback_root_box = total_cubes == 0

    def _register_bedrock_cube_uv(
        cname: str,
        cube: dict[str, Any],
        raw_sz: np.ndarray,
    ) -> tuple[float, float] | None:
        uv_raw = cube.get("uv", [0, 0])
        uv_dims_f = (float(raw_sz[0]), float(raw_sz[1]), float(raw_sz[2]))
        uv_map[cname] = {}
        if isinstance(uv_raw, dict):
            for f in FACES:
                if f in uv_raw and "uv" in uv_raw[f]:
                    u0, v0 = (float(x) for x in uv_raw[f]["uv"])
                    wu_def, hv_def = face_uv_extents_units(uv_dims_f, f)
                    su, sv = (float(x) for x in uv_raw[f].get("uv_size", [wu_def, hv_def]))
                    uv_map[cname][f] = [u0, v0, u0 + su, v0 + sv]
                else:
                    wu, hv = face_uv_extents_units(uv_dims_f, f)
                    uv_map[cname][f] = [0.0, 0.0, float(wu), float(hv)]
            return None
        ub, vb = float(uv_raw[0]), float(uv_raw[1])
        # Bedrock 1.12.0 Box-UV layout uses floor(size) UV strip offsets (0 for planar normal axes)
        sx = max(0, int(math.floor(float(raw_sz[0]) + 1e-9)))
        sy = max(0, int(math.floor(float(raw_sz[1]) + 1e-9)))
        sz = max(0, int(math.floor(float(raw_sz[2]) + 1e-9)))
        uv_map[cname] = {
            "up": [ub + sz, vb, ub + sz + sx, vb + sz],
            "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
            "east": [ub, vb + sz, ub + sz, vb + sz + sy],
            "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
            "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
            "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
        }
        return (ub, vb)

    def _emit_bone_locators(
        rec: dict[str, Any],
        piv_sync: np.ndarray,
    ) -> list[Locator]:
        locs_raw = rec.get("locators")
        if not isinstance(locs_raw, dict):
            return []
        out_locs: list[Locator] = []
        piv_bed_default = [-float(piv_sync[0]), float(piv_sync[1]), float(piv_sync[2])]
        for raw_lname, lspec in locs_raw.items():
            raw_str = str(raw_lname)
            base_lname = raw_str if raw_str.lower().startswith("loc_") else f"loc_{raw_str}"
            safe_lname = _sanitize_ident(base_lname, "locator", used_ids)
            if isinstance(lspec, dict):
                explicit_role = lspec.get("role")
                role = str(explicit_role) if explicit_role in LOCATOR_ROLES else _infer_locator_role(raw_str)
                pos_bed = np.asarray(lspec.get("offset", lspec.get("position", piv_bed_default)), dtype=np.float64)
            else:
                role = _infer_locator_role(raw_str)
                pos_bed = np.asarray(lspec, dtype=np.float64)
            pos_sync = bedrock_point_to_sync(pos_bed)
            loc_local = pos_sync - piv_sync
            out_locs.append(
                Locator(
                    name=safe_lname,
                    offset=(
                        0.0 if abs(float(loc_local[0])) < 1e-9 else round(float(loc_local[0]), 8),
                        0.0 if abs(float(loc_local[1])) < 1e-9 else round(float(loc_local[1]), 8),
                        0.0 if abs(float(loc_local[2])) < 1e-9 else round(float(loc_local[2]), 8),
                    ),
                    role=role,
                )
            )
        return out_locs

    def _can_mirror_bedrock_pair(
        raw_l: str,
        raw_r: str,
        parent_piv_sync: np.ndarray,
        parent_candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
    ) -> bool:
        if children_map.get(raw_l) or children_map.get(raw_r):
            return False
        rec_l, rec_r = bone_records[raw_l], bone_records[raw_r]
        if rec_l.get("locators") or rec_r.get("locators"):
            return False
        cubes_l = rec_l.get("cubes") or []
        cubes_r = rec_r.get("cubes") or []
        if not cubes_l or len(cubes_l) != len(cubes_r):
            return False
        if not any(abs(float(c[1][0])) <= 1e-5 and (c[3] or c[2] is None) for c in parent_candidates):
            return False
        piv_l = bedrock_point_to_sync(rec_l.get("pivot", [0.0, 0.0, 0.0]))
        piv_r = bedrock_point_to_sync(rec_r.get("pivot", [0.0, 0.0, 0.0]))
        lt_l = piv_l - parent_piv_sync
        lt_r = piv_r - parent_piv_sync
        if not np.allclose(lt_r, lt_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
            return False
        rot_l = bedrock_euler_to_sync(rec_l.get("rotation", [0.0, 0.0, 0.0]))
        rot_r = bedrock_euler_to_sync(rec_r.get("rotation", [0.0, 0.0, 0.0]))
        if not np.allclose(euler_xyz_matrix(rot_r), euler_xyz_matrix((rot_l[0], -rot_l[1], -rot_l[2])), atol=1e-4):
            return False
        for cb_l, cb_r in zip(cubes_l, cubes_r):
            sz_l = np.maximum(0.0, np.asarray(cb_l["size"], dtype=np.float64))
            sz_r = np.maximum(0.0, np.asarray(cb_r["size"], dtype=np.float64))
            if not np.allclose(sz_l, sz_r, atol=1e-4):
                return False
            if abs(float(cb_l.get("inflate", 0.0)) - float(cb_r.get("inflate", 0.0))) > 1e-5:
                return False
            cr_l = bedrock_euler_to_sync(cb_l.get("rotation") or [0.0, 0.0, 0.0])
            cr_r = bedrock_euler_to_sync(cb_r.get("rotation") or [0.0, 0.0, 0.0])
            if not _HAS_DSL_V32 and (np.any(np.abs(cr_l) > 1e-8) or np.any(np.abs(cr_r) > 1e-8)):
                return False
            if not np.allclose(euler_xyz_matrix(cr_r), euler_xyz_matrix((cr_l[0], -cr_l[1], -cr_l[2])), atol=1e-4):
                return False
            from_l = bedrock_origin_to_sync_cube_from(cb_l["origin"], sz_l)
            from_r = bedrock_origin_to_sync_cube_from(cb_r["origin"], sz_r)
            ul_l = (from_l + 0.5 * sz_l) - piv_l
            ul_r = (from_r + 0.5 * sz_r) - piv_r
            if not np.allclose(ul_r, ul_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
                return False
            if np.any(np.abs(cr_l) > 1e-8):
                orig_bed_l = np.asarray(cb_l["origin"], dtype=np.float64)
                orig_bed_r = np.asarray(cb_r["origin"], dtype=np.float64)
                cp_l = bedrock_point_to_sync(cb_l.get("pivot", (orig_bed_l + 0.5 * sz_l).tolist())) - piv_l
                cp_r = bedrock_point_to_sync(cb_r.get("pivot", (orig_bed_r + 0.5 * sz_r).tolist())) - piv_r
                if not np.allclose(cp_r, cp_l * np.array([-1.0, 1.0, 1.0]), atol=1e-4):
                    return False
        return True

    def _register_mirrored_bedrock_right_bone(
        raw_r: str,
        left_bone: Bone,
        safe_r: str,
        parent_piv_sync: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
    ) -> None:
        raw_to_safe_bone[raw_r] = safe_r
        rec_r = bone_records[raw_r]
        piv_r = bedrock_point_to_sync(rec_r.get("pivot", [0.0, 0.0, 0.0]))
        rot_r_arr = bedrock_euler_to_sync(rec_r.get("rotation", [0.0, 0.0, 0.0]))
        rot_r = (float(rot_r_arr[0]), float(rot_r_arr[1]), float(rot_r_arr[2]))
        w_piv_r = parent_world_piv + parent_world_rot @ (piv_r - parent_piv_sync)
        w_rot_r = parent_world_rot @ euler_xyz_matrix(rot_r)
        cubes_r = rec_r.get("cubes") or []
        unrot_r = [c for c in cubes_r if not any(abs(float(a)) > 1e-8 for a in (c.get("rotation") or [0.0, 0.0, 0.0]))]
        rot_r_list = [c for c in cubes_r if any(abs(float(a)) > 1e-8 for a in (c.get("rotation") or [0.0, 0.0, 0.0]))]
        ordered_r = [*unrot_r, *rot_r_list]
        left_boxes = [it for it in left_bone.items if isinstance(it, Box)]
        for bx_l, cb_r in zip(left_boxes, ordered_r):
            mir_cname = f"{safe_r}__{bx_l.name}"
            orig_bed_r = np.asarray(cb_r["origin"], dtype=np.float64)
            raw_sz_r = np.maximum(0.0, np.asarray(cb_r["size"], dtype=np.float64))
            _, _, eff_sz_r = _quantize_box_dimensions(raw_sz_r, size_step)
            _register_bedrock_cube_uv(mir_cname, cb_r, raw_sz_r)
            sync_from_r = bedrock_origin_to_sync_cube_from(orig_bed_r, raw_sz_r)
            ctr_r = sync_from_r + 0.5 * raw_sz_r
            c_r_raw = cb_r.get("rotation") or [0.0, 0.0, 0.0]
            if any(abs(float(a)) > 1e-8 for a in c_r_raw):
                cr_sync = bedrock_euler_to_sync(c_r_raw)
                cp_sync = bedrock_point_to_sync(cb_r.get("pivot", (orig_bed_r + 0.5 * raw_sz_r).tolist()))
                r_cb = euler_xyz_matrix((float(cr_sync[0]), float(cr_sync[1]), float(cr_sync[2])))
                w_rot_c = w_rot_r @ r_cb
                w_ctr_c = w_piv_r + w_rot_r @ (cp_sync - piv_r) + w_rot_c @ (ctr_r - cp_sync)
                expected_corners[mir_cname] = _obb_corners_world(w_ctr_c, w_rot_c, eff_sz_r)
            else:
                w_ctr_c = w_piv_r + w_rot_r @ (ctr_r - piv_r)
                expected_corners[mir_cname] = _obb_corners_world(w_ctr_c, w_rot_r, eff_sz_r)

    def _build_bedrock_bone(
        raw_name: str,
        parent_pivot: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
        parent_candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
        is_root: bool,
        mirror_partner_raw: str | None = None,
    ) -> Bone:
        rec = bone_records[raw_name]
        safe_bname = _sanitize_ident(raw_name, "bone", used_ids)
        raw_to_safe_bone[raw_name] = safe_bname
        piv = bedrock_point_to_sync(rec.get("pivot", [0.0, 0.0, 0.0]))
        rot_arr = bedrock_euler_to_sync(rec.get("rotation", [0.0, 0.0, 0.0]))
        rot = (
            0.0 if abs(float(rot_arr[0])) < 1e-9 else round(float(rot_arr[0]), 8),
            0.0 if abs(float(rot_arr[1])) < 1e-9 else round(float(rot_arr[1]), 8),
            0.0 if abs(float(rot_arr[2])) < 1e-9 else round(float(rot_arr[2]), 8),
        )
        local_trans = piv - parent_pivot
        if is_root:
            world_piv = piv.copy()
            world_rot = np.eye(3, dtype=np.float64)
        else:
            world_piv = parent_world_piv + parent_world_rot @ local_trans
            world_rot = parent_world_rot @ euler_xyz_matrix(rot)
        cubes = rec.get("cubes") or []

        direct_items: list[Box | Locator] = []
        rot_cube_bones: list[Bone] = []
        own_candidates: list[tuple[str, np.ndarray, np.ndarray | None, bool]] = []
        fb_name: str | None = None
        fb_loc: np.ndarray | None = None
        fb_sz: np.ndarray | None = None
        fb_is_unrot_box = False

        if cubes:
            unrot_cubes: list[tuple[int, dict[str, Any]]] = []
            rot_cubes: list[tuple[int, dict[str, Any], tuple[float, float, float]]] = []
            for c_idx, cube in enumerate(cubes):
                c_r = cube.get("rotation") or [0.0, 0.0, 0.0]
                if any(abs(float(a)) > 1e-8 for a in c_r):
                    cr_sync = bedrock_euler_to_sync(c_r)
                    c_rot = (
                        0.0 if abs(float(cr_sync[0])) < 1e-9 else round(float(cr_sync[0]), 8),
                        0.0 if abs(float(cr_sync[1])) < 1e-9 else round(float(cr_sync[1]), 8),
                        0.0 if abs(float(cr_sync[2])) < 1e-9 else round(float(cr_sync[2]), 8),
                    )
                    rot_cubes.append((c_idx, cube, c_rot))
                else:
                    unrot_cubes.append((c_idx, cube))

            for c_idx, cube in unrot_cubes:
                origin_bed = np.asarray(cube["origin"], dtype=np.float64)
                raw_sz = np.maximum(0.0, np.asarray(cube["size"], dtype=np.float64))
                ast_size, plane_normal, eff_sz = _quantize_box_dimensions(raw_sz, size_step)
                ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
                sync_from = bedrock_origin_to_sync_cube_from(origin_bed, raw_sz)
                ctr = sync_from + 0.5 * raw_sz
                loc_c = ctr - piv
                inflate = max(0.0, float(cube.get("inflate", 0.0)))
                cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                uv_pair = _register_bedrock_cube_uv(cname, cube, raw_sz)

                w_ctr = world_piv + world_rot @ loc_c
                expected_corners[cname] = _obb_corners_world(w_ctr, world_rot, eff_sz)

                if fb_name is None:
                    pl = _snap_place_without_base(loc_c, ast_sz_arr)
                    fb_name, fb_loc, fb_sz, fb_is_unrot_box = cname, loc_c, ast_sz_arr, True
                else:
                    assert fb_loc is not None
                    pl = _snap_place_with_base(loc_c, ast_sz_arr, fb_name, fb_loc, fb_sz, fb_is_unrot_box)
                direct_items.append(
                    _make_v32_box(
                        cname,
                        ast_size,
                        "mat_0",
                        place=pl,
                        inflate=inflate,
                        plane_normal=plane_normal,
                        uv=uv_pair,
                        uv_faces=None if uv_pair is not None else uv_map[cname],
                        texture=tex_decl_id,
                    )
                )
                own_candidates.append((cname, loc_c, ast_sz_arr, True))

            if _HAS_DSL_V32:
                # Emit per-cube local rotations directly on Box without extra *_crot_* dummy bones
                for c_idx, cube, c_rot in rot_cubes:
                    origin_bed = np.asarray(cube["origin"], dtype=np.float64)
                    raw_sz = np.maximum(0.0, np.asarray(cube["size"], dtype=np.float64))
                    ast_size, plane_normal, eff_sz = _quantize_box_dimensions(raw_sz, size_step)
                    ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
                    sync_from = bedrock_origin_to_sync_cube_from(origin_bed, raw_sz)
                    ctr = sync_from + 0.5 * raw_sz
                    c_piv_bed = np.asarray(cube.get("pivot", (origin_bed + 0.5 * raw_sz).tolist()), dtype=np.float64)
                    c_piv = bedrock_point_to_sync(c_piv_bed)
                    inflate = max(0.0, float(cube.get("inflate", 0.0)))
                    cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                    uv_pair = _register_bedrock_cube_uv(cname, cube, raw_sz)

                    unrot_loc_c = ctr - piv
                    c_piv_loc = c_piv - piv
                    r_cube = euler_xyz_matrix(c_rot)
                    actual_loc_c = c_piv_loc + r_cube @ (unrot_loc_c - c_piv_loc)
                    w_rot_c = world_rot @ r_cube
                    w_ctr_c = world_piv + world_rot @ actual_loc_c
                    expected_corners[cname] = _obb_corners_world(w_ctr_c, w_rot_c, eff_sz)

                    if fb_name is None:
                        pl = _snap_place_without_base(unrot_loc_c, ast_sz_arr)
                        fb_name, fb_loc, fb_sz, fb_is_unrot_box = cname, actual_loc_c, ast_sz_arr, False
                    else:
                        assert fb_loc is not None
                        pl = _snap_place_with_base(unrot_loc_c, ast_sz_arr, fb_name, fb_loc, fb_sz, fb_is_unrot_box)
                    piv_t = (
                        0.0 if abs(float(c_piv_loc[0])) < 1e-9 else round(float(c_piv_loc[0]), 8),
                        0.0 if abs(float(c_piv_loc[1])) < 1e-9 else round(float(c_piv_loc[1]), 8),
                        0.0 if abs(float(c_piv_loc[2])) < 1e-9 else round(float(c_piv_loc[2]), 8),
                    )
                    direct_items.append(
                        _make_v32_box(
                            cname,
                            ast_size,
                            "mat_0",
                            place=pl,
                            inflate=inflate,
                            rotate=c_rot,
                            pivot=piv_t,
                            plane_normal=plane_normal,
                            uv=uv_pair,
                            uv_faces=None if uv_pair is not None else uv_map[cname],
                            texture=tex_decl_id,
                        )
                    )
                    own_candidates.append((cname, actual_loc_c, ast_sz_arr, False))

            explicit_locs = _emit_bone_locators(rec, piv)
            direct_items.extend(explicit_locs)
            for eloc in explicit_locs:
                own_candidates.append((eloc.name, np.asarray(eloc.offset, dtype=np.float64), None, False))

            if fb_name is None:
                if explicit_locs:
                    fb_name = explicit_locs[0].name
                    fb_loc = np.asarray(explicit_locs[0].offset, dtype=np.float64)
                else:
                    piv_loc_name = _sanitize_ident(f"{safe_bname}_pivot", "locator", used_ids)
                    direct_items.append(Locator(name=piv_loc_name, offset=(0.0, 0.0, 0.0), role="socket"))
                    fb_name, fb_loc = piv_loc_name, np.zeros(3, dtype=np.float64)
                    own_candidates.append((piv_loc_name, fb_loc, None, False))

            if not _HAS_DSL_V32:
                assert fb_loc is not None
                fb_is_locator = not fb_is_unrot_box
                for c_idx, cube, c_rot in rot_cubes:
                    origin_bed = np.asarray(cube["origin"], dtype=np.float64)
                    raw_sz = np.maximum(0.0, np.asarray(cube["size"], dtype=np.float64))
                    ast_size, _, eff_sz = _quantize_box_dimensions(raw_sz, size_step)
                    sync_from = bedrock_origin_to_sync_cube_from(origin_bed, raw_sz)
                    ctr = sync_from + 0.5 * raw_sz
                    c_piv_bed = np.asarray(cube.get("pivot", (origin_bed + 0.5 * raw_sz).tolist()), dtype=np.float64)
                    c_piv = bedrock_point_to_sync(c_piv_bed)
                    inflate = max(0.0, float(cube.get("inflate", 0.0)))
                    cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                    _register_bedrock_cube_uv(cname, cube, raw_sz)

                    c_piv_loc = c_piv - piv
                    c_ctr_rel = ctr - c_piv
                    r_cube = euler_xyz_matrix(c_rot)
                    w_rot_c = world_rot @ r_cube
                    w_ctr_c = world_piv + world_rot @ c_piv_loc + w_rot_c @ c_ctr_rel
                    expected_corners[cname] = _obb_corners_world(w_ctr_c, w_rot_c, eff_sz)

                    rel_c = c_piv_loc - fb_loc
                    cbone_name = _sanitize_ident(f"{safe_bname}_crot_{c_idx}", "bone", used_ids)
                    att_c = Attach(
                        fb_name,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (float(rel_c[0]), float(rel_c[1]), float(rel_c[2])),
                        detached=(fb_is_locator or any(abs(float(x)) > 0.5 for x in rel_c)),
                    )
                    pl_c = Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (float(c_ctr_rel[0]), float(c_ctr_rel[1]), float(c_ctr_rel[2])))
                    rot_cube_bones.append(
                        Bone(cbone_name, rest_euler=c_rot, attach=att_c, items=[Box(cname, ast_size, "mat_0", place=pl_c, inflate=inflate)])  # type: ignore[arg-type]
                    )
        else:
            explicit_locs = _emit_bone_locators(rec, piv)
            if is_root and need_fallback_root_box:
                hub = _sanitize_ident(f"{safe_bname}_hub", "hub", used_ids)
                uv_map[hub] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
                direct_items.append(_make_v32_box(hub, (1, 1, 1), "mat_0", uv_faces=uv_map[hub], texture=tex_decl_id))
                fb_name, fb_loc = hub, np.zeros(3, dtype=np.float64)
                own_candidates.append((hub, fb_loc, np.ones(3, dtype=np.float64), True))
                direct_items.extend(explicit_locs)
                for eloc in explicit_locs:
                    own_candidates.append((eloc.name, np.asarray(eloc.offset, dtype=np.float64), None, False))
            elif explicit_locs:
                direct_items.extend(explicit_locs)
                for eloc in explicit_locs:
                    own_candidates.append((eloc.name, np.asarray(eloc.offset, dtype=np.float64), None, False))
            else:
                piv_loc_name = _sanitize_ident(f"{safe_bname}_pivot", "locator", used_ids)
                direct_items.append(Locator(name=piv_loc_name, offset=(0.0, 0.0, 0.0), role="socket"))
                own_candidates.append((piv_loc_name, np.zeros(3, dtype=np.float64), None, False))

        items: list[Box | Locator | Bone] = [*direct_items, *rot_cube_bones]
        child_raw_list = children_map.get(raw_name, [])
        x_sym_cands = [c for c in own_candidates if abs(float(c[1][0])) <= 1e-5 and (c[3] or c[2] is None)]
        parent_is_x_sym = abs(float(world_piv[0])) <= 1e-5 and abs(float(world_rot[0, 1])) <= 1e-5 and abs(float(world_rot[0, 2])) <= 1e-5
        paired_right_by_left: dict[int, int] = {}
        skip_right_indices: set[int] = set()
        if parent_is_x_sym and x_sym_cands:
            for i_idx, ch_i in enumerate(child_raw_list):
                if i_idx in skip_right_indices or i_idx in paired_right_by_left:
                    continue
                for j_idx in range(i_idx + 1, len(child_raw_list)):
                    if j_idx in skip_right_indices or j_idx in paired_right_by_left:
                        continue
                    ch_j = child_raw_list[j_idx]
                    lr_hit = _match_lr_bone_name_pair(ch_i, ch_j)
                    if lr_hit is None:
                        continue
                    if lr_hit[0] == ch_i and _can_mirror_bedrock_pair(ch_i, ch_j, piv, x_sym_cands):
                        paired_right_by_left[i_idx] = j_idx
                        skip_right_indices.add(j_idx)
                        break
                    if lr_hit[0] == ch_j and _can_mirror_bedrock_pair(ch_j, ch_i, piv, x_sym_cands):
                        paired_right_by_left[j_idx] = i_idx
                        skip_right_indices.add(i_idx)
                        break

        for idx_ch, child_raw in enumerate(child_raw_list):
            if idx_ch in skip_right_indices:
                continue
            partner_raw = child_raw_list[paired_right_by_left[idx_ch]] if idx_ch in paired_right_by_left else None
            cands_for_child = x_sym_cands if partner_raw is not None else own_candidates
            items.append(
                _build_bedrock_bone(
                    child_raw,
                    piv,
                    world_piv,
                    world_rot,
                    cands_for_child,
                    is_root=False,
                    mirror_partner_raw=partner_raw,
                )
            )

        if is_root:
            return Bone(safe_bname, pivot_local=(float(piv[0]), float(piv[1]), float(piv[2])), items=items)

        att = _snap_attach_to_parent(local_trans, parent_candidates, allow_connected=_HAS_DSL_V32)
        mirror_safe: str | None = None
        if mirror_partner_raw is not None:
            mirror_safe = _sanitize_ident(mirror_partner_raw, "bone", used_ids)
        built = Bone(safe_bname, rest_euler=rot, attach=att, mirror_name=mirror_safe, items=items)
        if mirror_partner_raw is not None and mirror_safe is not None:
            _register_mirrored_bedrock_right_bone(
                mirror_partner_raw, built, mirror_safe, parent_pivot, parent_world_piv, parent_world_rot
            )
        return built

    top_bones = children_map[None]
    if len(top_bones) == 1 and not any(abs(float(x)) > 1e-8 for x in bone_records[top_bones[0]].get("rotation", [0, 0, 0])):
        root_bone = _build_bedrock_bone(
            top_bones[0], np.zeros(3), np.zeros(3), np.eye(3), (), is_root=True
        )
    else:
        syn_root = _sanitize_ident("bedrock_root", "bone", used_ids)
        if need_fallback_root_box:
            syn_anchor = _sanitize_ident("bedrock_hub", "box", used_ids)
            uv_map[syn_anchor] = {f: [0.0, 0.0, 1.0, 1.0] for f in FACES}
            r_items: list[Box | Locator | Bone] = [
                _make_v32_box(syn_anchor, (1, 1, 1), "mat_0", uv_faces=uv_map[syn_anchor], texture=tex_decl_id)
            ]
            r_cands: list[tuple[str, np.ndarray, np.ndarray | None, bool]] = [
                (syn_anchor, np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64), True)
            ]
        else:
            syn_anchor = _sanitize_ident("bedrock_origin", "locator", used_ids)
            r_items = [Locator(name=syn_anchor, offset=(0.0, 0.0, 0.0), role="socket")]
            r_cands = [(syn_anchor, np.zeros(3, dtype=np.float64), None, False)]
        for tb in top_bones:
            r_items.append(
                _build_bedrock_bone(
                    tb, np.zeros(3), np.zeros(3), np.eye(3), r_cands, is_root=False
                )
            )
        root_bone = Bone(syn_root, pivot_local=(0.0, 0.0, 0.0), items=r_items)

    meta_node = _make_v32_meta("entity", "bedrock_entity", density, size_step)
    prog = validate_program(
        _make_v32_program(meta_node, [Palette("mat_0", (0.62, 0.06, 220.0), 6)], root_bone, textures_tuple)
    )
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)
    return prog, uv_map, max_err, raw_to_safe_bone


_ANIM_TIME_IDENTS: frozenset[str] = frozenset({
    "q.anim_time",
    "query.anim_time",
    "v.anim_time",
    "variable.anim_time",
    "q.life_time",
    "query.life_time",
    "v.life_time",
    "variable.life_time",
})
_ARITH_TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?|\.\d+)|([A-Za-z_][A-Za-z0-9_.]*)|([+\-*/()]))")
_TRIG_START_RE = re.compile(r"\bmath\.(sin|cos)\s*\(", re.IGNORECASE)
_ABS_START_RE = re.compile(r"\bmath\.abs\s*\(", re.IGNORECASE)


def _eval_linear_molang_expr(expr_str: str, t_val: float) -> float | None:
    """Zero-eval bounded recursive-descent arithmetic evaluator for affine Molang sub-expressions."""
    s = expr_str.strip()
    if not s or len(s) > 384:
        return None
    tokens: list[tuple[str, float | str]] = []
    pos = 0
    n_chars = len(s)
    while pos < n_chars:
        if s[pos:].isspace():
            break
        m = _ARITH_TOKEN_RE.match(s, pos)
        if m is None:
            return None
        num_s, ident_s, op_s = m.groups()
        if num_s is not None:
            tokens.append(("NUM", float(num_s)))
        elif ident_s is not None:
            low_id = ident_s.lower()
            if low_id in _ANIM_TIME_IDENTS:
                tokens.append(("NUM", float(t_val)))
            elif low_id == "math.pi":
                tokens.append(("NUM", 180.0))
            else:
                return None
        elif op_s is not None:
            tokens.append(("OP", op_s))
        pos = m.end()
        if len(tokens) > 128:
            return None

    if not tokens:
        return None

    idx = 0

    def _parse_expr(depth: int) -> float | None:
        nonlocal idx
        if depth > 32:
            return None
        lhs = _parse_mul(depth + 1)
        if lhs is None:
            return None
        while idx < len(tokens) and tokens[idx][0] == "OP" and tokens[idx][1] in ("+", "-"):
            op = tokens[idx][1]
            idx += 1
            rhs = _parse_mul(depth + 1)
            if rhs is None:
                return None
            lhs = lhs + rhs if op == "+" else lhs - rhs
        return lhs

    def _parse_mul(depth: int) -> float | None:
        nonlocal idx
        if depth > 32:
            return None
        lhs = _parse_unary(depth + 1)
        if lhs is None:
            return None
        while idx < len(tokens) and tokens[idx][0] == "OP" and tokens[idx][1] in ("*", "/"):
            op = tokens[idx][1]
            idx += 1
            rhs = _parse_unary(depth + 1)
            if rhs is None:
                return None
            if op == "*":
                lhs = lhs * rhs
            else:
                if abs(rhs) <= 1e-12:
                    return None
                lhs = lhs / rhs
        return lhs

    def _parse_unary(depth: int) -> float | None:
        nonlocal idx
        if depth > 32 or idx >= len(tokens):
            return None
        tok_kind, tok_val = tokens[idx]
        if tok_kind == "OP" and tok_val in ("+", "-"):
            idx += 1
            sub = _parse_unary(depth + 1)
            if sub is None:
                return None
            return -sub if tok_val == "-" else sub
        return _parse_primary(depth + 1)

    def _parse_primary(depth: int) -> float | None:
        nonlocal idx
        if depth > 32 or idx >= len(tokens):
            return None
        tok_kind, tok_val = tokens[idx]
        if tok_kind == "NUM":
            idx += 1
            return float(tok_val)
        if tok_kind == "OP" and tok_val == "(":
            idx += 1
            inner = _parse_expr(depth + 1)
            if inner is None or idx >= len(tokens) or tokens[idx] != ("OP", ")"):
                return None
            idx += 1
            return inner
        return None

    result = _parse_expr(0)
    if result is None or idx != len(tokens) or not math.isfinite(result):
        return None
    return float(result)


def _unwrap_molang_clamp(text: str) -> tuple[str, tuple[float, float] | None]:
    """Unwrap up to two nested math.clamp(expr, lo, hi) calls without eval."""
    cur = text.strip()
    bounds: tuple[float, float] | None = None
    for _ in range(2):
        if not (cur.lower().startswith("math.clamp(") and cur.endswith(")")):
            break
        inner = cur[11:-1]
        depth = 0
        commas: list[int] = []
        valid_parens = True
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    valid_parens = False
                    break
            elif ch == "," and depth == 0:
                commas.append(i)
        if not valid_parens or depth != 0 or len(commas) != 2:
            break
        arg0 = inner[: commas[0]].strip()
        lo_val = _eval_linear_molang_expr(inner[commas[0] + 1 : commas[1]], 0.0)
        hi_val = _eval_linear_molang_expr(inner[commas[1] + 1 :], 0.0)
        if lo_val is None or hi_val is None or lo_val > hi_val:
            break
        cur = arg0
        bounds = (lo_val, hi_val) if bounds is None else (max(bounds[0], lo_val), min(bounds[1], hi_val))
    return cur, bounds


def _split_top_level_additive_terms(expr_str: str) -> list[str]:
    """Split expression at top-level '+' or '-' (not after '*' or '/' or inside parens)."""
    terms: list[str] = []
    buf: list[str] = []
    depth = 0
    prev_nonspace = ""
    for ch in expr_str:
        if ch == "(":
            depth += 1
            buf.append(ch)
            prev_nonspace = ch
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            prev_nonspace = ch
        elif depth == 0 and ch in "+-" and buf and prev_nonspace not in ("", "+", "-", "*", "/", "("):
            term = "".join(buf).strip()
            if term:
                terms.append(term)
            buf = [ch]
            prev_nonspace = ch
        else:
            buf.append(ch)
            if not ch.isspace():
                prev_nonspace = ch
    tail = "".join(buf).strip()
    if tail:
        terms.append(tail)
    return terms


def _unwrap_signed_parens(expr_str: str) -> tuple[float, str]:
    """Strip leading unary signs and outer balanced parentheses while tracking sign."""
    sign = 1.0
    cur = expr_str.strip()
    for _ in range(8):
        if not cur:
            break
        if cur[0] in "+-":
            if cur[0] == "-":
                sign = -sign
            cur = cur[1:].strip()
            continue
        if cur.startswith("(") and cur.endswith(")"):
            depth = 0
            wraps_all = True
            for i, ch in enumerate(cur):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        wraps_all = False
                        break
                    if depth == 0 and i < len(cur) - 1:
                        wraps_all = False
                        break
            if wraps_all and depth == 0:
                cur = cur[1:-1].strip()
                continue
        break
    return sign, cur


def _flatten_additive_terms(expr_str: str, outer_sign: float = 1.0, depth: int = 0) -> list[tuple[float, str]]:
    """Flatten nested signed additive groups into (sign, term_core) pairs without eval."""
    raw = expr_str.strip()
    if not raw:
        return []
    if depth < 6:
        top_parts = _split_top_level_additive_terms(raw)
        if len(top_parts) > 1:
            out: list[tuple[float, str]] = []
            for p in top_parts:
                out.extend(_flatten_additive_terms(p, outer_sign, depth + 1))
            return out
    sgn, core = _unwrap_signed_parens(raw)
    eff_sign = outer_sign * sgn
    if not core:
        return []
    if depth < 6:
        inner_parts = _split_top_level_additive_terms(core)
        if len(inner_parts) > 1:
            out = []
            for p in inner_parts:
                out.extend(_flatten_additive_terms(p, eff_sign, depth + 1))
            return out
    return [(eff_sign, core)]


def _find_balanced_call(term: str, start_re: re.Pattern[str]) -> tuple[int, int, str, str] | None:
    """Locate a balanced function call matching ``start_re`` without eval."""
    m = start_re.search(term)
    if m is None:
        return None
    fn_name = (m.group(1) if m.lastindex else m.group(0)).lower()
    open_idx = m.end() - 1
    depth = 0
    close_idx = -1
    for i in range(open_idx, len(term)):
        ch = term[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx < 0 or depth != 0:
        return None
    inside_arg = term[open_idx + 1 : close_idx]
    return (m.start(), close_idx + 1, fn_name, inside_arg)


def _find_balanced_trig_call(term: str) -> tuple[int, int, str, str] | None:
    """Locate math.sin(...) or math.cos(...) with nested balanced parentheses."""
    return _find_balanced_call(term, _TRIG_START_RE)


def _split_top_level_ternary(expr_str: str) -> tuple[float, list[str]] | None:
    """Split top-level ``cond ? branch_a : branch_b`` ternary into ``(outer_sign, [branch_a, branch_b])``."""
    sign = 1.0
    cur = expr_str.strip()
    for _ in range(6):
        if cur.startswith("(") and cur.endswith(")"):
            _, unwrapped = _unwrap_signed_parens(cur)
            if unwrapped != cur:
                cur = unwrapped
                continue
        if cur and cur[0] in "+-":
            rest = cur[1:].strip()
            if rest.startswith("(") and rest.endswith(")"):
                _, unwrapped = _unwrap_signed_parens(rest)
                if unwrapped != rest:
                    if cur[0] == "-":
                        sign = -sign
                    cur = unwrapped
                    continue
        break
    depth = 0
    q_idx = -1
    colon_idx = -1
    for i, ch in enumerate(cur):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch == "?" and q_idx < 0:
                q_idx = i
            elif ch == ":" and q_idx >= 0 and colon_idx < 0:
                colon_idx = i
                break
    if q_idx >= 0 and colon_idx > q_idx + 1:
        b_true = cur[q_idx + 1 : colon_idx].strip()
        b_false = cur[colon_idx + 1 :].strip()
        branches = [b for b in (b_true, b_false) if b]
        return (sign, branches) if branches else None
    return None


def _eval_affine_wrapper(term_core: str, span_start: int, span_end: int) -> tuple[float, float] | None:
    """Evaluate outer affine wrapper ``f(u) = c0 + c1 * u`` around ``term_core[span_start:span_end]``."""
    prefix = term_core[:span_start]
    suffix = term_core[span_end:]
    f0_t0 = _eval_linear_molang_expr(f"{prefix}(0.0){suffix}", 0.0)
    f0_t1 = _eval_linear_molang_expr(f"{prefix}(0.0){suffix}", 1.0)
    f1_t0 = _eval_linear_molang_expr(f"{prefix}(1.0){suffix}", 0.0)
    f1_t1 = _eval_linear_molang_expr(f"{prefix}(1.0){suffix}", 1.0)
    f2_t0 = _eval_linear_molang_expr(f"{prefix}(2.0){suffix}", 0.0)
    if (
        f0_t0 is None
        or f0_t1 is None
        or f1_t0 is None
        or f1_t1 is None
        or f2_t0 is None
    ):
        return None
    if abs(f0_t1 - f0_t0) > 1e-6 or abs(f1_t1 - f1_t0) > 1e-6:
        return None
    if abs((f2_t0 - f1_t0) - (f1_t0 - f0_t0)) > 1e-4:
        return None
    return (f0_t0, f1_t0 - f0_t0)


def _parse_molang_inside_arg(inside_arg: str) -> tuple[float, float | None, float] | None:
    """Evaluate affine inside-trig argument f(t) = speed * t + phase0 without eval."""
    y0 = _eval_linear_molang_expr(inside_arg, 0.0)
    y1 = _eval_linear_molang_expr(inside_arg, 1.0)
    y2 = _eval_linear_molang_expr(inside_arg, 2.0)
    if y0 is None or y1 is None or y2 is None:
        return None
    if abs((y2 - y1) - (y1 - y0)) > 1e-4:
        return None
    speed = y1 - y0
    phase_deg = y0
    if abs(speed) <= 1e-6:
        return (phase_deg, None, 1.0)
    sin_sign = -1.0 if speed < 0.0 else 1.0
    eff_phase_deg = -phase_deg if speed < 0.0 else phase_deg
    freq_hz = round(float(np.clip(abs(speed) / 360.0, 0.05, 20.0)), 4)
    return (eff_phase_deg, freq_hz, sin_sign)


def _parse_molang_channel_zero_eval(
    expr: Any,
    bone_name: str,
    *,
    _depth: int = 0,
) -> tuple[str, float, float, float, float | None] | None:
    """Zero-eval bounded parser translating Bedrock/Cobblemon Molang rotation channel to TrackDecl parameters.

    Supports arbitrarily nested parentheses inside trig calls (e.g.
    ``math.sin((q.anim_time - 0.25) * 90 * 4) * 10``), affine-on-trig wrappers
    (``(1 - math.cos(q.anim_time * 360)) * 15``), ``q.life_time``, top-level
    ternary guards (``q.is_moving ? ... : 0``), ``math.abs(math.sin(...))``,
    unary/binary signs, multi-harmonic expressions, and ``math.clamp(...)``.
    Returns ``(law, amp_deg, phase_offset_unit, bias_deg, freq_hz_override)`` or ``None``.
    """
    if isinstance(expr, bool) or expr is None or _depth > 3:
        return None
    if isinstance(expr, (int, float)):
        val = float(expr)
        if not math.isfinite(val) or abs(val) < 0.25:
            return None
        return ("hinge", 0.0, 0.0, float(np.clip(val, -180.0, 180.0)), None)
    if not isinstance(expr, str) or len(expr) > 512:
        return None
    raw_text = expr.strip()
    if not raw_text:
        return None

    text, clamp_bounds = _unwrap_molang_clamp(raw_text)

    ternary_hit = _split_top_level_ternary(text)
    if ternary_hit is not None:
        outer_sign, ternary_branches = ternary_hit
        parsed_branches: list[tuple[str, float, float, float, float | None]] = []
        for br in ternary_branches:
            pb = _parse_molang_channel_zero_eval(br, bone_name, _depth=_depth + 1)
            if pb is not None:
                b_law, b_amp, b_ph, b_bs, b_freq = pb
                b_amp *= outer_sign
                b_bs *= outer_sign
                if clamp_bounds is not None:
                    c_lo, c_hi = clamp_bounds
                    lo_eff = float(np.clip(b_bs - abs(b_amp), c_lo, c_hi))
                    hi_eff = float(np.clip(b_bs + abs(b_amp), c_lo, c_hi))
                    b_bs = 0.5 * (lo_eff + hi_eff)
                    b_amp = math.copysign(0.5 * (hi_eff - lo_eff), b_amp if abs(b_amp) > 1e-9 else 1.0)
                parsed_branches.append((b_law, round(b_amp, 4), b_ph, round(b_bs, 4), b_freq))
        if not parsed_branches:
            return None
        return max(parsed_branches, key=lambda item: (abs(item[1]), abs(item[3])))

    # Pure constant expression (without anim_time or trig)
    if _TRIG_START_RE.search(text) is None:
        v0 = _eval_linear_molang_expr(text, 0.0)
        v1 = _eval_linear_molang_expr(text, 1.0)
        if v0 is not None and v1 is not None and abs(v1 - v0) <= 1e-6:
            const_val = v0
            if clamp_bounds is not None:
                const_val = float(np.clip(const_val, clamp_bounds[0], clamp_bounds[1]))
            if abs(const_val) < 0.25:
                return None
            return ("hinge", 0.0, 0.0, float(np.clip(const_val, -180.0, 180.0)), None)
        return None

    bias_total = 0.0
    # (amp_deg, phase_deg_effective, freq_hz_override, law_override_or_empty)
    trig_candidates: list[tuple[float, float, float | None, str]] = []

    for term_sign, term_core in _flatten_additive_terms(text):
        abs_hit = _find_balanced_call(term_core, _ABS_START_RE)
        if abs_hit is not None:
            a_start, a_end, _, abs_inner = abs_hit
            inner_trig = _find_balanced_trig_call(abs_inner)
            outer_aff = _eval_affine_wrapper(term_core, a_start, a_end)
            if inner_trig is not None and outer_aff is not None:
                it_start, it_end, it_fn, it_arg = inner_trig
                inner_aff = _eval_affine_wrapper(abs_inner, it_start, it_end)
                parsed_in = _parse_molang_inside_arg(it_arg)
                if inner_aff is not None and parsed_in is not None and abs(inner_aff[0]) <= 1e-6:
                    outer_c0, outer_c1 = outer_aff
                    bias_total += term_sign * outer_c0
                    arg_phase_deg, freq_override, _ = parsed_in
                    is_cos = it_fn == "cos"
                    if freq_override is None:
                        val_const = abs(
                            inner_aff[1]
                            * (math.cos(math.radians(arg_phase_deg)) if is_cos else math.sin(math.radians(arg_phase_deg)))
                        )
                        bias_total += term_sign * outer_c1 * val_const
                    else:
                        amp_rect = term_sign * outer_c1 * abs(inner_aff[1])
                        rect_freq = round(float(np.clip(2.0 * freq_override, 0.05, 20.0)), 4)
                        rect_phase_deg = 2.0 * (arg_phase_deg + (90.0 if is_cos else 0.0))
                        trig_candidates.append((amp_rect, rect_phase_deg, rect_freq, "hinge"))
                    continue

        trig_hit = _find_balanced_trig_call(term_core)
        if trig_hit is None:
            c0 = _eval_linear_molang_expr(term_core, 0.0)
            c1 = _eval_linear_molang_expr(term_core, 1.0)
            if c0 is not None and c1 is not None and abs(c1 - c0) <= 1e-6:
                bias_total += term_sign * c0
            continue

        t_start, t_end, trig_fn, inside_arg = trig_hit
        aff = _eval_affine_wrapper(term_core, t_start, t_end)
        if aff is None:
            continue
        aff_c0, aff_c1 = aff
        bias_total += term_sign * aff_c0
        scale = term_sign * aff_c1
        parsed_in = _parse_molang_inside_arg(inside_arg)
        if parsed_in is None:
            continue
        arg_phase_deg, freq_override, sin_sign = parsed_in
        is_cos = trig_fn == "cos"
        if freq_override is None:
            bias_total += scale * (math.cos(math.radians(arg_phase_deg)) if is_cos else math.sin(math.radians(arg_phase_deg)))
        else:
            amp_i = scale * (1.0 if is_cos else sin_sign)
            eff_phase_deg = (90.0 if is_cos else 0.0) + arg_phase_deg
            trig_candidates.append((amp_i, eff_phase_deg, freq_override, ""))

    if not trig_candidates:
        if abs(bias_total) >= 0.25:
            if clamp_bounds is not None:
                bias_total = float(np.clip(bias_total, clamp_bounds[0], clamp_bounds[1]))
            if abs(bias_total) >= 0.25:
                return ("hinge", 0.0, 0.0, round(float(np.clip(bias_total, -180.0, 180.0)), 4), None)
        return None

    amp_raw, phase_deg, freq_override, law_override = max(trig_candidates, key=lambda item: abs(item[0]))
    amp = float(np.clip(amp_raw, -180.0, 180.0))
    bias = float(np.clip(bias_total, -180.0, 180.0))

    if clamp_bounds is not None:
        c_lo, c_hi = clamp_bounds
        if law_override == "hinge":
            lo_raw = min(bias, bias + amp)
            hi_raw = max(bias, bias + amp)
            lo_eff = float(np.clip(lo_raw, c_lo, c_hi))
            hi_eff = float(np.clip(hi_raw, c_lo, c_hi))
            if amp >= 0.0:
                bias = lo_eff
                amp = hi_eff - lo_eff
            else:
                bias = hi_eff
                amp = lo_eff - hi_eff
        else:
            lo_eff = float(np.clip(bias - abs(amp), c_lo, c_hi))
            hi_eff = float(np.clip(bias + abs(amp), c_lo, c_hi))
            bias = 0.5 * (lo_eff + hi_eff)
            half_span = 0.5 * (hi_eff - lo_eff)
            amp = math.copysign(half_span, amp if abs(amp) > 1e-9 else 1.0)

    phase_offset = (phase_deg / 360.0) % 1.0
    if phase_offset >= 0.999999:
        phase_offset = 0.0

    wf = law_override or ("flap" if "wing" in bone_name.lower() else "wave")
    if abs(amp) < 0.1 and abs(bias) < 0.25:
        return None
    return (wf, round(amp, 4), round(phase_offset, 4), round(bias, 4), freq_override)


def _unwrap_keyframe_vec3(raw: Any) -> list[Any] | None:
    """Unwrap 3-component vector from plain list or Blockbench/Cobblemon 1-element nested list [[x, y, z]]."""
    cur = raw
    for _ in range(3):
        if isinstance(cur, (list, tuple)) and len(cur) == 1 and isinstance(cur[0], (list, tuple)):
            cur = cur[0]
        else:
            break
    if isinstance(cur, (list, tuple)) and len(cur) == 3:
        return [cur[0], cur[1], cur[2]]
    return None


def _extract_keyframe_vec3_entries(k_val: Any, *, allow_scalar: bool = False) -> list[list[Any]]:
    """Extract 3-channel keyframe vector(s) from plain vectors, Catmull-Rom pre/post dicts, or GeckoLib vector dicts."""
    if isinstance(k_val, dict):
        pre_raw = k_val.get("pre") if "pre" in k_val else None
        post_raw = k_val.get("post") if "post" in k_val else None
        pre_v = _unwrap_keyframe_vec3(pre_raw) if pre_raw is not None else None
        post_v = _unwrap_keyframe_vec3(post_raw) if post_raw is not None else None
        if allow_scalar:
            if pre_v is None and pre_raw is not None and _try_numeric_deg(pre_raw) is not None:
                s_pre = _try_numeric_deg(pre_raw)
                pre_v = [s_pre, s_pre, s_pre]
            if post_v is None and post_raw is not None and _try_numeric_deg(post_raw) is not None:
                s_post = _try_numeric_deg(post_raw)
                post_v = [s_post, s_post, s_post]
        if pre_v is not None and post_v is not None:
            return [pre_v, post_v] if pre_v != post_v else [post_v]
        if post_v is not None:
            return [post_v]
        if pre_v is not None:
            return [pre_v]
        if "vector" in k_val:
            vec_raw = k_val.get("vector")
            vec_v = _unwrap_keyframe_vec3(vec_raw)
            if vec_v is None and allow_scalar and _try_numeric_deg(vec_raw) is not None:
                s_vec = _try_numeric_deg(vec_raw)
                vec_v = [s_vec, s_vec, s_vec]
            if vec_v is not None:
                return [vec_v]
        return []
    plain = _unwrap_keyframe_vec3(k_val)
    if plain is not None:
        return [plain]
    if allow_scalar:
        s_plain = _try_numeric_deg(k_val)
        if s_plain is not None:
            return [[s_plain, s_plain, s_plain]]
    return []


def _try_numeric_deg(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, str):
        s = val.strip().replace(" ", "")
        if not s:
            return None
        try:
            f = float(s)
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def _extract_numeric_keyframe_knots(
    ch_raw: Any,
    *,
    dur: float,
    axis_sign: tuple[float, float, float],
    is_scale: bool = False,
) -> tuple[list[tuple[float, tuple[float, float, float]]], str] | None:
    """Parse numeric 3-vector keyframes into sorted (t, (x, y, z)) knots + interpolation mode.

    Returns ``None`` if the channel is absent or contains non-numeric Molang strings.
    """
    if ch_raw is None:
        return None

    def _convert_vec(raw_vec: list[Any]) -> tuple[float, float, float] | None:
        nums: list[float] = []
        for idx_c in range(3):
            nv = _try_numeric_deg(raw_vec[idx_c])
            if nv is None:
                return None
            signed = float(nv) * float(axis_sign[idx_c])
            if is_scale:
                signed = max(1e-4, signed)
            nums.append(0.0 if (not is_scale and abs(signed) < 1e-9) else round(signed, 6))
        return (nums[0], nums[1], nums[2])

    direct_entries = _extract_keyframe_vec3_entries(ch_raw, allow_scalar=is_scale)
    if direct_entries and not (isinstance(ch_raw, dict) and not ("vector" in ch_raw or "pre" in ch_raw or "post" in ch_raw)):
        cv = _convert_vec(direct_entries[-1])
        if cv is None:
            return None
        return ([(0.0, cv)], "linear")

    if not isinstance(ch_raw, dict) or not ch_raw:
        return None

    has_catmullrom = False
    raw_rows: list[tuple[float, int, tuple[float, float, float]]] = []
    seq = 0
    for t_str, k_val in ch_raw.items():
        try:
            t_num = float(t_str)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t_num):
            continue
        if isinstance(k_val, dict) and str(k_val.get("lerp_mode", "")).strip().lower() == "catmullrom":
            has_catmullrom = True
        entries = _extract_keyframe_vec3_entries(k_val, allow_scalar=is_scale)
        if not entries:
            return None
        for vec_e in entries:
            cv = _convert_vec(vec_e)
            if cv is None:
                return None
            t_clamped = round(float(np.clip(t_num, 0.0, dur)), 6)
            raw_rows.append((t_clamped, seq, cv))
            seq += 1

    if not raw_rows:
        return None
    raw_rows.sort(key=lambda item: (item[0], item[1]))
    dedup_knots: list[tuple[float, tuple[float, float, float]]] = []
    for t_c, _, vec_c in raw_rows:
        if dedup_knots and abs(t_c - dedup_knots[-1][0]) <= 1e-6:
            dedup_knots[-1] = (dedup_knots[-1][0], vec_c)
        else:
            dedup_knots.append((t_c, vec_c))
    return (dedup_knots, "catmullrom" if has_catmullrom else "linear")


def _fit_keyframe_channel_track(
    timed_vals: list[tuple[float, float]],
    *,
    dur: float,
    loop_val: bool,
    base_freq_hz: float,
    target_b: str,
    ax_name: str,
) -> TrackDecl | None:
    """Fit a phase-faithful TrackDecl (signed amplitude, phase_deg, bias_deg) to timestamped keyframes."""
    if not timed_vals:
        return None
    if len(timed_vals) == 1:
        v0 = timed_vals[0][1]
        if abs(v0) < 0.25:
            return None
        return TrackDecl(
            bone=target_b,
            law="hinge",
            axis=ax_name,
            amp_deg=0.0,
            freq_hz=base_freq_hz,
            phase_deg=0.0,
            bias_deg=round(float(np.clip(v0, -90.0, 90.0)), 4),
        )

    num_vals = [v for _, v in timed_vals]
    vmin, vmax = min(num_vals), max(num_vals)
    span = vmax - vmin
    if span < 0.25:
        if abs(num_vals[0]) >= 0.25:
            return TrackDecl(
                bone=target_b,
                law="hinge",
                axis=ax_name,
                amp_deg=0.0,
                freq_hz=base_freq_hz,
                phase_deg=0.0,
                bias_deg=round(float(np.clip(num_vals[0], -90.0, 90.0)), 4),
            )
        return None

    # Group (pre, post) pairs per distinct timestamp: pre = arriving limit, post = departing limit
    knots: list[tuple[float, float, float]] = []
    for t_val, v_val in timed_vals:
        if knots and abs(t_val - knots[-1][0]) <= 1e-6:
            t_prev, v_pre_prev, _ = knots[-1]
            knots[-1] = (t_prev, v_pre_prev, v_val)
        else:
            knots.append((t_val, v_val, v_val))

    if len(knots) == 1:
        bs_single = 0.5 * (vmax + vmin)
        amp_single = 0.5 * span
        ph_single = 0.0 if knots[0][2] >= knots[0][1] else 180.0
        wf_single = "flap" if "wing" in target_b.lower() else "wave"
        return TrackDecl(
            bone=target_b,
            law=wf_single,
            axis=ax_name,
            amp_deg=round(float(np.clip(amp_single, -120.0, 120.0)), 4),
            freq_hz=base_freq_hz,
            phase_deg=ph_single,
            bias_deg=round(float(np.clip(bs_single, -90.0, 90.0)), 4),
        )

    t_first, _, v_depart_start = knots[0]
    t_last, v_arrive_end, _ = knots[-1]
    t_span = t_last - t_first
    t_peak = max(timed_vals, key=lambda pair: pair[1])[0]
    t_trough = min(timed_vals, key=lambda pair: pair[1])[0]
    has_step_jump = any(abs(k_post - k_pre) > 0.15 * span for _, k_pre, k_post in knots)

    if not has_step_jump and t_span >= 0.45 * dur:
        if (
            abs(v_depart_start - vmin) < 0.15 * span
            and abs(v_arrive_end - vmin) < 0.15 * span
            and 0.20 * dur <= (t_peak - t_first) <= 0.80 * dur
        ):
            return TrackDecl(
                bone=target_b,
                law="hinge",
                axis=ax_name,
                amp_deg=round(float(np.clip(span, -120.0, 120.0)), 4),
                freq_hz=base_freq_hz,
                phase_deg=0.0,
                bias_deg=round(float(np.clip(vmin, -90.0, 90.0)), 4),
            )
        if (
            abs(v_depart_start - vmax) < 0.15 * span
            and abs(v_arrive_end - vmax) < 0.15 * span
            and 0.20 * dur <= (t_trough - t_first) <= 0.80 * dur
        ):
            return TrackDecl(
                bone=target_b,
                law="hinge",
                axis=ax_name,
                amp_deg=round(float(np.clip(-span, -120.0, 120.0)), 4),
                freq_hz=base_freq_hz,
                phase_deg=0.0,
                bias_deg=round(float(np.clip(vmax, -90.0, 90.0)), 4),
            )

    wf = "flap" if "wing" in target_b.lower() else "wave"
    safe_dur = max(1e-6, dur)
    segments: list[tuple[float, float, float, float]] = []
    for k_idx in range(len(knots) - 1):
        t0_s, _, y0_s = knots[k_idx]
        t1_s, y1_s, _ = knots[k_idx + 1]
        if t1_s - t0_s > 1e-6:
            segments.append((t0_s, t1_s, y0_s, y1_s))
    if loop_val and t_last < t_first + 0.85 * safe_dur and (t_first + safe_dur) - t_last > 1e-6:
        segments.append((t_last, t_first + safe_dur, knots[-1][2], knots[0][1]))

    if segments:
        seg_start = segments[0][0]
        seg_end = segments[-1][1]
        n_quad = 64
        s_grid = seg_start + (np.arange(n_quad, dtype=np.float64) + 0.5) * ((seg_end - seg_start) / n_quad)
        y_grid = np.empty(n_quad, dtype=np.float64)
        seg_ptr = 0
        for q_i, s_t in enumerate(s_grid):
            while seg_ptr + 1 < len(segments) and s_t > segments[seg_ptr][1]:
                seg_ptr += 1
            st0, st1, sy0, sy1 = segments[seg_ptr]
            alpha = float(np.clip((s_t - st0) / max(1e-9, st1 - st0), 0.0, 1.0))
            y_grid[q_i] = sy0 + alpha * (sy1 - sy0)

        ang = 2.0 * math.pi * (s_grid / safe_dur)
        design = np.column_stack([np.ones_like(ang), np.sin(ang), np.cos(ang)])
        coeffs, _, _, _ = np.linalg.lstsq(design, y_grid, rcond=1e-6)
        c0, cs, cc = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        amp_fit = math.hypot(cs, cc)
        if amp_fit >= 0.15 * span:
            bs = c0
            amp = max(amp_fit, 0.5 * span) if len(knots) <= 5 else amp_fit
            phase_deg = math.degrees(math.atan2(cc, cs)) % 360.0
        else:
            bs = 0.5 * (vmax + vmin)
            amp = 0.5 * span
            phase_deg = 0.0 if segments[0][3] >= segments[0][2] else 180.0
    else:
        bs = 0.5 * (vmax + vmin)
        amp = 0.5 * span
        phase_deg = 0.0

    if phase_deg >= 359.995:
        phase_deg = 0.0
    return TrackDecl(
        bone=target_b,
        law=wf,
        axis=ax_name,
        amp_deg=round(float(np.clip(amp, -120.0, 120.0)), 4),
        freq_hz=base_freq_hz,
        phase_deg=round(phase_deg, 2),
        bias_deg=round(float(np.clip(bs, -90.0, 90.0)), 4),
    )


def decompile_bedrock_animation_to_behavior(
    anim_doc: dict[str, Any],
    program: Program,
    *,
    raw_to_safe_bone: dict[str, str] | None = None,
    prefer_keyframes: bool = True,
    _bedrock_left_handed: bool = True,
) -> BehaviorDecl:
    """Convert Bedrock / Cobblemon / GeckoLib .animation.json into a validated @behavior BehaviorDecl (zero eval).

    Under ``_bedrock_left_handed=True`` (Bedrock ``1.8.0`` animation convention, C5),
    animated X & Y rotation deltas and X translation deltas are negated to map from
    Bedrock's left-handed frame into SyncCraft's right-handed model space.
    """
    anims = anim_doc.get("animations")
    if not isinstance(anims, dict) or not anims:
        raise ValueError("Bedrock .animation.json missing non-empty 'animations' dictionary")

    rot_axis_sign = (-1.0, -1.0, 1.0) if _bedrock_left_handed else (1.0, 1.0, 1.0)
    pos_axis_sign = (-1.0, 1.0, 1.0) if _bedrock_left_handed else (1.0, 1.0, 1.0)

    compiled = compile_geometry(program)
    valid_bones = set(compiled.bone_by_name)
    bone_lookup: dict[str, str] = {b: b for b in valid_bones}
    bone_lookup.update({b.lower(): b for b in valid_bones})
    if raw_to_safe_bone:
        for raw_k, safe_v in raw_to_safe_bone.items():
            if safe_v in valid_bones:
                bone_lookup[raw_k] = safe_v
                bone_lookup[raw_k.lower()] = safe_v

    # Seed used_clip_ids with all existing Program/Scene identifiers to prevent global AST namespace collisions
    used_clip_ids: set[str] = (
        set(compiled.bone_by_name)
        | set(compiled.box_by_name)
        | set(compiled.locator_by_name)
        | {p.name for p in program.palettes}
        | {m.name for m in program.materials}
        | {m.name for m in program.muscles}
        | {t.name for t in getattr(program, "textures", ())}
    )
    if program.kinematics is not None:
        used_clip_ids |= {c.name for c in program.kinematics.ik_chains}
        used_clip_ids |= {c.name for c in program.kinematics.look_chains}
    if program.behavior is not None:
        used_clip_ids |= {c.name for c in program.behavior.clips}
        used_clip_ids |= {s.name for s in program.behavior.states}

    clips: list[ClipDecl] = []

    for raw_clip_name, spec in anims.items():
        if not isinstance(spec, dict):
            continue
        short_name = str(raw_clip_name).split(".")[-1]
        cand_short = f"{short_name}_clip" if short_name in used_clip_ids else short_name
        clip_name = _sanitize_ident(cand_short, "clip", used_clip_ids)
        loop_raw = spec.get("loop", True)
        loop_val = loop_raw in (True, "loop", "hold_on_last_frame")

        bones_anim = spec.get("bones") or {}
        max_kf_t = 0.0
        if isinstance(bones_anim, dict):
            for ch_dict in bones_anim.values():
                if not isinstance(ch_dict, dict):
                    continue
                for ch_name in ("rotation", "position", "scale"):
                    ch_v = ch_dict.get(ch_name)
                    if isinstance(ch_v, dict):
                        for k_t in ch_v:
                            if k_t in ("vector", "pre", "post", "lerp_mode"):
                                continue
                            try:
                                tf = float(k_t)
                                if math.isfinite(tf) and tf > max_kf_t:
                                    max_kf_t = tf
                            except (TypeError, ValueError):
                                continue

        raw_dur = float(spec.get("animation_length", 0.0) or 0.0)
        eff_dur = max(raw_dur, max_kf_t) if (raw_dur > 0.0 or max_kf_t > 0.0) else 1.5
        dur = float(np.clip(eff_dur, 0.1, 60.0))
        base_freq_hz = round(float(np.clip(1.0 / dur, 0.05, 20.0)), 4)

        tracks: list[TrackDecl] = []
        keyframes: list[Any] = []
        if isinstance(bones_anim, dict):
            for raw_b, channels in bones_anim.items():
                target_b = bone_lookup.get(str(raw_b)) or bone_lookup.get(str(raw_b).lower())
                if target_b is None or not isinstance(channels, dict):
                    continue

                if _KeyframeTrackDecl is not None:
                    pos_parsed = _extract_numeric_keyframe_knots(
                        channels.get("position"),
                        dur=dur,
                        axis_sign=pos_axis_sign,
                        is_scale=False,
                    )
                    if pos_parsed is not None:
                        pos_knots, pos_interp = pos_parsed
                        if len(pos_knots) >= 2 or any(abs(c) > 1e-6 for _, v in pos_knots for c in v):
                            keyframes.append(
                                _KeyframeTrackDecl(
                                    bone=target_b,
                                    channel="position",
                                    keys=tuple(pos_knots),
                                    interpolation=pos_interp,
                                )
                            )
                    scale_parsed = _extract_numeric_keyframe_knots(
                        channels.get("scale"),
                        dur=dur,
                        axis_sign=(1.0, 1.0, 1.0),
                        is_scale=True,
                    )
                    if scale_parsed is not None:
                        sc_knots, sc_interp = scale_parsed
                        if len(sc_knots) >= 2 or any(abs(c - 1.0) > 1e-6 for _, v in sc_knots for c in v):
                            keyframes.append(
                                _KeyframeTrackDecl(
                                    bone=target_b,
                                    channel="scale",
                                    keys=tuple(sc_knots),
                                    interpolation=sc_interp,
                                )
                            )

                rot_ch = channels.get("rotation")
                if rot_ch is None:
                    continue

                if _KeyframeTrackDecl is not None:
                    rot_kf_parsed = _extract_numeric_keyframe_knots(
                        rot_ch,
                        dur=dur,
                        axis_sign=rot_axis_sign,
                        is_scale=False,
                    )
                    if rot_kf_parsed is not None:
                        rot_knots, rot_interp = rot_kf_parsed
                        has_nonzero_rot = any(abs(c) > 1e-6 for _, v in rot_knots for c in v)
                        if target_b == program.root.name:
                            if len(rot_knots) >= 2 or has_nonzero_rot:
                                keyframes.append(
                                    _KeyframeTrackDecl(
                                        bone=target_b,
                                        channel="rotation",
                                        keys=tuple(rot_knots),
                                        interpolation=rot_interp,
                                    )
                                )
                            continue
                        if prefer_keyframes and len(rot_knots) >= 2 and has_nonzero_rot:
                            keyframes.append(
                                _KeyframeTrackDecl(
                                    bone=target_b,
                                    channel="rotation",
                                    keys=tuple(rot_knots),
                                    interpolation=rot_interp,
                                )
                            )
                            continue

                if target_b == program.root.name:
                    continue

                rot_vec_direct = _unwrap_keyframe_vec3(rot_ch)
                if rot_vec_direct is not None:
                    for ax_idx, ax_name in enumerate(("x", "y", "z")):
                        sgn = rot_axis_sign[ax_idx]
                        parsed = _parse_molang_channel_zero_eval(rot_vec_direct[ax_idx], target_b)
                        if parsed is not None:
                            wf, amp, ph, bs, freq_ov = parsed
                            tracks.append(
                                TrackDecl(
                                    bone=target_b,
                                    law=wf,
                                    axis=ax_name,
                                    amp_deg=float(np.clip(round(amp * sgn, 4), -120.0, 120.0)),
                                    freq_hz=freq_ov if freq_ov is not None else base_freq_hz,
                                    phase_deg=round(ph * 360.0, 2),
                                    bias_deg=float(np.clip(round(bs * sgn, 4), -90.0, 90.0)),
                                )
                            )
                elif isinstance(rot_ch, dict) and rot_ch:
                    raw_kf_rows: list[tuple[float, int, list[Any]]] = []
                    seq_idx = 0
                    for t_str, k_val in rot_ch.items():
                        try:
                            t_num = float(t_str)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(t_num):
                            continue
                        for vec_entry in _extract_keyframe_vec3_entries(k_val):
                            raw_kf_rows.append((t_num, seq_idx, vec_entry))
                            seq_idx += 1
                    raw_kf_rows.sort(key=lambda item: (item[0], item[1]))
                    if raw_kf_rows:
                        for ax_idx, ax_name in enumerate(("x", "y", "z")):
                            sgn = rot_axis_sign[ax_idx]
                            timed_num_vals: list[tuple[float, float]] = []
                            molang_parsed_list: list[tuple[str, float, float, float, float | None]] = []
                            for t_row, _, vec_row in raw_kf_rows:
                                cell = vec_row[ax_idx]
                                num_v = _try_numeric_deg(cell)
                                if num_v is not None:
                                    timed_num_vals.append((t_row, num_v * sgn))
                                else:
                                    p_mol = _parse_molang_channel_zero_eval(cell, target_b)
                                    if p_mol is not None:
                                        molang_parsed_list.append(p_mol)

                            if molang_parsed_list:
                                wf, amp, ph, bs, freq_ov = max(
                                    molang_parsed_list, key=lambda p: (abs(p[1]), abs(p[3]))
                                )
                                tracks.append(
                                    TrackDecl(
                                        bone=target_b,
                                        law=wf,
                                        axis=ax_name,
                                        amp_deg=float(np.clip(round(amp * sgn, 4), -120.0, 120.0)),
                                        freq_hz=freq_ov if freq_ov is not None else base_freq_hz,
                                        phase_deg=round(ph * 360.0, 2),
                                        bias_deg=float(np.clip(round(bs * sgn, 4), -90.0, 90.0)),
                                    )
                                )
                            elif timed_num_vals:
                                fitted_tr = _fit_keyframe_channel_track(
                                    timed_num_vals,
                                    dur=dur,
                                    loop_val=loop_val,
                                    base_freq_hz=base_freq_hz,
                                    target_b=target_b,
                                    ax_name=ax_name,
                                )
                                if fitted_tr is not None:
                                    tracks.append(fitted_tr)

        if not tracks and not (_HAS_DSL_V32 and keyframes):
            non_root_bones = sorted(valid_bones - {program.root.name})
            if not non_root_bones:
                continue
            tracks.append(
                TrackDecl(
                    bone=non_root_bones[0],
                    law="wave",
                    axis="y",
                    amp_deg=0.0,
                    freq_hz=base_freq_hz,
                )
            )

        if _HAS_DSL_V32 and "keyframes" in ClipDecl.__dataclass_fields__:
            clips.append(
                ClipDecl(
                    name=clip_name,
                    duration_s=round(dur, 4),
                    loop=loop_val,
                    tracks=tuple(tracks),
                    keyframes=tuple(keyframes),
                )
            )
        else:
            clips.append(ClipDecl(name=clip_name, duration_s=round(dur, 4), loop=loop_val, tracks=tuple(tracks)))

    if not clips:
        raise ValueError("No valid animation clips found in .animation.json")
    return BehaviorDecl(clips=tuple(clips))


def decompile_bbmodel_animations_to_behavior(
    doc: dict[str, Any],
    program: Program,
    *,
    group_uuid_to_safe_bone: dict[str, str] | None = None,
) -> BehaviorDecl | None:
    """Convert Blockbench 5.0 .bbmodel ``animations`` list into a validated @behavior BehaviorDecl (zero eval)."""
    anims_list = doc.get("animations")
    if not isinstance(anims_list, list) or not anims_list:
        return None

    bed_anims: dict[str, Any] = {}
    raw_to_safe: dict[str, str] = dict(group_uuid_to_safe_bone or {})

    for idx, anim in enumerate(anims_list):
        if not isinstance(anim, dict):
            continue
        aname = str(anim.get("name") or f"clip_{idx}")
        loop_mode = anim.get("loop", "loop")
        is_loop = loop_mode in (True, "loop", "hold")
        length_s = float(anim.get("length", 1.5) or 1.5)
        animators = anim.get("animators") or {}
        bones_dict: dict[str, dict[str, Any]] = {}
        if isinstance(animators, dict):
            for anim_key, animator in animators.items():
                if not isinstance(animator, dict):
                    continue
                bone_key = raw_to_safe.get(str(anim_key)) or str(animator.get("name") or anim_key)
                kfs = animator.get("keyframes") or []
                if not isinstance(kfs, list):
                    continue
                ch_timelines: dict[str, dict[str, Any]] = {"rotation": {}, "position": {}, "scale": {}}
                for kf in kfs:
                    if not isinstance(kf, dict):
                        continue
                    ch_name = str(kf.get("channel", ""))
                    if ch_name not in ch_timelines:
                        continue
                    t_key = f"{float(kf.get('time', 0.0)):.6f}"
                    dps = kf.get("data_points") or []
                    if not isinstance(dps, list) or not dps or not isinstance(dps[0], dict):
                        continue
                    default_c = 1.0 if ch_name == "scale" else 0.0
                    vecs: list[list[Any]] = []
                    for dp in dps[:2]:
                        if isinstance(dp, dict):
                            vx = dp.get("x", default_c)
                            vy = dp.get("y", default_c)
                            vz = dp.get("z", default_c)
                            nx, ny, nz = _try_numeric_deg(vx), _try_numeric_deg(vy), _try_numeric_deg(vz)
                            vecs.append([
                                nx if nx is not None else vx,
                                ny if ny is not None else vy,
                                nz if nz is not None else vz,
                            ])
                    interp_mode = str(kf.get("interpolation", "linear")).strip().lower()
                    if len(vecs) == 2:
                        ch_timelines[ch_name][t_key] = {
                            "pre": vecs[0],
                            "post": vecs[1],
                            "lerp_mode": interp_mode,
                        }
                    elif len(vecs) == 1:
                        if interp_mode == "catmullrom":
                            ch_timelines[ch_name][t_key] = {
                                "post": vecs[0],
                                "lerp_mode": "catmullrom",
                            }
                        else:
                            ch_timelines[ch_name][t_key] = vecs[0]
                non_empty_chs = {k: v for k, v in ch_timelines.items() if v}
                if non_empty_chs:
                    bones_dict[bone_key] = non_empty_chs
        bed_anims[aname] = {
            "loop": is_loop,
            "animation_length": length_s,
            "bones": bones_dict,
        }

    if not bed_anims:
        return None
    try:
        return decompile_bedrock_animation_to_behavior(
            {"animations": bed_anims},
            program,
            raw_to_safe_bone=raw_to_safe,
            _bedrock_left_handed=False,
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Inverse Material Posterior, Forward OKLab Residual & Stamp Extraction
# ---------------------------------------------------------------------------


def _serialize_oklab_residual_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable summary of OKLab residual statistics (omitting raw ndarray buffers)."""
    out: dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 6)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _evaluate_material_oklab_residual(
    program: Program,
    scene: Any,
    domain: Any,
    posed_kinematics: Any,
    material_program: MaterialProgram,
    uv_map: dict[str, dict[str, list[float]]],
    atlas_rgba: np.ndarray,
    observed_crops: list[tuple[str, str, int, int, int, int, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Evaluate exact forward OKLab residual (observed - procedural) across all active UV texels (H6)."""
    h_img, w_img = atlas_rgba.shape[:2]
    residual_canvas = np.zeros((h_img, w_img, 3), dtype=np.float32)
    procedural_canvas = np.zeros((h_img, w_img, 3), dtype=np.float32)
    active_mask = np.zeros((h_img, w_img), dtype=bool)

    if not observed_crops:
        return {
            "rmse": 0.0,
            "rmse_delta_e": 0.0,
            "mae": 0.0,
            "mean_delta_e": 0.0,
            "p95_delta_e": 0.0,
            "max_delta_e": 0.0,
            "active_texel_count": 0,
            "explained_variance_ratio": 1.0,
            "residual_canvas_shape": (int(h_img), int(w_img), 3),
            "residual_oklab_texels": residual_canvas,
            "procedural_oklab_texels": procedural_canvas,
            "active_uv_mask": active_mask,
        }

    state = initialize_surface_state(domain)
    opt = evaluate_optical_fields(program, domain, material_program, state, posed_kinematics)

    delta_sq_channels: list[np.ndarray] = []
    delta_e_list: list[np.ndarray] = []
    obs_active_list: list[np.ndarray] = []

    for box_name, face, x0, y0, x1, y1, obs_lab, alpha_mask in observed_crops:
        fkey = (box_name, face)
        if fkey not in opt.patches:
            continue
        pred_lin = opt.patches[fkey].base_color_linear
        pred_lab = linear_srgb_to_oklab(pred_lin)
        crop_h = y1 - y0
        crop_w = x1 - x0
        if pred_lab.shape[0] != crop_h or pred_lab.shape[1] != crop_w:
            r_idx = np.clip(
                np.floor((np.arange(crop_h, dtype=np.float64) + 0.5) * pred_lab.shape[0] / max(1, crop_h)).astype(np.intp),
                0,
                pred_lab.shape[0] - 1,
            )
            c_idx = np.clip(
                np.floor((np.arange(crop_w, dtype=np.float64) + 0.5) * pred_lab.shape[1] / max(1, crop_w)).astype(np.intp),
                0,
                pred_lab.shape[1] - 1,
            )
            pred_lab_crop = pred_lab[np.ix_(r_idx, c_idx)]
        else:
            pred_lab_crop = pred_lab

        diff = (obs_lab - pred_lab_crop).astype(np.float64)
        mask_bool = np.asarray(alpha_mask, dtype=bool)
        if not np.any(mask_bool):
            continue

        cur_sub_mask = active_mask[y0:y1, x0:x1]
        write_mask = mask_bool & (~cur_sub_mask)
        if np.any(write_mask):
            residual_canvas[y0:y1, x0:x1][write_mask] = diff[write_mask].astype(np.float32)
            procedural_canvas[y0:y1, x0:x1][write_mask] = pred_lab_crop[write_mask].astype(np.float32)
            active_mask[y0:y1, x0:x1] |= write_mask

        active_diff = diff[mask_bool]
        delta_sq_channels.append((active_diff ** 2).ravel())
        delta_e_list.append(np.linalg.norm(active_diff, axis=-1).ravel())
        obs_active_list.append(obs_lab[mask_bool].reshape(-1, 3))

    if not delta_e_list:
        return {
            "rmse": 0.0,
            "rmse_delta_e": 0.0,
            "mae": 0.0,
            "mean_delta_e": 0.0,
            "p95_delta_e": 0.0,
            "max_delta_e": 0.0,
            "active_texel_count": 0,
            "explained_variance_ratio": 1.0,
            "residual_canvas_shape": (int(h_img), int(w_img), 3),
            "residual_oklab_texels": residual_canvas,
            "procedural_oklab_texels": procedural_canvas,
            "active_uv_mask": active_mask,
        }

    all_de = np.concatenate(delta_e_list, axis=0)
    all_sq = np.concatenate(delta_sq_channels, axis=0)
    all_obs = np.concatenate(obs_active_list, axis=0)

    rmse_channel = float(math.sqrt(float(np.mean(all_sq))))
    rmse_de = float(math.sqrt(float(np.mean(all_de ** 2))))
    mae_channel = float(np.mean(np.sqrt(all_sq)))
    mean_de = float(np.mean(all_de))
    p95_de = float(np.percentile(all_de, 95))
    max_de = float(np.max(all_de))
    obs_var = float(np.sum(np.var(all_obs, axis=0)))
    res_mse_sum = float(np.mean(all_de ** 2))
    r2 = 1.0 - (res_mse_sum / max(1e-9, obs_var)) if obs_var > 1e-7 else (1.0 if rmse_de < 1e-3 else 0.0)

    return {
        "rmse": rmse_channel,
        "rmse_delta_e": rmse_de,
        "mae": mae_channel,
        "mean_delta_e": mean_de,
        "p95_delta_e": p95_de,
        "max_delta_e": max_de,
        "active_texel_count": int(np.sum(active_mask)),
        "explained_variance_ratio": float(np.clip(r2, -1.0, 1.0)),
        "residual_canvas_shape": (int(h_img), int(w_img), 3),
        "residual_oklab_texels": residual_canvas,
        "procedural_oklab_texels": procedural_canvas,
        "active_uv_mask": active_mask,
    }


@dataclass(frozen=True)
class MaterialHypothesis:
    """Ranked candidate MaterialSpec inferred from observed surface maps."""

    spec: MaterialSpec
    posterior_weight: float
    oklab_residual_rmse: float
    rationale: str
    oklab_residual_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "posterior_weight": round(self.posterior_weight, 5),
            "oklab_residual_rmse": round(self.oklab_residual_rmse, 5),
            "rationale": self.rationale,
            "oklab_residual_stats": _serialize_oklab_residual_summary(self.oklab_residual_stats),
        }


@dataclass(frozen=True)
class IngestedAssetRecord:
    """Complete verified output of P1-03e / ADR 0006 real-asset inverse inference."""

    provenance: AssetProvenance
    format_kind: str
    program: Program
    vertex_reconstruction_error: float
    material_hypotheses: tuple[MaterialHypothesis, ...]
    best_material_program: MaterialProgram
    extracted_stamps: tuple[StampOverride, ...]
    identifiability_score: float
    has_labpbr_auxiliary_maps: bool
    license_track: str = "clean_room_permissive"
    commercial_weights_eligible: bool = True
    oklab_residual_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from synccraft.dsl.parser import format_program

        return {
            "asset_id": self.provenance.asset_id,
            "license_spdx": self.provenance.license_spdx,
            "provenance": self.provenance.to_dict(),
            "license_track": self.license_track,
            "commercial_weights_eligible": self.commercial_weights_eligible,
            "format_kind": self.format_kind,
            "program_dsl": format_program(self.program),
            "vertex_reconstruction_error": self.vertex_reconstruction_error,
            "identifiability_score": round(self.identifiability_score, 4),
            "has_labpbr_auxiliary_maps": self.has_labpbr_auxiliary_maps,
            "oklab_residual_stats": _serialize_oklab_residual_summary(self.oklab_residual_stats),
            "extracted_stamps": [s.to_dict() for s in self.extracted_stamps],
            "material_hypotheses": [h.to_dict() for h in self.material_hypotheses],
        }


def infer_material_posterior(
    program: Program,
    uv_map: dict[str, dict[str, list[float]]],
    atlas_rgba: np.ndarray,
    *,
    labpbr_n_rgba: np.ndarray | None = None,
    labpbr_s_rgba: np.ndarray | None = None,
    top_k: int = 3,
    seed: int = 404,
    return_residual: bool = False,
) -> (
    tuple[tuple[MaterialHypothesis, ...], tuple[StampOverride, ...], float]
    | tuple[tuple[MaterialHypothesis, ...], tuple[StampOverride, ...], float, dict[str, Any]]
):
    """Infer top-K MaterialSpec hypotheses, StampOverrides, and measured OKLab residuals from texture maps."""
    h_img, w_img = atlas_rgba.shape[:2]
    scene = compile_geometry(program)

    all_lab_pixels: list[np.ndarray] = []
    observed_crops: list[tuple[str, str, int, int, int, int, np.ndarray, np.ndarray]] = []
    grad_u_vals: list[float] = []
    grad_v_vals: list[float] = []
    high_freq_vals: list[float] = []
    extracted_stamps: list[StampOverride] = []

    spec_s_g: list[np.ndarray] = []
    spec_s_r: list[np.ndarray] = []
    spec_s_b: list[np.ndarray] = []

    for box in scene.boxes:
        if box.name not in uv_map:
            continue
        for face in FACES:
            uv_rect = uv_map[box.name].get(face)
            if not uv_rect:
                continue
            if abs(float(uv_rect[2]) - float(uv_rect[0])) < 1e-6 or abs(float(uv_rect[3]) - float(uv_rect[1])) < 1e-6:
                continue
            x0 = int(np.clip(min(uv_rect[0], uv_rect[2]), 0, w_img - 1))
            x1 = int(np.clip(math.ceil(max(uv_rect[0], uv_rect[2])), x0 + 1, w_img))
            y0 = int(np.clip(min(uv_rect[1], uv_rect[3]), 0, h_img - 1))
            y1 = int(np.clip(math.ceil(max(uv_rect[1], uv_rect[3])), y0 + 1, h_img))

            crop_rgba = atlas_rgba[y0:y1, x0:x1]
            alpha_mask = crop_rgba[..., 3] > 0
            if crop_rgba.size == 0 or not np.any(alpha_mask):
                continue

            lin = srgb_u8_to_linear(crop_rgba[..., :3])
            lab = linear_srgb_to_oklab(lin)
            observed_crops.append((box.name, face, x0, y0, x1, y1, lab, alpha_mask))
            all_lab_pixels.append(lab[alpha_mask].reshape(-1, 3))

            if lab.shape[1] >= 2:
                grad_u_vals.append(float(np.mean(np.abs(np.diff(lab[..., 0], axis=1)))))
            if lab.shape[0] >= 2:
                grad_v_vals.append(float(np.mean(np.abs(np.diff(lab[..., 0], axis=0)))))
            if lab.shape[0] >= 3 and lab.shape[1] >= 3:
                center_l = lab[1:-1, 1:-1, 0]
                lap_l = (
                    lab[:-2, 1:-1, 0] + lab[2:, 1:-1, 0] + lab[1:-1, :-2, 0] + lab[1:-1, 2:, 0] - 4.0 * center_l
                )
                high_freq_vals.append(float(np.mean(np.abs(lap_l))))

            # Detect localized high-contrast artist StampOverrides (eyes, emblems, warning stripes)
            if lab.shape[0] >= 3 and lab.shape[1] >= 3:
                med_lab = np.median(lab[alpha_mask].reshape(-1, 3), axis=0)
                delta_e = np.linalg.norm(lab - med_lab[None, None, :], axis=-1)
                outlier = (delta_e > 0.20) & alpha_mask
                outlier_frac = float(np.mean(outlier))
                if 0.02 <= outlier_frac <= 0.35:
                    rs, cs = np.where(outlier)
                    wu, hv = face_uv_extents_units(box.size, face)
                    u_min = float(cs.min()) / lab.shape[1] * wu
                    u_max = float(cs.max() + 1) / lab.shape[1] * wu
                    v_min = float(rs.min()) / lab.shape[0] * hv
                    v_max = float(rs.max() + 1) / lab.shape[0] * hv
                    mean_out_lin = lin[outlier].mean(axis=0)
                    out_lch = linear_srgb_to_oklch(mean_out_lin)
                    aspect = (u_max - u_min) / max(1e-6, v_max - v_min)
                    shape_guess = "stripe_u" if aspect > 2.2 else ("stripe_v" if aspect < 0.45 else "rect")
                    extracted_stamps.append(
                        StampOverride(
                            box_name=box.name,
                            face=face,
                            u_min=round(u_min, 4),
                            v_min=round(v_min, 4),
                            u_max=round(max(u_min + 0.1, u_max), 4),
                            v_max=round(max(v_min + 0.1, v_max), 4),
                            lch=(float(out_lch[0]), float(out_lch[1]), float(out_lch[2])),
                            shape=shape_guess,
                        )
                    )

            if labpbr_s_rgba is not None and labpbr_s_rgba.shape[:2] == (h_img, w_img):
                scrop = labpbr_s_rgba[y0:y1, x0:x1]
                spec_s_r.append(scrop[..., 0][alpha_mask].ravel())
                spec_s_g.append(scrop[..., 1][alpha_mask].ravel())
                spec_s_b.append(scrop[..., 2][alpha_mask].ravel())

    if not all_lab_pixels:
        mat_fallback_name = program.palettes[0].name if program.palettes else "mat_0"
        default_spec = MaterialSpec(name=mat_fallback_name, family="polymer_dielectric", seed=seed)
        empty_stats = {
            "rmse": 0.0,
            "rmse_delta_e": 0.0,
            "mae": 0.0,
            "mean_delta_e": 0.0,
            "p95_delta_e": 0.0,
            "max_delta_e": 0.0,
            "active_texel_count": 0,
            "explained_variance_ratio": 1.0,
            "residual_canvas_shape": (int(h_img), int(w_img), 3),
            "residual_oklab_texels": np.zeros((h_img, w_img, 3), dtype=np.float32),
            "procedural_oklab_texels": np.zeros((h_img, w_img, 3), dtype=np.float32),
            "active_uv_mask": np.zeros((h_img, w_img), dtype=bool),
        }
        hyp_fallback = (
            MaterialHypothesis(
                default_spec,
                1.0,
                0.0,
                "empty_or_transparent_atlas_fallback",
                oklab_residual_stats=empty_stats,
            ),
        )
        if return_residual:
            return hyp_fallback, (), 0.15, empty_stats
        return hyp_fallback, (), 0.15

    cat_lab = np.concatenate(all_lab_pixels, axis=0)
    q25 = np.percentile(cat_lab, 25, axis=0)
    q75 = np.percentile(cat_lab, 75, axis=0)
    med = np.median(cat_lab, axis=0)

    def _oklab_to_lch_tuple(lab_vec: np.ndarray) -> tuple[float, float, float]:
        L = float(np.clip(lab_vec[0], 0.04, 0.96))
        C = float(max(0.0, math.hypot(float(lab_vec[1]), float(lab_vec[2]))))
        H = float(math.degrees(math.atan2(float(lab_vec[2]), float(lab_vec[1]))) % 360.0)
        return (round(L, 4), round(C, 4), round(H, 2))

    dom_lch = _oklab_to_lch_tuple(q75)
    sec_lch = _oklab_to_lch_tuple(q25)
    med_lch = _oklab_to_lch_tuple(med)

    l_std = float(np.std(cat_lab[:, 0]))
    chroma_med = med_lch[1]
    hue_med = med_lch[2]
    gu = float(np.mean(grad_u_vals)) if grad_u_vals else 0.01
    gv = float(np.mean(grad_v_vals)) if grad_v_vals else 0.01
    aniso_ratio = max(gu, gv) / max(1e-4, min(gu, gv))
    hf_energy = float(np.mean(high_freq_vals)) if high_freq_vals else 0.01

    has_pbr = labpbr_s_rgba is not None or labpbr_n_rgba is not None
    obs_metal_frac = 0.0
    obs_rough = 0.45
    obs_sss_frac = 0.0
    obs_poros = 0.20

    if spec_s_g:
        g_all = np.concatenate(spec_s_g)
        r_all = np.concatenate(spec_s_r)
        b_all = np.concatenate(spec_s_b)
        obs_metal_frac = float(np.mean(g_all >= 230))
        obs_rough = float(np.clip(1.0 - np.mean(r_all) / 255.0, 0.04, 0.98))
        obs_sss_frac = float(np.mean(b_all >= 65))
        obs_poros = float(np.mean(np.where(b_all <= 64, b_all / 64.0, 0.1)))

    # Score all 10 physical material families against observed spectral/Oklab/LabPBR features
    scores: dict[str, tuple[float, str]] = {}
    for fam in sorted(MATERIAL_FAMILIES):
        log_score = 0.0
        notes: list[str] = []

        if fam == "brushed_metal":
            log_score += 2.2 * obs_metal_frac if spec_s_g else (0.9 if chroma_med < 0.04 else -0.8)
            if aniso_ratio > 1.35:
                log_score += 0.8
                notes.append("directional_groove_gradients")
        elif fam == "painted_metal":
            if spec_s_g:
                partial_metal = 1.0 if 0.05 < obs_metal_frac < 0.85 else (0.4 if obs_metal_frac >= 0.85 else -0.3)
                log_score += 1.8 * partial_metal
                notes.append("bimodal_dielectric_coat_conductor_f0")
            else:
                log_score += 0.75 if (chroma_med > 0.07 and l_std > 0.04) else 0.1
                notes.append("chromatic_coat_with_edge_contrast")
        elif fam == "solid_wood":
            warm_hue = 1.0 if 35.0 <= hue_med <= 90.0 and chroma_med >= 0.04 else -0.5
            log_score += 0.9 * warm_hue + (0.7 if aniso_ratio > 1.25 else 0.0)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("warm_orthotropic_ring_statistics")
        elif fam == "weathered_stone":
            low_chroma = 1.0 if chroma_med < 0.055 else -0.4
            log_score += 0.85 * low_chroma + (0.6 if obs_poros > 0.3 else 0.2)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("low_chroma_clastic_variance")
        elif fam == "patterned_skin":
            log_score += (1.1 if l_std > 0.08 else 0.1) + (1.4 * obs_sss_frac if spec_s_g else 0.3)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.5
            notes.append("bimodal_spot_stripe_contrast")
        elif fam == "keratin_scales":
            log_score += (0.95 if hf_energy > 0.06 else 0.0) + (0.6 * obs_sss_frac if spec_s_g else 0.2)
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.2
            notes.append("periodic_cellular_laplacian")
        elif fam == "woven_fabric":
            iso_weave = 1.0 if (hf_energy > 0.05 and aniso_ratio < 1.35) else 0.0
            log_score += 0.85 * iso_weave + (0.5 if obs_poros > 0.4 else 0.1)
            if spec_s_g and obs_metal_frac > 0.1:
                log_score -= 2.5
            notes.append("bidirectional_yarn_frequency")
        elif fam == "polymer_dielectric":
            smooth_uniform = 1.0 if l_std < 0.06 else 0.1
            log_score += 0.8 * smooth_uniform
            if spec_s_g and obs_metal_frac > 0.2:
                log_score -= 2.2
            notes.append("uniform_dielectric_reflectance")
        elif fam == "lichen_biofilm":
            green_hue = 1.0 if 100.0 <= hue_med <= 165.0 else -0.4
            log_score += 0.85 * green_hue + (0.4 if l_std > 0.06 else 0.0)
            if spec_s_g and obs_metal_frac > 0.1:
                log_score -= 2.5
            notes.append("chlorophyll_hue_cluster")
        elif fam == "thin_film_iridescence":
            multi_hue = float(np.std(cat_lab[:, 1]) + np.std(cat_lab[:, 2]))
            log_score += 1.3 if multi_hue > 0.10 else -0.3
            notes.append("high_chromatic_dispersion")

        scores[fam] = (log_score, ",".join(notes) if notes else "spectral_oklab_fit")

    ranked_fams = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)[: max(1, top_k)]
    temp = 0.65 if has_pbr else 1.15
    logits = np.array([sc / temp for _, (sc, _) in ranked_fams], dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    # Identifiability reflects entropy gap + availability of intrinsic normal/specular channels
    full_logits = np.array([sc / temp for sc, _ in scores.values()], dtype=np.float64)
    full_logits -= full_logits.max()
    full_p = np.exp(full_logits) / np.sum(np.exp(full_logits))
    norm_entropy = -float(np.sum(full_p * np.log(np.maximum(1e-12, full_p)))) / math.log(len(MATERIAL_FAMILIES))
    base_ident = 1.0 - norm_entropy
    identifiability = float(np.clip((0.55 + 0.42 * base_ident) if has_pbr else (0.28 + 0.40 * base_ident), 0.05, 0.98))

    domain = build_surface_domain(program, None, subdivisions_per_unit=1)
    _, posed_kin = evaluate_posed_surface_kinematics(domain, program)
    stamps_tuple = tuple(extracted_stamps)

    mat_name = program.palettes[0].name
    hypotheses: list[MaterialHypothesis] = []
    for idx_h, (fam, (_, note)) in enumerate(ranked_fams):
        is_metal = fam in ("brushed_metal",)
        is_coated = fam == "painted_metal"
        cand_spec = MaterialSpec(
            name=mat_name,
            family=fam,
            substrate_lch=(0.74, 0.02, 240.0) if is_coated else med_lch,
            secondary_lch=sec_lch,
            coating_lch=dom_lch if is_coated else med_lch,
            coating_thickness=0.55 if is_coated else 0.0,
            perceptual_roughness=round(obs_rough, 3),
            metalness=1.0 if is_metal else 0.0,
            substrate_metalness=1.0 if (is_metal or is_coated) else 0.0,
            porosity=round(obs_poros, 3),
            seed=seed + idx_h * 29,
        )
        cand_mp = MaterialProgram(
            materials={pal.name: replace(cand_spec, name=pal.name) for pal in program.palettes},
            stamps=stamps_tuple,
        )
        res_stats = _evaluate_material_oklab_residual(
            program,
            scene,
            domain,
            posed_kin,
            cand_mp,
            uv_map,
            atlas_rgba,
            observed_crops,
        )
        resid_rmse = float(res_stats["rmse_delta_e"])
        hypotheses.append(
            MaterialHypothesis(
                spec=cand_spec,
                posterior_weight=float(weights[idx_h]),
                oklab_residual_rmse=resid_rmse,
                rationale=note,
                oklab_residual_stats=res_stats,
            )
        )

    out_hyps = tuple(hypotheses)
    if return_residual:
        return out_hyps, stamps_tuple, identifiability, out_hyps[0].oklab_residual_stats
    return out_hyps, stamps_tuple, identifiability


def ingest_external_asset(
    payload: dict[str, Any],
    format_kind: str,
    provenance: AssetProvenance,
    *,
    atlas_png: bytes | None = None,
    labpbr_n_png: bytes | None = None,
    labpbr_s_png: bytes | None = None,
    animation_payload: dict[str, Any] | None = None,
    license_policy: str = "clean_room_permissive",
    top_k: int = 3,
    seed: int = 404,
) -> IngestedAssetRecord:
    """Gate provenance/license/NoAI, decompile geometry (+ optional animation), and infer MaterialProgram + OKLab residual."""
    payload_meta_text = _extract_payload_metadata_text(payload)
    if animation_payload is not None and isinstance(animation_payload, dict):
        anim_meta = _extract_payload_metadata_text(animation_payload)
        if anim_meta:
            payload_meta_text = f"{payload_meta_text} {anim_meta}".strip()
    resolved_track = verify_asset_provenance(
        provenance,
        license_policy=license_policy,
        metadata_text=payload_meta_text,
    )

    embedded_rgba: np.ndarray | None = None
    raw_to_safe_bone: dict[str, str] | None = None
    if format_kind == "bbmodel":
        program, uv_map, embedded_rgba, vert_err = _decompile_bbmodel(payload)
    elif format_kind == "java_json":
        program, uv_map, vert_err = _decompile_java_json(payload)
    elif format_kind == "bedrock_geojson":
        program, uv_map, vert_err, raw_to_safe_bone = _decompile_bedrock_geojson(payload)
    else:
        raise ValueError(f"Unsupported external format_kind {format_kind!r}; expected 'bbmodel', 'java_json', or 'bedrock_geojson'")

    if animation_payload is not None:
        beh = decompile_bedrock_animation_to_behavior(
            animation_payload,
            program,
            raw_to_safe_bone=raw_to_safe_bone,
        )
        program = validate_program(replace(program, behavior=beh))

    if atlas_png is not None:
        atlas_rgba = _decode_png_rgba(atlas_png)
    elif embedded_rgba is not None:
        atlas_rgba = embedded_rgba
    else:
        atlas_rgba = np.full((16, 16, 4), (160, 160, 165, 255), dtype=np.uint8)

    n_rgba = _decode_png_rgba(labpbr_n_png) if labpbr_n_png is not None else None
    s_rgba = _decode_png_rgba(labpbr_s_png) if labpbr_s_png is not None else None

    hyps, stamps, ident_score, resid_stats = infer_material_posterior(
        program,
        uv_map,
        atlas_rgba,
        labpbr_n_rgba=n_rgba,
        labpbr_s_rgba=s_rgba,
        top_k=top_k,
        seed=seed,
        return_residual=True,
    )
    best_mp = MaterialProgram(
        materials={pal.name: replace(hyps[0].spec, name=pal.name) for pal in program.palettes},
        stamps=stamps,
    )
    return IngestedAssetRecord(
        provenance=provenance,
        format_kind=format_kind,
        program=program,
        vertex_reconstruction_error=vert_err,
        material_hypotheses=hyps,
        best_material_program=best_mp,
        extracted_stamps=stamps,
        identifiability_score=ident_score,
        has_labpbr_auxiliary_maps=(n_rgba is not None or s_rgba is not None),
        license_track=resolved_track,
        commercial_weights_eligible=(resolved_track == "clean_room_permissive"),
        oklab_residual_stats=resid_stats,
    )


def build_quarantined_research_nc_dataset(
    assets: Sequence[dict[str, Any]],
    *,
    output_dir: Path | str | None = None,
    upgrade_with_artist_loop: bool = False,
    seed: int = 404,
) -> dict[str, Any]:
    """Ingest, verify, and package external CC-BY-NC (e.g. Cobblemon) assets in a quarantined split.

    Enforces ADR 0003 & ADR 0006 non-commercial quarantine + H7 NoAI/NoDerivatives rules:
    - Accepts ``CC-BY-NC-3.0`` / ``CC-BY-NC-4.0`` / ``CC-BY-NC-SA-3.0`` / ``CC-BY-NC-SA-4.0``
      while strictly rejecting ``-ND`` (``NoDerivatives``), ``NoAI`` opt-outs, and Mojang/proprietary sources.
    - Forces all output splits under ``research_nc/`` (never ``train``/``val``/``test_unseen``).
    - Marks the shard ``commercial_weights_eligible=False`` so Apache-2.0 commercial checkpoints
      cannot accidentally consume CC-BY-NC supervision.
    """
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.parser import format_program

    if not assets:
        raise ValueError("build_quarantined_research_nc_dataset requires at least one asset specification")

    rows: list[dict[str, Any]] = []
    seen_licenses: set[str] = set()

    for idx, spec in enumerate(assets):
        prov_raw = spec.get("provenance")
        if isinstance(prov_raw, AssetProvenance):
            prov = prov_raw
        elif isinstance(prov_raw, dict):
            prov = AssetProvenance(
                asset_id=str(prov_raw.get("asset_id", "")),
                source_url=str(prov_raw.get("source_url", "")),
                author=str(prov_raw.get("author", "")),
                license_spdx=str(prov_raw.get("license_spdx", "")),
                attribution_notice=str(prov_raw.get("attribution_notice", "")),
                title=str(prov_raw.get("title", "")),
                license_url=str(prov_raw.get("license_url", "")),
                license_text_sha256=str(prov_raw.get("license_text_sha256", "")),
                retrieved_at_iso=str(prov_raw.get("retrieved_at_iso", "")),
                ai_use_marker_status=str(prov_raw.get("ai_use_marker_status", "checked_none")),
                ai_marker_evidence_sha256=str(prov_raw.get("ai_marker_evidence_sha256", "")),
                is_ai_generated=bool(prov_raw.get("is_ai_generated", False)),
                changes_made_note=str(prov_raw.get("changes_made_note", "")),
                reviewer_id=str(prov_raw.get("reviewer_id", "")),
            )
        else:
            raise LicenseGateError(f"Asset index {idx} missing valid AssetProvenance")

        rec = ingest_external_asset(
            spec["payload"],
            str(spec.get("format_kind", "bedrock_geojson")),
            prov,
            atlas_png=spec.get("atlas_png"),
            labpbr_n_png=spec.get("labpbr_n_png"),
            labpbr_s_png=spec.get("labpbr_s_png"),
            animation_payload=spec.get("animation_payload"),
            license_policy="research_nc",
            seed=seed + idx * 17,
        )
        seen_licenses.add(prov.license_spdx)

        sc = compile_geometry(rec.program)
        static_aud = audit_scene(sc, enforce_physical_contact=False)
        raw_split = str(spec.get("split", "train")).strip().strip("/")
        if raw_split.startswith("research_nc/"):
            quarantined_split = raw_split
        else:
            quarantined_split = f"research_nc/{raw_split}"

        upgraded_dsl: str | None = None
        if upgrade_with_artist_loop:
            from synccraft.world.artist_loop import run_looped_artist_enhancement

            loop_res = run_looped_artist_enhancement(rec.program, max_iterations=4)
            upgraded_dsl = str(loop_res["final_program_dsl"])

        row: dict[str, Any] = {
            "sample_id": f"research_nc_{prov.asset_id}",
            "asset_id": prov.asset_id,
            "family": str(spec.get("family", prov.asset_id)),
            "split": quarantined_split,
            "license_spdx": prov.license_spdx,
            "license_track": "research_nc",
            "commercial_weights_eligible": False,
            "source_url": prov.source_url,
            "author": prov.author,
            "attribution_notice": prov.attribution_notice,
            "provenance": prov.to_dict(),
            "format_kind": rec.format_kind,
            "vertex_reconstruction_error": round(float(rec.vertex_reconstruction_error), 8),
            "box_count": len(sc.boxes),
            "bone_count": len(sc.bones),
            "locator_count": len(sc.locators),
            "clip_count": len(rec.program.behavior.clips) if rec.program.behavior is not None else 0,
            "static_sat_collision_free": len(static_aud["collision_pairs"]) == 0,
            "identifiability_score": round(float(rec.identifiability_score), 4),
            "dominant_material_family": rec.material_hypotheses[0].spec.family,
            "oklab_residual_stats": _serialize_oklab_residual_summary(rec.oklab_residual_stats),
            "program_dsl": format_program(rec.program),
        }
        if upgraded_dsl is not None:
            row["upgraded_program_dsl"] = upgraded_dsl
        rows.append(row)

    manifest: dict[str, Any] = {
        "schema_version": "synccraft.research_nc_quarantine.v3_2",
        "license_quarantine": "research_nc",
        "commercial_weights_eligible": False,
        "total_rows": len(rows),
        "spdx_licenses": sorted(seen_licenses),
        "rows": rows,
    }

    if output_dir is not None:
        q_dir = Path(output_dir) / "research_nc"
        q_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = q_dir / "quarantined_assets_v3_1.jsonl"
        manifest_path = q_dir / "quarantine_manifest.json"
        jsonl_path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["jsonl_path"] = str(jsonl_path)
        manifest["manifest_path"] = str(manifest_path)

    return manifest
