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
import struct
from typing import Any
import zipfile

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
    MaterialDecl,
    Meta,
    Palette,
    Place,
    Program,
    StyleDecl,
    TrackDecl,
    UnsupportedFeature,
    validate_program,
)
from synccraft.kinematics.se3_chain import (
    _recover_euler_xyz_degrees,
    compile_geometry,
    euler_xyz_matrix,
)
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
    FACE_AXES_BOX,
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

APPROVED_SHAREALIKE_LICENSES: frozenset[str] = frozenset({
    "CC-BY-SA-4.0",
    "CC-BY-SA-3.0",
})

APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES: frozenset[str] = frozenset({
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-NC-3.0",
    "CC-BY-NC-SA-3.0",
})

LICENSE_POLICIES: frozenset[str] = frozenset({
    "clean_room_permissive",
    "sharealike_ccbysa",
    "research_nc",
    "ai_assisted_quarantine",
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
    gltf_doc = payload.get("gltf") if isinstance(payload.get("gltf"), dict) else payload
    if isinstance(gltf_doc, dict) and isinstance(gltf_doc.get("asset"), dict):
        asset_hdr = gltf_doc["asset"]
        for ak, av in asset_hdr.items():
            if isinstance(av, str) and ak not in ("version", "minVersion"):
                chunks.append(av)
        extras = asset_hdr.get("extras")
        if isinstance(extras, dict):
            for ek, ev in extras.items():
                if isinstance(ev, str):
                    chunks.append(ev)
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
    ai_assisted: bool = False
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
            "ai_assisted": self.ai_assisted,
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

    Returns the resolved license track: ``'clean_room_permissive'``, ``'sharealike_ccbysa'``,
    ``'research_nc'``, ``'ai_assisted_quarantine'``, or ``'local_workspace_unverified'``.
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

    if type(provenance.is_ai_generated) is not bool or type(provenance.ai_assisted) is not bool:
        raise LicenseGateError("is_ai_generated and ai_assisted must be booleans")
    if provenance.is_ai_generated and license_policy != "ai_assisted_quarantine":
        raise LicenseGateError("AI-generated assets (is_ai_generated=True) are prohibited from ingestion")
    if provenance.ai_assisted and license_policy != "ai_assisted_quarantine":
        raise LicenseGateError(
            "AI-assisted assets (ai_assisted=True) require license_policy='ai_assisted_quarantine'"
        )

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
    if license_policy == "ai_assisted_quarantine":
        noai_hits = tuple(h for h in noai_hits if h != "ai_generated")
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

    if license_policy == "ai_assisted_quarantine":
        if spdx in (APPROVED_PERMISSIVE_LICENSES | APPROVED_SHAREALIKE_LICENSES):
            if spdx in APPROVED_SHAREALIKE_LICENSES and not provenance.attribution_notice.strip():
                raise LicenseGateError(f"{spdx} requires a non-empty attribution_notice")
            return "ai_assisted_quarantine"

    if spdx in APPROVED_PERMISSIVE_LICENSES:
        return "clean_room_permissive"
    if license_policy in ("sharealike_ccbysa", "research_nc", "local_workspace_unverified") and spdx in APPROVED_SHAREALIKE_LICENSES:
        if not provenance.attribution_notice.strip():
            raise LicenseGateError(f"{spdx} requires a non-empty attribution_notice under CC-BY-SA policy")
        return "sharealike_ccbysa"
    if license_policy in ("research_nc", "local_workspace_unverified") and spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES:
        if not provenance.attribution_notice.strip():
            raise LicenseGateError(
                f"{spdx} requires a non-empty attribution_notice under CC-BY-NC quarantine"
            )
        return "research_nc"
    if license_policy == "local_workspace_unverified" and spdx == "NOASSERTION":
        return "local_workspace_unverified"

    allowed = (
        sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_SHAREALIKE_LICENSES | APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES)
        if license_policy == "research_nc"
        else (
            sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_SHAREALIKE_LICENSES)
            if license_policy in ("sharealike_ccbysa", "ai_assisted_quarantine")
            else (
                sorted(APPROVED_PERMISSIVE_LICENSES | APPROVED_SHAREALIKE_LICENSES | APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES | {"NOASSERTION"})
                if license_policy == "local_workspace_unverified"
                else sorted(APPROVED_PERMISSIVE_LICENSES)
            )
        )
    )
    hint = (
        " (pass license_policy='research_nc' for quarantined non-commercial CC-BY-NC research splits)"
        if spdx in APPROVED_NONCOMMERCIAL_RESEARCH_LICENSES
        else (
            " (pass license_policy='sharealike_ccbysa' for quarantined CC-BY-SA splits)"
            if spdx in APPROVED_SHAREALIKE_LICENSES
            else ""
        )
    )
    raise LicenseGateError(
        f"Rejected license {spdx!r} under policy {license_policy!r}; "
        f"allowed licenses: {allowed}{hint}"
    )


def _sanitize_ident(raw: str, fallback: str, used: set[str]) -> str:
    import unicodedata

    ascii_raw = unicodedata.normalize("NFKD", str(raw).strip()).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in ascii_raw)
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
    """Choose coarsest v3.2 size_step dividing all positive dimensions (dyadic first, then decimal)."""
    if not _HAS_DSL_V32 or not raw_sizes:
        return 1.0
    pos_vals: list[float] = []
    for arr in raw_sizes:
        for v in np.asarray(arr, dtype=np.float64).ravel():
            if v > 1e-9:
                pos_vals.append(float(v))
    if not pos_vals:
        return 1.0
    for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.05, 0.01, 0.001):
        ok = True
        for v in pos_vals:
            q = round(v / step)
            if q < 1 or abs(q * step - v) > 1e-4:
                ok = False
                break
        if ok:
            return step
    return 0.001


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


def _clamp_authored_inflate(
    raw_inflate: float,
    raw_span: np.ndarray | Sequence[int | float],
    _plane_normal: str | None = None,
) -> float:
    """Preserve negative and positive inflate in authored v3.2 mode while keeping non-planar extents > 0."""
    inf = float(raw_inflate or 0.0)
    if not math.isfinite(inf) or abs(inf) <= 1e-9:
        return 0.0
    if not _HAS_DSL_V32:
        return max(0.0, inf)
    if inf < 0.0:
        span_arr = np.asarray(raw_span, dtype=np.float64)
        pos_dims = [float(v) for v in span_arr.ravel() if float(v) > 1e-6]
        if not pos_dims:
            return 0.0
        min_half = 0.5 * min(pos_dims)
        inf = max(inf, -min_half + min(1e-4, 0.25 * min_half))
        if inf >= -1e-7:
            return 0.0
    return round(inf, 6)


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
        uv_f_dict: Any = ()
        if uv_faces:
            uv_f_dict = {}
            for f, r in uv_faces.items():
                rot_i = int(round(float(r[4]))) % 360 if len(r) >= 5 else 0
                if rot_i in (90, 180, 270):
                    uv_f_dict[f] = (float(r[0]), float(r[1]), float(r[2]), float(r[3]), rot_i)
                else:
                    uv_f_dict[f] = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
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
    return Box(name, ast_size, material, place=place, inflate=max(0.0, float(inflate)))  # type: ignore[arg-type]


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


def _decompile_bbmodel(
    doc: dict[str, Any],
    *,
    external_atlas_rgba: np.ndarray | None = None,
) -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Decompile Blockbench 4/5 .bbmodel JSON -> validated Program + face UVs + max 8-corner vertex error."""
    used_ids: set[str] = set()
    res = doc.get("resolution") or {"width": 16, "height": 16}
    doc_res_w = max(1, int(res.get("width", 16) or 16))
    doc_res_h = max(1, int(res.get("height", 16) or 16))
    per_tex_uv_mode = bool((doc.get("meta") or {}).get("per_texture_uv_size", False))

    raw_elements = doc.get("elements") or []
    if not raw_elements:
        raise ValueError("bbmodel contains no elements")

    has_visible_cubes = any(
        isinstance(e, dict)
        and "from" in e
        and "to" in e
        and e.get("export") is not False
        and e.get("visibility") is not False
        for e in raw_elements
    )

    cube_uuid_order: list[tuple[str, dict[str, Any]]] = []
    locator_uuid_order: list[tuple[str, dict[str, Any]]] = []
    for idx, el in enumerate(raw_elements):
        if not isinstance(el, dict):
            continue
        if el.get("export") is False:
            continue
        el_type = str(el.get("type") or "cube").strip().lower()
        if el_type == "mesh" or ("vertices" in el and "from" not in el):
            if has_visible_cubes and el.get("visibility") is False:
                continue
            raise UnsupportedFeature("bbmodel contains non-cuboid free mesh element")
        uid_e = str(el.get("uuid") or f"__el_{idx}")
        if el_type in ("locator", "null_object") or ("position" in el and "from" not in el):
            locator_uuid_order.append((uid_e, el))
            continue
        if has_visible_cubes and el.get("visibility") is False:
            continue
        if "from" not in el or "to" not in el:
            continue
        lo_raw = np.asarray(el["from"], dtype=np.float64)
        hi_raw = np.asarray(el["to"], dtype=np.float64)
        sp_chk = np.abs(hi_raw - lo_raw)
        zero_cnt = sum(1 for c in sp_chk if float(c) <= 1e-9)
        if zero_cnt >= 2:
            # Skip invisible 1D line / 0D point click artifacts in Blockbench files
            continue
        cube_uuid_order.append((uid_e, el))

    # Support top-level doc["locators"] array/dict if present in custom Blockbench plugins
    top_locators = doc.get("locators")
    if isinstance(top_locators, list):
        for l_idx, loc_el in enumerate(top_locators):
            if isinstance(loc_el, dict):
                uid_l = str(loc_el.get("uuid") or f"__toploc_{l_idx}")
                locator_uuid_order.append((uid_l, loc_el))

    if not cube_uuid_order:
        raise ValueError("bbmodel contains no cube or planar quad elements")

    elements = [el for _, el in cube_uuid_order]

    # Resolve embedded or external textures and determine per-texture UV-to-pixel transforms
    textures = doc.get("textures") or []
    tex_key_to_idx: dict[Any, int] = {}
    for t_idx, t_entry in enumerate(textures):
        if not isinstance(t_entry, dict):
            continue
        tex_key_to_idx[t_idx] = t_idx
        tex_key_to_idx[str(t_idx)] = t_idx
        tex_key_to_idx[f"#{t_idx}"] = t_idx
        if t_entry.get("uuid") is not None:
            tex_key_to_idx[str(t_entry["uuid"])] = t_idx
        if t_entry.get("id") is not None:
            tex_key_to_idx[str(t_entry["id"])] = t_idx

    referenced_tex_indices: list[int] = []
    max_uv_by_tex: dict[int, tuple[float, float]] = {}
    for el in elements:
        f_d = el.get("faces")
        if isinstance(f_d, dict) and f_d:
            for f in FACES:
                fd_f = f_d.get(f)
                if isinstance(fd_f, dict):
                    tk = fd_f.get("texture", 0 if "uv" in fd_f and "texture" not in fd_f else None)
                    if tk is None:
                        continue
                    ti = tex_key_to_idx.get(tk, tex_key_to_idx.get(str(tk), 0))
                    if ti not in referenced_tex_indices:
                        referenced_tex_indices.append(ti)
                    uv_r = fd_f.get("uv")
                    if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                        mu = max(abs(float(uv_r[0])), abs(float(uv_r[2])))
                        mv = max(abs(float(uv_r[1])), abs(float(uv_r[3])))
                        old_u, old_v = max_uv_by_tex.get(ti, (0.0, 0.0))
                        max_uv_by_tex[ti] = (max(old_u, mu), max(old_v, mv))
        elif textures:
            if 0 not in referenced_tex_indices:
                referenced_tex_indices.append(0)
    if not referenced_tex_indices and textures:
        referenced_tex_indices.append(0)
    referenced_tex_indices.sort()

    decoded_by_idx: dict[int, np.ndarray] = {}
    for ti in referenced_tex_indices:
        if 0 <= ti < len(textures) and isinstance(textures[ti], dict):
            src_i = str(textures[ti].get("source", ""))
            if src_i.startswith("data:image/"):
                b64_pos = src_i.find("base64,")
                if b64_pos >= 0:
                    try:
                        decoded_by_idx[ti] = _decode_png_rgba(src_i[b64_pos + 7 :])
                    except Exception:
                        pass

    embedded_rgba: np.ndarray | None = None
    # Map ti -> (scale_u, scale_v, offset_u, offset_v)
    tex_uv_xform: dict[int, tuple[float, float, float, float]] = {}
    raw_tex_name = "atlas_0"
    raw_tex_path = "atlas_0.png"
    if textures and isinstance(textures[0], dict):
        t0_name = str(textures[0].get("name") or "atlas_0").strip()
        if t0_name:
            raw_tex_path = t0_name if t0_name.lower().endswith(".png") else f"{t0_name}.png"
            raw_tex_name = re.sub(r"\.[A-Za-z0-9]+$", "", t0_name) or "atlas_0"

    def _infer_tex_uv_space(ti: int, px_w: int, px_h: int) -> tuple[float, float]:
        t_ent = textures[ti] if (0 <= ti < len(textures) and isinstance(textures[ti], dict)) else {}
        tw_u = float(t_ent.get("uv_width") or 0)
        th_u = float(t_ent.get("uv_height") or 0)
        obs_u, obs_v = max_uv_by_tex.get(ti, (0.0, 0.0))
        if tw_u > 0 and th_u > 0 and (per_tex_uv_mode or (obs_u <= tw_u * 1.01 and obs_v <= th_u * 1.01)):
            return tw_u, th_u
        if (obs_u > doc_res_w * 1.01 or obs_v > doc_res_h * 1.01) and max(tw_u, float(px_w)) >= obs_u * 0.99:
            return (tw_u if tw_u >= obs_u * 0.99 else float(px_w)), (th_u if th_u >= obs_v * 0.99 else float(px_h))
        return float(doc_res_w), float(doc_res_h)

    if len(decoded_by_idx) == 1:
        ti0, img0 = next(iter(decoded_by_idx.items()))
        embedded_rgba = img0
        tex_h, tex_w = img0.shape[:2]
        uvw0, uvh0 = _infer_tex_uv_space(ti0, tex_w, tex_h)
        tex_uv_xform[ti0] = (tex_w / max(1.0, uvw0), tex_h / max(1.0, uvh0), 0.0, 0.0)
    elif len(decoded_by_idx) > 1:
        ordered_tis = sorted(decoded_by_idx.keys())
        comp_w = max(int(decoded_by_idx[ti].shape[1]) for ti in ordered_tis)
        comp_h = sum(int(decoded_by_idx[ti].shape[0]) for ti in ordered_tis)
        comp_rgba = np.zeros((comp_h, comp_w, 4), dtype=np.uint8)
        y_cursor = 0
        for ti in ordered_tis:
            im_i = decoded_by_idx[ti]
            ih, iw = im_i.shape[:2]
            comp_rgba[y_cursor : y_cursor + ih, 0:iw] = im_i
            uvw_i, uvh_i = _infer_tex_uv_space(ti, iw, ih)
            tex_uv_xform[ti] = (iw / max(1.0, uvw_i), ih / max(1.0, uvh_i), 0.0, float(y_cursor))
            y_cursor += ih
        embedded_rgba = comp_rgba
        tex_h, tex_w = comp_h, comp_w
        raw_tex_name = f"{raw_tex_name}_composite"
        raw_tex_path = f"{raw_tex_name}.png"
    elif external_atlas_rgba is not None:
        tex_h, tex_w = external_atlas_rgba.shape[:2]
        uvw_e, uvh_e = _infer_tex_uv_space(0, tex_w, tex_h)
        for ti in referenced_tex_indices or [0]:
            tex_uv_xform[ti] = (tex_w / max(1.0, uvw_e), tex_h / max(1.0, uvh_e), 0.0, 0.0)
    else:
        tex_w, tex_h = doc_res_w, doc_res_h

    all_raw_spans: list[np.ndarray] = []
    uv_ratios: list[float] = []
    total_area_u = 0.0
    for el in elements:
        lo_r = np.asarray(el["from"], dtype=np.float64)
        hi_r = np.asarray(el["to"], dtype=np.float64)
        sp_e = np.abs(hi_r - lo_r)
        all_raw_spans.append(sp_e)
        total_area_u += float(2.0 * (sp_e[0] * sp_e[1] + sp_e[1] * sp_e[2] + sp_e[2] * sp_e[0]))
        f_d = el.get("faces") or {}
        if isinstance(f_d, dict):
            for f in FACES:
                if f in f_d and isinstance(f_d[f], dict) and "uv" in f_d[f]:
                    if "texture" in f_d[f] and f_d[f]["texture"] is None:
                        continue
                    uv_r = f_d[f]["uv"]
                    if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                        tk_e = f_d[f].get("texture", 0)
                        ti_e = tex_key_to_idx.get(tk_e, tex_key_to_idx.get(str(tk_e), 0))
                        su_e, sv_e, _, _ = tex_uv_xform.get(ti_e, (1.0, 1.0, 0.0, 0.0))
                        wu_e, hv_e = face_uv_extents_units((float(sp_e[0]), float(sp_e[1]), float(sp_e[2])), f)
                        du = abs(float(uv_r[2]) - float(uv_r[0])) * su_e
                        dv = abs(float(uv_r[3]) - float(uv_r[1])) * sv_e
                        if wu_e > 0.25 and du > 0.25:
                            uv_ratios.append(16.0 * du / wu_e)
                        if hv_e > 0.25 and dv > 0.25:
                            uv_ratios.append(16.0 * dv / hv_e)

    size_step = _infer_dyadic_size_step(all_raw_spans)
    density = _infer_texel_density(tex_w, tex_h, uv_ratios, total_area_u, has_nonzero_box_uv=True)

    tex_decl_id: str | None = None
    tex_ref_map: dict[Any, str] = {}
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
        if len(decoded_by_idx) > 1:
            tid_comp = _sanitize_ident(raw_tex_name, "tex", used_ids)
            t_decls = [_TextureDecl(tid_comp, raw_tex_path, tex_w, tex_h)]
            for t_idx, t_entry in enumerate(textures):
                tex_ref_map[t_idx] = tid_comp
                tex_ref_map[str(t_idx)] = tid_comp
                tex_ref_map[f"#{t_idx}"] = tid_comp
                if isinstance(t_entry, dict):
                    if t_entry.get("uuid") is not None:
                        tex_ref_map[str(t_entry["uuid"])] = tid_comp
                    if t_entry.get("id") is not None:
                        tex_ref_map[str(t_entry["id"])] = tid_comp
        else:
            t_decls = []
            if textures and isinstance(textures, list):
                for t_idx, t_entry in enumerate(textures):
                    if not isinstance(t_entry, dict):
                        continue
                    tn = str(t_entry.get("name") or f"atlas_{t_idx}").strip() or f"atlas_{t_idx}"
                    tp = tn if tn.lower().endswith(".png") else f"{tn}.png"
                    t_stem = re.sub(r"\.[A-Za-z0-9]+$", "", tn) or f"atlas_{t_idx}"
                    tid = _sanitize_ident(t_stem, "tex", used_ids)
                    tw_i = max(1, int(t_entry.get("width") or t_entry.get("uv_width") or tex_w))
                    th_i = max(1, int(t_entry.get("height") or t_entry.get("uv_height") or tex_h))
                    if (t_idx in decoded_by_idx) or (t_idx == 0 and external_atlas_rgba is not None):
                        tw_i, th_i = tex_w, tex_h
                    t_decls.append(_TextureDecl(tid, tp, tw_i, th_i))
                    tex_ref_map[t_idx] = tid
                    tex_ref_map[str(t_idx)] = tid
                    tex_ref_map[f"#{t_idx}"] = tid
                    if t_entry.get("uuid") is not None:
                        tex_ref_map[str(t_entry["uuid"])] = tid
                    if t_entry.get("id") is not None:
                        tex_ref_map[str(t_entry["id"])] = tid
            if not t_decls:
                tid0 = _sanitize_ident(raw_tex_name, "tex", used_ids)
                t_decls.append(_TextureDecl(tid0, raw_tex_path, tex_w, tex_h))
        tex_decl_id = t_decls[0].name
        textures_tuple = tuple(t_decls)

    def _resolve_elem_tex_id(elem: dict[str, Any]) -> str | None:
        if tex_decl_id is None:
            return None
        f_d = elem.get("faces")
        if isinstance(f_d, dict):
            for f in FACES:
                fd_f = f_d.get(f)
                if isinstance(fd_f, dict) and fd_f.get("texture") is not None:
                    tk = fd_f["texture"]
                    if tk in tex_ref_map:
                        return tex_ref_map[tk]
                    if str(tk) in tex_ref_map:
                        return tex_ref_map[str(tk)]
        return tex_decl_id

    elem_uuid_order = cube_uuid_order
    elem_by_uuid = dict(cube_uuid_order)
    loc_by_uuid = dict(locator_uuid_order)
    groups = doc.get("groups") or []
    group_by_uuid: dict[str, dict[str, Any]] = {
        str(g.get("uuid", "")): dict(g)
        for g in groups
        if isinstance(g, dict) and str(g.get("uuid", "")).strip()
    }
    outliner = doc.get("outliner") or []

    # Support Blockbench 4 inline outliner group dicts and disambiguate copy-pasted duplicate group UUIDs
    inline_grp_seq = 0
    seen_outliner_grp_uuids: set[str] = set()

    def _register_inline_outliner_groups(items_list: list[Any]) -> None:
        nonlocal inline_grp_seq
        for entry in items_list:
            if isinstance(entry, dict):
                uid = str(entry.get("uuid") or "").strip()
                if not uid or uid in seen_outliner_grp_uuids:
                    inline_grp_seq += 1
                    orig_meta = group_by_uuid.get(uid, {})
                    uid = f"{uid}__dup_{inline_grp_seq}" if uid else f"__inline_grp_{inline_grp_seq}"
                    entry["uuid"] = uid
                    merged_dup = dict(orig_meta)
                    merged_dup.update(entry)
                    group_by_uuid[uid] = merged_dup
                else:
                    if uid not in group_by_uuid:
                        group_by_uuid[uid] = dict(entry)
                    else:
                        merged_g = dict(entry)
                        merged_g.update(group_by_uuid[uid])
                        group_by_uuid[uid] = merged_g
                seen_outliner_grp_uuids.add(uid)
                _register_inline_outliner_groups(entry.get("children") or [])

    _register_inline_outliner_groups(outliner)

    referenced_elem_uuids: set[str] = set()
    referenced_loc_uuids: set[str] = set()

    def _collect_outliner_refs(items_list: list[Any]) -> None:
        for entry in items_list:
            if isinstance(entry, str):
                if entry in elem_by_uuid:
                    referenced_elem_uuids.add(entry)
                elif entry in loc_by_uuid:
                    referenced_loc_uuids.add(entry)
            elif isinstance(entry, dict):
                _collect_outliner_refs(entry.get("children") or [])

    _collect_outliner_refs(outliner)

    uv_map: dict[str, dict[str, list[float]]] = {}
    palettes = [Palette("mat_0", (0.62, 0.06, 220.0), 6)]
    expected_corners: dict[str, np.ndarray] = {}
    group_uuid_to_safe_bone: dict[str, str] = {}

    def _compute_elem_uv_faces(
        elem: dict[str, Any],
        uv_dims: tuple[float, float, float],
    ) -> tuple[dict[str, list[float]], bool]:
        """Return (face_uv_dict_in_atlas_px, can_emit_compact_box_uv)."""
        f_dict = elem.get("faces") or {}
        uv_box_raw = elem.get("uv_offset")
        su0, sv0, ou0, ov0 = tex_uv_xform.get(0, (1.0, 1.0, 0.0, 0.0))
        if not f_dict and isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) >= 2:
            ub, vb = float(uv_box_raw[0]), float(uv_box_raw[1])
            sx = max(0, int(math.floor(uv_dims[0] + 1e-9)))
            sy = max(0, int(math.floor(uv_dims[1] + 1e-9)))
            sz = max(0, int(math.floor(uv_dims[2] + 1e-9)))
            base_net: dict[str, list[float]] = {
                "up": [ub + sz, vb, ub + sz + sx, vb + sz],
                "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
                "east": [ub, vb + sz, ub + sz, vb + sz + sy],
                "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
                "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
                "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
            }
            is_mir = bool(elem.get("mirror_uv"))
            if is_mir:
                swap_ew = {"east": "west", "west": "east"}
                base_net = {
                    swap_ew.get(fn, fn): [r[2], r[1], r[0], r[3]]
                    for fn, r in base_net.items()
                }
            scaled_net = {
                fn: [
                    round(ou0 + r[0] * su0, 5),
                    round(ov0 + r[1] * sv0, 5),
                    round(ou0 + r[2] * su0, 5),
                    round(ov0 + r[3] * sv0, 5),
                ]
                for fn, r in base_net.items()
            }
            unit_scale_identity = abs(su0 - 1.0) <= 1e-6 and abs(sv0 - 1.0) <= 1e-6 and abs(ou0) <= 1e-6 and abs(ov0) <= 1e-6
            return scaled_net, (not is_mir and unit_scale_identity)

        out_f: dict[str, list[float]] = {}
        if isinstance(f_dict, dict) and f_dict:
            has_explicit_face_tex = any(
                isinstance(v_fd, dict) and "texture" in v_fd and v_fd["texture"] is not None
                for v_fd in f_dict.values()
            )
            for f in FACES:
                fd_f = f_dict.get(f)
                if fd_f is None or not isinstance(fd_f, dict):
                    continue
                if fd_f.get("enabled") is False:
                    continue
                if "texture" in fd_f and fd_f["texture"] is None:
                    # Explicitly disabled/deleted face in Blockbench
                    continue
                if has_explicit_face_tex and "texture" not in fd_f:
                    # Sibling faces explicitly assign textures while this face was left unassigned
                    continue
                tk = fd_f.get("texture", 0)
                ti = tex_key_to_idx.get(tk, tex_key_to_idx.get(str(tk), 0))
                su, sv, ou, ov = tex_uv_xform.get(ti, (1.0, 1.0, 0.0, 0.0))
                uv_r = fd_f.get("uv")
                if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                    rect_px = [
                        round(ou + float(uv_r[0]) * su, 5),
                        round(ov + float(uv_r[1]) * sv, 5),
                        round(ou + float(uv_r[2]) * su, 5),
                        round(ov + float(uv_r[3]) * sv, 5),
                    ]
                else:
                    wu, hv = face_uv_extents_units(uv_dims, f)
                    rect_px = [round(ou, 5), round(ov, 5), round(ou + min(tex_w, wu), 5), round(ov + min(tex_h, hv), 5)]
                rot_uv = int(fd_f.get("rotation", 0) or 0) % 360
                if rot_uv in (90, 180, 270):
                    rect_px.append(rot_uv)
                out_f[f] = rect_px

        if not out_f:
            for f in FACES:
                wu, hv = face_uv_extents_units(uv_dims, f)
                out_f[f] = [0.0, 0.0, float(min(tex_w, wu)), float(min(tex_h, hv))]
        return out_f, False

    def _register_elem_uv(
        bname: str,
        elem: dict[str, Any],
        uv_dims: tuple[float, float, float],
    ) -> bool:
        f_uvs, compact_ok = _compute_elem_uv_faces(elem, uv_dims)
        uv_map[bname] = f_uvs
        return compact_ok

    def _emit_elems_for_bone(
        bone_label: str,
        elem_list: list[dict[str, Any]],
        loc_list: list[dict[str, Any]],
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
            lo_raw = np.asarray(el["from"], dtype=np.float64)
            hi_raw = np.asarray(el["to"], dtype=np.float64)
            lo = np.minimum(lo_raw, hi_raw)
            hi = np.maximum(lo_raw, hi_raw)
            raw_span = hi - lo
            ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
            ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
            ctr_unrot = 0.5 * (lo + hi)
            loc_ctr = ctr_unrot - bone_origin
            inflate = _clamp_authored_inflate(float(el.get("inflate", 0.0) or 0.0), raw_span)
            compact_box_uv_ok = _register_elem_uv(bname, el, (float(raw_span[0]), float(raw_span[1]), float(raw_span[2])))
            uv_box_raw = el.get("uv_offset")
            uv_pair = (
                (float(uv_box_raw[0]), float(uv_box_raw[1]))
                if compact_box_uv_ok and isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) == 2
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
            direct_items.append(
                _make_v32_box(
                    bname,
                    ast_size,
                    "mat_0",
                    place=pl,
                    inflate=inflate,
                    plane_normal=plane_normal,
                    uv=uv_pair,
                    uv_faces=None if uv_pair is not None else uv_map[bname],
                    texture=_resolve_elem_tex_id(el),
                )
            )
            anchor_candidates.append((bname, loc_ctr, ast_sz_arr, True))

        if _HAS_DSL_V32:
            # Emit per-cube rotations directly on Box without synthesizing extra rot_* dummy bones
            for el, el_rot in rot_elems:
                bname = _sanitize_ident(el.get("name", "cube"), "box", used_ids)
                lo_raw = np.asarray(el["from"], dtype=np.float64)
                hi_raw = np.asarray(el["to"], dtype=np.float64)
                lo = np.minimum(lo_raw, hi_raw)
                hi = np.maximum(lo_raw, hi_raw)
                raw_span = hi - lo
                ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
                ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
                ctr_unrot = 0.5 * (lo + hi)
                el_origin = np.asarray(el.get("origin", ctr_unrot.tolist()), dtype=np.float64)
                inflate = _clamp_authored_inflate(float(el.get("inflate", 0.0) or 0.0), raw_span)
                compact_box_uv_ok = _register_elem_uv(bname, el, (float(raw_span[0]), float(raw_span[1]), float(raw_span[2])))
                uv_box_raw = el.get("uv_offset")
                uv_pair = (
                    (float(uv_box_raw[0]), float(uv_box_raw[1]))
                    if compact_box_uv_ok and isinstance(uv_box_raw, (list, tuple)) and len(uv_box_raw) == 2
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
                        uv_faces=None if uv_pair is not None else uv_map[bname],
                        texture=_resolve_elem_tex_id(el),
                    )
                )
                anchor_candidates.append((bname, actual_loc_ctr, ast_sz_arr, False))

        # Emit explicit Blockbench locators attached to this bone
        for loc_el in loc_list:
            raw_lname = str(loc_el.get("name") or "locator").strip() or "locator"
            base_lname = raw_lname if raw_lname.lower().startswith("loc") else f"loc_{raw_lname}"
            safe_lname = _sanitize_ident(base_lname, "locator", used_ids)
            pos_raw = loc_el.get("position", loc_el.get("from", bone_origin.tolist()))
            pos_arr = np.asarray(pos_raw, dtype=np.float64)
            loc_off = pos_arr - bone_origin
            off_tuple = (
                0.0 if abs(float(loc_off[0])) < 1e-9 else round(float(loc_off[0]), 8),
                0.0 if abs(float(loc_off[1])) < 1e-9 else round(float(loc_off[1]), 8),
                0.0 if abs(float(loc_off[2])) < 1e-9 else round(float(loc_off[2]), 8),
            )
            role = _infer_locator_role(raw_lname)
            direct_items.append(Locator(name=safe_lname, offset=off_tuple, role=role))
            loc_vec = np.asarray(off_tuple, dtype=np.float64)
            if anchor_name is None:
                anchor_name = safe_lname
                anchor_loc = loc_vec
            anchor_candidates.append((safe_lname, loc_vec, None, False))

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
                lo_raw = np.asarray(el["from"], dtype=np.float64)
                hi_raw = np.asarray(el["to"], dtype=np.float64)
                lo = np.minimum(lo_raw, hi_raw)
                hi = np.maximum(lo_raw, hi_raw)
                raw_span = np.maximum(0.0, hi - lo)
                ast_size, _, eff_span = _quantize_box_dimensions(raw_span, size_step)
                ctr_unrot = 0.5 * (lo + hi)
                el_origin = np.asarray(el.get("origin", ctr_unrot.tolist()), dtype=np.float64)
                inflate = max(0.0, float(el.get("inflate", 0.0) or 0.0))
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
        ch_l = node_l.get("children") or []
        ch_r = node_r.get("children") or []
        if any(isinstance(c, dict) for c in ch_l) or any(isinstance(c, dict) for c in ch_r):
            return False
        if any(isinstance(c, str) and c in loc_by_uuid for c in ch_l) or any(
            isinstance(c, str) and c in loc_by_uuid for c in ch_r
        ):
            return False
        elems_l = [elem_by_uuid[c] for c in ch_l if isinstance(c, str) and c in elem_by_uuid]
        elems_r = [elem_by_uuid[c] for c in ch_r if isinstance(c, str) and c in elem_by_uuid]
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
        swap_ew = {"east": "west", "west": "east"}
        for el_l, el_r in zip(elems_l, elems_r):
            if _resolve_elem_tex_id(el_l) != _resolve_elem_tex_id(el_r):
                return False
            lo_l_raw, hi_l_raw = np.asarray(el_l["from"], dtype=np.float64), np.asarray(el_l["to"], dtype=np.float64)
            lo_r_raw, hi_r_raw = np.asarray(el_r["from"], dtype=np.float64), np.asarray(el_r["to"], dtype=np.float64)
            lo_l, hi_l = np.minimum(lo_l_raw, hi_l_raw), np.maximum(lo_l_raw, hi_l_raw)
            lo_r, hi_r = np.minimum(lo_r_raw, hi_r_raw), np.maximum(lo_r_raw, hi_r_raw)
            sz_l, sz_r = np.maximum(0.0, hi_l - lo_l), np.maximum(0.0, hi_r - lo_r)
            if not np.allclose(sz_l, sz_r, atol=1e-4):
                return False
            if abs(float(el_l.get("inflate", 0.0) or 0.0) - float(el_r.get("inflate", 0.0) or 0.0)) > 1e-5:
                return False
            rl_raw = el_l.get("rotation") or [0.0, 0.0, 0.0]
            rr_raw = el_r.get("rotation") or [0.0, 0.0, 0.0]
            rl = (float(rl_raw[0]), float(rl_raw[1]), float(rl_raw[2]))
            rr = (float(rr_raw[0]), float(rr_raw[1]), float(rr_raw[2]))
            if not _HAS_DSL_V32 and (any(abs(a) > 1e-8 for a in rl) or any(abs(a) > 1e-8 for a in rr)):
                return False
            if not np.allclose(euler_xyz_matrix(rr), euler_xyz_matrix((rl[0], -rl[1], -rl_raw[2] if isinstance(rl_raw[2], (int, float)) else -rl[2])), atol=1e-4):
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
            uv_l, _ = _compute_elem_uv_faces(el_l, (float(sz_l[0]), float(sz_l[1]), float(sz_l[2])))
            uv_r, _ = _compute_elem_uv_faces(el_r, (float(sz_r[0]), float(sz_r[1]), float(sz_r[2])))
            if {swap_ew.get(fn, fn) for fn in uv_l} != set(uv_r):
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
            lo_r_raw = np.asarray(el_r["from"], dtype=np.float64)
            hi_r_raw = np.asarray(el_r["to"], dtype=np.float64)
            lo_r = np.minimum(lo_r_raw, hi_r_raw)
            hi_r = np.maximum(lo_r_raw, hi_r_raw)
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
        if node.get("export") is False or grp.get("export") is False:
            return None
        if has_visible_outliner and (node.get("visibility") is False or grp.get("visibility") is False):
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

        raw_children = node.get("children") or []
        child_elems = [elem_by_uuid[c] for c in raw_children if isinstance(c, str) and c in elem_by_uuid]
        child_locs = [loc_by_uuid[c] for c in raw_children if isinstance(c, str) and c in loc_by_uuid]
        child_nodes = [
            c
            for c in raw_children
            if isinstance(c, dict)
            and c.get("export") is not False
            and group_by_uuid.get(str(c.get("uuid", "")), {}).get("export") is not False
            and not (
                has_visible_outliner
                and (
                    c.get("visibility") is False
                    or group_by_uuid.get(str(c.get("uuid", "")), {}).get("visibility") is False
                )
            )
        ]

        items, own_candidates = _emit_elems_for_bone(
            bone_name, child_elems, child_locs, g_origin, world_piv, world_rot
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

    raw_dict_roots = [
        n
        for n in outliner
        if isinstance(n, dict)
        and str(n.get("uuid", "")) in group_by_uuid
        and n.get("export") is not False
        and group_by_uuid[str(n.get("uuid", ""))].get("export") is not False
    ]
    loose_elems = [elem_by_uuid[u] for u in outliner if isinstance(u, str) and u in elem_by_uuid]
    loose_locs = [loc_by_uuid[u] for u in outliner if isinstance(u, str) and u in loc_by_uuid]
    for u_str, el_obj in elem_uuid_order:
        if u_str not in referenced_elem_uuids:
            loose_elems.append(el_obj)
    for u_str, loc_obj in locator_uuid_order:
        if u_str not in referenced_loc_uuids:
            loose_locs.append(loc_obj)

    has_visible_outliner = bool(loose_elems) or any(
        n.get("visibility") is not False
        and group_by_uuid[str(n.get("uuid", ""))].get("visibility") is not False
        for n in raw_dict_roots
    )
    dict_roots = [
        n
        for n in raw_dict_roots
        if not (
            has_visible_outliner
            and (
                n.get("visibility") is False
                or group_by_uuid[str(n.get("uuid", ""))].get("visibility") is False
            )
        )
    ]

    if len(dict_roots) == 1 and not loose_elems and not loose_locs:
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
                wrap_name, [], [], np.zeros(3), np.zeros(3), np.eye(3)
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
            wrap_name, loose_elems, loose_locs, np.zeros(3), np.zeros(3), np.eye(3)
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
            root_name, loose_elems, loose_locs, np.zeros(3), np.zeros(3), np.eye(3)
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


def _mojang_auto_uv_16(lo: np.ndarray, hi: np.ndarray, face: str) -> tuple[float, float, float, float]:
    """Compute Minecraft Java Edition BlockElementFace default UV in [0, 16] from element bounds."""
    x1, y1, z1 = float(lo[0]), float(lo[1]), float(lo[2])
    x2, y2, z2 = float(hi[0]), float(hi[1]), float(hi[2])
    if face == "north":
        raw = (16.0 - x2, 16.0 - y2, 16.0 - x1, 16.0 - y1)
    elif face == "south":
        raw = (x1, 16.0 - y2, x2, 16.0 - y1)
    elif face == "east":
        raw = (16.0 - z2, 16.0 - y2, 16.0 - z1, 16.0 - y1)
    elif face == "west":
        raw = (z1, 16.0 - y2, z2, 16.0 - y1)
    elif face == "up":
        raw = (x1, z1, x2, z2)
    else:  # "down"
        raw = (x1, 16.0 - z2, x2, 16.0 - z1)
    return (
        float(np.clip(raw[0], 0.0, 16.0)),
        float(np.clip(raw[1], 0.0, 16.0)),
        float(np.clip(raw[2], 0.0, 16.0)),
        float(np.clip(raw[3], 0.0, 16.0)),
    )


def _java_element_geometry(el: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray, bool]:
    """Return (lo_uv, hi_uv, raw_span_uv, raw_span_geom, ctr_geom, rot_euler, rot_origin, has_rot) honoring Mojang rescale."""
    lo_raw = np.asarray(el["from"], dtype=np.float64)
    hi_raw = np.asarray(el["to"], dtype=np.float64)
    lo = np.minimum(lo_raw, hi_raw)
    hi = np.maximum(lo_raw, hi_raw)
    raw_span_uv = np.maximum(0.0, hi - lo)
    ctr_unscaled = 0.5 * (lo + hi)

    rot_data = el.get("rotation")
    has_rot = False
    rot_euler = (0.0, 0.0, 0.0)
    rot_origin = ctr_unscaled.copy()
    raw_span_geom = raw_span_uv.copy()
    ctr_geom = ctr_unscaled.copy()

    if isinstance(rot_data, dict):
        rot_origin = np.asarray(rot_data.get("origin", ctr_unscaled.tolist()), dtype=np.float64)
        if "axis" in rot_data:
            ax = str(rot_data["axis"])
            ang = float(rot_data.get("angle", 0.0))
            rot_euler = (ang if ax == "x" else 0.0, ang if ax == "y" else 0.0, ang if ax == "z" else 0.0)
            if bool(rot_data.get("rescale", False)) and 1e-6 < abs(ang) < 89.0:
                cos_a = math.cos(math.radians(abs(ang)))
                if cos_a > 1e-6:
                    sf = 1.0 / cos_a
                    s_vec = np.ones(3, dtype=np.float64)
                    if ax == "x":
                        s_vec[1] = sf
                        s_vec[2] = sf
                    elif ax == "y":
                        s_vec[0] = sf
                        s_vec[2] = sf
                    elif ax == "z":
                        s_vec[0] = sf
                        s_vec[1] = sf
                    raw_span_geom = raw_span_uv * s_vec
                    ctr_geom = rot_origin + s_vec * (ctr_unscaled - rot_origin)
        else:
            rot_euler = (float(rot_data.get("x", 0.0)), float(rot_data.get("y", 0.0)), float(rot_data.get("z", 0.0)))
        has_rot = any(abs(a) > 1e-8 for a in rot_euler)

    return lo, hi, raw_span_uv, raw_span_geom, ctr_geom, rot_euler, rot_origin, has_rot


def _decompile_java_json(
    doc: dict[str, Any],
    *,
    external_atlas_rgba: np.ndarray | None = None,
    java_textures_rgba: dict[str, np.ndarray] | None = None,
    return_rgba: bool = False,
) -> (
    tuple[Program, dict[str, dict[str, list[float]]], float]
    | tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]
):
    """Decompile Minecraft Java block/item .json -> validated Program + face UVs (+ optional composite RGBA) + max 8-corner error."""
    used_ids: set[str] = set()
    elements = doc.get("elements") or []
    if not elements:
        raise ValueError("Java model JSON contains no elements")

    tex_dict = doc.get("textures") if isinstance(doc.get("textures"), dict) else {}

    def _resolve_java_tex_key(raw_ref: Any) -> str:
        cur = str(raw_ref or "").strip()
        if cur.startswith("#"):
            cur = cur[1:]
        for _ in range(6):
            nxt = tex_dict.get(cur)
            if isinstance(nxt, str) and nxt.startswith("#"):
                cur = nxt[1:]
            else:
                break
        return cur

    embedded_rgba: np.ndarray | None = None
    key_uv_xform: dict[str, tuple[float, float, float, float]] = {}

    if java_textures_rgba:
        # Collect active texture keys referenced by element faces in stable order
        active_keys: list[str] = []
        for el in elements:
            fd = el.get("faces")
            if isinstance(fd, dict):
                for f in FACES:
                    fdf = fd.get(f)
                    if isinstance(fdf, dict) and fdf.get("texture") is not None:
                        rk = _resolve_java_tex_key(fdf["texture"])
                        if rk in java_textures_rgba and rk not in active_keys:
                            active_keys.append(rk)
        if not active_keys:
            active_keys = sorted(java_textures_rgba.keys())

        if len(active_keys) == 1:
            only_k = active_keys[0]
            embedded_rgba = java_textures_rgba[only_k]
            tex_h, tex_w = embedded_rgba.shape[:2]
            default_xform = (tex_w / 16.0, tex_h / 16.0, 0.0, 0.0)
            for k in java_textures_rgba:
                key_uv_xform[k] = default_xform
        else:
            comp_w = max(int(java_textures_rgba[k].shape[1]) for k in active_keys)
            comp_h = sum(int(java_textures_rgba[k].shape[0]) for k in active_keys)
            comp = np.zeros((comp_h, comp_w, 4), dtype=np.uint8)
            y_cur = 0
            for k in active_keys:
                arr_k = java_textures_rgba[k]
                hk, wk = arr_k.shape[:2]
                comp[y_cur : y_cur + hk, :wk] = arr_k
                key_uv_xform[k] = (wk / 16.0, hk / 16.0, 0.0, float(y_cur))
                y_cur += hk
            embedded_rgba = comp
            tex_w, tex_h = comp_w, comp_h
            default_xform = key_uv_xform[active_keys[0]]
    elif external_atlas_rgba is not None:
        embedded_rgba = external_atlas_rgba
        tex_h, tex_w = external_atlas_rgba.shape[:2]
        default_xform = (tex_w / 16.0, tex_h / 16.0, 0.0, 0.0)
    else:
        tex_size = doc.get("texture_size") or [16, 16]
        tex_w, tex_h = max(1, int(tex_size[0])), max(1, int(tex_size[1]))
        default_xform = (tex_w / 16.0, tex_h / 16.0, 0.0, 0.0)

    parsed_geom = [_java_element_geometry(el) for el in elements]

    all_raw_spans: list[np.ndarray] = []
    uv_ratios: list[float] = []
    total_area_u = 0.0
    for el, (lo_e, hi_e, sp_uv, sp_geom, _, _, _, _) in zip(elements, parsed_geom):
        all_raw_spans.append(sp_geom)
        total_area_u += float(2.0 * (sp_geom[0] * sp_geom[1] + sp_geom[1] * sp_geom[2] + sp_geom[2] * sp_geom[0]))
        f_d = el.get("faces") or {}
        if isinstance(f_d, dict):
            for f in FACES:
                if f in f_d and isinstance(f_d[f], dict):
                    rk = _resolve_java_tex_key(f_d[f].get("texture", ""))
                    su_e, sv_e, _, _ = key_uv_xform.get(rk, default_xform)
                    if "uv" in f_d[f] and isinstance(f_d[f]["uv"], (list, tuple)) and len(f_d[f]["uv"]) == 4:
                        uv_r = f_d[f]["uv"]
                    else:
                        uv_r = _mojang_auto_uv_16(lo_e, hi_e, f)
                    wu_e, hv_e = face_uv_extents_units((float(sp_uv[0]), float(sp_uv[1]), float(sp_uv[2])), f)
                    du = abs(float(uv_r[2]) - float(uv_r[0])) * su_e
                    dv = abs(float(uv_r[3]) - float(uv_r[1])) * sv_e
                    if wu_e > 0.25 and du > 0.25:
                        uv_ratios.append(16.0 * du / wu_e)
                    if hv_e > 0.25 and dv > 0.25:
                        uv_ratios.append(16.0 * dv / hv_e)

    size_step = _infer_dyadic_size_step(all_raw_spans)
    density = _infer_texel_density(tex_w, tex_h, uv_ratios, total_area_u, has_nonzero_box_uv=True)

    tex_decl_id: str | None = None
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
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

    for idx, (el, (lo, hi, raw_span_uv, raw_span, ctr, rot_euler, rot_origin, has_rot)) in enumerate(zip(elements, parsed_geom)):
        ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
        ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
        bname = _sanitize_ident(el.get("name", f"elem_{idx}"), "box", used_ids)

        f_dict = el.get("faces") or {}
        uv_map[bname] = {}
        if isinstance(f_dict, dict) and f_dict:
            for f in FACES:
                fd_f = f_dict.get(f)
                if fd_f is None or not isinstance(fd_f, dict):
                    continue
                rk = _resolve_java_tex_key(fd_f.get("texture", ""))
                su, sv, ou, ov = key_uv_xform.get(rk, default_xform)
                if "uv" in fd_f and isinstance(fd_f["uv"], (list, tuple)) and len(fd_f["uv"]) == 4:
                    u0, v0, u1, v1 = (float(x) for x in fd_f["uv"])
                else:
                    u0, v0, u1, v1 = _mojang_auto_uv_16(lo, hi, f)
                rect_px = [
                    round(ou + u0 * su, 5),
                    round(ov + v0 * sv, 5),
                    round(ou + u1 * su, 5),
                    round(ov + v1 * sv, 5),
                ]
                rot_uv = int(fd_f.get("rotation", 0) or 0) % 360
                if rot_uv in (90, 180, 270):
                    rect_px.append(rot_uv)
                uv_map[bname][f] = rect_px
        else:
            for f in FACES:
                wu, hv = face_uv_extents_units((float(raw_span_uv[0]), float(raw_span_uv[1]), float(raw_span_uv[2])), f)
                uv_map[bname][f] = [0.0, 0.0, float(wu), float(hv)]

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
    if return_rgba:
        return prog, uv_map, embedded_rgba, max_err
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

    def _compute_bedrock_cube_uv(
        cube: dict[str, Any],
        raw_sz: np.ndarray,
        bone_mirror: bool = False,
    ) -> tuple[dict[str, list[float]], tuple[float, float] | None]:
        uv_raw = cube.get("uv", [0, 0])
        uv_dims_f = (float(raw_sz[0]), float(raw_sz[1]), float(raw_sz[2]))
        if isinstance(uv_raw, dict):
            out_f: dict[str, list[float]] = {}
            for f in FACES:
                fd_f = uv_raw.get(f)
                if fd_f is None or not isinstance(fd_f, dict):
                    continue
                if "uv" in fd_f:
                    u0, v0 = (float(x) for x in fd_f["uv"])
                    wu_def, hv_def = face_uv_extents_units(uv_dims_f, f)
                    su, sv = (float(x) for x in fd_f.get("uv_size", [wu_def, hv_def]))
                    rect_f = [u0, v0, u0 + su, v0 + sv]
                else:
                    wu, hv = face_uv_extents_units(uv_dims_f, f)
                    rect_f = [0.0, 0.0, float(wu), float(hv)]
                rot_uv = int(fd_f.get("uv_rotation", fd_f.get("rotation", 0)) or 0) % 360
                if rot_uv in (90, 180, 270):
                    rect_f.append(rot_uv)
                out_f[f] = rect_f
            if not out_f:
                for f in FACES:
                    wu, hv = face_uv_extents_units(uv_dims_f, f)
                    out_f[f] = [0.0, 0.0, float(wu), float(hv)]
            return out_f, None
        ub, vb = float(uv_raw[0]), float(uv_raw[1])
        # Bedrock 1.12.0 Box-UV layout uses floor(size) UV strip offsets (0 for planar normal axes)
        sx = max(0, int(math.floor(float(raw_sz[0]) + 1e-9)))
        sy = max(0, int(math.floor(float(raw_sz[1]) + 1e-9)))
        sz = max(0, int(math.floor(float(raw_sz[2]) + 1e-9)))
        base_net: dict[str, list[float]] = {
            "up": [ub + sz, vb, ub + sz + sx, vb + sz],
            "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
            "east": [ub, vb + sz, ub + sz, vb + sz + sy],
            "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
            "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
            "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
        }
        is_mir = bool(cube.get("mirror", bone_mirror))
        if is_mir:
            swap_ew = {"east": "west", "west": "east"}
            base_net = {
                swap_ew.get(fn, fn): [r[2], r[1], r[0], r[3]]
                for fn, r in base_net.items()
            }
            return base_net, None
        return base_net, (ub, vb)

    def _register_bedrock_cube_uv(
        cname: str,
        cube: dict[str, Any],
        raw_sz: np.ndarray,
        bone_mirror: bool = False,
    ) -> tuple[float, float] | None:
        f_uvs, uv_pair = _compute_bedrock_cube_uv(cube, raw_sz, bone_mirror=bone_mirror)
        uv_map[cname] = f_uvs
        return uv_pair

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
        swap_ew = {"east": "west", "west": "east"}
        bmir_l = bool(rec_l.get("mirror", False))
        bmir_r = bool(rec_r.get("mirror", False))
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
            uv_l, _ = _compute_bedrock_cube_uv(cb_l, sz_l, bone_mirror=bmir_l)
            uv_r, _ = _compute_bedrock_cube_uv(cb_r, sz_r, bone_mirror=bmir_r)
            if {swap_ew.get(fn, fn) for fn in uv_l} != set(uv_r):
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
        bmir_r = bool(rec_r.get("mirror", False))
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
            _register_bedrock_cube_uv(mir_cname, cb_r, raw_sz_r, bone_mirror=bmir_r)
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
        bone_mir = bool(rec.get("mirror", False))
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
                inflate = _clamp_authored_inflate(float(cube.get("inflate", 0.0) or 0.0), ast_size, plane_normal)
                cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                uv_pair = _register_bedrock_cube_uv(cname, cube, raw_sz, bone_mirror=bone_mir)

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
                    inflate = _clamp_authored_inflate(float(cube.get("inflate", 0.0) or 0.0), ast_size, plane_normal)
                    cname = _sanitize_ident(f"{safe_bname}_cube_{c_idx}", "box", used_ids)
                    uv_pair = _register_bedrock_cube_uv(cname, cube, raw_sz, bone_mirror=bone_mir)

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
    has_source_texture: bool = True
    resolved_atlas_rgba: np.ndarray | None = field(default=None, repr=False)
    resolved_uv_map: dict[str, dict[str, list[float]]] = field(default_factory=dict, repr=False)

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
            "has_source_texture": self.has_source_texture,
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
    if max(h_img, w_img) > 512:
        scale_f = 512.0 / float(max(h_img, w_img))
        new_w = max(1, int(round(w_img * scale_f)))
        new_h = max(1, int(round(h_img * scale_f)))
        sx = float(new_w) / float(w_img)
        sy = float(new_h) / float(h_img)
        atlas_rgba = np.asarray(
            Image.fromarray(atlas_rgba).resize((new_w, new_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        if labpbr_n_rgba is not None and labpbr_n_rgba.shape[:2] == (h_img, w_img):
            labpbr_n_rgba = np.asarray(
                Image.fromarray(labpbr_n_rgba).resize((new_w, new_h), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        if labpbr_s_rgba is not None and labpbr_s_rgba.shape[:2] == (h_img, w_img):
            labpbr_s_rgba = np.asarray(
                Image.fromarray(labpbr_s_rgba).resize((new_w, new_h), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        uv_map = {
            b_k: {
                f_k: (
                    [float(r[0]) * sx, float(r[1]) * sy, float(r[2]) * sx, float(r[3]) * sy, int(r[4])]
                    if len(r) >= 5
                    else [float(r[0]) * sx, float(r[1]) * sy, float(r[2]) * sx, float(r[3]) * sy]
                )
                for f_k, r in f_dict.items()
                if isinstance(r, (list, tuple)) and len(r) >= 4
            }
            for b_k, f_dict in uv_map.items()
        }
        h_img, w_img = new_h, new_w
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
                    px_unit = 16.0 / max(1, int(program.meta.texel_density))
                    wu_eff = max(px_unit, float(wu))
                    hv_eff = max(px_unit, float(hv))
                    u_min = float(cs.min()) / lab.shape[1] * wu_eff
                    u_max = float(cs.max() + 1) / lab.shape[1] * wu_eff
                    v_min = float(rs.min()) / lab.shape[0] * hv_eff
                    v_max = float(rs.max() + 1) / lab.shape[0] * hv_eff
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


# ---------------------------------------------------------------------------
# glTF 2.0 / Sketchfab Archive Cuboid Decompiler (format_kind="gltf_2")
# ---------------------------------------------------------------------------

_GLTF_COMPONENT_DTYPES: dict[int, Any] = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_GLTF_TYPE_NCOMP: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def extract_gltf_bundle_from_zip_or_glb(raw_bytes: bytes | Path | str) -> dict[str, Any]:
    """Unpack a Sketchfab ``gltf.zip``, ``.glb`` binary, or ``.gltf`` directory path into a memory bundle."""
    if isinstance(raw_bytes, (Path, str)):
        p = Path(raw_bytes)
        if p.is_dir():
            gltf_files = sorted(p.glob("*.gltf"))
            if not gltf_files:
                raise ValueError(f"No .gltf file found in directory {p}")
            gltf_doc = json.loads(gltf_files[0].read_text(encoding="utf-8"))
            buffers: dict[str, bytes] = {}
            images: dict[str, bytes] = {}
            for sub in p.rglob("*"):
                if sub.is_file():
                    rel = sub.relative_to(p).as_posix()
                    low = rel.lower()
                    if low.endswith(".bin"):
                        buffers[rel] = sub.read_bytes()
                        buffers[Path(rel).name] = buffers[rel]
                    elif low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        images[rel] = sub.read_bytes()
                        images[Path(rel).name] = images[rel]
            return {"gltf": gltf_doc, "buffers": buffers, "images": images}
        raw_data = p.read_bytes()
    else:
        raw_data = raw_bytes

    if raw_data[:4] == b"glTF":
        # Binary .glb container (glTF 2.0 spec section 4)
        if len(raw_data) < 20:
            raise ValueError("Truncated .glb header")
        _, version, total_len = struct.unpack("<4sII", raw_data[:12])
        if version != 2:
            raise UnsupportedFeature(f"Unsupported .glb version {version}; expected glTF 2.0")
        offset = 12
        gltf_json: dict[str, Any] | None = None
        glb_bin = b""
        while offset + 8 <= min(len(raw_data), total_len):
            chunk_len, chunk_type = struct.unpack("<II", raw_data[offset : offset + 8])
            offset += 8
            chunk_payload = raw_data[offset : offset + chunk_len]
            offset += chunk_len
            if chunk_type == 0x4E4F534A:  # JSON
                gltf_json = json.loads(chunk_payload.decode("utf-8"))
            elif chunk_type == 0x004E4942:  # BIN\0
                glb_bin = chunk_payload
        if gltf_json is None:
            raise ValueError("Missing JSON chunk in .glb")
        return {"gltf": gltf_json, "buffers": {"__glb_0__": glb_bin}, "images": {}}

    if raw_data[:4] == b"PK\x03\x04":
        buffers = {}
        images = {}
        gltf_doc_zip: dict[str, Any] | None = None
        with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
            for zname in zf.namelist():
                if zname.endswith("/"):
                    continue
                low = zname.lower()
                if low.endswith(".gltf") and gltf_doc_zip is None:
                    gltf_doc_zip = json.loads(zf.read(zname).decode("utf-8"))
                elif low.endswith(".glb") and gltf_doc_zip is None:
                    return extract_gltf_bundle_from_zip_or_glb(zf.read(zname))
                elif low.endswith(".bin"):
                    b_data = zf.read(zname)
                    buffers[zname] = b_data
                    buffers[Path(zname).name] = b_data
                elif low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    i_data = zf.read(zname)
                    images[zname] = i_data
                    images[Path(zname).name] = i_data
        if gltf_doc_zip is None:
            raise ValueError("Zip archive contains no .gltf or .glb scene")
        return {"gltf": gltf_doc_zip, "buffers": buffers, "images": images}

    gltf_plain = json.loads(raw_data.decode("utf-8"))
    return {"gltf": gltf_plain, "buffers": {}, "images": {}}


def _quat_xyzw_to_mat3(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]))
    nrm = math.sqrt(x * x + y * y + z * z + w * w)
    if nrm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / nrm, y / nrm, z / nrm, w / nrm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float64,
    )


def _gltf_node_local_mat4(node: dict[str, Any]) -> np.ndarray:
    mat_raw = node.get("matrix")
    if isinstance(mat_raw, (list, tuple)) and len(mat_raw) == 16:
        # glTF matrices are column-major
        return np.asarray(mat_raw, dtype=np.float64).reshape((4, 4), order="F")
    m = np.eye(4, dtype=np.float64)
    sc = node.get("scale")
    rot = node.get("rotation")
    tr = node.get("translation")
    r3 = _quat_xyzw_to_mat3(rot) if isinstance(rot, (list, tuple)) and len(rot) == 4 else np.eye(3, dtype=np.float64)
    if isinstance(sc, (list, tuple)) and len(sc) == 3:
        s_vec = np.asarray(sc, dtype=np.float64)
        r3 = r3 * s_vec[None, :]
    m[:3, :3] = r3
    if isinstance(tr, (list, tuple)) and len(tr) == 3:
        m[:3, 3] = np.asarray(tr, dtype=np.float64)
    return m


def _decompose_mat4_trs(m4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 affine matrix into (translation, proper_R_so3, signed_scale_vec3)."""
    t_vec = m4[:3, 3].astype(np.float64).copy()
    lin = m4[:3, :3].astype(np.float64)
    col_norms = np.linalg.norm(lin, axis=0)
    safe_norms = np.where(col_norms > 1e-12, col_norms, 1.0)
    r_raw = lin / safe_norms[None, :]
    u, _, vt = np.linalg.svd(r_raw)
    r_orth = u @ vt
    s_vec = col_norms.copy()
    if np.linalg.det(r_orth) < 0.0:
        r_orth[:, 0] *= -1.0
        s_vec[0] *= -1.0
    return t_vec, r_orth, s_vec


def _read_gltf_buffer_bytes(gltf: dict[str, Any], buffers_map: dict[str, bytes], buf_idx: int) -> bytes:
    buf_list = gltf.get("buffers") or []
    if buf_idx < 0 or buf_idx >= len(buf_list):
        if "__glb_0__" in buffers_map and buf_idx == 0:
            return buffers_map["__glb_0__"]
        raise ValueError(f"glTF buffer index {buf_idx} out of range")
    b_spec = buf_list[buf_idx]
    uri = str(b_spec.get("uri") or "")
    if not uri and "__glb_0__" in buffers_map:
        return buffers_map["__glb_0__"]
    if uri.startswith("data:"):
        b64_pos = uri.find("base64,")
        if b64_pos >= 0:
            return base64.b64decode(uri[b64_pos + 7 :])
    if uri in buffers_map:
        return buffers_map[uri]
    short = Path(uri).name
    if short in buffers_map:
        return buffers_map[short]
    if len(buffers_map) == 1:
        return next(iter(buffers_map.values()))
    raise ValueError(f"Unable to resolve glTF binary buffer URI {uri!r}")


def _read_gltf_accessor(gltf: dict[str, Any], buffers_map: dict[str, bytes], acc_idx: int) -> np.ndarray:
    acc = gltf["accessors"][acc_idx]
    cnt = int(acc["count"])
    ctype = _GLTF_COMPONENT_DTYPES.get(int(acc["componentType"]))
    if ctype is None:
        raise UnsupportedFeature(f"Unsupported glTF accessor componentType {acc.get('componentType')}")
    ncomp = _GLTF_TYPE_NCOMP.get(str(acc["type"]))
    if ncomp is None:
        raise UnsupportedFeature(f"Unsupported glTF accessor type {acc.get('type')!r}")
    if "bufferView" not in acc:
        out_z = np.zeros((cnt, ncomp), dtype=np.float64)
        return out_z[:, 0] if ncomp == 1 else out_z

    bv = gltf["bufferViews"][int(acc["bufferView"])]
    raw_buf = _read_gltf_buffer_bytes(gltf, buffers_map, int(bv.get("buffer", 0)))
    byte_offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    stride = int(bv.get("byteStride", 0) or 0)
    elem_bytes = np.dtype(ctype).itemsize * ncomp
    if stride and stride != elem_bytes:
        arr = np.empty((cnt, ncomp), dtype=ctype)
        for i in range(cnt):
            arr[i] = np.frombuffer(raw_buf, dtype=ctype, count=ncomp, offset=byte_offset + i * stride)
    else:
        arr_flat = np.frombuffer(raw_buf, dtype=ctype, count=cnt * ncomp, offset=byte_offset)
        arr = arr_flat.reshape((cnt, ncomp)).copy()

    if acc.get("normalized") and ctype != np.float32:
        info = np.iinfo(ctype)
        if info.min < 0:
            arr_f = np.maximum(-1.0, arr.astype(np.float64) / float(info.max))
        else:
            arr_f = arr.astype(np.float64) / float(info.max)
        return arr_f[:, 0] if ncomp == 1 else arr_f
    return arr[:, 0] if ncomp == 1 else arr


def _fit_exact_obb_from_triangles(
    pts_tri: np.ndarray,
    uv_tri: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]] | None] | None:
    """Verify that triangles form an exact cuboid or planar quad; return (ctr, R_so3, span, uv_norm_faces) or None."""
    flat_pts = pts_tri.reshape(-1, 3)
    aabb_lo = flat_pts.min(axis=0)
    aabb_hi = flat_pts.max(axis=0)
    aabb_sp = aabb_hi - aabb_lo
    max_dim = max(float(np.max(aabb_sp)), 1e-4)
    tol = 2.5e-3 * max(1.0, max_dim)

    # 1. Fast path: axis-aligned cuboid/quad in local frame
    aabb_ctr = 0.5 * (aabb_lo + aabb_hi)
    aabb_dist = np.linalg.norm(np.abs(flat_pts - aabb_ctr) - 0.5 * aabb_sp, axis=1)
    if float(np.max(aabb_dist)) <= tol:
        r_fit = np.eye(3, dtype=np.float64)
        ctr_fit = aabb_ctr
        sp_fit = np.maximum(0.0, aabb_sp)
    else:
        # 2. Oriented cuboid/quad from triangle face normals
        v0, v1, v2 = pts_tri[:, 0], pts_tri[:, 1], pts_tri[:, 2]
        cr = np.cross(v1 - v0, v2 - v0)
        nrms = np.linalg.norm(cr, axis=1)
        valid = nrms > 1e-11
        if int(valid.sum()) < 2:
            return None
        un = cr[valid] / nrms[valid, None]
        axes: list[np.ndarray] = [un[0]]
        for cand in un[1:]:
            if all(abs(float(np.dot(cand, ax))) < 0.05 for ax in axes):
                axes.append(cand)
                if len(axes) == 3:
                    break
        if len(axes) == 1:
            # Planar quad: derive tangent basis from longest edge
            edges = (v1 - v0).reshape(-1, 3)
            elens = np.linalg.norm(edges, axis=1)
            if float(np.max(elens)) <= 1e-9:
                return None
            t1 = edges[int(np.argmax(elens))] / float(np.max(elens))
            t1 = t1 - float(np.dot(t1, axes[0])) * axes[0]
            if float(np.linalg.norm(t1)) <= 1e-9:
                return None
            t1 = t1 / float(np.linalg.norm(t1))
            t2 = np.cross(axes[0], t1)
            axes = [t1, t2, axes[0]]
        elif len(axes) == 2:
            axes.append(np.cross(axes[0], axes[1]))
        r_cand = np.column_stack(axes)
        u_s, _, vt_s = np.linalg.svd(r_cand)
        r_fit = u_s @ vt_s
        if np.linalg.det(r_fit) < 0.0:
            r_fit[:, 2] *= -1.0
        loc_pts = flat_pts @ r_fit
        loc_lo = loc_pts.min(axis=0)
        loc_hi = loc_pts.max(axis=0)
        loc_ctr = 0.5 * (loc_lo + loc_hi)
        sp_fit = np.maximum(0.0, loc_hi - loc_lo)
        obb_dist = np.linalg.norm(np.abs(loc_pts - loc_ctr) - 0.5 * sp_fit, axis=1)
        if float(np.max(obb_dist)) > tol:
            return None
        ctr_fit = r_fit @ loc_ctr

    # Verify non-degenerate (at most 1 zero-thickness planar axis)
    zero_axes = sum(1 for c in sp_fit if float(c) <= 1e-7 * max(1.0, max_dim))
    if zero_axes >= 2:
        return None

    uv_faces_norm: dict[str, list[float]] | None = None
    if uv_tri is not None and uv_tri.shape[:2] == pts_tri.shape[:2]:
        uv_faces_norm = {}
        v0_l = (pts_tri[:, 0] - ctr_fit) @ r_fit
        v1_l = (pts_tri[:, 1] - ctr_fit) @ r_fit
        v2_l = (pts_tri[:, 2] - ctr_fit) @ r_fit
        n_l = np.cross(v1_l - v0_l, v2_l - v0_l)
        norms_l = np.linalg.norm(n_l, axis=1)
        for fname in FACES:
            t_r_b, t_d_b, n_o_b = FACE_AXES_BOX[fname]
            t_r = np.asarray(t_r_b, dtype=np.float64)
            t_d = np.asarray(t_d_b, dtype=np.float64)
            fvec = np.asarray(n_o_b, dtype=np.float64)
            dots = n_l @ fvec
            hit = (norms_l > 1e-12) & (dots > 0.85 * norms_l)
            if not np.any(hit):
                continue
            v_sel = np.stack([v0_l[hit], v1_l[hit], v2_l[hit]], axis=1).reshape(-1, 3)
            uv_sel = uv_tri[hit].reshape(-1, 2).copy()
            # Unwrap simple shifted [k, k+1] glTF REPEAT tiles into [0, 1]
            u_span = float(np.max(uv_sel[:, 0]) - np.min(uv_sel[:, 0]))
            v_span = float(np.max(uv_sel[:, 1]) - np.min(uv_sel[:, 1]))
            if (float(np.min(uv_sel[:, 0])) < -1e-4 or float(np.max(uv_sel[:, 0])) > 1.0001) and u_span <= 1.001:
                uv_sel[:, 0] -= math.floor(float(np.min(uv_sel[:, 0])) + 1e-5)
            if (float(np.min(uv_sel[:, 1])) < -1e-4 or float(np.max(uv_sel[:, 1])) > 1.0001) and v_span <= 1.001:
                uv_sel[:, 1] -= math.floor(float(np.min(uv_sel[:, 1])) + 1e-5)

            d_u = 0.5 * float(np.abs(t_r) @ sp_fit)
            d_v = 0.5 * float(np.abs(t_d) @ sp_fit)
            if d_u > 1e-8 and d_v > 1e-8:
                s_u = np.clip(0.5 + (v_sel @ t_r) / (2.0 * d_u), 0.0, 1.0)
                s_v = np.clip(0.5 + (v_sel @ t_d) / (2.0 * d_v), 0.0, 1.0)
                best_rect: list[float] | None = None
                best_err = 1e18
                for cand_rot in (0, 90):
                    if cand_rot == 0:
                        su_t, sv_t = s_u, s_v
                    else:
                        su_t, sv_t = s_v, 1.0 - s_u
                    m_u0, m_u1 = su_t < 0.25, su_t > 0.75
                    m_v0, m_v1 = sv_t < 0.25, sv_t > 0.75
                    if np.any(m_u0) and np.any(m_u1) and np.any(m_v0) and np.any(m_v1):
                        u0_c = float(np.clip(np.mean(uv_sel[m_u0, 0]), 0.0, 1.0))
                        u1_c = float(np.clip(np.mean(uv_sel[m_u1, 0]), 0.0, 1.0))
                        v0_c = float(np.clip(np.mean(uv_sel[m_v0, 1]), 0.0, 1.0))
                        v1_c = float(np.clip(np.mean(uv_sel[m_v1, 1]), 0.0, 1.0))
                        pred_u = u0_c + su_t * (u1_c - u0_c)
                        pred_v = v0_c + sv_t * (v1_c - v0_c)
                        err_c = float(np.max(np.abs(uv_sel[:, 0] - pred_u) + np.abs(uv_sel[:, 1] - pred_v)))
                        # Tie-break strongly in favor of cand_rot == 0
                        if err_c + (1e-4 if cand_rot == 90 else 0.0) < best_err:
                            best_err = err_c
                            best_rect = [u0_c, v0_c, u1_c, v1_c] if cand_rot == 0 else [u0_c, v0_c, u1_c, v1_c, 90]
                if best_rect is not None:
                    uv_faces_norm[fname] = best_rect
                    continue

            u0, v0_uv = float(np.min(uv_sel[:, 0])), float(np.min(uv_sel[:, 1]))
            u1, v1_uv = float(np.max(uv_sel[:, 0])), float(np.max(uv_sel[:, 1]))
            uv_faces_norm[fname] = [
                float(np.clip(u0, 0.0, 1.0)),
                float(np.clip(v0_uv, 0.0, 1.0)),
                float(np.clip(u1, 0.0, 1.0)),
                float(np.clip(v1_uv, 0.0, 1.0)),
            ]
    return ctr_fit, r_fit, sp_fit, uv_faces_norm


def _linear_channel_to_srgb_u8(val: float) -> int:
    c = float(np.clip(val, 0.0, 1.0))
    s = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(np.clip(round(s * 255.0), 0, 255))


def _extract_gltf_mesh_cuboids(
    gltf: dict[str, Any],
    buffers_map: dict[str, bytes],
    mesh_idx: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]] | None, str]]:
    meshes = gltf.get("meshes") or []
    if mesh_idx < 0 or mesh_idx >= len(meshes):
        raise ValueError(f"glTF mesh index {mesh_idx} out of range")
    prims = meshes[mesh_idx].get("primitives") or []
    if not prims:
        return []

    out_cuboids: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]] | None, str]] = []
    for prim in prims:
        mode = prim.get("mode", 4)
        if mode != 4:
            raise UnsupportedFeature(f"glTF primitive mode {mode} is not TRIANGLES")
        attrs = prim.get("attributes") or {}
        if "POSITION" not in attrs:
            continue
        pos = _read_gltf_accessor(gltf, buffers_map, int(attrs["POSITION"])).astype(np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 4:
            raise UnsupportedFeature("glTF primitive POSITION must be (N>=4, 3)")
        if "indices" in prim:
            idx = _read_gltf_accessor(gltf, buffers_map, int(prim["indices"])).astype(np.int64).ravel()
        else:
            idx = np.arange(len(pos), dtype=np.int64)
        if len(idx) < 6 or len(idx) % 3 != 0:
            raise UnsupportedFeature("glTF primitive triangle index count invalid for cuboid/plane")
        tris = idx.reshape(-1, 3)
        n_tris = len(tris)
        uv_arr: np.ndarray | None = None
        if "TEXCOORD_0" in attrs:
            uv_raw = _read_gltf_accessor(gltf, buffers_map, int(attrs["TEXCOORD_0"])).astype(np.float64)
            if uv_raw.ndim == 2 and uv_raw.shape == (len(pos), 2):
                uv_arr = uv_raw

        col_arr: np.ndarray | None = None
        if "COLOR_0" in attrs:
            c_raw = _read_gltf_accessor(gltf, buffers_map, int(attrs["COLOR_0"])).astype(np.float64)
            if c_raw.ndim == 2 and c_raw.shape[0] == len(pos) and c_raw.shape[1] in (3, 4):
                col_arr = c_raw

        base_mat_key = f"mat_{int(prim['material'])}" if isinstance(prim.get("material"), int) else "default"

        def _chunk_mat_key(tri_indices: np.ndarray) -> str:
            if base_mat_key != "default" or col_arr is None:
                return base_mat_key
            c_mean = np.mean(col_arr[tri_indices.ravel()], axis=0)
            ru = _linear_channel_to_srgb_u8(float(c_mean[0]))
            gu = _linear_channel_to_srgb_u8(float(c_mean[1]))
            bu = _linear_channel_to_srgb_u8(float(c_mean[2]))
            au = int(np.clip(round(float(c_mean[3]) * 255.0), 0, 255)) if len(c_mean) >= 4 else 255
            return f"vcol_{ru:02x}{gu:02x}{bu:02x}{au:02x}"

        if n_tris <= 12 and len(pos) <= 24:
            uv_tri = uv_arr[tris] if uv_arr is not None else None
            fitted = _fit_exact_obb_from_triangles(pos[tris], uv_tri)
            if fitted is None:
                raise UnsupportedFeature("gltf contains non-cuboid free mesh geometry")
            out_cuboids.append((*fitted, _chunk_mat_key(tris)))
        elif n_tris > 12 and n_tris % 12 == 0 and n_tris <= 12 * 512:
            for k in range(n_tris // 12):
                tri_chunk = tris[12 * k : 12 * (k + 1)]
                uv_chunk = uv_arr[tri_chunk] if uv_arr is not None else None
                fitted_k = _fit_exact_obb_from_triangles(pos[tri_chunk], uv_chunk)
                if fitted_k is None:
                    raise UnsupportedFeature("gltf contains non-cuboid merged free mesh geometry")
                out_cuboids.append((*fitted_k, _chunk_mat_key(tri_chunk)))
        else:
            raise UnsupportedFeature(f"gltf primitive with {len(pos)} vertices / {n_tris} triangles is a non-cuboid free mesh")
    return out_cuboids


def _decode_gltf_image_by_index(
    gltf: dict[str, Any],
    buffers_map: dict[str, bytes],
    images_map: dict[str, bytes],
    img_idx: int,
) -> tuple[np.ndarray | None, str]:
    images_list = gltf.get("images") or []
    if not (0 <= img_idx < len(images_list)):
        return None, "atlas_0.png"
    img_spec = images_list[img_idx]
    uri = str(img_spec.get("uri") or "")
    tex_name = Path(uri).name if (uri and not uri.startswith("data:")) else str(img_spec.get("name") or f"atlas_{img_idx}.png")
    if not tex_name.lower().endswith(".png"):
        tex_name = f"{re.sub(r'\\.[A-Za-z0-9]+$', '', tex_name) or f'atlas_{img_idx}'}.png"

    raw_img: bytes | None = None
    if uri.startswith("data:image/"):
        b64_pos = uri.find("base64,")
        if b64_pos >= 0:
            raw_img = base64.b64decode(uri[b64_pos + 7 :])
    elif uri:
        raw_img = images_map.get(uri) or images_map.get(Path(uri).name)
    elif "bufferView" in img_spec:
        bv = gltf["bufferViews"][int(img_spec["bufferView"])]
        b_all = _read_gltf_buffer_bytes(gltf, buffers_map, int(bv.get("buffer", 0)))
        off = int(bv.get("byteOffset", 0))
        ln = int(bv.get("byteLength", 0))
        raw_img = b_all[off : off + ln]

    if raw_img is None and img_idx == 0 and len(images_map) == 1:
        raw_img = next(iter(images_map.values()))
    if raw_img is None:
        return None, tex_name
    try:
        rgba = np.asarray(Image.open(io.BytesIO(raw_img)).convert("RGBA"), dtype=np.uint8)
        return rgba, tex_name
    except Exception:
        return None, tex_name


def _build_gltf_composite_atlas(
    gltf: dict[str, Any],
    buffers_map: dict[str, bytes],
    images_map: dict[str, bytes],
    active_mat_keys: Sequence[str],
) -> tuple[np.ndarray | None, str, dict[str, tuple[float, float, float, float]]]:
    """Build RGBA atlas (single or vertical multi-material composite) + per-material (ou, ov, sw, sh) UV slice."""
    textures_list = gltf.get("textures") or []
    materials_list = gltf.get("materials") or []
    images_list = gltf.get("images") or []

    img_cache: dict[int, tuple[np.ndarray | None, str]] = {}

    def _get_img(idx: int) -> tuple[np.ndarray | None, str]:
        if idx not in img_cache:
            img_cache[idx] = _decode_gltf_image_by_index(gltf, buffers_map, images_map, idx)
        return img_cache[idx]

    slices_by_sig: dict[str, np.ndarray] = {}
    sig_order: list[str] = []
    mat_to_sig: dict[str, str] = {}
    primary_tex_name = "atlas_0.png"

    for mk in (list(active_mat_keys) or ["default"]):
        if mk in mat_to_sig:
            continue
        if mk.startswith("mat_"):
            m_idx = int(mk[4:])
            mat = materials_list[m_idx] if 0 <= m_idx < len(materials_list) and isinstance(materials_list[m_idx], dict) else {}
            pbr = mat.get("pbrMetallicRoughness") if isinstance(mat.get("pbrMetallicRoughness"), dict) else {}
            bcf = pbr.get("baseColorFactor") if isinstance(pbr.get("baseColorFactor"), (list, tuple)) and len(pbr.get("baseColorFactor")) >= 3 else [1.0, 1.0, 1.0, 1.0]
            r_u8 = _linear_channel_to_srgb_u8(float(bcf[0]))
            g_u8 = _linear_channel_to_srgb_u8(float(bcf[1]))
            b_u8 = _linear_channel_to_srgb_u8(float(bcf[2]))
            a_u8 = int(np.clip(round(float(bcf[3]) * 255.0), 0, 255)) if len(bcf) >= 4 else 255
            if str(mat.get("alphaMode", "OPAQUE")).upper() == "OPAQUE":
                a_u8 = 255

            bct = pbr.get("baseColorTexture") if isinstance(pbr.get("baseColorTexture"), dict) else None
            src_img_idx: int | None = None
            if bct is not None and isinstance(bct.get("index"), int):
                t_i = int(bct["index"])
                if 0 <= t_i < len(textures_list) and isinstance(textures_list[t_i].get("source"), int):
                    src_img_idx = int(textures_list[t_i]["source"])
            if src_img_idx is not None:
                arr_i, tname_i = _get_img(src_img_idx)
                if primary_tex_name == "atlas_0.png":
                    primary_tex_name = tname_i
                if arr_i is not None:
                    has_tint = max(abs(r_u8 - 255), abs(g_u8 - 255), abs(b_u8 - 255), abs(a_u8 - 255)) > 3
                    sig = f"img_{src_img_idx}_{r_u8:02x}{g_u8:02x}{b_u8:02x}{a_u8:02x}" if has_tint else f"img_{src_img_idx}"
                    if sig not in slices_by_sig:
                        if has_tint:
                            scale_tint = np.array([r_u8 / 255.0, g_u8 / 255.0, b_u8 / 255.0, a_u8 / 255.0], dtype=np.float32)
                            slices_by_sig[sig] = np.clip(np.round(arr_i.astype(np.float32) * scale_tint[None, None, :]), 0, 255).astype(np.uint8)
                        else:
                            slices_by_sig[sig] = arr_i
                        sig_order.append(sig)
                    mat_to_sig[mk] = sig
                    continue

            # Untextured PBR material: synthesize exact 8x8 solid baseColorFactor patch
            sig_c = f"rgba_{r_u8:02x}{g_u8:02x}{b_u8:02x}{a_u8:02x}"
            if sig_c not in slices_by_sig:
                patch = np.empty((8, 8, 4), dtype=np.uint8)
                patch[:, :] = (r_u8, g_u8, b_u8, a_u8)
                slices_by_sig[sig_c] = patch
                sig_order.append(sig_c)
            mat_to_sig[mk] = sig_c
        elif mk.startswith("vcol_") and len(mk) == 13:
            hx = mk[5:]
            r_u8, g_u8, b_u8, a_u8 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16), int(hx[6:8], 16)
            sig_v = f"rgba_{hx}"
            if sig_v not in slices_by_sig:
                patch = np.empty((8, 8, 4), dtype=np.uint8)
                patch[:, :] = (r_u8, g_u8, b_u8, a_u8)
                slices_by_sig[sig_v] = patch
                sig_order.append(sig_v)
            mat_to_sig[mk] = sig_v
        else:
            if images_list:
                arr0, tname0 = _get_img(0)
                if primary_tex_name == "atlas_0.png":
                    primary_tex_name = tname0
                if arr0 is not None:
                    sig0 = "img_0"
                    if sig0 not in slices_by_sig:
                        slices_by_sig[sig0] = arr0
                        sig_order.append(sig0)
                    mat_to_sig[mk] = sig0
            elif images_map:
                raw_any = next(iter(images_map.values()))
                try:
                    arr_any = np.asarray(Image.open(io.BytesIO(raw_any)).convert("RGBA"), dtype=np.uint8)
                    sig_a = "img_ext_0"
                    if sig_a not in slices_by_sig:
                        slices_by_sig[sig_a] = arr_any
                        sig_order.append(sig_a)
                    mat_to_sig[mk] = sig_a
                except Exception:
                    pass

    if not sig_order:
        return None, primary_tex_name, {mk: (0.0, 0.0, 16.0, 16.0) for mk in active_mat_keys}

    if len(sig_order) == 1:
        only_arr = slices_by_sig[sig_order[0]]
        sh, sw = only_arr.shape[:2]
        return only_arr, primary_tex_name, {mk: (0.0, 0.0, float(sw), float(sh)) for mk in mat_to_sig}

    # Multi-material composite sheet: cap individual huge 2K/4K slices to <=1024px so pixel-art stays 1:1
    norm_slices: dict[str, np.ndarray] = {}
    for sig in sig_order:
        arr_s = slices_by_sig[sig]
        hs, ws = arr_s.shape[:2]
        if max(hs, ws) > 1024:
            sf = 1024.0 / float(max(hs, ws))
            nw, nh = max(1, int(round(ws * sf))), max(1, int(round(hs * sf)))
            arr_s = np.asarray(Image.fromarray(arr_s).resize((nw, nh), Image.Resampling.NEAREST), dtype=np.uint8)
        norm_slices[sig] = arr_s

    comp_w = max(int(norm_slices[s].shape[1]) for s in sig_order)
    comp_h = sum(int(norm_slices[s].shape[0]) for s in sig_order)
    comp = np.zeros((comp_h, comp_w, 4), dtype=np.uint8)
    sig_rect: dict[str, tuple[float, float, float, float]] = {}
    y_cur = 0
    for sig in sig_order:
        arr_s = norm_slices[sig]
        hs, ws = arr_s.shape[:2]
        comp[y_cur : y_cur + hs, :ws] = arr_s
        sig_rect[sig] = (0.0, float(y_cur), float(ws), float(hs))
        y_cur += hs

    mat_slice_map = {mk: sig_rect[sg] for mk, sg in mat_to_sig.items()}
    return comp, primary_tex_name, mat_slice_map


def _infer_gltf_unit_scale(raw_world_spans: Sequence[np.ndarray]) -> float:
    pos_vals = [float(v) for sp in raw_world_spans for v in sp.ravel() if float(v) > 1e-7]
    if not pos_vals:
        return 1.0
    med = float(np.median(np.asarray(pos_vals, dtype=np.float64)))
    if med >= 0.85:
        return 1.0
    # Blockbench glTF exporter maps 16 pixels = 1.0 meter
    scaled_16 = [v * 16.0 for v in pos_vals]
    dyadic_hits = sum(1 for v in scaled_16 if abs(round(v * 16.0) / 16.0 - v) <= 2e-3)
    if dyadic_hits >= max(1, int(0.6 * len(scaled_16))) or (0.02 <= med < 0.85):
        return 16.0
    return 1.0


def _decompile_gltf(
    payload: dict[str, Any],
) -> tuple[Program, dict[str, dict[str, list[float]]], np.ndarray | None, float]:
    """Decompile glTF 2.0 / Sketchfab cuboid scene -> validated Block-DSL v3.2 Program + UVs + 8-corner error."""
    if isinstance(payload.get("gltf"), dict):
        gltf = payload["gltf"]
        buffers_map: dict[str, bytes] = dict(payload.get("buffers") or {})
        images_map: dict[str, bytes] = dict(payload.get("images") or {})
    else:
        gltf = payload
        buffers_map = {}
        images_map = {}

    nodes = gltf.get("nodes") or []
    if not nodes:
        raise ValueError("glTF scene contains no nodes")

    # Identify animated node indices so we never collapse an animated bone into a static Box
    animated_nodes: set[int] = set()
    for anim in gltf.get("animations") or []:
        for ch in anim.get("channels") or []:
            tgt = ch.get("target") or {}
            if isinstance(tgt.get("node"), int):
                animated_nodes.add(int(tgt["node"]))

    # Locate clean Y-up model root(s), stripping Sketchfab viewer Z-up wrapper when present
    name_to_idx = {str(n.get("name") or ""): i for i, n in enumerate(nodes) if isinstance(n, dict)}
    if "GLTF_SceneRootNode" in name_to_idx:
        root_indices = list(nodes[name_to_idx["GLTF_SceneRootNode"]].get("children") or [])
    elif "RootNode" in name_to_idx:
        root_indices = list(nodes[name_to_idx["RootNode"]].get("children") or [])
    elif "Sketchfab_model" in name_to_idx:
        sf_children = list(nodes[name_to_idx["Sketchfab_model"]].get("children") or [])
        if len(sf_children) == 1 and nodes[sf_children[0]].get("mesh") is None and nodes[sf_children[0]].get("children"):
            c0 = sf_children[0]
            m_sf = _gltf_node_local_mat4(nodes[name_to_idx["Sketchfab_model"]])[:3, :3]
            m_c0 = _gltf_node_local_mat4(nodes[c0])[:3, :3]
            prod = m_sf @ m_c0
            s_est = float(np.linalg.norm(prod[:, 0]))
            if s_est > 1e-9 and np.allclose(prod / s_est, np.eye(3), atol=1e-2):
                root_indices = list(nodes[c0].get("children") or [])
            else:
                root_indices = sf_children
        else:
            root_indices = sf_children
    else:
        sc_idx = int(gltf.get("scene", 0) or 0)
        scenes = gltf.get("scenes") or []
        root_indices = list(scenes[sc_idx].get("children") or scenes[sc_idx].get("nodes") or []) if scenes else [0]

    # Unwrap anonymous identity container nodes (e.g. '_8', '_2111', 'root') at top of hierarchy
    for _ in range(4):
        if len(root_indices) == 1:
            r0 = root_indices[0]
            n0 = nodes[r0]
            if (
                r0 not in animated_nodes
                and n0.get("mesh") is None
                and n0.get("children")
                and np.allclose(_gltf_node_local_mat4(n0), np.eye(4), atol=1e-6)
            ):
                root_indices = list(n0["children"])
                continue
        break

    if not root_indices:
        raise ValueError("glTF scene resolved to zero root nodes")

    mesh_cuboids_cache: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]] | None, str]]] = {}

    def _get_mesh_cuboids(m_idx: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]] | None, str]]:
        if m_idx not in mesh_cuboids_cache:
            mesh_cuboids_cache[m_idx] = _extract_gltf_mesh_cuboids(gltf, buffers_map, m_idx)
        return mesh_cuboids_cache[m_idx]

    # Pre-traverse scene to verify all primitives are cuboids/planes and gather world spans + active materials
    raw_world_spans: list[np.ndarray] = []
    active_mat_keys: list[str] = []
    total_cuboid_count = 0

    def _precheck_subtree(n_idx: int, parent_m4: np.ndarray) -> None:
        nonlocal total_cuboid_count
        node = nodes[n_idx]
        cur_m4 = parent_m4 @ _gltf_node_local_mat4(node)
        if node.get("mesh") is not None:
            cubs = _get_mesh_cuboids(int(node["mesh"]))
            _, _, s_vec = _decompose_mat4_trs(cur_m4)
            abs_s = np.abs(s_vec)
            for _, r_loc, sp_loc, _, m_key in cubs:
                if m_key not in active_mat_keys:
                    active_mat_keys.append(m_key)
                # Effective axis lengths under node scale
                scaled_axes = np.linalg.norm((r_loc * sp_loc[None, :]) * abs_s[:, None], axis=0)
                raw_world_spans.append(scaled_axes)
                total_cuboid_count += 1
                if total_cuboid_count > 1500:
                    raise UnsupportedFeature("glTF exceeds 1500 cuboid primitive limit")
        for ch_i in node.get("children") or []:
            _precheck_subtree(int(ch_i), cur_m4)

    for r_i in root_indices:
        _precheck_subtree(int(r_i), np.eye(4, dtype=np.float64))

    if total_cuboid_count == 0:
        raise ValueError("glTF scene contains no cuboid or planar quad primitives")

    embedded_rgba, raw_tex_path, mat_slice_map = _build_gltf_composite_atlas(
        gltf, buffers_map, images_map, active_mat_keys
    )
    tex_h, tex_w = embedded_rgba.shape[:2] if embedded_rgba is not None else (16, 16)
    default_slice = (0.0, 0.0, float(tex_w), float(tex_h))

    unit_scale = _infer_gltf_unit_scale(raw_world_spans)
    scaled_spans = [sp * unit_scale for sp in raw_world_spans]
    size_step = _infer_dyadic_size_step(scaled_spans)

    def _is_leaf_cube_wrapper(n_idx: int) -> tuple[int, np.ndarray] | None:
        """If node n_idx is an unanimated wrapper around a single mesh leaf, return (mesh_idx, rel_mat4)."""
        if n_idx in animated_nodes:
            return None
        nd = nodes[n_idx]
        ch = nd.get("children") or []
        m_self = _gltf_node_local_mat4(nd)
        if nd.get("mesh") is not None and not ch:
            return (int(nd["mesh"]), m_self)
        if nd.get("mesh") is None and len(ch) == 1:
            c0 = int(ch[0])
            nc0 = nodes[c0]
            if c0 not in animated_nodes and nc0.get("mesh") is not None and not (nc0.get("children") or []):
                return (int(nc0["mesh"]), m_self @ _gltf_node_local_mat4(nc0))
        return None

    used_ids: set[str] = set()
    tex_decl_id: str | None = None
    textures_tuple: tuple[Any, ...] = ()
    if _HAS_DSL_V32 and _TextureDecl is not None:
        t_stem = re.sub(r"\.[A-Za-z0-9]+$", "", raw_tex_path) or "atlas_0"
        tex_decl_id = _sanitize_ident(t_stem, "tex", used_ids)
        textures_tuple = (_TextureDecl(tex_decl_id, raw_tex_path, tex_w, tex_h),)

    uv_map: dict[str, dict[str, list[float]]] = {}
    uv_ratios: list[float] = []
    total_area_u = 0.0
    expected_corners: dict[str, np.ndarray] = {}
    node_idx_to_bone_info: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}

    def _emit_gltf_boxes_for_bone(
        bone_label: str,
        mesh_entries: list[tuple[str, int, np.ndarray]],
        bone_world_piv: np.ndarray,
        bone_world_rot: np.ndarray,
        bone_world_scale: np.ndarray,
    ) -> tuple[list[Box | Locator], list[tuple[str, np.ndarray, np.ndarray | None, bool]]]:
        nonlocal total_area_u
        items_out: list[Box | Locator] = []
        cands_out: list[tuple[str, np.ndarray, np.ndarray | None, bool]] = []
        anchor_name: str | None = None
        anchor_loc: np.ndarray | None = None
        anchor_sz: np.ndarray | None = None
        anchor_is_unrot_box = False
        swap_ew = {"east": "west", "west": "east"}

        for raw_label, m_idx, rel_m4 in mesh_entries:
            cubs = _get_mesh_cuboids(m_idx)
            for c_i, (ctr_p, r_p, sp_p, uv_norm_raw, m_key) in enumerate(cubs):
                p_m4 = np.eye(4, dtype=np.float64)
                p_m4[:3, :3] = r_p
                p_m4[:3, 3] = ctr_p
                # Full transform from primitive local frame into bone frame (with bone scale & unit_scale)
                scale_m4 = np.diag([bone_world_scale[0], bone_world_scale[1], bone_world_scale[2], 1.0])
                full_rel = scale_m4 @ rel_m4 @ p_m4
                t_rel, r_box_loc, s_rel = _decompose_mat4_trs(full_rel)
                loc_ctr = t_rel * unit_scale
                raw_span = np.abs(s_rel) * sp_p * unit_scale
                total_area_u += float(2.0 * (raw_span[0] * raw_span[1] + raw_span[1] * raw_span[2] + raw_span[2] * raw_span[0]))

                ast_size, plane_normal, eff_span = _quantize_box_dimensions(raw_span, size_step)
                ast_sz_arr = np.asarray(ast_size, dtype=np.float64)
                b_raw = raw_label if len(cubs) == 1 else f"{raw_label}_{c_i}"
                bname = _sanitize_ident(b_raw, "box", used_ids)

                eul_arr = _recover_euler_xyz_degrees(r_box_loc)
                box_rot = (
                    0.0 if abs(float(eul_arr[0])) < 1e-7 else round(float(eul_arr[0]), 6),
                    0.0 if abs(float(eul_arr[1])) < 1e-7 else round(float(eul_arr[1]), 6),
                    0.0 if abs(float(eul_arr[2])) < 1e-7 else round(float(eul_arr[2]), 6),
                )
                is_unrot = not any(abs(a) > 1e-6 for a in box_rot)

                uv_norm = uv_norm_raw
                if s_rel[0] < 0.0 and uv_norm:
                    flipped_norm: dict[str, list[float]] = {}
                    for fn, rn in uv_norm.items():
                        fn_dst = swap_ew.get(fn, fn)
                        if len(rn) >= 5 and int(rn[4]) == 90:
                            flipped_norm[fn_dst] = [rn[0], rn[3], rn[2], rn[1], 90]
                        else:
                            flipped_norm[fn_dst] = [rn[2], rn[1], rn[0], rn[3]]
                    uv_norm = flipped_norm

                ou, ov, sw_m, sh_m = mat_slice_map.get(m_key, default_slice)
                is_multi_slice = ov > 0.0 or (ov + sh_m < float(tex_h) - 0.5)
                uv_map[bname] = {}
                if uv_norm:
                    for f, rn in uv_norm.items():
                        wu_e, hv_e = face_uv_extents_units((float(raw_span[0]), float(raw_span[1]), float(raw_span[2])), f)
                        if is_multi_slice:
                            u0_p = ou + float(np.clip(rn[0] * sw_m, 1e-4, max(1e-4, sw_m - 1e-4)))
                            v0_p = ov + float(np.clip(rn[1] * sh_m, 1e-4, max(1e-4, sh_m - 1e-4)))
                            u1_p = ou + float(np.clip(rn[2] * sw_m, 1e-4, max(1e-4, sw_m - 1e-4)))
                            v1_p = ov + float(np.clip(rn[3] * sh_m, 1e-4, max(1e-4, sh_m - 1e-4)))
                        else:
                            u0_p, v0_p = ou + rn[0] * sw_m, ov + rn[1] * sh_m
                            u1_p, v1_p = ou + rn[2] * sw_m, ov + rn[3] * sh_m
                        r_px: list[float] = [
                            round(u0_p, 4),
                            round(v0_p, 4),
                            round(u1_p, 4),
                            round(v1_p, 4),
                        ]
                        if len(rn) >= 5 and int(rn[4]) in (90, 180, 270):
                            r_px.append(int(rn[4]))
                        uv_map[bname][f] = r_px
                        du = abs(r_px[2] - r_px[0])
                        dv = abs(r_px[3] - r_px[1])
                        if wu_e > 0.25 and du > 0.25:
                            uv_ratios.append(16.0 * du / wu_e)
                        if hv_e > 0.25 and dv > 0.25:
                            uv_ratios.append(16.0 * dv / hv_e)
                else:
                    for f in FACES:
                        wu_e, hv_e = face_uv_extents_units((float(raw_span[0]), float(raw_span[1]), float(raw_span[2])), f)
                        if plane_normal is not None and (float(wu_e) <= 1e-8 or float(hv_e) <= 1e-8):
                            continue
                        uv_map[bname][f] = [
                            round(ou, 4),
                            round(ov, 4),
                            round(ou + min(sw_m, max(1.0, float(wu_e))), 4),
                            round(ov + min(sh_m, max(1.0, float(hv_e))), 4),
                        ]
                if not uv_map[bname]:
                    for f in FACES:
                        uv_map[bname][f] = [round(ou, 4), round(ov, 4), round(ou + sw_m, 4), round(ov + sh_m, 4)]

                w_ctr = bone_world_piv + bone_world_rot @ loc_ctr
                w_rot = bone_world_rot @ euler_xyz_matrix(box_rot)
                expected_corners[bname] = _obb_corners_world(w_ctr, w_rot, eff_span)

                if anchor_name is None:
                    pl = _snap_place_without_base(loc_ctr, ast_sz_arr)
                    anchor_name = bname
                    anchor_loc = loc_ctr
                    anchor_sz = ast_sz_arr
                    anchor_is_unrot_box = is_unrot
                else:
                    assert anchor_loc is not None
                    pl = _snap_place_with_base(loc_ctr, ast_sz_arr, anchor_name, anchor_loc, anchor_sz, anchor_is_unrot_box)

                items_out.append(
                    _make_v32_box(
                        bname,
                        ast_size,
                        "mat_0",
                        place=pl,
                        rotate=box_rot,
                        pivot=None,
                        plane_normal=plane_normal,
                        uv_faces=uv_map[bname],
                        texture=tex_decl_id,
                    )
                )
                cands_out.append((bname, loc_ctr, ast_sz_arr, is_unrot))

        if anchor_name is None:
            piv_loc = _sanitize_ident(f"{bone_label}_pivot", "locator", used_ids)
            items_out.append(Locator(name=piv_loc, offset=(0.0, 0.0, 0.0), role="socket"))
            z3 = np.zeros(3, dtype=np.float64)
            cands_out.append((piv_loc, z3, None, False))
        return items_out, cands_out

    def _build_gltf_bone(
        n_idx: int,
        parent_m4: np.ndarray,
        parent_world_piv: np.ndarray,
        parent_world_rot: np.ndarray,
        parent_candidates: Sequence[tuple[str, np.ndarray, np.ndarray | None, bool]],
        is_root: bool,
    ) -> Bone:
        nd = nodes[n_idx]
        cur_m4 = parent_m4 @ _gltf_node_local_mat4(nd)
        w_t_raw, w_rot, w_scale = _decompose_mat4_trs(cur_m4)
        w_piv = w_t_raw * unit_scale

        if is_root:
            g_rot = (0.0, 0.0, 0.0)
            local_trans = w_piv.copy()
            # Fold any root bone rotation into its child transforms so AST root has rest_euler=(0,0,0)
            eff_bone_rot = np.eye(3, dtype=np.float64)
            root_rot_fold = np.eye(4, dtype=np.float64)
            root_rot_fold[:3, :3] = w_rot
        else:
            r_rel = parent_world_rot.T @ w_rot
            eul_b = _recover_euler_xyz_degrees(r_rel)
            g_rot = (
                0.0 if abs(float(eul_b[0])) < 1e-7 else round(float(eul_b[0]), 6),
                0.0 if abs(float(eul_b[1])) < 1e-7 else round(float(eul_b[1]), 6),
                0.0 if abs(float(eul_b[2])) < 1e-7 else round(float(eul_b[2]), 6),
            )
            local_trans = parent_world_rot.T @ (w_piv - parent_world_piv)
            eff_bone_rot = parent_world_rot @ euler_xyz_matrix(g_rot)
            root_rot_fold = np.eye(4, dtype=np.float64)

        bname = _sanitize_ident(nd.get("name") or f"bone_{n_idx}", "bone", used_ids)
        node_idx_to_bone_info[n_idx] = (bname, np.asarray(g_rot, dtype=np.float64), local_trans.copy())

        mesh_entries: list[tuple[str, int, np.ndarray]] = []
        if nd.get("mesh") is not None:
            mesh_entries.append((str(nd.get("name") or f"mesh_{n_idx}"), int(nd["mesh"]), root_rot_fold))

        real_child_bones: list[int] = []
        for ch_raw in nd.get("children") or []:
            ch_i = int(ch_raw)
            wrap = _is_leaf_cube_wrapper(ch_i)
            if wrap is not None:
                m_i, rel_m = wrap
                lbl = str(nodes[ch_i].get("name") or f"cube_{ch_i}")
                mesh_entries.append((lbl, m_i, root_rot_fold @ rel_m))
            else:
                real_child_bones.append(ch_i)

        direct_items, own_cands = _emit_gltf_boxes_for_bone(
            bname, mesh_entries, w_piv, eff_bone_rot, w_scale
        )
        all_items: list[Box | Locator | Bone] = list(direct_items)
        for ch_b in real_child_bones:
            all_items.append(
                _build_gltf_bone(ch_b, cur_m4, w_piv, eff_bone_rot, own_cands, is_root=False)
            )

        if is_root:
            piv_t = (
                0.0 if abs(float(w_piv[0])) < 1e-8 else round(float(w_piv[0]), 6),
                0.0 if abs(float(w_piv[1])) < 1e-8 else round(float(w_piv[1]), 6),
                0.0 if abs(float(w_piv[2])) < 1e-8 else round(float(w_piv[2]), 6),
            )
            return Bone(name=bname, pivot_local=piv_t, items=all_items)

        att = _snap_attach_to_parent(local_trans, parent_candidates, allow_connected=_HAS_DSL_V32)
        return Bone(name=bname, rest_euler=g_rot, attach=att, items=all_items)

    # Check if all root_indices are leaf cube wrappers (static single-level model)
    root_wraps = [_is_leaf_cube_wrapper(int(r_i)) for r_i in root_indices]
    if all(w is not None for w in root_wraps):
        root_bname = _sanitize_ident("gltf_root", "bone", used_ids)
        m_entries = [
            (str(nodes[int(r_i)].get("name") or f"cube_{r_i}"), w[0], w[1])
            for r_i, w in zip(root_indices, root_wraps)
            if w is not None
        ]
        r_items, _ = _emit_gltf_boxes_for_bone(
            root_bname, m_entries, np.zeros(3), np.eye(3), np.ones(3)
        )
        root_bone = Bone(name=root_bname, pivot_local=(0.0, 0.0, 0.0), items=list(r_items))
    elif len(root_indices) == 1:
        r0 = int(root_indices[0])
        _, r0_rot, _ = _decompose_mat4_trs(_gltf_node_local_mat4(nodes[r0]))
        if r0 in animated_nodes or not np.allclose(r0_rot, np.eye(3), atol=1e-5):
            wrap_bname = _sanitize_ident("world_root", "bone", used_ids)
            w_items, w_cands = _emit_gltf_boxes_for_bone(
                wrap_bname, [], np.zeros(3), np.eye(3), np.ones(3)
            )
            cb0 = _build_gltf_bone(r0, np.eye(4), np.zeros(3), np.eye(3), w_cands, is_root=False)
            root_bone = Bone(name=wrap_bname, pivot_local=(0.0, 0.0, 0.0), items=[*w_items, cb0])
        else:
            root_bone = _build_gltf_bone(r0, np.eye(4), np.zeros(3), np.eye(3), (), is_root=True)
    else:
        wrap_bname = _sanitize_ident("world_root", "bone", used_ids)
        m_entries_root: list[tuple[str, int, np.ndarray]] = []
        child_roots: list[int] = []
        for r_i in root_indices:
            ri = int(r_i)
            w = _is_leaf_cube_wrapper(ri)
            if w is not None:
                m_entries_root.append((str(nodes[ri].get("name") or f"cube_{ri}"), w[0], w[1]))
            else:
                child_roots.append(ri)
        w_items, w_cands = _emit_gltf_boxes_for_bone(
            wrap_bname, m_entries_root, np.zeros(3), np.eye(3), np.ones(3)
        )
        full_root_items: list[Box | Locator | Bone] = list(w_items)
        for cr_i in child_roots:
            full_root_items.append(
                _build_gltf_bone(cr_i, np.eye(4), np.zeros(3), np.eye(3), w_cands, is_root=False)
            )
        root_bone = Bone(name=wrap_bname, pivot_local=(0.0, 0.0, 0.0), items=full_root_items)

    density = _infer_texel_density(tex_w, tex_h, uv_ratios, total_area_u, has_nonzero_box_uv=bool(uv_ratios))
    has_anims = bool(gltf.get("animations"))
    archetype = "entity" if (has_anims or len(node_idx_to_bone_info) >= 2) else "block_multipart"
    meta_node = _make_v32_meta(archetype, "bbmodel_5", density, size_step)
    prog = validate_program(_make_v32_program(meta_node, [Palette("mat_0", (0.62, 0.06, 220.0), 6)], root_bone, textures_tuple))
    compiled = compile_geometry(prog)
    max_err = _max_corner_reconstruction_error(compiled, expected_corners)

    if has_anims and _HAS_DSL_V32 and _KeyframeTrackDecl is not None:
        beh_gltf = _decompile_gltf_animations(gltf, buffers_map, prog, node_idx_to_bone_info, unit_scale)
        if beh_gltf is not None:
            prog = validate_program(replace(prog, behavior=beh_gltf))
    return prog, uv_map, embedded_rgba, max_err


def _decompile_gltf_animations(
    gltf: dict[str, Any],
    buffers_map: dict[str, bytes],
    program: Program,
    node_idx_to_bone_info: dict[int, tuple[str, np.ndarray, np.ndarray]],
    unit_scale: float,
) -> BehaviorDecl | None:
    anims = gltf.get("animations") or []
    if not anims or _KeyframeTrackDecl is None:
        return None
    compiled = compile_geometry(program)
    used_ids: set[str] = (
        set(compiled.bone_by_name)
        | set(compiled.box_by_name)
        | set(compiled.locator_by_name)
        | {p.name for p in program.palettes}
        | {t.name for t in getattr(program, "textures", ())}
    )
    clips: list[ClipDecl] = []
    for a_idx, anim in enumerate(anims):
        if not isinstance(anim, dict):
            continue
        samplers = anim.get("samplers") or []
        channels = anim.get("channels") or []
        if not samplers or not channels:
            continue
        raw_cname = str(anim.get("name") or f"clip_{a_idx}")
        cname = _sanitize_ident(f"{raw_cname}_clip" if raw_cname in used_ids else raw_cname, "clip", used_ids)

        max_t = 0.0
        raw_tracks: list[tuple[str, str, list[tuple[float, tuple[float, float, float]]]]] = []
        for ch in channels:
            tgt = ch.get("target") or {}
            n_idx = tgt.get("node")
            path = str(tgt.get("path") or "")
            s_idx = ch.get("sampler")
            if not isinstance(n_idx, int) or n_idx not in node_idx_to_bone_info or not isinstance(s_idx, int):
                continue
            if path not in ("rotation", "translation", "scale") or not (0 <= s_idx < len(samplers)):
                continue
            bname, rest_eul, rest_trans = node_idx_to_bone_info[n_idx]
            samp = samplers[s_idx]
            t_arr = _read_gltf_accessor(gltf, buffers_map, int(samp["input"])).astype(np.float64).ravel()
            v_arr = _read_gltf_accessor(gltf, buffers_map, int(samp["output"])).astype(np.float64)
            if len(t_arr) == 0 or len(v_arr) < len(t_arr):
                continue
            # Subsample ultra-dense baked 60fps timelines to <= 32 keyframes for concise DSL tokens
            if len(t_arr) > 32:
                sel = np.linspace(0, len(t_arr) - 1, 32).round().astype(int)
                sel = np.unique(sel)
                t_arr = t_arr[sel]
                v_arr = v_arr[sel]

            knots: list[tuple[float, tuple[float, float, float]]] = []
            for ti, vi in zip(t_arr, v_arr):
                tf = max(0.0, float(ti))
                if not math.isfinite(tf):
                    continue
                max_t = max(max_t, tf)
                if path == "rotation" and len(vi) == 4:
                    r_m = _quat_xyzw_to_mat3(vi)
                    tot_eul = _recover_euler_xyz_degrees(r_m)
                    d_eul = tot_eul - rest_eul
                    vec = (round(float(d_eul[0]), 4), round(float(d_eul[1]), 4), round(float(d_eul[2]), 4))
                elif path == "translation" and len(vi) == 3:
                    d_pos = vi * unit_scale - rest_trans
                    vec = (round(float(d_pos[0]), 4), round(float(d_pos[1]), 4), round(float(d_pos[2]), 4))
                elif path == "scale" and len(vi) == 3:
                    vec = (
                        round(max(1e-4, float(vi[0])), 4),
                        round(max(1e-4, float(vi[1])), 4),
                        round(max(1e-4, float(vi[2])), 4),
                    )
                else:
                    continue
                if knots and abs(tf - knots[-1][0]) <= 1e-5:
                    knots[-1] = (knots[-1][0], vec)
                else:
                    knots.append((round(tf, 5), vec))
            if not knots:
                continue
            ch_dsl = "position" if path == "translation" else path
            is_nontrivial = (
                any(abs(c - 1.0) > 1e-4 for _, v in knots for c in v)
                if ch_dsl == "scale"
                else any(abs(c) > 1e-3 for _, v in knots for c in v)
            )
            if is_nontrivial:
                raw_tracks.append((bname, ch_dsl, knots))

        if not raw_tracks:
            continue
        dur = round(float(np.clip(max_t if max_t > 0.0 else 1.0, 0.1, 60.0)), 4)
        kf_decls: list[Any] = []
        seen_bone_ch: set[tuple[str, str]] = set()
        for bname, ch_dsl, knots in raw_tracks:
            if (bname, ch_dsl) in seen_bone_ch:
                continue
            seen_bone_ch.add((bname, ch_dsl))
            clamped_knots = tuple((min(dur, max(0.0, t)), v) for t, v in knots)
            kf_decls.append(_KeyframeTrackDecl(bone=bname, channel=ch_dsl, keys=clamped_knots, interpolation="linear"))
        if kf_decls:
            clips.append(ClipDecl(name=cname, duration_s=dur, loop=True, keyframes=tuple(kf_decls)))
    return BehaviorDecl(clips=tuple(clips)) if clips else None


def ingest_gltf_archive(
    archive_bytes_or_path: bytes | Path | str,
    provenance: AssetProvenance,
    *,
    license_policy: str = "clean_room_permissive",
    top_k: int = 3,
    seed: int = 404,
) -> IngestedAssetRecord:
    """Unpack a Sketchfab ``gltf.zip`` or ``.glb`` archive and run full provenance-gated ``gltf_2`` ingestion."""
    bundle = extract_gltf_bundle_from_zip_or_glb(archive_bytes_or_path)
    return ingest_external_asset(
        bundle,
        "gltf_2",
        provenance,
        license_policy=license_policy,
        top_k=top_k,
        seed=seed,
    )


def _collect_ast_identifiers(program: Program) -> tuple[list[str], set[str]]:
    """Return (ordered_ast_box_names, all_non_palette_identifiers) for collision-free palette naming."""
    ast_boxes: list[str] = []
    other_ids: set[str] = {t.name for t in getattr(program, "textures", ())}
    if program.behavior is not None:
        for cl in program.behavior.clips:
            other_ids.add(cl.name)
    if getattr(program, "kinematics", None) is not None and program.kinematics is not None:
        for m in getattr(program.kinematics, "muscles", ()):
            other_ids.add(m.name)
        for ik in getattr(program.kinematics, "ik_chains", ()):
            other_ids.add(ik.name)
        for lk in getattr(program.kinematics, "look_chains", ()):
            other_ids.add(lk.name)

    def _walk(b: Bone) -> None:
        other_ids.add(b.name)
        if b.mirror_name:
            other_ids.add(b.mirror_name)
        if b.radial is not None:
            for k in range(1, b.radial.count):
                other_ids.add(f"{b.name}_r{k}")
                if b.mirror_name:
                    other_ids.add(f"{b.mirror_name}_r{k}")
        for it in b.items:
            if isinstance(it, Box):
                ast_boxes.append(it.name)
                other_ids.add(it.name)
                if b.mirror_name:
                    other_ids.add(f"{b.mirror_name}__{it.name}")
            elif isinstance(it, Locator):
                other_ids.add(it.name)
            elif isinstance(it, Bone):
                _walk(it)

    _walk(program.root)
    return ast_boxes, other_ids


def _infer_slot_family_and_props(
    base_spec: MaterialSpec,
    box_names: Sequence[str],
    lch: tuple[float, float, float],
    mean_alpha: float,
    domain: str,
) -> tuple[str, float, float, float, bool]:
    """Return (family, roughness, metalness, emissive_val, emissive_flag) for a clustered material slot."""
    joined = " ".join(n.lower() for n in box_names)
    l_val, c_val, _ = lch
    if mean_alpha < 210.0 or any(tok in joined for tok in ("glass", "window", "lens", "visor", "water", "slime", "ice")):
        return ("polymer_dielectric", 0.14, 0.0, 0.0, False)
    if any(tok in joined for tok in ("glow", "flame", "fire", "lamp", "lantern", "beam", "plasma", "ember", "core")) and l_val >= 0.65:
        return ("thin_film_iridescence" if c_val > 0.14 else "polymer_dielectric", 0.22, 0.0, 0.65, True)
    if any(tok in joined for tok in ("leaf", "leaves", "foliage", "frond", "vine", "moss", "bush", "grass", "hedge", "fern")):
        return ("lichen_biofilm", 0.76, 0.0, 0.0, False)
    if any(tok in joined for tok in ("wood", "plank", "trunk", "bark", "stem", "branch", "log", "timber", "shelf", "table")):
        return ("solid_wood", 0.68, 0.0, 0.0, False)
    if any(tok in joined for tok in ("stone", "rock", "brick", "pot", "cobble", "slate", "concrete", "marble")):
        return ("weathered_stone", 0.82, 0.0, 0.0, False)
    if any(tok in joined for tok in ("cloth", "fabric", "cape", "strap", "bag", "pack", "ribbon", "cushion", "pillow", "seat")):
        return ("woven_fabric", 0.80, 0.0, 0.0, False)
    if any(tok in joined for tok in ("blade", "steel", "iron", "metal", "gear", "axle", "bolt", "hinge", "brass", "gold", "chrome")):
        return ("brushed_metal", 0.30, 0.88, 0.0, False)
    if any(tok in joined for tok in ("horn", "claw", "fang", "talon", "shell", "beak", "carapace", "scale", "armor", "spike")):
        return ("keratin_scales", 0.42, 0.0, 0.0, False)
    if domain in ("dragon", "animal", "humanoid") and base_spec.family in ("brushed_metal", "painted_metal") and c_val > 0.05:
        return ("patterned_skin", 0.58, 0.0, 0.0, False)
    fam = base_spec.family if base_spec.family in MATERIAL_FAMILIES else "polymer_dielectric"
    return (
        fam,
        round(float(np.clip(base_spec.perceptual_roughness, 0.08, 0.96)), 3),
        round(float(np.clip(base_spec.metalness, 0.0, 1.0)), 3),
        0.0,
        False,
    )


def assign_multi_slot_materials_to_program(
    program: Program,
    uv_map: dict[str, dict[str, list[float]]],
    atlas_rgba: np.ndarray | None,
    base_spec: MaterialSpec,
    *,
    has_source_texture: bool = True,
    domain: str = "artificial",
    has_labpbr: bool = False,
    max_slots: int = 6,
    seed: int = 404,
) -> Program:
    """Decompose single-slot ``mat_0`` into up to ``max_slots`` part-coherent ``@palette`` + ``@materials`` slots."""
    ast_box_names, reserved_ids = _collect_ast_identifiers(program)
    if not ast_box_names:
        return program

    def _alloc_slot_name(k: int) -> str:
        cand = f"mat_{k}"
        while cand in reserved_ids:
            k += 1
            cand = f"mat_slot_{k}"
        reserved_ids.add(cand)
        return cand

    if not has_source_texture or atlas_rgba is None:
        # Clean domain-appropriate multi-slot assignment for geometry-only overrides (no vanilla Mojang PNGs)
        slot_defs: list[tuple[tuple[float, float, float], str, float, float]]
        if domain == "natural":
            slot_defs = [
                ((0.52, 0.14, 142.0), "lichen_biofilm", 0.76, 0.0),   # foliage / stems
                ((0.40, 0.09, 68.0), "solid_wood", 0.72, 0.0),        # woody stalks / roots
                ((0.68, 0.17, 35.0), "patterned_skin", 0.55, 0.0),    # blossoms / berries / petals
                ((0.48, 0.04, 240.0), "weathered_stone", 0.84, 0.0),  # pot / soil / stone
            ]
        elif domain in ("dragon", "animal", "humanoid"):
            slot_defs = [
                ((0.56, 0.11, 215.0), "patterned_skin", 0.58, 0.0),
                ((0.78, 0.06, 85.0), "keratin_scales", 0.42, 0.0),
                ((0.28, 0.08, 25.0), "polymer_dielectric", 0.25, 0.0),
            ]
        else:
            slot_defs = [
                ((0.55, 0.08, 65.0), "solid_wood", 0.68, 0.0),
                ((0.48, 0.03, 235.0), "brushed_metal", 0.34, 0.80),
                ((0.66, 0.04, 220.0), "weathered_stone", 0.82, 0.0),
            ]
        box_to_idx: dict[str, int] = {}
        used_indices: list[int] = []
        for bname in ast_box_names:
            low = bname.lower()
            if domain == "natural":
                if any(t in low for t in ("stem", "stalk", "trunk", "wood", "branch", "root", "bark")):
                    idx_s = 1
                elif any(t in low for t in ("flower", "petal", "bloom", "berry", "cap", "spore", "bud")):
                    idx_s = 2
                elif any(t in low for t in ("pot", "soil", "dirt", "stone", "base")):
                    idx_s = 3
                else:
                    idx_s = 0
            else:
                idx_s = 0
                if any(t in low for t in ("horn", "claw", "tooth", "eye", "metal", "iron", "handle", "trim", "wheel", "leg")):
                    idx_s = 1
                elif any(t in low for t in ("stone", "base", "glass", "top")):
                    idx_s = min(2, len(slot_defs) - 1)
            box_to_idx[bname] = idx_s
            if idx_s not in used_indices:
                used_indices.append(idx_s)
        used_indices.sort()
        idx_to_mat: dict[int, str] = {}
        new_pals: list[Palette] = []
        new_mats: list[MaterialDecl] = []
        for out_k, s_idx in enumerate(used_indices):
            mname = _alloc_slot_name(out_k)
            idx_to_mat[s_idx] = mname
            lch_s, fam_s, rgh_s, met_s = slot_defs[s_idx]
            new_pals.append(Palette(mname, lch_s, 6))
            new_mats.append(MaterialDecl(name=mname, family=fam_s, roughness=rgh_s, metalness=met_s, seed=seed + out_k * 13))
        ast_box_to_mat = {bn: idx_to_mat[box_to_idx[bn]] for bn in ast_box_names}
    else:
        scene = compile_geometry(program)
        ast_set = set(ast_box_names)
        wb_by_ast: dict[str, list[Any]] = {bn: [] for bn in ast_box_names}
        for wb in scene.boxes:
            if wb.name in ast_set:
                wb_by_ast[wb.name].append(wb)
            elif "__" in wb.name:
                tail = wb.name.split("__", 1)[1]
                if tail in ast_set:
                    wb_by_ast[tail].append(wb)

        h_img, w_img = atlas_rgba.shape[:2]
        box_lab: np.ndarray = np.zeros((len(ast_box_names), 3), dtype=np.float64)
        box_alpha: np.ndarray = np.full(len(ast_box_names), 255.0, dtype=np.float64)
        box_weight: np.ndarray = np.ones(len(ast_box_names), dtype=np.float64)
        valid_mask = np.zeros(len(ast_box_names), dtype=bool)

        for i_b, bn in enumerate(ast_box_names):
            wbs = wb_by_ast.get(bn) or []
            rgba_samples: list[np.ndarray] = []
            area_b = 0.0
            for wb in wbs:
                he = wb.half_extents
                area_b += float(8.0 * (he[0] * he[1] + he[1] * he[2] + he[2] * he[0]))
                f_uvs = uv_map.get(wb.name) or uv_map.get(bn) or {}
                for f in FACES:
                    rect = f_uvs.get(f)
                    if not rect or len(rect) < 4:
                        continue
                    x0 = int(np.clip(math.floor(min(rect[0], rect[2])), 0, w_img - 1))
                    x1 = int(np.clip(math.ceil(max(rect[0], rect[2])), x0 + 1, w_img))
                    y0 = int(np.clip(math.floor(min(rect[1], rect[3])), 0, h_img - 1))
                    y1 = int(np.clip(math.ceil(max(rect[1], rect[3])), y0 + 1, h_img))
                    crop = atlas_rgba[y0:y1, x0:x1].reshape(-1, 4)
                    vis = crop[crop[:, 3] >= 20]
                    if len(vis) > 0:
                        if len(vis) > 128:
                            step_s = max(1, len(vis) // 128)
                            vis = vis[::step_s]
                        rgba_samples.append(vis)
            box_weight[i_b] = max(0.25, area_b)
            if rgba_samples:
                cat = np.concatenate(rgba_samples, axis=0)
                lin_rgb = srgb_u8_to_linear(cat[:, :3])
                lab_pts = linear_srgb_to_oklab(lin_rgb)
                box_lab[i_b] = np.median(lab_pts, axis=0)
                box_alpha[i_b] = float(np.mean(cat[:, 3]))
                valid_mask[i_b] = True

        if not np.any(valid_mask):
            all_vis = atlas_rgba.reshape(-1, 4)
            all_vis = all_vis[all_vis[:, 3] >= 20]
            fallback_lab = (
                np.median(linear_srgb_to_oklab(srgb_u8_to_linear(all_vis[:: max(1, len(all_vis) // 512), :3])), axis=0)
                if len(all_vis) > 0
                else np.array([0.62, 0.0, 0.0], dtype=np.float64)
            )
            box_lab[:] = fallback_lab[None, :]
        elif not np.all(valid_mask):
            med_valid = np.median(box_lab[valid_mask], axis=0)
            box_lab[~valid_mask] = med_valid[None, :]

        # Chromatic-weighted OKLab clustering (separates distinct body/belly/eye/leaf/wood parts cleanly)
        w_feat = box_lab * np.array([1.0, 1.45, 1.45], dtype=np.float64)[None, :]
        k_cap = max(1, min(int(max_slots), len(ast_box_names)))
        first_idx = int(np.argmax(box_weight))
        centers: list[np.ndarray] = [w_feat[first_idx].copy()]
        min_sep_oklab = 0.055

        for _ in range(1, k_cap):
            d_mat = np.stack([np.linalg.norm(w_feat - c[None, :], axis=1) for c in centers], axis=1)
            min_d = np.min(d_mat, axis=1)
            # Slightly favor larger-area parts among clearly distinct color candidates
            cand_score = np.where(min_d >= min_sep_oklab, min_d * (1.0 + 0.12 * np.log1p(box_weight)), -1.0)
            best_i = int(np.argmax(cand_score))
            if cand_score[best_i] < 0.0:
                break
            centers.append(w_feat[best_i].copy())

        centers_arr = np.stack(centers, axis=0)
        for _ in range(8):
            dists = np.linalg.norm(w_feat[:, None, :] - centers_arr[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for c_i in range(len(centers_arr)):
                m_c = labels == c_i
                if np.any(m_c):
                    w_c = box_weight[m_c]
                    centers_arr[c_i] = np.average(w_feat[m_c], axis=0, weights=w_c)

        dists = np.linalg.norm(w_feat[:, None, :] - centers_arr[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Order active clusters by descending total part area so mat_0 is primary body material
        active_clusters = sorted(
            [c_i for c_i in range(len(centers_arr)) if np.any(labels == c_i)],
            key=lambda c_i: -float(np.sum(box_weight[labels == c_i])),
        )
        cluster_to_mat: dict[int, str] = {}
        new_pals = []
        new_mats = []
        for out_k, c_i in enumerate(active_clusters):
            mname = _alloc_slot_name(out_k)
            cluster_to_mat[c_i] = mname
            m_c = labels == c_i
            lab_c = np.average(box_lab[m_c], axis=0, weights=box_weight[m_c])
            l_v = round(float(np.clip(lab_c[0], 0.12, 0.95)), 4)
            c_v = round(float(np.clip(math.hypot(float(lab_c[1]), float(lab_c[2])), 0.0, 0.36)), 4)
            h_v = round(float(math.degrees(math.atan2(float(lab_c[2]), float(lab_c[1]))) % 360.0), 2) if c_v >= 1e-3 else 0.0
            bnames_c = [ast_box_names[idx_b] for idx_b in np.nonzero(m_c)[0]]
            alpha_c = float(np.mean(box_alpha[m_c]))
            fam_c, rgh_c, met_c, em_v, em_flag = _infer_slot_family_and_props(
                base_spec, bnames_c, (l_v, c_v, h_v), alpha_c, domain
            )
            steps_c = int(np.clip(getattr(base_spec, "ramp_steps", 6) if hasattr(base_spec, "ramp_steps") else 6, 4, 8))
            new_pals.append(Palette(mname, (l_v, c_v, h_v), steps_c, emissive=em_flag))
            new_mats.append(
                MaterialDecl(
                    name=mname,
                    family=fam_c,
                    roughness=rgh_c,
                    metalness=met_c,
                    coating=round(float(np.clip(base_spec.coating_thickness, 0.0, 1.0)), 3),
                    subsurface=round(float(np.clip(base_spec.subsurface, 0.0, 1.0)), 3),
                    emissive=em_v,
                    seed=int(seed + out_k * 17),
                )
            )
        ast_box_to_mat = {
            ast_box_names[i_b]: cluster_to_mat[int(labels[i_b])]
            for i_b in range(len(ast_box_names))
        }

    def _rewrite_bone_mats(b: Bone) -> Bone:
        new_items: list[Box | Locator | Bone] = []
        for it in b.items:
            if isinstance(it, Box):
                new_items.append(replace(it, material=ast_box_to_mat.get(it.name, new_pals[0].name)))
            elif isinstance(it, Bone):
                new_items.append(_rewrite_bone_mats(it))
            else:
                new_items.append(it)
        return replace(b, items=new_items)

    new_root = _rewrite_bone_mats(program.root)
    style_node = StyleDecl(
        target_encoding="labpbr_1_3" if has_labpbr else "vanilla",
        quantize_ramp=True,
        bevel_highlight=0.06,
        view_deg=20.0,
    )
    return validate_program(
        replace(
            program,
            palettes=new_pals,
            root=new_root,
            materials=tuple(new_mats),
            style=style_node,
        )
    )


def ingest_external_asset(
    payload: dict[str, Any],
    format_kind: str,
    provenance: AssetProvenance,
    *,
    atlas_png: bytes | None = None,
    java_textures_png: dict[str, bytes] | None = None,
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

    external_rgba = _decode_png_rgba(atlas_png) if atlas_png is not None else None
    java_rgba_map: dict[str, np.ndarray] | None = None
    if java_textures_png:
        decoded_j = {str(k): _decode_png_rgba(v) for k, v in java_textures_png.items() if v}
        if decoded_j:
            java_rgba_map = decoded_j

    embedded_rgba: np.ndarray | None = None
    raw_to_safe_bone: dict[str, str] | None = None
    if format_kind == "bbmodel":
        program, uv_map, embedded_rgba, vert_err = _decompile_bbmodel(
            payload, external_atlas_rgba=external_rgba
        )
    elif format_kind == "java_json":
        res_j = _decompile_java_json(
            payload,
            external_atlas_rgba=external_rgba,
            java_textures_rgba=java_rgba_map,
            return_rgba=True,
        )
        assert len(res_j) == 4
        program, uv_map, embedded_rgba, vert_err = res_j
    elif format_kind == "bedrock_geojson":
        program, uv_map, vert_err, raw_to_safe_bone = _decompile_bedrock_geojson(payload)
    elif format_kind == "gltf_2":
        program, uv_map, embedded_rgba, vert_err = _decompile_gltf(payload)
    else:
        raise ValueError(
            f"Unsupported external format_kind {format_kind!r}; "
            "expected 'bbmodel', 'java_json', 'bedrock_geojson', or 'gltf_2'"
        )

    if animation_payload is not None:
        beh = decompile_bedrock_animation_to_behavior(
            animation_payload,
            program,
            raw_to_safe_bone=raw_to_safe_bone,
        )
        program = validate_program(replace(program, behavior=beh))

    if embedded_rgba is not None:
        atlas_rgba = embedded_rgba
        has_source_texture = True
    elif external_rgba is not None:
        atlas_rgba = external_rgba
        has_source_texture = True
    else:
        atlas_rgba = np.full((16, 16, 4), (160, 160, 165, 255), dtype=np.uint8)
        has_source_texture = False

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
        has_source_texture=has_source_texture,
        resolved_atlas_rgba=atlas_rgba if has_source_texture else None,
        resolved_uv_map=uv_map,
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
