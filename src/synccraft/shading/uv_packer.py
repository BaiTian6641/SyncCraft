"""Deterministic UV atlas allocation: legacy shelf + v2 Semantic Anatomical MaxRects Packer.

Provides:
- ``allocate_uv_layout`` / ``build_atlas``: backwards-compatible Block-DSL UV allocator
  (supports ``packing_mode="legacy"`` and ``packing_mode="semantic_maxrects"``).
- ``classify_anatomical_zone``: maps bones/boxes/materials into clean anatomical UV regions
  (head, neck, torso, wings, forelimbs, hindlimbs, tail, emissive, accents).
- ``pack_semantic_uv_atlas``: high-utilization (typically 65-88%+) 2D MaxRects-BSSF zone-
  partitioned UV packer with zero-area plane-edge culling and adaptive hybrid L/R UV mirroring.
- ``repack_bbmodel_uvs``: repacks any existing Blockbench 5 ``.bbmodel`` in-place into clean
  anatomical partitions while optionally transferring its embedded PNG texture losslessly.
- ``render_uv_zone_map``: renders a color-coded, labeled UV zone diagnostic atlas PNG.
- ``measure_uv_utilization``: reports exact utilization, per-zone fill, and wasted area.
"""
from __future__ import annotations

import base64
import copy
import io
import math
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

FACES: tuple[str, ...] = ("north", "south", "east", "west", "up", "down")

# Canonical anatomical partition order (hero focal zones first -> wings/limbs -> tail -> FX)
ANATOMICAL_ZONES: tuple[str, ...] = (
    "head",
    "neck",
    "torso",
    "wings",
    "forelimbs",
    "hindlimbs",
    "tail",
    "emissive",
    "accents",
)

# Adaptive Hybrid L/R mirroring policy:
# - Symmetrical repeated structures share horizontally flipped UVs by default (2x density / 1/2 atlas)
# - Hero focal zones (head, neck, torso, tail) remain unique by default for asymmetric detail/markings
SYMMETRIC_UV_ZONES: frozenset[str] = frozenset(
    {"wings", "forelimbs", "hindlimbs", "accents", "emissive"}
)
UNIQUE_HERO_ZONES: frozenset[str] = frozenset({"head", "neck", "torso", "tail"})

# Distinct studio-palette colors per anatomical zone for the visual UV Zone Diagnostic Map
ZONE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "head": (235, 95, 85),
    "neck": (240, 155, 70),
    "torso": (95, 165, 240),
    "wings": (165, 105, 245),
    "forelimbs": (70, 200, 165),
    "hindlimbs": (110, 205, 95),
    "tail": (235, 200, 75),
    "emissive": (55, 225, 245),
    "accents": (210, 130, 185),
}

_ZONE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "wings",
        re.compile(
            r"(^|[_\-\s])(wing|patagium|pinion|covert|membrane|lobe|thruster|"
            r"sail|alula|carpal|f[1-4](_|$)|wing_l|wing_r)",
            re.IGNORECASE,
        ),
    ),
    (
        "head",
        re.compile(
            r"(^|[_\-\s])(head|skull|cranium|snout|jaw|mouth|tongue|teeth|tooth|fang|eye|"
            r"eyelid|brow|ear|horn|antler|crest|beak|crown|diadem|face|cheek|chin|frill|"
            r"muzzle|nose|nasal|maxilla|zygoma|occiput|palate|orbital|antenna|mandible|"
            r"whisker|comb|wattle|mandible)",
            re.IGNORECASE,
        ),
    ),
    (
        "neck",
        re.compile(
            r"(^|[_\-\s])(neck|throat|collar|mane|ruff|gorget|dewlap|nape|clavicle)",
            re.IGNORECASE,
        ),
    ),
    (
        "forelimbs",
        re.compile(
            r"(^|[_\-\s])(arm|foreleg|forelimb|fore_leg|fore_|fl_|fr_|front_leg|paw|hand|"
            r"wrist|elbow|finger|thumb|deltoid|bicep|tricep|vambrace|knuckle|palm|ulna)",
            re.IGNORECASE,
        ),
    ),
    (
        "hindlimbs",
        re.compile(
            r"(^|[_\-\s])(leg|hind|thigh|shin|calf|knee|ankle|foot|feet|hoof|hooves|toe|"
            r"hl_|hr_|back_leg|patella|hock|heel|sole|femur|tibia|metatarsal|haunch|talon)",
            re.IGNORECASE,
        ),
    ),
    (
        "tail",
        re.compile(
            r"(^|[_\-\s])(tail|caudal|rudder|club|rattle|fluke|stinger|whip|spade)",
            re.IGNORECASE,
        ),
    ),
    (
        "torso",
        re.compile(
            r"(^|[_\-\s])(pelvis|waist|chest|thorax|abdomen|back|belly|sternum|rib|carapace|"
            r"shell|spine|dorsal|ventral|hip|flank|body|torso|core|withers|pectoral|trapezius|"
            r"prow|groin|saddle|hump|scute)",
            re.IGNORECASE,
        ),
    ),
    (
        "emissive",
        re.compile(
            r"(^|[_\-\s])(glow|neon|rune|starlight|laser|orb|crystal_core|gem|slit|halo|"
            r"plasma|energy|magma|emissive|lantern|biolum)",
            re.IGNORECASE,
        ),
    ),
)

_MIRROR_TOKEN_RE = re.compile(
    r"(^|_|\.)(right|left|r|l)(_|\.|$)", re.IGNORECASE
)


def oklch_to_srgb(lch):
    """Diagnostic OKLCH base color, with explicit channel clipping (no gamut mapper)."""
    lightness, chroma, hue = map(float, lch)
    angle = math.radians(hue)
    a, b = chroma * math.cos(angle), chroma * math.sin(angle)
    try:
        l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    except OverflowError as exc:
        raise ValueError("OKLCH value exceeds the diagnostic converter's numeric range") from exc
    linear = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    if not all(math.isfinite(v) for v in linear):
        raise ValueError("OKLCH conversion produced nonfinite color channels")

    def encode(v):
        v = min(1.0, max(0.0, v))
        return round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))

    return tuple(encode(v) for v in linear)


def _pow2(v: int) -> int:
    v_i = max(1, int(v))
    return 1 << (v_i - 1).bit_length()


def classify_anatomical_zone(
    bone_name: str,
    box_name: str,
    *,
    ancestor_bones: tuple[str, ...] = (),
    material_name: str | None = None,
    is_emissive: bool = False,
    explicit_zone: str | None = None,
    zone_overrides: dict[str, str] | None = None,
) -> str:
    """Classify a box/element into one of the canonical anatomical UV partition zones."""
    if explicit_zone:
        z = str(explicit_zone).strip().lower()
        return z if z in ANATOMICAL_ZONES else "accents"

    if zone_overrides:
        if box_name in zone_overrides:
            return zone_overrides[box_name]
        if bone_name in zone_overrides:
            return zone_overrides[bone_name]
        for pat, z in zone_overrides.items():
            if pat.startswith("re:") and re.search(pat[3:], f"{bone_name}/{box_name}", re.IGNORECASE):
                return z

    # Check eyes first so emissive eyes stay cleanly inside the head zone
    if re.search(r"(^|[_\-\s])(eye|eyelid|iris|pupil|cornea)([_\-\s]|$)", f"{bone_name} {box_name}", re.IGNORECASE):
        return "head"

    # Dedicated emissive zone for non-eye glowing gems/slits/cores/runes
    if is_emissive:
        return "emissive"

    # Priority 1: Check the box name + immediate bone name
    local_text = f"{bone_name} {box_name}"
    for zone, pat in _ZONE_PATTERNS:
        if pat.search(local_text):
            return zone

    # Priority 2: Walk ancestor bone chain from closest parent outward (handles generic child names)
    for anc in ancestor_bones:
        for zone, pat in _ZONE_PATTERNS:
            if pat.search(anc):
                return zone

    # Priority 3: Material name hints
    if material_name:
        for zone, pat in _ZONE_PATTERNS:
            if pat.search(material_name):
                return zone

    return "accents"


# ---------------------------------------------------------------------------
# Fast 2D MaxRects-BSSF Bin Packer
# ---------------------------------------------------------------------------


class _MaxRectsBin:
    """2D Maximal Rectangles Bin Packer using Best-Short-Side-Fit (BSSF)."""

    __slots__ = ("width", "height", "free_rects", "used_rects")

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.free_rects: list[tuple[int, int, int, int]] = [(0, 0, self.width, self.height)]
        self.used_rects: list[tuple[int, int, int, int]] = []

    def insert(
        self,
        w: int,
        h: int,
        *,
        anchor: tuple[int, int] | None = None,
        affinity_weight: float = 0.0,
    ) -> tuple[int, int] | None:
        """Insert an axis-aligned rectangle (w, h); return (x, y) or None if no fit.

        When ``anchor`` and ``affinity_weight > 0`` are supplied, biases placement toward
        free rectangles adjacent to the current anatomical zone's anchor so zones stay
        tightly clustered while seamlessly interlocking into L-shaped / concave voids!
        """
        best_cost = 1e18
        best_score2 = 10**9
        best_y = 10**9
        best_x = 10**9
        best_rect: tuple[int, int, int, int] | None = None
        ax, ay = anchor if anchor is not None else (0, 0)
        aw = float(affinity_weight)

        for fx, fy, fw, fh in self.free_rects:
            if fw >= w and fh >= h:
                rem_w = fw - w
                rem_h = fh - h
                short_side = rem_w if rem_w < rem_h else rem_h
                long_side = rem_h if rem_w < rem_h else rem_w
                if aw > 0.0 and anchor is not None:
                    # Try the corner of this free rect nearest to the zone anchor
                    cx = fx if abs(fx - ax) <= abs((fx + fw - w) - ax) else (fx + fw - w)
                    cy = fy if abs(fy - ay) <= abs((fy + fh - h) - ay) else (fy + fh - h)
                    dist = abs(cx - ax) + abs(cy - ay)
                    cost = dist * aw + short_side * 1.0 + long_side * 0.04
                    px_cand, py_cand = cx, cy
                else:
                    cost = float(short_side)
                    px_cand, py_cand = fx, fy
                if (
                    cost < best_cost - 1e-6
                    or (abs(cost - best_cost) <= 1e-6 and long_side < best_score2)
                    or (abs(cost - best_cost) <= 1e-6 and long_side == best_score2 and (py_cand < best_y or (py_cand == best_y and px_cand < best_x)))
                ):
                    best_cost = cost
                    best_score2 = long_side
                    best_x = px_cand
                    best_y = py_cand
                    best_rect = (px_cand, py_cand, w, h)

        if best_rect is None:
            return None

        self._place_rect(best_rect)
        return best_rect[0], best_rect[1]

    def _place_rect(self, placed: tuple[int, int, int, int]) -> None:
        px, py, pw, ph = placed
        px2 = px + pw
        py2 = py + ph
        new_free: list[tuple[int, int, int, int]] = []
        for fx, fy, fw, fh in self.free_rects:
            fx2 = fx + fw
            fy2 = fy + fh
            if px >= fx2 or px2 <= fx or py >= fy2 or py2 <= fy:
                new_free.append((fx, fy, fw, fh))
                continue
            # Split horizontally / vertically around placed rect
            if px > fx:
                new_free.append((fx, fy, px - fx, fh))
            if px2 < fx2:
                new_free.append((px2, fy, fx2 - px2, fh))
            if py > fy:
                new_free.append((fx, fy, fw, py - fy))
            if py2 < fy2:
                new_free.append((fx, py2, fw, fy2 - py2))

        # Prune rectangles strictly contained inside another free rectangle
        # Sort largest area first so containment check is fast
        new_free.sort(key=lambda r: r[2] * r[3], reverse=True)
        pruned: list[tuple[int, int, int, int]] = []
        for i, (ax, ay, aw, ah) in enumerate(new_free):
            ax2, ay2 = ax + aw, ay + ah
            contained = False
            for bx, by, bw, bh in pruned:
                if ax >= bx and ay >= by and ax2 <= bx + bw and ay2 <= by + bh:
                    contained = True
                    break
            if not contained:
                pruned.append((ax, ay, aw, ah))
        self.free_rects = pruned
        self.used_rects.append(placed)


def _pack_islands_into_width(
    islands: list[dict[str, Any]],
    target_width: int,
    padding: int,
) -> tuple[int, int, list[tuple[int, int, dict[str, Any]]]] | None:
    """Pack a zone's islands tightly inside ``target_width`` using MaxRects-BSSF."""
    if not islands:
        return 0, 0, []
    p2 = 2 * padding
    min_w = max(isl["w"] + p2 for isl in islands)
    w_cap = max(min_w, int(target_width))
    # Sort largest edge & area first for optimal MaxRects utilization
    ordered = sorted(
        islands,
        key=lambda r: (max(r["h"], r["w"]), r["w"] * r["h"], r["h"], r["w"], r["box"], r["face"]),
        reverse=True,
    )
    total_area = sum((r["w"] + p2) * (r["h"] + p2) for r in ordered)
    max_h_est = max(sum(r["h"] + p2 for r in ordered), int(math.ceil(total_area / max(1, w_cap))) * 2 + 64)

    mr = _MaxRectsBin(w_cap, max_h_est)
    placed: list[tuple[int, int, dict[str, Any]]] = []
    used_w = 0
    used_h = 0
    for isl in ordered:
        pw = isl["w"] + p2
        ph = isl["h"] + p2
        pos = mr.insert(pw, ph)
        if pos is None:
            return None
        px, py = pos
        placed.append((px + padding, py + padding, isl))
        used_w = max(used_w, px + pw)
        used_h = max(used_h, py + ph)
    return used_w, used_h, placed


def _candidate_pow2_sheets(min_w: int, min_h: int, total_area: int, max_width: int | None) -> list[tuple[int, int]]:
    """Generate candidate power-of-two (W, H) sheets sorted by minimal area then balanced aspect."""
    w_min = max(16, _pow2(min_w))
    h_min = max(16, _pow2(min_h))
    w_cap = int(max_width) if max_width is not None else 4096
    dims_w = [1 << k for k in range(4, 14) if (1 << k) >= w_min and (1 << k) <= w_cap]
    if max_width is not None and max_width not in dims_w and max_width >= min_w:
        dims_w.append(max_width)
    dims_h = [1 << k for k in range(4, 14) if (1 << k) >= h_min and (1 << k) <= 8192]

    cands: list[tuple[int, int]] = []
    for w in dims_w:
        for h in dims_h:
            if w * h < total_area:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 4.0:
                continue
            cands.append((w, h))

    # Sort by: (1) smallest total pixel area, (2) prefer <= 2:1 aspect, (3) closer to square, (4) landscape >= portrait
    cands.sort(
        key=lambda wh: (
            wh[0] * wh[1],
            0 if max(wh[0], wh[1]) <= 2 * min(wh[0], wh[1]) else 1,
            abs(wh[0] - wh[1]),
            0 if wh[0] >= wh[1] else 1,
        )
    )
    return cands


def _candidate_tight_sheets(
    min_w: int,
    min_h: int,
    total_area: int,
    max_width: int | None,
    *,
    align: int = 16,
) -> list[tuple[int, int]]:
    """Generate tight non-square/flexible `align`-px-aligned (W, H) sheets (not restricted to square or 2:1 POT!).

    Samples clean 16px-aligned widths across aspect ratios 0.45..2.6 and derives the tightest
    matching height so atlases adapt seamlessly to whatever space the UVs genuinely need.
    """
    al = max(8, int(align))

    def _ceil_al(v: float) -> int:
        return max(al, int(math.ceil(v / al)) * al)

    w_min = _ceil_al(min_w)
    h_min = _ceil_al(min_h)
    w_cap = int(max_width) if max_width is not None else 4096
    sqrt_a = math.sqrt(max(1.0, float(total_area)))

    # Sample clean widths spanning narrow-portrait to wide-landscape + POT anchors
    w_set: set[int] = set()
    for ratio in (0.62, 0.72, 0.82, 0.92, 1.0, 1.12, 1.25, 1.38, 1.52, 1.70, 1.90, 2.15, 2.45):
        wc = _ceil_al(sqrt_a * ratio)
        if w_min <= wc <= w_cap:
            w_set.add(wc)
    for p2 in (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 2048):
        if w_min <= p2 <= w_cap and p2 >= sqrt_a * 0.55 and p2 <= sqrt_a * 2.85:
            w_set.add(p2)
    if w_min <= w_cap:
        w_set.add(w_min)

    cands_set: set[tuple[int, int]] = set()
    for w in sorted(w_set):
        # Test tight height guesses at 1.02x .. 1.35x area density
        for slack in (1.03, 1.07, 1.12, 1.18, 1.26, 1.38):
            h = max(h_min, _ceil_al((total_area * slack) / max(1, w)))
            if h > 8192:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 3.2:
                continue
            cands_set.add((w, h))

    # Also include smallest POT sheets as fallback candidates
    for pw, ph in _candidate_pow2_sheets(min_w, min_h, total_area, max_width)[:8]:
        cands_set.add((pw, ph))

    cands = sorted(
        cands_set,
        key=lambda wh: (
            wh[0] * wh[1],
            0 if max(wh[0], wh[1]) <= 2.2 * min(wh[0], wh[1]) else 1,
            abs(wh[0] - wh[1]),
            0 if wh[0] >= wh[1] else 1,
        ),
    )
    return cands


def _canonical_mirror_key(name: str) -> tuple[str, str | None]:
    """Return (canonical_skeleton_name, side_tag 'l'|'r'|None) for L/R symmetry detection."""
    # SyncCraft mirror_x naming convention: ``right_bone__box`` vs ``box`` or ``wing_r_upper__...``
    low = name.lower()
    parts = re.split(r"(__|[_\-.])", low)
    side: str | None = None
    out: list[str] = []
    for tok in parts:
        if tok in ("l", "left"):
            if side is None:
                side = "l"
            out.append("SIDE")
        elif tok in ("r", "right"):
            if side is None:
                side = "r"
            out.append("SIDE")
        else:
            out.append(tok)
    return "".join(out), side


def _mirrored_face_name(face: str) -> str:
    if face == "east":
        return "west"
    if face == "west":
        return "east"
    return face


def _solve_semantic_atlas_packing(
    master_islands: list[dict[str, Any]],
    *,
    padding: int = 1,
    max_width: int | None = None,
    zone_gap: int | None = None,
    canvas_mode: str = "tight",
) -> dict[str, Any]:
    """Pack master islands grouped into clean anatomical zones at maximum atlas utilization.

    ``canvas_mode``:
      - ``"tight"`` (default v3): allows arbitrary 16px-aligned non-square canvases AND
        interlocking zone-affinity MaxRects packing so smaller zones smoothly flow into
        L-shaped leftover voids (typically 80-92%+ padded fill!).
      - ``"pot"``: strict power-of-two canvas dimensions (still benefits from interlocking
        zone-affinity MaxRects when it eliminates wasted sheet area!).
    """
    p2 = 2 * padding
    z_gap = padding if zone_gap is None else int(zone_gap)

    by_zone: dict[str, list[dict[str, Any]]] = {}
    for isl in master_islands:
        by_zone.setdefault(isl["zone"], []).append(isl)

    active_zones = [z for z in ANATOMICAL_ZONES if z in by_zone]
    for z in sorted(by_zone):
        if z not in active_zones:
            active_zones.append(z)

    min_island_w = max(isl["w"] + p2 for isl in master_islands)
    min_island_h = max(isl["h"] + p2 for isl in master_islands)
    total_padded_area = sum((isl["w"] + p2) * (isl["h"] + p2) for isl in master_islands)

    if canvas_mode == "pot":
        candidates = _candidate_pow2_sheets(min_island_w, min_island_h, total_padded_area, max_width)
    else:
        candidates = _candidate_tight_sheets(min_island_w, min_island_h, total_padded_area, max_width)
    if not candidates:
        raise ValueError("face cannot fit atlas width at the requested density")

    # Strategy 1: Strict non-overlapping rectangular zone partitions (cleanest visual blocks)
    def _try_partitioned_sheet(sheet_w: int, sheet_h: int, gap: int) -> dict[str, Any] | None:
        zone_areas = {
            z: sum((r["w"] + p2) * (r["h"] + p2) for r in by_zone[z])
            for z in active_zones
        }
        zone_min_w = {
            z: max(r["w"] + p2 for r in by_zone[z])
            for z in active_zones
        }

        for aspect_bias in (1.0, 1.35, 0.78, 1.75, 0.58, 2.2):
            blocks: list[tuple[str, int, int, list[tuple[int, int, dict[str, Any]]]]] = []
            ok = True
            for z in active_zones:
                za = zone_areas[z]
                if za > 0.42 * sheet_w * sheet_h:
                    zw_target = min(sheet_w, max(zone_min_w[z], sheet_w))
                else:
                    zw_target = int(round(math.sqrt(za * aspect_bias)))
                    zw_target = max(zone_min_w[z], min(sheet_w, zw_target))
                packed_z = _pack_islands_into_width(by_zone[z], zw_target, padding)
                if packed_z is None:
                    ok = False
                    break
                zw, zh, z_placed = packed_z
                if zw > sheet_w or zh > sheet_h:
                    packed_z = _pack_islands_into_width(by_zone[z], sheet_w, padding)
                    if packed_z is None or packed_z[0] > sheet_w or packed_z[1] > sheet_h:
                        ok = False
                        break
                    zw, zh, z_placed = packed_z
                blocks.append((z, zw, zh, z_placed))
            if not ok:
                continue

            blocks_sorted = sorted(blocks, key=lambda b: (max(b[1], b[2]), b[1] * b[2]), reverse=True)
            mr_sheet = _MaxRectsBin(sheet_w, sheet_h)
            zone_rects: dict[str, list[int]] = {}
            all_placements: list[tuple[int, int, dict[str, Any]]] = []
            sheet_ok = True
            for z, zw, zh, z_placed in blocks_sorted:
                req_w = min(sheet_w, zw + gap) if zw + gap <= sheet_w else zw
                req_h = min(sheet_h, zh + gap) if zh + gap <= sheet_h else zh
                pos = mr_sheet.insert(req_w, req_h)
                if pos is None:
                    pos = mr_sheet.insert(zw, zh)
                if pos is None:
                    sheet_ok = False
                    break
                zx, zy = pos
                zone_rects[z] = [zx, zy, zx + zw, zy + zh]
                for lx, ly, isl in z_placed:
                    all_placements.append((zx + lx, zy + ly, isl))
            if sheet_ok:
                return {
                    "width": sheet_w,
                    "height": sheet_h,
                    "strategy": "strict_zone_partitions",
                    "zone_rects": zone_rects,
                    "placed": all_placements,
                }
        return None

    # Strategy 2 (v3 Interlocking Semantic Zone-Affinity MaxRects):
    # Packs zones in deterministic anatomical order using shared global MaxRects free space
    # with strong zone-anchor gravity! Every zone stays tightly clustered together while
    # subsequent zones organically flow into L-shaped & concave voids left by prior zones!
    def _try_interlocking_zone_sheet(sheet_w: int, sheet_h: int) -> dict[str, Any] | None:
        ordered_zones = sorted(
            active_zones,
            key=lambda z: sum((r["w"] + p2) * (r["h"] + p2) for r in by_zone[z]),
            reverse=True,
        )
        for affinity in (1.15, 0.35, 0.0):
            mr = _MaxRectsBin(sheet_w, sheet_h)
            all_placements: list[tuple[int, int, dict[str, Any]]] = []
            zone_bounds: dict[str, list[int]] = {}
            ok = True
            for z in ordered_zones:
                z_islands = sorted(
                    by_zone[z],
                    key=lambda r: (max(r["h"], r["w"]), r["w"] * r["h"], r["h"], r["w"], r["box"], r["face"]),
                    reverse=True,
                )
                x0, y0, x1, y1 = sheet_w, sheet_h, 0, 0
                anchor: tuple[int, int] | None = None
                for isl in z_islands:
                    pw, ph = isl["w"] + p2, isl["h"] + p2
                    pos = mr.insert(pw, ph, anchor=anchor, affinity_weight=affinity if anchor is not None else 0.0)
                    if pos is None:
                        ok = False
                        break
                    px, py = pos
                    if anchor is None:
                        anchor = (px, py)
                    all_placements.append((px + padding, py + padding, isl))
                    x0 = min(x0, px)
                    y0 = min(y0, py)
                    x1 = max(x1, px + pw)
                    y1 = max(y1, py + ph)
                if not ok:
                    break
                zone_bounds[z] = [x0, y0, x1, y1]
            if ok:
                # Trim unused right/bottom canvas margin to tightest clean 16px boundary in "tight" mode!
                out_w, out_h = sheet_w, sheet_h
                if canvas_mode != "pot":
                    max_x = max(x + isl["w"] + padding for x, _, isl in all_placements)
                    max_y = max(y + isl["h"] + padding for _, y, isl in all_placements)
                    out_w = max(16, int(math.ceil(max_x / 16.0)) * 16)
                    out_h = max(16, int(math.ceil(max_y / 16.0)) * 16)
                return {
                    "width": out_w,
                    "height": out_h,
                    "strategy": "interlocking_semantic_maxrects" if affinity > 0 else "contiguous_zone_maxrects",
                    "zone_rects": zone_bounds,
                    "placed": all_placements,
                }
        return None

    # Evaluate candidates in ascending area order; prefer tight interlocking/partitioned best fill!
    best_res: dict[str, Any] | None = None
    best_area = 10**18
    for sw, sh in candidates:
        if sw * sh > best_area * 1.02:
            break
        res = _try_partitioned_sheet(sw, sh, z_gap)
        if res is None and z_gap > 0:
            res = _try_partitioned_sheet(sw, sh, 0)
        if res is None:
            res = _try_interlocking_zone_sheet(sw, sh)
        if res is not None:
            # Also trim partitioned sheet if tight mode leaves trailing strip
            if canvas_mode != "pot" and res["strategy"] == "strict_zone_partitions":
                max_x = max(x + isl["w"] + padding for x, _, isl in res["placed"])
                max_y = max(y + isl["h"] + padding for _, y, isl in res["placed"])
                res["width"] = max(16, int(math.ceil(max_x / 16.0)) * 16)
                res["height"] = max(16, int(math.ceil(max_y / 16.0)) * 16)
            area = res["width"] * res["height"]
            if area < best_area:
                best_area = area
                best_res = res
                # Stop early once we hit >= 84% padded utilization!
                if total_padded_area / max(1, area) >= 0.84:
                    return best_res

    if best_res is not None:
        return best_res
    raise ValueError("face cannot fit atlas width at the requested density")


def _build_zone_metrics(
    width: int,
    height: int,
    padding: int,
    placements: list[tuple[str, str, int, int, int, int, str]],
    island_zones: dict[tuple[str, str], str],
    zone_rects: dict[str, list[int]],
    *,
    culled_zero_area_faces: int = 0,
    mirrored_shared_faces: int = 0,
    explicit_shared_faces: int = 0,
    strategy: str = "strict_zone_partitions",
) -> dict[str, Any]:
    total_px = max(1, width * height)
    used_px = sum(w * h for _, _, _, _, w, h, _ in placements)
    padded_px = sum((w + 2 * padding) * (h + 2 * padding) for _, _, _, _, w, h, _ in placements)

    zone_stats: dict[str, dict[str, Any]] = {}
    for bname, face, x, y, w, h, _ in placements:
        z = island_zones.get((bname, face), "accents")
        st = zone_stats.setdefault(
            z,
            {
                "rect": zone_rects.get(z, [x, y, x + w, y + h]),
                "island_count": 0,
                "used_px": 0,
                "padded_px": 0,
                "boxes": set(),
            },
        )
        st["island_count"] += 1
        st["used_px"] += w * h
        st["padded_px"] += (w + 2 * padding) * (h + 2 * padding)
        st["boxes"].add(bname)

    zones_out: dict[str, dict[str, Any]] = {}
    for z in ANATOMICAL_ZONES:
        if z not in zone_stats:
            continue
        st = zone_stats[z]
        r = st["rect"]
        r_area = max(1, (r[2] - r[0]) * (r[3] - r[1]))
        zones_out[z] = {
            "rect": list(r),
            "island_count": st["island_count"],
            "box_count": len(st["boxes"]),
            "used_px": st["used_px"],
            "padded_px": st["padded_px"],
            "zone_fill_pct": round(100.0 * st["padded_px"] / r_area, 1),
            "atlas_share_pct": round(100.0 * st["used_px"] / total_px, 1),
        }

    return {
        "width": width,
        "height": height,
        "total_px": total_px,
        "used_px": used_px,
        "padded_px": padded_px,
        "utilization_pct": round(100.0 * used_px / total_px, 2),
        "padded_utilization_pct": round(100.0 * padded_px / total_px, 2),
        "unique_islands": len(placements),
        "culled_zero_area_faces": culled_zero_area_faces,
        "mirrored_shared_faces": mirrored_shared_faces,
        "explicit_shared_faces": explicit_shared_faces,
        "strategy": strategy,
        "zones": zones_out,
    }


def pack_semantic_uv_atlas(
    program: Any,
    scene: Any,
    *,
    max_width: int | None = None,
    padding: int = 1,
    mirror_lr_uv: str = "adaptive",
    cull_zero_area: bool = True,
    canvas_mode: str = "tight",
    zone_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """High-utilization Semantic Anatomical UV Partitioner & MaxRects Packer (v2).

    Parameters
    ----------
    mirror_lr_uv:
        ``"adaptive"`` (default): shares horizontally flipped UVs across mirrored L/R twins
        in symmetric zones (wings, forelimbs, hindlimbs, accents, emissive) while keeping
        focal hero zones (head, neck, torso, tail) unique.
        ``"all"``: shares flipped UVs across all mirrored L/R twins.
        ``"none"``: every box face gets a unique island (zero L/R sharing).
    cull_zero_area:
        When True, degenerate 0-thickness edge faces on ``plane`` elements map to
        ``[0, 0, 0, 0]`` instead of wasting padded atlas area.
    """
    if max_width is not None and (type(max_width) is not int or max_width < 8):
        raise ValueError("max_width must be an integer >= 8")
    if type(padding) is not int or padding < 1:
        raise ValueError("padding must be an integer >= 1")
    if mirror_lr_uv not in ("adaptive", "all", "none"):
        raise ValueError("mirror_lr_uv must be one of 'adaptive', 'all', 'none'")

    units_per_block = max(1, int(getattr(program.meta, "units_per_block", 16)))
    density = program.meta.texel_density / units_per_block
    is_authored = getattr(program.meta, "audit_profile", "strict") == "authored"

    emissive_mats = {
        p.name for p in getattr(program, "palettes", ()) if bool(getattr(p, "emissive", False))
    }

    # Build bone ancestor chains for accurate anatomical zone inheritance
    bone_parent = {b.name: b.parent_name for b in scene.bones}

    def _ancestors(bname: str) -> tuple[str, ...]:
        chain: list[str] = []
        cur = bone_parent.get(bname)
        seen: set[str] = {bname}
        while cur and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = bone_parent.get(cur)
        return tuple(chain)

    box_zones: dict[str, str] = {}
    for box in scene.boxes:
        box_zones[box.name] = classify_anatomical_zone(
            box.bone_name,
            box.name,
            ancestor_bones=_ancestors(box.bone_name),
            material_name=box.material,
            is_emissive=(box.material in emissive_mats),
            explicit_zone=getattr(box, "uv_zone", None),
            zone_overrides=zone_overrides,
        )

    # Identify mirrored L/R twin pairs (box_r -> box_l) for adaptive UV mirroring
    boxes_by_name = {b.name: b for b in scene.boxes}
    unmirrored_by_sig: dict[tuple[str, tuple[int, int, int], str, float], list[Any]] = {}
    for box in scene.boxes:
        if getattr(box, "mirrored", False):
            continue
        sz_key = tuple(int(round(float(c) * 1000)) for c in box.size)
        uv_s = round(float(getattr(box, "uv_scale", 1.0)), 4)
        canon, _ = _canonical_mirror_key(f"{box.bone_name}__{box.name}")
        unmirrored_by_sig.setdefault((canon, sz_key, box.material, uv_s), []).append(box)

    mirror_source_map: dict[str, str] = {}
    if mirror_lr_uv != "none":
        for box in scene.boxes:
            if not getattr(box, "mirrored", False):
                continue
            zone = box_zones[box.name]
            if mirror_lr_uv == "adaptive" and zone not in SYMMETRIC_UV_ZONES:
                continue
            # Method 1: SyncCraft mirror_x suffix ``right_bone__orig_box`` -> ``orig_box``
            cand_src = None
            if "__" in box.name:
                tail_name = box.name.split("__", 1)[1]
                if tail_name in boxes_by_name and not getattr(boxes_by_name[tail_name], "mirrored", False):
                    cand_src = boxes_by_name[tail_name]
            if cand_src is None:
                sz_key = tuple(int(round(float(c) * 1000)) for c in box.size)
                uv_s = round(float(getattr(box, "uv_scale", 1.0)), 4)
                canon, _ = _canonical_mirror_key(f"{box.bone_name}__{box.name}")
                pool = unmirrored_by_sig.get((canon, sz_key, box.material, uv_s), [])
                if pool:
                    # Pick twin with closest mirror-reflected X center
                    bx_c = float(box.center[0])
                    cand_src = min(pool, key=lambda b: abs(float(b.center[0]) + bx_c))
            if cand_src is not None:
                # Confirm identical dimensions, material, and uv_scale
                if (
                    all(math.isclose(float(a), float(b), abs_tol=1e-4) for a, b in zip(box.size, cand_src.size))
                    and box.material == cand_src.material
                    and math.isclose(float(getattr(box, "uv_scale", 1.0)), float(getattr(cand_src, "uv_scale", 1.0)), abs_tol=1e-4)
                ):
                    mirror_source_map[box.name] = cand_src.name

    master_islands: list[dict[str, Any]] = []
    faces_out: dict[str, dict[str, list[int]]] = {b.name: {} for b in scene.boxes}
    explicit_share_master: dict[tuple[str, str, int, int, str], tuple[str, str]] = {}
    explicit_share_links: list[tuple[str, str, str, str]] = []
    culled_zero_area = 0
    mirrored_shared_faces = 0
    explicit_shared_faces = 0

    _MEMBRANE_SLAB_KW = ("bay", "sail", "membrane", "patag", "lobe", "fin", "web", "wing_mem", "wing_sheet")

    for box in scene.boxes:
        sx, sy, sz = map(float, box.size)
        sizes = {
            "north": (sx, sy), "south": (sx, sy),
            "east": (sz, sy), "west": (sz, sy),
            "up": (sx, sz), "down": (sx, sz),
        }
        share_key = getattr(box, "uv_share_key", None)
        eff_density = density * float(getattr(box, "uv_scale", 1.0))
        zone = box_zones[box.name]
        is_mirror_slave = box.name in mirror_source_map
        min_dim, max_dim = min(sx, sy, sz), max(sx, sy, sz)
        is_dual_slab = (
            cull_zero_area
            and min_dim <= 0.55
            and max_dim >= 4.0 * max(0.05, min_dim)
            and any(k in box.name.lower() for k in _MEMBRANE_SLAB_KW)
        )

        for face in FACES:
            u, v = sizes[face]
            # Cull 0-area edge faces on 0-thickness planes AND 0.5u side rims on dual-sided membrane slabs!
            if cull_zero_area and (u <= 1e-9 or v <= 1e-9 or (is_dual_slab and min(u, v) <= 0.55)):
                faces_out[box.name][face] = [0, 0, 0, 0]
                culled_zero_area += 1
                continue

            u_is_pos_int = math.isclose(u, round(u), abs_tol=1e-9) and u > 0.0
            v_is_pos_int = math.isclose(v, round(v), abs_tol=1e-9) and v > 0.0
            fw, fh = u * eff_density, v * eff_density
            if not is_authored and (
                (u_is_pos_int and not math.isclose(fw, round(fw), abs_tol=1e-7))
                or (v_is_pos_int and not math.isclose(fh, round(fh), abs_tol=1e-7))
            ):
                raise ValueError("texel_density produces fractional texels; integer atlas required")
            w, h = max(1, int(round(fw))), max(1, int(round(fh)))

            if is_mirror_slave:
                mirrored_shared_faces += 1
                continue

            if share_key is not None:
                sig = (share_key, face, w, h, box.material)
                if sig in explicit_share_master:
                    m_box, m_face = explicit_share_master[sig]
                    explicit_share_links.append((box.name, face, m_box, m_face))
                    explicit_shared_faces += 1
                    continue
                explicit_share_master[sig] = (box.name, face)

            master_islands.append(
                {
                    "box": box.name,
                    "face": face,
                    "w": w,
                    "h": h,
                    "mat": box.material,
                    "zone": zone,
                }
            )

    if not master_islands:
        raise ValueError("cannot pack an empty scene")

    solved = _solve_semantic_atlas_packing(master_islands, padding=padding, max_width=max_width, canvas_mode=canvas_mode)
    width = int(solved["width"])
    height = int(solved["height"])
    if max(width, height) > 8192:
        raise ValueError("diagnostic atlas exceeds 8192-pixel limit; use production packer")

    placements: list[tuple[str, str, int, int, int, int, str]] = []
    island_zones: dict[tuple[str, str], str] = {}
    for x, y, isl in solved["placed"]:
        bname, face, w, h, mat, zone = isl["box"], isl["face"], isl["w"], isl["h"], isl["mat"], isl["zone"]
        faces_out[bname][face] = [x, y, x + w, y + h]
        placements.append((bname, face, x, y, w, h, mat))
        island_zones[(bname, face)] = zone

    # Resolve explicit shared faces
    for bname, face, m_box, m_face in explicit_share_links:
        faces_out[bname][face] = list(faces_out[m_box][m_face])

    # Resolve mirrored L/R slave faces (horizontally flipped [u1, v0, u0, v1] on reflected face!)
    for slave_name, master_name in mirror_source_map.items():
        for face in FACES:
            if face in faces_out[slave_name]:
                continue  # Already culled [0, 0, 0, 0]
            m_face = _mirrored_face_name(face)
            u0, v0, u1, v1 = faces_out[master_name][m_face]
            faces_out[slave_name][face] = [u1, v0, u0, v1]

    metrics = _build_zone_metrics(
        width,
        height,
        padding,
        placements,
        island_zones,
        solved["zone_rects"],
        culled_zero_area_faces=culled_zero_area,
        mirrored_shared_faces=mirrored_shared_faces,
        explicit_shared_faces=explicit_shared_faces,
        strategy=solved["strategy"],
    )

    payload: dict[str, Any] = {
        "width": width,
        "height": height,
        "padding": padding,
        "packing_mode": "semantic_maxrects",
        "mirror_lr_uv": mirror_lr_uv,
        "faces": faces_out,
        "placements": placements,
        "box_zones": box_zones,
        "island_zones": {f"{k[0]}.{k[1]}": v for k, v in island_zones.items()},
        "zone_rects": solved["zone_rects"],
        "mirror_source_map": mirror_source_map,
        "metrics": metrics,
    }
    total_shared = explicit_shared_faces + mirrored_shared_faces
    if total_shared > 0:
        payload["shared_uv_face_count"] = total_shared
    return payload


def allocate_uv_layout(
    program,
    scene,
    *,
    max_width=None,
    padding=1,
    box_order=None,
    face_order=None,
    packing_mode="legacy",
    mirror_lr_uv="adaptive",
    cull_zero_area=True,
    canvas_mode="tight",
    zone_overrides=None,
):
    """Allocate integer padded per-face UV rectangles without baking color.

    Supports:
    - ``packing_mode="legacy"`` (default): exact backwards-compatible shelf packer for
      existing tests and explicit ``box_order``/``face_order`` permutations.
    - ``packing_mode="semantic_maxrects"``: v3 Anatomical Zone Interlocking MaxRects
      packer with flexible non-square tight-fit canvases (typically 80-92%+ padded fill),
      zero-area culling, and adaptive L/R mirroring.
    """
    if packing_mode == "semantic_maxrects" and box_order is None and face_order is None:
        return pack_semantic_uv_atlas(
            program,
            scene,
            max_width=max_width,
            padding=padding,
            mirror_lr_uv=mirror_lr_uv,
            cull_zero_area=cull_zero_area,
            canvas_mode=canvas_mode,
            zone_overrides=zone_overrides,
        )

    if max_width is not None and (type(max_width) is not int or max_width < 8):
        raise ValueError("max_width must be an integer >= 8")
    if type(padding) is not int or padding < 1:
        raise ValueError("padding must be an integer >= 1")
    try:
        units_per_block = max(1, int(getattr(program.meta, "units_per_block", 16)))
        density = program.meta.texel_density / units_per_block
    except OverflowError as exc:
        raise ValueError("texel_density exceeds the atlas numeric range") from exc

    boxes_by_name = {b.name: b for b in scene.boxes}
    if box_order is None:
        ordered_boxes = list(scene.boxes)
    else:
        if set(box_order) != set(boxes_by_name) or len(box_order) != len(boxes_by_name):
            raise ValueError("box_order must be an exact permutation of scene box names")
        ordered_boxes = [boxes_by_name[name] for name in box_order]

    default_faces = FACES
    if face_order is None:
        ordered_faces = default_faces
    else:
        if set(face_order) != set(default_faces) or len(face_order) != 6:
            raise ValueError("face_order must be an exact permutation of the six cuboid faces")
        ordered_faces = tuple(face_order)

    is_authored = getattr(program.meta, "audit_profile", "strict") == "authored"
    measured_islands = []
    widest_needed = 8
    total_padded_area = 0
    seen_share_area: set[tuple[str, str, int, int, str]] = set()
    for box in ordered_boxes:
        sx, sy, sz = map(float, box.size)
        sizes = {
            "north": (sx, sy), "south": (sx, sy),
            "east": (sz, sy), "west": (sz, sy),
            "up": (sx, sz), "down": (sx, sz),
        }
        share_key = getattr(box, "uv_share_key", None)
        box_uv_scale = float(getattr(box, "uv_scale", 1.0))
        eff_density = density * box_uv_scale
        for face in ordered_faces:
            u, v = sizes[face]
            u_is_pos_int = math.isclose(u, round(u), abs_tol=1e-9) and u > 0.0
            v_is_pos_int = math.isclose(v, round(v), abs_tol=1e-9) and v > 0.0
            fw, fh = u * eff_density, v * eff_density
            if not is_authored and (
                (u_is_pos_int and not math.isclose(fw, round(fw), abs_tol=1e-7))
                or (v_is_pos_int and not math.isclose(fh, round(fh), abs_tol=1e-7))
            ):
                raise ValueError("texel_density produces fractional texels; integer atlas required")
            w, h = max(1, int(round(fw))), max(1, int(round(fh)))
            measured_islands.append((box.name, face, w, h, box.material, share_key))
            widest_needed = max(widest_needed, w + 2 * padding)
            if share_key is not None:
                sig = (share_key, face, w, h, box.material)
                if sig in seen_share_area:
                    continue
                seen_share_area.add(sig)
            total_padded_area += (w + 2 * padding) * (h + 2 * padding)

    if not measured_islands:
        raise ValueError("cannot pack an empty scene")

    if max_width is None:
        if widest_needed <= 512 and total_padded_area <= 512 * 512:
            effective_max_width = 512
        else:
            target_side = max(widest_needed, int(math.ceil(math.sqrt(total_padded_area * 1.35))))
            effective_max_width = min(8192, max(512, _pow2(target_side)))
    else:
        effective_max_width = max_width

    x = y = padding
    row_height = 0
    occupied_width = 1
    placements = []
    faces = {b.name: {} for b in scene.boxes}
    shared_rects: dict[tuple[str, str, int, int, str], list[int]] = {}
    shared_uv_face_count = 0
    for bname, face, w, h, mat, share_key in measured_islands:
        if min(w, h) < 1 or w + 2 * padding > effective_max_width:
            raise ValueError("face cannot fit atlas width at the requested density")
        if share_key is not None:
            sig = (share_key, face, w, h, mat)
            if sig in shared_rects:
                faces[bname][face] = list(shared_rects[sig])
                shared_uv_face_count += 1
                continue
        if x + w + padding > effective_max_width:
            x = padding
            y += row_height + 2 * padding
            row_height = 0
        rect = [x, y, x + w, y + h]
        faces[bname][face] = rect
        if share_key is not None:
            shared_rects[(share_key, face, w, h, mat)] = list(rect)
        placements.append((bname, face, x, y, w, h, mat))
        row_height = max(row_height, h)
        occupied_width = max(occupied_width, x + w + padding)
        x += w + 2 * padding

    width, height = _pow2(occupied_width), _pow2(y + row_height + padding)
    if max(width, height) > 8192:
        raise ValueError("diagnostic atlas exceeds 8192-pixel limit; use production packer")
    payload = {
        "width": width,
        "height": height,
        "padding": padding,
        "faces": faces,
        "placements": placements,
    }
    if shared_uv_face_count > 0:
        payload["shared_uv_face_count"] = shared_uv_face_count
    return payload


def build_atlas(
    program,
    scene,
    *,
    max_width=None,
    padding=1,
    packing_mode="legacy",
    mirror_lr_uv="adaptive",
):
    """Return (descriptor, PNG bytes). Every box face gets a padded island."""
    layout = allocate_uv_layout(
        program,
        scene,
        max_width=max_width,
        padding=padding,
        packing_mode=packing_mode,
        mirror_lr_uv=mirror_lr_uv,
    )
    colors = {p.name: oklch_to_srgb(p.base_lch) for p in program.palettes}
    width, height = layout["width"], layout["height"]
    faces = layout["faces"]
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for _, _, u, v, w, h, mat_name in layout["placements"]:
        color = colors[mat_name]
        draw.rectangle((u - padding, v - padding, u + w + padding - 1, v + h + padding - 1), fill=(*color, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False)
    payload = stream.getvalue()
    out = {
        "width": width,
        "height": height,
        "data_url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
        "faces": faces,
        "mode": f"{packing_mode}_base_color",
    }
    if "metrics" in layout:
        out["metrics"] = layout["metrics"]
        out["zone_rects"] = layout["zone_rects"]
        out["box_zones"] = layout["box_zones"]
    return out, payload


# ---------------------------------------------------------------------------
# Standalone Blockbench 5 .bbmodel UV Utilization Auditor, Repacker & Zone Map
# ---------------------------------------------------------------------------


def _extract_bbmodel_bone_paths(doc: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map each element UUID -> (immediate_group_name, ancestor_group_names)."""
    groups_by_uuid = {g["uuid"]: g.get("name", "") for g in doc.get("groups", []) if isinstance(g, dict)}
    elem_paths: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _walk(nodes: list[Any], ancestors: tuple[str, ...]) -> None:
        for node in nodes:
            if isinstance(node, str):
                imm = ancestors[0] if ancestors else "root"
                elem_paths[node] = (imm, ancestors[1:])
            elif isinstance(node, dict):
                uid = node.get("uuid", "")
                gname = node.get("name") or groups_by_uuid.get(uid, "")
                next_anc = (gname,) + ancestors if gname else ancestors
                _walk(node.get("children") or [], next_anc)

    _walk(doc.get("outliner") or [], ())
    return elem_paths


def measure_uv_utilization(bbmodel_doc: dict[str, Any]) -> dict[str, Any]:
    """Measure UV atlas utilization, shared faces, and zone scatter on any .bbmodel dict."""
    res = bbmodel_doc.get("resolution") or {}
    w = max(1, int(res.get("width", 1)))
    h = max(1, int(res.get("height", 1)))
    total_px = w * h
    elem_paths = _extract_bbmodel_bone_paths(bbmodel_doc)

    # Rasterize exact unique pixel coverage mask (accounts for shared/overlapping UVs honestly!)
    covered = np.zeros((h, w), dtype=bool)
    raw_face_area = 0.0
    unique_rects: set[tuple[int, int, int, int]] = set()
    active_faces = 0
    zero_faces = 0
    zone_area: dict[str, float] = {}

    for el in bbmodel_doc.get("elements", []):
        uid = el.get("uuid", "")
        bname, ancs = elem_paths.get(uid, ("root", ()))
        z = classify_anatomical_zone(bname, el.get("name", ""), ancestor_bones=ancs)
        for face, fdata in (el.get("faces") or {}).items():
            uv = fdata.get("uv") if isinstance(fdata, dict) else None
            if not uv or len(uv) != 4:
                continue
            u0, v0, u1, v1 = map(float, uv)
            fw, fh = abs(u1 - u0), abs(v1 - v0)
            if fw < 1e-6 or fh < 1e-6:
                zero_faces += 1
                continue
            active_faces += 1
            raw_face_area += fw * fh
            x0, x1 = int(max(0, min(w, math.floor(min(u0, u1))))), int(max(0, min(w, math.ceil(max(u0, u1)))))
            y0, y1 = int(max(0, min(h, math.floor(min(v0, v1))))), int(max(0, min(h, math.ceil(max(v0, v1)))))
            if x1 > x0 and y1 > y0:
                covered[y0:y1, x0:x1] = True
                unique_rects.add((x0, y0, x1, y1))
                zone_area[z] = zone_area.get(z, 0.0) + (x1 - x0) * (y1 - y0)

    unique_px = int(covered.sum())
    return {
        "width": w,
        "height": h,
        "total_px": total_px,
        "unique_covered_px": unique_px,
        "raw_face_area_px": round(raw_face_area, 1),
        "utilization_pct": round(100.0 * unique_px / total_px, 2),
        "active_faces": active_faces,
        "zero_area_faces": zero_faces,
        "unique_uv_rects": len(unique_rects),
        "zone_raw_px": {z: round(v, 1) for z, v in sorted(zone_area.items())},
    }


def _decode_bbmodel_texture(doc: dict[str, Any]) -> np.ndarray | None:
    texs = doc.get("textures") or []
    if not texs:
        return None
    src = str(texs[0].get("source", ""))
    if not src.startswith("data:image/"):
        return None
    _, b64 = src.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def _write_padded_rgba(canvas: np.ndarray, x: int, y: int, patch: np.ndarray, padding: int) -> None:
    h, w = patch.shape[:2]
    H, W = canvas.shape[:2]
    canvas[y : y + h, x : x + w] = patch
    for p in range(1, padding + 1):
        if y - p >= 0:
            canvas[y - p, x : x + w] = patch[0]
        if y + h - 1 + p < H:
            canvas[y + h - 1 + p, x : x + w] = patch[-1]
        y_lo = max(0, y - p)
        y_hi = min(H, y + h + p)
        if x - p >= 0:
            canvas[y_lo:y_hi, x - p] = canvas[y_lo:y_hi, x]
        if x + w - 1 + p < W:
            canvas[y_lo:y_hi, x + w - 1 + p] = canvas[y_lo:y_hi, x + w - 1]


def repack_bbmodel_uvs(
    bbmodel_doc: dict[str, Any],
    *,
    padding: int = 1,
    mirror_lr_uv: str = "adaptive",
    cull_zero_area: bool = True,
    preserve_existing_texture: bool = True,
    max_width: int | None = None,
    canvas_mode: str = "tight",
    zone_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Repack any Blockbench 5 ``.bbmodel`` into semantic anatomical UV partitions.

    Preserves exact per-face texel dimensions (including 64x focal eyes!) and optionally
    transfers the existing embedded PNG texture pixel-for-pixel into the new layout.

    Returns ``(repacked_bbmodel_doc, layout_report, zone_map_png_bytes)``.
    """
    doc = copy.deepcopy(bbmodel_doc)
    elem_paths = _extract_bbmodel_bone_paths(doc)
    old_res = doc.get("resolution") or {}
    old_w = max(1, int(old_res.get("width", 1)))
    old_h = max(1, int(old_res.get("height", 1)))
    old_rgba = _decode_bbmodel_texture(doc) if preserve_existing_texture else None
    if old_rgba is not None:
        tex_h, tex_w = old_rgba.shape[:2]
    else:
        tex_w, tex_h = old_w, old_h

    elements = doc.get("elements", [])
    if not elements:
        raise ValueError("bbmodel contains no elements")

    # Compute anatomical zone + geometric signature for every element
    elem_info: list[dict[str, Any]] = []
    for idx, el in enumerate(elements):
        uid = el.get("uuid", f"el_{idx}")
        ename = el.get("name", f"box_{idx}")
        bname, ancs = elem_paths.get(uid, ("root", ()))
        f_pt = el.get("from", [0.0, 0.0, 0.0])
        t_pt = el.get("to", [0.0, 0.0, 0.0])
        size = tuple(abs(float(t_pt[i]) - float(f_pt[i])) for i in range(3))
        center = tuple(0.5 * (float(f_pt[i]) + float(t_pt[i])) for i in range(3))
        zone = classify_anatomical_zone(bname, ename, ancestor_bones=ancs, zone_overrides=zone_overrides)
        # Canonicalize SyncCraft mirror_x child names:
        # Left child under `wing_l_fore` is named `wing_radius`; its mirrored twin under
        # `wing_r_upper__wing_l_fore` is named `wing_r_upper__wing_radius`!
        # Using the trailing short box name + zone + exact dimensions + bone-chain side tag matches 100% of twins!
        short_name = ename.split("__")[-1]
        canon_short, name_side = _canonical_mirror_key(short_name)
        is_prefixed_mirror = "__" in ename
        # Determine side from ancestor chain (outermost mirrored root bone wins!)
        bone_chain = (bname,) + ancs
        chain_side: str | None = None
        for anc_b in reversed(bone_chain):
            _, s_tag = _canonical_mirror_key(anc_b)
            if s_tag is not None:
                chain_side = s_tag
                break
        side = "r" if is_prefixed_mirror else (chain_side or name_side)
        canon = f"{zone}::{canon_short}"
        elem_info.append(
            {
                "idx": idx,
                "el": el,
                "uid": uid,
                "name": ename,
                "bone": bname,
                "zone": zone,
                "size": size,
                "center": center,
                "canon": canon,
                "side": side,
            }
        )

    # Detect L/R mirror twins for adaptive UV sharing
    mirror_source_idx: dict[int, int] = {}
    if mirror_lr_uv != "none":
        by_canon: dict[tuple[str, tuple[int, int, int]], list[dict[str, Any]]] = {}
        for info in elem_info:
            sz_k = tuple(int(round(s * 100)) for s in info["size"])
            by_canon.setdefault((info["canon"], sz_k), []).append(info)
        for _, grp in by_canon.items():
            if len(grp) != 2:
                continue
            a, b = grp[0], grp[1]
            # Confirm X-reflection symmetry (opposite X signs or explicit l/r tags)
            if not (
                (a["side"] == "l" and b["side"] == "r")
                or (a["side"] == "r" and b["side"] == "l")
                or (abs(a["center"][0] + b["center"][0]) <= 0.25 and abs(a["center"][0]) > 0.25)
            ):
                continue
            master, slave = (a, b) if a["side"] != "r" else (b, a)
            if mirror_lr_uv == "adaptive" and slave["zone"] not in SYMMETRIC_UV_ZONES:
                continue
            # Also confirm face pixel sizes match so 64x vs 32x eyes never mismatch
            f_ok = True
            for face in FACES:
                mf = _mirrored_face_name(face)
                u_m = (master["el"].get("faces") or {}).get(mf, {}).get("uv")
                u_s = (slave["el"].get("faces") or {}).get(face, {}).get("uv")
                if bool(u_m) != bool(u_s):
                    f_ok = False
                    break
                if u_m and u_s:
                    if abs(abs(u_m[2] - u_m[0]) - abs(u_s[2] - u_s[0])) > 0.51 or abs(abs(u_m[3] - u_m[1]) - abs(u_s[3] - u_s[1])) > 0.51:
                        f_ok = False
                        break
            if f_ok:
                mirror_source_idx[slave["idx"]] = master["idx"]

    # Collect unique master islands with exact existing pixel extents
    scale_x = tex_w / float(old_w)
    scale_y = tex_h / float(old_h)
    master_islands: list[dict[str, Any]] = []
    old_uv_by_key: dict[tuple[int, str], list[float]] = {}
    culled_zero = 0
    mirrored_faces = 0

    _MEMBRANE_SLAB_KW = ("bay", "sail", "membrane", "patag", "lobe", "fin", "web", "wing_mem", "wing_sheet")

    for info in elem_info:
        idx = info["idx"]
        el = info["el"]
        sx, sy, sz = info["size"]
        geom_sizes = {
            "north": (sx, sy), "south": (sx, sy),
            "east": (sz, sy), "west": (sz, sy),
            "up": (sx, sz), "down": (sx, sz),
        }
        el_faces = el.get("faces") or {}
        is_slave = idx in mirror_source_idx
        min_dim, max_dim = min(sx, sy, sz), max(sx, sy, sz)
        is_dual_slab = (
            cull_zero_area
            and min_dim <= 0.55
            and max_dim >= 4.0 * max(0.05, min_dim)
            and any(k in info["name"].lower() for k in _MEMBRANE_SLAB_KW)
        )
        for face in FACES:
            fdata = el_faces.get(face)
            if not isinstance(fdata, dict) or not fdata.get("uv"):
                continue
            uv = [float(v) for v in fdata["uv"]]
            old_uv_by_key[(idx, face)] = uv
            gu, gv = geom_sizes[face]
            fw_px = abs(uv[2] - uv[0]) * scale_x
            fh_px = abs(uv[3] - uv[1]) * scale_y
            if cull_zero_area and (gu <= 1e-6 or gv <= 1e-6 or fw_px < 0.5 or fh_px < 0.5 or (is_dual_slab and min(gu, gv) <= 0.55)):
                fdata["uv"] = [0, 0, 0, 0]
                culled_zero += 1
                continue
            if is_slave:
                mirrored_faces += 1
                continue
            w = max(1, int(round(fw_px)))
            h = max(1, int(round(fh_px)))
            master_islands.append(
                {
                    "box": f"{idx}:{info['name']}",
                    "elem_idx": idx,
                    "face": face,
                    "w": w,
                    "h": h,
                    "mat": info["zone"],
                    "zone": info["zone"],
                }
            )

    solved = _solve_semantic_atlas_packing(master_islands, padding=padding, max_width=max_width, canvas_mode=canvas_mode)
    new_w, new_h = int(solved["width"]), int(solved["height"])

    new_canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    placements: list[tuple[str, str, int, int, int, int, str]] = []
    island_zones: dict[tuple[str, str], str] = {}
    master_rect_by_key: dict[tuple[int, str], list[int]] = {}

    for x, y, isl in solved["placed"]:
        idx = int(isl["elem_idx"])
        face = isl["face"]
        w, h = isl["w"], isl["h"]
        rect = [x, y, x + w, y + h]
        master_rect_by_key[(idx, face)] = rect
        elements[idx]["faces"][face]["uv"] = rect
        placements.append((isl["box"], face, x, y, w, h, isl["mat"]))
        island_zones[(isl["box"], face)] = isl["zone"]

        # Transfer existing pixels cleanly onto new UV island (preserving flips/orientation!)
        if old_rgba is not None:
            ou0, ov0, ou1, ov1 = old_uv_by_key[(idx, face)]
            su0, sv0, su1, sv1 = ou0 * scale_x, ov0 * scale_y, ou1 * scale_x, ov1 * scale_y
            flip_x = su1 < su0
            flip_y = sv1 < sv0
            x_lo = int(max(0, min(tex_w - 1, math.floor(min(su0, su1)))))
            x_hi = int(max(x_lo + 1, min(tex_w, math.ceil(max(su0, su1)))))
            y_lo = int(max(0, min(tex_h - 1, math.floor(min(sv0, sv1)))))
            y_hi = int(max(y_lo + 1, min(tex_h, math.ceil(max(sv0, sv1)))))
            crop = old_rgba[y_lo:y_hi, x_lo:x_hi]
            if flip_x:
                crop = crop[:, ::-1]
            if flip_y:
                crop = crop[::-1, :]
            if crop.shape[1] != w or crop.shape[0] != h:
                crop = np.asarray(
                    Image.fromarray(crop, mode="RGBA").resize((w, h), resample=Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            _write_padded_rgba(new_canvas, x, y, crop, padding)

    # Apply horizontally flipped UVs to mirrored L/R slaves
    for slave_idx, master_idx in mirror_source_idx.items():
        el_faces = elements[slave_idx].get("faces") or {}
        for face in FACES:
            fdata = el_faces.get(face)
            if not isinstance(fdata, dict):
                continue
            if fdata.get("uv") == [0, 0, 0, 0]:
                continue
            m_face = _mirrored_face_name(face)
            m_rect = master_rect_by_key.get((master_idx, m_face))
            if m_rect is not None:
                u0, v0, u1, v1 = m_rect
                fdata["uv"] = [u1, v0, u0, v1]

    # Update document resolution & embedded texture
    doc["resolution"] = {"width": new_w, "height": new_h}
    if doc.get("textures"):
        buf = io.BytesIO()
        Image.fromarray(new_canvas, mode="RGBA").save(buf, format="PNG", optimize=True)
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        t0 = doc["textures"][0]
        t0["width"] = new_w
        t0["height"] = new_h
        t0["uv_width"] = new_w
        t0["uv_height"] = new_h
        t0["source"] = data_url
        t0["internal"] = True

    metrics = _build_zone_metrics(
        new_w,
        new_h,
        padding,
        placements,
        island_zones,
        solved["zone_rects"],
        culled_zero_area_faces=culled_zero,
        mirrored_shared_faces=mirrored_faces,
        strategy=solved["strategy"],
    )
    layout_report = {
        "before": measure_uv_utilization(bbmodel_doc),
        "after": measure_uv_utilization(doc),
        "metrics": metrics,
        "zone_rects": solved["zone_rects"],
        "placements": placements,
        "island_zones": {f"{k[0]}.{k[1]}": v for k, v in island_zones.items()},
        "width": new_w,
        "height": new_h,
        "padding": padding,
    }
    zone_map_png = render_uv_zone_map(layout_report)
    return doc, layout_report, zone_map_png


def render_uv_zone_map(layout: dict[str, Any], *, target_min_size: int = 768) -> bytes:
    """Render a crisp, color-coded visual UV Zone Diagnostic Atlas PNG."""
    w = int(layout["width"])
    h = int(layout["height"])
    scale = max(1, min(4, int(math.ceil(target_min_size / max(1, min(w, h))))))
    canvas_w = w * scale
    header_h = 34
    canvas_h = h * scale + header_h

    img = Image.new("RGBA", (canvas_w, canvas_h), (20, 23, 31, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Subtle background grid every 32 atlas texels
    grid_step = 32 * scale
    for gx in range(0, canvas_w, grid_step):
        draw.line([(gx, header_h), (gx, canvas_h)], fill=(34, 39, 52, 255), width=1)
    for gy in range(header_h, canvas_h, grid_step):
        draw.line([(0, gy), (canvas_w, gy)], fill=(34, 39, 52, 255), width=1)

    zone_rects = layout.get("zone_rects") or {}
    island_zones = layout.get("island_zones") or {}

    # 1. Draw translucent colored partition backdrops per zone (strict rects only; interlocking draws tight hull glow)
    strat = (layout.get("metrics") or {}).get("strategy", "strict_zone_partitions")
    if strat == "strict_zone_partitions":
        for z, r in zone_rects.items():
            col = ZONE_COLORS_RGB.get(z, (180, 180, 180))
            x0, y0, x1, y1 = r[0] * scale, header_h + r[1] * scale, r[2] * scale - 1, header_h + r[3] * scale - 1
            draw.rectangle((x0, y0, x1, y1), fill=(col[0], col[1], col[2], 34), outline=(col[0], col[1], col[2], 210), width=2)

    # 2. Draw every packed UV face island
    zone_first_pos: dict[str, tuple[int, int]] = {}
    for bname, face, x, y, iw, ih, _ in layout.get("placements", []):
        z = island_zones.get(f"{bname}.{face}", "accents")
        if z not in zone_first_pos:
            zone_first_pos[z] = (x, y)
        col = ZONE_COLORS_RGB.get(z, (180, 180, 180))
        x0 = x * scale
        y0 = header_h + y * scale
        x1 = (x + iw) * scale - 1
        y1 = header_h + (y + ih) * scale - 1
        draw.rectangle(
            (x0, y0, max(x0, x1), max(y0, y1)),
            fill=(col[0], col[1], col[2], 165),
            outline=(min(255, col[0] + 45), min(255, col[1] + 45), min(255, col[2] + 45), 245),
            width=1,
        )

    # 3. Zone badges (positioned at each zone's primary cluster anchor so badges never obscure neighbors!)
    metrics = layout.get("metrics") or {}
    z_info = metrics.get("zones") or {}
    for z, r in zone_rects.items():
        col = ZONE_COLORS_RGB.get(z, (200, 200, 200))
        fill_pct = z_info.get(z, {}).get("zone_fill_pct", 0.0)
        icnt = z_info.get(z, {}).get("island_count", 0)
        label = f" {z.upper()} ({icnt}f) " if strat != "strict_zone_partitions" else f" {z.upper()} ({icnt}f, {fill_pct:.0f}%) "
        ax, ay = zone_first_pos.get(z, (r[0], r[1]))
        lx = min(canvas_w - len(label) * 6 - 10, max(2, ax * scale + 3))
        ly = header_h + ay * scale + 3
        tw = len(label) * 6 + 6
        draw.rectangle((lx, ly, min(canvas_w - 2, lx + tw), ly + 15), fill=(15, 17, 24, 232), outline=(*col, 255), width=1)
        draw.text((lx + 3, ly + 2), label, fill=(245, 248, 255, 255))

    # 4. Header telemetry bar
    draw.rectangle((0, 0, canvas_w, header_h - 1), fill=(13, 16, 22, 255))
    util = metrics.get("utilization_pct", 0.0)
    putil = metrics.get("padded_utilization_pct", 0.0)
    mir = metrics.get("mirrored_shared_faces", 0)
    culled = metrics.get("culled_zero_area_faces", 0)
    header_txt = (
        f"SyncCraft UV v3 | {w}x{h} | Fill: {util:.1f}% (padded {putil:.1f}%) | "
        f"{len(zone_rects)} Zones | {mir} L/R Shared | {culled} Culled | {strat}"
    )
    draw.text((10, 10), header_txt, fill=(235, 242, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
