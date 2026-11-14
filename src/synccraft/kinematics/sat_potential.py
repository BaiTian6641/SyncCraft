"""Reference OBB separating-axis audit and a separately labelled surrogate.

SAT penetration depth is a distance along a separating-axis candidate. It is
not intersection volume. The soft minimum is a lower bound and can miss small
positive overlaps; only the raw minimum is used by the deterministic audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .se3_chain import WorldBox


@dataclass(frozen=True)
class SATResult:
    raw_min_overlap: float
    penetration_depth: float
    softmin_overlap: float
    surrogate_energy: float
    tested_axes: int
    collision: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_min_overlap": self.raw_min_overlap,
            "penetration_depth": self.penetration_depth,
            "softmin_overlap": self.softmin_overlap,
            "surrogate_energy": self.surrogate_energy,
            "tested_axes": self.tested_axes,
            "collision": self.collision,
        }


def _validate_box(box: WorldBox) -> None:
    if box.center.shape != (3,) or box.rotation.shape != (3, 3):
        raise ValueError("Invalid OBB center or rotation shape")
    if not all(np.isfinite(value).all() for value in (box.center, box.rotation, box.half_extents)):
        raise ValueError("Nonfinite OBB data")
    if box.half_extents.shape != (3,) or np.any(box.half_extents <= 0):
        raise ValueError("OBB extents must be positive")
    if not np.allclose(box.rotation.T @ box.rotation, np.eye(3), atol=1e-9, rtol=0):
        raise ValueError("OBB rotation must be orthonormal")
    if not np.isclose(np.linalg.det(box.rotation), 1.0, atol=1e-9, rtol=0):
        raise ValueError("OBB rotation must be proper (determinant +1)")


def sat_obb(
    left: WorldBox,
    right: WorldBox,
    *,
    collision_tolerance: float = 1e-7,
    beta: float = 25.0,
    energy_tolerance: float = 0.01,
    cross_axis_tolerance: float = 1e-10,
) -> SATResult:
    """Evaluate up to 15 SAT axes, dropping numerically degenerate crosses.

    Touching or separated OBBs have exactly zero returned penetration depth and
    surrogate energy. Axes shorter than ``cross_axis_tolerance`` are omitted,
    never normalized to zero and never included in the minimum. This is needed
    for identical and parallel boxes to report positive intersection correctly.
    """
    _validate_box(left)
    _validate_box(right)
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if (
        not np.isfinite((collision_tolerance, energy_tolerance, cross_axis_tolerance)).all()
        or collision_tolerance < 0 or energy_tolerance < 0 or cross_axis_tolerance <= 0
    ):
        raise ValueError("Tolerances must be nonnegative (cross-axis tolerance positive)")
    axes: list[np.ndarray] = [left.rotation[:, i] for i in range(3)]
    axes.extend(right.rotation[:, i] for i in range(3))
    for i in range(3):
        for j in range(3):
            candidate = np.cross(left.rotation[:, i], right.rotation[:, j])
            norm = np.linalg.norm(candidate)
            if norm > cross_axis_tolerance:
                axes.append(candidate / norm)
    axis_array = np.asarray(axes)
    radii_left = np.abs(axis_array @ left.rotation) @ left.half_extents
    radii_right = np.abs(axis_array @ right.rotation) @ right.half_extents
    distance = np.abs(axis_array @ (right.center - left.center))
    overlaps = radii_left + radii_right - distance
    raw_min = float(overlaps.min())
    # Stable logsumexp: -log(sum(exp(-beta*d)))/beta.
    soft_min = raw_min - float(np.log(np.exp(-beta * (overlaps - raw_min)).sum())) / beta
    collision = raw_min > collision_tolerance
    penetration = max(raw_min, 0.0) if collision else 0.0
    energy = max(soft_min - energy_tolerance, 0.0) ** 2 if collision else 0.0
    return SATResult(raw_min, penetration, soft_min, energy, len(axes), collision)
