"""Model, constrained PGD projector, and bounded repair-loop boundaries."""

from .learned_refiner import LearnedKinematicRefiner
from .pgd_projector import (
    PGDProjectionResult,
    project_program,
    torch_euler_xyz_matrix,
    verify_torch_se3_parity,
)
from .repair_loop import MAX_ALLOWED_G2_REAUDITS, run_bounded_repair_policy

__all__ = [
    "MAX_ALLOWED_G2_REAUDITS",
    "LearnedKinematicRefiner",
    "PGDProjectionResult",
    "project_program",
    "run_bounded_repair_policy",
    "torch_euler_xyz_matrix",
    "verify_torch_se3_parity",
]
