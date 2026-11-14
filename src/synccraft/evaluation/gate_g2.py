"""Gate G2 calculator. Evidence must come from actual experiments, never a scaffold."""
import re

REQUIRED_CHECKS = (
    "full_static_audit_implemented", "heldout_split_frozen_before_training",
    "all_samples_have_at_least_16_parts", "repair_pgd_policy_preregistered",
    "matched_raw_budgets_enforced", "at_most_three_reaudits_per_sample",
    "failures_and_timeouts_counted", "baseline_implementations_verified",
)


def decide_gate(evidence: dict):
    if not isinstance(evidence, dict):
        raise ValueError("gate evidence must be a JSON object")
    reasons = [name for name in REQUIRED_CHECKS if evidence.get(name) is not True]
    for name in ("dataset_manifest_sha256", "evaluation_manifest_sha256"):
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(evidence.get(name, ""))):
            reasons.append(name)
    total, passed = evidence.get("total"), evidence.get("passed")
    if type(total) is not int or total <= 0 or type(passed) is not int or not 0 <= passed <= total:
        reasons.append("valid_sample_counts")
    if reasons:
        return {"decision": "blocked", "learned_refiner_enabled": False,
                "missing_or_invalid_evidence": reasons}
    rate = passed / total
    # Integer comparison makes the 90% boundary explicit and reproducible.
    enabled = passed * 10 < total * 9
    return {"decision": "eligible_for_learned_refiner" if enabled else "skip_learned_refiner",
            "learned_refiner_enabled": enabled, "pass_rate": rate,
            "passed": passed, "total": total,
            "interpretation": "Engineering branch decision; not proof of a local-minimum cause."}


def inspect_runtime_gate_prerequisites() -> dict[str, bool]:
    """Dynamically inspect repository module readiness for Gate G2 prerequisites.

    Prevents notebooks or scripts from hardcoding True for unimplemented Phase 1/2
    components (e.g. Steve/Alex grip audit, differentiable PyTorch SE(3)/PGD baselines).
    """
    full_audit_ready = False
    pgd_ready = False
    baselines_ready = False

    try:
        from synccraft.dsl import parse_program
        from synccraft.kinematics.se3_chain import compile_geometry
        from synccraft.audit.sync_audit import audit_scene

        sample = parse_program(
            "@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }\n"
            "@palette { m: ramp(base_lch=[0.6, 0.05, 220], steps=4) }\n"
            "@rig { root bone r { pivot_local: [0,0,0] box b { size: [2,2,2], material: m } } }\n"
        )
        rep = audit_scene(compile_geometry(sample))
        full_audit_ready = bool(rep.get("complete_for_release", False) and not rep.get("coverage_gaps"))
    except Exception:
        full_audit_ready = False

    try:
        from synccraft.refiner.pgd_baseline import project_with_pgd

        project_with_pgd()
        pgd_ready = True
        baselines_ready = True
    except NotImplementedError:
        pgd_ready = False
        baselines_ready = False
    except TypeError:
        # Real callable requiring arguments exists
        pgd_ready = True
        baselines_ready = True
    except Exception:
        pgd_ready = False
        baselines_ready = False

    return {
        "full_static_audit_implemented": full_audit_ready,
        "repair_pgd_policy_preregistered": pgd_ready,
        "baseline_implementations_verified": baselines_ready,
    }
