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
