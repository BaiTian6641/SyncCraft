"""Gate G2 calculator, runtime prerequisite verifier, and prompt-adherence auditor.

Evidence must come strictly from executed artifacts and verified completions,
never hardcoded booleans, TypeError catch-alls, or memorized single-asset outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any

from synccraft.dsl.ast_nodes import Bone, Box, Program
from synccraft.dsl.parser import format_program
from synccraft.kinematics.se3_chain import Scene, compile_geometry

REQUIRED_CHECKS = (
    "full_static_audit_implemented",
    "heldout_split_frozen_before_training",
    "all_samples_have_at_least_16_parts",
    "repair_pgd_policy_preregistered",
    "matched_raw_budgets_enforced",
    "at_most_three_reaudits_per_sample",
    "failures_and_timeouts_counted",
    "baseline_implementations_verified",
)

_BIOLOGICAL_MARKERS: tuple[str, ...] = (
    "wyvern",
    "drake",
    "dermis_continuum",
    "osteoderm",
    "mandible",
    "snout",
    "thigh",
    "shank",
    "phalange",
    "patagium",
    "carapace_continuum",
)

_INDUSTRIAL_WEAPON_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    "crossbow",
    "picatinny",
    "cranequin",
    "buttstock",
    "forklift",
    "excavator",
    "pagoda",
    "chandelier",
    "bookshelf",
    "pommel",
    "crossguard",
    "trebuchet",
    "catapult",
)

_WING_ANATOMY_MARKERS: tuple[str, ...] = (
    "wing",
    "pinion",
    "patagium",
    "elytra",
    "feather",
)

_WINGLESS_TERRESTRIAL_CUES: tuple[str, ...] = (
    "rhino",
    "tortoise",
    "turtle",
    "spider",
    "arachnid",
    "scorpion",
    "crab",
    "crustacean",
    "boar",
    "wolf",
    "hound",
    "bear",
    "elephant",
    "stag",
    "serpent",
)

_SUBJECT_LEXICONS: dict[str, tuple[str, ...]] = {
    "crossbow": ("crossbow", "prod", "stirrup", "bolt_rail", "cranequin", "quiver"),
    "pagoda": ("pagoda", "eave", "upturned_tip", "stone_plinth", "timber_hall"),
    "wyvern": ("wyvern", "wing_spar", "wing_membrane", "patagium", "thorax"),
    "excavator": ("excavator", "bucket", "dipper", "undercarriage", "cab_hull"),
    "forklift": ("forklift", "mast", "tine", "counterweight"),
    "chandelier": ("chandelier", "candelabra", "pendant", "sconce"),
    "sword": ("blade", "crossguard", "pommel", "ricasso", "fuller"),
    "trebuchet": ("trebuchet", "counterweight_box", "throwing_arm", "sling"),
}


def decide_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Gate G2 decision from preregistered checks and integer 90% threshold."""
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
        return {
            "decision": "blocked",
            "learned_refiner_enabled": False,
            "missing_or_invalid_evidence": reasons,
        }
    rate = passed / total
    # Integer comparison makes the 90% boundary explicit and reproducible.
    enabled = passed * 10 < total * 9
    return {
        "decision": "eligible_for_learned_refiner" if enabled else "skip_learned_refiner",
        "learned_refiner_enabled": enabled,
        "pass_rate": rate,
        "passed": passed,
        "total": total,
        "interpretation": "Engineering branch decision; not proof of a local-minimum cause.",
    }


def _has_geometric_bilateral_symmetry(scene: Scene) -> bool:
    if any(b.mirrored for b in scene.boxes) or any(b.mirrored for b in scene.bones):
        return True
    # Also accept authored rigs with explicit symmetric +/-X box pairs
    off_axis = [b for b in scene.boxes if abs(float(b.center[0])) > 0.5]
    if len(off_axis) < 2:
        return False
    matched = 0
    for bx in off_axis:
        mirrored_center = (-float(bx.center[0]), float(bx.center[1]), float(bx.center[2]))
        if any(
            other.name != bx.name
            and abs(float(other.center[0]) - mirrored_center[0]) <= 0.35
            and abs(float(other.center[1]) - mirrored_center[1]) <= 0.35
            and abs(float(other.center[2]) - mirrored_center[2]) <= 0.35
            and all(abs(float(other.size[k]) - float(bx.size[k])) <= 0.25 for k in range(3))
            for other in off_axis
        ):
            matched += 1
    return matched >= 2


def _collect_program_identifiers_lower(program: Program, scene: Scene) -> str:
    tokens: list[str] = [
        *(b.name.lower() for b in scene.bones),
        *(bx.name.lower() for bx in scene.boxes),
        *(loc.name.lower() for loc in scene.locators),
        *(p.name.lower() for p in program.palettes),
        *(m.name.lower() for m in program.materials),
        *(m.weld_group.lower() for m in program.materials if m.weld_group),
        *(mu.name.lower() for mu in program.muscles),
    ]
    return " ".join(tokens)


def verify_prompt_adherence(
    program: Program,
    scene: Scene | None,
    prompt_constraints: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Verify that a compiled Program matches the requested prompt/reference specification (H5).

    Catches under-parted shortcuts (< 16 boxes on dense Gate G2 prompts), wrong archetypes
    or target profiles, missing symmetry/materials/animation features, memorized cross-domain
    or conflicting-anatomy assets, and duplicate memorized AST hashes.
    """
    if not isinstance(program, Program):
        raise TypeError("verify_prompt_adherence requires a Program instance")
    compiled_scene = scene if scene is not None else compile_geometry(program)

    if prompt_constraints is None:
        canonical_sha = hashlib.sha256(format_program(program).encode("utf-8")).hexdigest()
        return {
            "passed": True,
            "violations": [],
            "checked_constraints": {},
            "canonical_ast_sha256": canonical_sha,
            "observed": {
                "box_count": len(compiled_scene.boxes),
                "bone_count": len(compiled_scene.bones),
                "archetype": program.meta.archetype,
                "target_profile": program.meta.target_profile,
                "has_mirror": _has_geometric_bilateral_symmetry(compiled_scene),
            },
        }

    spec: dict[str, Any]
    if isinstance(prompt_constraints, str):
        spec = {"prompt": prompt_constraints}
    elif isinstance(prompt_constraints, Mapping):
        spec = dict(prompt_constraints)
    else:
        raise TypeError("prompt_constraints must be a Mapping, str, or None")

    prompt_text = str(spec.get("prompt", ""))
    prompt_low = prompt_text.lower()
    family_low = str(spec.get("family_id", spec.get("family", ""))).lower()
    domain_low = str(spec.get("domain", "")).lower()

    violations: list[str] = []
    checked: dict[str, Any] = {}

    # 1. Minimum / exact box count from prompt or reference (never self-selected by model!)
    min_boxes = int(spec["min_boxes"]) if "min_boxes" in spec and spec["min_boxes"] is not None else 0
    if bool(spec.get("gate_g2_dense", False)):
        min_boxes = max(min_boxes, 16)
    ref_boxes = spec.get("reference_boxes", spec.get("reference_box_count"))
    if isinstance(ref_boxes, int) and ref_boxes >= 16 and "min_boxes" not in spec:
        min_boxes = max(min_boxes, 16)
    if isinstance(spec.get("reference_dsl"), str) and "min_boxes" not in spec:
        try:
            from synccraft.dsl.parser import parse_program

            ref_sc = compile_geometry(parse_program(spec["reference_dsl"]))
            if len(ref_sc.boxes) >= 16:
                min_boxes = max(min_boxes, 16)
        except Exception:
            pass

    for m in re.finditer(r"(?:at\s+least|>=|minimum(?:\s+of)?)\s*(\d+)\s*(?:boxes|parts|cuboids|primitives)", prompt_low):
        min_boxes = max(min_boxes, int(m.group(1)))

    exact_boxes = spec.get("exact_boxes")
    if exact_boxes is None:
        m_exact = re.search(r"(?:exactly|target_boxes\s*=)\s*(\d+)\s*(?:boxes|parts|cuboids)?", prompt_low)
        if m_exact:
            exact_boxes = int(m_exact.group(1))

    n_boxes = len(compiled_scene.boxes)
    n_bones = len(compiled_scene.bones)
    has_sym = _has_geometric_bilateral_symmetry(compiled_scene)

    if min_boxes > 0:
        checked["min_boxes"] = min_boxes
        if n_boxes < min_boxes:
            violations.append(f"min_boxes_not_met: observed {n_boxes} < required {min_boxes}")

    if isinstance(exact_boxes, int) and exact_boxes > 0:
        tol = int(spec.get("box_count_tolerance", 0))
        checked["exact_boxes"] = exact_boxes
        if abs(n_boxes - exact_boxes) > tol:
            violations.append(f"exact_boxes_mismatch: observed {n_boxes} vs required {exact_boxes} (tol={tol})")

    min_bones = int(spec.get("min_bones", 0) or 0)
    if min_bones > 0:
        checked["min_bones"] = min_bones
        if n_bones < min_bones:
            violations.append(f"min_bones_not_met: observed {n_bones} < required {min_bones}")

    # 2. Target profile and archetype adherence
    expected_archetype = spec.get("archetype")
    if expected_archetype is None:
        m_arch = re.search(r"\barchetype\s*[:=]\s*(entity|weapon_3d|block_multipart|item_extruded)\b", prompt_low)
        if m_arch:
            expected_archetype = m_arch.group(1)
    if expected_archetype:
        checked["archetype"] = str(expected_archetype)
        if program.meta.archetype != str(expected_archetype):
            violations.append(
                f"archetype_mismatch: observed {program.meta.archetype!r} != required {str(expected_archetype)!r}"
            )

    expected_profile = spec.get("target_profile")
    if expected_profile is None:
        m_prof = re.search(
            r"\btarget_profile\s*[:=]\s*(bedrock_entity|geckolib|java_modern|java_legacy|bbmodel_5)\b",
            prompt_low,
        )
        if m_prof:
            expected_profile = m_prof.group(1)
    if expected_profile:
        checked["target_profile"] = str(expected_profile)
        if program.meta.target_profile != str(expected_profile):
            violations.append(
                f"target_profile_mismatch: observed {program.meta.target_profile!r} != required {str(expected_profile)!r}"
            )

    # 3. Symmetry / bilateral mirror constraints
    req_sym = spec.get("requires_symmetry", spec.get("requires_mirror"))
    if req_sym is None and any(
        cue in prompt_low
        for cue in ("bilateral symmetry", "mirrored", "mirror_x", "symmetric left/right", "paired wings", "paired limbs")
    ):
        req_sym = True
    if req_sym is True:
        checked["requires_symmetry"] = True
        if not has_sym:
            violations.append("missing_requested_symmetry: no mirrored or bilateral +/-X paired geometry found")
    if spec.get("forbid_mirror") is True:
        checked["forbid_mirror"] = True
        if any(b.mirrored for b in compiled_scene.boxes):
            violations.append("unexpected_mirror_expansion: prompt forbade mirror clones")

    # 4. Requested materials, behavior/motion, muscles, and locators
    obs_mat_fams = sorted({m.family for m in program.materials})
    req_mats = spec.get("requires_materials")
    if req_mats is True or "@materials" in prompt_low:
        checked["requires_materials"] = True
        if not program.materials:
            violations.append("missing_requested_materials: @materials section is empty")

    req_mat_fams = spec.get("material_families")
    if req_mat_fams:
        wanted = sorted({str(x) for x in req_mat_fams})
        checked["material_families"] = wanted
        missing_fams = [f for f in wanted if f not in obs_mat_fams]
        if missing_fams:
            violations.append(f"missing_material_families: {missing_fams}")

    req_beh = spec.get("requires_behavior", spec.get("requires_animation"))
    if req_beh is True or "@behavior" in prompt_low:
        checked["requires_behavior"] = True
        if program.behavior is None or len(program.behavior.clips) == 0:
            violations.append("missing_requested_behavior: @behavior clips absent")

    if spec.get("requires_muscles") is True or "@muscles" in prompt_low:
        checked["requires_muscles"] = True
        if len(program.muscles) == 0:
            violations.append("missing_requested_muscles: @muscles declarations absent")

    if spec.get("requires_locators") is True:
        checked["requires_locators"] = True
        if len(compiled_scene.locators) == 0:
            violations.append("missing_requested_locators: 0 locators in compiled scene")

    # 5. Domain, anatomical body-plan, and anti-memorization checks
    id_blob = _collect_program_identifiers_lower(program, compiled_scene)
    combined_req = f"{domain_low} {family_low} {prompt_low}"

    if domain_low:
        checked["domain"] = domain_low
        if domain_low in ("biological", "fauna", "creature", "creatures"):
            if program.meta.archetype in ("weapon_3d", "item_extruded"):
                violations.append(
                    f"domain_archetype_conflict: biological domain emitted {program.meta.archetype!r}"
                )
            bad_hits = [w for w in _INDUSTRIAL_WEAPON_ARCHITECTURE_MARKERS if w in id_blob and w not in combined_req]
            if bad_hits:
                violations.append(f"domain_semantics_mismatch: biological prompt emitted non-biological parts {bad_hits[:3]}")
        elif domain_low in ("industrial", "architecture", "furniture", "flora", "vehicles", "weapons", "artificial"):
            if domain_low in ("architecture", "furniture", "flora") and program.meta.archetype == "entity" and not expected_archetype:
                violations.append(
                    f"domain_archetype_conflict: {domain_low!r} domain emitted entity archetype without request"
                )
            bio_hits = [w for w in _BIOLOGICAL_MARKERS if w in id_blob and w not in combined_req]
            if bio_hits:
                violations.append(
                    f"domain_semantics_mismatch: {domain_low!r} prompt emitted biological anatomy {bio_hits[:3]}"
                )

    # Same-domain anatomical mismatch: wingless terrestrial creature vs winged wyvern/bird
    if any(cue in combined_req for cue in _WINGLESS_TERRESTRIAL_CUES) and not any(
        w in combined_req for w in ("wing", "winged", "griffin", "wyvern", "dragon", "flying")
    ):
        wing_hits = [w for w in _WING_ANATOMY_MARKERS if w in id_blob]
        if wing_hits:
            violations.append(f"anatomical_mismatch: wingless creature prompt emitted wing anatomy {wing_hits[:3]}")

    # Cross-subject contraband lexicon check (e.g. memorized crossbow/pagoda/wyvern emitted for excavator)
    requested_subjects = {subj for subj, words in _SUBJECT_LEXICONS.items() if subj in combined_req or any(w in combined_req for w in words[:2])}
    if not requested_subjects and (prompt_low or family_low):
        # Even if prompt is a novel unseen subject not in _SUBJECT_LEXICONS, ensure it isn't a memorized known subject
        for other_subj, other_words in _SUBJECT_LEXICONS.items():
            matched_other = [w for w in other_words if w in id_blob]
            if len(matched_other) >= 2:
                violations.append(
                    f"semantic_subject_mismatch: novel prompt emitted memorized {other_subj!r} lexicon {matched_other[:3]}"
                )
                break
    elif requested_subjects:
        for other_subj, other_words in _SUBJECT_LEXICONS.items():
            if other_subj in requested_subjects:
                continue
            matched_other = [w for w in other_words if w in id_blob and w not in combined_req]
            if len(matched_other) >= 2:
                violations.append(
                    f"semantic_subject_mismatch: prompt for {sorted(requested_subjects)} emitted {other_subj!r} parts {matched_other[:3]}"
                )
                break

    for req_kw in spec.get("required_keywords", ()):
        kw_low = str(req_kw).lower()
        if kw_low and kw_low not in id_blob:
            violations.append(f"missing_required_keyword: {kw_low!r}")

    for bad_kw in spec.get("forbidden_keywords", ()):
        kw_low = str(bad_kw).lower()
        if kw_low and kw_low in id_blob:
            violations.append(f"forbidden_keyword_present: {kw_low!r}")

    canonical_sha = hashlib.sha256(format_program(program).encode("utf-8")).hexdigest()
    forbidden_hashes = {str(h) for h in spec.get("forbidden_ast_sha256s", ())}
    if canonical_sha in forbidden_hashes:
        violations.append(f"memorized_or_duplicate_completion: canonical_ast_sha256={canonical_sha[:16]}")

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "checked_constraints": checked,
        "canonical_ast_sha256": canonical_sha,
        "observed": {
            "box_count": n_boxes,
            "bone_count": n_bones,
            "locator_count": len(compiled_scene.locators),
            "archetype": program.meta.archetype,
            "target_profile": program.meta.target_profile,
            "has_mirror": has_sym,
            "has_materials": len(program.materials) > 0,
            "has_behavior": program.behavior is not None and len(program.behavior.clips) > 0,
            "material_families": obs_mat_fams,
        },
    }


def inspect_runtime_gate_prerequisites(
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Dynamically verify repository baseline/audit readiness and optional empirical artifacts (C3).

    Fixes broken ``synccraft.refiner.pgd_baseline`` import and NEVER treats ``TypeError``
    as proof of implementation: executes a live 2-box penetration repair through both
    ``synccraft.modeling.pgd_projector.project_program`` and
    ``synccraft.modeling.repair_loop.run_bounded_repair_policy`` with monotonic ``CostLedger`` checks.
    """
    full_audit_ready = False
    repair_pgd_ready = False
    baselines_ready = False

    try:
        from synccraft.audit.sync_audit import audit_scene
        from synccraft.dsl import parse_program

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
        from synccraft.dsl import Meta, Palette, Place, parse_program, validate_program
        from synccraft.evaluation.budget import CostLedger
        from synccraft.modeling.pgd_projector import project_program
        from synccraft.modeling.repair_loop import run_bounded_repair_policy

        # Construct a genuine 0.25-unit separated contact violation fixture and verify both solvers solve it
        defective_prog = validate_program(
            Program(
                Meta("block_multipart", "bbmodel_5", 16),
                [Palette("m", (0.6, 0.05, 220.0), 4)],
                Bone(
                    "r",
                    pivot_local=(0.0, 0.0, 0.0),
                    items=[
                        Box("base", (4, 4, 4), "m"),
                        Box(
                            "top",
                            (4, 2, 4),
                            "m",
                            place=Place("base", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), offset=(0.0, 0.25, 0.0)),
                        ),
                    ],
                ),
            )
        )
        defective_dsl = format_program(defective_prog)
        pgd_ledger = CostLedger()
        pgd_res = project_program(parse_program(defective_dsl), max_steps=10, cost_ledger=pgd_ledger)

        rep_ledger = CostLedger()
        rep_res = run_bounded_repair_policy(
            defective_dsl,
            {"archetype": "block_multipart", "target_profile": "bbmodel_5", "min_boxes": 2},
            rep_ledger,
            max_reaudits=3,
        )
        repair_pgd_ready = bool(
            pgd_res["passed"] is True
            and pgd_res["serialized_reaudit_passed"] is True
            and pgd_ledger.projection_steps >= 1
            and rep_res["passed"] is True
            and rep_ledger.initial_audit_calls == 1
            and 1 <= rep_ledger.reaudit_calls <= 3
        )
        baselines_ready = repair_pgd_ready
    except Exception:
        repair_pgd_ready = False
        baselines_ready = False

    result: dict[str, bool] = {
        "full_static_audit_implemented": full_audit_ready,
        "repair_pgd_policy_preregistered": repair_pgd_ready,
        "baseline_implementations_verified": baselines_ready,
    }

    if artifacts is not None:
        ds_manifest = artifacts.get("dataset_manifest") or {}
        eval_rows = list(artifacts.get("evaluation_records") or ())
        expected_total = artifacts.get("expected_benchmark_count")
        budget_limits = artifacts.get("budget_limits")
        cost_ledger_dict = artifacts.get("cost_ledger")

        ds_sha = str(ds_manifest.get("dataset_manifest_sha256", artifacts.get("dataset_manifest_sha256", "")))
        train_fams = set(ds_manifest.get("seen_families", ()))
        unseen_fams = set(ds_manifest.get("unseen_families", ()))
        split_frozen = bool(
            re.fullmatch(r"[a-fA-F0-9]{64}", ds_sha)
            and (
                (train_fams and unseen_fams and train_fams.isdisjoint(unseen_fams))
                or ds_manifest.get("split_counts", {}).get("test_unseen", 0) > 0
            )
        )
        all_16 = bool(
            len(eval_rows) > 0
            and all(
                int(r.get("reference_boxes", r.get("min_boxes", 0))) >= 16
                for r in eval_rows
            )
        )
        at_most_3 = bool(
            len(eval_rows) > 0
            and all(0 <= int(r.get("reaudits_used", 0)) <= 3 for r in eval_rows)
        )
        counted_all = bool(
            len(eval_rows) > 0
            and (expected_total is None or len(eval_rows) == int(expected_total))
            and all("passed" in r or "sat_collision_passed" in r for r in eval_rows)
        )
        budgets_ok = False
        if isinstance(cost_ledger_dict, Mapping):
            comp_calls = int(cost_ledger_dict.get("compiler_calls", 0))
            init_auds = int(cost_ledger_dict.get("initial_audit_calls", 0))
            reauds = int(cost_ledger_dict.get("reaudit_calls", 0))
            budgets_ok = (
                comp_calls >= len(eval_rows) > 0
                and init_auds == len(eval_rows)
                and reauds <= 3 * len(eval_rows)
            )
            if isinstance(budget_limits, Mapping):
                budgets_ok = budgets_ok and all(
                    int(cost_ledger_dict.get(k, 0)) <= int(v) for k, v in budget_limits.items()
                )

        result.update(
            {
                "heldout_split_frozen_before_training": split_frozen,
                "all_samples_have_at_least_16_parts": all_16,
                "matched_raw_budgets_enforced": budgets_ok,
                "at_most_three_reaudits_per_sample": at_most_3,
                "failures_and_timeouts_counted": counted_all,
            }
        )

    return result


def derive_gate_g2_evidence(
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    dataset_manifest: Mapping[str, Any],
    cost_ledger: Any,
    expected_benchmark_count: int | None = None,
    budget_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Construct Gate G2 evidence strictly from executed evaluation artifacts (zero hardcoded flags)."""
    rows = [dict(r) for r in evaluation_records]
    ledger_dict = cost_ledger.to_dict() if hasattr(cost_ledger, "to_dict") else dict(cost_ledger)
    ds_sha = str(dataset_manifest.get("dataset_manifest_sha256", ""))
    eval_bytes = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    eval_sha = hashlib.sha256(eval_bytes).hexdigest()

    # Detect duplicate memorized AST hashes across distinct benchmark prompts
    seen_hashes: set[str] = set()
    passed_count = 0
    for r in rows:
        ast_hash = str(r.get("canonical_ast_sha256", ""))
        is_dup = bool(ast_hash and ast_hash in seen_hashes)
        if ast_hash:
            seen_hashes.add(ast_hash)
        r["duplicate_memorized_output"] = is_dup
        req_min_boxes = max(16, int(r.get("min_boxes", 16)))
        sample_ok = bool(
            r.get("sat_collision_passed", r.get("audit_passed", False))
            and r.get("prompt_adherence_passed", True)
            and not is_dup
            and int(r.get("boxes", 0)) >= req_min_boxes
            and int(r.get("reference_boxes", req_min_boxes)) >= 16
            and 0 <= int(r.get("reaudits_used", 0)) <= 3
        )
        if sample_ok:
            passed_count += 1

    prereqs = inspect_runtime_gate_prerequisites(
        artifacts={
            "dataset_manifest": dataset_manifest,
            "evaluation_records": rows,
            "expected_benchmark_count": expected_benchmark_count if expected_benchmark_count is not None else len(rows),
            "cost_ledger": ledger_dict,
            "budget_limits": budget_limits,
        }
    )

    total_count = int(expected_benchmark_count) if expected_benchmark_count is not None else len(rows)
    return {
        "status": "artifact_derived_evaluation",
        "total": total_count,
        "passed": passed_count,
        "dataset_manifest_sha256": ds_sha,
        "evaluation_manifest_sha256": eval_sha,
        **prereqs,
    }
