"""Bounded Compiler-Audit + Constrained PGD + Optional Model-Callback Repair Loop (C3).

Implements the preregistered Gate G2 repair policy:
1. Initial parse, validation, SE(3) compilation, direct 15-axis SAT + P1-02 contact audit,
   and prompt-adherence check (charging ``compiler_calls=1``, ``initial_audit_calls=1``).
2. Up to ``max_reaudits <= 3`` repair iterations combining deterministic wrapper/brace
   cleanup, constrained PGD pose/offset projection (``project_program``), and optional
   model-guided repair callback, charging every ``compiler_calls``, ``reaudit_calls``,
   ``projection_steps``, ``discarded_tokens``, and ``repair_tokens`` monotonically to
   ``CostLedger``. Rollbacks never refund cost.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
import re
from typing import Any

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import Program, validate_program
from synccraft.dsl.parser import format_program, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.evaluation.gate_g2 import verify_prompt_adherence
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from synccraft.modeling.pgd_projector import project_program

MAX_ALLOWED_G2_REAUDITS = 3


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _strip_wrappers_and_fences(raw_text: str) -> str:
    """Strip <think>...</think> reasoning blocks and Markdown code fences deterministically."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:blockdsl|dsl|synccraft|text)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Strip leading prose before the first @meta declaration if present
        meta_idx = cleaned.find("@meta")
        if meta_idx > 0:
            cleaned = cleaned[meta_idx:].strip()
    # Balance up to 4 trailing unclosed braces when only closing braces were clipped
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")
    if 0 < (open_braces - close_braces) <= 4:
        cleaned = cleaned + ("\n}" * (open_braces - close_braces))
    return cleaned


def _unpack_callable_output(out: str | tuple[str, int]) -> tuple[str, int]:
    if isinstance(out, tuple) and len(out) == 2:
        text = str(out[0])
        toks = int(out[1])
        return text, max(_estimate_tokens(text), toks) if text else max(0, toks)
    text = str(out)
    return text, _estimate_tokens(text)


def _call_initial_generator(
    fn: Callable[..., str | tuple[str, int]],
    prompt_spec: Mapping[str, Any] | str | None,
) -> tuple[str, int]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if len(positional) == 0 and not has_varargs:
        return _unpack_callable_output(fn())
    if len(positional) == 1:
        return _unpack_callable_output(fn(prompt_spec))
    return _unpack_callable_output(fn(prompt_spec, None))


def _evaluate_candidate_dsl(
    dsl_text: str,
    prompt_spec: Mapping[str, Any] | str | None,
    *,
    enforce_physical_contact: bool,
) -> tuple[Program | None, Scene | None, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Parse, validate, compile, re-audit serialized roundtrip, and check prompt adherence."""
    try:
        prog = validate_program(parse_program(dsl_text))
        canonical_dsl = format_program(prog)
        prog = validate_program(parse_program(canonical_dsl))
        scene = compile_geometry(prog)
        use_contact = bool(
            enforce_physical_contact and getattr(prog.meta, "audit_profile", "strict") != "authored"
        )
        aud = audit_scene(scene, enforce_physical_contact=use_contact)
        adherence = verify_prompt_adherence(prog, scene, prompt_spec)
        err: str | None = None
        if not aud["passed"]:
            err = f"audit_failed: collisions={len(aud['collision_pairs'])}, contact_violations={len(aud['contact_violations'])}"
        elif not adherence["passed"]:
            err = f"prompt_adherence_failed: {adherence['violations']}"
        return prog, scene, aud, adherence, err
    except Exception as exc:
        return None, None, None, None, f"syntax_or_validation_error: {exc}"


def run_bounded_repair_policy(
    raw_dsl_or_fn: str | Callable[..., str | tuple[str, int]],
    prompt_spec: Mapping[str, Any] | str | None,
    cost_ledger: CostLedger,
    *,
    max_reaudits: int = 3,
    model_repair_fn: Callable[[str, dict[str, Any], int], str | tuple[str, int]] | None = None,
    max_projection_steps_per_reaudit: int = 8,
    enforce_physical_contact: bool = True,
) -> dict[str, Any]:
    """Execute preregistered bounded repair loop (<= 3 re-audits) with monotonic CostLedger accounting."""
    if not isinstance(cost_ledger, CostLedger):
        raise TypeError("cost_ledger must be a CostLedger instance")
    if type(max_reaudits) is not int or not (0 <= max_reaudits <= MAX_ALLOWED_G2_REAUDITS):
        raise ValueError(
            f"Gate G2 repair policy enforces 0 <= max_reaudits <= {MAX_ALLOWED_G2_REAUDITS}, got {max_reaudits!r}"
        )
    if type(max_projection_steps_per_reaudit) is not int or max_projection_steps_per_reaudit < 1:
        raise ValueError("max_projection_steps_per_reaudit must be a positive integer")

    active_repair_fn = model_repair_fn
    if isinstance(raw_dsl_or_fn, str):
        cur_dsl = raw_dsl_or_fn
    elif callable(raw_dsl_or_fn):
        cur_dsl, gen_tokens = _call_initial_generator(raw_dsl_or_fn, prompt_spec)
        if gen_tokens > 0:
            cost_ledger.record("accepted_tokens", gen_tokens)
        if active_repair_fn is None:
            sig = inspect.signature(raw_dsl_or_fn)
            pos = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(pos) >= 3:
                active_repair_fn = raw_dsl_or_fn  # type: ignore[assignment]
    else:
        raise TypeError("raw_dsl_or_fn must be a raw Block-DSL str or generation callable")

    # 1. Initial compilation + initial audit (Attempt 0)
    cost_ledger.record("compiler_calls", 1)
    cost_ledger.record("initial_audit_calls", 1)
    cur_prog, cur_scene, cur_aud, cur_adh, cur_err = _evaluate_candidate_dsl(
        cur_dsl,
        prompt_spec,
        enforce_physical_contact=enforce_physical_contact,
    )

    cur_audit_ok = bool(cur_aud is not None and cur_aud["passed"])
    cur_adh_ok = bool(cur_adh is not None and cur_adh["passed"])
    initial_passed = bool(cur_audit_ok and cur_adh_ok)
    reaudits_used = 0

    history: list[dict[str, Any]] = [
        {
            "attempt": 0,
            "stage": "initial_audit",
            "reaudits_used": 0,
            "syntax_valid": cur_prog is not None,
            "audit_passed": cur_audit_ok,
            "prompt_adherence_passed": cur_adh_ok,
            "error": cur_err,
            "cost_snapshot": cost_ledger.to_dict(),
        }
    ]

    tried_wrapper_cleanup = False

    # 2. Bounded repair iterations (strictly <= max_reaudits <= 3)
    while not (cur_audit_ok and cur_adh_ok) and reaudits_used < max_reaudits:
        next_attempt_idx = reaudits_used + 1
        feedback = {
            "attempt": next_attempt_idx,
            "syntax_valid": cur_prog is not None,
            "error": cur_err,
            "collision_pairs": list(cur_aud["collision_pairs"]) if cur_aud is not None else [],
            "contact_violations": list(cur_aud["contact_violations"]) if cur_aud is not None else [],
            "prompt_violations": list(cur_adh["violations"]) if cur_adh is not None else [],
        }

        stage_label = "noop"
        candidate_dsl: str | None = None

        if cur_prog is None:
            # Parse/validation failure: try deterministic wrapper/brace cleanup first, else model_repair_fn
            if not tried_wrapper_cleanup:
                tried_wrapper_cleanup = True
                cleaned = _strip_wrappers_and_fences(cur_dsl)
                if cleaned != cur_dsl and cleaned:
                    candidate_dsl = cleaned
                    stage_label = "deterministic_syntax_cleanup"

            if candidate_dsl is None and active_repair_fn is not None:
                discard_est = _estimate_tokens(cur_dsl)
                if discard_est > 0:
                    cost_ledger.record("discarded_tokens", discard_est)
                rep_dsl, rep_toks = _unpack_callable_output(active_repair_fn(cur_dsl, feedback, next_attempt_idx))
                cost_ledger.record("repair_tokens", max(1, rep_toks))
                candidate_dsl = _strip_wrappers_and_fences(rep_dsl)
                stage_label = "model_syntax_repair"

            if candidate_dsl is None:
                break

        elif not cur_audit_ok:
            # Valid AST failing SAT or P1-02 physical contact -> run constrained PGD projection
            pgd_res = project_program(
                cur_prog,
                max_steps=max_projection_steps_per_reaudit,
                enforce_physical_contact=enforce_physical_contact,
                cost_ledger=cost_ledger,
            )
            projected_dsl = str(pgd_res["final_dsl"])
            if pgd_res["passed"] or active_repair_fn is None:
                if not pgd_res["passed"] and projected_dsl == format_program(cur_prog):
                    # PGD made zero progress and no model repair callback exists
                    break
                candidate_dsl = projected_dsl
                stage_label = "pgd_geometric_projection"
            else:
                discard_est = _estimate_tokens(cur_dsl)
                if discard_est > 0:
                    cost_ledger.record("discarded_tokens", discard_est)
                rep_dsl, rep_toks = _unpack_callable_output(
                    active_repair_fn(projected_dsl, feedback, next_attempt_idx)
                )
                cost_ledger.record("repair_tokens", max(1, rep_toks))
                candidate_dsl = _strip_wrappers_and_fences(rep_dsl)
                stage_label = "pgd_plus_model_repair"

        else:
            # Valid AST & SAT passed, but failed prompt adherence (e.g. < 16 boxes or wrong domain)
            if active_repair_fn is None:
                break
            discard_est = _estimate_tokens(cur_dsl)
            if discard_est > 0:
                cost_ledger.record("discarded_tokens", discard_est)
            rep_dsl, rep_toks = _unpack_callable_output(active_repair_fn(cur_dsl, feedback, next_attempt_idx))
            cost_ledger.record("repair_tokens", max(1, rep_toks))
            candidate_dsl = _strip_wrappers_and_fences(rep_dsl)
            stage_label = "model_prompt_adherence_repair"

        # Charge compiler call + re-audit monotonically BEFORE inspecting outcome
        cost_ledger.record("compiler_calls", 1)
        cost_ledger.record("reaudit_calls", 1)
        reaudits_used += 1
        cur_dsl = candidate_dsl

        cur_prog, cur_scene, cur_aud, cur_adh, cur_err = _evaluate_candidate_dsl(
            cur_dsl,
            prompt_spec,
            enforce_physical_contact=enforce_physical_contact,
        )
        cur_audit_ok = bool(cur_aud is not None and cur_aud["passed"])
        cur_adh_ok = bool(cur_adh is not None and cur_adh["passed"])

        history.append(
            {
                "attempt": reaudits_used,
                "stage": stage_label,
                "reaudits_used": reaudits_used,
                "syntax_valid": cur_prog is not None,
                "audit_passed": cur_audit_ok,
                "prompt_adherence_passed": cur_adh_ok,
                "error": cur_err,
                "cost_snapshot": cost_ledger.to_dict(),
            }
        )

    final_passed = bool(cur_audit_ok and cur_adh_ok and reaudits_used <= MAX_ALLOWED_G2_REAUDITS)
    final_canonical_dsl = format_program(cur_prog) if cur_prog is not None else cur_dsl
    return {
        "passed": final_passed,
        "syntax_valid": cur_prog is not None,
        "audit_passed": cur_audit_ok,
        "prompt_adherence_passed": cur_adh_ok,
        "initial_passed": initial_passed,
        "reaudits_used": reaudits_used,
        "max_reaudits": max_reaudits,
        "at_most_three_reaudits_enforced": reaudits_used <= MAX_ALLOWED_G2_REAUDITS,
        "final_dsl": final_canonical_dsl,
        "program": cur_prog,
        "scene": cur_scene,
        "audit_report": cur_aud,
        "prompt_adherence": cur_adh,
        "boxes": len(cur_scene.boxes) if cur_scene is not None else 0,
        "bones": len(cur_scene.bones) if cur_scene is not None else 0,
        "canonical_ast_sha256": cur_adh["canonical_ast_sha256"] if cur_adh is not None else "",
        "failure_reason": None if final_passed else (cur_err or "repair_exhausted"),
        "cost_snapshot": cost_ledger.to_dict(),
        "history": history,
    }
