from dataclasses import FrozenInstanceError, replace
import unittest

from synccraft.dsl import (
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
    TrackDecl,
    format_program,
    parse_program,
    validate_program,
)
from synccraft.evaluation.budget import CostLedger
from synccraft.evaluation.gate_g2 import (
    REQUIRED_CHECKS,
    decide_gate,
    derive_gate_g2_evidence,
    inspect_runtime_gate_prerequisites,
    verify_prompt_adherence,
)
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.modeling import (
    project_program,
    run_bounded_repair_policy,
    verify_torch_se3_parity,
)


def _build_dense_symmetric_wyvern_program(*, num_spine_plates: int = 6) -> Program:
    """Construct a valid >= 16-box bilateral-symmetric biological rig for Gate G2 / H5 tests."""
    items: list[Box | Locator | Bone] = [
        Box("thorax_core", (6, 5, 8), "scales"),
        Locator("rider_mount", offset=(0.0, 2.5, 0.0), role="rider_seat"),
    ]
    prev = "thorax_core"
    for idx in range(num_spine_plates):
        pname = f"osteoderm_plate_{idx}"
        items.append(
            Box(
                pname,
                (2, 1, 2),
                "horn_trim",
                place=Place(prev, (0.0, 1.0, 0.0) if idx == 0 else (0.0, 0.0, 1.0), (0.0, -1.0, 0.0) if idx == 0 else (0.0, 0.0, -1.0)),
            )
        )
        prev = pname

    # Mirrored wings (2x3 = 6 boxes) + mirrored legs (2x2 = 4 boxes) -> 1 + 6 + 6 + 4 = 17 boxes!
    items.extend(
        [
            Bone(
                "wing_spar_l",
                attach=Attach("thorax_core", (-1.0, 0.6, -0.2), (1.0, 0.0, 0.0)),
                mirror_name="wing_spar_r",
                items=[
                    Box("wing_spar_inner", (5, 1, 4), "scales"),
                    Bone(
                        "patagium_tip_l",
                        attach=Attach("wing_spar_inner", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                        items=[
                            Box("wing_membrane_outer", (5, 1, 4), "horn_trim"),
                            Box("wing_claw", (1, 1, 2), "horn_trim", place=Place("wing_membrane_outer", (-1.0, 0.0, -1.0), (1.0, 0.0, 1.0))),
                        ],
                    ),
                ],
            ),
            Bone(
                "hind_thigh_l",
                attach=Attach("thorax_core", (-1.0, -0.6, 0.5), (1.0, 1.0, 0.0)),
                mirror_name="hind_thigh_r",
                items=[
                    Box("thigh_seg", (2, 4, 2), "scales"),
                    Bone(
                        "hind_shank_l",
                        attach=Attach("thigh_seg", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                        items=[Box("shank_seg", (2, 4, 2), "horn_trim")],
                    ),
                ],
            ),
        ]
    )
    return validate_program(
        Program(
            Meta("entity", "bedrock_entity", 16),
            [
                Palette("scales", (0.52, 0.11, 145.0), 6),
                Palette("horn_trim", (0.74, 0.05, 85.0), 4),
            ],
            Bone("wyvern_root", pivot_local=(0.0, 10.0, 0.0), items=items),
            materials=(MaterialDecl("scales", "keratin_scales", roughness=0.38),),
            behavior=BehaviorDecl(
                clips=(
                    ClipDecl(
                        "soar_flap",
                        duration_s=1.5,
                        tracks=(TrackDecl("wing_spar_l", "flap", "z", amp_deg=18.0, freq_hz=0.667),),
                    ),
                )
            ),
        )
    )


class BudgetGateTests(unittest.TestCase):
    def test_direct_constructor_and_mutation_cannot_refund_cost(self):
        with self.assertRaises(ValueError):
            CostLedger(discarded_tokens=-100)
        with self.assertRaises(ValueError):
            CostLedger(accepted_tokens=True)
        ledger = CostLedger(accepted_tokens=10)
        with self.assertRaises(FrozenInstanceError):
            ledger.accepted_tokens = 0

    def test_discarded_and_repair_tokens_consume_budget(self):
        ledger = CostLedger()
        ledger.record("accepted_tokens", 20)
        ledger.record("discarded_tokens", 10)
        ledger.record("repair_tokens", 7)
        self.assertEqual(ledger.total_generated_tokens, 37)
        self.assertFalse(ledger.within({"total_generated_tokens": 36}))
        self.assertTrue(ledger.within({"total_generated_tokens": 37}))
        with self.assertRaises(ValueError):
            ledger.record("discarded_tokens", -10)

    def test_empty_evidence_never_enables_refiner(self):
        self.assertEqual(decide_gate({})["decision"], "blocked")

    def test_ninety_percent_boundary_and_missing_audit(self):
        evidence = {key: True for key in REQUIRED_CHECKS}
        evidence.update(total=100, passed=90, dataset_manifest_sha256="a"*64,
                        evaluation_manifest_sha256="b"*64)
        self.assertEqual(decide_gate(evidence)["decision"], "skip_learned_refiner")
        evidence["passed"] = 89
        self.assertEqual(decide_gate(evidence)["decision"], "eligible_for_learned_refiner")
        evidence["full_static_audit_implemented"] = False
        self.assertEqual(decide_gate(evidence)["decision"], "blocked")

    def test_c3_torch_numpy_se3_parity_and_post_serialization_pgd_reaudit(self) -> None:
        parity = verify_torch_se3_parity(seed=42)
        self.assertTrue(parity["passed"], msg=str(parity))
        self.assertLess(parity["max_matrix_error"], 1e-12)
        self.assertLess(parity["max_grad_fd_error"], 1e-5)

        # Create a 2-arm rig with an explicit OBB collision + contact gap; verify PGD fixes it
        # without changing discrete integer box sizes and passes post-serialization roundtrip re-audit
        defective_prog = validate_program(
            Program(
                Meta("block_multipart", "bbmodel_5", 16),
                [Palette("m", (0.6, 0.06, 210.0), 4)],
                Bone(
                    "root",
                    pivot_local=(0.0, 4.0, 0.0),
                    items=[
                        Box("pedestal", (6, 4, 6), "m"),
                        Box(
                            "cap_block",
                            (4, 2, 4),
                            "m",
                            place=Place("pedestal", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), offset=(0.0, 0.35, 0.0)),
                        ),
                        Bone(
                            "arm_a",
                            rest_euler=(0.0, 0.0, -95.0),
                            attach=Attach("pedestal", (1.0, 0.8, 0.0), (-1.0, 0.0, 0.0)),
                            items=[Box("beam_a", (5, 2, 2), "m")],
                        ),
                    ],
                ),
            )
        )
        orig_sizes = {b.name: tuple(b.size) for b in compile_geometry(defective_prog).boxes}
        ledger = CostLedger()
        res = project_program(defective_prog, max_steps=16, cost_ledger=ledger)
        self.assertTrue(res.passed, msg=str(res.to_dict()))
        self.assertTrue(res.serialized_reaudit_passed)
        self.assertGreaterEqual(ledger.projection_steps, 1)

        repaired_scene = compile_geometry(parse_program(res.final_dsl))
        new_sizes = {b.name: tuple(b.size) for b in repaired_scene.boxes}
        self.assertEqual(orig_sizes, new_sizes)

    def test_c3_bounded_repair_loop_enforces_max_three_reaudits_and_monotonic_ledger(self) -> None:
        # 1. Reject max_reaudits > 3 (Gate G2 hard ceiling)
        with self.assertRaises(ValueError):
            run_bounded_repair_policy("", None, CostLedger(), max_reaudits=4)

        wyvern = _build_dense_symmetric_wyvern_program()
        clean_dsl = format_program(wyvern)
        prompt_spec = {
            "prompt": "Generate a symmetric biological wyvern entity with at least 16 parts, @materials, and @behavior.",
            "domain": "biological",
            "archetype": "entity",
            "target_profile": "bedrock_entity",
            "min_boxes": 16,
            "requires_symmetry": True,
            "requires_materials": True,
            "requires_behavior": True,
        }

        # 2. Wrapped markdown + <think> + missing final closing brace repaired in 1 deterministic re-audit
        wrapped = f"<think>Drafting 17-part wyvern rig...</think>\n```blockdsl\n{clean_dsl.rstrip()[:-1]}\n```"
        ledger_wrap = CostLedger()
        rep_wrap = run_bounded_repair_policy(wrapped, prompt_spec, ledger_wrap, max_reaudits=3)
        self.assertTrue(rep_wrap["passed"], msg=str(rep_wrap))
        self.assertFalse(rep_wrap["initial_passed"])
        self.assertEqual(rep_wrap["reaudits_used"], 1)
        self.assertEqual(ledger_wrap.initial_audits, 1)
        self.assertEqual(ledger_wrap.reaudits, 1)
        self.assertEqual(ledger_wrap.compiler_calls, 2)

        # 3. Colliding pose repaired via PGD within <= 3 re-audits
        folded_items = [
            replace(it, rest_euler=(0.0, 0.0, 125.0)) if (isinstance(it, Bone) and it.name == "wing_spar_l") else it
            for it in wyvern.root.items
        ]
        colliding_wyvern = validate_program(replace(wyvern, root=replace(wyvern.root, items=folded_items)))
        ledger_pgd = CostLedger()
        rep_pgd = run_bounded_repair_policy(
            format_program(colliding_wyvern),
            prompt_spec,
            ledger_pgd,
            max_reaudits=3,
            max_projection_steps_per_reaudit=12,
        )
        self.assertTrue(rep_pgd["passed"], msg=str(rep_pgd))
        self.assertLessEqual(rep_pgd["reaudits_used"], 3)
        self.assertGreaterEqual(ledger_pgd.projection_steps, 1)
        self.assertEqual(ledger_pgd.initial_audit_calls, 1)
        self.assertEqual(ledger_pgd.reaudit_calls, rep_pgd["reaudits_used"])

        # 4. Persistently failing model_repair_fn halts strictly at 3 re-audits and charges every attempt monotonically
        snapshots: list[int] = []

        def broken_repair_cb(prev_dsl: str, feedback: dict, attempt_idx: int) -> tuple[str, int]:
            snapshots.append(attempt_idx)
            return f"invalid_dsl_attempt_{attempt_idx}", 25

        ledger_fail = CostLedger()
        rep_fail = run_bounded_repair_policy(
            "completely broken dsl {{{",
            prompt_spec,
            ledger_fail,
            max_reaudits=3,
            model_repair_fn=broken_repair_cb,
        )
        self.assertFalse(rep_fail["passed"])
        self.assertEqual(rep_fail["reaudits_used"], 3)
        self.assertEqual(ledger_fail.initial_audit_calls, 1)
        self.assertEqual(ledger_fail.reaudit_calls, 3)
        self.assertEqual(ledger_fail.compiler_calls, 4)
        self.assertGreater(ledger_fail.discarded_tokens, 0)
        self.assertGreaterEqual(ledger_fail.repair_tokens, 50)
        # Verify monotonic non-decreasing total tokens across history snapshots
        hist_totals = [h["cost_snapshot"]["total_generated_tokens"] for h in rep_fail["history"]]
        self.assertEqual(hist_totals, sorted(hist_totals))

    def test_h5_verify_prompt_adherence_rejects_shortcuts_and_memorized_assets(self) -> None:
        wyvern = _build_dense_symmetric_wyvern_program()
        wyvern_scene = compile_geometry(wyvern)
        self.assertEqual(len(wyvern_scene.boxes), 17)

        valid_check = verify_prompt_adherence(
            wyvern,
            wyvern_scene,
            {
                "prompt": "Create a winged biological wyvern with bilateral symmetry, >= 16 parts, @materials, and @behavior.",
                "domain": "biological",
                "archetype": "entity",
                "target_profile": "bedrock_entity",
                "reference_boxes": 20,
                "requires_symmetry": True,
                "requires_materials": True,
                "requires_behavior": True,
                "requires_locators": True,
            },
        )
        self.assertTrue(valid_check["passed"], msg=str(valid_check))

        # 1. Under-parted 2-box valid asset MUST fail dense >=16 reference/prompt check
        tiny_prog = validate_program(
            Program(
                Meta("entity", "bedrock_entity", 16),
                [Palette("p", (0.5, 0.1, 120.0), 4)],
                Bone(
                    "r",
                    pivot_local=(0.0, 4.0, 0.0),
                    items=[
                        Box("body", (4, 4, 6), "p"),
                        Bone("leg_l", attach=Attach("body", (-1.0, -1.0, 0.0), (1.0, 1.0, 0.0)), mirror_name="leg_r", items=[Box("shin", (2, 4, 2), "p")]),
                    ],
                ),
            )
        )
        tiny_adh = verify_prompt_adherence(
            tiny_prog,
            None,
            {"prompt": "Generate a quadruped creature with at least 16 parts.", "reference_boxes": 18},
        )
        self.assertFalse(tiny_adh["passed"])
        self.assertTrue(any("min_boxes_not_met" in v for v in tiny_adh["violations"]))

        # 2. Memorized valid 17-box biological wyvern emitted for an industrial excavator prompt MUST fail
        cross_domain = verify_prompt_adherence(
            wyvern,
            wyvern_scene,
            {
                "prompt": "Build an articulated hydraulic excavator with cab_hull and dipper.",
                "domain": "vehicles",
                "family_id": "excavator",
                "min_boxes": 16,
            },
        )
        self.assertFalse(cross_domain["passed"])
        self.assertTrue(
            any("domain_semantics_mismatch" in v or "semantic_subject_mismatch" in v for v in cross_domain["violations"]),
            msg=str(cross_domain["violations"]),
        )

        # 3. Same-domain anatomical mismatch: winged wyvern emitted for a wingless armoured rhino prompt MUST fail
        same_domain_conflict = verify_prompt_adherence(
            wyvern,
            wyvern_scene,
            {
                "prompt": "Generate an armoured terrestrial rhino quadruped with >= 16 parts.",
                "domain": "biological",
                "min_boxes": 16,
            },
        )
        self.assertFalse(same_domain_conflict["passed"])
        self.assertTrue(any("anatomical_mismatch" in v for v in same_domain_conflict["violations"]))

        # 4. Duplicate memorized AST hash across distinct benchmark prompts MUST fail
        dup_check = verify_prompt_adherence(
            wyvern,
            wyvern_scene,
            {
                "prompt": "Create another distinct wyvern variant with >= 16 parts.",
                "min_boxes": 16,
                "forbidden_ast_sha256s": [valid_check["canonical_ast_sha256"]],
            },
        )
        self.assertFalse(dup_check["passed"])
        self.assertTrue(any("memorized_or_duplicate_completion" in v for v in dup_check["violations"]))

    def test_c3_runtime_gate_prerequisites_and_artifact_derived_evidence(self) -> None:
        prereqs = inspect_runtime_gate_prerequisites()
        self.assertTrue(prereqs["repair_pgd_policy_preregistered"])
        self.assertTrue(prereqs["baseline_implementations_verified"])
        # Honest reporting: static audit still tracks external runtime/grip coverage gaps
        self.assertFalse(prereqs["full_static_audit_implemented"])

        wyvern = _build_dense_symmetric_wyvern_program()
        sha_1 = verify_prompt_adherence(wyvern, None, None)["canonical_ast_sha256"]
        wyvern_2 = _build_dense_symmetric_wyvern_program(num_spine_plates=7)
        sha_2 = verify_prompt_adherence(wyvern_2, None, None)["canonical_ast_sha256"]

        ledger = CostLedger(
            accepted_tokens=400,
            compiler_calls=3,
            initial_audit_calls=3,
            reaudit_calls=2,
            projection_steps=6,
        )
        ds_manifest = {
            "dataset_manifest_sha256": "c" * 64,
            "seen_families": ["biped_guard"],
            "unseen_families": ["wyvern_alpha"],
        }
        eval_rows = [
            {
                "sample_id": "g2_001",
                "reference_boxes": 18,
                "boxes": 17,
                "sat_collision_passed": True,
                "prompt_adherence_passed": True,
                "reaudits_used": 1,
                "canonical_ast_sha256": sha_1,
            },
            {
                # Duplicate memorized completion of sample 1 -> must NOT count as passed!
                "sample_id": "g2_002",
                "reference_boxes": 20,
                "boxes": 17,
                "sat_collision_passed": True,
                "prompt_adherence_passed": True,
                "reaudits_used": 0,
                "canonical_ast_sha256": sha_1,
            },
            {
                # Failed syntax completion (0 boxes) -> counted in denominator while benchmark remains >=16-part valid
                "sample_id": "g2_003",
                "reference_boxes": 19,
                "boxes": 0,
                "sat_collision_passed": False,
                "prompt_adherence_passed": False,
                "reaudits_used": 1,
                "canonical_ast_sha256": "",
            },
        ]
        ev = derive_gate_g2_evidence(eval_rows, dataset_manifest=ds_manifest, cost_ledger=ledger, expected_benchmark_count=3)
        self.assertEqual(ev["total"], 3)
        self.assertEqual(ev["passed"], 1)
        self.assertTrue(ev["heldout_split_frozen_before_training"])
        self.assertTrue(ev["all_samples_have_at_least_16_parts"])
        self.assertTrue(ev["at_most_three_reaudits_per_sample"])
        self.assertTrue(ev["failures_and_timeouts_counted"])
        self.assertTrue(ev["matched_raw_budgets_enforced"])
        # Gate decision remains blocked until full_static_audit_implemented is true
        self.assertEqual(decide_gate(ev)["decision"], "blocked")
        _ = sha_2


if __name__ == "__main__":
    unittest.main()
