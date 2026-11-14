import unittest
from dataclasses import FrozenInstanceError

from synccraft.evaluation.budget import CostLedger
from synccraft.evaluation.gate_g2 import REQUIRED_CHECKS, decide_gate


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


if __name__ == "__main__":
    unittest.main()
