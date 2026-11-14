#!/usr/bin/env python3
"""Recompute supplied planning assumptions. Makes no provider or pricing requests."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/compute_budget.json").read_text())
base = sum(row["hours"] * row["cu_per_hour"] for row in config["rows"])
without_learned = sum(row["hours"] * row["cu_per_hour"] for row in config["rows"] if not row["conditional"])
print(json.dumps({
    "status": config["status"],
    "source_assumption_subtotal": round(base, 2),
    "with_exact_ten_percent_buffer": round(base * (1 + config["buffer_fraction"]), 2),
    "without_conditional_p2_with_buffer": round(without_learned * (1 + config["buffer_fraction"]), 2),
    "note": "The report rounds its buffer. Actual rates, allocation, throughput, and subscription terms require local verification."
}, indent=2))
