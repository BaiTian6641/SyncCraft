#!/usr/bin/env python3
"""Run the implemented CPU suite without pytest or an editable installation."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")


def main() -> int:
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    existing_pp = os.environ.get("PYTHONPATH", "")
    pp_parts = [p for p in existing_pp.split(os.pathsep) if p]
    if SRC not in pp_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([SRC, *pp_parts])
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
