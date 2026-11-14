#!/usr/bin/env bash
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
cd "$repo_dir"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check.py
printf '\nReady. Run: .venv/bin/python scripts/demo.py\n'
