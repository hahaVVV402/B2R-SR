#!/usr/bin/env python3
"""Test one canonical EDSR checkpoint on one paired SR dataset."""

import os
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().with_name("formal_recovery.py")
if not RUNNER.is_file():
    raise SystemExit(f"Missing EDSR evaluator: {RUNNER}")
os.execv(sys.executable, [sys.executable, str(RUNNER), "test-checkpoint", *sys.argv[1:]])
