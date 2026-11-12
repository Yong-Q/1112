#!/usr/bin/env python3
"""Thin wrapper: see `off-polar-run`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling.cli import run_main
raise SystemExit(run_main())
