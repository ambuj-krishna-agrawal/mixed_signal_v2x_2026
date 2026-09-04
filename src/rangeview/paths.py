"""Canonical project paths — used everywhere so nothing hardcodes CWD or module-local paths.

Relies on this file's location under `src/rangeview/`, so the project root is
`parents[2]` regardless of where the script is invoked from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS    = PROJECT_ROOT / "artifacts"
DEVKIT       = PROJECT_ROOT / "devkit_sources"
SCORING      = PROJECT_ROOT / "scoring_program"
DATA_RAW     = PROJECT_ROOT / "data_raw"
