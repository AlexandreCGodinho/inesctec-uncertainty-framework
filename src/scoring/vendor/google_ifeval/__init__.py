"""
Google Research Instruction Following Evaluation (IFEval) vendored harness.

Upstream: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Commit: e6890f85757dd84e27ca6df2dd30651dafad28e0 (2026-07-22)
Deterministic configuration: DetectorFactory.seed = 0
"""

import sys
from pathlib import Path
from langdetect import DetectorFactory

# Enforce deterministic behavior across all langdetect evaluations
DetectorFactory.seed = 0

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from instruction_following_eval import (  # noqa: E402
    evaluation_lib,
    instructions,
    instructions_registry,
    instructions_util,
)

__all__ = [
    "evaluation_lib",
    "instructions",
    "instructions_registry",
    "instructions_util",
]
