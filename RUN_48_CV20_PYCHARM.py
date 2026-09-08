"""PyCharm launcher for the fixed-condition CV_ext=20% validation panel.

Open this file in PyCharm and click Run.  Set SMALL_VALIDATION to True for a
quick code check or False for the complete 48-unit/30-repeat experiment.
No command-line arguments or editable installation are required.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


# ---------------------------------------------------------------------------
# User switch
# ---------------------------------------------------------------------------
SMALL_VALIDATION = False


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

output_name = (
    "results_48_cv20_small_validation"
    if SMALL_VALIDATION
    else "results_48_cv20_full"
)
arguments = [
    str(ROOT / "scripts" / "run_selective_panel_48_cv20.py"),
    "--output",
    str(ROOT / output_name),
]
if SMALL_VALIDATION:
    arguments.append("--smoke")

sys.argv = arguments
runpy.run_path(
    str(ROOT / "scripts" / "run_selective_panel_48_cv20.py"),
    run_name="__main__",
)
