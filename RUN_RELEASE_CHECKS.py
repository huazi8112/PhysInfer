"""One-click local verification for PyCharm or a standard Python interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=ENV, check=True)


if __name__ == "__main__":
    run("-m", "pytest", "-q")
    run("scripts/validate_installation.py")
    run("scripts/audit_release.py")
    run(
        "scripts/reproduce_figures.py",
        "--results",
        "frozen_results",
        "--output",
        "reproduced_figures",
    )
    print("All release checks passed.")
