"""Run a Monte Carlo batch of evacuation simulations.

Usage (PowerShell, .venv activated):
    python scripts/monte_carlo.py
    python scripts/monte_carlo.py -s lecture_hall --agents 150 --runs 50 --seed 7
    python scripts/monte_carlo.py --agents 150 --hazard x=10..40,y=2..28,tick=0..400,source_type=fire
    python scripts/monte_carlo.py --runs 5 --visual

See ``crowd_evac.montecarlo.cli`` for the flags.
"""
from __future__ import annotations

import sys

from crowd_evac.montecarlo.cli import main

# The guard is required: worker processes re-import this file on Windows.
if __name__ == "__main__":
    sys.exit(main())
