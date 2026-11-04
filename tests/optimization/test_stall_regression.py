"""Regression: a long single-agent walk must not be cut off by the stall rule.

With the old 200-tick (10 s) limit, ``lecture_hall`` with one agent and
seed 4 stopped early with nobody evacuated although the walk is only a
few seconds longer than the limit.
"""
from __future__ import annotations

from crowd_evac.domain.constants import STALL_TICKS
from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.harness import evaluate

_SCENARIO = "lecture_hall"
_SEED = 4
_MIN_STALL_TICKS = 1200


def test_stall_limit_covers_longest_known_walk() -> None:
    """Verify the limit is at least 60 s (1200 ticks at DT = 0.05 s)."""
    assert STALL_TICKS >= _MIN_STALL_TICKS


def test_single_agent_seed_4_evacuates() -> None:
    """Verify the one-agent run evacuates instead of stalling early."""
    result = evaluate(
        ForceParams.defaults(), _SCENARIO, _SEED, agent_count=1
    )
    assert result.evacuated_fraction == 1.0
