"""Tests for crowd_evac.metrics.speed -- moving-only mean speed tally.

Covers:
  - known velocities give a known mean
  - agents at or below the threshold and dead agents are excluded
  - nobody ever moving gives None
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from crowd_evac.domain.constants import MOVING_SPEED_EPS_MPS
from crowd_evac.metrics.speed import MovingSpeedTally


def _snap(speeds: list[float], alive: list[bool]) -> SimpleNamespace:
    """Build a snapshot stand-in with velocities along x."""
    vel = np.zeros((len(speeds), 2))
    vel[:, 0] = speeds
    return SimpleNamespace(
        velocities=vel, alive=np.asarray(alive, dtype=np.bool_)
    )


class TestMovingSpeedTally:
    """Test suite for MovingSpeedTally."""

    def test_mean_over_agents_and_ticks(self) -> None:
        """Mean covers every moving agent sample, not per-tick means."""
        tally = MovingSpeedTally()
        tally.add(_snap([1.0, 3.0], [True, True]))  # type: ignore[arg-type]
        tally.add(_snap([2.0, 0.0], [True, True]))  # type: ignore[arg-type]
        assert tally.mean_mps == pytest.approx(2.0)

    def test_threshold_and_dead_agents_excluded(self) -> None:
        """At-threshold speed and dead agents do not count."""
        tally = MovingSpeedTally()
        tally.add(
            _snap(  # type: ignore[arg-type]
                [MOVING_SPEED_EPS_MPS, 5.0, 1.0], [True, False, True]
            )
        )
        assert tally.mean_mps == pytest.approx(1.0)

    def test_nobody_moving_gives_none(self) -> None:
        """No moving sample yields None, not zero."""
        tally = MovingSpeedTally()
        assert tally.mean_mps is None
        tally.add(_snap([0.0, 0.01], [True, True]))  # type: ignore[arg-type]
        assert tally.mean_mps is None

    def test_empty_snapshot_is_ignored(self) -> None:
        """A snapshot with no agents adds nothing."""
        tally = MovingSpeedTally()
        tally.add(_snap([], []))  # type: ignore[arg-type]
        assert tally.mean_mps is None
