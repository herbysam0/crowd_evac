"""Tests for crowd_evac.dashboard.batch_process.

These start real ``spawn`` child processes on the small fixture room.

Covers:
  - events arrive in order and the final result equals a direct run
  - cancel returns the runs that finished
  - a bad hazard or room gives BatchFailed
  - a killed child gives BatchFailed
  - a headless batch does not import arcade
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from crowd_evac.dashboard.batch_process import (
    LOST_CHILD_MESSAGE,
    BatchEvent,
    BatchFailed,
    BatchFinished,
    BatchProcess,
    RunFinished,
    headless_workers,
)
from crowd_evac.montecarlo.config import HazardSpec, MonteCarloConfig, ValueRange
from crowd_evac.montecarlo.runner import run_monte_carlo

_ROOM = "fixture_minimal"
_TIMEOUT_S = 120.0
_POLL_S = 0.05


def _config(**overrides: object) -> MonteCarloConfig:
    """Small headless config; one worker keeps the child light."""
    base: dict[str, object] = {
        "scenario": _ROOM, "agents": 2, "runs": 3, "seed": 11,
        "max_workers": 1,
    }
    base.update(overrides)
    return MonteCarloConfig(**base)  # type: ignore[arg-type]


def _collect(batch: BatchProcess) -> list[BatchEvent]:
    """Poll until the final event arrives; fail on timeout."""
    events: list[BatchEvent] = []
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        events.extend(batch.poll())
        if events and isinstance(events[-1], (BatchFinished, BatchFailed)):
            return events
        time.sleep(_POLL_S)
    batch._process.kill()  # noqa: SLF001
    pytest.fail("batch never finished")


class TestEvents:
    """Event stream of a normal batch."""

    def test_events_in_order_and_result_matches_direct(self) -> None:
        """Verify progress events, then one result equal to a direct run."""
        config = _config()
        batch = BatchProcess(config)
        batch.start()
        events = _collect(batch)
        runs = [e for e in events if isinstance(e, RunFinished)]
        assert [(e.done, e.total) for e in runs] == [(1, 3), (2, 3), (3, 3)]
        final = events[-1]
        assert isinstance(final, BatchFinished)
        assert not any(isinstance(e, BatchFinished) for e in events[:-1])
        direct = run_monte_carlo(config)
        assert final.result.outcomes == direct.outcomes
        assert final.result.summary == direct.summary
        assert batch.poll() == []
        assert not batch.active

    def test_poll_before_start_is_empty(self) -> None:
        """Verify a batch that was never started reports nothing."""
        batch = BatchProcess(_config())
        assert batch.poll() == []
        assert not batch.active

    def test_headless_gets_default_workers(self) -> None:
        """Verify a missing worker count is filled in, keeping a UI core."""
        batch = BatchProcess(_config(max_workers=None))
        assert batch.config.max_workers == headless_workers()
        assert headless_workers() >= 1

    def test_visual_keeps_worker_setting(self) -> None:
        """Verify a visual batch is left alone."""
        batch = BatchProcess(_config(max_workers=None, visual=True))
        assert batch.config.max_workers is None


class TestCancel:
    """Cancelling keeps finished runs."""

    def test_cancel_returns_finished_runs(self) -> None:
        """Verify cancel after the first run ends the batch, flagged."""
        batch = BatchProcess(_config(runs=40))
        batch.start()
        events: list[BatchEvent] = []
        deadline = time.monotonic() + _TIMEOUT_S
        while not any(isinstance(e, RunFinished) for e in events):
            assert time.monotonic() < deadline
            events.extend(batch.poll())
            time.sleep(_POLL_S)
        batch.cancel()
        events.extend(_collect(batch))
        final = events[-1]
        assert isinstance(final, BatchFinished)
        assert final.result.cancelled
        assert 1 <= len(final.result.outcomes) < 40


class TestFailure:
    """Failures arrive as events, never as exceptions in the dashboard."""

    def test_unknown_room_fails(self) -> None:
        """Verify a missing room gives BatchFailed with a message."""
        batch = BatchProcess(_config(scenario="no_such_room"))
        batch.start()
        final = _collect(batch)[-1]
        assert isinstance(final, BatchFailed)
        assert "no_such_room" in final.message

    def test_hazard_stuck_in_wall_fails(self) -> None:
        """Verify an unsampleable range gives BatchFailed."""
        # fixture_minimal's west wall covers x in [0, 0.2].
        spec = HazardSpec(x=ValueRange(0.0, 0.1), y=ValueRange(1.0, 2.0))
        batch = BatchProcess(_config(hazards=(spec,)))
        batch.start()
        final = _collect(batch)[-1]
        assert isinstance(final, BatchFailed)
        assert final.message

    def test_killed_child_fails(self) -> None:
        """Verify a child that dies silently gives the generic failure."""
        batch = BatchProcess(_config(runs=200))
        batch.start()
        batch._process.kill()  # noqa: SLF001
        final = _collect(batch)[-1]
        assert final == BatchFailed(LOST_CHILD_MESSAGE)
        assert not batch.active


class TestHeadlessImport:
    """A headless batch must not need a display stack."""

    def test_headless_batch_does_not_load_arcade(self) -> None:
        """Verify importing the module and running a batch skips arcade."""
        code = (
            "import sys\n"
            "from crowd_evac.dashboard import batch_process\n"
            "from crowd_evac.montecarlo.config import MonteCarloConfig\n"
            "from crowd_evac.montecarlo.runner import run_monte_carlo\n"
            "run_monte_carlo(MonteCarloConfig(scenario='fixture_minimal',"
            " runs=1, seed=1, max_workers=1))\n"
            "sys.exit(1 if 'arcade' in sys.modules else 0)\n"
        )
        done = subprocess.run([sys.executable, "-c", code], check=False)
        assert done.returncode == 0
