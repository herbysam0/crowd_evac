"""Tests for crowd_evac.montecarlo.visual.

The window logic is driven frame by frame with arcade's GL parts mocked, so
most tests need no display.  ``render`` tests open a real window.
"""
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from crowd_evac.montecarlo.config import MonteCarloConfig
from crowd_evac.montecarlo.hazards import parse_hazard_spec
from crowd_evac.montecarlo.results import RunOutcome
from crowd_evac.montecarlo.runner import (
    RunTask,
    derive_seeds,
    run_monte_carlo,
    run_one,
)
from crowd_evac.montecarlo.visual import MonteCarloWindow
from crowd_evac.scenarios.schema import EmergencyEventData

_ROOM = "fixture_minimal"
_AGENTS = 4
_FRAME_S = 1 / 60
_MAX_FRAMES = 50_000
_HAZARD: EmergencyEventData = {
    "tick": 10, "type": "place_panic_source", "pos": [5.0, 4.0],
}


def _tasks(runs: int = 2) -> list[RunTask]:
    """Tasks on the small room; the first carries a hazard."""
    seeds = derive_seeds(5, runs)
    return [
        RunTask(_ROOM, _AGENTS, i, s, (_HAZARD,) if i == 0 else ())
        for i, s in enumerate(seeds)
    ]


def _mock_gl(mocker: MockerFixture) -> None:
    """Replace everything in EvacWindow that needs a GL context."""
    mocker.patch("crowd_evac.application.cli.compute_fit_ppm",
                 return_value=25.0)
    mocker.patch("crowd_evac.application.cli.ArcadeRenderer")
    mocker.patch("arcade.Window.__init__", return_value=None)
    mocker.patch("arcade.Window.set_minimum_size")
    mocker.patch("arcade.Window.close")


def _drive(window: MonteCarloWindow) -> None:
    """Call on_update until the window finishes and closes itself."""
    for _ in range(_MAX_FRAMES):
        if window.finished:
            return
        window.on_update(_FRAME_S)
    pytest.fail("window never closed")


class _Collector:
    """Collect notifications in order."""

    def __init__(self) -> None:
        """Start empty."""
        self.outcomes: list[RunOutcome] = []

    def __call__(self, outcome: RunOutcome) -> None:
        """Record one finished run."""
        self.outcomes.append(outcome)


class TestWindowLogic:
    """Frame-by-frame behaviour with GL mocked."""

    def test_plays_all_runs_and_matches_headless(
        self, mocker: MockerFixture
    ) -> None:
        """Verify automatic start and results equal to headless runs."""
        _mock_gl(mocker)
        tasks = _tasks()
        seen = _Collector()
        window = MonteCarloWindow(tasks, seen, lambda: False)
        _drive(window)
        assert tuple(window.outcomes) == tuple(run_one(t) for t in tasks)
        assert seen.outcomes == window.outcomes

    def test_closes_after_last_run(self, mocker: MockerFixture) -> None:
        """Verify the window closes itself when the last run ends."""
        _mock_gl(mocker)
        window = MonteCarloWindow(_tasks(1), _Collector(), lambda: False)
        _drive(window)
        assert window.finished
        assert len(window.outcomes) == 1
        window.close.assert_called_once()  # type: ignore[attr-defined]

    def test_cancel_between_runs_keeps_finished(
        self, mocker: MockerFixture
    ) -> None:
        """Verify cancelling after one run reports it and stops."""
        _mock_gl(mocker)
        seen = _Collector()
        window = MonteCarloWindow(
            _tasks(3), seen, lambda: len(seen.outcomes) >= 1
        )
        _drive(window)
        assert [o.run_index for o in window.outcomes] == [0]

    def test_mouse_hazards_ignored(self, mocker: MockerFixture) -> None:
        """Verify queued mouse events do not change the run."""
        _mock_gl(mocker)
        tasks = _tasks(1)
        window = MonteCarloWindow(tasks, _Collector(), lambda: False)
        window._input_source.on_mouse_press(100.0, 100.0, 1, 0)
        _drive(window)
        assert window.outcomes[0] == run_one(tasks[0])

    def test_space_does_not_pause(self, mocker: MockerFixture) -> None:
        """Verify SPACE is ignored so the batch cannot stall."""
        _mock_gl(mocker)
        window = MonteCarloWindow(_tasks(1), _Collector(), lambda: False)
        window.on_key_press(32, 0)
        window.on_update(_FRAME_S)
        assert window._sim.tick == 1


@pytest.mark.render
class TestRealWindow:
    """Open an actual arcade window (needs a display)."""

    def test_visual_matches_headless(self) -> None:
        """Verify a visual batch equals the headless batch for one seed."""
        hazard = parse_hazard_spec("x=1..9,y=1..7,tick=0..40")

        def config(visual: bool, workers: int | None) -> MonteCarloConfig:
            """Same batch, headless or visual."""
            return MonteCarloConfig(
                scenario=_ROOM, agents=_AGENTS, runs=2, seed=9,
                hazards=(hazard,), visual=visual, max_workers=workers,
            )

        headless = run_monte_carlo(config(False, 1))
        visual = run_monte_carlo(config(True, None))
        assert visual.outcomes == headless.outcomes

    def test_closing_early_returns_finished_runs(self) -> None:
        """Verify a cancel after run one returns it, flagged cancelled."""
        done: list[int] = []

        class _Count:
            """Listener counting finished runs."""

            def on_run_complete(
                self, outcome: RunOutcome, n: int, total: int
            ) -> None:
                """Record the run."""
                done.append(outcome.run_index)

        config = MonteCarloConfig(
            scenario=_ROOM, agents=_AGENTS, runs=3, seed=9, visual=True
        )
        result = run_monte_carlo(
            config, listener=_Count(), should_cancel=lambda: bool(done)
        )
        assert result.cancelled
        assert [o.run_index for o in result.outcomes] == [0]
