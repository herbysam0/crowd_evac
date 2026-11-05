"""Tests for crowd_evac.montecarlo.runner.

Covers:
  - Seeds: given vs random base seed, stable distinct per-run seeds.
  - run_monte_carlo: determinism for any worker count, hazards that never
    fire, listener calls, cancellation, early usage errors.
  - Headless import does not load arcade.
  - Trapped agents: the run ends at the stall limit and is incomplete.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from crowd_evac.application.hazard_args import HazardOutOfRoomError
from crowd_evac.application.rng import SeededRNG
from crowd_evac.domain.constants import STALL_TICKS
from crowd_evac.montecarlo.config import MonteCarloConfig
from crowd_evac.montecarlo.hazards import (
    HazardRangeError,
    parse_hazard_spec,
)
from crowd_evac.montecarlo.results import RunOutcome
from crowd_evac.montecarlo.runner import (
    RunTask,
    derive_seeds,
    resolve_base_seed,
    run_monte_carlo,
    run_one,
)
from crowd_evac.optimization.harness import DEFAULT_MAX_TICKS

_SEED = 7
_RUNS = 3
_ROOM = "fixture_minimal"  # 10 x 8 m: runs take well under a second


class _Recorder:
    """Listener that records every callback."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[int, int, int]] = []

    def on_run_complete(
        self, outcome: RunOutcome, done: int, total: int
    ) -> None:
        """Record ``(run_index, done, total)``."""
        self.calls.append((outcome.run_index, done, total))


def _config(**overrides: object) -> MonteCarloConfig:
    """Small, fast, in-process config."""
    base: dict[str, object] = {
        "scenario": _ROOM, "runs": _RUNS, "seed": _SEED, "max_workers": 1,
    }
    base.update(overrides)
    return MonteCarloConfig(**base)  # type: ignore[arg-type]


class TestSeeds:
    """Base and per-run seeds."""

    def test_given_seed_used(self) -> None:
        """Verify an explicit seed is returned and not flagged random."""
        assert resolve_base_seed(5) == (5, False)

    def test_random_seed_recorded(self) -> None:
        """Verify no seed gives a flagged value in the uint32 range."""
        seed, was_random = resolve_base_seed(None)
        assert was_random
        assert 0 <= seed < 2**32

    def test_run_seeds_stable_and_distinct(self) -> None:
        """Verify per-run seeds repeat and do not collide."""
        seeds = derive_seeds(42, 50)
        assert seeds == derive_seeds(42, 50)
        assert len(set(seeds)) == 50
        assert seeds != derive_seeds(43, 50)

    def test_run_seeds_valid_for_game(self) -> None:
        """Verify each seed is a plain non-negative int the game accepts."""
        for seed in derive_seeds(42, 10):
            assert type(seed) is int
            SeededRNG(seed)

    def test_prefix_is_stable_when_runs_change(self) -> None:
        """Verify asking for fewer runs gives the same first seeds."""
        assert derive_seeds(9, 5) == derive_seeds(9, 8)[:5]


class TestRunMonteCarlo:
    """The dashboard-facing entry point."""

    def test_outcomes_in_run_order_with_seeds(self) -> None:
        """Verify one outcome per run, carrying the derived seeds."""
        result = run_monte_carlo(_config())
        assert [o.run_index for o in result.outcomes] == [0, 1, 2]
        assert tuple(o.seed for o in result.outcomes) == derive_seeds(
            _SEED, _RUNS
        )
        assert result.base_seed == _SEED
        assert not result.seed_was_random
        assert not result.cancelled
        assert result.summary.complete == _RUNS
        assert result.elapsed_s > 0.0
        assert result.curves.per_run.shape[0] == _RUNS

    def test_random_seed_reported(self) -> None:
        """Verify a generated seed is flagged and reproduces the batch."""
        first = run_monte_carlo(_config(seed=None, runs=2))
        assert first.seed_was_random
        again = run_monte_carlo(_config(seed=first.base_seed, runs=2))
        assert first.outcomes == again.outcomes

    @pytest.mark.parametrize("workers", [1, 2])
    def test_same_results_for_any_worker_count(self, workers: int) -> None:
        """Verify worker count never changes results, hazards included."""
        hazard = parse_hazard_spec("x=1..9,y=1..7,tick=0..100")
        reference = run_monte_carlo(
            _config(agents=3, hazards=(hazard,), max_workers=1)
        )
        result = run_monte_carlo(
            _config(agents=3, hazards=(hazard,), max_workers=workers)
        )
        assert result.outcomes == reference.outcomes

    def test_hazard_after_run_end_not_fired(self) -> None:
        """Verify a hazard scheduled past the run is reported as unfired."""
        late = parse_hazard_spec("x=2,y=2,tick=9000")
        early = parse_hazard_spec("x=2,y=2,tick=0")
        result = run_monte_carlo(_config(runs=1, hazards=(late, early)))
        assert result.outcomes[0].hazards_fired == 1
        assert len(result.outcomes[0].hazards) == 2

    def test_listener_called_once_per_run(self) -> None:
        """Verify progress callbacks arrive once per run, counting up."""
        listener = _Recorder()
        run_monte_carlo(_config(), listener=listener)
        assert listener.calls == [(0, 1, 3), (1, 2, 3), (2, 3, 3)]

    def test_cancel_returns_finished_runs(self) -> None:
        """Verify cancelling after one run returns it, flagged cancelled."""
        listener = _Recorder()
        result = run_monte_carlo(
            _config(),
            listener=listener,
            should_cancel=lambda: len(listener.calls) >= 1,
        )
        assert result.cancelled
        assert [o.run_index for o in result.outcomes] == [0]
        assert result.summary.complete == 1

    def test_cancel_before_start(self) -> None:
        """Verify cancelling at once returns an empty, flagged result."""
        result = run_monte_carlo(_config(max_workers=2),
                                 should_cancel=lambda: True)
        assert result.cancelled
        assert result.outcomes == ()

    def test_hazard_outside_room_fails_before_any_run(self) -> None:
        """Verify an out-of-room bound is raised before running."""
        listener = _Recorder()
        spec = parse_hazard_spec("x=0..500,y=1")
        with pytest.raises(HazardOutOfRoomError):
            run_monte_carlo(_config(hazards=(spec,)), listener=listener)
        assert listener.calls == []

    def test_walled_range_fails_before_any_run(self) -> None:
        """Verify an unsampleable range is raised before running."""
        # fixture_minimal's west wall covers x in [0, 0.2].
        spec = parse_hazard_spec("x=0..0.1,y=1..2")
        with pytest.raises(HazardRangeError):
            run_monte_carlo(_config(hazards=(spec,)))


class TestDirectSeed:
    """``direct_seed`` uses the base seed as the spawn seed."""

    def test_single_run_uses_exact_seed(self) -> None:
        """Verify the run's spawn seed is the configured seed itself."""
        result = run_monte_carlo(_config(runs=1, direct_seed=True))
        assert result.outcomes[0].seed == _SEED
        assert result.base_seed == _SEED

    def test_direct_matches_run_one_with_that_seed(self) -> None:
        """Verify the outcome equals a plain run of that seed."""
        result = run_monte_carlo(_config(runs=1, agents=3, direct_seed=True))
        task = RunTask(_ROOM, 3, 0, _SEED, ())
        assert result.outcomes[0] == run_one(task)

    def test_default_path_unchanged(self) -> None:
        """Verify without the flag a single run still derives its seed."""
        result = run_monte_carlo(_config(runs=1))
        assert result.outcomes[0].seed == derive_seeds(_SEED, 1)[0]
        assert result.outcomes[0].seed != _SEED

    def test_more_than_one_run_rejected(self) -> None:
        """Verify direct_seed with several runs raises ValueError."""
        with pytest.raises(ValueError, match="direct_seed"):
            _config(runs=2, direct_seed=True)

    def test_start_paused_needs_one_run(self) -> None:
        """Verify start_paused with several runs raises ValueError."""
        with pytest.raises(ValueError, match="start_paused"):
            _config(runs=2, start_paused=True)

    def test_visual_run_receives_start_paused(
        self, mocker: MockerFixture
    ) -> None:
        """Verify the runner hands start_paused to the visual window."""
        play = mocker.patch(
            "crowd_evac.montecarlo.visual.run_visual", return_value=[]
        )
        run_monte_carlo(
            _config(runs=1, visual=True, direct_seed=True, start_paused=True)
        )
        assert play.call_args.kwargs["start_paused"] is True

    def test_random_seed_is_drawn_and_used(self) -> None:
        """Verify a blank seed is drawn, flagged, and used as spawn seed."""
        result = run_monte_carlo(_config(runs=1, seed=None, direct_seed=True))
        assert result.seed_was_random
        assert result.outcomes[0].seed == result.base_seed


class TestSpeedPassThrough:
    """Headless runs report a moving speed."""

    def test_outcome_has_moving_speed(self) -> None:
        """Verify a finished run carries a positive moving speed."""
        result = run_monte_carlo(_config(runs=1))
        speed = result.outcomes[0].moving_speed_mps
        assert speed is not None and speed > 0.0
        assert result.summary.speed_mean_mps == pytest.approx(speed)


class TestHeadlessImport:
    """Headless batches must not need a display stack."""

    def test_runner_import_does_not_load_arcade(self) -> None:
        """Verify importing the runner leaves arcade unloaded."""
        code = (
            "import sys, crowd_evac.montecarlo.runner;"
            "sys.exit(1 if 'arcade' in sys.modules else 0)"
        )
        done = subprocess.run([sys.executable, "-c", code], check=False)
        assert done.returncode == 0


class TestTrappedAgents:
    """Agents that cannot reach an exit end the run at the stall limit."""

    def test_run_ends_at_stall_limit_and_is_incomplete(
        self, tmp_path: Path
    ) -> None:
        """Verify a sealed-off half leaves the run incomplete, not capped."""
        room = tmp_path / "sealed.json"
        room.write_text(json.dumps(_SEALED_ROOM), encoding="utf-8")
        outcome = run_one(RunTask(room, 12, 0, 1, ()))
        assert not outcome.complete
        assert 0 < outcome.evacuated < outcome.total
        # Ended by the stall rule, long before the tick cap.
        assert len(outcome.exits_per_tick) < DEFAULT_MAX_TICKS
        assert len(outcome.exits_per_tick) >= STALL_TICKS


def _wall(x: float, y: float, w: float, h: float, label: str) -> dict:  # type: ignore[type-arg]
    """One wall rectangle for the scenario JSON."""
    return {"x": x, "y": y, "width": w, "height": h, "label": label}


_SEALED_ROOM = {
    "schema_version": "2.0",
    "name": "sealed",
    "floor_plan": {
        "width_m": 12.0,
        "height_m": 6.0,
        "walls": [
            _wall(0.0, 0.0, 12.0, 0.5, "south"),
            _wall(0.0, 5.5, 12.0, 0.5, "north"),
            _wall(0.0, 0.0, 0.5, 6.0, "west"),
            _wall(11.5, 0.0, 0.5, 6.0, "east"),
            _wall(5.75, 0.0, 0.5, 6.0, "divider"),
        ],
        "obstacles": [],
        "exits": [{"x": 11.5, "y": 3.0, "width_m": 2.0, "side": "east",
                   "capacity_per_second": 5, "label": "exit"}],
    },
}
