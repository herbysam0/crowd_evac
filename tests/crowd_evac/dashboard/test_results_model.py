"""Tests for crowd_evac.dashboard.results_model.

Covers: one run, batch, incomplete single run, no complete runs, cancelled,
and the seed line.
"""
from __future__ import annotations

from crowd_evac.dashboard.results_model import NO_VALUE, build_view
from crowd_evac.domain.constants import DT
from crowd_evac.montecarlo.results import RunOutcome, make_outcome
from crowd_evac.scenarios.schema import EmergencyEventData

_HAZARD: EmergencyEventData = {
    "tick": 0, "type": "place_panic_source", "pos": [1.0, 1.0],
}


def _run(
    index: int, exits: tuple[int, ...], total: int = 2,
    speed: float | None = 1.0,
) -> RunOutcome:
    """Outcome with one hazard firing at tick 0."""
    return make_outcome(index, 100 + index, (_HAZARD,), total, exits, speed)


class TestSingle:
    """One run shows its value."""

    def test_complete_run_shows_time_and_speed(self) -> None:
        """Verify time and velocity for a finished single run."""
        view = build_view(
            [_run(0, (0, 2))], total_runs=1, single=True, base_seed=7,
            seed_was_random=False, finished=True,
        )
        assert view.time_lines == (f"{2 * DT:.1f} s",)
        assert view.speed_lines == ("1.00 m/s",)
        assert view.status == "Done 1/1"
        assert view.count_lines == ("hazards fired 1/1",)

    def test_incomplete_run_shows_count_not_time(self) -> None:
        """Verify 'n/m evacuated' replaces the time."""
        view = build_view(
            [_run(0, (0, 1), total=150)], total_runs=1, single=True,
            base_seed=7, seed_was_random=False, finished=True,
        )
        assert view.time_lines == ("1/150 evacuated",)

    def test_nothing_yet(self) -> None:
        """Verify the panel shows placeholders before the first result."""
        view = build_view(
            [], total_runs=1, single=True, base_seed=7,
            seed_was_random=False,
        )
        assert view.time_lines == (NO_VALUE,)
        assert view.speed_lines == (NO_VALUE,)
        assert view.status == "Run 0/1"


class TestBatch:
    """A batch shows statistics."""

    def test_statistics_over_complete_runs(self) -> None:
        """Verify mean, std dev, min and max lines."""
        runs = [_run(0, (0, 2)), _run(1, (0, 0, 2))]
        view = build_view(
            runs, total_runs=20, single=False, base_seed=7,
            seed_was_random=False,
        )
        assert view.status == "Run 2/20"
        assert view.time_lines[0].startswith("mean ")
        assert any(line.startswith("std dev") for line in view.time_lines)
        assert view.time_lines[-2].startswith("min ")
        assert view.time_lines[-1].startswith("max ")
        assert view.count_lines[0] == "complete 2, incomplete 0"

    def test_one_complete_run_has_no_std(self) -> None:
        """Verify std dev is omitted with a single complete run."""
        view = build_view(
            [_run(0, (0, 2))], total_runs=20, single=False, base_seed=7,
            seed_was_random=False,
        )
        assert not any(
            line.startswith("std dev") for line in view.time_lines
        )

    def test_no_complete_runs(self) -> None:
        """Verify the time shows a message but speed still shows."""
        view = build_view(
            [_run(0, (0, 1), speed=0.8)], total_runs=3, single=False,
            base_seed=7, seed_was_random=False,
        )
        assert view.time_lines == ("no complete runs",)
        assert view.speed_lines == ("0.80 m/s",)
        assert view.count_lines[0] == "complete 0, incomplete 1"

    def test_speed_std_for_batch(self) -> None:
        """Verify velocity gains a std dev line with two values."""
        runs = [_run(0, (0, 2), speed=1.0), _run(1, (0, 2), speed=2.0)]
        view = build_view(
            runs, total_runs=2, single=False, base_seed=7,
            seed_was_random=False,
        )
        assert view.speed_lines[0] == "1.50 m/s"
        assert view.speed_lines[1].startswith("std dev")

    def test_cancelled_status(self) -> None:
        """Verify a cancelled batch says how far it got."""
        view = build_view(
            [_run(0, (0, 2))], total_runs=20, single=False, base_seed=7,
            seed_was_random=False, finished=True, cancelled=True,
        )
        assert view.status == "Cancelled after 1/20"

    def test_hazards_fired_summed(self) -> None:
        """Verify fired and planned hazards are totalled over runs."""
        runs = [_run(0, (0, 2)), _run(1, (0, 2))]
        view = build_view(
            runs, total_runs=2, single=False, base_seed=7,
            seed_was_random=False,
        )
        assert view.count_lines[1] == "hazards fired 2/2"


class TestSeed:
    """Seed line."""

    def test_random_seed_marked(self) -> None:
        """Verify a drawn seed is flagged."""
        view = build_view(
            [], total_runs=1, single=True, base_seed=42,
            seed_was_random=True,
        )
        assert view.seed_line == "seed 42 (random)"

    def test_given_seed_plain(self) -> None:
        """Verify a typed seed has no marker."""
        view = build_view(
            [], total_runs=1, single=True, base_seed=42,
            seed_was_random=False,
        )
        assert view.seed_line == "seed 42"
