"""Tests for crowd_evac.montecarlo.report.

Covers: base seed and ``(random)`` label, per-run table, std dev ``n/a``,
incomplete count, sampled-hazard lines, cancelled batches, and that no graph
data reaches the console.
"""
from __future__ import annotations

from crowd_evac.domain.constants import DT
from crowd_evac.montecarlo.config import (
    HazardSpec,
    MonteCarloConfig,
    ValueRange,
)
from crowd_evac.montecarlo.report import format_report
from crowd_evac.montecarlo.results import (
    MonteCarloResult,
    RunOutcome,
    build_curves,
    make_outcome,
    summarize,
)
from crowd_evac.scenarios.schema import EmergencyEventData


def _outcome(
    index: int,
    last_tick: int,
    total: int = 2,
    left: int | None = None,
    hazards: tuple[EmergencyEventData, ...] = (),
) -> RunOutcome:
    """Outcome whose last person leaves at ``last_tick``."""
    left = total if left is None else left
    exits = [0] * last_tick
    exits[-1] = left
    return make_outcome(index, 1000 + index, hazards, total, exits)


def _result(
    outcomes: list[RunOutcome],
    *,
    random: bool = True,
    hazards: tuple[HazardSpec, ...] = (),
    runs: int | None = None,
    elapsed_s: float = 0.0,
) -> MonteCarloResult:
    """Wrap outcomes in a result."""
    config = MonteCarloConfig(
        scenario="lecture_hall", agents=2,
        runs=runs if runs is not None else len(outcomes),
        hazards=hazards,
    )
    return MonteCarloResult(
        config=config,
        base_seed=2718281828,
        seed_was_random=random,
        outcomes=tuple(outcomes),
        summary=summarize(outcomes),
        curves=build_curves(outcomes),
        cancelled=len(outcomes) < config.runs,
        elapsed_s=elapsed_s,
    )


def _hazard_event(x: float, y: float, tick: int) -> EmergencyEventData:
    """Concrete hazard event."""
    return {"tick": tick, "type": "place_panic_source", "pos": [x, y]}


class TestHeader:
    """Header lines."""

    def test_random_seed_labelled(self) -> None:
        """Verify a generated seed is labelled (random)."""
        text = format_report(_result([_outcome(0, 20)]))
        assert "Base seed: 2718281828 (random)" in text

    def test_given_seed_not_labelled(self) -> None:
        """Verify a given seed has no label."""
        text = format_report(_result([_outcome(0, 20)], random=False))
        assert "Base seed: 2718281828\n" in text
        assert "(random)" not in text

    def test_settings_line(self) -> None:
        """Verify scenario, agents, runs and mode are shown."""
        text = format_report(_result([_outcome(0, 20)]))
        assert text.splitlines()[0] == (
            "Scenario: lecture_hall  Agents: 2  Runs: 1  Mode: headless"
        )


class TestTables:
    """Per-run and summary tables."""

    def test_run_table_lists_seed_and_time(self) -> None:
        """Verify each run shows seed, evacuated and last evac time."""
        text = format_report(_result([_outcome(0, 20), _outcome(1, 30)]))
        assert "Run  Seed  Evacuated  Last evac (s)" in text
        assert "1    1000  2/2        1.00" in text
        assert "2    1001  2/2        1.50" in text

    def test_summary_values(self) -> None:
        """Verify mean, std dev, min and max appear to 2 places."""
        text = format_report(_result([_outcome(0, 20), _outcome(1, 40)]))
        assert "Time to last evacuation (2 complete runs)" in text
        # times 1.00 s and 2.00 s -> mean 1.50, sample std 0.71
        assert "1.50      0.71         1.00     2.00" in text

    def test_std_dev_not_available_for_one_complete_run(self) -> None:
        """Verify std dev is n/a with a single complete run."""
        text = format_report(_result([_outcome(0, 20)]))
        assert "n/a" in text

    def test_incomplete_run_counted_and_excluded(self) -> None:
        """Verify incomplete runs are counted, shown as n/m, not averaged."""
        text = format_report(
            _result([_outcome(0, 20), _outcome(1, 400, left=1)])
        )
        assert "1/2" in text
        assert "Incomplete runs: 1 of 2" in text
        assert "(1 complete runs)" in text or "(1 complete run)" in text
        assert f"{400 * DT:.2f}" in text  # shown in the run table

    def test_nobody_evacuated(self) -> None:
        """Verify a run without exits shows a dash and no statistics."""
        silent = make_outcome(0, 5, (), 2, (0, 0, 0))
        text = format_report(_result([silent]))
        assert "0/2" in text
        assert "-" in text
        assert "Incomplete runs: 1 of 1" in text
        assert "Mean (s)" not in text


class TestHazards:
    """Hazard columns and sampled-position lines."""

    def test_no_hazard_column_without_hazards(self) -> None:
        """Verify the Hazards fired column only appears with hazards."""
        assert "Hazards fired" not in format_report(
            _result([_outcome(0, 20)])
        )

    def test_hazards_fired_column(self) -> None:
        """Verify fired / given is shown per run."""
        spec = HazardSpec(x=1.0, y=1.0)
        late = _outcome(0, 20, hazards=(_hazard_event(1, 1, 5000),))
        text = format_report(_result([late], hazards=(spec,)))
        assert "Hazards fired" in text
        assert "0/1" in text

    def test_randomized_hazards_listed_per_run(self) -> None:
        """Verify sampled position and tick are printed for each run."""
        spec = HazardSpec(x=ValueRange(1.0, 9.0), y=2.0, tick=ValueRange(0, 9))
        run = _outcome(0, 20, hazards=(_hazard_event(3.5, 2.0, 7),))
        text = format_report(_result([run], hazards=(spec,)))
        assert "Run 1: hazard 1 at (3.50, 2.00) tick 7" in text

    def test_fixed_hazards_not_listed(self) -> None:
        """Verify fixed hazards add no per-run position lines."""
        spec = HazardSpec(x=1.0, y=2.0)
        run = _outcome(0, 20, hazards=(_hazard_event(1.0, 2.0, 0),))
        assert "hazard 1 at" not in format_report(
            _result([run], hazards=(spec,))
        )


class TestElapsed:
    """Wall-clock time of the batch."""

    def test_elapsed_shown_to_two_places(self) -> None:
        """Verify the elapsed time is printed in seconds."""
        text = format_report(_result([_outcome(0, 20)], elapsed_s=12.3456))
        assert "Elapsed: 12.35 s" in text

    def test_elapsed_comes_last_when_complete(self) -> None:
        """Verify it is the final line of an uncancelled report."""
        text = format_report(_result([_outcome(0, 20)], elapsed_s=1.0))
        assert text.splitlines()[-1] == "Elapsed: 1.00 s"


class TestCancelledAndGraph:
    """Early stop and the dashboard-only graph data."""

    def test_cancelled_batch_reports_progress(self) -> None:
        """Verify a stopped batch says how many runs finished."""
        text = format_report(_result([_outcome(0, 20)], runs=5))
        assert "Stopped early: 1 of 5 runs finished" in text

    def test_no_graph_data_printed(self) -> None:
        """Verify curves never reach the console report."""
        text = format_report(_result([_outcome(0, 20)]))
        assert "curve" not in text.lower()
        assert "graph" not in text.lower()
        assert len(text.splitlines()) < 20
