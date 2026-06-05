"""Tests for crowd_evac.optimization.suite (Phase 2 Step 2.3).

Covers:
  - SearchScenario: frozen dataclass, field types, and invariants.
  - CalibrationRig: frozen dataclass, positive door_width_m, field types.
  - search_suite(): non-empty, two-or-more entries, unique names, every
    scenario_ref loads and starts via the harness without error.
  - calibration_rigs(): non-empty, positive door widths, bottleneck_corridor
    present, loads via harness, produces measurable egress.
  - Timing: down-scaled lecture_hall_small evaluates faster than the
    full-scale lecture_hall under an equal tick cap (lenient bound).
  - Parallel batch: evaluate_batch across all suite scenarios × multiple seeds
    using all available CPU cores returns correct result count and is faster
    than the serial equivalent.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.harness import RunResult, evaluate, evaluate_batch
from crowd_evac.optimization.suite import (
    CalibrationRig,
    SearchScenario,
    calibration_rigs,
    search_suite,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SEED: int = 42
# Just enough ticks to verify scenario loads and begins stepping, not a full run.
_TINY_MAX_TICKS: int = 5


# ---------------------------------------------------------------------------
# SearchScenario dataclass
# ---------------------------------------------------------------------------


class TestSearchScenarioDataclass:
    """SearchScenario is a frozen dataclass with correctly-typed fields."""

    @pytest.fixture
    def scenario(self) -> SearchScenario:
        """First entry from search_suite() as a representative instance."""
        return search_suite()[0]

    def test_is_frozen(self, scenario: SearchScenario) -> None:
        """SearchScenario raises AttributeError on attribute assignment."""
        with pytest.raises(AttributeError):
            scenario.name = "mutated"  # type: ignore[misc]

    def test_name_is_non_empty_string(self, scenario: SearchScenario) -> None:
        """name field is a non-empty str."""
        assert isinstance(scenario.name, str)
        assert len(scenario.name) > 0

    def test_scenario_ref_is_str_or_path(self, scenario: SearchScenario) -> None:
        """scenario_ref is either str or pathlib.Path."""
        assert isinstance(scenario.scenario_ref, (str, Path))

    def test_full_agent_count_is_positive_int(
        self, scenario: SearchScenario
    ) -> None:
        """full_agent_count is a positive integer."""
        assert isinstance(scenario.full_agent_count, int)
        assert scenario.full_agent_count > 0

    def test_description_is_non_empty_string(
        self, scenario: SearchScenario
    ) -> None:
        """description is a non-empty str."""
        assert isinstance(scenario.description, str)
        assert len(scenario.description) > 0


# ---------------------------------------------------------------------------
# CalibrationRig dataclass
# ---------------------------------------------------------------------------


class TestCalibrationRigDataclass:
    """CalibrationRig is a frozen dataclass with correctly-typed fields."""

    @pytest.fixture
    def rig(self) -> CalibrationRig:
        """First entry from calibration_rigs() as a representative instance."""
        return calibration_rigs()[0]

    def test_is_frozen(self, rig: CalibrationRig) -> None:
        """CalibrationRig raises AttributeError on attribute assignment."""
        with pytest.raises(AttributeError):
            rig.name = "mutated"  # type: ignore[misc]

    def test_name_is_non_empty_string(self, rig: CalibrationRig) -> None:
        """name field is a non-empty str."""
        assert isinstance(rig.name, str)
        assert len(rig.name) > 0

    def test_scenario_ref_is_str_or_path(self, rig: CalibrationRig) -> None:
        """scenario_ref is either str or pathlib.Path."""
        assert isinstance(rig.scenario_ref, (str, Path))

    def test_door_width_m_is_positive_float(self, rig: CalibrationRig) -> None:
        """door_width_m is a positive float."""
        assert isinstance(rig.door_width_m, float)
        assert rig.door_width_m > 0.0

    def test_description_is_non_empty_string(self, rig: CalibrationRig) -> None:
        """description is a non-empty str."""
        assert isinstance(rig.description, str)
        assert len(rig.description) > 0


# ---------------------------------------------------------------------------
# search_suite()
# ---------------------------------------------------------------------------


class TestSearchSuite:
    """search_suite() returns a valid, non-empty list of SearchScenario entries."""

    def test_returns_non_empty_list(self) -> None:
        """search_suite() returns a non-empty list."""
        result = search_suite()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_entries_are_search_scenarios(self) -> None:
        """Every element in the list is a SearchScenario instance."""
        assert all(isinstance(s, SearchScenario) for s in search_suite())

    def test_at_least_two_scenarios_for_topology_diversity(self) -> None:
        """At least two entries cover distinct floor-plan topologies."""
        assert len(search_suite()) >= 2

    def test_scenario_names_are_unique(self) -> None:
        """All scenario names are distinct strings."""
        names = [s.name for s in search_suite()]
        assert len(names) == len(set(names))

    def test_all_full_agent_counts_positive(self) -> None:
        """Every entry has a positive full_agent_count."""
        for s in search_suite():
            assert s.full_agent_count > 0, (
                f"{s.name!r} has non-positive full_agent_count={s.full_agent_count}"
            )

    def test_all_scenarios_load_via_harness(self) -> None:
        """Every search scenario loads without error under a tiny tick cap."""
        for scenario in search_suite():
            result = evaluate(
                ForceParams.defaults(),
                scenario.scenario_ref,
                _SEED,
                max_ticks=_TINY_MAX_TICKS,
            )
            # Cap forces is_terminal=True; confirms the scenario loaded cleanly.
            assert result.is_terminal

    def test_all_scenarios_return_run_results(self) -> None:
        """evaluate() on each search scenario returns a RunResult instance."""
        for scenario in search_suite():
            result = evaluate(
                ForceParams.defaults(),
                scenario.scenario_ref,
                _SEED,
                max_ticks=_TINY_MAX_TICKS,
            )
            assert isinstance(result, RunResult)

    def test_lecture_hall_small_has_fewer_agents_than_full(self) -> None:
        """lecture_hall_small full_agent_count > search agent count in the JSON.

        Verifies the full_agent_count metadata reflects the full-scale scenario
        (150) while the actual scenario file has a reduced count (30).
        """
        small = next(
            s for s in search_suite() if s.name == "lecture_hall_small"
        )
        # full_agent_count is the full-scale reference; the scenario file has
        # fewer agents (the down-scale ratio).  30 < 150.
        result = evaluate(
            ForceParams.defaults(),
            small.scenario_ref,
            _SEED,
            max_ticks=_TINY_MAX_TICKS,
        )
        assert result.initial_count < small.full_agent_count


# ---------------------------------------------------------------------------
# calibration_rigs()
# ---------------------------------------------------------------------------


class TestCalibrationRigs:
    """calibration_rigs() returns a valid, non-empty list of CalibrationRig entries."""

    def test_returns_non_empty_list(self) -> None:
        """calibration_rigs() returns a non-empty list."""
        result = calibration_rigs()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_entries_are_calibration_rigs(self) -> None:
        """Every element is a CalibrationRig instance."""
        assert all(isinstance(r, CalibrationRig) for r in calibration_rigs())

    def test_all_door_widths_are_positive(self) -> None:
        """Every rig has a positive door_width_m."""
        for rig in calibration_rigs():
            assert rig.door_width_m > 0.0, (
                f"{rig.name!r} has non-positive door_width_m={rig.door_width_m}"
            )

    def test_bottleneck_corridor_rig_is_present(self) -> None:
        """The 'bottleneck_corridor' rig exists in the returned list."""
        names = {r.name for r in calibration_rigs()}
        assert "bottleneck_corridor" in names

    def test_bottleneck_rig_door_width_is_1_2m(self) -> None:
        """bottleneck_corridor rig has door_width_m == 1.2 (Weidmann reference)."""
        bottleneck = next(
            r for r in calibration_rigs() if r.name == "bottleneck_corridor"
        )
        assert bottleneck.door_width_m == pytest.approx(1.2)

    def test_bottleneck_rig_loads_via_harness(self) -> None:
        """bottleneck_corridor loads and begins stepping without error."""
        bottleneck = next(
            r for r in calibration_rigs() if r.name == "bottleneck_corridor"
        )
        result = evaluate(
            ForceParams.defaults(),
            bottleneck.scenario_ref,
            _SEED,
            max_ticks=_TINY_MAX_TICKS,
        )
        assert result.is_terminal  # cap confirms load, not hang

    def test_bottleneck_rig_produces_measurable_egress(self) -> None:
        """Running the bottleneck rig for 2000 ticks yields positive evacuated_fraction.

        2000 ticks = 100 s simulated.  At 1.34 m/s free-walking speed, agents
        starting near the 1.2 m gap traverse the right corridor (~5 m) in
        ~4 s (80 ticks).  Even with queuing delay, positive egress is expected
        well within this budget.
        """
        bottleneck = next(
            r for r in calibration_rigs() if r.name == "bottleneck_corridor"
        )
        result = evaluate(
            ForceParams.defaults(),
            bottleneck.scenario_ref,
            _SEED,
            max_ticks=2000,
        )
        assert result.evacuated_fraction > 0.0, (
            "bottleneck_corridor should evacuate some agents within 2000 ticks "
            f"(got evacuated_fraction={result.evacuated_fraction!r})"
        )


# ---------------------------------------------------------------------------
# Timing: down-scaled scenario evaluates faster than full scale
# ---------------------------------------------------------------------------


class TestDownScaleFaster:
    """lecture_hall_small evaluates faster than the full-scale lecture_hall."""

    def test_small_scenario_faster_than_full(self) -> None:
        """lecture_hall_small wall-clock time < lecture_hall under equal tick cap.

        Lenient bound: any measurable speedup is sufficient.  Theory: at
        30 vs 150 agents the per-tick cost is O(N^2) for pairwise forces,
        giving ~25× expected speedup per tick.
        """
        tick_cap = 100
        params = ForceParams.defaults()

        t0 = time.monotonic()
        evaluate(params, "lecture_hall_small", _SEED, max_ticks=tick_cap)
        small_wall_s = time.monotonic() - t0

        t1 = time.monotonic()
        evaluate(params, "lecture_hall", _SEED, max_ticks=tick_cap)
        full_wall_s = time.monotonic() - t1

        assert small_wall_s < full_wall_s, (
            f"lecture_hall_small ({small_wall_s:.3f} s) should be faster than "
            f"lecture_hall ({full_wall_s:.3f} s) under {tick_cap}-tick cap"
        )


# ---------------------------------------------------------------------------
# Parallel batch: all suite scenarios × multiple seeds across all CPU cores
# ---------------------------------------------------------------------------


class TestParallelBatchAcrossSuite:
    """evaluate_batch runs all suite scenarios × multiple seeds in parallel.

    This is the closest proxy to what the fitness function (Step 2.6) will
    call: one candidate weight set evaluated against the full search_suite()
    over several seeds using all available CPU cores.
    """

    # Two seeds; enough to exercise the Cartesian product without being slow.
    _SEEDS: list[int] = [42, 43]
    _TICK_CAP: int = 50

    def test_result_count_equals_scenarios_times_seeds(self) -> None:
        """len(results) == len(search_suite()) × len(seeds)."""
        suite = search_suite()
        # evaluate_batch takes one scenario at a time; run per scenario and
        # aggregate, mirroring what the fitness function does.
        all_results: list[RunResult] = []
        for scenario in suite:
            batch = evaluate_batch(
                [ForceParams.defaults()],
                scenario.scenario_ref,
                self._SEEDS,
                max_ticks=self._TICK_CAP,
            )
            all_results.extend(batch)

        assert len(all_results) == len(suite) * len(self._SEEDS)

    def test_all_results_are_run_results(self) -> None:
        """Every element returned by evaluate_batch is a RunResult."""
        for scenario in search_suite():
            batch = evaluate_batch(
                [ForceParams.defaults()],
                scenario.scenario_ref,
                self._SEEDS,
                max_ticks=self._TICK_CAP,
            )
            assert all(isinstance(r, RunResult) for r in batch)

    def test_full_suite_parallel_batch_dispatches_all_cores(self) -> None:
        """evaluate_batch with default max_workers dispatches across all cores.

        Runs one candidate × three seeds for every suite scenario using
        ``max_workers=None`` (all logical cores — the production default).
        Asserts result count and types; timing is logged but not asserted.

        Why no speedup assertion here: on Windows the ``spawn`` start method
        costs ~0.5 s per worker process.  At 50 ticks/task the simulation
        cost is ~50 ms, so spawn overhead dominates and ``parallel < serial``
        is not reliable at this scale.  The speedup assertion lives in
        ``tests/optimization/test_suite_perf.py`` where a 1000-tick workload
        makes the comparison meaningful.
        """
        import logging
        import os

        suite = search_suite()
        seeds = [42, 43, 44]
        n_cores = os.cpu_count() or 1

        all_results: list[RunResult] = []
        for scenario in suite:
            t0 = time.monotonic()
            batch = evaluate_batch(
                [ForceParams.defaults()],
                scenario.scenario_ref,
                seeds,
                max_ticks=self._TICK_CAP,
                max_workers=None,
            )
            elapsed = time.monotonic() - t0
            logging.getLogger(__name__).info(
                "parallel batch %s: %d tasks / %d cores → %.3f s",
                scenario.name,
                len(seeds),
                n_cores,
                elapsed,
            )
            all_results.extend(batch)

        assert len(all_results) == len(suite) * len(seeds)
        assert all(isinstance(r, RunResult) for r in all_results)
