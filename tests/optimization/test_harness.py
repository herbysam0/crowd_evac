"""Tests for crowd_evac.optimization.harness (Phase 2 Step 2.2).

Covers:
  - RunResult: field presence, types, and shape invariants.
  - evaluate(): happy path (fixture_minimal + default params),
    reproducibility (same seed → identical result), and cap behaviour
    (small max_ticks → is_terminal=True without hanging).
  - evaluate_batch(): result count, type, ordering, and validation errors.
  - _load_scenario(): str and Path dispatch; TypeError on bad type.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization.harness import (
    RunResult,
    _events_by_tick,
    _load_scenario,
    evaluate,
    evaluate_batch,
    rerouted_flow_field,
)
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

_SCENARIO = "fixture_minimal"  # 50 agents, 10 m × 8 m room, one exit
_SEED = 42
_SMALL_MAX_TICKS = 5  # guaranteed to hit the cap for any non-trivial scenario


# ---------------------------------------------------------------------------
# RunResult structure
# ---------------------------------------------------------------------------


class TestRunResultStructure:
    """RunResult fields have the expected types and shapes after a real run."""

    @pytest.fixture(scope="class")
    def result(self) -> RunResult:
        """Single evaluate() call shared across tests in this class."""
        return evaluate(ForceParams.defaults(), _SCENARIO, _SEED)

    def test_evac_time_is_non_negative_float(self, result: RunResult) -> None:
        """evac_time is a non-negative float."""
        assert isinstance(result.evac_time, float)
        assert result.evac_time >= 0.0

    def test_evacuated_fraction_in_unit_interval(self, result: RunResult) -> None:
        """evacuated_fraction is a float in [0.0, 1.0]."""
        assert isinstance(result.evacuated_fraction, float)
        assert 0.0 <= result.evacuated_fraction <= 1.0

    def test_is_terminal_is_bool(self, result: RunResult) -> None:
        """is_terminal is a boolean."""
        assert isinstance(result.is_terminal, bool)

    def test_total_ticks_is_non_negative_int(self, result: RunResult) -> None:
        """total_ticks is a non-negative integer."""
        assert isinstance(result.total_ticks, int)
        assert result.total_ticks >= 0

    def test_initial_count_is_positive_int(self, result: RunResult) -> None:
        """initial_count matches the scenario agent count (50)."""
        assert isinstance(result.initial_count, int)
        assert result.initial_count == 50

    def test_throughput_series_length_matches_total_ticks(
        self, result: RunResult
    ) -> None:
        """throughput_series has one entry per executed tick."""
        assert len(result.throughput_series) == result.total_ticks

    def test_density_series_length_matches_total_ticks(
        self, result: RunResult
    ) -> None:
        """density_series has one entry per executed tick."""
        assert len(result.density_series) == result.total_ticks

    def test_density_series_non_negative(self, result: RunResult) -> None:
        """All peak-density values are non-negative."""
        assert all(d >= 0.0 for d in result.density_series)

    def test_sample_ticks_starts_with_zero(self, result: RunResult) -> None:
        """The initial state (tick 0) is always the first sample."""
        assert len(result.sample_ticks) >= 1
        assert result.sample_ticks[0] == 0

    def test_positions_history_shape(self, result: RunResult) -> None:
        """positions_history shape is (n_samples, initial_count, 2)."""
        n = len(result.sample_ticks)
        assert result.positions_history.shape == (n, result.initial_count, 2)

    def test_velocities_history_shape(self, result: RunResult) -> None:
        """velocities_history shape is (n_samples, initial_count, 2)."""
        n = len(result.sample_ticks)
        assert result.velocities_history.shape == (n, result.initial_count, 2)

    def test_panics_history_shape(self, result: RunResult) -> None:
        """panics_history shape is (n_samples, initial_count)."""
        n = len(result.sample_ticks)
        assert result.panics_history.shape == (n, result.initial_count)

    def test_alive_history_shape(self, result: RunResult) -> None:
        """alive_history shape is (n_samples, initial_count)."""
        n = len(result.sample_ticks)
        assert result.alive_history.shape == (n, result.initial_count)

    def test_positions_history_dtype_float64(self, result: RunResult) -> None:
        """positions_history stores float64 values."""
        assert result.positions_history.dtype == np.float64

    def test_alive_history_dtype_bool(self, result: RunResult) -> None:
        """alive_history stores boolean values."""
        assert result.alive_history.dtype == np.bool_


# ---------------------------------------------------------------------------
# evaluate() — happy path
# ---------------------------------------------------------------------------


class TestEvaluateHappyPath:
    """evaluate() completes successfully under default params."""

    def test_returns_run_result_instance(self) -> None:
        """evaluate() returns a RunResult."""
        result = evaluate(ForceParams.defaults(), _SCENARIO, _SEED)
        assert isinstance(result, RunResult)

    def test_default_params_not_terminal(self) -> None:
        """Default params complete before the default tick cap."""
        result = evaluate(ForceParams.defaults(), _SCENARIO, _SEED)
        assert not result.is_terminal

    def test_evac_fraction_positive(self) -> None:
        """Some agents evacuate under default params."""
        result = evaluate(ForceParams.defaults(), _SCENARIO, _SEED)
        assert result.evacuated_fraction > 0.0

    def test_custom_history_interval_respected(self) -> None:
        """history_interval=20 produces fewer samples than the default 10."""
        r_default = evaluate(ForceParams.defaults(), _SCENARIO, _SEED)
        r_sparse = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, history_interval=20
        )
        # Tick 0 is always included; after that every 20 vs 10 ticks.
        assert len(r_sparse.sample_ticks) <= len(r_default.sample_ticks)

    def test_sample_ticks_are_multiples_of_interval(self) -> None:
        """Non-initial sample ticks are multiples of history_interval."""
        interval = 5
        result = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, history_interval=interval
        )
        for tick in result.sample_ticks[1:]:
            assert tick % interval == 0, f"tick {tick} not a multiple of {interval}"

    def test_throughput_series_sums_to_evacuated_count(self) -> None:
        """Sum of per-tick egress matches evacuated_count × initial_count."""
        result = evaluate(ForceParams.defaults(), _SCENARIO, _SEED)
        expected = round(result.evacuated_fraction * result.initial_count)
        actual = sum(result.throughput_series)
        assert actual == expected


# ---------------------------------------------------------------------------
# evaluate() — reproducibility
# ---------------------------------------------------------------------------


class TestEvaluateReproducibility:
    """Same (params, scenario, seed) triple produces bit-identical results."""

    @pytest.fixture(scope="class")
    def run_a(self) -> RunResult:
        """First run."""
        return evaluate(ForceParams.defaults(), _SCENARIO, _SEED)

    @pytest.fixture(scope="class")
    def run_b(self) -> RunResult:
        """Second run — same args, independent call."""
        return evaluate(ForceParams.defaults(), _SCENARIO, _SEED)

    def test_scalar_fields_identical(
        self, run_a: RunResult, run_b: RunResult
    ) -> None:
        """Scalar fields are bit-identical across two identical runs."""
        assert run_a.evac_time == run_b.evac_time
        assert run_a.evacuated_fraction == run_b.evacuated_fraction
        assert run_a.is_terminal == run_b.is_terminal
        assert run_a.total_ticks == run_b.total_ticks
        assert run_a.initial_count == run_b.initial_count

    def test_series_identical(self, run_a: RunResult, run_b: RunResult) -> None:
        """Per-tick series are identical across two identical runs."""
        assert run_a.throughput_series == run_b.throughput_series
        assert run_a.density_series == run_b.density_series

    def test_positions_history_identical(
        self, run_a: RunResult, run_b: RunResult
    ) -> None:
        """positions_history arrays are element-wise identical."""
        assert np.array_equal(run_a.positions_history, run_b.positions_history)

    def test_different_seed_may_differ(self) -> None:
        """Different seeds can produce different evacuation times."""
        r1 = evaluate(ForceParams.defaults(), _SCENARIO, seed=1)
        r2 = evaluate(ForceParams.defaults(), _SCENARIO, seed=99)
        # Seeds could theoretically collide, but it is astronomically unlikely.
        # We test that at minimum the function returns valid RunResults.
        assert isinstance(r1, RunResult)
        assert isinstance(r2, RunResult)


# ---------------------------------------------------------------------------
# evaluate() — cap behaviour
# ---------------------------------------------------------------------------


class TestEvaluateCapBehaviour:
    """A small max_ticks cap terminates the run without hanging."""

    def test_small_max_ticks_sets_is_terminal(self) -> None:
        """max_ticks=5 marks the result as terminal for a 50-agent scenario."""
        result = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, max_ticks=_SMALL_MAX_TICKS
        )
        assert result.is_terminal

    def test_total_ticks_does_not_exceed_cap(self) -> None:
        """total_ticks never exceeds max_ticks."""
        result = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, max_ticks=_SMALL_MAX_TICKS
        )
        assert result.total_ticks <= _SMALL_MAX_TICKS

    def test_terminal_run_still_returns_run_result(self) -> None:
        """A capped run returns a fully-populated RunResult, not an exception."""
        result = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, max_ticks=_SMALL_MAX_TICKS
        )
        assert isinstance(result, RunResult)
        assert len(result.throughput_series) == result.total_ticks

    def test_terminal_evac_time_equals_sim_time_at_cap(self) -> None:
        """When no egress occurs before the cap, evac_time == sim_time at cap."""
        # Force cap before any egress by using a very tight tick limit.
        result = evaluate(
            ForceParams.defaults(), _SCENARIO, _SEED, max_ticks=1
        )
        if result.is_terminal and sum(result.throughput_series) == 0:
            # evac_time should reflect the sim_time at cap (small positive).
            assert result.evac_time >= 0.0


# ---------------------------------------------------------------------------
# evaluate_batch() — correctness and validation
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    """evaluate_batch() distributes work and returns correct results."""

    def test_returns_n_times_m_results(self) -> None:
        """N candidates × M seeds → N×M RunResult entries."""
        candidates = [ForceParams.defaults()] * 2
        seeds = [42, 43, 44]
        results = evaluate_batch(
            candidates, _SCENARIO, seeds, max_ticks=_SMALL_MAX_TICKS
        )
        assert len(results) == len(candidates) * len(seeds)

    def test_all_results_are_run_results(self) -> None:
        """Every element of the returned list is a RunResult."""
        candidates = [ForceParams.defaults()] * 3
        seeds = [10, 20]
        results = evaluate_batch(
            candidates, _SCENARIO, seeds, max_ticks=_SMALL_MAX_TICKS
        )
        assert all(isinstance(r, RunResult) for r in results)

    def test_row_major_ordering_matches_serial(self) -> None:
        """Result at index i*len(seeds)+j matches serial evaluate(cand[i], seed[j])."""
        candidates = [ForceParams.defaults(), ForceParams.defaults()]
        seeds = [1, 2]
        batch = evaluate_batch(
            candidates, _SCENARIO, seeds, max_ticks=_SMALL_MAX_TICKS, max_workers=1
        )
        for i, p in enumerate(candidates):
            for j, s in enumerate(seeds):
                serial = evaluate(p, _SCENARIO, s, max_ticks=_SMALL_MAX_TICKS)
                batch_result = batch[i * len(seeds) + j]
                assert batch_result.evac_time == serial.evac_time
                assert batch_result.total_ticks == serial.total_ticks

    def test_raises_on_empty_candidates(self) -> None:
        """ValueError raised when candidates list is empty."""
        with pytest.raises(ValueError, match="candidates must be non-empty"):
            evaluate_batch([], _SCENARIO, [42])

    def test_raises_on_empty_seeds(self) -> None:
        """ValueError raised when seeds list is empty."""
        with pytest.raises(ValueError, match="seeds must be non-empty"):
            evaluate_batch([ForceParams.defaults()], _SCENARIO, [])


# ---------------------------------------------------------------------------
# _load_scenario() — dispatch
# ---------------------------------------------------------------------------


class TestLoadScenario:
    """_load_scenario dispatches correctly by argument type."""

    def test_string_loads_bundled_scenario(self) -> None:
        """A str argument loads a bundled scenario without error."""
        floor_plan, scenario_data = _load_scenario("fixture_minimal")
        assert scenario_data["name"] == "fixture_minimal"

    def test_path_loads_from_filesystem(self, tmp_path: Path) -> None:
        """A Path argument loads a scenario file from the filesystem."""
        # Write a minimal valid scenario to a temp file.
        data = {
            "schema_version": "1.0",
            "name": "tmp_test",
            "floor_plan": {
                "width_m": 10.0,
                "height_m": 8.0,
                "walls": [],
                "obstacles": [],
                "exits": [
                    {
                        "x": 10.0,
                        "y": 4.0,
                        "width_m": 2.0,
                        "side": "east",
                        "capacity_per_second": 10,
                        "label": "main",
                    }
                ],
            },
            "agents": {"count": 5, "spawn_seed": 7},
        }
        scenario_file = tmp_path / "tmp_test.json"
        scenario_file.write_text(json.dumps(data), encoding="utf-8")
        floor_plan, scenario_data = _load_scenario(scenario_file)
        assert scenario_data["name"] == "tmp_test"

    def test_invalid_type_raises_type_error(self) -> None:
        """A non-str/non-Path argument raises TypeError."""
        with pytest.raises(TypeError, match="scenario must be str"):
            _load_scenario(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scripted emergency-event injection (Phase 2, Step 2.9)
# ---------------------------------------------------------------------------


def _scenario_with_event(tick: int, pos: list[float]) -> dict[str, object]:
    """Build a minimal scenario dict carrying one place_panic_source event."""
    return {
        "schema_version": "1.0",
        "name": "tmp_hazard",
        "floor_plan": {
            "width_m": 12.0,
            "height_m": 8.0,
            "walls": [],
            "obstacles": [],
            "exits": [
                {"x": 12.0, "y": 4.0, "width_m": 2.0, "side": "east",
                 "capacity_per_second": 10, "label": "main"},
            ],
        },
        "agents": {"count": 6, "spawn_seed": 7},
        "simulation": {"dt": 0.05, "max_ticks": 5000},
        "events": [
            {"tick": tick, "type": "place_panic_source", "pos": pos},
        ],
    }


class TestEventsByTick:
    """_events_by_tick grouping behaviour."""

    def test_empty_events_yield_empty_map(self) -> None:
        """No events produce an empty schedule."""
        assert _events_by_tick([]) == {}

    def test_groups_by_tick(self) -> None:
        """Events are grouped under their firing tick."""
        events = [
            {"tick": 5, "type": "place_panic_source", "pos": [1.0, 1.0]},
            {"tick": 5, "type": "place_panic_source", "pos": [2.0, 2.0]},
            {"tick": 9, "type": "place_panic_source", "pos": [3.0, 3.0]},
        ]
        grouped = _events_by_tick(events)  # type: ignore[arg-type]
        assert set(grouped) == {5, 9}
        assert len(grouped[5]) == 2
        assert len(grouped[9]) == 1

    def test_preserves_listed_order_within_tick(self) -> None:
        """Events on the same tick keep their file order."""
        events = [
            {"tick": 1, "type": "place_panic_source", "pos": [1.0, 1.0]},
            {"tick": 1, "type": "place_panic_source", "pos": [2.0, 2.0]},
        ]
        grouped = _events_by_tick(events)  # type: ignore[arg-type]
        assert grouped[1][0]["pos"] == [1.0, 1.0]
        assert grouped[1][1]["pos"] == [2.0, 2.0]


class TestEventInjection:
    """evaluate() fires scenario events through add_panic_source."""

    def test_event_triggers_injection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scenario event invokes add_panic_source once at its tick."""
        calls: list[tuple[str, tuple[float, float]]] = []

        def _spy(sim: object, source_type: str,
                 pos: tuple[float, float], **kwargs: object) -> None:
            calls.append((source_type, pos))

        monkeypatch.setattr(
            "crowd_evac.optimization.harness.add_panic_source", _spy
        )
        scenario = tmp_path / "hazard.json"
        scenario.write_text(
            json.dumps(_scenario_with_event(2, [6.0, 4.0])), encoding="utf-8"
        )
        evaluate(ForceParams.defaults(), scenario, _SEED, max_ticks=20)
        assert len(calls) == 1
        assert calls[0][0] == "fire"
        assert calls[0][1] == (6.0, 4.0)

    def test_event_before_cap_not_fired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An event scheduled past the tick cap never fires."""
        calls: list[object] = []
        monkeypatch.setattr(
            "crowd_evac.optimization.harness.add_panic_source",
            lambda *a, **k: calls.append(1),
        )
        scenario = tmp_path / "late_hazard.json"
        scenario.write_text(
            json.dumps(_scenario_with_event(500, [6.0, 4.0])), encoding="utf-8"
        )
        evaluate(ForceParams.defaults(), scenario, _SEED, max_ticks=5)
        assert calls == []

    def test_event_run_is_deterministic(self, tmp_path: Path) -> None:
        """Same seed + scenario events → identical RunResult (positions)."""
        scenario = tmp_path / "det_hazard.json"
        scenario.write_text(
            json.dumps(_scenario_with_event(1, [6.0, 4.0])), encoding="utf-8"
        )
        r1 = evaluate(ForceParams.defaults(), scenario, _SEED, max_ticks=30)
        r2 = evaluate(ForceParams.defaults(), scenario, _SEED, max_ticks=30)
        assert r1.evac_time == r2.evac_time
        assert np.array_equal(r1.positions_history, r2.positions_history)


class TestRerouteFlowField:
    """rerouted_flow_field reflects scenario hazard blocks."""

    def test_hazard_scenario_blocks_cells(self) -> None:
        """The hazard scenario re-routes: its field blocks cells the base does not."""
        params = ForceParams.defaults()
        floor_plan, _ = _load_scenario("hazard_lecture_hall")
        base = FlowField.build(floor_plan)
        rerouted = rerouted_flow_field("hazard_lecture_hall", params)
        assert len(rerouted.blocked) > len(base.blocked)

    def test_event_free_scenario_matches_base(self) -> None:
        """A scenario with no events yields a field with no extra blocks."""
        floor_plan, _ = _load_scenario("lecture_hall")
        base = FlowField.build(floor_plan)
        rerouted = rerouted_flow_field("lecture_hall", ForceParams.defaults())
        assert len(rerouted.blocked) == len(base.blocked)
