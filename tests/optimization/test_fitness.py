"""Tests for crowd_evac.optimization.fitness (Phase 2 Step 2.6).

Covers the plan success criteria:
  - returns a 2-objective + 1-constraint vector of correct shape and sign;
  - identical params + seeds -> identical result (determinism / CRN);
  - raising K reduces the aggregated-objective variance (directional);
  - a stuck-prone param set reports a violated constraint;
plus the configuration validation, aggregation/penalty helpers, and the
flat-job regrouping.

Fast tests monkeypatch the module-level ``evaluate`` and ``_flow_field_for``
with synthetic hand-built fixtures, so no real simulation runs and the serial
(``max_workers=1``) dispatch path is exercised.  One integration test runs the
real harness on a small scenario at a tiny tick cap to prove the wiring holds
end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.params import ForceParams
from crowd_evac.optimization import fitness as F
from crowd_evac.optimization.fitness import (
    FitnessConfig,
    FitnessResult,
    evaluate_fitness,
)
from crowd_evac.optimization.harness import RunResult
from crowd_evac.optimization.suite import SearchScenario
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Synthetic RunResult builders (mirrors the realism/stuck test fixtures)
# ---------------------------------------------------------------------------

_GRID = 10
_CELL = 1.0


def _open_field() -> FlowField:
    """A 10x10 unit-cell field whose only exits are column 0 (descent = -x)."""
    walkable = np.ones((_GRID, _GRID), dtype=np.bool_)
    exits = [(r, 0) for r in range(_GRID)]
    return FlowField(_CELL, walkable, exits)


def _seed_speed(seed: int) -> float:
    """Deterministic per-seed free-walking speed spanning the realism band.

    Maps a seed to ``[1.0, 1.6)`` m/s so some seeds fall inside the
    ``[1.2, 1.4]`` reference band (zero speed-distance) and others outside it
    (positive distance), giving the realism objective genuine per-seed spread.
    """
    frac = ((seed * 2654435761) % 1000) / 1000.0
    return 1.0 + 0.6 * frac


def _moving_run(seed: int, *, n_agents: int = 6) -> RunResult:
    """A run of alive, moving agents whose free speed depends on the seed.

    Agents move along +x at the seed's speed across four samples spanning a
    free-flow-to-jam density sweep, so the realism extractors see a moving
    cohort (free speed = the seed speed) and the stuck detector sees no stall.
    """
    speed = _seed_speed(seed)
    densities = [0.1, 0.5, 1.0, 2.0]
    n_samples = len(densities)
    vel = np.zeros((n_samples, n_agents, 2), dtype=np.float64)
    vel[:, :, 0] = speed
    pos = np.zeros((n_samples, n_agents, 2), dtype=np.float64)
    panic = np.zeros((n_samples, n_agents), dtype=np.float64)
    alive = np.ones((n_samples, n_agents), dtype=np.bool_)
    throughput = [n_agents] * n_samples
    return RunResult(
        evac_time=40.0,
        evacuated_fraction=1.0,
        is_terminal=False,
        total_ticks=n_samples,
        initial_count=n_agents,
        throughput_series=tuple(throughput),
        density_series=tuple(densities),
        sample_ticks=tuple(t + 1 for t in range(n_samples)),
        positions_history=pos,
        velocities_history=vel,
        panics_history=panic,
        alive_history=alive,
    )


def _deadlock_run(n_samples: int = 6) -> RunResult:
    """A single stalled agent at an interior cell with a clear route west.

    Mirrors the stuck-detector deadlock fixture: motionless at (5.5, 5.5) over
    six samples at tick spacing 10, so it satisfies every deadlock condition on
    the open field and is flagged stuck.
    """
    pos = np.tile(
        np.asarray([[5.5, 5.5]], dtype=np.float64)[np.newaxis, :, :],
        (n_samples, 1, 1),
    )
    vel = np.zeros((n_samples, 1, 2), dtype=np.float64)
    panic = np.zeros((n_samples, 1), dtype=np.float64)
    alive = np.ones((n_samples, 1), dtype=np.bool_)
    return RunResult(
        evac_time=10.0,
        evacuated_fraction=0.0,
        is_terminal=True,
        total_ticks=10 * n_samples,
        initial_count=1,
        throughput_series=(0,) * n_samples,
        density_series=(0.1,) * n_samples,
        sample_ticks=tuple(10 * (k + 1) for k in range(n_samples)),
        positions_history=pos,
        velocities_history=vel,
        panics_history=panic,
        alive_history=alive,
    )


def _one_scenario() -> SearchScenario:
    """A single search scenario (its ref is ignored by the mocked evaluator)."""
    return SearchScenario(
        name="mock",
        scenario_ref="open_room_search",
        full_agent_count=25,
        description="mock scenario for fitness unit tests",
    )


@pytest.fixture
def patch_moving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the evaluator/flow-field with the moving-run synthetic fixtures."""

    def fake_eval(
        params: ForceParams,
        scenario_ref: object,
        seed: int,
        max_ticks: int,
        *,
        wall_clock_cap_s: float,
        history_interval: int,
    ) -> RunResult:
        return _moving_run(seed)

    monkeypatch.setattr(F, "evaluate", fake_eval)
    monkeypatch.setattr(F, "_flow_field_for", lambda ref: _open_field())


@pytest.fixture
def patch_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the evaluator/flow-field with the deadlock synthetic fixture."""

    def fake_eval(
        params: ForceParams,
        scenario_ref: object,
        seed: int,
        max_ticks: int,
        *,
        wall_clock_cap_s: float,
        history_interval: int,
    ) -> RunResult:
        return _deadlock_run()

    monkeypatch.setattr(F, "evaluate", fake_eval)
    monkeypatch.setattr(F, "_flow_field_for", lambda ref: _open_field())


# ---------------------------------------------------------------------------
# Plan success criteria (mocked, fast)
# ---------------------------------------------------------------------------


class TestFitnessVectorShape:
    """The objective/constraint vector has the correct shape and sign."""

    def test_shape_and_sign(self, patch_moving: None) -> None:
        """Two non-negative objectives, one non-negative constraint, K-sized."""
        cfg = FitnessConfig(seeds=(0, 1), scenarios=(_one_scenario(),), max_workers=1)
        res = evaluate_fitness(ForceParams.defaults(), cfg)
        assert isinstance(res, FitnessResult)
        assert len(res.objectives) == 2
        assert len(res.constraints) == 1
        assert res.objectives[0] >= 0.0  # realism distance
        assert res.objectives[1] >= 0.0  # effective evac time
        assert res.objectives == (res.realism_distance, res.evac_time)
        assert res.constraints == (float(res.stuck_count),)
        assert len(res.per_seed_realism) == 2
        assert 0.0 <= res.evacuated_fraction <= 1.0

    def test_feasible_when_not_stuck(self, patch_moving: None) -> None:
        """A moving, non-deadlocked candidate satisfies the constraint (== 0)."""
        cfg = FitnessConfig(seeds=(0,), scenarios=(_one_scenario(),), max_workers=1)
        res = evaluate_fitness(ForceParams.defaults(), cfg)
        assert res.stuck_count == 0
        assert res.constraints[0] <= 0.0


class TestDeterminism:
    """Identical params + seeds reproduce an identical result (CRN)."""

    def test_repeat_equal(self, patch_moving: None) -> None:
        """Two evaluations of the same candidate compare equal."""
        cfg = FitnessConfig(
            seeds=(0, 1, 2), scenarios=(_one_scenario(),), max_workers=1
        )
        a = evaluate_fitness(ForceParams.defaults(), cfg)
        b = evaluate_fitness(ForceParams.defaults(), cfg)
        assert a == b  # wall_clock_s excluded from equality


class TestConstraintViolation:
    """A stuck-prone candidate reports a violated constraint."""

    def test_deadlock_violates(self, patch_deadlock: None) -> None:
        """A deadlocked run drives stuck_count > 0 and the constraint > 0."""
        cfg = FitnessConfig(seeds=(0, 1), scenarios=(_one_scenario(),), max_workers=1)
        res = evaluate_fitness(ForceParams.defaults(), cfg)
        assert res.stuck_count > 0
        assert res.constraints[0] > 0.0


class TestSeedVarianceReduction:
    """Raising K reduces the variance of the aggregated objective."""

    def test_more_seeds_lower_variance(self, patch_moving: None) -> None:
        """The K=8 realism estimator varies less across seed sets than K=1."""
        params = ForceParams.defaults()

        def realism_obj(seeds: tuple[int, ...]) -> float:
            cfg = FitnessConfig(
                seeds=seeds,
                scenarios=(_one_scenario(),),
                max_workers=1,
                robustness_weight=0.0,  # pure mean, so var falls as ~1/K
            )
            return evaluate_fitness(params, cfg).realism_distance

        k1 = [realism_obj((s,)) for s in range(40)]
        k8 = [realism_obj(tuple(range(i * 8, i * 8 + 8))) for i in range(5)]
        assert np.var(k8) < np.var(k1)


# ---------------------------------------------------------------------------
# Configuration validation (failure paths)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """FitnessConfig rejects out-of-range or empty inputs."""

    def test_empty_seeds(self) -> None:
        """No seeds raises."""
        with pytest.raises(ValueError, match="seeds must be non-empty"):
            FitnessConfig(seeds=())

    def test_empty_scenarios(self) -> None:
        """No scenarios raises."""
        with pytest.raises(ValueError, match="scenarios must be non-empty"):
            FitnessConfig(scenarios=())

    def test_bad_quantile(self) -> None:
        """A quantile outside [0, 1] raises."""
        with pytest.raises(ValueError, match="robustness_quantile"):
            FitnessConfig(robustness_quantile=1.5)

    def test_negative_weight(self) -> None:
        """A negative robustness weight raises."""
        with pytest.raises(ValueError, match="robustness_weight"):
            FitnessConfig(robustness_weight=-0.1)

    def test_negative_penalty(self) -> None:
        """A negative incomplete-evac penalty raises."""
        with pytest.raises(ValueError, match="incomplete_evac_penalty_s"):
            FitnessConfig(incomplete_evac_penalty_s=-1.0)

    def test_bad_max_workers(self) -> None:
        """A worker count below 1 raises."""
        with pytest.raises(ValueError, match="max_workers"):
            FitnessConfig(max_workers=0)

    def test_non_positive_max_ticks(self) -> None:
        """A non-positive tick cap raises."""
        with pytest.raises(ValueError, match="max_ticks"):
            FitnessConfig(max_ticks=0)

    def test_defaults_build(self) -> None:
        """The default config wires the search suite and a rig without error."""
        cfg = FitnessConfig()
        assert len(cfg.seeds) >= 1
        assert len(cfg.scenarios) >= 1
        assert cfg.rig.door_width_m > 0.0


# ---------------------------------------------------------------------------
# Aggregation and penalty helpers
# ---------------------------------------------------------------------------


class TestAggregate:
    """_aggregate blends the mean with an upper-quantile robustness term."""

    def test_zero_weight_is_mean(self) -> None:
        """Weight 0 returns the plain mean."""
        assert F._aggregate([1.0, 3.0, 5.0], 0.75, 0.0) == pytest.approx(3.0)

    def test_single_value(self) -> None:
        """A single sample returns itself regardless of weight."""
        assert F._aggregate([2.5], 0.9, 1.0) == pytest.approx(2.5)

    def test_positive_weight_penalises_spread(self) -> None:
        """A positive weight pushes the score above the mean for a spread set."""
        mean = 3.0
        blended = F._aggregate([1.0, 3.0, 5.0], 1.0, 0.5)
        assert blended > mean  # upper quantile (5) drags the score up

    def test_no_spread_equals_mean(self) -> None:
        """A constant set returns the mean even with a positive weight."""
        assert F._aggregate([4.0, 4.0, 4.0], 0.75, 1.0) == pytest.approx(4.0)


class TestEffectiveEvacTime:
    """_effective_evac_time penalises unevacuated agents."""

    def test_full_evac_is_raw(self) -> None:
        """A fully-evacuated run keeps its raw evac time."""
        run = _moving_run(0)  # evacuated_fraction == 1.0
        assert F._effective_evac_time(run, 500.0) == pytest.approx(run.evac_time)

    def test_partial_evac_penalised(self) -> None:
        """A half-evacuated run adds half the penalty to its time."""
        run = _deadlock_run()  # evacuated_fraction == 0.0
        assert F._effective_evac_time(run, 500.0) == pytest.approx(
            run.evac_time + 500.0
        )


class TestRegroup:
    """_regroup inverts the seed-major, stride M+1 flat layout."""

    def test_grouping(self) -> None:
        """Each seed's M suite runs precede its single rig run."""
        runs = [_moving_run(i) for i in range(6)]  # K=2, M=2 -> stride 3
        grouped = F._regroup(runs, n_seeds=2, n_scenarios=2)
        assert len(grouped) == 2
        suite0, rig0 = grouped[0]
        assert suite0 == [runs[0], runs[1]]
        assert rig0 is runs[2]
        suite1, rig1 = grouped[1]
        assert suite1 == [runs[3], runs[4]]
        assert rig1 is runs[5]


class TestFlowFieldFor:
    """_flow_field_for dispatches by reference type and rejects others."""

    def test_bad_type_raises(self) -> None:
        """A non-str, non-Path reference raises TypeError."""
        with pytest.raises(TypeError, match="scenario_ref must be str or Path"):
            F._flow_field_for(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration (real harness, small scale)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end wiring on a real scenario at a tiny tick cap."""

    def test_real_evaluation(self) -> None:
        """A real run yields a finite, correctly-shaped fitness vector."""
        cfg = FitnessConfig(
            seeds=(0,),
            scenarios=(_one_scenario(),),
            max_ticks=40,
            max_workers=1,
        )
        res = evaluate_fitness(ForceParams.defaults(), cfg)
        assert len(res.objectives) == 2
        assert np.isfinite(res.objectives[0])
        assert np.isfinite(res.objectives[1])
        assert res.objectives[0] >= 0.0
        assert res.stuck_count >= 0
        assert 0.0 <= res.evacuated_fraction <= 1.0
