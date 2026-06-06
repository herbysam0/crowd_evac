"""Tests for crowd_evac.optimization.realism (Phase 2 Step 2.4).

Covers:
  - Reference constants: unit-checked in their documented empirical ranges.
  - free_walking_speed(): high-percentile of moving-agent speeds; edge cases.
  - specific_flow(): mean-flow definition; zero-egress and bad-width paths.
  - speed_density_trace(): density/speed pairing and tick alignment.
  - _band_distance / _fd_distance / _eb_penalty: zero at target, monotonic
    rise when a single statistic is detuned away from its reference.
  - realism_report / realism_distance: composite is ~0 for a synthetic
    target-matching run and rises when each component is individually detuned.

All RunResults are hand-built synthetic fixtures so each statistic is driven
in isolation; no simulation is executed.  The builders are exposed as factory
fixtures (``make_run``, ``scaled_velocity_run``) and the ready-made runs as
value fixtures (``weidmann_matching_run``, ``matching_runset``).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from crowd_evac.optimization import realism as R
from crowd_evac.optimization.harness import RunResult
from crowd_evac.optimization.realism import (
    CalibrationRunSet,
    RealismReport,
    free_walking_speed,
    peak_run_density,
    realism_distance,
    realism_report,
    specific_flow,
    speed_density_trace,
)

# ---------------------------------------------------------------------------
# Factory fixtures (parametrised builders) and value fixtures (ready runs)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_run() -> Callable[..., RunResult]:
    """Return a builder for synthetic RunResults that drive the extractors.

    The returned callable accepts keyword arguments ``speeds_per_sample``,
    ``densities``, ``sample_ticks``, ``throughput``, ``evac_time``, and an
    optional ``n_agents``.  All agents at a sample share that sample's speed
    (moving along +x) and are alive; ``densities`` is the per-tick
    peak-density series and ``sample_ticks`` indexes velocity samples into it.

    Returns:
        A callable ``(**kwargs) -> RunResult``.
    """

    def _build(
        *,
        speeds_per_sample: list[float],
        densities: list[float],
        sample_ticks: list[int],
        throughput: list[int],
        evac_time: float = 1.0,
        n_agents: int = 4,
    ) -> RunResult:
        n_samples = len(speeds_per_sample)
        vel = np.zeros((n_samples, n_agents, 2), dtype=np.float64)
        for k, s in enumerate(speeds_per_sample):
            vel[k, :, 0] = s
        pos = np.zeros((n_samples, n_agents, 2), dtype=np.float64)
        panic = np.zeros((n_samples, n_agents), dtype=np.float64)
        alive = np.ones((n_samples, n_agents), dtype=np.bool_)
        denom = float(n_agents) if n_agents else 1.0
        return RunResult(
            evac_time=evac_time,
            evacuated_fraction=min(sum(throughput) / denom, 1.0),
            is_terminal=False,
            total_ticks=len(densities),
            initial_count=n_agents,
            throughput_series=tuple(throughput),
            density_series=tuple(densities),
            sample_ticks=tuple(sample_ticks),
            positions_history=pos,
            velocities_history=vel,
            panics_history=panic,
            alive_history=alive,
        )

    return _build


@pytest.fixture
def weidmann_matching_run(make_run: Callable[..., RunResult]) -> RunResult:
    """A bottleneck run whose statistics all match their references.

    The first sample sits at a free-flow density *below* the FD density floor
    (so it anchors the 85th-percentile free-speed estimate at the target but is
    excluded from the curve fit); every remaining sample's speed is the exact
    Weidmann prediction at its density, so the FD-shape residual is zero.  The
    peak density clears the EB congestion floor, and ``N / (T·w)`` gives a
    specific flow in the target band (N=60, T=40 s, w=1.2 m → 1.25 P/(m·s)).
    """
    v0 = R.FREE_WALK_SPEED_TARGET_MPS
    # First density is below FD_DENSITY_FLOOR_M2 (0.2) → excluded from the FD
    # fit; the rest lie on the Weidmann curve and stay above the moving floor.
    densities = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    speeds = [v0] + [
        float(R._weidmann_speed(np.array([d], dtype=np.float64), v0)[0])
        for d in densities[1:]
    ]
    return make_run(
        speeds_per_sample=speeds,
        densities=densities,
        sample_ticks=[t + 1 for t in range(len(densities))],
        throughput=[10, 10, 10, 10, 10, 10],
        evac_time=40.0,
        n_agents=60,
    )


@pytest.fixture
def matching_runset(weidmann_matching_run: RunResult) -> CalibrationRunSet:
    """A CalibrationRunSet whose every realism component matches its target."""
    return CalibrationRunSet(
        flow_results=(weidmann_matching_run,),
        bottleneck_result=weidmann_matching_run,
        bottleneck_door_width_m=1.2,
    )


@pytest.fixture
def scaled_velocity_run() -> Callable[[RunResult, float], RunResult]:
    """Return a builder that copies a run with its velocities rescaled.

    Returns:
        A callable ``(run, factor) -> RunResult`` yielding a copy of ``run``
        whose velocity history is multiplied by ``factor``.
    """

    def _scale(run: RunResult, factor: float) -> RunResult:
        return dataclasses.replace(
            run, velocities_history=run.velocities_history * factor
        )

    return _scale


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------


class TestReferenceConstants:
    """Empirical reference constants sit in their documented ranges."""

    def test_free_walk_target_within_band(self) -> None:
        """Free-walking target lies inside its accepted band."""
        lo, hi = R.FREE_WALK_SPEED_BAND_MPS
        assert lo <= R.FREE_WALK_SPEED_TARGET_MPS <= hi

    def test_free_walk_band_is_literature_range(self) -> None:
        """Free-walking band matches the cited 1.2-1.4 m/s range."""
        assert R.FREE_WALK_SPEED_BAND_MPS == (1.20, 1.40)

    def test_specific_flow_target_within_band(self) -> None:
        """Specific-flow target lies inside its accepted band."""
        lo, hi = R.SPECIFIC_FLOW_BAND_PMS
        assert lo <= R.SPECIFIC_FLOW_TARGET_PMS <= hi

    def test_specific_flow_band_is_plausible(self) -> None:
        """Specific-flow band sits in the 1.0-2.0 P/(m·s) plausible range."""
        lo, hi = R.SPECIFIC_FLOW_BAND_PMS
        assert 1.0 <= lo < hi <= 2.0

    def test_weidmann_jam_density_in_range(self) -> None:
        """Weidmann jam density is in the canonical 4-7 agents/m² range."""
        assert 4.0 <= R.WEIDMANN_JAM_DENSITY_M2 <= 7.0

    def test_weidmann_gamma_positive(self) -> None:
        """Weidmann shape coefficient is strictly positive."""
        assert R.WEIDMANN_GAMMA > 0.0

    def test_component_weights_sum_to_one(self) -> None:
        """Composite component weights form a convex combination."""
        total = R.W_SPEED + R.W_FLOW + R.W_FD + R.W_EB
        assert total == pytest.approx(1.0)

    def test_all_component_weights_non_negative(self) -> None:
        """No component weight is negative."""
        assert min(R.W_SPEED, R.W_FLOW, R.W_FD, R.W_EB) >= 0.0


# ---------------------------------------------------------------------------
# Weidmann curve
# ---------------------------------------------------------------------------


class TestWeidmannSpeed:
    """The Weidmann speed-density curve behaves as documented."""

    def test_zero_at_jam_density(self) -> None:
        """Speed is ~0 at the jam density."""
        v = R._weidmann_speed(
            np.array([R.WEIDMANN_JAM_DENSITY_M2], dtype=np.float64), 1.34
        )
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_monotone_decreasing_in_density(self) -> None:
        """Predicted speed falls as density rises."""
        rho = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        v = R._weidmann_speed(rho, 1.34)
        assert np.all(np.diff(v) < 0.0)

    def test_clamped_non_negative_beyond_jam(self) -> None:
        """Speed is clamped to zero past the jam density."""
        v = R._weidmann_speed(np.array([8.0], dtype=np.float64), 1.34)
        assert v[0] == 0.0


# ---------------------------------------------------------------------------
# free_walking_speed()
# ---------------------------------------------------------------------------


class TestFreeWalkingSpeed:
    """free_walking_speed isolates the free-walking cohort."""

    def test_uniform_speed_recovered(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """When all moving agents share a speed, it is recovered."""
        run = make_run(
            speeds_per_sample=[1.34, 1.34, 1.34],
            densities=[0.5, 0.5, 0.5],
            sample_ticks=[1, 2, 3],
            throughput=[0, 0, 0],
        )
        assert free_walking_speed(run) == pytest.approx(1.34, abs=1e-6)

    def test_high_percentile_ignores_slow_congested(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """A high percentile reflects fast free walkers, not slow ones."""
        run = make_run(
            speeds_per_sample=[1.30, 0.10, 0.10, 0.10],
            densities=[0.5, 5.0, 5.0, 5.0],
            sample_ticks=[1, 2, 3, 4],
            throughput=[0, 0, 0, 0],
        )
        assert free_walking_speed(run) > 0.5

    def test_stationary_agents_yield_zero(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """All-stationary history returns 0.0 (no free walkers)."""
        run = make_run(
            speeds_per_sample=[0.0, 0.0],
            densities=[0.5, 0.5],
            sample_ticks=[1, 2],
            throughput=[0, 0],
        )
        assert free_walking_speed(run) == 0.0

    def test_empty_history_yields_zero(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """Empty velocity history returns 0.0 without raising."""
        run = make_run(
            speeds_per_sample=[],
            densities=[],
            sample_ticks=[],
            throughput=[],
        )
        assert free_walking_speed(run) == 0.0


# ---------------------------------------------------------------------------
# specific_flow()
# ---------------------------------------------------------------------------


class TestSpecificFlow:
    """specific_flow applies the q = N / (T·w) definition."""

    def test_known_value(self, make_run: Callable[..., RunResult]) -> None:
        """N=60, T=40 s, w=1.2 m → q = 1.25 P/(m·s)."""
        run = make_run(
            speeds_per_sample=[1.0],
            densities=[1.0],
            sample_ticks=[1],
            throughput=[60],
            evac_time=40.0,
            n_agents=60,
        )
        assert specific_flow(run, 1.2) == pytest.approx(1.25, rel=1e-6)

    def test_zero_egress_yields_zero(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """No egress → zero specific flow."""
        run = make_run(
            speeds_per_sample=[1.0],
            densities=[1.0],
            sample_ticks=[1],
            throughput=[0],
            evac_time=40.0,
        )
        assert specific_flow(run, 1.2) == 0.0

    def test_non_positive_width_raises(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """A non-positive door width raises ValueError."""
        run = make_run(
            speeds_per_sample=[1.0],
            densities=[1.0],
            sample_ticks=[1],
            throughput=[5],
            evac_time=10.0,
        )
        with pytest.raises(ValueError, match="door_width_m must be > 0"):
            specific_flow(run, 0.0)


# ---------------------------------------------------------------------------
# speed_density_trace() and peak_run_density()
# ---------------------------------------------------------------------------


class TestSpeedDensityTrace:
    """speed_density_trace pairs density with mean moving speed per sample."""

    def test_pairs_density_with_speed(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """Returned arrays align density[k] with the sample's mean speed."""
        run = make_run(
            speeds_per_sample=[1.2, 0.8, 0.4],
            densities=[1.0, 2.0, 3.0],
            sample_ticks=[1, 2, 3],
            throughput=[0, 0, 0],
        )
        density, speed = speed_density_trace(run)
        assert list(density) == [1.0, 2.0, 3.0]
        assert speed == pytest.approx([1.2, 0.8, 0.4])

    def test_skips_tick_zero(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """Tick 0 (no density entry) is excluded from the trace."""
        run = make_run(
            speeds_per_sample=[1.5, 1.0],
            densities=[2.0],
            sample_ticks=[0, 1],
            throughput=[0],
        )
        density, speed = speed_density_trace(run)
        assert list(density) == [2.0]
        assert speed == pytest.approx([1.0])

    def test_empty_when_no_moving_agents(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """All-stationary samples yield empty trace arrays."""
        run = make_run(
            speeds_per_sample=[0.0, 0.0],
            densities=[1.0, 2.0],
            sample_ticks=[1, 2],
            throughput=[0, 0],
        )
        density, speed = speed_density_trace(run)
        assert density.size == 0
        assert speed.size == 0


class TestPeakRunDensity:
    """peak_run_density returns the maximum of the density series."""

    def test_returns_max(self, make_run: Callable[..., RunResult]) -> None:
        """Peak equals the largest density-series entry."""
        run = make_run(
            speeds_per_sample=[1.0, 1.0],
            densities=[1.0, 4.5],
            sample_ticks=[1, 2],
            throughput=[0, 0],
        )
        assert peak_run_density(run) == 4.5

    def test_empty_series_yields_zero(
        self, make_run: Callable[..., RunResult]
    ) -> None:
        """Empty density series returns 0.0."""
        run = make_run(
            speeds_per_sample=[],
            densities=[],
            sample_ticks=[],
            throughput=[],
        )
        assert peak_run_density(run) == 0.0


# ---------------------------------------------------------------------------
# Component distances: zero at target, monotonic when detuned
# ---------------------------------------------------------------------------


class TestBandDistance:
    """_band_distance is zero inside the band and rises monotonically out."""

    def test_zero_inside_band(self) -> None:
        """A value inside the band has zero distance."""
        assert R._band_distance(1.34, 1.20, 1.40) == 0.0

    def test_rises_below_band(self) -> None:
        """Distance grows as the value drops further below the band."""
        d1 = R._band_distance(1.10, 1.20, 1.40)
        d2 = R._band_distance(1.00, 1.20, 1.40)
        assert 0.0 < d1 < d2

    def test_rises_above_band(self) -> None:
        """Distance grows as the value climbs further above the band."""
        d1 = R._band_distance(1.50, 1.20, 1.40)
        d2 = R._band_distance(1.80, 1.20, 1.40)
        assert 0.0 < d1 < d2


class TestFdDistance:
    """_fd_distance is zero on the curve and rises as the trace is detuned."""

    def test_zero_on_curve(self) -> None:
        """A trace lying exactly on Weidmann has zero shape distance."""
        rho = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        v = R._weidmann_speed(rho, 1.34)
        assert R._fd_distance(rho, v, 1.34) == pytest.approx(0.0, abs=1e-9)

    def test_rises_as_curve_flattens(self) -> None:
        """A flat speed trace has positive distance growing with the offset."""
        rho = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        v_curve = R._weidmann_speed(rho, 1.34)
        flat_mild = np.full_like(rho, 1.0)
        flat_hard = np.full_like(rho, 1.34)
        d_mild = R._fd_distance(rho, flat_mild, 1.34)
        d_hard = R._fd_distance(rho, flat_hard, 1.34)
        on_curve = R._fd_distance(rho, v_curve, 1.34)
        assert on_curve < d_mild < d_hard

    def test_no_samples_above_floor_yields_zero(self) -> None:
        """A trace entirely below the density floor carries no FD evidence."""
        rho = np.array([0.05, 0.1], dtype=np.float64)
        v = np.array([1.3, 1.3], dtype=np.float64)
        assert R._fd_distance(rho, v, 1.34) == 0.0


class TestEbPenalty:
    """_eb_penalty punishes runs that never form congestion."""

    def test_zero_when_floor_reached(self) -> None:
        """Reaching the congestion floor yields zero penalty."""
        assert R._eb_penalty(R.EB_CONGESTION_FLOOR_M2) == 0.0
        assert R._eb_penalty(R.EB_CONGESTION_FLOOR_M2 + 2.0) == 0.0

    def test_rises_as_peak_density_falls(self) -> None:
        """Penalty grows as the peak density falls further below the floor."""
        d1 = R._eb_penalty(0.6)
        d2 = R._eb_penalty(0.2)
        assert 0.0 < d1 < d2

    def test_frictionless_run_near_max_penalty(self) -> None:
        """A near-zero-density (frictionless) run is heavily penalised."""
        assert R._eb_penalty(0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composite realism_report / realism_distance
# ---------------------------------------------------------------------------


class TestRealismComposite:
    """Composite distance is ~0 at target and rises when a component detunes."""

    def test_matching_run_distance_near_zero(
        self, matching_runset: CalibrationRunSet
    ) -> None:
        """A target-matching synthetic run yields ~0 composite distance."""
        report = realism_report(matching_runset)
        assert report.distance == pytest.approx(0.0, abs=1e-6)

    def test_report_is_dataclass_with_components(
        self, matching_runset: CalibrationRunSet
    ) -> None:
        """realism_report returns a RealismReport exposing each component."""
        report = realism_report(matching_runset)
        assert isinstance(report, RealismReport)
        assert report.free_walk_speed_mps == pytest.approx(
            R.FREE_WALK_SPEED_TARGET_MPS, abs=1e-6
        )

    def test_realism_distance_matches_report(
        self, matching_runset: CalibrationRunSet
    ) -> None:
        """realism_distance equals realism_report(...).distance."""
        assert realism_distance(matching_runset) == realism_report(
            matching_runset
        ).distance

    def test_detuned_free_speed_raises_distance(
        self,
        matching_runset: CalibrationRunSet,
        scaled_velocity_run: Callable[[RunResult, float], RunResult],
    ) -> None:
        """Inflating the free-walking speed raises the composite distance."""
        fast = scaled_velocity_run(matching_runset.bottleneck_result, 2.0)
        detuned = CalibrationRunSet(
            flow_results=(fast,),
            bottleneck_result=matching_runset.bottleneck_result,
            bottleneck_door_width_m=matching_runset.bottleneck_door_width_m,
        )
        assert realism_distance(detuned) > realism_distance(matching_runset)

    def test_detuned_flow_raises_distance(
        self,
        matching_runset: CalibrationRunSet,
        make_run: Callable[..., RunResult],
    ) -> None:
        """Cutting egress below the flow band raises the composite distance."""
        base_run = matching_runset.bottleneck_result
        speeds = list(
            np.linalg.norm(base_run.velocities_history, axis=-1)[:, 0]
        )
        slow_run = make_run(
            speeds_per_sample=speeds,
            densities=list(base_run.density_series),
            sample_ticks=list(base_run.sample_ticks),
            throughput=[3, 3, 3, 3, 3, 3],  # N=18 → q well below 1.2
            evac_time=40.0,
            n_agents=60,
        )
        detuned = CalibrationRunSet(
            flow_results=(base_run,),
            bottleneck_result=slow_run,
            bottleneck_door_width_m=1.2,
        )
        assert realism_distance(detuned) > realism_distance(matching_runset)

    def test_frictionless_run_raises_distance(
        self,
        matching_runset: CalibrationRunSet,
        make_run: Callable[..., RunResult],
    ) -> None:
        """A run that never congests (low peak density) raises the distance."""
        frictionless = make_run(
            speeds_per_sample=[R.FREE_WALK_SPEED_TARGET_MPS] * 6,
            densities=[0.2, 0.3, 0.4, 0.3, 0.2, 0.1],  # never reaches floor
            sample_ticks=[1, 2, 3, 4, 5, 6],
            throughput=[10, 10, 10, 10, 10, 10],
            evac_time=40.0,
            n_agents=60,
        )
        detuned = CalibrationRunSet(
            flow_results=(matching_runset.bottleneck_result,),
            bottleneck_result=frictionless,
            bottleneck_door_width_m=1.2,
        )
        assert realism_distance(detuned) > realism_distance(matching_runset)

    def test_empty_flow_results_raises(
        self, weidmann_matching_run: RunResult
    ) -> None:
        """An empty flow_results bundle raises ValueError."""
        bad = CalibrationRunSet(
            flow_results=(),
            bottleneck_result=weidmann_matching_run,
            bottleneck_door_width_m=1.2,
        )
        with pytest.raises(ValueError, match="flow_results must be non-empty"):
            realism_distance(bad)

    def test_non_positive_width_raises(
        self, weidmann_matching_run: RunResult
    ) -> None:
        """A non-positive door width raises ValueError."""
        bad = CalibrationRunSet(
            flow_results=(weidmann_matching_run,),
            bottleneck_result=weidmann_matching_run,
            bottleneck_door_width_m=0.0,
        )
        with pytest.raises(
            ValueError, match="bottleneck_door_width_m must be > 0"
        ):
            realism_distance(bad)
