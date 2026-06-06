"""Tests for crowd_evac.domain.constants."""
from __future__ import annotations

from crowd_evac.domain import constants


class TestConstants:
    """Test suite for global simulation constants."""

    def test_constants_are_positive(self) -> None:
        """Verify all numeric constants are positive."""
        positive_constants = [
            constants.DT,
            constants.MAX_SPEED,
            constants.MAX_ACCEL,
            constants.REPULSION_RADIUS,
            constants.REPULSION_STRENGTH,
            constants.PANIC_RANGE,
            constants.PANIC_DECAY_RATE,
            constants.PANIC_REPULSION_STRENGTH,
            constants.GRID_CELL_SIZE,
            constants.PIXELS_PER_METER,
        ]
        for const in positive_constants:
            assert const > 0, f"constant {const} must be positive"

    def test_dt_is_reasonable_timestep(self) -> None:
        """Verify DT is in a realistic range for crowd simulation."""
        assert 0.01 <= constants.DT <= 0.1, "DT should be 10-100 ms"

    def test_speed_limits_are_sensible(self) -> None:
        """Verify speed constraints are physically plausible."""
        assert constants.MAX_SPEED <= 4.0, "max speed under 4 m/s (realistic)"
        assert constants.PANIC_SPEED_MULTIPLIER >= 1.0, "panic boost >= 1x"
        assert (
            constants.PANIC_SPEED_MULTIPLIER <= 2.5
        ), "panic boost <= 2.5x (search bound upper edge)"

    def test_repulsion_radius_less_than_panic_range(self) -> None:
        """Verify short-range repulsion is closer than panic influence."""
        assert (
            constants.REPULSION_RADIUS < constants.PANIC_RANGE
        ), "agents repel before panic affects them"

    def test_exit_capacity_is_positive_integer(self) -> None:
        """Verify exit capacity is a reasonable positive integer."""
        assert isinstance(constants.EXIT_CAPACITY_PER_SECOND, int)
        assert constants.EXIT_CAPACITY_PER_SECOND > 0
        assert constants.EXIT_CAPACITY_PER_SECOND <= 100, "cap under 100/tick"


class TestEB16Thresholds:
    """Unit tests for the §8 EB-1..6 empirical detection thresholds.

    Verifies each threshold is in range and internally consistent with the
    reference statistics from ``optimization.realism``.
    """

    def test_eb1_upstream_density_threshold_in_range(self) -> None:
        """EB-1 threshold must be a positive, sub-jam density."""
        from crowd_evac.optimization.realism import WEIDMANN_JAM_DENSITY_M2

        assert constants.EB1_UPSTREAM_DENSITY_THRESHOLD > 0.0
        assert constants.EB1_UPSTREAM_DENSITY_THRESHOLD < WEIDMANN_JAM_DENSITY_M2

    def test_eb2_collapse_panic_threshold_is_normalised(self) -> None:
        """EB-2 panic threshold must be in the normalised [0, 1] range."""
        assert 0.0 < constants.EB2_COLLAPSE_PANIC_THRESHOLD < 1.0

    def test_eb3_flow_split_fraction_is_sensible(self) -> None:
        """EB-3 flow-split fraction must be positive and below full-crowd diversion."""
        assert 0.0 < constants.EB3_FLOW_SPLIT_FRACTION < 1.0

    def test_eb4_panic_wave_speed_is_positive(self) -> None:
        """EB-4 wave speed threshold must be positive and below free-walking speed."""
        from crowd_evac.optimization.realism import FREE_WALK_SPEED_BAND_MPS

        assert constants.EB4_PANIC_WAVE_MIN_SPEED_MPS > 0.0
        assert constants.EB4_PANIC_WAVE_MIN_SPEED_MPS < FREE_WALK_SPEED_BAND_MPS[1]

    def test_eb5_interference_deviation_is_positive_fraction(self) -> None:
        """EB-5 deviation threshold must be a small positive fraction."""
        assert 0.0 < constants.EB5_INTERFERENCE_DEVIATION < 1.0

    def test_eb6_false_route_fraction_in_range(self) -> None:
        """EB-6 false-route fraction must be in (0, 1)."""
        assert 0.0 < constants.EB6_FALSE_ROUTE_FRACTION < 1.0

    def test_eb1_threshold_matches_realism_congestion_floor(self) -> None:
        """EB-1 threshold is anchored to the realism metric's congestion floor."""
        from crowd_evac.optimization.realism import EB_CONGESTION_FLOOR_M2

        assert constants.EB1_UPSTREAM_DENSITY_THRESHOLD == EB_CONGESTION_FLOOR_M2
