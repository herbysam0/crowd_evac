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
            constants.PANIC_SPEED_MULTIPLIER <= 2.0
        ), "panic boost <= 2x (realistic)"

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
