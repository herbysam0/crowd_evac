"""Tests for crowd_evac.domain.panic_source.PanicSource.

Covers:
  - Construction: valid defaults, explicit parameters, and all invalid inputs.
  - pos property: returns (x, y) tuple.
  - decay(): intensity reduces by decay_rate * dt, clamps to zero, raises on
    non-positive dt.
  - is_active: correct True/False around the expiry threshold.

Note on pytest.raises(match=...): the match string is a *regex substring*
check — it passes when the pattern appears anywhere in the error message.
Partial patterns are intentional to avoid hard-coding the repr of the
offending value into every test.
"""
from __future__ import annotations

import pytest

from crowd_evac.domain.constants import DT, PANIC_DECAY_RATE, PANIC_RANGE
from crowd_evac.domain.panic_source import PanicSource, _EXPIRED_THRESHOLD


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPanicSourceConstruction:
    """PanicSource must validate its fields on construction."""

    def test_valid_defaults(self) -> None:
        """Source constructed with just position uses correct defaults."""
        s = PanicSource(x=1.0, y=2.0)
        assert s.intensity == 1.0
        assert s.radius == PANIC_RANGE
        assert s.decay_rate == PANIC_DECAY_RATE
        assert s.is_active

    def test_explicit_parameters_stored(self) -> None:
        """Explicit arguments are preserved exactly."""
        s = PanicSource(
            x=3.0, y=4.0, intensity=0.5, radius=5.0, decay_rate=0.1
        )
        assert s.x == 3.0
        assert s.y == 4.0
        assert s.intensity == 0.5
        assert s.radius == 5.0
        assert s.decay_rate == 0.1

    def test_pos_property(self) -> None:
        """pos returns the (x, y) tuple in metres."""
        s = PanicSource(x=7.0, y=8.0)
        assert s.pos == (7.0, 8.0)

    def test_zero_radius_raises(self) -> None:
        """radius=0.0 must raise ValueError (message: 'radius must be positive')."""
        with pytest.raises(ValueError, match="radius must be positive"):
            PanicSource(x=0.0, y=0.0, radius=0.0)

    def test_negative_radius_raises(self) -> None:
        """Negative radius must raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            PanicSource(x=0.0, y=0.0, radius=-1.0)

    def test_negative_decay_rate_raises(self) -> None:
        """Negative decay_rate raises ValueError (message: 'decay_rate must be >= 0')."""
        with pytest.raises(ValueError, match="decay_rate must be >= 0"):
            PanicSource(x=0.0, y=0.0, decay_rate=-0.1)

    def test_intensity_above_one_raises(self) -> None:
        """intensity > 1.0 raises ValueError (message: 'intensity must be in')."""
        with pytest.raises(ValueError, match=r"intensity must be in \[0, 1\]"):
            PanicSource(x=0.0, y=0.0, intensity=1.5)

    def test_negative_intensity_raises(self) -> None:
        """intensity < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"intensity must be in \[0, 1\]"):
            PanicSource(x=0.0, y=0.0, intensity=-0.1)

    def test_zero_intensity_is_valid(self) -> None:
        """intensity=0.0 is a valid (but expired) source state."""
        s = PanicSource(x=0.0, y=0.0, intensity=0.0, decay_rate=0.0)
        assert s.intensity == 0.0
        assert not s.is_active


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


class TestPanicSourceDecay:
    """PanicSource.decay() must reduce intensity correctly (R11.1)."""

    def test_decay_reduces_by_rate_times_dt(self) -> None:
        """Single decay call reduces intensity by decay_rate * dt."""
        s = PanicSource(x=0.0, y=0.0, intensity=1.0, decay_rate=0.5)
        s.decay(DT)
        assert s.intensity == pytest.approx(1.0 - 0.5 * DT, rel=1e-9)

    def test_decay_clamps_to_zero(self) -> None:
        """Intensity never goes negative when decay exceeds remaining value."""
        s = PanicSource(x=0.0, y=0.0, intensity=0.001, decay_rate=10.0)
        s.decay(1.0)
        assert s.intensity == 0.0

    def test_zero_decay_rate_unchanged(self) -> None:
        """A source with decay_rate=0 retains its intensity indefinitely."""
        s = PanicSource(x=0.0, y=0.0, intensity=0.7, decay_rate=0.0)
        for _ in range(100):
            s.decay(DT)
        assert s.intensity == pytest.approx(0.7, rel=1e-9)

    def test_repeated_decay_monotonically_decreasing(self) -> None:
        """Each tick lowers intensity; it never rises."""
        s = PanicSource(x=0.0, y=0.0, intensity=1.0, decay_rate=0.1)
        prev = s.intensity
        for _ in range(20):
            s.decay(DT)
            assert s.intensity <= prev
            prev = s.intensity

    def test_zero_dt_raises(self) -> None:
        """dt=0.0 raises ValueError (message: 'dt must be positive')."""
        s = PanicSource(x=0.0, y=0.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            s.decay(0.0)

    def test_negative_dt_raises(self) -> None:
        """Negative dt raises ValueError."""
        s = PanicSource(x=0.0, y=0.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            s.decay(-DT)


# ---------------------------------------------------------------------------
# is_active
# ---------------------------------------------------------------------------


class TestPanicSourceIsActive:
    """is_active must reflect the expired-threshold crossing."""

    def test_full_intensity_is_active(self) -> None:
        """Freshly created source with intensity=1.0 is active."""
        assert PanicSource(x=0.0, y=0.0, intensity=1.0).is_active

    def test_zero_intensity_is_inactive(self) -> None:
        """Source at intensity=0.0 is inactive."""
        s = PanicSource(x=0.0, y=0.0, intensity=0.0, decay_rate=0.0)
        assert not s.is_active

    def test_source_expires_after_enough_ticks(self) -> None:
        """Source with high decay_rate eventually becomes inactive (R11.1)."""
        s = PanicSource(x=0.0, y=0.0, intensity=1.0, decay_rate=1.0)
        # decay_rate=1.0 drains 1.0 intensity per second; 30 ticks * DT > 1 s.
        for _ in range(30):
            s.decay(DT)
        assert not s.is_active

    def test_just_above_threshold_is_active(self) -> None:
        """Intensity just above _EXPIRED_THRESHOLD is still active."""
        s = PanicSource(
            x=0.0, y=0.0,
            intensity=_EXPIRED_THRESHOLD * 10,
            decay_rate=0.0,
        )
        assert s.is_active

    def test_at_threshold_is_inactive(self) -> None:
        """Intensity at exactly _EXPIRED_THRESHOLD is below the active boundary."""
        s = PanicSource(
            x=0.0, y=0.0,
            intensity=_EXPIRED_THRESHOLD,
            decay_rate=0.0,
        )
        assert not s.is_active
