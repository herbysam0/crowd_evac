"""Panic source model: point hazard driving the panic gradient field (FR-11 subset).

A :class:`PanicSource` represents a positioned hazard — fire, alarm — that
generates a radially-decaying scalar panic field consumed by
:class:`~crowd_evac.domain.panic_field.PanicField` and ultimately
:func:`~crowd_evac.domain.forces.f_panic_repulsion`.

R11.1 — intensity decay: each call to :meth:`PanicSource.decay` reduces
intensity by ``decay_rate * dt``; once intensity falls below
:data:`_EXPIRED_THRESHOLD` the source is considered inactive and contributes
nothing to the field.

Phase 1 scope: fire_visual type is sufficient for Phase 1.  Full PanicSource
type hierarchy (smoke, structural collapse, multiple visual types) is Phase 3
work.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from crowd_evac.domain.constants import PANIC_DECAY_RATE, PANIC_RANGE

logger = logging.getLogger(__name__)

_EXPIRED_THRESHOLD: float = 1e-6
"""Intensity floor; sources below this are treated as expired (inactive)."""


@dataclass
class PanicSource:
    """A point hazard driving a radially-decaying panic gradient field (FR-11).

    The source sits at world position ``(x, y)`` and has an influence
    radius ``radius`` in metres.  The scalar field contribution from this
    source at any world point ``p`` is::

        v(p) = intensity * max(0, 1 − ||p − (x, y)|| / radius)

    which is ``intensity`` at the source and linearly decays to zero at the
    boundary of the influence disc.

    Each simulation tick the caller should invoke :meth:`decay` to reduce
    intensity (R11.1).  The source becomes inactive once intensity falls
    below :data:`_EXPIRED_THRESHOLD`.

    Attributes:
        x: Horizontal world position in metres.
        y: Vertical world position in metres.
        intensity: Current source intensity in ``[0, 1]``.
        radius: Influence radius in metres.  Must be positive.
        decay_rate: Intensity reduction per simulated second.  Must be >= 0.
        source_type: Hazard type tag (e.g. ``"fire"``).  Selects the
            symbol rendered at the source position.  Unused by domain logic.
    """

    x: float
    y: float
    intensity: float = 1.0
    radius: float = PANIC_RANGE
    decay_rate: float = PANIC_DECAY_RATE
    source_type: str = "fire"
    """Human-readable hazard type tag used for symbol selection and logging."""

    def __post_init__(self) -> None:
        """Validate field constraints on construction.

        Raises:
            ValueError: If ``radius`` is not positive, ``decay_rate`` is
                negative, or ``intensity`` is outside ``[0, 1]``.
        """
        if self.radius <= 0.0:
            raise ValueError(
                f"radius must be positive, got {self.radius!r}"
            )
        if self.decay_rate < 0.0:
            raise ValueError(
                f"decay_rate must be >= 0, got {self.decay_rate!r}"
            )
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(
                f"intensity must be in [0, 1], got {self.intensity!r}"
            )

    @property
    def pos(self) -> tuple[float, float]:
        """World position as an ``(x, y)`` tuple in metres."""
        return (self.x, self.y)

    @property
    def is_active(self) -> bool:
        """``True`` while intensity is above the expiry threshold."""
        return self.intensity > _EXPIRED_THRESHOLD

    def decay(self, dt: float) -> None:
        """Reduce intensity by ``decay_rate * dt``, clamped to zero (R11.1).

        Args:
            dt: Elapsed simulation time in seconds.  Must be positive.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self.intensity = max(0.0, self.intensity - self.decay_rate * dt)
