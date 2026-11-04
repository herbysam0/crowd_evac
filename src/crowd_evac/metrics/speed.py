"""Average person velocity over a run (moving agents only).

Queueing agents are ignored: a speed counts only above
``MOVING_SPEED_EPS_MPS``.  Headless and visual loops both call
:meth:`MovingSpeedTally.add` after each step so their results are equal.
"""
from __future__ import annotations

import numpy as np

from crowd_evac.application.simulation import SimSnapshot
from crowd_evac.domain.constants import MOVING_SPEED_EPS_MPS


class MovingSpeedTally:
    """Running mean of speeds of alive, moving agents over all ticks.

    Attributes:
        mean_mps: Mean speed (m/s) over every counted sample, or ``None``
            when no agent ever moved.
    """

    def __init__(self) -> None:
        """Start with no samples."""
        self._sum: float = 0.0
        self._count: int = 0

    def add(self, snapshot: SimSnapshot) -> None:
        """Add the speeds of alive agents above the moving threshold.

        Args:
            snapshot: State after one step.
        """
        speeds = np.linalg.norm(snapshot.velocities, axis=-1)
        moving = snapshot.alive & (speeds > MOVING_SPEED_EPS_MPS)
        count = int(np.count_nonzero(moving))
        if count:
            self._sum += float(speeds[moving].sum())
            self._count += count

    @property
    def mean_mps(self) -> float | None:
        """Mean moving speed so far, or ``None`` if nobody has moved."""
        if self._count == 0:
            return None
        return self._sum / self._count
