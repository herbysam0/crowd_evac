"""Clock port — simulation timing interface."""
from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    """Provide simulation timing (fixed-step loop tick management)."""

    def tick(self) -> float:
        """Execute one simulation step and return elapsed wall time.

        Returns:
            Elapsed wall time in seconds for this step. Allows render/IO loop
            to throttle or adjust frame pacing.
        """
        ...

    @property
    def dt(self) -> float:
        """Fixed time step (seconds) per simulation step.

        Returns:
            The fixed time step (e.g., 1/60 for 60 Hz simulation).
        """
        ...

    @property
    def current_tick(self) -> int:
        """Current monotonic simulation tick.

        Returns:
            Tick number, starting at 0, incremented by 1 per step.
        """
        ...

    @property
    def elapsed_time(self) -> float:
        """Total elapsed simulation time.

        Returns:
            Sum of all dt values for all completed steps (seconds).
        """
        ...
