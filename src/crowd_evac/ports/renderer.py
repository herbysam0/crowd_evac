"""Renderer port — read-only snapshot visualization interface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crowd_evac.application.simulation import SimSnapshot


class Renderer(Protocol):
    """Render a simulation snapshot without mutating it."""

    def render(self, snapshot: SimSnapshot) -> None:
        """Render the current simulation state.

        Args:
            snapshot: Immutable read-only view of simulation state. The
                renderer must not mutate this snapshot or any objects within it.
        """
        ...
