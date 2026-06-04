"""InputSource port — user input event interface."""
from __future__ import annotations

from typing import Protocol


class InputEvent:
    """Base class for input events (placeholder for Phase 1)."""

    pass


class InputSource(Protocol):
    """Provide user input events from the rendering/UI layer."""

    def poll(self) -> list[InputEvent]:
        """Poll for pending input events.

        Returns:
            List of InputEvent objects. Empty list if no events pending.
            Concrete events (PlacePanicSource, MovePanicSource, etc.) are
            defined by adapters and logged by the application layer.
        """
        ...
