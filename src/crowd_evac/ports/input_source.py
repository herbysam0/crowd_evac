"""InputSource port — user input event interface and concrete event types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class InputEvent:
    """Base class for all user input events.

    Concrete subclasses carry per-event payload. The application layer
    dispatches on the concrete type via ``isinstance`` checks in
    :func:`~crowd_evac.application.injection.process_input_events`.
    """


@dataclass(frozen=True)
class PlacePanicSourceEvent(InputEvent):
    """Left-button press: place a new panic source at a world position.

    Attributes:
        pos_m: World-space position ``(x, y)`` in metres where the new
            panic source should be created.
    """

    pos_m: tuple[float, float]


@dataclass(frozen=True)
class MovePanicSourceEvent(InputEvent):
    """Left-button drag: move the current panic source to a new position.

    Attributes:
        pos_m: Updated world-space position ``(x, y)`` in metres for the
            currently active panic source.
    """

    pos_m: tuple[float, float]


class InputSource(Protocol):
    """Provide user input events from the rendering/UI layer.

    Implementations translate engine-specific events (mouse, keyboard) into
    typed :class:`InputEvent` objects.  The sim loop drains the queue each
    tick by calling :meth:`poll` and routes events through the application
    injection API (R7.2).
    """

    def poll(self) -> list[InputEvent]:
        """Return and clear all pending input events.

        Returns:
            List of :class:`InputEvent` objects since the last call.
            Empty list if no events are pending.
        """
        ...
