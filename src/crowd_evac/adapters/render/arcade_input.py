"""Arcade mouse input adapter implementing the InputSource port (FR-7 R7.2 / FR-15 subset).

Mouse left-button interactions are converted to typed
:class:`~crowd_evac.ports.input_source.PlacePanicSourceEvent` and
:class:`~crowd_evac.ports.input_source.MovePanicSourceEvent` objects,
buffered internally, and drained by the sim loop via :meth:`ArcadeInputSource.poll`.

Commands generated from events are routed through the application injection API
(:func:`~crowd_evac.application.injection.process_input_events`) — adapters
must never mutate domain state directly (R7.2).

Each user action is INFO-logged with its CLI-equivalent syntax as an early
seed for the replay scripting feature (R15.4).

Coordinate conversion
---------------------
Arcade reports mouse positions in pixels with origin at the window
bottom-left and y increasing upward.  The domain uses metres, so each
coordinate is divided by ``pixels_per_meter``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from crowd_evac.domain.constants import PIXELS_PER_METER
from crowd_evac.ports.input_source import (
    InputEvent,
    MovePanicSourceEvent,
    PlacePanicSourceEvent,
)

if TYPE_CHECKING:
    import arcade

logger = logging.getLogger(__name__)

# Left mouse button; value 1 is platform-independent and matches arcade's
# MOUSE_BUTTON_LEFT constant defined in arcade.types.
MOUSE_BUTTON_LEFT: int = 1
"""Left mouse button identifier (matches ``arcade.MOUSE_BUTTON_LEFT``)."""


class ArcadeInputSource:
    """Translate arcade mouse events into :class:`InputEvent` objects (FR-7 R7.2).

    Maintains an internal event queue.  Attach to an arcade Window either by
    calling :meth:`register` (which assigns the mouse handlers automatically)
    or by delegating the window's own ``on_mouse_press`` / ``on_mouse_drag``
    calls to this source's methods manually.

    The sim loop calls :meth:`poll` each tick to drain the queue and route
    events through the application injection API — not directly into domain
    state (R7.2).

    Only left-button events generate commands; all other buttons are ignored.

    Args:
        pixels_per_meter: World-to-pixel scale factor.  Must match the
            :class:`~crowd_evac.adapters.render.arcade_renderer.ArcadeRenderer`
            configured for the same session.

    Attributes:
        pixels_per_meter: Stored scale factor (read-only by convention).
    """

    def __init__(self, pixels_per_meter: float = PIXELS_PER_METER) -> None:
        self.pixels_per_meter: float = pixels_per_meter
        self._queue: list[InputEvent] = []

    # ------------------------------------------------------------------
    # Event handlers — called from arcade window callbacks
    # ------------------------------------------------------------------

    def on_mouse_press(
        self,
        x: float,
        y: float,
        button: int,
        modifiers: int,
    ) -> None:
        """Emit a :class:`PlacePanicSourceEvent` on left-button press.

        Converts pixel coordinates to world-space metres and enqueues the
        event.  Right- and middle-button presses are silently ignored.

        Args:
            x: Pixel x-coordinate (arcade origin: window bottom-left).
            y: Pixel y-coordinate (arcade origin: window bottom-left, y up).
            button: Arcade mouse button identifier (1 = left).
            modifiers: Keyboard modifier bitmask (unused).
        """
        if button != MOUSE_BUTTON_LEFT:
            return
        pos_m = (x / self.pixels_per_meter, y / self.pixels_per_meter)
        self._queue.append(PlacePanicSourceEvent(pos_m=pos_m))
        logger.info(
            "input: place fire %.4f %.4f  # add_panic_source fire %.4f %.4f",
            pos_m[0],
            pos_m[1],
            pos_m[0],
            pos_m[1],
        )

    def on_mouse_drag(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        buttons: int,
        modifiers: int,
    ) -> None:
        """Emit a :class:`MovePanicSourceEvent` when the left button is dragged.

        Only fires when the left button is held (``buttons & MOUSE_BUTTON_LEFT``
        is non-zero); right- and middle-button drags are silently ignored.

        Args:
            x: Current pixel x-coordinate.
            y: Current pixel y-coordinate.
            dx: Change in x since last drag event (unused, provided by arcade).
            dy: Change in y since last drag event (unused, provided by arcade).
            buttons: Bitmask of currently held mouse buttons.
            modifiers: Keyboard modifier bitmask (unused).
        """
        if not (buttons & MOUSE_BUTTON_LEFT):
            return
        pos_m = (x / self.pixels_per_meter, y / self.pixels_per_meter)
        self._queue.append(MovePanicSourceEvent(pos_m=pos_m))
        logger.info(
            "input: move fire %.4f %.4f  # move_panic_source fire %.4f %.4f",
            pos_m[0],
            pos_m[1],
            pos_m[0],
            pos_m[1],
        )

    # ------------------------------------------------------------------
    # InputSource protocol
    # ------------------------------------------------------------------

    def poll(self) -> list[InputEvent]:
        """Return and clear all buffered input events.

        Atomically swaps the internal queue with an empty list so no events
        are lost if new events arrive concurrently (in the single-threaded
        arcade loop this is equivalent to a simple clear).

        Returns:
            All :class:`InputEvent` objects buffered since the last call.
            Empty list if no events are pending.
        """
        events, self._queue = self._queue, []
        return events

    # ------------------------------------------------------------------
    # Registration helper
    # ------------------------------------------------------------------

    def register(self, window: arcade.Window) -> None:
        """Attach this source's mouse handlers to an arcade Window.

        Assigns :meth:`on_mouse_press` and :meth:`on_mouse_drag` directly to
        the window instance, so arcade dispatches mouse events to this source
        automatically.

        Args:
            window: Live arcade Window to attach handlers to.  Must be
                fully constructed (GL context active) before calling this.
        """
        window.on_mouse_press = self.on_mouse_press  # type: ignore[method-assign]
        window.on_mouse_drag = self.on_mouse_drag  # type: ignore[method-assign]
        logger.debug(
            "ArcadeInputSource registered on %r (ppm=%.1f).",
            window,
            self.pixels_per_meter,
        )
