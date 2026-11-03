"""Metre/pixel mapping shared by the simulation window and the designer.

Pure helpers with no GL dependency, so they are testable headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: a runtime import would load the designer package with the
    # CLI, which must stay lazy.
    from crowd_evac.designer.geometry import Rect

MIN_AVAILABLE_PX: float = 1.0
"""Floor for the drawable extent so degenerate windows stay finite."""


@dataclass(frozen=True)
class Viewport:
    """Linear mapping between world metres and window pixels."""

    ppm: float
    ox: float
    oy: float

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        """Convert window pixels to world metres."""
        return ((px - self.ox) / self.ppm, (py - self.oy) / self.ppm)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Convert world metres to window pixels."""
        return (self.ox + x * self.ppm, self.oy + y * self.ppm)

    def lrbt(self, rect: Rect) -> tuple[float, float, float, float]:
        """Return a world rectangle as pixel ``(left, right, bottom, top)``."""
        x, y, w, h = rect
        left, bottom = self.to_px(x, y)
        return (left, left + w * self.ppm, bottom, bottom + h * self.ppm)


def fit_viewport(
    room_w: float,
    room_h: float,
    canvas_w: float,
    canvas_h: float,
    margin_px: float = 0.0,
) -> Viewport:
    """Scale the room uniformly to fill the canvas, centred.

    Args:
        room_w: Room width in metres. Must be positive.
        room_h: Room height in metres. Must be positive.
        canvas_w: Canvas width in pixels.
        canvas_h: Canvas height in pixels.
        margin_px: Blank border kept on every side, in pixels.

    Returns:
        Viewport with a uniform scale; spare space becomes equal bars on
        the non-limiting axis. Always finite and positive, even for a
        zero-sized canvas.
    """
    avail_w = max(canvas_w - 2 * margin_px, MIN_AVAILABLE_PX)
    avail_h = max(canvas_h - 2 * margin_px, MIN_AVAILABLE_PX)
    ppm = min(avail_w / room_w, avail_h / room_h)
    return Viewport(
        ppm,
        (canvas_w - room_w * ppm) / 2.0,
        (canvas_h - room_h * ppm) / 2.0,
    )
