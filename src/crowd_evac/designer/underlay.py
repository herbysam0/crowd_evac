"""Session-local photo underlay state for the room designer.

The underlay is a visual aid only: it is never part of the ``Layout``,
the ``DesignModel`` or the exported scenario JSON. This module holds the
GL-free state and the fit geometry; drawing lives in the render adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from crowd_evac.designer.geometry import Rect
from crowd_evac.domain.errors import CrowdEvacError

DEFAULT_OPACITY: float = 0.5
OPACITY_STEP: float = 0.1
MIN_OPACITY: float = 0.0
MAX_OPACITY: float = 1.0
SUPPORTED_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


class UnderlayError(CrowdEvacError):
    """Raised when an underlay photo cannot be loaded or fitted."""


@dataclass(frozen=True)
class Underlay:
    """A loaded photo shown beneath the design.

    Attributes:
        path: Source file (kept for display only; never saved).
        px_w: Photo width in pixels.
        px_h: Photo height in pixels.
        visible: Whether the photo is drawn.
        opacity: Alpha in ``[0, 1]``.
    """

    path: Path
    px_w: int
    px_h: int
    visible: bool = True
    opacity: float = DEFAULT_OPACITY


def fit_rect(
    px_w: int, px_h: int, room_w_m: float, room_h_m: float
) -> Rect:
    """Return the largest aspect-preserving rect at the room origin.

    Args:
        px_w: Photo width in pixels. Must be positive.
        px_h: Photo height in pixels. Must be positive.
        room_w_m: Room width in metres. Must be positive.
        room_h_m: Room height in metres. Must be positive.

    Returns:
        ``(0, 0, width_m, height_m)`` fitting inside the room.

    Raises:
        UnderlayError: If any size is not positive.
    """
    if min(px_w, px_h, room_w_m, room_h_m) <= 0:
        raise UnderlayError("photo and room sizes must be positive")
    scale = min(room_w_m / px_w, room_h_m / px_h)
    return (0.0, 0.0, px_w * scale, px_h * scale)


def with_opacity(underlay: Underlay, value: float) -> Underlay:
    """Return a copy with opacity clamped to ``[0, 1]``."""
    clamped = min(MAX_OPACITY, max(MIN_OPACITY, value))
    return replace(underlay, opacity=clamped)


def toggled(underlay: Underlay) -> Underlay:
    """Return a copy with ``visible`` flipped."""
    return replace(underlay, visible=not underlay.visible)
