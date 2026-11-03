"""Canvas drawing and metre/pixel mapping for the room designer.

Pure helpers (:class:`Viewport`, :func:`selection_rect`,
:func:`band_label`) need no GL context; the ``draw_*`` functions issue
arcade draw calls and run inside the window's ``on_draw``.
"""
from __future__ import annotations

from dataclasses import dataclass

import arcade
from arcade.types import RGBA255

from crowd_evac.adapters.render.arcade_renderer import (
    EXIT_COLOR,
    FLOOR_COLOR,
    OBSTACLE_COLOR,
    WALL_COLOR,
)
from crowd_evac.adapters.render.designer_style import LABEL_BG_COLOR
from crowd_evac.designer.controller import RubberBand, Tool
from crowd_evac.designer.geometry import Rect, wall_rect
from crowd_evac.designer.layout import Layout
from crowd_evac.designer.model import DesignModel
from crowd_evac.designer.placement import Selection

CANVAS_MARGIN_PX: float = 40.0
GRID_LINE_COLOR: RGBA255 = (60, 60, 72, 255)
SELECTION_COLOR: RGBA255 = (255, 220, 60, 255)
BAND_COLOR: RGBA255 = (120, 180, 255, 150)
SELECTION_BORDER_PX: float = 2.0
LABEL_PAD_PX: float = 4.0
ALERT_BG_COLOR: RGBA255 = (200, 30, 30, 245)
NOTICE_BG_COLOR: RGBA255 = (30, 140, 70, 245)
ALERT_PAD_PX: float = 14.0
LABEL_OFFSET_PX: float = 10.0

DESIRED_WINDOW_W: int = 1320
DESIRED_WINDOW_H: int = 800
MIN_WINDOW_W: int = 940
MIN_WINDOW_H: int = 560
# Space the OS needs around the client area (frame, title bar, taskbar).
SCREEN_MARGIN_X: int = 24
SCREEN_MARGIN_Y: int = 110

TOOL_HINTS: dict[Tool, str] = {
    Tool.SELECT: "No tool: click an item, drag to move it, Del deletes",
    Tool.WALL: "Wall: drag; snaps horizontal or vertical (0.1 m grid)",
    Tool.DOORWAY: "Doorway: click a wall on the room boundary",
    Tool.OBSTACLE: "Obstacle: drag a rectangle",
}


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
    room_w: float, room_h: float, canvas_w: float, canvas_h: float
) -> Viewport:
    """Scale the room to fill the canvas area, centred, with a margin."""
    avail_w = max(canvas_w - 2 * CANVAS_MARGIN_PX, 1.0)
    avail_h = max(canvas_h - 2 * CANVAS_MARGIN_PX, 1.0)
    ppm = min(avail_w / room_w, avail_h / room_h)
    return Viewport(
        ppm,
        (canvas_w - room_w * ppm) / 2.0,
        (canvas_h - room_h * ppm) / 2.0,
    )


def fit_window_size(
    screen_px: tuple[int, int],
    pixel_ratio: float,
    desired: tuple[int, int] = (DESIRED_WINDOW_W, DESIRED_WINDOW_H),
) -> tuple[int, int]:
    """Return a window size (logical units) that fits the display.

    The window is shrunk when ``desired`` would not fit once the OS display
    scaling is applied, leaving room for the frame, title bar and taskbar.

    Args:
        screen_px: Display size in physical pixels.
        pixel_ratio: Physical pixels per logical unit (e.g. 1.5 at 150%).
        desired: Preferred size in logical units.

    Returns:
        ``(width, height)``, never below the minimum usable size.
    """
    avail_w = int(screen_px[0] / pixel_ratio) - SCREEN_MARGIN_X
    avail_h = int(screen_px[1] / pixel_ratio) - SCREEN_MARGIN_Y
    return (
        max(MIN_WINDOW_W, min(desired[0], avail_w)),
        max(MIN_WINDOW_H, min(desired[1], avail_h)),
    )


def selection_rect(model: DesignModel, sel: Selection | None) -> Rect | None:
    """Return the world rectangle of the selected entity, if it exists."""
    if sel is None:
        return None
    layout = model.layout
    if sel.kind == "wall" and sel.id in layout.walls:
        return model.wall_rect(sel.id)
    if sel.kind == "doorway" and sel.id in layout.doorways:
        return model.doorway_rect(sel.id)
    if sel.kind == "obstacle" and sel.id in layout.obstacles:
        return layout.obstacles[sel.id].rect
    return None


def band_label(band: RubberBand) -> str:
    """Return the live readout shown beside a rubber band."""
    if band.kind == "wall":
        return f"{band.length_m:.1f} m"
    _, _, w, h = band.rect
    return f"{w:.1f} x {h:.1f} m"


def band_preview_rect(band: RubberBand, layout: Layout) -> Rect | None:
    """Return the rectangle previewed for a drag, or None if degenerate."""
    if band.kind == "obstacle":
        return band.rect
    if band.length_m <= 0:
        return None
    room = layout.room
    return wall_rect(
        band.x0, band.y0, band.x1, band.y1, room.width_m, room.height_m,
        band.thickness,
    )


def _fill(vp: Viewport, rect: Rect, color: RGBA255) -> None:
    """Fill a world rectangle."""
    left, right, bottom, top = vp.lrbt(rect)
    arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, color)


def _outline(vp: Viewport, rect: Rect, color: RGBA255, width: float) -> None:
    """Outline a world rectangle."""
    left, right, bottom, top = vp.lrbt(rect)
    arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, color, width)


def draw_room(vp: Viewport, model: DesignModel) -> None:
    """Draw floor, 1 m grid, walls, obstacles and doorways."""
    layout = model.layout
    room = layout.room
    _fill(vp, (0.0, 0.0, room.width_m, room.height_m), FLOOR_COLOR)
    for i in range(1, int(room.width_m)):
        a, b = vp.to_px(i, 0.0), vp.to_px(i, room.height_m)
        arcade.draw_line(a[0], a[1], b[0], b[1], GRID_LINE_COLOR)
    for j in range(1, int(room.height_m)):
        a, b = vp.to_px(0.0, j), vp.to_px(room.width_m, j)
        arcade.draw_line(a[0], a[1], b[0], b[1], GRID_LINE_COLOR)
    for obs in layout.obstacles.values():
        _fill(vp, obs.rect, OBSTACLE_COLOR)
    for wall_id in layout.walls:
        _fill(vp, model.wall_rect(wall_id), WALL_COLOR)
    for door_id in layout.doorways:
        _fill(vp, model.doorway_rect(door_id), EXIT_COLOR)


def draw_alert(
    label: arcade.Text,
    canvas_w: float,
    canvas_h: float,
    color: RGBA255 = ALERT_BG_COLOR,
) -> None:
    """Draw a full-width banner across the top of the canvas."""
    height = label.content_height + 2 * ALERT_PAD_PX
    arcade.draw_lrbt_rectangle_filled(
        0.0, canvas_w, canvas_h - height, canvas_h, color
    )
    label.x = ALERT_PAD_PX
    # Text is anchored at its top, so wrapped lines grow downward inside
    # the banner instead of spilling below it.
    label.y = canvas_h - ALERT_PAD_PX
    label.draw()


def draw_overlays(
    vp: Viewport,
    model: DesignModel,
    sel: Selection | None,
    band: RubberBand | None,
    label: arcade.Text,
) -> None:
    """Draw the selection outline and the rubber band with its readout."""
    rect = selection_rect(model, sel)
    if rect is not None:
        _outline(vp, rect, SELECTION_COLOR, SELECTION_BORDER_PX)
    if band is None:
        return
    preview = band_preview_rect(band, model.layout)
    if preview is not None:
        _fill(vp, preview, BAND_COLOR)
    label.text = band_label(band)
    px, py = vp.to_px(band.x1, band.y1)
    label.x, label.y = px + LABEL_OFFSET_PX, py + LABEL_OFFSET_PX
    # Backing plate keeps the readout legible over walls and the grid.
    arcade.draw_lrbt_rectangle_filled(
        label.x - LABEL_PAD_PX,
        label.x + label.content_width + LABEL_PAD_PX,
        label.y - LABEL_PAD_PX,
        label.y + label.content_height + LABEL_PAD_PX,
        LABEL_BG_COLOR,
    )
    label.draw()
