"""Arcade renderer adapter implementing the Renderer port (FR-7 R7.1 / R7.3).

Draws the walkable region, walls, obstacles, exits, agents, and panic
sources from a read-only :class:`~crowd_evac.application.simulation.SimSnapshot`
using arcade's :class:`~arcade.SpriteList` of :class:`~arcade.SpriteCircle`
instances — the same rendering path validated to ≥ 30 FPS at 10k agents by
the Step 1.3 spike.

Architecture
------------
The module is split into two layers:

- **Pure helpers** (:func:`build_agent_render_items`) — no GL dependency,
  safely callable in headless CI. These compute pixel-space render data from
  the snapshot alone and are unit-tested without a display.
- :class:`ArcadeRenderer` — holds arcade sprite lists and issues draw calls.
  Intended to be constructed inside an arcade :class:`~arcade.Window` context
  (after the GL context is live) and called from the window's ``on_draw``
  handler.

Coordinate system
-----------------
Domain coordinates are in metres with origin at the room bottom-left.
Arcade's default origin is also bottom-left with y increasing upward, so
conversion is simply ``pixel = metre × pixels_per_meter`` on both axes —
no y-axis flip is required.

Immutability (R7.3)
-------------------
:meth:`ArcadeRenderer.render` only reads from the snapshot. All NumPy arrays
in a :class:`~crowd_evac.application.simulation.SimSnapshot` are already
independent copies produced by
:meth:`~crowd_evac.application.simulation.Simulation.snapshot`, so the
renderer cannot mutate live simulation state even by accident; we read them
as read-only by convention regardless.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import arcade
from arcade.types import RGBA255

from crowd_evac.domain.constants import (
    AGENT_RADIUS,
    FIRE_SYMBOL_SIZE_M,
    PIXELS_PER_METER,
)
from crowd_evac.domain.floor_plan import ExitSide, FloorPlan
from crowd_evac.domain.panic_source import PanicSource

if TYPE_CHECKING:
    from crowd_evac.application.simulation import SimSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

EXIT_DEPTH_M: float = 0.5
"""Depth of the exit-opening rectangle drawn perpendicular to the wall (m)."""

CALM_COLOR: RGBA255 = (50, 120, 220, 220)
"""Agent colour at zero panic (calm — blue)."""

PANIC_COLOR: RGBA255 = (230, 50, 30, 220)
"""Agent colour at maximum panic (panicked — red)."""

FLOOR_COLOR: RGBA255 = (38, 38, 48, 255)
"""Room bounding-box background colour."""

WALL_COLOR: RGBA255 = (90, 90, 100, 255)
"""Wall segment fill colour."""

OBSTACLE_COLOR: RGBA255 = (65, 65, 75, 255)
"""Interior obstacle fill colour."""

EXIT_COLOR: RGBA255 = (55, 190, 75, 220)
"""Exit opening highlight colour (green)."""

PANIC_SOURCE_COLOR: RGBA255 = (235, 110, 25, 100)
"""Panic source base colour (orange); alpha is scaled by source intensity."""

# ---------------------------------------------------------------------------
# Emergency symbol configuration
# ---------------------------------------------------------------------------

_SYMBOL_CHAR: dict[str, str] = {
    "fire": "\U0001f525",  # U+1F525 FIRE — rendered by Segoe UI Emoji on Windows 11
}
"""Maps source_type → Unicode character drawn at the hazard position.

Each entry is rendered by :meth:`ArcadeRenderer._draw_panic_sources` using the
OS emoji font.  Add entries here to support additional hazard types.
"""

_SYMBOL_SIZE_M: dict[str, float] = {
    "fire": FIRE_SYMBOL_SIZE_M,
}
"""Maps source_type → symbol world-space diameter in metres.

The rendered symbol is scaled to ``size_m × pixels_per_meter`` pixels on each
call so it stays fixed in world space regardless of the rendering scale.
Add or override entries to configure the visual size per hazard type.
"""

_SYMBOL_FONTS: tuple[str, ...] = (
    "Segoe UI Emoji",    # Windows 11
    "Apple Color Emoji", # macOS / iOS
    "Noto Color Emoji",  # Linux / Android
)
"""Font fallback chain for emoji symbol rendering.

Pyglet tries each name in order; the first one found by the OS is used.
The chain covers Windows, macOS, and common Linux emoji fonts.
"""


# ---------------------------------------------------------------------------
# Pure data types and helpers (no GL dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentDrawItem:
    """Pixel-space render data for one active agent (no GL dependency).

    Produced by :func:`build_agent_render_items`.  Using a plain dataclass
    keeps the pure computation layer decoupled from arcade's GL types so it
    can be tested without a display.

    Attributes:
        x_px: Pixel x-coordinate (right = positive).
        y_px: Pixel y-coordinate (up = positive, matching arcade's origin).
        radius_px: Circle sprite radius in pixels.
        color: RGBA colour interpolated from :data:`CALM_COLOR` to
            :data:`PANIC_COLOR` based on the agent's panic level.
    """

    x_px: float
    y_px: float
    radius_px: int
    color: RGBA255


def _interpolate_color(c0: RGBA255, c1: RGBA255, t: float) -> RGBA255:
    """Linearly interpolate between two RGBA colours.

    Args:
        c0: Start colour at ``t = 0``.
        c1: End colour at ``t = 1``.
        t: Interpolation parameter, clamped to ``[0, 1]``.

    Returns:
        Interpolated :data:`~arcade.types.RGBA255` colour.
    """
    t = max(0.0, min(1.0, t))
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
        int(c0[3] + (c1[3] - c0[3]) * t),
    )


def build_agent_render_items(
    snapshot: SimSnapshot,
    pixels_per_meter: float = PIXELS_PER_METER,
    agent_radius_m: float = AGENT_RADIUS,
) -> list[AgentDrawItem]:
    """Map alive agents from a snapshot to pixel-space render items.

    Pure function — no GL or arcade dependency.  Returns exactly one
    :class:`AgentDrawItem` per alive agent (``snapshot.alive[i] is True``),
    in ascending agent-index order.  Dead / egressed agents are omitted, so
    ``len(result) == snapshot.active_count``.

    The snapshot is never mutated; this function only reads from it.

    Agent radius in pixels is computed as ``agent_radius_m * pixels_per_meter``,
    so visual size scales with the rendering scale factor and matches the
    agent's physical size (0.2 m = 40 cm diameter).

    Args:
        snapshot: Read-only simulation state at the current tick.
        pixels_per_meter: World-to-pixel scale factor.
        agent_radius_m: Agent radius in metres. Pixel radius is computed as
            ``agent_radius_m * pixels_per_meter``.

    Returns:
        List of :class:`AgentDrawItem` instances, one per alive agent.

    Example:
        >>> items = build_agent_render_items(snap, pixels_per_meter=40.0)
        >>> len(items) == snap.active_count
        True
    """
    items: list[AgentDrawItem] = []
    positions = snapshot.positions
    panics = snapshot.panics
    alive = snapshot.alive
    agent_radius_px = int(round(agent_radius_m * pixels_per_meter))

    for i in range(len(alive)):
        if not alive[i]:
            continue
        x_px = float(positions[i, 0]) * pixels_per_meter
        y_px = float(positions[i, 1]) * pixels_per_meter
        p = max(0.0, min(1.0, float(panics[i])))
        color = _interpolate_color(CALM_COLOR, PANIC_COLOR, p)
        items.append(
            AgentDrawItem(
                x_px=x_px, y_px=y_px, radius_px=agent_radius_px, color=color
            )
        )

    return items


# ---------------------------------------------------------------------------
# ArcadeRenderer
# ---------------------------------------------------------------------------


def _draw_filled_rect(
    left: float, bottom: float, width: float, height: float, color: RGBA255
) -> None:
    """Draw a filled axis-aligned rectangle via arcade.

    Wraps the arcade 3.x draw API so callers use left/bottom/width/height
    (LBWH) rather than left/right/bottom/top (which varies by arcade version).

    Args:
        left: Left edge x-coordinate in pixels.
        bottom: Bottom edge y-coordinate in pixels.
        width: Rectangle width in pixels.
        height: Rectangle height in pixels.
        color: Fill colour.
    """
    arcade.draw_lrbt_rectangle_filled(
        left, left + width, bottom, bottom + height, color
    )


class ArcadeRenderer:
    """Arcade-backed Renderer adapter (FR-7 R7.1 / R7.3).

    Draws the simulation walkable region, walls, obstacles, exits, panic
    sources, and agents from a read-only :class:`SimSnapshot`.  The renderer
    never mutates the snapshot or any domain object (R7.3).

    Agent sprites are managed as an :class:`~arcade.SpriteList` of
    :class:`~arcade.SpriteCircle` instances — the same batch-draw path
    validated to ≥ 30 FPS at 10k agents by the Step 1.3 spike.  Agent visual
    size is proportional to the rendering scale (``pixels_per_meter``), so it
    always displays at the agent's physical radius (0.2 m = 40 cm diameter).
    Alive agents are colour-coded by panic level (calm = blue → panicked = red);
    dead / egressed agents are hidden by setting ``visible = False``.

    Geometry (floor, walls, obstacles, exits) is drawn using arcade's
    immediate-mode draw calls on every frame.  The geometry is static over a
    scenario so the overhead is constant and negligible at Phase 1 scales.

    Headless mode
    -------------
    Pass ``headless=True`` to skip all GL initialisation.  The pure
    computation path (:func:`build_agent_render_items`) is still exercised
    inside :meth:`render`, keeping it testable in headless CI without a
    display.

    Args:
        floor_plan: Static floor geometry for this scenario.  Stored as a
            reference; the renderer reads it on every :meth:`render` call but
            never mutates it.
        pixels_per_meter: World-to-pixel scale factor (default 40 px/m).
        agent_radius_m: Agent radius in metres (default 0.2 m = 40 cm diameter).
            Pixel radius is computed as ``agent_radius_m * pixels_per_meter``.
        background_image: Optional filesystem path to a background image
            (R7.1).  Loaded once at GL init; silently skipped on load failure.
        headless: If ``True``, skip all GL operations. For testing only.
        y_offset_px: Vertical offset in pixels applied to all drawn geometry
            and agents.  Use this when the window reserves space at the bottom
            (e.g. a control panel) so the simulation viewport starts above y=0.

    Attributes:
        floor_plan: The :class:`~crowd_evac.domain.floor_plan.FloorPlan`
            this renderer is bound to.
    """

    def __init__(
        self,
        floor_plan: FloorPlan,
        *,
        pixels_per_meter: float = PIXELS_PER_METER,
        agent_radius_m: float = AGENT_RADIUS,
        background_image: str | None = None,
        headless: bool = False,
        y_offset_px: float = 0.0,
    ) -> None:
        self.floor_plan = floor_plan
        self._ppm = pixels_per_meter
        self._agent_radius_m = agent_radius_m
        self._agent_radius_px = int(round(agent_radius_m * pixels_per_meter))
        self._background_image = background_image
        self._headless = headless
        self._y_offset_px = y_offset_px

        self._agent_sprites: arcade.SpriteList[arcade.SpriteCircle] | None = None
        # Background image sprite is held in a SpriteList so arcade 3.x
        # batch-draw API can render it (Sprite has no standalone draw() in 3.x).
        self._background_list: arcade.SpriteList[arcade.Sprite] = (
            arcade.SpriteList()
        )

        if not headless:
            self._init_gl()

        logger.debug(
            "ArcadeRenderer constructed (headless=%s, ppm=%.1f, agent_radius=%.2f m (%.1f px), "
            "floor=%.1f×%.1f m, y_offset=%.0f px).",
            headless,
            pixels_per_meter,
            agent_radius_m,
            self._agent_radius_px,
            floor_plan.width_m,
            floor_plan.height_m,
            y_offset_px,
        )

    def _init_gl(self) -> None:
        """Initialise GL resources; requires an active arcade GL context.

        Creates the empty agent :class:`~arcade.SpriteList` and, if a
        background image path was provided, loads it as an
        :class:`~arcade.Sprite` centred on the room bounding box and adds it
        to the background sprite list.
        """
        self._agent_sprites = arcade.SpriteList()

        if self._background_image is not None:
            cx = self.floor_plan.width_m * self._ppm / 2.0
            cy = self.floor_plan.height_m * self._ppm / 2.0
            try:
                bg_sprite = arcade.Sprite(
                    self._background_image, center_x=cx, center_y=cy
                )
                self._background_list.append(bg_sprite)
            except Exception:
                logger.warning(
                    "Background image '%s' could not be loaded; skipping.",
                    self._background_image,
                )

    def render(self, snapshot: SimSnapshot) -> None:
        """Draw all simulation layers from the snapshot (read-only, R7.3).

        Rendering order:

        1. Optional background image (R7.1).
        2. Room floor.
        3. Walls and interior obstacles.
        4. Exit openings.
        5. Active panic sources.
        6. Agents (alive = panic-coded colour; dead = hidden).

        In headless mode the pure computation step still runs; GL draw calls
        are skipped.

        Args:
            snapshot: Immutable simulation state.  This method reads but
                never mutates any field or array within the snapshot.
        """
        # Pure computation path — exercised even headless so it remains testable
        build_agent_render_items(snapshot, self._ppm, self._agent_radius_m)

        if self._headless:
            return

        self._draw_background()
        self._draw_floor()
        self._draw_geometry()
        self._draw_panic_sources(snapshot.panic_sources)
        self._sync_and_draw_agents(snapshot)

        logger.debug(
            "render tick=%d active=%d evacuated=%d",
            snapshot.tick,
            snapshot.active_count,
            snapshot.evacuated_count,
        )

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_background(self) -> None:
        """Draw the optional background image below all geometry layers (R7.1)."""
        if len(self._background_list) > 0:
            self._background_list.draw()

    def _draw_floor(self) -> None:
        """Fill the room bounding box with the floor background colour."""
        fp = self.floor_plan
        ppm = self._ppm
        _draw_filled_rect(0.0, self._y_offset_px, fp.width_m * ppm, fp.height_m * ppm, FLOOR_COLOR)

    def _draw_geometry(self) -> None:
        """Draw walls, interior obstacles, and exit-opening rectangles."""
        fp = self.floor_plan
        ppm = self._ppm
        y_off = self._y_offset_px

        for wall in fp.walls:
            _draw_filled_rect(
                wall.x * ppm,
                wall.y * ppm + y_off,
                wall.width * ppm,
                wall.height * ppm,
                WALL_COLOR,
            )

        for obs in fp.obstacles:
            _draw_filled_rect(
                obs.x * ppm,
                obs.y * ppm + y_off,
                obs.width * ppm,
                obs.height * ppm,
                OBSTACLE_COLOR,
            )

        for exit_ in fp.exits:
            half = exit_.width_m / 2.0
            depth = EXIT_DEPTH_M
            if exit_.side in (ExitSide.SOUTH, ExitSide.NORTH):
                # Opening spans along x; depth extends along y
                x0 = (exit_.x - half) * ppm
                y0 = (exit_.y - depth / 2.0) * ppm + y_off
                w = exit_.width_m * ppm
                h = depth * ppm
            else:
                # EAST or WEST: opening spans along y; depth extends along x
                x0 = (exit_.x - depth / 2.0) * ppm
                y0 = (exit_.y - half) * ppm + y_off
                w = depth * ppm
                h = exit_.width_m * ppm
            _draw_filled_rect(x0, y0, w, h, EXIT_COLOR)

    def _draw_panic_sources(
        self, sources: tuple[PanicSource, ...]
    ) -> None:
        """Draw active panic sources: influence circle + hazard symbol.

        Each active source is rendered as two layers:

        1. A semi-transparent orange circle whose radius scales with
           ``source.radius × intensity`` so it visually shrinks as the source
           decays.  Alpha is also scaled by intensity so expired sources fade
           before disappearing.
        2. A hazard-type symbol (e.g. 🔥 for ``source_type="fire"``) drawn
           centred at the source position using the OS emoji font.  Symbol size
           is looked up from :data:`_SYMBOL_SIZE_M` (world-space metres),
           defaulting to 1 m when the type is not registered.

        Args:
            sources: Panic sources from the current snapshot (read-only).
        """
        ppm = self._ppm
        y_off = self._y_offset_px
        r_base, g_base, b_base, _ = PANIC_SOURCE_COLOR
        for src in sources:
            if not src.is_active:
                continue
            intensity = max(0.0, min(1.0, src.intensity))
            alpha = int(PANIC_SOURCE_COLOR[3] * intensity)
            color: RGBA255 = (r_base, g_base, b_base, alpha)
            cx_px = src.x * ppm
            cy_px = src.y * ppm + y_off
            arcade.draw_circle_filled(
                cx_px,
                cy_px,
                src.radius * intensity * ppm,
                color,
            )
            symbol = _SYMBOL_CHAR.get(src.source_type)
            if symbol is not None:
                size_m = _SYMBOL_SIZE_M.get(src.source_type, 1.0)
                font_size_px = int(size_m * ppm)
                arcade.draw_text(
                    symbol,
                    cx_px,
                    cy_px,
                    arcade.color.WHITE,
                    font_size=font_size_px,
                    font_name=_SYMBOL_FONTS,
                    anchor_x="center",
                    anchor_y="center",
                )

    def _sync_and_draw_agents(self, snapshot: SimSnapshot) -> None:
        """Synchronise the agent sprite list to snapshot state and draw it.

        The sprite list is rebuilt whenever the total agent-slot count
        changes (only on first call, since ``len(alive)`` is constant over
        a scenario).  Alive agents are positioned and colour-coded; dead
        agents are hidden via ``sprite.visible = False``.

        Args:
            snapshot: Current simulation state (read-only).
        """
        assert self._agent_sprites is not None

        n_total = len(snapshot.alive)

        # Rebuild on first call (or if slot count ever changes)
        if len(self._agent_sprites) != n_total:
            self._agent_sprites = arcade.SpriteList()
            for _ in range(n_total):
                self._agent_sprites.append(
                    arcade.SpriteCircle(self._agent_radius_px, CALM_COLOR)
                )

        positions = snapshot.positions
        panics = snapshot.panics
        alive = snapshot.alive
        ppm = self._ppm
        y_off = self._y_offset_px

        for i, sprite in enumerate(self._agent_sprites):
            is_alive = bool(alive[i])
            sprite.visible = is_alive
            if is_alive:
                sprite.center_x = float(positions[i, 0]) * ppm
                sprite.center_y = float(positions[i, 1]) * ppm + y_off
                p = max(0.0, min(1.0, float(panics[i])))
                sprite.color = _interpolate_color(CALM_COLOR, PANIC_COLOR, p)

        self._agent_sprites.draw()
