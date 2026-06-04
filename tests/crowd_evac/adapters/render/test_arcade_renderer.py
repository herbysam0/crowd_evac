"""Tests for crowd_evac.adapters.render.arcade_renderer (Step 1.16).

Covers:
  - :func:`build_agent_render_items` — pure function, no GL required.
  - :class:`ArcadeRenderer` in ``headless=True`` mode — construction and
    render without a display.
  - Snapshot immutability: ``render()`` must not alter any array in the
    snapshot (R7.3).
  - Module import safety: importing the module must not require a GL context.

Actual GL draw calls are tested in :class:`TestArcadeRendererGL`, which is
marked ``@pytest.mark.render`` and skipped in headless CI.  Run them with::

    pytest tests/crowd_evac/adapters/render/test_arcade_renderer.py -m render
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.testing as npt
import pytest

from crowd_evac.adapters.render.arcade_renderer import (
    CALM_COLOR,
    PANIC_COLOR,
    AgentDrawItem,
    ArcadeRenderer,
    build_agent_render_items,
)
from crowd_evac.domain.constants import AGENT_RADIUS
from crowd_evac.application.simulation import SimSnapshot
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MakeSnapshot = Callable[..., SimSnapshot]


@pytest.fixture
def floor_plan() -> FloorPlan:
    """Return a 10×10 m FloorPlan with a single north exit."""
    exit_ = Exit(
        x=5.0,
        y=10.0,
        width_m=2.0,
        side=ExitSide.NORTH,
        capacity_per_second=3,
    )
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(),
        obstacles=(),
        exits=(exit_,),
    )


@pytest.fixture
def make_snapshot() -> MakeSnapshot:
    """Return a factory that builds a minimal SimSnapshot.

    The factory accepts:
      - ``n_agents``: total agent slots (alive + dead), default 5.
      - ``n_dead``: how many of the first n_dead agents are marked dead, default 0.
      - ``panic_level``: uniform panic value for all alive agents, default 0.0.

    Agents are spread 0.5 m apart along the x-axis at y = 1.0 m.
    """

    def _make(
        n_agents: int = 5,
        n_dead: int = 0,
        panic_level: float = 0.0,
    ) -> SimSnapshot:
        """Build a :class:`SimSnapshot` for testing."""
        positions = np.zeros((n_agents, 2), dtype=np.float64)
        for i in range(n_agents):
            positions[i] = [float(i) * 0.5 + 0.5, 1.0]
        velocities = np.zeros((n_agents, 2), dtype=np.float64)
        panics = np.full(n_agents, panic_level, dtype=np.float64)
        alive = np.ones(n_agents, dtype=np.bool_)
        alive[:n_dead] = False
        goals = np.zeros(n_agents, dtype=np.intp)
        return SimSnapshot(
            tick=0,
            sim_time=0.0,
            positions=positions,
            velocities=velocities,
            panics=panics,
            alive=alive,
            goals=goals,
            evacuated_count=n_dead,
            active_count=n_agents - n_dead,
            panic_sources=(),
            events=(),
        )

    return _make


# ---------------------------------------------------------------------------
# Tests: build_agent_render_items (pure, no GL)
# ---------------------------------------------------------------------------


class TestBuildAgentRenderItems:
    """Test suite for build_agent_render_items."""

    # -- Happy path --------------------------------------------------------

    def test_all_alive_produces_one_item_per_agent(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Each alive agent maps to exactly one AgentDrawItem."""
        snap = make_snapshot(n_agents=5, n_dead=0)
        items = build_agent_render_items(snap, pixels_per_meter=40.0)
        assert len(items) == 5

    def test_item_count_equals_active_count(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """len(items) must equal snapshot.active_count."""
        snap = make_snapshot(n_agents=8, n_dead=3)
        items = build_agent_render_items(snap, pixels_per_meter=40.0)
        assert len(items) == snap.active_count

    def test_pixel_coords_are_scaled_by_ppm(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Pixel coords equal world metres multiplied by pixels_per_meter."""
        snap = make_snapshot(n_agents=1, n_dead=0)
        ppm = 50.0
        items = build_agent_render_items(snap, pixels_per_meter=ppm)
        assert len(items) == 1
        # Agent 0 spawns at (0.5 m, 1.0 m)
        assert items[0].x_px == pytest.approx(0.5 * ppm)
        assert items[0].y_px == pytest.approx(1.0 * ppm)

    def test_calm_agent_receives_calm_color(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """A panic=0 agent gets exactly CALM_COLOR."""
        snap = make_snapshot(n_agents=1, n_dead=0, panic_level=0.0)
        items = build_agent_render_items(snap)
        assert items[0].color == CALM_COLOR

    def test_fully_panicked_agent_receives_panic_color(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """A panic=1 agent gets exactly PANIC_COLOR."""
        snap = make_snapshot(n_agents=1, n_dead=0, panic_level=1.0)
        items = build_agent_render_items(snap)
        assert items[0].color == PANIC_COLOR

    def test_custom_radius_propagated_to_items(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Custom agent_radius_m is reflected in every item (converted to pixels)."""
        snap = make_snapshot(n_agents=3)
        ppm = 40.0
        agent_radius_m = 0.25  # 0.25 m * 40 ppm = 10 px
        items = build_agent_render_items(snap, pixels_per_meter=ppm, agent_radius_m=agent_radius_m)
        assert all(item.radius_px == 10 for item in items)

    # -- Edge cases --------------------------------------------------------

    def test_zero_agent_snapshot_returns_empty_list(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Zero-agent snapshot produces an empty list."""
        snap = make_snapshot(n_agents=0)
        assert build_agent_render_items(snap) == []

    def test_all_dead_returns_empty_list(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """If every agent is dead, no items are produced."""
        snap = make_snapshot(n_agents=4, n_dead=4)
        assert build_agent_render_items(snap) == []

    def test_partial_dead_produces_alive_subset(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Only alive agents appear in the result."""
        snap = make_snapshot(n_agents=6, n_dead=2)
        assert len(build_agent_render_items(snap)) == 4

    # -- Return type -------------------------------------------------------

    def test_items_are_agent_draw_item_instances(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """All returned objects are AgentDrawItem instances."""
        snap = make_snapshot(n_agents=3)
        items = build_agent_render_items(snap)
        assert all(isinstance(item, AgentDrawItem) for item in items)

    def test_agent_draw_item_is_immutable(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """AgentDrawItem must be a frozen (immutable) dataclass."""
        snap = make_snapshot(n_agents=1)
        item = build_agent_render_items(snap)[0]
        with pytest.raises((AttributeError, TypeError)):
            item.x_px = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: ArcadeRenderer (headless=True, no GL)
# ---------------------------------------------------------------------------


class TestArcadeRendererHeadless:
    """Test ArcadeRenderer in headless=True mode; no GL context required."""

    # -- Happy path --------------------------------------------------------

    def test_headless_construction_does_not_raise(
        self, floor_plan: FloorPlan
    ) -> None:
        """ArcadeRenderer(floor_plan, headless=True) must not raise."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        assert renderer is not None

    def test_floor_plan_attribute_is_stored(
        self, floor_plan: FloorPlan
    ) -> None:
        """renderer.floor_plan must be the exact FloorPlan passed in."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        assert renderer.floor_plan is floor_plan

    def test_headless_render_does_not_raise(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() in headless mode completes without exception."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        renderer.render(make_snapshot(n_agents=5))

    # -- Snapshot immutability (R7.3) --------------------------------------

    def test_render_does_not_mutate_positions(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() must not alter snapshot.positions."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=4)
        pos_before = snap.positions.copy()
        renderer.render(snap)
        npt.assert_array_equal(snap.positions, pos_before)

    def test_render_does_not_mutate_panics(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() must not alter snapshot.panics."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=4, panic_level=0.6)
        panics_before = snap.panics.copy()
        renderer.render(snap)
        npt.assert_array_equal(snap.panics, panics_before)

    def test_render_does_not_mutate_alive(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() must not alter snapshot.alive."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=4, n_dead=1)
        alive_before = snap.alive.copy()
        renderer.render(snap)
        npt.assert_array_equal(snap.alive, alive_before)

    def test_render_does_not_mutate_velocities(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() must not alter snapshot.velocities."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=3)
        vel_before = snap.velocities.copy()
        renderer.render(snap)
        npt.assert_array_equal(snap.velocities, vel_before)

    def test_render_does_not_mutate_tick(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() must leave snapshot.tick unchanged."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=2)
        renderer.render(snap)
        assert snap.tick == 0

    # -- Edge cases --------------------------------------------------------

    def test_render_with_zero_agents_does_not_raise(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() with an empty population must not raise."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        renderer.render(make_snapshot(n_agents=0))

    def test_render_with_all_dead_does_not_raise(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """render() when all agents are dead must not raise."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        renderer.render(make_snapshot(n_agents=5, n_dead=5))

    def test_custom_ppm_is_stored(self, floor_plan: FloorPlan) -> None:
        """pixels_per_meter kwarg is stored on the renderer."""
        renderer = ArcadeRenderer(floor_plan, headless=True, pixels_per_meter=80.0)
        assert renderer._ppm == 80.0  # noqa: SLF001

    def test_default_agent_radius_computed_from_meters(
        self, floor_plan: FloorPlan
    ) -> None:
        """Default agent_radius_px is computed as agent_radius_m * pixels_per_meter."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        # Default: AGENT_RADIUS (0.2 m) * PIXELS_PER_METER (40 px/m) = 8 px
        expected_px = int(round(AGENT_RADIUS * 40.0))
        assert renderer._agent_radius_px == expected_px  # noqa: SLF001

    def test_render_called_multiple_times_does_not_raise(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """Calling render() repeatedly with the same snapshot is idempotent."""
        renderer = ArcadeRenderer(floor_plan, headless=True)
        snap = make_snapshot(n_agents=3)
        renderer.render(snap)
        renderer.render(snap)
        renderer.render(snap)


# ---------------------------------------------------------------------------
# Tests: module import safety
# ---------------------------------------------------------------------------


class TestModuleImportSafety:
    """Verify the module is import-safe without a GL context."""

    def test_arcade_renderer_class_is_importable(self) -> None:
        """ArcadeRenderer must be importable without a display."""
        import crowd_evac.adapters.render.arcade_renderer as mod

        assert hasattr(mod, "ArcadeRenderer")
        assert callable(mod.ArcadeRenderer)

    def test_build_agent_render_items_is_importable(self) -> None:
        """build_agent_render_items must be importable without a display."""
        import crowd_evac.adapters.render.arcade_renderer as mod

        assert hasattr(mod, "build_agent_render_items")
        assert callable(mod.build_agent_render_items)

    def test_agent_draw_item_is_importable(self) -> None:
        """AgentDrawItem must be importable without a display."""
        import crowd_evac.adapters.render.arcade_renderer as mod

        assert hasattr(mod, "AgentDrawItem")

    def test_color_constants_are_valid_rgba_tuples(self) -> None:
        """Module-level colour constants are 4-element int tuples in [0, 255]."""
        import crowd_evac.adapters.render.arcade_renderer as mod

        for name in ("CALM_COLOR", "PANIC_COLOR", "FLOOR_COLOR", "WALL_COLOR"):
            value = getattr(mod, name)
            assert isinstance(value, tuple), f"{name} must be a tuple"
            assert len(value) == 4, f"{name} must have 4 components"
            assert all(isinstance(c, int) and 0 <= c <= 255 for c in value), (
                f"{name} components must be int in [0, 255]"
            )


# ---------------------------------------------------------------------------
# Tests: ArcadeRenderer with a real GL window (require --display)
# Run with: pytest -m render
# ---------------------------------------------------------------------------


@pytest.mark.render
class TestArcadeRendererGL:
    """Tests that open a real arcade window and exercise the GL draw path.

    These tests require a display and a working GPU/Mesa driver.  They are
    excluded from headless CI.  Run them manually on the target laptop::

        pytest tests/crowd_evac/adapters/render/test_arcade_renderer.py -m render -v
    """

    def test_renderer_draws_one_frame_without_crash(
        self, floor_plan: FloorPlan, make_snapshot: MakeSnapshot
    ) -> None:
        """Open a tiny window, render one frame from a snapshot, then close.

        Verifies the full GL draw path: sprite list build, geometry draw,
        and sprite draw all succeed without exception.
        """
        import arcade

        class _OneFrameWindow(arcade.Window):
            """Minimal window that renders one snapshot frame and exits."""

            def __init__(self) -> None:
                """Initialise the window and renderer."""
                super().__init__(400, 400, "arcade_renderer GL test")
                self.renderer = ArcadeRenderer(
                    floor_plan,
                    pixels_per_meter=40.0,
                    headless=False,
                )
                self.snap = make_snapshot(n_agents=10, n_dead=2)
                self._done = False

            def on_draw(self) -> None:
                """Render one frame and signal completion."""
                self.clear()
                self.renderer.render(self.snap)
                self._done = True
                arcade.exit()

        window = _OneFrameWindow()
        try:
            arcade.run()
        finally:
            window.close()
        assert window._done, "on_draw was never called"  # noqa: SLF001

    def test_renderer_with_walls_and_obstacles_does_not_crash(
        self, make_snapshot: MakeSnapshot
    ) -> None:
        """Geometry rendering (walls + obstacles) must not raise on a real window."""
        import arcade

        from crowd_evac.domain.floor_plan import Obstacle, Wall

        fp = FloorPlan(
            width_m=12.0,
            height_m=8.0,
            walls=(Wall(x=0.0, y=0.0, width=0.3, height=8.0, label="west"),),
            obstacles=(Obstacle(x=5.0, y=2.0, width=1.0, height=3.0, label="pillar"),),
            exits=(
                Exit(x=12.0, y=4.0, width_m=1.5, side=ExitSide.EAST, capacity_per_second=3),
            ),
        )

        class _GeomWindow(arcade.Window):
            """Window that renders one frame with geometry and exits."""

            def __init__(self) -> None:
                """Initialise with complex floor plan."""
                super().__init__(480, 320, "geom GL test")
                self.renderer = ArcadeRenderer(fp, headless=False)
                self.snap = make_snapshot(n_agents=5)
                self._done = False

            def on_draw(self) -> None:
                """Render one frame and exit."""
                self.clear()
                self.renderer.render(self.snap)
                self._done = True
                arcade.exit()

        window = _GeomWindow()
        try:
            arcade.run()
        finally:
            window.close()
        assert window._done  # noqa: SLF001
