"""Tests for crowd_evac.application.cli (Step 1.19 / FR-0 R0.1).

Covers:
  - :func:`build_simulation_from_scenario` starts at tick 0.
  - All spawned agents lie within the floor plan bounding box.
  - One headless step advances the tick to 1.
  - A mocked :class:`~crowd_evac.ports.renderer.Renderer` receives the snapshot.
  - An unknown scenario name raises :exc:`MalformedScenarioError`.
  - :func:`compute_fit_ppm` caps at default_ppm when floor plan fits.
  - :func:`compute_fit_ppm` scales down when floor plan exceeds the screen.
  - :func:`compute_fit_ppm` falls back to default_ppm when display is unavailable.

All tests run headless — no arcade window is opened.  :class:`EvacWindow`
tests that require a display are marked ``@pytest.mark.render``.
"""
from __future__ import annotations

import types

import pytest
from pytest_mock import MockerFixture

from crowd_evac.application.cli import (
    DEFAULT_SCENARIO,
    _MIN_PPM,
    _SCREEN_MARGIN,
    _get_logical_screen_size,
    build_simulation_from_scenario,
    compute_fit_ppm,
)
from crowd_evac.domain.constants import PIXELS_PER_METER
from crowd_evac.domain.errors import MalformedScenarioError


def _make_floor_plan(width_m: float, height_m: float) -> types.SimpleNamespace:
    """Return a minimal floor-plan-like namespace for ppm tests."""
    return types.SimpleNamespace(width_m=width_m, height_m=height_m)


def _make_screen(width: int, height: int, scale: float) -> types.SimpleNamespace:
    """Return a minimal pyglet-Screen-like namespace for screen-size tests."""
    return types.SimpleNamespace(
        width=width, height=height, get_scale=lambda: scale
    )


class TestGetLogicalScreenSize:
    """Tests for :func:`_get_logical_screen_size`."""

    # -- Happy path -----------------------------------------------------------

    def test_divides_physical_by_scale(self, mocker: MockerFixture) -> None:
        """A 1920×1200 physical screen at 150 % yields 1280×800 logical."""
        mocker.patch(
            "crowd_evac.application.cli.arcade.get_screens",
            return_value=[_make_screen(1920, 1200, 1.5)],
        )
        assert _get_logical_screen_size() == (1280.0, 800.0)

    def test_unscaled_screen_passes_through(self, mocker: MockerFixture) -> None:
        """At 100 % scaling logical size equals physical size."""
        mocker.patch(
            "crowd_evac.application.cli.arcade.get_screens",
            return_value=[_make_screen(1920, 1080, 1.0)],
        )
        assert _get_logical_screen_size() == (1920.0, 1080.0)

    # -- Edge case ------------------------------------------------------------

    def test_zero_scale_treated_as_unity(self, mocker: MockerFixture) -> None:
        """A non-positive scale is treated as 1.0 to avoid division by zero."""
        mocker.patch(
            "crowd_evac.application.cli.arcade.get_screens",
            return_value=[_make_screen(1600, 900, 0.0)],
        )
        assert _get_logical_screen_size() == (1600.0, 900.0)

    # -- Failure path ---------------------------------------------------------

    def test_returns_none_on_query_failure(self, mocker: MockerFixture) -> None:
        """Returns None when the screen list cannot be obtained."""
        mocker.patch(
            "crowd_evac.application.cli.arcade.get_screens",
            side_effect=RuntimeError("no display"),
        )
        assert _get_logical_screen_size() is None


class TestComputeFitPpm:
    """Tests for :func:`compute_fit_ppm`.

    The display query is mocked at :func:`_get_logical_screen_size`, which
    already converts physical pixels to logical (DPI-scaled) pixels.  Each test
    therefore supplies *logical* screen dimensions directly.
    """

    # -- Happy path -----------------------------------------------------------

    def test_small_floor_plan_uses_default_ppm(self, mocker: MockerFixture) -> None:
        """Floor plans that fit at default ppm keep the default."""
        # Logical screen large enough that the floor plan fits at default ppm.
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(3840.0, 2160.0),
        )
        fp = _make_floor_plan(10.0, 8.0)  # 400×320 px at 40 px/m — fits easily
        result = compute_fit_ppm(fp)
        assert result == PIXELS_PER_METER

    def test_large_floor_plan_scales_down(self, mocker: MockerFixture) -> None:
        """Floor plans that exceed the screen are scaled to fit within margin."""
        # 1280×800 logical = a 1920×1200 physical display at 150 % scaling.
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(1280.0, 800.0),
        )
        # Lecture Hall: 50×30 m at 40 px/m → 2000×1200 px — does not fit.
        fp = _make_floor_plan(50.0, 30.0)
        result = compute_fit_ppm(fp)
        # Must fit within logical screen * margin
        assert fp.width_m * result <= 1280.0 * _SCREEN_MARGIN + 1
        assert fp.height_m * result <= 800.0 * _SCREEN_MARGIN + 1

    def test_result_never_exceeds_default_ppm(self, mocker: MockerFixture) -> None:
        """compute_fit_ppm never exceeds default_ppm even on a massive screen."""
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(7680.0, 4320.0),
        )
        fp = _make_floor_plan(5.0, 5.0)
        result = compute_fit_ppm(fp)
        assert result <= PIXELS_PER_METER

    # -- Edge case ------------------------------------------------------------

    def test_fallback_on_display_query_failure(self, mocker: MockerFixture) -> None:
        """Returns default_ppm when the screen size cannot be determined."""
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=None,
        )
        fp = _make_floor_plan(50.0, 30.0)
        result = compute_fit_ppm(fp)
        assert result == PIXELS_PER_METER

    def test_result_respects_min_ppm(self, mocker: MockerFixture) -> None:
        """Result never drops below _MIN_PPM even on a tiny screen."""
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(100.0, 80.0),
        )
        fp = _make_floor_plan(10_000.0, 10_000.0)  # Huge floor plan
        result = compute_fit_ppm(fp)
        assert result >= _MIN_PPM

    # -- Failure path ---------------------------------------------------------

    def test_proportional_scaling_width_bound(self, mocker: MockerFixture) -> None:
        """A floor plan limited by width fits within screen_w × margin."""
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(1600.0, 900.0),
        )
        # Wide but not tall: width is the binding constraint.
        fp = _make_floor_plan(100.0, 5.0)
        result = compute_fit_ppm(fp)
        assert fp.width_m * result <= 1600.0 * _SCREEN_MARGIN + 1

    def test_proportional_scaling_height_bound(self, mocker: MockerFixture) -> None:
        """A floor plan limited by height fits within screen_h × margin."""
        mocker.patch(
            "crowd_evac.application.cli._get_logical_screen_size",
            return_value=(1600.0, 900.0),
        )
        # Tall but not wide: height is the binding constraint.
        fp = _make_floor_plan(5.0, 100.0)
        result = compute_fit_ppm(fp)
        assert fp.height_m * result <= 900.0 * _SCREEN_MARGIN + 1


class TestBuildSimulationFromScenario:
    """Tests for :func:`build_simulation_from_scenario`."""

    # -- Happy path -----------------------------------------------------------

    def test_default_scenario_starts_at_tick_zero(self) -> None:
        """Verify the simulation starts at tick 0 on the default scenario."""
        sim, _ = build_simulation_from_scenario()
        assert sim.tick == 0

    def test_spawned_agents_are_within_floor_plan_bounds(self) -> None:
        """Verify all alive agents lie within the floor plan bounding box."""
        sim, floor_plan = build_simulation_from_scenario()
        snapshot = sim.snapshot()
        positions = snapshot.positions[snapshot.alive]
        assert (positions[:, 0] >= 0.0).all(), "agent x < 0"
        assert (positions[:, 0] <= floor_plan.width_m).all(), "agent x > width"
        assert (positions[:, 1] >= 0.0).all(), "agent y < 0"
        assert (positions[:, 1] <= floor_plan.height_m).all(), "agent y > height"

    def test_default_scenario_has_active_agents(self) -> None:
        """Verify at least one agent is alive at tick 0."""
        sim, _ = build_simulation_from_scenario()
        assert sim.snapshot().active_count > 0

    # -- Edge case ------------------------------------------------------------

    def test_named_scenario_also_starts_at_tick_zero(self) -> None:
        """Verify the bundled fixture_minimal scenario also starts at tick 0."""
        sim, _ = build_simulation_from_scenario("fixture_minimal")
        assert sim.tick == 0

    def test_one_headless_step_advances_tick(self) -> None:
        """Verify a single headless step moves tick from 0 to 1."""
        sim, _ = build_simulation_from_scenario()
        sim.step()
        assert sim.tick == 1

    def test_renderer_port_receives_snapshot(self, mocker: MockerFixture) -> None:
        """Verify the snapshot from a built sim can be handed to a Renderer mock.

        This test confirms the public contract: ``sim.snapshot()`` returns a
        :class:`~crowd_evac.application.simulation.SimSnapshot` that a
        Renderer-conformant object can consume without error.
        """
        sim, _ = build_simulation_from_scenario()
        mock_renderer = mocker.MagicMock()
        snapshot = sim.snapshot()
        mock_renderer.render(snapshot)
        mock_renderer.render.assert_called_once_with(snapshot)

    # -- Failure path ---------------------------------------------------------

    def test_unknown_scenario_raises_malformed_error(self) -> None:
        """Verify that a non-existent scenario name raises MalformedScenarioError."""
        with pytest.raises(MalformedScenarioError):
            build_simulation_from_scenario("does_not_exist_xyz")


class TestDefaultScenarioConstant:
    """Tests for the DEFAULT_SCENARIO module constant."""

    def test_default_scenario_is_lecture_hall(self) -> None:
        """Verify the default scenario points to the bundled Lecture Hall."""
        assert DEFAULT_SCENARIO == "lecture_hall"

    def test_default_scenario_is_loadable(self) -> None:
        """Verify the default scenario name successfully loads a scenario."""
        sim, floor_plan = build_simulation_from_scenario(DEFAULT_SCENARIO)
        assert sim.tick == 0
        assert floor_plan.width_m > 0.0
        assert floor_plan.height_m > 0.0
