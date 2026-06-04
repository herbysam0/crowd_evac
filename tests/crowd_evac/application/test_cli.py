"""Tests for crowd_evac.application.cli (Step 1.19 / FR-0 R0.1).

Covers:
  - :func:`build_simulation_from_scenario` starts at tick 0.
  - All spawned agents lie within the floor plan bounding box.
  - One headless step advances the tick to 1.
  - A mocked :class:`~crowd_evac.ports.renderer.Renderer` receives the snapshot.
  - An unknown scenario name raises :exc:`MalformedScenarioError`.

All tests run headless — no arcade window is opened.  :class:`EvacWindow`
tests that require a display are marked ``@pytest.mark.render``.
"""
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from crowd_evac.application.cli import DEFAULT_SCENARIO, build_simulation_from_scenario
from crowd_evac.domain.errors import MalformedScenarioError


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
