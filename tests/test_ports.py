"""Tests for ports — verify Protocol implementations type-check."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.ports import Clock, InputEvent, InputSource, Renderer, ScenarioRepository

if TYPE_CHECKING:
    from crowd_evac.application.simulation import SimSnapshot


class FakeRenderer:
    """Trivial Renderer implementation."""

    def render(self, snapshot: SimulationSnapshot) -> None:
        """Do nothing."""
        pass


class FakeClock:
    """Trivial Clock implementation."""

    def __init__(self, dt: float = 1 / 60) -> None:
        """Initialize with fixed time step."""
        self._dt = dt
        self._tick = 0

    def tick(self) -> float:
        """Return dt and increment tick."""
        self._tick += 1
        return self._dt

    @property
    def dt(self) -> float:
        """Return fixed time step."""
        return self._dt

    @property
    def current_tick(self) -> int:
        """Return current tick."""
        return self._tick

    @property
    def elapsed_time(self) -> float:
        """Return total elapsed time."""
        return self._tick * self._dt


class FakeInputSource:
    """Trivial InputSource implementation."""

    def poll(self) -> list[InputEvent]:
        """Return empty event list."""
        return []


class FakeScenarioRepository:
    """Trivial ScenarioRepository implementation."""

    def load_scenario(self, scenario_id: str) -> tuple[FloorPlan, int]:
        """Return a minimal FloorPlan and agent count."""
        from crowd_evac.domain.floor_plan import Exit, ExitSide

        exit_obj = Exit(
            x=5.0,
            y=5.0,
            width_m=2.0,
            side=ExitSide.NORTH,
            capacity_per_second=2,
        )
        floor = FloorPlan(
            width_m=10.0,
            height_m=10.0,
            walls=(),
            obstacles=(),
            exits=(exit_obj,),
        )
        return floor, 0


class TestRendererProtocol:
    """Test Renderer protocol conformance."""

    def test_fake_renderer_is_renderer(self) -> None:
        """Verify FakeRenderer implements Renderer protocol."""
        renderer: Renderer = FakeRenderer()
        assert renderer is not None

    def test_renderer_has_render_method(self) -> None:
        """Verify Renderer protocol has render method."""
        renderer = FakeRenderer()
        assert hasattr(renderer, "render")
        assert callable(renderer.render)


class TestClockProtocol:
    """Test Clock protocol conformance."""

    def test_fake_clock_is_clock(self) -> None:
        """Verify FakeClock implements Clock protocol."""
        clock: Clock = FakeClock()
        assert clock is not None

    def test_clock_has_tick_method(self) -> None:
        """Verify Clock has tick method."""
        clock = FakeClock()
        assert hasattr(clock, "tick")
        assert callable(clock.tick)

    def test_clock_tick_returns_float(self) -> None:
        """Verify tick returns a float."""
        clock = FakeClock(dt=0.016)
        elapsed = clock.tick()
        assert isinstance(elapsed, float)
        assert elapsed == 0.016

    def test_clock_dt_property(self) -> None:
        """Verify Clock has dt property."""
        clock = FakeClock(dt=0.02)
        assert clock.dt == 0.02

    def test_clock_current_tick_property(self) -> None:
        """Verify Clock has current_tick property."""
        clock = FakeClock()
        assert clock.current_tick == 0
        clock.tick()
        assert clock.current_tick == 1

    def test_clock_elapsed_time_property(self) -> None:
        """Verify Clock has elapsed_time property."""
        clock = FakeClock(dt=0.016)
        assert clock.elapsed_time == 0.0
        clock.tick()
        assert clock.elapsed_time == pytest.approx(0.016)
        clock.tick()
        assert clock.elapsed_time == pytest.approx(0.032)


class TestInputSourceProtocol:
    """Test InputSource protocol conformance."""

    def test_fake_input_source_is_input_source(self) -> None:
        """Verify FakeInputSource implements InputSource protocol."""
        source: InputSource = FakeInputSource()
        assert source is not None

    def test_input_source_has_poll_method(self) -> None:
        """Verify InputSource has poll method."""
        source = FakeInputSource()
        assert hasattr(source, "poll")
        assert callable(source.poll)

    def test_input_source_poll_returns_list(self) -> None:
        """Verify poll returns a list of InputEvent."""
        source = FakeInputSource()
        events = source.poll()
        assert isinstance(events, list)


class TestScenarioRepositoryProtocol:
    """Test ScenarioRepository protocol conformance."""

    def test_fake_scenario_repository_is_repository(self) -> None:
        """Verify FakeScenarioRepository implements ScenarioRepository protocol."""
        repo: ScenarioRepository = FakeScenarioRepository()
        assert repo is not None

    def test_scenario_repository_has_load_scenario(self) -> None:
        """Verify ScenarioRepository has load_scenario method."""
        repo = FakeScenarioRepository()
        assert hasattr(repo, "load_scenario")
        assert callable(repo.load_scenario)

    def test_scenario_repository_load_scenario_returns_tuple(self) -> None:
        """Verify load_scenario returns (FloorPlan, int) tuple."""
        repo = FakeScenarioRepository()
        floor, count = repo.load_scenario("test")
        assert isinstance(floor, FloorPlan)
        assert isinstance(count, int)
