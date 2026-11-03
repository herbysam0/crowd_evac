"""Tests for the scripted-event helpers in crowd_evac.application.injection.

Covers:
  - events_by_tick: grouping, ordering, empty input.
  - apply_event: panic source placed with defaults and with overrides.
  - apply_due_events: fires only the events scheduled for the current tick.
"""
from __future__ import annotations

import pytest

from crowd_evac.application.injection import (
    apply_due_events,
    apply_event,
    events_by_tick,
)
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.constants import (
    HAZARD_BLOCK_RADIUS,
    PANIC_DECAY_RATE,
    PANIC_RANGE,
)
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField
from crowd_evac.scenarios.schema import EmergencyEventData

_AGENTS = 4


def _event(tick: int, pos: list[float]) -> EmergencyEventData:
    """Build one minimal place_panic_source event."""
    return {"tick": tick, "type": "place_panic_source", "pos": pos}


@pytest.fixture
def sim() -> Simulation:
    """Return a tick-0 simulation in a 10 x 5 m room with an east exit."""
    floor = FloorPlan(
        width_m=10.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(Exit(10.0, 2.5, 2.0, ExitSide.EAST, 10, "east"),),
    )
    rng = SeededRNG(3)
    return Simulation(
        state=spawn(floor, _AGENTS, rng.generator),
        flow_field=FlowField.build(floor),
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


class TestEventsByTick:
    """events_by_tick groups a flat event list into a tick schedule."""

    def test_groups_by_tick(self) -> None:
        """Events land under their firing tick."""
        grouped = events_by_tick(
            [_event(5, [1.0, 1.0]), _event(5, [2.0, 2.0]), _event(9, [3.0, 3.0])]
        )
        assert set(grouped) == {5, 9}
        assert len(grouped[5]) == 2
        assert len(grouped[9]) == 1

    def test_empty_events_yield_empty_map(self) -> None:
        """No events produce an empty schedule."""
        assert events_by_tick([]) == {}
        assert events_by_tick(()) == {}

    def test_preserves_order_within_tick(self) -> None:
        """Events on one tick keep the order they were given."""
        grouped = events_by_tick([_event(1, [1.0, 1.0]), _event(1, [2.0, 2.0])])
        assert [e["pos"] for e in grouped[1]] == [[1.0, 1.0], [2.0, 2.0]]


class TestApplyEvent:
    """apply_event turns an event into a panic source."""

    def test_minimal_event_uses_defaults(self, sim: Simulation) -> None:
        """Only tick/type/pos given: defaults match add_panic_source."""
        apply_event(sim, _event(0, [4.0, 2.0]))
        (source,) = sim.panic_field.sources
        assert (source.x, source.y) == (4.0, 2.0)
        assert source.intensity == 1.0
        assert source.radius == PANIC_RANGE
        assert source.decay_rate == PANIC_DECAY_RATE
        assert source.block_radius == HAZARD_BLOCK_RADIUS
        assert source.source_type == "fire"
        assert source.blocks_navigation is True

    def test_overrides_are_applied(self, sim: Simulation) -> None:
        """Optional fields override the defaults."""
        event: EmergencyEventData = {
            "tick": 0,
            "type": "place_panic_source",
            "pos": [4.0, 2.0],
            "intensity": 0.5,
            "radius": 3.0,
            "decay_rate": 0.1,
            "block_radius": 0.75,
            "blocks_navigation": False,
            "source_type": "smoke",
        }
        apply_event(sim, event)
        (source,) = sim.panic_field.sources
        assert (source.intensity, source.radius) == (0.5, 3.0)
        assert source.decay_rate == 0.1
        assert source.block_radius == 0.75
        assert source.blocks_navigation is False
        assert source.source_type == "smoke"

    def test_out_of_range_value_raises(self, sim: Simulation) -> None:
        """An intensity outside [0, 1] is rejected by PanicSource."""
        event: EmergencyEventData = {
            "tick": 0, "type": "place_panic_source",
            "pos": [4.0, 2.0], "intensity": 2.0,
        }
        with pytest.raises(ValueError, match="intensity"):
            apply_event(sim, event)


class TestApplyDueEvents:
    """apply_due_events fires only what is scheduled for sim.tick."""

    def test_fires_nothing_before_its_tick(self, sim: Simulation) -> None:
        """An event for a later tick does not fire at tick 0."""
        apply_due_events(sim, events_by_tick([_event(3, [4.0, 2.0])]))
        assert len(sim.panic_field.sources) == 0

    def test_fires_at_its_tick(self, sim: Simulation) -> None:
        """An event for tick 2 fires once the sim reaches tick 2."""
        schedule = events_by_tick([_event(2, [4.0, 2.0])])
        for _ in range(2):
            apply_due_events(sim, schedule)
            sim.step()
        assert len(sim.panic_field.sources) == 0
        assert sim.tick == 2
        apply_due_events(sim, schedule)
        assert len(sim.panic_field.sources) == 1

    def test_tick_zero_fires_immediately(self, sim: Simulation) -> None:
        """A tick-0 event is due before the first step."""
        apply_due_events(sim, events_by_tick([_event(0, [4.0, 2.0])]))
        assert len(sim.panic_field.sources) == 1

    def test_empty_schedule_is_a_noop(self, sim: Simulation) -> None:
        """No schedule, no sources."""
        apply_due_events(sim, {})
        assert len(sim.panic_field.sources) == 0
