"""Tests for crowd_evac.adapters.render.arcade_input (Step 1.17).

Covers:
  - ArcadeInputSource.on_mouse_press: left-button press emits PlacePanicSourceEvent.
  - ArcadeInputSource.on_mouse_press: non-left-button press is ignored.
  - ArcadeInputSource.on_mouse_drag: left-button drag emits MovePanicSourceEvent.
  - ArcadeInputSource.on_mouse_drag: non-left-button drag is ignored.
  - ArcadeInputSource.poll: drains queue and returns buffered events.
  - ArcadeInputSource.poll: subsequent call returns empty list.
  - ArcadeInputSource.register: attaches on_mouse_press handler to window.
  - ArcadeInputSource.register: attaches on_mouse_drag handler to window.
  - ArcadeInputSource.register: press via registered handler queues event.
  - ArcadeInputSource.register: drag via registered handler queues event.
  - process_input_events: PlacePanicSourceEvent calls add_panic_source on sim.
  - process_input_events: MovePanicSourceEvent removes current source then adds new.
  - process_input_events: returns the new PanicSource after place event.
  - process_input_events: returns the new PanicSource after move event.
  - process_input_events: empty event list leaves sim unchanged.
  - process_input_events: unknown InputEvent subclass is silently skipped.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from crowd_evac.adapters.render.arcade_input import (
    MOUSE_BUTTON_LEFT,
    ArcadeInputSource,
)
from crowd_evac.application.injection import (
    add_panic_source,
    process_input_events,
)
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField
from crowd_evac.ports.input_source import (
    InputEvent,
    MovePanicSourceEvent,
    PlacePanicSourceEvent,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_PPM = 40.0
_SEED = 42
_RIGHT_BUTTON = 4
_NO_BUTTONS = 0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def src() -> ArcadeInputSource:
    """Return an ArcadeInputSource at 40 px/m."""
    return ArcadeInputSource(pixels_per_meter=_PPM)


@pytest.fixture
def floor() -> FloorPlan:
    """Return a 10×5 m open floor with one east exit."""
    return FloorPlan(
        width_m=10.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=10.0,
                y=2.5,
                width_m=2.0,
                side=ExitSide.EAST,
                capacity_per_second=10,
            ),
        ),
    )


@pytest.fixture
def sim(floor: FloorPlan) -> Simulation:
    """Return a running Simulation from the shared floor plan."""
    rng = SeededRNG(_SEED)
    return Simulation(
        state=spawn(floor, 5, rng.generator),
        flow_field=FlowField.build(floor),
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Tests: on_mouse_press
# ---------------------------------------------------------------------------


class TestOnMousePress:
    """Test suite for ArcadeInputSource.on_mouse_press."""

    # -- Happy path --------------------------------------------------------

    def test_left_press_emits_place_event(self, src: ArcadeInputSource) -> None:
        """Left-button press produces exactly one PlacePanicSourceEvent."""
        src.on_mouse_press(120.0, 80.0, MOUSE_BUTTON_LEFT, 0)
        events = src.poll()
        assert len(events) == 1
        assert isinstance(events[0], PlacePanicSourceEvent)

    def test_left_press_converts_pixels_to_metres(
        self, src: ArcadeInputSource
    ) -> None:
        """PlacePanicSourceEvent pos_m equals pixel coords divided by ppm."""
        src.on_mouse_press(120.0, 80.0, MOUSE_BUTTON_LEFT, 0)
        event = src.poll()[0]
        assert isinstance(event, PlacePanicSourceEvent)
        assert event.pos_m == pytest.approx((120.0 / _PPM, 80.0 / _PPM))

    def test_multiple_presses_accumulate(self, src: ArcadeInputSource) -> None:
        """Multiple presses before poll all appear in the queue."""
        src.on_mouse_press(40.0, 40.0, MOUSE_BUTTON_LEFT, 0)
        src.on_mouse_press(80.0, 80.0, MOUSE_BUTTON_LEFT, 0)
        assert len(src.poll()) == 2

    # -- Edge cases --------------------------------------------------------

    def test_right_press_is_ignored(self, src: ArcadeInputSource) -> None:
        """Non-left-button press produces no event."""
        src.on_mouse_press(120.0, 80.0, _RIGHT_BUTTON, 0)
        assert src.poll() == []

    def test_place_event_is_immutable(self, src: ArcadeInputSource) -> None:
        """PlacePanicSourceEvent is a frozen dataclass (immutable)."""
        src.on_mouse_press(40.0, 40.0, MOUSE_BUTTON_LEFT, 0)
        event = src.poll()[0]
        assert isinstance(event, PlacePanicSourceEvent)
        with pytest.raises((AttributeError, TypeError)):
            event.pos_m = (0.0, 0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: on_mouse_drag
# ---------------------------------------------------------------------------


class TestOnMouseDrag:
    """Test suite for ArcadeInputSource.on_mouse_drag."""

    # -- Happy path --------------------------------------------------------

    def test_left_drag_emits_move_event(self, src: ArcadeInputSource) -> None:
        """Left-button drag produces exactly one MovePanicSourceEvent."""
        src.on_mouse_drag(160.0, 100.0, 5.0, 3.0, MOUSE_BUTTON_LEFT, 0)
        events = src.poll()
        assert len(events) == 1
        assert isinstance(events[0], MovePanicSourceEvent)

    def test_left_drag_converts_pixels_to_metres(
        self, src: ArcadeInputSource
    ) -> None:
        """MovePanicSourceEvent pos_m equals pixel coords divided by ppm."""
        src.on_mouse_drag(160.0, 100.0, 0.0, 0.0, MOUSE_BUTTON_LEFT, 0)
        event = src.poll()[0]
        assert isinstance(event, MovePanicSourceEvent)
        assert event.pos_m == pytest.approx((160.0 / _PPM, 100.0 / _PPM))

    # -- Edge cases --------------------------------------------------------

    def test_right_drag_is_ignored(self, src: ArcadeInputSource) -> None:
        """Non-left-button drag produces no event."""
        src.on_mouse_drag(160.0, 100.0, 5.0, 3.0, _RIGHT_BUTTON, 0)
        assert src.poll() == []

    def test_no_buttons_drag_is_ignored(self, src: ArcadeInputSource) -> None:
        """Drag with no buttons held produces no event."""
        src.on_mouse_drag(160.0, 100.0, 5.0, 3.0, _NO_BUTTONS, 0)
        assert src.poll() == []


# ---------------------------------------------------------------------------
# Tests: poll
# ---------------------------------------------------------------------------


class TestPoll:
    """Test suite for ArcadeInputSource.poll."""

    # -- Happy path --------------------------------------------------------

    def test_poll_returns_queued_events(self, src: ArcadeInputSource) -> None:
        """poll() returns all buffered events in insertion order."""
        src.on_mouse_press(40.0, 40.0, MOUSE_BUTTON_LEFT, 0)
        src.on_mouse_drag(80.0, 80.0, 0.0, 0.0, MOUSE_BUTTON_LEFT, 0)
        events = src.poll()
        assert len(events) == 2
        assert isinstance(events[0], PlacePanicSourceEvent)
        assert isinstance(events[1], MovePanicSourceEvent)

    def test_poll_clears_queue(self, src: ArcadeInputSource) -> None:
        """Second poll after a non-empty first poll returns empty list."""
        src.on_mouse_press(40.0, 40.0, MOUSE_BUTTON_LEFT, 0)
        src.poll()
        assert src.poll() == []

    # -- Edge cases --------------------------------------------------------

    def test_poll_on_empty_queue_returns_empty_list(
        self, src: ArcadeInputSource
    ) -> None:
        """poll() on an empty queue returns [] without raising."""
        assert src.poll() == []


# ---------------------------------------------------------------------------
# Tests: register
# ---------------------------------------------------------------------------


class TestRegister:
    """Test suite for ArcadeInputSource.register."""

    # -- Happy path --------------------------------------------------------

    def test_register_attaches_press_handler(
        self, src: ArcadeInputSource
    ) -> None:
        """register sets window.on_mouse_press to on_mouse_press method."""
        window = MagicMock()
        src.register(window)
        assert window.on_mouse_press == src.on_mouse_press

    def test_register_attaches_drag_handler(
        self, src: ArcadeInputSource
    ) -> None:
        """register sets window.on_mouse_drag to on_mouse_drag method."""
        window = MagicMock()
        src.register(window)
        assert window.on_mouse_drag == src.on_mouse_drag

    def test_registered_press_handler_queues_event(
        self, src: ArcadeInputSource
    ) -> None:
        """Press dispatched via registered handler reaches the queue."""
        window = MagicMock()
        src.register(window)
        # Simulate arcade dispatching the mouse event to the registered handler
        window.on_mouse_press(80.0, 120.0, MOUSE_BUTTON_LEFT, 0)
        events = src.poll()
        assert len(events) == 1
        assert isinstance(events[0], PlacePanicSourceEvent)
        assert events[0].pos_m == pytest.approx((80.0 / _PPM, 120.0 / _PPM))

    def test_registered_drag_handler_queues_event(
        self, src: ArcadeInputSource
    ) -> None:
        """Drag dispatched via registered handler reaches the queue."""
        window = MagicMock()
        src.register(window)
        window.on_mouse_drag(200.0, 160.0, 5.0, 3.0, MOUSE_BUTTON_LEFT, 0)
        events = src.poll()
        assert len(events) == 1
        assert isinstance(events[0], MovePanicSourceEvent)


# ---------------------------------------------------------------------------
# Tests: process_input_events
# ---------------------------------------------------------------------------


class TestProcessInputEvents:
    """Test suite for process_input_events routing through the app API."""

    # -- Happy path: PlacePanicSourceEvent --------------------------------

    def test_place_event_adds_source_to_sim(
        self, sim: Simulation
    ) -> None:
        """PlacePanicSourceEvent causes a source to appear in sim.panic_field."""
        event = PlacePanicSourceEvent(pos_m=(3.0, 2.0))
        assert len(sim.panic_field.sources) == 0
        process_input_events(sim, [event])
        assert len(sim.panic_field.sources) == 1

    def test_place_event_source_position_matches_event(
        self, sim: Simulation
    ) -> None:
        """Source created by PlacePanicSourceEvent has the event's position."""
        event = PlacePanicSourceEvent(pos_m=(4.5, 1.5))
        process_input_events(sim, [event])
        src = sim.panic_field.sources[0]
        assert src.x == pytest.approx(4.5)
        assert src.y == pytest.approx(1.5)

    def test_place_event_returns_new_source(
        self, sim: Simulation
    ) -> None:
        """process_input_events returns the PanicSource created by place event."""
        event = PlacePanicSourceEvent(pos_m=(3.0, 2.0))
        result = process_input_events(sim, [event])
        assert result is not None
        assert result is sim.panic_field.sources[0]

    def test_place_event_calls_add_panic_source_via_api(
        self, sim: Simulation, mocker: MockerFixture
    ) -> None:
        """PlacePanicSourceEvent routes through add_panic_source (R7.2)."""
        mock_add = mocker.patch(
            "crowd_evac.application.injection.add_panic_source"
        )
        event = PlacePanicSourceEvent(pos_m=(3.0, 2.5))
        process_input_events(sim, [event])
        mock_add.assert_called_once()
        call_args = mock_add.call_args
        assert call_args.args[0] is sim
        assert call_args.args[1] == "fire"
        assert call_args.args[2] == pytest.approx((3.0, 2.5))

    # -- Happy path: MovePanicSourceEvent ---------------------------------

    def test_move_event_removes_old_source(
        self, sim: Simulation
    ) -> None:
        """MovePanicSourceEvent removes current_source from sim.panic_field."""
        existing = add_panic_source(
            sim, "fire", (2.0, 1.0), block_cells=False
        )
        assert len(sim.panic_field.sources) == 1
        move_event = MovePanicSourceEvent(pos_m=(5.0, 3.0))
        process_input_events(sim, [move_event], current_source=existing)
        # The original source should be gone; a new one at (5.0, 3.0) present.
        positions = [(s.x, s.y) for s in sim.panic_field.sources]
        assert (2.0, 1.0) not in positions
        assert (5.0, 3.0) in positions

    def test_move_event_adds_source_at_new_position(
        self, sim: Simulation
    ) -> None:
        """MovePanicSourceEvent adds a new source at the drag position."""
        existing = add_panic_source(
            sim, "fire", (2.0, 1.0), block_cells=False
        )
        move_event = MovePanicSourceEvent(pos_m=(6.0, 3.5))
        process_input_events(sim, [move_event], current_source=existing)
        src = sim.panic_field.sources[0]
        assert src.x == pytest.approx(6.0)
        assert src.y == pytest.approx(3.5)

    def test_move_event_returns_new_source(
        self, sim: Simulation
    ) -> None:
        """process_input_events returns the new source after a move event."""
        existing = add_panic_source(
            sim, "fire", (1.0, 1.0), block_cells=False
        )
        move_event = MovePanicSourceEvent(pos_m=(5.0, 2.5))
        result = process_input_events(
            sim, [move_event], current_source=existing
        )
        assert result is not None
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(2.5)

    def test_move_event_no_current_source_does_not_raise(
        self, sim: Simulation
    ) -> None:
        """Move with no current source skips removal without raising."""
        move_event = MovePanicSourceEvent(pos_m=(3.0, 2.0))
        result = process_input_events(sim, [move_event], current_source=None)
        assert result is not None
        assert len(sim.panic_field.sources) == 1

    # -- Edge cases -------------------------------------------------------

    def test_empty_event_list_leaves_sim_unchanged(
        self, sim: Simulation
    ) -> None:
        """Empty event list produces no side effects and returns current_source."""
        dummy = PanicSource(x=1.0, y=1.0)
        result = process_input_events(sim, [], current_source=dummy)
        assert result is dummy
        assert len(sim.panic_field.sources) == 0

    def test_unknown_event_subclass_is_silently_skipped(
        self, sim: Simulation
    ) -> None:
        """Unrecognised InputEvent subclasses do not raise or mutate sim."""

        class _UnknownEvent(InputEvent):
            """Unknown subclass for testing."""

        result = process_input_events(sim, [_UnknownEvent()])
        assert result is None
        assert len(sim.panic_field.sources) == 0
