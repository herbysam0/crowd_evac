"""Tests for crowd_evac.application.injection (FR-12.1 / R4.3).

Covers:
  - add_panic_source: source appended to sim.panic_field.
  - add_panic_source: return value is the added source object.
  - add_panic_source: source attributes match call arguments.
  - add_panic_source: agents near source have elevated panic after one tick.
  - add_panic_source: ValueError propagates for out-of-range arguments.
  - add_panic_source: 'panic_source_added' event logged with correct tick.
  - add_panic_source: event payload carries all required fields.
  - add_panic_source: cells within radius have cost=inf after injection (R4.3).
  - add_panic_source: cells outside radius retain finite cost.
  - add_panic_source: direction at probe point flips when only path is blocked
    (directional re-route, R4.3 / R12.2).
  - add_panic_source: block_cells=False leaves flow field unchanged.
  - add_panic_source: PathfindingError (all exits blocked) is caught silently;
    panic source still added.
  - add_panic_source: multiple calls accumulate independent sources.
  - remove_panic_source: source removed from panic_field.
  - remove_panic_source: removal event logged at current tick.
  - remove_panic_source: removing absent source raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.application.injection import add_panic_source, remove_panic_source
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEED = 7
_N_AGENTS = 10


@pytest.fixture
def one_exit_floor() -> FloorPlan:
    """10 m × 5 m open room with a single east exit."""
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
                label="east",
            ),
        ),
    )


@pytest.fixture
def two_exit_floor() -> FloorPlan:
    """10 m × 5 m open room with west exit and east exit."""
    return FloorPlan(
        width_m=10.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=0.0,
                y=2.5,
                width_m=2.0,
                side=ExitSide.WEST,
                capacity_per_second=10,
                label="west",
            ),
            Exit(
                x=10.0,
                y=2.5,
                width_m=2.0,
                side=ExitSide.EAST,
                capacity_per_second=10,
                label="east",
            ),
        ),
    )


def _build_sim(
    floor: FloorPlan,
    seed: int = _SEED,
    n: int = _N_AGENTS,
) -> Simulation:
    """Construct a fully-wired Simulation from a floor plan."""
    rng = SeededRNG(seed)
    flow = FlowField.build(floor)
    state = spawn(floor, n, rng.generator)
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Panic-field mutation
# ---------------------------------------------------------------------------


class TestPanicFieldMutation:
    """add_panic_source correctly mutates the simulation's panic field."""

    def test_source_appended_to_panic_field(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Source appears in sim.panic_field.sources immediately after injection."""
        sim = _build_sim(one_exit_floor)
        assert len(sim.panic_field.sources) == 0

        add_panic_source(sim, "fire", (5.0, 2.5))

        assert len(sim.panic_field.sources) == 1

    def test_return_value_is_added_source(self, one_exit_floor: FloorPlan) -> None:
        """Return value is the exact object appended to panic_field.sources."""
        sim = _build_sim(one_exit_floor)
        returned = add_panic_source(sim, "fire", (5.0, 2.5))

        assert returned is sim.panic_field.sources[0]

    def test_source_attributes_match_call_args(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Returned source carries the position, intensity, and radius passed in."""
        sim = _build_sim(one_exit_floor)
        source = add_panic_source(
            sim, "smoke", (3.0, 1.5), intensity=0.7, radius=4.0
        )

        assert source.x == pytest.approx(3.0)
        assert source.y == pytest.approx(1.5)
        assert source.intensity == pytest.approx(0.7)
        assert source.radius == pytest.approx(4.0)

    def test_agents_near_source_have_elevated_panic_after_one_tick(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """All active agents inside a room-spanning source have panic > 0 after step.

        A non-decaying source placed at the room centre with radius 15 m
        covers the entire 10 × 5 floor.  Before the first tick panic is 0;
        after one step the propagation phase raises it.
        """
        sim = _build_sim(one_exit_floor)
        add_panic_source(
            sim,
            "fire",
            (5.0, 2.5),
            intensity=1.0,
            radius=15.0,
            decay_rate=0.0,
        )
        assert np.all(sim.state.panic == 0.0)

        sim.step()

        active = sim.state.active_indices
        if active.size > 0:
            assert np.all(sim.state.panic[active] > 0.0)

    def test_invalid_intensity_raises_value_error(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Intensity > 1 raises ValueError before any field mutation."""
        sim = _build_sim(one_exit_floor)
        with pytest.raises(ValueError, match="intensity"):
            add_panic_source(sim, "fire", (5.0, 2.5), intensity=1.5)

    def test_multiple_injections_accumulate_sources(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Each call adds an independent source; total count accumulates."""
        sim = _build_sim(one_exit_floor)
        add_panic_source(sim, "fire", (2.0, 1.0), radius=0.5, block_cells=False)
        add_panic_source(sim, "fire", (4.0, 3.0), radius=0.5, block_cells=False)

        assert len(sim.panic_field.sources) == 2


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------


class TestEventLogging:
    """add_panic_source logs a tick-stamped event with the required payload."""

    def test_event_kind_is_panic_source_added(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """A 'panic_source_added' event appears in the snapshot event log."""
        sim = _build_sim(one_exit_floor)
        add_panic_source(sim, "fire", (5.0, 2.5))

        kinds = [e.kind for e in sim.snapshot().events]
        assert "panic_source_added" in kinds

    def test_event_stamped_at_current_tick(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Event tick matches the simulation tick at the moment of injection."""
        sim = _build_sim(one_exit_floor)
        for _ in range(3):
            sim.step()

        add_panic_source(sim, "fire", (5.0, 2.5))

        snap = sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "panic_source_added")
        assert evt.tick == 3

    def test_event_payload_contains_required_fields(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Payload has source_type, pos, intensity, radius, and blocked_cells."""
        sim = _build_sim(one_exit_floor)
        add_panic_source(sim, "alarm", (2.0, 1.0), intensity=0.5, radius=3.0)

        snap = sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "panic_source_added")
        assert evt.payload["source_type"] == "alarm"
        assert evt.payload["pos"] == pytest.approx([2.0, 1.0])
        assert evt.payload["intensity"] == pytest.approx(0.5)
        assert evt.payload["radius"] == pytest.approx(3.0)
        assert "blocked_cells" in evt.payload


# ---------------------------------------------------------------------------
# Flow-field re-route (R4.3 / R12.2)
# ---------------------------------------------------------------------------


class TestFlowFieldReroute:
    """add_panic_source triggers a bounded flow-field recompute."""

    def test_cells_within_radius_have_infinite_cost_after_injection(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Grid cell at the source position has cost=inf after injection.

        Before injection the cell is walkable (finite cost); after injection
        with radius 1.0 m the cell centre at (5.125, 2.625) — which lies
        within 1.0 m of (5.0, 2.5) — must be blocked.
        """
        sim = _build_sim(one_exit_floor)
        cs = GRID_CELL_SIZE
        c = int(5.0 / cs)
        r = int(2.5 / cs)

        assert np.isfinite(sim.flow_field.cost[r, c])

        add_panic_source(sim, "fire", (5.0, 2.5), radius=1.0)

        assert not np.isfinite(sim.flow_field.cost[r, c])

    def test_cells_outside_radius_retain_finite_cost(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Cells far from the source remain reachable after injection."""
        sim = _build_sim(one_exit_floor)
        add_panic_source(sim, "fire", (5.0, 2.5), radius=1.0)

        # (1.0, 2.5) is ~4 m from the source — well outside the 1 m radius.
        c = int(1.0 / GRID_CELL_SIZE)
        r = int(2.5 / GRID_CELL_SIZE)
        assert np.isfinite(sim.flow_field.cost[r, c])

    def test_injection_reroutes_agents_to_opposite_exit(
        self, two_exit_floor: FloorPlan
    ) -> None:
        """Blocking the east exit area causes agents to re-route toward the west exit.

        With west exit at x=0 and east exit at x=10, position (7.0, 2.5)
        is initially 3 m from east and 7 m from west, so the flow field
        points east (dx > 0).  After injecting a source at (9.0, 2.5) with
        radius 2.0 m — which covers all east exit cells — the only reachable
        exit is west and the direction at (7.0, 2.5) must flip to dx < 0.
        """
        sim = _build_sim(two_exit_floor)
        probe = np.array([[7.0, 2.5]])

        dir_before = sim.flow_field.sample(probe)
        assert dir_before[0, 0] > 0.0, (
            "Initial direction at (7, 2.5) should point east (dx > 0)"
        )

        add_panic_source(sim, "fire", (9.0, 2.5), radius=2.0)

        dir_after = sim.flow_field.sample(probe)
        assert dir_after[0, 0] < 0.0, (
            "After blocking east exit, direction at (7, 2.5) should point west "
            "(dx < 0)"
        )

    def test_block_cells_false_leaves_flow_field_unchanged(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """block_cells=False adds panic pressure without altering the flow field."""
        sim = _build_sim(one_exit_floor)
        cs = GRID_CELL_SIZE
        c = int(5.0 / cs)
        r = int(2.5 / cs)
        cost_before = float(sim.flow_field.cost[r, c])

        add_panic_source(sim, "fire", (5.0, 2.5), radius=1.0, block_cells=False)

        assert sim.flow_field.cost[r, c] == pytest.approx(cost_before)

    def test_pathfinding_error_does_not_propagate(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Injection that blocks all exits raises no exception.

        A source at (9.5, 2.5) with radius 1.5 m covers the single east exit
        cells on the 10 × 5 floor.  The recompute would raise PathfindingError;
        the injection API must catch it silently.  The panic source must still
        be added to the panic field.
        """
        sim = _build_sim(one_exit_floor)
        # Should not raise — PathfindingError is caught internally.
        add_panic_source(sim, "fire", (9.5, 2.5), radius=1.5)

        assert len(sim.panic_field.sources) == 1

    def test_flow_field_unchanged_when_pathfinding_error_caught(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """When blocking all exits, the original flow field is preserved."""
        sim = _build_sim(one_exit_floor)
        cs = GRID_CELL_SIZE
        # Sample cost at a cell near the room centre before injection.
        c = int(5.0 / cs)
        r = int(2.5 / cs)
        cost_before = float(sim.flow_field.cost[r, c])

        add_panic_source(sim, "fire", (9.5, 2.5), radius=1.5)

        # The interior cell should still have its original finite cost.
        assert sim.flow_field.cost[r, c] == pytest.approx(cost_before)


# ---------------------------------------------------------------------------
# remove_panic_source
# ---------------------------------------------------------------------------


class TestRemovePanicSource:
    """remove_panic_source removes a source and logs the event."""

    def test_source_absent_from_panic_field_after_removal(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Source is no longer in panic_field.sources after removal."""
        sim = _build_sim(one_exit_floor)
        source = add_panic_source(
            sim, "fire", (5.0, 2.5), block_cells=False
        )
        assert source in sim.panic_field.sources

        remove_panic_source(sim, source)

        assert source not in sim.panic_field.sources

    def test_removal_event_logged_at_current_tick(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """'panic_source_removed' event is stamped at the tick of removal."""
        sim = _build_sim(one_exit_floor)
        source = add_panic_source(
            sim, "fire", (5.0, 2.5), block_cells=False
        )
        for _ in range(2):
            sim.step()

        remove_panic_source(sim, source)

        snap = sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "panic_source_removed")
        assert evt.tick == 2

    def test_remove_absent_source_raises_value_error(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Removing a source that was never added raises ValueError."""
        sim = _build_sim(one_exit_floor)
        orphan = PanicSource(x=1.0, y=1.0)
        with pytest.raises(ValueError):
            remove_panic_source(sim, orphan)
