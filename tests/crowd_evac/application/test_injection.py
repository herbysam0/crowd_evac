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
from crowd_evac.domain.params import ForceParams
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


# ---------------------------------------------------------------------------
# Hazard-block restoration (regression: blocks were permanent — agents froze)
# ---------------------------------------------------------------------------


class TestHazardBlockRestoration:
    """A hazard's flow-field block is restored when it decays or is removed.

    Directly guards the diagnosed Phase-1 defect: injected blocks were never
    cleared, so an agent stranded in the zero-direction region never
    recalculated a route. With the fix, the flow field re-solves from the
    pristine mask once the hazard clears.
    """

    def _blocked_cell(self) -> tuple[int, int]:
        """Return the (row, col) cell covering the (5.0, 2.5) injection point."""
        cs = GRID_CELL_SIZE
        return int(2.5 / cs), int(5.0 / cs)

    def test_decayed_source_restores_blocked_cell(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Once a source decays to inactive, its blocked cell becomes finite.

        After restoration the flow field also yields a non-zero exit direction
        at that cell — the agent can seek the exit again (recalculation).
        """
        sim = _build_sim(one_exit_floor)
        r, c = self._blocked_cell()
        original = float(sim.flow_field.cost[r, c])

        source = add_panic_source(sim, "fire", (5.0, 2.5), decay_rate=5.0)
        assert not np.isfinite(sim.flow_field.cost[r, c])  # blocked on inject

        # Drain the source to inactivity, then refresh as a tick would.
        source.intensity = 0.0
        sim.refresh_hazard_blocks()

        assert sim.flow_field.cost[r, c] == pytest.approx(original)
        probe = np.array([[5.0, 2.5]])
        assert np.linalg.norm(sim.flow_field.sample(probe)[0]) > 0.0

    def test_removed_source_restores_blocked_cell(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Removing a blocking source restores its footprint immediately."""
        sim = _build_sim(one_exit_floor)
        r, c = self._blocked_cell()
        original = float(sim.flow_field.cost[r, c])

        source = add_panic_source(sim, "fire", (5.0, 2.5))
        assert not np.isfinite(sim.flow_field.cost[r, c])

        remove_panic_source(sim, source)

        assert sim.flow_field.cost[r, c] == pytest.approx(original)

    def test_block_radius_decoupled_from_panic_radius(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """A large panic radius with a small block_radius blocks only the core.

        A cell ~3 m from the source is within the 8 m panic radius but outside
        the 1 m block footprint, so it must remain reachable — the fix that
        stops the crowd being engulfed.
        """
        sim = _build_sim(one_exit_floor)
        cs = GRID_CELL_SIZE
        add_panic_source(
            sim, "fire", (5.0, 2.5), radius=8.0, block_radius=1.0
        )
        # Core cell blocked.
        assert not np.isfinite(sim.flow_field.cost[int(2.5 / cs), int(5.0 / cs)])
        # Cell ~3 m away (inside panic radius, outside block radius) reachable.
        assert np.isfinite(sim.flow_field.cost[int(2.5 / cs), int(2.0 / cs)])

    def test_refresh_is_noop_without_sources(
        self, one_exit_floor: FloorPlan
    ) -> None:
        """Stepping a source-less sim leaves the flow-field object identity.

        Guards determinism: the per-tick refresh must not rebuild the field
        when the hazard set is empty.
        """
        sim = _build_sim(one_exit_floor)
        before = sim.flow_field
        sim.step()
        assert sim.flow_field is before


# ---------------------------------------------------------------------------
# Hazard avoidance routing (fire near one exit -> crowd diverts to the other)
# ---------------------------------------------------------------------------


class TestHazardAvoidanceRouting:
    """The danger-cost field diverts agents to the next-best exit."""

    def test_fire_near_exit_reroutes_to_other_exit(
        self, two_exit_floor: FloorPlan
    ) -> None:
        """A fire by the east exit flips a mid-room east-bound cell westward.

        With the default (overpowering) avoidance weight, the route through the
        hazard becomes far more expensive than the westward detour, so agents
        head for the unobstructed exit — the real-world response.
        """
        sim = _build_sim(two_exit_floor)
        probe = np.array([[6.0, 2.5]])
        assert sim.flow_field.sample(probe)[0, 0] > 0.0  # east by default

        # Tiny physical core (block_radius 0.1 m) so the divert is driven by
        # the danger cost over the 4 m panic radius, not the hard block.
        add_panic_source(
            sim, "fire", (7.5, 2.5), radius=4.0, block_radius=0.1
        )

        assert sim.flow_field.sample(probe)[0, 0] < 0.0  # diverted west

    def test_zero_avoidance_weight_keeps_nearest_exit(
        self, two_exit_floor: FloorPlan
    ) -> None:
        """With avoidance disabled the crowd still takes the nearest exit.

        Confirms the rerouting is driven by the danger cost, not the small
        physical core block: a fire whose core does not reach the exit leaves
        the east-bound route intact when hazard_avoidance_cost is 0.
        """
        rng = SeededRNG(_SEED)
        flow = FlowField.build(two_exit_floor)
        state = spawn(two_exit_floor, _N_AGENTS, rng.generator)
        sim = Simulation(
            state=state,
            flow_field=flow,
            panic_field=PanicField(),
            exit_model=ExitModel(two_exit_floor),
            rng=rng,
            params=ForceParams(hazard_avoidance_cost=0.0),
        )
        probe = np.array([[6.0, 2.5]])
        assert sim.flow_field.sample(probe)[0, 0] > 0.0

        add_panic_source(
            sim, "fire", (7.5, 2.5), radius=4.0, block_radius=0.1
        )

        # A 0.1 m core blocks a single cell — not enough to wall the corridor;
        # with no danger cost the eastward route to the nearer exit survives.
        assert sim.flow_field.sample(probe)[0, 0] > 0.0
