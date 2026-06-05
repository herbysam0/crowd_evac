"""Tests for crowd_evac.application.simulation (FR-6).

Covers:
  - Simulation.tick increments by 1 per step (R6.3).
  - sim_time equals tick x DT (R6.3).
  - snapshot() is idempotent between steps — frame-rate independence (R6.1).
  - snapshot() arrays are copies; mutating them does not alter live state.
  - Reproducibility: same seed produces same outcome (NFR-R1 best-effort).
  - SimEvent records carry their tick field (NFR-R3).
  - "tick_advanced" event is appended each step with the correct tick.
  - log_event() stamps events at the current tick.
  - step() raises RuntimeError when simulation is already complete.
  - Panic propagates from a nearby source onto agent panic levels.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import SimEvent, SimSnapshot, Simulation
from crowd_evac.domain.agent_state import AgentState, spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.constants import AGENT_PANIC_DECAY_RATE, AGENT_RADIUS, DT
from crowd_evac.domain.overlap import MIN_AGENT_SEPARATION
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Obstacle
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_N_AGENTS = 8


@pytest.fixture
def sim_floor() -> FloorPlan:
    """10 m x 5 m open room with a single exit on the east wall."""
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
def sim_flow(sim_floor: FloorPlan) -> FlowField:
    """Flow field built from the standard test floor."""
    return FlowField.build(sim_floor)


def _build_sim(
    floor: FloorPlan,
    flow: FlowField,
    seed: int = _SEED,
    n_agents: int = _N_AGENTS,
    panic_field: PanicField | None = None,
    collision_map: CollisionMap | None = None,
) -> Simulation:
    """Construct a fully-wired Simulation from floor and flow fixtures."""
    rng = SeededRNG(seed)
    state = spawn(floor, n_agents, rng.generator)
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=panic_field if panic_field is not None else PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
        collision_map=collision_map,
    )


@pytest.fixture
def sim(sim_floor: FloorPlan, sim_flow: FlowField) -> Simulation:
    """Standard simulation with 8 agents, seed 42, no panic sources."""
    return _build_sim(sim_floor, sim_flow)


# ---------------------------------------------------------------------------
# Tick counter and sim_time (R6.3)
# ---------------------------------------------------------------------------


class TestTickCounter:
    """tick increments monotonically by 1 per step; sim_time tracks it."""

    def test_initial_tick_is_zero(self, sim: Simulation) -> None:
        """tick starts at 0 before any steps."""
        assert sim.tick == 0

    def test_tick_increments_by_one_per_step(self, sim: Simulation) -> None:
        """Each step() call increments tick by exactly 1."""
        for expected in range(1, 5):
            sim.step()
            assert sim.tick == expected

    def test_sim_time_zero_before_stepping(self, sim: Simulation) -> None:
        """sim_time is 0.0 before any steps."""
        assert sim.sim_time == 0.0

    def test_sim_time_equals_tick_times_dt(self, sim: Simulation) -> None:
        """sim_time is derived from the tick counter, not accumulated additions.

        Because sim_time is computed as float(tick) * DT on every read rather
        than being summed one DT at a time, it matches the reference
        n_steps * DT value to within floating-point rounding of a single
        multiplication.  pytest.approx with the default relative tolerance
        of 1e-6 is more than sufficient for any reasonable step count.
        """
        n_steps = 6
        for _ in range(n_steps):
            sim.step()
        # Both sides evaluate the same single multiplication; rounding is
        # identical, so pytest.approx is used only as a safety margin.
        assert sim.sim_time == pytest.approx(n_steps * DT)


# ---------------------------------------------------------------------------
# Snapshot idempotency — frame-rate independence (R6.1)
# ---------------------------------------------------------------------------


class TestSnapshotIdempotency:
    """snapshot() is a pure read; calling it repeatedly does not alter state."""

    def test_snapshot_returns_sim_snapshot(self, sim: Simulation) -> None:
        """snapshot() returns a SimSnapshot instance."""
        snap = sim.snapshot()
        assert isinstance(snap, SimSnapshot)

    def test_multiple_snapshots_between_steps_are_equal(
        self, sim: Simulation
    ) -> None:
        """Three consecutive snapshots between steps carry identical data."""
        sim.step()
        s1 = sim.snapshot()
        s2 = sim.snapshot()
        s3 = sim.snapshot()

        assert s1.tick == s2.tick == s3.tick
        np.testing.assert_array_equal(s1.positions, s2.positions)
        np.testing.assert_array_equal(s2.positions, s3.positions)
        np.testing.assert_array_equal(s1.alive, s3.alive)

    def test_reading_snapshot_does_not_change_step_outcome(
        self,
        sim_floor: FloorPlan,
        sim_flow: FlowField,
    ) -> None:
        """A sim that reads snapshots produces the same state as one that does not.

        Both sims use the same seed so they start identically.  After the
        same number of steps the positions must be bit-for-bit equal.
        """
        sim_read = _build_sim(sim_floor, sim_flow, seed=_SEED)
        sim_plain = _build_sim(sim_floor, sim_flow, seed=_SEED)

        for _ in range(5):
            # Read snapshot many times before stepping
            sim_read.snapshot()
            sim_read.snapshot()
            sim_read.step()
            sim_plain.step()

        np.testing.assert_array_equal(sim_read.state.pos, sim_plain.state.pos)
        np.testing.assert_array_equal(sim_read.state.vel, sim_plain.state.vel)

    def test_snapshot_arrays_are_copies(self, sim: Simulation) -> None:
        """Mutating a snapshot array does not alter live simulation state."""
        snap = sim.snapshot()
        original_pos = snap.positions.copy()

        # Overwrite the snapshot array entirely
        snap.positions[:] = 999.0

        # Live state must be untouched
        np.testing.assert_array_equal(sim.state.pos, original_pos)


# ---------------------------------------------------------------------------
# Snapshot field correctness
# ---------------------------------------------------------------------------


class TestSnapshotFields:
    """snapshot() exposes the correct values at the current tick."""

    def test_snapshot_tick_matches_sim_tick(self, sim: Simulation) -> None:
        """snapshot().tick reflects the current simulation tick."""
        sim.step()
        sim.step()
        snap = sim.snapshot()
        assert snap.tick == sim.tick

    def test_snapshot_sim_time_matches(self, sim: Simulation) -> None:
        """snapshot().sim_time equals sim.sim_time at snapshot time."""
        for _ in range(3):
            sim.step()
        snap = sim.snapshot()
        assert snap.sim_time == pytest.approx(sim.sim_time)

    def test_snapshot_active_count_reflects_alive(self, sim: Simulation) -> None:
        """active_count matches the number of True entries in the alive array."""
        sim.step()
        snap = sim.snapshot()
        expected = int(np.sum(snap.alive))
        assert snap.active_count == expected

    def test_snapshot_evacuated_count_matches_exit_model(
        self, sim: Simulation
    ) -> None:
        """evacuated_count in snapshot matches the exit model's tally."""
        for _ in range(10):
            sim.step()
        snap = sim.snapshot()
        assert snap.evacuated_count == sim.exit_model.evacuated_count


# ---------------------------------------------------------------------------
# Reproducibility (NFR-R1 best-effort)
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same seed and same event sequence produce the same outcome."""

    def test_same_seed_same_positions_after_n_steps(
        self,
        sim_floor: FloorPlan,
        sim_flow: FlowField,
    ) -> None:
        """Two sims with the same seed produce identical positions after N steps."""
        sim_a = _build_sim(sim_floor, sim_flow, seed=99)
        sim_b = _build_sim(sim_floor, sim_flow, seed=99)

        for _ in range(20):
            sim_a.step()
            sim_b.step()

        np.testing.assert_array_equal(sim_a.state.pos, sim_b.state.pos)
        np.testing.assert_array_equal(sim_a.state.vel, sim_b.state.vel)
        np.testing.assert_array_equal(sim_a.state.alive, sim_b.state.alive)

    def test_different_seeds_differ(
        self,
        sim_floor: FloorPlan,
        sim_flow: FlowField,
    ) -> None:
        """Different seeds produce distinct initial layouts at tick 0."""
        sim_a = _build_sim(sim_floor, sim_flow, seed=1)
        sim_b = _build_sim(sim_floor, sim_flow, seed=2)
        # With 8 agents in a 10x5 room, identical positions are astronomically
        # unlikely; this distinguishes the two seeds at spawn time.
        assert not np.array_equal(sim_a.state.pos, sim_b.state.pos)


# ---------------------------------------------------------------------------
# Event log (R6.3 / NFR-R3)
# ---------------------------------------------------------------------------


class TestEventLog:
    """Every event in the log carries its tick; tick_advanced is logged each step."""

    def test_all_events_have_tick_attribute(self, sim: Simulation) -> None:
        """Every SimEvent in the snapshot log has an integer tick field."""
        for _ in range(3):
            sim.step()
        snap = sim.snapshot()
        assert len(snap.events) > 0
        for evt in snap.events:
            assert isinstance(evt, SimEvent)
            assert isinstance(evt.tick, int)

    def test_tick_advanced_event_logged_each_step(self, sim: Simulation) -> None:
        """Each step appends exactly one 'tick_advanced' event at the new tick."""
        n_steps = 4
        for _ in range(n_steps):
            sim.step()

        snap = sim.snapshot()
        tick_events = [e for e in snap.events if e.kind == "tick_advanced"]
        assert len(tick_events) == n_steps

        # Events are stamped at ticks 1, 2, 3, 4 (post-increment in step()).
        for i, evt in enumerate(tick_events):
            assert evt.tick == i + 1

    def test_tick_advanced_payload_has_egressed_key(self, sim: Simulation) -> None:
        """tick_advanced payload includes an integer 'egressed' key."""
        sim.step()
        snap = sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "tick_advanced")
        assert "egressed" in evt.payload
        assert isinstance(evt.payload["egressed"], int)

    def test_log_event_stamps_current_tick(self, sim: Simulation) -> None:
        """log_event() stamps the event at the current tick value."""
        # Before any steps: tick == 0
        sim.log_event("before_first_step", value=1)
        sim.step()
        # After one step: tick == 1
        sim.log_event("after_first_step", value=2)

        snap = sim.snapshot()
        before = next(e for e in snap.events if e.kind == "before_first_step")
        after = next(e for e in snap.events if e.kind == "after_first_step")
        assert before.tick == 0
        assert after.tick == 1

    def test_log_event_payload_round_trips(self, sim: Simulation) -> None:
        """Payload kwargs are stored and retrievable from the event log."""
        sim.log_event("test_event", x=3.14, label="fire", active=True)
        snap = sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "test_event")
        assert evt.payload["x"] == pytest.approx(3.14)
        assert evt.payload["label"] == "fire"
        assert evt.payload["active"] is True


# ---------------------------------------------------------------------------
# Completion guard
# ---------------------------------------------------------------------------


class TestCompletionGuard:
    """step() raises RuntimeError when the simulation is already complete."""

    def test_step_raises_on_zero_agent_sim(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """A simulation with 0 agents is immediately complete; step() raises."""
        rng = SeededRNG(_SEED)
        empty_state = AgentState(
            pos=np.empty((0, 2), dtype=np.float64),
            vel=np.empty((0, 2), dtype=np.float64),
            panic=np.empty(0, dtype=np.float64),
            goal=np.empty(0, dtype=np.intp),
            alive=np.empty(0, dtype=np.bool_),
        )
        sim_empty = Simulation(
            state=empty_state,
            flow_field=sim_flow,
            panic_field=PanicField(),
            exit_model=ExitModel(sim_floor),
            rng=rng,
        )
        assert sim_empty.is_complete
        with pytest.raises(RuntimeError, match="completed"):
            sim_empty.step()

    def test_is_complete_false_while_agents_remain(self, sim: Simulation) -> None:
        """is_complete is False for a freshly-built sim with live agents."""
        assert not sim.is_complete


# ---------------------------------------------------------------------------
# Panic propagation
# ---------------------------------------------------------------------------


class TestPanicPropagation:
    """Agents near an active panic source have their panic raised each tick."""

    def test_panic_raised_near_source(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """After one step, agents within a room-spanning source have panic > 0.

        A full-intensity non-decaying source placed at the room centre with
        a radius large enough to cover the entire floor.  All active agents
        must have panic > 0 after the first tick.
        """
        source = PanicSource(
            x=5.0,
            y=2.5,
            intensity=1.0,
            radius=15.0,  # covers 10x5 room with margin
            decay_rate=0.0,
        )
        pf = PanicField([source])
        sim_p = _build_sim(sim_floor, sim_flow, seed=_SEED, panic_field=pf)

        assert np.all(sim_p.state.panic == 0.0)

        sim_p.step()

        active = sim_p.state.active_indices
        if active.size > 0:
            assert np.all(sim_p.state.panic[active] > 0.0), (
                "All active agents should have panic > 0 after one tick "
                "inside a room-spanning panic source."
            )

    def test_no_panic_without_sources(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """Without panic sources agent panic levels remain zero throughout."""
        sim_clean = _build_sim(sim_floor, sim_flow, seed=_SEED)
        for _ in range(5):
            sim_clean.step()
        active = sim_clean.state.active_indices
        if active.size > 0:
            assert np.all(sim_clean.state.panic[active] == 0.0)

    def test_panic_decays_after_source_expires(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """Panic falls by AGENT_PANIC_DECAY_RATE*DT per tick once the source is gone.

        One step with a room-spanning source raises every agent's panic above zero.
        The source is then cleared and a single additional step is taken.
        For agents still active, panic must be strictly lower than it was after
        the exposure tick, proving the decay formula runs (not the old raise-only
        np.maximum).
        """
        source = PanicSource(
            x=5.0,
            y=2.5,
            intensity=1.0,
            radius=15.0,  # covers the whole 10x5 room
            decay_rate=0.0,  # pinned at full intensity for the exposure tick
        )
        pf = PanicField([source])
        sim_p = _build_sim(sim_floor, sim_flow, seed=_SEED, panic_field=pf)

        # Step 1: expose all agents — panic rises to field value at each position
        sim_p.step()
        active = sim_p.state.active_indices
        assert active.size > 0, "fixture must have active agents after tick 1"
        panic_after_exposure = sim_p.state.panic.copy()  # full array for indexing
        assert np.all(panic_after_exposure[active] > 0.0)

        # Remove source so field is zero everywhere
        pf.sources.clear()

        # Step 2: one decay tick — each agent's panic must strictly decrease
        sim_p.step()
        active2 = sim_p.state.active_indices
        assert active2.size > 0, "agents should still be active after only 2 ticks"
        expected_upper = panic_after_exposure[active2] - AGENT_PANIC_DECAY_RATE * DT
        assert np.all(sim_p.state.panic[active2] <= expected_upper), (
            "Panic must have decayed by AGENT_PANIC_DECAY_RATE*DT after one "
            f"source-free tick; delta was too small: "
            f"before={panic_after_exposure[active2]}, "
            f"after={sim_p.state.panic[active2]}."
        )


# ---------------------------------------------------------------------------
# Collision constraint (step 1.19a item 1) — obstacles are never crossed
# ---------------------------------------------------------------------------


class TestCollisionConstraint:
    """A wired collision map keeps every agent out of wall/obstacle cells."""

    @pytest.fixture
    def obstacle_floor(self) -> FloorPlan:
        """10 m x 6 m room split by a near-full-height obstacle wall.

        The obstacle spans x in [4.5, 5.5] and y in [0, 4.5], leaving a 1.5 m
        gap at the top, so west-side agents must route around it to reach the
        east exit rather than walking straight through.
        """
        return FloorPlan(
            width_m=10.0,
            height_m=6.0,
            walls=(),
            obstacles=(Obstacle(x=4.5, y=0.0, width=1.0, height=4.5),),
            exits=(
                Exit(
                    x=10.0,
                    y=3.0,
                    width_m=2.0,
                    side=ExitSide.EAST,
                    capacity_per_second=10,
                    label="east",
                ),
            ),
        )

    def test_agents_never_occupy_blocked_cell(
        self, obstacle_floor: FloorPlan
    ) -> None:
        """Across the full run no active agent ever lands in a blocked cell.

        Steps the simulation until evacuation completes (or a tick ceiling),
        checking the invariant before the first step and after every step.
        """
        flow = FlowField.build(obstacle_floor)
        cmap = CollisionMap.from_floor_plan(obstacle_floor)
        sim = _build_sim(
            obstacle_floor, flow, n_agents=40, collision_map=cmap
        )

        def _assert_all_clear() -> None:
            snap = sim.snapshot()
            alive_pos = snap.positions[snap.alive]
            if alive_pos.shape[0] > 0:
                assert not np.any(cmap.is_blocked(alive_pos))

        _assert_all_clear()
        for _ in range(400):
            if sim.is_complete:
                break
            sim.step()
            _assert_all_clear()

    def test_without_map_agents_can_enter_obstacle(
        self, obstacle_floor: FloorPlan
    ) -> None:
        """Control: with no collision map the constraint is absent.

        Confirms the obstacle floor and exit pull agents *through* the
        obstacle region when collision resolution is disabled, so the
        passing :meth:`test_agents_never_occupy_blocked_cell` is attributable
        to the collision map and not to the geometry trivially avoiding it.
        """
        flow = FlowField.build(obstacle_floor)
        cmap = CollisionMap.from_floor_plan(obstacle_floor)
        sim = _build_sim(obstacle_floor, flow, n_agents=40, seed=1)

        entered_block = False
        for _ in range(400):
            if sim.is_complete:
                break
            sim.step()
            snap = sim.snapshot()
            alive_pos = snap.positions[snap.alive]
            if alive_pos.shape[0] > 0 and np.any(cmap.is_blocked(alive_pos)):
                entered_block = True
                break
        assert entered_block


# ---------------------------------------------------------------------------
# No-overlap invariant (step 1.19a item 12) — agents never overlap
# ---------------------------------------------------------------------------


class TestNoOverlapInvariant:
    """The hard no-overlap projection keeps agents and walls non-overlapping."""

    def _min_pairwise_alive(self, sim: Simulation) -> float:
        """Smallest centre distance among the currently-alive agents."""
        pos = sim.state.pos[sim.state.alive]
        if pos.shape[0] < 2:
            return np.inf
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        return float(dist.min())

    def _build_overlap_sim(
        self,
        floor: FloorPlan,
        flow: FlowField,
        *,
        enable_no_overlap: bool,
    ) -> Simulation:
        """Build a 24-agent sim (seed 7) with the projection on or off."""
        rng = SeededRNG(7)
        return Simulation(
            state=spawn(floor, 24, rng.generator),
            flow_field=flow,
            panic_field=PanicField(),
            exit_model=ExitModel(floor),
            rng=rng,
            enable_no_overlap=enable_no_overlap,
        )

    def test_projection_enforces_separation_unlike_force_alone(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """The projection holds agents apart where the force alone lets them pack.

        Two seed-matched 24-agent runs step in lock-step toward a shared exit.
        After a short warm-up that relaxes any deep spawn overlap, the
        projection run keeps its closest pair near one diameter (documented
        0.85x tolerance for finite Jacobi passes at the jam), while the
        force-only control lets agents pack visibly closer.
        """
        sim_on = self._build_overlap_sim(
            sim_floor, sim_flow, enable_no_overlap=True
        )
        sim_off = self._build_overlap_sim(
            sim_floor, sim_flow, enable_no_overlap=False
        )

        warmup, measured = 5, 60
        min_on, min_off = np.inf, np.inf
        for tick in range(warmup + measured):
            if sim_on.is_complete or sim_off.is_complete:
                break
            sim_on.step()
            sim_off.step()
            if tick >= warmup:
                min_on = min(min_on, self._min_pairwise_alive(sim_on))
                min_off = min(min_off, self._min_pairwise_alive(sim_off))

        assert min_on >= 0.85 * MIN_AGENT_SEPARATION
        assert min_off < min_on

    def test_wall_bodies_do_not_overlap_in_full_run(
        self, sim_floor: FloorPlan, sim_flow: FlowField
    ) -> None:
        """No alive agent centre stays nearer than the radius to a wall.

        The room's bounding box is solid geometry; with the projection enabled
        every alive centre stays at least one radius inside it for the whole
        run (allowing a small finite-iteration tolerance).
        """
        cmap = CollisionMap.from_floor_plan(sim_floor)
        sim = _build_sim(
            sim_floor, sim_flow, n_agents=24, collision_map=cmap
        )
        # With no interior walls the grid's bounding box is the only geometry,
        # so every alive centre must stay one radius inside all four faces of
        # the 10 x 5 room (small finite-iteration tolerance).
        margin = AGENT_RADIUS - 0.05
        for _ in range(120):
            if sim.is_complete:
                break
            sim.step()
            pos = sim.state.pos[sim.state.alive]
            if pos.shape[0] == 0:
                continue
            assert np.all(pos[:, 0] >= margin)
            assert np.all(pos[:, 0] <= 10.0 - margin)
            assert np.all(pos[:, 1] >= margin)
            assert np.all(pos[:, 1] <= 5.0 - margin)
