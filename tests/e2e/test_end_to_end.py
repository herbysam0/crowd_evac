"""End-to-end headless tests for the full evacuation pipeline (Step 1.20 / FR-5 R5.3 / FR-6 R6.1 / NFR-R1).

Covers:
  - Default (lecture_hall) scenario builds and runs headless to evacuation-complete.
  - evacuated_count == initial_count after a fully-solvable run (R5.3).
  - active_count reaches zero confirming full evacuation, not stall-guard (R5.3).
  - Same seed + same events → identical tick-by-tick outcome (NFR-R1).
  - Different seeds → distinct initial layouts.
  - Halving snapshot cadence does not alter simulation outcome (R6.1).
  - Multiple snapshots between steps are bit-for-bit identical (R6.1).
  - Panic-source injection mid-run is logged and visible in the snapshot (FR-12.1).

All tests run headless — no arcade window is opened.  The ``e2e`` mark is
registered in ``pyproject.toml`` and excluded from the default fast run.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.application.cli import build_simulation_from_scenario
from crowd_evac.application.injection import add_panic_source
from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.constants import PANIC_RANGE
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FULL_EVAC_CEILING: int = 12_000
"""Maximum ticks for the lecture_hall scenario to fully evacuate all agents."""

_SMALL_EVAC_CEILING: int = 3_000
"""Maximum ticks for the small synthetic room to fully evacuate all agents."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _run_to_completion(sim: Simulation, ceiling: int) -> None:
    """Step sim until is_complete or ceiling ticks (in-place)."""
    for _ in range(ceiling):
        if sim.is_complete:
            break
        sim.step()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def open_room_floor() -> FloorPlan:
    """10 m × 5 m open room with a single wide east exit (capacity 40 agents/s).

    No walls or obstacles; every agent reaches the exit within a small
    number of ticks at the default speed cap, making this an ideal
    fully-solvable synthetic scenario.
    """
    return FloorPlan(
        width_m=10.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=10.0,
                y=2.5,
                width_m=4.0,
                side=ExitSide.EAST,
                capacity_per_second=40,
                label="east",
            ),
        ),
    )


@pytest.fixture
def open_room_flow(open_room_floor: FloorPlan) -> FlowField:
    """Flow field for the open-room floor plan."""
    return FlowField.build(open_room_floor)


@pytest.fixture
def small_sim(open_room_floor: FloorPlan, open_room_flow: FlowField) -> Simulation:
    """A small (20 agents, seed=42) fully-solvable simulation on the open room."""
    rng = SeededRNG(42)
    state = spawn(open_room_floor, 20, rng.generator)
    return Simulation(
        state=state,
        flow_field=open_room_flow,
        panic_field=PanicField(),
        exit_model=ExitModel(open_room_floor),
        rng=rng,
    )


def _make_open_sim(
    floor: FloorPlan,
    flow: FlowField,
    seed: int,
    n_agents: int,
) -> Simulation:
    """Build a fresh open-room simulation from the given fixtures and parameters."""
    rng = SeededRNG(seed)
    state = spawn(floor, n_agents, rng.generator)
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Default scenario (lecture_hall) E2E
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDefaultScenarioE2E:
    """End-to-end tests using the bundled lecture_hall default scenario."""

    def test_default_scenario_builds_headless(self) -> None:
        """build_simulation_from_scenario() produces a sim at tick 0 with agents."""
        sim, floor_plan = build_simulation_from_scenario()
        assert sim.tick == 0
        assert sim.snapshot().active_count > 0
        assert floor_plan.width_m > 0.0
        assert floor_plan.height_m > 0.0

    def test_default_scenario_runs_to_completion(self) -> None:
        """The lecture_hall scenario reaches is_complete within the tick ceiling."""
        sim, _ = build_simulation_from_scenario()
        initial_count = sim.state.count
        _run_to_completion(sim, _FULL_EVAC_CEILING)
        assert sim.is_complete, (
            f"Simulation did not complete within {_FULL_EVAC_CEILING} ticks; "
            f"tick={sim.tick}, active={sim.state.active_indices.size}, "
            f"evacuated={sim.exit_model.evacuated_count}/{initial_count}"
        )

    def test_default_scenario_full_evacuation_r5_3(self) -> None:
        """All agents evacuate cleanly (R5.3): active_count == 0, evacuated == initial.

        Distinguishes true full evacuation from a stall-guard completion:
        active_indices.size == 0 is only reachable when every agent has
        egressed — the stall-guard fires while agents are still alive.
        """
        sim, _ = build_simulation_from_scenario()
        initial_count = sim.state.count
        _run_to_completion(sim, _FULL_EVAC_CEILING)
        assert sim.state.active_indices.size == 0, (
            f"{sim.state.active_indices.size} agents still active after "
            "completion; stall-guard may have fired before full evacuation."
        )
        assert sim.exit_model.evacuated_count == initial_count, (
            f"evacuated={sim.exit_model.evacuated_count} != "
            f"initial={initial_count}"
        )

    def test_default_scenario_headless_step_advances_tick(self) -> None:
        """One headless step on the default scenario increments tick from 0 to 1."""
        sim, _ = build_simulation_from_scenario()
        sim.step()
        assert sim.tick == 1


# ---------------------------------------------------------------------------
# Seed reproducibility (NFR-R1)
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Same seed + same event sequence → identical simulation outcome."""

    def test_same_seed_two_complete_runs_match(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """Two runs from the same seed step in lock-step to identical positions.

        Both simulations start with seed=42 and 20 agents.  At every tick the
        positions and alive flags must be bit-for-bit equal.
        """
        sim_a = _make_open_sim(open_room_floor, open_room_flow, seed=42, n_agents=20)
        sim_b = _make_open_sim(open_room_floor, open_room_flow, seed=42, n_agents=20)

        for _ in range(_SMALL_EVAC_CEILING):
            done_a = sim_a.is_complete
            done_b = sim_b.is_complete
            if done_a and done_b:
                break
            assert done_a == done_b, (
                f"Completion diverged at tick {sim_a.tick}: "
                f"A complete={done_a}, B complete={done_b}"
            )
            sim_a.step()
            sim_b.step()
            np.testing.assert_array_equal(
                sim_a.state.pos,
                sim_b.state.pos,
                err_msg=f"positions diverged at tick {sim_a.tick}",
            )
            np.testing.assert_array_equal(
                sim_a.state.alive,
                sim_b.state.alive,
                err_msg=f"alive flags diverged at tick {sim_a.tick}",
            )

        assert sim_a.tick == sim_b.tick, "runs completed at different ticks"
        assert sim_a.exit_model.evacuated_count == sim_b.exit_model.evacuated_count

    def test_different_seeds_produce_distinct_initial_positions(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """Two runs with different seeds start with different agent layouts."""
        sim_a = _make_open_sim(open_room_floor, open_room_flow, seed=1, n_agents=20)
        sim_b = _make_open_sim(open_room_floor, open_room_flow, seed=2, n_agents=20)
        assert not np.array_equal(sim_a.state.pos, sim_b.state.pos), (
            "Different seeds produced identical starting positions — "
            "RNG seeding is not working correctly."
        )

    def test_fixture_minimal_same_seed_reproduces(self) -> None:
        """fixture_minimal loaded twice terminates at the same tick (NFR-R1)."""
        sim_a, _ = build_simulation_from_scenario("fixture_minimal")
        sim_b, _ = build_simulation_from_scenario("fixture_minimal")

        _run_to_completion(sim_a, _SMALL_EVAC_CEILING)
        _run_to_completion(sim_b, _SMALL_EVAC_CEILING)

        assert sim_a.tick == sim_b.tick, (
            f"fixture_minimal runs terminated at different ticks: "
            f"A={sim_a.tick}, B={sim_b.tick}"
        )
        assert sim_a.exit_model.evacuated_count == sim_b.exit_model.evacuated_count


# ---------------------------------------------------------------------------
# Snapshot cadence independence (R6.1)
# ---------------------------------------------------------------------------


class TestSnapshotCadenceIndependence:
    """Calling snapshot() at any cadence does not alter simulation outcome."""

    def test_frequent_snapshots_do_not_change_outcome(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """A sim that snapshots every tick produces the same result as one that never does.

        sim_heavy takes ten snapshots before each step; sim_plain takes none.
        Positions and alive flags must be bit-for-bit equal after every step.
        """
        sim_plain = _make_open_sim(open_room_floor, open_room_flow, seed=7, n_agents=15)
        sim_heavy = _make_open_sim(open_room_floor, open_room_flow, seed=7, n_agents=15)

        for _ in range(_SMALL_EVAC_CEILING):
            if sim_plain.is_complete and sim_heavy.is_complete:
                break
            if sim_plain.is_complete or sim_heavy.is_complete:
                break
            sim_plain.step()
            for _ in range(10):
                sim_heavy.snapshot()
            sim_heavy.step()
            np.testing.assert_array_equal(
                sim_plain.state.pos,
                sim_heavy.state.pos,
                err_msg=(
                    f"Positions diverged at tick {sim_plain.tick} — "
                    "snapshot() is mutating simulation state."
                ),
            )
            np.testing.assert_array_equal(
                sim_plain.state.alive,
                sim_heavy.state.alive,
            )

        assert sim_plain.exit_model.evacuated_count == sim_heavy.exit_model.evacuated_count

    def test_multiple_snapshots_between_steps_are_identical(
        self, small_sim: Simulation
    ) -> None:
        """Three consecutive snapshots between steps carry identical data (R6.1)."""
        small_sim.step()
        snap1 = small_sim.snapshot()
        snap2 = small_sim.snapshot()
        snap3 = small_sim.snapshot()

        assert snap1.tick == snap2.tick == snap3.tick
        np.testing.assert_array_equal(snap1.positions, snap2.positions)
        np.testing.assert_array_equal(snap2.positions, snap3.positions)
        np.testing.assert_array_equal(snap1.alive, snap3.alive)
        assert snap1.evacuated_count == snap3.evacuated_count

    def test_snapshot_counts_match_at_completion(
        self, small_sim: Simulation
    ) -> None:
        """snapshot().evacuated_count matches exit_model.evacuated_count at completion."""
        _run_to_completion(small_sim, _SMALL_EVAC_CEILING)
        snap = small_sim.snapshot()
        assert snap.evacuated_count == small_sim.exit_model.evacuated_count
        assert snap.active_count == small_sim.state.active_indices.size

    def test_halved_snapshot_cadence_same_final_state(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """Halving render cadence leaves final state unchanged (R6.1 proxy).

        sim_a snapshots every tick; sim_b snapshots every other tick.
        Positions and evacuated count must be identical at each step.
        """
        sim_a = _make_open_sim(open_room_floor, open_room_flow, seed=5, n_agents=15)
        sim_b = _make_open_sim(open_room_floor, open_room_flow, seed=5, n_agents=15)

        for i in range(100):
            if sim_a.is_complete or sim_b.is_complete:
                break
            sim_a.snapshot()
            if i % 2 == 0:
                sim_b.snapshot()
            sim_a.step()
            sim_b.step()

        np.testing.assert_array_equal(sim_a.state.pos, sim_b.state.pos)
        np.testing.assert_array_equal(sim_a.state.alive, sim_b.state.alive)
        assert sim_a.exit_model.evacuated_count == sim_b.exit_model.evacuated_count


# ---------------------------------------------------------------------------
# Small synthetic scenario — guaranteed full evacuation
# ---------------------------------------------------------------------------


class TestSmallSyntheticFullEvacuation:
    """A small fully-solvable open-room scenario evacuates all agents cleanly."""

    def test_all_agents_evacuated_open_room(self, small_sim: Simulation) -> None:
        """Every agent egresses cleanly: evacuated == initial and active_count == 0 (R5.3)."""
        initial_count = small_sim.state.count
        _run_to_completion(small_sim, _SMALL_EVAC_CEILING)

        assert small_sim.is_complete
        assert small_sim.state.active_indices.size == 0, (
            f"{small_sim.state.active_indices.size} agents still active; "
            "stall-guard fired before full evacuation."
        )
        assert small_sim.exit_model.evacuated_count == initial_count

    def test_evacuation_completes_before_ceiling(self, small_sim: Simulation) -> None:
        """Full evacuation finishes well short of the ceiling (no timeout)."""
        _run_to_completion(small_sim, _SMALL_EVAC_CEILING)
        assert small_sim.tick < _SMALL_EVAC_CEILING, (
            f"Evacuation hit the {_SMALL_EVAC_CEILING}-tick ceiling."
        )

    def test_tick_advances_monotonically(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """Tick counter increments by exactly 1 per step."""
        sim = _make_open_sim(open_room_floor, open_room_flow, seed=1, n_agents=5)
        prev = sim.tick
        for _ in range(50):
            if sim.is_complete:
                break
            sim.step()
            assert sim.tick == prev + 1
            prev = sim.tick

    def test_evacuated_count_monotonically_non_decreasing(
        self,
        open_room_floor: FloorPlan,
        open_room_flow: FlowField,
    ) -> None:
        """Evacuated count never decreases between ticks."""
        sim = _make_open_sim(open_room_floor, open_room_flow, seed=2, n_agents=15)
        prev_evac = 0
        for _ in range(200):
            if sim.is_complete:
                break
            sim.step()
            current = sim.exit_model.evacuated_count
            assert current >= prev_evac, (
                f"Evacuated count decreased from {prev_evac} to {current} "
                f"at tick {sim.tick}."
            )
            prev_evac = current


# ---------------------------------------------------------------------------
# Panic injection mid-run (FR-12.1)
# ---------------------------------------------------------------------------


class TestInjectionMidRun:
    """Injecting a panic source mid-run is logged and visible in the snapshot."""

    def test_injection_logged_with_correct_tick(self, small_sim: Simulation) -> None:
        """add_panic_source appends a panic_source_added event at the injection tick."""
        for _ in range(10):
            small_sim.step()
        injection_tick = small_sim.tick

        add_panic_source(small_sim, "fire", pos=(5.0, 2.5), intensity=1.0, radius=3.0)

        snap = small_sim.snapshot()
        logged = [e for e in snap.events if e.kind == "panic_source_added"]
        assert len(logged) == 1, "Expected exactly one panic_source_added event."
        assert logged[0].tick == injection_tick, (
            f"Event tick {logged[0].tick} != injection tick {injection_tick}"
        )

    def test_injection_appears_in_snapshot_panic_sources(
        self, small_sim: Simulation
    ) -> None:
        """A placed panic source is visible in the snapshot's panic_sources tuple."""
        small_sim.step()
        assert len(small_sim.snapshot().panic_sources) == 0

        add_panic_source(small_sim, "fire", pos=(5.0, 2.5), radius=PANIC_RANGE)

        snap = small_sim.snapshot()
        assert len(snap.panic_sources) == 1
        src = snap.panic_sources[0]
        assert abs(src.x - 5.0) < 1e-9
        assert abs(src.y - 2.5) < 1e-9

    def test_injection_raises_agent_panic_next_tick(
        self, small_sim: Simulation
    ) -> None:
        """Agents near the injection point have panic > 0 after the next step."""
        assert np.all(small_sim.state.panic == 0.0), "panic must start at 0"

        add_panic_source(
            small_sim, "fire", pos=(5.0, 2.5), intensity=1.0, radius=20.0
        )
        small_sim.step()

        active = small_sim.state.active_indices
        if active.size > 0:
            assert np.any(small_sim.state.panic[active] > 0.0), (
                "No agent raised their panic after a room-wide injection."
            )

    def test_injection_event_payload_has_expected_keys(
        self, small_sim: Simulation
    ) -> None:
        """panic_source_added payload contains pos, intensity, and radius keys."""
        add_panic_source(
            small_sim, "fire", pos=(3.0, 1.0), intensity=0.8, radius=4.0
        )
        snap = small_sim.snapshot()
        evt = next(e for e in snap.events if e.kind == "panic_source_added")
        assert "pos" in evt.payload
        assert "intensity" in evt.payload
        assert "radius" in evt.payload
        assert evt.payload["intensity"] == pytest.approx(0.8)
        assert evt.payload["radius"] == pytest.approx(4.0)
