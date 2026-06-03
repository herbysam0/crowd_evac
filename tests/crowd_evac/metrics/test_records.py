"""Tests for crowd_evac.metrics.records (FR-8 / R8.2 / R8.3).

Covers:
  - make_record returns a MetricRecord from any snapshot (headless, R8.2).
  - MetricRecord contains the four required FR-8 fields.
  - MetricRecord is JSON-serializable.
  - egressed_this_tick matches the tick_advanced event payload.
  - per_exit_egress sums to egressed_this_tick.
  - peak_density_m2 rises when agents are clustered (directional, R8.3).
  - evac_progress is clamped to [0.0, 1.0] and equals 1.0 when count is 0.
  - make_record raises ValueError on invalid initial_count or cell_size.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import SimEvent, SimSnapshot, Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.constants import DT
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.metrics.records import MetricRecord, make_record
from crowd_evac.pathfinding.flow_field import FlowField

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_N_AGENTS = 8


@pytest.fixture
def floor() -> FloorPlan:
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
def flow(floor: FloorPlan) -> FlowField:
    """Flow field for the standard test floor."""
    return FlowField.build(floor)


@pytest.fixture
def sim(floor: FloorPlan, flow: FlowField) -> Simulation:
    """Standard simulation with 8 agents, seed 42, no panic sources."""
    rng = SeededRNG(_SEED)
    state = spawn(floor, _N_AGENTS, rng.generator)
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
    )


def _make_synthetic_snapshot(
    positions: list[list[float]],
    *,
    tick: int = 1,
    evacuated: int = 0,
    egressed_this_tick: int = 0,
    per_exit_egress: list[int] | None = None,
) -> SimSnapshot:
    """Build a SimSnapshot with synthetic agent positions for unit tests.

    Agents are always alive; panic and velocities are zero.  A synthetic
    ``tick_advanced`` event is appended so make_record can extract
    throughput fields without a real Simulation.

    Args:
        positions: List of ``[x, y]`` world positions.
        tick: Snapshot tick value.
        evacuated: Cumulative evacuated count to embed in the snapshot.
        egressed_this_tick: Value to store in the tick_advanced payload.
        per_exit_egress: Per-exit egress list for the tick_advanced payload.

    Returns:
        A fully-populated :class:`SimSnapshot`.
    """
    pos = np.array(positions, dtype=np.float64).reshape(-1, 2)
    n = pos.shape[0]
    evt = SimEvent(
        tick=tick,
        kind="tick_advanced",
        payload={
            "egressed": egressed_this_tick,
            "active": n,
            "evacuated": evacuated,
            "per_exit_egress": per_exit_egress if per_exit_egress is not None else [],
        },
    )
    return SimSnapshot(
        tick=tick,
        sim_time=tick * DT,
        positions=pos,
        velocities=np.zeros((n, 2), dtype=np.float64),
        panics=np.zeros(n, dtype=np.float64),
        alive=np.ones(n, dtype=np.bool_),
        goals=np.full(n, -1, dtype=np.intp),
        evacuated_count=evacuated,
        active_count=n,
        panic_sources=(),
        events=(evt,),
    )


# ---------------------------------------------------------------------------
# TestMakeRecordBasic — headless production and field presence (R8.2 / R8.3)
# ---------------------------------------------------------------------------


class TestMakeRecordBasic:
    """make_record produces a MetricRecord that is available headless."""

    def test_returns_dict(self, sim: Simulation) -> None:
        """make_record returns a plain dict (TypedDict is a dict subclass)."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert isinstance(record, dict)

    def test_contains_tick(self, sim: Simulation) -> None:
        """Record contains tick field matching the snapshot tick."""
        sim.step()
        sim.step()
        snap = sim.snapshot()
        record = make_record(snap, initial_count=_N_AGENTS)
        assert record["tick"] == snap.tick

    def test_contains_evac_progress(self, sim: Simulation) -> None:
        """Record contains evac_progress field."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert "evac_progress" in record
        assert 0.0 <= record["evac_progress"] <= 1.0

    def test_contains_total_throughput(self, sim: Simulation) -> None:
        """Record contains egressed_this_tick (total throughput) field."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert "egressed_this_tick" in record
        assert isinstance(record["egressed_this_tick"], int)

    def test_contains_per_exit_egress(self, sim: Simulation) -> None:
        """Record contains per_exit_egress (per-exit throughput) field."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert "per_exit_egress" in record
        assert isinstance(record["per_exit_egress"], list)

    def test_contains_density(self, sim: Simulation) -> None:
        """Record contains peak_density_m2 field."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert "peak_density_m2" in record
        assert record["peak_density_m2"] >= 0.0

    def test_record_before_any_steps(self, sim: Simulation) -> None:
        """make_record works on a tick-0 snapshot (no tick_advanced events yet)."""
        snap = sim.snapshot()
        record = make_record(snap, initial_count=_N_AGENTS)
        assert record["tick"] == 0
        assert record["egressed_this_tick"] == 0
        assert record["per_exit_egress"] == []


# ---------------------------------------------------------------------------
# TestMetricRecordSerialization — JSON-serializable (R8.2)
# ---------------------------------------------------------------------------


class TestMetricRecordSerialization:
    """MetricRecord can be serialized to JSON without error."""

    def test_json_serializable(self, sim: Simulation) -> None:
        """json.dumps succeeds on a MetricRecord from a live snapshot."""
        sim.step()
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        payload = json.dumps(record)
        assert isinstance(payload, str)

    def test_json_round_trip_preserves_tick(self, sim: Simulation) -> None:
        """Tick value survives a JSON round-trip."""
        sim.step()
        sim.step()
        snap = sim.snapshot()
        record = make_record(snap, initial_count=_N_AGENTS)
        reloaded: dict[str, object] = json.loads(json.dumps(record))
        assert reloaded["tick"] == record["tick"]


# ---------------------------------------------------------------------------
# TestThroughputFields — egress counts match event log
# ---------------------------------------------------------------------------


class TestThroughputFields:
    """egressed_this_tick and per_exit_egress are consistent with events."""

    def test_egressed_this_tick_matches_event_payload(self) -> None:
        """egressed_this_tick equals the egressed value in tick_advanced."""
        snap = _make_synthetic_snapshot(
            [[1.0, 1.0], [2.0, 2.0]],
            egressed_this_tick=3,
            per_exit_egress=[3],
        )
        record = make_record(snap, initial_count=10)
        assert record["egressed_this_tick"] == 3

    def test_per_exit_egress_matches_event_payload(self) -> None:
        """per_exit_egress equals the list from tick_advanced."""
        snap = _make_synthetic_snapshot(
            [[1.0, 1.0]],
            egressed_this_tick=5,
            per_exit_egress=[2, 3],
        )
        record = make_record(snap, initial_count=10)
        assert record["per_exit_egress"] == [2, 3]

    def test_per_exit_egress_sums_to_total(self) -> None:
        """sum(per_exit_egress) equals egressed_this_tick."""
        snap = _make_synthetic_snapshot(
            [[1.0, 1.0], [3.0, 3.0]],
            egressed_this_tick=7,
            per_exit_egress=[4, 3],
        )
        record = make_record(snap, initial_count=10)
        assert sum(record["per_exit_egress"]) == record["egressed_this_tick"]

    def test_throughput_zero_at_tick_zero(self, sim: Simulation) -> None:
        """egressed_this_tick is 0 on the tick-0 snapshot (no events yet)."""
        record = make_record(sim.snapshot(), initial_count=_N_AGENTS)
        assert record["egressed_this_tick"] == 0

    def test_live_sim_throughput_matches_last_tick_event(
        self, sim: Simulation
    ) -> None:
        """egressed_this_tick matches the most recent tick_advanced payload."""
        sim.step()
        snap = sim.snapshot()
        record = make_record(snap, initial_count=_N_AGENTS)
        # Find the tick_advanced event for this tick.
        evt = next(
            e
            for e in reversed(snap.events)
            if e.kind == "tick_advanced" and e.tick == snap.tick
        )
        assert record["egressed_this_tick"] == evt.payload["egressed"]


# ---------------------------------------------------------------------------
# TestEvacProgress — fraction and edge cases
# ---------------------------------------------------------------------------


class TestEvacProgress:
    """evac_progress is in [0.0, 1.0] and tracks the evacuation fraction."""

    def test_zero_evacuated_gives_zero_progress(self) -> None:
        """No evacuations → evac_progress == 0.0."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]], evacuated=0)
        record = make_record(snap, initial_count=10)
        assert record["evac_progress"] == pytest.approx(0.0)

    def test_full_evacuation_gives_one(self) -> None:
        """All agents evacuated → evac_progress == 1.0."""
        snap = _make_synthetic_snapshot(
            [[1.0, 1.0]], evacuated=10, egressed_this_tick=10
        )
        record = make_record(snap, initial_count=10)
        assert record["evac_progress"] == pytest.approx(1.0)

    def test_partial_evacuation(self) -> None:
        """Partial evacuation → evac_progress == evacuated / initial."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]], evacuated=4)
        record = make_record(snap, initial_count=8)
        assert record["evac_progress"] == pytest.approx(0.5)

    def test_zero_initial_count_gives_one(self) -> None:
        """Zero initial_count → evac_progress == 1.0 (vacuously complete)."""
        snap = _make_synthetic_snapshot([], evacuated=0)
        record = make_record(snap, initial_count=0)
        assert record["evac_progress"] == pytest.approx(1.0)

    def test_progress_clamped_at_one(self) -> None:
        """evac_progress never exceeds 1.0 even if evacuated > initial."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]], evacuated=12)
        record = make_record(snap, initial_count=10)
        assert record["evac_progress"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestPeakDensity — directional density test (R8.3)
# ---------------------------------------------------------------------------


class TestPeakDensity:
    """peak_density_m2 rises when agents are clustered together."""

    def test_density_positive_with_agents(self) -> None:
        """Any snapshot with live agents has peak_density_m2 > 0."""
        snap = _make_synthetic_snapshot([[1.0, 1.0], [1.1, 1.1]])
        record = make_record(snap, initial_count=2)
        assert record["peak_density_m2"] > 0.0

    def test_density_zero_with_no_agents(self) -> None:
        """Snapshot with no live agents has peak_density_m2 == 0.0."""
        pos = np.empty((0, 2), dtype=np.float64)
        snap = SimSnapshot(
            tick=1,
            sim_time=DT,
            positions=pos,
            velocities=np.empty((0, 2), dtype=np.float64),
            panics=np.empty(0, dtype=np.float64),
            alive=np.empty(0, dtype=np.bool_),
            goals=np.empty(0, dtype=np.intp),
            evacuated_count=0,
            active_count=0,
            panic_sources=(),
            events=(),
        )
        record = make_record(snap, initial_count=0)
        assert record["peak_density_m2"] == pytest.approx(0.0)

    def test_clustered_higher_density_than_spread(self) -> None:
        """Clustered agents produce a higher peak_density_m2 than spread-out ones.

        Eight agents placed within a 0.5 m × 0.5 m cluster must yield a
        higher density than the same eight placed at the corners of a
        10 m × 5 m room.
        """
        # Spread: one agent per corner + midpoints of a 10 × 5 room.
        spread_positions = [
            [0.5, 0.5], [5.0, 0.5], [9.5, 0.5], [0.5, 2.5],
            [9.5, 2.5], [0.5, 4.5], [5.0, 4.5], [9.5, 4.5],
        ]
        # Clustered: all eight agents within a 0.4 m × 0.4 m patch.
        clustered_positions = [
            [5.0 + 0.05 * i, 2.5 + 0.05 * j]
            for i in range(4)
            for j in range(2)
        ]
        snap_spread = _make_synthetic_snapshot(spread_positions)
        snap_clustered = _make_synthetic_snapshot(clustered_positions)

        record_spread = make_record(snap_spread, initial_count=8)
        record_clustered = make_record(snap_clustered, initial_count=8)

        assert record_clustered["peak_density_m2"] > record_spread["peak_density_m2"]


# ---------------------------------------------------------------------------
# TestMakeRecordValidation — invalid argument rejection
# ---------------------------------------------------------------------------


class TestMakeRecordValidation:
    """make_record raises ValueError on bad arguments."""

    def test_negative_initial_count_raises(self) -> None:
        """initial_count < 0 raises ValueError."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]])
        with pytest.raises(ValueError, match="initial_count"):
            make_record(snap, initial_count=-1)

    def test_zero_cell_size_raises(self) -> None:
        """cell_size == 0 raises ValueError."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]])
        with pytest.raises(ValueError, match="cell_size"):
            make_record(snap, initial_count=1, cell_size=0.0)

    def test_negative_cell_size_raises(self) -> None:
        """cell_size < 0 raises ValueError."""
        snap = _make_synthetic_snapshot([[1.0, 1.0]])
        with pytest.raises(ValueError, match="cell_size"):
            make_record(snap, initial_count=1, cell_size=-0.5)
