"""Headless room_model_3 regression run for issue #17 (no egress via walls).

Covers:
  - 40 agents, fixed seed: every agent evacuates.
  - Every removed agent was last seen inside a door opening (past its plane,
    within its span) — never in the lower-left room's corner.
  - No active agent centre ever rests in a blocked (wall) cell.

Uses the pre-apron geometry from the shared ``room_model_3`` fixture; the
migrated (apron) variant is covered in the apron-migration phase.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import spawn
from crowd_evac.domain.collision import CollisionMap
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import FloorPlan
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.pathfinding.flow_field import FlowField

AGENTS: int = 40
SEED: int = 7
TICK_CEILING: int = 3_000
DOOR_Y_RANGE: tuple[float, float] = (7.4, 9.0)
WEST_PLANE_X: float = 1.0
EAST_PLANE_X: float = 15.2


def _build(plan: FloorPlan) -> tuple[Simulation, CollisionMap]:
    """Wire a full headless simulation on *plan*."""
    rng = SeededRNG(SEED)
    collision = CollisionMap.from_floor_plan(plan)
    sim = Simulation(
        state=spawn(plan, AGENTS, rng.generator),
        flow_field=FlowField.build(plan),
        panic_field=PanicField(),
        exit_model=ExitModel(plan),
        rng=rng,
        collision_map=collision,
    )
    return sim, collision


@pytest.mark.e2e
def test_room_model_3_all_agents_exit_through_doors(
    room_model_3: FloorPlan,
) -> None:
    """All 40 agents leave via the door gaps; none enters a wall."""
    sim, collision = _build(room_model_3)
    state = sim.state
    removed: list[npt.NDArray[np.float64]] = []
    for _ in range(TICK_CEILING):
        if sim.is_complete:
            break
        alive_before = state.alive.copy()
        pos_before = state.pos.copy()
        sim.step()
        gone = alive_before & ~state.alive
        removed.extend(pos_before[gone])
        assert not collision.is_blocked(state.pos[state.alive]).any()

    assert sim.exit_model.evacuated_count == AGENTS
    for x, y in removed:
        assert DOOR_Y_RANGE[0] <= y <= DOOR_Y_RANGE[1]
        assert x < WEST_PLANE_X or x > EAST_PLANE_X
