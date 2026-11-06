"""Shared pytest fixtures for crowd_evac tests."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Wall

# A builder taking (pos, vel=None, panic=None, alive=None) -> AgentState.
MakeState = Callable[..., AgentState]


@pytest.fixture
def make_state() -> MakeState:
    """Return a factory that builds an AgentState from array-like fields.

    The returned callable accepts ``pos`` (required, shape ``(N, 2)``) and
    optional ``vel``, ``panic``, and ``alive``; omitted fields default to
    zero velocity, zero panic, all-alive, and an unassigned goal (``-1``).

    Returns:
        A callable ``make(pos, vel=None, panic=None, alive=None)``
        returning a fully-populated :class:`AgentState`.
    """

    def _make(
        pos: npt.ArrayLike,
        vel: npt.ArrayLike | None = None,
        panic: npt.ArrayLike | None = None,
        alive: npt.ArrayLike | None = None,
    ) -> AgentState:
        """Build an AgentState, broadcasting sensible defaults."""
        pos_arr = np.asarray(pos, dtype=np.float64)
        if pos_arr.size == 0:
            pos_arr = pos_arr.reshape(0, 2)
        n = pos_arr.shape[0]

        vel_arr = (
            np.asarray(vel, dtype=np.float64)
            if vel is not None
            else np.zeros((n, 2), dtype=np.float64)
        )
        if vel_arr.size == 0:
            vel_arr = vel_arr.reshape(0, 2)

        panic_arr = (
            np.asarray(panic, dtype=np.float64)
            if panic is not None
            else np.zeros(n, dtype=np.float64)
        )
        alive_arr = (
            np.asarray(alive, dtype=np.bool_)
            if alive is not None
            else np.ones(n, dtype=np.bool_)
        )
        return AgentState(
            pos=pos_arr,
            vel=vel_arr,
            panic=panic_arr,
            goal=np.full(n, -1, dtype=np.intp),
            alive=alive_arr,
        )

    return _make


@pytest.fixture
def room_model_3() -> FloorPlan:
    """Pre-apron ``room_model_3`` geometry (issue #17 regression fixture).

    16.2 x 10 m: 1 m perimeter walls, a hall along the top (y 7.4-9.0)
    separated from three lower rooms by 0.1 m walls at y 7.3-7.4, and
    1.6 m doors through the west and east perimeter walls spanning
    y 7.4-9.0. ``wall_5`` touches the west door's lower edge.
    """
    walls = (
        Wall(0.0, 9.0, 16.2, 1.0, "wall_1"),
        Wall(15.2, 0.0, 1.0, 10.0, "wall_2"),
        Wall(0.0, 0.0, 16.2, 1.0, "wall_3"),
        Wall(0.0, 0.0, 1.0, 10.0, "wall_4"),
        Wall(1.0, 7.3, 3.7, 0.1, "wall_5"),
        Wall(5.6, 7.3, 3.7, 0.1, "wall_6"),
        Wall(11.1, 7.3, 4.1, 0.1, "wall_7"),
        Wall(5.55, 1.0, 0.1, 6.4, "wall_8"),
        Wall(10.1, 1.0, 0.1, 6.4, "wall_9"),
    )
    exits = (
        Exit(1.0, 8.2, 1.6, ExitSide.WEST, 1.0, "door_10"),
        Exit(15.2, 8.2, 1.6, ExitSide.EAST, 1.0, "door_11"),
    )
    return FloorPlan(16.2, 10.0, walls, (), exits)
