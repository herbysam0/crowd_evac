"""Shared fixtures for domain-layer tests.

Provides a factory fixture for building :class:`AgentState` instances from
plain Python lists or arrays, so individual tests request a builder rather
than calling a module-level helper.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.agent_state import AgentState

# A builder taking (pos, vel=None, panic=None, alive=None) -> AgentState.
MakeState = Callable[..., AgentState]


@pytest.fixture
def make_state() -> MakeState:
    """Return a factory that builds an AgentState from array-like fields.

    The returned callable accepts ``pos`` (required, shape ``(N, 2)``) and
    optional ``vel``, ``panic``, and ``alive``; omitted fields default to
    zero velocity, zero panic, all-alive, and an unassigned goal.

    Returns:
        A callable ``make(pos, vel=None, panic=None, alive=None)`` returning
        a fully-populated :class:`AgentState`.
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

        if vel is not None:
            vel_arr = np.asarray(vel, dtype=np.float64)
            if vel_arr.size == 0:
                vel_arr = vel_arr.reshape(0, 2)
        else:
            vel_arr = np.zeros((n, 2), dtype=np.float64)

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
