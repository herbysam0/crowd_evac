"""Shared fixtures for domain-layer tests.

Provides factory fixtures for building :class:`AgentState` and
:class:`~crowd_evac.domain.panic_source.PanicSource` instances from plain
Python values, so individual tests request a builder rather than calling a
module-level helper.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.panic_source import PanicSource

# A builder taking (pos, vel=None, panic=None, alive=None) -> AgentState.
MakeState = Callable[..., AgentState]

# A builder taking keyword overrides -> PanicSource (non-decaying by default).
MakeSource = Callable[..., PanicSource]


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


@pytest.fixture
def make_source() -> MakeSource:
    """Return a factory that builds a :class:`PanicSource` from keyword args.

    Defaults: origin position, full intensity, ``PANIC_RANGE`` radius, and
    ``decay_rate=0.0`` (non-decaying) so test assertions stay deterministic
    unless the caller explicitly requests decay.

    Returns:
        A callable with signature
        ``make(x=0.0, y=0.0, intensity=1.0, radius=10.0, decay_rate=0.0)``
        returning a :class:`PanicSource`.
    """

    def _make(
        x: float = 0.0,
        y: float = 0.0,
        intensity: float = 1.0,
        radius: float = 10.0,
        decay_rate: float = 0.0,
    ) -> PanicSource:
        """Build a PanicSource with sensible test defaults."""
        return PanicSource(
            x=x,
            y=y,
            intensity=intensity,
            radius=radius,
            decay_rate=decay_rate,
        )

    return _make
