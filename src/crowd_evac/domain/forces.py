"""Force terms acting on the agent population (FR-1 R1.2 / FR-14).

Each function returns a ``(N, 2)`` float64 acceleration array, where
``N == state.count``.  Dead agents (``alive[i] == False``) always receive
a zero row.  Force terms are additive; callers sum them before passing the
total to :func:`~crowd_evac.domain.integrator.step`.

Functions defined here (step 1.7):

- :func:`f_exit`: Exit-seeking self-driven force from the flow field.

Further terms — ``f_crowd``, ``f_panic_repulsion``, ``compose`` — will be
added in steps 1.8–1.11 to this same module without changing the integrator.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState, Vec2Array
from crowd_evac.domain.constants import (
    MAX_SPEED,
    PANIC_SPEED_MULTIPLIER,
    RELAXATION_TIME,
)
from crowd_evac.pathfinding.flow_field import FlowField


def f_exit(
    state: AgentState,
    field: FlowField,
    relaxation_time: float = RELAXATION_TIME,
) -> Vec2Array:
    """Compute exit-seeking acceleration for all agents (FR-1 R1.2).

    Samples the flow field at each active agent's world position to get the
    desired exit-seeking direction, scales it by the agent's panic-modulated
    desired speed, and returns the social-force steering acceleration::

        a_i = (v_desired_i * direction_i − v_i) / relaxation_time

    where::

        v_desired_i = MAX_SPEED * (1 + panic_i * (PANIC_SPEED_MULTIPLIER − 1))

    Dead agents (``alive[i] == False``) receive a zero row in the output.

    Args:
        state: Current agent state (positions, velocities, panic levels).
        field: Pre-computed flow field providing exit-seeking unit directions.
        relaxation_time: Characteristic time in seconds for velocity
            relaxation. Must be positive.

    Returns:
        Float64 array of shape ``(N, 2)`` with per-agent acceleration vectors
        in m/s². Dead agents have zero rows.

    Raises:
        ValueError: If relaxation_time is not positive.
    """
    if relaxation_time <= 0.0:
        raise ValueError(
            f"relaxation_time must be positive, got {relaxation_time!r}"
        )
    n = state.count
    out: Vec2Array = np.zeros((n, 2), dtype=np.float64)
    active = state.active_indices
    if active.size == 0:
        return out

    dirs: npt.NDArray[np.float64] = field.sample(state.pos[active])  # (A, 2)

    # Linearly interpolate desired speed between MAX_SPEED (no panic) and
    # MAX_SPEED * PANIC_SPEED_MULTIPLIER (full panic).
    desired_speed: npt.NDArray[np.float64] = MAX_SPEED * (
        1.0 + state.panic[active] * (PANIC_SPEED_MULTIPLIER - 1.0)
    )  # (A,)

    desired_vel: Vec2Array = dirs * desired_speed[:, np.newaxis]  # (A, 2)
    out[active] = (desired_vel - state.vel[active]) / relaxation_time
    return out
