"""Semi-implicit Euler integrator with speed and acceleration clamping (FR-1 R1.3).

One fixed-timestep step:

1. Clamp the total per-agent acceleration to ``MAX_ACCEL`` magnitude.
2. Semi-implicit Euler velocity update: ``v_new = v + a_clamped * DT``.
3. Clamp per-agent speed to the panic-modulated cap
   ``MAX_SPEED * (1 + panic * (PANIC_SPEED_MULTIPLIER − 1))``.
4. Position update: ``pos_new = pos + v_new * DT``.

Only alive agents are updated; dead agents' state is left unchanged.  The
panic-modulated cap means a fully-calm agent (panic=0) never exceeds
``MAX_SPEED``, while a fully-panicked agent may reach
``MAX_SPEED * PANIC_SPEED_MULTIPLIER`` (R1.2/R1.3).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import (
    DT,
    MAX_ACCEL,
    MAX_SPEED,
    PANIC_SPEED_MULTIPLIER,
)


def step(
    state: AgentState,
    acceleration: npt.NDArray[np.float64],
    *,
    max_accel: float = MAX_ACCEL,
    max_speed: float = MAX_SPEED,
    panic_speed_multiplier: float = PANIC_SPEED_MULTIPLIER,
) -> None:
    """Apply one semi-implicit Euler step to all active agents in place.

    Updates ``state.vel`` and ``state.pos``; dead agents are not touched.

    Clamping order:

    - Acceleration magnitude is clamped to ``max_accel`` before integration.
    - After the velocity update, speed is clamped to the per-agent
      panic-modulated cap ``max_speed * (1 + panic * (panic_speed_multiplier - 1))``.

    Args:
        state: Agent state to update in place.
        acceleration: Total per-agent acceleration in m/s², shape ``(N, 2)``.
            Must have the same first-dimension length as ``state.count``.
        max_accel: Maximum acceleration magnitude clamp in m/s². Defaults to
            the module constant ``MAX_ACCEL``.
        max_speed: Base maximum agent speed in m/s for the speed cap. Defaults
            to the module constant ``MAX_SPEED``.
        panic_speed_multiplier: Speed boost factor at full panic for the cap.
            Defaults to the module constant ``PANIC_SPEED_MULTIPLIER``.

    Raises:
        ValueError: If ``acceleration.shape[0] != state.count``.
    """
    if acceleration.shape[0] != state.count:
        raise ValueError(
            f"acceleration.shape[0] ({acceleration.shape[0]!r}) must equal "
            f"state.count ({state.count!r})"
        )
    active = state.active_indices
    if active.size == 0:
        return

    a: npt.NDArray[np.float64] = acceleration[active].copy()

    # -- Clamp acceleration magnitude to max_accel ------------------------
    a_mag: npt.NDArray[np.float64] = np.linalg.norm(a, axis=1, keepdims=True)
    safe_amag = np.where(a_mag > 0.0, a_mag, 1.0)
    a = np.where(a_mag > max_accel, a * (max_accel / safe_amag), a)

    # -- Semi-implicit Euler: update velocity first, then position --------
    v_new: npt.NDArray[np.float64] = state.vel[active] + a * DT

    # -- Clamp speed to panic-modulated cap --------------------------------
    speed_cap: npt.NDArray[np.float64] = max_speed * (
        1.0 + state.panic[active] * (panic_speed_multiplier - 1.0)
    )  # (A,)
    v_mag: npt.NDArray[np.float64] = np.linalg.norm(
        v_new, axis=1, keepdims=True
    )
    safe_vmag = np.where(v_mag > 0.0, v_mag, 1.0)
    v_new = np.where(
        v_mag > speed_cap[:, np.newaxis],
        v_new * (speed_cap[:, np.newaxis] / safe_vmag),
        v_new,
    )

    state.vel[active] = v_new
    state.pos[active] = state.pos[active] + v_new * DT
