"""Evacuation-complete predicate (FR-5 R5.3).

Exports a single function :func:`is_evacuation_complete` that the
simulation loop calls each tick to decide whether to halt.  Two
terminal conditions are recognised:

1. **All evacuated** — no active agents remain.
2. **Stalled** — alive agents exist but no egress has occurred for
   *stall_ticks* consecutive ticks, indicating the remainder are
   trapped and will never reach an exit.
"""
from __future__ import annotations

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import STALL_TICKS
from crowd_evac.domain.exit_model import ExitModel


def is_evacuation_complete(
    state: AgentState,
    exit_model: ExitModel,
    stall_ticks: int = STALL_TICKS,
) -> bool:
    """Return ``True`` when the evacuation has terminated (R5.3).

    Evacuation is considered complete when either:

    1. No active agents remain — all have successfully egressed, or
    2. No agent has egressed for *stall_ticks* consecutive ticks —
       the remaining alive agents are unreachable and will never egress.

    Args:
        state: Current agent population.
        exit_model: Exit manager tracking per-tick egress events.
        stall_ticks: Consecutive no-egress ticks before declaring the
            simulation stalled.  Must be positive.

    Returns:
        ``True`` if evacuation is complete; ``False`` otherwise.

    Raises:
        ValueError: If *stall_ticks* is not positive.
    """
    if stall_ticks <= 0:
        raise ValueError(
            f"stall_ticks must be positive, got {stall_ticks!r}"
        )
    if state.active_indices.size == 0:
        return True
    return exit_model.ticks_since_last_egress >= stall_ticks
