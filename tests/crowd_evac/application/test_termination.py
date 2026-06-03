"""Tests for crowd_evac.application.termination (FR-5 R5.3).

Covers:
  - Complete when no active agents remain
  - Incomplete when active agents exist and stall_ticks not yet elapsed
  - Complete when stall_ticks consecutive no-egress ticks have elapsed
  - Stall counter resets after egress; needs a fresh stall_ticks streak
  - ValueError on non-positive stall_ticks
"""
from __future__ import annotations

import pytest

from crowd_evac.application.termination import is_evacuation_complete
from crowd_evac.domain.constants import DT
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan

from tests.conftest import MakeState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_floor() -> FloorPlan:
    """5 × 5 m room with one SOUTH exit, capacity 5 agents/s."""
    return FloorPlan(
        width_m=5.0,
        height_m=5.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=2.5,
                y=0.0,
                width_m=1.0,
                side=ExitSide.SOUTH,
                capacity_per_second=5,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsEvacuationComplete:
    """is_evacuation_complete — all terminal / non-terminal paths."""

    def test_complete_when_no_active_agents(
        self, minimal_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Returns True when all agents have been removed from state."""
        state = make_state([(2.5, 0.3)])
        model = ExitModel(minimal_floor, dt=DT)
        state.remove([0])
        assert is_evacuation_complete(state, model)

    def test_incomplete_with_active_agents_and_no_stall(
        self, minimal_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Returns False when agents are alive and stall_ticks not elapsed."""
        state = make_state([(2.5, 4.0)])  # far from exit
        model = ExitModel(minimal_floor, dt=DT)
        # Zero ticks run: stall counter == 0 < stall_ticks.
        assert not is_evacuation_complete(state, model, stall_ticks=5)

    def test_complete_when_stall_ticks_consecutive_no_egress(
        self, minimal_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Returns True after stall_ticks consecutive ticks with no egress."""
        stall = 5
        state = make_state([(2.5, 4.0)])  # far from exit — never captured
        model = ExitModel(minimal_floor, dt=DT)
        for _ in range(stall):
            model.step(state)
        assert is_evacuation_complete(state, model, stall_ticks=stall)

    def test_stall_resets_after_egress_needs_fresh_streak(
        self, minimal_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """Counter resets on egress; needs stall_ticks more ticks to trigger."""
        stall = 5
        # Agent 0 near exit (egresses early); agent 1 far away (never moves).
        state = make_state([(2.5, 0.3), (2.5, 4.0)])
        model = ExitModel(minimal_floor, dt=DT)
        # Run 8 ticks: agent 0 egresses on tick 4, stall counter = 4 after tick 8.
        for _ in range(8):
            model.step(state)
        assert not is_evacuation_complete(state, model, stall_ticks=stall)
        # Run stall more ticks: counter reaches stall.
        for _ in range(stall):
            model.step(state)
        assert is_evacuation_complete(state, model, stall_ticks=stall)

    def test_nonpositive_stall_ticks_raises(
        self, minimal_floor: FloorPlan, make_state: MakeState
    ) -> None:
        """stall_ticks ≤ 0 raises ValueError."""
        state = make_state([(2.5, 0.3)])
        model = ExitModel(minimal_floor, dt=DT)
        with pytest.raises(ValueError, match="stall_ticks must be positive"):
            is_evacuation_complete(state, model, stall_ticks=0)
