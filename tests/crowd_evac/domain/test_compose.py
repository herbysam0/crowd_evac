"""Tests for crowd_evac.domain.forces.compose (FR-14 R14.1 subset).

Covers:
  - compose(): output shape, dtype, zero rows for dead agents, empty population.
  - Additive correctness: compose with all terms enabled equals the explicit
    sum of individual force functions.
  - Toggle semantics: disabling each term removes exactly its contribution.
  - Panic-repulsion trajectory: an agent near an active panic source is pushed
    away when enable_panic_repulsion=True vs a baseline with it disabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.forces import (
    compose,
    f_crowd,
    f_density,
    f_exit,
    f_herd,
    f_panic_repulsion,
)
from crowd_evac.domain.integrator import step
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource
from crowd_evac.pathfinding.flow_field import FlowField

from .conftest import MakeState


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corridor_floor() -> FloorPlan:
    """10 m × 2 m open corridor with a single exit on the east wall."""
    return FloorPlan(
        width_m=10.0,
        height_m=2.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=10.0,
                y=1.0,
                width_m=1.0,
                side=ExitSide.EAST,
                capacity_per_second=10,
                label="east",
            ),
        ),
    )


@pytest.fixture
def field(corridor_floor: FloorPlan) -> FlowField:
    """Flow field built from the corridor floor plan."""
    return FlowField.build(corridor_floor)


@pytest.fixture
def empty_panic_field() -> PanicField:
    """Panic field with no sources — all contributions are zero."""
    return PanicField()


@pytest.fixture
def west_panic_field() -> PanicField:
    """Panic field with a strong, non-decaying source at the west end."""
    source = PanicSource(x=0.0, y=1.0, intensity=1.0, radius=12.0, decay_rate=0.0)
    return PanicField([source])


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestComposeShape:
    """compose() must return (N, 2) float64 for any agent count."""

    def test_shape_two_agents(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """Two active agents produce shape (2, 2)."""
        state = make_state([[1.0, 1.0], [3.0, 1.0]])
        out = compose(state, field, empty_panic_field)
        assert out.shape == (2, 2)

    def test_dtype_float64(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """Output dtype must be float64."""
        state = make_state([[1.0, 1.0]])
        out = compose(state, field, empty_panic_field)
        assert out.dtype == np.float64

    def test_empty_population(
        self, field: FlowField, empty_panic_field: PanicField
    ) -> None:
        """Empty population produces shape (0, 2)."""
        state = AgentState(
            pos=np.empty((0, 2), dtype=np.float64),
            vel=np.empty((0, 2), dtype=np.float64),
            panic=np.empty(0, dtype=np.float64),
            goal=np.empty(0, dtype=np.intp),
            alive=np.empty(0, dtype=np.bool_),
        )
        out = compose(state, field, empty_panic_field)
        assert out.shape == (0, 2)

    def test_dead_agents_receive_zero_rows(
        self,
        make_state: MakeState,
        field: FlowField,
        west_panic_field: PanicField,
    ) -> None:
        """Dead agents get a zero row regardless of enabled terms."""
        state = make_state(
            [[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]],
            alive=[False, True, False],
        )
        out = compose(state, field, west_panic_field)
        np.testing.assert_array_equal(out[0], [0.0, 0.0])
        np.testing.assert_array_equal(out[2], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Additive correctness
# ---------------------------------------------------------------------------


class TestComposeAdditive:
    """compose() must equal the explicit sum of all enabled individual terms."""

    def test_all_terms_match_explicit_sum(
        self,
        make_state: MakeState,
        field: FlowField,
    ) -> None:
        """compose(all enabled) == f_exit + f_crowd + f_density + f_herd + f_panic_repulsion."""
        # Agents close enough to generate crowd forces; panic source nearby.
        state = make_state(
            [[1.0, 1.0], [1.3, 1.0], [4.0, 1.0]],
            vel=[[0.4, 0.0], [0.5, 0.0], [0.3, 0.1]],
            panic=[0.2, 0.5, 0.1],
        )
        source = PanicSource(x=0.0, y=1.0, intensity=1.0, radius=12.0, decay_rate=0.0)
        pf = PanicField([source])

        expected = (
            f_exit(state, field)
            + f_crowd(state)
            + f_density(state)
            + f_herd(state)
            + f_panic_repulsion(state, pf)
        )
        result = compose(state, field, pf)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_all_terms_disabled_returns_zeros(
        self,
        make_state: MakeState,
        field: FlowField,
        west_panic_field: PanicField,
    ) -> None:
        """With every term disabled the output is all zeros."""
        state = make_state([[1.0, 1.0], [1.3, 1.0]], panic=[0.5, 0.5])
        out = compose(
            state,
            field,
            west_panic_field,
            enable_exit=False,
            enable_crowd=False,
            enable_density=False,
            enable_herd=False,
            enable_panic_repulsion=False,
        )
        np.testing.assert_array_equal(out, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Toggle semantics — disabling a term removes exactly its contribution
# ---------------------------------------------------------------------------


class TestComposeToggles:
    """Disabling each term removes exactly its contribution from the total."""

    def _two_close_agents(self, make_state: MakeState) -> AgentState:
        """Two agents at 0.3 m separation, panic 0.4, moving right."""
        return make_state(
            [[2.0, 1.0], [2.3, 1.0]],
            vel=[[0.5, 0.0], [0.4, 0.0]],
            panic=[0.4, 0.4],
        )

    def test_disable_exit_removes_exit_term(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """compose(all) − compose(no exit) == f_exit."""
        state = self._two_close_agents(make_state)
        all_on = compose(state, field, empty_panic_field)
        no_exit = compose(state, field, empty_panic_field, enable_exit=False)
        np.testing.assert_allclose(all_on - no_exit, f_exit(state, field), atol=1e-12)

    def test_disable_crowd_removes_crowd_term(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """compose(all) − compose(no crowd) == f_crowd."""
        state = self._two_close_agents(make_state)
        all_on = compose(state, field, empty_panic_field)
        no_crowd = compose(state, field, empty_panic_field, enable_crowd=False)
        np.testing.assert_allclose(
            all_on - no_crowd, f_crowd(state), atol=1e-12
        )

    def test_disable_density_removes_density_term(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """compose(all) − compose(no density) == f_density."""
        state = self._two_close_agents(make_state)
        all_on = compose(state, field, empty_panic_field)
        no_density = compose(state, field, empty_panic_field, enable_density=False)
        np.testing.assert_allclose(
            all_on - no_density, f_density(state), atol=1e-12
        )

    def test_disable_herd_removes_herd_term(
        self,
        make_state: MakeState,
        field: FlowField,
        empty_panic_field: PanicField,
    ) -> None:
        """compose(all) − compose(no herd) == f_herd."""
        state = self._two_close_agents(make_state)
        all_on = compose(state, field, empty_panic_field)
        no_herd = compose(state, field, empty_panic_field, enable_herd=False)
        np.testing.assert_allclose(
            all_on - no_herd, f_herd(state), atol=1e-12
        )

    def test_disable_panic_repulsion_removes_panic_term(
        self,
        make_state: MakeState,
        field: FlowField,
        west_panic_field: PanicField,
    ) -> None:
        """compose(all) − compose(no panic) == f_panic_repulsion."""
        state = self._two_close_agents(make_state)
        all_on = compose(state, field, west_panic_field)
        no_panic = compose(
            state, field, west_panic_field, enable_panic_repulsion=False
        )
        np.testing.assert_allclose(
            all_on - no_panic,
            f_panic_repulsion(state, west_panic_field),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Panic-repulsion trajectory: agents flee active sources
# ---------------------------------------------------------------------------


class TestComposePanicRepulsionTrajectory:
    """With panic repulsion enabled an agent near a source is pushed away."""

    def test_panic_repulsion_bends_trajectory_northward(
        self,
        make_state: MakeState,
        field: FlowField,
    ) -> None:
        """A south-of-corridor panic source deflects an agent northward.

        The exit force is purely eastward (corridor centre); panic repulsion
        from a source directly south is purely northward.  With repulsion
        active the agent accumulates y-velocity that the baseline lacks,
        so y_with > y_without after a handful of ticks.
        """
        # Source directly south; repulsion vector = (0, +1) at agent position.
        source = PanicSource(x=5.0, y=-3.0, intensity=1.0, radius=10.0, decay_rate=0.0)
        pf = PanicField([source])

        state_with = make_state([[5.0, 1.0]], panic=[1.0])
        state_without = make_state([[5.0, 1.0]], panic=[1.0])

        for _ in range(5):
            step(state_with, compose(state_with, field, pf))
            step(
                state_without,
                compose(state_without, field, pf, enable_panic_repulsion=False),
            )

        y_with = float(state_with.pos[0, 1])
        y_without = float(state_without.pos[0, 1])
        assert y_with > y_without, (
            f"South panic source should push agent north of baseline: "
            f"y_with={y_with:.4f}, y_without={y_without:.4f}"
        )
