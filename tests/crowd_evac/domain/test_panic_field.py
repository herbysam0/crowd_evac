"""Tests for PanicField and f_panic_repulsion (FR-11 subset, step 1.10).

Covers:
  - PanicField.value_at(): max at source, linear decay with distance, zero
    outside radius, multi-source clamping, empty field, inactive source.
  - PanicField.repulsion_at(): away-from-source direction, zero outside
    radius, stronger closer to source.
  - PanicField mutation: add_source, remove_source, decay_all.
  - f_panic_repulsion(): shape, dtype, dead-agent zeros, correct direction
    (R11.4), strength scaling, zero for out-of-range agents, failure paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.agent_state import AgentState
from crowd_evac.domain.constants import DT
from crowd_evac.domain.forces import f_panic_repulsion
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.panic_source import PanicSource

from .conftest import MakeSource, MakeState


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def origin_source(make_source: MakeSource) -> PanicSource:
    """Non-decaying source at the origin with radius 10 m, intensity 1.0."""
    return make_source(x=0.0, y=0.0, intensity=1.0, radius=10.0)


@pytest.fixture
def single_source_field(origin_source: PanicSource) -> PanicField:
    """PanicField containing only the origin source."""
    return PanicField([origin_source])


# ---------------------------------------------------------------------------
# PanicField.value_at
# ---------------------------------------------------------------------------


class TestPanicFieldValueAt:
    """PanicField.value_at() must produce the correct scalar decay field."""

    def test_value_equals_intensity_at_source_position(
        self, single_source_field: PanicField
    ) -> None:
        """Field value at the source position equals source intensity."""
        vals = single_source_field.value_at(
            np.array([[0.0, 0.0]], dtype=np.float64)
        )
        assert vals[0] == pytest.approx(1.0)

    def test_value_decays_linearly_with_distance(
        self, single_source_field: PanicField
    ) -> None:
        """Value at d = R/2 is half the source intensity (linear decay)."""
        positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
        vals = single_source_field.value_at(positions)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(0.5)

    def test_value_zero_at_radius_boundary(
        self, single_source_field: PanicField
    ) -> None:
        """Field value is zero at exactly the influence radius distance."""
        vals = single_source_field.value_at(
            np.array([[10.0, 0.0]], dtype=np.float64)
        )
        assert vals[0] == pytest.approx(0.0, abs=1e-9)

    def test_value_zero_outside_radius(
        self, make_source: MakeSource
    ) -> None:
        """Field value is strictly zero beyond the influence radius."""
        field = PanicField([make_source(radius=5.0)])
        vals = field.value_at(np.array([[6.0, 0.0]], dtype=np.float64))
        assert vals[0] == 0.0

    def test_empty_field_returns_zeros(self) -> None:
        """Field with no sources returns zero for all query positions."""
        field = PanicField()
        vals = field.value_at(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        )
        np.testing.assert_array_equal(vals, np.zeros(2))

    def test_inactive_source_contributes_nothing(
        self, make_source: MakeSource
    ) -> None:
        """An expired source (intensity=0.0) contributes zero to the field."""
        field = PanicField([make_source(intensity=0.0)])
        vals = field.value_at(np.array([[0.0, 0.0]], dtype=np.float64))
        assert vals[0] == pytest.approx(0.0, abs=1e-9)

    def test_two_coincident_sources_clamp_to_one(
        self, make_source: MakeSource
    ) -> None:
        """Two overlapping full-intensity sources sum to > 1 but are clamped."""
        field = PanicField([make_source(), make_source()])
        vals = field.value_at(np.array([[0.0, 0.0]], dtype=np.float64))
        # Raw sum = 2.0; clamped output must be exactly 1.0.
        assert vals[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PanicField.repulsion_at
# ---------------------------------------------------------------------------


class TestPanicFieldRepulsionAt:
    """PanicField.repulsion_at() must point away from each source."""

    def test_repulsion_points_away_from_source(
        self, single_source_field: PanicField
    ) -> None:
        """Agent to the right of origin gets a positive x repulsion component."""
        rep = single_source_field.repulsion_at(
            np.array([[1.0, 0.0]], dtype=np.float64)
        )
        assert rep[0, 0] > 0.0, "Expected +x direction (away from origin)"
        assert abs(rep[0, 1]) < 1e-9

    def test_repulsion_zero_outside_radius(
        self, make_source: MakeSource
    ) -> None:
        """Agent beyond source radius receives zero repulsion."""
        field = PanicField([make_source(radius=5.0)])
        rep = field.repulsion_at(np.array([[6.0, 0.0]], dtype=np.float64))
        np.testing.assert_array_almost_equal(rep[0], [0.0, 0.0])

    def test_repulsion_stronger_nearer_source(
        self, single_source_field: PanicField
    ) -> None:
        """Closer agent receives a larger repulsion magnitude than far agent."""
        positions = np.array([[1.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        rep = single_source_field.repulsion_at(positions)
        # d=1: field_val ≈ 0.9; d=4: field_val = 0.6.
        assert np.linalg.norm(rep[0]) > np.linalg.norm(rep[1])


# ---------------------------------------------------------------------------
# PanicField mutation
# ---------------------------------------------------------------------------


class TestPanicFieldMutation:
    """add_source, remove_source, and decay_all must update state correctly."""

    def test_add_source_contributes_immediately(
        self, make_source: MakeSource
    ) -> None:
        """A newly added source affects value_at on the very next call."""
        field = PanicField()
        before = field.value_at(np.array([[0.0, 0.0]], dtype=np.float64))[0]
        assert before == 0.0
        field.add_source(make_source())
        after = field.value_at(np.array([[0.0, 0.0]], dtype=np.float64))[0]
        assert after > 0.0

    def test_remove_source_stops_contribution(
        self, make_source: MakeSource
    ) -> None:
        """Removing a source zeroes the field at its former position."""
        s = make_source()
        field = PanicField([s])
        field.remove_source(s)
        vals = field.value_at(np.array([[0.0, 0.0]], dtype=np.float64))
        assert vals[0] == pytest.approx(0.0, abs=1e-9)

    def test_remove_absent_source_raises(self) -> None:
        """Removing a source not in the field raises ValueError."""
        field = PanicField()
        with pytest.raises(ValueError):
            field.remove_source(PanicSource(x=0.0, y=0.0))

    def test_decay_all_reduces_each_source(self) -> None:
        """decay_all applies decay_rate * dt to every source in the field."""
        s1 = PanicSource(x=0.0, y=0.0, intensity=1.0, decay_rate=0.2)
        s2 = PanicSource(x=5.0, y=0.0, intensity=0.8, decay_rate=0.1)
        field = PanicField([s1, s2])
        field.decay_all(DT)
        assert s1.intensity == pytest.approx(1.0 - 0.2 * DT, rel=1e-9)
        assert s2.intensity == pytest.approx(0.8 - 0.1 * DT, rel=1e-9)


# ---------------------------------------------------------------------------
# f_panic_repulsion — shape and dtype
# ---------------------------------------------------------------------------


class TestFPanicRepulsionShape:
    """f_panic_repulsion must return an (N, 2) float64 array."""

    def test_single_agent_shape(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Single active agent produces shape (1, 2)."""
        result = f_panic_repulsion(make_state([[1.0, 0.0]]), single_source_field)
        assert result.shape == (1, 2)

    def test_multiple_agents_shape(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Five active agents produce shape (5, 2)."""
        state = make_state([[float(i), 0.0] for i in range(1, 6)])
        result = f_panic_repulsion(state, single_source_field)
        assert result.shape == (5, 2)

    def test_dtype_float64(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Output dtype must be float64."""
        result = f_panic_repulsion(make_state([[1.0, 0.0]]), single_source_field)
        assert result.dtype == np.float64

    def test_empty_population_shape(
        self, single_source_field: PanicField
    ) -> None:
        """Empty population (N=0) produces shape (0, 2)."""
        state = AgentState(
            pos=np.empty((0, 2), dtype=np.float64),
            vel=np.empty((0, 2), dtype=np.float64),
            panic=np.empty(0, dtype=np.float64),
            goal=np.empty(0, dtype=np.intp),
            alive=np.empty(0, dtype=np.bool_),
        )
        result = f_panic_repulsion(state, single_source_field)
        assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# f_panic_repulsion — dead agents
# ---------------------------------------------------------------------------


class TestFPanicRepulsionDeadAgents:
    """Dead agents must receive zero rows (R1.4)."""

    def test_dead_agent_gets_zero(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Agent with alive=False gets a (0, 0) row."""
        result = f_panic_repulsion(
            make_state([[1.0, 0.0]], alive=[False]), single_source_field
        )
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_all_dead_returns_all_zeros(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """All-dead population produces an all-zero output."""
        state = make_state([[1.0, 0.0], [2.0, 0.0]], alive=[False, False])
        result = f_panic_repulsion(state, single_source_field)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# f_panic_repulsion — direction correctness (R11.4)
# ---------------------------------------------------------------------------


class TestFPanicRepulsionDirection:
    """Active agents near a source must accelerate away from it (R11.4)."""

    def test_agent_right_of_source_gets_positive_x(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Agent at (1, 0) gets a positive x-acceleration (away from origin)."""
        result = f_panic_repulsion(
            make_state([[1.0, 0.0]]), single_source_field
        )
        assert result[0, 0] > 0.0, "Expected +x direction (away from origin)"
        assert abs(result[0, 1]) < 1e-9

    def test_agent_above_source_gets_positive_y(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Agent at (0, 2) gets a positive y-acceleration (away from origin)."""
        result = f_panic_repulsion(
            make_state([[0.0, 2.0]]), single_source_field
        )
        assert result[0, 1] > 0.0, "Expected +y direction (away from origin)"

    def test_out_of_range_agent_gets_zero(
        self,
        make_state: MakeState,
        make_source: MakeSource,
    ) -> None:
        """Agent beyond source radius receives no repulsion force."""
        field = PanicField([make_source(radius=10.0)])
        result = f_panic_repulsion(make_state([[20.0, 0.0]]), field)
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# f_panic_repulsion — strength scaling
# ---------------------------------------------------------------------------


class TestFPanicRepulsionStrength:
    """strength parameter must scale the output proportionally."""

    def test_doubled_strength_doubles_magnitude(
        self,
        make_state: MakeState,
        make_source: MakeSource,
    ) -> None:
        """Doubling strength doubles the acceleration magnitude."""
        state = make_state([[1.0, 0.0]])
        # Build two independent fields with the same source parameters.
        field1 = PanicField([make_source()])
        field2 = PanicField([make_source()])
        a1 = f_panic_repulsion(state, field1, strength=1.0)
        a2 = f_panic_repulsion(state, field2, strength=2.0)
        np.testing.assert_allclose(
            np.linalg.norm(a2[0]),
            np.linalg.norm(a1[0]) * 2.0,
            rtol=1e-9,
        )

    def test_zero_strength_returns_zeros(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """strength=0.0 produces zero acceleration for all agents."""
        result = f_panic_repulsion(
            make_state([[1.0, 0.0]]), single_source_field, strength=0.0
        )
        np.testing.assert_array_equal(result, np.zeros((1, 2)))


# ---------------------------------------------------------------------------
# f_panic_repulsion — failure paths
# ---------------------------------------------------------------------------


class TestFPanicRepulsionFailurePaths:
    """f_panic_repulsion must raise on invalid arguments."""

    def test_negative_strength_raises(
        self,
        make_state: MakeState,
        single_source_field: PanicField,
    ) -> None:
        """Negative strength raises ValueError (message: 'strength must be non-negative')."""
        with pytest.raises(ValueError, match="strength must be non-negative"):
            f_panic_repulsion(
                make_state([[1.0, 0.0]]),
                single_source_field,
                strength=-1.0,
            )
