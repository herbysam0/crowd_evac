"""Tests for crowd_evac.domain.params.ForceParams (Phase 2 Step 2.1).

Covers:
  - ForceParams.defaults(): returns correct type with Phase-1 constant values.
  - ForceParams(): field validation raises ValueError on constraint violations.
  - to_array() / from_array(): ordered round-trip and failure paths.
  - compose() with ForceParams.defaults() produces bit-identical output to
    compose() with no params argument (regression lock).
  - Simulation accepts a ForceParams and produces bit-identical ticks to the
    no-params baseline when defaults are used.
  - Custom params propagate into compose output (non-regression check).
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.application.rng import SeededRNG
from crowd_evac.application.simulation import Simulation
from crowd_evac.domain.agent_state import AgentState, spawn
from crowd_evac.domain.constants import (
    DENSITY_PRESSURE_STRENGTH,
    DENSITY_SENSING_RADIUS,
    HERD_ATTRACTION_STRENGTH,
    HERD_PERCEPTION_RADIUS,
    HIGH_DENSITY_THRESHOLD,
    MAX_ACCEL,
    MAX_SPEED,
    PANIC_REPULSION_STRENGTH,
    PANIC_SPEED_MULTIPLIER,
    RELAXATION_TIME,
    REPULSION_RADIUS,
    REPULSION_STRENGTH,
)
from crowd_evac.domain.exit_model import ExitModel
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
from crowd_evac.domain.forces import compose
from crowd_evac.domain.panic_field import PanicField
from crowd_evac.domain.params import N_PARAMS, ForceParams, _FIELD_ORDER
from crowd_evac.pathfinding.flow_field import FlowField


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_N_AGENTS = 10


@pytest.fixture
def open_floor() -> FloorPlan:
    """8 m × 4 m room with a single exit on the east wall."""
    return FloorPlan(
        width_m=8.0,
        height_m=4.0,
        walls=(),
        obstacles=(),
        exits=(
            Exit(
                x=8.0,
                y=2.0,
                width_m=2.0,
                side=ExitSide.EAST,
                capacity_per_second=10,
                label="east",
            ),
        ),
    )


@pytest.fixture
def flow_field(open_floor: FloorPlan) -> FlowField:
    """Flow field built from the test floor."""
    return FlowField.build(open_floor)


@pytest.fixture
def empty_panic() -> PanicField:
    """Panic field with no sources."""
    return PanicField()


@pytest.fixture
def agent_state(open_floor: FloorPlan) -> AgentState:
    """10 agents spawned deterministically for force-term tests."""
    rng = np.random.default_rng(_SEED)
    return spawn(open_floor, _N_AGENTS, rng)


def _build_sim(
    floor: FloorPlan,
    flow: FlowField,
    seed: int = _SEED,
    n: int = _N_AGENTS,
    params: ForceParams | None = None,
) -> Simulation:
    """Construct a fully-wired Simulation with optional ForceParams."""
    rng = SeededRNG(seed)
    state = spawn(floor, n, rng.generator)
    kwargs = {}
    if params is not None:
        kwargs["params"] = params
    return Simulation(
        state=state,
        flow_field=flow,
        panic_field=PanicField(),
        exit_model=ExitModel(floor),
        rng=rng,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ForceParams.defaults()
# ---------------------------------------------------------------------------


class TestForceParamsDefaults:
    """defaults() returns a ForceParams whose fields match Phase-1 constants."""

    def test_defaults_returns_force_params_instance(self) -> None:
        """defaults() returns a ForceParams object."""
        result = ForceParams.defaults()
        assert isinstance(result, ForceParams)

    def test_defaults_equals_bare_constructor(self) -> None:
        """ForceParams() and ForceParams.defaults() are equal."""
        assert ForceParams() == ForceParams.defaults()

    def test_defaults_field_values_match_constants(self) -> None:
        """Every field in defaults() equals its corresponding domain constant."""
        p = ForceParams.defaults()
        assert p.relaxation_time == RELAXATION_TIME
        assert p.panic_speed_multiplier == PANIC_SPEED_MULTIPLIER
        assert p.repulsion_strength == REPULSION_STRENGTH
        assert p.repulsion_radius == REPULSION_RADIUS
        assert p.high_density_threshold == HIGH_DENSITY_THRESHOLD
        assert p.density_pressure_strength == DENSITY_PRESSURE_STRENGTH
        assert p.density_sensing_radius == DENSITY_SENSING_RADIUS
        assert p.herd_attraction_strength == HERD_ATTRACTION_STRENGTH
        assert p.herd_perception_radius == HERD_PERCEPTION_RADIUS
        assert p.panic_repulsion_strength == PANIC_REPULSION_STRENGTH
        assert p.max_accel == MAX_ACCEL
        assert p.max_speed == MAX_SPEED

    def test_defaults_is_frozen(self) -> None:
        """ForceParams instances are frozen; attribute assignment raises."""
        p = ForceParams.defaults()
        with pytest.raises(Exception):
            p.relaxation_time = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Physical constraint validation
# ---------------------------------------------------------------------------


class TestForceParamsValidation:
    """Invalid field values raise ValueError during construction."""

    def test_valid_params_do_not_raise(self) -> None:
        """The default constructor succeeds without raising."""
        ForceParams()  # must not raise

    def test_zero_relaxation_time_raises(self) -> None:
        """relaxation_time=0 raises ValueError."""
        with pytest.raises(ValueError, match="relaxation_time"):
            ForceParams(relaxation_time=0.0)

    def test_negative_relaxation_time_raises(self) -> None:
        """Negative relaxation_time raises ValueError."""
        with pytest.raises(ValueError, match="relaxation_time"):
            ForceParams(relaxation_time=-0.1)

    def test_panic_speed_multiplier_below_one_raises(self) -> None:
        """panic_speed_multiplier < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="panic_speed_multiplier"):
            ForceParams(panic_speed_multiplier=0.9)

    def test_panic_speed_multiplier_exactly_one_is_valid(self) -> None:
        """panic_speed_multiplier=1.0 (no boost) is physically valid."""
        ForceParams(panic_speed_multiplier=1.0)  # must not raise

    def test_negative_repulsion_strength_raises(self) -> None:
        """Negative repulsion_strength raises ValueError."""
        with pytest.raises(ValueError, match="repulsion_strength"):
            ForceParams(repulsion_strength=-0.1)

    def test_zero_repulsion_radius_raises(self) -> None:
        """repulsion_radius=0 raises ValueError."""
        with pytest.raises(ValueError, match="repulsion_radius"):
            ForceParams(repulsion_radius=0.0)

    def test_negative_repulsion_radius_raises(self) -> None:
        """Negative repulsion_radius raises ValueError."""
        with pytest.raises(ValueError, match="repulsion_radius"):
            ForceParams(repulsion_radius=-1.0)

    def test_negative_high_density_threshold_raises(self) -> None:
        """Negative high_density_threshold raises ValueError."""
        with pytest.raises(ValueError, match="high_density_threshold"):
            ForceParams(high_density_threshold=-1.0)

    def test_negative_density_pressure_strength_raises(self) -> None:
        """Negative density_pressure_strength raises ValueError."""
        with pytest.raises(ValueError, match="density_pressure_strength"):
            ForceParams(density_pressure_strength=-0.1)

    def test_zero_density_sensing_radius_raises(self) -> None:
        """density_sensing_radius=0 raises ValueError."""
        with pytest.raises(ValueError, match="density_sensing_radius"):
            ForceParams(density_sensing_radius=0.0)

    def test_negative_herd_attraction_strength_raises(self) -> None:
        """Negative herd_attraction_strength raises ValueError."""
        with pytest.raises(ValueError, match="herd_attraction_strength"):
            ForceParams(herd_attraction_strength=-0.1)

    def test_zero_herd_perception_radius_raises(self) -> None:
        """herd_perception_radius=0 raises ValueError."""
        with pytest.raises(ValueError, match="herd_perception_radius"):
            ForceParams(herd_perception_radius=0.0)

    def test_negative_panic_repulsion_strength_raises(self) -> None:
        """Negative panic_repulsion_strength raises ValueError."""
        with pytest.raises(ValueError, match="panic_repulsion_strength"):
            ForceParams(panic_repulsion_strength=-0.1)

    def test_zero_max_accel_raises(self) -> None:
        """max_accel=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_accel"):
            ForceParams(max_accel=0.0)

    def test_zero_max_speed_raises(self) -> None:
        """max_speed=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_speed"):
            ForceParams(max_speed=0.0)

    def test_zero_strength_fields_are_valid(self) -> None:
        """Strengths of exactly 0.0 are physically valid (term is disabled)."""
        ForceParams(
            repulsion_strength=0.0,
            density_pressure_strength=0.0,
            herd_attraction_strength=0.0,
            panic_repulsion_strength=0.0,
        )  # must not raise


# ---------------------------------------------------------------------------
# to_array / from_array round-trip
# ---------------------------------------------------------------------------


class TestForceParamsArrayRoundTrip:
    """to_array() and from_array() form an exact round-trip."""

    def test_to_array_has_correct_length(self) -> None:
        """to_array() returns a vector of length N_PARAMS."""
        arr = ForceParams.defaults().to_array()
        assert arr.shape == (N_PARAMS,)

    def test_to_array_dtype_is_float64(self) -> None:
        """to_array() output is float64."""
        arr = ForceParams.defaults().to_array()
        assert arr.dtype == np.float64

    def test_n_params_matches_field_order_length(self) -> None:
        """N_PARAMS equals the number of canonical fields."""
        assert N_PARAMS == len(_FIELD_ORDER)

    def test_defaults_round_trip(self) -> None:
        """ForceParams.defaults() survives a to_array → from_array cycle."""
        p = ForceParams.defaults()
        assert ForceParams.from_array(p.to_array()) == p

    def test_custom_params_round_trip(self) -> None:
        """Non-default params survive the round-trip exactly."""
        p = ForceParams(
            relaxation_time=0.3,
            panic_speed_multiplier=1.8,
            repulsion_strength=5.0,
            repulsion_radius=1.0,
            high_density_threshold=3.0,
            density_pressure_strength=1.5,
            density_sensing_radius=2.0,
            herd_attraction_strength=0.5,
            herd_perception_radius=8.0,
            panic_repulsion_strength=3.0,
            max_accel=4.0,
            max_speed=1.5,
        )
        assert ForceParams.from_array(p.to_array()) == p

    def test_to_array_field_order_matches_field_order_constant(self) -> None:
        """Each element of to_array() matches the value of the corresponding field."""
        p = ForceParams.defaults()
        arr = p.to_array()
        for i, field_name in enumerate(_FIELD_ORDER):
            assert arr[i] == pytest.approx(getattr(p, field_name))

    def test_from_array_wrong_size_raises(self) -> None:
        """from_array() raises ValueError for wrong-length input."""
        with pytest.raises(ValueError, match="shape"):
            ForceParams.from_array(np.zeros(N_PARAMS - 1, dtype=np.float64))

    def test_from_array_nan_raises(self) -> None:
        """from_array() raises ValueError if the array contains NaN."""
        arr = ForceParams.defaults().to_array()
        arr[0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            ForceParams.from_array(arr)

    def test_from_array_inf_raises(self) -> None:
        """from_array() raises ValueError if the array contains inf."""
        arr = ForceParams.defaults().to_array()
        arr[3] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            ForceParams.from_array(arr)

    def test_from_array_invalid_physical_value_raises(self) -> None:
        """from_array() raises ValueError if a field violates physical constraints."""
        arr = ForceParams.defaults().to_array()
        # Set relaxation_time (index 0) to zero — must fail validation
        arr[0] = 0.0
        with pytest.raises(ValueError, match="relaxation_time"):
            ForceParams.from_array(arr)


# ---------------------------------------------------------------------------
# compose() bit-identity with and without explicit params
# ---------------------------------------------------------------------------


class TestComposeBitIdentical:
    """compose() with ForceParams.defaults() is bit-identical to no-params."""

    def test_default_params_produces_identical_output(
        self,
        agent_state: AgentState,
        flow_field: FlowField,
        empty_panic: PanicField,
    ) -> None:
        """compose(..., params=ForceParams.defaults()) == compose(...) element-wise."""
        baseline = compose(agent_state, flow_field, empty_panic)
        with_defaults = compose(
            agent_state,
            flow_field,
            empty_panic,
            params=ForceParams.defaults(),
        )
        np.testing.assert_array_equal(baseline, with_defaults)

    def test_custom_relaxation_time_changes_output(
        self,
        agent_state: AgentState,
        flow_field: FlowField,
        empty_panic: PanicField,
    ) -> None:
        """Halving relaxation_time changes the compose output for all active agents."""
        baseline = compose(agent_state, flow_field, empty_panic)
        fast = compose(
            agent_state,
            flow_field,
            empty_panic,
            params=ForceParams(relaxation_time=RELAXATION_TIME * 0.5),
        )
        # f_exit scales by 1/relaxation_time — halving it doubles exit force.
        # Any active agent pointing toward an exit will have a different row.
        assert not np.array_equal(baseline, fast), (
            "Halved relaxation_time should double exit force magnitude"
        )

    def test_zero_strengths_zeroes_crowd_terms(
        self,
        agent_state: AgentState,
        flow_field: FlowField,
        empty_panic: PanicField,
    ) -> None:
        """Setting all crowd-force strengths to zero removes their contribution."""
        no_crowd = compose(
            agent_state,
            flow_field,
            empty_panic,
            params=ForceParams(
                repulsion_strength=0.0,
                density_pressure_strength=0.0,
                herd_attraction_strength=0.0,
                panic_repulsion_strength=0.0,
            ),
        )
        exit_only = compose(
            agent_state,
            flow_field,
            empty_panic,
            enable_crowd=False,
            enable_density=False,
            enable_herd=False,
            enable_panic_repulsion=False,
        )
        np.testing.assert_allclose(no_crowd, exit_only, atol=1e-12)


# ---------------------------------------------------------------------------
# Simulation accepts ForceParams; default produces bit-identical ticks
# ---------------------------------------------------------------------------


class TestSimulationForceParams:
    """Simulation honours ForceParams; defaults reproduce pre-Phase-2 ticks."""

    def test_simulation_accepts_default_params(
        self, open_floor: FloorPlan, flow_field: FlowField
    ) -> None:
        """Simulation.__init__ succeeds with explicit ForceParams.defaults()."""
        sim = _build_sim(open_floor, flow_field, params=ForceParams.defaults())
        sim.step()  # must not raise

    def test_default_params_produces_bit_identical_sequence(
        self, open_floor: FloorPlan, flow_field: FlowField
    ) -> None:
        """A sim with params=ForceParams.defaults() equals one with no params arg.

        Both sims use the same seed and step the same number of ticks.
        Positions and velocities must be bit-for-bit identical, confirming
        the refactor is behaviour-preserving at floating-point precision.
        """
        sim_no_param = _build_sim(open_floor, flow_field)
        sim_with_param = _build_sim(
            open_floor, flow_field, params=ForceParams.defaults()
        )

        for _ in range(20):
            if sim_no_param.is_complete or sim_with_param.is_complete:
                break
            sim_no_param.step()
            sim_with_param.step()

        np.testing.assert_array_equal(
            sim_no_param.state.pos, sim_with_param.state.pos
        )
        np.testing.assert_array_equal(
            sim_no_param.state.vel, sim_with_param.state.vel
        )
        np.testing.assert_array_equal(
            sim_no_param.state.alive, sim_with_param.state.alive
        )

    def test_custom_params_affects_simulation_trajectory(
        self, open_floor: FloorPlan, flow_field: FlowField
    ) -> None:
        """Changing relaxation_time produces a different trajectory after a few ticks.

        relaxation_time affects the exit-seeking force for every active agent,
        so even a single step on a sparse population will diverge.
        """
        sim_default = _build_sim(open_floor, flow_field)
        sim_custom = _build_sim(
            open_floor,
            flow_field,
            params=ForceParams(relaxation_time=RELAXATION_TIME * 2.0),
        )

        for _ in range(10):
            if sim_default.is_complete or sim_custom.is_complete:
                break
            sim_default.step()
            sim_custom.step()

        assert not np.array_equal(
            sim_default.state.pos, sim_custom.state.pos
        ), "Doubled relaxation_time should produce a distinct agent trajectory"

    def test_params_stored_on_simulation(
        self, open_floor: FloorPlan, flow_field: FlowField
    ) -> None:
        """The params argument is accessible as sim._params after construction."""
        p = ForceParams(relaxation_time=0.8)
        rng = SeededRNG(_SEED)
        state = spawn(open_floor, _N_AGENTS, rng.generator)
        sim = Simulation(
            state=state,
            flow_field=flow_field,
            panic_field=PanicField(),
            exit_model=ExitModel(open_floor),
            rng=rng,
            params=p,
        )
        assert sim._params == p
