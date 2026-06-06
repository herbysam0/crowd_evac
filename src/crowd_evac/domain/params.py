"""Injectable value object for all tunable behavioural force weights (Phase 2).

:class:`ForceParams` is a frozen dataclass that collects the decision
variables the Phase-2 optimiser varies.  Every tunable constant from
``domain.constants`` becomes an injectable field here; the module constants
remain the authoritative source for the defaults returned by
:meth:`ForceParams.defaults`.

The ordered vector representation (:meth:`to_array` / :meth:`from_array`)
is used by the optimiser search space (``optimization.space``) and the
evaluation harness (``optimization.harness``).  The canonical field ordering
matches the decision-variable table in ``docs/plan_phase_2.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.constants import (
    DENSITY_PRESSURE_STRENGTH,
    DENSITY_SENSING_RADIUS,
    HAZARD_AVOIDANCE_COST,
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

# Canonical field ordering for to_array / from_array.  Matches the
# decision-variable table in docs/plan_phase_2.md §Decision variables.
_FIELD_ORDER: tuple[str, ...] = (
    "relaxation_time",
    "panic_speed_multiplier",
    "repulsion_strength",
    "repulsion_radius",
    "high_density_threshold",
    "density_pressure_strength",
    "density_sensing_radius",
    "herd_attraction_strength",
    "herd_perception_radius",
    "panic_repulsion_strength",
    "max_accel",
    "max_speed",
    "hazard_avoidance_cost",
)

N_PARAMS: int = len(_FIELD_ORDER)
"""Number of optimisable parameters represented in :class:`ForceParams`."""


@dataclass(frozen=True)
class ForceParams:
    """Frozen value object holding all tunable behavioural force weights.

    Fields mirror the decision variables in ``docs/plan_phase_2.md``
    §Decision variables.  Defaults are the Phase-1 hand-tuned constants from
    ``domain.constants``; those constants are the single source of truth for
    :meth:`defaults`.

    Physical invariants (positivity, non-negativity) are checked in
    ``__post_init__``; a violated constraint raises :exc:`ValueError`
    immediately.  Optimiser search bounds are stored separately in
    ``optimization.space`` and are deliberately wider than these physical
    constraints.

    Attributes:
        relaxation_time: Characteristic time (s) for velocity relaxation in
            ``f_exit``. Must be > 0.
        panic_speed_multiplier: Speed boost factor at full panic in ``f_exit``
            and the integrator speed cap. Must be >= 1.0.
        repulsion_strength: Magnitude scale for short-range agent-agent
            repulsion in ``f_crowd``. Must be >= 0.
        repulsion_radius: Cut-off radius (m) for agent-agent repulsion in
            ``f_crowd``. Must be > 0.
        high_density_threshold: Density (agents/m²) above which ``f_density``
            applies drag. Must be >= 0.
        density_pressure_strength: Drag scale for ``f_density``. Must be >= 0.
        density_sensing_radius: Radius (m) for local density estimation in
            ``f_density``. Must be > 0.
        herd_attraction_strength: Alignment scale for ``f_herd``. Must be >= 0.
        herd_perception_radius: Radius (m) for herd-alignment perception in
            ``f_herd``. Must be > 0.
        panic_repulsion_strength: Repulsion scale for ``f_panic_repulsion``.
            Must be >= 0.
        max_accel: Maximum agent acceleration magnitude (m/s²) clamped in the
            integrator. Must be > 0.
        max_speed: Base maximum agent speed (m/s) used in ``f_exit`` desired
            velocity and the integrator speed cap. Must be > 0.
        hazard_avoidance_cost: Strength of hazard route-avoidance in the
            flow-field solve (the danger-cost multiplier). ``0`` routes by
            distance only; larger values divert the crowd to the next-best exit
            around a hazard. Must be >= 0. Unlike the force-term weights this
            shapes navigation (the flow field), not the additive accelerations.
    """

    relaxation_time: float = RELAXATION_TIME
    panic_speed_multiplier: float = PANIC_SPEED_MULTIPLIER
    repulsion_strength: float = REPULSION_STRENGTH
    repulsion_radius: float = REPULSION_RADIUS
    high_density_threshold: float = HIGH_DENSITY_THRESHOLD
    density_pressure_strength: float = DENSITY_PRESSURE_STRENGTH
    density_sensing_radius: float = DENSITY_SENSING_RADIUS
    herd_attraction_strength: float = HERD_ATTRACTION_STRENGTH
    herd_perception_radius: float = HERD_PERCEPTION_RADIUS
    panic_repulsion_strength: float = PANIC_REPULSION_STRENGTH
    max_accel: float = MAX_ACCEL
    max_speed: float = MAX_SPEED
    hazard_avoidance_cost: float = HAZARD_AVOIDANCE_COST

    def __post_init__(self) -> None:
        """Validate all physical constraints; raise ValueError on violation."""
        _check_positive("relaxation_time", self.relaxation_time)
        _check_ge("panic_speed_multiplier", self.panic_speed_multiplier, 1.0)
        _check_non_negative("repulsion_strength", self.repulsion_strength)
        _check_positive("repulsion_radius", self.repulsion_radius)
        _check_non_negative("high_density_threshold", self.high_density_threshold)
        _check_non_negative(
            "density_pressure_strength", self.density_pressure_strength
        )
        _check_positive("density_sensing_radius", self.density_sensing_radius)
        _check_non_negative("herd_attraction_strength", self.herd_attraction_strength)
        _check_positive("herd_perception_radius", self.herd_perception_radius)
        _check_non_negative(
            "panic_repulsion_strength", self.panic_repulsion_strength
        )
        _check_positive("max_accel", self.max_accel)
        _check_positive("max_speed", self.max_speed)
        _check_non_negative("hazard_avoidance_cost", self.hazard_avoidance_cost)

    @classmethod
    def defaults(cls) -> ForceParams:
        """Return a ForceParams with all Phase-1 constant values.

        Equivalent to ``ForceParams()``; provided as a stable named entry-point
        so callers do not depend on the bare constructor default order.

        Returns:
            A ``ForceParams`` instance with every field set to its
            ``domain.constants`` default value.
        """
        return cls()

    def to_array(self) -> npt.NDArray[np.float64]:
        """Serialise the parameter vector in canonical field order.

        The field order matches ``_FIELD_ORDER`` and mirrors the search-space
        bounds in ``optimization.space``.  Use :meth:`from_array` to invert.

        Returns:
            Float64 array of shape ``(N_PARAMS,)`` containing all field values.

        Example:
            >>> p = ForceParams.defaults()
            >>> arr = p.to_array()
            >>> arr.shape
            (13,)
            >>> arr.dtype
            dtype('float64')
        """
        return np.array(
            [getattr(self, f) for f in _FIELD_ORDER],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> ForceParams:
        """Deserialise a ForceParams from a flat float array.

        The array must follow the canonical order produced by :meth:`to_array`.
        Physical validation is applied via the dataclass constructor.

        Args:
            arr: Float64 array of shape ``(N_PARAMS,)`` with field values in
                canonical order.  All values must be finite.

        Returns:
            A ``ForceParams`` instance built from the array values.

        Raises:
            ValueError: If ``arr`` does not have exactly ``N_PARAMS`` elements,
                if any value is non-finite, or if any reconstructed field
                fails physical validation (e.g. negative radius).

        Example:
            >>> import numpy as np
            >>> p = ForceParams.defaults()
            >>> ForceParams.from_array(p.to_array()) == p
            True
        """
        if arr.shape != (N_PARAMS,):
            raise ValueError(
                f"arr must have shape ({N_PARAMS},), got {arr.shape!r}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("arr must contain only finite values, got NaN or inf")
        kwargs = {f: float(arr[i]) for i, f in enumerate(_FIELD_ORDER)}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------


def _check_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _check_non_negative(name: str, value: float) -> None:
    """Raise ValueError if value is negative."""
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _check_ge(name: str, value: float, minimum: float) -> None:
    """Raise ValueError if value is strictly less than minimum."""
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
