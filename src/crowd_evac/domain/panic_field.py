"""Scalar panic gradient field aggregating PanicSource contributions (FR-11 subset).

Aggregates one or more :class:`~crowd_evac.domain.panic_source.PanicSource`
instances into a continuous scalar field over the floor.  The field value at
world point ``p`` is the clamped sum of per-source contributions::

    V(p) = clamp( sum_s[ I_s * max(0, 1 − ||p − p_s|| / R_s) ], 0, 1 )

where ``I_s`` is source intensity and ``R_s`` is the influence radius.

The field gradient points *toward* each source (increasing intensity).
:meth:`PanicField.repulsion_at` returns the **negative** gradient direction
— i.e., away from each source — weighted by the local field value, giving
:func:`~crowd_evac.domain.forces.f_panic_repulsion` its down-gradient push.

Phase 1 uses straight-line (Euclidean) distance.  Full obstacle-aware field
propagation (line-of-sight blocking through walls) and wall reflection are
Phase 3 features.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.agent_state import Float1D, Vec2Array
from crowd_evac.domain.panic_source import PanicSource

logger = logging.getLogger(__name__)

_MIN_DIST: float = 1e-6
"""Distance floor preventing division-by-zero in direction normalisation."""


class PanicField:
    """Aggregate scalar panic field built from :class:`PanicSource` objects.

    Sources may be added and removed at runtime to support the FR-12.1
    injection API (step 1.14).  Only sources whose
    :attr:`~PanicSource.is_active` flag is ``True`` contribute to the field.

    Attributes:
        sources: Mutable list of :class:`PanicSource` instances.
    """

    def __init__(
        self,
        sources: Sequence[PanicSource] | None = None,
    ) -> None:
        """Initialise the field, optionally seeding it with existing sources.

        Args:
            sources: Initial panic sources.  Defaults to an empty list.
        """
        self.sources: list[PanicSource] = list(sources) if sources else []

    # ------------------------------------------------------------------
    # Runtime mutation helpers (FR-12.1 injection API in step 1.14)
    # ------------------------------------------------------------------

    def add_source(self, source: PanicSource) -> None:
        """Append a source to the field.

        Args:
            source: The :class:`PanicSource` to add.
        """
        self.sources.append(source)

    def remove_source(self, source: PanicSource) -> None:
        """Remove a source from the field.

        Args:
            source: The :class:`PanicSource` to remove.

        Raises:
            ValueError: If ``source`` is not present in :attr:`sources`.
        """
        self.sources.remove(source)

    def decay_all(self, dt: float) -> None:
        """Advance intensity decay on every source by ``dt`` seconds.

        Delegates to :meth:`~PanicSource.decay` on each source unchanged.
        Expired sources remain in the list but no longer contribute to the
        field; callers may prune them between ticks if desired.

        Args:
            dt: Elapsed simulation time in seconds.  Forwarded verbatim to
                each source's :meth:`~PanicSource.decay`.
        """
        for source in self.sources:
            source.decay(dt)

    # ------------------------------------------------------------------
    # Field evaluation
    # ------------------------------------------------------------------

    def value_at(self, positions: Vec2Array) -> Float1D:
        """Compute scalar field values at an array of world positions.

        Inactive sources (intensity at or below threshold) are skipped.
        The per-source contribution is::

            v_s(p) = I_s * max(0, 1 − d(p, p_s) / R_s)

        The total is the sum of all active contributions, clamped to
        ``[0, 1]``.

        Args:
            positions: World positions of shape ``(P, 2)`` in metres.

        Returns:
            Float64 array of shape ``(P,)`` with field values in
            ``[0, 1]``.  All-zero when no active sources exist or when all
            positions lie outside every source's influence radius.
        """
        pos = np.asarray(positions, dtype=np.float64)
        n: int = pos.shape[0]
        total: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)

        for source in self.sources:
            if not source.is_active:
                continue
            sp = np.array([source.x, source.y], dtype=np.float64)
            dist: npt.NDArray[np.float64] = np.linalg.norm(
                pos - sp[np.newaxis, :], axis=1
            )
            contrib: npt.NDArray[np.float64] = source.intensity * np.maximum(
                0.0, 1.0 - dist / source.radius
            )
            total += contrib

        clipped: Float1D = np.clip(total, 0.0, 1.0)
        return clipped

    def repulsion_at(self, positions: Vec2Array) -> Vec2Array:
        """Compute weighted down-gradient repulsion vectors at world positions.

        For each active source, the per-position contribution is::

            v_s(p) * (p − p_s) / max(||p − p_s||, ε)

        where ``v_s(p)`` is the scalar field contribution from source ``s``
        and the direction vector points *away* from the source (down the
        panic gradient).  Contributions from all active sources are summed.

        The output is **not** normalised.  The caller scales it by
        ``PANIC_REPULSION_STRENGTH`` (see
        :func:`~crowd_evac.domain.forces.f_panic_repulsion`).

        Args:
            positions: World positions of shape ``(P, 2)`` in metres.

        Returns:
            Float64 array of shape ``(P, 2)`` with summed away-from-source
            repulsion vectors.  Zero rows where no active source is within
            range.
        """
        pos = np.asarray(positions, dtype=np.float64)
        n: int = pos.shape[0]
        total: Vec2Array = np.zeros((n, 2), dtype=np.float64)

        for source in self.sources:
            if not source.is_active:
                continue
            sp = np.array([source.x, source.y], dtype=np.float64)
            delta: Vec2Array = pos - sp[np.newaxis, :]
            dist: npt.NDArray[np.float64] = np.linalg.norm(delta, axis=1)
            in_range: npt.NDArray[np.bool_] = dist < source.radius
            if not np.any(in_range):
                continue

            field_val: npt.NDArray[np.float64] = source.intensity * (
                1.0 - dist[in_range] / source.radius
            )
            d_clamped: npt.NDArray[np.float64] = np.maximum(
                dist[in_range], _MIN_DIST
            )
            direction: Vec2Array = delta[in_range] / d_clamped[:, np.newaxis]
            total[in_range] += field_val[:, np.newaxis] * direction

        return total
