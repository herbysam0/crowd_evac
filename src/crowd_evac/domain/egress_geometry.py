"""Vectorised door-line geometry for the gated egress model (issue #17).

Each exit's door line is a plane with an outward unit normal and a lateral
span. Three questions are answered here, all vectorised over agents x exits:

- **Signed outward distance** of every agent centre from every door plane
  (negative inside the room).
- **Band membership**: whether an agent's body overlaps the opening
  laterally (centre within the span widened by one radius).
- **Gating**: clamp agents that have not been released by the door back to
  the line (body flush against it, centre one radius inside), so nobody
  passes a door faster than its capacity allows.

Per exit it also fixes where a released agent is placed (centre one radius
past the line, laterally inside the span) and how far past the line it must
walk to count as evacuated (outer end of the gap / apron). A door with no
room behind it for a whole body (a boundary door with no apron) cannot take a
placed agent; release there means immediate evacuation.

Pure NumPy — no framework or I/O imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crowd_evac.domain.constants import AGENT_RADIUS
from crowd_evac.domain.door_geometry import (
    DOOR_CLEAR_MARGIN_M,
    RectLike,
    egress_reach,
)
from crowd_evac.domain.floor_grid import exit_frame, host_flags
from crowd_evac.domain.floor_plan import FloorPlan

Float1D = npt.NDArray[np.float64]
Bool1D = npt.NDArray[np.bool_]
Vec2Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EgressGeometry:
    """Per-exit door lines packed as arrays. Build via :meth:`from_floor_plan`.

    Indices follow :attr:`FloorPlan.exits`.

    Attributes:
        radius: Agent body radius in metres.
        points: Door-plane points (exit centres), shape ``(n, 2)``.
        normals: Outward unit normals, shape ``(n, 2)``.
        lateral_axis: Coordinate index running along each opening, ``(n,)``.
        band_lo: Span lower bound minus the radius, ``(n,)``.
        band_hi: Span upper bound plus the radius, ``(n,)``.
        place_lo: Lowest lateral centre for a placed agent, ``(n,)``.
        place_hi: Highest lateral centre for a placed agent, ``(n,)``.
        place_after: ``True`` where a released agent fits past the line;
            ``False`` means release evacuates immediately, ``(n,)``.
        remove_depth: Outward distance a released agent must reach to be
            evacuated (outer end of the egress path), ``(n,)``.
    """

    radius: float
    points: Vec2Array
    normals: Vec2Array
    lateral_axis: npt.NDArray[np.intp]
    band_lo: Float1D
    band_hi: Float1D
    place_lo: Float1D
    place_hi: Float1D
    place_after: Bool1D
    remove_depth: Float1D

    @classmethod
    def from_floor_plan(
        cls, floor_plan: FloorPlan, radius: float = AGENT_RADIUS
    ) -> EgressGeometry:
        """Derive door lines, placement and evacuation depths for every exit.

        Args:
            floor_plan: Floor plan whose exits are evacuated through.
            radius: Agent body radius in metres. Must be positive.

        Returns:
            Packed per-exit geometry.

        Raises:
            ValueError: If *radius* is not positive.
        """
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius!r}")
        n = len(floor_plan.exits)
        cols = {k: np.empty(n, dtype=np.float64) for k in (
            "band_lo", "band_hi", "place_lo", "place_hi", "remove_depth"
        )}
        points = np.empty((n, 2), dtype=np.float64)
        normals = np.empty((n, 2), dtype=np.float64)
        lateral = np.empty(n, dtype=np.intp)
        place_after = np.empty(n, dtype=np.bool_)
        flags = host_flags(floor_plan)
        for i, exit_ in enumerate(floor_plan.exits):
            frame = exit_frame(floor_plan, exit_)
            blockers: list[RectLike] = [
                w for w, host in zip(floor_plan.walls, flags[i]) if not host
            ]
            blockers.extend(floor_plan.obstacles)
            reach = egress_reach(
                frame, blockers, floor_plan.width_m, floor_plan.height_m
            )
            # Deepest centre position the wall push-out allows, minus slack.
            deepest = reach - radius - DOOR_CLEAR_MARGIN_M
            points[i] = (frame.px, frame.py)
            normals[i] = frame.normal
            lateral[i] = frame.lateral_axis
            cols["band_lo"][i] = frame.lo - radius
            cols["band_hi"][i] = frame.hi + radius
            mid = (frame.lo + frame.hi) / 2.0
            cols["place_lo"][i] = min(frame.lo + radius, mid)
            cols["place_hi"][i] = max(frame.hi - radius, mid)
            place_after[i] = deepest >= radius
            cols["remove_depth"][i] = deepest
        return cls(
            radius, points, normals, lateral, cols["band_lo"],
            cols["band_hi"], cols["place_lo"], cols["place_hi"], place_after,
            cols["remove_depth"],
        )


def outward_distances(pos: Vec2Array, geometry: EgressGeometry) -> Vec2Array:
    """Signed distance of each centre past each door plane, shape ``(A, n)``.

    Example:
        >>> from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan
        >>> fp = FloorPlan(10.0, 10.0, (), (),
        ...                (Exit(2.0, 5.0, 1.0, ExitSide.WEST, 1.0),))
        >>> g = EgressGeometry.from_floor_plan(fp, radius=0.25)
        >>> outward_distances(np.array([[1.75, 5.0], [3.0, 5.0]]), g).ravel()
        array([ 0.25, -1.  ])
    """
    offset = pos[:, np.newaxis, :] - geometry.points[np.newaxis, :, :]
    result: Vec2Array = np.einsum("aek,ek->ae", offset, geometry.normals)
    return result


def in_band(pos: Vec2Array, geometry: EgressGeometry) -> npt.NDArray[np.bool_]:
    """Whether each body overlaps each opening laterally, shape ``(A, n)``."""
    lateral = pos[:, geometry.lateral_axis]
    result: npt.NDArray[np.bool_] = (
        (lateral >= geometry.band_lo[np.newaxis, :])
        & (lateral <= geometry.band_hi[np.newaxis, :])
    )
    return result


def gate(
    pos: Vec2Array,
    vel: Vec2Array,
    released: Bool1D,
    geometry: EgressGeometry,
) -> npt.NDArray[np.intp]:
    """Hold unreleased agents at door lines; detect agents already past them.

    An unreleased agent in an opening's band whose centre is less than one
    radius past the plane (its body straddles the line, or is about to) is
    moved back along the normal to exactly one radius inside, and its outward
    velocity component is zeroed. An unreleased agent already a full radius
    past a line (e.g. spawned in the gap) is reported so the caller can treat
    it as released.

    Args:
        pos: Agent centres ``(A, 2)``; modified in place.
        vel: Agent velocities ``(A, 2)``; modified in place.
        released: ``(A,)`` flags; released agents are never gated.
        geometry: Packed exit geometry.

    Returns:
        ``(A,)`` exit index for each agent already past a line, else ``-1``.
    """
    r = geometry.radius
    s = outward_distances(pos, geometry)
    band = in_band(pos, geometry) & ~released[:, np.newaxis]
    straddle = band & (s > -r) & (s < r)
    for e in range(geometry.points.shape[0]):
        rows = np.flatnonzero(straddle[:, e])
        if rows.size == 0:
            continue
        n = geometry.normals[e]
        pos[rows] -= (s[rows, e] + r)[:, np.newaxis] * n
        v_out = vel[rows] @ n
        vel[rows] -= np.maximum(v_out, 0.0)[:, np.newaxis] * n
        straddle[rows, :] = False  # one door clamps an agent per tick
    past = band & (s >= r)
    return np.where(past.any(axis=1), past.argmax(axis=1), -1).astype(np.intp)


def placement(
    pos: Vec2Array, exit_idx: int, geometry: EgressGeometry
) -> Vec2Array:
    """Return centres one radius past door *exit_idx*, lateral kept in span.

    Args:
        pos: Current centres of the released agents, shape ``(k, 2)``.
        exit_idx: Door index.
        geometry: Packed exit geometry.

    Returns:
        New centres, shape ``(k, 2)``.
    """
    out = pos.copy()
    axis = int(geometry.lateral_axis[exit_idx])
    normal_axis = 1 - axis
    point = geometry.points[exit_idx]
    sign = geometry.normals[exit_idx, normal_axis]
    out[:, normal_axis] = point[normal_axis] + sign * geometry.radius
    out[:, axis] = np.clip(
        out[:, axis], geometry.place_lo[exit_idx], geometry.place_hi[exit_idx]
    )
    return out
