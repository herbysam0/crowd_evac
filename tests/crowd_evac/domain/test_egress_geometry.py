"""Tests for crowd_evac.domain.egress_geometry — door lines and gating.

Covers:
  - EgressGeometry.from_floor_plan: normals, placement flag, removal depth,
    span bounds, invalid radius
  - outward_distances / in_band on all four sides
  - gate: clamps straddlers flush to the line, zeroes outward velocity only,
    ignores released agents and agents beside the opening, reports agents
    already past the line
  - placement: one radius past the line, lateral clamped into the span
"""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.domain.door_geometry import DOOR_CLEAR_MARGIN_M
from crowd_evac.domain.egress_geometry import (
    EgressGeometry,
    gate,
    in_band,
    outward_distances,
    placement,
)
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan

RADIUS = 0.25  # exactly representable: plane ± R round-trips


def _floating(side: ExitSide, x: float, y: float) -> FloorPlan:
    """10 x 10 m wall-free room with one 1 m door away from the boundary."""
    return FloorPlan(10.0, 10.0, (), (), (Exit(x, y, 1.0, side, 1.0, "d"),))


# Door centre, outward unit normal, lateral unit vector per side.
SIDES = [
    (ExitSide.WEST, (2.0, 5.0), (-1.0, 0.0), (0.0, 1.0)),
    (ExitSide.EAST, (8.0, 5.0), (1.0, 0.0), (0.0, 1.0)),
    (ExitSide.SOUTH, (5.0, 2.0), (0.0, -1.0), (1.0, 0.0)),
    (ExitSide.NORTH, (5.0, 8.0), (0.0, 1.0), (1.0, 0.0)),
]
SIDE_IDS = ["west", "east", "south", "north"]


def _at(
    centre: tuple[float, float],
    n: tuple[float, float],
    lat: tuple[float, float],
    depth: float,
    lateral: float = 0.0,
) -> np.ndarray:
    """One position *depth* past the plane and *lateral* along the span."""
    return np.array([[
        centre[0] + n[0] * depth + lat[0] * lateral,
        centre[1] + n[1] * depth + lat[1] * lateral,
    ]])


class TestEgressGeometry:
    """Construction of the packed per-exit arrays."""

    def test_room_model_3_fields(self, room_model_3: FloorPlan) -> None:
        """1 m perimeter-wall doors take a placed agent; remove at gap end."""
        geom = EgressGeometry.from_floor_plan(room_model_3, 0.28)
        np.testing.assert_allclose(geom.normals, [[-1.0, 0.0], [1.0, 0.0]])
        assert geom.place_after.tolist() == [True, True]
        np.testing.assert_allclose(
            geom.remove_depth, [1.0 - 0.28 - DOOR_CLEAR_MARGIN_M] * 2
        )
        np.testing.assert_allclose(geom.place_lo, [7.68, 7.68])
        np.testing.assert_allclose(geom.place_hi, [8.72, 8.72])

    def test_boundary_door_cannot_place(self) -> None:
        """A door on the room edge has no room behind it: immediate removal."""
        fp = FloorPlan(
            10.0, 10.0, (), (), (Exit(5.0, 0.0, 2.0, ExitSide.SOUTH, 1.0),)
        )
        geom = EgressGeometry.from_floor_plan(fp, RADIUS)
        assert not geom.place_after[0]

    def test_narrow_door_places_on_centre(self) -> None:
        """A span narrower than a body places agents on its centre line."""
        fp = _floating(ExitSide.WEST, 2.0, 5.0)
        geom = EgressGeometry.from_floor_plan(fp, 0.6)
        assert geom.place_lo[0] == geom.place_hi[0] == 5.0

    def test_nonpositive_radius_raises(self, room_model_3: FloorPlan) -> None:
        """radius ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            EgressGeometry.from_floor_plan(room_model_3, 0.0)


class TestDistancesAndBand:
    """Signed outward distance and lateral band on every side."""

    @pytest.mark.parametrize("side, centre, n, lat", SIDES, ids=SIDE_IDS)
    def test_outward_distance_sign(
        self,
        side: ExitSide,
        centre: tuple[float, float],
        n: tuple[float, float],
        lat: tuple[float, float],
    ) -> None:
        """Positive past the plane, negative inside the room."""
        geom = EgressGeometry.from_floor_plan(_floating(side, *centre), RADIUS)
        assert outward_distances(_at(centre, n, lat, 0.5), geom)[0, 0] == 0.5
        assert outward_distances(_at(centre, n, lat, -1.0), geom)[0, 0] == -1.0

    @pytest.mark.parametrize("side, centre, n, lat", SIDES, ids=SIDE_IDS)
    @pytest.mark.parametrize(
        "lateral, expected",
        [(0.0, True), (0.5 + RADIUS, True), (0.5 + RADIUS + 0.01, False)],
        ids=["centre", "band-edge", "beside"],
    )
    def test_band_is_span_plus_radius(
        self,
        side: ExitSide,
        centre: tuple[float, float],
        n: tuple[float, float],
        lat: tuple[float, float],
        lateral: float,
        expected: bool,
    ) -> None:
        """Band covers bodies overlapping the 1 m opening laterally."""
        geom = EgressGeometry.from_floor_plan(_floating(side, *centre), RADIUS)
        pos = _at(centre, n, lat, 0.0, lateral)
        assert bool(in_band(pos, geom)[0, 0]) is expected


class TestGate:
    """Unreleased agents never straddle or pass a door line."""

    @pytest.mark.parametrize("side, centre, n, lat", SIDES, ids=SIDE_IDS)
    def test_straddler_clamped_flush_inside(
        self,
        side: ExitSide,
        centre: tuple[float, float],
        n: tuple[float, float],
        lat: tuple[float, float],
    ) -> None:
        """Centre 0.1 m past the line → moved to one radius inside."""
        geom = EgressGeometry.from_floor_plan(_floating(side, *centre), RADIUS)
        pos = _at(centre, n, lat, 0.1)
        vel = np.array([[n[0] + lat[0], n[1] + lat[1]]])
        past = gate(pos, vel, np.array([False]), geom)
        np.testing.assert_allclose(pos, _at(centre, n, lat, -RADIUS))
        np.testing.assert_allclose(vel, [lat])  # outward part removed
        assert past.tolist() == [-1]

    def test_inward_velocity_kept(self) -> None:
        """A straddler already moving back into the room keeps its velocity."""
        geom = EgressGeometry.from_floor_plan(
            _floating(ExitSide.WEST, 2.0, 5.0), RADIUS
        )
        pos = np.array([[1.9, 5.0]])
        vel = np.array([[0.4, 0.0]])
        gate(pos, vel, np.array([False]), geom)
        np.testing.assert_allclose(vel, [[0.4, 0.0]])

    @pytest.mark.parametrize(
        "x, y, released",
        [(1.9, 5.0, True), (1.9, 6.0, False), (2.5, 5.0, False)],
        ids=["released", "beside-span", "inside-room"],
    )
    def test_not_gated(self, x: float, y: float, released: bool) -> None:
        """Released, laterally-outside and well-inside agents are untouched."""
        geom = EgressGeometry.from_floor_plan(
            _floating(ExitSide.WEST, 2.0, 5.0), RADIUS
        )
        pos = np.array([[x, y]])
        gate(pos, np.zeros((1, 2)), np.array([released]), geom)
        np.testing.assert_allclose(pos, [[x, y]])

    def test_agent_fully_past_reported(self) -> None:
        """A body wholly past the line is reported with its door index."""
        geom = EgressGeometry.from_floor_plan(
            _floating(ExitSide.WEST, 2.0, 5.0), RADIUS
        )
        pos = np.array([[2.0 - RADIUS, 5.0]])
        assert gate(pos, np.zeros((1, 2)), np.array([False]), geom).tolist() == [0]
        np.testing.assert_allclose(pos, [[2.0 - RADIUS, 5.0]])


class TestPlacement:
    """Released agents are placed directly after the line."""

    def test_room_model_3_west_placement(self, room_model_3: FloorPlan) -> None:
        """Centre one radius past x = 1.0; lateral clamped to [7.68, 8.72]."""
        geom = EgressGeometry.from_floor_plan(room_model_3, 0.28)
        pos = np.array([[1.28, 8.0], [1.28, 7.2], [1.28, 9.2]])
        out = placement(pos, 0, geom)
        np.testing.assert_allclose(
            out, [[0.72, 8.0], [0.72, 7.68], [0.72, 8.72]]
        )
        np.testing.assert_allclose(pos[:, 0], 1.28)  # input not mutated
