"""Tests for crowd_evac.domain.door_geometry — door frames, hosts, depth.

Covers:
  - door_frame: axis from side label, outward sign from geometry, span
  - is_host: perimeter wall, wall behind a floating door, interior walls
  - egress_reach: to the boundary through a carved host, zero at a
    boundary door, limited by a blocker on the centre line
"""
from __future__ import annotations

import pytest

from crowd_evac.domain.door_geometry import (
    DoorFrame,
    ExitSide,
    door_frame,
    egress_reach,
    is_host,
)
from crowd_evac.domain.floor_plan import Wall


class TestDoorFrame:
    """door_frame derives axis, outward sign and lateral span."""

    @pytest.mark.parametrize(
        "x, y, side, normal",
        [
            (1.0, 8.2, ExitSide.WEST, (-1.0, 0.0)),
            (15.2, 8.2, ExitSide.EAST, (1.0, 0.0)),
            (8.0, 0.5, ExitSide.SOUTH, (0.0, -1.0)),
            (8.0, 9.5, ExitSide.NORTH, (0.0, 1.0)),
        ],
        ids=["west", "east", "bottom", "top"],
    )
    def test_normal_points_to_nearer_boundary(
        self, x: float, y: float, side: ExitSide, normal: tuple[float, float]
    ) -> None:
        """Outward normal follows geometry on all four sides."""
        assert door_frame(x, y, 1.0, side, 16.2, 10.0).normal == normal

    def test_mislabelled_side_uses_geometry(self) -> None:
        """lecture_hall labels its y = 0.5 door 'north'; outward is still -y."""
        frame = door_frame(25.0, 0.5, 3.0, ExitSide.NORTH, 50.0, 30.0)
        assert frame.normal == (0.0, -1.0)

    def test_midline_tie_uses_label(self) -> None:
        """Exactly mid-room, the label breaks the tie."""
        assert door_frame(5.0, 5.0, 1.0, ExitSide.EAST, 10.0, 10.0).sign == 1.0
        assert door_frame(5.0, 5.0, 1.0, ExitSide.WEST, 10.0, 10.0).sign == -1.0

    def test_lateral_span_is_authored_width(self) -> None:
        """Span is centre ± width / 2 along the opening axis."""
        frame = door_frame(1.0, 8.2, 1.6, ExitSide.WEST, 16.2, 10.0)
        assert frame.lo == pytest.approx(7.4)
        assert frame.hi == pytest.approx(9.0)
        assert frame.lateral_axis == 1


@pytest.fixture
def west_door() -> DoorFrame:
    """room_model_3 west door: plane x = 1.0, span y 7.4-9.0, outward -x."""
    return door_frame(1.0, 8.2, 1.6, ExitSide.WEST, 16.2, 10.0)


class TestIsHost:
    """is_host identifies the wall the opening is cut through."""

    def test_perimeter_wall_touching_plane_is_host(
        self, west_door: DoorFrame
    ) -> None:
        """wall_4 (x 0-1) contains the door centre on its face."""
        assert is_host(west_door, Wall(0.0, 0.0, 1.0, 10.0))

    def test_wall_behind_floating_door_is_host(self) -> None:
        """lecture_hall: door at x = 1.0, west wall x 0-0.5 (0.5 m gap)."""
        frame = door_frame(1.0, 5.0, 1.5, ExitSide.WEST, 50.0, 30.0)
        assert is_host(frame, Wall(0.0, 0.0, 0.5, 30.0))

    @pytest.mark.parametrize(
        "wall",
        [
            Wall(1.0, 7.3, 3.7, 0.1, "wall_5"),  # perpendicular, touches span
            Wall(0.0, 9.0, 16.2, 1.0, "wall_1"),  # perpendicular corner wall
            Wall(3.0, 0.0, 0.1, 10.0, "inside"),  # parallel, inside room
            Wall(-3.0, 0.0, 0.5, 10.0, "far"),  # parallel, > max gap behind
            Wall(0.0, 0.0, 1.0, 5.0, "short"),  # does not cover the centre
        ],
        ids=["wall_5", "corner", "inside", "too-far", "lateral-miss"],
    )
    def test_non_host_walls(self, west_door: DoorFrame, wall: Wall) -> None:
        """Interior, perpendicular, distant and laterally-missing walls."""
        assert not is_host(west_door, wall)


class TestEgressReach:
    """egress_reach is the free depth behind the door plane."""

    def test_reach_to_boundary_through_carved_host(
        self, west_door: DoorFrame
    ) -> None:
        """Host wall_4 is not a blocker: 1 m to the x = 0 boundary."""
        assert egress_reach(west_door, (), 16.2, 10.0) == pytest.approx(1.0)

    def test_boundary_door_reach_is_zero(self) -> None:
        """Door on the room boundary has nothing behind it."""
        frame = door_frame(5.0, 0.0, 2.0, ExitSide.SOUTH, 10.0, 10.0)
        assert egress_reach(frame, (), 10.0, 10.0) == 0.0

    def test_blocker_behind_door_limits_reach(
        self, west_door: DoorFrame
    ) -> None:
        """A rect whose near face is 0.4 m behind the plane caps the reach."""
        blocker = Wall(0.0, 7.0, 0.6, 3.0)
        assert egress_reach(west_door, (blocker,), 16.2, 10.0) == pytest.approx(
            0.4
        )

    def test_blocker_off_centre_line_ignored(
        self, west_door: DoorFrame
    ) -> None:
        """A rect behind the door but beside its centre line is ignored."""
        blocker = Wall(0.0, 9.0, 0.6, 1.0)
        assert egress_reach(west_door, (blocker,), 16.2, 10.0) == pytest.approx(
            1.0
        )
