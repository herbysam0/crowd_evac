"""Headless room-design model: add, edit and delete walls, doorways, obstacles.

Every mutation builds a candidate :class:`Layout`, validates it in full and
only then commits, so a rejected change leaves the model untouched.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from crowd_evac.designer import wall_ops
from crowd_evac.designer.geometry import (
    DEFAULT_WALL_THICKNESS_M,
    Rect,
    Side,
)
from crowd_evac.designer.layout import (
    Doorway,
    Layout,
    Obstacle,
    RoomSettings,
    Wall,
    rect_of_doorway,
    rect_of_wall,
    side_of_wall,
    validate,
)

DEFAULT_ROOM_WIDTH_M: float = 15.0
DEFAULT_ROOM_HEIGHT_M: float = 10.0
DEFAULT_DOOR_CAPACITY: float = 1.05


class DesignModel:
    """Mutable design state with validated edits.

    Attributes:
        layout: Current immutable snapshot (read-only for callers).
    """

    def __init__(
        self,
        room: RoomSettings | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Create an empty design, or wrap an existing layout.

        Args:
            room: Initial room settings; defaults to a 15 x 10 m room.
            layout: Existing layout (e.g. imported from a file); overrides
                ``room``.

        Raises:
            DesignValidationError: If ``room`` is invalid.
        """
        room = room or RoomSettings(
            "my_room",
            DEFAULT_ROOM_WIDTH_M,
            DEFAULT_ROOM_HEIGHT_M,
        )
        self.layout = layout or Layout(room, {}, {}, {})
        validate(self.layout)
        ids = [
            *self.layout.walls,
            *self.layout.obstacles,
            *self.layout.doorways,
        ]
        self._next_id = max(ids, default=0) + 1

    # -- Queries -----------------------------------------------------------

    def wall_rect(self, wall_id: int) -> Rect:
        """Return the drawn rectangle of a wall."""
        return rect_of_wall(self.layout, self.layout.walls[wall_id])

    def doorway_rect(self, door_id: int) -> Rect:
        """Return the footprint rectangle of a doorway."""
        return rect_of_doorway(self.layout, self.layout.doorways[door_id])

    def side_of(self, wall_id: int) -> Side | None:
        """Return the perimeter side of a wall, or None if interior."""
        return side_of_wall(self.layout, self.layout.walls[wall_id])

    # -- Room --------------------------------------------------------------

    def update_room(self, **changes: Any) -> None:
        """Change room fields (name, width_m, height_m).

        Args:
            **changes: Fields of :class:`RoomSettings` to replace.

        Raises:
            DesignValidationError: If an entity would leave the room or a
                value is invalid.
        """
        self.commit(room=replace(self.layout.room, **changes))

    # -- Walls -------------------------------------------------------------

    def add_wall(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        thickness: float = DEFAULT_WALL_THICKNESS_M,
    ) -> Wall:
        """Add a horizontal or vertical wall; return it.

        Raises:
            DesignValidationError: On any violated wall rule.
        """
        wall = Wall(self._new_id(), x0, y0, x1, y1, thickness)
        self.commit(walls={**self.layout.walls, wall.id: wall})
        return wall

    def edit_wall(
        self, wall_id: int, x0: float, y0: float, x1: float, y1: float
    ) -> Wall:
        """Replace a wall's endpoints.

        Raises:
            DesignValidationError: If invalid, or an attached doorway would
                no longer fit or would change side.
        """
        thickness = self.layout.walls[wall_id].thickness
        new = Wall(wall_id, x0, y0, x1, y1, thickness)
        walls = {**self.layout.walls, wall_id: new}
        wall_ops.require_same_side(self, wall_id, walls)
        self.commit(walls=walls)
        return new

    def move_wall(self, wall_id: int, dx: float, dy: float) -> Wall:
        """Translate a wall; its doorways slide along (see ``wall_ops``)."""
        return wall_ops.move_wall(self, wall_id, dx, dy)

    def set_wall_length(self, wall_id: int, length: float) -> Wall:
        """Set a wall's length, keeping its start fixed."""
        return wall_ops.set_wall_length(self, wall_id, length)

    def set_wall_thickness(self, wall_id: int, thickness: float) -> Wall:
        """Change a wall's thickness about its centreline."""
        return wall_ops.set_wall_thickness(self, wall_id, thickness)

    def delete_wall(self, wall_id: int) -> int:
        """Delete a wall and its doorways.

        Returns:
            Number of doorways removed with it.
        """
        if wall_id not in self.layout.walls:
            raise KeyError(wall_id)
        walls = {k: v for k, v in self.layout.walls.items() if k != wall_id}
        doors = {
            k: v
            for k, v in self.layout.doorways.items()
            if v.wall_id != wall_id
        }
        removed = len(self.layout.doorways) - len(doors)
        self.commit(walls=walls, doorways=doors)
        return removed

    # -- Doorways ----------------------------------------------------------

    def add_doorway(
        self,
        wall_id: int,
        center: float,
        width_m: float,
        capacity_per_second: float = DEFAULT_DOOR_CAPACITY,
    ) -> Doorway:
        """Add a doorway on a perimeter wall; return it.

        Raises:
            DesignValidationError: On any violated doorway rule.
        """
        door = Doorway(
            self._new_id(), wall_id, center, width_m, capacity_per_second
        )
        self.commit(doorways={**self.layout.doorways, door.id: door})
        return door

    def edit_doorway(
        self,
        door_id: int,
        center: float,
        width_m: float,
        capacity_per_second: float,
    ) -> Doorway:
        """Replace a doorway's centre, width and capacity."""
        new = replace(
            self.layout.doorways[door_id],
            center=center,
            width_m=width_m,
            capacity_per_second=capacity_per_second,
        )
        self.commit(doorways={**self.layout.doorways, door_id: new})
        return new

    def move_doorway(self, door_id: int, delta: float) -> Doorway:
        """Slide a doorway ``delta`` metres along its wall."""
        d = self.layout.doorways[door_id]
        return self.edit_doorway(
            door_id, wall_ops.rounded(d.center + delta), d.width_m, d.capacity_per_second
        )

    def delete_doorway(self, door_id: int) -> None:
        """Delete a doorway."""
        if door_id not in self.layout.doorways:
            raise KeyError(door_id)
        doors = {k: v for k, v in self.layout.doorways.items() if k != door_id}
        self.commit(doorways=doors)

    # -- Obstacles ---------------------------------------------------------

    def add_obstacle(
        self, x: float, y: float, width: float, height: float
    ) -> Obstacle:
        """Add a rectangular obstacle; return it.

        Raises:
            DesignValidationError: On any violated obstacle rule.
        """
        obs = Obstacle(self._new_id(), x, y, width, height)
        self.commit(obstacles={**self.layout.obstacles, obs.id: obs})
        return obs

    def edit_obstacle(
        self, obs_id: int, x: float, y: float, width: float, height: float
    ) -> Obstacle:
        """Replace an obstacle's rectangle."""
        if obs_id not in self.layout.obstacles:
            raise KeyError(obs_id)
        new = Obstacle(obs_id, x, y, width, height)
        self.commit(obstacles={**self.layout.obstacles, obs_id: new})
        return new

    def move_obstacle(self, obs_id: int, dx: float, dy: float) -> Obstacle:
        """Translate an obstacle."""
        o = self.layout.obstacles[obs_id]
        return self.edit_obstacle(
            obs_id, wall_ops.rounded(o.x + dx), wall_ops.rounded(o.y + dy), o.width, o.height
        )

    def delete_obstacle(self, obs_id: int) -> None:
        """Delete an obstacle."""
        if obs_id not in self.layout.obstacles:
            raise KeyError(obs_id)
        kept = {k: v for k, v in self.layout.obstacles.items() if k != obs_id}
        self.commit(obstacles=kept)

    # -- Internals ---------------------------------------------------------

    def _new_id(self) -> int:
        """Return a fresh id; consumed even if the add is rejected."""
        value = self._next_id
        self._next_id += 1
        return value

    def commit(self, **changes: Any) -> None:
        """Validate a candidate layout and adopt it on success."""
        candidate = replace(self.layout, **changes)
        validate(candidate)
        self.layout = candidate
