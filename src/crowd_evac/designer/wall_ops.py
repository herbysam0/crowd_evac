"""Wall edits that need more than a plain replace: move, resize, thicken.

Functions operate on a :class:`~crowd_evac.designer.model.DesignModel` via its
public ``layout`` and ``commit``; every change is validated before adoption.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from crowd_evac.designer.layout import Wall, side_of_wall
from crowd_evac.domain.errors import DesignValidationError

if TYPE_CHECKING:
    from crowd_evac.designer.model import DesignModel

MOVE_DECIMALS = 6


def rounded(value: float) -> float:
    """Round away float noise from accumulated moves."""
    return round(value, MOVE_DECIMALS)


def require_same_side(
    model: DesignModel, wall_id: int, walls: dict[int, Wall]
) -> None:
    """Reject a wall change that moves attached doorways off their side."""
    layout = model.layout
    if not any(d.wall_id == wall_id for d in layout.doorways.values()):
        return
    probe = replace(layout, walls=walls)
    if side_of_wall(probe, walls[wall_id]) != model.side_of(wall_id):
        raise DesignValidationError(
            "Wall edit would move its doorways to another side"
        )


def move_wall(model: DesignModel, wall_id: int, dx: float, dy: float) -> Wall:
    """Translate a wall; its doorways slide along with it.

    Raises:
        DesignValidationError: If the result is invalid, e.g. the wall would
            leave the room or its doorways' boundary side.
    """
    w = model.layout.walls[wall_id]
    new = Wall(
        wall_id,
        rounded(w.x0 + dx), rounded(w.y0 + dy),
        rounded(w.x1 + dx), rounded(w.y1 + dy),
        w.thickness,
    )
    along = dx if w.horizontal else dy
    doors = {
        k: replace(d, center=rounded(d.center + along))
        if d.wall_id == wall_id
        else d
        for k, d in model.layout.doorways.items()
    }
    walls = {**model.layout.walls, wall_id: new}
    require_same_side(model, wall_id, walls)
    model.commit(walls=walls, doorways=doors)
    return new


def set_wall_length(model: DesignModel, wall_id: int, length: float) -> Wall:
    """Set a wall's length; the start stays fixed and the end moves.

    Raises:
        DesignValidationError: If ``length`` is not positive or the result
            is invalid.
    """
    w = model.layout.walls[wall_id]
    if length <= 0:
        raise DesignValidationError("Wall length must be positive")
    if w.horizontal:
        sign = 1.0 if w.x1 >= w.x0 else -1.0
        x1, y1 = w.x0 + sign * length, w.y1
    else:
        sign = 1.0 if w.y1 >= w.y0 else -1.0
        x1, y1 = w.x1, w.y0 + sign * length
    return model.edit_wall(wall_id, w.x0, w.y0, x1, y1)


def set_wall_thickness(
    model: DesignModel, wall_id: int, thickness: float
) -> Wall:
    """Change a wall's thickness about its centreline.

    Raises:
        DesignValidationError: If not positive, or the thicker wall leaves
            the room or overlaps an obstacle.
    """
    new = replace(model.layout.walls[wall_id], thickness=thickness)
    model.commit(walls={**model.layout.walls, wall_id: new})
    return new
