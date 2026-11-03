"""Numeric fields of the selected entity: reading and applying edits."""
from __future__ import annotations

from crowd_evac.designer.model import DesignModel
from crowd_evac.designer.placement import Selection
from crowd_evac.domain.errors import DesignValidationError


def selected_values(
    model: DesignModel, sel: Selection | None
) -> dict[str, float]:
    """Return the editable numeric fields of ``sel`` (empty if None).

    Wall X/Y is its start point, obstacle X/Y its bottom-left corner and
    doorway X/Y its centre.
    """
    if sel is None:
        return {}
    layout = model.layout
    if sel.kind == "wall":
        w = layout.walls[sel.id]
        return {
            "x": w.x0,
            "y": w.y0,
            "length": w.length,
            "thickness": w.thickness,
        }
    if sel.kind == "obstacle":
        o = layout.obstacles[sel.id]
        return {"x": o.x, "y": o.y, "width": o.width, "height": o.height}
    d = layout.doorways[sel.id]
    wall = layout.walls[d.wall_id]
    x, y = (d.center, wall.y0) if wall.horizontal else (wall.x0, d.center)
    return {
        "x": x,
        "y": y,
        "width_m": d.width_m,
        "capacity_per_second": float(d.capacity_per_second),
    }


def apply_field(
    model: DesignModel, sel: Selection, field: str, value: float
) -> None:
    """Apply one numeric edit to the selected entity.

    Raises:
        DesignValidationError: If the model rejects the change.
    """
    if sel.kind == "wall":
        _edit_wall(model, sel.id, field, value)
    elif sel.kind == "obstacle":
        o = model.layout.obstacles[sel.id]
        vals = {"x": o.x, "y": o.y, "width": o.width, "height": o.height}
        vals[field] = value
        model.edit_obstacle(sel.id, **vals)
    else:
        _edit_doorway(model, sel.id, field, value)


def _edit_wall(
    model: DesignModel, wall_id: int, field: str, value: float
) -> None:
    """Edit a wall's start point (moves it), length or thickness."""
    w = model.layout.walls[wall_id]
    if field == "length":
        model.set_wall_length(wall_id, value)
    elif field == "thickness":
        model.set_wall_thickness(wall_id, value)
    elif field == "x":
        model.move_wall(wall_id, value - w.x0, 0.0)
    else:
        model.move_wall(wall_id, 0.0, value - w.y0)


def _edit_doorway(
    model: DesignModel, door_id: int, field: str, value: float
) -> None:
    """Edit a doorway; only its along-wall coordinate can change."""
    d = model.layout.doorways[door_id]
    wall = model.layout.walls[d.wall_id]
    centre, width, capacity = d.center, d.width_m, d.capacity_per_second
    if field in ("x", "y"):
        if (field == "x") != wall.horizontal:
            raise DesignValidationError(
                "A doorway can only slide along its wall"
            )
        centre = value
    elif field == "width_m":
        width = value
    else:
        capacity = int(value)
    model.edit_doorway(door_id, centre, width, capacity)
