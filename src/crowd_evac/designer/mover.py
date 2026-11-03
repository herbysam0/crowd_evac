"""Drag-to-move for the selected entity.

Each pointer step is applied incrementally to the model. A step the model
rejects is retried one axis at a time, so an entity slides along a boundary
instead of freezing, and it catches up once the pointer is valid again.
"""
from __future__ import annotations

from crowd_evac.designer.model import DesignModel
from crowd_evac.designer.placement import Selection
from crowd_evac.domain.errors import DesignValidationError

MOVE_DECIMALS = 6


class MoveDrag:
    """State of one move gesture.

    Attributes:
        applied: Total ``(dx, dy)`` currently applied to the entity.
    """

    def __init__(
        self, model: DesignModel, sel: Selection, ax: float, ay: float
    ) -> None:
        """Start a move.

        Args:
            model: Design to edit.
            sel: Entity being moved.
            ax: Snapped pointer x at press.
            ay: Snapped pointer y at press.
        """
        self._model = model
        self._sel = sel
        self._anchor = (ax, ay)
        self.applied = (0.0, 0.0)

    @property
    def moved(self) -> bool:
        """Whether the entity has moved from where it started."""
        return self.applied != (0.0, 0.0)

    def update(self, px: float, py: float) -> str | None:
        """Move toward the pointer.

        Args:
            px: Snapped pointer x.
            py: Snapped pointer y.

        Returns:
            None on success or no-op; the rejection reason if no variant of
            the step was valid.
        """
        sx = round(px - self._anchor[0] - self.applied[0], MOVE_DECIMALS)
        sy = round(py - self._anchor[1] - self.applied[1], MOVE_DECIMALS)
        if sx == 0.0 and sy == 0.0:
            return None
        error = ""
        for dx, dy in ((sx, sy), (sx, 0.0), (0.0, sy)):
            if dx == 0.0 and dy == 0.0:
                continue
            try:
                self._apply(dx, dy)
            except DesignValidationError as exc:
                error = str(exc)
                continue
            self.applied = (
                round(self.applied[0] + dx, MOVE_DECIMALS),
                round(self.applied[1] + dy, MOVE_DECIMALS),
            )
            return None
        return error

    def _apply(self, dx: float, dy: float) -> None:
        """Apply one translation to the selected entity."""
        sel, model = self._sel, self._model
        if sel.kind == "wall":
            model.move_wall(sel.id, dx, dy)
        elif sel.kind == "obstacle":
            model.move_obstacle(sel.id, dx, dy)
        else:
            wall = model.layout.walls[model.layout.doorways[sel.id].wall_id]
            model.move_doorway(sel.id, dx if wall.horizontal else dy)
