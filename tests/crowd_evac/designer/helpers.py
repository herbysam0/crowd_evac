"""Shared constants and helpers for the designer tests."""
from __future__ import annotations

from crowd_evac.designer.controller import DesignController, Tool
from crowd_evac.designer.layout import Layout
from crowd_evac.designer.model import DesignModel

ROOM_W = 15.0
ROOM_H = 10.0


def snapshot(model: DesignModel) -> Layout:
    """Return the current layout for before/after comparison."""
    return model.layout


def drag(
    ctl: DesignController,
    tool: Tool,
    a: tuple[float, float],
    b: tuple[float, float],
) -> None:
    """Press at ``a``, drag and release at ``b`` with ``tool``."""
    ctl.set_tool(tool)
    ctl.press(*a)
    ctl.drag(*b)
    ctl.release(*b)


def bottom_wall_with_door(ctl: DesignController) -> None:
    """Draw the bottom wall, then click a door at x=5."""
    drag(ctl, Tool.WALL, (0.0, 0.0), (ROOM_W, 0.0))
    ctl.set_tool(Tool.DOORWAY)
    ctl.press(5.0, 0.25)
