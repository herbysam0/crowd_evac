"""Shared helpers for the designer window/IO tests."""
from __future__ import annotations

from crowd_evac.designer.controller import DesignController, Tool


def ready_controller() -> DesignController:
    """Return a controller with a closed-enough room that validates."""
    ctl = DesignController()
    ctl.update_room(name="saved_room")
    ctl.set_tool(Tool.WALL)
    ctl.press(0.0, 0.0)
    ctl.release(15.0, 0.0)
    ctl.set_tool(Tool.DOORWAY)
    ctl.press(7.5, 0.2)
    return ctl
