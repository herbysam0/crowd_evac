"""File dialog, texture loading and drawing of the designer photo underlay."""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog

import arcade
from PIL import UnidentifiedImageError

from crowd_evac.adapters.render.viewport import Viewport
from crowd_evac.designer.controller import DesignController
from crowd_evac.designer.layout import RoomSettings
from crowd_evac.designer.underlay import (
    OPACITY_STEP,
    SUPPORTED_SUFFIXES,
    Underlay,
    UnderlayError,
    fit_rect,
    toggled,
    with_opacity,
)

MAX_ALPHA = 255
DIALOG_TITLE = "Select underlay photo"


def pick_photo() -> Path | None:
    """Ask the user for a photo file with the native dialog.

    Returns:
        Chosen path, or None if the dialog was cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    # Without topmost the dialog can open behind the arcade window.
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askopenfilename(
            title=DIALOG_TITLE,
            filetypes=[
                ("Images", " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES))
            ],
        )
    finally:
        root.destroy()
    return Path(chosen) if chosen else None


def load_underlay(path: Path) -> tuple[Underlay, arcade.Texture]:
    """Load a photo as an underlay and GPU texture.

    Args:
        path: Image file (png, jpg, jpeg or bmp).

    Returns:
        The underlay state and its texture.

    Raises:
        UnderlayError: If the type is unsupported or the file is unreadable.
    """
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnderlayError(f"unsupported file type '{path.suffix}'")
    try:
        texture = arcade.load_texture(path)
    except (OSError, UnidentifiedImageError) as exc:
        raise UnderlayError(f"cannot read image ({exc})") from exc
    return Underlay(path, texture.width, texture.height), texture


def draw_underlay(
    vp: Viewport,
    underlay: Underlay | None,
    texture: arcade.Texture | None,
    room: RoomSettings,
) -> None:
    """Draw the photo fitted into the room when it is visible."""
    if underlay is None or texture is None or not underlay.visible:
        return
    rect = fit_rect(underlay.px_w, underlay.px_h, room.width_m, room.height_m)
    left, right, bottom, top = vp.lrbt(rect)
    arcade.draw_texture_rect(
        texture,
        arcade.LRBT(left, right, bottom, top),
        alpha=round(underlay.opacity * MAX_ALPHA),
    )


class UnderlaySession:
    """Underlay state and actions for one designer window.

    Attributes:
        underlay: Current photo state, or None when nothing is loaded.
        texture: GPU texture for ``underlay``.
    """

    def __init__(
        self,
        ctl: DesignController,
        picker: Callable[[], Path | None] = pick_photo,
    ) -> None:
        """Create an empty session.

        Args:
            ctl: Controller used for alert and notice banners.
            picker: Returns a chosen file or None; injectable for tests.
        """
        self._ctl = ctl
        self._picker = picker
        self.underlay: Underlay | None = None
        self.texture: arcade.Texture | None = None

    def load(self) -> None:
        """Pick and load a photo; cancel is a no-op, failure alerts."""
        path = self._picker()
        if path is None:
            return
        try:
            self.underlay, self.texture = load_underlay(path)
        except UnderlayError as exc:
            self._ctl.show_alert(f"PHOTO NOT LOADED - {exc}")
            return
        self._ctl.show_notice(f"Underlay loaded: {path.name}")

    def toggle(self) -> None:
        """Show or hide the photo."""
        if self.underlay is not None:
            self.underlay = toggled(self.underlay)

    def step_opacity(self, direction: int) -> None:
        """Change opacity by one step (+1 or -1)."""
        if self.underlay is not None:
            value = self.underlay.opacity + direction * OPACITY_STEP
            self.underlay = with_opacity(self.underlay, value)

    def clear(self) -> None:
        """Drop the photo."""
        self.underlay, self.texture = None, None

    def draw(self, vp: Viewport, room: RoomSettings) -> None:
        """Draw the photo if loaded and visible."""
        draw_underlay(vp, self.underlay, self.texture, room)
