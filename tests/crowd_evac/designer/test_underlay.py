"""Tests for crowd_evac.designer.underlay (pure state and geometry)."""
from __future__ import annotations

from pathlib import Path

import pytest

from crowd_evac.designer.underlay import (
    Underlay,
    UnderlayError,
    fit_rect,
    toggled,
    with_opacity,
)


def _u() -> Underlay:
    return Underlay(Path("plan.png"), 100, 50)


class TestFitRect:
    """fit_rect keeps aspect ratio, anchored at the origin."""

    @pytest.mark.parametrize(
        "px, room, expected",
        [
            ((2000, 1000), (10.0, 10.0), (0.0, 0.0, 10.0, 5.0)),
            ((1000, 2000), (10.0, 10.0), (0.0, 0.0, 5.0, 10.0)),
            ((500, 500), (8.0, 8.0), (0.0, 0.0, 8.0, 8.0)),
        ],
        ids=["wide", "tall", "square"],
    )
    def test_fit(self, px, room, expected) -> None:  # type: ignore[no-untyped-def]
        """Largest aspect-preserving rect fits the room."""
        assert fit_rect(*px, *room) == expected

    @pytest.mark.parametrize(
        "args",
        [(0, 10, 5.0, 5.0), (10, -1, 5.0, 5.0), (10, 10, 0.0, 5.0)],
    )
    def test_non_positive_raises(
        self, args: tuple[int, int, float, float]
    ) -> None:
        """Non-positive sizes are rejected."""
        with pytest.raises(UnderlayError):
            fit_rect(*args)


class TestOpacityAndToggle:
    """with_opacity clamps; toggled flips visibility only."""

    @pytest.mark.parametrize(
        "value, expected", [(-0.2, 0.0), (1.3, 1.0), (0.4, 0.4)]
    )
    def test_clamp(self, value: float, expected: float) -> None:
        """Opacity stays within [0, 1]."""
        assert with_opacity(_u(), value).opacity == expected

    def test_toggle_flips_visible_only(self) -> None:
        """Toggling leaves other fields intact and is reversible."""
        u = _u()
        t = toggled(u)
        assert t.visible is False
        assert (t.path, t.px_w, t.opacity) == (u.path, u.px_w, u.opacity)
        assert toggled(t) == u
