"""Tests for crowd_evac.adapters.render.grid_overlay (issue #19)."""
from __future__ import annotations

import pytest

from crowd_evac.adapters.render.grid_overlay import (
    GRID_LINE_COLOR,
    GRID_LINE_WIDTH_PX,
    grid_segments,
)


class TestGridSegments:
    """Pure segment maths."""

    def test_counts_and_extent(self) -> None:
        pts = grid_segments(10.0, 6.0, 40.0, 0.0, 0.0)
        assert len(pts) == (9 + 5) * 2
        verticals = pts[: 9 * 2]
        assert verticals[0] == (40.0, 0.0)
        assert verticals[1] == (40.0, 240.0)
        horizontals = pts[9 * 2:]
        assert horizontals[0] == (0.0, 40.0)
        assert horizontals[1] == (400.0, 40.0)

    def test_offsets_and_ppm_applied(self) -> None:
        pts = grid_segments(4.0, 4.0, 25.0, 100.0, 50.0)
        assert pts[0] == (125.0, 50.0)
        assert pts[1] == (125.0, 150.0)

    def test_non_integer_room(self) -> None:
        pts = grid_segments(5.5, 3.2, 10.0, 0.0, 0.0)
        assert len(pts) == (4 + 2) * 2

    @pytest.mark.parametrize(
        ("w", "h", "count"),
        [(1.0, 5.0, 4), (5.0, 1.0, 4), (0.0, 0.0, 0)],
        ids=["narrow", "short", "zero"],
    )
    def test_small_rooms(self, w: float, h: float, count: int) -> None:
        assert len(grid_segments(w, h, 10.0, 0.0, 0.0)) == count * 2

    @pytest.mark.parametrize("ppm", [0.0, -1.0])
    def test_non_positive_ppm_raises(self, ppm: float) -> None:
        with pytest.raises(ValueError, match="ppm"):
            grid_segments(5.0, 5.0, ppm, 0.0, 0.0)


def test_style_is_faint_and_thin() -> None:
    assert GRID_LINE_COLOR[3] < 64
    assert GRID_LINE_WIDTH_PX <= 1.0
