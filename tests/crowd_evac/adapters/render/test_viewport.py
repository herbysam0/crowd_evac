"""Tests for crowd_evac.adapters.render.viewport -- shared pixel mapping.

Covers:
  - Viewport round trips and rectangle conversion.
  - fit_viewport: centring, margin, aspect, degenerate canvases.
"""
from __future__ import annotations

import math

import pytest

from crowd_evac.adapters.render.viewport import Viewport, fit_viewport

_MARGIN = 40.0


class TestViewport:
    """Metre/pixel mapping."""

    def test_round_trip(self) -> None:
        """Verify to_px / to_world are inverses."""
        vp = fit_viewport(15.0, 10.0, 980.0, 760.0, margin_px=_MARGIN)
        px = vp.to_px(3.2, 4.1)
        assert vp.to_world(*px) == pytest.approx((3.2, 4.1))

    def test_lrbt_scales_rect(self) -> None:
        """Verify rectangle conversion uses ppm."""
        vp = fit_viewport(10.0, 10.0, 500.0, 500.0, margin_px=_MARGIN)
        left, right, _, _ = vp.lrbt((1.0, 1.0, 2.0, 2.0))
        assert right - left == pytest.approx(2.0 * vp.ppm)

    def test_to_world_inverts_offsets(self) -> None:
        """Verify offsets are removed before dividing by ppm."""
        vp = Viewport(ppm=10.0, ox=100.0, oy=50.0)
        assert vp.to_world(150.0, 70.0) == pytest.approx((5.0, 2.0))


class TestFitViewport:
    """Uniform fit, centring and margins."""

    def test_room_fits_inside_canvas_with_margin(self) -> None:
        """Verify the scaled room stays within the canvas margin."""
        vp = fit_viewport(50.0, 30.0, 980.0, 760.0, margin_px=_MARGIN)
        left, right, bottom, top = vp.lrbt((0.0, 0.0, 50.0, 30.0))
        assert 0 < left and right < 980 and 0 < bottom and top < 760

    def test_wider_window_centres_horizontally(self) -> None:
        """Verify spare width becomes equal bars left and right."""
        vp = fit_viewport(10.0, 10.0, 400.0, 200.0)
        assert vp.ppm == pytest.approx(20.0)
        assert vp.ox == pytest.approx(100.0)
        assert vp.oy == pytest.approx(0.0)

    def test_taller_window_centres_vertically(self) -> None:
        """Verify spare height becomes equal bars top and bottom."""
        vp = fit_viewport(10.0, 10.0, 200.0, 400.0)
        assert vp.ppm == pytest.approx(20.0)
        assert vp.ox == pytest.approx(0.0)
        assert vp.oy == pytest.approx(100.0)

    def test_matching_aspect_has_no_bars(self) -> None:
        """Verify margin 0 and matching aspect leave no offset."""
        vp = fit_viewport(50.0, 30.0, 500.0, 300.0, margin_px=0.0)
        assert vp.ox == pytest.approx(0.0)
        assert vp.oy == pytest.approx(0.0)
        assert vp.ppm == pytest.approx(10.0)

    @pytest.mark.parametrize("size", [(0.0, 0.0), (1.0, 1.0)])
    def test_degenerate_canvas_is_finite_and_positive(
        self, size: tuple[float, float]
    ) -> None:
        """Verify tiny canvases never yield zero, negative or NaN scale."""
        vp = fit_viewport(10.0, 10.0, *size, margin_px=_MARGIN)
        assert vp.ppm > 0.0
        assert all(math.isfinite(v) for v in (vp.ppm, vp.ox, vp.oy))
