"""Tests for the designer underlay adapter (texture loading; no display)."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from crowd_evac.adapters.render.designer_underlay import load_underlay
from crowd_evac.designer.underlay import UnderlayError


def test_load_underlay_reads_pixel_size(tmp_path: Path) -> None:
    """A real PNG yields matching pixel dimensions and default state."""
    png = tmp_path / "plan.png"
    Image.new("RGB", (40, 20), (200, 10, 10)).save(png)
    underlay, texture = load_underlay(png)
    assert (underlay.px_w, underlay.px_h) == (40, 20)
    assert underlay.visible is True
    assert texture.width == 40


def test_load_underlay_rejects_fake_image(tmp_path: Path) -> None:
    """Text content with an image suffix raises UnderlayError."""
    bad = tmp_path / "fake.png"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(UnderlayError):
        load_underlay(bad)


def test_load_underlay_rejects_missing_file(tmp_path: Path) -> None:
    """A missing file raises UnderlayError."""
    with pytest.raises(UnderlayError):
        load_underlay(tmp_path / "nope.png")


def test_load_underlay_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """Unsupported suffixes are rejected before any decoding."""
    with pytest.raises(UnderlayError, match="unsupported"):
        load_underlay(tmp_path / "plan.pdf")


def test_short_name_caps_length() -> None:
    """Long names are cut so the panel label cannot widen the column."""
    from crowd_evac.adapters.render.designer_underlay_panel import short_name

    assert short_name("plan.png") == "plan.png"
    cut = short_name("a_very_long_floor_plan_scan_2026.jpeg")
    assert len(cut) == 18 and cut.endswith("...")
