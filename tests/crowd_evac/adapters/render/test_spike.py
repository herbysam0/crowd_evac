"""Tests for crowd_evac.adapters.render.spike.

Covers the pure timing/aggregation helper (the unit the RK-1 spike is
graded on) and confirms the module is import-safe headless. The arcade
window itself needs a GL context and is exercised manually via
``scripts/bench_render.py`` on the target laptop, so it is not constructed
here.
"""
from __future__ import annotations

import arcade
import pytest

from crowd_evac.adapters.render import spike
from crowd_evac.adapters.render.spike import (
    FrameStats,
    RenderSpikeWindow,
    SpikeConfig,
    format_report,
    run_spike,
    summarize_frame_times,
)


class TestSummarizeFrameTimes:
    """Test suite for summarize_frame_times."""

    # -- Happy path ----------------------------------------------------

    def test_uniform_frames_yield_exact_fps(self) -> None:
        """Verify constant 10 ms frames report 100 FPS across all metrics."""
        stats = summarize_frame_times([0.01] * 50)
        assert stats.frame_count == 50
        assert stats.mean_fps == pytest.approx(100.0)
        assert stats.min_fps == pytest.approx(100.0)
        assert stats.max_fps == pytest.approx(100.0)
        assert stats.p1_low_fps == pytest.approx(100.0)
        assert stats.mean_frame_ms == pytest.approx(10.0)

    def test_mixed_frames_compute_expected_aggregates(self) -> None:
        """Verify mean/min/max/1%-low over a known mixed sequence."""
        stats = summarize_frame_times([0.01, 0.01, 0.02, 0.02])
        assert stats.duration_s == pytest.approx(0.06)
        assert stats.mean_fps == pytest.approx(4 / 0.06)
        assert stats.min_fps == pytest.approx(50.0)
        assert stats.max_fps == pytest.approx(100.0)
        # n_low = max(1, 4 // 100) = 1 -> the slowest frame (0.02 s).
        assert stats.p1_low_fps == pytest.approx(50.0)
        assert stats.mean_frame_ms == pytest.approx(15.0)

    def test_one_percent_low_averages_slowest_bucket(self) -> None:
        """Verify the 1% low uses count // 100 slowest frames."""
        # 200 fast frames (5 ms) + 2 slow frames (50 ms). n_low = 202 // 100 = 2,
        # so the 1% low is the mean FPS of exactly the two 50 ms frames.
        times = [0.005] * 200 + [0.05, 0.05]
        stats = summarize_frame_times(times)
        assert stats.p1_low_fps == pytest.approx(20.0)
        assert stats.min_fps == pytest.approx(20.0)
        assert stats.max_fps == pytest.approx(200.0)

    # -- Edge cases ----------------------------------------------------

    def test_single_frame_is_supported(self) -> None:
        """Verify a one-frame run produces consistent scalar stats."""
        stats = summarize_frame_times([0.04])
        assert stats.frame_count == 1
        assert stats.mean_fps == pytest.approx(25.0)
        assert stats.min_fps == stats.max_fps == pytest.approx(25.0)
        assert stats.p1_low_fps == pytest.approx(25.0)

    def test_returns_frozen_frame_stats(self) -> None:
        """Verify the result is an immutable FrameStats instance."""
        stats = summarize_frame_times([0.01])
        assert isinstance(stats, FrameStats)
        with pytest.raises((AttributeError, TypeError)):
            stats.mean_fps = 1.0  # type: ignore[misc]

    # -- Failure path --------------------------------------------------

    def test_empty_sequence_raises_value_error(self) -> None:
        """Verify an empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            summarize_frame_times([])

    @pytest.mark.parametrize(
        "bad_times",
        [[0.0], [0.01, 0.0, 0.02], [-0.01], [0.01, -0.5]],
        ids=["single-zero", "embedded-zero", "single-neg", "embedded-neg"],
    )
    def test_non_positive_frame_time_raises(
        self, bad_times: list[float]
    ) -> None:
        """Verify any non-positive frame time raises ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            summarize_frame_times(bad_times)


class TestFormatReport:
    """Test suite for format_report."""

    def test_report_contains_agent_count_and_metrics(self) -> None:
        """Verify the report names the agent count and key metrics."""
        stats = summarize_frame_times([0.02] * 10)
        report = format_report(stats, agent_count=10000)
        assert "10000 agents" in report
        assert "mean FPS" in report
        assert "1% low FPS" in report

    def test_report_is_single_block_without_trailing_newline(self) -> None:
        """Verify the report is multi-line with no trailing newline."""
        report = format_report(summarize_frame_times([0.01]), agent_count=1)
        assert "\n" in report
        assert not report.endswith("\n")


class TestHeadlessImport:
    """Verify the spike module is safe to import without a display."""

    def test_spike_config_defaults_and_overrides(self) -> None:
        """Verify SpikeConfig builds with defaults and accepts overrides."""
        assert SpikeConfig().agent_count == spike.DEFAULT_AGENT_COUNT
        custom = SpikeConfig(agent_count=10000, seed=7)
        assert custom.agent_count == 10000
        assert custom.seed == 7

    def test_window_class_is_arcade_subclass_not_instantiated(self) -> None:
        """Verify the window type exists and subclasses arcade.Window.

        The class is only referenced, never constructed, since constructing
        it requires a GL context unavailable in headless CI.
        """
        assert issubclass(RenderSpikeWindow, arcade.Window)

    def test_run_spike_is_callable(self) -> None:
        """Verify run_spike is exposed as a callable entry point."""
        assert callable(run_spike)
