"""Render tests for the performance dashboard window (need a display).

Covers: opening with defaults, hazard Add/Remove, field errors in the status
line, chart cache, a headless batch and a visual batch through real child
processes, and Cancel.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from crowd_evac.dashboard.batch_process import BatchProcess
from crowd_evac.dashboard.form import MAX_HAZARDS
from crowd_evac.montecarlo.runner import run_monte_carlo

if TYPE_CHECKING:
    from crowd_evac.adapters.render.dashboard_window import DashboardWindow

_ROOM = "fixture_minimal"
_TIMEOUT_S = 180.0

pytestmark = pytest.mark.render


@pytest.fixture
def window() -> Iterator[DashboardWindow]:
    """Open a dashboard window and close it afterwards."""
    from crowd_evac.adapters.render.dashboard_window import DashboardWindow

    win = DashboardWindow()
    try:
        yield win
    finally:
        win.close()


def _frame(win: DashboardWindow) -> None:
    """Run one event/update/draw cycle."""
    win.dispatch_events()
    win.on_update(1 / 60)
    win.on_draw()
    win.flip()


def _pump_until_done(win: DashboardWindow) -> int:
    """Run frames until the batch ends; return how many ran."""
    frames = 0
    deadline = time.monotonic() + _TIMEOUT_S
    while win.running and time.monotonic() < deadline:
        _frame(win)
        frames += 1
        time.sleep(0.01)
    assert not win.running, "batch never finished"
    _frame(win)
    return frames


def _setup_small_batch(win: DashboardWindow, *, visual: bool = False) -> None:
    """Fill the form for a quick batch on the small room."""
    panel = win.panel
    panel._dropdown.value = _ROOM
    panel._persons.text = "2"
    panel._iterations.text = "3"
    panel._seed.text = "5"
    if visual:
        panel._toggle_mode()
    win.hazards.add_row()
    row = win.hazards.rows[0]
    for key, text in {
        "x_min": "1", "x_max": "9", "y_min": "1", "y_max": "7",
        "t_min": "0", "t_max": "2",
    }.items():
        row.inputs[key].text = text


class TestOpening:
    """Initial state."""

    def test_defaults_and_room_hint(self, window: DashboardWindow) -> None:
        """Verify issue defaults and the room-size hint."""
        panel = window.panel
        fields = panel.read_fields(())
        assert (fields.scenario, fields.persons, fields.visual) == (
            "lecture_hall", "1", False
        )
        assert (fields.seed, fields.iterations) == ("", "20")
        assert not window.running
        assert window.hazards.rows == []
        _frame(window)

    def test_scenario_change_updates_hint(self, window: DashboardWindow) -> None:
        """Verify choosing another room changes the size hint."""
        panel = window.panel
        panel._dropdown.value = _ROOM
        assert "10 x 8 m" in panel._hint.text
        assert panel.cancel_button.disabled is True


class TestHazardList:
    """Add and Remove."""

    def test_add_and_remove(self, window: DashboardWindow) -> None:
        """Verify rows are added, centred in the room and removed."""
        hazards = window.hazards
        hazards.add_row()
        hazards.add_row()
        assert len(hazards.rows) == 2
        assert hazards.read()[0].source_type == "fire"
        assert hazards.read()[0].x_min == "25"  # lecture_hall is 50 x 30
        hazards.remove_row(hazards.rows[0])
        assert len(hazards.rows) == 1
        _frame(window)

    def test_limit_disables_add(self, window: DashboardWindow) -> None:
        """Verify the list stops at MAX_HAZARDS and re-enables on remove."""
        hazards = window.hazards
        for _ in range(MAX_HAZARDS + 2):
            hazards.add_row()
        assert len(hazards.rows) == MAX_HAZARDS
        assert hazards._add.disabled is True
        hazards.remove_row(hazards.rows[0])
        assert hazards._add.disabled is False

    def test_toggle_blocking(self, window: DashboardWindow) -> None:
        """Verify the blocks-route button flips the field."""
        hazards = window.hazards
        hazards.add_row()
        row = hazards.rows[0]
        assert row.read().blocks_navigation is True
        row._toggle_nav()
        assert row.read().blocks_navigation is False


class TestErrors:
    """Bad input shows its message and starts nothing."""

    def test_bad_persons(self, window: DashboardWindow) -> None:
        """Verify a zero persons field shows its error."""
        window.panel._persons.text = "0"
        window.run_single()
        assert "Persons" in window.panel.status_text
        assert not window.running

    def test_bad_hazard_field_names_the_row(self, window: DashboardWindow) -> None:
        """Verify a hazard error is prefixed with its row number."""
        window.hazards.add_row()
        window.hazards.rows[0].inputs["x_min"].text = "abc"
        window.run_batch()
        assert window.panel.status_text.startswith("Hazard 1")
        assert not window.running

    def test_hazard_outside_room_reported_before_start(
        self, window: DashboardWindow
    ) -> None:
        """Verify an out-of-room hazard never starts a process."""
        window.panel._dropdown.value = _ROOM
        window.hazards.add_row()
        window.hazards.rows[0].inputs["x_min"].text = "500"
        window.run_single()
        assert "outside" in window.panel.status_text
        assert window._batch is None


class TestChartCache:
    """The chart rebuilds only when something changes."""

    def test_cache_holds_between_frames(self, window: DashboardWindow) -> None:
        """Verify repeated frames do not rebuild, new data does."""
        chart = window.chart
        _frame(window)
        _frame(window)
        count = chart.rebuild_count
        _frame(window)
        assert chart.rebuild_count == count
        from crowd_evac.montecarlo.results import build_curves, make_outcome

        chart.set_curves(build_curves([make_outcome(0, 1, (), 1, (0, 1))]))
        _frame(window)
        assert chart.rebuild_count == count + 1
        chart.set_rect(*chart._rect)
        _frame(window)
        assert chart.rebuild_count == count + 1


class TestBatches:
    """Whole batches through real child processes."""

    def test_headless_batch_fills_results(self, window: DashboardWindow) -> None:
        """Verify results, chart and buttons after a headless batch."""
        _setup_small_batch(window)
        window.run_batch()
        assert window.running
        assert window.panel.single_button.disabled
        _pump_until_done(window)
        assert window.results.status_text == "Done 3/3"
        assert window.results.time_text.startswith("mean")
        assert "m/s" in window.results.speed_text
        assert not window.panel.single_button.disabled
        direct = run_monte_carlo(window._batch.config)
        assert tuple(window._outcomes) == direct.outcomes

    def test_visual_batch_runs_beside_dashboard(self, window: DashboardWindow) -> None:
        """Verify a visual batch finishes while the dashboard keeps updating."""
        _setup_small_batch(window, visual=True)
        window.run_batch()
        frames = _pump_until_done(window)
        assert frames > 0
        assert window.results.status_text == "Done 3/3"
        assert window._batch is not None
        headless = run_monte_carlo(
            replace(window._batch.config, visual=False, max_workers=1)
        )
        assert tuple(window._outcomes) == headless.outcomes

    def test_cancel_keeps_finished_runs(self, window: DashboardWindow) -> None:
        """Verify Cancel shows 'Cancelling...' and keeps finished runs."""
        _setup_small_batch(window)
        window.panel._iterations.text = "60"
        window.run_batch()
        deadline = time.monotonic() + _TIMEOUT_S
        while not window._outcomes and time.monotonic() < deadline:
            _frame(window)
            time.sleep(0.01)
        window.cancel()
        assert window.results.status_text == "Cancelling..."
        _pump_until_done(window)
        assert window.results.status_text.startswith("Cancelled")
        assert 1 <= len(window._outcomes) < 60

    def test_ctrl_c_closes_dashboard_and_cancels_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify Ctrl+C in the terminal ends run_dashboard without raising."""
        import arcade

        from crowd_evac.adapters.render import dashboard_window as dw

        created: list[dw.DashboardWindow] = []
        real = dw.DashboardWindow

        def make() -> dw.DashboardWindow:
            """Open a real window and remember it."""
            win = real()
            created.append(win)
            return win

        def interrupted() -> None:
            """Stand in for the event loop receiving Ctrl+C mid-batch."""
            _setup_small_batch(created[0])
            created[0].run_batch()
            raise KeyboardInterrupt

        cancels: list[bool] = []
        real_cancel = BatchProcess.cancel

        def spy_cancel(batch: BatchProcess) -> None:
            """Record the call, then really cancel so the child ends."""
            cancels.append(True)
            real_cancel(batch)

        monkeypatch.setattr(dw, "DashboardWindow", make)
        monkeypatch.setattr(arcade, "run", interrupted)
        monkeypatch.setattr(BatchProcess, "cancel", spy_cancel)
        dw.run_dashboard()
        assert cancels == [True]
        assert created[0].has_exit

    def test_failed_batch_shows_message(self, window: DashboardWindow) -> None:
        """Verify a batch the child cannot run shows its error."""
        _setup_small_batch(window)
        window.hazards.rows[0].inputs["x_min"].text = "0"
        window.hazards.rows[0].inputs["x_max"].text = "0.1"
        window.run_batch()
        _pump_until_done(window)
        assert window.panel.status_text.startswith("Batch failed")
        assert not window.panel.single_button.disabled
