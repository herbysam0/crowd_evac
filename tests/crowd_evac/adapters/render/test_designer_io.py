"""Tests for designer save/load flows and their feedback.
"""
from __future__ import annotations

from pathlib import Path

from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.adapters.render.designer_io import save_design
from crowd_evac.designer.controller import DesignController, Tool
from crowd_evac.designer.layout import Layout

from .designer_helpers import ready_controller


class TestSaveDesign:
    """Save flow with overwrite confirmation (tmp_path only)."""

    def test_saves_and_reports(self, tmp_path: Path) -> None:
        """Verify a valid design is written and loadable."""
        ctl = ready_controller()
        assert save_design(ctl, None, tmp_path) == (None, True)
        assert ctl.status == "Saved saved_room.json"
        load_scenario_file(tmp_path / "saved_room.json")

    def test_existing_name_needs_second_click(self, tmp_path: Path) -> None:
        """Verify overwrite is armed first, then performed."""
        ctl = ready_controller()
        save_design(ctl, None, tmp_path)
        pending, saved = save_design(ctl, None, tmp_path)
        assert pending is not None and not saved and "again" in ctl.status
        assert save_design(ctl, pending, tmp_path) == (None, True)
        assert ctl.status.startswith("Saved")

    def test_edit_after_arming_rearms(self, tmp_path: Path) -> None:
        """Verify changing the design invalidates the confirmation."""
        ctl = ready_controller()
        save_design(ctl, None, tmp_path)
        pending, _ = save_design(ctl, None, tmp_path)
        ctl.update_room(width_m=16.0)
        again, _ = save_design(ctl, pending, tmp_path)
        assert again is not None and "again" in ctl.status

    def test_invalid_design_writes_nothing(self, tmp_path: Path) -> None:
        """Verify a design without doorway shows an error, no file."""
        ctl = DesignController()
        assert save_design(ctl, None, tmp_path) == (None, False)
        assert ctl.status and list(tmp_path.iterdir()) == []

    def test_bad_name_reported(self, tmp_path: Path) -> None:
        """Verify an unsafe name is rejected on Save."""
        ctl = ready_controller()
        ctl.update_room(name="../evil")
        save_design(ctl, None, tmp_path)
        assert "Invalid scenario name" in ctl.status
        assert list(tmp_path.iterdir()) == []


class TestLoadDesign:
    """Load flow with discard confirmation (read-only on bundled files)."""

    def _fresh(self) -> tuple[DesignController, Layout]:
        """Return a controller and its clean (as-loaded) layout."""
        ctl = DesignController()
        return ctl, ctl.model.layout

    def test_loads_bundled_scenario(self) -> None:
        """Verify a clean design loads straight away."""
        from crowd_evac.adapters.render.designer_io import load_design

        ctl, clean = self._fresh()
        pending, loaded = load_design(ctl, "fixture_minimal", None, clean)
        assert (pending, loaded) == (None, True)
        assert ctl.model.layout.room.name == "fixture_minimal"
        assert ctl.model.layout.extras.agent_count == 50

    def test_unsaved_changes_need_second_click(self) -> None:
        """Verify load asks before discarding edits, then proceeds."""
        from crowd_evac.adapters.render.designer_io import load_design

        ctl, clean = self._fresh()
        ctl.set_tool(Tool.OBSTACLE)
        ctl.press(5.0, 4.0)
        ctl.release(7.0, 5.0)
        pending, loaded = load_design(ctl, "fixture_minimal", None, clean)
        assert pending is not None and not loaded
        assert "UNSAVED" in ctl.alert
        assert ctl.model.layout.room.name != "fixture_minimal"
        assert load_design(ctl, "fixture_minimal", pending, clean)[1]

    def test_missing_file_raises_alert_and_keeps_design(self) -> None:
        """Verify a missing scenario is a loud failure, design untouched."""
        from crowd_evac.adapters.render.designer_io import load_design

        ctl, clean = self._fresh()
        before = ctl.model.layout
        assert load_design(ctl, "nope_not_here", None, clean) == (None, False)
        assert "LOAD FAILED" in ctl.alert and ctl.model.layout is before

    def test_empty_name_is_ignored(self) -> None:
        """Verify no selection gives a status, not an exception."""
        from crowd_evac.adapters.render.designer_io import load_design

        ctl, clean = self._fresh()
        assert load_design(ctl, "", None, clean) == (None, False)
        assert ctl.status == "No scenario selected"

    def test_list_scenarios_sorted_and_tolerates_missing_dir(
        self, tmp_path: Path
    ) -> None:
        """Verify directory listing."""
        from crowd_evac.adapters.render.designer_io import list_scenarios

        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        (tmp_path / "a.json").write_text("{}", encoding="utf-8")
        (tmp_path / "skip.txt").write_text("", encoding="utf-8")
        assert list_scenarios(tmp_path) == ["a", "b"]
        assert list_scenarios(tmp_path / "missing") == []


class TestSaveFeedback:
    """Save results are loud: banner text, not just the status line."""

    def test_success_notice_shows_full_path(self, tmp_path: Path) -> None:
        """Verify a green notice names the file's full path."""
        ctl = ready_controller()
        save_design(ctl, None, tmp_path)
        assert ctl.notice == f"SAVED  {tmp_path / 'saved_room.json'}"
        assert ctl.notice_seq == 1

    def test_missing_doorway_alert_is_plain_language(
        self, tmp_path: Path
    ) -> None:
        """Verify the cryptic loader message becomes an actionable alert."""
        ctl = DesignController()
        ctl.set_tool(Tool.WALL)
        ctl.press(0.0, 0.0)
        ctl.release(15.0, 0.0)
        save_design(ctl, None, tmp_path)
        assert ctl.alert.startswith("NOT SAVED")
        assert "at least one doorway" in ctl.alert
        assert "FloorPlan" not in ctl.alert

    def test_overwrite_prompt_raises_alert(self, tmp_path: Path) -> None:
        """Verify the overwrite confirmation is a banner too."""
        ctl = ready_controller()
        save_design(ctl, None, tmp_path)
        save_design(ctl, None, tmp_path)
        assert "ALREADY EXISTS" in ctl.alert

    def test_bad_name_alert(self, tmp_path: Path) -> None:
        """Verify an invalid name is reported as NOT SAVED."""
        ctl = ready_controller()
        ctl.update_room(name="Bad Name")
        save_design(ctl, None, tmp_path)
        assert ctl.alert.startswith("NOT SAVED")
        assert "Invalid scenario name" in ctl.alert
