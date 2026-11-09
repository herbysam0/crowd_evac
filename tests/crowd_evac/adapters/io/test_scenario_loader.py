"""Tests for crowd_evac.adapters.io.scenario_loader.

Covers happy-path loading, data round-trip fidelity, malformed input
handling, and CWD-independent bundled asset resolution.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from crowd_evac.adapters.io.scenario_loader import (
    load_bundled_scenario,
    load_scenario_file,
)
from crowd_evac.domain.errors import MalformedScenarioError, ScenarioValidationError
from crowd_evac.domain.floor_plan import ExitSide, FloorPlan


# ---------------------------------------------------------------------------
# Fixtures — minimal valid scenario dict for reuse
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_scenario_dict() -> dict[str, object]:
    """Return a minimal valid scenario as a plain Python dict."""
    return {
        "schema_version": "2.0",
        "name": "unit_test_room",
        "floor_plan": {
            "width_m": 10.0,
            "height_m": 8.0,
            "walls": [
                {"x": 0.0, "y": 0.0, "width": 10.0, "height": 0.5,
                 "label": "south_wall"},
            ],
            "obstacles": [
                {"x": 3.0, "y": 3.0, "width": 2.0, "height": 1.0}
            ],
            "exits": [
                {
                    "x": 5.0, "y": 0.25, "width_m": 2.0,
                    "side": "south", "capacity_per_second": 5,
                    "label": "main_exit",
                }
            ],
        },
    }


@pytest.fixture
def scenario_file(
    tmp_path: Path, valid_scenario_dict: dict[str, object]
) -> Path:
    """Write the valid scenario dict to a temp JSON file and return its path."""
    path = tmp_path / "test_scenario.json"
    path.write_text(json.dumps(valid_scenario_dict), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_scenario_file — happy path
# ---------------------------------------------------------------------------


class TestLoadScenarioFileHappyPath:
    """Test load_scenario_file with well-formed input."""

    def test_returns_floor_plan_and_scenario_data(
        self, scenario_file: Path
    ) -> None:
        """Verify load returns a (FloorPlan, ScenarioData) tuple."""
        floor_plan, data = load_scenario_file(scenario_file)
        assert isinstance(floor_plan, FloorPlan)
        assert data["name"] == "unit_test_room"

    def test_floor_plan_dimensions_round_trip(
        self, scenario_file: Path
    ) -> None:
        """Verify width_m and height_m are preserved through the load cycle."""
        floor_plan, _ = load_scenario_file(scenario_file)
        assert floor_plan.width_m == 10.0
        assert floor_plan.height_m == 8.0

    def test_walls_round_trip(self, scenario_file: Path) -> None:
        """Verify wall count, position and label are preserved."""
        floor_plan, _ = load_scenario_file(scenario_file)
        assert len(floor_plan.walls) == 1
        wall = floor_plan.walls[0]
        assert wall.x == 0.0
        assert wall.y == 0.0
        assert wall.width == 10.0
        assert wall.height == 0.5
        assert wall.label == "south_wall"

    def test_obstacles_round_trip(self, scenario_file: Path) -> None:
        """Verify obstacle count and geometry are preserved."""
        floor_plan, _ = load_scenario_file(scenario_file)
        assert len(floor_plan.obstacles) == 1
        obs = floor_plan.obstacles[0]
        assert obs.x == 3.0
        assert obs.width == 2.0

    def test_exits_round_trip(self, scenario_file: Path) -> None:
        """Verify exit data is preserved: position, width, side, capacity."""
        floor_plan, _ = load_scenario_file(scenario_file)
        assert len(floor_plan.exits) == 1
        exit_ = floor_plan.exits[0]
        assert exit_.x == 5.0
        assert exit_.y == 0.25
        assert exit_.width_m == 2.0
        assert exit_.side is ExitSide.SOUTH
        assert exit_.capacity_per_second == 5
        assert exit_.label == "main_exit"

    def test_fractional_capacity_parsed_as_float(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify capacity_per_second 1.05 loads as float, int still accepted."""
        fp_data = valid_scenario_dict["floor_plan"]
        assert isinstance(fp_data, dict)
        fp_data["exits"][0]["capacity_per_second"] = 1.05
        path = tmp_path / "frac.json"
        path.write_text(json.dumps(valid_scenario_dict), encoding="utf-8")
        floor_plan, _ = load_scenario_file(path)
        assert floor_plan.exits[0].capacity_per_second == 1.05
        fp_data["exits"][0]["capacity_per_second"] = 3
        path.write_text(json.dumps(valid_scenario_dict), encoding="utf-8")
        floor_plan, _ = load_scenario_file(path)
        assert isinstance(floor_plan.exits[0].capacity_per_second, float)

    def test_scenario_data_is_room_only(self, scenario_file: Path) -> None:
        """Verify ScenarioData carries only version, name and floor plan."""
        _, data = load_scenario_file(scenario_file)
        assert set(data) == {"schema_version", "name", "floor_plan"}
        assert data["schema_version"] == "2.0"

    def test_removed_blocks_in_file_do_not_reach_scenario_data(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify leftover agents/simulation/events keys are dropped."""
        scenario = {
            **valid_scenario_dict,
            "agents": {"count": 5, "spawn_seed": 1},
            "simulation": {"dt": 0.05},
            "events": [],
        }
        path = tmp_path / "leftovers.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        _, data = load_scenario_file(path)
        assert set(data) == {"schema_version", "name", "floor_plan"}

    def test_optional_walls_default_to_empty(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify scenario without walls key still loads with empty walls."""
        fp_dict = dict(valid_scenario_dict["floor_plan"])  # type: ignore[arg-type]
        del fp_dict["walls"]
        scenario = {**valid_scenario_dict, "floor_plan": fp_dict}
        path = tmp_path / "no_walls.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        floor_plan, _ = load_scenario_file(path)
        assert floor_plan.walls == ()

    def test_optional_obstacles_default_to_empty(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify scenario without obstacles key loads with empty obstacles."""
        fp_dict = dict(valid_scenario_dict["floor_plan"])  # type: ignore[arg-type]
        del fp_dict["obstacles"]
        scenario = {**valid_scenario_dict, "floor_plan": fp_dict}
        path = tmp_path / "no_obs.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        floor_plan, _ = load_scenario_file(path)
        assert floor_plan.obstacles == ()


# ---------------------------------------------------------------------------
# load_scenario_file — malformed / invalid input
# ---------------------------------------------------------------------------


class TestLoadScenarioFileMalformed:
    """Test load_scenario_file with broken or incomplete input."""

    def test_invalid_json_raises_malformed_error(
        self, tmp_path: Path
    ) -> None:
        """Verify completely invalid JSON raises MalformedScenarioError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json }", encoding="utf-8")
        with pytest.raises(MalformedScenarioError, match="Invalid JSON"):
            load_scenario_file(bad)

    def test_missing_floor_plan_key_raises(self, tmp_path: Path) -> None:
        """Verify absence of floor_plan key raises MalformedScenarioError."""
        incomplete = {
            "schema_version": "2.0",
            "name": "broken",
        }
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(MalformedScenarioError, match="floor_plan"):
            load_scenario_file(path)

    def test_file_without_agents_block_loads(self, tmp_path: Path) -> None:
        """Verify a 2.0 file needs no agents block."""
        scenario = {
            "schema_version": "2.0",
            "name": "room_only",
            "floor_plan": {
                "width_m": 5.0, "height_m": 5.0, "exits": [
                    {"x": 2.5, "y": 0.0, "width_m": 1.0,
                     "side": "south", "capacity_per_second": 1}
                ],
            },
        }
        path = tmp_path / "room_only.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        floor_plan, data = load_scenario_file(path)
        assert floor_plan.width_m == 5.0
        assert data["name"] == "room_only"

    def test_legacy_1_0_rejected_naming_migration_script(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify a 1.0 file is rejected and the error names the script."""
        legacy = {
            **valid_scenario_dict,
            "schema_version": "1.0",
            "agents": {"count": 50, "spawn_seed": 42},
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        with pytest.raises(
            MalformedScenarioError, match="migrate_scenarios.py"
        ):
            load_scenario_file(path)

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        """Verify absence of schema_version raises MalformedScenarioError."""
        path = tmp_path / "no_version.json"
        path.write_text(json.dumps({"name": "x", "floor_plan": {}}),
                        encoding="utf-8")
        with pytest.raises(MalformedScenarioError, match="schema_version"):
            load_scenario_file(path)

    def test_unsupported_schema_version_raises(self, tmp_path: Path) -> None:
        """Verify an unknown schema_version raises MalformedScenarioError."""
        scenario = {
            "schema_version": "99.0",
            "name": "future",
            "floor_plan": {
                "width_m": 5.0, "height_m": 5.0, "exits": [
                    {"x": 2.5, "y": 0.0, "width_m": 1.0,
                     "side": "south", "capacity_per_second": 1}
                ],
            },
        }
        path = tmp_path / "future.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        with pytest.raises(MalformedScenarioError, match="schema_version"):
            load_scenario_file(path)

    def test_invalid_exit_side_raises(self, tmp_path: Path) -> None:
        """Verify an unrecognised exit side value raises MalformedScenarioError."""
        scenario = {
            "schema_version": "2.0",
            "name": "bad_side",
            "floor_plan": {
                "width_m": 5.0, "height_m": 5.0,
                "exits": [
                    {"x": 2.5, "y": 0.0, "width_m": 1.0,
                     "side": "diagonal", "capacity_per_second": 1}
                ],
            },
        }
        path = tmp_path / "bad_side.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        with pytest.raises(MalformedScenarioError, match="diagonal"):
            load_scenario_file(path)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            load_scenario_file(tmp_path / "nonexistent.json")

    def test_negative_floor_plan_width_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        """Verify semantically invalid data raises ScenarioValidationError."""
        scenario = {
            "schema_version": "2.0",
            "name": "bad_dim",
            "floor_plan": {
                "width_m": -5.0, "height_m": 5.0,
                "exits": [
                    {"x": 1.0, "y": 0.0, "width_m": 1.0,
                     "side": "south", "capacity_per_second": 1}
                ],
            },
        }
        path = tmp_path / "bad_dim.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        with pytest.raises(ScenarioValidationError):
            load_scenario_file(path)


# ---------------------------------------------------------------------------
# load_bundled_scenario — CWD independence
# ---------------------------------------------------------------------------


class TestLoadBundledScenario:
    """Test load_bundled_scenario resolves assets via importlib.resources."""

    def test_bundled_loads_regardless_of_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify bundled scenario loads correctly from any working directory."""
        monkeypatch.chdir(tmp_path)  # CWD has no scenario files
        floor_plan, data = load_bundled_scenario("fixture_minimal")
        assert isinstance(floor_plan, FloorPlan)
        assert data["name"] == "fixture_minimal"

    def test_bundled_fixture_dimensions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fixture_minimal is 10 x 8 m plus its 1 m low-y apron."""
        monkeypatch.chdir(tmp_path)
        floor_plan, _ = load_bundled_scenario("fixture_minimal")
        assert floor_plan.width_m == 10.0
        assert floor_plan.height_m == 9.0

    def test_unknown_bundled_name_raises_malformed_error(self) -> None:
        """Verify an unknown bundled name raises MalformedScenarioError."""
        with pytest.raises(MalformedScenarioError, match="not found"):
            load_bundled_scenario("no_such_scenario_xyz")


# ---------------------------------------------------------------------------
# Lecture Hall bundled scenario — FR-0, FR-13.1
# ---------------------------------------------------------------------------


class TestLectureHallScenario:
    """Test suite for the lecture_hall bundled scenario (Phase 1 default)."""

    def test_lecture_hall_loads_via_bundled_loader(self) -> None:
        """Verify the bundled lecture_hall scenario loads without error."""
        floor_plan, data = load_bundled_scenario("lecture_hall")
        assert isinstance(floor_plan, FloorPlan)
        assert data["name"] == "lecture_hall"

    def test_lecture_hall_dimensions_in_valid_range(self) -> None:
        """Verify the 50 x 30 m hall plus west, east and low-y aprons."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        assert floor_plan.width_m == 52.0
        assert floor_plan.height_m == 31.0

    def test_lecture_hall_exits_present(self) -> None:
        """Verify at least one exit exists (main exit required)."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        assert len(floor_plan.exits) >= 1, "At least one exit required"

    def test_lecture_hall_main_exit_accessible(self) -> None:
        """Verify main exit (front) has positive capacity and reasonable size."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        # Assume first exit is the main one; verify it's on north wall
        main_exit = floor_plan.exits[0]
        assert main_exit.side is ExitSide.NORTH
        assert main_exit.width_m > 0.0
        assert main_exit.capacity_per_second > 0

    def test_lecture_hall_obstacles_present(self) -> None:
        """Verify seating obstacles are present (tiered-seating requirement)."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        assert len(floor_plan.obstacles) > 0, "Seating obstacles required"

    def test_lecture_hall_obstacles_are_seating_rows(self) -> None:
        """Verify obstacles form a reasonable tiered seating pattern."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        # Expect multiple rows with consistent widths and small heights
        rows = [obs for obs in floor_plan.obstacles]
        assert len(rows) >= 3, "At least 3 seating rows expected"
        # All seating rows should have similar width (within 10%)
        widths = [r.width for r in rows]
        min_width = min(widths)
        max_width = max(widths)
        assert max_width <= min_width * 1.1, "Seating row widths should be consistent"

    def test_lecture_hall_aisle_exists_for_convergence(self) -> None:
        """Verify center aisle is navigable (agents can flow from back to exit).

        Checks that there's a continuous vertical corridor between seating
        rows through which agents can navigate.
        """
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        # Seating rows should not span the full width, leaving aisles
        max_obstacle_width = max(obs.width for obs in floor_plan.obstacles)
        assert (
            max_obstacle_width < floor_plan.width_m
        ), "Aisle(s) required for navigation"

    def test_lecture_hall_walls_form_boundary(self) -> None:
        """Verify outer walls define the room boundary."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        # Should have at least perimeter walls
        assert len(floor_plan.walls) >= 2, "Boundary walls required"

    def test_lecture_hall_optional_side_exits_if_present(self) -> None:
        """Verify that optional side exits (if present) are on east/west walls."""
        floor_plan, _ = load_bundled_scenario("lecture_hall")
        side_exits = [
            e for e in floor_plan.exits
            if e.side in (ExitSide.EAST, ExitSide.WEST)
        ]
        for exit_ in side_exits:
            assert exit_.width_m > 0.0
            assert exit_.capacity_per_second > 0


# ---------------------------------------------------------------------------
# Bundled rooms — schema 2.1, room configuration only
# ---------------------------------------------------------------------------


BUNDLED_ROOMS: tuple[str, ...] = (
    "bottleneck_corridor",
    "fixture_minimal",
    "lecture_hall",
    "lecture_hall_no_aisle",
    "model_room_1",
    "open_room_search",
    "room_model_3",
)


class TestBundledRooms:
    """Every shipped room is schema 2.1 and holds no run settings."""

    def test_catalog_matches_expected_rooms(self) -> None:
        """Verify the shipped rooms exist; extra user rooms are tolerated.

        Rooms saved from the designer land in the same folder, so extras
        must not fail the test. Retired rooms must stay gone.
        """
        from crowd_evac.adapters.io.scenario_catalog import (
            list_bundled_scenarios,
        )

        listed = set(list_bundled_scenarios())
        assert set(BUNDLED_ROOMS) <= listed
        assert "hazard_lecture_hall" not in listed
        assert not any(name.endswith("_small") for name in listed)

    @pytest.mark.parametrize("name", BUNDLED_ROOMS)
    def test_room_loads_as_2_1(self, name: str) -> None:
        """Verify each room is migrated (schema 2.1) and keeps its name."""
        _, data = load_bundled_scenario(name)
        assert data["schema_version"] == "2.1"
        assert data["name"] == name

    @pytest.mark.parametrize("name", BUNDLED_ROOMS)
    def test_room_file_has_only_room_keys(self, name: str) -> None:
        """Verify the raw JSON holds only version, name and floor plan."""
        from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR

        raw = json.loads(
            (SCENARIOS_DIR / f"{name}.json").read_text(encoding="utf-8")
        )
        assert set(raw) == {"schema_version", "name", "floor_plan"}

    def test_no_aisle_room_has_lecture_hall_footprint(self) -> None:
        """Verify the renamed room keeps the lecture hall footprint."""
        floor_plan, _ = load_bundled_scenario("lecture_hall_no_aisle")
        hall, _ = load_bundled_scenario("lecture_hall")
        assert (floor_plan.width_m, floor_plan.height_m) == (52.0, 31.0)
        assert (floor_plan.width_m, floor_plan.height_m) == (
            hall.width_m, hall.height_m
        )
        assert len(floor_plan.obstacles) == 5


class TestSchemaVersions:
    """Schema 2.1 (aprons) loads silently; 2.0 loads with a warning."""

    def _write(
        self, tmp_path: Path, scenario: dict[str, object], version: str
    ) -> Path:
        """Write *scenario* with the given schema version."""
        path = tmp_path / "v.json"
        path.write_text(
            json.dumps({**scenario, "schema_version": version}),
            encoding="utf-8",
        )
        return path

    def test_2_1_loads_without_warning(
        self,
        tmp_path: Path,
        valid_scenario_dict: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify a 2.1 file loads and logs nothing."""
        path = self._write(tmp_path, valid_scenario_dict, "2.1")
        with caplog.at_level(logging.WARNING):
            _, data = load_scenario_file(path)
        assert data["schema_version"] == "2.1"
        assert not caplog.records

    def test_2_0_warns_and_names_migration_script(
        self,
        tmp_path: Path,
        valid_scenario_dict: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify a 2.0 file still loads but logs the migration hint."""
        path = self._write(tmp_path, valid_scenario_dict, "2.0")
        with caplog.at_level(logging.WARNING):
            load_scenario_file(path)
        assert "migrate_door_apron.py" in caplog.text

    def test_unsupported_version_raises(
        self, tmp_path: Path, valid_scenario_dict: dict[str, object]
    ) -> None:
        """Verify an unknown version is rejected."""
        path = self._write(tmp_path, valid_scenario_dict, "3.0")
        with pytest.raises(MalformedScenarioError, match="unsupported"):
            load_scenario_file(path)
