"""Tests for crowd_evac.application.hazard_args and the run-option flags.

Covers:
  - parse_hazard: minimal, all keys, unknown key, duplicate key, bad number,
    missing x/y, negative tick, out-of-range value.
  - check_hazards_in_room: inside, boundary, outside.
  - parse_args: defaults, --agents/--seed/--hazard, rejection rules, and the
    rule that run flags cannot be combined with --design or --list.
"""
from __future__ import annotations

import pytest

from crowd_evac.application.cli_args import parse_args
from crowd_evac.application.hazard_args import (
    HazardOutOfRoomError,
    HazardSpecError,
    check_hazards_in_room,
    parse_hazard,
)
from crowd_evac.domain.constants import DEFAULT_AGENT_COUNT, DEFAULT_SPAWN_SEED
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan

_FULL_SPEC = (
    "x=25,y=3.5,tick=100,intensity=0.5,radius=4,decay_rate=0.1,"
    "block_radius=0.8,blocks_navigation=false,source_type=smoke"
)


class TestParseHazard:
    """parse_hazard builds a scripted event from key=value text."""

    def test_minimal_spec(self) -> None:
        """Verify only x and y are required; tick defaults to 0."""
        assert parse_hazard("x=1,y=2") == {
            "tick": 0, "type": "place_panic_source", "pos": [1.0, 2.0],
        }

    def test_all_keys(self) -> None:
        """Verify every optional key is converted to its type."""
        event = parse_hazard(_FULL_SPEC)
        assert event == {
            "tick": 100,
            "type": "place_panic_source",
            "pos": [25.0, 3.5],
            "intensity": 0.5,
            "radius": 4.0,
            "decay_rate": 0.1,
            "block_radius": 0.8,
            "blocks_navigation": False,
            "source_type": "smoke",
        }

    def test_old_hazard_file_equivalent(self) -> None:
        """Verify the README command reproduces the old scenario event."""
        assert parse_hazard("x=25,y=3.5,tick=100,source_type=fire") == {
            "tick": 100,
            "type": "place_panic_source",
            "pos": [25.0, 3.5],
            "source_type": "fire",
        }

    @pytest.mark.parametrize("word", ["true", "TRUE", "1", "yes"])
    def test_true_words(self, word: str) -> None:
        """Verify truthy spellings parse to True."""
        assert parse_hazard(f"x=1,y=1,blocks_navigation={word}")[
            "blocks_navigation"
        ] is True

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("x=1,y=2,speed=3", "unknown key 'speed'"),
            ("x=1,x=2,y=2", "duplicate key 'x'"),
            ("x=abc,y=2", "not a number"),
            ("x=1", "missing required key 'y'"),
            ("y=1", "missing required key 'x'"),
            ("x=1,y=2,tick=-1", "tick must be >= 0"),
            ("x=1,y=2,tick=1.5", "not an integer"),
            ("x=1,y=2,intensity=2", "intensity must be in"),
            ("x=1,y=2,radius=0", "radius must be positive"),
            ("x=1,y=2,decay_rate=-1", "decay_rate must be >= 0"),
            ("x=nan,y=2", "must be finite"),
            ("x=1,y=2,blocks_navigation=maybe", "true or false"),
            ("x=1,y=2,source_type=", "must not be empty"),
            ("x=1,y", "expected key=value"),
            ("", "expected key=value"),
        ],
    )
    def test_rejects_bad_specs(self, spec: str, message: str) -> None:
        """Verify each malformed spec fails with a specific message."""
        with pytest.raises(HazardSpecError, match=message):
            parse_hazard(spec)


@pytest.fixture
def room() -> FloorPlan:
    """Return a 10 x 8 m room."""
    return FloorPlan(
        width_m=10.0, height_m=8.0, walls=(), obstacles=(),
        exits=(Exit(10.0, 4.0, 2.0, ExitSide.EAST, 5, "east"),),
    )


class TestCheckHazardsInRoom:
    """check_hazards_in_room rejects positions off the floor plan."""

    def test_inside_passes(self, room: FloorPlan) -> None:
        """Verify an interior position passes."""
        check_hazards_in_room([parse_hazard("x=5,y=4")], room)

    def test_boundary_passes(self, room: FloorPlan) -> None:
        """Verify the room edge counts as inside."""
        check_hazards_in_room([parse_hazard("x=10,y=8")], room)

    def test_empty_passes(self, room: FloorPlan) -> None:
        """Verify no hazards is fine."""
        check_hazards_in_room([], room)

    @pytest.mark.parametrize("spec", ["x=10.1,y=4", "x=-0.1,y=4", "x=5,y=8.5"])
    def test_outside_raises(self, room: FloorPlan, spec: str) -> None:
        """Verify positions beyond the room raise a clear error."""
        with pytest.raises(HazardOutOfRoomError, match="outside"):
            check_hazards_in_room([parse_hazard(spec)], room)


class TestRunFlags:
    """--agents, --seed and --hazard on parse_args."""

    def test_defaults(self) -> None:
        """Verify 3 agents, seed 42 and no hazards when unspecified."""
        args = parse_args([])
        assert (args.agents, args.seed, args.hazard) == (3, 42, [])
        assert DEFAULT_AGENT_COUNT == 3
        assert DEFAULT_SPAWN_SEED == 42

    def test_agents_and_seed(self) -> None:
        """Verify explicit values are used."""
        args = parse_args(["--agents", "150", "--seed", "7"])
        assert (args.agents, args.seed) == (150, 7)

    def test_hazard_is_repeatable(self) -> None:
        """Verify each --hazard adds an event, in order."""
        args = parse_args(["--hazard", "x=1,y=1", "--hazard", "x=2,y=2,tick=9"])
        assert [e["pos"] for e in args.hazard] == [[1.0, 1.0], [2.0, 2.0]]
        assert args.hazard[1]["tick"] == 9

    @pytest.mark.parametrize(
        "argv",
        [
            ["--agents", "0"],
            ["--agents", "-3"],
            ["--agents", "many"],
            ["--seed", "-1"],
            ["--seed", "x"],
            ["--hazard", "x=1"],
            ["--hazard", "x=1,y=1,bogus=2"],
            ["--hazard", "x=1,y=1,x=2"],
            ["--hazard", "x=1,y=1,tick=-4"],
        ],
    )
    def test_invalid_values_exit_2(self, argv: list[str]) -> None:
        """Verify bad values are argparse errors (status 2)."""
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2

    @pytest.mark.parametrize("mode", ["--design", "--list"])
    @pytest.mark.parametrize(
        "extra",
        [["--agents", "5"], ["--seed", "1"], ["--hazard", "x=1,y=1"]],
        ids=["agents", "seed", "hazard"],
    )
    def test_run_flags_rejected_with_design_or_list(
        self, mode: str, extra: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify run flags cannot be mixed with --design or --list."""
        with pytest.raises(SystemExit) as exc:
            parse_args([mode, *extra])
        assert exc.value.code == 2
        assert mode in capsys.readouterr().err

    def test_design_alone_still_works(self) -> None:
        """Verify --design without run flags is unaffected."""
        assert parse_args(["--design"]).design is True
