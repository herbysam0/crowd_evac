"""Tests for scenario selection in crowd_evac.application.cli."""
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from crowd_evac.application import cli
from crowd_evac.application.cli import DEFAULT_SCENARIO, main, parse_args


class TestParseArgs:
    """Suite for parse_args."""

    def test_no_args_selects_default(self) -> None:
        """Verify omitted flag keeps the Lecture Hall default."""
        args = parse_args([])
        assert args.scenario == DEFAULT_SCENARIO
        assert args.list is False

    @pytest.mark.parametrize("flag", ["--scenario", "-s"])
    def test_scenario_flag_selects_bundled(self, flag: str) -> None:
        """Verify long and short flags select a bundled scenario."""
        assert parse_args([flag, "lecture_hall_small"]).scenario == (
            "lecture_hall_small"
        )

    def test_list_flag_sets_list(self) -> None:
        """Verify --list is parsed."""
        assert parse_args(["--list"]).list is True

    def test_unknown_scenario_exits_with_status_2(self) -> None:
        """Verify an unknown name is rejected by argparse."""
        with pytest.raises(SystemExit) as exc:
            parse_args(["--scenario", "no_such_scenario"])
        assert exc.value.code == 2


class TestMain:
    """Suite for main() scenario wiring (no window opened)."""

    def test_list_prints_names_without_opening_window(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify --list prints names and never builds a simulation."""
        build = mocker.patch.object(cli, "build_simulation_from_scenario")
        main(["--list"])
        assert "lecture_hall" in capsys.readouterr().out.splitlines()
        build.assert_not_called()

    def test_main_builds_selected_scenario(self, mocker: MockerFixture) -> None:
        """Verify the chosen scenario reaches build and the window."""
        build = mocker.patch.object(
            cli, "build_simulation_from_scenario", return_value=(object(), object())
        )
        window = mocker.patch.object(cli, "EvacWindow")
        mocker.patch.object(cli.arcade, "run")
        main(["-s", "bottleneck_corridor"])
        build.assert_called_once_with("bottleneck_corridor")
        assert window.call_args.args[2] == "bottleneck_corridor"
