"""Tests for scenario selection in crowd_evac.application.cli."""
from __future__ import annotations

import subprocess
import sys

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

    def test_design_defaults_off(self) -> None:
        """Verify existing invocations are unchanged by the new flag."""
        assert parse_args([]).design is False
        assert parse_args(["-s", "lecture_hall_small"]).design is False

    @pytest.mark.parametrize("flag", ["--design", "-d"])
    def test_design_flag_sets_design(self, flag: str) -> None:
        """Verify long and short flags enable the designer."""
        assert parse_args([flag]).design is True

    @pytest.mark.parametrize(
        "argv",
        [
            ["--design", "--list"],
            ["--design", "--scenario", "lecture_hall_small"],
        ],
        ids=["with-list", "with-scenario"],
    )
    def test_design_cannot_combine_with_run_options(
        self, argv: list[str]
    ) -> None:
        """Verify mixing designer and run options exits with status 2."""
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2

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


class TestMainDesign:
    """Suite for the --design dispatch."""

    def test_design_opens_designer_only(self, mocker: MockerFixture) -> None:
        """Verify --design runs the designer and builds no simulation."""
        run = mocker.patch(
            "crowd_evac.adapters.render.designer_window.run_designer"
        )
        build = mocker.patch.object(cli, "build_simulation_from_scenario")
        window = mocker.patch.object(cli, "EvacWindow")
        main(["--design"])
        run.assert_called_once_with()
        build.assert_not_called()
        window.assert_not_called()

    def test_normal_run_does_not_open_designer(
        self, mocker: MockerFixture
    ) -> None:
        """Verify a scenario run never touches the designer."""
        run = mocker.patch(
            "crowd_evac.adapters.render.designer_window.run_designer"
        )
        mocker.patch.object(
            cli, "build_simulation_from_scenario", return_value=(object(), object())
        )
        mocker.patch.object(cli, "EvacWindow")
        mocker.patch.object(cli.arcade, "run")
        main([])
        run.assert_not_called()

    def test_cli_import_does_not_load_designer(self) -> None:
        """Verify the designer is imported lazily, not with the CLI."""
        code = (
            "import sys, crowd_evac.application.cli;"
            "print(any('designer' in m for m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"
