"""Tests for crowd_evac.montecarlo.cli and scripts/monte_carlo.py.

Covers: all defaults, rejected values, repeatable ranged hazards, usage
errors for bad rooms/hazards, and a full script run printing the report.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from crowd_evac.montecarlo.cli import main, parse_args
from crowd_evac.montecarlo.config import ValueRange

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "monte_carlo.py"
_USAGE_EXIT = 2


class TestParseArgs:
    """Flags map onto MonteCarloConfig."""

    def test_defaults(self) -> None:
        """Verify every default from the flag table."""
        config = parse_args([])
        assert config.scenario == "lecture_hall"
        assert config.agents == 1
        assert config.runs == 20
        assert config.visual is False
        assert config.seed is None
        assert config.hazards == ()

    def test_all_flags(self) -> None:
        """Verify each flag lands in the config."""
        config = parse_args([
            "-s", "fixture_minimal", "--agents", "150", "--runs", "50",
            "--seed", "7", "--visual",
        ])
        assert config.scenario == "fixture_minimal"
        assert (config.agents, config.runs, config.seed) == (150, 50, 7)
        assert config.visual is True

    def test_hazard_ranges_and_repeat(self) -> None:
        """Verify --hazard repeats and keeps ranges."""
        config = parse_args([
            "--hazard", "x=10..40,y=2..28,tick=0..400,source_type=fire",
            "--hazard", "x=1,y=1",
        ])
        assert len(config.hazards) == 2
        assert config.hazards[0].x == ValueRange(10.0, 40.0)
        assert config.hazards[0].source_type == "fire"
        assert config.hazards[1].x == 1.0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--agents", "0"],
            ["--runs", "0"],
            ["--seed", "-1"],
            ["--agents", "many"],
            ["-s", "no_such_room"],
            ["--hazard", "x=9..1,y=1"],
            ["--hazard", "x=1,y=1,radius=1..2"],
            ["--hazard", "y=1"],
        ],
    )
    def test_rejects_bad_values(self, argv: list[str]) -> None:
        """Verify bad values exit with the argparse usage status."""
        with pytest.raises(SystemExit) as caught:
            parse_args(argv)
        assert caught.value.code == _USAGE_EXIT


class TestMain:
    """main runs a batch and prints the report."""

    def test_usage_error_for_hazard_outside_room(self) -> None:
        """Verify an out-of-room hazard is a usage error, before runs."""
        argv = ["-s", "fixture_minimal", "--runs", "1",
                "--hazard", "x=0..500,y=1"]
        with pytest.raises(SystemExit) as caught:
            main(argv)
        assert caught.value.code == _USAGE_EXIT

    def test_prints_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify a small batch prints header, tables and returns 0."""
        argv = ["-s", "fixture_minimal", "--runs", "2", "--seed", "3"]
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "Scenario: fixture_minimal" in out
        assert "Base seed: 3\n" in out


class TestScript:
    """The thin script entry point."""

    def test_randomized_hazard_prints_full_report(self) -> None:
        """Verify the script runs a ranged hazard and exits 0."""
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "-s", "fixture_minimal",
             "--agents", "2", "--runs", "2", "--seed", "3",
             "--hazard", "x=1..9,y=1..7,tick=0..50"],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        assert done.returncode == 0, done.stderr
        for text in ("Base seed: 3", "Hazards fired", "hazard 1 at",
                     "Mean (s)", "Incomplete runs:", "Elapsed:"):
            assert text in done.stdout
