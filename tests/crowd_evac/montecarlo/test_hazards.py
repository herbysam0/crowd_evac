"""Tests for crowd_evac.montecarlo.config and crowd_evac.montecarlo.hazards.

Covers:
  - parse_hazard_spec: fixed value, range, lo == hi, lo > hi, range on
    another key, bad number, out-of-room bound.
  - sample_events: stays in range, repeatable per seed, wall redraw,
    fully walled range, independence from other runs.
  - MonteCarloConfig validation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from crowd_evac.application.hazard_args import (
    HazardOutOfRoomError,
    HazardSpecError,
    split_pairs,
)
from crowd_evac.domain.constants import GRID_CELL_SIZE
from crowd_evac.domain.floor_plan import Exit, ExitSide, FloorPlan, Wall
from crowd_evac.montecarlo.config import (
    MAX_HAZARD_SAMPLE_ATTEMPTS,
    HazardSpec,
    MonteCarloConfig,
    ValueRange,
)
from crowd_evac.montecarlo.hazards import (
    HazardRangeError,
    check_specs_in_room,
    parse_hazard_spec,
    sample_events,
    validate_spec,
)

_BASE_SEED = 1234
_RANGED = "x=0..3,y=0..9,tick=0..100"


def _room() -> FloorPlan:
    """10 x 10 m room, one east exit, a full-height wall at x in [4, 6]."""
    return FloorPlan(
        width_m=10.0,
        height_m=10.0,
        walls=(Wall(x=4.0, y=0.0, width=2.0, height=10.0),),
        obstacles=(),
        exits=(
            Exit(x=10.0, y=5.0, width_m=2.0, side=ExitSide.EAST,
                 capacity_per_second=5),
        ),
    )


def _mask() -> npt.NDArray[np.bool_]:
    """Walkable mask for :func:`_room`."""
    return _room().walkable_mask(GRID_CELL_SIZE)


class TestSplitPairs:
    """split_pairs is the public key=value splitter."""

    def test_splits_pairs(self) -> None:
        """Verify pairs are split and stripped."""
        assert split_pairs("a=1, b=2") == {"a": "1", "b": "2"}

    def test_rejects_duplicate(self) -> None:
        """Verify duplicate keys are rejected."""
        with pytest.raises(HazardSpecError, match="duplicate"):
            split_pairs("a=1,a=2")

    def test_rejects_missing_equals(self) -> None:
        """Verify a bare token is rejected."""
        with pytest.raises(HazardSpecError, match="key=value"):
            split_pairs("a")


class TestParseHazardSpec:
    """parse_hazard_spec reads fixed values and lo..hi ranges."""

    def test_fixed_values(self) -> None:
        """Verify plain values stay plain."""
        spec = parse_hazard_spec("x=1,y=2,tick=5,source_type=fire")
        assert (spec.x, spec.y, spec.tick) == (1.0, 2.0, 5)
        assert spec.source_type == "fire"

    def test_ranges_for_x_y_tick(self) -> None:
        """Verify x, y and tick each accept a range."""
        spec = parse_hazard_spec("x=10..40,y=2.5..28,tick=0..400")
        assert spec.x == ValueRange(10.0, 40.0)
        assert spec.y == ValueRange(2.5, 28.0)
        assert spec.tick == ValueRange(0, 400)

    def test_equal_bounds_allowed(self) -> None:
        """Verify lo == hi is a valid range."""
        assert parse_hazard_spec("x=3..3,y=1").x == ValueRange(3.0, 3.0)

    def test_optional_fields_carried(self) -> None:
        """Verify fixed optional keys are kept for the event."""
        spec = parse_hazard_spec("x=1,y=1,intensity=0.5,radius=4")
        assert spec.intensity == 0.5
        assert spec.radius == 4.0

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("x=5..1,y=1", "lower bound"),
            ("x=1,y=1,tick=9..2", "lower bound"),
            ("x=1,y=1,radius=1..3", "range"),
            ("x=1,y=1,source_type=a..b", "range"),
            ("x=a..3,y=1", "not a number"),
            ("x=1,y=1,tick=1.5..3", "integer"),
            ("x=1..2..3,y=1", "not a number"),
            ("y=1", "missing required"),
            ("x=1,y=1,intensity=7", "intensity"),
            ("x=1,y=1,tick=-4..3", "tick"),
        ],
    )
    def test_rejects_bad_input(self, text: str, message: str) -> None:
        """Verify malformed hazards raise HazardSpecError."""
        with pytest.raises(HazardSpecError, match=message):
            parse_hazard_spec(text)


class TestValidation:
    """validate_spec and check_specs_in_room reject bad specs."""

    def test_reversed_range_rejected(self) -> None:
        """Verify a ValueRange with lo > hi cannot be built."""
        with pytest.raises(ValueError, match="lower bound"):
            ValueRange(5.0, 1.0)

    def test_bad_intensity_rejected(self) -> None:
        """Verify PanicSource ranges apply to hand-built specs."""
        with pytest.raises(HazardSpecError):
            validate_spec(HazardSpec(x=1.0, y=1.0, intensity=3.0))

    @pytest.mark.parametrize(
        "text", ["x=-1..3,y=1", "x=1,y=2..11", "x=11,y=1", "x=1..11,y=1..2"]
    )
    def test_out_of_room_bounds(self, text: str) -> None:
        """Verify any bound outside the room is a usage error."""
        with pytest.raises(HazardOutOfRoomError):
            check_specs_in_room([parse_hazard_spec(text)], _room())

    def test_room_bounds_inclusive(self) -> None:
        """Verify the whole room, x=0..10, is accepted."""
        check_specs_in_room([parse_hazard_spec("x=0..10,y=0..10")], _room())


class TestSampling:
    """sample_events draws one concrete event per spec per run."""

    def test_stays_in_range(self) -> None:
        """Verify sampled x, y and tick fall inside their bounds."""
        spec = parse_hazard_spec("x=0..3,y=2..8,tick=10..20")
        for run in range(50):
            (event,) = sample_events([spec], _mask(), _BASE_SEED, run)
            x, y = event["pos"]
            assert 0.0 <= x <= 3.0
            assert 2.0 <= y <= 8.0
            assert 10 <= event["tick"] <= 20

    def test_same_seed_repeats(self) -> None:
        """Verify identical inputs give identical events."""
        spec = parse_hazard_spec(_RANGED)
        first = sample_events([spec], _mask(), _BASE_SEED, 3)
        assert first == sample_events([spec], _mask(), _BASE_SEED, 3)

    def test_runs_differ(self) -> None:
        """Verify different run indices give different events."""
        spec = parse_hazard_spec(_RANGED)
        ticks = {
            sample_events([spec], _mask(), _BASE_SEED, run)[0]["tick"]
            for run in range(20)
        }
        assert len(ticks) > 1

    def test_base_seed_changes_events(self) -> None:
        """Verify another base seed gives another sample."""
        spec = parse_hazard_spec(_RANGED)
        one = sample_events([spec], _mask(), 1, 0)
        two = sample_events([spec], _mask(), 2, 0)
        assert one != two

    def test_fixed_spec_is_not_sampled(self) -> None:
        """Verify an all-fixed hazard passes through unchanged."""
        spec = parse_hazard_spec("x=1,y=2,tick=7,intensity=0.5")
        (event,) = sample_events([spec], _mask(), _BASE_SEED, 0)
        assert event["pos"] == [1.0, 2.0]
        assert event["tick"] == 7
        assert event["intensity"] == 0.5
        assert event["type"] == "place_panic_source"

    def test_fixed_position_in_wall_accepted(self) -> None:
        """Verify a fixed position inside a wall is kept."""
        spec = parse_hazard_spec("x=5,y=5,tick=0..10")
        (event,) = sample_events([spec], _mask(), _BASE_SEED, 0)
        assert event["pos"] == [5.0, 5.0]

    def test_redraws_out_of_walls(self) -> None:
        """Verify a range overlapping a wall never lands inside it."""
        spec = parse_hazard_spec("x=0..10,y=0..10")
        mask = _mask()
        for run in range(100):
            (event,) = sample_events([spec], mask, _BASE_SEED, run)
            assert not 4.0 <= event["pos"][0] < 6.0

    def test_fully_walled_range_raises(self) -> None:
        """Verify a range with no walkable point raises after the cap."""
        spec = parse_hazard_spec("x=4.5..5.5,y=1..9")
        with pytest.raises(HazardRangeError, match=str(
                MAX_HAZARD_SAMPLE_ATTEMPTS)):
            sample_events([spec], _mask(), _BASE_SEED, 0)

    def test_run_independent_of_other_runs(self) -> None:
        """Verify run k's hazard does not depend on sampling other runs."""
        spec = parse_hazard_spec(_RANGED)
        direct = sample_events([spec], _mask(), _BASE_SEED, 5)
        for run in range(5):
            sample_events([spec], _mask(), _BASE_SEED, run)
        assert sample_events([spec], _mask(), _BASE_SEED, 5) == direct

    def test_no_specs_gives_no_events(self) -> None:
        """Verify an empty spec list gives an empty tuple."""
        assert sample_events([], _mask(), _BASE_SEED, 0) == ()


class TestConfig:
    """MonteCarloConfig defaults and validation."""

    def test_defaults(self) -> None:
        """Verify defaults match the CLI table."""
        config = MonteCarloConfig()
        assert config.scenario == "lecture_hall"
        assert config.agents == 1
        assert config.runs == 20
        assert config.visual is False
        assert config.seed is None
        assert config.hazards == ()

    @pytest.mark.parametrize(
        "kwargs",
        [{"agents": 0}, {"runs": 0}, {"seed": -1}, {"max_workers": 0}],
    )
    def test_rejects_invalid(self, kwargs: dict[str, int]) -> None:
        """Verify out-of-range settings raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(**kwargs)  # type: ignore[arg-type]
