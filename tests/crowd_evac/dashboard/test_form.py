"""Tests for crowd_evac.dashboard.form.

Covers:
  - defaults, blank seed, single vs Monte Carlo config
  - hazard rows: fixed vs range, blank max, seconds to ticks, reversed range
  - field-specific errors and the hazard limit
  - add/remove helpers, type prefill, room check
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from crowd_evac.adapters.io.hazard_catalog import HazardType
from crowd_evac.dashboard.form import (
    MAX_HAZARDS,
    FormError,
    FormFields,
    HazardFields,
    add_hazard,
    build_config,
    hazard_from_type,
    remove_hazard,
    room_errors,
)
from crowd_evac.montecarlo.config import ValueRange

_ROOMS = ("lecture_hall", "fixture_minimal")


def _row(**overrides: object) -> HazardFields:
    """A valid fixed hazard row with overrides."""
    base = HazardFields(x_min="2", y_min="3", t_min="1")
    return replace(base, **overrides)  # type: ignore[arg-type]


def _build(fields: FormFields, mc: bool = True) -> object:
    """Build with the fixed room list."""
    return build_config(fields, monte_carlo=mc, scenarios=_ROOMS)


def _errors(fields: FormFields, mc: bool = True) -> dict[str, str]:
    """Return the errors build_config raises."""
    with pytest.raises(FormError) as info:
        build_config(fields, monte_carlo=mc, scenarios=_ROOMS)
    return info.value.errors


class TestDefaults:
    """The issue's defaults."""

    def test_monte_carlo_defaults(self) -> None:
        """Verify lecture_hall, 1 person, headless, no seed, 20 runs."""
        cfg = build_config(FormFields(), monte_carlo=True, scenarios=_ROOMS)
        assert (cfg.scenario, cfg.agents, cfg.runs) == ("lecture_hall", 1, 20)
        assert cfg.visual is False
        assert cfg.seed is None
        assert cfg.hazards == ()
        assert cfg.direct_seed is False

    def test_single_run_uses_direct_seed(self) -> None:
        """Verify single run is one run with the exact seed."""
        cfg = build_config(
            FormFields(seed="7", iterations="500"),
            monte_carlo=False, scenarios=_ROOMS,
        )
        assert cfg.runs == 1
        assert cfg.direct_seed is True
        assert cfg.seed == 7

    def test_blank_seed_is_random(self) -> None:
        """Verify a whitespace seed means None."""
        cfg = build_config(
            FormFields(seed="  "), monte_carlo=True, scenarios=_ROOMS
        )
        assert cfg.seed is None

    def test_visual_and_workers_pass_through(self) -> None:
        """Verify mode and worker count reach the config."""
        cfg = build_config(
            FormFields(visual=True), monte_carlo=True,
            scenarios=_ROOMS, max_workers=3,
        )
        assert cfg.visual is True and cfg.max_workers == 3

    def test_iterations_ignored_for_single_run(self) -> None:
        """Verify a bad iterations text does not block a single run."""
        cfg = build_config(
            FormFields(iterations="abc"), monte_carlo=False, scenarios=_ROOMS
        )
        assert cfg.runs == 1


class TestScalarErrors:
    """Field-specific messages."""

    @pytest.mark.parametrize(
        ("fields", "key"),
        [
            (FormFields(persons="0"), "persons"),
            (FormFields(persons="x"), "persons"),
            (FormFields(persons="1.5"), "persons"),
            (FormFields(seed="-1"), "seed"),
            (FormFields(seed="abc"), "seed"),
            (FormFields(iterations="0"), "iterations"),
            (FormFields(scenario="nope"), "scenario"),
        ],
        ids=["zero", "text", "float", "neg-seed", "text-seed", "zero-runs",
             "room"],
    )
    def test_bad_field_names_its_key(
        self, fields: FormFields, key: str
    ) -> None:
        """Verify each bad field reports under its own key."""
        assert key in _errors(fields)

    def test_all_errors_reported_together(self) -> None:
        """Verify several bad fields are all listed at once."""
        errors = _errors(FormFields(persons="0", seed="z"))
        assert {"persons", "seed"} <= set(errors)


class TestHazardRows:
    """Hazard row parsing."""

    def test_fixed_hazard(self) -> None:
        """Verify blank max gives fixed values and seconds become ticks."""
        cfg = build_config(
            FormFields(hazards=(_row(),)),
            monte_carlo=True, scenarios=_ROOMS,
        )
        spec = cfg.hazards[0]
        assert (spec.x, spec.y, spec.tick) == (2.0, 3.0, 20)

    def test_equal_max_means_fixed(self) -> None:
        """Verify max equal to min collapses to a fixed value."""
        cfg = build_config(
            FormFields(hazards=(_row(x_max="2"),)),
            monte_carlo=True, scenarios=_ROOMS,
        )
        assert cfg.hazards[0].x == 2.0

    def test_ranges(self) -> None:
        """Verify min/max become ValueRange; time is converted to ticks."""
        row = _row(x_max="5", y_max="6", t_min="0", t_max="2.5")
        cfg = build_config(
            FormFields(hazards=(row,)), monte_carlo=True, scenarios=_ROOMS
        )
        spec = cfg.hazards[0]
        assert spec.x == ValueRange(2.0, 5.0)
        assert spec.y == ValueRange(3.0, 6.0)
        assert spec.tick == ValueRange(0, 50)

    def test_type_and_parameters(self) -> None:
        """Verify type and numeric fields reach the spec; blank stays None."""
        row = _row(source_type="smoke", radius="4", blocks_navigation=False)
        spec = build_config(
            FormFields(hazards=(row,)), monte_carlo=True, scenarios=_ROOMS
        ).hazards[0]
        assert spec.source_type == "smoke"
        assert spec.radius == 4.0
        assert spec.intensity is None
        assert spec.blocks_navigation is False

    @pytest.mark.parametrize(
        ("row", "key"),
        [
            (_row(x_min="5", x_max="2"), "hazard.0.x"),
            (_row(y_min="abc"), "hazard.0.y"),
            (_row(x_min=""), "hazard.0.x"),
            (_row(t_min="-1"), "hazard.0"),
            (_row(radius="-3"), "hazard.0"),
            (_row(intensity="2"), "hazard.0"),
            (_row(t_max="x"), "hazard.0.time"),
            (_row(radius="big"), "hazard.0.radius"),
            (_row(x_min="nan"), "hazard.0.x"),
        ],
        ids=["reversed", "text", "blank-min", "neg-time", "neg-radius",
             "intensity", "bad-max", "text-radius", "nan"],
    )
    def test_bad_row_names_its_field(self, row: HazardFields, key: str) -> None:
        """Verify each bad hazard field reports under hazard.<i>.<field>."""
        assert key in _errors(FormFields(hazards=(row,)))

    def test_second_row_indexed(self) -> None:
        """Verify the error key carries the row index."""
        fields = FormFields(hazards=(_row(), _row(y_min="")))
        assert "hazard.1.y" in _errors(fields)

    def test_limit(self) -> None:
        """Verify more than MAX_HAZARDS rows is rejected."""
        fields = FormFields(hazards=(_row(),) * (MAX_HAZARDS + 1))
        assert "hazards" in _errors(fields)


class TestRowHelpers:
    """add / remove / prefill."""

    def test_add_until_full(self) -> None:
        """Verify rows are added up to the limit and no further."""
        fields = FormFields()
        for _ in range(MAX_HAZARDS + 2):
            fields = add_hazard(fields, _row())
        assert len(fields.hazards) == MAX_HAZARDS

    def test_remove(self) -> None:
        """Verify removal by index and that a bad index is ignored."""
        fields = FormFields(hazards=(_row(x_min="1"), _row(x_min="2")))
        assert remove_hazard(fields, 0).hazards[0].x_min == "2"
        assert remove_hazard(fields, 5) is fields

    def test_prefill_from_type(self) -> None:
        """Verify a type fills name and numeric defaults, not position."""
        row = hazard_from_type(
            HazardType(name="smoke", label="Smoke", radius=4.5,
                       blocks_navigation=False)
        )
        assert row.source_type == "smoke"
        assert row.radius == "4.5"
        assert row.blocks_navigation is False
        assert row.x_min == "" and row.y_min == ""


class TestRoomErrors:
    """The pre-start room check."""

    def test_inside_room_is_fine(self) -> None:
        """Verify an in-room hazard gives no error."""
        cfg = build_config(
            FormFields(scenario="fixture_minimal", hazards=(_row(),)),
            monte_carlo=True, scenarios=_ROOMS,
        )
        assert room_errors(cfg) == {}

    def test_outside_room_reported(self) -> None:
        """Verify a hazard beyond the room edge is reported at once."""
        cfg = build_config(
            FormFields(
                scenario="fixture_minimal", hazards=(_row(x_min="500"),)
            ),
            monte_carlo=True, scenarios=_ROOMS,
        )
        assert "form" in room_errors(cfg)


class TestDescribeErrors:
    """Status-line text."""

    def test_hazard_errors_are_numbered_from_one(self) -> None:
        """Verify hazard.N keys become 'Hazard N+1:' lines."""
        from crowd_evac.dashboard.form import describe_errors

        text = describe_errors(
            {"persons": "Persons must be >= 1", "hazard.1.x": "x bad"}
        )
        assert text == "Persons must be >= 1\nHazard 2: x bad"

    def test_form_level_and_empty(self) -> None:
        """Verify a whole-form message passes through and none gives ''."""
        from crowd_evac.dashboard.form import describe_errors

        assert describe_errors({"form": "outside room"}) == "outside room"
        assert describe_errors({}) == ""
