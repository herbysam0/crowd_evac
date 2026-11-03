"""Save and load flows for the designer window (no GL dependencies).

Both flows guard against data loss with a second-click confirmation: Save
before overwriting an existing file, Load before discarding unsaved edits.
"""
from __future__ import annotations

from pathlib import Path

from crowd_evac.adapters.io.scenario_catalog import SCENARIOS_DIR
from crowd_evac.adapters.io.scenario_loader import load_scenario_file
from crowd_evac.adapters.io.scenario_writer import (
    scenario_exists,
    scenario_path,
    write_scenario,
)
from crowd_evac.designer.controller import DesignController
from crowd_evac.designer.export import build_validated_scenario
from crowd_evac.designer.importer import layout_from_scenario
from crowd_evac.designer.layout import Layout
from crowd_evac.domain.errors import CrowdEvacError

_SUFFIX = ".json"
_NO_EXIT_HINT = "at least one exit"

# (name, layout) awaiting a second click to confirm.
Pending = tuple[str, Layout]


def list_scenarios(directory: Path = SCENARIOS_DIR) -> list[str]:
    """Return the names of saved scenarios in ``directory``, sorted."""
    if not directory.is_dir():
        return []
    return sorted(
        p.name[: -len(_SUFFIX)]
        for p in directory.iterdir()
        if p.is_file() and p.name.endswith(_SUFFIX)
    )


def save_design(
    ctl: DesignController,
    pending: Pending | None,
    directory: Path = SCENARIOS_DIR,
) -> tuple[Pending | None, bool]:
    """Validate and save the design, asking before overwriting.

    Args:
        ctl: Controller whose model is saved; ``status`` receives the result.
        pending: Overwrite confirmation armed by a previous Save click.
        directory: Target scenarios directory.

    Returns:
        ``(pending, saved)``: a confirmation to arm when the name exists and
        the user must press Save again with the design unchanged, and
        whether the file was written.
    """
    layout = ctl.model.layout
    name = layout.room.name
    try:
        data = build_validated_scenario(layout)
        exists = scenario_exists(name, directory)
        if exists and pending != (name, layout):
            ctl.status = f"'{name}' exists - press Save again to overwrite"
            ctl.show_alert(
                f"'{name}' ALREADY EXISTS - press Save again to overwrite it"
            )
            return (name, layout), False
        path = write_scenario(data, directory, overwrite=exists)
    except CrowdEvacError as exc:
        ctl.status = _explain(exc)
        ctl.show_alert(f"NOT SAVED - {ctl.status}")
        return None, False
    ctl.status = f"Saved {path.name}"
    ctl.show_notice(f"SAVED  {path}")
    return None, True


def _explain(exc: CrowdEvacError) -> str:
    """Return a user-facing reason for a rejected save."""
    message = str(exc)
    if _NO_EXIT_HINT in message:
        return (
            "the room needs at least one doorway (use the Doorway tool on a "
            "wall at the room edge)"
        )
    return message


def load_design(
    ctl: DesignController,
    name: str,
    pending: Pending | None,
    clean: Layout,
    directory: Path = SCENARIOS_DIR,
) -> tuple[Pending | None, bool]:
    """Load a saved scenario into the designer.

    Args:
        ctl: Controller that receives the loaded design.
        name: Scenario name (without ``.json``).
        pending: Discard confirmation armed by a previous Load click.
        clean: Layout as of the last save or load, to detect unsaved edits.
        directory: Scenarios directory.

    Returns:
        ``(pending, loaded)``: a confirmation to arm when there are unsaved
        edits, and whether the design was replaced. Failures raise an alert
        on the controller and keep the current design.
    """
    layout = ctl.model.layout
    if not name:
        ctl.status = "No scenario selected"
        return None, False
    if layout != clean and pending != (name, layout):
        ctl.show_alert(
            f"UNSAVED CHANGES - press Load again to discard them and open "
            f"'{name}'"
        )
        return (name, layout), False
    return None, _try_load(ctl, name, directory)


def _try_load(ctl: DesignController, name: str, directory: Path) -> bool:
    """Load ``name`` into ``ctl``; raise an alert and return False on error."""
    try:
        plan, data = load_scenario_file(scenario_path(name, directory))
        ctl.load_layout(layout_from_scenario(plan, data))
    except FileNotFoundError:
        ctl.show_alert(f"LOAD FAILED - scenario '{name}' not found")
        return False
    except CrowdEvacError as exc:
        ctl.show_alert(f"LOAD FAILED - '{name}': {exc}")
        return False
    return True
