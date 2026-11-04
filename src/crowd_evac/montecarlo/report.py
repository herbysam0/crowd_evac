"""Console report of a Monte Carlo batch: tables only, no graph.

The graph data stays in :attr:`MonteCarloResult.curves` for the dashboard.
"""
from __future__ import annotations

from collections.abc import Sequence

from crowd_evac.montecarlo.config import ValueRange
from crowd_evac.montecarlo.results import MonteCarloResult, RunOutcome

_COLUMN_GAP: str = "  "
_NOT_AVAILABLE: str = "n/a"
_NO_TIME: str = "-"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-aligned columns separated by two spaces, no trailing blanks."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    return [
        _COLUMN_GAP.join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
        for row in (headers, *rows)
    ]


def _seconds(value: float | None, missing: str = _NOT_AVAILABLE) -> str:
    """Format seconds to two places, or ``missing`` for ``None``."""
    return missing if value is None else f"{value:.2f}"


def _run_rows(result: MonteCarloResult) -> list[list[str]]:
    """One table row per finished run."""
    show_hazards = bool(result.config.hazards)
    rows: list[list[str]] = []
    for o in result.outcomes:
        row = [
            str(o.run_index + 1), str(o.seed), f"{o.evacuated}/{o.total}",
            _seconds(o.last_evac_s, _NO_TIME),
        ]
        if show_hazards:
            row.append(f"{o.hazards_fired}/{len(o.hazards)}")
        rows.append(row)
    return rows


def _sampled_lines(result: MonteCarloResult) -> list[str]:
    """One line per run naming each hazard's sampled position and tick."""
    if not any(
        isinstance(v, ValueRange)
        for spec in result.config.hazards
        for v in (spec.x, spec.y, spec.tick)
    ):
        return []

    def describe(outcome: RunOutcome) -> str:
        """Join the run's hazards into one line."""
        parts = [
            f"hazard {i} at ({h['pos'][0]:.2f}, {h['pos'][1]:.2f}) "
            f"tick {h['tick']}"
            for i, h in enumerate(outcome.hazards, start=1)
        ]
        return f"Run {outcome.run_index + 1}: " + "; ".join(parts)

    return [describe(o) for o in result.outcomes]


def _summary_lines(result: MonteCarloResult) -> list[str]:
    """The time-to-last-evacuation block."""
    s = result.summary
    noun = "run" if s.complete == 1 else "runs"
    title = f"Time to last evacuation ({s.complete} complete {noun})"
    if s.complete == 0:
        return [title, "No run evacuated everyone."]
    row = [_seconds(s.mean_s), _seconds(s.std_s), _seconds(s.min_s),
           _seconds(s.max_s)]
    return [title, *_table(
        ["Mean (s)", "Std dev (s)", "Min (s)", "Max (s)"], [row]
    )]


def format_report(result: MonteCarloResult) -> str:
    """Render a batch as console text.

    Args:
        result: A finished or cancelled batch.

    Returns:
        The report: settings, base seed, one row per run, a line per run of
        sampled hazard positions (only when some hazard is randomized),
        time-to-last-evacuation statistics over complete runs, and the
        incomplete-run count and the elapsed wall-clock time.  Ends with a
        newline.
    """
    c = result.config
    mode = "visual" if c.visual else "headless"
    label = " (random)" if result.seed_was_random else ""
    headers = ["Run", "Seed", "Evacuated", "Last evac (s)"]
    if c.hazards:
        headers.append("Hazards fired")
    lines = [
        f"Scenario: {c.scenario}  Agents: {c.agents}  Runs: {c.runs}  "
        f"Mode: {mode}",
        f"Base seed: {result.base_seed}{label}",
        "",
        *_table(headers, _run_rows(result)),
    ]
    sampled = _sampled_lines(result)
    if sampled:
        lines += ["", *sampled]
    lines += ["", *_summary_lines(result), ""]
    total = len(result.outcomes)
    lines.append(f"Incomplete runs: {result.summary.incomplete} of {total}")
    lines.append(f"Elapsed: {result.elapsed_s:.2f} s")
    if result.cancelled:
        lines.append(
            f"Stopped early: {total} of {c.runs} runs finished"
        )
    return "\n".join(lines) + "\n"
