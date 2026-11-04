"""MonteCarloListener port -- progress callbacks for a Monte Carlo batch."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crowd_evac.montecarlo.results import RunOutcome


class MonteCarloListener(Protocol):
    """Receive progress while :func:`run_monte_carlo` works.

    A dashboard implements this to draw a progress bar and to add each run
    to its graph as it finishes.
    """

    def on_run_complete(
        self, outcome: RunOutcome, done: int, total: int
    ) -> None:
        """Handle one finished run.

        Args:
            outcome: The run that just finished.
            done: Runs finished so far, including this one.
            total: Runs in the batch.  Headless runs may finish out of order.
        """
        ...
