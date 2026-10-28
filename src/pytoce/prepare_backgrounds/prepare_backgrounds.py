"""Prepare the background scenarios used by the LCA calculations."""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from pytoce.shared.runtime.manifests import OperationStatus

if TYPE_CHECKING:
    from .preparation.prepared_state.member_preparation import PreparedBackgroundResult


@dataclass(frozen=True, slots=True)
class PrepareBackgroundsReport:
    """Result of :func:`pytoce.prepare_backgrounds`.

    Attributes:
        status: ``computed`` when every background is newly computed, ``reused``
            when every background is reused, or ``updated`` for a mixture.
            Completed backgrounds are reused when other selected backgrounds are prepared.
        workspace: Active workspace root.
        elapsed_seconds: Time used to prepare the background scenarios.
        results: One result for each requested background scenario.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    results: tuple["PreparedBackgroundResult", ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize paths as text and background results as mappings."""

        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "results": [item.as_dict() for item in self.results],
        }


def _aggregate_status(
    results: tuple["PreparedBackgroundResult", ...],
) -> OperationStatus:
    """Return the aggregate status for the supplied records."""
    statuses = {item.status for item in results}
    if statuses == {"computed"}:
        return "computed"
    if statuses == {"reused"}:
        return "reused"
    return "updated"


def prepare_backgrounds(
    *,
    background_scenario: str | Sequence[str] = (
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
        "remind_ssp2_pkbudg1000",
        "remind_ssp2_pkbudg650",
    ),
    premise_key: str | None = None,
) -> PrepareBackgroundsReport:
    """Prepare background inventories for time-explicit prospective LCA.

    The function imports ecoinvent 3.12 Allocation, cut-off by classification
    into Brightway and uses premise 2.4.6 to generate prospective background
    inventories. By default, it prepares all four REMIND SSP2 background
    scenarios available in this premise version, covering climate policy
    pathways ranging from below 1.5C to above 3C of warming by 2100.
    Completed background scenarios are reused.

    Args:
        background_scenario: One background scenario name, a sequence, or
            ``"all"``. The default selects ``"remind_ssp2_ndc"``,
            ``"remind_ssp2_npi"``, ``"remind_ssp2_pkbudg1000"``, and
            ``"remind_ssp2_pkbudg650"``. Historical REMIND-EU names select
            previously prepared backgrounds.
        premise_key: 44-byte Fernet key supplied as text, required to decrypt
            the official integrated assessment model (IAM) files when
            preparing new backgrounds. Defaults to ``None``, which can be
            used when all selected backgrounds are already prepared. Request
            the key from the premise developers as described at
            https://premise.readthedocs.io/en/latest/getting_started/preparation.html.

    Returns:
        ``PrepareBackgroundsReport`` describing completion status, elapsed
        time in seconds, and the directory and manifest for each requested
        background scenario.

    Raises:
        RuntimeError: If no active workspace has been selected, scientific
            dependency versions are incompatible, prepared data are
            incomplete or inconsistent, or Brightway preparation fails.
        FileNotFoundError: If required ecoinvent or PB-LCIA inputs are missing,
            or a selected historical REMIND-EU background is unavailable.
        ValueError: If a background selection, premise key, source file, or
            inventory provider is invalid.
        OSError: If input or prepared background files cannot be read or
            written.

    Notes:
        Call ``set_workspace()`` and ``download_data()`` before preparing new
        backgrounds.

        Ecoinvent 3.12 is used as the general background system through 2025.
        Linear interpolation provides a gradual transition between ecoinvent
        in 2025 and premise in 2030 for 2026--2029. From 2030 onward,
        intermediate years are represented by interpolation between
        successive premise databases. Specific foreground technologies
        required before 2030 use inventories from premise 2020 and 2025.

    Example:
        Prepare the four default backgrounds using a premise key stored in an
        environment variable::

            import os
            from pytoce import prepare_backgrounds

            report = prepare_backgrounds(premise_key=os.environ["PREMISE_KEY"])
    """

    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    from .preparation.premise import (
        GENERAL_BACKGROUND_ANCHOR_YEARS,
        LEGACY_BACKGROUND_IDS,
        select_background_scenarios,
    )
    from .preparation.prepared_state.batch_preparation import prepare_selected
    from .preparation.prepared_state.member_preparation import (
        selected_prepared_background,
    )
    from .preparation.prepared_state.source_authority import resolve_sources

    started = perf_counter()
    root = get_active_workspace()
    if premise_key is not None and (
        not isinstance(premise_key, str) or len(premise_key.encode("utf-8")) != 44
    ):
        raise ValueError(
            "premise_key must be the 44-byte Fernet key supplied with the premise IAM data. "
            "Pass that key, or pass None when the installed premise data do not require it."
        )
    scenarios = select_background_scenarios(background_scenario)
    completed_by_name = {}
    for scenario in scenarios:
        selected = selected_prepared_background(
            root,
            scenario.background_scenario,
        )
        if selected is not None:
            completed_by_name[scenario.background_scenario] = selected
    for name in completed_by_name:
        print(
            f"prepare_backgrounds: {name}: reusing the completed background scenario",
            file=sys.stderr,
        )
    pending = tuple(
        scenario for scenario in scenarios if scenario.background_scenario not in completed_by_name
    )
    legacy_pending = tuple(
        scenario.background_scenario
        for scenario in pending
        if scenario.background_scenario in LEGACY_BACKGROUND_IDS
    )
    if legacy_pending:
        raise FileNotFoundError(
            "The selected historical REMIND-EU background scenario is not available: "
            f"{legacy_pending!r}. Choose a completed historical scenario or one of the "
            "four current REMIND SSP2 scenarios."
        )
    if pending:
        print(
            "prepare_backgrounds: preparing ecoinvent and LCIA data",
            file=sys.stderr,
        )
        sources = resolve_sources(root)
        created = prepare_selected(
            root,
            sources,
            pending,
            years=GENERAL_BACKGROUND_ANCHOR_YEARS,
            premise_key=premise_key,
        )
        completed_by_name.update({item.background_scenario: item for item in created})
    results = tuple(completed_by_name[scenario.background_scenario] for scenario in scenarios)
    return PrepareBackgroundsReport(
        status=_aggregate_status(results),
        workspace=root,
        elapsed_seconds=perf_counter() - started,
        results=results,
    )
