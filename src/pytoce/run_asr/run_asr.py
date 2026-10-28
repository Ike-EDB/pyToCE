"""Public coordinator for monitoring absolute environmental sustainability."""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pytoce.shared.runtime.manifests import ItemStatus, OperationStatus


@dataclass(frozen=True, slots=True)
class RunASRItemReport:
    """One foreground/background comparison with the selected aCC cases.

    Attributes:
        status: Whether this result was computed, updated, or reused.
        foreground_scenario: Completed foreground scenario name.
        background_scenario: Completed background scenario name.
        result_directory: Directory containing the assessment results.
        manifest_path: Completion manifest recording the LCA and aCC inputs.
        annual_asr: Annual regional and global ASRs for 2019--2100.
        cumulative_gwp100_asr: Independent cumulative ASRs for 2019--2100.
    """

    status: ItemStatus
    foreground_scenario: str
    background_scenario: str
    result_directory: Path
    manifest_path: Path
    annual_asr: Path
    cumulative_gwp100_asr: Path

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""
        return {
            "status": self.status,
            "foreground_scenario": self.foreground_scenario,
            "background_scenario": self.background_scenario,
            "result_directory": str(self.result_directory),
            "manifest_path": str(self.manifest_path),
            "annual_asr": str(self.annual_asr),
            "cumulative_gwp100_asr": str(self.cumulative_gwp100_asr),
        }


@dataclass(frozen=True, slots=True)
class RunASRReport:
    """Result of :func:`pytoce.run_asr`.

    Attributes:
        status: Computed when every item is new, reused when every item is
            reused, or updated otherwise, including a mixture of item statuses.
        workspace: Active workspace root.
        elapsed_seconds: Time used for the assessment request.
        results: One result per requested foreground/background pair.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    results: tuple[RunASRItemReport, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""
        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "results": [item.as_dict() for item in self.results],
        }


def run_asr(
    *,
    allocated_cc_result: str,
    foreground_scenario: str | Sequence[str] = "remind_mfa_default",
    background_scenario: str | Sequence[str] = (
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
        "remind_ssp2_pkbudg1000",
        "remind_ssp2_pkbudg650",
    ),
    lcia_method: str | Sequence[str] = ("ef31", "pb_lcia"),
    allocated_cc_case: str | Sequence[str] | None = None,
    refresh: bool = False,
) -> RunASRReport:
    """Calculate annual and cumulative Absolute Sustainability Ratios (ASRs).

    The function compares completed LCA environmental burdens with their
    allocated carrying capacities to assess the absolute environmental
    sustainability of user-defined circular economy (CE) performance
    trajectories. Allocated carrying capacities can be calculated through the
    pyaesa tutorial workflow or supplied by the user.

    Annual ASRs divide environmental burdens by the allocated carrying
    capacities for the same region, year, and impact category. An ASR <= 1
    indicates compliance with the allocated carrying capacities, whereas an
    ASR > 1 indicates transgression.

    Args:
        allocated_cc_result: Required directory name under
            ``data_raw/allocated_cc/results``. The directory contains a YAML
            definition and two CSV files providing complete, finite, positive
            annual and cumulative allocated carrying capacities. Follow the
            templates installed under ``data_raw/allocated_cc/templates``.
        foreground_scenario: One foreground scenario name with completed LCA
            results, or a sequence of names. Defaults to
            ``"remind_mfa_default"``.
        background_scenario: One background scenario name with completed LCA
            results, or a sequence of names. Defaults to all four REMIND SSP2
            background scenarios used by ``run_lca()``.
        lcia_method: Method selection identifying the completed LCA results.
            Use the same selection as ``run_lca()``: ``"ef31"`` or both
            ``"ef31"`` and ``"pb_lcia"``. Defaults to ``("ef31", "pb_lcia")``.
            ASRs use the 16 EF 3.1 categories and their corresponding allocated
            carrying capacities.
        allocated_cc_case: One supplied case label, a sequence of labels, or
            ``None``. Defaults to ``None``, which selects every available
            case separately. The source documentation defines the allocation
            method and statistic represented by each label.
        refresh: If ``True``, recalculate the requested ASR results from the
            completed LCA and allocated carrying capacity inputs. Defaults to
            ``False``, which reuses completed results.

    Returns:
        ``RunASRReport`` describing completion status, elapsed time in seconds,
        and the annual ASR table, cumulative GWP100 ASR table, and manifest
        paths for each foreground and background scenario pair. Results are
        written under ``results/<foreground>/asr_results``.

    Raises:
        RuntimeError: If no active workspace has been selected or a result
            writer is locked.
        ValueError: If selections or inputs are invalid, required LCA results
            are missing or ambiguous, or an ASR cannot be represented as a
            finite value.
        OSError: If input or result files cannot be read or written.

    Notes:
        Call ``run_lca()`` with the selected foregrounds, backgrounds, and
        LCIA methods before calculating new ASR results.

        Annual results include 2019 for reporting; compliance with annual
        allocated carrying capacities is assessed over 2020--2100. The
        additional cumulative climate change ASR divides summed all-GHG GWP100
        burdens over 2019--2100 by the cumulative allocated carrying capacity
        supplied for that period. Global ASRs divide the sum of regional
        burdens by the sum of their allocated carrying capacities. ASRs are
        dimensionless. Monitoring and target setting use the same definitions.

        Environmental burdens and allocated carrying capacities must refer to
        the same activity, functional unit, and system boundaries, with
        matching LCIA factors and units. Compiled allocated carrying capacity
        inputs are stored under ``data_processed/allocated_cc/<name>``.
        Supply revised inputs under a new ``allocated_cc_result`` name.

    Example:
        Assess a completed LCA result against a supplied allocated carrying
        capacity input, using defaults where omitted::

            from pytoce import run_asr

            report = run_asr(
                allocated_cc_result="example_allocated_cc",
                background_scenario="remind_ssp2_npi",
            )
    """
    if not isinstance(refresh, bool):
        raise ValueError("refresh must be True to recalculate or False to reuse ASR results")

    from pytoce.run_lca.calculation.lcia import normalized_lcia_method
    from pytoce.run_lca.run_lca import _names
    from pytoce.run_mfa.scenario.scenario_contract import validate_foreground_scenario_name
    from pytoce.run_optimization.input.allocated_cc_paths import _readable_name
    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    from .results import write_asr_result

    started = perf_counter()
    foregrounds = tuple(
        validate_foreground_scenario_name(name)
        for name in _names(foreground_scenario, "foreground_scenario")
    )
    backgrounds = _names(background_scenario, "background_scenario")
    name = _readable_name(allocated_cc_result, field="allocated_cc_result")
    cases = None if allocated_cc_case is None else _names(allocated_cc_case, "allocated_cc_case")
    selection = normalized_lcia_method(lcia_method)
    if "ef31" not in selection.groups:
        raise ValueError("lcia_method must include the complete 'ef31' group used by the aCC input")
    workspace = get_active_workspace()
    results = []
    for foreground in foregrounds:
        for background in backgrounds:
            status, directory = write_asr_result(
                workspace,
                foreground=foreground,
                background=background,
                allocated_cc_result=name,
                cases=cases,
                selection=selection,
                refresh=refresh,
            )
            results.append(
                RunASRItemReport(
                    status,
                    foreground,
                    background,
                    directory,
                    directory / "manifest.json",
                    directory / "annual_asr.parquet",
                    directory / "cumulative_gwp100_asr.parquet",
                )
            )
            print(f"run_asr: {foreground}, {background}: {status}", file=sys.stderr)
    statuses = {item.status for item in results}
    status: OperationStatus = next(iter(statuses)) if len(statuses) == 1 else "updated"
    return RunASRReport(status, workspace, perf_counter() - started, tuple(results))
