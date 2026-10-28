"""Calculate and reuse foreground MFA results."""

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.manifests import (
    ItemStatus,
    OperationStatus,
)

if TYPE_CHECKING:
    from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe

    from .model.materials import MaterialComposition
    from .model.mfa import MFAEvaluationSession
    from .remind_mfa.execution import RemindMFAExecutionRequest
    from .scenario.compiler import CompiledForegroundScenario


@dataclass(frozen=True, slots=True)
class MFAResult:
    """One completed foreground MFA result.

    Attributes:
        status: Whether this foreground result was ``computed``, ``updated``,
            or ``reused``.
        foreground_scenario: Readable foreground scenario name.
        ssp_scenario: Shared socioeconomic pathway, always ``SSP2``.
        result_directory: Directory containing the MFA result.
        manifest_path: Path to the result manifest.
        resolved_input_parameters: Path to annual input values used by the MFA.
        derived_annual_outputs: Path to annual values calculated by the MFA.
        waste_trade_propensity_ledger: Path to annual waste trade propensities and amounts.
    """

    status: ItemStatus
    foreground_scenario: str
    ssp_scenario: str
    result_directory: Path
    manifest_path: Path
    resolved_input_parameters: Path
    derived_annual_outputs: Path
    waste_trade_propensity_ledger: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the MFA result with paths represented as text."""

        return {
            "status": self.status,
            "foreground_scenario": self.foreground_scenario,
            "ssp_scenario": self.ssp_scenario,
            "result_directory": str(self.result_directory),
            "manifest_path": str(self.manifest_path),
            "resolved_input_parameters": str(self.resolved_input_parameters),
            "derived_annual_outputs": str(self.derived_annual_outputs),
            "waste_trade_propensity_ledger": str(self.waste_trade_propensity_ledger),
        }


@dataclass(frozen=True, slots=True)
class RunMFAReport:
    """Result of :func:`pytoce.run_mfa`.

    Attributes:
        status: Overall status: ``computed``, ``reused``, or ``updated``.
        workspace: Active workspace root.
        elapsed_seconds: Time used for the MFA request.
        results: One result for each requested foreground scenario.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    results: tuple[MFAResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""

        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "results": [item.as_dict() for item in self.results],
        }


def _scenario_names(foreground_scenario: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize readable scenario names, remove duplicates, and sort them."""
    from .scenario.scenario_contract import ScenarioError, validate_foreground_scenario_name

    try:
        raw = (
            (foreground_scenario,)
            if isinstance(foreground_scenario, str)
            else tuple(foreground_scenario)
        )
    except TypeError as exc:
        raise ScenarioError(
            "foreground_scenario must be one readable name or a sequence of readable names"
        ) from exc
    if not raw:
        raise ScenarioError(
            "foreground_scenario is empty. Select at least one YAML filename stem from "
            "data_raw/foreground_scenarios/scenarios."
        )
    normalized: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ScenarioError(
                f"Invalid foreground_scenario entry {value!r}. Select a non-empty YAML "
                "filename stem from data_raw/foreground_scenarios/scenarios."
            )
        normalized.append(validate_foreground_scenario_name(value.strip()))
    return tuple(sorted(set(normalized)))


def _mfa_result_report(
    foreground_scenario: str,
    result_directory: Path,
    *,
    status: ItemStatus,
) -> MFAResult:
    """Build the public report for one MFA result directory."""
    return MFAResult(
        status=status,
        foreground_scenario=foreground_scenario,
        ssp_scenario="SSP2",
        result_directory=result_directory,
        manifest_path=result_directory / "manifest.json",
        resolved_input_parameters=result_directory / "resolved_annual_input_parameters.parquet",
        derived_annual_outputs=result_directory / "derived_annual_outputs.parquet",
        waste_trade_propensity_ledger=result_directory / "waste_trade_propensity_ledger.parquet",
    )


def _run_one(
    root: Path,
    compiled: "CompiledForegroundScenario",
    *,
    request: "RemindMFAExecutionRequest",
    session: "MFAEvaluationSession",
    recipe: "EcoinventMaterialRecipe",
    composition: "MaterialComposition",
    refresh: bool,
    progress: Callable[[str, int | None], None],
) -> MFAResult:
    """Compute or reuse one foreground scenario's MFA result and manifest."""
    import pandas as pd

    from .model.results import (
        build_mfa_result_target,
        selected_mfa_result,
        validate_mfa_result_for_refresh,
        write_mfa_result,
    )

    target = build_mfa_result_target(
        root,
        compiled,
        request,
        material_recipe_identity=recipe.identity,
        material_composition_identity=composition.identity,
    )
    target.result_directory.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(target.result_directory.parent / ".writer.lock"):
        reference = selected_mfa_result(root, compiled.foreground_scenario)
        if not refresh and reference is not None:
            progress("reuse", None)
            return _mfa_result_report(
                compiled.foreground_scenario,
                target.result_directory,
                status="reused",
            )
        replacing = refresh and reference is not None
        if replacing:
            validate_mfa_result_for_refresh(target)

        progress("load_compiled_inputs", None)
        compiled_table = pd.read_parquet(compiled.compiled_parameters_path)
        computation = session.compute(
            foreground_scenario=compiled.foreground_scenario,
            compiled=compiled_table,
            scenario_change_points=compiled.normalized_change_points,
            progress=progress,
        )

        progress("write_result_files", None)
        status = write_mfa_result(
            target,
            computation,
            replacing=replacing,
        )
        return _mfa_result_report(
            compiled.foreground_scenario,
            target.result_directory,
            status=status,
        )


def _progress_writer(
    foreground_scenario: str,
) -> Callable[[str, int | None], None]:
    """Return live progress output for one MFA result."""

    messages = {
        "reuse": "reusing the completed result",
        "load_compiled_inputs": "reading scenario inputs",
        "write_result_files": "writing result files",
        "compile_scenario": "reading the scenario YAML",
        "source_authority": "reading REMIND-MFA source data",
        "physical_material_realization": "calculating polymer and monomer balances",
        "result_tables": "formatting result tables",
        "cohort_continuation": "calculating stocks and cohorts",
    }

    def write(phase: str, year: int | None) -> None:
        """Print one MFA phase and optional year to standard error."""
        year_text = "" if year is None else f" for {year}"
        print(
            f"run_mfa: {foreground_scenario}: {messages[phase]}{year_text}",
            file=sys.stderr,
        )

    return write


def _aggregate_status(results: tuple[MFAResult, ...]) -> OperationStatus:
    """Return the aggregate status for the supplied records."""
    statuses = {result.status for result in results}
    if statuses == {"computed"}:
        return "computed"
    if statuses == {"reused"}:
        return "reused"
    return "updated"


def run_mfa(
    *,
    foreground_scenario: str | Sequence[str] = "remind_mfa_default",
    refresh: bool = False,
) -> RunMFAReport:
    """Calculate annual plastics stocks and flows through dynamic MFA.

    The function reads foreground scenario YAML files from the selected
    workspace's ``data_raw/foreground_scenarios/scenarios`` directory and
    translates circular economy (CE) performance parameters into annual
    plastics stocks and flows. It uses REMIND-MFA 0.2.0 and its input data
    1.0.0, with the world divided into 12 regions, plastics divided into 12
    families, and seven end-use groups. Foreground scenarios modify the
    plastics system from 2020 onward while retaining historical data through
    2019.

    Args:
        foreground_scenario: One runnable YAML filename stem or a sequence of
            stems. Defaults to ``"remind_mfa_default"``. Duplicate names are
            removed before calculation. ``set_workspace()`` installs examples
            in ``scenarios`` and editable templates in the adjacent
            ``templates`` directory.
        refresh: If ``True``, recompile the selected scenarios and recalculate
            their MFA results. Defaults to ``False``, which reuses completed
            results. Use ``True`` to apply changes to an existing scenario.

    Returns:
        ``RunMFAReport`` describing completion status, elapsed time in seconds,
        and one result per foreground scenario, with paths to the MFA result
        directory, manifest, resolved annual inputs, derived annual outputs,
        and waste trade table.

    Raises:
        RuntimeError: If no active workspace has been selected, a result
            writer is locked, or the plastics MFA calculation fails.
        FileNotFoundError: If required REMIND-MFA files or the derived
            ecoinvent material recipe are missing.
        ValueError: If ``refresh`` is not a Boolean, a scenario is unavailable,
            or scenario inputs, annual schedules, stocks, flows, or
            mass-balance checks are invalid.
        OSError: If input or result files cannot be read or written.

    Notes:
        Call ``set_workspace()`` and ``download_data()`` before calculating
        new MFA results.

        In each year, the stock remaining in use from previous cohorts is
        retained. The new plastics inflow is calculated to meet the in-use
        stock target after accounting for these surviving cohorts.
        Collection and treatment parameters determine the end-of-life flows.
        The resulting secondary material contributes to supplying the
        required new inflow, while the remaining requirement is met by virgin
        production according to the selected virgin feedstock shares.

    Example:
        Calculate the default foreground scenario, using defaults where
        omitted::

            from pytoce import run_mfa

            report = run_mfa(foreground_scenario="remind_mfa_default")
    """

    if not isinstance(refresh, bool):
        raise ValueError(
            f"refresh={refresh!r} is invalid. Use refresh=True to recalculate selected "
            "MFA results or refresh=False to reuse completed results."
        )

    from pytoce.download_data.sources.ecoinvent_materials import load_material_recipe
    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    from .model.materials import load_material_composition
    from .model.mfa import MFAEvaluationSession
    from .model.results import selected_mfa_result
    from .remind_mfa.execution import prepare_execution, prepare_workspace_source
    from .scenario.catalogue import load_parameter_catalogue
    from .scenario.compiler import CompiledForegroundScenario, compile_foreground_scenario

    started = perf_counter()
    root = get_active_workspace()
    names = _scenario_names(foreground_scenario)
    completed_by_name: dict[str, MFAResult] = {}
    for name in names:
        reference = selected_mfa_result(root, name)
        if not refresh and reference is not None:
            _progress_writer(name)("reuse", None)
            completed_by_name[name] = _mfa_result_report(
                name,
                reference.parent,
                status="reused",
            )
    pending_names = tuple(name for name in names if name not in completed_by_name)
    if not pending_names:
        results = tuple(completed_by_name[name] for name in names)
        return RunMFAReport(
            status="reused",
            workspace=root,
            elapsed_seconds=perf_counter() - started,
            results=results,
        )
    recipe = load_material_recipe(root)
    composition = load_material_composition()
    compiled_items: list[CompiledForegroundScenario] = []
    for name in pending_names:
        _progress_writer(name)("compile_scenario", None)
        compiled_items.append(compile_foreground_scenario(root, name, refresh=refresh))
    compiled = tuple(compiled_items)
    compiled = tuple(sorted(compiled, key=lambda item: item.foreground_scenario_identity))
    _progress_writer("shared")("source_authority", None)
    request = prepare_execution(prepare_workspace_source(root))
    session = MFAEvaluationSession(
        request,
        tuple(load_parameter_catalogue()),
        recipe=recipe,
        composition=composition,
    )

    completed: list[MFAResult] = list(completed_by_name.values())
    for item in compiled:
        progress = _progress_writer(item.foreground_scenario)
        result = _run_one(
            root,
            item,
            request=request,
            session=session,
            recipe=recipe,
            composition=composition,
            refresh=refresh,
            progress=progress,
        )
        completed.append(result)
        print(
            f"run_mfa {result.status}: {result.foreground_scenario}",
            file=sys.stderr,
        )
    completed_by_name.update({result.foreground_scenario: result for result in completed})
    results = tuple(completed_by_name[name] for name in names)
    return RunMFAReport(
        status=_aggregate_status(results),
        workspace=root,
        elapsed_seconds=perf_counter() - started,
        results=results,
    )
