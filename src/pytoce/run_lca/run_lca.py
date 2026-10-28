"""Public coordinator for regional time-explicit LCA results."""

import json
import re
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.manifests import ItemStatus, OperationStatus

if TYPE_CHECKING:
    import pandas as pd

    from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
    from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
    from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
        PreparedManifest,
    )
    from pytoce.run_mfa.model.mfa import MFAEvaluation
    from pytoce.run_mfa.model.result_inputs import MFAResultReference

    from .calculation.lcia import ImpactCategory, LCIAMethodSelection

_BACKGROUND_DIMENSIONS = (
    "ssp_scenario",
    "iam_model",
    "iam_scenario",
    "gmst",
    "gmst_source",
    "premise_version",
    "ecoinvent_version",
    "system_model",
)

_READABLE_NAME = re.compile(r"[a-z0-9_]+")
DEFAULT_UPSTREAM_ANALYSIS_YEARS = (2019, 2050, 2100)
# Atomic refresh adds 43 characters to the final name in its backup directory.
_MAX_RESULT_NAME_LENGTH = 212


@dataclass(frozen=True, slots=True)
class RunLCAItemReport:
    """One LCA result for a foreground and background scenario.

    The result directory, manifest and table attributes are filesystem paths.

    Attributes:
        status: ``computed`` for a new result, ``updated`` for replacement of a
            completed result, or ``reused`` when its completion manifest already
            exists and refresh is not requested.
        foreground_scenario: Readable foreground scenario name.
        background_scenario: Readable REMIND premise scenario name.
        result_directory: Directory containing the LCA result.
        manifest_path: Path to the result manifest.
        scores: Regional and global environmental burdens summed over 2019--2100.
        annual_scores: Regional and global environmental burdens for each year
            from 2019 to 2100.
        annual_life_cycle_stage_scores: Annual environmental burdens separated
            by life cycle stage.
        life_cycle_stage_scores: Environmental burdens summed over 2019--2100
            for each life cycle stage.
        contributions: Contribution detail for the selected upstream analysis years.
        stage_contributions: Contribution detail for the same selected years,
            separated by life cycle stage.
        carbon_consistency: Comparison of MFA carbon and direct treatment emissions.
        direct_treatment_carbon_flows: Direct carbon flows from treatment.
        mechanical_recycling_loss: Mechanical-recycling loss accounting.
        mismanaged_waste_fate: Fate allocation for fixed kilogram-service demands.
        activation_lineage: Providers supplying fixed kilogram-service demands.
        mismanaged_waste_fate_closure: Mass balance for fixed kilogram-service demands.
        mfa_to_lca_flow_coverage: Connection between MFA flows and LCA processes.
        structural_paths: Non-additive rows linking each regional root and dated
            provider, with descriptors of the stage and material.
        structural_edges: Non-additive edges for these paths; the stage endpoint
            retains the regional root's Brightway key.
        structural_attribution: Life-cycle-stage score rows and the explicit
            zero residual for each region, year, and LCIA category.
    """

    status: ItemStatus
    foreground_scenario: str
    background_scenario: str
    result_directory: Path
    manifest_path: Path
    scores: Path
    annual_scores: Path
    annual_life_cycle_stage_scores: Path
    life_cycle_stage_scores: Path
    contributions: Path
    stage_contributions: Path
    carbon_consistency: Path
    direct_treatment_carbon_flows: Path
    mechanical_recycling_loss: Path
    mismanaged_waste_fate: Path
    activation_lineage: Path
    mismanaged_waste_fate_closure: Path
    mfa_to_lca_flow_coverage: Path
    structural_paths: Path
    structural_edges: Path
    structural_attribution: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the item with every product path represented as text."""

        return {
            "status": self.status,
            "foreground_scenario": self.foreground_scenario,
            "background_scenario": self.background_scenario,
            "result_directory": str(self.result_directory),
            "manifest_path": str(self.manifest_path),
            "scores": str(self.scores),
            "annual_scores": str(self.annual_scores),
            "annual_life_cycle_stage_scores": str(self.annual_life_cycle_stage_scores),
            "life_cycle_stage_scores": str(self.life_cycle_stage_scores),
            "contributions": str(self.contributions),
            "stage_contributions": str(self.stage_contributions),
            "carbon_consistency": str(self.carbon_consistency),
            "direct_treatment_carbon_flows": str(self.direct_treatment_carbon_flows),
            "mechanical_recycling_loss": str(self.mechanical_recycling_loss),
            "mismanaged_waste_fate": str(self.mismanaged_waste_fate),
            "activation_lineage": str(self.activation_lineage),
            "mismanaged_waste_fate_closure": str(self.mismanaged_waste_fate_closure),
            "mfa_to_lca_flow_coverage": str(self.mfa_to_lca_flow_coverage),
            "structural_paths": str(self.structural_paths),
            "structural_edges": str(self.structural_edges),
            "structural_attribution": str(self.structural_attribution),
        }


@dataclass(frozen=True, slots=True)
class RunLCAReport:
    """Result of :func:`pytoce.run_lca`.

    Attributes:
        status: ``computed`` when every item is newly computed, ``reused`` when
            every item is reused, or ``updated`` otherwise, including a mixture
            of computed and reused items.
        workspace: Active workspace root.
        elapsed_seconds: Time used for the LCA request.
        results: One result for each requested foreground and background scenario.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    results: tuple[RunLCAItemReport, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""

        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "results": [item.as_dict() for item in self.results],
        }


@dataclass(frozen=True, slots=True)
class _Request:
    """One foreground, background, and destination passed to an LCA worker."""

    foreground_scenario: str
    background_scenario: str
    result_directory: Path


_ISOLATED_PRODUCT_COMMAND = (
    "from pytoce.run_lca.run_lca import _isolated_product_entry; "
    "import sys; "
    "raise SystemExit(_isolated_product_entry(sys.argv[1], sys.argv[2]))"
)


def _names(value: str | Sequence[str], parameter: str) -> tuple[str, ...]:
    """Normalize readable scenario names, remove duplicates, and sort them."""

    try:
        raw: tuple[object, ...] = (value,) if isinstance(value, str) else tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{parameter} must be one lowercase snake_case name or a sequence of names"
        ) from exc
    if not raw:
        raise ValueError(
            f"{parameter} must contain at least one readable lowercase snake_case name"
        )
    if any(not isinstance(item, str) or not _READABLE_NAME.fullmatch(item) for item in raw):
        raise ValueError(f"{parameter}={value!r} must contain only lowercase snake_case names")
    names = tuple(sorted(set(cast(tuple[str, ...], raw))))
    return names


def _years(value: int | Sequence[int]) -> tuple[int, ...]:
    """Return sorted unique upstream-analysis years from 2019 through 2100."""

    try:
        raw: tuple[object, ...] = (value,) if isinstance(value, int) else tuple(value)
    except TypeError as exc:
        raise ValueError(
            "upstream_analysis_year must be one integer or a sequence of integers "
            "from 2019 through 2100"
        ) from exc
    if any(isinstance(year, bool) or not isinstance(year, int) for year in raw):
        raise ValueError(
            f"upstream_analysis_year={value!r} must contain only integers from 2019 through 2100"
        )
    years = tuple(sorted(set(raw)))
    if not years or years[0] < 2019 or years[-1] > 2100:
        raise ValueError("upstream_analysis_year must be within 2019-2100")
    return years


def _year_name(years: tuple[int, ...]) -> str:
    """Compress consecutive analysis years for an LCA result directory name."""
    runs: list[tuple[int, int]] = []
    start = previous = years[0]
    for year in years[1:]:
        if year == previous + 1:
            previous = year
            continue
        runs.append((start, previous))
        start = previous = year
    runs.append((start, previous))
    return "_".join(str(start) if start == end else f"{start}-{end}" for start, end in runs)


def lca_result_directory(
    workspace: Path,
    foreground_scenario: str,
    background_scenario: str,
    selection: "LCIAMethodSelection",
    years: tuple[int, ...],
) -> Path:
    """Return the canonical readable directory for one exact LCA request."""

    from .calculation.lcia import readable_lcia_method_selection

    readable_name = (
        f"{background_scenario}__{readable_lcia_method_selection(selection)}"
        f"__upstream_analysis_{_year_name(years)}"
    )
    if len(readable_name) > _MAX_RESULT_NAME_LENGTH:
        raise ValueError(
            f"LCA result name {readable_name!r} has {len(readable_name)} characters; "
            f"at most {_MAX_RESULT_NAME_LENGTH} are supported, including room for refresh. "
            "Split analysis years across calls. For a custom method, register the same "
            "factors under a shorter method label."
        )
    return workspace / "results" / foreground_scenario / "lca_results" / readable_name


def _requests(
    workspace: Path,
    foreground_scenarios: tuple[str, ...],
    background_scenarios: tuple[str, ...],
    selection: "LCIAMethodSelection",
    years: tuple[int, ...],
) -> tuple[_Request, ...]:
    """Build foreground and background product requests in stable order."""
    return tuple(
        _Request(
            foreground_scenario=foreground,
            background_scenario=background,
            result_directory=lca_result_directory(
                workspace,
                foreground,
                background,
                selection,
                years,
            ),
        )
        for foreground in foreground_scenarios
        for background in background_scenarios
    )


def _result(request: _Request, status: ItemStatus) -> RunLCAItemReport:
    """Build one LCA item report from its request and completion status."""
    directory = request.result_directory
    return RunLCAItemReport(
        status=status,
        foreground_scenario=request.foreground_scenario,
        background_scenario=request.background_scenario,
        result_directory=directory,
        manifest_path=directory / "manifest.json",
        scores=directory / "scores.parquet",
        annual_scores=directory / "annual_scores.parquet",
        annual_life_cycle_stage_scores=(directory / "annual_life_cycle_stage_scores.parquet"),
        life_cycle_stage_scores=directory / "life_cycle_stage_scores.parquet",
        contributions=directory / "contributions.parquet",
        stage_contributions=directory / "stage_contributions.parquet",
        carbon_consistency=directory / "carbon_consistency.csv",
        direct_treatment_carbon_flows=(directory / "direct_treatment_carbon_flows.parquet"),
        mechanical_recycling_loss=directory / "mechanical_recycling_loss.parquet",
        mismanaged_waste_fate=directory / "mismanaged_waste_fate.parquet",
        activation_lineage=directory / "activation_lineage.parquet",
        mismanaged_waste_fate_closure=(directory / "mismanaged_waste_fate_closure.parquet"),
        mfa_to_lca_flow_coverage=directory / "mfa_to_lca_flow_coverage.parquet",
        structural_paths=directory / "structural_paths.parquet",
        structural_edges=directory / "structural_edges.parquet",
        structural_attribution=directory / "structural_attribution.parquet",
    )


def _aggregate_status(results: tuple[RunLCAItemReport, ...]) -> OperationStatus:
    """Return the aggregate status for the supplied records."""
    statuses = {item.status for item in results}
    if statuses == {"computed"}:
        return "computed"
    if statuses == {"reused"}:
        return "reused"
    return "updated"


def _mfa_parent(
    workspace: Path,
    foreground_scenario: str,
) -> tuple["MFAResultReference", "MFAEvaluation", "EcoinventMaterialRecipe"]:
    """Read the selected MFA tables and material recipe for foreground calculation."""
    from pytoce.download_data.sources.ecoinvent_materials import load_material_recipe
    from pytoce.run_mfa.model.result_inputs import read_mfa_result, read_mfa_result_manifest
    from pytoce.run_mfa.model.results import selected_mfa_result

    path = selected_mfa_result(workspace, foreground_scenario)
    if path is None:
        raise FileNotFoundError(
            f"No completed MFA result was found for foreground_scenario="
            f"{foreground_scenario!r}. Run run_mfa(foreground_scenario=...) first."
        )
    reference = read_mfa_result_manifest(path)
    return reference, read_mfa_result(reference), load_material_recipe(workspace)


def _background_dimensions(member: "PreparedManifest") -> dict[str, object]:
    """Read the prepared member's row of LCA reporting labels."""
    import pyarrow.parquet as pq

    table = pq.read_table(member.path.parent / "background_scenarios.parquet").to_pandas()
    return {name: table.iloc[0][name] for name in _BACKGROUND_DIMENSIONS}


def _method_selection(
    names: "BrightwayNames",
    authority: "pd.DataFrame",
    selection: "LCIAMethodSelection",
) -> tuple["ImpactCategory", ...]:
    """Resolve admitted LCIA rows in their recorded Brightway project."""
    from pytoce.prepare_backgrounds.preparation.ecoinvent import selected_brightway_project

    from .calculation.lcia import resolve_lcia_methods_from_authority

    with selected_brightway_project(names) as bd:
        return resolve_lcia_methods_from_authority(authority, selection, bd)


def _calculate_and_write_product(
    workspace: Path,
    request: _Request,
    selection: "LCIAMethodSelection",
    years: tuple[int, ...],
    *,
    refresh: bool,
) -> ItemStatus:
    """Calculate one foreground and background LCA result and write it once.

    Args:
        workspace: Active pyToCE workspace.
        request: Foreground scenario, background scenario, and result directory.
        selection: Selected pyToCE method groups or one exact Brightway method.
        years: Years selected for detailed upstream contribution tables.
        refresh: Whether to replace a completed result after protecting unrelated
            files and linked paths at the destination.

    Returns:
        ``"computed"``, ``"reused"``, or ``"updated"`` for the requested result.

    Raises:
        FileNotFoundError: If the completed MFA result or prepared background is absent.
        ValueError: If an incomplete result directory exists, the selected exact
            method is unavailable, or the existing result cannot be refreshed safely.

    Notes:
        The result lock prevents two processes from writing the same directory.
        Completed parent tables supply the foreground inputs. The completion
        manifest follows every result table.
    """

    import pyarrow.parquet as pq

    from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
        read_prepared_manifest,
    )

    from .calculation.foreground import build_foreground_database
    from .calculation.providers import (
        load_prepared_provider_product,
        load_provider_routes,
        prepared_brightway_names,
    )
    from .calculation.results import validate_lca_result_for_refresh, write_lca_result

    request.result_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = request.result_directory.parent / f".{request.result_directory.name}.writer.lock"
    with writer_lock(lock_path):
        manifest_path = request.result_directory / "manifest.json"
        if not refresh and manifest_path.is_file():
            return "reused"
        result_exists = request.result_directory.exists() or request.result_directory.is_symlink()
        if result_exists and not manifest_path.is_file():
            raise ValueError(
                f"The LCA result directory {request.result_directory} has no valid completion "
                "manifest. Keep it unchanged and create a new workspace."
            )
        replacing = refresh and result_exists
        if replacing:
            validate_lca_result_for_refresh(request.result_directory)

        mfa_reference, mfa, recipe = _mfa_parent(
            workspace,
            request.foreground_scenario,
        )
        mfa_manifest = mfa_reference.manifest
        member = read_prepared_manifest(
            workspace
            / "data_processed/prepared_backgrounds"
            / request.background_scenario
            / "manifest.json"
        )
        shared = read_prepared_manifest(
            workspace / "data_processed/prepared_backgrounds/shared/manifest.json"
        )
        names = prepared_brightway_names(workspace, shared)
        providers = load_prepared_provider_product(shared, member)
        dimensions = _background_dimensions(member)
        routes = load_provider_routes(shared)
        methods = _method_selection(
            names,
            pq.read_table(member.path.parent / "impact_categories.parquet").to_pandas(),
            selection,
        )
        foreground = build_foreground_database(
            request.foreground_scenario,
            names=names,
            routes=routes,
            mfa_result_identity=str(mfa_manifest["identity"]),
            mfa=mfa,
            recipe=recipe,
        )
        pa_dump_shares = routes.open_dump_shares()
        del mfa, routes
        return write_lca_result(
            request.result_directory,
            request.foreground_scenario,
            request.background_scenario,
            foreground,
            methods,
            years,
            mfa_manifest,
            mfa_reference.sha256,
            member.payload,
            member.sha256,
            names=names,
            providers=providers,
            pa_dump_shares=pa_dump_shares,
            background_dimensions=dimensions,
            replacing=replacing,
        )


def _isolated_product_entry(payload_path: str, response_path: str) -> int:
    """Execute one package-generated product payload in a fresh interpreter."""

    from .calculation.lcia import LCIAMethodSelection

    response: dict[str, object]
    try:
        payload = cast(
            dict[str, object],
            json.loads(Path(payload_path).read_text(encoding="utf-8")),
        )
        request = _Request(
            foreground_scenario=str(payload["foreground_scenario"]),
            background_scenario=str(payload["background_scenario"]),
            result_directory=Path(str(payload["result_directory"])),
        )
        method = cast(dict[str, list[str]], payload["lcia_method"])
        years = tuple(int(year) for year in cast(list[int], payload["years"]))
        status = _calculate_and_write_product(
            Path(str(payload["workspace"])),
            request,
            LCIAMethodSelection(
                groups=tuple(method["groups"]), exact_method=tuple(method["exact_method"])
            ),
            years,
            refresh=bool(payload["refresh"]),
        )
        response = {
            "schema_version": "pytoce_lca_product_response_schema_v1",
            "status": status,
        }
        return_code = 0
    except Exception as exc:
        response = {
            "schema_version": "pytoce_lca_product_response_schema_v1",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 1
    Path(response_path).write_text(
        json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return return_code


def _isolated_product(
    workspace: Path,
    request: _Request,
    selection: "LCIAMethodSelection",
    years: tuple[int, ...],
    *,
    refresh: bool = False,
) -> ItemStatus:
    """Run one pending product with an interpreter-scoped memory lifetime."""

    # Preserve the exact tuple containing "ef31" when passing the selected methods to the child.
    payload = {
        "schema_version": "pytoce_lca_product_request_schema_v1",
        "workspace": str(workspace),
        "foreground_scenario": request.foreground_scenario,
        "background_scenario": request.background_scenario,
        "result_directory": str(request.result_directory),
        "lcia_method": {
            "groups": list(selection.groups),
            "exact_method": list(selection.exact_method),
        },
        "years": list(years),
        "refresh": refresh,
    }
    print(
        f"run_lca: {request.foreground_scenario}, {request.background_scenario}: "
        "starting calculation",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(
        prefix="pytoce_run_lca_", ignore_cleanup_errors=True
    ) as temporary:
        temporary_path = Path(temporary)
        payload_path = temporary_path / "request.json"
        response_path = temporary_path / "response.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _ISOLATED_PRODUCT_COMMAND,
                str(payload_path),
                str(response_path),
            ],
            check=False,
        )
        response = (
            cast(
                dict[str, object],
                json.loads(response_path.read_text(encoding="utf-8")),
            )
            if response_path.is_file()
            else {}
        )
    if completed.returncode != 0:
        error_type = str(response.get("error_type", "calculation process failure"))
        message = str(response.get("message", "the calculation returned no error message"))
        child_traceback = str(response.get("traceback", ""))
        failure = RuntimeError(
            "run_lca calculation process failed for "
            f"foreground_scenario={request.foreground_scenario!r}, "
            f"background_scenario={request.background_scenario!r}: "
            f"{error_type}: {message}"
        )
        if child_traceback:
            failure.add_note(f"Calculation process traceback:\n{child_traceback}")
        raise failure
    print(
        f"run_lca: {request.foreground_scenario}, {request.background_scenario}: "
        "calculation complete",
        file=sys.stderr,
    )
    return cast(ItemStatus, response["status"])


def run_lca(
    *,
    foreground_scenario: str | Sequence[str] = "remind_mfa_default",
    background_scenario: str | Sequence[str] = (
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
        "remind_ssp2_pkbudg1000",
        "remind_ssp2_pkbudg650",
    ),
    lcia_method: str | Sequence[str] = (
        "ef31",
        "pb_lcia",
    ),
    upstream_analysis_year: int | Sequence[int] = DEFAULT_UPSTREAM_ANALYSIS_YEARS,
    refresh: bool = False,
) -> RunLCAReport:
    """Calculate environmental burdens through time-explicit prospective LCA.

    The function translates annual MFA flows into a foreground life cycle
    inventory and links each activity to the background inventory
    corresponding to its year of occurrence. It calculates each requested
    foreground scenario with each requested background scenario over
    2019--2100, with the world divided into 12 regions.

    Args:
        foreground_scenario: One foreground scenario name with completed
            ``run_mfa()`` results, or a sequence of names. Defaults to
            ``"remind_mfa_default"``.
        background_scenario: One prepared background scenario name or a
            sequence. The default selects ``"remind_ssp2_ndc"``,
            ``"remind_ssp2_npi"``, ``"remind_ssp2_pkbudg1000"``, and
            ``"remind_ssp2_pkbudg650"``. Historical REMIND-EU names select
            completed LCA results with ``refresh=False``.
        lcia_method: Life cycle impact assessment (LCIA) method selection.
            ``"ef31"`` selects all 16 EF 3.1 midpoint impact categories, and
            ``"pb_lcia"`` selects all 10 supported PB-LCIA impact categories.
            A sequence of these group names selects the listed groups.
            Defaults to ``("ef31", "pb_lcia")``. A full registered Brightway
            method tuple selects one impact category; the same tuple can be
            supplied as a JSON array string.
        upstream_analysis_year: Years for detailed upstream contribution
            analysis. Accepts one integer year or a sequence of years from
            2019 through 2100. Defaults to ``(2019, 2050, 2100)``. Annual
            environmental burden totals cover the full 2019--2100 period.
        refresh: If ``True``, recalculate the requested LCA results using the
            completed MFA and prepared backgrounds. Defaults to ``False``,
            which reuses completed results.

    Returns:
        ``RunLCAReport`` describing completion status, elapsed time in seconds,
        and one result per foreground and background scenario pair, including
        paths to annual and period-total scores, life cycle stage scores,
        contributions, diagnostics, and the manifest.

    Raises:
        RuntimeError: If no active workspace has been selected, a required
            MFA or prepared background result is unavailable, or an LCA
            calculation process fails.
        ValueError: If ``refresh`` is not a Boolean, scenario names, years,
            or method selections are invalid, the assembled result directory
            name exceeds 212 characters, or a historical REMIND-EU selection
            requires calculation or refresh.
        OSError: If workspace or calculation process files cannot be read or
            written, or a calculation process cannot be started.

    Notes:
        Call ``set_workspace()``, ``download_data()``, ``prepare_backgrounds()``,
        and ``run_mfa()`` before calculating new LCA results.

        Production, conversion, recycling, and end-of-life treatment are
        evaluated with the background system corresponding to their year of
        occurrence. Emissions are assigned to the year of the corresponding
        activity, and characterization factors are kept constant across years.
        Environmental burdens retain the unit of each impact category.
        Global results sum the 12 regional results.

        Each LCIA method selection and set of upstream analysis years has
        its own result directory. The selected LCIA methods characterize
        the same inventory within that result.

    Example:
        Calculate environmental burdens for a completed foreground scenario
        under one prepared background::

            from pytoce import run_lca

            report = run_lca(
                foreground_scenario="remind_mfa_default",
                background_scenario="remind_ssp2_npi",
                lcia_method=("ef31", "pb_lcia"),
                upstream_analysis_year=(2019, 2050, 2100),
            )
    """

    if not isinstance(refresh, bool):
        raise ValueError(
            f"refresh={refresh!r} is invalid. Use refresh=True to recalculate selected "
            "LCA results or refresh=False to reuse completed results."
        )

    from pytoce.run_mfa.scenario.scenario_contract import validate_foreground_scenario_name
    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    from .calculation.lcia import normalized_lcia_method

    started = perf_counter()
    workspace = get_active_workspace()
    foregrounds = tuple(
        validate_foreground_scenario_name(name)
        for name in _names(foreground_scenario, "foreground_scenario")
    )
    backgrounds = _names(background_scenario, "background_scenario")
    selection = normalized_lcia_method(lcia_method)
    years = _years(upstream_analysis_year)
    requests = _requests(workspace, foregrounds, backgrounds, selection, years)

    from pytoce.prepare_backgrounds.preparation.premise import (
        LEGACY_BACKGROUND_IDS,
    )

    completed: dict[tuple[str, str], RunLCAItemReport] = {}
    pending: list[_Request] = []
    for request in requests:
        if not refresh and (request.result_directory / "manifest.json").is_file():
            result = _result(request, "reused")
            completed[(request.foreground_scenario, request.background_scenario)] = result
            print(
                f"run_lca: {request.foreground_scenario}, "
                f"{request.background_scenario}: reusing the completed result",
                file=sys.stderr,
            )
        else:
            pending.append(request)

    if pending:
        legacy_pending = tuple(
            sorted(
                {
                    request.background_scenario
                    for request in pending
                    if request.background_scenario in LEGACY_BACKGROUND_IDS
                }
            )
        )
        if legacy_pending:
            raise ValueError(
                "REMIND-EU LCA products are retained for reuse only and cannot be "
                f"computed or updated: {legacy_pending!r}"
            )

    for request in pending:
        status = _isolated_product(
            workspace,
            request,
            selection,
            years,
            refresh=refresh,
        )
        completed[(request.foreground_scenario, request.background_scenario)] = _result(
            request, status
        )

    results = tuple(
        completed[(request.foreground_scenario, request.background_scenario)]
        for request in requests
    )
    return RunLCAReport(
        status=_aggregate_status(results),
        workspace=workspace,
        elapsed_seconds=perf_counter() - started,
        results=results,
    )
