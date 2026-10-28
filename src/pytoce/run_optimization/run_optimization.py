"""Public coordinator for the deterministic optimization workflow."""

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from pytoce.run_optimization.blocks import (
    OPTIMIZATION_BLOCK_ORDER,
    OptimizationBlock,
)
from pytoce.shared.runtime.manifests import ItemStatus, OperationStatus

if TYPE_CHECKING:
    from pytoce.run_lca.calculation.lcia import LCIAMethodSelection
    from pytoce.run_mfa.model.result_inputs import MFAResultReference
    from pytoce.run_optimization.input.allocated_cc_contract import AllocatedCCResult
    from pytoce.run_optimization.runtime.result_store import SavedOptimizationResult

_BLOCK_LABELS = {
    "demand_lifetime_frontier": "demand_lifetime",
    "circularity_feedstock_anchor": "circularity_feedstock",
    "integrated_frontier": "integrated",
}
_READABLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_OPTIMIZATION_LCIA_SELECTION = "ef31"
_REAL_BACKGROUNDS = (
    "remind_ssp2_ndc",
    "remind_ssp2_npi",
    "remind_ssp2_pkbudg1000",
    "remind_ssp2_pkbudg650",
)


class RunOptimizationError(RuntimeError):
    """A public optimization request cannot be completed."""


@dataclass(frozen=True, slots=True)
class RunOptimizationRequest:
    """Selections for one optimization request."""

    foreground_scenarios: tuple[str, ...]
    background_scenarios: tuple[str, ...]
    allocated_cc_result: str
    lcia_method: "LCIAMethodSelection"
    result_lcia_selection: str
    optimization_blocks: tuple[OptimizationBlock, ...]
    allocated_cc_cases: tuple[str, ...] | None
    initial_grid_step_pct: int
    endpoint_refinement_steps_pct: tuple[int, ...]
    readable_request_selection: str
    candidate_workers: int
    refresh: bool


@dataclass(frozen=True, slots=True)
class RunOptimizationProductReport:
    """One selected optimization result.

    Attributes:
        status: Whether the selected result was ``computed``,
            ``updated``, or ``reused``.
        foreground_scenario: Readable foreground scenario name.
        product_identity: Content identifier calculated from the selected result
            and its recorded inputs.
        result_directory: Directory containing the optimization result.
        manifest_path: Path to the result manifest.
    """

    status: ItemStatus
    foreground_scenario: str
    product_identity: str
    result_directory: Path
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the product record with paths represented as text."""

        return {
            "status": self.status,
            "foreground_scenario": self.foreground_scenario,
            "product_identity": self.product_identity,
            "result_directory": str(self.result_directory),
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True, slots=True)
class RunOptimizationReport:
    """Result of :func:`pytoce.run_optimization`.

    Attributes:
        status: ``computed`` when every product is newly computed, ``reused``
            when every product is reused, or ``updated`` otherwise, including a
            mixture of computed and reused products.
        workspace: Active workspace root.
        elapsed_seconds: Time used for the optimization request.
        products: One selected result for each requested foreground scenario.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    products: tuple[RunOptimizationProductReport, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary with paths converted to text."""

        return {
            "status": self.status,
            "workspace": str(self.workspace),
            "elapsed_seconds": self.elapsed_seconds,
            "products": [item.as_dict() for item in self.products],
        }


def normalize_run_optimization_request(
    *,
    foreground_scenario: str | Sequence[str],
    allocated_cc_result: str,
    background_scenario: str | Sequence[str] | None = None,
    optimization_lcia_method: str | Sequence[str] = "ef31",
    lcia_method: str | Sequence[str] = ("ef31", "pb_lcia"),
    optimization_block: OptimizationBlock | Sequence[OptimizationBlock] = OPTIMIZATION_BLOCK_ORDER,
    allocated_cc_case: str | Sequence[str] | None = None,
    initial_grid_step_pct: int = 20,
    endpoint_refinement_steps_pct: Sequence[int] = (10, 5),
    candidate_workers: int = 1,
    refresh: bool = False,
) -> RunOptimizationRequest:
    """Validate optimization selections.

    The function accepts readable foreground scenario names, EF v3.1 for
    constraints, the three documented optimization blocks, readable aCC case
    names, and a Boolean refresh value. Selected-result LCIA methods may include
    EF v3.1, PB-LCIA, or one exact category from the package's EF/PB catalogue.

    Returns:
        Ordered, duplicate-free selections and the readable result directory
        names used by :func:`run_optimization`.

    Raises:
        RunOptimizationError: If a value is invalid or outside the supported
            optimization selections. Each message states the accepted choice.
    """

    from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
    from pytoce.run_lca.calculation.lcia import readable_lcia_method_selection
    from pytoce.run_mfa.scenario.scenario_contract import (
        ScenarioError,
        validate_foreground_scenario_name,
    )

    try:
        foregrounds = tuple(
            sorted(
                {
                    validate_foreground_scenario_name(name)
                    for name in _selection_names(
                        foreground_scenario, parameter="foreground_scenario"
                    )
                }
            )
        )
    except (ScenarioError, RunOptimizationError) as exc:
        raise RunOptimizationError(f"Invalid foreground_scenario selection: {exc}") from exc
    backgrounds = (
        _REAL_BACKGROUNDS
        if background_scenario is None
        else tuple(
            sorted(set(_selection_names(background_scenario, parameter="background_scenario")))
        )
    )
    if any(background not in _REAL_BACKGROUNDS for background in backgrounds):
        raise RunOptimizationError(f"background_scenario must select from {_REAL_BACKGROUNDS!r}.")
    allocated_result = _readable_name(allocated_cc_result, parameter="allocated_cc_result")
    optimization_methods = _lcia_selection(
        optimization_lcia_method, parameter="optimization_lcia_method"
    )
    if optimization_methods.groups != ("ef31",):
        raise RunOptimizationError(
            "optimization_lcia_method must be 'ef31'. Use lcia_method to report EF v3.1, "
            "PB-LCIA, or an exact category from the package's EF/PB catalogue."
        )
    result_methods = _lcia_selection(lcia_method, parameter="lcia_method")
    if result_methods.exact_method:
        supported_methods, _ = load_lcia_catalogue()
        if result_methods.exact_method not in {
            method.impact_category_tuple for method in supported_methods
        }:
            raise RunOptimizationError(
                f"lcia_method={result_methods.exact_method!r} is not supported by optimization. "
                "Select 'ef31', 'pb_lcia', both groups, or an exact category from the "
                "package's sixteen EF v3.1 and ten PB-LCIA categories."
            )
    blocks = _optimization_blocks(optimization_block)
    cases = (
        None
        if allocated_cc_case is None
        else tuple(sorted(set(_selection_names(allocated_cc_case, parameter="allocated_cc_case"))))
    )
    if type(initial_grid_step_pct) is not int or initial_grid_step_pct not in (10, 20, 25, 50, 100):
        raise RunOptimizationError(
            "initial_grid_step_pct must be one of 10, 20, 25, 50 or 100 percentage points."
        )
    steps = _endpoint_refinement_steps(endpoint_refinement_steps_pct, initial_grid_step_pct)
    if type(candidate_workers) is not int or not 1 <= candidate_workers <= 61:
        raise RunOptimizationError(
            "candidate_workers must be an integer from 1 through 61; Boolean values are invalid."
        )
    if not isinstance(refresh, bool):
        raise RunOptimizationError(
            f"refresh={refresh!r} is invalid. Use refresh=True to recalculate this selected "
            "result or refresh=False to reuse it."
        )
    return RunOptimizationRequest(
        foreground_scenarios=foregrounds,
        background_scenarios=backgrounds,
        allocated_cc_result=allocated_result,
        lcia_method=result_methods,
        result_lcia_selection=readable_lcia_method_selection(result_methods),
        optimization_blocks=blocks,
        allocated_cc_cases=cases,
        initial_grid_step_pct=initial_grid_step_pct,
        endpoint_refinement_steps_pct=steps,
        readable_request_selection=_readable_request_selection(blocks, cases, backgrounds),
        candidate_workers=candidate_workers,
        refresh=refresh,
    )


def run_optimization(
    *,
    foreground_scenario: str | Sequence[str],
    allocated_cc_result: str,
    background_scenario: str | Sequence[str] | None = None,
    optimization_lcia_method: str | Sequence[str] = "ef31",
    lcia_method: str | Sequence[str] = ("ef31", "pb_lcia"),
    optimization_block: OptimizationBlock | Sequence[OptimizationBlock] = OPTIMIZATION_BLOCK_ORDER,
    allocated_cc_case: str | Sequence[str] | None = None,
    initial_grid_step_pct: int = 20,
    endpoint_refinement_steps_pct: Sequence[int] = (10, 5),
    candidate_workers: int = 1,
    refresh: bool = False,
) -> RunOptimizationReport:
    """Search for CE performance levels for target setting.

    Target setting starts from allocated carrying capacities and searches
    for circular economy (CE) performance levels whose environmental burdens
    comply with those carrying capacities. The user selects a completed
    foreground scenario as the parent, supplying its MFA and baseline LCA
    results. Allocated carrying capacities can be calculated through the
    pyaesa tutorial workflow or supplied by the user. The search is performed
    separately for each selected background scenario and allocated carrying
    capacity case.

    Args:
        foreground_scenario: Required foreground scenario name or a sequence
            of names. Each foreground is optimized from its own completed MFA
            and baseline LCA results. CE performance families outside the
            selected block retain their parent trajectories.
        allocated_cc_result: Required directory name of a complete allocated
            carrying capacity input under ``data_raw/allocated_cc/results``.
            Follow the YAML and CSV templates installed under
            ``data_raw/allocated_cc/templates``.
        background_scenario: One REMIND SSP2 background scenario name or a
            sequence. Defaults to ``None``, which selects all four scenarios
            supported by ``prepare_backgrounds()``.
        optimization_lcia_method: LCIA method used for constraints, search,
            ranking, and selection. The accepted value and default is
            ``"ef31"``, which uses all 16 EF 3.1 impact categories and
            cumulative GWP100.
        lcia_method: Impact categories used to report environmental burdens
            for the selected CE transition pathway. Accepts ``"ef31"``,
            ``"pb_lcia"``, both groups, or the full Brightway method tuple for
            one of the package's 16 EF 3.1 or ten PB-LCIA impact categories.
            A full method tuple can also be supplied as a JSON array string.
            Defaults to ``("ef31", "pb_lcia")``.
        optimization_block: One block name or a sequence. Defaults to all
            three blocks:

            - ``"demand_lifetime_frontier"``: Block I, demand reduction and
              lifetime extension.
            - ``"circularity_feedstock_anchor"``: Block II, collection,
              treatment composition, and virgin feedstock composition.
            - ``"integrated_frontier"``: Block III, all five CE performance
              families together.

        allocated_cc_case: One supplied case label, a sequence of labels, or
            ``None``. Defaults to ``None``, which selects every available
            case separately. The source documentation defines the allocation
            method and statistic represented by each label.
        initial_grid_step_pct: Initial exploration spacing in percentage
            points. Accepts 10, 20, 25, 50, or 100; defaults to 20. Smaller
            spacings increase the number of initial combinations evaluated.
        endpoint_refinement_steps_pct: Nonempty, strictly decreasing sequence
            of distinct positive integer spacings, in percentage points,
            below ``initial_grid_step_pct``. Defaults to ``(10, 5)``;
            ``(5,)`` requests a single refinement stage. To add finer
            refinement later, supply the full schedule, for example
            ``(10, 5, 1)``. Compatible completed stages and candidate
            calculations are reused. Each schedule has its own selected
            result. Demand-timing repair uses one-percentage-point intervals.
        candidate_workers: Maximum number of concurrent candidate evaluation
            processes. Accepts integers from 1 through 61, following the
            Windows process-pool limit; defaults to 1. Each process retains
            its own scientific inputs. Choose a count that fits the available
            CPU cores and memory.
        refresh: If ``True``, rewrite outputs for the selected CE transition
            pathways using completed prerequisites and reusing completed
            candidate calculations. Defaults to ``False``, which reuses
            completed selected results.

    Returns:
        ``RunOptimizationReport`` describing completion status, the workspace,
        elapsed time in seconds, and one result per foreground scenario,
        including its content identifier, directory, and manifest path.

    Raises:
        RuntimeError: If no active workspace has been selected, a selection
            is invalid, a required MFA, background, LCA, or allocated carrying
            capacity input is missing or inconsistent, or the search or
            result writing fails.
        OSError: If a required workspace path cannot be read or written.

    Notes:
        Baseline LCA results must match each selected foreground, background,
        and reporting ``lcia_method`` selection. Building the shared LCA
        responses requires all four prepared REMIND SSP2 backgrounds.
        When several LCA results differ only in upstream analysis years,
        the result with default analysis years is preferred, followed by
        result directory name order. These LCA results must share the same
        inputs and calculation settings apart from upstream analysis years.

        Each tested combination is evaluated through the dynamic MFA before
        its environmental burdens and ASRs are calculated. Demand-timing
        repair can adjust the annual timing of demand reduction while
        retaining the target levels and completion year.

        Compliance requires an Absolute Sustainability Ratio (ASR) at most 1
        for all 16 EF 3.1 impact categories in each of the 12 regions for
        every year from 2020 through 2100, and for cumulative GWP100 in each
        region over 2019--2100. The unchanged parent is selected when it
        meets every constraint. Among compliant combinations, selection
        minimizes intervention distance, with equal weight given to the CE
        performance families considered in the corresponding block. Remaining
        ties favor the combination whose highest ASR is closest to one.
        If none of the evaluated combinations complies, a noncompliant
        fallback is selected and labeled ``selected_noncompliant``.

        The search selects CE performance levels with completion in 2050.
        The same levels are also evaluated with completion in 2100 to assess
        the effect of the transition rate. The :doc:`workflow_reference`
        describes the search, refinement, demand-timing repair, and selection
        criteria. The article and its Methodological appendix are available
        from :doc:`publications_and_resources`.

    Example:
        Search from a completed foreground scenario and a supplied allocated
        carrying capacity input, using defaults where omitted::

            from pytoce import run_optimization

            report = run_optimization(
                foreground_scenario="frozen_2019_default_baseline",
                allocated_cc_result="study_allocated_cc",
            )
    """

    from pytoce.set_workspace.runtime.workspace import get_active_workspace

    started = perf_counter()
    workspace = get_active_workspace()
    request = normalize_run_optimization_request(
        foreground_scenario=foreground_scenario,
        allocated_cc_result=allocated_cc_result,
        background_scenario=background_scenario,
        optimization_lcia_method=optimization_lcia_method,
        lcia_method=lcia_method,
        optimization_block=optimization_block,
        allocated_cc_case=allocated_cc_case,
        initial_grid_step_pct=initial_grid_step_pct,
        endpoint_refinement_steps_pct=endpoint_refinement_steps_pct,
        candidate_workers=candidate_workers,
        refresh=refresh,
    )
    products = tuple(
        _run_foreground(workspace, request, foreground)
        for foreground in request.foreground_scenarios
    )
    statuses = {product.status for product in products}
    return RunOptimizationReport(
        status=(
            "computed"
            if statuses == {"computed"}
            else "reused"
            if statuses == {"reused"}
            else "updated"
        ),
        workspace=workspace,
        elapsed_seconds=perf_counter() - started,
        products=products,
    )


def _run_foreground(
    workspace: Path,
    request: RunOptimizationRequest,
    foreground: str,
) -> RunOptimizationProductReport:
    """Run the optimization request for one selected foreground scenario.

    A completed selected result is reused unless ``refresh=True``. For a new
    calculation, the function checks the required LCA results, loads the MFA
    result and selected aCC cases, prepares the response tables used by each
    candidate process, evaluates the permitted intervention combinations, and
    writes only the selected result tables. Candidate result files are retained
    to avoid repeating completed candidate calculations for the same
    optimization request.

    Args:
        workspace: Active pyToCE workspace.
        request: Validated method, aCC case, search-block, and refresh selections.
        foreground: Selected parent of this independent optimization result.

    Returns:
        One report for the newly calculated or reused selected optimization result.

    Raises:
        RunOptimizationError: If a required input is missing or inconsistent, a
            candidate process stops, or the selected result cannot be written.
    """
    from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
        PreparedDataError,
    )

    from .calculation.candidate_authority import CandidateEvaluationError
    from .calculation.candidate_search import CANDIDATE_SELECTION_METHOD_IDENTITY
    from .calculation.candidate_worker import (
        CandidateWorkerRequestBridge,
        prepare_candidate_worker_request_context,
    )
    from .calculation.request_search import (
        build_candidate_input_identity,
        build_candidate_worker_metric_scopes,
        build_optimization_request_identity,
        run_optimization_search,
    )
    from .calculation.response_bank_contract import ResponseBankError
    from .calculation.response_kernel import ResponseBuilderError
    from .calculation.results import build_selected_result_tables
    from .calculation.selected_parent import (
        OptimizationResultError as CalculationResultError,
    )
    from .calculation.selected_parent import (
        load_baseline_lca_result,
        select_baseline_lca_result,
    )
    from .input.allocated_cc import select_allocated_cc_cases
    from .input.allocated_cc_contract import AllocatedCCError
    from .input.allocated_cc_product import compile_allocated_cc_result
    from .runtime.candidate_results import CandidateResultStore, CandidateResultStoreError
    from .runtime.result_store import (
        OptimizationResultError as StoreError,
    )
    from .runtime.result_store import (
        OptimizationResultTarget,
        ensure_optimization_results_guide,
        product_complete,
        read_product_manifest,
        read_refinement_prefix,
        stage_selected_life_cycle_scores,
        validate_result_for_refresh,
        write_optimization_result,
    )
    from .runtime.scheduler import CandidateProcessPool, SchedulerError

    target = OptimizationResultTarget(
        workspace=workspace,
        foreground_scenario=foreground,
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection=_OPTIMIZATION_LCIA_SELECTION,
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=request.initial_grid_step_pct,
        endpoint_refinement_steps_pct=request.endpoint_refinement_steps_pct,
    )
    selected_directory = target.selected_results(request.result_lcia_selection)
    try:
        selected_complete = product_complete(selected_directory)
        if selected_complete:
            if not request.refresh:
                _progress(foreground, "reusing the completed result")
                return _product_report(foreground, read_product_manifest(selected_directory))
            validate_result_for_refresh(selected_directory)
        elif selected_directory.exists() or selected_directory.is_symlink():
            raise StoreError(
                f"The optimization result directory {selected_directory} exists without a "
                "completion manifest. It has not been changed. Create a new workspace and run "
                "the workflows there."
            )
        ensure_optimization_results_guide(target)
    except StoreError as exc:
        raise RunOptimizationError(f"Cannot use {selected_directory}: {exc}") from exc

    _progress(foreground, "checking the required LCA results")
    try:
        baseline_references = tuple(
            select_baseline_lca_result(
                workspace,
                foreground_scenario=foreground,
                background_scenario=background,
                selection=request.lcia_method,
            )
            for background in request.background_scenarios
        )
    except CalculationResultError as exc:
        raise RunOptimizationError(f"Cannot use the required LCA results: {exc}") from exc
    _progress(foreground, "loading the MFA result")
    from pytoce.run_mfa.model.result_inputs import read_mfa_result, read_mfa_result_manifest
    from pytoce.run_mfa.model.results import selected_mfa_result

    mfa_path = selected_mfa_result(workspace, foreground)
    if mfa_path is None:
        raise RunOptimizationError(
            "No completed MFA result was found for "
            f"foreground_scenario={foreground!r}. Run run_mfa for that foreground scenario, "
            "then run run_optimization again."
        )
    mfa_reference = read_mfa_result_manifest(mfa_path)
    parent_mfa = read_mfa_result(mfa_reference)
    _progress(foreground, "loading the aCC result")
    try:
        allocated_cc = compile_allocated_cc_result(workspace, request.allocated_cc_result)
        allocated_cc_cases = select_allocated_cc_cases(
            allocated_cc,
            request.allocated_cc_cases,
        )
    except (AllocatedCCError, OSError) as exc:
        raise RunOptimizationError(
            f"Cannot use allocated_cc_result={request.allocated_cc_result!r}: {exc}"
        ) from exc
    _progress(foreground, "preparing optimization response tables")
    try:
        initializer = _prepare_worker_initializer(
            workspace,
            request,
            foreground,
            allocated_cc,
            allocated_cc_cases,
            mfa_reference,
        )
        context = prepare_candidate_worker_request_context(
            initializer,
            allocated_cc,
            parent_mfa=parent_mfa,
            result_lcia_method=request.lcia_method,
        )
        del parent_mfa
    except (
        AllocatedCCError,
        CandidateEvaluationError,
        ResponseBankError,
        ResponseBuilderError,
        PreparedDataError,
        OSError,
    ) as exc:
        raise RunOptimizationError(
            f"Cannot prepare optimization inputs for {foreground}: {exc}"
        ) from exc
    metric_scopes = build_candidate_worker_metric_scopes(request.optimization_blocks)
    candidate_input_identity = build_candidate_input_identity(
        planner=context.planner,
        worker_scientific_authority_identity=context.worker_scientific_authority_identity,
        search_scoring_identity=context.search_scoring.identity,
        metric_scopes=metric_scopes,
    )
    try:
        prefix = read_refinement_prefix(
            target,
            result_lcia_selection=request.result_lcia_selection,
            candidate_input_identity=candidate_input_identity,
            selection_method_identity=CANDIDATE_SELECTION_METHOD_IDENTITY,
            requested_blocks=request.optimization_blocks,
        )
        continuation_parent = None if prefix is None else prefix.product_identity
        request_identity = build_optimization_request_identity(
            foreground_scenario=foreground,
            allocated_cc_result=request.allocated_cc_result,
            optimization_lcia_selection=_OPTIMIZATION_LCIA_SELECTION,
            search_scoring_identity=context.search_scoring.identity,
            readable_request_selection=request.readable_request_selection,
            planner=context.planner,
            requested_blocks=request.optimization_blocks,
            initial_grid_step_pct=request.initial_grid_step_pct,
            endpoint_refinement_steps_pct=request.endpoint_refinement_steps_pct,
            continuation_parent_identity=continuation_parent,
        )
        with CandidateResultStore.open(
            target.root,
            candidate_input_identity=candidate_input_identity,
            worker_authority_identity=context.planner.worker_authority_identity,
        ) as result_store:
            with CandidateProcessPool(
                worker_initializer_payload=context.worker_initializer_payload,
                candidate_workers=request.candidate_workers,
            ) as scheduler:
                bridge = CandidateWorkerRequestBridge(
                    context.planner,
                    metric_scopes,
                    result_store=result_store,
                    worker_scientific_authority_identity=(
                        context.worker_scientific_authority_identity
                    ),
                    allocated_cc_context=context.allocated_cc_context,
                    search_scoring=context.search_scoring,
                )
                _progress(foreground, "testing intervention combinations")
                execution = run_optimization_search(
                    request_identity=request_identity,
                    planner=context.planner,
                    bridge=bridge,
                    scheduler=scheduler,
                    result_store=result_store,
                    requested_blocks=request.optimization_blocks,
                    candidate_input_identity=candidate_input_identity,
                    initial_grid_step_pct=request.initial_grid_step_pct,
                    endpoint_refinement_steps_pct=request.endpoint_refinement_steps_pct,
                    continuation_parent_identity=continuation_parent,
                    completed_search=None if prefix is None else prefix.completed_search,
                )
            _progress(foreground, "reading the required LCA results")
            baseline_lca_results = tuple(
                load_baseline_lca_result(reference) for reference in baseline_references
            )
            _progress(foreground, "writing selected results")
            with stage_selected_life_cycle_scores(selected_directory) as stage_writer:
                selected = build_selected_result_tables(
                    execution,
                    request_context=context,
                    baseline_lca_results=baseline_lca_results,
                    stage_writer=stage_writer,
                    stage_progress=lambda completed, total: _progress(
                        foreground,
                        f"writing selected results ({completed}/{total})",
                    ),
                )
                selected_receipt = write_optimization_result(
                    selected_directory,
                    product_identity=selected.identity,
                    parent_identities=selected.parent_identities,
                    method_identity=selected.method_identity,
                    tables=selected.tables,
                    summary=selected.summary,
                    refresh=request.refresh,
                )
            product = _product_report(foreground, selected_receipt)
    except (
        CandidateEvaluationError,
        AllocatedCCError,
        CandidateResultStoreError,
        CalculationResultError,
        ResponseBankError,
        ResponseBuilderError,
        PreparedDataError,
        SchedulerError,
        StoreError,
        OSError,
    ) as exc:
        raise RunOptimizationError(f"Optimization failed for {foreground}: {exc}") from exc
    _progress(foreground, "calculation complete")
    return product


def _prepare_worker_initializer(
    workspace: Path,
    request: RunOptimizationRequest,
    foreground: str,
    allocated_cc: "AllocatedCCResult",
    allocated_cc_cases: tuple[str, ...],
    mfa_reference: "MFAResultReference",
) -> bytes:
    """Prepare responses and serialize their references, selected allocated carrying
    capacity cases, and foreground input identities for workers."""
    from .calculation.response_builder import (
        ensure_optimization_response_bank,
        response_method_selection,
    )
    from .calculation.worker_inputs import prepare_candidate_worker_initializer

    selection = response_method_selection(request.lcia_method)
    bank = ensure_optimization_response_bank(
        workspace,
        selection=selection,
        member_progress=lambda background: _progress(
            foreground, f"preparing response tables for {background}"
        ),
    )
    return prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=foreground,
        mfa_reference=mfa_reference,
        allocated_cc=allocated_cc,
        allocated_cc_case=allocated_cc_cases,
        response_method_union=bank.response_method_union,
        background_scenarios=request.background_scenarios,
    )


def _product_report(
    foreground: str,
    receipt: "SavedOptimizationResult",
) -> RunOptimizationProductReport:
    """Build the public report for one completed optimization result."""
    return RunOptimizationProductReport(
        status=receipt.status,
        foreground_scenario=foreground,
        product_identity=receipt.product_identity,
        result_directory=receipt.directory,
        manifest_path=receipt.manifest_path,
    )


def _progress(foreground: str, message: str) -> None:
    """Print one concise optimization progress message."""
    print(
        f"run_optimization: {foreground}: {message}",
        file=sys.stderr,
    )


def _selection_names(value: str | Sequence[str], *, parameter: str) -> tuple[str, ...]:
    """Normalize one name or a sequence of names for a public argument."""
    try:
        raw: tuple[object, ...] = (value,) if isinstance(value, str) else tuple(value)
    except TypeError as exc:
        raise RunOptimizationError(
            f"{parameter}={value!r} is invalid. Use one readable name or a sequence of "
            "readable names."
        ) from exc
    if not raw:
        raise RunOptimizationError(
            f"{parameter}={value!r} is empty. Select at least one readable name."
        )
    return tuple(_readable_name(item, parameter=parameter) for item in raw)


def _readable_name(value: object, *, parameter: str) -> str:
    """Validate one lowercase ``snake_case`` public name."""
    if not isinstance(value, str) or not _READABLE_NAME.fullmatch(value):
        raise RunOptimizationError(
            f"{parameter}={value!r} is invalid. Use a lowercase snake_case readable name."
        )
    return value


def _lcia_selection(value: str | Sequence[str], *, parameter: str) -> "LCIAMethodSelection":
    """Normalize one public LCIA method selection."""
    from pytoce.run_lca.calculation.lcia import normalized_lcia_method

    try:
        return normalized_lcia_method(value)
    except (TypeError, ValueError) as exc:
        raise RunOptimizationError(f"Invalid {parameter} selection: {exc}") from exc


def _optimization_blocks(
    value: OptimizationBlock | Sequence[OptimizationBlock],
) -> tuple[OptimizationBlock, ...]:
    """Validate and order the requested optimization blocks."""
    try:
        raw: tuple[object, ...] = (value,) if isinstance(value, str) else tuple(value)
    except TypeError as exc:
        raise RunOptimizationError(
            f"optimization_block={value!r} is invalid. Select one block or a sequence from "
            f"{OPTIMIZATION_BLOCK_ORDER!r}."
        ) from exc
    if not raw:
        raise RunOptimizationError(
            "optimization_block is empty. Select at least one block from "
            f"{OPTIMIZATION_BLOCK_ORDER!r}."
        )
    unknown = tuple(item for item in raw if item not in OPTIMIZATION_BLOCK_ORDER)
    if unknown:
        raise RunOptimizationError(
            f"Unknown optimization_block value(s): {unknown!r}. Choose from "
            f"{OPTIMIZATION_BLOCK_ORDER!r}."
        )
    selected = set(raw)
    return tuple(block for block in OPTIMIZATION_BLOCK_ORDER if block in selected)


def _readable_request_selection(
    blocks: tuple[OptimizationBlock, ...],
    cases: tuple[str, ...] | None,
    backgrounds: tuple[str, ...],
) -> str:
    """Build the directory name that identifies the requested blocks and cases."""
    block_name = "blocks_" + "_".join(_BLOCK_LABELS[block] for block in blocks)
    case_name = "all" if cases is None else "_".join(f"{len(case)}_{case}" for case in cases)
    selection = f"{block_name}__cases_{case_name}"
    if backgrounds != _REAL_BACKGROUNDS:
        selection += "__backgrounds_" + "_".join(
            background.removeprefix("remind_ssp2_") for background in backgrounds
        )
    return selection


def _endpoint_refinement_steps(value: Sequence[int], initial_grid_step_pct: int) -> tuple[int, ...]:
    """Admit the ordered endpoint search schedule once at the public boundary."""
    message = (
        "endpoint_refinement_steps_pct must be a nonempty, strictly decreasing sequence "
        "of distinct positive integer percentage points smaller than initial_grid_step_pct, "
        "for example (10, 5) "
        "or (5,). Boolean values are not percentage points."
    )
    try:
        steps = tuple(value)
    except TypeError as exc:
        raise RunOptimizationError(message) from exc
    if not steps or any(
        type(step) is not int or not 1 <= step < initial_grid_step_pct for step in steps
    ):
        raise RunOptimizationError(message)
    if any(left <= right for left, right in zip(steps, steps[1:])):
        raise RunOptimizationError(message)
    return steps
