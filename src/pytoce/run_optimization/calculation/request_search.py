"""Run the intervention candidate search for one optimization request."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from pytoce.run_optimization.blocks import (
    OPTIMIZATION_BLOCK_ORDER,
    OptimizationBlock,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateInterventionAuthority,
)
from pytoce.run_optimization.calculation.candidate_search import (
    _REFINEMENT_STEPS_PCT,
    CANDIDATE_REFINEMENT_METHOD_IDENTITY,
    CANDIDATE_SELECTION_METHOD_IDENTITY,
    CIRCULARITY_FEEDSTOCK_DESIGN_METHOD_IDENTITY,
    DEMAND_LIFETIME_DESIGN_METHOD_IDENTITY,
    CandidateEvaluation,
    CandidateRefinement,
    CandidateSearchState,
    CandidateStateMetrics,
    CompetitiveCandidateBatch,
    InterventionCandidate,
    advance_candidate_search,
    candidate_coordinate_key,
    circularity_feedstock_pairwise_design,
    demand_lifetime_initial_design,
    finish_candidate_search,
    initialize_candidate_search,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerMetricScope,
    CandidateWorkerPlanner,
    CandidateWorkerRequestBridge,
    CandidateWorkerResultSummary,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    DEMAND_REPAIR_METHOD_IDENTITY,
    STOCK_PROPOSAL_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.integrated_design import (
    INTEGRATED_DESIGN_METHOD_IDENTITY,
    JointStageDesign,
    build_joint_stage_design,
    joint_pair_cover,
)
from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
    IntegratedSearchState,
    advance_integrated_search,
    finish_integrated_search,
    initialize_integrated_search,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.regional_metrics import intervention_distance
from pytoce.run_optimization.calculation.worker_protocol import (
    DemandRepairOutcome,
    SelectedCandidateResult,
    _selected_candidate_result,
)
from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
    CandidateTask,
    CandidateTaskRecord,
)
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool
from pytoce.shared.runtime.filesystem import payload_sha256

CANDIDATE_SEARCH_CONFIGURATION_METHOD_IDENTITY: Final = "pytoce_candidate_search_configuration_v9"
OPTIMIZATION_REQUEST_METHOD_IDENTITY: Final = "pytoce_optimization_request_v5"
METRIC_SCOPE_METHOD_IDENTITY: Final = "pytoce_metric_scope_v3"
_REQUEST_IDENTITY_PREFIX = "pytoce_optimization_request_v5::"
_SEARCH_IDENTITY_PREFIX = "pytoce_candidate_search_v6::"
_SEARCH_TARGET_YEAR = 2050


def search_configuration_identity(
    endpoint_refinement_steps_pct: tuple[int, ...], *, initial_grid_step_pct: int = 20
) -> str:
    """Bind the full endpoint schedule and fixed scientific search methods."""
    return (
        CANDIDATE_SEARCH_CONFIGURATION_METHOD_IDENTITY
        + "::"
        + payload_sha256(
            {
                "method_identity": CANDIDATE_SEARCH_CONFIGURATION_METHOD_IDENTITY,
                "target_year": _SEARCH_TARGET_YEAR,
                "initial_grid_step_pct": initial_grid_step_pct,
                "demand_lifetime_design_method": DEMAND_LIFETIME_DESIGN_METHOD_IDENTITY,
                "circularity_feedstock_design_method": CIRCULARITY_FEEDSTOCK_DESIGN_METHOD_IDENTITY,
                "refinement_method": CANDIDATE_REFINEMENT_METHOD_IDENTITY,
                "integrated_search_method": INTEGRATED_SEARCH_METHOD_IDENTITY,
                "integrated_design_method": INTEGRATED_DESIGN_METHOD_IDENTITY,
                "refinement_steps_pct": list(endpoint_refinement_steps_pct),
                "selection_method": CANDIDATE_SELECTION_METHOD_IDENTITY,
                "stock_proposal_method": STOCK_PROPOSAL_METHOD_IDENTITY,
                "demand_repair_method": DEMAND_REPAIR_METHOD_IDENTITY,
                "stock_proposal_solver": "scipy1.13.1:highs-ds",
            }
        )
    )


@dataclass(frozen=True, slots=True)
class SearchBlockResult:
    """One block's completed refinement and selected candidates."""

    optimization_block: OptimizationBlock
    refinement: CandidateRefinement
    selected_candidates: tuple[InterventionCandidate, ...]
    selected_states: tuple[CandidateStateMetrics, ...]


@dataclass(frozen=True, slots=True)
class RequestSearchStage:
    """Keep completed single-block and joint stages, including hidden prerequisites."""

    step_pct: int
    single_blocks: tuple[CandidateSearchState, ...]
    integrated: IntegratedSearchState | None
    demand_anchors: tuple[InterventionCandidate, ...]
    circularity_anchors: tuple[InterventionCandidate, ...]


@dataclass(frozen=True, slots=True)
class CompletedCandidateSearch:
    """Bind completed stages to the compact values and tasks needed for continuation."""

    initial_grid_step_pct: int
    endpoint_refinement_steps_pct: tuple[int, ...]
    executed_blocks: tuple[OptimizationBlock, ...]
    stages: tuple[RequestSearchStage, ...]
    candidate_records: tuple[tuple[str, str, str], ...]
    worker_summaries: tuple[CandidateWorkerResultSummary, ...]
    task_records: tuple[CandidateTaskRecord, ...]


@dataclass(frozen=True, slots=True)
class RequestSearchResult:
    """Completed candidate search for one optimization request."""

    identity: str
    block_results: tuple[SearchBlockResult, ...]
    initial_grid_step_pct: int = 20
    endpoint_refinement_steps_pct: tuple[int, ...] = _REFINEMENT_STEPS_PCT
    candidate_input_identity: str = ""
    continuation_parent_identity: str | None = None
    completed_stages: tuple[RequestSearchStage, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedCandidateEvaluation:
    """Keep a selected candidate with its saved calculation and selection identities."""

    candidate: InterventionCandidate
    selection_identities: tuple[str, ...]
    selected_result: SelectedCandidateResult


@dataclass(frozen=True, slots=True)
class SearchExecutionProducts:
    """Selected 2050 and fixed-target timing evaluations ready for result tables."""

    search: RequestSearchResult
    selected_2050: tuple[SelectedCandidateEvaluation, ...]
    selected_timing: tuple[SelectedCandidateEvaluation, ...]
    worker_result_set_identity: str
    request_coordinate_count: int
    request_unique_physical_candidate_count: int
    request_physical_candidate_deduplication_count: int
    physical_repair_elapsed_seconds: float
    environmental_evaluation_elapsed_seconds: float
    mfa_evaluation_count: int
    stock_lp_count: int
    environmental_evaluation_count: int
    repair_status_counts: tuple[tuple[str, int], ...]
    completed_search: CompletedCandidateSearch | None = None
    newly_computed_worker_result_count: int = 0
    reused_worker_result_count: int = 0
    incremental_physical_repair_elapsed_seconds: float = 0.0
    incremental_environmental_evaluation_elapsed_seconds: float = 0.0
    incremental_mfa_evaluation_count: int = 0
    incremental_stock_lp_count: int = 0


def build_optimization_request_identity(
    *,
    foreground_scenario: str,
    allocated_cc_result: str,
    optimization_lcia_selection: str,
    search_scoring_identity: str,
    readable_request_selection: str,
    planner: CandidateWorkerPlanner,
    requested_blocks: Sequence[OptimizationBlock],
    initial_grid_step_pct: int = 20,
    endpoint_refinement_steps_pct: tuple[int, ...] = _REFINEMENT_STEPS_PCT,
    continuation_parent_identity: str | None = None,
) -> str:
    """Build the optimization request identity from its checked inputs."""
    blocks = tuple(requested_blocks)
    return _REQUEST_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": OPTIMIZATION_REQUEST_METHOD_IDENTITY,
            "foreground_scenario": foreground_scenario,
            "allocated_cc_result": allocated_cc_result,
            "optimization_lcia_selection": optimization_lcia_selection,
            "search_scoring_identity": search_scoring_identity,
            "readable_request_selection": readable_request_selection,
            "planner_identity": planner.identity,
            "requested_blocks": list(blocks),
            "configuration_identity": search_configuration_identity(
                endpoint_refinement_steps_pct, initial_grid_step_pct=initial_grid_step_pct
            ),
            "continuation_parent_identity": continuation_parent_identity,
        }
    )


def build_candidate_worker_metric_scopes(
    requested_blocks: Sequence[OptimizationBlock],
) -> tuple[CandidateWorkerMetricScope, ...]:
    """Build metric scopes for requested blocks and any required single-block prerequisites."""
    return tuple(
        CandidateWorkerMetricScope(
            optimization_block=block,
            metric_registry_identity=METRIC_SCOPE_METHOD_IDENTITY
            + "::"
            + payload_sha256(
                {
                    "method_identity": METRIC_SCOPE_METHOD_IDENTITY,
                    "optimization_block": block,
                }
            ),
        )
        for block in _execution_blocks(requested_blocks)
    )


def build_candidate_input_identity(
    *,
    planner: CandidateWorkerPlanner,
    worker_scientific_authority_identity: str,
    search_scoring_identity: str,
    metric_scopes: Sequence[CandidateWorkerMetricScope],
) -> str:
    """Bind reusable MFA/scoring inputs independently of endpoint refinement.

    The planner binds the actual parent, worker methods, and selected allocated
    carrying capacity labels. Scientific authority binds recipes and
    characterized responses.
    Repair and metric methods remain explicit because changing either can alter
    a saved representative or its interpretation. Only outer search scheduling
    is omitted, so later finer refinement can read matching completed results.
    """
    return "pytoce_candidate_calculation_inputs_v1::" + payload_sha256(
        {
            "planner_identity": planner.identity,
            "worker_authority_identity": planner.worker_authority_identity,
            "worker_scientific_authority_identity": worker_scientific_authority_identity,
            "search_scoring_identity": search_scoring_identity,
            "stock_proposal_method": STOCK_PROPOSAL_METHOD_IDENTITY,
            "demand_repair_method": DEMAND_REPAIR_METHOD_IDENTITY,
            "metric_scopes": [
                [scope.optimization_block, scope.metric_registry_identity]
                for scope in metric_scopes
            ],
        }
    )


def _compatible_metric_scopes(
    candidate: InterventionCandidate,
    request_scopes: Sequence[CandidateWorkerMetricScope],
) -> tuple[CandidateWorkerMetricScope, ...]:
    """Return every requested block whose distance family fits the intervention."""

    allowed: set[OptimizationBlock] = {"integrated_frontier"}
    if candidate.circularity.uses_exact_parent:
        allowed.add("demand_lifetime_frontier")
    if candidate.demand_lifetime.uses_exact_parent:
        allowed.add("circularity_feedstock_anchor")
    return tuple(scope for scope in request_scopes if scope.optimization_block in allowed)


class CandidateWorkerEvaluator:
    """Evaluate each intervention once and reuse equivalent candidates."""

    def __init__(
        self,
        *,
        planner: CandidateWorkerPlanner,
        bridge: CandidateWorkerRequestBridge,
        scheduler: CandidateProcessPool,
        result_store: CandidateResultStore,
    ) -> None:
        """Bind candidate planning, process execution, and reusable result summaries."""
        self._planner = planner
        self._bridge = bridge
        self._scheduler = scheduler
        self._result_store = result_store
        self._summary_by_intervention: dict[str, CandidateWorkerResultSummary] = {}
        self._candidate_records: dict[str, tuple[str, str]] = {}

    @property
    def candidate_records(self) -> tuple[tuple[str, str, str], ...]:
        """List each candidate with its intervention and result summary identities."""
        return tuple(
            (candidate, intervention, summary)
            for candidate, (intervention, summary) in sorted(self._candidate_records.items())
        )

    @property
    def result_summary_identities(self) -> tuple[str, ...]:
        """List the unique result summary identities in stable order."""
        return tuple(sorted(item.identity for item in self._summary_by_intervention.values()))

    @property
    def repair_outcomes(self) -> tuple[DemandRepairOutcome, ...]:
        """Expose embodied producer work, including reused completed endpoint results."""
        return tuple(item.repair for item in self._summary_by_intervention.values())

    @property
    def worker_summaries(self) -> tuple[CandidateWorkerResultSummary, ...]:
        """Return compact summaries once in original task order for completed-stage custody."""
        return tuple(
            self._summary_by_intervention[item.candidate_identity]
            for item in self._bridge.task_records
        )

    @property
    def task_records(self) -> tuple[CandidateTaskRecord, ...]:
        """Expose completed task sequence."""
        return self._bridge.task_records

    @property
    def newly_computed_repairs(self) -> tuple[DemandRepairOutcome, ...]:
        """Separate new worker effort from costs embodied in restored or cached results."""
        computed = self._scheduler.computed_candidate_identities
        return tuple(
            summary.repair
            for identity, summary in self._summary_by_intervention.items()
            if identity in computed
        )

    def restore_completed_search(self, completed: CompletedCandidateSearch) -> None:
        """Restore completed compact values and references for selected-result loading."""
        self._bridge.restore_task_records(completed.task_records)
        self._summary_by_intervention = {
            item.requested_intervention_identity: item for item in completed.worker_summaries
        }
        self._candidate_records = {
            candidate: (intervention, summary)
            for candidate, intervention, summary in completed.candidate_records
        }

    def worker_result_set_identity(self) -> str:
        """Identify the complete set of saved candidate results."""

        return self._bridge.finalize_result_set(self._summary_by_intervention)

    def _plan(
        self,
        candidates: tuple[InterventionCandidate, ...],
    ) -> dict[str, CandidateInterventionAuthority]:
        """Compile each candidate into the intervention expected by the worker."""
        return {
            candidate.identity: self._planner.plan_intervention(
                demand_lifetime=candidate.demand_lifetime,
                circularity=candidate.circularity,
            )
            for candidate in candidates
        }

    def _evaluate_planned(
        self,
        candidates: tuple[InterventionCandidate, ...],
        interventions: Mapping[str, CandidateInterventionAuthority],
    ) -> tuple[CandidateEvaluation, ...]:
        """Evaluate new interventions and convert their summaries to candidate metrics."""
        missing: dict[str, CandidateTask] = {}
        for candidate in candidates:
            intervention_input = interventions[candidate.identity]
            intervention = intervention_input.identity
            if intervention not in self._summary_by_intervention and intervention not in missing:
                missing[intervention] = self._bridge.build_task(
                    intervention_input,
                    metric_scopes=_compatible_metric_scopes(
                        candidate,
                        self._bridge.metric_scopes,
                    ),
                )
        if missing:
            tasks = tuple(sorted(missing.values(), key=lambda item: item.sequence))
            summaries = self._scheduler.evaluate_batch(
                tasks,
                result_store=self._result_store,
                read_result=self._bridge.read_saved_result,
                prepare_dispatch=self._bridge.prepare_dispatch,
            )
            for summary in summaries:
                self._summary_by_intervention[summary.requested_intervention_identity] = summary
            repairs = self.repair_outcomes
            counts = Counter(item.status for item in repairs)
            print(
                f"run_optimization: endpoints={len(repairs)} "
                f"physical_checks={sum(item.mfa_evaluation_count for item in repairs)} "
                f"stock_solves={sum(item.stock_lp_count for item in repairs)} "
                "physical_seconds="
                f"{sum(item.physical_repair_elapsed_seconds for item in repairs):.1f} "
                "environmental_seconds="
                f"{sum(item.environmental_evaluation_elapsed_seconds for item in repairs):.1f} "
                + " ".join(f"{status}={count}" for status, count in sorted(counts.items())),
                flush=True,
            )
        outputs: list[CandidateEvaluation] = []
        for candidate in candidates:
            intervention = interventions[candidate.identity].identity
            summary = self._summary_by_intervention[intervention]
            self._candidate_records[candidate.identity] = (intervention, summary.identity)
            outputs.append(_candidate_evaluation(candidate, summary))
        return tuple(outputs)

    def __call__(
        self, candidates: tuple[InterventionCandidate, ...]
    ) -> tuple[CandidateEvaluation, ...]:
        """Plan and evaluate a batch of intervention candidates."""
        return self._evaluate_planned(candidates, self._plan(candidates))

    def evaluate_competitive(
        self,
        candidates: tuple[InterventionCandidate, ...],
        distance_ceiling: float,
    ) -> CompetitiveCandidateBatch:
        """Plan once and evaluate only candidates that can beat a compliant incumbent."""

        interventions = self._plan(candidates)
        distances = tuple(
            (
                candidate.identity,
                intervention_distance(
                    physical_candidate_identity=candidate.identity,
                    intervention_authority_identity=(interventions[candidate.identity].identity),
                    distance_inputs=interventions[candidate.identity].distance_inputs,
                    optimization_block=candidate.optimization_block,
                ).total_distance,
            )
            for candidate in candidates
        )
        distance_by_candidate = dict(distances)
        retained = tuple(
            candidate
            for candidate in candidates
            if distance_by_candidate[candidate.identity] <= distance_ceiling
        )
        retained_interventions = {
            candidate.identity: interventions[candidate.identity] for candidate in retained
        }
        return CompetitiveCandidateBatch(
            candidate_distances=distances,
            evaluations=self._evaluate_planned(retained, retained_interventions),
        )

    def selected(
        self,
        candidates: Sequence[InterventionCandidate],
        *,
        selection_identities: Mapping[str, Sequence[str]],
    ) -> tuple[SelectedCandidateEvaluation, ...]:
        """Load previously evaluated selections."""
        ordered = tuple(candidates)
        result_by_intervention = {
            intervention: self._bridge.load_candidate(intervention)
            for intervention in {self._candidate_records[item.identity][0] for item in ordered}
        }
        return tuple(
            SelectedCandidateEvaluation(
                candidate=candidate,
                selection_identities=tuple(sorted(selection_identities[candidate.identity])),
                selected_result=_selected_candidate_result(
                    result_by_intervention[self._candidate_records[candidate.identity][0]],
                    self._summary_by_intervention[self._candidate_records[candidate.identity][0]],
                    selection_identities=selection_identities[candidate.identity],
                ),
            )
            for candidate in ordered
        )


def run_optimization_search(
    *,
    request_identity: str,
    planner: CandidateWorkerPlanner,
    bridge: CandidateWorkerRequestBridge,
    scheduler: CandidateProcessPool,
    result_store: CandidateResultStore,
    requested_blocks: Sequence[OptimizationBlock],
    candidate_input_identity: str,
    initial_grid_step_pct: int = 20,
    endpoint_refinement_steps_pct: tuple[int, ...] = _REFINEMENT_STEPS_PCT,
    continuation_parent_identity: str | None = None,
    completed_search: CompletedCandidateSearch | None = None,
) -> SearchExecutionProducts:
    """Search requested decision groups and calculate the selected timing cases.

    At each spacing the two single-group searches run before the joint stage;
    their actual selections seed its combinations and routes. Intervention identities
    permit reuse across blocks; search-candidate identities also bind the block.
    The selected 2050 endpoints start fresh linear ramps for all active controls
    with completion year 2100, followed by the same conditional demand-only repair.
    Rejected timings and failed demand repairs remain recorded outcomes with the 2050 selection
    retained. The unchanged baseline is reused.

    Args:
        request_identity: Content identifier of the complete optimization request.
        planner: Converts candidates to worker tasks and shared input identities.
        bridge: Loads completed candidate results and calculates their metrics.
        scheduler: Runs candidate tasks in the process pool.
        result_store: Completed calculations with compatible scientific input authority.
        requested_blocks: Ordered decision groups requested by the user.
        candidate_input_identity: Authority of reusable MFA and environmental calculations.
        initial_grid_step_pct: Admitted coarse percentage-point spacing.
        endpoint_refinement_steps_pct: Ordered endpoint neighborhood increments.
        continuation_parent_identity: Explicit compatible prefix result, when available.
        completed_search: Complete compatible stages and compact worker custody from a prefix.

    Returns:
        Search results, selected 2050 calculations, selected 2100
        calculations, and request-level candidate reuse counts.
    """

    evaluator = CandidateWorkerEvaluator(
        planner=planner,
        bridge=bridge,
        scheduler=scheduler,
        result_store=result_store,
    )
    if completed_search is not None:
        evaluator.restore_completed_search(completed_search)
    search = _run_candidate_search(
        request_identity=request_identity,
        requested_blocks=requested_blocks,
        endpoint_refinement_steps_pct=endpoint_refinement_steps_pct,
        candidate_input_identity=candidate_input_identity,
        initial_grid_step_pct=initial_grid_step_pct,
        continuation_parent_identity=continuation_parent_identity,
        prior_stages=() if completed_search is None else completed_search.stages,
        evaluator=evaluator,
    )
    candidates_by_identity = {
        candidate.identity: candidate
        for block in search.block_results
        for candidate in block.selected_candidates
    }
    selection_states: dict[str, list[str]] = {}
    for block in search.block_results:
        for state in block.refinement.selection.states:
            selection_states.setdefault(state.candidate_identity, []).append(state.state_identity)
    selected_candidates = tuple(
        candidates_by_identity[identity] for identity in sorted(selection_states)
    )
    timing_candidates, timing_roles = _selected_timing_candidates(
        selected_candidates,
        selection_identities=selection_states,
    )
    print("run_optimization: evaluating selected endpoint timings", flush=True)
    evaluator(timing_candidates)
    # Release worker inputs before loading full selected payloads for reporting.
    scheduler.close()
    selected_2050 = evaluator.selected(
        selected_candidates,
        selection_identities=selection_states,
    )
    selected_timing = evaluator.selected(
        timing_candidates,
        selection_identities=timing_roles,
    )
    worker_result_set_identity = evaluator.worker_result_set_identity()
    request_coordinate_count = len(evaluator.candidate_records)
    request_unique_physical_candidate_count = len(evaluator.result_summary_identities)
    repairs = evaluator.repair_outcomes
    new_repairs = evaluator.newly_computed_repairs
    return SearchExecutionProducts(
        search=search,
        selected_2050=selected_2050,
        selected_timing=selected_timing,
        worker_result_set_identity=worker_result_set_identity,
        physical_repair_elapsed_seconds=sum(
            item.physical_repair_elapsed_seconds for item in repairs
        ),
        environmental_evaluation_elapsed_seconds=sum(
            item.environmental_evaluation_elapsed_seconds for item in repairs
        ),
        mfa_evaluation_count=sum(item.mfa_evaluation_count for item in repairs),
        stock_lp_count=sum(item.stock_lp_count for item in repairs),
        environmental_evaluation_count=sum(
            item.status in {"linear_feasible", "stock_projection_feasible", "local_pool_feasible"}
            for item in repairs
        ),
        repair_status_counts=tuple(sorted(Counter(item.status for item in repairs).items())),
        request_coordinate_count=request_coordinate_count,
        request_unique_physical_candidate_count=request_unique_physical_candidate_count,
        request_physical_candidate_deduplication_count=(
            request_coordinate_count - request_unique_physical_candidate_count
        ),
        completed_search=CompletedCandidateSearch(
            initial_grid_step_pct=initial_grid_step_pct,
            endpoint_refinement_steps_pct=endpoint_refinement_steps_pct,
            executed_blocks=_execution_blocks(requested_blocks),
            stages=search.completed_stages,
            candidate_records=evaluator.candidate_records,
            worker_summaries=evaluator.worker_summaries,
            task_records=evaluator.task_records,
        ),
        newly_computed_worker_result_count=len(new_repairs),
        reused_worker_result_count=len(repairs) - len(new_repairs),
        incremental_physical_repair_elapsed_seconds=sum(
            item.physical_repair_elapsed_seconds for item in new_repairs
        ),
        incremental_environmental_evaluation_elapsed_seconds=sum(
            item.environmental_evaluation_elapsed_seconds for item in new_repairs
        ),
        incremental_mfa_evaluation_count=sum(item.mfa_evaluation_count for item in new_repairs),
        incremental_stock_lp_count=sum(item.stock_lp_count for item in new_repairs),
    )


def _selected_timing_candidates(
    selected_2050: Sequence[InterventionCandidate],
    *,
    selection_identities: Mapping[str, Sequence[str]],
) -> tuple[tuple[InterventionCandidate, ...], dict[str, list[str]]]:
    """Construct selected 2100 endpoints and roles.

    Unchanged parents reuse their existing path. Shared timing candidates retain
    every originating selection identity in the same canonical order.
    """
    timing_candidates: dict[str, InterventionCandidate] = {}
    timing_roles: dict[str, list[str]] = {}
    for selected in selected_2050:
        timing = _timing_candidate(selected, target_year=2100)
        if timing is None:
            continue
        timing_candidates.setdefault(timing.identity, timing)
        timing_roles.setdefault(timing.identity, []).extend(
            sorted(selection_identities[selected.identity])
        )
    return (
        tuple(timing_candidates[identity] for identity in sorted(timing_candidates)),
        timing_roles,
    )


def _timing_candidate(
    source: InterventionCandidate,
    *,
    target_year: Literal[2100],
) -> InterventionCandidate | None:
    """Retain selected endpoints and construct fresh linear controls for completion in 2100."""

    if source.demand_lifetime.uses_exact_parent and source.circularity.uses_exact_parent:
        return None
    demand = DemandLifetimeDecision(
        source.demand_lifetime.demand_reduction_pct,
        source.demand_lifetime.lifetime_extension_pct,
        target_year,
    )
    circularity = (
        CircularityDecision.parent(target_year=target_year)
        if source.circularity.uses_exact_parent
        else replace(source.circularity, target_year=target_year)
    )
    return InterventionCandidate(source.optimization_block, demand, circularity)


def _run_candidate_search(
    *,
    request_identity: str,
    requested_blocks: Sequence[OptimizationBlock],
    evaluator: CandidateWorkerEvaluator,
    candidate_input_identity: str,
    initial_grid_step_pct: int = 20,
    endpoint_refinement_steps_pct: tuple[int, ...] = _REFINEMENT_STEPS_PCT,
    continuation_parent_identity: str | None = None,
    prior_stages: tuple[RequestSearchStage, ...] = (),
) -> RequestSearchResult:
    """Complete I/II then III at each spacing, appending only missing stages.

    Single-block state contains only proposals from that block's own search.
    Each single-block state retains its own evaluated candidates. The coarse phase admits the joint
    design and routes; local
    branch polling begins at the first configured refinement spacing.
    """
    blocks = tuple(requested_blocks)
    executed = _execution_blocks(blocks)
    stages = list(prior_stages)
    single: dict[OptimizationBlock, CandidateSearchState] = {}
    integrated: IntegratedSearchState | None = None
    demand_anchors: dict[str, InterventionCandidate] = {}
    circularity_anchors: dict[str, InterventionCandidate] = {}
    if stages:
        last = stages[-1]
        single = {item.optimization_block: item for item in last.single_blocks}
        integrated = last.integrated
        demand_anchors = {item.identity: item for item in last.demand_anchors}
        circularity_anchors = {item.identity: item for item in last.circularity_anchors}
    else:
        for block in executed:
            if block == "integrated_frontier":
                continue
            if block == "demand_lifetime_frontier":
                initial = demand_lifetime_initial_design(
                    target_year=_SEARCH_TARGET_YEAR, step_pct=initial_grid_step_pct
                )
            else:
                initial = (
                    *circularity_feedstock_pairwise_design(
                        target_year=_SEARCH_TARGET_YEAR, step_pct=initial_grid_step_pct
                    ),
                    InterventionCandidate(
                        block,
                        DemandLifetimeDecision(0, 0, _SEARCH_TARGET_YEAR),
                        CircularityDecision.parent(target_year=_SEARCH_TARGET_YEAR),
                    ),
                )
            print(f"run_optimization: {block}: {len(initial)} coarse endpoints", flush=True)
            single[block] = initialize_candidate_search(
                optimization_block=block, initial_candidates=initial, evaluate=evaluator
            )
        _retain_single_block_anchors(single, demand_anchors, circularity_anchors)
        if "integrated_frontier" in executed:
            design = build_joint_stage_design(
                block_i_candidates=tuple(demand_anchors.values()),
                block_ii_candidates=tuple(circularity_anchors.values()),
                pair_cover=joint_pair_cover(initial_grid_step_pct=initial_grid_step_pct),
                step_pct=initial_grid_step_pct,
                target_year=_SEARCH_TARGET_YEAR,
            )
            _announce_joint_stage(design, already_observed=set())
            integrated = initialize_integrated_search(design=design, evaluate=evaluator)
        stages.append(
            RequestSearchStage(
                initial_grid_step_pct,
                tuple(single.values()),
                integrated,
                tuple(demand_anchors.values()),
                tuple(circularity_anchors.values()),
            )
        )
    for step_pct in endpoint_refinement_steps_pct[len(stages) - 1 :]:
        for block, state in single.items():
            if any(item.status != "baseline_compliant" for item in state.selection.states):
                single[block] = advance_candidate_search(
                    state,
                    step_pct=step_pct,
                    unpruned=step_pct <= 5 or step_pct == endpoint_refinement_steps_pct[-1],
                    evaluate=evaluator,
                    evaluate_competitive=evaluator.evaluate_competitive,
                )
        _retain_single_block_anchors(single, demand_anchors, circularity_anchors)
        if integrated is not None:
            candidates = {
                item.candidate.identity: item.candidate for item in integrated.evaluations
            }
            design = build_joint_stage_design(
                block_i_candidates=tuple(demand_anchors.values()),
                block_ii_candidates=tuple(circularity_anchors.values()),
                pair_cover=integrated.completed_stages[0].design.pair_cover,
                step_pct=step_pct,
                target_year=_SEARCH_TARGET_YEAR,
            )
            _announce_joint_stage(design, already_observed=set(candidates))
            integrated = advance_integrated_search(integrated, design=design, evaluate=evaluator)
        stages.append(
            RequestSearchStage(
                step_pct,
                tuple(single.values()),
                integrated,
                tuple(demand_anchors.values()),
                tuple(circularity_anchors.values()),
            )
        )
    results = {
        block: _search_block_result(
            block,
            finish_candidate_search(
                state, endpoint_refinement_steps_pct=endpoint_refinement_steps_pct
            ),
            state.evaluations,
        )
        for block, state in single.items()
    }
    if integrated is not None:
        results["integrated_frontier"] = _search_block_result(
            "integrated_frontier",
            finish_integrated_search(
                integrated, endpoint_refinement_steps_pct=endpoint_refinement_steps_pct
            ),
            integrated.evaluations,
        )
    requested_results = tuple(results[block] for block in blocks)
    identity = _SEARCH_IDENTITY_PREFIX + payload_sha256(
        {
            "request_identity": request_identity,
            "configuration_identity": search_configuration_identity(
                endpoint_refinement_steps_pct, initial_grid_step_pct=initial_grid_step_pct
            ),
            "continuation_parent_identity": continuation_parent_identity,
            "requested_blocks": list(blocks),
            "block_refinement_identities": [item.refinement.identity for item in requested_results],
            "candidate_records": [list(item) for item in evaluator.candidate_records],
            "result_summary_identities": list(evaluator.result_summary_identities),
        }
    )
    return RequestSearchResult(
        identity=identity,
        block_results=requested_results,
        endpoint_refinement_steps_pct=endpoint_refinement_steps_pct,
        candidate_input_identity=candidate_input_identity,
        initial_grid_step_pct=initial_grid_step_pct,
        continuation_parent_identity=continuation_parent_identity,
        completed_stages=tuple(stages),
    )


def _announce_joint_stage(design: JointStageDesign, *, already_observed: set[str]) -> None:
    """Report base coverage before admission determines the stage's frozen route guides."""
    reused = sum(item.identity in already_observed for item in design.candidates)
    print(
        f"run_optimization: Block III step={design.step_pct}: "
        f"pair_requirements={design.pair_cover.requirement_count} "
        f"independent_starts={len(design.pair_cover.rows)} "
        f"runtime_ii_anchors={len(design.actual_ii_candidates)} "
        f"base_proposals={len(design.candidates)} "
        f"already_evaluated={reused} pending_admission={len(design.candidates) - reused} "
        "(before shared cache reuse)",
        flush=True,
    )


def _retain_single_block_anchors(
    states: Mapping[OptimizationBlock, CandidateSearchState],
    demand: dict[str, InterventionCandidate],
    circularity: dict[str, InterventionCandidate],
) -> None:
    """Accumulate actual I/II winners."""
    for block, state in states.items():
        selected = set(state.selection.selected_candidate_identities)
        destination = demand if block == "demand_lifetime_frontier" else circularity
        for evaluation in sorted(
            state.evaluations, key=lambda item: candidate_coordinate_key(item.candidate)
        ):
            if evaluation.candidate.identity in selected:
                destination.setdefault(evaluation.candidate.identity, evaluation.candidate)


def _search_block_result(
    block: OptimizationBlock,
    refinement: CandidateRefinement,
    evaluations: Sequence[CandidateEvaluation],
) -> SearchBlockResult:
    """Project selected compact values to requested public block reports."""
    by_candidate = {item.candidate.identity: item for item in evaluations}
    selected = tuple(
        by_candidate[identity].candidate
        for identity in refinement.selection.selected_candidate_identities
    )
    state_index = {
        (metric.state_identity, metric.candidate_identity): metric
        for evaluation in evaluations
        for metric in evaluation.state_metrics
    }
    return SearchBlockResult(
        optimization_block=block,
        refinement=refinement,
        selected_candidates=selected,
        selected_states=tuple(
            state_index[(state.state_identity, state.candidate_identity)]
            for state in refinement.selection.states
        ),
    )


def _candidate_evaluation(
    candidate: InterventionCandidate,
    summary: CandidateWorkerResultSummary,
) -> CandidateEvaluation:
    """Build search values for one checked worker result.

    Only metric rows belonging to the candidate's optimization block are used.
    Each row retains its background, allocated carrying capacity label, regional
    compliance counts, ASR exceedance, and binding coordinates. Intervention
    distance is recalculated from the saved five changes so selection compares the
    same distance used by the root's metric calculation.
    """
    entries = tuple(
        item
        for item in summary.member_metrics
        if item.optimization_block == candidate.optimization_block
    )
    metrics: list[CandidateStateMetrics] = []
    for entry in entries:
        metrics.append(
            CandidateStateMetrics(
                state_identity=entry.optimization_item_identity,
                candidate_identity=candidate.identity,
                physical_candidate_identity=entry.physical_candidate_identity,
                optimization_block=candidate.optimization_block,
                background_scenario=entry.background_scenario,
                allocated_cc_case=entry.allocated_cc_case,
                is_zero_intervention_parent=entry.is_zero_intervention_parent,
                intervention_distance=entry.intervention_distance,
                all_region_joint_compliant=entry.all_region_joint_compliant,
                fully_compliant_regions=entry.fully_compliant_regions,
                fully_compliant_impact_targets=(entry.fully_compliant_impact_targets),
                compliant_region_count=entry.compliant_region_count,
                fully_compliant_impact_target_count=(entry.fully_compliant_impact_target_count),
                compliant_region_impact_group_count=(entry.compliant_region_impact_group_count),
                compliant_coordinate_count=entry.compliant_coordinate_count,
                region_count=entry.region_count,
                impact_target_count=entry.impact_target_count,
                region_impact_group_count=entry.region_impact_group_count,
                coordinate_count=entry.coordinate_count,
                worst_asr=entry.z_worst,
                binding_slack=entry.binding_slack,
                maximum_positive_exceedance=entry.maximum_positive_exceedance,
                balanced_mean_positive_exceedance=(entry.balanced_mean_positive_exceedance),
                binding_coordinates=tuple(
                    (
                        item.region,
                        item.constraint_type,
                        item.optimization_lcia_method,
                        item.impact_category_id,
                        item.year,
                        item.asr,
                    )
                    for item in entry.binding_constraints
                ),
            )
        )
    distance = intervention_distance(
        physical_candidate_identity=summary.physical.physical_candidate_identity,
        intervention_authority_identity=summary.intervention.identity,
        distance_inputs=summary.intervention.distance_inputs,
        optimization_block=candidate.optimization_block,
    ).total_distance
    return CandidateEvaluation(
        candidate=candidate,
        physical_candidate_identity=summary.physical.physical_candidate_identity,
        state_metrics=tuple(metrics),
        intervention_distance=distance,
        diagnostic=summary.physical.diagnostic,
        repair_unresolved=summary.repair.status == "repair_unresolved",
    )


def _execution_blocks(values: Sequence[OptimizationBlock]) -> tuple[OptimizationBlock, ...]:
    """Include both single-group searches when the combined block is requested."""
    requested = tuple(values)
    if "integrated_frontier" not in requested:
        return requested
    return OPTIMIZATION_BLOCK_ORDER
