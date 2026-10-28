"""Evaluate optimization candidates in persistent worker processes.

Each worker loads the completed MFA result and response tables once and derives
the baseline scores for the selected background scenarios. For a new MFA calculation, the worker
tests the requested linear trajectory first. It attempts bounded
demand repair only after MFA rejection when positive demand reduction
allows it. Compatible completed repair outcomes can be reused. The chosen
admitted trajectory is scored once across the selected backgrounds. The root
process compares those saved scores with each selected set of allocated carrying
capacities.

The module creates tasks, reads saved candidate results, and connects the
calculation to the spawned process pool.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import cast

import pandas as pd

from pytoce.run_lca.calculation.lcia import LCIAMethodSelection, canonical_lcia_method
from pytoce.run_mfa.model.mfa import MFAEvaluation, MFAEvaluationSession
from pytoce.run_mfa.scenario.parent_relative import (
    compile_parent_relative_scenario_assignments,
)
from pytoce.run_mfa.scenario.scenario_contract import OptimizationPhysicalInfeasibility
from pytoce.run_optimization.calculation.candidate_acc import (
    CandidateACCContext,
    _assert_acc_scoring_compatibility,
    calculate_candidate_acc_metrics,
    prepare_candidate_acc_context,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateInterventionAuthority,
    CandidateInterventionParent,
    CandidateParentAuthority,
    build_candidate_intervention_from_parent,
    build_candidate_intervention_parent,
)
from pytoce.run_optimization.calculation.candidate_evaluation import (
    AdmittedCandidateCalculation,
    CandidateEvaluationSession,
    InfeasibleCandidateCalculation,
    feedstock_requirements_identity,
)
from pytoce.run_optimization.calculation.candidate_metrics import (
    CandidateMetricSummary,
    candidate_metric_summary,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    SearchScoringContext,
    build_response_parent_score_projection,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    CandidateParentScoreProjection,
)
from pytoce.run_optimization.calculation.candidate_repair import (
    CandidateRepairFamilyContext,
    candidate_repair_family_identity,
    prepare_repair_family_context,
    repair_reference_from_result,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    STOCK_PROPOSAL_METHOD_IDENTITY,
    demand_path_departure,
    demand_repair_neighbors,
    propose_annual_demand,
)
from pytoce.run_optimization.calculation.interventions import (
    CIRCULARITY_METHOD_IDENTITY,
    DEMAND_LIFETIME_METHOD_IDENTITY,
    CircularityDecision,
    DemandLifetimeDecision,
    build_circularity_trajectories,
    build_demand_lifetime_trajectories,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    PaCandidateScores,
    build_pa_candidate_projection,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
    response_projection_selection_identity,
    select_response_bank_method_registry,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.run_optimization.calculation.worker_inputs import (
    _initializer_with_pa_parent,
    _prepare_parent_search_inputs,
    _prepare_search_inputs,
    _PreparedPhysicalInputs,
    _PreparedSearchInputs,
    candidate_worker_authority_identity,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    _WORKER_RESULT_SUMMARY_PREFIX,
    CANDIDATE_WORKER_METHOD_IDENTITY,
    CANDIDATE_WORKER_RESULT_METHOD_IDENTITY,
    CandidateInterventionSummary,
    CandidateRepairReference,
    CandidateWorkerMetricScope,
    CandidateWorkerResult,
    CandidateWorkerResultSummary,
    CandidateWorkerScoreSummary,
    CandidateWorkerTask,
    DemandRepairOutcome,
    RepairStatus,
    _candidate_worker_physical_summary,
    candidate_worker_result_identity,
    decode_candidate_repair_reference,
    decode_candidate_worker_result,
    decode_candidate_worker_task,
    encode_candidate_repair_reference,
    encode_candidate_worker_result,
    encode_candidate_worker_task,
)
from pytoce.run_optimization.input.allocated_cc import select_allocated_cc_result
from pytoce.run_optimization.input.allocated_cc_contract import (
    AllocatedCCResult,
    AllocatedCCSelection,
)
from pytoce.run_optimization.runtime.candidate_results import (
    CandidateRepairProvenance,
    CandidateResultStore,
    CandidateTask,
    CandidateTaskDispatch,
    CandidateTaskRecord,
)
from pytoce.shared.runtime.filesystem import payload_sha256

_WORKER_PLANNER_PREFIX = "pytoce_optimization_candidate_worker_planner_v2::"
_WORKER_REQUEST_BRIDGE_PREFIX = "pytoce_optimization_candidate_worker_request_bridge_v8::"
_WORKER_RESULT_SET_PREFIX = "pytoce_optimization_candidate_worker_result_set_v1::"


def _demand_stock_multipliers(decision: DemandLifetimeDecision) -> tuple[float, ...]:
    """Return exact annual stock multipliers within one fixed repair context.

    The parent, lifetime and circularity must remain fixed. Distinct demand
    fractions can round to the same numerical stock assignments; retain all
    years, at their admitted precision, when recognizing rejected aliases.
    """
    return tuple(1.0 - float(value) for value in decision.demand_fraction)


def _build_candidate_intervention(
    parent: CandidateInterventionParent,
    *,
    demand_lifetime: DemandLifetimeDecision,
    circularity: CircularityDecision,
) -> CandidateInterventionAuthority:
    """Build one intervention through the canonical trajectory owners."""

    demand = build_demand_lifetime_trajectories(
        parent_stock_targets=parent.demand_lifetime_parent.stock_targets,
        parent_lifetimes=parent.demand_lifetime_parent.lifetimes,
        parent_stock_target_identity=parent.demand_lifetime_parent.stock_target_identity,
        parent_lifetime_identity=parent.demand_lifetime_parent.lifetime_identity,
        decision=demand_lifetime,
    )
    circularity_trajectories = build_circularity_trajectories(
        parent=parent.circularity_parent.table,
        parent_identity=parent.circularity_parent.identity,
        decision=circularity,
    )
    return build_candidate_intervention_from_parent(
        parent,
        demand_lifetime=demand,
        circularity=circularity_trajectories,
    )


@dataclass(frozen=True, slots=True)
class CandidateWorkerPlanner:
    """Build candidate interventions from the completed MFA result."""

    identity: str
    worker_authority_identity: str
    allocated_cc_cases: tuple[str, ...]
    intervention_parent: CandidateInterventionParent
    repair_family_context: CandidateRepairFamilyContext

    def plan_intervention(
        self,
        *,
        demand_lifetime: DemandLifetimeDecision,
        circularity: CircularityDecision,
    ) -> CandidateInterventionAuthority:
        """Build one intervention for the requested demand and circularity changes."""

        return _build_candidate_intervention(
            self.intervention_parent,
            demand_lifetime=demand_lifetime,
            circularity=circularity,
        )


@dataclass(frozen=True, slots=True)
class CandidateWorkerRequestContext:
    """Inputs used by candidate search and final reporting."""

    planner: CandidateWorkerPlanner
    allocated_cc_context: CandidateACCContext
    search_scoring: SearchScoringContext
    result_lcia_selection_identity: str
    result_parent: CandidateParentScoreProjection
    result_projection: ResponseBankProjection
    result_pa_projection: PaCandidateProjection
    result_pa_parent: PaCandidateScores
    worker_scientific_authority_identity: str
    worker_initializer_payload: bytes

    @property
    def physical_parent(self) -> CandidateParentAuthority:
        """Return the completed MFA inputs used by reporting."""

        return self.search_scoring.physical_parent


class CandidateWorkerRequestBridge:
    """Retain compact summaries for search and complete candidate results on disk.

    Returned and reused payloads are decoded transiently to derive summaries.
    A restored family representative may also be loaded to establish repair reuse.
    Selected reporting reloads its selected full results. The bridge keeps compact summaries between
    evaluations.
    """

    def __init__(
        self,
        planner: CandidateWorkerPlanner,
        metric_scopes: Sequence[CandidateWorkerMetricScope],
        *,
        result_store: CandidateResultStore,
        worker_scientific_authority_identity: str,
        allocated_cc_context: CandidateACCContext,
        search_scoring: SearchScoringContext,
    ) -> None:
        """Connect task planning and result decoding to the candidate result cache."""
        scopes = tuple(metric_scopes)
        self._planner = planner
        self._metric_scopes = scopes
        self._result_store = result_store
        self._tasks: dict[str, CandidateTaskRecord] = {}
        self._repair_representatives: dict[str, CandidateTaskRecord] = {}
        self._repair_references: dict[str, CandidateRepairReference] = {}
        self._restored_candidates: set[str] = set()
        self._next_sequence = 0
        self._worker_scientific_authority_identity = worker_scientific_authority_identity
        self._allocated_cc_context = allocated_cc_context
        self._scoring = search_scoring
        self._identity = _WORKER_REQUEST_BRIDGE_PREFIX + payload_sha256(
            {
                "method_identity": "pytoce_optimization_candidate_worker_request_bridge_v8",
                "planner_identity": planner.identity,
                "worker_authority_identity": planner.worker_authority_identity,
                "allocated_cc_cases": list(planner.allocated_cc_cases),
                "metric_scopes": [
                    {
                        "optimization_block": scope.optimization_block,
                        "metric_registry_identity": scope.metric_registry_identity,
                    }
                    for scope in scopes
                ],
            }
        )

    @property
    def identity(self) -> str:
        """Return the exact request bridge identity."""

        return self._identity

    @property
    def metric_scopes(self) -> tuple[CandidateWorkerMetricScope, ...]:
        """Return optimization metric scopes in their recorded request order."""

        return self._metric_scopes

    @property
    def task_records(self) -> tuple[CandidateTaskRecord, ...]:
        """Retain task identities and their execution order for a completed stage."""

        return tuple(sorted(self._tasks.values(), key=lambda item: item.sequence))

    def restore_task_records(self, records: Sequence[CandidateTaskRecord]) -> None:
        """Restore a fresh bridge from an accepted compatible completed stage.

        Records retain the original task sequence from the completed-stage records. The
        completed-stage
        owner supplies nonempty ordered records after matching scientific
        authority; every completed search evaluated its parent first.
        """

        self._tasks = {record.candidate_identity: record for record in records}
        self._restored_candidates = set(self._tasks)
        for record in records:
            if record.repair_family_identity is not None:
                self._repair_representatives.setdefault(record.repair_family_identity, record)
        self._next_sequence = records[-1].sequence + 1

    def build_task(
        self,
        intervention: CandidateInterventionAuthority,
        metric_scopes: Sequence[CandidateWorkerMetricScope],
    ) -> CandidateTask:
        """Create one deterministic worker task for an intervention."""

        selected_scopes = tuple(metric_scopes)
        candidate = intervention.identity
        payload = encode_candidate_worker_task(
            CandidateWorkerTask(intervention=intervention, metric_scopes=selected_scopes)
        )
        task = CandidateTask(
            sequence=self._next_sequence,
            candidate_identity=candidate,
            payload=payload,
            repair_family_identity=candidate_repair_family_identity(
                intervention,
                self._planner.repair_family_context,
                worker_authority_identity=self._planner.worker_authority_identity,
            ),
        )
        record = task.record
        self._tasks[candidate] = record
        if record.repair_family_identity is not None:
            self._repair_representatives.setdefault(record.repair_family_identity, record)
        self._next_sequence += 1
        return task

    def read_saved_result(self, payload: bytes) -> CandidateWorkerResultSummary:
        """Calculate compact request metrics from a saved candidate result."""
        result = decode_candidate_worker_result(payload)
        record = self._tasks.get(result.requested_intervention_identity)
        if record is not None and record.repair_family_identity is not None:
            self._repair_references.setdefault(
                record.repair_family_identity,
                repair_reference_from_result(result, record.repair_family_identity),
            )
        return candidate_worker_result_summary(
            result, allocated_cc_context=self._allocated_cc_context, search_scoring=self._scoring
        )

    def prepare_dispatch(self, task: CandidateTask) -> CandidateTaskDispatch | None:
        """Dispatch one member or hold it until its canonical repair basis is saved.

        Only requested results can establish a reference. A restored stage may
        supply its first family result lazily; the reference uses that requested result.
        """
        family = task.repair_family_identity
        if family is None:
            return CandidateTaskDispatch(task)
        representative = self._repair_representatives[family]
        if family not in self._repair_references:
            if representative.candidate_identity in self._restored_candidates:
                result = self.load_candidate(representative.candidate_identity)
                self._repair_references[family] = repair_reference_from_result(result, family)
            elif representative.candidate_identity == task.candidate_identity:
                return CandidateTaskDispatch(task)
            else:
                return None
        reference = self._repair_references[family]
        return CandidateTaskDispatch(
            task,
            encode_candidate_repair_reference(reference),
            CandidateRepairProvenance(
                reference.family_identity,
                reference.requested_intervention_identity,
                reference.worker_result_identity,
            ),
        )

    def load_candidate(self, candidate_identity: str) -> CandidateWorkerResult:
        """Load one candidate result."""

        candidate = candidate_identity
        task = self._tasks[candidate]
        saved = cast(bytes, self._result_store.read_record(task))
        return decode_candidate_worker_result(saved)

    def finalize_result_set(
        self,
        summaries: Mapping[str, CandidateWorkerResultSummary],
    ) -> str:
        """Return the identity of the complete worker result set."""

        records = tuple(
            (
                task.candidate_identity,
                summaries[task.candidate_identity].worker_result_identity,
            )
            for task in self.task_records
        )
        payload = {
            "method_identity": "pytoce_optimization_candidate_worker_result_set_v1",
            "bridge_identity": self.identity,
            "worker_authority_identity": self._planner.worker_authority_identity,
            "worker_scientific_authority_identity": self._worker_scientific_authority_identity,
            "worker_method_identity": CANDIDATE_WORKER_METHOD_IDENTITY,
            "worker_result_method_identity": CANDIDATE_WORKER_RESULT_METHOD_IDENTITY,
            "metric_scope_records": [
                [scope.optimization_block, scope.metric_registry_identity]
                for scope in self._metric_scopes
            ],
            "result_records": [list(item) for item in records],
        }
        return _WORKER_RESULT_SET_PREFIX + payload_sha256(payload)


def candidate_worker_result_summary(
    result: CandidateWorkerResult,
    *,
    allocated_cc_context: CandidateACCContext,
    search_scoring: SearchScoringContext,
) -> CandidateWorkerResultSummary:
    """Derive search metrics from saved scores and the request's allocated carrying capacities."""

    intervention = result.intervention
    physical = result.physical
    intervention_summary = CandidateInterventionSummary(
        identity=intervention.identity,
        physical_parent_identity=intervention.physical_parent_identity,
        parent_foreground_scenario_identity=(intervention.parent_foreground_scenario_identity),
        demand_lifetime_parent_identity=intervention.demand_lifetime_parent_identity,
        circularity_parent_identity=intervention.circularity_parent_identity,
        demand_lifetime_trajectory_identity=intervention.demand_lifetime_trajectory_identity,
        circularity_trajectory_identity=intervention.circularity_trajectory_identity,
        distance_inputs=intervention.distance_inputs,
        uses_exact_parent=intervention.uses_exact_parent,
    )
    physical_summary = _candidate_worker_physical_summary(physical)
    score_summary = None
    member_entries: tuple[CandidateMetricSummary, ...] = ()
    if result.scores is not None:
        scores = result.scores
        score_summary = CandidateWorkerScoreSummary(
            physical_candidate_identity=scores.physical_candidate_identity,
            environmental_identity=scores.environmental_identity,
            backgrounds=scores.backgrounds,
            optimization_method_registry_identity=(scores.optimization_method_registry_identity),
            cumulative_climate_method_registry_identity=(
                scores.cumulative_climate_method_registry_identity
            ),
            annual_scores_identity=scores.annual_scores_identity,
            annual_climate_scores_identity=scores.annual_climate_scores_identity,
            score_identity=scores.score_identity,
        )
        member_entries = tuple(
            candidate_metric_summary(item.metrics)
            for scope in result.metric_scopes
            for item in calculate_candidate_acc_metrics(
                scores,
                allocated_cc_context,
                environmental_parent_members=search_scoring.optimization_parent.member_parents,
                scoring_context_identity=search_scoring.identity,
                intervention_authority_identity=intervention.identity,
                distance_inputs=intervention.distance_inputs,
                is_zero_intervention_parent=intervention.uses_exact_parent,
                optimization_block=scope.optimization_block,
                metric_registry_identity=scope.metric_registry_identity,
            )
        )
    identity_payload = {
        "method_identity": "pytoce_optimization_candidate_worker_summary_v4",
        "worker_result_identity": result.identity,
        "worker_authority_identity": result.worker_authority_identity,
        "worker_scientific_authority_identity": result.worker_scientific_authority_identity,
        "worker_method_identity": CANDIDATE_WORKER_METHOD_IDENTITY,
        "worker_result_method_identity": CANDIDATE_WORKER_RESULT_METHOD_IDENTITY,
        "requested_intervention_identity": result.requested_intervention_identity,
        "intervention_authority_identity": intervention.identity,
        "physical_request_identity": physical.physical_request_identity,
        "physical_candidate_identity": physical.physical_candidate_identity,
        "score_identity": (None if score_summary is None else score_summary.score_identity),
        "feedstock_requirements_identity": result.feedstock_requirements_identity,
        "member_metric_entry_identities": [item.identity for item in member_entries],
    }
    return CandidateWorkerResultSummary(
        identity=_WORKER_RESULT_SUMMARY_PREFIX + payload_sha256(identity_payload),
        worker_result_identity=result.identity,
        worker_authority_identity=result.worker_authority_identity,
        worker_scientific_authority_identity=result.worker_scientific_authority_identity,
        worker_method_identity=CANDIDATE_WORKER_METHOD_IDENTITY,
        worker_result_method_identity=CANDIDATE_WORKER_RESULT_METHOD_IDENTITY,
        requested_intervention_identity=result.requested_intervention_identity,
        repair=result.repair,
        intervention=intervention_summary,
        physical=physical_summary,
        scores=score_summary,
        feedstock_requirements_identity=result.feedstock_requirements_identity,
        member_metrics=member_entries,
    )


def _worker_scientific_authority_identity(
    inputs: _PreparedSearchInputs,
    *,
    candidate_mfa_session_identity: str,
) -> str:
    """Identify the MFA and response tables used by one worker."""

    physical = inputs.physical
    payload = physical.payload
    authority_payload = {
        "method_identity": CANDIDATE_WORKER_METHOD_IDENTITY,
        "stock_proposal_method": STOCK_PROPOSAL_METHOD_IDENTITY,
        "foreground_scenario_identity": physical.scenario.foreground_scenario_identity,
        "mfa_result_identity": payload["mfa_result_identity"],
        "mfa_result_manifest_sha256": payload["mfa_result_manifest_sha256"],
        "candidate_mfa_session_identity": candidate_mfa_session_identity,
        "physical_parent_identity": physical.physical_parent.identity,
        "optimization_parent_scores_identity": inputs.optimization_parent.identity,
        "climate_parent_scores_identity": inputs.climate_parent.identity,
        "optimization_projection_identity": inputs.optimization_projection.projection_identity,
        "climate_projection_identity": inputs.climate_projection.projection_identity,
        "prepared_receiving_identity": inputs.pa_projection.prepared_receiving_identity,
    }
    return CANDIDATE_WORKER_METHOD_IDENTITY + "::" + payload_sha256(authority_payload)


def _planner_from_physical_inputs(
    inputs: _PreparedPhysicalInputs,
    acc_selection: AllocatedCCSelection,
    *,
    optimization_projection_identity: str,
) -> CandidateWorkerPlanner:
    """Bind the candidate planner to baseline inputs and trajectory methods."""
    intervention_parent = build_candidate_intervention_parent(
        inputs.scenario,
        physical_parent=inputs.physical_parent,
    )
    worker_authority = candidate_worker_authority_identity(
        inputs.payload, optimization_projection_identity=optimization_projection_identity
    )
    identity_payload = {
        "method_identity": "pytoce_optimization_candidate_worker_planner_v2",
        "worker_authority_identity": worker_authority,
        "intervention_parent_identity": intervention_parent.identity,
        "demand_lifetime_method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
        "circularity_method_identity": CIRCULARITY_METHOD_IDENTITY,
        "allocated_cc_cases": list(acc_selection.allocated_cc_cases),
        "allocated_cc_selection_identity": acc_selection.allocated_cc_selection_identity,
    }
    return CandidateWorkerPlanner(
        identity=_WORKER_PLANNER_PREFIX + payload_sha256(identity_payload),
        worker_authority_identity=worker_authority,
        allocated_cc_cases=acc_selection.allocated_cc_cases,
        intervention_parent=intervention_parent,
        repair_family_context=prepare_repair_family_context(inputs.scenario),
    )


def prepare_candidate_worker_request_context(
    initializer_payload: bytes,
    allocated_cc: AllocatedCCResult,
    *,
    parent_mfa: MFAEvaluation,
    result_lcia_method: LCIAMethodSelection,
) -> CandidateWorkerRequestContext:
    """Prepare the shared search inputs and selected-result LCIA inputs."""

    inputs = _prepare_parent_search_inputs(initializer_payload, parent_mfa)
    methods = canonical_lcia_method(result_lcia_method)
    result_registry = select_response_bank_method_registry(
        inputs.method_registry, result_lcia_method
    )
    result_projection = project_optimization_response_bank(
        inputs.response_bank,
        result_registry,
        role="result",
    )
    if (
        result_projection.method_registry_identity
        == inputs.optimization_projection.method_registry_identity
    ):
        result_pa_projection, result_pa_parent = inputs.pa_projection, inputs.pa_parent
    else:
        result_pa_projection = build_pa_candidate_projection(
            result_projection,
            inputs.pa_projection.providers_by_background,
            inputs.pa_projection.dump_shares,
            prepared_receiving_identity=inputs.pa_projection.prepared_receiving_identity,
        )
        parent = inputs.physical.physical_parent
        result_pa_parent = calculate_pa_candidate_projection(
            result_pa_projection,
            spinning_events=parent.pa_spinning_events,
            recycling_events=parent.pa_recycling_events,
            disposal_events=parent.pa_disposal_events,
        )
    result_parent = build_response_parent_score_projection(
        inputs.physical.physical_parent,
        result_projection,
        pa_parent=result_pa_parent,
        pa_method_registry=result_registry,
    )
    result_selection_identity = response_projection_selection_identity(
        methods,
        result_projection,
    )
    physical = inputs.physical
    allocated_cc_selection = select_allocated_cc_result(
        allocated_cc,
        tuple(cast(list[str], physical.payload["allocated_cc_case"])),
    )
    acc_context = prepare_candidate_acc_context(allocated_cc_selection)
    scoring = prepare_search_scoring(
        physical_parent=physical.physical_parent,
        optimization_parent=inputs.optimization_parent,
        climate_parent=inputs.climate_parent,
        optimization_projection=inputs.optimization_projection,
        climate_projection=inputs.climate_projection,
        pa_projection=inputs.pa_projection,
        pa_parent=inputs.pa_parent,
    )
    _assert_acc_scoring_compatibility(acc_context, scoring)
    return CandidateWorkerRequestContext(
        planner=_planner_from_physical_inputs(
            physical,
            allocated_cc_selection,
            optimization_projection_identity=inputs.optimization_projection.projection_identity,
        ),
        allocated_cc_context=acc_context,
        search_scoring=scoring,
        result_lcia_selection_identity=result_selection_identity,
        result_parent=result_parent,
        result_projection=result_projection,
        result_pa_projection=result_pa_projection,
        result_pa_parent=result_pa_parent,
        worker_initializer_payload=_initializer_with_pa_parent(
            initializer_payload, inputs.pa_parent
        ),
        worker_scientific_authority_identity=_worker_scientific_authority_identity(
            inputs,
            candidate_mfa_session_identity=cast(
                str,
                physical.payload["candidate_mfa_session_identity"],
            ),
        ),
    )


class CandidateEvaluationWorker:
    """One persistent MFA and response calculation in a worker process."""

    def __init__(self, initializer_payload: bytes) -> None:
        """Decode one worker request and prepare its MFA and response calculations."""
        inputs = _prepare_search_inputs(initializer_payload)
        physical = inputs.physical
        mfa_session = MFAEvaluationSession(
            physical.mfa_request,
            physical.parameter_catalogue,
            recipe=physical.recipe,
            composition=physical.composition,
        )
        session = CandidateEvaluationSession(
            mfa_session=mfa_session,
            material_recipe=physical.recipe,
            parent_scenario=physical.scenario,
            parent_mfa=physical.parent_mfa,
            parent_authority=physical.physical_parent,
            optimization_parent_scores=inputs.optimization_parent,
            cumulative_climate_parent_scores=inputs.climate_parent,
            provider_routes=physical.routes,
            optimization_activations=physical.optimization_activations,
            optimization_projection=inputs.optimization_projection,
            cumulative_climate_projection=inputs.climate_projection,
            pa_projection=inputs.pa_projection,
            pa_parent=inputs.pa_parent,
        )
        self._worker_scientific_authority_identity = _worker_scientific_authority_identity(
            inputs,
            candidate_mfa_session_identity=mfa_session.authority_identity,
        )
        self._worker_authority_identity = candidate_worker_authority_identity(
            physical.payload,
            optimization_projection_identity=inputs.optimization_projection.projection_identity,
        )
        self._session = session
        self._mfa_session = mfa_session
        self._proposal_scenario = physical.scenario
        self._proposal_parent = physical.physical_parent.demand_lifetime_parent
        self._intervention_parent = build_candidate_intervention_parent(
            physical.scenario, physical_parent=physical.physical_parent
        )
        self._stock_proposals: dict[tuple[int, int, int], DemandLifetimeDecision | None] = {}
        self._repair_family_context = prepare_repair_family_context(physical.scenario)

    def _stock_proposal(
        self, decision: DemandLifetimeDecision
    ) -> tuple[DemandLifetimeDecision | None, bool]:
        """Reuse one conditional demand solve for fixed linear lifetime and target year."""
        key = (
            decision.demand_reduction_pct,
            decision.lifetime_extension_pct,
            decision.target_year,
        )
        if key in self._stock_proposals:
            return self._stock_proposals[key], False
        parent = self._proposal_parent
        lifetime = build_demand_lifetime_trajectories(
            parent_stock_targets=parent.stock_targets,
            parent_lifetimes=parent.lifetimes,
            parent_stock_target_identity=parent.stock_target_identity,
            parent_lifetime_identity=parent.lifetime_identity,
            decision=DemandLifetimeDecision(
                0, decision.lifetime_extension_pct, decision.target_year
            ),
        )
        scenario = compile_parent_relative_scenario_assignments(
            self._proposal_scenario,
            self._proposal_scenario.foreground_scenario,
            lifetime.assignments,
        )
        system = self._mfa_session.fixed_lifetime_stock_system(
            compiled=scenario.compiled_parameters
        )
        coordinates = pd.MultiIndex.from_product(
            (system.years, system.regions, system.goods), names=("year", "region", "good")
        )
        stock = (
            parent.stock_targets.set_index(["year", "region", "good"])["stock_projection"]
            .reindex(coordinates)
            .to_numpy(dtype=float)
            .reshape(len(system.years), len(system.regions), len(system.goods))
        )
        proposal = propose_annual_demand(
            system,
            parent_stock=stock,
            demand_reduction_pct=decision.demand_reduction_pct,
            lifetime_extension_pct=decision.lifetime_extension_pct,
            lifetime_fraction=decision.lifetime_fraction,
            target_year=decision.target_year,
        )
        self._stock_proposals[key] = proposal
        return proposal, True

    def _repair(
        self, requested: CandidateInterventionAuthority
    ) -> tuple[
        CandidateInterventionAuthority,
        AdmittedCandidateCalculation | InfeasibleCandidateCalculation,
        DemandRepairOutcome,
    ]:
        """Choose one MFA trajectory using the demand-repair rules."""
        started = perf_counter()
        decision = requested.demand_lifetime_evaluation_projection.decision
        circularity = requested.circularity_evaluation_projection.decision
        initial = self._session.admit_physical(requested)
        physical_count = int(not requested.uses_exact_parent)
        linear_diagnostic = (
            initial.physical.diagnostic
            if isinstance(initial, InfeasibleCandidateCalculation)
            else None
        )
        lp_diagnostic = None
        lp_count = 0
        pool_count = 0
        pool_evaluations = 0

        def outcome(
            status: RepairStatus, chosen: DemandLifetimeDecision | None
        ) -> DemandRepairOutcome:
            """Bind actual bounded-work counts and direct-owner failure evidence."""
            return DemandRepairOutcome(
                status=status,
                physical_repair_elapsed_seconds=perf_counter() - started,
                environmental_evaluation_elapsed_seconds=0.0,
                mfa_evaluation_count=physical_count,
                stock_lp_count=lp_count,
                pool_candidate_count=pool_count,
                pool_evaluation_count=pool_evaluations,
                demand_departure_pp=None if chosen is None else demand_path_departure(chosen),
                linear_diagnostic=linear_diagnostic,
                stock_projection_diagnostic=lp_diagnostic,
            )

        if isinstance(initial, AdmittedCandidateCalculation):
            return requested, initial, outcome("linear_feasible", decision)
        if decision.demand_reduction_pct == 0:
            return requested, initial, outcome("unique_fixed_path_infeasible", None)
        proposal, solved = self._stock_proposal(decision)
        lp_count = int(solved)
        if proposal is None:
            return requested, initial, outcome("stock_infeasible_fixed_controls", None)
        intervention = _build_candidate_intervention(
            self._intervention_parent, demand_lifetime=proposal, circularity=circularity
        )
        projected: AdmittedCandidateCalculation | InfeasibleCandidateCalculation
        if intervention.identity == requested.identity:
            projected = initial
        else:
            projected = self._session.admit_physical(intervention)
            physical_count += 1
        if isinstance(projected, AdmittedCandidateCalculation):
            return intervention, projected, outcome("stock_projection_feasible", proposal)
        lp_diagnostic = projected.physical.diagnostic
        failure_year = cast(
            int, dict(cast(OptimizationPhysicalInfeasibility, lp_diagnostic).coordinates)["year"]
        )
        pool = demand_repair_neighbors(proposal, failure_year=failure_year)
        pool_count = len(pool)
        # Both seeds have completed MFA rejection. Only demand varies in
        # this immutable pool, so identical full-horizon multipliers reproduce
        # the rejected assignments. Keep this knowledge local to this request.
        rejected_controls = {
            _demand_stock_multipliers(decision),
            _demand_stock_multipliers(proposal),
        }
        for alternative in pool:
            controls = _demand_stock_multipliers(alternative)
            if controls in rejected_controls:
                continue
            candidate = _build_candidate_intervention(
                self._intervention_parent, demand_lifetime=alternative, circularity=circularity
            )
            admitted = self._session.admit_physical(candidate)
            physical_count += 1
            pool_evaluations += 1
            if isinstance(admitted, AdmittedCandidateCalculation):
                return candidate, admitted, outcome("local_pool_feasible", alternative)
            rejected_controls.add(controls)
        return intervention, projected, outcome("repair_unresolved", None)

    def _reuse_repair(
        self, requested: CandidateInterventionAuthority, reference: CandidateRepairReference
    ) -> tuple[
        CandidateInterventionAuthority,
        AdmittedCandidateCalculation | InfeasibleCandidateCalculation,
        DemandRepairOutcome,
    ]:
        """Apply a completed finite repair while retaining this member's full scoring path."""
        started = perf_counter()
        family = candidate_repair_family_identity(
            requested,
            self._repair_family_context,
            worker_authority_identity=self._worker_authority_identity,
        )
        if family != reference.family_identity:
            raise ValueError("A candidate repair reference belongs to different controls")
        intervention = _build_candidate_intervention(
            self._intervention_parent,
            demand_lifetime=reference.demand_lifetime,
            circularity=requested.circularity_evaluation_projection.decision,
        )
        feasible = reference.status in {
            "linear_feasible",
            "stock_projection_feasible",
            "local_pool_feasible",
        }
        if feasible:
            admitted = self._session.calculate_physical(intervention)
        else:
            admitted = self._session.reuse_physical_rejection(
                intervention, cast(OptimizationPhysicalInfeasibility, reference.physical_diagnostic)
            )
        return (
            intervention,
            admitted,
            DemandRepairOutcome(
                status=reference.status,
                physical_repair_elapsed_seconds=perf_counter() - started,
                environmental_evaluation_elapsed_seconds=0.0,
                mfa_evaluation_count=int(feasible),
                stock_lp_count=0,
                pool_candidate_count=reference.pool_candidate_count,
                pool_evaluation_count=0,
                demand_departure_pp=reference.demand_departure_pp,
                linear_diagnostic=reference.linear_diagnostic,
                stock_projection_diagnostic=reference.stock_projection_diagnostic,
            ),
        )

    def __call__(self, payload: bytes, repair_reference: bytes | None = None) -> bytes:
        """Admit or repair one endpoint, then score only its chosen MFA state."""
        task = decode_candidate_worker_task(payload)
        intervention, admitted, repair = (
            self._repair(task.intervention)
            if repair_reference is None
            else self._reuse_repair(
                task.intervention, decode_candidate_repair_reference(repair_reference)
            )
        )
        scores = None
        pa_scores = None
        requirements = None
        requirements_identity = None
        if isinstance(admitted, AdmittedCandidateCalculation):
            started = perf_counter()
            calculation = self._session.complete_physical(admitted)
            physical = calculation.physical
            scores, pa_scores = self._session.score_optimization(calculation)
            requirements = calculation.feedstock_requirements
            requirements_identity = feedstock_requirements_identity(requirements)
            repair = replace(
                repair, environmental_evaluation_elapsed_seconds=perf_counter() - started
            )
        else:
            physical = admitted.physical
        result = CandidateWorkerResult(
            identity="",
            worker_authority_identity=self._worker_authority_identity,
            worker_scientific_authority_identity=self._worker_scientific_authority_identity,
            requested_intervention_identity=task.intervention.identity,
            repair=repair,
            intervention=intervention,
            physical=physical,
            scores=scores,
            pa_scores=pa_scores,
            metric_scopes=task.metric_scopes,
            feedstock_requirements=requirements,
            feedstock_requirements_identity=requirements_identity,
        )
        result = replace(result, identity=candidate_worker_result_identity(result))
        return encode_candidate_worker_result(result)


def initialize_candidate_worker(payload: bytes) -> CandidateEvaluationWorker:
    """Importable spawned-process initializer used by the scheduler."""

    return CandidateEvaluationWorker(payload)
