"""Typed worker messages, compact summaries, serialization, and parsing."""

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import ipc

from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_mfa.scenario.scenario_contract import OptimizationPhysicalInfeasibility
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateInterventionAuthority,
    PhysicalEvaluation,
    PhysicalStatus,
)
from pytoce.run_optimization.calculation.candidate_metrics import CandidateMetricSummary
from pytoce.run_optimization.calculation.candidate_projection import (
    OptimizationScores,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import _array_identity
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    CircularityEvaluationProjection,
    DemandLifetimeDecision,
    DemandLifetimeEvaluationProjection,
    InterventionDistanceInputs,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import PaCandidateScores
from pytoce.run_optimization.calculation.regional_metrics import (
    AnnualCategory,
    BindingConstraint,
    ConstraintType,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

CANDIDATE_WORKER_METHOD_IDENTITY = "pytoce_optimization_candidate_worker_v29"
CANDIDATE_WORKER_TASK_SCHEMA = "pytoce_candidate_worker_task_schema_v1"
CANDIDATE_REPAIR_REFERENCE_SCHEMA = "pytoce_candidate_repair_reference_schema_v1"
CANDIDATE_REPAIR_PROVENANCE_SCHEMA = "pytoce_candidate_repair_provenance_schema_v1"
CANDIDATE_WORKER_RESULT_SCHEMA = "pytoce_single_stock_projection_worker_result_schema_v1"
CANDIDATE_WORKER_SUMMARY_SCHEMA = "pytoce_completed_stage_worker_summary_schema_v1"
CANDIDATE_WORKER_INITIALIZER_SCHEMA = (
    "pytoce_optimization_selected_backgrounds_initializer_schema_v1"
)
CANDIDATE_WORKER_RESULT_METHOD_IDENTITY = "pytoce_optimization_candidate_worker_result_v23"
_WORKER_RESULT_PREFIX = CANDIDATE_WORKER_RESULT_METHOD_IDENTITY + "::"
_WORKER_RESULT_SUMMARY_PREFIX = "pytoce_optimization_candidate_worker_summary_v4::"
_SELECTED_CANDIDATE_RESULT_PREFIX = "pytoce_optimization_selected_candidate_result_v5::"
_TASK_MAGIC = b"CEPTASK1"
_RESULT_MAGIC = b"CEPRESU1"
_RESULT_SECTION_COUNT = 7


@dataclass(frozen=True, slots=True)
class CandidateWorkerMetricScope:
    """Identify the metrics calculated for one optimization decision group."""

    optimization_block: OptimizationBlock
    metric_registry_identity: str


@dataclass(frozen=True, slots=True)
class CandidateWorkerTask:
    """One complete intervention and its metric scopes sent to a persistent worker."""

    intervention: CandidateInterventionAuthority
    metric_scopes: tuple[CandidateWorkerMetricScope, ...]


RepairStatus = Literal[
    "linear_feasible",
    "stock_projection_feasible",
    "local_pool_feasible",
    "stock_infeasible_fixed_controls",
    "unique_fixed_path_infeasible",
    "repair_unresolved",
]


@dataclass(frozen=True, slots=True)
class DemandRepairOutcome:
    """Record finite demand-repair outcomes separately from MFA infeasibility.

    MFA, stock LP, and pool-evaluation counts record actual calculation invocations,
    excluding reused conditional proposals and the unchanged parent. The pool
    candidate count records the generated pool size, including candidates skipped
    or not reached after a feasible path is found. This size is retained when the
    repair outcome is reused. Departure is the admitted path's mean absolute demand
    departure from its linear reference over 2020 through T-1,
    in percentage points; it is absent when no path is admitted. Diagnostics retain
    the rejected linear and stock-projection paths' direct equations. Solver
    infeasibility is conditional on the submitted fixed-control stock formulation;
    exhausting the finite pool records a failed demand repair.

    ``physical_repair_elapsed_seconds`` measures repair or repair reuse, including
    admission, stock solving and pool evaluation where required.
    ``environmental_evaluation_elapsed_seconds`` measures completion,
    scoring and feedstock-identity calculation. Both are worker elapsed durations
    excluding initialization and final result encoding. Durations from concurrent
    workers can overlap.
    """

    status: RepairStatus
    physical_repair_elapsed_seconds: float
    environmental_evaluation_elapsed_seconds: float
    mfa_evaluation_count: int
    stock_lp_count: int
    pool_candidate_count: int
    pool_evaluation_count: int
    demand_departure_pp: Fraction | None
    linear_diagnostic: OptimizationPhysicalInfeasibility | None
    stock_projection_diagnostic: OptimizationPhysicalInfeasibility | None


@dataclass(frozen=True, slots=True)
class CandidateRepairReference:
    """Completed family repair outcome sent separately from an unchanged logical task."""

    family_identity: str
    requested_intervention_identity: str
    worker_result_identity: str
    demand_lifetime: DemandLifetimeDecision
    status: RepairStatus
    pool_candidate_count: int
    demand_departure_pp: Fraction | None
    linear_diagnostic: OptimizationPhysicalInfeasibility | None
    stock_projection_diagnostic: OptimizationPhysicalInfeasibility | None
    physical_diagnostic: OptimizationPhysicalInfeasibility | None


def encode_candidate_repair_reference(reference: CandidateRepairReference) -> bytes:
    """Encode compact execution reuse."""
    return _canonical_json_bytes(
        {"schema_version": CANDIDATE_REPAIR_REFERENCE_SCHEMA, **asdict(reference)}
    )


def decode_candidate_repair_reference(payload: bytes) -> CandidateRepairReference:
    """Restore the current producer-owned reference and its exact rational demand path."""
    item = json.loads(payload)
    departure = item["demand_departure_pp"]
    return CandidateRepairReference(
        family_identity=item["family_identity"],
        requested_intervention_identity=item["requested_intervention_identity"],
        worker_result_identity=item["worker_result_identity"],
        demand_lifetime=_decode_demand_lifetime(item["demand_lifetime"]),
        status=item["status"],
        pool_candidate_count=item["pool_candidate_count"],
        demand_departure_pp=None if departure is None else Fraction(*departure),
        linear_diagnostic=_decode_diagnostic(item["linear_diagnostic"]),
        stock_projection_diagnostic=_decode_diagnostic(item["stock_projection_diagnostic"]),
        physical_diagnostic=_decode_diagnostic(item["physical_diagnostic"]),
    )


@dataclass(frozen=True, slots=True)
class CandidateWorkerResult:
    """Store one candidate calculation returned by a worker process.

    Attributes:
        identity: Identifier of scientific content, excluding elapsed times and
            counts of MFA evaluations, stock LP solves and local pool evaluations.
        worker_authority_identity: Identifier of the worker files and request inputs.
        worker_scientific_authority_identity: Identifier of scientific inputs and methods.
        requested_intervention_identity: Identifier of the intervention submitted before repair.
        repair: Demand-timing repair outcome, diagnostics and observed calculation work.
        intervention: Validated demand, lifetime, and circularity trajectories.
        physical: Candidate MFA status, MFA state identity and activations.
        scores: Annual EF and climate scores when the candidate is feasible in the MFA.
        pa_scores: EF search-method PA impacts, absolute totals and stage rows;
            zero arrays and an empty table if feasible with no PA, or ``None`` if infeasible.
            Scientific authority binds EF methods and sources; reporting uses its own methods.
        metric_scopes: Requested optimization blocks and their metric identities.
        feedstock_requirements: Virgin monomer or whole-polymer requirements by route.
        feedstock_requirements_identity: Content identifier of ``feedstock_requirements``.
    """

    identity: str
    worker_authority_identity: str
    worker_scientific_authority_identity: str
    requested_intervention_identity: str
    repair: DemandRepairOutcome
    intervention: CandidateInterventionAuthority
    physical: PhysicalEvaluation
    scores: OptimizationScores | None
    pa_scores: PaCandidateScores | None
    metric_scopes: tuple[CandidateWorkerMetricScope, ...]
    feedstock_requirements: pd.DataFrame | None
    feedstock_requirements_identity: str | None


@dataclass(frozen=True, slots=True)
class CandidateInterventionSummary:
    """Compact intervention values retained after a worker message is released."""

    identity: str
    physical_parent_identity: str
    parent_foreground_scenario_identity: str
    demand_lifetime_parent_identity: str
    circularity_parent_identity: str
    demand_lifetime_trajectory_identity: str
    circularity_trajectory_identity: str
    distance_inputs: InterventionDistanceInputs
    uses_exact_parent: bool


@dataclass(frozen=True, slots=True)
class CandidateWorkerPhysicalSummary:
    """Store MFA identities and status after candidate tables are released.

    Attributes:
        physical_request_identity: Identifier binding parent, intervention, MFA session,
            method, and provider-activation inputs.
        physical_candidate_identity: Identifier of a feasible MFA outcome or
            recorded rejection, bound to its request.
        physical_parent_identity: Identifier binding the parent MFA inputs and provider
            activations.
        intervention_authority_identity: Content identifier of the candidate trajectories.
        status: Whether the candidate satisfies all MFA feasibility checks.
        diagnostic: Complete direct-owner failure, or ``None`` when feasible.
        physical_state_identity: Identity of the candidate MFA state,
            or ``None`` when infeasible.
        activation_identity: Content identifier of the provider activations, or ``None`` when
            infeasible.
    """

    physical_request_identity: str
    physical_candidate_identity: str
    physical_parent_identity: str
    intervention_authority_identity: str
    status: PhysicalStatus
    diagnostic: OptimizationPhysicalInfeasibility | None
    physical_state_identity: str | None
    activation_identity: str | None


def _candidate_worker_physical_summary(
    physical: PhysicalEvaluation,
) -> CandidateWorkerPhysicalSummary:
    """Keep scalar values."""

    return CandidateWorkerPhysicalSummary(
        physical_request_identity=physical.physical_request_identity,
        physical_candidate_identity=physical.physical_candidate_identity,
        physical_parent_identity=physical.physical_parent_identity,
        intervention_authority_identity=physical.intervention_authority_identity,
        status=physical.status,
        diagnostic=physical.diagnostic,
        physical_state_identity=physical.physical_state_identity,
        activation_identity=physical.activation_identity,
    )


@dataclass(frozen=True, slots=True)
class CandidateWorkerScoreSummary:
    """Environmental score identities retained as a compact summary."""

    physical_candidate_identity: str
    environmental_identity: str
    backgrounds: tuple[str, ...]
    optimization_method_registry_identity: str
    cumulative_climate_method_registry_identity: str
    annual_scores_identity: str
    annual_climate_scores_identity: str
    score_identity: str


@dataclass(frozen=True, slots=True)
class CandidateWorkerResultSummary:
    """Store searchable values after the full candidate result is released.

    Attributes:
        identity: Identifier derived from method, worker, request, intervention,
            MFA, score, feedstock and member-metric identities. Execution-work metadata is retained
            as provenance.
        worker_result_identity: Content identifier of the saved candidate result.
        worker_authority_identity: Content identifier calculated from the files and request inputs.
        worker_scientific_authority_identity: Identifier of scientific inputs and methods.
        worker_method_identity: Candidate worker calculation method.
        worker_result_method_identity: Candidate result serialization method.
        requested_intervention_identity: Identifier of the intervention submitted before repair.
        repair: Demand-timing repair outcome, diagnostics and observed calculation work.
        intervention: Candidate percentages, trajectories, and baseline identities.
        physical: Candidate MFA status and MFA identities.
        scores: Environmental score identities for a feasible candidate.
        feedstock_requirements_identity: Content identifier of the saved feedstock rows.
        member_metrics: Compliance with allocated carrying capacities and ranking
            values for each requested state.
    """

    identity: str
    worker_result_identity: str
    worker_authority_identity: str
    worker_scientific_authority_identity: str
    worker_method_identity: str
    worker_result_method_identity: str
    requested_intervention_identity: str
    repair: DemandRepairOutcome
    intervention: CandidateInterventionSummary
    physical: CandidateWorkerPhysicalSummary
    scores: CandidateWorkerScoreSummary | None
    feedstock_requirements_identity: str | None
    member_metrics: tuple[CandidateMetricSummary, ...]


def encode_candidate_worker_summary(summary: CandidateWorkerResultSummary) -> dict[str, object]:
    """Embed an exact compact summary in its completed selected-product record.

    Annual arrays and candidate tables remain in the existing worker product.
    The exact rational repair departure is encoded as its numerator/denominator
    pair, as in the full worker protocol; the encoded values retain their stored precision.
    """

    departure = summary.repair.demand_departure_pp
    return {
        "schema_version": CANDIDATE_WORKER_SUMMARY_SCHEMA,
        **asdict(summary),
        "repair": {
            **asdict(summary.repair),
            "demand_departure_pp": (
                None if departure is None else (departure.numerator, departure.denominator)
            ),
        },
    }


def decode_candidate_worker_summary(record: Mapping[str, object]) -> CandidateWorkerResultSummary:
    """Restore trusted compact values.

    The selected-product owner admits a completed compatible stage. Identities,
    tuple order, metrics, work counters and diagnostics are restored exactly;
    the schema has only this current package-owned representation.
    """

    intervention = cast(dict[str, object], record["intervention"])
    distance = cast(dict[str, object], intervention["distance_inputs"])
    physical = cast(dict[str, object], record["physical"])
    scores = cast(dict[str, object] | None, record["scores"])
    return CandidateWorkerResultSummary(
        identity=cast(str, record["identity"]),
        worker_result_identity=cast(str, record["worker_result_identity"]),
        worker_authority_identity=cast(str, record["worker_authority_identity"]),
        worker_scientific_authority_identity=cast(
            str, record["worker_scientific_authority_identity"]
        ),
        worker_method_identity=cast(str, record["worker_method_identity"]),
        worker_result_method_identity=cast(str, record["worker_result_method_identity"]),
        requested_intervention_identity=cast(str, record["requested_intervention_identity"]),
        repair=_decode_repair(cast(dict[str, object], record["repair"])),
        intervention=CandidateInterventionSummary(
            identity=cast(str, intervention["identity"]),
            physical_parent_identity=cast(str, intervention["physical_parent_identity"]),
            parent_foreground_scenario_identity=cast(
                str, intervention["parent_foreground_scenario_identity"]
            ),
            demand_lifetime_parent_identity=cast(
                str, intervention["demand_lifetime_parent_identity"]
            ),
            circularity_parent_identity=cast(str, intervention["circularity_parent_identity"]),
            demand_lifetime_trajectory_identity=cast(
                str, intervention["demand_lifetime_trajectory_identity"]
            ),
            circularity_trajectory_identity=cast(
                str, intervention["circularity_trajectory_identity"]
            ),
            distance_inputs=InterventionDistanceInputs(
                identity=cast(str, distance["identity"]),
                physical_parent_identity=cast(str, distance["physical_parent_identity"]),
                demand_lifetime_parent_identity=cast(
                    str, distance["demand_lifetime_parent_identity"]
                ),
                demand_lifetime_trajectory_identity=cast(
                    str, distance["demand_lifetime_trajectory_identity"]
                ),
                circularity_trajectory_identity=cast(
                    str, distance["circularity_trajectory_identity"]
                ),
                target_year=cast(int, distance["target_year"]),
                demand_reduction_fraction=cast(float, distance["demand_reduction_fraction"]),
                lifetime_extension_fraction=cast(float, distance["lifetime_extension_fraction"]),
                collection_distance=cast(float, distance["collection_distance"]),
                treatment_share_distance=cast(float, distance["treatment_share_distance"]),
                feedstock_share_distance=cast(float, distance["feedstock_share_distance"]),
            ),
            uses_exact_parent=cast(bool, intervention["uses_exact_parent"]),
        ),
        physical=CandidateWorkerPhysicalSummary(
            physical_request_identity=cast(str, physical["physical_request_identity"]),
            physical_candidate_identity=cast(str, physical["physical_candidate_identity"]),
            physical_parent_identity=cast(str, physical["physical_parent_identity"]),
            intervention_authority_identity=cast(str, physical["intervention_authority_identity"]),
            status=cast(PhysicalStatus, physical["status"]),
            diagnostic=_decode_diagnostic(cast(dict[str, object] | None, physical["diagnostic"])),
            physical_state_identity=cast(str | None, physical["physical_state_identity"]),
            activation_identity=cast(str | None, physical["activation_identity"]),
        ),
        scores=(
            None
            if scores is None
            else CandidateWorkerScoreSummary(
                physical_candidate_identity=cast(str, scores["physical_candidate_identity"]),
                environmental_identity=cast(str, scores["environmental_identity"]),
                backgrounds=tuple(cast(Sequence[str], scores["backgrounds"])),
                optimization_method_registry_identity=cast(
                    str, scores["optimization_method_registry_identity"]
                ),
                cumulative_climate_method_registry_identity=cast(
                    str, scores["cumulative_climate_method_registry_identity"]
                ),
                annual_scores_identity=cast(str, scores["annual_scores_identity"]),
                annual_climate_scores_identity=cast(str, scores["annual_climate_scores_identity"]),
                score_identity=cast(str, scores["score_identity"]),
            )
        ),
        feedstock_requirements_identity=cast(str | None, record["feedstock_requirements_identity"]),
        member_metrics=tuple(
            _decode_worker_metric_summary(item)
            for item in cast(Sequence[Mapping[str, object]], record["member_metrics"])
        ),
    )


def _decode_worker_metric_summary(item: Mapping[str, object]) -> CandidateMetricSummary:
    """Restore every saved ranking coordinate and its exact binding constraints."""

    return CandidateMetricSummary(
        identity=cast(str, item["identity"]),
        physical_candidate_identity=cast(str, item["physical_candidate_identity"]),
        intervention_authority_identity=cast(str, item["intervention_authority_identity"]),
        physical_parent_identity=cast(str, item["physical_parent_identity"]),
        distance_inputs_identity=cast(str, item["distance_inputs_identity"]),
        demand_lifetime_parent_identity=cast(str, item["demand_lifetime_parent_identity"]),
        demand_lifetime_trajectory_identity=cast(str, item["demand_lifetime_trajectory_identity"]),
        circularity_trajectory_identity=cast(str, item["circularity_trajectory_identity"]),
        target_year=cast(int, item["target_year"]),
        demand_reduction_fraction=cast(float, item["demand_reduction_fraction"]),
        lifetime_extension_fraction=cast(float, item["lifetime_extension_fraction"]),
        collection_distance=cast(float, item["collection_distance"]),
        treatment_share_distance=cast(float, item["treatment_share_distance"]),
        feedstock_share_distance=cast(float, item["feedstock_share_distance"]),
        evaluation_identity=cast(str, item["evaluation_identity"]),
        metric_identity=cast(str, item["metric_identity"]),
        background_scenario=cast(str, item["background_scenario"]),
        allocated_cc_case=cast(str, item["allocated_cc_case"]),
        allocated_cc_context_identity=cast(str, item["allocated_cc_context_identity"]),
        optimization_item_identity=cast(str, item["optimization_item_identity"]),
        comparison_scope_identity=cast(str, item["comparison_scope_identity"]),
        optimization_block=cast(OptimizationBlock, item["optimization_block"]),
        is_zero_intervention_parent=cast(bool, item["is_zero_intervention_parent"]),
        intervention_distance_identity=cast(str, item["intervention_distance_identity"]),
        intervention_distance=cast(float, item["intervention_distance"]),
        regions=tuple(cast(Sequence[str], item["regions"])),
        all_region_joint_compliant=cast(bool, item["all_region_joint_compliant"]),
        fully_compliant_regions=tuple(cast(Sequence[str], item["fully_compliant_regions"])),
        fully_compliant_impact_targets=tuple(
            cast(Sequence[str], item["fully_compliant_impact_targets"])
        ),
        compliant_region_count=cast(int, item["compliant_region_count"]),
        fully_compliant_impact_target_count=cast(int, item["fully_compliant_impact_target_count"]),
        compliant_region_impact_group_count=cast(int, item["compliant_region_impact_group_count"]),
        compliant_coordinate_count=cast(int, item["compliant_coordinate_count"]),
        region_count=cast(int, item["region_count"]),
        impact_target_count=cast(int, item["impact_target_count"]),
        region_impact_group_count=cast(int, item["region_impact_group_count"]),
        coordinate_count=cast(int, item["coordinate_count"]),
        z_worst=cast(float, item["z_worst"]),
        binding_slack=cast(float | None, item["binding_slack"]),
        maximum_positive_exceedance=cast(float, item["maximum_positive_exceedance"]),
        balanced_mean_positive_exceedance=cast(float, item["balanced_mean_positive_exceedance"]),
        binding_constraints=tuple(
            BindingConstraint(
                region=cast(str, binding["region"]),
                constraint_type=cast(ConstraintType, binding["constraint_type"]),
                optimization_lcia_method=cast(str, binding["optimization_lcia_method"]),
                impact_category_id=cast(str, binding["impact_category_id"]),
                year=cast(int | None, binding["year"]),
                asr=cast(float, binding["asr"]),
            )
            for binding in cast(Sequence[Mapping[str, object]], item["binding_constraints"])
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectedCandidateResult:
    """Selection bound to its original saved worker result.

    ``result`` is decoded from the saved candidate result. Loading restores its saved numerical
    values.
    The compact ``summary`` retains the metrics used to select this result.
    ``demand_lifetime`` retains the realized representative path; the worker
    result separately identifies its requested linear endpoint task.
    """

    identity: str
    selection_identities: tuple[str, ...]
    candidate_identity: str
    summary: CandidateWorkerResultSummary
    result: CandidateWorkerResult
    demand_lifetime: DemandLifetimeDecision


def _selected_candidate_result(
    result: CandidateWorkerResult,
    summary: CandidateWorkerResultSummary,
    *,
    selection_identities: Sequence[str],
) -> SelectedCandidateResult:
    """Bind each selection or delayed timing to its saved MFA outcome."""

    selections = tuple(sorted(selection_identities))
    demand_lifetime = result.intervention.demand_lifetime_evaluation_projection.decision
    candidate = result.intervention.identity
    scores = result.scores
    identity_payload = {
        "method_identity": "pytoce_optimization_selected_candidate_result_v5",
        "realized_demand_lifetime": demand_lifetime.coordinates(),
        "requested_intervention_identity": result.requested_intervention_identity,
        "selection_identities": list(selections),
        "candidate_identity": candidate,
        "worker_result_summary_identity": summary.identity,
        "worker_result_identity": result.identity,
        "worker_authority_identity": result.worker_authority_identity,
        "worker_scientific_authority_identity": result.worker_scientific_authority_identity,
        "intervention_authority_identity": result.intervention.identity,
        "demand_lifetime_evaluation_projection_identity": (
            result.intervention.demand_lifetime_evaluation_projection.identity
        ),
        "circularity_evaluation_projection_identity": (
            result.intervention.circularity_evaluation_projection.identity
        ),
        "demand_lifetime_trajectory_identity": (
            result.intervention.demand_lifetime_trajectory_identity
        ),
        "circularity_trajectory_identity": result.intervention.circularity_trajectory_identity,
        "distance_inputs_identity": result.intervention.distance_inputs.identity,
        "physical_request_identity": result.physical.physical_request_identity,
        "physical_candidate_identity": result.physical.physical_candidate_identity,
        "physical_state_identity": result.physical.physical_state_identity,
        "activation_identity": result.physical.activation_identity,
        "environmental_identity": None if scores is None else scores.environmental_identity,
        "score_identity": None if scores is None else scores.score_identity,
        "annual_scores_identity": None if scores is None else scores.annual_scores_identity,
        "annual_climate_scores_identity": (
            None if scores is None else scores.annual_climate_scores_identity
        ),
        "metric_scope_records": [
            [item.optimization_block, item.metric_registry_identity]
            for item in result.metric_scopes
        ],
        "member_metric_entry_identities": [item.identity for item in summary.member_metrics],
        "feedstock_requirements_identity": result.feedstock_requirements_identity,
    }
    return SelectedCandidateResult(
        identity=_SELECTED_CANDIDATE_RESULT_PREFIX + payload_sha256(identity_payload),
        selection_identities=selections,
        candidate_identity=candidate,
        summary=summary,
        result=result,
        demand_lifetime=demand_lifetime,
    )


def encode_candidate_worker_task(task: CandidateWorkerTask) -> bytes:
    """Encode one candidate task as canonical JSON."""

    return _TASK_MAGIC + _canonical_json_bytes(
        {
            "schema_version": CANDIDATE_WORKER_TASK_SCHEMA,
            "intervention": asdict(task.intervention),
            "metric_scopes": [asdict(scope) for scope in task.metric_scopes],
        }
    )


def _decode_demand_lifetime(item: dict[str, Any]) -> DemandLifetimeDecision:
    """Restore the sole exact annual path state from generated rational pairs."""
    return DemandLifetimeDecision(
        item["demand_reduction_pct"],
        item["lifetime_extension_pct"],
        item["target_year"],
        tuple(Fraction(*value) for value in item["demand_fraction"]),
        tuple(Fraction(*value) for value in item["lifetime_fraction"]),
    )


def decode_candidate_worker_task(payload: bytes) -> CandidateWorkerTask:
    """Decode the intervention and metric scopes written by the parent process."""
    root = json.loads(payload[len(_TASK_MAGIC) :])
    return CandidateWorkerTask(
        intervention=_decode_intervention(root["intervention"]),
        metric_scopes=tuple(CandidateWorkerMetricScope(**item) for item in root["metric_scopes"]),
    )


def encode_candidate_worker_result(result: CandidateWorkerResult) -> bytes:
    """Encode one candidate result with seven fixed binary sections.

    PA annual impacts, absolute contribution totals and sparse stages are the
    already calculated EF search projection. Selected reporting can reuse those methods
    or characterize the stored PA events with its requested methods;
    serialization stores the completed projection.
    """

    activation = (
        b""
        if result.physical.activation_delta is None
        else _arrow_frame_bytes(result.physical.activation_delta)
    )
    scores = result.scores
    annual_scores = b"" if scores is None else _float64_bytes(scores.annual_scores)
    annual_climate_scores = b"" if scores is None else _float64_bytes(scores.annual_climate_scores)
    requirements = (
        b""
        if result.feedstock_requirements is None
        else _arrow_frame_bytes(result.feedstock_requirements)
    )
    pa_scores = result.pa_scores
    pa_annual = b"" if pa_scores is None else _float64_bytes(pa_scores.annual_scores)
    pa_norms = b"" if pa_scores is None else _float64_bytes(pa_scores.annual_contribution_norms)
    pa_stages = b"" if pa_scores is None else _arrow_frame_bytes(pa_scores.stage_scores)
    header = _canonical_json_bytes(_result_header(result))
    output = bytearray(_RESULT_MAGIC)
    output.extend(struct.pack("<Q", len(header)))
    output.extend(header)
    for section in (
        activation,
        annual_scores,
        annual_climate_scores,
        requirements,
        pa_annual,
        pa_norms,
        pa_stages,
    ):
        output.extend(struct.pack("<Q", len(section)))
        output.extend(section)
    return bytes(output)


def candidate_worker_result_identity(result: CandidateWorkerResult) -> str:
    """Bind physical, EF search/PA score and feedstock content at its producer."""
    header = _result_header(result)
    del header["identity"]
    repair = dict(asdict(result.repair))
    # Actual work varies when an initialized worker reuses a conditional solve; it is retained in
    # the receipt as execution metadata.
    for name in (
        "mfa_evaluation_count",
        "stock_lp_count",
        "pool_evaluation_count",
        "physical_repair_elapsed_seconds",
        "environmental_evaluation_elapsed_seconds",
    ):
        del repair[name]
    header["repair"] = repair
    return _WORKER_RESULT_PREFIX + sha256(_canonical_json_bytes(header)).hexdigest()


def decode_candidate_worker_result(payload: bytes) -> CandidateWorkerResult:
    """Reconstruct the saved candidate fields, tables and score arrays."""
    header, sections = _result_message(payload)
    activation = None if not sections[0] else _arrow_frame(sections[0])
    requirements = None if not sections[3] else _arrow_frame(sections[3])
    physical = header["physical"]
    diagnostic = physical["diagnostic"]
    if diagnostic is not None:
        physical["diagnostic"] = OptimizationPhysicalInfeasibility(
            **{
                **diagnostic,
                "coordinates": tuple(tuple(row) for row in diagnostic["coordinates"]),
                "details": tuple(tuple(row) for row in diagnostic["details"]),
            }
        )
    physical["pa_spinning_events"] = tuple(
        _decode_pa_spinning_event(row) for row in physical["pa_spinning_events"]
    )
    physical["pa_recycling_events"] = tuple(
        PaRecyclingEvent(
            **{
                **row,
                "principal_by_origin_kg": tuple(row["principal_by_origin_kg"]),
                "airborne_principal_by_origin_kg": tuple(row["airborne_principal_by_origin_kg"]),
            }
        )
        for row in physical["pa_recycling_events"]
    )
    physical["pa_disposal_events"] = tuple(
        PaDisposalEvent(**{**row, "principal_by_origin_kg": tuple(row["principal_by_origin_kg"])})
        for row in physical["pa_disposal_events"]
    )
    return CandidateWorkerResult(
        identity=header["identity"],
        worker_authority_identity=header["worker_authority_identity"],
        worker_scientific_authority_identity=header["worker_scientific_authority_identity"],
        requested_intervention_identity=header["requested_intervention_identity"],
        repair=_decode_repair(header["repair"]),
        intervention=_decode_intervention(header["intervention"]),
        physical=PhysicalEvaluation(**physical, activation_delta=activation),
        scores=_decode_scores(
            header["scores"], annual_scores=sections[1], annual_climate_scores=sections[2]
        ),
        pa_scores=_decode_pa_scores(
            header["pa_scores"],
            annual_scores=sections[4],
            annual_norms=sections[5],
            stages=sections[6],
        ),
        metric_scopes=tuple(CandidateWorkerMetricScope(**item) for item in header["metric_scopes"]),
        feedstock_requirements=requirements,
        feedstock_requirements_identity=header["feedstock_requirements_identity"],
    )


def _decode_diagnostic(item: dict[str, Any] | None) -> OptimizationPhysicalInfeasibility | None:
    """Restore a direct-owner diagnostic retained in a completed repair result."""
    if item is None:
        return None
    return OptimizationPhysicalInfeasibility(
        **{
            **item,
            "coordinates": tuple(tuple(row) for row in item["coordinates"]),
            "details": tuple(tuple(row) for row in item["details"]),
        }
    )


def _decode_repair(item: dict[str, Any]) -> DemandRepairOutcome:
    """Decode exact departure and compact repair outcome metadata."""
    return DemandRepairOutcome(
        **{
            **item,
            "demand_departure_pp": (
                None
                if item["demand_departure_pp"] is None
                else Fraction(*item["demand_departure_pp"])
            ),
            "linear_diagnostic": _decode_diagnostic(item["linear_diagnostic"]),
            "stock_projection_diagnostic": _decode_diagnostic(item["stock_projection_diagnostic"]),
        }
    )


def _decode_pa_spinning_event(row: dict[str, Any]) -> PaSpinningEvent:
    """Restore fixed fossil/biogenic/DAC tuples from the worker's own JSON record."""
    return PaSpinningEvent(
        **{
            **row,
            **{
                field: tuple(row[field])
                for field in (
                    "maintenance_principal_by_origin_kg",
                    "finished_principal_carbon_by_origin_kg",
                    "caprolactam_air_by_origin_kg",
                    "caprolactam_water_by_origin_kg",
                )
            },
        }
    )


def _result_header(result: CandidateWorkerResult) -> dict[str, object]:
    """Return the JSON portion of one candidate result."""

    physical = result.physical
    scores = result.scores
    pa_scores = result.pa_scores
    return {
        "schema_version": CANDIDATE_WORKER_RESULT_SCHEMA,
        "identity": result.identity,
        "worker_authority_identity": result.worker_authority_identity,
        "worker_scientific_authority_identity": result.worker_scientific_authority_identity,
        "requested_intervention_identity": result.requested_intervention_identity,
        "repair": asdict(result.repair),
        "intervention": asdict(result.intervention),
        "physical": {
            "physical_request_identity": physical.physical_request_identity,
            "physical_candidate_identity": physical.physical_candidate_identity,
            "physical_parent_identity": physical.physical_parent_identity,
            "intervention_authority_identity": physical.intervention_authority_identity,
            "status": physical.status,
            "diagnostic": None if physical.diagnostic is None else asdict(physical.diagnostic),
            "physical_state_identity": physical.physical_state_identity,
            "activation_identity": physical.activation_identity,
            "pa_spinning_events": [asdict(event) for event in physical.pa_spinning_events],
            "pa_recycling_events": [asdict(event) for event in physical.pa_recycling_events],
            "pa_disposal_events": [asdict(event) for event in physical.pa_disposal_events],
        },
        "scores": (
            None
            if scores is None
            else {
                "physical_candidate_identity": scores.physical_candidate_identity,
                "environmental_identity": scores.environmental_identity,
                "member_identities": scores.member_identities,
                "backgrounds": scores.backgrounds,
                "regions": scores.regions,
                "years": scores.years,
                "annual_categories": [asdict(item) for item in scores.annual_categories],
                "optimization_method_registry_identity": (
                    scores.optimization_method_registry_identity
                ),
                "cumulative_climate_method_registry_identity": (
                    scores.cumulative_climate_method_registry_identity
                ),
                "annual_scores_identity": scores.annual_scores_identity,
                "annual_climate_scores_identity": scores.annual_climate_scores_identity,
                "score_identity": scores.score_identity,
            }
        ),
        "pa_scores": (
            None
            if pa_scores is None
            else {
                "shape": list(pa_scores.annual_scores.shape),
                "annual_scores_identity": _array_identity(pa_scores.annual_scores),
                "annual_contribution_norms_identity": _array_identity(
                    pa_scores.annual_contribution_norms
                ),
                "stage_scores_identity": canonical_table_sha256(pa_scores.stage_scores),
            }
        ),
        "metric_scopes": [asdict(item) for item in result.metric_scopes],
        "feedstock_requirements_identity": result.feedstock_requirements_identity,
    }


def _result_message(payload: bytes) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """Read the JSON header and seven length-prefixed sections written by a worker."""
    stream = BytesIO(payload)
    stream.seek(len(_RESULT_MAGIC))
    header_length = struct.unpack("<Q", stream.read(8))[0]
    header = json.loads(stream.read(header_length))
    sections = tuple(
        stream.read(struct.unpack("<Q", stream.read(8))[0]) for _ in range(_RESULT_SECTION_COUNT)
    )
    return header, sections


def _decode_intervention(item: dict[str, Any]) -> CandidateInterventionAuthority:
    """Restore nested intervention records and assignment tuples from generated JSON."""
    demand = item["demand_lifetime_evaluation_projection"]
    circularity = item["circularity_evaluation_projection"]
    decision = demand["decision"]
    return CandidateInterventionAuthority(
        **{
            **item,
            "demand_lifetime_evaluation_projection": DemandLifetimeEvaluationProjection(
                **{
                    **demand,
                    "decision": _decode_demand_lifetime(decision),
                }
            ),
            "circularity_evaluation_projection": CircularityEvaluationProjection(
                **{**circularity, "decision": CircularityDecision(**circularity["decision"])}
            ),
            "distance_inputs": InterventionDistanceInputs(**item["distance_inputs"]),
            "assignment_records": tuple(tuple(row) for row in item["assignment_records"]),
        }
    )


def _decode_scores(
    item: dict[str, Any] | None,
    *,
    annual_scores: bytes,
    annual_climate_scores: bytes,
) -> OptimizationScores | None:
    """Restore native LCIA arrays and coordinates from a generated worker result.

    Scores are reported for candidates admitted by the MFA. Feasible scores use little-endian
    float64 with axes [background, world region, year, category]; annual climate
    scores omit the category axis. Stored field and content identities are
    retained.
    """
    if item is None:
        return None
    backgrounds = tuple(item["backgrounds"])
    regions = tuple(item["regions"])
    years = tuple(item["years"])
    categories = tuple(AnnualCategory(**record) for record in item["annual_categories"])
    shape = (len(backgrounds), len(regions), len(years))
    return OptimizationScores(
        **{
            **item,
            "member_identities": tuple(item["member_identities"]),
            "backgrounds": backgrounds,
            "regions": regions,
            "years": years,
            "annual_categories": categories,
            "annual_scores": np.frombuffer(annual_scores, dtype="<f8").reshape(
                (*shape, len(categories))
            ),
            "annual_climate_scores": np.frombuffer(annual_climate_scores, dtype="<f8").reshape(
                shape
            ),
        }
    )


def _decode_pa_scores(
    item: dict[str, Any] | None,
    *,
    annual_scores: bytes,
    annual_norms: bytes,
    stages: bytes,
) -> PaCandidateScores | None:
    """Restore completed EF search PA arrays and stages.

    The producer already bound their content and the worker scientific authority
    binds the background, method and source ordering. Little-endian float64
    buffers are read-only; the Arrow section retains the native stage schema,
    including a typed empty table when a feasible candidate has no PA stream.
    The reader restores the completed worker payload.
    """
    if item is None:
        return None
    shape = tuple(item["shape"])
    return PaCandidateScores(
        np.frombuffer(annual_scores, dtype="<f8").reshape(shape),
        np.frombuffer(annual_norms, dtype="<f8").reshape(shape),
        _arrow_frame(stages),
    )


def _float64_bytes(values: np.ndarray) -> bytes:
    """Return contiguous little-endian float64 bytes."""

    return np.ascontiguousarray(values, dtype="<f8").tobytes()


def _arrow_frame(data: bytes) -> pd.DataFrame:
    """Decode one generated Arrow table section."""
    return ipc.open_stream(BytesIO(data)).read_all().to_pandas()  # type: ignore[no-untyped-call]


def _canonical_json_bytes(value: object) -> bytes:
    """Encode a worker header as sorted compact UTF-8 JSON."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: (item.numerator, item.denominator),
        )
        + "\n"
    ).encode("utf-8")


def _arrow_frame_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as an Arrow IPC stream with data columns only."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    stream = BytesIO()
    options = pa.ipc.IpcWriteOptions(compression="zstd", use_threads=True)
    with ipc.new_stream(  # type: ignore[no-untyped-call]
        stream,
        table.schema,
        options=options,
    ) as writer:
        writer.write_table(table)
    return stream.getvalue()
