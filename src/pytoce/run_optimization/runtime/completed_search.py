"""Compact completed-stage records embedded in the selected optimization product.

The result-store owner admits a compatible completed product before decoding.
This codec restores its saved values from the compact completed-stage records.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
from typing import Final, Literal, cast

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CandidateRefinementStage,
    CandidateSearchState,
    CandidateSelection,
    InterventionCandidate,
    RefinementPass,
    SelectedCandidateState,
    SelectionStatus,
)
from pytoce.run_optimization.calculation.integrated_design import (
    INTEGRATED_DESIGN_METHOD_IDENTITY,
    EndpointCoordinates,
    JointPairCover,
    JointRoute,
    JointStageDesign,
    RouteFamily,
    explicit_endpoint_coordinates,
)
from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
    IntegratedBranchPoll,
    IntegratedBranchSeed,
    IntegratedBranchTerminal,
    IntegratedFamilySeed,
    IntegratedGuideRole,
    IntegratedPollBatch,
    IntegratedSearchState,
    IntegratedStageState,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CompletedCandidateSearch,
    RequestSearchStage,
    _candidate_evaluation,
    search_configuration_identity,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    decode_candidate_worker_summary,
    encode_candidate_worker_summary,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateTaskRecord

COMPLETED_SEARCH_SCHEMA: Final = "pytoce_completed_domain_search_schema_v1"


def encode_completed_search(completed: CompletedCandidateSearch) -> dict[str, object]:
    """Encode shared catalogs once and retain complete stage order and obligations."""
    candidates: dict[str, InterventionCandidate] = {}
    joint_stages: dict[str, IntegratedStageState] = {}
    passes: dict[RefinementPass, int] = {}
    for stage in completed.stages:
        for candidate in (*stage.demand_anchors, *stage.circularity_anchors):
            candidates[candidate.identity] = candidate
        for single in stage.single_blocks:
            for evaluation in single.evaluations:
                candidates[evaluation.candidate.identity] = evaluation.candidate
            for refinement_pass in single.passes:
                passes.setdefault(refinement_pass, len(passes))
        if stage.integrated is not None:
            for evaluation in stage.integrated.evaluations:
                candidates[evaluation.candidate.identity] = evaluation.candidate
            for joint in stage.integrated.completed_stages:
                joint_stages[joint.design.identity] = joint
                for candidate in joint.design.candidates:
                    candidates[candidate.identity] = candidate
                for poll in joint.polls:
                    passes.setdefault(poll.refinement_pass, len(passes))
    candidate_values = tuple(candidates[key] for key in sorted(candidates))
    candidate_indices = {item.identity: index for index, item in enumerate(candidate_values)}
    demands: dict[DemandLifetimeDecision, int] = {}
    for candidate in candidate_values:
        demands.setdefault(candidate.demand_lifetime, len(demands))
    covers: dict[JointPairCover, int] = {}
    for joint in joint_stages.values():
        covers.setdefault(joint.design.pair_cover, len(covers))
    joint_indices = {identity: index for index, identity in enumerate(joint_stages)}
    return {
        "schema_version": COMPLETED_SEARCH_SCHEMA,
        "integrated_search_method": INTEGRATED_SEARCH_METHOD_IDENTITY,
        "integrated_design_method": INTEGRATED_DESIGN_METHOD_IDENTITY,
        "search_configuration_identity": search_configuration_identity(
            completed.endpoint_refinement_steps_pct,
            initial_grid_step_pct=completed.initial_grid_step_pct,
        ),
        "initial_grid_step_pct": completed.initial_grid_step_pct,
        "endpoint_refinement_steps_pct": completed.endpoint_refinement_steps_pct,
        "single_block_stage_policies": [
            [step, step <= 5 or step == completed.endpoint_refinement_steps_pct[-1]]
            for step in completed.endpoint_refinement_steps_pct
        ],
        "executed_blocks": completed.executed_blocks,
        "demand_lifetime_decisions": [item.coordinates() for item in demands],
        "candidates": [
            {
                "identity": item.identity,
                "optimization_block": item.optimization_block,
                "demand_lifetime": demands[item.demand_lifetime],
                "circularity": item.circularity.percentages(),
                "circularity_target_year": item.circularity.target_year,
            }
            for item in candidate_values
        ],
        "pair_covers": [asdict(item) for item in covers],
        "refinement_passes": [asdict(item) for item in passes],
        "joint_stages": [
            _encode_joint_stage(item, candidate_indices, passes, covers)
            for item in joint_stages.values()
        ],
        "stages": [
            _encode_request_stage(item, candidate_indices, passes, joint_indices)
            for item in completed.stages
        ],
        "candidate_records": completed.candidate_records,
        "worker_summaries": [
            encode_candidate_worker_summary(item) for item in completed.worker_summaries
        ],
        "task_records": [asdict(item) for item in completed.task_records],
    }


def _encode_request_stage(
    stage: RequestSearchStage,
    candidate_indices: Mapping[str, int],
    passes: Mapping[RefinementPass, int],
    joint_indices: Mapping[str, int],
) -> dict[str, object]:
    """Reference shared candidates and joint stages from one completed resolution."""
    integrated = stage.integrated
    return {
        "step_pct": stage.step_pct,
        "single_blocks": [
            {
                "optimization_block": item.optimization_block,
                "evaluations": [
                    candidate_indices[value.candidate.identity] for value in item.evaluations
                ],
                "selection": asdict(item.selection),
                "completed_stages": [asdict(value) for value in item.completed_stages],
                "passes": [passes[value] for value in item.passes],
                "reused_candidate_count": item.reused_candidate_count,
                "final_neighbor_candidate_identities": item.final_neighbor_candidate_identities,
                "final_anchor_candidate_identities": item.final_anchor_candidate_identities,
                "final_selection_stable": item.final_selection_stable,
            }
            for item in stage.single_blocks
        ],
        "integrated": None
        if integrated is None
        else {
            "evaluations": [
                candidate_indices[value.candidate.identity] for value in integrated.evaluations
            ],
            "completed_stages": [
                joint_indices[value.design.identity] for value in integrated.completed_stages
            ],
            "reused_candidate_count": integrated.reused_candidate_count,
        },
        "demand_anchors": [candidate_indices[value.identity] for value in stage.demand_anchors],
        "circularity_anchors": [
            candidate_indices[value.identity] for value in stage.circularity_anchors
        ],
    }


def _encode_joint_stage(
    stage: IntegratedStageState,
    candidate_indices: Mapping[str, int],
    passes: Mapping[RefinementPass, int],
    covers: Mapping[JointPairCover, int],
) -> dict[str, object]:
    """Keep one copy of each stage's route geometry, guide set and branch terminals."""
    design = stage.design
    coordinates = {
        explicit_endpoint_coordinates(item): candidate_indices[item.identity]
        for item in design.candidates
        if not item.circularity.uses_exact_parent
    }
    return {
        "design": {
            "identity": design.identity,
            "step_pct": design.step_pct,
            "candidates": [candidate_indices[item.identity] for item in design.candidates],
            "starting_candidate_identities": design.starting_candidate_identities,
            "source_candidate_identities": design.source_candidate_identities,
            "geometric_guide_candidate_identities": design.geometric_guide_candidate_identities,
            "actual_ii_candidates": [
                candidate_indices[item.identity] for item in design.actual_ii_candidates
            ],
            "actual_i_pairs": design.actual_i_pairs,
            "routes": [
                {
                    "source": coordinates[item.source],
                    "target": coordinates[item.target],
                    "step_pct": item.step_pct,
                    "points": [coordinates[point] for point in item.points],
                    "origins": [[coordinates[guide], family] for guide, family in item.origins],
                }
                for item in design.routes
            ],
            "proposal_origins": design.proposal_origins,
            "pair_cover": covers[design.pair_cover],
            "explicit_ii_routes_absent": design.explicit_ii_routes_absent,
        },
        "selection": asdict(stage.selection),
        "admitted_starting_candidate_identities": stage.admitted_starting_candidate_identities,
        "local_guide_candidate_identities": stage.local_guide_candidate_identities,
        "guide_roles": [asdict(item) for item in stage.guide_roles],
        "family_seeds": [asdict(item) for item in stage.family_seeds],
        "pending_branches": [asdict(item) for item in stage.pending_branches],
        "terminal_branches": [asdict(item) for item in stage.terminal_branches],
        "polls": [
            {
                "state_identity": item.state_identity,
                "refinement_pass": passes[item.refinement_pass],
                "batches": [asdict(batch) for batch in item.batches],
                "evaluated_candidate_identities": item.evaluated_candidate_identities,
                "skipped_candidate_identities": item.skipped_candidate_identities,
                "purpose": item.purpose,
            }
            for item in stage.polls
        ],
        "parent_shortcut": stage.parent_shortcut,
        "local_polling_complete": stage.local_polling_complete,
    }


def decode_completed_search(record: Mapping[str, object]) -> CompletedCandidateSearch:
    """Restore a compatible completed product."""
    demands = tuple(
        _decode_demand_lifetime(item)
        for item in cast(Sequence[Sequence[object]], record["demand_lifetime_decisions"])
    )
    candidates = tuple(
        _decode_candidate(item, demands)
        for item in cast(Sequence[Mapping[str, object]], record["candidates"])
    )
    summaries = tuple(
        decode_candidate_worker_summary(item)
        for item in cast(Sequence[Mapping[str, object]], record["worker_summaries"])
    )
    summaries_by_identity = {item.identity: item for item in summaries}
    candidate_records = tuple(
        cast(tuple[str, str, str], tuple(item))
        for item in cast(Sequence[tuple[str, str, str]], record["candidate_records"])
    )
    summaries_by_candidate = {candidate: summary for candidate, _, summary in candidate_records}
    evaluations = {
        index: _candidate_evaluation(
            item, summaries_by_identity[summaries_by_candidate[item.identity]]
        )
        for index, item in enumerate(candidates)
        if item.identity in summaries_by_candidate
    }
    passes = tuple(
        _decode_refinement_pass(item)
        for item in cast(Sequence[Mapping[str, object]], record["refinement_passes"])
    )
    covers = tuple(
        JointPairCover(
            rows=tuple(
                cast(EndpointCoordinates, tuple(row))
                for row in cast(Sequence[EndpointCoordinates], item["rows"])
            ),
            requirement_count=cast(int, item["requirement_count"]),
            covered_requirement_count=cast(int, item["covered_requirement_count"]),
            initial_grid_step_pct=cast(int, item["initial_grid_step_pct"]),
        )
        for item in cast(Sequence[Mapping[str, object]], record["pair_covers"])
    )
    joint_stages = tuple(
        _decode_joint_stage(item, candidates, passes, covers)
        for item in cast(Sequence[Mapping[str, object]], record["joint_stages"])
    )
    return CompletedCandidateSearch(
        initial_grid_step_pct=cast(int, record["initial_grid_step_pct"]),
        endpoint_refinement_steps_pct=tuple(
            cast(Sequence[int], record["endpoint_refinement_steps_pct"])
        ),
        executed_blocks=tuple(cast(Sequence[OptimizationBlock], record["executed_blocks"])),
        stages=tuple(
            _decode_request_stage(item, candidates, evaluations, passes, joint_stages)
            for item in cast(Sequence[Mapping[str, object]], record["stages"])
        ),
        candidate_records=candidate_records,
        worker_summaries=summaries,
        task_records=tuple(
            CandidateTaskRecord(
                cast(int, item["sequence"]),
                cast(str, item["candidate_identity"]),
                cast(str | None, item["repair_family_identity"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["task_records"])
        ),
    )


def _decode_demand_lifetime(values: Sequence[object]) -> DemandLifetimeDecision:
    """Restore exact rational annual controls once per distinct retained decision."""
    return DemandLifetimeDecision(
        cast(int, values[0]),
        cast(int, values[1]),
        cast(int, values[2]),
        tuple(
            Fraction(numerator, denominator)
            for numerator, denominator in cast(Sequence[tuple[int, int]], values[3])
        ),
        tuple(
            Fraction(numerator, denominator)
            for numerator, denominator in cast(Sequence[tuple[int, int]], values[4])
        ),
    )


def _decode_candidate(
    record: Mapping[str, object], demands: Sequence[DemandLifetimeDecision]
) -> InterventionCandidate:
    """Reference shared annual controls and preserve categorical parent circularity."""
    circularity = cast(tuple[int, int, int, int, int, int, int, int] | None, record["circularity"])
    target_year = cast(int, record["circularity_target_year"])
    return InterventionCandidate(
        cast(OptimizationBlock, record["optimization_block"]),
        demands[cast(int, record["demand_lifetime"])],
        CircularityDecision.parent(target_year=target_year)
        if circularity is None
        else CircularityDecision(*circularity, target_year),
    )


def _decode_selection(record: Mapping[str, object]) -> CandidateSelection:
    """Restore exact selected state identities and statuses."""
    return CandidateSelection(
        identity=cast(str, record["identity"]),
        states=tuple(
            SelectedCandidateState(
                cast(str, item["state_identity"]),
                cast(str, item["candidate_identity"]),
                cast(SelectionStatus, item["status"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["states"])
        ),
        selected_candidate_identities=tuple(
            cast(Sequence[str], record["selected_candidate_identities"])
        ),
    )


def _decode_refinement_pass(record: Mapping[str, object]) -> RefinementPass:
    """Preserve proposal, pruning, evaluation and selection order from a completed poll."""
    return RefinementPass(
        step_pct=cast(int, record["step_pct"]),
        iteration=cast(int, record["iteration"]),
        anchor_candidate_identities=tuple(
            cast(Sequence[str], record["anchor_candidate_identities"])
        ),
        proposed_candidate_identities=tuple(
            cast(Sequence[str], record["proposed_candidate_identities"])
        ),
        pruned_candidate_identities=tuple(
            cast(Sequence[str], record["pruned_candidate_identities"])
        ),
        newly_evaluated_candidate_identities=tuple(
            cast(Sequence[str], record["newly_evaluated_candidate_identities"])
        ),
        selected_candidate_identities=tuple(
            cast(Sequence[str], record["selected_candidate_identities"])
        ),
        coordinated=cast(bool, record["coordinated"]),
    )


def _decode_request_stage(
    record: Mapping[str, object],
    candidates: Sequence[InterventionCandidate],
    evaluations: Mapping[int, CandidateEvaluation],
    passes: Sequence[RefinementPass],
    joint_stages: Sequence[IntegratedStageState],
) -> RequestSearchStage:
    """Restore single-block prerequisites and joint continuation references for a stage."""
    integrated = cast(Mapping[str, object] | None, record["integrated"])
    restored_joint = None
    if integrated is not None:
        completed = tuple(
            joint_stages[index] for index in cast(Sequence[int], integrated["completed_stages"])
        )
        restored_joint = IntegratedSearchState(
            evaluations=tuple(
                evaluations[index] for index in cast(Sequence[int], integrated["evaluations"])
            ),
            selection=completed[-1].selection,
            completed_stages=completed,
            terminal_branches=completed[-1].terminal_branches,
            pending_branches=completed[-1].pending_branches,
            reused_candidate_count=cast(int, integrated["reused_candidate_count"]),
        )
    return RequestSearchStage(
        step_pct=cast(int, record["step_pct"]),
        single_blocks=tuple(
            CandidateSearchState(
                optimization_block=cast(OptimizationBlock, item["optimization_block"]),
                evaluations=tuple(
                    evaluations[index] for index in cast(Sequence[int], item["evaluations"])
                ),
                selection=_decode_selection(cast(Mapping[str, object], item["selection"])),
                completed_stages=tuple(
                    CandidateRefinementStage(
                        cast(int, stage["step_pct"]), cast(bool, stage["unpruned"])
                    )
                    for stage in cast(Sequence[Mapping[str, object]], item["completed_stages"])
                ),
                passes=tuple(passes[index] for index in cast(Sequence[int], item["passes"])),
                reused_candidate_count=cast(int, item["reused_candidate_count"]),
                final_neighbor_candidate_identities=tuple(
                    cast(Sequence[str], item["final_neighbor_candidate_identities"])
                ),
                final_anchor_candidate_identities=tuple(
                    cast(Sequence[str], item["final_anchor_candidate_identities"])
                ),
                final_selection_stable=cast(bool, item["final_selection_stable"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["single_blocks"])
        ),
        integrated=restored_joint,
        demand_anchors=tuple(
            candidates[index] for index in cast(Sequence[int], record["demand_anchors"])
        ),
        circularity_anchors=tuple(
            candidates[index] for index in cast(Sequence[int], record["circularity_anchors"])
        ),
    )


def _decode_joint_stage(
    record: Mapping[str, object],
    candidates: Sequence[InterventionCandidate],
    passes: Sequence[RefinementPass],
    covers: Sequence[JointPairCover],
) -> IntegratedStageState:
    """Restore one immutable joint design, its guide admission and branch obligations."""
    design = cast(Mapping[str, object], record["design"])
    return IntegratedStageState(
        design=JointStageDesign(
            identity=cast(str, design["identity"]),
            step_pct=cast(int, design["step_pct"]),
            candidates=tuple(
                candidates[index] for index in cast(Sequence[int], design["candidates"])
            ),
            starting_candidate_identities=tuple(
                cast(Sequence[str], design["starting_candidate_identities"])
            ),
            source_candidate_identities=tuple(
                cast(Sequence[str], design["source_candidate_identities"])
            ),
            geometric_guide_candidate_identities=tuple(
                cast(Sequence[str], design["geometric_guide_candidate_identities"])
            ),
            actual_ii_candidates=tuple(
                candidates[index] for index in cast(Sequence[int], design["actual_ii_candidates"])
            ),
            actual_i_pairs=tuple(
                cast(tuple[int, int], tuple(pair))
                for pair in cast(Sequence[Sequence[int]], design["actual_i_pairs"])
            ),
            routes=tuple(
                JointRoute(
                    source=explicit_endpoint_coordinates(candidates[cast(int, item["source"])]),
                    target=explicit_endpoint_coordinates(candidates[cast(int, item["target"])]),
                    step_pct=cast(int, item["step_pct"]),
                    points=tuple(
                        explicit_endpoint_coordinates(candidates[index])
                        for index in cast(Sequence[int], item["points"])
                    ),
                    origins=tuple(
                        (explicit_endpoint_coordinates(candidates[index]), family)
                        for index, family in cast(
                            Sequence[tuple[int, RouteFamily]], item["origins"]
                        )
                    ),
                )
                for item in cast(Sequence[Mapping[str, object]], design["routes"])
            ),
            proposal_origins=tuple(
                (identity, tuple(families))
                for identity, families in cast(
                    Sequence[tuple[str, Sequence[str]]], design["proposal_origins"]
                )
            ),
            pair_cover=covers[cast(int, design["pair_cover"])],
            explicit_ii_routes_absent=cast(bool, design["explicit_ii_routes_absent"]),
        ),
        selection=_decode_selection(cast(Mapping[str, object], record["selection"])),
        admitted_starting_candidate_identities=tuple(
            cast(Sequence[str], record["admitted_starting_candidate_identities"])
        ),
        local_guide_candidate_identities=tuple(
            cast(Sequence[str], record["local_guide_candidate_identities"])
        ),
        guide_roles=tuple(
            IntegratedGuideRole(
                role=cast(Literal["actual_ii", "independent_joint", "conditional_i"], item["role"]),
                state_identity=cast(str | None, item["state_identity"]),
                demand_lifetime_pair=None
                if item["demand_lifetime_pair"] is None
                else cast(
                    tuple[int, int], tuple(cast(Sequence[int], item["demand_lifetime_pair"]))
                ),
                candidate_identity=cast(str | None, item["candidate_identity"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["guide_roles"])
        ),
        family_seeds=tuple(
            IntegratedFamilySeed(
                state_identity=cast(str, item["state_identity"]),
                family=cast(str, item["family"]),
                candidate_identity=cast(str | None, item["candidate_identity"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["family_seeds"])
        ),
        pending_branches=tuple(
            IntegratedBranchSeed(
                state_identity=cast(str, item["state_identity"]),
                candidate_identity=cast(str, item["candidate_identity"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["pending_branches"])
        ),
        terminal_branches=tuple(
            IntegratedBranchTerminal(
                cast(str, item["state_identity"]),
                cast(str, item["candidate_identity"]),
                tuple(cast(Sequence[str], item["neighbor_candidate_identities"])),
            )
            for item in cast(Sequence[Mapping[str, object]], record["terminal_branches"])
        ),
        polls=tuple(
            IntegratedBranchPoll(
                state_identity=cast(str, item["state_identity"]),
                refinement_pass=passes[cast(int, item["refinement_pass"])],
                batches=tuple(
                    IntegratedPollBatch(
                        family=cast(str, batch["family"]),
                        proposed_candidate_identities=tuple(
                            cast(Sequence[str], batch["proposed_candidate_identities"])
                        ),
                        newly_evaluated_candidate_identities=tuple(
                            cast(Sequence[str], batch["newly_evaluated_candidate_identities"])
                        ),
                    )
                    for batch in cast(Sequence[Mapping[str, object]], item["batches"])
                ),
                evaluated_candidate_identities=tuple(
                    cast(Sequence[str], item["evaluated_candidate_identities"])
                ),
                skipped_candidate_identities=tuple(
                    cast(Sequence[str], item["skipped_candidate_identities"])
                ),
                purpose=cast(Literal["domain_refinement"], item["purpose"]),
            )
            for item in cast(Sequence[Mapping[str, object]], record["polls"])
        ),
        parent_shortcut=cast(bool, record["parent_shortcut"]),
        local_polling_complete=cast(bool, record["local_polling_complete"]),
    )
