"""Synthetic endpoint responses for completed-stage search tests.

The finite response domain supplies synthetic worker records. Distance projection,
ranking, proposal generation, and stage continuation use the package implementation.
Requested annual controls follow linear paths.
"""

from dataclasses import asdict, replace
from functools import lru_cache
from typing import cast

from candidate_worker_fixtures import _result, _summary

from pytoce.run_mfa.scenario.scenario_contract import OptimizationPhysicalInfeasibility
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_metrics import (
    _candidate_metric_summary_identity,
)
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CompetitiveCandidateBatch,
    InterventionCandidate,
    circularity_feedstock_pairwise_design,
)
from pytoce.run_optimization.calculation.interventions import (
    intervention_distance_inputs_identity,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    BindingConstraint,
    intervention_distance,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    CompletedCandidateSearch,
    _candidate_evaluation,
    _execution_blocks,
    _run_candidate_search,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerResultSummary,
    DemandRepairOutcome,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateTaskRecord
from pytoce.shared.runtime.filesystem import payload_sha256


@lru_cache(maxsize=1)
def _summary_template() -> CandidateWorkerResultSummary:
    """Reuse the compact worker-response structure."""
    return _summary(_result())


@lru_cache(maxsize=1)
def _toy_circularity_anchor() -> tuple[int, ...] | None:
    """Choose one declared coarse explicit vector for the finite response domain."""
    return circularity_feedstock_pairwise_design(step_pct=100)[0].circularity.percentages()


class CompletedSearchOracle:
    """Supply a bounded synthetic endpoint response while retaining real search state."""

    def __init__(self, *, parent_compliant: bool) -> None:
        """Declare whether the toy parent terminates search or requires finite alternatives."""
        self.parent_compliant = parent_compliant
        self.summaries: dict[str, CandidateWorkerResultSummary] = {}
        self.records: dict[str, tuple[str, str]] = {}

    @property
    def candidate_records(self) -> tuple[tuple[str, str, str], ...]:
        """Retain exact candidate-to-request/summary links in canonical order."""
        return tuple((key, *value) for key, value in sorted(self.records.items()))

    @property
    def result_summary_identities(self) -> tuple[str, ...]:
        """Expose the observed worker-summary identities consumed by request hashing."""
        return tuple(sorted(item.identity for item in self.summaries.values()))

    def __call__(
        self, candidates: tuple[InterventionCandidate, ...]
    ) -> tuple[CandidateEvaluation, ...]:
        """Evaluate each toy request once, sharing its block-specific metric projections."""
        evaluations = []
        for candidate in candidates:
            requested = "toy-request::" + payload_sha256(
                {
                    "demand_lifetime": candidate.demand_lifetime.coordinates(),
                    "circularity": candidate.circularity.percentages(),
                    "target_year": candidate.target_year,
                }
            )
            if requested not in self.summaries:
                self.summaries[requested] = self._response(candidate, requested)
            summary = self.summaries[requested]
            self.records[candidate.identity] = (requested, summary.identity)
            evaluations.append(_candidate_evaluation(candidate, summary))
        return tuple(evaluations)

    def evaluate_competitive(
        self, candidates: tuple[InterventionCandidate, ...], distance_ceiling: float
    ) -> CompetitiveCandidateBatch:
        """Reject pruning outside this parent-terminal or wholly noncompliant toy domain."""
        raise AssertionError("The declared toy response domain cannot enter distance pruning.")

    def _response(
        self, candidate: InterventionCandidate, requested: str
    ) -> CandidateWorkerResultSummary:
        """Produce internally consistent toy IDs, distances, statuses and ranking summaries."""
        template = _summary_template()
        percentages = candidate.circularity.percentages()
        demand = candidate.demand_lifetime.demand_reduction_pct / 100
        lifetime = candidate.demand_lifetime.lifetime_extension_pct / 100
        collection = 0.0 if percentages is None else abs(percentages[0] / 100 - 0.5)
        treatment = (
            0.0
            if percentages is None
            else sum(abs(value / 100 - 0.25) for value in percentages[1:5]) / 2
        )
        feedstock = 0.0 if percentages is None else (100 - percentages[5]) / 100
        physical_identity = "toy-physical::" + requested
        distance = replace(
            template.intervention.distance_inputs,
            identity="",
            demand_lifetime_trajectory_identity="toy-demand::" + requested,
            circularity_trajectory_identity="toy-circularity::" + requested,
            target_year=candidate.target_year,
            demand_reduction_fraction=demand,
            lifetime_extension_fraction=lifetime,
            collection_distance=collection,
            treatment_share_distance=treatment,
            feedstock_share_distance=feedstock,
        )
        distance = replace(distance, identity=intervention_distance_inputs_identity(distance))
        admitted = (
            candidate.demand_lifetime.demand_reduction_pct in (0, 90)
            and candidate.demand_lifetime.lifetime_extension_pct in (0, 100)
            and (percentages is None or percentages == _toy_circularity_anchor())
        )
        diagnostic = (
            None
            if admitted
            else OptimizationPhysicalInfeasibility(
                code="toy_domain_rejection",
                equation="endpoint belongs to declared toy domain",
                observed=1.0,
                bound=0.0,
                tolerance=0.0,
                shortfall=1.0,
                status="toy_domain_rejection",
                count=1,
                coordinates=(("year", candidate.target_year),),
                details=(("scope", "synthetic codec fixture"),),
            )
        )
        worst = (
            0.5
            if self.parent_compliant
            else 3.0 - (demand + lifetime + collection + treatment + feedstock) / 4
        )
        compliant = worst <= 1.0
        member_metrics = []
        for block in _execution_blocks(("integrated_frontier",)) if admitted else ():
            if block == "demand_lifetime_frontier" and percentages is not None:
                continue
            if (
                block == "circularity_feedstock_anchor"
                and not candidate.demand_lifetime.uses_exact_parent
            ):
                continue
            projected = intervention_distance(
                physical_candidate_identity=physical_identity,
                intervention_authority_identity=requested,
                distance_inputs=distance,
                optimization_block=block,
            )
            metric = replace(
                template.member_metrics[0],
                identity="",
                physical_candidate_identity=physical_identity,
                intervention_authority_identity=requested,
                distance_inputs_identity=distance.identity,
                demand_lifetime_trajectory_identity=distance.demand_lifetime_trajectory_identity,
                circularity_trajectory_identity=distance.circularity_trajectory_identity,
                target_year=candidate.target_year,
                demand_reduction_fraction=demand,
                lifetime_extension_fraction=lifetime,
                collection_distance=collection,
                treatment_share_distance=treatment,
                feedstock_share_distance=feedstock,
                evaluation_identity="toy-evaluation::" + requested,
                metric_identity="toy-metric::" + requested + "::" + block,
                allocated_cc_context_identity="toy-allocated-capacity",
                comparison_scope_identity="toy-comparison-scope",
                optimization_item_identity="toy-state::" + block,
                optimization_block=block,
                is_zero_intervention_parent=candidate.is_exact_parent,
                intervention_distance_identity=projected.identity,
                intervention_distance=projected.total_distance,
                regions=("EUR",),
                all_region_joint_compliant=compliant,
                fully_compliant_regions=("EUR",) if compliant else (),
                fully_compliant_impact_targets=("annual::ef31::climate_change",)
                if compliant
                else (),
                compliant_region_count=int(compliant),
                fully_compliant_impact_target_count=int(compliant),
                compliant_region_impact_group_count=int(compliant),
                compliant_coordinate_count=int(compliant),
                region_count=1,
                impact_target_count=1,
                region_impact_group_count=1,
                coordinate_count=1,
                z_worst=worst,
                binding_slack=1.0 - worst if compliant else None,
                maximum_positive_exceedance=max(worst - 1.0, 0.0),
                balanced_mean_positive_exceedance=max(worst - 1.0, 0.0),
                binding_constraints=(
                    BindingConstraint("EUR", "annual", "ef31", "climate_change", 2050, worst),
                ),
            )
            member_metrics.append(
                replace(metric, identity=_candidate_metric_summary_identity(metric))
            )
        scores = template.scores
        assert scores is not None
        summary = replace(
            template,
            identity="",
            worker_result_identity="toy-worker-result::" + requested,
            requested_intervention_identity=requested,
            repair=DemandRepairOutcome(
                "linear_feasible" if admitted else "unique_fixed_path_infeasible",
                0.0,
                0.0,
                0,
                0,
                0,
                0,
                None,
                diagnostic,
                None,
            ),
            intervention=replace(
                template.intervention,
                identity=requested,
                demand_lifetime_trajectory_identity=distance.demand_lifetime_trajectory_identity,
                circularity_trajectory_identity=distance.circularity_trajectory_identity,
                distance_inputs=distance,
                uses_exact_parent=candidate.is_exact_parent,
            ),
            physical=replace(
                template.physical,
                physical_request_identity="toy-physical-request::" + requested,
                physical_candidate_identity=physical_identity,
                intervention_authority_identity=requested,
                status="physical_feasible" if admitted else "physical_infeasible",
                diagnostic=diagnostic,
                physical_state_identity="toy-state::" + requested if admitted else None,
                activation_identity="toy-activation::" + requested if admitted else None,
            ),
            scores=replace(
                scores,
                physical_candidate_identity=physical_identity,
                environmental_identity="toy-environment::" + requested,
                backgrounds=(template.member_metrics[0].background_scenario,),
                annual_scores_identity="toy-annual::" + requested,
                annual_climate_scores_identity="toy-annual-climate::" + requested,
                score_identity="toy-score::" + requested,
            )
            if admitted
            else None,
            feedstock_requirements_identity="toy-feedstock::" + requested if admitted else None,
            member_metrics=tuple(member_metrics) if admitted else (),
        )
        return replace(summary, identity="toy-summary::" + payload_sha256(asdict(summary)))


def make_completed_search(
    *,
    steps: tuple[int, ...] = (10, 5),
    requested_blocks: tuple[OptimizationBlock, ...] = ("integrated_frontier",),
    parent_compliant: bool = True,
) -> CompletedCandidateSearch:
    """Run the actual coarse/stage owners against a tiny explicit synthetic response domain."""
    oracle = CompletedSearchOracle(parent_compliant=parent_compliant)
    search = _run_candidate_search(
        request_identity="toy-request-search",
        requested_blocks=requested_blocks,
        evaluator=cast(CandidateWorkerEvaluator, oracle),
        candidate_input_identity="toy-input-authority",
        initial_grid_step_pct=100,
        endpoint_refinement_steps_pct=steps,
    )
    return CompletedCandidateSearch(
        initial_grid_step_pct=100,
        endpoint_refinement_steps_pct=steps,
        executed_blocks=_execution_blocks(requested_blocks),
        stages=search.completed_stages,
        candidate_records=oracle.candidate_records,
        worker_summaries=tuple(oracle.summaries.values()),
        task_records=tuple(
            CandidateTaskRecord(index, identity) for index, identity in enumerate(oracle.summaries)
        ),
    )
