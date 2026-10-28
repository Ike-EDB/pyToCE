"""Deterministic candidate evaluations shared by search owner tests."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CandidateStateMetrics,
    CompetitiveCandidateBatch,
    InterventionCandidate,
)
from pytoce.run_optimization.calculation.candidate_worker import CandidateWorkerPlanner


def _planner(identity: str = "planner") -> CandidateWorkerPlanner:
    """Return the minimal planner identity consumed by request tests."""

    return cast(CandidateWorkerPlanner, SimpleNamespace(identity=identity))


def _compliant_evaluation(candidate: InterventionCandidate) -> CandidateEvaluation:
    """Return one compliant environmental state for a candidate."""

    distance = (
        candidate.demand_lifetime.demand_reduction_pct
        + candidate.demand_lifetime.lifetime_extension_pct
    ) / 200.0
    metric = CandidateStateMetrics(
        state_identity="state::ndc::p50",
        candidate_identity=candidate.identity,
        physical_candidate_identity=f"physical::{candidate.identity}",
        optimization_block=candidate.optimization_block,
        background_scenario="remind_ssp2_ndc",
        allocated_cc_case="p50",
        is_zero_intervention_parent=candidate.is_exact_parent,
        intervention_distance=distance,
        all_region_joint_compliant=True,
        fully_compliant_regions=("EUR",),
        fully_compliant_impact_targets=("annual::ef31::climate_change",),
        compliant_region_count=1,
        fully_compliant_impact_target_count=1,
        compliant_region_impact_group_count=1,
        compliant_coordinate_count=1,
        region_count=1,
        impact_target_count=1,
        region_impact_group_count=1,
        coordinate_count=1,
        worst_asr=0.9,
        binding_slack=0.1,
        maximum_positive_exceedance=0.0,
        balanced_mean_positive_exceedance=0.0,
        binding_coordinates=(("EUR", "annual", "ef31", "climate_change", 2050, 0.9),),
    )
    return CandidateEvaluation(
        candidate=candidate,
        physical_candidate_identity=metric.physical_candidate_identity,
        state_metrics=(metric,),
        intervention_distance=distance,
    )


def _competitive_evaluator(
    evaluate: Callable[[tuple[InterventionCandidate, ...]], tuple[CandidateEvaluation, ...]],
    distance: Callable[[InterventionCandidate], float],
) -> Callable[[tuple[InterventionCandidate, ...], float], CompetitiveCandidateBatch]:
    """Apply the worker's distance-filter protocol to cheap synthetic search responses."""

    def evaluate_competitive(
        candidates: tuple[InterventionCandidate, ...], distance_ceiling: float
    ) -> CompetitiveCandidateBatch:
        """Return every supplied distance and evaluate only competitive fixture candidates."""
        distances = tuple((candidate.identity, distance(candidate)) for candidate in candidates)
        retained = tuple(
            candidate
            for candidate, (_identity, value) in zip(candidates, distances, strict=True)
            if value <= distance_ceiling
        )
        return CompetitiveCandidateBatch(distances, evaluate(retained))

    return evaluate_competitive


def _unexpected_competitive(
    candidates: tuple[InterventionCandidate, ...], distance_ceiling: float
) -> CompetitiveCandidateBatch:
    """Reject pruning in a test with only noncompliant or terminal baseline selections."""
    raise AssertionError("This selection must not prune candidate evaluation")
