"""Joint orchestration uses runtime single-block winners and preserves completed stages."""

import json
from dataclasses import replace
from itertools import product
from typing import cast

from completed_search_fixtures import CompletedSearchOracle, make_completed_search
from request_search_fixtures import _competitive_evaluator, _compliant_evaluation

from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    finish_candidate_search,
    select_candidate_states,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    _run_candidate_search,
)
from pytoce.run_optimization.runtime.completed_search import (
    decode_completed_search,
    encode_completed_search,
)


class ResourceLandscape:
    """Supply a small response surface with a shared resource constraint and synergy."""

    candidate_records: tuple[tuple[str, str, str], ...] = ()
    result_summary_identities: tuple[str, ...] = ()

    def __init__(self, *, optimum_demand: int = 50) -> None:
        """Track numerical requests through the package search."""
        self.observed: list[str] = []
        self.optimum_demand = optimum_demand

    def __call__(
        self, candidates: tuple[InterventionCandidate, ...]
    ) -> tuple[CandidateEvaluation, ...]:
        """Admit D+C<=100 and score a known joint improvement over either single block."""
        values = []
        for candidate in candidates:
            self.observed.append(candidate.identity)
            demand = candidate.demand_lifetime.demand_reduction_pct
            shares = candidate.circularity.percentages()
            collection = 0 if shares is None else shares[0]
            allowed = (
                candidate.demand_lifetime.lifetime_extension_pct == 0
                and demand + collection <= 100
                and (shares is None or shares[1:] == (100, 0, 0, 0, 100, 0, 0))
            )
            if not allowed:
                values.append(
                    CandidateEvaluation(candidate, "rejected::" + candidate.identity, (), 0.0)
                )
                continue
            synergy = min(
                demand * 50 / self.optimum_demand,
                collection * 50 / (100 - self.optimum_demand),
            )
            asr = 3 - (demand + collection + synergy) / 100
            distance = self.distance(candidate)
            base = _compliant_evaluation(candidate)
            metric = replace(
                base.state_metrics[0],
                state_identity="state::" + candidate.optimization_block,
                all_region_joint_compliant=False,
                fully_compliant_regions=(),
                fully_compliant_impact_targets=(),
                compliant_region_count=0,
                fully_compliant_impact_target_count=0,
                compliant_region_impact_group_count=0,
                compliant_coordinate_count=0,
                worst_asr=asr,
                binding_slack=None,
                maximum_positive_exceedance=asr - 1,
                balanced_mean_positive_exceedance=asr - 1,
                intervention_distance=distance,
                binding_coordinates=(("EUR", "annual", "ef31", "climate_change", 2050, asr),),
            )
            values.append(replace(base, state_metrics=(metric,), intervention_distance=distance))
        return tuple(values)

    @staticmethod
    def distance(candidate: InterventionCandidate) -> float:
        """Respect the different active-group weights while keeping scoring identical."""
        demand = candidate.demand_lifetime.demand_reduction_pct
        shares = candidate.circularity.percentages()
        collection = 0 if shares is None else shares[0]
        group_count = {
            "demand_lifetime_frontier": 2,
            "circularity_feedstock_anchor": 3,
            "integrated_frontier": 5,
        }[candidate.optimization_block]
        return (demand + collection) / (100 * group_count)

    def evaluate_competitive(self, candidates, distance_ceiling):
        """Use the existing admitted-distance protocol for unchanged I/II polling."""
        return _competitive_evaluator(self, self.distance)(candidates, distance_ceiling)


def test_joint_request_combines_stage_winners_and_preserves_single_block_search() -> None:
    """The infeasible I-by-II maximum yields a better admitted intermediate joint vector."""
    evaluator = ResourceLandscape()
    result = _run_candidate_search(
        request_identity="joint::fixture",
        requested_blocks=("integrated_frontier",),
        evaluator=cast(CandidateWorkerEvaluator, evaluator),
        candidate_input_identity="inputs::fixture",
        initial_grid_step_pct=100,
        endpoint_refinement_steps_pct=(10, 5),
    )
    assert len(result.block_results) == 1
    assert [item.step_pct for item in result.completed_stages] == [100, 10, 5]
    final = result.block_results[0]
    selected = final.selected_candidates[0]
    assert selected.demand_lifetime.demand_reduction_pct == 50
    assert selected.circularity.collection_pct == 50
    assert final.selected_states[0].worst_asr == 1.5
    assert final.refinement.state_verification[0].declared_neighborhood_verified
    for block in ("demand_lifetime_frontier", "circularity_feedstock_anchor"):
        independent = _run_candidate_search(
            request_identity="single::" + block,
            requested_blocks=(block,),
            evaluator=cast(CandidateWorkerEvaluator, ResourceLandscape()),
            candidate_input_identity="inputs::fixture",
            initial_grid_step_pct=100,
            endpoint_refinement_steps_pct=(10, 5),
        )
        hidden = next(
            item
            for item in result.completed_stages[-1].single_blocks
            if item.optimization_block == block
        )
        assert (
            finish_candidate_search(hidden, endpoint_refinement_steps_pct=(10, 5))
            == independent.block_results[0].refinement
        )
        assert (
            final.selected_states[0].worst_asr
            < independent.block_results[0].selected_states[0].worst_asr
        )
    for stage in result.completed_stages:
        assert {item.optimization_block for item in stage.single_blocks} == {
            "demand_lifetime_frontier",
            "circularity_feedstock_anchor",
        }
        assert stage.integrated is not None
        assert stage.integrated.completed_stages[-1].design.actual_ii_candidates
    assert len(evaluator.observed) == len(set(evaluator.observed))


def test_joint_search_matches_closed_small_domain_exhaustive_reference() -> None:
    """Known resource synergy is recovered against every endpoint in a closed test domain.

    Initial spacing 100 and local spacing 50 produce demand levels 0, 40, 50,
    and 90: 40 is the permitted step down from the 90-percent boundary. Other
    scalar levels and shares remain multiples of 50. This synthetic evaluator
    exposes algorithm behavior under the declared synthetic inputs.
    """
    levels = (0, 50, 100)
    treatment = tuple(values for values in product(levels, repeat=4) if sum(values) == 100)
    feedstock = tuple(values for values in product(levels, repeat=3) if sum(values) == 100)
    requests = []
    for demand, lifetime in product((0, 40, 50, 90), levels):
        decision = DemandLifetimeDecision(demand, lifetime, 2050)
        requests.append(
            InterventionCandidate(
                "integrated_frontier", decision, CircularityDecision.parent(target_year=2050)
            )
        )
        for collection, treatments, feedstocks in product(levels, treatment, feedstock):
            requests.append(
                InterventionCandidate(
                    "integrated_frontier",
                    decision,
                    CircularityDecision(
                        collection,
                        treatments[0],
                        treatments[1],
                        treatments[2],
                        treatments[3],
                        feedstocks[0],
                        feedstocks[1],
                        feedstocks[2],
                        2050,
                    ),
                )
            )
    assert len(requests) == 2172
    reference = ResourceLandscape()(tuple(requests))
    assert sum(bool(item.state_metrics) for item in reference) == 12
    exhaustive = select_candidate_states(reference)
    result = _run_candidate_search(
        request_identity="closed-domain::fixture",
        requested_blocks=("integrated_frontier",),
        evaluator=cast(CandidateWorkerEvaluator, ResourceLandscape()),
        candidate_input_identity="inputs::fixture",
        initial_grid_step_pct=100,
        endpoint_refinement_steps_pct=(50,),
    )
    actual = result.block_results[0]
    assert actual.refinement.selection == exhaustive
    assert actual.selected_candidates[0].demand_lifetime.demand_reduction_pct == 50
    assert actual.selected_candidates[0].circularity.collection_pct == 50
    final = result.completed_stages[-1].integrated
    assert final is not None
    assert select_candidate_states(final.evaluations) == exhaustive
    alternatives = []
    for single in result.completed_stages[-1].single_blocks:
        selected_ids = set(single.selection.selected_candidate_identities)
        alternatives.extend(
            replace(item.candidate, optimization_block="integrated_frontier")
            for item in single.evaluations
            if item.candidate.identity in selected_ids
        )
    assert (
        select_candidate_states(ResourceLandscape()((*actual.selected_candidates, *alternatives)))
        == exhaustive
    )
    closed_identities = {item.identity for item in requests}
    assert {item.candidate.identity for item in final.evaluations} <= closed_identities
    assert actual.refinement.state_verification[0].declared_neighborhood_verified
    assert set(actual.refinement.final_neighbor_candidate_identities) <= {
        item.candidate.identity for item in final.evaluations
    }


def test_completed_joint_stages_append_without_replaying_earlier_proposals() -> None:
    """The request owner retains prior branches and passes when adding a finer stage."""
    # This deliberately synthetic optimum is off the five-point grid.
    evaluator = ResourceLandscape(optimum_demand=52)
    arguments = dict(
        requested_blocks=("integrated_frontier",),
        evaluator=cast(CandidateWorkerEvaluator, evaluator),
        candidate_input_identity="inputs::fixture",
        initial_grid_step_pct=100,
    )
    prefix = _run_candidate_search(
        request_identity="coarse", endpoint_refinement_steps_pct=(10, 5), **arguments
    )
    prior_calls = set(evaluator.observed)
    evaluator.observed.clear()
    extension = _run_candidate_search(
        request_identity="finer",
        endpoint_refinement_steps_pct=(10, 5, 1),
        continuation_parent_identity="completed::five",
        prior_stages=prefix.completed_stages,
        **arguments,
    )
    assert extension.completed_stages[:3] == prefix.completed_stages
    assert [item.step_pct for item in extension.completed_stages] == [100, 10, 5, 1]
    assert evaluator.observed
    assert not prior_calls.intersection(evaluator.observed)
    selected = extension.block_results[0].selected_candidates[0]
    assert selected.demand_lifetime.demand_reduction_pct == 52
    assert selected.circularity.collection_pct == 48
    assert (
        extension.block_results[0].selected_states[0].worst_asr
        < prefix.block_results[0].selected_states[0].worst_asr
    )
    assert (
        extension.block_results[0].selected_states[0].maximum_positive_exceedance
        < prefix.block_results[0].selected_states[0].maximum_positive_exceedance
    )
    assert extension.continuation_parent_identity == "completed::five"


def test_decoded_nonterminal_stages_match_uninterrupted_extension() -> None:
    """Serialization preserves every live branch and obligation before new-stage proposals."""
    completed = make_completed_search(parent_compliant=False)
    restored = decode_completed_search(json.loads(json.dumps(encode_completed_search(completed))))
    oracle = CompletedSearchOracle(parent_compliant=False)
    oracle.summaries = {
        value.requested_intervention_identity: value for value in restored.worker_summaries
    }
    oracle.records = {
        identity: (requested, summary)
        for identity, requested, summary in restored.candidate_records
    }
    extension = _run_candidate_search(
        request_identity="decoded-extension",
        requested_blocks=("integrated_frontier",),
        evaluator=cast(CandidateWorkerEvaluator, oracle),
        candidate_input_identity="toy-input-authority",
        initial_grid_step_pct=100,
        endpoint_refinement_steps_pct=(10, 5, 4),
        prior_stages=restored.stages,
        continuation_parent_identity="completed::five",
    )
    uninterrupted = make_completed_search(steps=(10, 5, 4), parent_compliant=False)
    assert extension.completed_stages[:3] == restored.stages
    assert extension.completed_stages == uninterrupted.stages
    assert oracle.candidate_records == uninterrupted.candidate_records
    assert tuple(oracle.summaries.values()) == uninterrupted.worker_summaries
