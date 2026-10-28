"""Append work to completed endpoint stages and reuse prior evaluations."""

import json
from collections.abc import Callable
from dataclasses import asdict, replace

import pytest
from request_search_fixtures import _competitive_evaluator, _compliant_evaluation

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    advance_candidate_search,
    finish_candidate_search,
    initialize_candidate_search,
    refine_candidates,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)


def _distance(candidate: InterventionCandidate) -> float:
    """Use a nonnegative synthetic intervention metric for the owner-level landscape."""
    shares = candidate.circularity.percentages()
    return (
        candidate.demand_lifetime.demand_reduction_pct
        + candidate.demand_lifetime.lifetime_extension_pct
        + (shares[0] if shares else 0)
    ) / 300


def _two_state_landscape(
    observed: list[str],
) -> Callable[[tuple[InterventionCandidate, ...]], tuple[CandidateEvaluation, ...]]:
    """Return two small optima missed at five points but reachable at one point."""

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Score supplied endpoints and expose every new numerical request to the test."""
        observed.extend(item.identity for item in candidates)
        values = []
        for candidate in candidates:
            base = _compliant_evaluation(candidate)
            metrics = []
            for state, demand, lifetime, collection in (("a", 13, 17, 43), ("b", 32, 7, 67)):
                if candidate.optimization_block == "demand_lifetime_frontier":
                    error = abs(candidate.demand_lifetime.demand_reduction_pct - demand) + abs(
                        candidate.demand_lifetime.lifetime_extension_pct - lifetime
                    )
                else:
                    shares = candidate.circularity.percentages()
                    target = (collection, 20, 30, 30, 20, 40, 30, 30)
                    error = sum(
                        abs(actual - wanted)
                        for actual, wanted in zip(shares or (0,) * 8, target, strict=True)
                    )
                compliant = error == 0
                asr = 0.9 if compliant else 1 + error / 100
                metrics.append(
                    replace(
                        base.state_metrics[0],
                        state_identity=state,
                        allocated_cc_case=state,
                        intervention_distance=_distance(candidate),
                        all_region_joint_compliant=compliant,
                        fully_compliant_regions=("EUR",) if compliant else (),
                        fully_compliant_impact_targets=("annual::ef31::climate_change",)
                        if compliant
                        else (),
                        compliant_region_count=int(compliant),
                        fully_compliant_impact_target_count=int(compliant),
                        compliant_region_impact_group_count=int(compliant),
                        compliant_coordinate_count=int(compliant),
                        worst_asr=asr,
                        binding_slack=0.1 if compliant else None,
                        maximum_positive_exceedance=error / 100,
                        balanced_mean_positive_exceedance=error / 100,
                        binding_coordinates=(
                            ("EUR", "annual", "ef31", "climate_change", 2050, asr),
                        ),
                    )
                )
            values.append(
                replace(
                    base, state_metrics=tuple(metrics), intervention_distance=_distance(candidate)
                )
            )
        return tuple(values)

    return evaluate


@pytest.mark.parametrize("block", ("demand_lifetime_frontier", "circularity_feedstock_anchor"))
def test_completed_five_point_state_appends_one_point_without_replay(
    block: OptimizationBlock,
) -> None:
    """Appending preserves the five-point state and matches an uninterrupted full schedule."""
    parent = InterventionCandidate(
        block, DemandLifetimeDecision(0, 0), CircularityDecision.parent()
    )
    seed = InterventionCandidate(
        block,
        DemandLifetimeDecision(20, 20)
        if block == "demand_lifetime_frontier"
        else DemandLifetimeDecision(0, 0),
        CircularityDecision.parent()
        if block == "demand_lifetime_frontier"
        else CircularityDecision(50, 20, 30, 30, 20, 40, 30, 30),
    )
    calls: list[str] = []
    evaluate = _two_state_landscape(calls)
    competitive = _competitive_evaluator(evaluate, _distance)
    state = initialize_candidate_search(
        optimization_block=block, initial_candidates=(parent, seed), evaluate=evaluate
    )
    for step in (10, 5):
        state = advance_candidate_search(
            state,
            step_pct=step,
            unpruned=step <= 5,
            evaluate=evaluate,
            evaluate_competitive=competitive,
        )
    prefix_bytes = json.dumps(asdict(state), sort_keys=True, default=str)
    prefix_calls = tuple(calls)
    prefix_result = finish_candidate_search(state, endpoint_refinement_steps_pct=(10, 5))
    assert all(item.status == "selected_noncompliant" for item in prefix_result.selection.states)
    assert all(item.declared_neighborhood_verified for item in prefix_result.state_verification)
    assert [(item.step_pct, item.unpruned) for item in state.completed_stages] == [
        (10, False),
        (5, True),
    ]

    extended = advance_candidate_search(
        state,
        step_pct=1,
        unpruned=True,
        evaluate=evaluate,
        evaluate_competitive=competitive,
    )
    actual = finish_candidate_search(extended, endpoint_refinement_steps_pct=(10, 5, 1))
    assert json.dumps(asdict(state), sort_keys=True, default=str) == prefix_bytes
    assert extended.passes[: len(state.passes)] == state.passes
    assert tuple(calls[: len(prefix_calls)]) == prefix_calls
    assert len(calls) > len(prefix_calls)
    assert len(calls) == len(set(calls))
    assert all(item.status == "intervention_compliant" for item in actual.selection.states)
    assert all(item.declared_neighborhood_verified for item in actual.state_verification)
    selected = {
        item.candidate.identity: item.candidate
        for item in extended.evaluations
        if item.candidate.identity in actual.selection.selected_candidate_identities
    }
    if block == "demand_lifetime_frontier":
        assert {
            (item.demand_lifetime.demand_reduction_pct, item.demand_lifetime.lifetime_extension_pct)
            for item in selected.values()
        } == {(13, 17), (32, 7)}
    else:
        assert {item.circularity.collection_pct for item in selected.values()} == {43, 67}

    fresh_calls: list[str] = []
    fresh_evaluate = _two_state_landscape(fresh_calls)
    fresh = refine_candidates(
        optimization_block=block,
        initial_candidates=(parent, seed),
        evaluate=fresh_evaluate,
        evaluate_competitive=_competitive_evaluator(fresh_evaluate, _distance),
        endpoint_refinement_steps_pct=(10, 5, 1),
    )
    assert actual == fresh
    assert calls == fresh_calls
