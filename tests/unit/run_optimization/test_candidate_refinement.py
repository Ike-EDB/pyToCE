"""Endpoint-neighborhood checks with failed demand repairs."""

from dataclasses import replace

import pytest
from request_search_fixtures import _competitive_evaluator, _compliant_evaluation

from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    advance_candidate_search,
    finish_candidate_search,
    initialize_candidate_search,
    neighboring_candidates,
    refine_candidates,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)


@pytest.mark.parametrize("unresolved", (False, True))
@pytest.mark.parametrize("steps", ((10, 5), (5,), (10, 5, 1), (10,)))
def test_unresolved_endpoint_prevents_neighborhood_verification(
    unresolved: bool, steps: tuple[int, ...]
) -> None:
    """Require an admitted demand path for neighborhood completion."""
    seed = InterventionCandidate(
        "demand_lifetime_frontier", DemandLifetimeDecision(50, 50), CircularityDecision.parent()
    )
    unknown = replace(seed, demand_lifetime=DemandLifetimeDecision(50 - steps[-1], 50))

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Supply one admitted optimum and explicit rejected endpoint classifications."""
        return tuple(
            _compliant_evaluation(item)
            if item == seed
            else CandidateEvaluation(
                item,
                "physical::" + item.identity,
                (),
                0.5,
                repair_unresolved=unresolved and item == unknown,
            )
            for item in candidates
        )

    result = refine_candidates(
        optimization_block=seed.optimization_block,
        initial_candidates=(seed,),
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(evaluate, lambda item: 0.5),
        endpoint_refinement_steps_pct=steps,
    )
    assert result.selection.selected_candidate_identities == (seed.identity,)
    assert {item.step_pct for item in result.passes} == set(steps)
    assert all(not item.pruned_candidate_identities for item in result.passes if item.step_pct <= 5)
    assert set(result.final_neighbor_candidate_identities) == {
        item.identity for item in neighboring_candidates(seed, step_pct=steps[-1], coordinated=True)
    }
    assert result.state_verification[0].selection_stable
    assert result.state_verification[0].declared_neighborhood_verified is (not unresolved)
    assert result.unsuccessful_path_search_count == int(unresolved)


@pytest.mark.parametrize(
    ("block", "scalar", "coordinated"),
    (
        ("demand_lifetime_frontier", 4, 8),
        ("circularity_feedstock_anchor", 20, 20),
        ("integrated_frontier", 24, 108),
    ),
)
def test_outer_endpoint_neighbor_counts(block: str, scalar: int, coordinated: int) -> None:
    """The declared direct endpoint domains contain no environmental annual-path search."""
    demand = (
        DemandLifetimeDecision(0, 0)
        if block == "circularity_feedstock_anchor"
        else DemandLifetimeDecision(50, 50)
    )
    circularity = (
        CircularityDecision.parent()
        if block == "demand_lifetime_frontier"
        else CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25)
    )
    seed = InterventionCandidate(block, demand, circularity)
    assert len(neighboring_candidates(seed, step_pct=1)) == scalar
    assert len(neighboring_candidates(seed, step_pct=1, coordinated=True)) == coordinated


@pytest.mark.parametrize("collection", (0, 100))
def test_circularity_neighborhood_respects_closed_collection_domain(collection: int) -> None:
    """A bounded collection endpoint has exactly one collection-direction neighbor."""
    seed = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision(collection, 25, 25, 25, 25, 50, 25, 25),
    )
    neighbors = neighboring_candidates(seed, step_pct=1)
    changed = [item for item in neighbors if item.circularity.collection_pct != collection]
    assert len(changed) == 1
    assert changed[0].circularity.collection_pct == (1 if collection == 0 else 99)


def test_completed_initial_neighbors_are_not_evaluated_again() -> None:
    """Prior selected starts may already cover an entire competitive neighborhood."""
    seed = InterventionCandidate(
        "demand_lifetime_frontier", DemandLifetimeDecision(50, 50), CircularityDecision.parent()
    )
    initial = (seed, *neighboring_candidates(seed, step_pct=10, coordinated=True))
    observed: list[str] = []

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Keep the only admitted optimum and record unique numerical requests."""
        observed.extend(item.identity for item in candidates)
        return tuple(
            _compliant_evaluation(item)
            if item == seed
            else CandidateEvaluation(item, "physical::" + item.identity, (), 0.5)
            for item in candidates
        )

    result = refine_candidates(
        optimization_block=seed.optimization_block,
        initial_candidates=initial,
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(evaluate, lambda item: 0.5),
    )
    assert result.selection.selected_candidate_identities == (seed.identity,)
    assert len(observed) == len(set(observed))
    coarse_passes = [item for item in result.passes if item.step_pct == 10]
    assert len(coarse_passes) == 2
    assert all(not item.newly_evaluated_candidate_identities for item in coarse_passes)
    assert all(not item.pruned_candidate_identities for item in coarse_passes)


def test_coarse_selection_and_early_parent_preserve_truthful_stage_verification() -> None:
    """A compliant parent stops refinement and retains the current stage state."""
    seed = InterventionCandidate(
        "demand_lifetime_frontier", DemandLifetimeDecision(10, 10), CircularityDecision.parent()
    )
    parent = replace(seed, demand_lifetime=DemandLifetimeDecision(0, 0))
    observed: list[str] = []

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Admit the starting intervention and unchanged parent on this bounded fixture."""
        observed.extend(item.identity for item in candidates)
        return tuple(
            _compliant_evaluation(item)
            if item in (seed, parent)
            else CandidateEvaluation(item, "physical::" + item.identity, (), 0.5)
            for item in candidates
        )

    state = initialize_candidate_search(
        optimization_block=seed.optimization_block,
        initial_candidates=(seed,),
        evaluate=evaluate,
    )
    coarse = finish_candidate_search(state, endpoint_refinement_steps_pct=())
    assert coarse.selection.selected_candidate_identities == (seed.identity,)
    assert coarse.passes == ()
    assert coarse.final_neighbor_candidate_identities == ()
    assert not coarse.state_verification[0].selection_stable
    assert not coarse.state_verification[0].declared_neighborhood_verified

    stage = advance_candidate_search(
        state,
        step_pct=10,
        unpruned=False,
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(
            evaluate,
            lambda item: (
                (
                    item.demand_lifetime.demand_reduction_pct
                    + item.demand_lifetime.lifetime_extension_pct
                )
                / 200
            ),
        ),
    )
    terminal = finish_candidate_search(stage, endpoint_refinement_steps_pct=(10, 5))
    assert terminal.selection.selected_candidate_identities == (parent.identity,)
    assert terminal.selection.states[0].status == "baseline_compliant"
    assert terminal.passes[-1].coordinated
    assert {item.step_pct for item in terminal.passes} == {10}
    assert terminal.final_neighbor_candidate_identities == ()
    assert not terminal.state_verification[0].selection_stable
    assert not terminal.state_verification[0].declared_neighborhood_verified
    assert len(observed) == len(set(observed))
