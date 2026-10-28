"""Finite line geometry and ranking valleys at the admitted evaluator boundary."""

from collections import Counter
from dataclasses import replace

import pytest
from request_search_fixtures import _compliant_evaluation, _unexpected_competitive

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    candidate_coordinate_key,
    line_candidates,
    neighboring_candidates,
    refine_candidates,
)
from pytoce.run_optimization.calculation.integrated_search import _poll_batches
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)


@pytest.mark.parametrize("step", (7, 16, 50))
@pytest.mark.parametrize("block", ("circularity_feedstock_anchor", "integrated_frontier"))
def test_complete_lines_match_bounded_scalar_and_exact_partition_domains(
    step: int, block: OptimizationBlock
) -> None:
    """Compare line vectors with exhaustive one-axis/one-transfer algebraic requirements."""
    candidate = InterventionCandidate(
        block,
        DemandLifetimeDecision(23, 38)
        if block == "integrated_frontier"
        else DemandLifetimeDecision(0, 0),
        CircularityDecision(19, 17, 23, 29, 31, 37, 41, 22),
    )
    base = (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        *candidate.circularity.percentages(),  # type: ignore[misc]
    )
    expected = set()
    scalar_bounds = ((0, 90), (1, 100), (2, 100)) if block == "integrated_frontier" else ((2, 100),)
    for axis, bound in scalar_bounds:
        for value in range(bound + 1):
            if value != base[axis] and (value in (0, bound) or (value - base[axis]) % step == 0):
                coordinates = list(base)
                coordinates[axis] = value
                expected.add(tuple(coordinates))
    for partition in (range(3, 7), range(7, 10)):
        for donor in partition:
            for recipient in partition:
                if donor == recipient:
                    continue
                for amount in range(1, base[donor] + 1):
                    if amount % step == 0 or amount == base[donor]:
                        coordinates = list(base)
                        coordinates[donor] -= amount
                        coordinates[recipient] += amount
                        expected.add(tuple(coordinates))
    neighbors = line_candidates(candidate, step_pct=step)
    actual = {
        (
            item.demand_lifetime.demand_reduction_pct,
            item.demand_lifetime.lifetime_extension_pct,
            *item.circularity.percentages(),  # type: ignore[misc]
        )
        for item in neighbors
    }
    assert actual == expected
    assert len(neighbors) == len(actual)
    assert tuple(sorted(neighbors, key=candidate_coordinate_key)) == neighbors
    assert all(sum(row[3:7]) == sum(row[7:]) == 100 for row in actual)
    assert all(item.target_year == candidate.target_year for item in neighbors)


def test_parent_lines_do_not_invent_shares_or_expand_coordinated_geometry() -> None:
    """Long scalar moves add no Cartesian products to existing one-step joint moves."""
    parent = InterventionCandidate(
        "integrated_frontier", DemandLifetimeDecision(23, 38), CircularityDecision.parent()
    )
    lines = line_candidates(parent, step_pct=16)
    assert all(item.circularity.uses_exact_parent for item in lines)
    explicit = replace(parent, circularity=CircularityDecision(19, 17, 23, 29, 31, 37, 41, 22))
    scalar = line_candidates(explicit, step_pct=16)
    batches = _poll_batches(explicit, guides=(), step_pct=16, coordinated=True)
    observed = {item for _, batch in batches for item in batch}
    expected = set(neighboring_candidates(explicit, step_pct=16, coordinated=True)) - set(scalar)
    assert observed == expected
    assert all(
        abs(item.demand_lifetime.demand_reduction_pct - 23) <= 16
        and abs(item.demand_lifetime.lifetime_extension_pct - 38) <= 16
        for item in observed
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_block_ii_lines_cross_worse_rejected_and_unresolved_intermediates(reverse: bool) -> None:
    """A farther improvement remains eligible after all three distinct intermediate outcomes."""
    seed = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision(100 if reverse else 0, 100, 0, 0, 0, 100, 0, 0),
    )
    scores = {100 if reverse else 0: 2.0, 80 if reverse else 20: 5.0, 40 if reverse else 60: 1.0}
    unresolved_collection = 70 if reverse else 30
    calls: Counter[str] = Counter()

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Score the fixed-partition collection line through the package search."""
        observations = []
        for candidate in candidates:
            calls[candidate.identity] += 1
            collection = candidate.circularity.collection_pct
            fixed = (
                candidate.circularity.treatment_percentages
                == seed.circularity.treatment_percentages
                and candidate.circularity.feedstock_percentages
                == seed.circularity.feedstock_percentages
            )
            if fixed and collection in scores:
                base = _compliant_evaluation(candidate)
                metric = replace(
                    base.state_metrics[0],
                    all_region_joint_compliant=False,
                    fully_compliant_regions=(),
                    fully_compliant_impact_targets=(),
                    compliant_region_count=0,
                    fully_compliant_impact_target_count=0,
                    compliant_region_impact_group_count=0,
                    compliant_coordinate_count=0,
                    worst_asr=1.0 + scores[collection],
                    binding_slack=None,
                    maximum_positive_exceedance=scores[collection],
                    balanced_mean_positive_exceedance=scores[collection],
                )
                observations.append(replace(base, state_metrics=(metric,)))
            else:
                observations.append(
                    CandidateEvaluation(
                        candidate,
                        "rejected",
                        (),
                        0.5,
                        repair_unresolved=fixed and collection == unresolved_collection,
                    )
                )
        return tuple(reversed(observations)) if reverse else tuple(observations)

    report = refine_candidates(
        optimization_block=seed.optimization_block,
        initial_candidates=(seed,),
        evaluate=evaluate,
        evaluate_competitive=_unexpected_competitive,
        endpoint_refinement_steps_pct=(10,),
    )
    winner = replace(
        seed, circularity=replace(seed.circularity, collection_pct=40 if reverse else 60)
    )
    assert report.selection.selected_candidate_identities == (winner.identity,)
    assert set(report.final_neighbor_candidate_identities) == {
        item.identity for item in line_candidates(winner, step_pct=10)
    }
    assert report.unsuccessful_path_search_count == 1
    assert report.state_verification[0].selection_stable
    assert not report.state_verification[0].declared_neighborhood_verified
    assert set(calls.values()) == {1}
