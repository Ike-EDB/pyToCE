from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from regional_metric_fixtures import H12_REGIONS, YEARS, _inputs
from request_search_fixtures import _competitive_evaluator, _unexpected_competitive

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CandidateStateMetrics,
    CompetitiveCandidateBatch,
    InterventionCandidate,
    circularity_feedstock_domain,
    circularity_feedstock_pairwise_design,
    demand_lifetime_initial_design,
    neighboring_candidates,
    pair_requirements,
    percentage_compositions,
    refine_candidates,
    select_candidate_states,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import endpoint_neighbors
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    AnnualCategory,
    calculate_regional_metrics,
    intervention_distance,
)


def _reported(candidate: InterventionCandidate) -> tuple[int, ...]:
    return (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        *candidate.circularity.percentages(),  # type: ignore[misc]
    )


def _metric(
    candidate: InterventionCandidate,
    *,
    distance: float,
    worst_asr: float,
    region_count: int = 12,
    impact_count: int = 17,
    group_count: int = 204,
    coordinate_count: int = 15_564,
    compliant_regions: int | None = None,
    compliant_impacts: int | None = None,
    compliant_groups: int | None = None,
    compliant_coordinates: int | None = None,
    balanced_exceedance: float | None = None,
    state_identity: str = "state::ndc::p50",
    background_scenario: str = "remind_ssp2_ndc",
    allocated_cc_case: str = "p50",
) -> CandidateStateMetrics:
    compliant = worst_asr <= 1.0
    n_region = region_count if compliant_regions is None and compliant else compliant_regions or 0
    n_impact = impact_count if compliant_impacts is None and compliant else compliant_impacts or 0
    n_group = group_count if compliant_groups is None and compliant else compliant_groups or 0
    n_coordinate = (
        coordinate_count
        if compliant_coordinates is None and compliant
        else compliant_coordinates or 0
    )
    return CandidateStateMetrics(
        state_identity=state_identity,
        candidate_identity=candidate.identity,
        physical_candidate_identity=f"physical::{candidate.identity}",
        optimization_block=candidate.optimization_block,
        background_scenario=background_scenario,
        allocated_cc_case=allocated_cc_case,
        is_zero_intervention_parent=candidate.is_exact_parent,
        intervention_distance=distance,
        all_region_joint_compliant=compliant,
        fully_compliant_regions=tuple(f"region-{index}" for index in range(n_region)),
        fully_compliant_impact_targets=tuple(f"target-{index}" for index in range(n_impact)),
        compliant_region_count=n_region,
        fully_compliant_impact_target_count=n_impact,
        compliant_region_impact_group_count=n_group,
        compliant_coordinate_count=n_coordinate,
        region_count=region_count,
        impact_target_count=impact_count,
        region_impact_group_count=group_count,
        coordinate_count=coordinate_count,
        worst_asr=worst_asr,
        binding_slack=1.0 - worst_asr if compliant else None,
        maximum_positive_exceedance=max(worst_asr - 1.0, 0.0),
        balanced_mean_positive_exceedance=(
            max(worst_asr - 1.0, 0.0) if balanced_exceedance is None else balanced_exceedance
        ),
        binding_coordinates=(("region-0", "annual", "ef31", "climate_change", 2050, worst_asr),),
    )


def _evaluation(
    candidate: InterventionCandidate,
    *,
    distance: float,
    worst_asr: float,
    **coverage: float,
) -> CandidateEvaluation:
    metric = _metric(
        candidate,
        distance=distance,
        worst_asr=worst_asr,
        **coverage,  # type: ignore[arg-type]
    )
    return CandidateEvaluation(
        candidate=candidate,
        physical_candidate_identity=metric.physical_candidate_identity,
        state_metrics=(metric,),
        intervention_distance=distance,
    )


def _expected_final_reference_neighborhood(candidate: InterventionCandidate) -> set[str]:
    """Require the declared final poll and exact linear references for its endpoint pairs."""
    neighbors = neighboring_candidates(candidate, step_pct=5, coordinated=True)
    references = {
        InterventionCandidate(
            item.optimization_block,
            DemandLifetimeDecision(
                item.demand_lifetime.demand_reduction_pct,
                item.demand_lifetime.lifetime_extension_pct,
            ),
            item.circularity,
        ).identity
        for item in neighbors
    }
    return {item.identity for item in neighbors} | references


def test_coarse_domains_and_pairwise_designs_are_exact() -> None:
    demand_lifetime = demand_lifetime_initial_design()
    assert len(demand_lifetime) == 36
    assert {item.demand_lifetime for item in demand_lifetime} == {
        DemandLifetimeDecision(demand, lifetime)
        for demand in (0, 20, 40, 60, 80, 90)
        for lifetime in (0, 20, 40, 60, 80, 100)
    }
    assert len(percentage_compositions(component_count=4, step_pct=20)) == 56
    assert len(percentage_compositions(component_count=3, step_pct=20)) == 21

    circularity_domain = circularity_feedstock_domain()
    circularity_design = circularity_feedstock_pairwise_design()
    domain_rows = tuple(item.circularity.percentages() for item in circularity_domain)
    design_rows = tuple(item.circularity.percentages() for item in circularity_design)
    assert len(circularity_domain) == 7_056
    assert len(set(domain_rows)) == len(domain_rows)
    for row in domain_rows:
        assert row is not None
        assert all(value in (0, 20, 40, 60, 80, 100) for value in row)
        assert sum(row[1:5]) == 100
        assert sum(row[5:]) == 100
    assert len(circularity_design) == 132
    assert len(pair_requirements(domain_rows)) == 873  # type: ignore[arg-type]
    assert pair_requirements(design_rows) == pair_requirements(domain_rows)  # type: ignore[arg-type]


def test_integrated_neighbors_preserve_scalar_and_coordinated_moves() -> None:
    """Combine every valid circularity move with each single-endpoint move."""
    candidate = InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(50, 50),
        CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25),
    )
    neighbors = neighboring_candidates(candidate, step_pct=10)
    assert len(neighbors) == 4 + 2 + 12 + 6
    changes: Counter[int] = Counter()
    base = _reported(candidate)
    for neighbor in neighbors:
        changed = [
            index
            for index, values in enumerate(zip(base, _reported(neighbor), strict=True))
            if values[0] != values[1]
        ]
        changes[len(changed)] += 1
        if len(changed) == 2:
            delta = tuple(_reported(neighbor)[index] - base[index] for index in changed)
            assert sorted(delta) == [-10, 10]
    assert changes == {1: 6, 2: 18}

    coordinated = neighboring_candidates(candidate, step_pct=10, coordinated=True)
    assert set(neighbors).issubset(coordinated)
    # Enumerate nearby percentage vectors by their partition constraints,
    # independently of the owner's donor/recipient transfer generator.
    circularity_vectors: set[tuple[int, ...]] = {
        (collection, 25, 25, 25, 25, 50, 25, 25) for collection in (40, 60)
    }
    circularity_vectors.update(
        (50, *shares, 50, 25, 25)
        for shares in product((15, 25, 35), repeat=4)
        if sum(shares) == 100 and sum(value != 25 for value in shares) == 2
    )
    circularity_vectors.update(
        (50, 25, 25, 25, 25, *shares)
        for shares in product((40, 50, 60), (15, 25, 35), (15, 25, 35))
        if sum(shares) == 100
        and sum(value != original for value, original in zip(shares, (50, 25, 25), strict=True))
        == 2
    )
    expected_joint = {
        (DemandLifetimeDecision(demand, lifetime), CircularityDecision(*shares))
        for demand, lifetime in ((40, 50), (60, 50), (50, 40), (50, 60))
        for shares in circularity_vectors
    }
    observed_joint = {
        (neighbor.demand_lifetime, neighbor.circularity)
        for neighbor in coordinated
        if neighbor.demand_lifetime != candidate.demand_lifetime
        and neighbor.circularity != candidate.circularity
    }
    assert len(expected_joint) == 4 * (2 + 12 + 6)
    assert observed_joint == expected_joint


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("demand_lifetime_frontier", ((0, 1), (1, 0))),
        ("integrated_frontier", ((0, 1), (1, 0))),
        ("circularity_feedstock_anchor", ()),
    ],
)
def test_exact_parent_preserves_active_demand_lifetime_neighbors(
    block: OptimizationBlock, expected: tuple[tuple[int, int], ...]
) -> None:
    """Generate neighbors for zero demand and lifetime controls."""
    parent = InterventionCandidate(
        block,
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    neighbors = neighboring_candidates(parent, step_pct=1)
    assert (
        tuple(
            (item.demand_lifetime.demand_reduction_pct, item.demand_lifetime.lifetime_extension_pct)
            for item in neighbors
        )
        == expected
    )
    assert all(item.circularity == parent.circularity for item in neighbors)


@pytest.mark.parametrize("block", ["demand_lifetime_frontier", "integrated_frontier"])
@pytest.mark.parametrize("best_demand", [0, 5])
def test_noncompliant_parent_refines_or_verifies_its_actual_neighbors(
    block: OptimizationBlock,
    best_demand: int,
) -> None:
    """Evaluate neighbors before retaining the parent or selecting a nearby improvement."""
    initial = tuple(
        InterventionCandidate(block, item.demand_lifetime, item.circularity)
        for item in demand_lifetime_initial_design()
    )
    calls: Counter[str] = Counter()
    observed_candidates: dict[str, InterventionCandidate] = {}

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Supply responses that retain the noncompliant parent or favor a nearby candidate."""
        evaluations = []
        for candidate in batch:
            calls[candidate.identity] += 1
            observed_candidates[candidate.identity] = candidate
            demand = candidate.demand_lifetime.demand_reduction_pct
            lifetime = candidate.demand_lifetime.lifetime_extension_pct
            error = (demand - best_demand) ** 2 + lifetime**2
            evaluations.append(
                _evaluation(
                    candidate,
                    distance=(demand / 90 + lifetime / 100) / 2,
                    worst_asr=(
                        2 + error / 100
                        if best_demand == 0
                        else 0.9
                        if error == 0
                        else 1 + error / 100
                    ),
                )
            )
        return tuple(evaluations)

    result = refine_candidates(
        optimization_block=block,
        initial_candidates=initial,
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(
            evaluate,
            lambda candidate: (
                (
                    candidate.demand_lifetime.demand_reduction_pct / 90
                    + candidate.demand_lifetime.lifetime_extension_pct / 100
                )
                / 2
            ),
        ),
    )
    selected = observed_candidates[result.selection.selected_candidate_identities[0]]
    assert selected.demand_lifetime.demand_reduction_pct == best_demand
    assert selected.demand_lifetime.lifetime_extension_pct == 0
    assert result.selection.states[0].status == (
        "selected_noncompliant" if best_demand == 0 else "intervention_compliant"
    )
    assert all(count == 1 for count in calls.values())
    assert all(identity in calls for identity in result.final_neighbor_candidate_identities)
    assert set(
        result.final_neighbor_candidate_identities
    ) == _expected_final_reference_neighborhood(selected)
    assert result.state_verification[0].declared_neighborhood_verified == (
        block == "demand_lifetime_frontier"
    )
    assert result.state_verification[0].selection_stable


def test_feasible_selection_uses_distance_then_slack_then_identity() -> None:
    candidates = demand_lifetime_initial_design()[1:4]
    evaluations = (
        _evaluation(candidates[0], distance=0.2, worst_asr=1.0),
        _evaluation(candidates[1], distance=0.1, worst_asr=0.7),
        _evaluation(candidates[2], distance=0.1, worst_asr=0.9),
    )
    selection = select_candidate_states(evaluations)
    assert selection.states[0].candidate_identity == candidates[2].identity
    assert selection.states[0].status == "intervention_compliant"


@pytest.mark.parametrize(
    "tier",
    range(8),
    ids=("regions", "targets", "groups", "coordinates", "maximum", "mean", "distance", "identity"),
)
def test_noncompliant_selection_uses_all_coverage_and_severity_tiers(tier: int) -> None:
    """Each ranking criterion defeats a contrary later preference with consistent ASRs."""
    candidates = tuple(
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(value, value),
            CircularityDecision.parent(),
        )
        for value in (20, 10)
    )
    if tier == 6:
        # Choose real candidate identities that oppose their intervention distances.
        domain = demand_lifetime_initial_design()[1:]
        candidates = next(
            (left, right)
            for left, right in product(domain, repeat=2)
            if sum(
                (
                    left.demand_lifetime.demand_reduction_pct,
                    left.demand_lifetime.lifetime_extension_pct,
                )
            )
            < sum(
                (
                    right.demand_lifetime.demand_reduction_pct,
                    right.demand_lifetime.lifetime_extension_pct,
                )
            )
            and left.identity > right.identity
        )
    elif tier == 7:
        candidates = tuple(
            sorted(
                (
                    replace(candidates[0], demand_lifetime=DemandLifetimeDecision(20, 0)),
                    replace(candidates[1], demand_lifetime=DemandLifetimeDecision(0, 20)),
                ),
                key=lambda item: item.identity,
            )
        )
    annual = np.full((2, 12, 82, 1), 0.5, dtype=np.float64)
    climate = np.full((2, 12, 82), 0.5, dtype=np.float64)
    year = YEARS.index(2050)
    annual[:, 0, year, 0] = 2.0
    # At cumulative allocated carrying capacity 100, one value 159.5 plus 81 values 0.5 gives ASR 2.
    if tier == 0:
        climate[0, 0, year] = 159.5
        annual[1, 1, year, 0] = 2.0
    elif tier == 1:
        annual[0, :2, year, 0] = 3.0
        climate[1, 1, year] = 159.5
    elif tier == 2:
        annual[0, 0, 1:, 0] = 3.0
        climate[0, 1, year] = 159.5
        annual[1, 1, year, 0] = 2.0
        climate[1, 0, year] = 159.5
    elif tier == 3:
        annual[0, 0, year, 0] = 3.0
        annual[1, 0, year + 1, 0] = 2.0
    elif tier == 4:
        annual[0, 0, year + 1, 0] = 2.0
        annual[1, 0, year : year + 2, 0] = (2.5, 1.125)
    elif tier == 5:
        annual[0, 0, year + 1, 0] = 1.125
        annual[1, 0, year + 1, 0] = 2.0
    evaluations = []
    ranks = []
    for index, candidate in enumerate(candidates):
        distance = intervention_distance(
            physical_candidate_identity=candidate.identity,
            intervention_authority_identity=candidate.identity,
            distance_inputs=_inputs(
                demand=candidate.demand_lifetime.demand_reduction_pct / 100,
                lifetime=candidate.demand_lifetime.lifetime_extension_pct / 100,
            ),
            optimization_block=candidate.optimization_block,
        )
        metrics = calculate_regional_metrics(
            evaluation_identity=candidate.identity,
            background_scenario="remind_ssp2_ndc",
            allocated_cc_case="p50",
            allocated_cc_context_identity="capacity::ranking",
            physical_candidate_identity=candidate.identity,
            intervention_authority_identity=candidate.identity,
            optimization_item_identity="state::ranking",
            optimization_block=candidate.optimization_block,
            is_zero_intervention_parent=False,
            intervention_distance=distance,
            optimization_lcia_method_registry_identity="annual::ranking",
            cumulative_climate_method_registry_identity="climate::ranking",
            comparison_scope_identity="scope::ranking",
            annual_scores=annual[index],
            annual_allocated_cc=np.ones_like(annual[index]),
            annual_budget_compatible_gwp100_scores=climate[index],
            cumulative_gwp100_allocated_cc=np.full(12, 100.0),
            regions=H12_REGIONS,
            years=YEARS,
            annual_categories=(AnnualCategory("ef31", "acidification"),),
        )
        state = CandidateStateMetrics(
            state_identity=metrics.optimization_item_identity,
            candidate_identity=candidate.identity,
            physical_candidate_identity=metrics.physical_candidate_identity,
            optimization_block=candidate.optimization_block,
            background_scenario=metrics.background_scenario,
            allocated_cc_case=metrics.allocated_cc_case,
            is_zero_intervention_parent=False,
            intervention_distance=distance.total_distance,
            all_region_joint_compliant=metrics.all_region_joint_compliant,
            fully_compliant_regions=metrics.fully_compliant_regions,
            fully_compliant_impact_targets=metrics.fully_compliant_impact_targets,
            compliant_region_count=metrics.compliant_region_count,
            fully_compliant_impact_target_count=metrics.fully_compliant_impact_target_count,
            compliant_region_impact_group_count=metrics.compliant_region_impact_group_count,
            compliant_coordinate_count=metrics.compliant_coordinate_count,
            region_count=metrics.region_count,
            impact_target_count=metrics.impact_target_count,
            region_impact_group_count=metrics.region_impact_group_count,
            coordinate_count=metrics.coordinate_count,
            worst_asr=metrics.z_worst,
            binding_slack=metrics.binding_slack,
            maximum_positive_exceedance=metrics.maximum_positive_exceedance,
            balanced_mean_positive_exceedance=metrics.balanced_mean_positive_exceedance,
            binding_coordinates=tuple(
                (
                    item.region,
                    item.constraint_type,
                    item.optimization_lcia_method,
                    item.impact_category_id,
                    item.year,
                    item.asr,
                )
                for item in metrics.binding_constraints
            ),
        )
        evaluations.append(
            CandidateEvaluation(candidate, candidate.identity, (state,), distance.total_distance)
        )
        ranks.append(
            (
                -metrics.compliant_region_count,
                -metrics.fully_compliant_impact_target_count,
                -metrics.compliant_region_impact_group_count,
                -metrics.compliant_coordinate_count,
                metrics.maximum_positive_exceedance,
                metrics.balanced_mean_positive_exceedance,
                distance.total_distance,
                candidate.identity,
            )
        )
    assert ranks[0][:tier] == ranks[1][:tier]
    assert ranks[0][tier] < ranks[1][tier]
    if tier < 7:
        assert ranks[0][tier + 1 :] > ranks[1][tier + 1 :]
    for ordered in (evaluations, list(reversed(evaluations))):
        selection = select_candidate_states(ordered)
        assert selection.states[0].candidate_identity == candidates[0].identity
        assert selection.states[0].status == "selected_noncompliant"


def test_refinement_closes_five_point_neighbors_without_claiming_a_finer_optimum() -> None:
    """A stable five-point endpoint can miss an intervening compliant optimum."""
    initial = (
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(20, 20),
            CircularityDecision.parent(),
        ),
    )
    calls: Counter[str] = Counter()
    observed_candidates: dict[str, InterventionCandidate] = {}

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        results = []
        for candidate in candidates:
            calls[candidate.identity] += 1
            observed_candidates[candidate.identity] = candidate
            error = abs(candidate.demand_lifetime.demand_reduction_pct - 12) + abs(
                candidate.demand_lifetime.lifetime_extension_pct - 7
            )
            distance = (
                candidate.demand_lifetime.demand_reduction_pct
                + candidate.demand_lifetime.lifetime_extension_pct
            ) / 200.0
            results.append(
                _evaluation(
                    candidate,
                    distance=distance,
                    worst_asr=0.9 if error == 0 else 1.0 + error,
                )
            )
        return tuple(results)

    result = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=initial,
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(
            evaluate,
            lambda candidate: (
                (
                    candidate.demand_lifetime.demand_reduction_pct
                    + candidate.demand_lifetime.lifetime_extension_pct
                )
                / 200.0
            ),
        ),
    )
    assert result.selection.selected_candidate_identities[0] in calls
    selected = next(
        candidate
        for candidate in observed_candidates.values()
        if candidate.identity == result.selection.selected_candidate_identities[0]
    )
    assert selected.demand_lifetime.demand_reduction_pct == 10
    assert selected.demand_lifetime.lifetime_extension_pct == 5
    assert result.selection.states[0].status == "selected_noncompliant"
    assert {item.step_pct for item in result.passes} == {10, 5}
    assert result.state_verification[0].declared_neighborhood_verified
    assert result.state_verification[0].selection_stable
    assert result.final_neighbor_candidate_identities
    assert all(count == 1 for count in calls.values())


def test_coordinated_search_crosses_a_scalar_stall_and_verifies_the_final_anchor() -> None:
    """A synthetic diagonal valley requires a joint move; scalar intermediates are worse."""
    initial = InterventionCandidate(
        "demand_lifetime_frontier", DemandLifetimeDecision(50, 50), CircularityDecision.parent()
    )
    observed: dict[str, InterventionCandidate] = {}

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Give the diagonal a better noncompliance ranking in the synthetic response domain."""
        results = []
        for candidate in batch:
            observed[candidate.identity] = candidate
            demand = candidate.demand_lifetime.demand_reduction_pct
            lifetime = candidate.demand_lifetime.lifetime_extension_pct
            error = abs(demand - 30) + abs(lifetime - 30) + 100 * abs(demand - lifetime)
            results.append(
                _evaluation(candidate, distance=(demand + lifetime) / 200, worst_asr=2 + error)
            )
        return tuple(results)

    result = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=(initial,),
        evaluate=evaluate,
        evaluate_competitive=_unexpected_competitive,
    )
    selected = observed[result.selection.selected_candidate_identities[0]]
    assert (
        selected.demand_lifetime.demand_reduction_pct,
        selected.demand_lifetime.lifetime_extension_pct,
    ) == (30, 30)
    assert result.passes[0].selected_candidate_identities == (initial.identity,) or all(
        observed[identity].demand_lifetime.demand_reduction_pct == 50
        and observed[identity].demand_lifetime.lifetime_extension_pct == 50
        for identity in result.passes[0].selected_candidate_identities
    )
    assert any(
        item.coordinated and item.selected_candidate_identities != item.anchor_candidate_identities
        for item in result.passes
    )
    assert result.passes[-1].coordinated
    assert result.state_verification[0].declared_neighborhood_verified
    assert set(
        result.final_neighbor_candidate_identities
    ) == _expected_final_reference_neighborhood(selected)


def test_competitive_refinement_keeps_every_five_percent_neighbor() -> None:
    """Final five-point verification retains worse-distance alternatives too."""
    initial = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(50, 50),
        CircularityDecision.parent(),
    )
    neighbors = endpoint_neighbors(initial.demand_lifetime, step_pct=10)
    equal_distance_winner = replace(
        initial, demand_lifetime=next(item for item in neighbors if item.demand_reduction_pct == 40)
    )
    strictly_worse = replace(
        initial, demand_lifetime=next(item for item in neighbors if item.demand_reduction_pct == 60)
    )

    def distance(candidate: InterventionCandidate) -> float:
        if candidate.identity in {initial.identity, equal_distance_winner.identity}:
            return 0.2
        if candidate.identity == strictly_worse.identity:
            return 0.3
        return 0.4

    def worst_asr(candidate: InterventionCandidate) -> float:
        return 0.99 if candidate == equal_distance_winner else 0.9

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        return tuple(
            _evaluation(
                candidate,
                distance=distance(candidate),
                worst_asr=worst_asr(candidate),
            )
            for candidate in candidates
        )

    def evaluate_competitive(
        candidates: tuple[InterventionCandidate, ...],
        distance_ceiling: float,
    ) -> CompetitiveCandidateBatch:
        distances = tuple((candidate.identity, distance(candidate)) for candidate in candidates)
        retained = tuple(
            candidate for candidate in candidates if distance(candidate) <= distance_ceiling
        )
        return CompetitiveCandidateBatch(distances, evaluate(retained))

    filtered = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=(initial,),
        evaluate=evaluate,
        evaluate_competitive=evaluate_competitive,
    )

    expected = select_candidate_states(evaluate((initial, equal_distance_winner, strictly_worse)))
    assert filtered.selection == expected
    assert filtered.selection.selected_candidate_identities == (equal_distance_winner.identity,)
    assert filtered.state_verification[0].declared_neighborhood_verified
    assert filtered.state_verification[0].selection_stable
    first_pass = filtered.passes[0]
    assert equal_distance_winner.identity in first_pass.newly_evaluated_candidate_identities
    assert equal_distance_winner.identity not in first_pass.pruned_candidate_identities
    assert strictly_worse.identity in first_pass.pruned_candidate_identities
    assert first_pass.selected_candidate_identities == (equal_distance_winner.identity,)
    expected_final_neighbors = _expected_final_reference_neighborhood(equal_distance_winner)
    assert set(filtered.final_neighbor_candidate_identities) == expected_final_neighbors
    assert all(
        not item.pruned_candidate_identities for item in filtered.passes if item.step_pct == 5
    )


def test_mixed_noncompliant_states_keep_the_neighborhood_unpruned() -> None:
    initial = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(50, 50),
        CircularityDecision.parent(),
    )
    competitive_calls = 0

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        outputs: list[CandidateEvaluation] = []
        for candidate in candidates:
            distance = 0.1 if candidate == initial else 0.2
            metrics = (
                _metric(
                    candidate,
                    distance=distance,
                    worst_asr=0.9,
                    state_identity="state::ndc::p50",
                ),
                _metric(
                    candidate,
                    distance=distance,
                    worst_asr=1.2,
                    state_identity="state::npi::p50",
                    background_scenario="remind_ssp2_npi",
                ),
            )
            outputs.append(
                CandidateEvaluation(
                    candidate=candidate,
                    physical_candidate_identity=f"physical::{candidate.identity}",
                    state_metrics=metrics,
                    intervention_distance=distance,
                )
            )
        return tuple(outputs)

    def evaluate_competitive(
        candidates: tuple[InterventionCandidate, ...],
        distance_ceiling: float,
    ) -> CompetitiveCandidateBatch:
        nonlocal competitive_calls
        competitive_calls += 1
        return CompetitiveCandidateBatch((), ())

    result = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=(initial,),
        evaluate=evaluate,
        evaluate_competitive=evaluate_competitive,
    )

    assert competitive_calls == 0
    assert {item.status for item in result.selection.states} == {
        "intervention_compliant",
        "selected_noncompliant",
    }
    assert all(not item.pruned_candidate_identities for item in result.passes)
    assert set(
        result.final_neighbor_candidate_identities
    ) == _expected_final_reference_neighborhood(initial)


@pytest.mark.parametrize(
    "block", ["demand_lifetime_frontier", "circularity_feedstock_anchor", "integrated_frontier"]
)
def test_compliant_parent_is_terminal_without_neighbor_work(block: OptimizationBlock) -> None:
    parent = InterventionCandidate(
        block,
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    calls: Counter[str] = Counter()

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        for candidate in batch:
            calls[candidate.identity] += 1
        return tuple(_evaluation(item, distance=0.0, worst_asr=0.8) for item in batch)

    result = refine_candidates(
        optimization_block=block,
        initial_candidates=(parent,),
        evaluate=evaluate,
        evaluate_competitive=_unexpected_competitive,
    )
    assert calls == {parent.identity: 1}
    assert result.selection.states[0].status == "baseline_compliant"
    assert result.passes == ()
    assert result.state_verification[0].selected_is_exact_parent
    assert not result.state_verification[0].declared_neighborhood_verified


@pytest.mark.parametrize("block", ["circularity_feedstock_anchor", "integrated_frontier"])
@pytest.mark.parametrize("parent_asr", (0.8, 1.1))
def test_exact_parent_precedes_compliant_paths_with_equal_endpoints(
    block: OptimizationBlock, parent_asr: float
) -> None:
    """Retain a compliant varying parent even when an active path has smaller slack."""
    parent = InterventionCandidate(
        block, DemandLifetimeDecision(0, 0), CircularityDecision.parent()
    )
    active = InterventionCandidate(
        block, DemandLifetimeDecision(0, 0), CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25)
    )
    # A parent reaching these shares in 2050 has zero endpoint distance from
    # this supported active target despite different earlier annual shares.
    # The trajectory owner's separate test establishes that distinction.
    evaluations = (
        _evaluation(parent, distance=0.0, worst_asr=parent_asr),
        _evaluation(active, distance=0.0, worst_asr=0.9),
    )
    expected = parent if parent_asr <= 1 else active
    for ordered in (evaluations, tuple(reversed(evaluations))):
        selected = select_candidate_states(ordered)
        assert selected.selected_candidate_identities == (expected.identity,)
        assert selected.states[0].status == (
            "baseline_compliant" if expected is parent else "intervention_compliant"
        )


@pytest.mark.parametrize("block", ["circularity_feedstock_anchor", "integrated_frontier"])
def test_compliant_parent_stops_refinement_with_zero_distance_competition(
    block: OptimizationBlock,
) -> None:
    """Do no neighborhood work after selecting the unchanged compliant parent."""
    parent = InterventionCandidate(
        block, DemandLifetimeDecision(0, 0), CircularityDecision.parent()
    )
    active = InterventionCandidate(
        block, DemandLifetimeDecision(0, 0), CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25)
    )
    calls: list[str] = []

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Return the two bounded compliant metrics and record actual search requests."""
        calls.extend(item.identity for item in batch)
        return tuple(
            _evaluation(item, distance=0.0, worst_asr=0.8 if item.is_exact_parent else 0.9)
            for item in batch
        )

    result = refine_candidates(
        optimization_block=block,
        initial_candidates=(parent, active),
        evaluate=evaluate,
        evaluate_competitive=_unexpected_competitive,
    )
    assert Counter(calls) == {parent.identity: 1}
    assert result.evaluation_count == 1
    assert result.reused_candidate_count == 0
    assert result.selection.states[0].status == "baseline_compliant"
    assert result.passes == ()
    assert result.state_verification[0].selected_is_exact_parent
    assert not result.state_verification[0].declared_neighborhood_verified


@pytest.mark.parametrize("all_states_compliant", [True, False])
def test_terminal_parent_skips_full_design_only_when_every_state_is_compliant(
    all_states_compliant: bool,
) -> None:
    """Retain all 36 starts when even one parent state requires intervention."""
    design = demand_lifetime_initial_design()
    assert len(design) == 36
    parent = next(item for item in design if item.is_exact_parent)
    calls: Counter[str] = Counter()

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Supply one compliant case and one independently selected case."""
        outputs = []
        for candidate in batch:
            calls[candidate.identity] += 1
            metrics = tuple(
                _metric(
                    candidate,
                    distance=0.0 if candidate.is_exact_parent else 1.0,
                    worst_asr=0.8 if case == "generous" or all_states_compliant else 1.2,
                    state_identity=f"state::ndc::{case}",
                    allocated_cc_case=case,
                )
                for case in ("generous", "direct")
            )
            outputs.append(
                CandidateEvaluation(
                    candidate,
                    f"physical::{candidate.identity}",
                    metrics,
                    metrics[0].intervention_distance,
                )
            )
        return tuple(outputs)

    result = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=design,
        evaluate=evaluate,
        evaluate_competitive=_unexpected_competitive,
    )
    assert set(calls.values()) == {1}
    assert result.evaluation_count == len(calls)
    if all_states_compliant:
        assert calls == {parent.identity: 1}
        assert result.reused_candidate_count == 0
        assert result.passes == ()
        assert result.final_neighbor_candidate_identities == ()
        assert all(state.status == "baseline_compliant" for state in result.selection.states)
        assert all(not state.declared_neighborhood_verified for state in result.state_verification)
    else:
        assert {item.identity for item in design}.issubset(calls)
        assert result.passes
        assert {state.status for state in result.selection.states} == {
            "baseline_compliant",
            "selected_noncompliant",
        }
        assert set(result.final_neighbor_candidate_identities) == (
            _expected_final_reference_neighborhood(parent)
        )


def test_noncompliant_baseline_and_integrated_baseline_are_not_locally_verified() -> None:
    parent = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    parent_result = refine_candidates(
        optimization_block="circularity_feedstock_anchor",
        initial_candidates=(parent,),
        evaluate=lambda batch: tuple(
            _evaluation(item, distance=0.0, worst_asr=1.2) for item in batch
        ),
        evaluate_competitive=_unexpected_competitive,
    )
    assert parent_result.selection.states[0].status == "selected_noncompliant"
    assert not parent_result.state_verification[0].declared_neighborhood_verified

    integrated = InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(50, 50),
        CircularityDecision.parent(),
    )

    def distance(candidate: InterventionCandidate) -> float:
        """Return the fixture's fixed equal-control intervention distance."""
        return (
            candidate.demand_lifetime.demand_reduction_pct
            + candidate.demand_lifetime.lifetime_extension_pct
        ) / 500.0

    def evaluate(batch: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Retain a compliant baseline-circularity anchor."""
        return tuple(
            _evaluation(
                item,
                distance=distance(item),
                worst_asr=(
                    0.8
                    if (
                        item.demand_lifetime.demand_reduction_pct,
                        item.demand_lifetime.lifetime_extension_pct,
                    )
                    == (50, 50)
                    else 1.2
                ),
            )
            for item in batch
        )

    integrated_result = refine_candidates(
        optimization_block="integrated_frontier",
        initial_candidates=(integrated,),
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(evaluate, distance),
    )
    proof = integrated_result.state_verification[0]
    assert proof.selected_has_non_grid_baseline_component
    assert proof.selection_stable
    assert not proof.declared_neighborhood_verified


def test_alternative_coarse_grid_covers_its_feasible_pairs() -> None:
    """A user-selected 50-point design retains the demand cap and all feasible pairs."""
    demand = demand_lifetime_initial_design(step_pct=50)
    assert {item.demand_lifetime for item in demand} == {
        DemandLifetimeDecision(d, lifetime) for d in (0, 50, 90) for lifetime in (0, 50, 100)
    }
    domain = circularity_feedstock_domain(step_pct=50)
    design = circularity_feedstock_pairwise_design(step_pct=50)
    assert len(domain) == 180
    domain_rows = tuple(item.circularity.percentages() for item in domain)
    design_rows = tuple(item.circularity.percentages() for item in design)
    assert pair_requirements(design_rows) == pair_requirements(domain_rows)  # type: ignore[arg-type]
