"""Algebraic coverage and finite route contracts for the integrated endpoint search."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations, product
from math import ceil

import pytest

from pytoce.run_optimization.calculation.candidate_search import InterventionCandidate
from pytoce.run_optimization.calculation.integrated_design import (
    BASE_PROPOSAL_FAMILIES,
    ROUTE_PROPOSAL_FAMILIES,
    build_joint_stage_design,
    complete_joint_stage_design,
    explicit_endpoint_coordinates,
    guided_joint_candidate,
    joint_pair_cover,
    move_joint_coordinates,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)


def _candidate(
    demand: int,
    lifetime: int,
    circularity: tuple[int, int, int, int, int, int, int, int] | None = None,
) -> InterventionCandidate:
    """Construct the requested endpoint."""
    return InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(demand, lifetime),
        CircularityDecision.parent() if circularity is None else CircularityDecision(*circularity),
    )


def _coordinates(candidate: InterventionCandidate) -> tuple[int, int, tuple[int, ...] | None]:
    """Read categorical controls for comparisons independent of candidate hashes."""
    return (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        candidate.circularity.percentages(),
    )


@pytest.mark.parametrize("step", (10, 20, 25, 50, 100))
def test_implicit_joint_design_covers_every_compatible_pair(step: int) -> None:
    """Independent simplex arithmetic defines complete pair requirements on each public grid."""
    cover = joint_pair_cover(initial_grid_step_pct=step)
    demand = tuple(dict.fromkeys((*range(0, 91, step), 90)))
    levels = tuple(range(0, 101, step))
    domains = (demand, *((levels,) * 9))
    requirements = set()
    for left, right in combinations(range(10), 2):
        same_partition = (3 <= left < right <= 6) or (7 <= left < right <= 9)
        for a, b in product(domains[left], domains[right]):
            if not same_partition or a + b <= 100:
                requirements.add((left, a, right, b))
    observed = {
        (left, row[left], right, row[right])
        for row in cover.rows
        for left, right in combinations(range(10), 2)
    }
    assert observed == requirements
    assert cover.requirement_count == cover.covered_requirement_count == len(requirements)
    assert len(cover.rows) == len(set(cover.rows))
    assert all(sum(row[3:7]) == sum(row[7:10]) == 100 for row in cover.rows)
    assert all(row[index] in domains[index] for row in cover.rows for index in range(10))
    assert {row[0] for row in cover.rows} == set(demand)
    assert cover == joint_pair_cover(initial_grid_step_pct=step)


@pytest.mark.parametrize("step", (20, 7, 1))
def test_joint_moves_bound_scalar_and_total_partition_transfers(step: int) -> None:
    """Reach both exact boundaries for every requested spacing."""
    source = (90, 0, 100, 45, 35, 20, 0, 60, 25, 15)
    target = (0, 100, 0, 0, 0, 5, 95, 0, 10, 90)
    current = source
    expected_transitions = max(
        *(ceil(abs(a - b) / step) for a, b in zip(source[:3], target[:3], strict=True)),
        ceil(sum(max(b - a, 0) for a, b in zip(source[3:7], target[3:7], strict=True)) / step),
        ceil(sum(max(b - a, 0) for a, b in zip(source[7:], target[7:], strict=True)) / step),
    )
    for _ in range(expected_transitions):
        moved = move_joint_coordinates(current, target, step_pct=step)
        assert moved != current
        assert all(
            min(a, b) <= c <= max(a, b) for a, b, c in zip(current, target, moved, strict=True)
        )
        assert all(abs(a - b) <= step for a, b in zip(current[:3], moved[:3], strict=True))
        for part in (slice(3, 7), slice(7, 10)):
            assert sum(moved[part]) == 100
            assert (
                sum(max(b - a, 0) for a, b in zip(current[part], moved[part], strict=True)) <= step
            )
        current = moved
    assert current == target
    assert move_joint_coordinates(target, source, step_pct=step) != target
    assert move_joint_coordinates(target, target, step_pct=step) == target


def test_stage_base_uses_runtime_anchors_and_exact_six_family_memberships() -> None:
    """All coarse crosses and J+ switches precede rank-selected routes and local carry."""
    first = _candidate(37, 23)
    second = _candidate(0, 0, (63, 17, 28, 31, 24, 61, 23, 16))
    cover = joint_pair_cover(initial_grid_step_pct=100)
    design = build_joint_stage_design(
        block_i_candidates=(first,),
        block_ii_candidates=(second,),
        pair_cover=cover,
        step_pct=7,
    )
    coordinates = {_coordinates(item): item for item in design.candidates}
    origins = dict(design.proposal_origins)
    assert (37, 23, None) in coordinates
    assert design.actual_i_pairs == ((37, 23),)
    assert design.routes == design.geometric_guide_candidate_identities == ()
    assert set(design.starting_candidate_identities) == {
        item.identity for item in design.candidates
    }
    assert {family for families in origins.values() for family in families} == set(
        BASE_PROPOSAL_FAMILIES
    )
    assert origins[coordinates[(37, 23, second.circularity.percentages())].identity] == (
        "actual_i_by_represented_circularity",
        "direct_combination",
    )
    for pair in product((0, 90), (0, 100)):
        assert origins[coordinates[(*pair, None)].identity] == ("parent_circularity",)
        assert (
            "coarse_demand_lifetime_by_ii"
            in origins[coordinates[(*pair, second.circularity.percentages())].identity]
        )
    for row in cover.rows:
        for demand, lifetime in ((row[0], row[1]), (0, 0)):
            assert (
                "independent_pair_cover"
                in origins[coordinates[(demand, lifetime, row[2:])].identity]
            )
        assert (
            "actual_i_by_represented_circularity"
            in origins[coordinates[(37, 23, row[2:])].identity]
        )
    assert set(design.source_candidate_identities) == {
        coordinates[(demand, lifetime, second.circularity.percentages())].identity
        for demand, lifetime in ((0, 0), (37, 23))
    }
    assert design.actual_ii_candidates == (coordinates[_coordinates(second)],)
    assert not design.explicit_ii_routes_absent
    assert len({id(item.demand_lifetime) for item in design.candidates}) == len(
        {key[:2] for key in coordinates}
    )


def test_complete_routes_preserve_hybrid_reversals_all_families_and_endpoint_origins() -> None:
    """Projected segments sample every bounded transition in both exact orientations."""
    second = _candidate(0, 0, (63, 17, 28, 31, 24, 61, 23, 16))
    base = build_joint_stage_design(
        block_i_candidates=(_candidate(37, 23),),
        block_ii_candidates=(second,),
        pair_cover=joint_pair_cover(initial_grid_step_pct=20),
        step_pct=7,
    )
    guide = next(
        item
        for item in base.candidates
        if _coordinates(item)[:2] == (0, 100) and not item.circularity.uses_exact_parent
    )
    design = complete_joint_stage_design(base, guides=(guide, second, guide))
    rows = {_coordinates(item): item for item in design.candidates}
    origins = dict(design.proposal_origins)
    routes = {(route.source, route.target): route for route in design.routes}
    assert len(routes) == len(design.routes)
    assert {family for route in design.routes for _, family in route.origins} == set(
        ROUTE_PROPOSAL_FAMILIES
    )
    assert set(design.geometric_guide_candidate_identities) == {guide.identity, second.identity}
    assert design.starting_candidate_identities == base.starting_candidate_identities
    for route in design.routes:
        assert route.source != route.target
        assert (route.target, route.source) in routes
        assert route.points[-1] == route.target
        assert route.step_pct == 20
        assert route.identity != replace(route, step_pct=10).identity
        previous = route.source
        for point in route.points:
            assert point == move_joint_coordinates(previous, route.target, step_pct=20)
            previous = point
        for point in (route.source, *route.points):
            assert {family for _, family in route.origins} <= set(
                origins[rows[(point[0], point[1], point[2:])].identity]
            )
    source = (37, 23, *explicit_endpoint_coordinates(second)[2:])
    hybrid = (37, 23, *explicit_endpoint_coordinates(guide)[2:])
    assert (hybrid, source) in routes
    assert (explicit_endpoint_coordinates(guide), source) in routes
    assert design == complete_joint_stage_design(base, guides=(second, guide))


def test_circularity_boundaries_and_own_ii_axes_do_not_depend_on_admitted_guides() -> None:
    """Missing conditional guides cannot suppress II-start axes or circularity compromises."""
    second = _candidate(0, 0, (63, 17, 28, 31, 24, 61, 23, 16))
    base = build_joint_stage_design(
        block_i_candidates=(_candidate(37, 23),),
        block_ii_candidates=(second,),
        pair_cover=joint_pair_cover(initial_grid_step_pct=20),
        step_pct=5,
    )
    completed = complete_joint_stage_design(base, guides=())
    routes = {(route.source, route.target): route for route in completed.routes}
    source = explicit_endpoint_coordinates(second)
    for family, pair in (("demand", (37, 0)), ("lifetime", (0, 23)), ("demand_lifetime", (37, 23))):
        target = (*pair, *source[2:])
        route = routes[source, target]
        assert family in {origin[1] for origin in route.origins}
        assert routes[target, source].points[-1] == source
        assert len(route.points) == 2
    for scalar in (0, 100):
        target = (*source[:2], scalar, *source[3:])
        assert (source, target) in routes
    for partition in (range(3, 7), range(7, 10)):
        for donor, recipient in product(partition, repeat=2):
            if donor == recipient:
                continue
            target = list(source)
            target[recipient] += target[donor]
            target[donor] = 0
            assert (source, tuple(target)) in routes
    coincident = complete_joint_stage_design(base, guides=(second,))
    assert set(routes) <= {(route.source, route.target) for route in coincident.routes}


def test_route_reuse_keeps_initial_spacing_and_adds_only_new_segments() -> None:
    """Finer local stages preserve completed coarse routes and cumulative family origins."""
    second = _candidate(0, 0, (63, 17, 28, 31, 24, 61, 23, 16))
    cover = joint_pair_cover(initial_grid_step_pct=50)
    base = build_joint_stage_design(
        block_i_candidates=(_candidate(37, 23),),
        block_ii_candidates=(second,),
        pair_cover=cover,
        step_pct=20,
    )
    complete = complete_joint_stage_design(base, guides=(second,))
    fine_base = build_joint_stage_design(
        block_i_candidates=(_candidate(37, 23),),
        block_ii_candidates=(second,),
        pair_cover=cover,
        step_pct=7,
    )
    fine = complete_joint_stage_design(fine_base, guides=(second,), previous_routes=complete.routes)
    assert fine.identity != complete.identity
    assert fine.routes == complete.routes
    assert fine.candidates == complete.candidates
    updated_base = build_joint_stage_design(
        block_i_candidates=(_candidate(37, 23), _candidate(11, 49)),
        block_ii_candidates=(second,),
        pair_cover=cover,
        step_pct=5,
    )
    updated = complete_joint_stage_design(updated_base, guides=(), previous_routes=fine.routes)
    by_segment = {(route.source, route.target): route for route in updated.routes}
    assert len(updated.routes) > len(fine.routes)
    for route in fine.routes:
        retained = by_segment[route.source, route.target]
        assert retained.points == route.points
        assert retained.identity == route.identity
        assert set(route.origins) <= set(retained.origins)
    assert {route.step_pct for route in updated.routes} == {50}
    assert updated.geometric_guide_candidate_identities == ()


def test_parent_only_ii_keeps_explicit_design_and_duplicate_family_memberships() -> None:
    """Retain origins for categorical group-II combinations."""
    cover = joint_pair_cover(initial_grid_step_pct=100)
    base = build_joint_stage_design(
        block_i_candidates=(_candidate(17, 29),),
        block_ii_candidates=(_candidate(0, 0),),
        pair_cover=cover,
        step_pct=20,
    )
    guide = next(item for item in base.candidates if not item.circularity.uses_exact_parent)
    design = complete_joint_stage_design(base, guides=(guide,))
    assert design.explicit_ii_routes_absent
    assert design.routes == design.source_candidate_identities == ()
    assert design.actual_ii_candidates[0].is_exact_parent
    coordinates = {_coordinates(item): item for item in design.candidates}
    origins = dict(design.proposal_origins)
    assert origins[coordinates[(17, 29, None)].identity] == (
        "direct_combination",
        "parent_circularity",
    )
    assert "coarse_demand_lifetime_by_ii" in origins[coordinates[(0, 0, None)].identity]
    assert all((row[0], row[1], row[2:]) in coordinates for row in cover.rows)
    assert all((17, 29, row[2:]) in coordinates for row in cover.rows)


def test_noop_projections_add_no_changed_control_family() -> None:
    """A coincident guide contributes no demand/lifetime route origins to its source."""
    second = _candidate(0, 0, (0, 100, 0, 0, 0, 100, 0, 0))
    base = build_joint_stage_design(
        block_i_candidates=(_candidate(0, 0),),
        block_ii_candidates=(second,),
        pair_cover=joint_pair_cover(initial_grid_step_pct=100),
        step_pct=20,
    )
    completed = complete_joint_stage_design(base, guides=(second,))
    assert {family for route in completed.routes for _, family in route.origins} == {"circularity"}
    assert set(dict(completed.proposal_origins)[second.identity]) & set(
        ROUTE_PROPOSAL_FAMILIES
    ) == {"circularity"}


def test_stage_order_and_duplicate_anchors_do_not_change_requests_or_identity() -> None:
    """Equivalent runtime selection order and repeated state winners share one base design."""
    first = (_candidate(19, 33), _candidate(27, 41))
    second = (_candidate(0, 0, (50, 0, 100, 0, 0, 100, 0, 0)), _candidate(0, 0))
    cover = joint_pair_cover(initial_grid_step_pct=100)
    forward = build_joint_stage_design(
        block_i_candidates=first,
        block_ii_candidates=second,
        pair_cover=cover,
        step_pct=20,
    )
    reversed_design = build_joint_stage_design(
        block_i_candidates=(*reversed(first), first[0]),
        block_ii_candidates=(*reversed(second), second[0]),
        pair_cover=cover,
        step_pct=20,
    )
    assert forward == reversed_design


def test_requested_paths_are_reconstructed_and_guided_noops_are_omitted() -> None:
    """A repaired anchor's annual demand shape cannot migrate to a different request."""
    source = _candidate(37, 23, (63, 17, 28, 31, 24, 61, 23, 16))
    shaped = replace(
        source,
        demand_lifetime=replace(
            source.demand_lifetime,
            demand_fraction=(Fraction(0), *((Fraction(37, 100),) * 81)),
        ),
    )
    guide = _candidate(0, 100, (0, 100, 0, 0, 0, 0, 0, 100))
    moved = guided_joint_candidate(shaped, guide, step_pct=7)
    assert moved is not None
    assert moved.demand_lifetime == DemandLifetimeDecision(30, 30)
    assert guided_joint_candidate(source, source, step_pct=7) is None
    design = build_joint_stage_design(
        block_i_candidates=(shaped,),
        block_ii_candidates=(_candidate(0, 0, source.circularity.percentages()),),
        pair_cover=joint_pair_cover(initial_grid_step_pct=100),
        step_pct=20,
    )
    rebuilt = next(item for item in design.candidates if _coordinates(item) == _coordinates(source))
    assert rebuilt == source
    assert rebuilt != shaped
