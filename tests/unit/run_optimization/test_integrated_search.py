"""Joint search preserves both domains, finite routes and state-specific closure."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest
from request_search_fixtures import _compliant_evaluation

from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    line_candidates,
    neighboring_candidates,
)
from pytoce.run_optimization.calculation.integrated_design import (
    JointPairCover,
    JointStageDesign,
    build_joint_stage_design,
    joint_pair_cover,
)
from pytoce.run_optimization.calculation.integrated_search import (
    advance_integrated_search,
    finish_integrated_search,
    initialize_integrated_search,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)


def _candidate(
    demand: int, lifetime: int = 0, collection: int | None = None
) -> InterventionCandidate:
    """Create linear requests in either the parent or explicit circularity domain."""
    circularity = (
        CircularityDecision.parent()
        if collection is None
        else CircularityDecision(collection, 100, 0, 0, 0, 100, 0, 0)
    )
    return InterventionCandidate(
        "integrated_frontier", DemandLifetimeDecision(demand, lifetime), circularity
    )


def _design(
    *candidates: InterventionCandidate,
    step: int,
    actual_i_pairs: tuple[tuple[int, int], ...] = (),
) -> JointStageDesign:
    """Supply a bounded declared geometric fixture; geometry coverage has its own owner."""
    identities = tuple(item.identity for item in candidates)
    return JointStageDesign(
        identity=f"fixture-stage-{step}::" + "::".join(identities),
        step_pct=step,
        candidates=candidates,
        starting_candidate_identities=identities,
        source_candidate_identities=(),
        geometric_guide_candidate_identities=(),
        actual_ii_candidates=(),
        routes=(),
        proposal_origins=tuple(
            (
                item.identity,
                ("parent_circularity",)
                if item.circularity.uses_exact_parent
                else ("actual_i_by_represented_circularity",)
                if (
                    item.demand_lifetime.demand_reduction_pct,
                    item.demand_lifetime.lifetime_extension_pct,
                )
                in actual_i_pairs
                else ("independent_pair_cover",),
            )
            for item in candidates
        ),
        pair_cover=JointPairCover((), 0, 0, 20),
        explicit_ii_routes_absent=True,
        actual_i_pairs=actual_i_pairs,
    )


def _evaluation(
    candidate: InterventionCandidate, scores: tuple[float, ...], *, compliant: bool = False
) -> CandidateEvaluation:
    """Supply small synthetic environmental states at the real search ranking boundary."""
    base = _compliant_evaluation(candidate)
    metrics = tuple(
        replace(
            base.state_metrics[0],
            state_identity=f"state::{index}",
            all_region_joint_compliant=compliant,
            fully_compliant_regions=("EUR",) if compliant else (),
            fully_compliant_impact_targets=("target",) if compliant else (),
            compliant_region_count=int(compliant),
            fully_compliant_impact_target_count=int(compliant),
            compliant_region_impact_group_count=int(compliant),
            compliant_coordinate_count=int(compliant),
            worst_asr=score if compliant else 1.0 + score,
            binding_slack=1.0 - score if compliant else None,
            maximum_positive_exceedance=0.0 if compliant else score,
            balanced_mean_positive_exceedance=0.0 if compliant else score,
        )
        for index, score in enumerate(scores)
    )
    return replace(base, state_metrics=metrics)


def _landscape(
    values: Mapping[InterventionCandidate, tuple[float, ...]],
    observed: list[str],
    *,
    unresolved: frozenset[InterventionCandidate] = frozenset(),
    reverse: bool = False,
) -> Callable[[tuple[InterventionCandidate, ...]], tuple[CandidateEvaluation, ...]]:
    """Return synthetic scored, rejected and failed-repair outcomes."""

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Score admitted requests once; the remaining requests are rejected."""
        observed.extend(item.identity for item in candidates)
        results = tuple(
            _evaluation(item, values[item])
            if item in values
            else CandidateEvaluation(
                item,
                "physical::" + item.identity,
                (),
                0.5,
                repair_unresolved=item in unresolved,
            )
            for item in candidates
        )
        return tuple(reversed(results)) if reverse else results

    return evaluate


def test_inferior_domain_crosses_a_full_line_to_outperform_the_initial_winner() -> None:
    """An initially inferior explicit domain remains active and polls beyond intermediates."""
    parent, first = _candidate(0), _candidate(10)
    inferior, intermediate, winner = (_candidate(value, collection=50) for value in (50, 45, 40))
    observed: list[str] = []
    evaluate = _landscape(
        {parent: (20,), first: (3,), inferior: (5,), intermediate: (4,), winner: (2,)}, observed
    )
    coarse = initialize_integrated_search(
        design=_design(parent, first, inferior, step=20), evaluate=evaluate
    )
    assert coarse.selection.selected_candidate_identities == (first.identity,)
    assert not coarse.completed_stages[0].polls
    coarse_report = finish_integrated_search(coarse, endpoint_refinement_steps_pct=())
    assert not coarse_report.state_verification[0].selection_stable
    state = advance_integrated_search(
        coarse, design=_design(parent, first, inferior, step=5), evaluate=evaluate
    )
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(5,))
    assert report.selection.selected_candidate_identities == (winner.identity,)
    assert not any(
        item.refinement_pass.anchor_candidate_identities == (intermediate.identity,)
        for item in state.completed_stages[-1].polls
    )
    assert all(not item.pruned_candidate_identities for item in report.passes)
    assert report.state_verification[0].declared_neighborhood_verified
    assert not report.state_verification[0].selected_has_non_grid_baseline_component
    assert len(observed) == len(set(observed))
    assert {item.candidate_identity for item in state.terminal_branches} >= {
        first.identity,
        winner.identity,
    }


def test_parent_domain_winner_keeps_explicit_branches_and_guided_joint_moves() -> None:
    """A three-family improvement survives even when parent circularity wins initially."""
    parent = _candidate(0)
    incumbent = _candidate(10)
    explicit = _candidate(10, 10, 50)
    guide = _candidate(20, 20, 60)
    winner = _candidate(15, 15, 55)
    assert winner not in neighboring_candidates(explicit, step_pct=5, coordinated=True)
    observed: list[str] = []
    evaluate = _landscape(
        {parent: (20,), incumbent: (2,), explicit: (5,), guide: (10,), winner: (1,)}, observed
    )
    coarse = initialize_integrated_search(
        design=_design(parent, incumbent, explicit, guide, step=20, actual_i_pairs=((20, 20),)),
        evaluate=evaluate,
    )
    assert coarse.selection.selected_candidate_identities == (incumbent.identity,)
    state = advance_integrated_search(
        coarse,
        design=_design(parent, incumbent, explicit, guide, step=5, actual_i_pairs=((20, 20),)),
        evaluate=evaluate,
    )
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(5,))
    assert report.selection.selected_candidate_identities == (winner.identity,)
    assert any(
        winner.identity in item.proposed_candidate_identities and item.coordinated
        for item in report.passes
    )
    assert report.state_verification[0].declared_neighborhood_verified
    assert set(state.completed_stages[-1].local_guide_candidate_identities) == {
        explicit.identity,
        guide.identity,
    }


@pytest.mark.parametrize("reverse", (False, True))
def test_cross_state_winners_receive_own_pending_search(reverse: bool) -> None:
    """An incidental neighbor winning another state must not remain unrefined there."""
    parent, a, b, incidental, winner = (_candidate(value) for value in (0, 20, 70, 25, 30))
    observed: list[str] = []
    evaluate = _landscape(
        {parent: (20, 20), a: (2, 10), b: (10, 2), incidental: (8, 1), winner: (9, 0.5)},
        observed,
        reverse=reverse,
    )
    coarse = initialize_integrated_search(design=_design(parent, a, b, step=20), evaluate=evaluate)
    state = advance_integrated_search(
        coarse, design=_design(parent, a, b, step=5), evaluate=evaluate
    )
    assert tuple(item.candidate_identity for item in state.selection.states) == (
        a.identity,
        winner.identity,
    )
    assert any(
        item.state_identity == "state::1"
        and item.refinement_pass.anchor_candidate_identities == (winner.identity,)
        for item in state.completed_stages[-1].polls
    )
    assert len(observed) == len(set(observed))
    serial = _landscape(
        {parent: (20, 20), a: (2, 10), b: (10, 2), incidental: (8, 1), winner: (9, 0.5)}, []
    )
    reference = advance_integrated_search(
        initialize_integrated_search(design=_design(parent, a, b, step=20), evaluate=serial),
        design=_design(parent, a, b, step=5),
        evaluate=serial,
    )
    assert state == reference


@pytest.mark.parametrize("unknown_is_neighbor", (False, True))
def test_unresolved_proposals_do_not_stop_later_points_and_local_flags_are_scoped(
    unknown_is_neighbor: bool,
) -> None:
    """Attempt every base proposal; a failed local repair prevents neighborhood completion."""
    parent, rejected, worse, winner = (_candidate(value) for value in (0, 20, 40, 60))
    unknown = _candidate(55) if unknown_is_neighbor else _candidate(30, 10)
    observed: list[str] = []
    evaluate = _landscape(
        {parent: (10,), worse: (11,), winner: (2,)},
        observed,
        unresolved=frozenset({unknown}),
    )
    proposals = (parent, rejected, unknown, worse, winner)
    coarse = initialize_integrated_search(design=_design(*proposals, step=20), evaluate=evaluate)
    assert {item.identity for item in proposals}.issubset(observed)
    state = advance_integrated_search(coarse, design=_design(*proposals, step=5), evaluate=evaluate)
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(5,))
    assert report.selection.selected_candidate_identities == (winner.identity,)
    assert report.state_verification[0].selection_stable
    assert report.state_verification[0].declared_neighborhood_verified is not unknown_is_neighbor
    assert report.unsuccessful_path_search_count == 1


def test_numerical_failure_propagates_instead_of_becoming_a_route_rejection() -> None:
    """Search must not catch numerical execution errors or continue a failed attempt."""
    parent, first, failing = _candidate(0), _candidate(20), _candidate(40)

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Represent the declared numerical failure with a test double."""
        if failing in candidates:
            raise RuntimeError("numerical solver failure")
        return tuple(_evaluation(item, (2,)) for item in candidates)

    with pytest.raises(RuntimeError, match="numerical solver failure"):
        initialize_integrated_search(
            design=_design(parent, first, failing, step=20), evaluate=evaluate
        )


def test_compliant_parent_shortcut_never_claims_executed_neighborhoods() -> None:
    """A complete compliant unchanged parent is terminal before any other proposal."""
    parent, unused = _candidate(0), _candidate(60, 50, 100)
    observed: list[str] = []

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Return a compliant parent in two states."""
        observed.extend(item.identity for item in candidates)
        return tuple(_evaluation(item, (0.8, 0.9), compliant=True) for item in candidates)

    coarse = initialize_integrated_search(
        design=_design(parent, unused, step=20), evaluate=evaluate
    )
    state = advance_integrated_search(
        coarse, design=_design(parent, unused, step=5), evaluate=evaluate
    )
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(5,))
    assert observed == [parent.identity]
    assert all(item.parent_shortcut for item in state.completed_stages)
    assert report.evaluation_count == 1
    assert not report.passes
    assert not report.final_neighbor_candidate_identities
    assert not any(item.selection_stable for item in report.state_verification)
    assert not any(item.declared_neighborhood_verified for item in report.state_verification)


def test_empty_metrics_do_not_invent_a_successful_parent_or_neighborhood() -> None:
    """Record an MFA rejection as a failed candidate outcome."""
    candidate = _candidate(20, 20, 50)
    observed: list[str] = []
    evaluate = _landscape({}, observed)
    coarse = initialize_integrated_search(design=_design(candidate, step=20), evaluate=evaluate)
    state = advance_integrated_search(coarse, design=_design(candidate, step=5), evaluate=evaluate)
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(5,))
    assert not report.selection.states
    assert not report.state_verification
    assert observed == [candidate.identity]


def test_completed_stage_extension_keeps_old_stage_bytes_and_reuses_observations() -> None:
    """Appending a one-point fixture stage preserves the completed five-point prefix."""
    parent, first, fine = (_candidate(value) for value in (0, 20, 19))
    observed: list[str] = []
    evaluate = _landscape({parent: (10,), first: (2,), fine: (1,)}, observed)
    coarse = initialize_integrated_search(design=_design(parent, first, step=20), evaluate=evaluate)
    five = advance_integrated_search(
        coarse, design=_design(parent, first, step=5), evaluate=evaluate
    )
    frozen_stages = five.completed_stages
    frozen_report = finish_integrated_search(five, endpoint_refinement_steps_pct=(5,))
    one = advance_integrated_search(five, design=_design(parent, first, step=1), evaluate=evaluate)
    assert one.completed_stages[:-1] == frozen_stages
    assert finish_integrated_search(five, endpoint_refinement_steps_pct=(5,)) == frozen_report
    assert finish_integrated_search(
        one, endpoint_refinement_steps_pct=(5, 1)
    ).selection.selected_candidate_identities == (fine.identity,)
    assert (
        not finish_integrated_search(five, endpoint_refinement_steps_pct=(5, 1))
        .state_verification[0]
        .selection_stable
    )
    assert len(observed) == len(set(observed))
    assert one.reused_candidate_count > five.reused_candidate_count


def test_same_family_better_basin_is_honestly_unexplored() -> None:
    """Compare evaluated domain winners with a separate higher-order basin."""
    parent, inferior, intermediate, missed = (
        _candidate(0),
        _candidate(90),
        _candidate(40),
        _candidate(90, 50),
    )
    values = {parent: (3,), inferior: (5,), intermediate: (4,), missed: (2,)}
    observed: list[str] = []
    evaluate = _landscape(values, observed)
    pair_cover = joint_pair_cover(initial_grid_step_pct=100)

    def design(step: int) -> JointStageDesign:
        """Use the actual supported design with the selected unchanged single-block anchors."""
        return build_joint_stage_design(
            block_i_candidates=(replace(parent, optimization_block="demand_lifetime_frontier"),),
            block_ii_candidates=(
                replace(parent, optimization_block="circularity_feedstock_anchor"),
            ),
            pair_cover=pair_cover,
            step_pct=step,
        )

    coarse = initialize_integrated_search(design=design(100), evaluate=evaluate)
    assert not coarse.terminal_branches
    assert {item.candidate_identity for item in coarse.pending_branches} == {parent.identity}
    assert inferior.identity in observed
    state = advance_integrated_search(coarse, design=design(50), evaluate=evaluate)
    report = finish_integrated_search(state, endpoint_refinement_steps_pct=(50,))
    assert report.selection.selected_candidate_identities == (parent.identity,)
    assert report.state_verification[0].declared_neighborhood_verified
    assert intermediate.identity not in observed and missed.identity not in observed
    assert min(values, key=lambda item: values[item]) == missed
    assert intermediate in neighboring_candidates(inferior, step_pct=50)
    assert missed in neighboring_candidates(intermediate, step_pct=50, coordinated=True)
    seeds = state.completed_stages[-1].family_seeds
    assert len(seeds) == 13
    assert len({item.candidate_identity for item in seeds if item.candidate_identity}) == 1
    assert any(item.candidate_identity is None for item in seeds)


def test_opportunistic_batches_ignore_external_cache_order_and_report_skips() -> None:
    """A later cached improvement cannot pre-empt the fixed first improving batch."""
    parent, seed = _candidate(0), _candidate(20, collection=20)
    axial, collection = _candidate(15, collection=20), _candidate(20, collection=25)
    values = {parent: (10,), seed: (4,), axial: (3,), collection: (1,)}
    states = []
    for warm in (False, True):
        cache = {collection.identity: _evaluation(collection, values[collection])} if warm else {}
        evaluator = _landscape(values, [])

        def evaluate(
            candidates: tuple[InterventionCandidate, ...],
        ) -> tuple[CandidateEvaluation, ...]:
            """Return only the requested batch, with warm external values in reversed order."""
            missing = tuple(item for item in candidates if item.identity not in cache)
            cache.update((item.candidate.identity, item) for item in evaluator(missing))
            return tuple(cache[item.identity] for item in reversed(candidates))

        coarse = initialize_integrated_search(
            design=_design(parent, seed, step=20), evaluate=evaluate
        )
        states.append(
            advance_integrated_search(
                coarse, design=_design(parent, seed, step=5), evaluate=evaluate
            )
        )
    assert states[0] == states[1]
    poll = next(
        item
        for item in states[0].completed_stages[-1].polls
        if (
            item.refinement_pass.anchor_candidate_identities == (seed.identity,)
            and not item.refinement_pass.coordinated
        )
    )
    assert poll.refinement_pass.selected_candidate_identities == (axial.identity,)
    assert tuple(item.family for item in poll.batches) == ("demand_lifetime",)
    assert collection.identity in poll.skipped_candidate_identities
    assert collection.identity not in poll.evaluated_candidate_identities
    assert set(poll.refinement_pass.proposed_candidate_identities) == set(
        (*poll.evaluated_candidate_identities, *poll.skipped_candidate_identities)
    )


def test_final_closure_revisits_changed_states_without_carrying_transient_winners() -> None:
    """Cross-state improvements require repeated closure but no permanent transient branches."""
    parent, a, b, x, y, z, final = (
        _candidate(0),
        _candidate(20),
        _candidate(70, 70),
        _candidate(25),
        _candidate(25, 30),
        _candidate(40, 30),
        _candidate(40, 45),
    )
    evaluate = _landscape(
        {
            parent: (20, 20),
            a: (2, 10),
            b: (10, 2),
            x: (8, 1),
            y: (1, 0.5),
            z: (0.5, 5),
            final: (0.4, 0.2),
        },
        [],
    )
    coarse = initialize_integrated_search(design=_design(parent, a, b, step=20), evaluate=evaluate)
    state = advance_integrated_search(
        coarse, design=_design(parent, a, b, step=5), evaluate=evaluate
    )
    assert {item.candidate_identity for item in state.selection.states} == {final.identity}
    assert {(item.state_identity, item.candidate_identity) for item in state.terminal_branches} == {
        ("state::0", final.identity),
        ("state::1", final.identity),
    }
    polishing = [item.state_identity for item in state.completed_stages[-1].polls]
    assert polishing[0] == "state::0"
    assert "state::0" in polishing[polishing.index("state::1") + 1 :]
    assert polishing[-1] == "state::1"
    assert all(item.purpose == "domain_refinement" for item in state.completed_stages[-1].polls)
    assert all(
        item.declared_neighborhood_verified
        for item in finish_integrated_search(
            state, endpoint_refinement_steps_pct=(5,)
        ).state_verification
    )


def test_compliant_parent_state_does_not_disable_other_states_or_supply_extra_roles() -> None:
    """Guide and seed roles are designated only for states requiring intervention search."""
    parent, seed = _candidate(0), _candidate(20, collection=50)

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Keep the first parent state compliant while the second benefits from intervention."""
        result = []
        for candidate in candidates:
            if candidate not in (parent, seed):
                result.append(CandidateEvaluation(candidate, "rejected", (), 0.5))
                continue
            evaluation = _evaluation(candidate, (2, 3 if candidate == parent else 1))
            if candidate == parent:
                evaluation = replace(
                    evaluation,
                    state_metrics=(
                        replace(
                            _evaluation(candidate, (0.5,), compliant=True).state_metrics[0],
                            state_identity="state::0",
                        ),
                        evaluation.state_metrics[1],
                    ),
                )
            result.append(evaluation)
        return tuple(result)

    coarse = initialize_integrated_search(design=_design(parent, seed, step=20), evaluate=evaluate)
    stage = advance_integrated_search(
        coarse, design=_design(parent, seed, step=5), evaluate=evaluate
    ).completed_stages[-1]
    assert {item.state_identity for item in stage.family_seeds} == {"state::1"}
    assert {item.state_identity for item in stage.guide_roles} == {"state::1"}
    assert {item.state_identity for item in stage.terminal_branches} == {"state::1"}


@pytest.mark.parametrize("new_score", (1.0, 3.0))
def test_completed_stage_candidates_remain_eligible_without_inferior_branch_carry(
    new_score: float,
) -> None:
    """Re-select all saved observations after admission; carry no superseded polling duty."""
    parent, old, new = _candidate(0), _candidate(20), _candidate(45, 35)
    evaluate = _landscape({parent: (10,), old: (2,), new: (new_score,)}, [])
    coarse = initialize_integrated_search(design=_design(parent, old, step=20), evaluate=evaluate)
    ten = advance_integrated_search(coarse, design=_design(parent, old, step=10), evaluate=evaluate)
    assert {item.candidate_identity for item in ten.terminal_branches} == {old.identity}
    five = advance_integrated_search(ten, design=_design(parent, new, step=5), evaluate=evaluate)
    winner, inferior = (new, old) if new_score < 2 else (old, new)
    assert five.selection.selected_candidate_identities == (winner.identity,)
    assert old.identity in {item.candidate.identity for item in five.evaluations}
    assert not any(
        poll.refinement_pass.anchor_candidate_identities == (inferior.identity,)
        for poll in five.completed_stages[-1].polls
    )
    assert {item.candidate_identity for item in five.terminal_branches} == {winner.identity}


def test_nonwinning_explicit_domain_has_its_own_complete_terminal() -> None:
    """A globally inferior categorical winner is still closed at its actual incumbent."""
    parent, explicit, improved = _candidate(0), _candidate(20, 20, 50), _candidate(20, 40, 50)
    evaluate = _landscape({parent: (1,), explicit: (3,), improved: (2,)}, [])
    coarse = initialize_integrated_search(
        design=_design(parent, explicit, step=20), evaluate=evaluate
    )
    state = advance_integrated_search(
        coarse, design=_design(parent, explicit, step=5), evaluate=evaluate
    )
    assert state.selection.selected_candidate_identities == (parent.identity,)
    terminals = {item.candidate_identity: item for item in state.terminal_branches}
    assert set(terminals) == {parent.identity, improved.identity}
    expected_scalar = {item.identity for item in line_candidates(improved, step_pct=5)}
    assert expected_scalar.issubset(terminals[improved.identity].neighbor_candidate_identities)
    assert explicit.identity not in terminals


def test_numerical_failure_on_a_distant_line_point_is_not_a_rejection() -> None:
    """A one-attempt numerical failure beyond the immediate neighbor remains fatal."""
    parent, failing = _candidate(0), _candidate(40)

    def evaluate(candidates: tuple[InterventionCandidate, ...]) -> tuple[CandidateEvaluation, ...]:
        """Raise the simulated failure during full-line refinement."""
        if failing in candidates:
            raise RuntimeError("numerical line failure")
        return tuple(_evaluation(item, (2,)) for item in candidates)

    coarse = initialize_integrated_search(design=_design(parent, step=20), evaluate=evaluate)
    with pytest.raises(RuntimeError, match="numerical line failure"):
        advance_integrated_search(coarse, design=_design(parent, step=10), evaluate=evaluate)
