"""Refine Block III states with unchanged parent circularity or explicit circularity targets."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, TypeAlias

from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CandidateRefinement,
    CandidateSelection,
    CandidateStateCheck,
    InterventionCandidate,
    RefinementPass,
    candidate_coordinate_key,
    line_candidates,
    neighboring_candidates,
    select_candidate_states,
)
from pytoce.run_optimization.calculation.integrated_design import (
    BASE_PROPOSAL_FAMILIES,
    ROUTE_PROPOSAL_FAMILIES,
    JointRoute,
    JointStageDesign,
    complete_joint_stage_design,
    guided_joint_candidate,
)
from pytoce.shared.runtime.filesystem import payload_sha256

INTEGRATED_SEARCH_METHOD_IDENTITY: Final = "pytoce_integrated_domain_search_v3"
CandidateEvaluator: TypeAlias = Callable[
    [tuple[InterventionCandidate, ...]], Sequence[CandidateEvaluation]
]
BranchKey: TypeAlias = tuple[str, str]


@dataclass(frozen=True, slots=True)
class IntegratedGuideRole:
    """Record a named guide role, including an empty conditional or independent role."""

    role: Literal["actual_ii", "independent_joint", "conditional_i"]
    state_identity: str | None
    demand_lifetime_pair: tuple[int, int] | None
    candidate_identity: str | None


@dataclass(frozen=True, slots=True)
class IntegratedFamilySeed:
    """Record a scientific family's minimum as search provenance."""

    state_identity: str
    family: str
    candidate_identity: str | None


@dataclass(frozen=True, slots=True)
class IntegratedBranchSeed:
    """Retain a state/domain winner awaiting local polling."""

    state_identity: str
    candidate_identity: str


@dataclass(frozen=True, slots=True)
class IntegratedPollBatch:
    """Record one fully processed scientific-family batch, independent of workers."""

    family: str
    proposed_candidate_identities: tuple[str, ...]
    newly_evaluated_candidate_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegratedBranchTerminal:
    """Retain one state-specific incumbent and its complete final local proposals.

    Coarse seeds have a separate type and are distinguished from local terminals.
    """

    state_identity: str
    candidate_identity: str
    neighbor_candidate_identities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntegratedBranchPoll:
    """Associate a scalar or coordinated poll with its own environmental state."""

    state_identity: str
    refinement_pass: RefinementPass
    batches: tuple[IntegratedPollBatch, ...] = ()
    evaluated_candidate_identities: tuple[str, ...] = ()
    skipped_candidate_identities: tuple[str, ...] = ()
    purpose: Literal["domain_refinement"] = "domain_refinement"


@dataclass(frozen=True, slots=True)
class IntegratedStageState:
    """Preserve frozen proposal geometry, admitted guides and completed branches."""

    design: JointStageDesign
    selection: CandidateSelection
    admitted_starting_candidate_identities: tuple[str, ...]
    local_guide_candidate_identities: tuple[str, ...]
    terminal_branches: tuple[IntegratedBranchTerminal, ...]
    polls: tuple[IntegratedBranchPoll, ...] = ()
    parent_shortcut: bool = False
    local_polling_complete: bool = False
    guide_roles: tuple[IntegratedGuideRole, ...] = ()
    family_seeds: tuple[IntegratedFamilySeed, ...] = ()
    pending_branches: tuple[IntegratedBranchSeed, ...] = ()


@dataclass(frozen=True, slots=True)
class IntegratedSearchState:
    """Store compact joint observations and immutable completed-stage evidence."""

    evaluations: tuple[CandidateEvaluation, ...]
    selection: CandidateSelection
    completed_stages: tuple[IntegratedStageState, ...]
    terminal_branches: tuple[IntegratedBranchTerminal, ...]
    reused_candidate_count: int = 0
    pending_branches: tuple[IntegratedBranchSeed, ...] = ()


def initialize_integrated_search(
    *, design: JointStageDesign, evaluate: CandidateEvaluator
) -> IntegratedSearchState:
    """Admit the base, freeze role guides, traverse routes and retain pending seeds."""
    evaluations: dict[str, CandidateEvaluation] = {}
    parents = tuple(item for item in design.candidates if item.is_exact_parent)
    _evaluate_missing(parents, evaluations=evaluations, evaluate=evaluate)
    selection = select_candidate_states(evaluations.values())
    shortcut = _parent_terminal(selection)
    reused = 0
    roles: tuple[IntegratedGuideRole, ...] = ()
    families: tuple[IntegratedFamilySeed, ...] = ()
    if not shortcut:
        design, selection, roles, families, reused = _admit_stage(
            design, previous_routes=(), evaluations=evaluations, evaluate=evaluate
        )
    pending = _domain_branches(evaluations=evaluations, selection=selection)
    admitted = _admitted_candidates(design.candidates, evaluations)
    stage = IntegratedStageState(
        design=design,
        selection=selection,
        admitted_starting_candidate_identities=tuple(item.candidate.identity for item in admitted),
        local_guide_candidate_identities=design.geometric_guide_candidate_identities,
        terminal_branches=(),
        parent_shortcut=shortcut,
        guide_roles=roles,
        family_seeds=families,
        pending_branches=pending,
    )
    return IntegratedSearchState(
        evaluations=_ordered_evaluations(evaluations.values()),
        selection=selection,
        completed_stages=(stage,),
        terminal_branches=(),
        reused_candidate_count=reused,
        pending_branches=pending,
    )


def advance_integrated_search(
    state: IntegratedSearchState,
    *,
    design: JointStageDesign,
    evaluate: CandidateEvaluator,
) -> IntegratedSearchState:
    """Admit all stage proposals, then close every current state/domain winner."""
    evaluations = {item.candidate.identity: item for item in state.evaluations}
    shortcut = _parent_terminal(state.selection)
    reused = 0
    roles: tuple[IntegratedGuideRole, ...] = ()
    families: tuple[IntegratedFamilySeed, ...] = ()
    selection = state.selection
    if not shortcut:
        design, selection, roles, families, reused = _admit_stage(
            design,
            previous_routes=state.completed_stages[-1].design.routes,
            evaluations=evaluations,
            evaluate=evaluate,
        )
    guides = tuple(
        evaluations[identity].candidate for identity in design.geometric_guide_candidate_identities
    )
    starts = _domain_branches(evaluations=evaluations, selection=selection)
    terminals, polls, selection, poll_reuse = _complete_branches(
        starts,
        evaluations=evaluations,
        selection=selection,
        guides=guides,
        step_pct=design.step_pct,
        evaluate=evaluate,
    )
    admitted = _admitted_candidates(design.candidates, evaluations)
    stage = IntegratedStageState(
        design=design,
        selection=selection,
        admitted_starting_candidate_identities=tuple(item.candidate.identity for item in admitted),
        local_guide_candidate_identities=tuple(item.identity for item in guides),
        terminal_branches=terminals,
        polls=polls,
        parent_shortcut=shortcut,
        local_polling_complete=bool(starts),
        guide_roles=roles,
        family_seeds=families,
    )
    return IntegratedSearchState(
        evaluations=_ordered_evaluations(evaluations.values()),
        selection=selection,
        completed_stages=(*state.completed_stages, stage),
        terminal_branches=terminals,
        reused_candidate_count=state.reused_candidate_count + reused + poll_reuse,
    )


def _admit_stage(
    design: JointStageDesign,
    *,
    previous_routes: tuple[JointRoute, ...],
    evaluations: dict[str, CandidateEvaluation],
    evaluate: CandidateEvaluator,
) -> tuple[
    JointStageDesign,
    CandidateSelection,
    tuple[IntegratedGuideRole, ...],
    tuple[IntegratedFamilySeed, ...],
    int,
]:
    """Admit base values before selecting guides and freezing the complete coarse routes."""
    _, base_reuse = _evaluate_missing(design.candidates, evaluations=evaluations, evaluate=evaluate)
    selection = select_candidate_states(evaluations.values())
    roles = _role_guides(design, evaluations=evaluations, selection=selection)
    guide_ids = sorted(
        {item.candidate_identity for item in roles if item.candidate_identity is not None}
    )
    design = complete_joint_stage_design(
        design,
        guides=tuple(evaluations[identity].candidate for identity in guide_ids),
        previous_routes=previous_routes,
    )
    missing_count = sum(item.identity not in evaluations for item in design.candidates)
    print(
        f"Block III spacing {design.step_pct}: frozen routes={len(design.routes)}, "
        f"distinct proposals={len(design.candidates)}, "
        f"new logical evaluations={missing_count}",
        flush=True,
    )
    _, route_reuse = _evaluate_missing(
        design.candidates, evaluations=evaluations, evaluate=evaluate
    )
    selection = select_candidate_states(evaluations.values())
    families = _family_seeds(design, evaluations=evaluations, selection=selection)
    return design, selection, roles, families, base_reuse + route_reuse


def _role_guides(
    design: JointStageDesign,
    *,
    evaluations: dict[str, CandidateEvaluation],
    selection: CandidateSelection,
) -> tuple[IntegratedGuideRole, ...]:
    """Retain actual Block II guides and rank independent and conditional joint guides.

    For each active state, select the best admitted independent joint candidate and
    the best admitted explicit candidate at each retained Block I endpoint pair.
    """
    roles = [
        IntegratedGuideRole("actual_ii", None, None, item.identity)
        for item in design.actual_ii_candidates
        if not item.circularity.uses_exact_parent
    ]
    origins = dict(design.proposal_origins)
    explicit = tuple(item for item in design.candidates if not item.circularity.uses_exact_parent)
    independent = tuple(
        item.identity for item in explicit if "independent_pair_cover" in origins[item.identity]
    )
    for state in selection.states:
        if state.status == "baseline_compliant":
            continue
        roles.append(
            IntegratedGuideRole(
                "independent_joint",
                state.state_identity,
                None,
                _state_minimum(state.state_identity, independent, evaluations),
            )
        )
        for pair in design.actual_i_pairs:
            candidates = tuple(
                item.identity
                for item in explicit
                if (
                    item.demand_lifetime.demand_reduction_pct,
                    item.demand_lifetime.lifetime_extension_pct,
                )
                == pair
            )
            roles.append(
                IntegratedGuideRole(
                    "conditional_i",
                    state.state_identity,
                    pair,
                    _state_minimum(state.state_identity, candidates, evaluations),
                )
            )
    return tuple(roles)


def _family_seeds(
    design: JointStageDesign,
    *,
    evaluations: dict[str, CandidateEvaluation],
    selection: CandidateSelection,
) -> tuple[IntegratedFamilySeed, ...]:
    """Record each proposal family's best-ranked admitted candidate, or None."""
    return tuple(
        IntegratedFamilySeed(
            state.state_identity,
            family,
            _state_minimum(
                state.state_identity,
                tuple(
                    identity for identity, origins in design.proposal_origins if family in origins
                ),
                evaluations,
            ),
        )
        for state in selection.states
        if state.status != "baseline_compliant"
        for family in (*BASE_PROPOSAL_FAMILIES, *ROUTE_PROPOSAL_FAMILIES)
    )


def _domain_branches(
    *, evaluations: dict[str, CandidateEvaluation], selection: CandidateSelection
) -> tuple[IntegratedBranchSeed, ...]:
    """Select both categorical minima from all observations."""
    active = {
        item.state_identity for item in selection.states if item.status != "baseline_compliant"
    }
    branches: list[IntegratedBranchSeed] = []
    for parent in (True, False):
        domain = select_candidate_states(
            item
            for item in evaluations.values()
            if item.candidate.circularity.uses_exact_parent == parent
        )
        branches.extend(
            IntegratedBranchSeed(state.state_identity, state.candidate_identity)
            for state in domain.states
            if state.state_identity in active
        )
    return tuple(sorted(branches, key=lambda item: (item.state_identity, item.candidate_identity)))


def _state_minimum(
    state_identity: str,
    identities: Iterable[str],
    evaluations: dict[str, CandidateEvaluation],
) -> str | None:
    """Use the complete common III ranking, or report an empty admitted role."""
    observations = tuple(
        replace(
            evaluations[identity],
            state_metrics=tuple(
                metric
                for metric in evaluations[identity].state_metrics
                if metric.state_identity == state_identity
            ),
        )
        for identity in identities
    )
    selection = select_candidate_states(observations)
    return selection.states[0].candidate_identity if selection.states else None


def _complete_branches(
    starts: tuple[IntegratedBranchSeed, ...],
    *,
    evaluations: dict[str, CandidateEvaluation],
    selection: CandidateSelection,
    guides: tuple[InterventionCandidate, ...],
    step_pct: int,
    evaluate: CandidateEvaluator,
) -> tuple[
    tuple[IntegratedBranchTerminal, ...], tuple[IntegratedBranchPoll, ...], CandidateSelection, int
]:
    """Close current categorical minima, including discoveries from other states or domains."""
    resolved: dict[BranchKey, IntegratedBranchTerminal] = {}
    polls: list[IntegratedBranchPoll] = []
    reused = 0

    def follow(
        state_identity: str,
        incumbent_identity: str,
    ) -> IntegratedBranchTerminal:
        """Follow a current domain minimum until both polls close at the same incumbent."""
        nonlocal selection, reused
        coordinated = False
        while True:
            incumbent = evaluations[incumbent_identity].candidate
            batches = _poll_batches(
                incumbent, guides=guides, step_pct=step_pct, coordinated=coordinated
            )
            processed: list[IntegratedPollBatch] = []
            best = incumbent_identity
            for family, proposals in batches:
                new, reused_here = _evaluate_missing(
                    proposals, evaluations=evaluations, evaluate=evaluate
                )
                reused += reused_here
                selection = select_candidate_states(
                    evaluations[identity]
                    for identity in (*selection.selected_candidate_identities, *new)
                )
                best = _branch_winner(
                    state_identity, incumbent_identity, proposals=proposals, evaluations=evaluations
                )
                processed.append(
                    IntegratedPollBatch(family, tuple(item.identity for item in proposals), new)
                )
                if best != incumbent_identity:
                    break
            full_ids = tuple(item.identity for _, batch in batches for item in batch)
            processed_ids = tuple(
                identity for item in processed for identity in item.proposed_candidate_identities
            )
            skipped = full_ids[len(processed_ids) :]
            polls.append(
                IntegratedBranchPoll(
                    state_identity,
                    RefinementPass(
                        step_pct=step_pct,
                        iteration=len(polls) + 1,
                        anchor_candidate_identities=(incumbent_identity,),
                        proposed_candidate_identities=full_ids,
                        pruned_candidate_identities=(),
                        newly_evaluated_candidate_identities=tuple(
                            identity
                            for item in processed
                            for identity in item.newly_evaluated_candidate_identities
                        ),
                        selected_candidate_identities=(best,),
                        coordinated=coordinated,
                    ),
                    batches=tuple(processed),
                    evaluated_candidate_identities=processed_ids,
                    skipped_candidate_identities=skipped,
                )
            )
            if best != incumbent_identity:
                incumbent_identity = best
                coordinated = False
            elif not coordinated:
                coordinated = True
            else:
                scalar = _poll_batches(
                    incumbent, guides=guides, step_pct=step_pct, coordinated=False
                )
                terminal = IntegratedBranchTerminal(
                    state_identity,
                    incumbent_identity,
                    tuple(
                        sorted(
                            {*full_ids, *(item.identity for _, batch in scalar for item in batch)}
                        )
                    ),
                )
                break
        resolved[(state_identity, incumbent_identity)] = terminal
        return terminal

    pending = starts
    current = starts
    while pending:
        winner = pending[0]
        follow(winner.state_identity, winner.candidate_identity)
        # A new observation can change another state's best in either domain.
        # Recompute before starting another branch instead of carrying stale starts.
        current = _domain_branches(evaluations=evaluations, selection=selection)
        pending = tuple(
            item
            for item in current
            if (item.state_identity, item.candidate_identity) not in resolved
        )
    terminals = tuple(resolved[(item.state_identity, item.candidate_identity)] for item in current)
    return terminals, tuple(polls), selection, reused


def _branch_winner(
    state_identity: str,
    incumbent_identity: str,
    *,
    proposals: tuple[InterventionCandidate, ...],
    evaluations: dict[str, CandidateEvaluation],
) -> str:
    """Apply the existing complete ranking only to this branch's incumbent and poll."""
    identities = (incumbent_identity, *(item.identity for item in proposals))
    observations = tuple(
        replace(
            evaluations[identity],
            state_metrics=tuple(
                item
                for item in evaluations[identity].state_metrics
                if item.state_identity == state_identity
            ),
        )
        for identity in identities
    )
    return select_candidate_states(observations).states[0].candidate_identity


def _poll_batches(
    candidate: InterventionCandidate,
    *,
    guides: tuple[InterventionCandidate, ...],
    step_pct: int,
    coordinated: bool,
) -> tuple[tuple[str, tuple[InterventionCandidate, ...]], ...]:
    """Partition the declared poll in fixed scientific order."""
    scalar = line_candidates(candidate, step_pct=step_pct)
    scalar_ids = {item.identity for item in scalar}
    names = (
        (
            "demand_lifetime_diagonal",
            "demand_lifetime_collection",
            "demand_lifetime_treatment",
            "demand_lifetime_feedstock",
            "guided_joint",
        )
        if coordinated
        else ("demand_lifetime", "collection", "treatment", "feedstock")
    )
    groups: dict[str, list[InterventionCandidate]] = {name: [] for name in names}
    proposals = (
        neighboring_candidates(candidate, step_pct=step_pct, coordinated=True)
        if coordinated
        else scalar
    )
    seen = scalar_ids.copy() if coordinated else set()
    for proposal in proposals:
        if proposal.identity in seen:
            continue
        seen.add(proposal.identity)
        if proposal.circularity == candidate.circularity:
            group = names[0]
        elif proposal.circularity.collection_pct != candidate.circularity.collection_pct:
            group = names[1]
        elif (
            proposal.circularity.treatment_percentages
            != candidate.circularity.treatment_percentages
        ):
            group = names[2]
        else:
            group = names[3]
        groups[group].append(proposal)
    if coordinated and not candidate.circularity.uses_exact_parent:
        for guide in guides:
            guided = guided_joint_candidate(candidate, guide, step_pct=step_pct)
            if guided is not None and guided.identity not in seen:
                seen.add(guided.identity)
                groups["guided_joint"].append(guided)
    return tuple(
        (name, tuple(sorted(groups[name], key=candidate_coordinate_key)))
        for name in names
        if groups[name]
    )


def _evaluate_missing(
    candidates: Iterable[InterventionCandidate],
    *,
    evaluations: dict[str, CandidateEvaluation],
    evaluate: CandidateEvaluator,
) -> tuple[tuple[str, ...], int]:
    """Evaluate each distinct complete requested vector once in canonical order."""
    unique = {item.identity: item for item in candidates}
    missing = tuple(
        sorted(
            (item for identity, item in unique.items() if identity not in evaluations),
            key=candidate_coordinate_key,
        )
    )
    if missing:
        for observation in _ordered_evaluations(evaluate(missing)):
            evaluations[observation.candidate.identity] = observation
    return tuple(item.identity for item in missing), len(unique) - len(missing)


def _ordered_evaluations(
    evaluations: Iterable[CandidateEvaluation],
) -> tuple[CandidateEvaluation, ...]:
    """Make saved observations independent of worker completion order."""
    return tuple(sorted(evaluations, key=lambda item: candidate_coordinate_key(item.candidate)))


def _admitted_candidates(
    candidates: Iterable[InterventionCandidate], evaluations: dict[str, CandidateEvaluation]
) -> tuple[CandidateEvaluation, ...]:
    """Retain scored stage proposals, keeping rejection and failed demand repair distinct."""
    identities = {item.identity for item in candidates}
    return _ordered_evaluations(
        item
        for identity, item in evaluations.items()
        if identity in identities and item.state_metrics
    )


def _parent_terminal(selection: CandidateSelection) -> bool:
    """Use the unchanged-parent shortcut only for a nonempty complete compliant set."""
    return bool(selection.states) and all(
        item.status == "baseline_compliant" for item in selection.states
    )


def finish_integrated_search(
    state: IntegratedSearchState, *, endpoint_refinement_steps_pct: tuple[int, ...]
) -> CandidateRefinement:
    """Report the tested final-state neighborhoods."""
    evaluations = {item.candidate.identity: item for item in state.evaluations}
    stage = state.completed_stages[-1]
    final_stage = bool(
        endpoint_refinement_steps_pct
        and stage.design.step_pct == endpoint_refinement_steps_pct[-1]
        and stage.local_polling_complete
    )
    terminals = {
        (item.state_identity, item.candidate_identity): item for item in stage.terminal_branches
    }
    verification: list[CandidateStateCheck] = []
    final_neighbors: set[str] = set()
    for selected in state.selection.states:
        candidate = evaluations[selected.candidate_identity].candidate
        terminal = terminals.get((selected.state_identity, selected.candidate_identity))
        stable = final_stage and terminal is not None
        neighbors = terminal.neighbor_candidate_identities if stable and terminal else ()
        final_neighbors.update(neighbors)
        verification.append(
            CandidateStateCheck(
                state_identity=selected.state_identity,
                candidate_identity=selected.candidate_identity,
                selected_is_exact_parent=candidate.is_exact_parent,
                selected_has_non_grid_baseline_component=candidate.circularity.uses_exact_parent,
                declared_neighborhood_verified=bool(neighbors)
                and not any(evaluations[identity].repair_unresolved for identity in neighbors),
                selection_stable=stable,
            )
        )
    passes = tuple(
        poll.refinement_pass for completed in state.completed_stages for poll in completed.polls
    )
    unsuccessful = sum(item.repair_unresolved for item in state.evaluations)
    identity = (
        INTEGRATED_SEARCH_METHOD_IDENTITY
        + "::"
        + payload_sha256(
            {
                "method_identity": INTEGRATED_SEARCH_METHOD_IDENTITY,
                "endpoint_refinement_steps_pct": endpoint_refinement_steps_pct,
                "selection_identity": state.selection.identity,
                "evaluation_identities": sorted(evaluations),
                "stages": [
                    {
                        "design_identity": item.design.identity,
                        "selection_identity": item.selection.identity,
                        "admitted_starts": item.admitted_starting_candidate_identities,
                        "local_guides": item.local_guide_candidate_identities,
                        "guide_roles": [
                            (
                                role.role,
                                role.state_identity,
                                role.demand_lifetime_pair,
                                role.candidate_identity,
                            )
                            for role in item.guide_roles
                        ],
                        "family_seeds": [
                            (seed.state_identity, seed.family, seed.candidate_identity)
                            for seed in item.family_seeds
                        ],
                        "pending_branches": [
                            (seed.state_identity, seed.candidate_identity)
                            for seed in item.pending_branches
                        ],
                        "terminals": [
                            (
                                branch.state_identity,
                                branch.candidate_identity,
                                branch.neighbor_candidate_identities,
                            )
                            for branch in item.terminal_branches
                        ],
                        "polls": [
                            (
                                poll.state_identity,
                                poll.refinement_pass.anchor_candidate_identities,
                                poll.refinement_pass.proposed_candidate_identities,
                                poll.refinement_pass.selected_candidate_identities,
                                poll.refinement_pass.coordinated,
                                poll.evaluated_candidate_identities,
                                poll.skipped_candidate_identities,
                                poll.purpose,
                                [
                                    (batch.family, batch.proposed_candidate_identities)
                                    for batch in poll.batches
                                ],
                            )
                            for poll in item.polls
                        ],
                        "parent_shortcut": item.parent_shortcut,
                        "local_polling_complete": item.local_polling_complete,
                    }
                    for item in state.completed_stages
                ],
                "unresolved_count": unsuccessful,
                "verification": [
                    (
                        item.state_identity,
                        item.candidate_identity,
                        item.declared_neighborhood_verified,
                        item.selection_stable,
                    )
                    for item in verification
                ],
            }
        )
    )
    return CandidateRefinement(
        identity=identity,
        optimization_block="integrated_frontier",
        evaluation_count=len(evaluations),
        reused_candidate_count=state.reused_candidate_count,
        selection=state.selection,
        state_verification=tuple(verification),
        passes=passes,
        final_neighbor_candidate_identities=tuple(sorted(final_neighbors)),
        unsuccessful_path_search_count=unsuccessful,
    )
