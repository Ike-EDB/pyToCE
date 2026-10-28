"""Deterministic candidate combinations, selection, and local refinement."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations, product
from typing import Final, Literal, TypeAlias, cast

from pytoce.run_mfa.scenario.scenario_contract import OptimizationPhysicalInfeasibility
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    endpoint_neighbors,
    initial_demand_lifetime_decisions,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.shared.runtime.filesystem import payload_sha256

DEMAND_LIFETIME_DESIGN_METHOD_IDENTITY: Final = "pytoce_demand_lifetime_candidate_design_v5"
CIRCULARITY_FEEDSTOCK_DESIGN_METHOD_IDENTITY: Final = (
    "pytoce_circularity_feedstock_pairwise_design_v2"
)
CANDIDATE_REFINEMENT_METHOD_IDENTITY: Final = "pytoce_candidate_refinement_v12"
CANDIDATE_SELECTION_METHOD_IDENTITY: Final = "pytoce_candidate_selection_v3"
_CANDIDATE_IDENTITY_PREFIX = "pytoce_intervention_candidate_v4::"
_REFINEMENT_IDENTITY_PREFIX = CANDIDATE_REFINEMENT_METHOD_IDENTITY + "::"
_SELECTION_IDENTITY_PREFIX = "pytoce_candidate_selection_v3::"
_REFINEMENT_STEPS_PCT = (10, 5)

FactorValue: TypeAlias = tuple[int, int]
PairRequirement: TypeAlias = tuple[FactorValue, FactorValue]
SelectionStatus: TypeAlias = Literal[
    "intervention_compliant",
    "baseline_compliant",
    "selected_noncompliant",
]
BindingCoordinate: TypeAlias = tuple[
    str,
    Literal["annual", "cumulative_gwp100"],
    str,
    str,
    int | None,
    float,
]


@dataclass(frozen=True, slots=True)
class InterventionCandidate:
    """Store one complete annual demand/lifetime path and circularity intervention."""

    optimization_block: OptimizationBlock
    demand_lifetime: DemandLifetimeDecision
    circularity: CircularityDecision
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Attach the identity for a candidate created by the registered designs."""

        object.__setattr__(self, "identity", candidate_identity(self))

    @property
    def target_year(self) -> int:
        """Return the completion year of the active intervention group."""
        if not self.demand_lifetime.uses_exact_parent:
            return self.demand_lifetime.target_year
        return self.circularity.target_year

    @property
    def is_exact_parent(self) -> bool:
        """Return whether both intervention groups keep the baseline values."""

        return self.demand_lifetime.uses_exact_parent and self.circularity.uses_exact_parent


@dataclass(frozen=True, slots=True)
class CandidateStateMetrics:
    """Store compliance and ranking values for one optimization state.

    Attributes:
        state_identity: Content identifier calculated from the optimization block,
            background scenario, and allocated carrying capacity label, together with
            the scoring context, supplied allocated carrying capacities, and metric
            methods.
        candidate_identity: Identifier binding the block, endpoint percentages,
            completion year, and exact annual demand and lifetime controls.
        physical_candidate_identity: Identifier of the candidate's MFA outcome.
        optimization_block: Decision group being searched.
        background_scenario: REMIND background used for LCIA scores.
        allocated_cc_case: Selected label from the supplied allocated carrying capacities.
        is_zero_intervention_parent: Whether the candidate retains all baseline values.
        intervention_distance: Distance from baseline intervention values.
        all_region_joint_compliant: Whether every required coordinate in the 12 world regions has an
            absolute sustainability ratio (ASR) at most 1.
        fully_compliant_regions: Regions compliant for every required impact target.
        fully_compliant_impact_targets: Impact targets compliant in every world region.
        compliant_region_count: Number of fully compliant world regions.
        fully_compliant_impact_target_count: Number of fully compliant impact targets.
        compliant_region_impact_group_count: Compliant region and impact-target groups.
        compliant_coordinate_count: Compliant annual and cumulative ASR coordinates.
        region_count: Total number of world regions assessed.
        impact_target_count: Total annual categories plus the cumulative climate target.
        region_impact_group_count: Total region and impact-target groups.
        coordinate_count: Total annual and cumulative ASR coordinates.
        worst_asr: Largest required regional ASR across the annual and cumulative
            compliance coordinates.
        binding_slack: ``1 - worst_asr`` when all required coordinates comply;
            otherwise ``None``.
        maximum_positive_exceedance: ``max(worst_asr - 1, 0)``.
        balanced_mean_positive_exceedance: Mean positive exceedance with equal
            weight across impact targets and 12 world regions.
        binding_coordinates: Coordinates with the smallest slack or largest exceedance.
    """

    state_identity: str
    candidate_identity: str
    physical_candidate_identity: str
    optimization_block: OptimizationBlock
    background_scenario: str
    allocated_cc_case: str
    is_zero_intervention_parent: bool
    intervention_distance: float
    all_region_joint_compliant: bool
    fully_compliant_regions: tuple[str, ...]
    fully_compliant_impact_targets: tuple[str, ...]
    compliant_region_count: int
    fully_compliant_impact_target_count: int
    compliant_region_impact_group_count: int
    compliant_coordinate_count: int
    region_count: int
    impact_target_count: int
    region_impact_group_count: int
    coordinate_count: int
    worst_asr: float
    binding_slack: float | None
    maximum_positive_exceedance: float
    balanced_mean_positive_exceedance: float
    binding_coordinates: tuple[BindingCoordinate, ...]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Store one candidate, its direct failure or feasible state metrics."""

    candidate: InterventionCandidate
    physical_candidate_identity: str
    state_metrics: tuple[CandidateStateMetrics, ...]
    intervention_distance: float
    diagnostic: OptimizationPhysicalInfeasibility | None = None
    repair_unresolved: bool = False


@dataclass(frozen=True, slots=True)
class SelectedCandidateState:
    """Record the selected candidate and status for one optimization state."""

    state_identity: str
    candidate_identity: str
    status: SelectionStatus


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Record the selected candidate for every evaluated optimization state."""

    identity: str
    states: tuple[SelectedCandidateState, ...]
    selected_candidate_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateStateCheck:
    """Distinguish stable endpoint selection from fully resolved neighbors."""

    state_identity: str
    candidate_identity: str
    selected_is_exact_parent: bool
    selected_has_non_grid_baseline_component: bool
    declared_neighborhood_verified: bool
    selection_stable: bool


@dataclass(frozen=True, slots=True)
class RefinementPass:
    """Candidates proposed, evaluated, and retained at one refinement step."""

    step_pct: int
    iteration: int
    anchor_candidate_identities: tuple[str, ...]
    proposed_candidate_identities: tuple[str, ...]
    pruned_candidate_identities: tuple[str, ...]
    newly_evaluated_candidate_identities: tuple[str, ...]
    selected_candidate_identities: tuple[str, ...]
    coordinated: bool = False


@dataclass(frozen=True, slots=True)
class CompetitiveCandidateBatch:
    """Store distances and evaluations for one refinement candidate set."""

    candidate_distances: tuple[tuple[str, float], ...]
    evaluations: tuple[CandidateEvaluation, ...]


@dataclass(frozen=True, slots=True)
class CandidateRefinement:
    """Selected states, completed neighborhoods and failed demand repairs.

    The unsuccessful count covers endpoints whose finite demand-repair pool
    found no feasible member among the tested paths.
    """

    identity: str
    optimization_block: OptimizationBlock
    evaluation_count: int
    reused_candidate_count: int
    selection: CandidateSelection
    state_verification: tuple[CandidateStateCheck, ...]
    passes: tuple[RefinementPass, ...]
    final_neighbor_candidate_identities: tuple[str, ...]
    unsuccessful_path_search_count: int = 0


@dataclass(frozen=True, slots=True)
class CandidateRefinementStage:
    """Record completed spacing and the full request's distance-pruning policy."""

    step_pct: int
    unpruned: bool


@dataclass(frozen=True, slots=True)
class CandidateSearchState:
    """Retain compact observations and completed polling for one search block.

    The latest terminal neighborhood belongs to the completed stage, even when
    the request includes later stages. The state retains compact search summaries.
    """

    optimization_block: OptimizationBlock
    evaluations: tuple[CandidateEvaluation, ...]
    selection: CandidateSelection
    completed_stages: tuple[CandidateRefinementStage, ...] = ()
    passes: tuple[RefinementPass, ...] = ()
    reused_candidate_count: int = 0
    final_neighbor_candidate_identities: tuple[str, ...] = ()
    final_anchor_candidate_identities: tuple[str, ...] = ()
    final_selection_stable: bool = False


def candidate_identity(candidate: InterventionCandidate) -> str:
    """Return the intervention candidate identity."""
    return _CANDIDATE_IDENTITY_PREFIX + payload_sha256(
        {
            "optimization_block": candidate.optimization_block,
            "demand_lifetime": list(candidate.demand_lifetime.coordinates()),
            "circularity": list(candidate.circularity.percentages() or ()),
            "circularity_target_year": candidate.circularity.target_year,
        }
    )


def demand_lifetime_initial_design(
    *, target_year: int = 2050, step_pct: int = 20
) -> tuple[InterventionCandidate, ...]:
    """Create the exact joint linear reference at every coarse endpoint pair."""
    return tuple(
        InterventionCandidate(
            optimization_block="demand_lifetime_frontier",
            demand_lifetime=decision,
            circularity=CircularityDecision.parent(target_year=target_year),
        )
        for decision in initial_demand_lifetime_decisions(
            target_year=target_year, step_pct=step_pct
        )
    )


def percentage_compositions(*, component_count: int, step_pct: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate percentage combinations whose components sum to 100."""

    levels = tuple(range(0, 101, step_pct))
    return tuple(values for values in product(levels, repeat=component_count) if sum(values) == 100)


def circularity_feedstock_domain(
    *, target_year: int = 2050, step_pct: int = 20
) -> tuple[InterventionCandidate, ...]:
    """Create the complete admitted-grid collection, treatment, and feedstock set."""
    treatments = percentage_compositions(component_count=4, step_pct=step_pct)
    feedstocks = percentage_compositions(component_count=3, step_pct=step_pct)
    candidates = tuple(
        InterventionCandidate(
            optimization_block="circularity_feedstock_anchor",
            demand_lifetime=DemandLifetimeDecision(0, 0, target_year),
            circularity=CircularityDecision(
                collection,
                *cast(tuple[int, int, int, int], treatment),
                *cast(tuple[int, int, int], feedstock),
                target_year,
            ),
        )
        for collection, treatment, feedstock in product(
            range(0, 101, step_pct),
            treatments,
            feedstocks,
        )
    )
    return candidates


def circularity_feedstock_pairwise_design(
    *, target_year: int = 2050, step_pct: int = 20
) -> tuple[InterventionCandidate, ...]:
    """Cover every pair of circularity values admitted by the share constraints."""
    domain = circularity_feedstock_domain(target_year=target_year, step_pct=step_pct)
    rows = tuple(cast(tuple[int, ...], item.circularity.percentages()) for item in domain)
    selected = greedy_pairwise_indices(rows)
    return tuple(domain[index] for index in selected)


def pair_requirements(rows: Sequence[tuple[int, ...]]) -> frozenset[PairRequirement]:
    """Return every pair of control values present in the admitted ``rows``."""

    width = len(rows[0])
    return frozenset(
        ((left, row[left]), (right, row[right]))
        for row in rows
        for left, right in combinations(range(width), 2)
    )


def greedy_pairwise_indices(
    rows: Sequence[tuple[int, ...]],
) -> tuple[int, ...]:
    """Select a canonical greedy maximum-uncovered-pair design."""

    requirements = tuple(sorted(pair_requirements(rows)))
    bit_by_requirement = {requirement: 1 << index for index, requirement in enumerate(requirements)}
    coverage = tuple(
        sum(
            bit_by_requirement[((left, row[left]), (right, row[right]))]
            for left, right in combinations(range(len(row)), 2)
        )
        for row in rows
    )
    uncovered = (1 << len(requirements)) - 1
    remaining = list(range(len(rows)))
    selected: list[int] = []
    while uncovered:
        best_index = min(
            remaining,
            key=lambda index: (-(coverage[index] & uncovered).bit_count(), rows[index]),
        )
        gain = coverage[best_index] & uncovered
        selected.append(best_index)
        remaining.remove(best_index)
        uncovered &= ~gain
    return tuple(selected)


def neighboring_candidates(
    candidate: InterventionCandidate, *, step_pct: int, coordinated: bool = False
) -> tuple[InterventionCandidate, ...]:
    """Generate scalar and optionally coordinated neighbors of one candidate.

    Demand reduction, lifetime extension, and collection may move up or down
    within their bounds. Treatment and feedstock changes transfer the step from
    one share to another so each partition continues to total 100 percent.

    Args:
        candidate: Selected candidate around which alternatives are generated.
        step_pct: Endpoint and circularity percentage points.
        coordinated: Include joint demand/lifetime and circularity endpoint proposals.

    Returns:
        Unique valid alternatives in deterministic coordinate order. Zero demand
        and lifetime controls retain their valid positive neighbors. An unchanged regional
        circularity parent is a separate categorical choice.
    """

    demand = candidate.demand_lifetime
    circularity = candidate.circularity
    neighbors: dict[str, InterventionCandidate] = {}

    if candidate.optimization_block in {"demand_lifetime_frontier", "integrated_frontier"}:
        for updated in endpoint_neighbors(demand, step_pct=step_pct, coordinated=coordinated):
            item = InterventionCandidate(candidate.optimization_block, updated, circularity)
            neighbors[item.identity] = item

    if (
        candidate.optimization_block
        in {
            "circularity_feedstock_anchor",
            "integrated_frontier",
        }
        and not circularity.uses_exact_parent
    ):
        collection = cast(int, circularity.collection_pct)
        for direction in (-step_pct, step_pct):
            collection_changed = collection + direction
            if 0 <= collection_changed <= 100:
                item = _with_circularity_components(
                    candidate,
                    collection=collection_changed,
                    treatment=circularity.treatment_percentages,
                    feedstock=circularity.feedstock_percentages,
                )
                neighbors[item.identity] = item
        treatment = circularity.treatment_percentages
        for treatment_changed in _share_transfers(treatment, step_pct=step_pct):
            item = _with_circularity_components(
                candidate,
                collection=collection,
                treatment=cast(tuple[int, int, int, int], treatment_changed),
                feedstock=circularity.feedstock_percentages,
            )
            neighbors[item.identity] = item
        feedstock = circularity.feedstock_percentages
        for feedstock_changed in _share_transfers(feedstock, step_pct=step_pct):
            item = _with_circularity_components(
                candidate,
                collection=collection,
                treatment=treatment,
                feedstock=cast(tuple[int, int, int], feedstock_changed),
            )
            neighbors[item.identity] = item
    if coordinated and candidate.optimization_block == "integrated_frontier":
        circularity_neighbors = tuple(
            item for item in neighbors.values() if item.circularity != circularity
        )
        for circularity_neighbor, endpoint in product(
            circularity_neighbors, endpoint_neighbors(demand, step_pct=step_pct)
        ):
            item = replace(circularity_neighbor, demand_lifetime=endpoint)
            neighbors[item.identity] = item
    return tuple(sorted(neighbors.values(), key=candidate_coordinate_key))


def line_candidates(
    candidate: InterventionCandidate, *, step_pct: int
) -> tuple[InterventionCandidate, ...]:
    """Enumerate complete finite scalar and donor-recipient lines independently.

    Scalar lines retain both exact bounds as well as all incumbent-relative step
    multiples. Share transfers additionally retain the exact donor-empty point.
    The complete proposed line remains available after each intermediate evaluation.
    Parent circularity remains a categorical choice.
    """
    demand = candidate.demand_lifetime
    circularity = candidate.circularity
    neighbors: dict[str, InterventionCandidate] = {}
    if candidate.optimization_block == "integrated_frontier":
        endpoints = (demand.demand_reduction_pct, demand.lifetime_extension_pct)
        for index, bound in enumerate((90, 100)):
            for level in _line_levels(endpoints[index], bound=bound, step_pct=step_pct):
                changed = list(endpoints)
                changed[index] = level
                item = InterventionCandidate(
                    candidate.optimization_block,
                    DemandLifetimeDecision(changed[0], changed[1], demand.target_year),
                    circularity,
                )
                neighbors[item.identity] = item
    if not circularity.uses_exact_parent:
        collection = cast(int, circularity.collection_pct)
        treatment = circularity.treatment_percentages
        feedstock = circularity.feedstock_percentages
        for level in _line_levels(collection, bound=100, step_pct=step_pct):
            item = _with_circularity_components(
                candidate, collection=level, treatment=treatment, feedstock=feedstock
            )
            neighbors[item.identity] = item
        for index, shares in enumerate((treatment, feedstock)):
            for transferred in _share_transfers(shares, step_pct=step_pct, full_line=True):
                item = _with_circularity_components(
                    candidate,
                    collection=collection,
                    treatment=cast(tuple[int, int, int, int], transferred)
                    if index == 0
                    else treatment,
                    feedstock=cast(tuple[int, int, int], transferred) if index == 1 else feedstock,
                )
                neighbors[item.identity] = item
    return tuple(sorted(neighbors.values(), key=candidate_coordinate_key))


def _line_levels(value: int, *, bound: int, step_pct: int) -> tuple[int, ...]:
    """Return other levels at bounded steps from the current level and exact bounds."""
    return tuple(sorted({0, bound, *range(value % step_pct, bound + 1, step_pct)} - {value}))


def _share_transfers(
    values: tuple[int, ...], *, step_pct: int, full_line: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Transfer a step, or all step multiples and the donor boundary, between shares."""
    results: set[tuple[int, ...]] = set()
    for source, target in product(range(len(values)), repeat=2):
        # The admitted shares sum to 100, so a transfer bounded by the donor
        # also keeps the receiving share at or below 100 percent.
        if source == target:
            continue
        amounts = (
            {*range(step_pct, values[source], step_pct), values[source]}
            if full_line
            else {step_pct}
        )
        for amount in amounts:
            if 0 < amount <= values[source]:
                changed = list(values)
                changed[source] -= amount
                changed[target] += amount
                results.add(tuple(changed))
    return tuple(sorted(results))


def _with_circularity_components(
    candidate: InterventionCandidate,
    *,
    collection: int,
    treatment: tuple[int, int, int, int],
    feedstock: tuple[int, int, int],
) -> InterventionCandidate:
    """Replace a candidate's collection, treatment, and feedstock percentages."""
    circularity = CircularityDecision(
        collection,
        *treatment,
        *feedstock,
        candidate.target_year,
    )
    return InterventionCandidate(
        candidate.optimization_block, candidate.demand_lifetime, circularity
    )


def candidate_coordinate_key(candidate: InterventionCandidate) -> tuple[int | Fraction, ...]:
    """Order endpoints and exact annual controls deterministically before evaluation."""
    circularity = candidate.circularity.percentages()
    return (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        *candidate.demand_lifetime.demand_fraction,
        *candidate.demand_lifetime.lifetime_fraction,
        *(circularity or (-1,) * 8),
        candidate.target_year,
    )


def select_candidate_states(
    evaluations: Iterable[CandidateEvaluation],
) -> CandidateSelection:
    """Select one candidate per block, background, and allocated carrying capacity label.

    A compliant unchanged parent is retained. Otherwise compliant selection
    minimizes the CE intervention distance for the corresponding block, then
    retains the combination whose highest ASR is closest to one.
    Noncompliant fallback combinations are ranked successively by the number of
    fully compliant regions, fully compliant impact categories, compliant
    region-impact category pairs, and individual compliant ASRs. Remaining ties
    minimize maximum positive ASR exceedance, mean positive ASR exceedance, and
    CE intervention distance. Cumulative GWP100 is counted separately from annual
    climate change. Candidate identity resolves exact ties.

    Args:
        evaluations: Candidate calculations with metrics for every requested state.

    Returns:
        The selected candidate and status for each state, plus their identity.
    """

    state_metrics: dict[str, list[CandidateStateMetrics]] = {}
    for evaluation in evaluations:
        for metric in evaluation.state_metrics:
            state_metrics.setdefault(metric.state_identity, []).append(metric)
    selected: list[SelectedCandidateState] = []
    for state_identity, metrics in sorted(state_metrics.items()):
        compliant = [item for item in metrics if item.all_region_joint_compliant]
        if compliant:
            # A varying parent can share an active candidate's target-year
            # endpoint while their earlier paths differ. Retain the compliant
            # unchanged parent before applying ordinary intervention tie rules.
            chosen = min(
                compliant,
                key=lambda item: (
                    not item.is_zero_intervention_parent,
                    item.intervention_distance,
                    cast(float, item.binding_slack),
                    item.candidate_identity,
                ),
            )
            status: SelectionStatus = (
                "baseline_compliant"
                if chosen.is_zero_intervention_parent
                else "intervention_compliant"
            )
        else:
            # Compare coverage before exceedance so partial compliance is ranked
            # consistently across regions and impact targets.
            chosen = min(
                metrics,
                key=lambda item: (
                    -item.compliant_region_count,
                    -item.fully_compliant_impact_target_count,
                    -item.compliant_region_impact_group_count,
                    -item.compliant_coordinate_count,
                    item.maximum_positive_exceedance,
                    item.balanced_mean_positive_exceedance,
                    item.intervention_distance,
                    item.candidate_identity,
                ),
            )
            status = "selected_noncompliant"
        selected.append(
            SelectedCandidateState(
                state_identity=state_identity,
                candidate_identity=chosen.candidate_identity,
                status=status,
            )
        )
    unique = tuple(sorted({item.candidate_identity for item in selected}))
    identity = _SELECTION_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": CANDIDATE_SELECTION_METHOD_IDENTITY,
            "states": [
                [item.state_identity, item.candidate_identity, item.status] for item in selected
            ],
            "selected_candidate_identities": list(unique),
        }
    )
    return CandidateSelection(
        identity=identity,
        states=tuple(selected),
        selected_candidate_identities=unique,
    )


def refine_candidates(
    *,
    optimization_block: OptimizationBlock,
    initial_candidates: Sequence[InterventionCandidate],
    evaluate: Callable[[tuple[InterventionCandidate, ...]], Sequence[CandidateEvaluation]],
    evaluate_competitive: Callable[
        [tuple[InterventionCandidate, ...], float], CompetitiveCandidateBatch
    ],
    endpoint_refinement_steps_pct: tuple[int, ...] = _REFINEMENT_STEPS_PCT,
) -> CandidateRefinement:
    """Refine endpoints with one MFA representative per admitted endpoint.

    A compliant exact parent is terminal. Otherwise retain every starting
    coordinate. Improvements restart refinement. Block II checks complete lines;
    the other blocks check coordinated endpoints after scalar refinement stalls.
    Final unpruned neighbors refer to the same selected endpoint. Failed demand
    repairs prevent the neighborhood from being marked complete.
    """

    state = initialize_candidate_search(
        optimization_block=optimization_block,
        initial_candidates=initial_candidates,
        evaluate=evaluate,
    )
    for step_pct in endpoint_refinement_steps_pct:
        state = advance_candidate_search(
            state,
            step_pct=step_pct,
            unpruned=step_pct <= 5 or step_pct == endpoint_refinement_steps_pct[-1],
            evaluate=evaluate,
            evaluate_competitive=evaluate_competitive,
        )
        if not any(item.status != "baseline_compliant" for item in state.selection.states):
            break
    return finish_candidate_search(
        state, endpoint_refinement_steps_pct=endpoint_refinement_steps_pct
    )


def initialize_candidate_search(
    *,
    optimization_block: OptimizationBlock,
    initial_candidates: Sequence[InterventionCandidate],
    evaluate: Callable[[tuple[InterventionCandidate, ...]], Sequence[CandidateEvaluation]],
) -> CandidateSearchState:
    """Admit the unchanged parent first, then the required complete starting design."""
    candidates = _unique_candidates(initial_candidates)
    parents = tuple(item for item in candidates if item.is_exact_parent)
    evaluations = {item.candidate.identity: item for item in (evaluate(parents) if parents else ())}
    selection = select_candidate_states(evaluations.values())
    reused_candidate_count = 0
    if not selection.states or any(
        item.status != "baseline_compliant" for item in selection.states
    ):
        missing = tuple(item for item in candidates if item.identity not in evaluations)
        reused_candidate_count = len(candidates) - len(missing)
        if missing:
            evaluations.update((item.candidate.identity, item) for item in evaluate(missing))
        selection = select_candidate_states(evaluations.values())
    return CandidateSearchState(
        optimization_block=optimization_block,
        evaluations=tuple(evaluations.values()),
        selection=selection,
        reused_candidate_count=reused_candidate_count,
    )


def advance_candidate_search(
    state: CandidateSearchState,
    *,
    step_pct: int,
    unpruned: bool,
    evaluate: Callable[[tuple[InterventionCandidate, ...]], Sequence[CandidateEvaluation]],
    evaluate_competitive: Callable[
        [tuple[InterventionCandidate, ...], float], CompetitiveCandidateBatch
    ],
) -> CandidateSearchState:
    """Complete one spacing using its unchanged full-request pruning policy.

    ``unpruned`` is true for spacings at most five and the request's final
    spacing. Reuse by a longer request requires the same pruning policy in every completed stage.
    """
    optimization_block = state.optimization_block
    evaluations = {item.candidate.identity: item for item in state.evaluations}
    reused_candidate_count = state.reused_candidate_count

    def evaluate_missing(batch: tuple[InterventionCandidate, ...]) -> tuple[str, ...]:
        """Evaluate newly proposed candidates in this search."""
        nonlocal reused_candidate_count
        missing = tuple(item for item in batch if item.identity not in evaluations)
        reused_candidate_count += len(batch) - len(missing)
        if missing:
            observed = tuple(evaluate(missing))
            evaluations.update((item.candidate.identity, item) for item in observed)
        return tuple(item.identity for item in missing)

    def evaluate_competitive_candidates(
        batch: tuple[InterventionCandidate, ...],
        *,
        distance_ceiling: float,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Evaluate candidates within the distance ceiling and record candidates skipped."""
        nonlocal reused_candidate_count
        missing = tuple(item for item in batch if item.identity not in evaluations)
        reused_candidate_count += len(batch) - len(missing)
        if not missing:
            return (), ()
        observed_batch = evaluate_competitive(missing, distance_ceiling)
        distance_records = observed_batch.candidate_distances
        distance_by_candidate = dict(distance_records)
        retained = {
            identity
            for identity, distance in distance_by_candidate.items()
            if distance <= distance_ceiling
        }
        observed = observed_batch.evaluations
        evaluations.update((item.candidate.identity, item) for item in observed)
        newly_evaluated = tuple(item.identity for item in missing if item.identity in retained)
        pruned = tuple(item.identity for item in missing if item.identity not in retained)
        return newly_evaluated, pruned

    selection = state.selection
    passes = list(state.passes)
    final_neighbors: tuple[str, ...] = ()
    final_anchors: tuple[str, ...] = ()
    final_selection_stable = False
    iteration = 0
    coordinated = False
    while True:
        iteration += 1
        print(
            f"run_optimization: {optimization_block}: step={step_pct} "
            f"iteration={iteration} neighborhood={'coordinated' if coordinated else 'scalar'}",
            flush=True,
        )
        active_candidate_identities = tuple(
            sorted(
                {
                    item.candidate_identity
                    for item in selection.states
                    if item.status != "baseline_compliant"
                }
            )
        )
        if not active_candidate_identities:
            break
        anchors = tuple(evaluations[identity].candidate for identity in active_candidate_identities)
        proposed = _unique_candidates(
            neighbor
            for anchor in anchors
            for neighbor in (
                line_candidates(anchor, step_pct=step_pct)
                if optimization_block == "circularity_feedstock_anchor"
                else neighboring_candidates(anchor, step_pct=step_pct, coordinated=coordinated)
            )
        )
        distance_ceiling = (
            None if unpruned else _competitive_distance_ceiling(selection, evaluations=evaluations)
        )
        if distance_ceiling is None:
            newly_evaluated = evaluate_missing(proposed)
            pruned: tuple[str, ...] = ()
        else:
            newly_evaluated, pruned = evaluate_competitive_candidates(
                proposed, distance_ceiling=distance_ceiling
            )
        next_selection = select_candidate_states(evaluations.values())
        passes.append(
            RefinementPass(
                step_pct=step_pct,
                iteration=iteration,
                anchor_candidate_identities=tuple(item.identity for item in anchors),
                proposed_candidate_identities=tuple(item.identity for item in proposed),
                pruned_candidate_identities=pruned,
                newly_evaluated_candidate_identities=newly_evaluated,
                selected_candidate_identities=next_selection.selected_candidate_identities,
                coordinated=coordinated,
            )
        )
        changed = next_selection.states != selection.states
        selection = next_selection
        if changed:
            coordinated = False
            continue
        if not coordinated and optimization_block != "circularity_feedstock_anchor":
            coordinated = True
            continue
        final_neighbors = tuple(item.identity for item in proposed)
        final_anchors = active_candidate_identities
        final_selection_stable = True
        break
    return CandidateSearchState(
        optimization_block=optimization_block,
        evaluations=tuple(evaluations.values()),
        selection=selection,
        completed_stages=(*state.completed_stages, CandidateRefinementStage(step_pct, unpruned)),
        passes=tuple(passes),
        reused_candidate_count=reused_candidate_count,
        final_neighbor_candidate_identities=final_neighbors,
        final_anchor_candidate_identities=final_anchors,
        final_selection_stable=final_selection_stable,
    )


def finish_candidate_search(
    state: CandidateSearchState,
    *,
    endpoint_refinement_steps_pct: tuple[int, ...],
) -> CandidateRefinement:
    """Describe the selection with verification tied to the requested final stage."""
    optimization_block = state.optimization_block
    evaluations = {item.candidate.identity: item for item in state.evaluations}
    selection = state.selection
    passes = state.passes
    final_stage = bool(
        state.completed_stages
        and endpoint_refinement_steps_pct
        and state.completed_stages[-1].step_pct == endpoint_refinement_steps_pct[-1]
    )
    final_neighbors = state.final_neighbor_candidate_identities if final_stage else ()
    final_anchors = state.final_anchor_candidate_identities if final_stage else ()
    final_selection_stable = state.final_selection_stable and final_stage
    unpruned = final_stage and state.completed_stages[-1].unpruned
    verification = tuple(
        _state_verification(
            item,
            evaluations=evaluations,
            selection_stable=final_selection_stable,
            final_anchors=final_anchors,
            final_neighbors=final_neighbors,
            unpruned=unpruned,
        )
        for item in selection.states
    )
    unsuccessful_path_search_count = sum(item.repair_unresolved for item in evaluations.values())
    payload = {
        "method_identity": CANDIDATE_REFINEMENT_METHOD_IDENTITY,
        "endpoint_refinement_steps_pct": list(endpoint_refinement_steps_pct),
        "optimization_block": optimization_block,
        "selection_identity": selection.identity,
        "evaluation_identities": sorted(evaluations),
        "passes": [
            {
                "step_pct": item.step_pct,
                "iteration": item.iteration,
                "coordinated": item.coordinated,
                "anchors": list(item.anchor_candidate_identities),
                "proposals": list(item.proposed_candidate_identities),
                "pruned": list(item.pruned_candidate_identities),
                "new": list(item.newly_evaluated_candidate_identities),
                "selected": list(item.selected_candidate_identities),
            }
            for item in passes
        ],
        "final_neighbors": list(final_neighbors),
        "unsuccessful_path_search_count": unsuccessful_path_search_count,
        "state_verification": [
            [
                item.state_identity,
                item.candidate_identity,
                item.selected_is_exact_parent,
                item.selected_has_non_grid_baseline_component,
                item.declared_neighborhood_verified,
                item.selection_stable,
            ]
            for item in verification
        ],
    }
    return CandidateRefinement(
        identity=_REFINEMENT_IDENTITY_PREFIX + payload_sha256(payload),
        optimization_block=optimization_block,
        evaluation_count=len(evaluations),
        reused_candidate_count=state.reused_candidate_count,
        selection=selection,
        state_verification=verification,
        passes=tuple(passes),
        final_neighbor_candidate_identities=final_neighbors,
        unsuccessful_path_search_count=unsuccessful_path_search_count,
    )


def _competitive_distance_ceiling(
    selection: CandidateSelection,
    *,
    evaluations: Mapping[str, CandidateEvaluation],
) -> float | None:
    """Return the largest compliant distance, or leave the current neighborhood unpruned."""

    if any(item.status == "selected_noncompliant" for item in selection.states):
        return None
    distances: list[float] = []
    for selected in selection.states:
        evaluation = evaluations[selected.candidate_identity]
        metric = next(
            item
            for item in evaluation.state_metrics
            if item.state_identity == selected.state_identity
        )
        distances.append(metric.intervention_distance)
    return max(distances)


def _state_verification(
    selected: SelectedCandidateState,
    *,
    evaluations: Mapping[str, CandidateEvaluation],
    selection_stable: bool,
    final_anchors: tuple[str, ...],
    final_neighbors: tuple[str, ...],
    unpruned: bool,
) -> CandidateStateCheck:
    """Bind verification to the exact anchors and expanded final unpruned pass."""
    evaluation = evaluations[selected.candidate_identity]
    candidate = evaluation.candidate
    exact_parent = candidate.is_exact_parent
    non_grid_baseline = (
        candidate.optimization_block == "integrated_frontier"
        and candidate.circularity.uses_exact_parent
    )
    locally_verified = (
        selection_stable
        and unpruned
        and selected.status != "baseline_compliant"
        and not non_grid_baseline
        and selected.candidate_identity in final_anchors
        and bool(final_neighbors)
        and set(final_neighbors).issubset(evaluations)
        and not any(evaluations[identity].repair_unresolved for identity in final_neighbors)
    )
    return CandidateStateCheck(
        state_identity=selected.state_identity,
        candidate_identity=selected.candidate_identity,
        selected_is_exact_parent=exact_parent,
        selected_has_non_grid_baseline_component=non_grid_baseline,
        declared_neighborhood_verified=locally_verified,
        selection_stable=selection_stable,
    )


def _unique_candidates(
    candidates: Iterable[InterventionCandidate],
) -> tuple[InterventionCandidate, ...]:
    """Deduplicate candidate identities and sort by endpoints and annual controls."""
    unique: dict[str, InterventionCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.identity, candidate)
    return tuple(sorted(unique.values(), key=candidate_coordinate_key))
