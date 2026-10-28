"""Algebraic joint endpoint coverage and finite, share-preserving compromise routes.

These designs prescribe requested endpoints. Each endpoint then receives an MFA
feasibility check and, when admitted, an environmental assessment. Parent
circularity remains a separate category.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Final, Literal, TypeAlias, cast

from pytoce.run_optimization.calculation.candidate_search import InterventionCandidate
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    initial_demand_lifetime_decisions,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.shared.runtime.filesystem import payload_sha256

INTEGRATED_DESIGN_METHOD_IDENTITY: Final = "pytoce_integrated_endpoint_design_v2"
EndpointCoordinates: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int]
RouteFamily: TypeAlias = Literal[
    "circularity",
    "demand",
    "lifetime",
    "demand_lifetime",
    "demand_circularity",
    "lifetime_circularity",
    "demand_lifetime_circularity",
]
_PairRequirement: TypeAlias = tuple[int, int, int, int]
_RequestCoordinates: TypeAlias = tuple[int, int, tuple[int, ...] | None]
_PARTITIONS = ((3, 4, 5, 6), (7, 8, 9))
_ROUTE_PROJECTIONS: tuple[tuple[RouteFamily, bool, bool, bool], ...] = (
    ("circularity", False, False, True),
    ("demand", True, False, False),
    ("lifetime", False, True, False),
    ("demand_lifetime", True, True, False),
    ("demand_circularity", True, False, True),
    ("lifetime_circularity", False, True, True),
    ("demand_lifetime_circularity", True, True, True),
)
BASE_PROPOSAL_FAMILIES: Final = (
    "parent_circularity",
    "actual_ii",
    "direct_combination",
    "coarse_demand_lifetime_by_ii",
    "independent_pair_cover",
    "actual_i_by_represented_circularity",
)
ROUTE_PROPOSAL_FAMILIES: tuple[RouteFamily, ...] = tuple(
    projection[0] for projection in _ROUTE_PROJECTIONS
)


@dataclass(frozen=True, slots=True)
class JointPairCover:
    """Cover every algebraically compatible pair through progressive construction."""

    rows: tuple[EndpointCoordinates, ...]
    requirement_count: int
    covered_requirement_count: int
    initial_grid_step_pct: int


@dataclass(frozen=True, slots=True)
class JointRoute:
    """Record a deduplicated route and every geometric guide/family that requested it.

    Points exclude the source and include the exact target. The geometric sequence retains all
    proposed points through its target.
    """

    source: EndpointCoordinates
    target: EndpointCoordinates
    points: tuple[EndpointCoordinates, ...]
    origins: tuple[tuple[EndpointCoordinates, RouteFamily], ...]
    step_pct: int

    @property
    def identity(self) -> str:
        """Bind the oriented segment, initial spacing and exact movement rule."""
        return (
            INTEGRATED_DESIGN_METHOD_IDENTITY
            + "::"
            + payload_sha256(
                {"source": self.source, "target": self.target, "step_pct": self.step_pct}
            )
        )


@dataclass(frozen=True, slots=True)
class JointStageDesign:
    """Record a stage base or its completed role-guided route geometry and origins."""

    identity: str
    step_pct: int
    candidates: tuple[InterventionCandidate, ...]
    starting_candidate_identities: tuple[str, ...]
    source_candidate_identities: tuple[str, ...]
    geometric_guide_candidate_identities: tuple[str, ...]
    actual_ii_candidates: tuple[InterventionCandidate, ...]
    routes: tuple[JointRoute, ...]
    proposal_origins: tuple[tuple[str, tuple[str, ...]], ...]
    pair_cover: JointPairCover
    explicit_ii_routes_absent: bool
    actual_i_pairs: tuple[tuple[int, int], ...]


def joint_pair_cover(*, initial_grid_step_pct: int) -> JointPairCover:
    """Construct the canonical implicit pair-cover over the admitted coarse grid.

    A row starts with the first uncovered compatible pair. Each subsequent
    field/value choice maximizes its additional uncovered pairs with assigned
    fields, breaking ties by field then value, subject to exact simplex completion.
    The public input owner admits the grid spacing; the construction adds compatible pairs
    progressively.
    """
    decisions = initial_demand_lifetime_decisions(target_year=2050, step_pct=initial_grid_step_pct)
    domains = (
        tuple(sorted({item.demand_reduction_pct for item in decisions})),
        tuple(sorted({item.lifetime_extension_pct for item in decisions})),
        *((tuple(range(0, 101, initial_grid_step_pct)),) * 8),
    )
    sums: list[frozenset[int]] = [frozenset({0})]
    for _ in range(4):
        sums.append(frozenset(a + b for a in sums[-1] for b in domains[2] if a + b <= 100))
    requirements: set[_PairRequirement] = set()
    for left, right in combinations(range(10), 2):
        for left_value, right_value in product(domains[left], domains[right]):
            partial = {left: left_value, right: right_value}
            if _share_completion_possible(partial, sums):
                requirements.add((left, left_value, right, right_value))
    uncovered = requirements.copy()
    rows: list[EndpointCoordinates] = []
    while uncovered:
        left, left_value, right, right_value = min(uncovered)
        assigned = {left: left_value, right: right_value}
        while len(assigned) < 10:
            choices: list[tuple[int, int, int]] = []
            for field in range(10):
                if field in assigned:
                    continue
                for value in domains[field]:
                    proposal = assigned | {field: value}
                    if _share_completion_possible(proposal, sums):
                        gain = sum(
                            _ordered_pair(field, value, other, other_value) in uncovered
                            for other, other_value in assigned.items()
                        )
                        choices.append((-gain, field, value))
            _, field, value = min(choices)
            assigned[field] = value
        row = cast(EndpointCoordinates, tuple(assigned[field] for field in range(10)))
        rows.append(row)
        uncovered.difference_update(_row_pairs(row))
    return JointPairCover(tuple(rows), len(requirements), len(requirements), initial_grid_step_pct)


def _share_completion_possible(
    assigned: dict[int, int], achievable_sums: Sequence[frozenset[int]]
) -> bool:
    """Check each partial partition against the exact sums of its remaining levels."""
    for partition in _PARTITIONS:
        fixed = [assigned[field] for field in partition if field in assigned]
        if 100 - sum(fixed) not in achievable_sums[len(partition) - len(fixed)]:
            return False
    return True


def _ordered_pair(left: int, value: int, right: int, other: int) -> _PairRequirement:
    """Place a pair of distinct assigned fields in canonical field order."""
    return (left, value, right, other) if left < right else (right, other, left, value)


def _row_pairs(row: EndpointCoordinates) -> Iterable[_PairRequirement]:
    """Yield the two-field requirements represented by one complete endpoint row."""
    return ((left, row[left], right, row[right]) for left, right in combinations(range(10), 2))


def explicit_endpoint_coordinates(candidate: InterventionCandidate) -> EndpointCoordinates:
    """Read the admitted explicit endpoint controls."""
    return (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        *cast(tuple[int, int, int, int, int, int, int, int], candidate.circularity.percentages()),
    )


def _move_partition(
    source: tuple[int, ...], target: tuple[int, ...], step_pct: int
) -> tuple[int, ...]:
    """Transfer at most one step of total share in canonical donor/recipient order."""
    values = list(source)
    remaining = step_pct
    for donor, recipient in product(range(len(values)), repeat=2):
        transfer = min(
            max(values[donor] - target[donor], 0),
            max(target[recipient] - values[recipient], 0),
            remaining,
        )
        values[donor] -= transfer
        values[recipient] += transfer
        remaining -= transfer
    return tuple(values)


def _bounded_scalar(source: int, target: int, step_pct: int) -> int:
    """Move toward a scalar target by at most its remaining distance."""
    return (
        source + min(step_pct, target - source)
        if target >= source
        else source - min(step_pct, source - target)
    )


def move_joint_coordinates(
    source: EndpointCoordinates, target: EndpointCoordinates, *, step_pct: int
) -> EndpointCoordinates:
    """Simultaneously advance all scalars and both exact-share partitions."""
    return cast(
        EndpointCoordinates,
        (
            *(_bounded_scalar(source[index], target[index], step_pct) for index in range(3)),
            *_move_partition(source[3:7], target[3:7], step_pct),
            *_move_partition(source[7:10], target[7:10], step_pct),
        ),
    )


def guided_joint_candidate(
    candidate: InterventionCandidate, guide: InterventionCandidate, *, step_pct: int
) -> InterventionCandidate | None:
    """Propose one explicit guided move through the ordinary linear endpoint owner.

    The search owner supplies explicit candidates only; parent circularity is a separate categorical
    choice. Return ``None`` for a no-op.
    """
    source = explicit_endpoint_coordinates(candidate)
    moved = move_joint_coordinates(source, explicit_endpoint_coordinates(guide), step_pct=step_pct)
    if moved == source:
        return None
    return InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(moved[0], moved[1], candidate.target_year),
        CircularityDecision(*moved[2:], candidate.target_year),
    )


def _request_key(values: _RequestCoordinates) -> tuple[int, ...]:
    """Order categorical parent controls before explicit circularity endpoints."""
    return (values[0], values[1], *((-1,) * 8 if values[2] is None else values[2]))


def _as_request(row: EndpointCoordinates) -> _RequestCoordinates:
    """Separate scalar endpoints from the eight explicit circularity percentages."""
    return (row[0], row[1], row[2:])


def _requested_coordinates(candidate: InterventionCandidate) -> _RequestCoordinates:
    """Read endpoint requests while preserving categorical parent circularity."""
    return (
        candidate.demand_lifetime.demand_reduction_pct,
        candidate.demand_lifetime.lifetime_extension_pct,
        candidate.circularity.percentages(),
    )


def _projected_segments(
    sources: Sequence[EndpointCoordinates], guides: Sequence[EndpointCoordinates]
) -> Iterable[tuple[EndpointCoordinates, EndpointCoordinates, EndpointCoordinates, RouteFamily]]:
    """Yield seven hybrid segments; reversal later preserves each fixed-control face."""
    for source, guide in product(sources, guides):
        for family, demand, lifetime, circularity in _ROUTE_PROJECTIONS:
            target = cast(
                EndpointCoordinates,
                (
                    guide[0] if demand else source[0],
                    guide[1] if lifetime else source[1],
                    *(guide[2:] if circularity else source[2:]),
                ),
            )
            yield source, target, guide, family


def _boundary_segments(
    sources: Sequence[EndpointCoordinates],
) -> Iterable[tuple[EndpointCoordinates, EndpointCoordinates, EndpointCoordinates, RouteFamily]]:
    """Reach collection bounds and each simplex donor-empty boundary independently."""
    for source in sources:
        for collection in (0, 100):
            target = cast(EndpointCoordinates, (*source[:2], collection, *source[3:]))
            yield source, target, target, "circularity"
        for partition in _PARTITIONS:
            for donor, recipient in product(partition, repeat=2):
                if donor == recipient:
                    continue
                values = list(source)
                values[recipient] += values[donor]
                values[donor] = 0
                target = cast(EndpointCoordinates, tuple(values))
                yield source, target, target, "circularity"


def _own_ii_segments(
    sources: Sequence[EndpointCoordinates], actual_i_pairs: Sequence[tuple[int, int]]
) -> Iterable[tuple[EndpointCoordinates, EndpointCoordinates, EndpointCoordinates, RouteFamily]]:
    """Start each demand/lifetime axis at II itself, including off-grid I endpoints."""
    for source in sources:
        if source[:2] != (0, 0):
            continue
        for demand, lifetime in actual_i_pairs:
            for family, pair in (
                ("demand", (demand, 0)),
                ("lifetime", (0, lifetime)),
                ("demand_lifetime", (demand, lifetime)),
            ):
                target = cast(EndpointCoordinates, (*pair, *source[2:]))
                yield source, target, target, cast(RouteFamily, family)


def _finite_routes(
    sources: Sequence[EndpointCoordinates],
    guides: Sequence[EndpointCoordinates],
    *,
    step_pct: int,
    actual_i_pairs: Sequence[tuple[int, int]],
    previous_routes: Sequence[JointRoute],
) -> tuple[JointRoute, ...]:
    """Keep cumulative complete segments and add only new initial-spacing routes.

    Both directions traverse the exact same projected segment. Previously
    completed points are retained rather than regenerated at a finer spacing.
    Origins may grow when a new scientific family requests an existing segment.
    """
    previous = {(route.source, route.target): route for route in previous_routes}
    origins = {key: set(route.origins) for key, route in previous.items()}
    segments = (
        _projected_segments(sources, guides),
        _boundary_segments(sources),
        _own_ii_segments(sources, actual_i_pairs),
    )
    for family_segments in segments:
        for source, target, guide, family in family_segments:
            if target == source:
                continue
            for endpoints in ((source, target), (target, source)):
                origins.setdefault(endpoints, set()).add((guide, family))
    routes: list[JointRoute] = []
    for (source, target), route_origins in sorted(origins.items()):
        ordered_origins = tuple(sorted(route_origins))
        if (source, target) in previous:
            routes.append(replace(previous[source, target], origins=ordered_origins))
            continue
        current = source
        points: list[EndpointCoordinates] = []
        while current != target:
            current = move_joint_coordinates(current, target, step_pct=step_pct)
            points.append(current)
        routes.append(JointRoute(source, target, tuple(points), ordered_origins, step_pct))
    return tuple(routes)


def _candidate_catalog(
    coordinates: Iterable[_RequestCoordinates], *, target_year: int
) -> dict[_RequestCoordinates, InterventionCandidate]:
    """Build fresh linear requests once per coordinate, sharing scalar trajectories."""
    demand_decisions: dict[tuple[int, int], DemandLifetimeDecision] = {}
    candidates: dict[_RequestCoordinates, InterventionCandidate] = {}
    for values in sorted(coordinates, key=_request_key):
        pair = values[:2]
        if pair not in demand_decisions:
            demand_decisions[pair] = DemandLifetimeDecision(*pair, target_year)
        circularity = (
            CircularityDecision.parent(target_year=target_year)
            if values[2] is None
            else CircularityDecision(
                *cast(tuple[int, int, int, int, int, int, int, int], values[2]), target_year
            )
        )
        candidates[values] = InterventionCandidate(
            "integrated_frontier", demand_decisions[pair], circularity
        )
    return candidates


def build_joint_stage_design(
    *,
    block_i_candidates: Sequence[InterventionCandidate],
    block_ii_candidates: Sequence[InterventionCandidate],
    pair_cover: JointPairCover,
    step_pct: int,
    target_year: int = 2050,
) -> JointStageDesign:
    """Freeze the complete algebraic base before admission and rank-selected guides.

    The caller supplies cumulative actual I/II anchors. The supplied anchors define the represented
    circularity settings. Every requested trajectory
    is freshly linear, even when an anchor's realized path needed demand repair.
    """
    coarse_pairs = {
        (item.demand_reduction_pct, item.lifetime_extension_pct)
        for item in initial_demand_lifetime_decisions(
            target_year=target_year, step_pct=pair_cover.initial_grid_step_pct
        )
    }
    actual_i = {
        (item.demand_lifetime.demand_reduction_pct, item.demand_lifetime.lifetime_extension_pct)
        for item in block_i_candidates
    }
    actual_ii = {_requested_coordinates(item) for item in block_ii_candidates}
    explicit_ii = {values[2] for values in actual_ii if values[2] is not None}
    sources = {
        cast(EndpointCoordinates, (*pair, *circularity))
        for pair, circularity in product(actual_i | {(0, 0)}, explicit_ii)
    }
    independent = set(pair_cover.rows) | {
        cast(EndpointCoordinates, (0, 0, *row[2:])) for row in pair_cover.rows
    }
    represented_circularity = {row[2:] for row in pair_cover.rows} | explicit_ii
    origins: dict[_RequestCoordinates, set[str]] = {}

    def add(values: _RequestCoordinates, family: str) -> None:
        """Retain every scientific family contributing a duplicate endpoint request."""
        origins.setdefault(values, set()).add(family)

    for pair in coarse_pairs | actual_i:
        add((*pair, None), "parent_circularity")
    for values in actual_ii:
        add(values, "actual_ii")
    for pair, circularity in product(actual_i, {values[2] for values in actual_ii}):
        add((*pair, circularity), "direct_combination")
    for pair, circularity in product(coarse_pairs, {values[2] for values in actual_ii}):
        add((*pair, circularity), "coarse_demand_lifetime_by_ii")
    for row in independent:
        add(_as_request(row), "independent_pair_cover")
    for pair, circularity in product(actual_i, represented_circularity):
        add((*pair, circularity), "actual_i_by_represented_circularity")
    candidates = _candidate_catalog(origins, target_year=target_year)
    identity = (
        INTEGRATED_DESIGN_METHOD_IDENTITY
        + "::"
        + payload_sha256(
            {
                "step_pct": step_pct,
                "target_year": target_year,
                "pair_cover": pair_cover.rows,
                "initial_grid_step_pct": pair_cover.initial_grid_step_pct,
                "actual_i": sorted(actual_i),
                "actual_ii": sorted(actual_ii, key=_request_key),
            }
        )
    )
    return JointStageDesign(
        identity=identity,
        step_pct=step_pct,
        candidates=tuple(candidates.values()),
        starting_candidate_identities=tuple(item.identity for item in candidates.values()),
        source_candidate_identities=tuple(
            candidates[_as_request(row)].identity for row in sorted(sources)
        ),
        geometric_guide_candidate_identities=(),
        actual_ii_candidates=tuple(
            candidates[item] for item in sorted(actual_ii, key=_request_key)
        ),
        routes=(),
        proposal_origins=tuple(
            (candidate.identity, tuple(sorted(origins[values])))
            for values, candidate in candidates.items()
        ),
        pair_cover=pair_cover,
        explicit_ii_routes_absent=not explicit_ii,
        actual_i_pairs=tuple(sorted(actual_i)),
    )


def complete_joint_stage_design(
    design: JointStageDesign,
    *,
    guides: Sequence[InterventionCandidate],
    previous_routes: Sequence[JointRoute] = (),
) -> JointStageDesign:
    """Freeze all coarse route proposals after the search owner selects role guides.

    Guides are explicit admitted base candidates. Every endpoint and intermediate
    point of each nontrivial cumulative route retains all route-family origins,
    regardless of MFA validation or its rank. Only new segments are generated;
    completed routes retain their original spacing.
    """
    by_identity = {item.identity: item for item in design.candidates}
    source_rows = tuple(
        explicit_endpoint_coordinates(by_identity[key])
        for key in design.source_candidate_identities
    )
    guide_rows = tuple(sorted({explicit_endpoint_coordinates(item) for item in guides}))
    routes = _finite_routes(
        source_rows,
        guide_rows,
        step_pct=design.pair_cover.initial_grid_step_pct,
        actual_i_pairs=design.actual_i_pairs,
        previous_routes=previous_routes,
    )
    origins = {
        _requested_coordinates(by_identity[key]): set(families)
        for key, families in design.proposal_origins
    }
    for route in routes:
        for point in (route.source, *route.points):
            origins.setdefault(_as_request(point), set()).update(
                family for _, family in route.origins
            )
    candidates = _candidate_catalog(origins, target_year=design.candidates[0].target_year)
    guide_identities = tuple(candidates[_as_request(row)].identity for row in guide_rows)
    identity = (
        INTEGRATED_DESIGN_METHOD_IDENTITY
        + "::"
        + payload_sha256(
            {
                "base_identity": design.identity,
                "guides": guide_rows,
                "routes": [(route.identity, route.origins) for route in routes],
            }
        )
    )
    return replace(
        design,
        identity=identity,
        candidates=tuple(candidates.values()),
        geometric_guide_candidate_identities=guide_identities,
        routes=routes,
        proposal_origins=tuple(
            (candidate.identity, tuple(sorted(origins[values])))
            for values, candidate in candidates.items()
        ),
    )
