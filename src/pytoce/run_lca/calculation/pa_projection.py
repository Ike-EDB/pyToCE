"""Project actual PA receiving mixtures through their dated native inventories.

Material receiving precedes temporal weighting and LCIA. This owner retains
activities and source context, and groups each function's native quantities.
"""

import json
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import fsum

from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from .pa_event_receiving import (
    PaReceivingEvent,
    PaReceivingRequest,
    PaReceivingResult,
    iter_pa_receiving,
    iter_pa_receiving_requests,
)
from .pa_receiving import PaDisposalEvent, PaReceivingAmount, PaRecyclingEvent, PaSpinningEvent


@dataclass(frozen=True, slots=True)
class PaProjectedEvent:
    """A complete actual receiving result before its stated temporal share.

    The activity includes any dump-class allocation. Receiving amounts
    and mass-balance ledgers have that same unweighted basis. ``temporal_share``
    applies once when entering the dated native inventory or response score.
    """

    event: PaReceivingEvent
    infiltration_class: str
    source_anchor_year: int
    source_database: str
    matrix_database: str
    temporal_share: float
    receiving: PaReceivingResult


@dataclass(frozen=True, slots=True)
class PaProjectedRequest:
    """An actual routed mixture and temporal context before receiving."""

    receiving_request: PaReceivingRequest
    source_anchor_year: int
    source_database: str
    matrix_database: str
    temporal_share: float


@dataclass(frozen=True, slots=True)
class _PaProjectionBatch:
    """Active mixtures at one anchor, with their original disposal positions."""

    source_anchor_year: int
    spinning_events: tuple[PaSpinningEvent, ...]
    recycling_events: tuple[PaRecyclingEvent, ...]
    disposal_events: tuple[tuple[int, PaDisposalEvent], ...]
    temporal_shares: dict[int, float]


@dataclass(frozen=True, slots=True)
class PaReceivingGroup:
    """One function's native quantity and its original attribution members.

    Numerical characterization uses the endpoint and accurately summed amount.
    The original rows remain available to inventory and carbon consumers.
    """

    receiving_function: str
    endpoint: PaReceivingActivity | PaReceivingFlow
    amount: float
    members: list[PaReceivingAmount]


@dataclass(frozen=True, slots=True)
class PaFunctionAmount:
    """One function's summed native-unit quantity before temporal weighting.

    Principal amounts are the portions attributable to fossil, biogenic and
    atmospheric-DAC principal carbon, in the endpoint's unit. These fields attribute the same
    inventory quantity by carbon origin.
    Source records identify equations, coefficients and dated operations.
    Native distributions remain on the receiving rows; the aggregate sum
    is deterministic.
    """

    receiving_function: str
    endpoint: PaReceivingActivity | PaReceivingFlow
    amount: float
    principal_amounts: tuple[float, float, float]
    source_records_json: str


def pa_source_weights(year: int, anchors: tuple[int, ...]) -> tuple[tuple[int, float], ...]:
    """Map an event year to the admitted general-background anchors.

    ``anchors`` is the prepared, increasing set starting at 2025 and ending at
    2100. Ecoinvent supplies all years through 2025. During 2026--2029 its 2025
    reference interpolates to the matching premise 2030 anchor; later intervals
    use that same member's anchors. The receiving calculation uses the ordinary background for each
    activity year. Fractions are elapsed years
    divided by the interval duration, as in annual linear Timex interpolation.
    """
    if year <= anchors[0]:
        return ((anchors[0], 1.0),)
    index = bisect_left(anchors, year)
    upper = anchors[index]
    if year == upper:
        return ((upper, 1.0),)
    lower = anchors[index - 1]
    upper_share = (year - lower) / (upper - lower)
    return ((lower, 1.0 - upper_share), (upper, upper_share))


def _iter_pa_projection_batches(
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[PaDisposalEvent, ...],
    sources_by_year: Mapping[int, PaReceivingSources],
) -> Iterator[_PaProjectionBatch]:
    """Select active anchors."""
    anchors = tuple(sorted(year for year in sources_by_year if year >= 2025))
    weights = {
        year: dict(pa_source_weights(year, anchors))
        for year in {
            event.year for event in (*spinning_events, *recycling_events, *disposal_events)
        }
    }
    for anchor in anchors:
        shares = {year: values[anchor] for year, values in weights.items() if anchor in values}
        yield _PaProjectionBatch(
            anchor,
            tuple(event for event in spinning_events if event.year in shares),
            tuple(event for event in recycling_events if event.year in shares),
            tuple(
                (index, event)
                for index, event in enumerate(disposal_events)
                if event.year in shares
            ),
            shares,
        )


def iter_pa_projection_requests(
    *,
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[PaDisposalEvent, ...],
    reference: PaReceivingReference,
    sources_by_year: Mapping[int, PaReceivingSources],
    matrix_by_year: Mapping[int, str],
    dump_shares: Mapping[str, float],
) -> Iterator[PaProjectedRequest]:
    """Yield routed requests in the complete projection's anchor/event order.

    This record prepares the receiving inputs. Optimization can reuse an identical original-only
    disposal result before evaluation; dated functions still receive each
    actual mixture against their own source.
    """
    for batch in _iter_pa_projection_batches(
        spinning_events, recycling_events, disposal_events, sources_by_year
    ):
        anchor = batch.source_anchor_year
        source = sources_by_year[anchor]
        for request in iter_pa_receiving_requests(
            spinning_events=batch.spinning_events,
            recycling_events=batch.recycling_events,
            disposal_events=batch.disposal_events,
            reference=reference,
            dump_shares=dump_shares,
        ):
            yield PaProjectedRequest(
                request,
                anchor,
                source.general_background_database,
                matrix_by_year[anchor],
                batch.temporal_shares[request.event.year],
            )


def iter_pa_projection(
    *,
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[PaDisposalEvent, ...],
    reference: PaReceivingReference,
    sources_by_year: Mapping[int, PaReceivingSources],
    matrix_by_year: Mapping[int, str],
    dump_shares: Mapping[str, float],
) -> Iterator[PaProjectedEvent]:
    """Evaluate each actual mixture at its active source anchors exactly once.

    The caller supplies completed sources and provenance endpoints. Ordinary LCA and optimization
    use these same
    results. Candidate processing uses the retained parent MFA and prepared endpoint responses.
    """
    for batch in _iter_pa_projection_batches(
        spinning_events, recycling_events, disposal_events, sources_by_year
    ):
        anchor = batch.source_anchor_year
        source = sources_by_year[anchor]
        for event, infiltration, receiving in iter_pa_receiving(
            spinning_events=batch.spinning_events,
            recycling_events=batch.recycling_events,
            disposal_events=tuple(event for _, event in batch.disposal_events),
            reference=reference,
            sources=source,
            spinning_providers=source.spinning_providers,
            dump_shares=dump_shares,
        ):
            yield PaProjectedEvent(
                event,
                infiltration,
                anchor,
                source.general_background_database,
                matrix_by_year[anchor],
                batch.temporal_shares[event.year],
                receiving,
            )


def group_pa_receiving_quantities(
    amounts: tuple[PaReceivingAmount, ...],
) -> Iterator[PaReceivingGroup]:
    """Sum duplicate endpoints within each already calculated process.

    Call separately for each actual event, dump class and source anchor. Native
    product/flow keys and function identity define the groups. Different source citations and carbon
    origins may describe the same quantity. Sum that quantity before calculating its absolute
    contribution.
    Signed and explicit-zero sums retain their endpoint. Original row order
    within each canonical function/endpoint group is unchanged for ``fsum``
    and for the inventory owner's subsequent attribution and provenance.
    """
    groups: dict[tuple[str, tuple[str, str]], list[PaReceivingAmount]] = {}
    for row in amounts:
        groups.setdefault((row.function, row.endpoint.key), []).append(row)
    for (function, _key), rows in sorted(groups.items()):
        yield PaReceivingGroup(function, rows[0].endpoint, fsum(row.amount for row in rows), rows)


def group_pa_receiving_amounts(
    amounts: tuple[PaReceivingAmount, ...],
) -> Iterator[PaFunctionAmount]:
    """Add source records and principal-origin attribution to grouped quantities.

    Ordinary inventories and carbon reporting retain their full provenance.
    Numerical characterization uses the same grouping and amount sum.
    """
    for group in group_pa_receiving_quantities(amounts):
        rows = group.members
        sources = {row.source.key: row.source for row in rows}
        records = [
            {
                "database": source.key[0],
                "code": source.key[1],
                "name": source.name,
                "location": source.location,
                "unit": source.unit,
            }
            for _, source in sorted(sources.items())
        ]
        principal = tuple(
            fsum(
                row.amount * row.principal_carbon_origin_fractions[origin]
                for row in rows
                if row.principal_carbon_origin_fractions is not None
            )
            for origin in range(3)
        )
        yield PaFunctionAmount(
            group.receiving_function,
            group.endpoint,
            group.amount,
            (principal[0], principal[1], principal[2]),
            json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        )
