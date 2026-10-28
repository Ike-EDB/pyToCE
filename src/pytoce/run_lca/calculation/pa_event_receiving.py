"""Receive actual PA fiber events at one admitted source anchor before attribution."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from ...prepare_backgrounds.preparation.providers.pa_receiving_records import PaReceivingActivity
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from .pa_dump_receiving import PaDumpReceiving, receive_pa_dump
from .pa_landfill_receiving import PaLandfillReceiving, receive_pa_landfill
from .pa_municipal_receiving import PaMunicipalReceiving, receive_pa_municipal_incineration
from .pa_open_burning_receiving import PaOpenBurningReceiving, receive_pa_open_burning
from .pa_receiving import PaDisposalEvent, PaRecyclingEvent, PaSpinningEvent
from .pa_recycling_receiving import PaRecyclingReceiving, receive_pa_recycling
from .pa_spinning_receiving import PaSpinningReceiving, receive_pa_spinning

PaReceivingEvent = PaSpinningEvent | PaRecyclingEvent | PaDisposalEvent
PaReceivingResult = (
    PaSpinningReceiving
    | PaRecyclingReceiving
    | PaMunicipalReceiving
    | PaLandfillReceiving
    | PaOpenBurningReceiving
    | PaDumpReceiving
)


@dataclass(frozen=True, slots=True)
class PaReceivingRequest:
    """One actual mixture awaiting its complete receiving calculation.

    ``disposal_index`` locates the original disposal event before anchor
    filtering. It is transient scheduling metadata.
    Dump events already include their class share in every mass coordinate.
    """

    event: PaReceivingEvent
    infiltration_class: str
    disposal_index: int | None


def iter_pa_receiving_requests(
    *,
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[tuple[int, PaDisposalEvent], ...],
    reference: PaReceivingReference,
    dump_shares: Mapping[str, float],
) -> Iterator[PaReceivingRequest]:
    """Route mixtures in input order to the corresponding receivers.

    The supplied disposal indices precede any temporal filtering. Class order
    follows the original sites; zero shares remain absent streams. Scale all
    mass coordinates before receiving.
    """
    for event in spinning_events:
        yield PaReceivingRequest(event, "", None)
    for event in recycling_events:
        yield PaReceivingRequest(event, "", None)
    for index, event in disposal_events:
        if event.demand_type != "mismanaged_open_dump_treatment":
            yield PaReceivingRequest(event, "", index)
            continue
        for site in reference.dump.sites:
            share = dump_shares[site.infiltration_class]
            if share == 0:
                continue
            routed = replace(
                event,
                principal_by_origin_kg=(
                    event.principal_by_origin_kg[0] * share,
                    event.principal_by_origin_kg[1] * share,
                    event.principal_by_origin_kg[2] * share,
                ),
                total_finish_dose_kg=event.total_finish_dose_kg * share,
                water_kg=event.water_kg * share,
            )
            yield PaReceivingRequest(routed, site.infiltration_class, index)


def receive_pa_request(
    request: PaReceivingRequest,
    *,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    spinning_providers: Mapping[str, PaReceivingActivity],
) -> PaReceivingResult:
    """Evaluate one routed mixture with its unchanged complete material-balance calculation."""
    event = request.event
    if isinstance(event, PaSpinningEvent):
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        return receive_pa_spinning(
            event, reference, sources, spinning_provider=spinning_providers[grade]
        )
    if isinstance(event, PaRecyclingEvent):
        return receive_pa_recycling(event, reference, sources)
    # These names exhaust the disposal events produced by the material-balance calculation.
    receiving = {
        "incineration_treatment": "municipal",
        "mechanical_recycling_reject_incineration_treatment": "municipal",
        "landfill_treatment": "landfill",
        "mismanaged_open_burning_treatment": "burning",
        "mismanaged_open_dump_treatment": "dump",
    }[event.demand_type]
    if receiving == "municipal":
        return receive_pa_municipal_incineration(event, reference.municipal, sources.municipal)
    if receiving == "landfill":
        return receive_pa_landfill(
            constituent=event.constituent,
            principal_by_origin_kg={
                "fossil": event.principal_by_origin_kg[0],
                "biogenic": event.principal_by_origin_kg[1],
                "atmospheric_dac": event.principal_by_origin_kg[2],
            },
            total_finish_dose_kg=event.total_finish_dose_kg,
            water_kg=event.water_kg,
            reference=reference,
            sources=sources,
        )
    if receiving == "burning":
        return receive_pa_open_burning(event, reference.open_burning)
    return receive_pa_dump(
        constituent=event.constituent,
        principal_by_origin_kg={
            "fossil": event.principal_by_origin_kg[0],
            "biogenic": event.principal_by_origin_kg[1],
            "atmospheric_dac": event.principal_by_origin_kg[2],
        },
        total_finish_dose_kg=event.total_finish_dose_kg,
        water_kg=event.water_kg,
        infiltration_class=request.infiltration_class,
        reference=reference.dump,
    )


def iter_pa_receiving(
    *,
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[PaDisposalEvent, ...],
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    spinning_providers: Mapping[str, PaReceivingActivity],
    dump_shares: Mapping[str, float],
) -> Iterator[tuple[PaReceivingEvent, str, PaReceivingResult]]:
    """Yield complete receiving results and their actual activity basis.

    Inputs retain their activity year and world region; ``sources`` supplies one
    selected background anchor. Each receiver consumes the unsplit principal,
    inherited finish and water together. Origin attribution and interpolation
    occur after these potentially nonlinear calculations. Spinning providers
    are keyed by ``pa6`` and ``pa66`` and identify provenance. Existing receiving results retain all
    child mass-balance ledgers.

    Mismanaged fate shares already belong to the disposal events. Only the
    independently admitted dump-class shares remain here: scale every mass
    before calculation and yield that scaled event, so both emitted amounts
    and mass-balance ledgers share its basis. The admitted five-class portfolio is
    fixed by the selected original source, independently of scenario parameters.
    A zero original class weight denotes an absent stream and is not received.
    Class order follows the five original source sites; other results carry an
    empty infiltration-class string. Input order is preserved within spinning,
    recycling and disposal groups. Receiving uses the prepared source references.
    """
    for request in iter_pa_receiving_requests(
        spinning_events=spinning_events,
        recycling_events=recycling_events,
        disposal_events=tuple(enumerate(disposal_events)),
        reference=reference,
        dump_shares=dump_shares,
    ):
        yield (
            request.event,
            request.infiltration_class,
            receive_pa_request(
                request,
                reference=reference,
                sources=sources,
                spinning_providers=spinning_providers,
            ),
        )
