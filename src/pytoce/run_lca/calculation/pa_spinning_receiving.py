"""Connect actual PA spinning to complete maintenance and Swiss wastewater treatment."""

from dataclasses import dataclass
from math import fsum

from ...prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_load,
    _pa_finish_reference,
    _pa_spinning_water,
    _PASpinningWater,
)
from ...prepare_backgrounds.preparation.providers.pa_finish_water import _pa_finish_water_load
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.waste_gas_load import _NMVOC_NAME
from .pa_landfarming_receiving import receive_pa_landfarmed_sludge
from .pa_receiving import PaReceivingAmount, PaSpinningEvent
from .pa_sludge_receiving import receive_pa_incinerated_sludge
from .pa_thermal_receiving import receive_pa_maintenance
from .pa_water_receiving import receive_pa_water


@dataclass(frozen=True, slots=True)
class PaSpinningReceiving:
    """Complete actual receiving amounts and the site's water allocation.

    The separate fixed spinning provider owns operating energy, fresh concentrate,
    capital and thermal-fluid makeup/retirement per kg finished principal.
    These ``amounts`` own the actual site resource/emissions and maintenance,
    collected water, gas, sludge and ash. They supply the corresponding variable receiving
    quantities. ``water`` distinguishes its internal maintenance transfer from final
    cooling, exhaust, collected water and conditioned-fiber moisture.
    """

    amounts: tuple[PaReceivingAmount, ...]
    water: _PASpinningWater


def receive_pa_spinning(
    event: PaSpinningEvent,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    *,
    spinning_provider: PaReceivingActivity,
) -> PaSpinningReceiving:
    """Calculate one complete manufacturing mixture before any dated interpolation.

    ``spinning_provider`` identifies the prepared grade's BREF operating function
    for provenance. Each final endpoint remains a selected native supplier or
    elementary flow. No store query or repeated principal/finish loss occurs.
    All quantities use the event's actual kg basis and original finish composition.
    Recycling wash is a different activity; the incoming recovered pellets
    have already undergone that wash and are not washed again here.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
    finish = _pa_finish_reference(grade)
    principal = event.finished_principal_kg
    maintenance = receive_pa_maintenance(event, reference, sources)
    site = _pa_spinning_water(
        finish,
        finished_principal_kg=principal,
        recovered_principal_kg=event.recovered_principal_kg,
        embedded_dose_kg=event.embedded_finish_dose_kg,
        maintenance_principal_kg=fsum(event.maintenance_principal_by_origin_kg),
        maintenance_embedded_dose_kg=event.maintenance_finish_dose_kg,
        maintenance_water_kg=maintenance.site_water_kg,
    )
    collected_finish = _pa_finish_load(
        {name: principal * mass for name, mass in finish.collected.compounds_kg.items()}
    )
    # Principal caprolactam and the fixed auxiliary mixture enter one nonlinear
    # wastewater calculation. The event already carries the principal exit by
    # fossil/biogenic/DAC origin; the finish retains its own compound origins.
    leachate = _pa_finish_water_load(
        collected_finish,
        caprolactam_by_origin_kg=dict(
            zip(
                ("fossil", "biogenic", "atmospheric_dac"),
                event.caprolactam_water_by_origin_kg,
                strict=True,
            )
        ),
    )
    water = receive_pa_water(
        leachate,
        volume_m3=principal * finish.collected_solution_m3,
        incoming_water_kg=site.collected_water_kg,
        reference=reference,
        sources=sources,
        receiving="swiss",
    )
    sludge = receive_pa_incinerated_sludge(water, reference, sources, receiving="swiss")
    landfarming = receive_pa_landfarmed_sludge(water, reference, sources, receiving="swiss")
    amounts = list((*maintenance.amounts, *water.amounts, *sludge.amounts, *landfarming.amounts))
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in sources.added_biosphere}
    # Polymers BREF2007 section9.3.3.2 describes once-through river/well water.
    # The selected integrated maintenance source already admits this well-water
    # identity. Its ordinary control inlet is inside the whole site withdrawal.
    well = next(
        row.endpoint
        for row in sources.fixed.rows
        if isinstance(row.endpoint, PaReceivingFlow)
        and row.endpoint.name == "Water, well, in ground"
    )
    # Table9.4 HTM midpoints (g/t -> kg/kg) enter one NMVOC total with principal caprolactam and
    # ester-equivalent finish. The fixed support supplies lost fluid.
    thermal_fluid_air = {"pa6": 32.5e-6, "pa66": 30e-6}[grade]
    nmvoc = fsum(event.caprolactam_air_by_origin_kg) + principal * (
        fsum(finish.air.compounds_kg.values()) + thermal_fluid_air
    )
    # Water uses 1000 kg/m3. Conditioned-fiber moisture remains with the product;
    # collected and maintenance water are internal transfers already treated.
    # NMVOC provides aggregate characterization; caprolactam carbon
    # remains separately recorded in the event.
    for endpoint, amount in (
        (well, site.site_water_kg / 1000),
        (flows["Water", "cubic meter", ("water",)], site.cooling_water_kg / 1000),
        (flows["Water", "cubic meter", ("air",)], site.exhaust_water_kg / 1000),
        (flows[_NMVOC_NAME, "kilogram", ("air",)], nmvoc),
    ):
        amounts.append(
            PaReceivingAmount(
                "pa_spinning_site",
                spinning_provider,
                endpoint,
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    return PaSpinningReceiving(tuple(amounts), site)
