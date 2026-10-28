"""Receive actual PA fiber washing, pelletizing, dust and wet-pellet delivery."""

from dataclasses import dataclass
from math import fsum

from ...prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
    _pa_recycling_reference,
    _PARecyclingReference,
)
from ...prepare_backgrounds.preparation.providers.pa_finish_water import _pa_finish_water_load
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from .pa_landfarming_receiving import PaLandfarmingReceiving, receive_pa_landfarmed_sludge
from .pa_receiving import PaReceivingAmount, PaRecyclingEvent
from .pa_sludge_receiving import PaSludgeReceiving, receive_pa_incinerated_sludge
from .pa_water_receiving import PaWaterReceiving, receive_pa_water


@dataclass(frozen=True, slots=True)
class PaRecyclingReceiving:
    """Actual operating inventory and wash, pellet and receiving ledgers.

    ``load`` retains actual dry material and water on the event's kg basis.
    ``water``, ``sludge`` and ``landfarming`` preserve their conditional carbon,
    phosphorus and material fates; their final amounts occur once in ``amounts``.
    Principal recovery, airborne material and managed rejects remain with MFA
    custody. This result uses the exits allocated by the material calculation.
    """

    amounts: tuple[PaReceivingAmount, ...]
    load: _PARecyclingReference
    water: PaWaterReceiving
    sludge: PaSludgeReceiving
    landfarming: PaLandfarmingReceiving


def receive_pa_recycling(
    event: PaRecyclingEvent,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
) -> PaRecyclingReceiving:
    """Wash one actual PA fiber mixture and supply its complete dated operations.

    The activity precedes origin attribution and temporal interpolation.
    Original WEEE dry-feed and water quantities determine actual washing water;
    dated complete operations retain their utilities, equipment, cooling and
    individual native uncertainties. Surface washing acts once, using the already allocated airborne
    finish dose.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
    finish = _pa_finish_reference(grade)
    original = reference.recycling
    native = sources.recycling
    principal = fsum(event.principal_by_origin_kg)
    load = _pa_recycling_reference(
        grade,
        finish,
        principal_kg=principal,
        embedded_dose_kg=event.embedded_finish_dose_kg,
        recovery_fraction=event.recovered_principal_kg / principal,
        wash_water_per_dry_kg=original.wash_water_kg_per_reference_kg
        / original.dry_feed_kg_per_reference_kg,
    )
    # The original allocated dry-feed/reference ratio converts incoming dry mixture into WEEE
    # operating kg. Dated rows are already normalized by production. Their water portfolio supplies
    # the original wash demand.
    preparation_amount = load.incoming_dry_kg / original.dry_feed_kg_per_reference_kg
    dated_water = fsum(
        native.preparation.rows[index].amount for index in native.preparation_water_rows
    )
    replaced_preparation = (*native.preparation_pm_rows, *native.preparation_wastewater_rows)
    amounts = [
        PaReceivingAmount(
            "pa_recycling_preparation",
            native.preparation.activity,
            row.endpoint,
            row.amount,
            row.metadata,
        ).scaled(
            load.fresh_wash_water_kg / dated_water
            if index in native.preparation_water_rows
            else preparation_amount
        )
        for index, row in enumerate(native.preparation.rows)
        if index not in replaced_preparation
    ]
    # Pelletizing supplies the complete melt/dry function on actual dry pellet
    # output. Its separate cooling-water supply and whole cooling-water receiver
    # stay intact; the incoming wash/film water belongs to the load.
    amounts.extend(
        PaReceivingAmount(
            "pa_recycling_finishing",
            native.finishing.activity,
            row.endpoint,
            row.amount,
            row.metadata,
        ).scaled(load.pellet_dry_kg)
        for index, row in enumerate(native.finishing.rows)
        if index not in native.finishing_pm_rows
    )
    # Allocate postwash principal and retained finish to aggregate dust.
    # One release replaces washer and extruder particulate emissions and
    # retains both operation identities.
    dust = fsum(event.airborne_principal_by_origin_kg) + (
        fsum(finish.retained.compounds_kg.values()) * event.airborne_finish_dose_kg
    )
    dust_endpoint = native.preparation.rows[native.preparation_pm_rows[0]].endpoint
    amounts.append(
        PaReceivingAmount(
            "pa_recycling_dust",
            native.preparation.activity,
            dust_endpoint,
            dust,
            {
                "uncertainty type": 0,
                "loc": dust,
                "source_operations": (
                    native.preparation.activity.key,
                    native.finishing.activity.key,
                ),
                "comment": "Aggregate postwash dust from PA preparation and pelletizing.",
            },
        )
    )
    # The reference specifies 4 wt% film and 500 ppm pellet endpoints. Their water difference uses
    # native m3 (1000 kg).
    air_water_m3 = load.air_water_kg / 1000
    amounts.append(
        PaReceivingAmount(
            "pa_recycling_drying",
            native.finishing.activity,
            native.evaporated_water,
            air_water_m3,
            {"uncertainty type": 0, "loc": air_water_m3},
        )
    )
    # Carried pellet water belongs to the delivered payload. The complete
    # prepared delivery endpoint retains its native production normalization
    # and supplier uncertainties; actual wet cargo is demanded once.
    delivered_kg = load.pellet_dry_kg + load.pellet_water_kg
    amounts.append(
        PaReceivingAmount(
            "pa_recycling_delivery",
            native.pellet_delivery,
            native.pellet_delivery,
            delivered_kg,
            {"uncertainty type": 0, "loc": delivered_kg},
        )
    )
    # Surface finish replaces the donor PP wastewater composition. The actual solvent water and
    # removed-solute solution volume enter one complete Swiss
    # collection/biological/gas/sludge/ash/soil receiving chain.
    leachate = _pa_finish_water_load(load.removed_finish, caprolactam_by_origin_kg={})
    water = receive_pa_water(
        leachate,
        volume_m3=load.collected_solution_m3,
        incoming_water_kg=load.collected_water_kg,
        reference=reference,
        sources=sources,
        receiving="swiss",
    )
    sludge = receive_pa_incinerated_sludge(water, reference, sources, receiving="swiss")
    soil = receive_pa_landfarmed_sludge(water, reference, sources, receiving="swiss")
    amounts.extend((*water.amounts, *sludge.amounts, *soil.amounts))
    return PaRecyclingReceiving(tuple(amounts), load, water, sludge, soil)
