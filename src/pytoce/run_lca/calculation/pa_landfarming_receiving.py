"""Calculate inventories and constituent fates for PA sludge applied to soil."""

from dataclasses import dataclass, replace
from typing import Literal

from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.sludge_landfarming import (
    LandfarmingLoad,
    calculate_landfarming,
)
from .pa_receiving import PaReceivingAmount
from .pa_water_receiving import PaWaterReceiving


@dataclass(frozen=True, slots=True)
class PaLandfarmingReceiving:
    """Actual inventory plus the conditional soil/crop ledger before collection.

    Final native amounts include the collection treatment fraction once. The
    ``load`` preserves the same conditional basis as its wastewater
    parent; crop uptake remains a fate.
    """

    amounts: tuple[PaReceivingAmount, ...]
    load: LandfarmingLoad


def receive_pa_landfarmed_sludge(
    water: PaWaterReceiving,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    *,
    receiving: Literal["swiss", "sanitary"],
) -> PaLandfarmingReceiving:
    """Receive the actual agricultural sludge share through its selected function.

    Swiss manufacturing/recycling and RoW sanitary leachate retain separate
    original sources and dated spreading portfolios. Both PA paths
    use complete carbon oxidation and the Doka report's phosphorus allocation.
    The calculation uses the supplied event composition; all quantities remain on the actual
    activity basis.
    """
    if receiving == "swiss":
        soil_reference, water_sources = reference.water.landfarming, sources.water
        # The CH average-sewage source supplies both native CO2 templates.
        templates = {"fossil": "fossil", "non-fossil": "non-fossil"}
    else:
        soil_reference, water_sources = reference.landfill.landfarming, sources.landfill
        # The selected plastic-leachate donor has fossil CO2 only. Its same
        # total oxidation function uses the already admitted exact non-fossil
        # counterpart for actual biological/DAC carbon, retaining auxiliaries.
        templates = {"fossil": "fossil", "non-fossil": "fossil"}
    load = calculate_landfarming(
        water.wastewater,
        soil_reference,
        complete_carbon_oxidation=True,
        phosphorus_equilibrium=True,
    )
    treated = water.treatment_fraction
    market = water_sources.sludge_landfarming_market
    operation = water_sources.sludge_landfarming
    function = f"pa_{receiving}_sludge_landfarming"
    amounts = [
        PaReceivingAmount(
            function + "_market", market.activity, row.endpoint, row.amount, row.metadata
        ).scaled(load.wet_kg * treated)
        for row in market.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.key != operation.activity.key
    ]
    # The native negative-kg market is expanded here. The actual wet mass
    # already includes the agricultural share and its 97% moisture reference.
    # Its complete spreading/transport rows retain their own product units.
    amounts.extend(
        PaReceivingAmount(
            function, operation.activity, row.endpoint, row.amount, row.metadata
        ).scaled(load.wet_kg * treated)
        for row in operation.rows
        if isinstance(row.endpoint, PaReceivingActivity)
    )
    original = soil_reference.operation
    for release in load.native_releases:
        row = original.rows[release.source_row]
        amounts.append(
            PaReceivingAmount(
                function, original.activity, row.endpoint, row.amount, row.metadata
            ).scaled(release.amount * treated / row.amount)
        )
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in water_sources.biosphere}
    for release in load.carbon_dioxide:
        template_name = f"Carbon dioxide, {templates[release.origin]}"
        template = next(
            row
            for row in original.rows
            if isinstance(row.endpoint, PaReceivingFlow) and row.endpoint.name == template_name
        )
        amount = PaReceivingAmount(
            function,
            original.activity,
            template.endpoint,
            template.amount,
            template.metadata,
            release.principal_carbon_origin_fractions,
        ).scaled(release.amount_kg * treated / template.amount)
        name = f"Carbon dioxide, {release.origin}"
        if name != template_name:
            amount = replace(
                amount,
                endpoint=flows[name, "kilogram", ("air", "non-urban air or from high stacks")],
                metadata={
                    key: value
                    for key, value in amount.metadata.items()
                    if key not in {"flow", "activity"}
                },
            )
        amounts.append(amount)
    for compartment, phosphate in load.phosphate_water_kg.items():
        amount = phosphate * treated
        amounts.append(
            PaReceivingAmount(
                function,
                original.activity,
                flows["Phosphate", "kilogram", compartment],
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    return PaLandfarmingReceiving(tuple(amounts), load)
