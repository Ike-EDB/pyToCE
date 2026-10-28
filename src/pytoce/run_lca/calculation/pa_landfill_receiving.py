"""Calculate PA fiber inventories for sanitary landfill and connected treatments."""

from dataclasses import dataclass
from math import fsum
from typing import cast

from ...prepare_backgrounds.preparation.providers.landfill_phases import _LandfillLoad
from ...prepare_backgrounds.preparation.providers.landfill_site import _sanitary_site_emissions
from ...prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from ...prepare_backgrounds.preparation.providers.pa_finish_material import _pa_finish_reference
from ...prepare_backgrounds.preparation.providers.pa_landfill_load import _pa_landfill_load
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import PaReceivingActivity
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from .pa_gas_receiving import receive_pa_landfill_gas
from .pa_landfarming_receiving import PaLandfarmingReceiving, receive_pa_landfarmed_sludge
from .pa_receiving import PaReceivingAmount
from .pa_sludge_receiving import PaSludgeReceiving, receive_pa_incinerated_sludge
from .pa_water_receiving import PaWaterReceiving, receive_pa_water


@dataclass(frozen=True, slots=True)
class PaLandfillReceiving:
    """Complete actual amounts and separate landfill material/carbon phases.

    ``load`` retains the placed water, principal and auxiliary carbon origins,
    short and later releases and stored material on the caller's kg basis.
    ``water``, ``sludge`` and ``landfarming`` preserve conditional downstream
    ledgers, including unreleased ash phosphorus and crop uptake. Their native
    amounts already include collection and occur once in the combined inventory;
    these references identify the source records used by that calculation.
    """

    amounts: tuple[PaReceivingAmount, ...]
    load: _LandfillLoad
    water: PaWaterReceiving
    sludge: PaSludgeReceiving
    landfarming: PaLandfarmingReceiving


def receive_pa_landfill(
    *,
    constituent: str,
    principal_by_origin_kg: dict[_CarbonOrigin, float],
    total_finish_dose_kg: float,
    water_kg: float,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
) -> PaLandfillReceiving:
    """Receive one nonzero PA mixture at the selected RoW sanitary site.

    All inputs share one activity before principal-origin attribution or
    temporal interpolation. The finish dose multiplies the original retained
    compound vector; it already includes any surface finish appropriate to this
    event. Wet end-of-life fiber and dry recycling reject therefore provide
    their own dose and water.
    Original site climate and liquid volume remain distinct from regional custody
    and dated operating supplies. The five open dumps have separate owners.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[constituent]
    finish = _pa_finish_reference(grade)
    original = reference.landfill
    native = sources.landfill
    load = _pa_landfill_load(
        finish,
        principal_by_origin_kg=principal_by_origin_kg,
        finish_dose_kg=total_finish_dose_kg,
        water_kg=water_kg,
        site=original.climate,
    )
    gas = receive_pa_landfill_gas(load.gas, reference.gas, sources.gas)
    # The original negative leachate/waste ratio is m3/kg wet waste. At Doka's 1000 kg/m3 solution
    # reference, solutes displace carrier water; placed moisture is included.
    volume = original.leachate_volume_m3_per_wet_kg * load.wet_mass_kg
    water = receive_pa_water(
        load.leachate,
        volume_m3=volume,
        incoming_water_kg=1000 * volume - fsum(load.leachate.elements_kg.values()),
        reference=reference,
        sources=sources,
        receiving="sanitary",
    )
    sludge = receive_pa_incinerated_sludge(water, reference, sources, receiving="sanitary")
    soil = receive_pa_landfarmed_sludge(water, reference, sources, receiving="sanitary")
    function = "pa_sanitary_landfill"
    site = native.landfill
    # Admission retains only technosphere rows here and normalizes them by
    # absolute source production. Preserve all dated support on wet throughput;
    # collection is expanded above instead of buying its whole native chain.
    retained = tuple(row for row in site.rows if row.endpoint.key != native.collection.activity.key)
    amounts = [
        PaReceivingAmount(function, site.activity, row.endpoint, row.amount, row.metadata).scaled(
            load.wet_mass_kg
        )
        for row in retained
    ]
    administration_kwh = load.wet_mass_kg * fsum(
        row.amount
        for row in retained
        if cast(PaReceivingActivity, row.endpoint).reference_product == "electricity, low voltage"
    )
    emissions = _sanitary_site_emissions(
        wet_mass_kg=load.wet_mass_kg,
        carbon_kg=load.elements_kg["C"],
        later_elements_kg={
            element: load.phases[element].later_leachate_kg
            for element in ("C", "N", "S", "P", "Na")
        },
        gas_elements_kg={"Na": load.phases["Na"].gas_kg},
    )
    # Doka's consuming-meter heat uses 3.6 MJ/kWh for site administration. Gas
    # heat has already removed non-credited electricity export. Decomposition
    # heat belongs to soil; leachate energy and its receiving remain separate.
    emissions["Heat, waste", ("air", "non-urban air or from high stacks")] = (
        gas.heat_air_mj + 3.6 * administration_kwh
    )
    emissions["Heat, waste", ("soil", "industrial")] = gas.heat_soil_mj
    flows = {(flow.name, flow.categories): flow for flow in native.biosphere}
    amounts.extend(
        PaReceivingAmount(
            function,
            original.landfill,
            flows[key],
            amount,
            {"uncertainty type": 0, "loc": amount},
        )
        for key, amount in emissions.items()
    )
    amounts.extend((*gas.amounts, *water.amounts, *sludge.amounts, *soil.amounts))
    return PaLandfillReceiving(tuple(amounts), load, water, sludge, soil)
