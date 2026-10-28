"""Calculate inventories for PA fiber wastewater collection, treatment and digestion.

Manufacturing and recycling use the selected Swiss co-treatment function;
sanitary landfill uses its separate RoW leachate function. Event loads are
calculated before carbon-origin attribution. The returned wastewater load and
treatment fraction supply sludge calculations, which determine the ash loads.
"""

from dataclasses import dataclass
from math import fsum
from typing import Literal

from ...prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _ROW_WASTEWATER_PROFILE,
    _LeachateLoad,
    _wastewater_load,
    _WastewaterLoad,
)
from ...prepare_backgrounds.preparation.providers.leachate_treatment import (
    _CARBON_INDICATORS,
    _RAW_ELEMENTS,
    _wastewater_discharges,
)
from ...prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    PaLandfillReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    PaReceivingOperation,
    PaReceivingRow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    PaWaterReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.water_treatment import _HEAT_PRODUCTS
from .pa_gas_receiving import receive_pa_wastewater_gas
from .pa_receiving import PaReceivingAmount


@dataclass(frozen=True, slots=True)
class PaWaterReceiving:
    """Actual collection/plant inventory and its conditional sludge receiving load.

    ``amounts`` include collection bypass plus treated plant and gas quantities.
    ``wastewater`` retains the complete incoming reference before collection;
    its sludge children must apply ``treatment_fraction`` exactly once. Sludge
    incineration/landfarming shares are already in that load's respective masses.
    """

    amounts: tuple[PaReceivingAmount, ...]
    wastewater: _WastewaterLoad
    treatment_fraction: float


def _native_water_amount(
    function: str,
    operation: PaReceivingOperation,
    row: PaReceivingRow,
    multiplier: float,
) -> PaReceivingAmount:
    """Scale an admitted water-source row in its endpoint's native unit."""
    return PaReceivingAmount(
        function, operation.activity, row.endpoint, row.amount, row.metadata
    ).scaled(multiplier)


def _collection_amounts(
    leachate: _LeachateLoad,
    wastewater: _WastewaterLoad,
    sources: PaWaterReceivingSources | PaLandfillReceivingSources,
    function: str,
) -> tuple[list[PaReceivingAmount], float]:
    """Keep sewer support and send the untreated fraction to its native water flows.

    The admitted child coefficient is negative per positive m3 collection.
    Its opposite is the treated fraction; child quantities use that production basis. Raw phosphorus
    is elemental P, while the plant's separate
    discharge owner calculates phosphate. Actual phosphorus follows the event's composition.
    """
    collection = sources.collection
    treatment_fraction = -fsum(
        row.amount for row in collection.rows if row.endpoint.key == sources.wastewater.activity.key
    )
    amounts = [
        _native_water_amount(function, collection, row, wastewater.volume_m3)
        for row in collection.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.key != sources.wastewater.activity.key
    ]
    bypass = 1 - treatment_fraction
    # Actual raw H/O are the admitted aqueous descriptors. Ionic chloride
    # stays separate from elemental chlorine, as in the shared collection owner.
    raw = {name: leachate.elements_kg.get(element, 0.0) for name, element in _RAW_ELEMENTS.items()}
    raw["Chlorine"] -= leachate.chloride_kg
    raw["Chloride"] = leachate.chloride_kg
    raw.update(
        {name: float(getattr(leachate, field)) for name, field in _CARBON_INDICATORS.items()}
    )
    # Carrier water kg is converted to the existing elementary-flow unit m3.
    raw["Water"] = wastewater.water.incoming_kg / 1000
    for row in collection.rows:
        if not isinstance(row.endpoint, PaReceivingFlow):
            continue
        value = raw[row.endpoint.name] * bypass
        if value != 0:
            # Retain the collection source's exact compartment, including its native unspecified
            # water compartment.
            amounts.append(_native_water_amount(function, collection, row, value / row.amount))
    # The RoW plastic collector has no P row; its separately admitted raw P
    # endpoint closes the actual PA phosphorus bypass. Both chains admit the
    # ionic chloride endpoint separately from the native elemental descriptors.
    added = {"Chloride": leachate.chloride_kg}
    if isinstance(sources, PaLandfillReceivingSources):
        added["Phosphorus"] = leachate.elements_kg["P"]
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in sources.biosphere}
    for name, value in added.items():
        if value == 0:
            continue
        amount = value * bypass
        amounts.append(
            PaReceivingAmount(
                function,
                collection.activity,
                flows[name, "kilogram", ("water", "surface water")],
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    return amounts, treatment_fraction


def receive_pa_water(
    leachate: _LeachateLoad,
    *,
    volume_m3: float,
    incoming_water_kg: float,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    receiving: Literal["swiss", "sanitary"],
) -> PaWaterReceiving:
    """Calculate one complete actual PA water inlet before sludge/ash receiving.

    All input quantities share the caller's activity. The explicit receiving selection fixes the
    admitted
    CH or RoW function, original climate and dated portfolio. No donor composition,
    current database query or interpolation of nonlinear load coefficients enters
    this calculation. An absent inlet is handled by its activity owner.
    """
    if receiving == "swiss":
        water_reference, water_sources = reference.water, sources.water
        profile = _CH_WASTEWATER_PROFILE
    else:
        water_reference, water_sources = reference.landfill, sources.landfill
        profile = _ROW_WASTEWATER_PROFILE
    function = f"pa_{receiving}_wastewater"
    wastewater = _wastewater_load(
        leachate,
        volume_m3=volume_m3,
        incoming_water_kg=incoming_water_kg,
        evaporation_fraction=water_reference.evaporation_fraction,
        profile=profile,
    )
    amounts, treated = _collection_amounts(
        leachate, wastewater, water_sources, f"pa_{receiving}_collection"
    )
    gas = receive_pa_wastewater_gas(wastewater, profile, reference.gas, sources.gas)
    operation = water_sources.wastewater
    grouped: dict[str, list[PaReceivingRow]] = {}
    for row in operation.rows:
        if isinstance(row.endpoint, PaReceivingActivity):
            product = row.endpoint.reference_product
            if product in _HEAT_PRODUCTS:
                product = "heat, district or industrial"
            grouped.setdefault(product, []).append(row)
    # Doka Infra G28/G33 allocates 23375 kWh construction electricity per
    # 13250 kg limestone. Source rows are already per m3; apply actual volume.
    limestone = fsum(row.amount for row in grouped["limestone, crushed, washed"])
    gross_electricity = wastewater.duties.electricity_kwh + volume_m3 * limestone * 23375 / 13250
    purchased_heat = wastewater.duties.heat_mj - gas.internal_heat_mj
    target = {
        "electricity, medium voltage": gross_electricity - gas.internal_electricity_kwh,
        "heat, district or industrial": purchased_heat,
        "iron sulfate": wastewater.duties.iron_sulfate_kg,
        "polyacrylamide": wastewater.duties.polyacrylamide_kg,
        "inert waste, for final disposal": -wastewater.duties.sand_kg,
        "waste graphical paper": -wastewater.duties.grit_biomass_kg,
        "waste plastic, mixture": -wastewater.duties.grit_plastic_kg,
    }
    # Connected child functions handle actual sludge.
    children = {
        water_sources.sludge_incineration_market.activity.key,
        water_sources.sludge_landfarming_market.activity.key,
    }
    for product, rows in grouped.items():
        if product in target:
            quantity = target[product]
            if quantity != 0:
                total = fsum(row.amount for row in rows)
                amounts.extend(
                    _native_water_amount(function, operation, row, treated * quantity / total)
                    for row in rows
                )
        else:
            amounts.extend(
                _native_water_amount(function, operation, row, treated * volume_m3)
                for row in rows
                if row.endpoint.key not in children
            )
    # Admission retained only the native land/infrastructure biosphere rows;
    # all donor biological, gas and hydraulic emissions are replaced below.
    amounts.extend(
        _native_water_amount(function, operation, row, treated * volume_m3)
        for row in operation.rows
        if isinstance(row.endpoint, PaReceivingFlow)
    )
    amounts.extend(row.scaled(treated) for row in gas.amounts)
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in water_sources.biosphere}
    for discharge in _wastewater_discharges(
        wastewater,
        profile,
        gross_electricity_kwh=gross_electricity,
        gas_heat_air_mj=gas.heat_air_mj,
        purchased_heat_mj=purchased_heat,
    ):
        amount = treated * discharge.amount
        amounts.append(
            PaReceivingAmount(
                function,
                operation.activity,
                flows[discharge.name, discharge.unit, discharge.categories],
                amount,
                {"uncertainty type": 0, "loc": amount},
                discharge.principal_carbon_origin_fractions,
            )
        )
    return PaWaterReceiving(tuple(amounts), wastewater, treated)
