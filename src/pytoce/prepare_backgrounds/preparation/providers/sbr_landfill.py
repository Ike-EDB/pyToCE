"""Connect filled SBR trim and retired compound to their complete landfill function.

The Others non-tire compounding operation owns manufacturing trim and its
136 km haul. Retired compound enters through its own principal-kg service.
Both use the original RoW plastic-landfill site,
its actual leachate composition and its complete gas/sludge/ash descendants.
The GLO thermal/ash children retain their actual geographic identities.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

from .ash_treatment import _PLASTIC_ASH_RELEASES, _ash_dataset, _ash_market_dataset
from .catalogue import provider_service_code
from .landfill_site import _sanitary_site_emissions
from .leachate_load import _ROW_WASTEWATER_PROFILE, _CarbonOrigin, _wastewater_load, _WastewaterLoad
from .leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .sbr_landfill_load import _sbr_landfill_load, _SBRLandfillLoad
from .sbr_material import _sbr_compound_material
from .sludge_incineration import _sludge_dataset, _sludge_source
from .sludge_incineration_load import calculate_sludge_incineration
from .sludge_landfarming import _landfarming_dataset
from .waste_gas import _GasTreatment, _landfill_gas, _wastewater_gas
from .water_treatment import _ROW_PROFILE, _water_evaporation

_AIR = ("air", "non-urban air or from high stacks")
_SOIL = ("soil", "industrial")
_LONG = ("water", "ground-, long-term")
_ELEMENTS = {
    "C": "carbon",
    "H": "hydrogen",
    "O": "oxygen",
    "N": "nitrogen",
    "S": "sulfur",
    "Zn": "zinc",
}


def _sbr_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Compile filled-SBR receiving on its declared principal or whole-waste basis.

    The original landfill supplies liquid volume/kg waste and the original WWT
    supplies its climate.
    All descendant loads are conditional per retained principal kg; collection
    alone applies the treated share. The calculation consumes the completed package inputs.
    """
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
    if kind == "sbr_compound_landfill":
        # The existing site produces negative kg of whole filled waste. This service receives +1
        # principal kg with its retained auxiliaries; purchasing -M converts that unit once. The
        # shared site already normalizes its homogeneous trim.
        service_id = str(service["provider_service_id"])
        key = (database_name, provider_service_code(service_id))
        mass = _sbr_compound_material(origin, principal_kg=1.0).mass_kg
        treatment = str(cast(dict[str, object], construction["treatment"])["service_id"])
        _service_trace(context, service)
        return {
            "name": str(service["name"]),
            "reference product": str(service["reference_product"]),
            "unit": "kilogram",
            "location": str(service["location"]),
            "type": "process",
            "provider_service_id": service_id,
            "provider_stage_owner": str(service["stage_owner"]),
            "exchanges": [
                {"input": key, "amount": 1.0, "type": "production"},
                {
                    "input": (database_name, provider_service_code(treatment)),
                    "amount": -mass,
                    "type": "technosphere",
                },
            ],
        }
    formulation = str(construction["formulation_service_id"])
    profile = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], context.catalogue["semantic_contract"])["sbr_filled_landfill"],
    )
    load = _sbr_landfill_load(origin)
    source = context.source.activity(profile["landfill_source"], year=2025)
    rows = context.exchanges(source)
    production = next(ex for ex in rows if ex["type"] == "production")
    liquid = next(
        ex
        for ex in rows
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"]
        == profile["wastewater_source"]["reference_product"]
    )
    # Both original coefficients are negative: m3 leachate/kg actual scrap. At 1000 kg/m3, actual
    # solutes displace carrier water.
    volume = float(liquid["amount"]) / float(production["amount"]) * load.mass_kg
    plant = context.source.activity(profile["wastewater_source"], year=2025)
    wastewater = _wastewater_load(
        load.leachate,
        volume_m3=volume,
        incoming_water_kg=1000 * volume - fsum(load.leachate.elements_kg.values()),
        evaporation_fraction=_water_evaporation(str(plant["comment"]), _ROW_PROFILE),
        profile=_ROW_WASTEWATER_PROFILE,
    )
    service_id = str(service["provider_service_id"])
    if kind == "sbr_filled_residue_landfill":
        gas = _landfill_gas(context, construction, load.gas, service_id)
        return _sbr_landfill_dataset(context, service, load, wastewater, gas, database_name)
    if kind == "sbr_filled_leachate_collection":
        return _leachate_market_dataset(context, service, load.leachate, wastewater, database_name)
    if kind == "sbr_filled_wastewater_treatment":
        gas = _wastewater_gas(
            context, construction, wastewater, service_id, profile=_ROW_WASTEWATER_PROFILE
        )
        return _wastewater_dataset(
            context,
            service,
            load.leachate,
            wastewater,
            gas,
            database_name,
            profile=_ROW_WASTEWATER_PROFILE,
        )
    if kind == "sbr_filled_sludge_collection":
        return _sludge_market_dataset(context, service, wastewater, database_name)
    if kind == "sbr_filled_sludge_landfarming":
        return _landfarming_dataset(
            context, service, wastewater, formulation, database_name, complete_carbon_oxidation=True
        )
    thermal = (
        construction
        if kind == "sbr_filled_sludge_incineration"
        else cast(
            dict[str, object],
            services[str(construction["incineration_service_id"])]["construction"],
        )
    )
    sludge = calculate_sludge_incineration(
        _sludge_source(context, thermal),
        wastewater,
        formulation,
        source_carbon_origins=tuple(
            dict.fromkeys(cast(dict[str, str], thermal["carbon_templates"]).values())
        ),
        additional_apc_water_flows=tuple(cast(list[str], thermal["additional_apc_water_flows"])),
        phosphorus_phase_reference=False,
        # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
        bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
    )
    if kind == "sbr_filled_sludge_incineration":
        return _sludge_dataset(context, service, sludge, database_name)
    ash = sludge.bottom_ash if construction["ash_fraction"] == "bottom" else sludge.residual_ash
    if kind == "sbr_filled_ash_market":
        return _ash_market_dataset(context, service, ash, database_name)
    return _ash_dataset(
        context,
        service,
        ash,
        database_name,
        release_profile=_PLASTIC_ASH_RELEASES[str(construction["ash_fraction"])],
    )


def _sbr_landfill_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: _SBRLandfillLoad,
    wastewater: _WastewaterLoad,
    gas: _GasTreatment,
    database_name: str,
) -> dict[str, object]:
    """Replace waste chemistry and retain every initial site support function.

    Inputs use one retained principal kg. The site's original negative waste
    production persists; its parent purchases actual filled mass with a
    negative coefficient. Gas supports and liquid collection occur once.
    The source's initial land allocation excludes separately constructed
    sludge/ash disposal. All elementary identities remain native and EF-ready.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="filled_SBR_manufacturing_scrap_landfill",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = next(ex for ex in native if ex["type"] == "production")
    scale = -_self_production_amount(activity, native, method="filled SBR landfill") / load.mass_kg
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": load.elements_kg["C"] / load.mass_kg, "unit": "kg"},
    }
    properties.update(
        {
            f"{_ELEMENTS[el]} content": {"amount": amount / load.mass_kg, "unit": "dimensionless"}
            for el, amount in load.elements_kg.items()
        }
    )
    for origin in ("fossil", "non-fossil"):
        carbon = fsum(
            x.carbon_kg for x in load.carbon if (x.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": carbon / load.mass_kg,
            "unit": "dimensionless",
        }
    reference = _copied_native_exchange(production)
    for field in ("flow", "activity", "production volume", "classifications"):
        reference.pop(field, None)
    reference.update(
        input=(database_name, provider_service_code(service_id)),
        name=str(service["reference_product"]),
        properties=properties,
        comment=("Dry filled SBR finishing scrap with nominal ingredient composition."),
    )
    generated = [reference]
    technosphere = [ex for ex in native if ex["type"] == "technosphere"]
    leachate = next(
        ex
        for ex in technosphere
        if context.input(ex)["reference product"] == "leachate, SLF, waste plastic, mixture"
    )
    collection = _scaled_native_exchange(
        leachate, -scale * wastewater.volume_m3 / float(leachate["amount"])
    )
    for field in ("flow", "activity", "properties"):
        collection.pop(field, None)
    collection.update(
        input=(
            database_name,
            provider_service_code(
                str(cast(dict[str, object], construction["leachate"])["service_id"])
            ),
        ),
        name="filled SBR leachate collection",
    )
    generated.append(collection)
    retained = [ex for ex in technosphere if ex is not leachate]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    generated.extend(_scaled_native_exchange(ex, scale) for ex in gas.biosphere_exchanges)
    administration = fsum(
        float(ex["amount"])
        for ex in retained
        if context.input(ex)["reference product"] == "electricity, low voltage"
    )
    emissions = _sanitary_site_emissions(
        wet_mass_kg=load.mass_kg,
        carbon_kg=load.elements_kg["C"],
        later_elements_kg={
            element: load.phases[element].later_leachate_kg for element in ("C", "N", "S", "Zn")
        },
        gas_elements_kg={"Zn": load.phases["Zn"].gas_kg},
    )
    # Doka consuming-meter heat: 3.6 MJ/kWh, once. Gas heat already subtracts
    # non-credited electricity export. Short-leachate energy stays with receiving.
    emissions["Heat, waste", _AIR] = gas.heat_air_mj + 3.6 * administration / scale
    emissions["Heat, waste", _SOIL] = gas.heat_soil_mj
    # Doka (2003), volume III, section 6.5.1/C135 assigns heat to long-term groundwater. Apply that
    # convention to the actual filled manufacturing scrap's phase HHV: this allocation precedes
    # dissipation.
    emissions["Heat, waste", _LONG] = load.gas.decomposition_trace["later_leachate"]
    for ex in native:
        if ex["type"] != "biosphere":
            continue
        flow = context.input(ex)
        key = (str(flow["name"]), tuple(flow["categories"]))
        if key not in emissions:
            # Gas and short-term leachate use their separate calculations; this loop handles the
            # admitted metal releases.
            continue
        row = _scaled_native_exchange(ex, emissions[key] * scale / float(ex["amount"]))
        row.pop("properties", None)
        generated.append(row)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json([leachate])
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {
            "scrap_per_retained_principal_kg": asdict(load),
            "leachate_m3": wastewater.volume_m3,
            "gas": gas.trace,
            "administration_kwh_per_native_reference": administration,
            "placement_boundary": (
                "Initial site only; sludge and ash indicators remain "
                "with their own terminal functions."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
