"""Compile actual cured materials at sanitary placement and complete receiving.

One public unit receives one kg principal resin plus its actual auxiliaries.
Native selected site, collection, wastewater, sludge and ash functions retain
their supplied operating chains. The actual cured material defines waste chemistry. Principal demand
retains its stated reference basis.
"""

import json
import re
from dataclasses import asdict
from math import fsum
from typing import cast

from .ash_treatment import (
    _ELEMENTS,
    _ROAD_ASH_RELEASES,
    _UPR_ASH_RELEASES,
    _ash_dataset,
    _ash_market_dataset,
    _AshReleaseProfile,
)
from .catalogue import provider_service_code
from .epoxy_landfill_load import _epoxy_landfill_load
from .landfill_phases import _LandfillLoad, _LandfillSite
from .landfill_site import _sanitary_site_emissions
from .leachate_load import _ROW_WASTEWATER_PROFILE, _CarbonOrigin, _wastewater_load, _WastewaterLoad
from .leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .road_landfill_load import _road_landfill_load
from .sludge_incineration import _sludge_dataset, _sludge_source
from .sludge_incineration_load import calculate_sludge_incineration
from .sludge_landfarming import _landfarming_dataset
from .treatment_compilation import _direct_principal_carbon_owner
from .upr_landfill_load import _upr_landfill_load
from .waste_gas import _GasTreatment, _landfill_gas, _wastewater_gas
from .water_treatment import _ROW_PROFILE, _water_evaporation


def _resin_disposal_reference(
    load: _LandfillLoad, key: tuple[str, str], product: str, material_name: str
) -> dict[str, object]:
    """Describe actual cured material on a positive principal-kg service unit.

    Native element contents are fractions of dry matter. Carrier water stays
    in the wet-mass property; geological C is metadata outside the
    organic carbon content and fate model.
    """
    dry_mass = load.wet_mass_kg - load.water_kg
    properties: dict[str, object] = {
        "wet mass": {"amount": load.wet_mass_kg, "unit": "kg"},
        "dry mass": {"amount": dry_mass, "unit": "kg"},
        "water in wet mass": {"amount": load.water_kg, "unit": "kg"},
        "carbon allocation": {"amount": load.elements_kg["C"], "unit": "kg"},
        "geological carbon outside organic fate": {
            "amount": load.geological_carbon_kg,
            "unit": "kg",
        },
    }
    for element in load.elements_kg:
        name = {"H": "hydrogen", "O": "oxygen", **_ELEMENTS}[element]
        properties[f"{name} content"] = {
            "amount": load.elements_kg[element] / dry_mass,
            "unit": "dimensionless",
        }
    for category in ("fossil", "non-fossil"):
        properties[f"carbon content, {category}"] = {
            "amount": fsum(
                pool.phases.input_kg
                for pool in load.carbon
                if (pool.origin == "fossil") == (category == "fossil")
            )
            / dry_mass,
            "unit": "dimensionless",
        }
    return {
        "input": key,
        "type": "production",
        "amount": 1.0,
        "unit": "kilogram",
        "name": product,
        "properties": properties,
        "comment": (f"One kg principal {material_name} plus its admitted retained auxiliaries."),
    }


def _resin_landfill_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Connect the selected dated receiving functions to actual cured resin.

    Fixed original site climate/hydraulics and source WWT process fractions
    determine the load. Dated complete suppliers retain their own
    signs, production references and uncertainty; collection is applied once.
    """
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    material_name = kind.split("_", 1)[0]
    origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
    profile = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], context.catalogue["semantic_contract"])[
            f"{material_name}_landfill"
        ],
    )
    source = context.source.activity(profile["landfill_source"], year=2025)
    climate = {
        name: float(value)
        for name, value in re.findall(r"\((MAT|MAP)\) of ([0-9.]+)", str(source["comment"]))
    }
    site = _LandfillSite(climate["MAT"], climate["MAP"], 1.0)
    if material_name == "road":
        load = _road_landfill_load(
            next(
                row
                for row in context.material_recipe.conversion
                if row.operation == "others_road_marking_conversion"
            ),
            origin,
            site,
            carbonate_end_year=float(cast(float, profile["carbonate_end_year"])),
        )
    elif material_name == "upr":
        load = _upr_landfill_load(
            next(
                row
                for row in context.material_recipe.conversion
                if row.operation == "others_upr_hand_lay_up_conversion"
            ),
            origin,
            site,
            carbonate_end_year=float(cast(float, profile["carbonate_end_year"])),
        )
    else:
        load = _epoxy_landfill_load(origin, site)
    rows = context.exchanges(source)
    production = _self_production_amount(source, rows, method=f"{material_name} sanitary source")
    liquid = next(
        row
        for row in rows
        if row["type"] == "technosphere"
        and context.input(row)["reference product"]
        == profile["wastewater_source"]["reference_product"]
    )
    # The selected native site supplies leachate m3/kg actual matrix.
    # Its negative wastewater and production signs cancel once. Actual solutes
    # displace water within that volume.
    volume = float(liquid["amount"]) / production * load.wet_mass_kg
    plant = context.source.activity(profile["wastewater_source"], year=2025)
    wastewater = _wastewater_load(
        load.leachate,
        volume_m3=volume,
        incoming_water_kg=1000 * volume - fsum(load.leachate.elements_kg.values()),
        evaporation_fraction=_water_evaporation(str(plant["comment"]), _ROW_PROFILE),
        profile=_ROW_WASTEWATER_PROFILE,
    )
    service_id = str(service["provider_service_id"])
    if kind == f"{material_name}_landfill":
        gas = _landfill_gas(context, construction, load.gas, service_id)
        return _resin_landfill_dataset(context, service, load, wastewater, gas, database_name)
    if kind == f"{material_name}_leachate_collection":
        return _leachate_market_dataset(context, service, load.leachate, wastewater, database_name)
    if kind == f"{material_name}_wastewater_treatment":
        gas = _wastewater_gas(
            context, construction, wastewater, service_id, profile=_ROW_WASTEWATER_PROFILE
        )
        data = _wastewater_dataset(
            context,
            service,
            load.leachate,
            wastewater,
            gas,
            database_name,
            profile=_ROW_WASTEWATER_PROFILE,
        )
    elif kind == f"{material_name}_sludge_collection":
        return _sludge_market_dataset(context, service, wastewater, database_name)
    elif kind == f"{material_name}_sludge_landfarming":
        data = _landfarming_dataset(
            context,
            service,
            wastewater,
            f"{material_name}_cured_matrix",
            database_name,
            complete_carbon_oxidation=cast(bool, construction["complete_carbon_oxidation"]),
        )
    else:
        thermal = (
            construction
            if kind == f"{material_name}_sludge_incineration"
            else cast(
                dict[str, object],
                services[str(construction["incineration_service_id"])]["construction"],
            )
        )
        sludge = calculate_sludge_incineration(
            _sludge_source(context, thermal),
            wastewater,
            f"{material_name}_cured_matrix",
            source_carbon_origins=tuple(
                dict.fromkeys(cast(dict[str, str], thermal["carbon_templates"]).values())
            ),
            additional_apc_water_flows=tuple(
                cast(list[str], thermal["additional_apc_water_flows"])
            ),
            phosphorus_phase_reference=False,
            # Original MIXED and paint slag give water/dry mass, respectively:
            # .197802/.802198 and .186047/.813953 kg/kg dry slag.
            bottom_ash_water_to_dry_ratio=(
                0.197802 / 0.802198 if material_name == "upr" else 0.186047 / 0.813953
            ),
            # Actual road glass/rutile supplies these known elements. Other
            # materials retain their independently admitted receiving choices.
            additional_mineral_elements=(
                ("Al", "Ti", "K", "Mg", "Na", "S") if material_name == "road" else ()
            ),
        )
        if kind == f"{material_name}_sludge_incineration":
            data = _sludge_dataset(context, service, sludge, database_name)
        else:
            ash = (
                sludge.bottom_ash
                if construction["ash_fraction"] == "bottom"
                else sludge.residual_ash
            )
            if kind == f"{material_name}_sludge_ash_market":
                return _ash_market_dataset(context, service, ash, database_name)
            # Paint ash retains its native subunit N release. Only actual fresh
            # catalyst/lime elements absent from that donor need source factors:
            # Doka 2003 III Table A8, p118, short / cumulative elemental release.
            ash_profile = _AshReleaseProfile(
                ()
                if construction["ash_fraction"] == "bottom"
                else (
                    ("Titanium ion", "Ti", 1, 0.0004956, 0.2973),
                    ("Vanadium V", "V", 1, 0.002454, 0.7711),
                    ("Calcium II", "Ca", 1, 0.0001508, 0.09051),
                ),
            )
            return _ash_dataset(
                context,
                service,
                ash,
                database_name,
                release_profile=(
                    _UPR_ASH_RELEASES[str(construction["ash_fraction"])]
                    if material_name == "upr"
                    else _ROAD_ASH_RELEASES[str(construction["ash_fraction"])]
                    if material_name == "road"
                    else ash_profile
                ),
            )
    data["direct_treatment_carbon_owners_json"] = json.dumps(
        [_direct_principal_carbon_owner(data, database_name, origin, material_name)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return data


def _resin_landfill_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: _LandfillLoad,
    wastewater: _WastewaterLoad,
    gas: _GasTreatment,
    database_name: str,
) -> dict[str, object]:
    """Retain complete initial site operation per principal kg received."""
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    material_name = str(construction["kind"]).split("_", 1)[0]
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id=f"actual_cured_{material_name}_landfill",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = _self_production_amount(
        activity, native, method=f"{material_name} sanitary landfill"
    )
    scale = load.wet_mass_kg / -production
    key = database_name, provider_service_code(service_id)
    generated = [
        _resin_disposal_reference(load, key, str(service["reference_product"]), material_name)
    ]
    leachate_id = str(cast(dict[str, object], construction["leachate"])["service_id"])
    profile = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], context.catalogue["semantic_contract"])[
            f"{material_name}_landfill"
        ],
    )
    technosphere = [row for row in native if row["type"] == "technosphere"]
    leachate = next(
        row
        for row in technosphere
        if context.input(row)["reference product"]
        == profile["wastewater_source"]["reference_product"]
    )
    generated.append(
        {
            "input": (database_name, provider_service_code(leachate_id)),
            "type": "technosphere",
            "amount": -wastewater.volume_m3,
            "loc": -wastewater.volume_m3,
            "uncertainty type": 0,
        }
    )
    retained = [row for row in technosphere if row is not leachate]
    generated.extend(_scaled_native_exchange(row, scale) for row in retained)
    generated.extend(_copied_native_exchange(row) for row in gas.biosphere_exchanges)
    emissions = _sanitary_site_emissions(
        wet_mass_kg=load.wet_mass_kg,
        carbon_kg=load.elements_kg["C"],
        later_elements_kg={
            element: phase.later_leachate_kg for element, phase in load.phases.items()
        },
        gas_elements_kg={element: phase.gas_kg for element, phase in load.phases.items()},
    )
    administration = fsum(
        float(row["amount"]) * scale
        for row in retained
        if context.input(row)["reference product"] == "electricity, low voltage"
    )
    source_outputs: dict[tuple[str, tuple[str, ...]], list[dict[str, object]]] = {}
    for row in native:
        if row["type"] != "biosphere":
            continue
        flow = context.input(row)
        flow_key = str(flow["name"]), tuple(flow["categories"])
        if flow_key not in emissions:
            continue
        source_outputs.setdefault(flow_key, []).append(row)
    for flow_key, source_rows in source_outputs.items():
        multiplier = emissions[flow_key] / fsum(
            float(cast(float, row["amount"])) for row in source_rows
        )
        for row in source_rows:
            if multiplier == 0.0:
                # Retain explicit zero amounts as deterministic exchanges.
                generated_row = _copied_native_exchange(row)
                for field in ("scale", "shape", "minimum", "maximum", "negative"):
                    generated_row.pop(field, None)
                generated_row.update(amount=0.0, loc=0.0, **{"uncertainty type": 0})
            else:
                generated_row = _scaled_native_exchange(row, multiplier)
            generated_row.pop("properties", None)
            generated.append(generated_row)
    if material_name in {"upr", "road"}:
        # Actual UPR glass/silica requires B/Si/Ca receiving beyond the MIXED
        # source vector. Native Al/Co above retain their compatible uncertainty.
        additions = [
            ("Boron", ("water", "ground-, long-term")),
            ("Silicon", ("water", "ground-, long-term")),
            ("Calcium II", ("water", "ground-, long-term")),
            ("Silicon", ("air", "non-urban air or from high stacks")),
            ("Calcium II", ("air", "non-urban air or from high stacks")),
        ]
        if material_name == "road":
            # The retained road binder includes glass, TiO2 and limestone.
            # Their known atoms use the corresponding landfill functions.
            # Carbonate C remains outside the organic-release calculation.
            additions = [
                (name, compartment)
                for name in (
                    "Silicon",
                    "Calcium II",
                    "Aluminium III",
                    "Titanium ion",
                    "Potassium I",
                    "Magnesium",
                    "Sodium I",
                )
                for compartment in (
                    ("water", "ground-, long-term"),
                    ("air", "non-urban air or from high stacks"),
                )
            ] + [
                ("Sulfate", ("water", "ground-, long-term")),
                ("Hydrogen sulfide", ("water", "ground-, long-term")),
            ]
        for name, categories in additions:
            if (name, categories) in source_outputs:
                continue  # A native positive template already owns this actual release.
            amount = emissions[name, categories]
            flow = context.biosphere.flow(
                {"name": name, "unit": "kilogram", "categories": list(categories)}
            )
            generated.append(
                {
                    "input": flow.key,
                    "type": "biosphere",
                    "name": name,
                    "unit": "kilogram",
                    "amount": amount,
                    "loc": amount,
                    "uncertainty type": 0,
                }
            )
    # Doka (2003) and Landfill2020 separate short and later ground heat.
    # Bind matrix/gas quantities and soil heat to the selected native sites.
    # Short-leachate HHV belongs to its receiver. Convert administrative
    # electricity from kWh to MJ heat using 3.6.
    heat = {
        ("air", "non-urban air or from high stacks"): gas.heat_air_mj + 3.6 * administration,
        ("water", "ground-, long-term"): load.gas.decomposition_trace["later_leachate_hhv_mj"],
    }
    for compartment, amount in heat.items():
        source_rows = [
            row
            for row in native
            if row["type"] == "biosphere"
            and context.input(row)["name"] == "Heat, waste"
            and tuple(context.input(row)["categories"]) == compartment
        ]
        multiplier = amount / fsum(float(row["amount"]) for row in source_rows)
        for row in source_rows:
            generated_row = _scaled_native_exchange(row, multiplier)
            generated_row.pop("properties", None)
            generated.append(generated_row)
    soil_heat = context.biosphere.flow(
        {"name": "Heat, waste", "unit": "megajoule", "categories": ["soil", "industrial"]}
    )
    generated.append(
        {
            "input": cast(tuple[str, str], soil_heat.key),
            "type": "biosphere",
            "name": "Heat, waste",
            "unit": "megajoule",
            "amount": gas.heat_soil_mj,
            "loc": gas.heat_soil_mj,
            "uncertainty type": 0,
        }
    )
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json([leachate])
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {
            "schema": f"pytoce_{material_name}_landfill_reference_schema_v1",
            "actual_matrix_per_principal_kg": asdict(load),
            "leachate_m3": wastewater.volume_m3,
            "gas": gas.trace,
            "native_production": production,
            "administration_kwh": administration,
            "bound_heat_mj": {
                "rural_air": heat["air", "non-urban air or from high stacks"],
                "industrial_soil": gas.heat_soil_mj,
                "long_groundwater": heat["water", "ground-, long-term"],
            },
            "short_leachate_hhv_mj_at_receiver": load.gas.decomposition_trace[
                "short_leachate_hhv_mj"
            ],
            "placement_boundary": (
                "Initial site; sludge and ash disposal remain at their own recipients."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    data: dict[str, object] = {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
    data["direct_treatment_carbon_owners_json"] = json.dumps(
        [
            _direct_principal_carbon_owner(
                data,
                database_name,
                cast(_CarbonOrigin, construction["principal_carbon_origin"]),
                material_name,
            )
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return data
