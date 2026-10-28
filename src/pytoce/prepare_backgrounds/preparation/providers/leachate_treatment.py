"""Compile selected municipal collection and wastewater functions for actual loads.

Material calculations supply residue loads with their admitted compositions,
including thermoplastic starch, rubber, polyamide, resins and road materials.
Each retains its selected native treatment chain.
Doka functions act on the distinct loads and the caller's declared reference;
a collection market applies its treatment fraction once.
Chemical manufacture supplies separate ingredient loads, such as polyamide
phosphate-ester manufacturing wash. These municipal treatment functions are
applied to the declared loads.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import Any, cast

from .catalogue import provider_service_code
from .leachate_load import _LeachateLoad, _WastewaterLoad, _WastewaterProcessProfile
from .material_carbon import CARBON_ORIGINS, PrincipalCarbonOriginFractions
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .waste_gas import _GasTreatment
from .water_treatment import _HEAT_PRODUCTS, _LAND_FLOWS, _water_emission

# The native collection model reports elements. H/O follow the caller's elemental load.
_RAW_ELEMENTS = {
    "Hydrogen": "H",
    "Oxygen": "O",
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Sulfur": "S",
    "Chlorine": "Cl",
    "Sodium I": "Na",
    "Magnesium": "Mg",
    "Zinc II": "Zn",
    "Cobalt II": "Co",
    "Calcium II": "Ca",
    "Potassium I": "K",
    "Boron": "B",
    "Silicon": "Si",
    "Aluminium III": "Al",
    "Titanium ion": "Ti",
}
_EFFLUENT_ELEMENTS = {
    "Chloride": ("Cl", 1.0),
    "Sodium I": ("Na", 1.0),
    "Magnesium": ("Mg", 1.0),
    "Zinc II": ("Zn", 1.0),
    "Iron ion": ("Fe", 1.0),
    # Native Co(II), Ca(II), and K(I) discharges are in kg element.
    "Cobalt II": ("Co", 1.0),
    "Calcium II": ("Ca", 1.0),
    "Potassium I": ("K", 1.0),
    "Boron": ("B", 1.0),
    "Silicon": ("Si", 1.0),
    "Aluminium III": ("Al", 1.0),
    "Titanium ion": ("Ti", 1.0),
    # WWcalc FA/EX rows 24:26 convert kg S/N/P to kg SO4/NO3/PO4
    # using element masses 32.07/14.01/30.97 g/mol.
    "Sulfate": ("S", 96.07 / 32.07),
    "Nitrate": ("N", 62.01 / 14.01),
    "Phosphate": ("P", 94.97 / 30.97),
}
_CARBON_INDICATORS = {
    "TOC, Total Organic Carbon": "toc_kg",
    "DOC, Dissolved Organic Carbon": "doc_kg",
    "COD, Chemical Oxygen Demand": "cod_kg_o2",
    "BOD5, Biological Oxygen Demand": "bod5_kg_o2",
}


def _leachate_reference(
    exchange: Any,
    key: tuple[str, str],
    product: str,
    leachate: _LeachateLoad,
    wastewater: _WastewaterLoad,
) -> dict[str, object]:
    """Replace municipal composition on a signed cubic-meter reference.

    Wet liquid is carrier water plus element loads divided by solution volume.
    Residue loads include empirical descriptors; ingredient manufacturing
    supplies element masses. Carbon allocation uses kg C/m3, and carbon content
    uses kg C/kg dry descriptor.
    """
    dry = fsum(leachate.elements_kg.values())
    volume = wastewater.volume_m3
    properties: dict[str, object] = {
        "wet mass": {"amount": (wastewater.water.incoming_kg + dry) / volume, "unit": "kg"},
        "dry mass": {"amount": dry / volume, "unit": "kg"},
        "water in wet mass": {"amount": wastewater.water.incoming_kg / volume, "unit": "kg"},
        "carbon allocation": {"amount": leachate.toc_kg / volume, "unit": "kg"},
        "carbon content": {"amount": leachate.toc_kg / dry, "unit": "dimensionless"},
    }
    for origin in ("fossil", "non-fossil"):
        carbon = fsum(
            pool.carbon_kg
            for pool in leachate.carbon
            if (pool.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": carbon / dry,
            "unit": "dimensionless",
        }
    copied = _copied_native_exchange(exchange)
    for field in ("flow", "activity", "production volume", "classifications"):
        copied.pop(field, None)
    copied.update(
        input=key,
        name=product,
        properties=properties,
        comment="Actual wastewater on the declared municipal hydraulic reference.",
    )
    return copied


def _leachate_market_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    leachate: _LeachateLoad,
    wastewater: _WastewaterLoad,
    database_name: str,
    *,
    original_physical_reference: bool = False,
) -> dict[str, object]:
    """Retain both sewer services and apply the source collection fraction once.

    Raw bypass receives the actual pre-treatment element/indicator loads and
    incoming water. The plant's own negative production is normalized by
    Brightway; it is not another factor in the market's collection fraction.
    Constructed CH chemical wastewater keeps its original collection fraction
    when dated sewer suppliers are used; other receiving retains dated ratios.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    plant_id = str(cast(dict[str, object], construction["plant"])["service_id"])
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="leachate_collection",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="actual leachate collection")
    production = next(ex for ex in native if ex["type"] == "production")
    dated_plant = next(
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == selector["reference_product"]
    )
    plant = dated_plant
    reference_terms: dict[str, object] = {}
    if original_physical_reference:
        original = context.source.activity(selector, year=2025)
        reference = context.exchanges(original)
        p0 = float(next(ex["amount"] for ex in reference if ex["type"] == "production"))
        original_plant = next(
            ex
            for ex in reference
            if ex["type"] == "technosphere"
            and context.input(ex)["reference product"] == selector["reference_product"]
        )
        # The constructed CH water load retains its original treated fraction
        # c0/p0. Only the collection's signed reference basis changes to p.
        plant = _scaled_native_exchange(original_plant, p / p0)
        reference_terms = {"physical_reference_source": list(original.key)}
    treated = float(cast(float, plant["amount"])) / p
    bypass_scale = -p * (1 - treated) / wastewater.volume_m3
    generated = [
        _leachate_reference(
            ex,
            (database_name, provider_service_code(sid)),
            str(service["reference_product"]),
            leachate,
            wastewater,
        )
        for ex, sid in ((production, service_id), (plant, plant_id))
    ]
    retained = [ex for ex in native if ex["type"] == "technosphere" and ex is not dated_plant]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    raw = leachate
    amounts = {name: raw.elements_kg.get(element, 0.0) for name, element in _RAW_ELEMENTS.items()}
    amounts["Chlorine"] -= raw.chloride_kg
    amounts.update({name: float(getattr(raw, field)) for name, field in _CARBON_INDICATORS.items()})
    # Native Water is m3. Divide carrier kg by the admitted water
    # density, 1000 kg/m3; the caller supplies total solution volume separately.
    amounts["Water"] = wastewater.water.incoming_kg / 1000
    native_names = set()
    for ex in native:
        if ex["type"] != "biosphere":
            continue
        name = str(context.input(ex)["name"])
        native_names.add(name)
        if name in amounts and amounts[name] != 0:
            generated.append(
                _scaled_native_exchange(ex, amounts[name] * bypass_scale / float(ex["amount"]))
            )
    # Actual UPR glass and road minerals introduce these loads beyond their
    # selected donors. Preserve any native bypass template; otherwise bind the
    # admitted actual raw element to its existing surface-water identity once.
    for name in (
        "Boron",
        "Silicon",
        "Calcium II",
        "Aluminium III",
        "Titanium ion",
        "Potassium I",
        "Magnesium",
        "Sodium I",
        "Sulfur",
    ):
        if name not in native_names and amounts[name] != 0:
            generated.append(
                _water_emission(
                    context,
                    name,
                    "kilogram",
                    ["water", "surface water"],
                    amounts[name] * bypass_scale,
                )
            )
    # Known aqueous chloride bypasses as its existing ion flow, independently of the native Chlorine
    # (Cl2) template. It remains part of total Cl in the conditional plant load.
    if raw.chloride_kg:
        generated.append(
            _water_emission(
                context,
                "Chloride",
                "kilogram",
                ["water", "surface water"],
                raw.chloride_kg * bypass_scale,
            )
        )
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json([dated_plant])
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {
            "treated_fraction": treated,
            "raw_load_per_caller_reference": asdict(raw),
            "plant_service_id": plant_id,
            **reference_terms,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "cubic meter",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }


@dataclass(frozen=True, slots=True)
class _WastewaterDischarge:
    """One calculated wastewater release on the actual treated-load reference.

    Name, unit and categories describe an existing native elementary flow;
    its database identity belongs to the compiler. Biological CO2 alone carries
    principal-carbon origin fractions, leaving auxiliary carbon outside that
    numerator. No source production or native exchange amount enters this record.
    """

    name: str
    amount: float
    unit: str
    categories: tuple[str, ...]
    principal_carbon_origin_fractions: PrincipalCarbonOriginFractions | None = None


def _wastewater_discharges(
    wastewater: _WastewaterLoad,
    profile: _WastewaterProcessProfile,
    *,
    gross_electricity_kwh: float,
    gas_heat_air_mj: float,
    purchased_heat_mj: float,
) -> list[_WastewaterDischarge]:
    """Calculate effluent and plant releases before collection or native scaling.

    All quantities share the actual ``wastewater`` volume/load reference.
    Gross electricity includes the source's allocated construction electricity
    before subtraction of internal digester generation. Purchased heat already
    excludes internally supplied heat. Gas heat is the separate gas owner's
    atmospheric heat output; its other emissions and supplies remain there.

    Preserve source descriptor order and the distinct biological/hydraulic air
    allocations. The result excludes native land/infrastructure and gas rows,
    and contains no database access, flow binding or source-production basis.
    """
    generated: list[_WastewaterDischarge] = []
    effluent = wastewater.effluent
    aqueous = {
        name: effluent.elements_kg[element] * factor
        for name, (element, factor) in _EFFLUENT_ELEMENTS.items()
    }
    aqueous.update(
        {name: float(getattr(effluent, field)) for name, field in _CARBON_INDICATORS.items()}
    )
    # Native Water is m3; divide the carrier by 1000 kg/m3.
    aqueous["Water"] = wastewater.water.effluent_kg / 1000
    # WWcalc B140 transfers the C/N/S biological-energy allocation to water.
    biological = wastewater.biological_air_elements_kg
    aqueous["Heat, waste"] = (
        34.016 * biological["C"] + 19.079 * biological["S"] + 6.276 * biological["N"]
    )
    for name, mass in aqueous.items():
        if mass != 0:
            unit = (
                "cubic meter"
                if name == "Water"
                else "megajoule"
                if name == "Heat, waste"
                else "kilogram"
            )
            generated.append(_WastewaterDischarge(name, mass, unit, ("water", "surface water")))
    for compartment, bio_share, hydraulic_share in (
        (
            "urban air close to ground",
            profile.hydraulic.urban_treatment,
            profile.hydraulic.urban_emissions,
        ),
        (
            "non-urban air or from high stacks",
            1 - profile.hydraulic.urban_treatment,
            1 - profile.hydraulic.urban_emissions,
        ),
    ):
        air = ("air", compartment)
        for origin in ("fossil", "non-fossil"):
            carbon = fsum(
                pool.biological_air_carbon_kg
                for pool in wastewater.carbon
                if (pool.origin == "fossil") == (origin == "fossil")
            )
            if carbon != 0:
                # Biological CO2 combines actual pools in each native origin group. Auxiliaries
                # share the flow; principal numerators retain their own carbon.
                principal_origins = cast(
                    PrincipalCarbonOriginFractions,
                    tuple(
                        fsum(
                            pool.biological_air_carbon_kg
                            for pool in wastewater.carbon
                            if pool.principal
                            and pool.origin == actual_origin
                            and (pool.origin == "fossil") == (origin == "fossil")
                        )
                        / carbon
                        for actual_origin in CARBON_ORIGINS
                    ),
                )
                # Source gas convention: 44 kg CO2 contains 12 kg carbon.
                generated.append(
                    _WastewaterDischarge(
                        f"Carbon dioxide, {origin}",
                        bio_share * carbon * 44 / 12,
                        "kilogram",
                        air,
                        principal_origins,
                    )
                )
        if wastewater.biological_n2o_nitrogen_kg != 0:
            # Two N atoms: 28 kg N per 44 kg N2O on Doka's nominal mass basis.
            generated.append(
                _WastewaterDischarge(
                    "Dinitrogen monoxide",
                    bio_share * wastewater.biological_n2o_nitrogen_kg * 44 / 28,
                    "kilogram",
                    air,
                )
            )
        generated.append(
            _WastewaterDischarge(
                "Water",
                hydraulic_share * wastewater.water.hydraulic_evaporation_kg / 1000,
                "cubic meter",
                air,
            )
        )
        # WWcalc B133:B138 allocates electricity (3.6 MJ/kWh) and gas heat together by Infra AT16.
        # Construction electricity remains a lifecycle allocation. Delivered external heat is
        # added at consumption; its complete supplier retains upstream losses.
        heat = hydraulic_share * (3.6 * gross_electricity_kwh + gas_heat_air_mj + purchased_heat_mj)
        generated.append(_WastewaterDischarge("Heat, waste", heat, "megajoule", air))
    generated.append(
        _WastewaterDischarge(
            "Oxygen",
            wastewater.duties.biological_oxygen_kg,
            "kilogram",
            ("natural resource", "in air"),
        )
    )
    # The source disposal indicator uses 1000 kg per m3 treatment service.
    generated.append(
        _WastewaterDischarge(
            "Non-hazardous waste disposed",
            1000 * wastewater.volume_m3,
            "kilogram",
            ("inventory indicator", "waste"),
        )
    )
    return generated


def _wastewater_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    leachate: _LeachateLoad,
    wastewater: _WastewaterLoad,
    gas: _GasTreatment,
    database_name: str,
    profile: _WastewaterProcessProfile,
    *,
    original_physical_reference: bool = False,
) -> dict[str, object]:
    """Compile complete conditional treatment before the collection-market share.

    Actual loads replace municipal biological emissions, purchased processing
    energy, Fe/PAM, and sludge. Native infrastructure, retirement, land use and
    its embodied operating electricity remain on their hydraulic basis. The
    gas record supplies gas-related elementary flows, including plant O2/H2O.
    Native operating supplies remain at this treatment owner.
    Constructed CH chemical wastewater retains the original embodied-electricity
    duty per treated cubic meter while its complete suppliers follow the date.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="actual_wastewater_treatment",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="leachate wastewater treatment")
    scale = -p / wastewater.volume_m3
    production = next(ex for ex in native if ex["type"] == "production")
    generated = [
        _leachate_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            leachate,
            wastewater,
        )
    ]
    technosphere = [ex for ex in native if ex["type"] == "technosphere"]
    grouped: dict[str, list[Any]] = {}
    for ex in technosphere:
        product = str(context.input(ex)["reference product"])
        if product in _HEAT_PRODUCTS:
            product = "heat, district or industrial"
        grouped.setdefault(product, []).append(ex)
    # Infra G28/G33 allocates 23375 kWh embodied construction electricity to
    # 13250 kg limestone. Keep this separate from WWcalc's actual-load meters.
    limestone = fsum(float(ex["amount"]) for ex in grouped["limestone, crushed, washed"])
    reference_terms: dict[str, object] = {}
    if original_physical_reference:
        original = context.source.activity(
            cast(dict[str, object], construction["source"]), year=2025
        )
        reference = context.exchanges(original)
        p0 = float(next(ex["amount"] for ex in reference if ex["type"] == "production"))
        original_limestone = fsum(
            float(ex["amount"])
            for ex in reference
            if ex["type"] == "technosphere"
            and context.input(ex)["reference product"] == "limestone, crushed, washed"
        )
        # L0/(-p0) is the original construction allocation per treated m3. Apply it to -p m3.
        limestone = p / p0 * original_limestone
        reference_terms = {
            "physical_reference_source": list(original.key),
            "embodied_electricity_kwh_per_treated_m3": (original_limestone / -p0 * 23375 / 13250),
        }
    gross_electricity = scale * wastewater.duties.electricity_kwh + limestone * 23375 / 13250
    # Doka 2021 section 10 consumes recovered digester energy inside the plant.
    # Subtract it once from purchased utilities; no energy output or credit is
    # created. The gas owner reports separate HHV heat and LHV generation.
    electricity = gross_electricity - scale * gas.internal_electricity_kwh
    purchased_heat = wastewater.duties.heat_mj - gas.internal_heat_mj
    target = {
        "electricity, medium voltage": electricity,
        "heat, district or industrial": scale * purchased_heat,
        "iron sulfate": scale * wastewater.duties.iron_sulfate_kg,
        "polyacrylamide": scale * wastewater.duties.polyacrylamide_kg,
        "inert waste, for final disposal": -scale * wastewater.duties.sand_kg,
        # Rows 61846:61861 and 61864:61879 provide the regional inventories for
        # macro-biomass and macroplastic hydraulic screenings, respectively.
        "waste graphical paper": -scale * wastewater.duties.grit_biomass_kg,
        "waste plastic, mixture": -scale * wastewater.duties.grit_plastic_kg,
    }
    sludge = {
        str(construction["incineration_reference_product"]): (
            "incineration",
            wastewater.incinerated_sludge_wet_kg,
        ),
        str(construction["landfarming_reference_product"]): (
            "landfarming",
            wastewater.landfarmed_sludge_wet_kg,
        ),
    }
    retained: list[Any] = []
    replaced: list[Any] = []
    for ex in technosphere:
        node = context.input(ex)
        product = str(node["reference product"])
        if product in _HEAT_PRODUCTS:
            product = "heat, district or industrial"
        if product in sludge:
            role, wet = sludge[product]
            sid = str(cast(dict[str, object], construction[role])["service_id"])
            row = _scaled_native_exchange(ex, -scale * wet / float(ex["amount"]))
            for field in ("flow", "activity", "properties"):
                row.pop(field, None)
            row.update(input=(database_name, provider_service_code(sid)), name=sid)
            generated.append(row)
            replaced.append(ex)
        elif product in target:
            if target[product] != 0:
                total = fsum(float(row["amount"]) for row in grouped[product])
                generated.append(_scaled_native_exchange(ex, target[product] / total))
            replaced.append(ex)
        else:
            generated.append(_copied_native_exchange(ex))
            retained.append(ex)
    generated.extend(_scaled_native_exchange(ex, scale) for ex in gas.biosphere_exchanges)
    retained_land = [
        ex
        for ex in native
        if ex["type"] == "biosphere" and context.input(ex)["name"] in _LAND_FLOWS
    ]
    generated.extend(_copied_native_exchange(ex) for ex in retained_land)
    # The pure discharge owner uses one actual-load basis. Native production
    # scaling stays here; divide the already scaled gross electrical allocation
    # once, then apply the same scale when binding its calculated releases.
    discharges = _wastewater_discharges(
        wastewater,
        profile,
        gross_electricity_kwh=gross_electricity / scale,
        gas_heat_air_mj=gas.heat_air_mj,
        purchased_heat_mj=purchased_heat,
    )
    for discharge in discharges:
        row = _water_emission(
            context,
            discharge.name,
            discharge.unit,
            list(discharge.categories),
            scale * discharge.amount,
        )
        if discharge.principal_carbon_origin_fractions is not None:
            row["principal_carbon_origin_fractions"] = discharge.principal_carbon_origin_fractions
        generated.append(row)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json(replaced)
    trace["retained_biosphere_json"] = context.exchange_json(retained_land)
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere" and ex not in retained_land
    )
    trace["project_terms_json"] = json.dumps(
        {
            "load_per_caller_reference": asdict(wastewater),
            "gas": gas.trace,
            "electricity_kwh_per_native_reference": electricity,
            "internal_electricity_kwh": gas.internal_electricity_kwh,
            "internal_heat_mj": gas.internal_heat_mj,
            "hydraulic_urban_fraction": profile.hydraulic.urban_emissions,
            "biological_gas_urban_fraction": profile.hydraulic.urban_treatment,
            **reference_terms,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "cubic meter",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }


def _sludge_market_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    wastewater: _WastewaterLoad,
    database_name: str,
    *,
    original_physical_reference: bool = False,
) -> dict[str, object]:
    """Carry actual leachate sludge through its existing complete terminal haulage.

    Incineration and landfarming have separate 70% and 97% water references.
    Keep each market's negative production and signed child amount; Brightway
    alone divides by the child's production. The market uses the disposal quantity already
    calculated by the plant.
    Constructed CH wastewater preserves the original terminal-to-market ratio
    on the dated signed production basis; its dated haulage remains unchanged.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    child = str(cast(dict[str, object], construction["treatment"])["service_id"])
    if construction["sludge_fraction"] == "incineration":
        fraction = wastewater.incineration_dry_share
        wet = wastewater.incinerated_sludge_wet_kg
        water = wastewater.water.incinerated_sludge_water_kg
    else:
        fraction = wastewater.landfarming_dry_share
        wet = wastewater.landfarmed_sludge_wet_kg
        water = wastewater.water.landfarmed_sludge_water_kg
    dry = wastewater.final_sludge.dry_mass_kg * fraction
    carbon = wastewater.final_sludge.elements_kg["C"] * fraction
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": dry / wet, "unit": "kg"},
        "water in wet mass": {"amount": water / wet, "unit": "kg"},
        "carbon allocation": {"amount": carbon / wet, "unit": "kg"},
        "carbon content": {"amount": carbon / dry, "unit": "dimensionless"},
    }
    for origin in ("fossil", "non-fossil"):
        pool_carbon = fsum(
            pool.final_sludge_carbon_kg * fraction
            for pool in wastewater.carbon
            if (pool.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": pool_carbon / dry,
            "unit": "dimensionless",
        }
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="sludge_collection",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    reference_terms: dict[str, object] = {}
    rows = native
    if original_physical_reference:
        original = context.source.activity(selector, year=2025)
        reference = context.exchanges(original)
        p = float(next(ex["amount"] for ex in native if ex["type"] == "production"))
        p0 = float(next(ex["amount"] for ex in reference if ex["type"] == "production"))
        original_children = [
            ex
            for ex in reference
            if ex["type"] == "technosphere"
            and context.input(ex)["reference product"] == selector["reference_product"]
        ]
        # Preserve original kg terminal/kg collection on its specified production reference.
        rows = tuple(
            ex
            for ex in native
            if ex["type"] == "production"
            or context.input(ex)["reference product"] != selector["reference_product"]
        ) + tuple(_scaled_native_exchange(ex, p / p0) for ex in original_children)
        reference_terms = {
            "physical_reference_source": list(original.key),
            "terminal_amount_per_collection_unit": (
                fsum(float(ex["amount"]) for ex in original_children) / p0
            ),
        }
    generated: list[dict[str, object]] = []
    retained: list[Any] = []
    for ex in rows:
        node = context.input(ex)
        if ex["type"] == "production" or node["reference product"] == selector["reference_product"]:
            target = service_id if ex["type"] == "production" else child
            row = _copied_native_exchange(ex)
            for field in ("flow", "activity", "production volume", "classifications"):
                row.pop(field, None)
            row.update(
                input=(database_name, provider_service_code(target)),
                name=str(service["reference_product"]),
                properties=properties,
                comment=(
                    "Actual leachate sludge and carried water; dry mass contains the separate "
                    "source organic-H/oxide-O allocations."
                ),
            )
            generated.append(row)
        else:
            generated.append(_copied_native_exchange(ex))
            retained.append(ex)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json(
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == selector["reference_product"]
    )
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = "[]"
    trace["project_terms_json"] = json.dumps(
        {
            "sludge_fraction": construction["sludge_fraction"],
            "wet_kg_per_caller_reference": wet,
            "dry_kg_per_caller_reference": dry,
            "carbon_kg_per_caller_reference": carbon,
            "treatment_service_id": child,
            **reference_terms,
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
