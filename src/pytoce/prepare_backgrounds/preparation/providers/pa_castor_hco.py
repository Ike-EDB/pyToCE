"""Supply crude castor oil and hydrogenated castor oil for PA fiber spin finish.

The constructed castor mill uses combined crude-oil/seed-cake allocation. Its
output feeds the separately owned complete native refinery proxy. Integrated
oil hardening consumes refined oil and a dry supported-Ni powder, and passes
retained oil, catalyst, bleaching earth and citric-acid-equivalent organic
material to its combined receiver.
"""

import json
from dataclasses import asdict, dataclass
from typing import cast

from .catalogue import provider_service_code
from .hco_nickel import _HCO_NICKEL_SHARE
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange

# The admitted PA nominal triglycerides use CIAAW masses in kg/kmol: original castor C57H104O9 and
# hydrogenated C57H110O9. These references distinguish oil carbon from native crop allocation.
_ATOMS = {"C": 12.011, "H": 1.008, "O": 15.999}
_CASTOR_MASS = 57 * _ATOMS["C"] + 104 * _ATOMS["H"] + 9 * _ATOMS["O"]
_HCO_MASS = _CASTOR_MASS + 6 * _ATOMS["H"]
# CN103305339A Table 8 fixes the refined-oil inlet to HCO at 0.05% water
# on a wet-mass basis, separately from the combined crude annotation below.
_PRODUCT_WATER = 0.0005 / (1 - 0.0005)
# Table 1's combined crude oil contains 0.32% water on its wet-mass basis.
_CRUDE_WATER = 0.0032 / (1 - 0.0032)
_RESOURCE_CORRECTION = "Carbon dioxide, non-fossil, resource correction"


@dataclass(frozen=True, slots=True)
class _CastorReference:
    """Fixed unallocated and allocated quantities per kg dry combined crude oil.

    Unallocated quantities and the source dry throughput are in kg.
    Dryer inputs are MJ fuel and kWh before local
    oil/cake allocation. Seed, crude moisture and coproduct quantities
    remain separate from allocated LCI demands and the downstream refinery.
    """

    physical_kg: dict[str, float]
    allocation: dict[str, float]
    dryer_fuel_mj: float
    dryer_electricity_kwh: float
    dryer_heat_mj: float


@dataclass(frozen=True, slots=True)
class _HcoReference:
    """One kg dry HCO and its complete upstream oil/catalyst/cake quantities.

    All quantities are kg. The selected dry catalyst contains nickel and silica. Water accompanying
    the dry refined oil is condensed separately at hydrogenation using the fixed refined-oil
    moisture.
    """

    castor_oil_kg: float
    castor_water_kg: float
    hydrogen_net_kg: float
    hydrogen_supply_kg: float
    hydrogen_vent_kg: float
    catalyst_kg: float
    filter_aid_kg: float
    citric_acid_kg: float
    cake_kg: float
    cake_components_kg: dict[str, float]


def _original_properties(
    context: _RecipeContext, selector: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Read original material properties retained before premise transformations."""
    node = context.source.activity(selector, year=2025)
    rows = context.exchanges(node)
    _self_production_amount(node, rows, method="PA castor original material")
    production = next(row for row in rows if row["type"] == "production")
    return cast(dict[str, dict[str, object]], production["properties"])


def _castor_reference(context: _RecipeContext, construction: dict[str, object]) -> _CastorReference:
    """Solve the admitted dry-seed and combined-crude-oil/cake allocation.

    Seber et al. (2022) supply 47% dry-seed oil, 96% extraction and 3.5% processing
    moisture. CN103305339A Table 1 supplies the combined crude's moisture
    annotation. Native extraction is deallocated through its stated dry
    throughput, then reallocated once by actual dry oil and cake. The complete
    refinery owns its separate native allocation; no castor price vector is used.
    """
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    seed = _original_properties(context, selectors["seed"])
    seed_dry = 1 / (0.47 * 0.96)
    seed_wet = seed_dry / float(cast(float, seed["dry mass"]["amount"]))
    seed_processing_wet = seed_dry / 0.965
    evaporated_seed_water = seed_wet - seed_processing_wet

    source = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    rows = context.exchanges(source)
    output = _self_production_amount(source, rows, method="PA castor extraction")
    production = next(row for row in rows if row["type"] == "production")
    properties = cast(dict[str, dict[str, object]], production["properties"])
    native_allocation = float(cast(float, properties["allocation factor"]["amount"]))
    soy = next(
        row
        for row in rows
        if row["type"] == "technosphere" and context.input(row)["reference product"] == "soybean"
    )
    soy_properties = cast(dict[str, dict[str, object]], soy["properties"])
    source_dry = (
        float(soy["amount"])
        / output
        * float(cast(float, soy_properties["dry mass"]["amount"]))
        / native_allocation
    )
    physical_factor = seed_dry / (native_allocation * source_dry)
    oil_water = next(
        row
        for row in rows
        if row["type"] == "biosphere" and context.input(row)["name"] == "Oils, non-fossil"
    )
    # The native final oil-water indicator defines an oil-equivalent withdrawal
    # separately from collected wastewater.
    oil_loss = float(oil_water["amount"]) / output * physical_factor
    seed_cake_dry = seed_dry - 1 - oil_loss
    seed_cake_water = seed_processing_wet - seed_dry - _CRUDE_WATER
    total_dry_products = 1 + seed_cake_dry
    oil_allocation = 1 / total_dry_products
    # NREL's 1940 kcal/kg evaporated water is fuel input; .004184 MJ/kcal. Its 15.5 kWh/t beans and
    # 5907/94697 evaporation ratio give a separate motor meter.
    fuel = 1940 * 0.004184 * evaporated_seed_water
    electricity = (0.0155 / (5907 / 94697)) * evaporated_seed_water
    furnace = context.source.activity(selectors["drying_heat"], year=2025)
    furnace_rows = context.exchanges(furnace)
    furnace_output = _self_production_amount(furnace, furnace_rows, method="PA seed dryer")
    gas = next(
        row
        for row in furnace_rows
        if row["type"] == "technosphere"
        and context.input(row)["reference product"] == "natural gas, high pressure"
    )
    gas_properties = cast(dict[str, dict[str, object]], gas["properties"])
    fuel_per_mj = (
        float(gas["amount"])
        / furnace_output
        * float(cast(float, gas_properties["heating value, net"]["amount"]))
    )
    return _CastorReference(
        physical_kg={
            "crude_oil_dry": 1.0,
            "crude_oil_water": _CRUDE_WATER,
            "crude_oil_wet": 1 + _CRUDE_WATER,
            "seed_dry": seed_dry,
            "seed_wet": seed_wet,
            "seed_processing_wet": seed_processing_wet,
            "seed_water_evaporated": evaporated_seed_water,
            "seed_cake_dry": seed_cake_dry,
            "seed_cake_water": seed_cake_water,
            "oil_final_water": oil_loss,
        },
        allocation={
            "native_extraction": native_allocation,
            "native_unallocated_dry_seed_kg": source_dry,
            "physical_source_rows": physical_factor,
            "crude_oil": oil_allocation,
            "seed_cake": seed_cake_dry / total_dry_products,
            "allocated_source_rows": physical_factor * oil_allocation,
        },
        dryer_fuel_mj=fuel,
        dryer_electricity_kwh=electricity,
        dryer_heat_mj=fuel / fuel_per_mj,
    )


def _hco_reference() -> _HcoReference:
    """Combine nominal saturation with the oil-hardening consumable reference.

    PBL/TNO supplies .002 kg dry catalyst, .0015 kg bleaching earth and .0005 kg
    citric acid per kg feed. The catalyst dose uses the selected 13%-Ni powder.
    US4584140A Example 11 supplies the 40/60 oil/non-fat cake ratio for collected
    matter. Kumar supplies the separate 1.1/.93 hydrogen allowance.
    """
    catalyst_per_oil = 0.002
    earth_per_oil = 0.0015
    acid_per_oil = 0.0005
    nonfat_per_oil = catalyst_per_oil + earth_per_oil + acid_per_oil
    cake_oil_per_oil = 0.40 / 0.60 * nonfat_per_oil
    hydrogen_per_oil = 6 * _ATOMS["H"] / _CASTOR_MASS
    # Reference output is filtered dry HCO: added reaction H2 increases oil
    # mass, while oil retained in the whole cake reduces saleable dry product.
    oil = 1 / (1 + hydrogen_per_oil - cake_oil_per_oil)
    catalyst = catalyst_per_oil * oil
    cake = (nonfat_per_oil + cake_oil_per_oil) * oil
    net_hydrogen = hydrogen_per_oil * oil
    supplied_hydrogen = net_hydrogen * 1.1 / 0.93
    return _HcoReference(
        castor_oil_kg=oil,
        castor_water_kg=oil * _PRODUCT_WATER,
        hydrogen_net_kg=net_hydrogen,
        hydrogen_supply_kg=supplied_hydrogen,
        hydrogen_vent_kg=supplied_hydrogen - net_hydrogen,
        catalyst_kg=catalyst,
        filter_aid_kg=earth_per_oil * oil,
        citric_acid_kg=acid_per_oil * oil,
        cake_kg=cake,
        cake_components_kg={
            "oil": cake_oil_per_oil * oil,
            "unknown_mineral": earth_per_oil * oil,
            "citric_acid": acid_per_oil * oil,
            "Ni": _HCO_NICKEL_SHARE * catalyst,
            "SiO2": (1 - _HCO_NICKEL_SHARE) * catalyst,
        },
    )


def _pa_castor_hco_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Bind one dry kg crude castor oil or HCO to dated supplies and receivers.

    Original references are fixed; native operating rows keep their
    current-anchor amounts/uncertainty. Signed receiver demands own the whole
    target waste, while the operating service owns only its source support.
    No source crop uptake, generic final-water identity or coproduct credit
    is changed to reconcile the nominal molecular carbon reference.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    grade = str(construction["grade"])
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    key = database_name, provider_service_code(service_id)
    is_crude = grade == "castor_crude_oil"
    water = _CRUDE_WATER if is_crude else 0.0
    carbon = 57 * _ATOMS["C"] / (_CASTOR_MASS if is_crude else _HCO_MASS)
    rows: list[dict[str, object]] = [
        {
            "input": key,
            "type": "production",
            "amount": 1.0,
            "properties": {
                "wet mass": {"amount": 1 + water, "unit": "kg"},
                "dry mass": {"amount": 1.0, "unit": "kg"},
                "water in wet mass": {"amount": water, "unit": "kg"},
                "carbon allocation": {"amount": carbon, "unit": "kg"},
                "carbon content, non-fossil": {"amount": carbon, "unit": "dimensionless"},
                "carbon content, fossil": {"amount": 0.0, "unit": "dimensionless"},
            },
        }
    ]
    if is_crude:
        reference = _castor_reference(context, construction)
        physical, allocation = reference.physical_kg, reference.allocation
        share = allocation["crude_oil"]
        supplies = {
            "seed": share * physical["seed_wet"],
            "drying_heat": share * reference.dryer_heat_mj,
            "drying_electricity": share * reference.dryer_electricity_kwh,
        }
        dependencies: dict[str, tuple[float, str]] = {}
        source, trace = context.activity(
            cast(dict[str, object], construction["source"]),
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id="allocated_castor_extraction",
            semantic_class="service_normalization",
        )
        native = context.exchanges(source)
        output = _self_production_amount(source, native, method="dated PA castor extraction")
        retained, removed = [], []
        for exchange in native:
            kind = str(exchange["type"])
            if kind == "production":
                continue
            node = context.input(exchange)
            omit = (
                node["reference product"] == "soybean"
                if kind == "technosphere"
                else node["name"] == _RESOURCE_CORRECTION
            )
            if omit:
                removed.append(exchange)
            else:
                retained.append(exchange)
                rows.append(
                    _scaled_native_exchange(exchange, allocation["allocated_source_rows"] / output)
                )
        for kind in ("technosphere", "biosphere"):
            trace[f"retained_{kind}_json"] = context.exchange_json(
                r for r in retained if r["type"] == kind
            )
            trace[f"removed_{kind}_json"] = context.exchange_json(
                r for r in removed if r["type"] == kind
            )
        trace["project_terms_json"] = json.dumps(
            {
                **asdict(reference),
                "dated_source_production_kg": output,
                "dated_operating_row_multiplier": allocation["allocated_source_rows"] / output,
            },
            sort_keys=True,
        )
        # Native Water uses liquid-equivalent m3, including emissions to air:
        # divide kg evaporated water by the freshwater reference 1000 kg/m3.
        air_name, air_amount, air_unit = (
            "Water",
            share * physical["seed_water_evaporated"] / 1000,
            "cubic meter",
        )
        record = asdict(reference)
    else:
        hco = _hco_reference()
        # The PBL/TNO meter includes hardening and post-treatment. It is per kg feed oil.
        supplies = {
            "hydrogen": hco.hydrogen_supply_kg,
            "filter_aid": hco.filter_aid_kg,
            "citric_acid": hco.citric_acid_kg,
            "steam": 0.030 * hco.castor_oil_kg,
            "electricity": 0.010 * hco.castor_oil_kg,
        }
        # Condensed refined-oil moisture uses the same 1000 kg/m3 freshwater
        # reference; the native receiving service has negative m3 production.
        dependencies = {
            "castor": (hco.castor_oil_kg, "kilogram"),
            "nickel": (hco.catalyst_kg, "kilogram"),
            "condensate": (-hco.castor_water_kg / 1000, "cubic meter"),
            "treatment": (-hco.cake_kg, "kilogram"),
        }
        air_name, air_amount, air_unit = "Hydrogen", hco.hydrogen_vent_kg, "kilogram"
        record = asdict(hco)
    for label, amount in supplies.items():
        node, trace = context.activity(
            selectors[label],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=label,
            semantic_class="service_normalization",
        )
        rows.append(
            {
                "input": cast(tuple[str, str], node.key),
                "type": "technosphere",
                "amount": amount,
                "unit": str(node["unit"]),
            }
        )
        trace["project_terms_json"] = json.dumps(
            {"demand_per_dry_oil_kg": amount, "unit": node["unit"]}
        )
    for label, (amount, unit) in dependencies.items():
        target = cast(dict[str, object], construction[label])
        rows.append(
            {
                "input": (database_name, provider_service_code(str(target["service_id"]))),
                "type": "technosphere",
                "amount": amount,
                "unit": unit,
            }
        )
    flow = context.biosphere.flow({"name": air_name, "unit": air_unit, "categories": ["air"]})
    rows.append(
        {
            "input": cast(tuple[str, str], flow.key),
            "type": "biosphere",
            "amount": air_amount,
            "unit": air_unit,
        }
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "castor_hco_reference": record,
        "comment": (
            "Technical oil for PA fiber spin finish. The castor mill uses dry-product "
            "allocation with operating and capital proxies, retaining native crop emissions "
            "and uptake. Combined crude oil feeds native oil refining. HCO uses the PBL/TNO "
            "hardening and post-treatment quantities per feed oil, with a dry 13% Ni/silica "
            "catalyst. Refined-oil moisture condenses at hardening. HCO, catalyst, earth and "
            "citric-acid-equivalent material enter the connected cake treatment. The modeled "
            "chain includes one factory, with no recovered-oil or metal credit."
        ),
    }
