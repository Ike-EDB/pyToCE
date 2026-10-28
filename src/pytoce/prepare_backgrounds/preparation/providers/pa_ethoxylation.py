"""Supply PA ethoxylates through the complete native surfactant function.

The complete native ethoxylation operation supplies its dated utilities and
environmental function. Main reactant supplies are adapted for Oleth-10
and PEG-40 hydrogenated castor oil. Their fixed references use original ecoinvent
quantities; the two native mist/water substances remain explicit environmental
proxies.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .treatment_compilation import _replace_carbon_identity

# Admitted PA formula-mass convention, g/mol; these selected atomic weights
# differ from the rounded integer weights used by some native carbon accounts.
_ATOMS = {"C": 12.011, "H": 1.008, "O": 15.999}
# Nominal molecular references: C18H36O + 10 EO and C57H110O9 + 40 EO.
_PRECURSORS = {"oleth10": (18, 36, 1, 10), "peg40_hco": (57, 110, 9, 40)}
_MAIN_FEEDS = {"dodecanol", "ethylene oxide"}
_MIST = "Octaethylene glycol monododecyl ether"
_FOSSIL_CO2 = "Carbon dioxide, fossil"


@dataclass(frozen=True, slots=True)
class _EthoxylationReference:
    """Quantities per nominal kg ethoxylate.

    All carbon quantities are kg C. The two unassigned main-carbon quantities are reported as
    diagnostics.
    """

    main_gross: float
    precursor_kg: float
    ethylene_oxide_kg: float
    nominal_fossil_kg_c: float
    nominal_biogenic_kg_c: float
    main_aerosol_kg: float
    precursor_water_kg: float
    ethylene_oxide_air_kg: float
    original_co2_kg: float
    unassigned_main_fossil_kg_c: float
    unassigned_main_biogenic_kg_c: float


def _ethoxylation_reference(
    context: _RecipeContext, construction: dict[str, object]
) -> _EthoxylationReference:
    """Derive the fixed reference from the original source inventory.

    The original main-reactant/output mass ratio scales stoichiometric precursor and
    ethylene oxide demands. The native operation retains its workup and environmental
    functions. Stoichiometric target molecules define the material reference.
    """
    source = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    native = context.exchanges(source)
    output = _self_production_amount(source, native, method="PA ethoxylation")
    gross = (
        fsum(
            float(ex["amount"])
            for ex in native
            if ex["type"] == "technosphere"
            and context.input(ex)["reference product"] in _MAIN_FEEDS
        )
        / output
    )
    releases = {
        str(context.input(ex)["name"]): float(ex["amount"]) / output
        for ex in native
        if ex["type"] == "biosphere"
    }
    carbon, hydrogen, oxygen = (_ATOMS[e] for e in ("C", "H", "O"))
    c, h, o, eo_count = _PRECURSORS[str(construction["grade"])]
    precursor_mass = c * carbon + h * hydrogen + o * oxygen
    eo_mass = 2 * carbon + 4 * hydrogen + oxygen
    product_mass = precursor_mass + eo_count * eo_mass
    biogenic = c * carbon / product_mass
    fossil = 2 * eo_count * carbon / product_mass
    co2_carbon = carbon / (carbon + 2 * oxygen)
    co2 = releases[_FOSSIL_CO2]
    mist = releases[_MIST]
    precursor_water = releases["Dodecanol"]
    eo_air = releases["Ethylene oxide"]
    return _EthoxylationReference(
        main_gross=gross,
        precursor_kg=gross * precursor_mass / product_mass,
        ethylene_oxide_kg=gross * eo_count * eo_mass / product_mass,
        nominal_fossil_kg_c=fossil,
        nominal_biogenic_kg_c=biogenic,
        main_aerosol_kg=mist,
        precursor_water_kg=precursor_water,
        ethylene_oxide_air_kg=eo_air,
        original_co2_kg=co2,
        unassigned_main_fossil_kg_c=(gross - 1 - mist) * fossil
        - eo_air * 2 * carbon / eo_mass
        - co2 * fossil / (biogenic + fossil) * co2_carbon,
        unassigned_main_biogenic_kg_c=(gross - 1 - mist) * biogenic
        - precursor_water * c * carbon / precursor_mass
        - co2 * biogenic / (biogenic + fossil) * co2_carbon,
    )


def _pa_ethoxylation_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Compile one nominal kg Oleth-10 or PEG-40 HCO.

    Current-anchor utilities, wastewater and final environmental proxies are
    preserved per native output. Current CO2 retains its source total and is attributed using the
    fixed main-product carbon moieties. Native workup,
    losses and final emissions remain within the
    supplied surfactant function.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    reference = _ethoxylation_reference(context, construction)
    source, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="integrated_ethoxylation_environmental_function",
        semantic_class="service_normalization",
    )
    native = context.exchanges(source)
    output = _self_production_amount(source, native, method="PA ethoxylation")
    eo_supplies = [
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == "ethylene oxide"
    ]
    eo_multiplier = reference.ethylene_oxide_kg / fsum(float(ex["amount"]) for ex in eo_supplies)
    key = (database_name, provider_service_code(service_id))
    fossil_c = reference.nominal_fossil_kg_c
    biogenic_c = reference.nominal_biogenic_kg_c
    properties = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": fossil_c + biogenic_c, "unit": "kg"},
        "carbon content": {
            "amount": fossil_c + biogenic_c,
            "unit": "dimensionless",
        },
        "carbon content, fossil": {"amount": fossil_c, "unit": "dimensionless"},
        "carbon content, non-fossil": {
            "amount": biogenic_c,
            "unit": "dimensionless",
        },
    }
    generated: list[dict[str, object]] = [
        {"input": key, "amount": 1.0, "type": "production", "properties": properties}
    ]
    retained = []
    removed = []
    dated_co2 = (
        float(
            next(
                ex
                for ex in native
                if ex["type"] == "biosphere" and context.input(ex)["name"] == _FOSSIL_CO2
            )["amount"]
        )
        / output
    )
    for exchange in native:
        kind = str(exchange["type"])
        if kind == "production":
            continue
        node = context.input(exchange)
        if kind == "technosphere" and node["reference product"] in _MAIN_FEEDS:
            removed.append(exchange)
            continue
        retained.append(exchange)
        if kind == "biosphere" and node["name"] == _FOSSIL_CO2:
            fraction = reference.nominal_biogenic_kg_c / (
                reference.nominal_biogenic_kg_c + reference.nominal_fossil_kg_c
            )
            fossil = _scaled_native_exchange(exchange, (1 - fraction) / output)
            biogenic = _scaled_native_exchange(exchange, fraction / output)
            counterpart = context.biosphere.flow(
                {
                    "name": "Carbon dioxide, non-fossil",
                    "unit": str(node["unit"]),
                    "categories": list(node["categories"]),
                }
            )
            _replace_carbon_identity(biogenic, counterpart)
            generated.extend((fossil, biogenic))
        else:
            generated.append(_scaled_native_exchange(exchange, 1 / output))
    generated.extend(_scaled_native_exchange(ex, eo_multiplier) for ex in eo_supplies)
    if construction["grade"] == "peg40_hco":
        generated.append(
            {
                "input": (
                    database_name,
                    provider_service_code(
                        str(cast(dict[str, object], construction["hco"])["service_id"])
                    ),
                ),
                "type": "technosphere",
                "amount": reference.precursor_kg,
            }
        )
    else:
        alcohol, _ = context.activity(
            cast(dict[str, object], construction["oleyl_alcohol_source"]),
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id="oleyl_alcohol_manufacturing_function",
            semantic_class="explicit_cross_material_analogue",
        )
        generated.append(
            {"input": alcohol.key, "type": "technosphere", "amount": reference.precursor_kg}
        )
    trace["retained_technosphere_json"] = context.exchange_json(
        ex for ex in retained if ex["type"] == "technosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(removed)
    trace["retained_biosphere_json"] = context.exchange_json(
        ex for ex in retained if ex["type"] == "biosphere"
    )
    physical = {
        "schema": "pa_ethoxylation_reference_schema_v1",
        "grade": str(construction["grade"]),
        **asdict(reference),
        "dated_native_co2_kg": dated_co2,
        "main_carbon_remainder_is_additional_waste": False,
        "source_mist_and_precursor_water_are_environmental_proxies": True,
    }
    trace["project_terms_json"] = json.dumps(physical, sort_keys=True, separators=(",", ":"))
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "pa_ethoxylation_reference_json": trace["project_terms_json"],
        "exchanges": generated,
    }
