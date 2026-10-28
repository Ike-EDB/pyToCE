"""Adapt native dioctyl adipate production to nominal PA tridecyl stearate.

Original adipic-acid utilization sets the main-feed reference. Native operating
and environmental quantities, including ethylene releases, provide the proxy.
"""

import json
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .pa_ester_ingredients import _ester_distribution
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .treatment_compilation import _replace_carbon_identity

# Admitted CIAAW formula convention, g/mol. Native DOA is C22H42O4 from
# adipic acid C6H10O4 + two C8H18O, losing two waters. Target C31H62O2 is
# stearic acid C18H36O2 + C13H28O, losing one water. Raw feed sums therefore
# cannot transfer the native stoichiometric utilization to this target.
_C, _H, _O = 12.011, 1.008, 15.999
_DOA_MASS = 22 * _C + 42 * _H + 4 * _O
_ADIPIC_MASS = 6 * _C + 10 * _H + 4 * _O
_TARGET_MASS = 31 * _C + 62 * _H + 2 * _O
_MAIN_MASSES = {
    "stearic_acid": 18 * _C + 36 * _H + 2 * _O,
    "fatty_alcohol": 13 * _C + 28 * _H + _O,
}
_NATIVE_MAIN_FEEDS = {"adipic acid", "ethylene"}


def _pa_tridecyl_stearate_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Replace native main feeds and retain dated operating quantities.

    The original 2025 acid input per output sets the molar utilization.
    Each acid record supplies a scaled uncertainty proxy for both reactants.
    Native process CO2 is attributed to the nominal 18/31 biogenic and 13/31
    fossil carbon moieties, preserving its total and upstream suppliers.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    original = context.source.activity(selector, year=2025)
    original_native = context.exchanges(original)
    original_output = _self_production_amount(
        original, original_native, method="PA tridecyl stearate original DOA"
    )
    acid_rows = [
        ex
        for ex in original_native
        if ex["type"] == "technosphere" and context.input(ex)["reference product"] == "adipic acid"
    ]
    original_acid = fsum(float(ex["amount"]) for ex in acid_rows) / original_output
    utilization = original_acid * _DOA_MASS / _ADIPIC_MASS
    main_inputs = {field: utilization * mass / _TARGET_MASS for field, mass in _MAIN_MASSES.items()}
    source, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="integrated_tridecyl_stearate_environmental_function",
        semantic_class="service_normalization",
    )
    native = context.exchanges(source)
    output = _self_production_amount(source, native, method="PA tridecyl stearate DOA")
    key = database_name, provider_service_code(service_id)
    fossil_c, biogenic_c = 13 * _C / _TARGET_MASS, 18 * _C / _TARGET_MASS
    properties = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": fossil_c + biogenic_c, "unit": "kg"},
        "carbon content": {"amount": fossil_c + biogenic_c, "unit": "dimensionless"},
        "carbon content, fossil": {"amount": fossil_c, "unit": "dimensionless"},
        "carbon content, non-fossil": {"amount": biogenic_c, "unit": "dimensionless"},
    }
    generated: list[dict[str, object]] = [
        {"input": key, "type": "production", "amount": 1.0, "properties": properties}
    ]
    retained, removed = [], []
    for exchange in native:
        kind = str(exchange["type"])
        if kind == "production":
            continue
        node = context.input(exchange)
        if kind == "technosphere" and node["reference product"] in _NATIVE_MAIN_FEEDS:
            removed.append(exchange)
            continue
        retained.append(exchange)
        if kind == "biosphere" and node["name"] == "Carbon dioxide, fossil":
            # Attribute native process-loss carbon by target carbon origin, preserving
            # each record and total while leaving utility carbon unchanged.
            fossil = _scaled_native_exchange(exchange, 13 / (31 * output))
            biogenic = _scaled_native_exchange(exchange, 18 / (31 * output))
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
    sources = cast(dict[str, dict[str, object]], construction["sources"])
    for field, mass in _MAIN_MASSES.items():
        supplier, supply_trace = context.activity(
            sources[field],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=field,
            semantic_class="explicit_cross_material_analogue",
        )
        supply_trace["project_terms_json"] = json.dumps(
            {"supplied_kg": main_inputs[field]}, sort_keys=True
        )
        # Preserve each acid record's multiplicative uncertainty. Brightway normalizes the supplier
        # production reference for these kg demands.
        factor = _DOA_MASS * mass / (_ADIPIC_MASS * _TARGET_MASS * original_output)
        for exchange in acid_rows:
            scaled = _scaled_native_exchange(exchange, factor)
            # Native acid properties, classifications, CAS/flow/activity IDs and comments describe
            # adipic acid. Transfer only its admitted uncertainty fields.
            target = {
                field: scaled[field]
                for field in (
                    "amount",
                    "uncertainty type",
                    "loc",
                    "scale",
                    "scale without pedigree",
                    "minimum",
                    "maximum",
                    "shape",
                    "negative",
                    "pedigree",
                )
                if field in scaled
            }
            target.update(
                input=supplier.key,
                name=str(supplier["name"]),
                unit=str(supplier["unit"]),
                type="technosphere",
            )
            generated.append(target)
    generated.extend(
        _ester_distribution(
            context,
            cast(dict[str, object], construction["freight_source"]),
            service_id=service_id,
            amount_kg=main_inputs["fatty_alcohol"],
        )
    )
    trace["retained_technosphere_json"] = context.exchange_json(
        ex for ex in retained if ex["type"] == "technosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(removed)
    trace["retained_biosphere_json"] = context.exchange_json(
        ex for ex in retained if ex["type"] == "biosphere"
    )
    physical = {
        "schema": "pa_tridecyl_stearate_reference_schema_v1",
        "original_adipic_acid_kg_per_kg_doa": original_acid,
        "original_main_molar_utilization": utilization,
        "native_inputs_kg": main_inputs,
        "product_fossil_kg_c": fossil_c,
        "product_biogenic_kg_c": biogenic_c,
        "native_environmental_function_is_target_proxy": True,
        "main_feed_uncertainty_is_original_acid_proxy": True,
        "technical_manufacturing_function_is_not_nominal_molecular_assay": True,
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
        "pa_tridecyl_stearate_reference_json": trace["project_terms_json"],
        "exchanges": generated,
    }
