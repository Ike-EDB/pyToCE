"""Supply operating functions for PA fiber-finish ingredients.

DOA and fatty-acid-amide producers supply operations per kg isolated output.
PA ingredient calculations supply reactants and composition-dependent releases.
Nitrogen has a separate supply and vent function.
"""

import json
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .water_treatment import _water_reference

# The retained DOA 14-row and fatty-amide 14-row operating functions include
# factory, electricity, both heat families, external wastewater and water
# balances. All other donor rows describe the replaced chemical reaction.
_OPERATION_PRODUCTS = {
    "chemical factory, organics",
    "electricity, medium voltage",
    "heat, district or industrial, natural gas",
    # premise 2.4.6 heat.py relinks this native heat supply to the member's
    # industrial-heat product. This is the same duty at future anchors.
    "heat, district or industrial",
    "heat, from steam, in chemical industry",
    "wastewater, average",
}
_OPERATION_WATER_FLOWS = {
    "Water",
    "Water, cooling, unspecified natural origin",
    "Water, river",
    "Water, well, in ground",
}


def _pa_finish_operation_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Transfer dated operating support with its complete native wastewater supply.

    Preserve native support amounts and uncertainty per kg donor output.
    The aggregate site wastewater retains its regional suppliers and treatment
    inventories. Actual ingredient condensate, salts, organic residues, and site wastewater have
    separate receiving demands.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    source, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="ingredient_operating_function",
        semantic_class="operation_only_extraction",
    )
    native = context.exchanges(source)
    production = _self_production_amount(source, native, method="PA ingredient operating function")
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    retained = []
    removed = []
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        is_supply = exchange["type"] == "technosphere"
        keep = (
            str(node["reference product"]) in _OPERATION_PRODUCTS
            if is_supply
            else str(node["name"]) in _OPERATION_WATER_FLOWS
        )
        if not keep:
            removed.append(exchange)
            continue
        retained.append(exchange)
        rows.append(_scaled_native_exchange(exchange, 1.0 / production))
    trace["retained_technosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "technosphere"
    )
    trace["retained_biosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "biosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(
        row for row in removed if row["type"] == "technosphere"
    )
    trace["removed_biosphere_json"] = context.exchange_json(
        row for row in removed if row["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {
            "source_production_kg": production,
            "output_reference_kg": 1.0,
            "target_chemistry_owned_by_consuming_ingredient": True,
        },
        sort_keys=True,
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
    }


def _pa_condensate_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Collect one m3 nominal pure condensate through the four admitted regions.

    The DOA producer supplies the dated relative wastewater-service portfolio.
    Actual PA ingredient owners specify their separate condensate volumes.
    The complete hydraulic recipients replace source sewage composition; they
    retain their own signed production, infrastructure and regional treatment.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    source, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="condensate_collection_portfolio",
        semantic_class="source_backed_portfolio",
    )
    native = context.exchanges(source)
    supplies = [
        exchange
        for exchange in native
        if exchange["type"] == "technosphere"
        and context.input(exchange)["reference product"] == "wastewater, average"
    ]
    # Each waste demand is negative. Their negative total defines a positive scale to the negative
    # one-m3 collection reference.
    total = fsum(float(exchange["amount"]) for exchange in supplies)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": -1.0}]
    recipients = cast(dict[str, dict[str, object]], construction["water"])
    for exchange in supplies:
        location = str(context.input(exchange)["location"])
        child = str(recipients[location]["service_id"])
        record = _scaled_native_exchange(exchange, -1.0 / total)
        rows.append(
            _water_reference(
                record,
                (database_name, provider_service_code(child)),
                "nominal pure-water treatment",
            )
        )
    trace["retained_technosphere_json"] = context.exchange_json(supplies)
    trace["removed_technosphere_json"] = context.exchange_json(
        exchange
        for exchange in native
        if exchange["type"] == "technosphere" and exchange not in supplies
    )
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        exchange for exchange in native if exchange["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {"native_signed_water_m3": total, "output_signed_water_m3": -1.0}, sort_keys=True
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "cubic meter",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
    }


def _pa_finish_nitrogen_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Supply and vent one kg nitrogen through the selected native portfolio.

    The NaHSO4 producer supplies its dated liquid-N2 purchase shares and
    matching N2 vent. Ingredient owners set the admitted purge dose.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    source, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="nitrogen_supply_and_vent",
        semantic_class="source_backed_portfolio",
    )
    native = context.exchanges(source)
    supplies = [
        exchange
        for exchange in native
        if exchange["type"] == "technosphere"
        and context.input(exchange)["reference product"] == "nitrogen, liquid"
    ]
    vents = [
        exchange
        for exchange in native
        if exchange["type"] == "biosphere" and context.input(exchange)["name"] == "Nitrogen"
    ]
    # Normalize the portfolio by its actual purchased N2 mass. Preserve its vent and all uncertainty
    # metadata.
    factor = 1.0 / fsum(float(exchange["amount"]) for exchange in supplies)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    rows.extend(_scaled_native_exchange(exchange, factor) for exchange in (*supplies, *vents))
    trace["retained_technosphere_json"] = context.exchange_json(supplies)
    trace["retained_biosphere_json"] = context.exchange_json(vents)
    trace["removed_technosphere_json"] = context.exchange_json(
        exchange
        for exchange in native
        if exchange["type"] == "technosphere" and exchange not in supplies
    )
    trace["removed_biosphere_json"] = context.exchange_json(
        exchange for exchange in native if exchange["type"] == "biosphere" and exchange not in vents
    )
    trace["project_terms_json"] = json.dumps(
        {"native_nitrogen_supply_kg": 1.0 / factor, "output_reference_nitrogen_kg": 1.0},
        sort_keys=True,
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
    }
