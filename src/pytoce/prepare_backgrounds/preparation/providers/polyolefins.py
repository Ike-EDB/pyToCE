"""Separate native polyolefin operations from allocated virgin and recovered monomers."""

import json
from typing import Any, cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext
from .treatment_compilation import _replace_carbon_identity


def _native_operation(
    context: _RecipeContext, service: dict[str, object]
) -> tuple[Any, tuple[Any, ...], dict[str, object]]:
    """Return the native activity, its complete exchanges and source record."""
    construction = cast(dict[str, object], service["construction"])
    base = cast(dict[str, object], construction["base"])
    _service_trace(context, service)
    activity, row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=str(service["provider_service_id"]),
        component_id="native_operation",
        semantic_class="operation_only_extraction",
    )
    exchanges = context.exchanges(activity)
    _self_production_amount(activity, exchanges, method=str(service["name"]))
    return activity, exchanges, row


def _dataset(
    service: dict[str, object],
    database_name: str,
    production: Any,
    exchanges: list[dict[str, object]],
) -> dict[str, object]:
    """Preserve the native production basis and exchange metadata in one component."""
    service_id = str(service["provider_service_id"])
    output = _copied_native_exchange(production)
    output["input"] = (database_name, provider_service_code(service_id))
    output["name"] = str(service["reference_product"])
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": [output, *exchanges],
    }


def _polymer_components(
    context: _RecipeContext,
    operation: dict[str, object],
    supplies: list[dict[str, object]],
    database_name: str,
) -> list[dict[str, object]]:
    """Separate the allocated polymer operation from ethylene and propylene supplies.

    Each service uses kg of destination polymer as its functional unit. Preserve
    the native production q and every separate monomer exchange A*q, including
    uncertainty. Foreground activation applies the recovered fraction;
    no recipe coefficient B enters this allocated inventory.
    """
    activity, exchanges, row = _native_operation(context, operation)
    production = next(exchange for exchange in exchanges if exchange["type"] == "production")
    monomers: dict[str, list[Any]] = {"ethylene": [], "propylene": []}
    retained: list[Any] = []
    for exchange in exchanges:
        kind = str(exchange["type"])
        if kind == "production":
            continue
        node = context.input(exchange)
        product = str(node.get("reference product", ""))
        if kind == "technosphere" and product in monomers:
            monomers[product].append(exchange)
        else:
            retained.append(exchange)
    row["retained_technosphere_json"] = context.exchange_json(
        exchange for exchange in retained if exchange["type"] == "technosphere"
    )
    row["removed_technosphere_json"] = context.exchange_json(
        exchange for inputs in monomers.values() for exchange in inputs
    )
    datasets = [
        _dataset(
            operation, database_name, production, [_copied_native_exchange(e) for e in retained]
        )
    ]
    for service in supplies:
        _service_trace(context, service)
        construction = cast(dict[str, object], service["construction"])
        inputs = monomers[str(construction["monomer"])]
        copied = [_copied_native_exchange(exchange) for exchange in inputs]
        supplier = cast(dict[str, object] | None, construction.get("supplier"))
        if supplier is not None:
            for exchange in copied:
                exchange["input"] = (
                    database_name,
                    provider_service_code(str(supplier["service_id"])),
                )
        context.semantic_rows.append(
            {
                "background_scenario": context.background_scenario,
                "anchor_year": context.year,
                "ledger_entry_type": "allocated_polymer_monomer_supply",
                "provider_service_id": str(service["provider_service_id"]),
                "native_activity_database": str(activity.key[0]),
                "native_activity_code": str(activity.key[1]),
                "native_reference_production_amount": float(production["amount"]),
                "native_monomer_inputs_json": context.exchange_json(inputs),
                "generated_exchanges_json": json.dumps(copied, sort_keys=True),
                "allocation_metadata_preserved": True,
                "uncertainty_metadata_preserved": True,
            }
        )
        datasets.append(_dataset(service, database_name, production, copied))
    return datasets


def _cracker_operation(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Retain the allocated South Korean cracker operation; supply naphtha separately.

    Source comments assign direct CO2, CO and CH4 to cracker feed and internal
    byproduct fuel. Only these species change identity for non-fossil feed, in
    their actual native compartments. Preserve other flows, auxiliary supplies,
    native allocation and uncertainty; invent no emission for unassigned carbon.
    """
    activity, exchanges, row = _native_operation(context, service)
    construction = cast(dict[str, object], service["construction"])
    non_fossil = construction["ultimate_carbon_class"] == "non_fossil"
    species = {
        "Carbon dioxide, fossil": "Carbon dioxide, non-fossil",
        "Carbon monoxide, fossil": "Carbon monoxide, non-fossil",
        "Methane, fossil": "Methane, non-fossil",
    }
    feeds: list[Any] = []
    retained: list[Any] = []
    generated: list[dict[str, object]] = []
    production = next(exchange for exchange in exchanges if exchange["type"] == "production")
    for exchange in exchanges:
        kind = str(exchange["type"])
        if kind == "production":
            continue
        node = context.input(exchange)
        if kind == "technosphere" and node.get("reference product") == "naphtha":
            feeds.append(exchange)
            continue
        retained.append(exchange)
        copied = _copied_native_exchange(exchange)
        name = str(node.get("name", ""))
        if kind == "biosphere" and name in species:
            categories = tuple(node.get("categories", ()))
            counterpart = context.biosphere.flow(
                {"name": species[name], "unit": "kilogram", "categories": categories}
            )
            if non_fossil:
                _replace_carbon_identity(copied, counterpart)
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "ledger_entry_type": "cracker_direct_carbon_mapping",
                    "provider_service_id": str(service["provider_service_id"]),
                    "source_flow_key": tuple(node.key),
                    "generated_flow_key": copied["input"],
                    "source_amount": float(exchange["amount"]),
                    "generated_amount": float(exchange["amount"]),
                    "mapping_status": (
                        "exact_non_fossil_counterpart" if non_fossil else "native_fossil_identity"
                    ),
                }
            )
        generated.append(copied)
    row["removed_technosphere_json"] = context.exchange_json(feeds)
    row["retained_technosphere_json"] = context.exchange_json(
        exchange for exchange in retained if exchange["type"] == "technosphere"
    )
    return _dataset(service, database_name, production, generated)


def polyolefin_datasets(
    context: _RecipeContext, services: dict[str, dict[str, object]], database_name: str
) -> dict[tuple[str, str], dict[str, object]]:
    """Compile each dated native polymer once and its linked monomer components."""
    operations = []
    supplies: dict[str, list[dict[str, object]]] = {}
    datasets = []
    for service in services.values():
        construction = cast(dict[str, object], service["construction"])
        kind = str(construction["kind"])
        if kind == "polyolefin_polymerization_operation":
            operations.append(service)
        elif kind == "polyolefin_monomer_supply":
            operation = cast(dict[str, object], construction["polymerization"])
            supplies.setdefault(str(operation["service_id"]), []).append(service)
        elif kind == "recovered_monomer_cracker_operation":
            datasets.append(_cracker_operation(context, service, database_name))
    for operation in operations:
        datasets.extend(
            _polymer_components(
                context, operation, supplies[str(operation["provider_service_id"])], database_name
            )
        )
    return {
        (database_name, provider_service_code(str(dataset["provider_service_id"]))): dataset
        for dataset in datasets
    }
