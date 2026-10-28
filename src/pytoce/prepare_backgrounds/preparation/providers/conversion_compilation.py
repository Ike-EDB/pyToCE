"""Native input and saleable conversion bases."""

import json
import math
from typing import Any, cast

from .catalogue import provider_service_code
from .native_recipe import _service_trace
from .native_resolution import _exact_exchange_matches, _input_semantics, _matches
from .recipe_context import _copied_native_exchange, _RecipeContext
from .treatment_compilation import _direct_carbon_owners_json, _waste_carbon_owners


def _input_normalized_operation_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Compile a native operation on its declared material or fuel input basis.

    Project production equals the central native normalization-input amount
    ``q`` in the service unit. Retained exchanges keep their original amounts
    and distribution fields; Brightway divides them by ``q``. The derived
    denominator is fixed.
    Declared omissions belong to another identified operation.

    Args:
        context: Native resolvers and source records for one member and year.
        service: Admitted material-processing or fuel-normalized definition.
        database_name: Project database receiving the operation.

    Returns:
        A provider dataset with native retained exchanges and production ``q``.

    Raises:
        ValueError: If the native normalization input is absent, nonpositive or nonfinite,
            or a declared omission is missing from the selected operation.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    base = cast(dict[str, object], construction["base"])
    activity, resolution_row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="native_input_normalized_operation",
        semantic_class="service_normalization",
    )
    denominator_selector = cast(
        dict[str, object],
        construction["denominator_selector"],
    )
    omitted_selectors = cast(
        list[dict[str, object]],
        construction.get("omit_technosphere", []),
    )
    native_exchanges = tuple(
        exchange
        for exchange in context.exchanges(activity)
        if str(exchange["type"]) != "production"
    )
    denominator = sum(
        float(exchange["amount"])
        for exchange in native_exchanges
        if str(exchange["type"]) == "technosphere"
        and _matches(context.input(exchange), denominator_selector)
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "The declared normalization input is absent or nonpositive, or its sum is nonfinite in "
            f"{activity.get('name')} ({activity.get('reference product')}): {denominator!r} "
            f"{denominator_selector['unit']}. Use a source operation with a finite, "
            "positive total for the required normalization input."
        )

    omitted: set[int] = set()
    retained_technosphere: list[Any] = []
    removed_technosphere: list[Any] = []
    retained: list[dict[str, object]] = []
    for exchange in native_exchanges:
        exchange_type = str(exchange["type"])
        input_node = context.input(exchange)
        omission = next(
            (
                index
                for index, selector in enumerate(omitted_selectors)
                if exchange_type == "technosphere" and _matches(input_node, selector)
            ),
            None,
        )
        if omission is None:
            if exchange_type == "technosphere":
                retained_technosphere.append(exchange)
            retained.append(_copied_native_exchange(exchange))
            continue

        semantics = _input_semantics(input_node)
        removed_technosphere.append(exchange)
        omitted.add(omission)
        context.semantic_rows.append(
            {
                "background_scenario": context.background_scenario,
                "anchor_year": context.year,
                "provider_service_id": service_id,
                "semantic_class": "operation_only_extraction",
                "ledger_entry_type": "native_exchange",
                "native_activity_database": str(activity.key[0]),
                "native_activity_code": str(activity.key[1]),
                "removed_input_name": semantics[0],
                "removed_input_reference_product": semantics[1],
                "removed_input_unit": semantics[2],
                "removed_input_location": semantics[3],
                "removed_native_amount": float(exchange["amount"]),
                "removed_amount_per_normalization_input": (float(exchange["amount"]) / denominator),
            }
        )

    missing = [selector for index, selector in enumerate(omitted_selectors) if index not in omitted]
    if missing:
        raise ValueError(
            "A declared omitted technosphere exchange is absent from "
            f"{activity.get('name')} ({activity.get('reference product')}): "
            f"{missing}"
        )

    resolution_row["retained_technosphere_json"] = context.exchange_json(retained_technosphere)
    resolution_row["removed_technosphere_json"] = context.exchange_json(removed_technosphere)
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "provider_service_id": service_id,
            "semantic_class": "service_normalization",
            "ledger_entry_type": "service_normalization",
            "native_activity_database": str(activity.key[0]),
            "native_activity_code": str(activity.key[1]),
            "denominator_input_name": str(denominator_selector["name"]),
            "denominator_input_reference_product": str(denominator_selector["reference_product"]),
            "denominator_input_unit": str(denominator_selector["unit"]),
            "native_normalization_input_amount": denominator,
            "project_functional_unit": (f"one {denominator_selector['unit']} normalized input"),
            "scaling_algebra": "project production = native input amount; retain native exchanges",
            "normalization_basis": "central native input amount; fixed derived denominator",
        }
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": [
            {
                "input": (database_name, provider_service_code(service_id)),
                "amount": denominator,
                "type": "production",
            },
            *retained,
        ],
    }


def _conversion_operation_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Normalize a native conversion operation to one kg finished plastic.

    The source recipe gives finished kg per native service kg, ``y``. Project
    production is ``p*y`` for native production ``p``; Brightway therefore
    supplies ``F/(p*y)`` operations for finished demand ``F``. Native waste treatment is a separate
    inventory exchange.
    Every nonproduction exchange keeps its native amount and uncertainty.
    For biogenic and DAC plastic, only the declared waste input selects the
    corresponding non-fossil treatment chain. Utilities and other emissions
    retain their original identities.

    A changed production basis uses the fixed central denominator because no
    joint distribution for the derived quantity is established. A unit basis
    preserves the original production metadata. Principal resin is supplied
    separately by the MFA model.

    Args:
        context: Dated native resolvers and admitted conversion requirements.
        service: The selected generic processing operation and source binding.
        database_name: Project database receiving the finished-output service.

    Returns:
        A dataset with source-derived production and all native burdens.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    requirement = next(
        item
        for item in context.material_recipe.conversion
        if item.operation == construction["operation"]
    )
    _service_trace(context, service)
    base = cast(dict[str, object], construction["base"])
    activity, resolution_row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="native_conversion_operation",
        semantic_class="operation_only_extraction",
    )
    native_exchanges = context.exchanges(activity)
    native_production = next(
        exchange for exchange in native_exchanges if str(exchange["type"]) == "production"
    )
    gross_production = float(native_production["amount"])
    finished_fraction = requirement.finished_kg_per_native_service_kg
    finished_production = gross_production * finished_fraction
    production = (
        _copied_native_exchange(native_production)
        if finished_fraction == 1.0
        else {"amount": finished_production, "type": "production"}
    )
    production["input"] = (database_name, provider_service_code(service_id))
    replacements = cast(list[dict[str, object]], construction["waste_replacements"])
    non_fossil = str(construction["ultimate_carbon_class"]) == "non_fossil"
    retained = []
    owners = []
    replacement_rows = []
    for exchange in native_exchanges:
        if str(exchange["type"]) == "production":
            continue
        copied = _copied_native_exchange(exchange)
        for replacement in replacements:
            if _exact_exchange_matches(
                exchange, context.input(exchange), cast(dict[str, object], replacement["selector"])
            ):
                waste_service_id = str(replacement["service_id"])
                owners.extend(
                    _waste_carbon_owners(
                        context,
                        context.input(exchange),
                        waste_service_id,
                        scale=float(exchange["amount"]) / finished_production,
                        non_fossil=non_fossil,
                        database_name=database_name,
                    )
                )
                if non_fossil:
                    copied["input"] = (database_name, provider_service_code(waste_service_id))
                project_input = cast(tuple[str, str], copied["input"])
                replacement_rows.append(
                    {
                        "native_input_database": str(exchange["input"][0]),
                        "native_input_code": str(exchange["input"][1]),
                        "project_input_database": project_input[0],
                        "project_input_code": project_input[1],
                        "amount": float(exchange["amount"]),
                        "exchange_type": "technosphere",
                        "metadata_preserved": True,
                    }
                )
        retained.append(copied)
    resolution_row["project_terms_json"] = json.dumps(
        replacement_rows, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "provider_service_id": service_id,
            "semantic_class": "service_normalization",
            "ledger_entry_type": "conversion_service_balance",
            "native_activity_database": str(activity.key[0]),
            "native_activity_code": str(activity.key[1]),
            "conversion_operation": requirement.operation,
            "native_gross_production": gross_production,
            "finished_kg_per_native_service_kg": finished_fraction,
            "finished_production": finished_production,
            "project_functional_unit": "kilogram finished converted plastic",
            "scaling_algebra": "finished production = native production * source finished fraction",
            "normalization_basis": (
                "unchanged native output, including production uncertainty"
                if finished_fraction == 1.0
                else "source finished fraction; fixed derived denominator"
            ),
        }
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "direct_treatment_carbon_owners_json": _direct_carbon_owners_json(tuple(owners)),
        "exchanges": [production, *retained],
    }
