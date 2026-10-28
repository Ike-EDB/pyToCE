"""Chemical-recycling preparation and conversion-chain compilation."""

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from .catalogue import provider_service_code
from .native_recipe import _service_trace
from .native_resolution import _exact_exchange_matches, _exchange_semantic_key
from .recipe_context import _copied_native_exchange, _RecipeContext
from .treatment_compilation import (
    _matches_treatment_carbon_selector,
    _replace_carbon_identity,
    _waste_carbon_owners,
)


def _chemical_native_resolution_row(
    context: _RecipeContext,
    *,
    service_id: str,
    source_role: str,
    activity: Any,
    retained: tuple[Any, ...],
    removed: tuple[Any, ...],
    semantic_class: str,
) -> dict[str, object]:
    """Record one native chemical operation and return its mutable provenance row."""
    row: dict[str, object] = {
        "background_scenario": context.background_scenario,
        "anchor_year": context.year,
        "ledger_entry_type": "native_component",
        "semantic_class": semantic_class,
        "provider_service_id": service_id,
        "component_id": "source_defined_native_operation",
        "source_class": source_role,
        "native_database_name": str(activity.key[0]),
        "activity_code": str(activity["code"]),
        "activity_name": str(activity.get("name", "")),
        "reference_product": str(activity.get("reference product", "")),
        "native_unit": str(activity.get("unit", "")),
        "location": str(activity.get("location", "")),
        "retained_technosphere_json": context.exchange_json(
            exchange for exchange in retained if str(exchange["type"]) == "technosphere"
        ),
        "removed_technosphere_json": context.exchange_json(
            exchange for exchange in removed if str(exchange["type"]) == "technosphere"
        ),
        "retained_biosphere_json": context.exchange_json(
            exchange for exchange in retained if str(exchange["type"]) == "biosphere"
        ),
        "removed_biosphere_json": context.exchange_json(
            exchange for exchange in removed if str(exchange["type"]) == "biosphere"
        ),
        "project_terms_json": "[]",
    }
    context.technology_rows.append(row)
    return row


@dataclass(frozen=True, slots=True)
class _ChemicalPreparationDetails:
    """Preparation amounts for one background on the tracked-plastic basis."""

    activity: Any
    retained: tuple[Any, ...]
    removed: tuple[Any, ...]
    native_reference_production_amount: float
    processed_plastic_output_amount: float
    plastic_reject_output_amount: float
    tracked_plastic_denominator: float
    processed_feed_per_kg_tracked_plastic: float
    plastic_reject_per_kg_tracked_plastic: float


def _chemical_output_matches(
    exchange: Any,
    input_node: Any,
    selector: dict[str, object],
) -> bool:
    """Match a preparation output by its declared classification."""

    amount = float(exchange["amount"])
    node_fields = {
        "name": "name",
        "reference_product": "reference product",
        "unit": "unit",
        "location": "location",
    }
    return (
        str(exchange["type"]) == str(selector["exchange_type"])
        and all(
            key not in selector or str(input_node.get(node_field, "")) == str(selector[key])
            for key, node_field in node_fields.items()
        )
        and ((amount < 0.0) if str(selector["sign"]) == "negative" else (amount > 0.0))
    )


def _chemical_preparation_details(
    context: _RecipeContext,
    construction: dict[str, object],
) -> _ChemicalPreparationDetails:
    """Resolve and normalize the chemical recycling preparation activity.

    The selected activity must have one self-production exchange, one processed
    plastic output, and one plastic reject output matching ``construction``.
    The tracked-plastic denominator sums the processed and reject output
    magnitudes. Coefficients express feed and reject per kg of tracked plastic;
    retained exchanges keep their native units for later compilation.

    Raises:
        ValueError: If the source activity lacks the required exchanges or has
            zero production magnitude or a nonpositive tracked-plastic basis.
    """
    activity = context.resolver(str(construction["source_role"])).activity(
        cast(dict[str, object], construction["selector"]), year=context.year
    )
    native = context.exchanges(activity)
    production = tuple(exchange for exchange in native if str(exchange["type"]) == "production")
    if len(production) != 1 or cast(tuple[str, str], production[0]["input"]) != cast(
        tuple[str, str], activity.key
    ):
        raise ValueError("Chemical-recycling preparation must have one self-production exchange.")
    processed_selector = cast(dict[str, object], construction["processed_output_selector"])
    processed = tuple(
        exchange
        for exchange in native
        if _chemical_output_matches(exchange, context.input(exchange), processed_selector)
    )
    if len(processed) != 1:
        raise ValueError(
            "Chemical-recycling preparation must have one source-defined processed output."
        )
    reject_selector = cast(dict[str, object], construction["plastic_reject_output_selector"])
    plastic_reject = tuple(
        exchange
        for exchange in native
        if _chemical_output_matches(exchange, context.input(exchange), reject_selector)
    )
    if len(plastic_reject) != 1:
        raise ValueError(
            "Chemical-recycling preparation must have one source-defined plastic reject."
        )
    removed = (production[0], processed[0])
    retained = tuple(exchange for exchange in native if exchange not in removed)
    native_reference_production_amount = abs(float(production[0]["amount"]))
    processed_amount = abs(float(processed[0]["amount"]))
    plastic_reject_amount = abs(float(plastic_reject[0]["amount"]))
    tracked_plastic_denominator = processed_amount + plastic_reject_amount
    if native_reference_production_amount <= 0.0 or tracked_plastic_denominator <= 0.0:
        raise ValueError("Chemical-recycling preparation requires a positive mass basis.")
    processed_coefficient = processed_amount / tracked_plastic_denominator
    plastic_reject_coefficient = plastic_reject_amount / tracked_plastic_denominator
    return _ChemicalPreparationDetails(
        activity=activity,
        retained=retained,
        removed=removed,
        native_reference_production_amount=native_reference_production_amount,
        processed_plastic_output_amount=processed_amount,
        plastic_reject_output_amount=plastic_reject_amount,
        tracked_plastic_denominator=tracked_plastic_denominator,
        processed_feed_per_kg_tracked_plastic=processed_coefficient,
        plastic_reject_per_kg_tracked_plastic=plastic_reject_coefficient,
    )


def _chemical_recycling_preparation_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Compile the chemical recycling preparation-stage provider dataset."""
    service_id = str(service["provider_service_id"])
    code = provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    details = _chemical_preparation_details(context, construction)
    resolution_row = _chemical_native_resolution_row(
        context,
        service_id=service_id,
        source_role=str(construction["source_role"]),
        activity=details.activity,
        retained=details.retained,
        removed=details.removed,
        semantic_class="explicit_cross_material_operation_analogue",
    )
    reject_selector = cast(dict[str, object], construction["plastic_reject_output_selector"])
    reject_exchange = next(
        exchange
        for exchange in details.retained
        if _chemical_output_matches(exchange, context.input(exchange), reject_selector)
    )
    waste_service_id = str(cast(dict[str, object], construction["reject_market"])["service_id"])
    non_fossil = str(construction["ultimate_carbon_class"]) == "non_fossil"
    reject_input = (
        (database_name, provider_service_code(waste_service_id))
        if non_fossil
        else cast(tuple[str, str], reject_exchange["input"])
    )
    direct_owners = _waste_carbon_owners(
        context,
        context.input(reject_exchange),
        waste_service_id,
        scale=float(reject_exchange["amount"]) / details.tracked_plastic_denominator,
        non_fossil=non_fossil,
        database_name=database_name,
    )
    resolution_row["project_terms_json"] = json.dumps(
        [
            {
                "native_input_database": str(reject_exchange["input"][0]),
                "native_input_code": str(reject_exchange["input"][1]),
                "project_input_database": reject_input[0],
                "project_input_code": reject_input[1],
                "amount": float(reject_exchange["amount"]),
                "exchange_type": "technosphere",
                "metadata_preserved": True,
            }
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    generated_exchanges = []
    for exchange in details.retained:
        generated = _copied_native_exchange(exchange)
        if exchange is reject_exchange:
            generated["input"] = reject_input
        generated_exchanges.append(generated)
    normalization = cast(
        dict[str, object],
        construction["tracked_plastic_denominator"],
    )
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "provider_service_id": service_id,
            "semantic_class": "explicit_cross_material_operation_analogue",
            "ledger_entry_type": "chemical_recycling_preparation_alignment",
            "native_reference_production_amount": (details.native_reference_production_amount),
            "processed_plastic_output_amount": details.processed_plastic_output_amount,
            "plastic_reject_output_amount": details.plastic_reject_output_amount,
            "tracked_plastic_denominator": details.tracked_plastic_denominator,
            "native_processed_feed_coefficient": (details.processed_feed_per_kg_tracked_plastic),
            "native_plastic_reject_coefficient": (details.plastic_reject_per_kg_tracked_plastic),
            "excluded_denominator_classes_json": json.dumps(
                normalization["excluded_output_classes"],
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "normalization_equation": str(normalization["equation"]),
            "normalization_status": str(normalization["normalization_status"]),
            "resolution_status": str(construction["resolution_status"]),
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
        "native_reference_production_amount": (details.native_reference_production_amount),
        "provider_source_database": str(details.activity.key[0]),
        "provider_source_code": str(details.activity.key[1]),
        "tracked_plastic_denominator": details.tracked_plastic_denominator,
        "native_processed_plastic_output_amount": (details.processed_plastic_output_amount),
        "native_plastic_reject_output_amount": details.plastic_reject_output_amount,
        "native_processed_feed_coefficient": (details.processed_feed_per_kg_tracked_plastic),
        "native_plastic_reject_coefficient": (details.plastic_reject_per_kg_tracked_plastic),
        "chemical_preparation_normalization_status": str(normalization["normalization_status"]),
        "chemical_recycling_resolution_status": str(construction["resolution_status"]),
        "direct_treatment_carbon_owners_json": json.dumps(
            direct_owners,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "exchanges": [
            {
                "input": (database_name, code),
                "amount": details.tracked_plastic_denominator,
                "type": "production",
            },
            *generated_exchanges,
        ],
    }


def _conversion_sibling_vector(
    context: _RecipeContext,
    activity: Any,
    denominator_selector: dict[str, object],
) -> tuple[
    float,
    tuple[Any, ...],
    tuple[Any, ...],
    dict[tuple[str, str, str, str, tuple[str, ...], str], float],
]:
    """Normalize one coproduct record by its processed plastic feed amount."""
    native = context.exchanges(activity)
    production = tuple(exchange for exchange in native if str(exchange["type"]) == "production")
    denominator = tuple(
        exchange
        for exchange in native
        if _exact_exchange_matches(exchange, context.input(exchange), denominator_selector)
    )
    if len(production) != 1 or len(denominator) != 1:
        raise ValueError(
            "Each hydrothermal sibling must have one production and one processed-feed denominator."
        )
    removed = (*production, denominator[0])
    retained = tuple(exchange for exchange in native if exchange not in removed)
    denominator_amount = float(denominator[0]["amount"])
    vector: dict[tuple[str, str, str, str, tuple[str, ...], str], float] = {}
    for exchange in retained:
        key = _exchange_semantic_key(str(exchange["type"]), context.input(exchange))
        if key in vector:
            raise ValueError("A hydrothermal sibling repeats a retained semantic exchange.")
        vector[key] = float(exchange["amount"]) / denominator_amount
    return denominator_amount, retained, tuple(removed), vector


def _chemical_conversion_exchanges(
    context: _RecipeContext,
    construction: dict[str, object],
    retained: tuple[Any, ...],
    *,
    service_id: str,
    activity: Any,
    database_name: str,
    code: str,
) -> list[dict[str, object]]:
    """Copy hydrothermal conversion exchanges and map direct plastic-carbon flows.

    Args:
        context: Prepared activity and biosphere resolvers for one background scenario
            and anchor year.
        construction: Hydrothermal conversion definition from the provider catalog.
        retained: Exchanges retained after removing production and processed-plastic feed.
        service_id: Identifier of the project conversion service.
        activity: Native hydrothermal activity used for trace records.
        database_name: Project provider database receiving the service.
        code: Brightway code of the project conversion service.

    Returns:
        Copied Brightway exchanges with their native amounts, uncertainty, and
        allocation data. For non-fossil plastic, only the declared direct carbon flow
        identities are replaced by the corresponding non-fossil biosphere flows.

    Raises:
        ValueError: If a declared native direct carbon flow does not occur exactly once.
    """

    rules = tuple(
        cast(
            list[dict[str, object]],
            construction["direct_treatment_carbon_mappings"],
        )
    )
    non_fossil = str(construction["ultimate_carbon_class"]) == "non_fossil"
    matches = [0 for _rule in rules]
    generated: list[dict[str, object]] = []
    for exchange in retained:
        input_node = context.input(exchange)
        matching = [
            (index, rule)
            for index, rule in enumerate(rules)
            if _matches_treatment_carbon_selector(str(exchange["type"]), input_node, rule)
        ]
        copied = _copied_native_exchange(exchange)
        if not matching:
            generated.append(copied)
            continue
        index, rule = matching[0]
        matches[index] += 1
        source_name = str(input_node.get("name", ""))
        replacement = (
            context.biosphere.flow(cast(dict[str, object], rule["replacement_selector"]))
            if non_fossil
            else input_node
        )
        replacement_name = str(replacement.get("name", ""))
        if non_fossil:
            _replace_carbon_identity(copied, replacement)
        generated.append(copied)
        for origin in ("biogenic", "atmospheric_dac") if non_fossil else ("fossil",):
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "provider_demand_type": "chemical_recycling_conversion",
                    "provider_stage_owner": str(construction["stage_owner"]),
                    "ultimate_carbon_origin": origin,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "treatment_carbon_mapping",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    "source_flow_database": str(input_node.key[0]),
                    "source_flow_code": str(input_node.key[1]),
                    "source_flow_name": source_name,
                    "replacement_flow_database": str(
                        replacement.key[0] if non_fossil else input_node.key[0]
                    ),
                    "replacement_flow_code": str(
                        replacement.key[1] if non_fossil else input_node.key[1]
                    ),
                    "replacement_flow_name": (replacement_name if non_fossil else source_name),
                    "source_amount": float(exchange["amount"]),
                    "generated_amount": float(exchange["amount"]),
                    "uncertainty_metadata_preserved": True,
                    "allocation_metadata_preserved": True,
                    "mapping_status": (
                        "exact_non_fossil_counterpart" if non_fossil else "native_fossil_identity"
                    ),
                    "direct_treatment_carbon_owner_database": database_name,
                    "direct_treatment_carbon_owner_code": code,
                    "direct_treatment_carbon_owner_scope": "direct_biosphere_only",
                }
            )
    if any(count != 1 for count in matches):
        raise ValueError("Each declared conversion carbon flow must resolve exactly once.")
    return generated


def _chemical_recycling_conversion_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Compile the chemical recycling conversion service for one anchor year.

    The source activity has several coproduct records for the same hydrothermal
    operation. The function removes each reference production and the processed
    plastic feed, verifies that the remaining operation vectors agree within the
    documented source tolerance, and uses the named canonical coproduct record.
    The service is normalized per kilogram of plastic entering preparation, then
    receives the retained conversion exchanges and the direct treated-plastic
    carbon flows for the selected fossil or non-fossil service.

    Args:
        context: Resolved ecoinvent or premise activities and reporting records.
        service: Chemical recycling conversion recipe from the provider catalog.
        database_name: Prepared provider database receiving the compiled activity.

    Returns:
        Brightway dataset for one kilogram of tracked plastic entering the full
        preparation and conversion chain.

    Raises:
        ValueError: If coproduct records do not reduce to the same conversion
            operation or a declared direct carbon flow is not found exactly once.
    """
    service_id = str(service["provider_service_id"])
    code = provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    denominator_selector = cast(dict[str, object], construction["denominator_selector"])
    siblings: list[
        tuple[
            Any,
            float,
            tuple[Any, ...],
            tuple[Any, ...],
            dict[tuple[str, str, str, str, tuple[str, ...], str], float],
        ]
    ] = []
    resolver = context.resolver(str(construction["source_role"]))
    for reference_product in cast(list[object], construction["reference_products"]):
        activity = resolver.activity(
            {
                "name": construction["activity_name"],
                "reference_product": reference_product,
                "unit": construction["unit"],
                "exact_location": construction["exact_location"],
            },
            year=context.year,
        )
        denominator, retained, removed, vector = _conversion_sibling_vector(
            context,
            activity,
            denominator_selector,
        )
        siblings.append((activity, denominator, retained, removed, vector))
    canonical_reference_product = str(construction["canonical_reference_product"])
    canonical = next(
        row
        for row in siblings
        if str(row[0].get("reference product", "")) == canonical_reference_product
    )
    canonical_vector = canonical[4]
    tolerance = cast(dict[str, object], construction["premise_tolerance"])
    relative = float(cast(float | int | str, tolerance["relative"]))
    absolute = float(cast(float | int | str, tolerance["absolute"]))
    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    for _activity, _denominator, _retained, _removed, vector in siblings:
        if vector.keys() != canonical_vector.keys():
            raise ValueError("Hydrothermal siblings do not retain one semantic operation vector.")
        for key, expected in canonical_vector.items():
            actual = vector[key]
            difference = abs(actual - expected)
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
            divisor = max(abs(actual), abs(expected), absolute)
            maximum_relative_difference = max(maximum_relative_difference, difference / divisor)
            if not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
                raise ValueError(
                    "Hydrothermal sibling operations differ beyond the accepted tolerance."
                )
    preparation_service = next(
        row
        for row in cast(list[dict[str, object]], context.catalogue["services"])
        if row["provider_service_id"] == "chemical_recycling_preparation_fossil"
    )
    preparation_construction = cast(dict[str, object], preparation_service["construction"])
    preparation_details = _chemical_preparation_details(
        context,
        preparation_construction,
    )
    preparation_normalization = cast(
        dict[str, object],
        preparation_construction["tracked_plastic_denominator"],
    )
    activity, denominator, retained, removed, _vector = canonical
    production_amount = denominator / preparation_details.processed_feed_per_kg_tracked_plastic
    _chemical_native_resolution_row(
        context,
        service_id=service_id,
        source_role=str(construction["source_role"]),
        activity=activity,
        retained=retained,
        removed=removed,
        semantic_class="explicit_cross_material_operation_analogue",
    )
    generated_exchanges = _chemical_conversion_exchanges(
        context,
        construction,
        retained,
        service_id=service_id,
        activity=activity,
        database_name=database_name,
        code=code,
    )
    direct_owners = (
        {
            "owner_database": database_name,
            "owner_code": code,
            "owner_scale_per_provider_unit": 1.0,
            "source_role": "member_local_frontier_service",
            "owner_scope": "direct_biosphere_only",
            "origin_mapping_status": (
                "exact_non_fossil_counterpart"
                if str(construction["ultimate_carbon_class"]) == "non_fossil"
                else "native_identity"
            ),
        },
    )
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "provider_service_id": service_id,
            "semantic_class": "explicit_cross_material_operation_analogue",
            "ledger_entry_type": "chemical_recycling_conversion_alignment",
            "native_reference_production_amount": (
                preparation_details.native_reference_production_amount
            ),
            "tracked_plastic_denominator": (preparation_details.tracked_plastic_denominator),
            "processed_plastic_output_amount": (
                preparation_details.processed_plastic_output_amount
            ),
            "plastic_reject_output_amount": (preparation_details.plastic_reject_output_amount),
            "native_processed_feed_coefficient": (
                preparation_details.processed_feed_per_kg_tracked_plastic
            ),
            "native_plastic_reject_coefficient": (
                preparation_details.plastic_reject_per_kg_tracked_plastic
            ),
            "normalization_equation": str(preparation_normalization["equation"]),
            "normalization_status": str(preparation_normalization["normalization_status"]),
            "canonical_denominator": denominator,
            "project_production_amount": production_amount,
            "maximum_absolute_sibling_difference": maximum_absolute_difference,
            "maximum_relative_sibling_difference": maximum_relative_difference,
            "relative_tolerance": relative,
            "absolute_tolerance": absolute,
            "resolution_status": str(construction["resolution_status"]),
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
        "native_reference_production_amount": (
            preparation_details.native_reference_production_amount
        ),
        "tracked_plastic_denominator": (preparation_details.tracked_plastic_denominator),
        "native_processed_plastic_output_amount": (
            preparation_details.processed_plastic_output_amount
        ),
        "native_plastic_reject_output_amount": (preparation_details.plastic_reject_output_amount),
        "native_processed_feed_coefficient": (
            preparation_details.processed_feed_per_kg_tracked_plastic
        ),
        "native_plastic_reject_coefficient": (
            preparation_details.plastic_reject_per_kg_tracked_plastic
        ),
        "chemical_preparation_normalization_status": str(
            preparation_normalization["normalization_status"]
        ),
        "native_conversion_denominator": denominator,
        "chemical_recycling_resolution_status": str(construction["resolution_status"]),
        "direct_treatment_carbon_owners_json": json.dumps(
            direct_owners,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "exchanges": [
            {
                "input": (database_name, code),
                "amount": production_amount,
                "type": "production",
            },
            *generated_exchanges,
        ],
    }
