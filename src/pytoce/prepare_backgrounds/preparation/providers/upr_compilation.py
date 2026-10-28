"""Compile actual UPR formulation and the complete native hand-lay-up operation.

The material recipe purchases principal resin separately. This owner supplies
the actual auxiliaries, retains the native operation's independent materials
and receiving, and passes only the actual homogeneous article trim to its
complete receiving service. The article's composition defines the modeled atoms and carbon origins.
"""

import json
from dataclasses import asdict
from typing import cast

from .catalogue import provider_service_code
from .leachate_load import _CarbonOrigin
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .upr_material import _upr_material, _upr_reference


def _upr_conversion_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Build one application activity per kg retained principal UPR.

    Native FR operation quantities scale by actual gross article throughput
    divided by native positive production. Its full mold/tool/cleaning duties
    remain. Actual styrene replaces the source's styrene amount once, retaining
    its uncertainty; white spirit uses the admitted existing NMVOC-class flow,
    and free hardener water uses its exact water-air identity.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    conversion = next(
        row
        for row in context.material_recipe.conversion
        if row.operation == str(construction["operation"])
    )
    reference = _upr_reference(
        conversion, cast(_CarbonOrigin, construction["principal_carbon_origin"])
    )
    _service_trace(context, service)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    for label, amount in reference.supply_amounts.items():
        selector = selectors[label]
        if "service_id" in selector:
            # Peroxide and Co2EH have separate manufacture and delivery. Their demands are kg actual
            # ingredient.
            supplier_key = database_name, provider_service_code(str(selector["service_id"]))
        else:
            supplier, trace = context.activity(
                selector,
                source_role="native_ecoinvent_provider",
                service_id=service_id,
                component_id=label,
                semantic_class="source_backed_portfolio",
            )
            supplier_key = cast(tuple[str, str], supplier.key)
            trace["project_terms_json"] = json.dumps(
                {"actual_input_kg_per_retained_principal_kg": amount}, sort_keys=True
            )
        rows.append(
            {"input": supplier_key, "type": "technosphere", "amount": amount, "unit": "kilogram"}
        )

    operation, trace = context.activity(
        cast(dict[str, object], construction["operation_source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_hand_lay_up_operation",
        semantic_class="service_normalization",
    )
    native = context.exchanges(operation)
    production = _self_production_amount(operation, native, method="UPR hand lay-up")
    operation_scale = reference.operation_kg / production
    styrene_total = reference.principal_styrene_air_kg + reference.accelerator_styrene_air_kg
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        multiplier = operation_scale
        if (
            exchange["type"] == "technosphere"
            and node["reference product"] == "waste plastic, mixture"
        ):
            # Native -.186 kg/service combines article trim with auxiliaries.
            # The admitted source split retains -.08803 kg/service nonarticle
            # mold/tool/packaging waste; actual article trim is linked below.
            multiplier *= 0.08803 / 0.186
        elif exchange["type"] == "biosphere" and node["name"] == "Styrene":
            # The material recipe owns the source-total transfer and equal fractional
            # principal/accelerator loss. The material fates retain their source definitions across
            # backgrounds. Preserve source spread while setting this one native row to its actual
            # total kg.
            multiplier = styrene_total / float(exchange["amount"])
        rows.append(_scaled_native_exchange(exchange, multiplier))
    trace["project_terms_json"] = json.dumps(
        {
            "native_positive_production_kg": production,
            "operation_kg_per_retained_principal_kg": reference.operation_kg,
            "operation_multiplier": operation_scale,
            "mixed_waste_retained_fraction": 0.08803 / 0.186,
            "actual_styrene_kg": styrene_total,
        },
        sort_keys=True,
    )

    for name, unit, amount in (
        (
            "NMVOC, non-methane volatile organic compounds",
            "kilogram",
            reference.white_spirit_air_kg,
        ),
        # Existing Water-air is m3 liquid-equivalent water. The source-based freshwater convention
        # is 1000 kg/m3.
        ("Water", "cubic meter", reference.water_air_kg / 1000),
    ):
        flow = context.biosphere.flow(
            {"name": name, "unit": unit, "categories": ["air", "urban air close to ground"]}
        )
        rows.append(
            {
                "input": cast(tuple[str, str], flow.key),
                "type": "biosphere",
                "amount": amount,
                "unit": unit,
            }
        )
    rows.append(
        {
            "input": (
                database_name,
                provider_service_code(
                    str(cast(dict[str, object], construction["actual_residue"])["service_id"])
                ),
            ),
            "type": "technosphere",
            # This complete receiver's +1 unit is principal kg, including its fixed associated
            # auxiliaries.
            "amount": reference.external_solid.components_kg["principal"],
            "unit": "kilogram",
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
        "exchanges": rows,
        "upr_reference_schema": "pytoce_upr_conversion_reference_schema_v1",
        "upr_reference_per_retained_principal_kg": asdict(reference),
        "comment": (
            "Actual UPR/glass application per kg retained principal. Principal resin is "
            "purchased by the material recipe; complete native FR hand lay-up retains its "
            "mould resin, glass, tools, acetone, sanitation water and independent receiving. "
            "Only the article part of its mixed-waste row is replaced by actual homogeneous "
            "trim receiving. Actual target ingredients keep their independent complete "
            "supply proxies and fixed fossil auxiliary carbon. Principal and accelerator "
            "styrene share one native urban emission with source uncertainty retained. "
            "White-spirit evaporation uses the existing urban NMVOC class as a "
            "whole-mixture characterization proxy. Free-water evaporation uses the "
            "existing urban Water flow. "
            "All native EF factors remain unchanged. Retained and trimmed dry material "
            "share one fixed composition. Nominal elemental composition and tracked "
            "principal carbon are recorded separately."
        ),
    }


def _upr_residue_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Keep the complete FR disposal mix per kg homogeneous UPR principal trim.

    Each positive terminal service already includes the fixed cured auxiliaries,
    so its share is the original negative treatment amount divided by negative
    market production. Freight alone scales to whole article mass. The native
    open-burning share remains explicit despite rounding to zero in its comment.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    conversion = next(
        row
        for row in context.material_recipe.conversion
        if row.operation == "others_upr_hand_lay_up_conversion"
    )
    origin = str(construction["principal_carbon_origin"])
    material = _upr_material(
        conversion,
        (
            1.0 if origin == "fossil" else 0.0,
            1.0 if origin == "biogenic" else 0.0,
            1.0 if origin == "atmospheric_dac" else 0.0,
        ),
    )
    _service_trace(context, service)
    source, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_manufacturing_disposal_market",
        semantic_class="service_normalization",
    )
    native = context.exchanges(source)
    production = _self_production_amount(source, native, method="UPR residue market")
    treatments = cast(dict[str, dict[str, object]], construction["treatments"])
    targets = {
        "treatment of waste plastic, mixture, open burning": treatments["open_burning"],
        "treatment of waste plastic, mixture, sanitary landfill": treatments["sanitary_landfill"],
        "treatment of waste plastic, mixture, municipal incineration": treatments[
            "municipal_incineration"
        ],
    }
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    rebound = []
    retained = []
    bindings = []
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        if (
            exchange["type"] == "technosphere"
            and node["reference product"] == "waste plastic, mixture"
        ):
            target = str(targets[str(node["name"])]["service_id"])
            # Normalize the native negative amount by positive |production|
            # before changing reference sign; lognormal loc remains log|amount|.
            record = _scaled_native_exchange(exchange, -1 / production)
            record["input"] = database_name, provider_service_code(target)
            record["amount"] = -float(cast(float, record["amount"]))
            if "loc" in record and record.get("uncertainty type", 0) != 2:
                record["loc"] = -float(cast(float, record["loc"]))
            if "negative" in record:
                record["negative"] = False
            minimum = record.pop("minimum", None)
            maximum = record.pop("maximum", None)
            if maximum is not None:
                record["minimum"] = -float(cast(float, maximum))
            if minimum is not None:
                record["maximum"] = -float(cast(float, minimum))
            # The new input is a positive principal-unit cured-composite service. Its exact input
            # key owns identity; donor composition metadata no longer applies.
            for field in (
                "activity",
                "name",
                "reference product",
                "flow",
                "CAS number",
                "chemical formula",
                "properties",
                "classifications",
            ):
                record.pop(field, None)
            rows.append(record)
            rebound.append(exchange)
            bindings.append(
                {"native_input": node.key, "service_id": target, "principal_kg": record["amount"]}
            )
        else:
            rows.append(_scaled_native_exchange(exchange, -material.mass_kg / production))
            retained.append(exchange)
    trace["retained_technosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "technosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(rebound)
    trace["project_terms_json"] = json.dumps(
        {
            "native_negative_production_kg": production,
            "whole_material_kg_per_principal_kg": material.mass_kg,
            "terminal_bindings": bindings,
        },
        sort_keys=True,
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "upr_residue_schema": "pytoce_upr_manufacturing_residue_schema_v1",
        "whole_material_kg_per_principal_kg": material.mass_kg,
        "comment": (
            "Complete native FR manufacturing-disposal market adapted to homogeneous "
            "cured UPR/glass trim. Each original treatment share, including open burning, "
            "purchases a positive principal-unit actual-material receiver. Collection and "
            "all native freight scale once to the associated whole article mass; no extra "
            "terminal mass factor, formulation purchase or EOL collection is added. "
            "Native shares are neither rounded nor renormalized. Source uncertainty and "
            "allocation remain, with only the negative-to-positive reference conversion."
        ),
    }
