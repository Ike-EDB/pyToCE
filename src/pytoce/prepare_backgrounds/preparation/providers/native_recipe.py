"""Native source-backed recipe tracing and operation extraction."""

import json
from typing import Any, cast

from .catalogue import (
    _service_direct_source_ids,
    _source_uses_json,
)
from .native_resolution import _add_amount
from .recipe_context import _RecipeContext


def _construction_semantic_class(
    construction: dict[str, object],
) -> str:
    """Classify one native construction by provider source and recipe kind."""
    kind = str(construction["kind"])
    if (
        kind == "native_activity"
        and construction.get("analogue_classification") == "explicit_cross_material_analogue"
    ):
        return "explicit_cross_material_analogue"
    if kind in {
        "operation_only",
        "treatment_origin_operation",
        "conversion_operation",
        "chemical_recycling_preparation_operation",
        "chemical_recycling_conversion_operation",
        "waste_market_origin_operation",
        "polyolefin_polymerization_operation",
        "recovered_monomer_cracker_operation",
        "pa_finish_operation",
    }:
        return "operation_only_extraction"
    if kind in {
        "phb_incineration",
        "phb_ash_market",
        "phb_ash_treatment",
        "pbs_incineration",
        "pbs_ash_market",
        "pbs_ash_treatment",
        "epoxy_incineration",
        "epoxy_ash_market",
        "epoxy_ash_treatment",
        "input_normalized_operation",
        "epoxy_coating",
        "road_coating",
        "upr_conversion",
        "upr_manufacturing_residue",
        "upr_incineration",
        "road_incineration",
        "upr_ash_market",
        "road_ash_market",
        "upr_ash_treatment",
        "road_ash_treatment",
        "upr_open_burning",
        "road_open_burning",
        "upr_open_dump",
        "road_open_dump",
        "upr_landfill",
        "road_landfill",
        "upr_leachate_collection",
        "road_leachate_collection",
        "upr_wastewater_treatment",
        "road_wastewater_treatment",
        "upr_sludge_collection",
        "road_sludge_collection",
        "upr_sludge_incineration",
        "road_sludge_incineration",
        "upr_sludge_landfarming",
        "road_sludge_landfarming",
        "upr_sludge_ash_market",
        "road_sludge_ash_market",
        "upr_sludge_ash_treatment",
        "road_sludge_ash_treatment",
        "polyolefin_monomer_supply",
        "polyethylene_recovery_operation",
        "pellet_recovery_operation",
        "pet_recovery_operation",
        "weee_recovery_operation",
        "phb_recovery_operation",
        "tps_recovery_operation",
        "tps_residue_treatment",
        "tps_open_burning",
        "tps_native_treatment",
        "polyester_open_burning",
        "sbr_open_burning",
        "sbr_compound_landfill",
        "sbr_incineration",
        "sbr_ash_market",
        "sbr_ash_treatment",
        "sbr_dump",
        "epoxy_open_burning",
        "epoxy_open_dump",
        "epoxy_landfill",
        "epoxy_leachate_collection",
        "epoxy_wastewater_treatment",
        "epoxy_sludge_collection",
        "epoxy_sludge_incineration",
        "epoxy_sludge_landfarming",
        "epoxy_sludge_ash_market",
        "epoxy_sludge_ash_treatment",
        "tps_sack_delivery",
        "tps_residue_landfill",
        "tps_leachate_collection",
        "tps_wastewater_treatment",
        "tps_sludge_collection",
        "tps_sludge_incineration",
        "tps_sludge_landfarming",
        "tps_ash_market",
        "tps_ash_treatment",
        "plastic_residue_haul",
        "pet_food_grade_finish",
        "water_collection",
        "hydraulic_water_treatment",
        "pa_ethoxylation",
        "pdo_manufacture",
        "hco_nickel_manufacture",
        "pa_castor_hco",
        "pa_ester_ingredient",
        "pa_tridecyl_stearate",
        "pa_spinning_support",
        "pa_finish_concentrate",
        "pa_finish_residue_incineration",
        "pa_finish_residue_ash",
        "swiss_sludge_incineration",
        "chemical_water_collection",
        "ptt_renewable_steam",
        "pa_finish_container",
        "chemical_wastewater_treatment",
        "chemical_sludge_collection",
        "chemical_sludge_landfarming",
        "chemical_ash_market",
        "chemical_ash_treatment",
        "sbr_compounding",
        "sbr_filled_residue_landfill",
        "sbr_filled_leachate_collection",
        "sbr_filled_wastewater_treatment",
        "sbr_filled_sludge_collection",
        "sbr_filled_sludge_incineration",
        "sbr_filled_sludge_landfarming",
        "sbr_filled_ash_market",
        "sbr_filled_ash_treatment",
        "sbr_dust_incineration",
        "sbr_dust_ash_treatment",
    }:
        return "service_normalization"
    if kind in {
        "project_recipe",
        "pa_finish_nitrogen",
        "pa_condensate_collection",
    }:
        return "source_backed_portfolio"
    return (
        "premise_transformed_same_product"
        if str(construction["source_role"]) == "premise_only_provider"
        else "unchanged_same_product"
    )


def _service_trace(
    context: _RecipeContext,
    service: dict[str, object],
) -> None:
    """Trace nested provider services used by one native recipe."""
    service_id = str(service["provider_service_id"])
    if service_id in context.traced_services:
        return
    context.traced_services.add(service_id)
    construction = cast(dict[str, object], service["construction"])
    method_authority_id = str(service["citation"])
    method_authority = context.method_authorities[method_authority_id]
    direct_source_ids = _service_direct_source_ids(
        context.catalogue,
        service,
    )
    complete_source_ids = context.complete_source_ids[service_id]
    method_details = cast(dict[str, object], service.get("representative_method", {}))
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "ledger_entry_type": "service_construction",
            "semantic_class": _construction_semantic_class(construction),
            "provider_service_id": service_id,
            "project_output": str(service["reference_product"]),
            "project_unit": str(service["unit"]),
            "foreground_stage_owner": str(service["stage_owner"]),
            "bundled_stage_coverage": (
                "complete_native_supply_chain_retained; project plastics stages "
                "follow the declared foreground stage owner"
            ),
            "exact_once_decision": (
                f"owned once by {service['stage_owner']}; nested project plastics "
                "stages are omitted only when replaced or separately routed"
            ),
            "scaling_algebra": json.dumps(
                construction,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "functional_equivalence_claim": str(
                method_details.get(
                    "functional_equivalence_claim",
                    f"{service['reference_product']} in {service['unit']}",
                )
            ),
            "reason": str(service["name"]),
            "method_authority_id": method_authority_id,
            "source_authority_ids_json": json.dumps(
                complete_source_ids,
                separators=(",", ":"),
            ),
            "direct_source_authority_ids_json": json.dumps(
                direct_source_ids,
                separators=(",", ":"),
            ),
            "complete_source_authority_ids_json": json.dumps(
                complete_source_ids,
                separators=(",", ":"),
            ),
            "source_uses_json": _source_uses_json(
                context.catalogue,
                direct_source_ids=direct_source_ids,
                complete_source_ids=complete_source_ids,
            ),
            "method_technology_boundary": str(method_authority["technology_boundary"]),
            "method_allocation_boundary": str(method_authority["allocation_boundary"]),
            "method_geography_year_boundary": str(method_authority["geography_year_boundary"]),
            "method_details_json": json.dumps(
                method_details,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "limitation": str(method_authority["limitation"]),
            "failure_rule": (
                "fail when an exact declared native functional product, unit, "
                "location, or complete project component is unavailable"
            ),
            "retained_technosphere_json": "[]",
            "removed_technosphere_json": "[]",
            "retained_biosphere_json": "[]",
            "project_terms_json": json.dumps(
                construction.get("biosphere_terms", []),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    )


def _native_exchange(
    context: _RecipeContext,
    component: dict[str, object],
    *,
    service_id: str,
    scale: float,
    exchanges: dict[tuple[tuple[str, str], str], float],
) -> None:
    """Format one resolved native exchange for a provider recipe."""
    if component.get("analogue_classification") == "explicit_cross_material_analogue":
        semantic_class = "explicit_cross_material_analogue"
    elif str(component["component_id"]) == "native_activity":
        semantic_class = _construction_semantic_class(component)
    else:
        semantic_class = "source_backed_portfolio"
    activity, _row = context.activity(
        cast(dict[str, object], component["selector"]),
        source_role=str(component["source_role"]),
        service_id=service_id,
        component_id=str(component["component_id"]),
        semantic_class=semantic_class,
    )
    input_key = cast(tuple[str, str], activity.key)
    if (
        str(component["source_role"]) == "premise_only_provider"
        and context.ordinary.database_name != context.technology.database_name
    ):
        # Keep the selected premise activity as a project database copy even when
        # ecoinvent contains an activity with the same product. Replace only its
        # ordinary technosphere inputs with matching ecoinvent activities.
        input_key = context._hybrid_clone_activity(activity, service_id=service_id)
    _add_amount(
        exchanges,
        input_key,
        "technosphere",
        scale * float(cast(float | int | str, component["amount"])),
    )


def _self_production_amount(
    activity: Any,
    exchanges: tuple[Any, ...],
    *,
    method: str,
) -> float:
    """Return the sole native reference-production amount.

    These source-dependent constructions preserve a selected native operation.
    A coproduct activity would require a different allocation method and is
    therefore not supported by these constructions.
    """

    production_exchanges = tuple(
        exchange for exchange in exchanges if str(exchange["type"]) == "production"
    )
    self_production = tuple(
        exchange
        for exchange in production_exchanges
        if cast(tuple[str, str], exchange["input"]) == cast(tuple[str, str], activity.key)
    )
    if len(production_exchanges) != 1 or len(self_production) != 1:
        raise ValueError(
            f"The native {method} must have one self-production exchange "
            f"and no coproduct: {activity.get('name')}."
        )
    return float(self_production[0]["amount"])
