"""Compile source-backed provider services from native activities and recipes."""

import json
from collections.abc import Sequence
from math import fsum
from typing import cast

from ..pa_finish_container import pa_finish_container
from ..ptt_steam import ptt_steam_dataset
from .ash_treatment import _MUNICIPAL_ASH_RELEASES, _ash_dataset, _ash_market_dataset
from .catalogue import (
    _method_authority,
    _service_dependencies,
    _service_direct_source_ids,
    _service_source_closures,
    _source_authority_closure,
    _source_uses_json,
    provider_service_code,
)
from .chemical_recycling import (
    _chemical_recycling_conversion_dataset,
    _chemical_recycling_preparation_dataset,
)
from .conversion_compilation import (
    _conversion_operation_dataset,
    _input_normalized_operation_dataset,
)
from .hco_nickel import _hco_nickel_dataset
from .leachate_load import _ROW_WASTEWATER_PROFILE, _wastewater_load
from .leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from .native_recipe import (
    _construction_semantic_class,
    _native_exchange,
    _service_trace,
)
from .native_resolution import _add_amount
from .open_burning_compilation import _open_burning_dataset
from .operation_compilation import _operation_only_dataset
from .pa_castor_hco import _pa_castor_hco_dataset
from .pa_ester_ingredients import _pa_ester_dataset
from .pa_ethoxylation import _pa_ethoxylation_dataset
from .pa_finish_concentrate import _pa_finish_concentrate_dataset
from .pa_finish_residues import _pa_finish_residue_dataset
from .pa_operations import (
    _pa_condensate_dataset,
    _pa_finish_nitrogen_dataset,
    _pa_finish_operation_dataset,
)
from .pa_spinning_support import _pa_spinning_support_dataset
from .pa_tridecyl_stearate import _pa_tridecyl_stearate_dataset
from .pdo_manufacture import _pdo_dataset
from .phb_incineration import _phb_treatment_dataset
from .recipe_context import _RecipeContext
from .recovery_compilation import (
    _pellet_recovery_recipe,
    _pet_food_grade_finish_recipe,
    _pet_recovery_recipe,
    _phb_recovery_recipe,
    _polyethylene_recovery_recipe,
    _residue_haul_dataset,
    _tps_recovery_recipe,
    _tps_residue_treatment_recipe,
    _weee_recovery_recipe,
)
from .resin_dump import _resin_dump_dataset
from .resin_incineration import _resin_treatment_dataset
from .resin_landfill import _resin_landfill_treatment_dataset
from .sack_delivery import _tps_sack_delivery_recipe
from .sbr_compounding import _sbr_compounding_recipe
from .sbr_dust_treatment import _sbr_dust_ash_dataset, _sbr_dust_incineration_dataset
from .sbr_landfill import _sbr_treatment_dataset
from .sludge_incineration import _sludge_dataset, _sludge_source
from .sludge_incineration_load import calculate_sludge_incineration
from .sludge_landfarming import _landfarming_dataset
from .tps_native_treatment import _tps_native_treatment_dataset
from .tps_residue_load import _tps_landfill_gas_load, _tps_residue_load
from .tps_site import _tps_landfill_dataset
from .treatment_compilation import (
    _native_treatment_carbon_owner,
    _treatment_origin_operation_dataset,
    _waste_market_dataset,
)
from .upr_compilation import _upr_conversion_dataset, _upr_residue_dataset
from .waste_gas import _landfill_gas, _wastewater_gas
from .water_treatment import (
    _ROW_PROFILE,
    _WATER_PROFILES,
    _water_evaporation,
    _water_market_dataset,
    _water_plant_dataset,
)


def _tps_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Connect each TPS treatment owner to one actual formulation and site basis.

    The original sanitary-landfill reference defines hydraulic volume per wet
    residue kg; the original wastewater climate defines evaporation.  All material
    and conditional treatment loads then share one kg principal TPS. The
    collection market alone applies its treatment fraction. Native source rows
    are already cached by the preparation context; no product cache is needed
    for these small deterministic load calculations.
    """
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    formulation_id = str(construction["formulation_service_id"])
    profile = cast(
        dict[str, object],
        cast(dict[str, object], context.catalogue["semantic_contract"])["tps_residue_treatment"],
    )
    residue = _tps_residue_load(services, formulation_id)
    landfill = context.source.activity(
        cast(dict[str, object], profile["landfill_source"]), year=2025
    )
    source_rows = context.exchanges(landfill)
    # Both source exchanges are negative: m3 leachate / kg wet residue. The
    # actual wet formulation includes its carried water and retained auxiliaries.
    production = next(ex for ex in source_rows if ex["type"] == "production")
    leachate = next(
        ex
        for ex in source_rows
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == "leachate, SLF, municipal solid waste"
    )
    volume = float(leachate["amount"]) / float(production["amount"]) * residue.wet_mass_kg
    plant = context.source.activity(
        cast(dict[str, object], profile["wastewater_source"]), year=2025
    )
    wastewater = _wastewater_load(
        residue.leachate,
        volume_m3=volume,
        # Source leachate reference: 1000 kg wet liquid/m3. Actual solutes and carried residue water
        # occupy their respective parts of that mass.
        incoming_water_kg=1000 * volume - fsum(residue.leachate.elements_kg.values()),
        evaporation_fraction=_water_evaporation(str(plant["comment"]), _ROW_PROFILE),
        profile=_ROW_WASTEWATER_PROFILE,
    )
    service_id = str(service["provider_service_id"])
    if kind == "tps_residue_landfill":
        gas = _landfill_gas(context, construction, _tps_landfill_gas_load(residue), service_id)
        return _tps_landfill_dataset(context, service, residue, wastewater, gas, database_name)
    if kind == "tps_leachate_collection":
        return _leachate_market_dataset(
            context, service, residue.leachate, wastewater, database_name
        )
    if kind == "tps_wastewater_treatment":
        gas = _wastewater_gas(
            context, construction, wastewater, service_id, profile=_ROW_WASTEWATER_PROFILE
        )
        return _wastewater_dataset(
            context,
            service,
            residue.leachate,
            wastewater,
            gas,
            database_name,
            profile=_ROW_WASTEWATER_PROFILE,
        )
    if kind == "tps_sludge_collection":
        return _sludge_market_dataset(context, service, wastewater, database_name)
    if kind == "tps_sludge_landfarming":
        return _landfarming_dataset(
            context,
            service,
            wastewater,
            formulation_id,
            database_name,
            complete_carbon_oxidation=False,
        )
    # The thermal calculation supplies ash composition from the same formulation.
    thermal = (
        construction
        if kind == "tps_sludge_incineration"
        else cast(
            dict[str, object],
            services[str(construction["incineration_service_id"])]["construction"],
        )
    )
    load = calculate_sludge_incineration(
        _sludge_source(context, thermal),
        wastewater,
        formulation_id,
        source_carbon_origins=tuple(
            dict.fromkeys(cast(dict[str, str], thermal["carbon_templates"]).values())
        ),
        additional_apc_water_flows=tuple(cast(list[str], thermal["additional_apc_water_flows"])),
        phosphorus_phase_reference=False,
        # The retained TPS receiving slag has .115628 kg water/.884372 kg dry ash.
        bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
    )
    if kind == "tps_sludge_incineration":
        return _sludge_dataset(context, service, load, database_name)
    ash = load.bottom_ash if construction["ash_fraction"] == "bottom" else load.residual_ash
    if kind == "tps_ash_market":
        return _ash_market_dataset(context, service, ash, database_name)
    return _ash_dataset(
        context,
        service,
        ash,
        database_name,
        release_profile=_MUNICIPAL_ASH_RELEASES[str(construction["ash_fraction"])],
    )


def _service_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Compile one provider service into a Brightway dataset.

    Chemical recycling, treatment and native operations use their
    dedicated scientific compilers. Other services flatten declared components,
    combine identical technosphere and biosphere exchanges, and add one
    unit production exchange in ``database_name``.
    """
    service_id = str(service["provider_service_id"])
    code = provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    if construction["kind"] == "ptt_renewable_steam":
        _service_trace(context, service)
        return ptt_steam_dataset(context)
    if construction["kind"] == "pa_finish_container":
        _service_trace(context, service)
        return pa_finish_container(context)
    if construction["kind"] == "tps_native_treatment":
        return _tps_native_treatment_dataset(context, service, services, database_name)
    if construction["kind"] == "upr_conversion":
        return _upr_conversion_dataset(context, service, database_name)
    if construction["kind"] == "upr_manufacturing_residue":
        return _upr_residue_dataset(context, service, database_name)
    if construction["kind"] in {
        "tps_open_burning",
        "polyester_open_burning",
        "sbr_open_burning",
        "epoxy_open_burning",
        "upr_open_burning",
        "road_open_burning",
    }:
        return _open_burning_dataset(context, service, services, database_name)
    if str(construction["kind"]) in {
        "epoxy_incineration",
        "epoxy_ash_market",
        "epoxy_ash_treatment",
        "pbs_incineration",
        "pbs_ash_market",
        "pbs_ash_treatment",
        "upr_incineration",
        "road_incineration",
        "upr_ash_market",
        "road_ash_market",
        "upr_ash_treatment",
        "road_ash_treatment",
        "sbr_incineration",
        "sbr_ash_market",
        "sbr_ash_treatment",
    }:
        return _resin_treatment_dataset(context, service, services, database_name)
    if construction["kind"] in {"epoxy_open_dump", "upr_open_dump", "road_open_dump", "sbr_dump"}:
        return _resin_dump_dataset(context, service, database_name)
    if construction["kind"] in {
        "epoxy_landfill",
        "epoxy_leachate_collection",
        "epoxy_wastewater_treatment",
        "epoxy_sludge_collection",
        "epoxy_sludge_incineration",
        "epoxy_sludge_landfarming",
        "epoxy_sludge_ash_market",
        "epoxy_sludge_ash_treatment",
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
    }:
        return _resin_landfill_treatment_dataset(context, service, services, database_name)
    if str(construction["kind"]) in {"phb_incineration", "phb_ash_market", "phb_ash_treatment"}:
        return _phb_treatment_dataset(context, service, services, database_name)
    if str(construction["kind"]) in {
        "tps_residue_landfill",
        "tps_leachate_collection",
        "tps_wastewater_treatment",
        "tps_sludge_collection",
        "tps_sludge_landfarming",
        "tps_sludge_incineration",
        "tps_ash_market",
        "tps_ash_treatment",
    }:
        return _tps_treatment_dataset(context, service, services, database_name)
    if str(construction["kind"]) in {
        "sbr_compound_landfill",
        "sbr_filled_residue_landfill",
        "sbr_filled_leachate_collection",
        "sbr_filled_wastewater_treatment",
        "sbr_filled_sludge_collection",
        "sbr_filled_sludge_incineration",
        "sbr_filled_sludge_landfarming",
        "sbr_filled_ash_market",
        "sbr_filled_ash_treatment",
    }:
        return _sbr_treatment_dataset(context, service, services, database_name)
    if str(construction["kind"]) == "pa_ethoxylation":
        return _pa_ethoxylation_dataset(context, service, database_name)
    if str(construction["kind"]) == "pdo_manufacture":
        return _pdo_dataset(context, service, database_name)
    if str(construction["kind"]) == "hco_nickel_manufacture":
        return _hco_nickel_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_castor_hco":
        return _pa_castor_hco_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_ester_ingredient":
        return _pa_ester_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_tridecyl_stearate":
        return _pa_tridecyl_stearate_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_spinning_support":
        return _pa_spinning_support_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_finish_concentrate":
        return _pa_finish_concentrate_dataset(context, service, database_name)
    if str(construction["kind"]) in {"pa_finish_residue_incineration", "pa_finish_residue_ash"}:
        return _pa_finish_residue_dataset(context, service, services, database_name)
    if str(construction["kind"]) == "pa_finish_operation":
        return _pa_finish_operation_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_finish_nitrogen":
        return _pa_finish_nitrogen_dataset(context, service, database_name)
    if str(construction["kind"]) == "pa_condensate_collection":
        return _pa_condensate_dataset(context, service, database_name)
    if str(construction["kind"]) == "sbr_dust_incineration":
        return _sbr_dust_incineration_dataset(context, service, services, database_name)
    if str(construction["kind"]) == "sbr_dust_ash_treatment":
        return _sbr_dust_ash_dataset(context, service, services, database_name)
    if str(construction["kind"]) == "water_collection":
        return _water_market_dataset(context, service, database_name)
    if str(construction["kind"]) == "hydraulic_water_treatment":
        return _water_plant_dataset(
            context, service, database_name, _WATER_PROFILES[str(construction["source_profile"])]
        )
    if str(construction["kind"]) == "plastic_residue_haul":
        return _residue_haul_dataset(context, service, database_name)
    if str(construction["kind"]) == "operation_only":
        return _operation_only_dataset(context, service, services, database_name)
    if str(construction["kind"]) == "input_normalized_operation":
        return _input_normalized_operation_dataset(context, service, database_name)
    if str(construction["kind"]) == "conversion_operation":
        return _conversion_operation_dataset(context, service, database_name)
    if str(construction["kind"]) == "waste_market_origin_operation":
        return _waste_market_dataset(context, service, database_name)
    if str(construction["kind"]) == "chemical_recycling_preparation_operation":
        return _chemical_recycling_preparation_dataset(context, service, database_name)
    if str(construction["kind"]) == "chemical_recycling_conversion_operation":
        return _chemical_recycling_conversion_dataset(context, service, database_name)
    if str(construction["kind"]) == "treatment_origin_operation":
        return _treatment_origin_operation_dataset(
            context,
            service,
            database_name,
        )
    exchanges: dict[tuple[tuple[str, str], str], float] = {}
    _flatten_service(
        context,
        service_id,
        scale=1.0,
        services=services,
        exchanges=exchanges,
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        **_native_treatment_carbon_owner(context, service),
        "exchanges": [
            {
                "input": (database_name, code),
                "amount": 1.0,
                "type": "production",
            },
            *(
                {
                    "input": input_key,
                    "amount": amount,
                    "type": exchange_type,
                }
                for (input_key, exchange_type), amount in sorted(exchanges.items())
                if amount != 0.0
            ),
        ],
    }


def _route_contexts(
    routes: Sequence[dict[str, object]],
    services: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group route rows by provider service and construction context."""
    contexts: dict[str, list[dict[str, object]]] = {}
    for route in routes:
        pending = [str(route["provider_service_id"])]
        visited: set[str] = set()
        while pending:
            service_id = pending.pop()
            if service_id in visited:
                continue
            visited.add(service_id)
            contexts.setdefault(service_id, []).append(route)
            pending.extend(_service_dependencies(services[service_id]))
    return contexts


def _route_semantic_class(
    catalogue: dict[str, object],
    route: dict[str, object],
    service: dict[str, object],
) -> str:
    """Classify a route from explicit catalog overrides or its service construction."""
    contract = cast(dict[str, object], catalogue["semantic_contract"])
    route_id = str(route["route_id"])
    if route_id in cast(list[str], contract["explicit_cross_material_route_ids"]):
        return "explicit_cross_material_analogue"
    if route_id in cast(list[str], contract["service_normalization_route_ids"]):
        return "service_normalization"
    return _construction_semantic_class(cast(dict[str, object], service["construction"]))


def _chemical_route_semantics(
    route: dict[str, object],
    service: dict[str, object],
) -> dict[str, object]:
    """Return the input-operation boundary for preparation and hydrothermal treatment."""

    demand_type = str(route["provider_demand_type"])
    if demand_type not in {
        "chemical_recycling_preparation",
        "chemical_recycling_conversion",
    }:
        return {}
    construction = cast(dict[str, object], service["construction"])
    return {
        "chemical_recycling_resolution_status": str(route["resolution_status"]),
        "chemical_recycling_stage": str(route["component_role"]),
        "chemical_recycling_mapping_basis": str(route["mapping_basis"]),
        "chemical_recycling_analogue_status": str(route["analogue_status"]),
        "chemical_recycling_ultimate_carbon_origin": str(route["ultimate_carbon_origin"]),
        "chemical_recycling_exact_once_owner": str(service["stage_owner"]),
        "chemical_recycling_boundary": str(construction.get("retained_boundary", "")),
    }


def _route_semantic_rows(
    catalogue: dict[str, object],
    services: dict[str, dict[str, object]],
    *,
    routes: Sequence[dict[str, object]],
    background_scenario: str,
    year: int,
) -> list[dict[str, object]]:
    """Describe every supplied provider route at one background anchor year.

    Each row links an MFA demand and material to its provider service, functional
    unit, valid years, component share, carbon origin, and treatment fate. It
    also records the direct and complete source lists, method boundaries,
    analog status, scaling inputs, and known limitation. Chemical recycling
    fields are added for preparation and conversion routes. SBR fields are
    present on every row, with default values where the route omits them.
    Validity bounds are recorded as route metadata.

    Args:
        catalogue: Complete provider service, route, source, and method catalog.
        services: Provider service entries indexed by their identifier.
        routes: Resolved route shares with their original catalog metadata.
        background_scenario: REMIND background scenario prepared for these rows.
        year: Anchor year recorded on each route row.

    Returns:
        One descriptive record per supplied provider route.
    """
    rows: list[dict[str, object]] = []
    source_closures = _service_source_closures(catalogue)
    for route in routes:
        service = services[str(route["provider_service_id"])]
        construction = cast(dict[str, object], service["construction"])
        method_authority_id = str(route.get("method_authority_id", service["citation"]))
        method_authority = _method_authority(catalogue, method_authority_id)
        direct_source_ids = tuple(
            sorted(
                {
                    *_service_direct_source_ids(catalogue, service),
                    *(
                        (str(route["infiltration_source_id"]),)
                        if "infiltration_source_id" in route
                        else ()
                    ),
                    *(
                        str(value)
                        for value in json.loads(str(method_authority["source_authority_ids_json"]))
                    ),
                }
            )
        )
        complete_source_ids = tuple(
            sorted(
                {
                    *source_closures[str(route["provider_service_id"])],
                    *_source_authority_closure(catalogue, list(direct_source_ids)),
                }
            )
        )
        rows.append(
            {
                "background_scenario": background_scenario,
                "ssp_scenario": "SSP2",
                "anchor_year": year,
                "ledger_entry_type": "route_mapping",
                "semantic_class": _route_semantic_class(
                    catalogue,
                    route,
                    service,
                ),
                "route_id": str(route["route_id"]),
                "route_component_id": str(route["route_component_id"]),
                "provider_service_id": str(route["provider_service_id"]),
                "provider_demand_type": str(route["provider_demand_type"]),
                "virgin_feedstock_route": str(route["virgin_feedstock_route"]),
                "ultimate_carbon_origin": str(route.get("ultimate_carbon_origin", "")),
                "treatment_carbon_mapping_status": str(
                    route.get("treatment_carbon_mapping_status", "")
                ),
                "modeled_stage": str(route["component_role"]),
                "material": str(route["material"]),
                "constituent": str(route["constituent"]),
                "product": str(route["product"]),
                "good": str(route["good"]),
                "lca_fate": str(route.get("lca_fate", "")),
                "infiltration_class": str(route.get("infiltration_class", "")),
                "provider_class": str(route.get("provider_class", "")),
                "analogue_status": str(route.get("analogue_status", "")),
                "material_functional_unit": str(route.get("material_functional_unit", "")),
                "styrene_component_share": float(
                    cast(float | int | str, route.get("styrene_component_share", 0.0))
                ),
                "butadiene_analogue_component_share": float(
                    cast(
                        float | int | str,
                        route.get("butadiene_analogue_component_share", 0.0),
                    )
                ),
                "sbr_share_in_material": float(
                    cast(float | int | str, route.get("sbr_share_in_material", 0.0))
                ),
                "sbr_proxy_component_role": str(route.get("sbr_proxy_component_role", "")),
                "whole_tyre_constituents_excluded": bool(
                    route.get("whole_tyre_constituents_excluded", False)
                ),
                "no_treatment_sensitivity_executed": bool(
                    route.get("no_treatment_sensitivity_executed", False)
                ),
                "h12_applicability": str(route["region"]),
                "valid_from_year": int(cast(int | str, route["valid_from_year"])),
                "valid_to_year": int(cast(int | str, route["valid_to_year"])),
                "source_class": (
                    "project_composed_provider"
                    if str(construction["kind"]) != "native_activity"
                    else str(
                        construction.get(
                            "source_role",
                            "native_ecoinvent_provider",
                        )
                    )
                ),
                "interpolation_rule": (
                    "matching-member linear interpolation between adjacent "
                    "provider frontiers; 2019 uses the 2020 forward proxy"
                ),
                "project_demand": str(route["mapping_basis"]),
                "project_output": str(service["reference_product"]),
                "project_unit": str(service["unit"]),
                "foreground_stage_owner": str(service["stage_owner"]),
                "bundled_stage_coverage": (
                    "native ordinary background supply chains retained; modeled "
                    "plastics stage assigned once to the declared route"
                ),
                "exact_once_decision": (
                    f"{route['provider_demand_type']} is owned once by {service['stage_owner']}"
                ),
                "scaling_algebra": json.dumps(
                    {
                        "route_share": float(cast(float | int | str, route["share"])),
                        "throughput_multiplier": float(
                            cast(
                                float | int | str,
                                route["throughput_multiplier"],
                            )
                        ),
                        "mapping_basis": route["mapping_basis"],
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "functional_equivalence_claim": str(
                    route.get(
                        "functional_equivalence_claim",
                        f"{route['mapping_basis']} maps to {service['reference_product']} "
                        f"in {service['unit']}",
                    )
                ),
                "reason": str(route.get("selection_reason", service["name"])),
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
                    catalogue,
                    direct_source_ids=direct_source_ids,
                    complete_source_ids=complete_source_ids,
                ),
                "method_technology_boundary": str(method_authority["technology_boundary"]),
                "method_allocation_boundary": str(method_authority["allocation_boundary"]),
                "method_geography_year_boundary": str(method_authority["geography_year_boundary"]),
                "method_details_json": json.dumps(
                    service.get("representative_method", {}),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "limitation": str(route.get("limitation", method_authority["limitation"])),
                "failure_rule": (
                    "fail when the route service or an exact declared native "
                    "functional product is unavailable in the selected member"
                ),
                "retained_technosphere_json": "[]",
                "removed_technosphere_json": "[]",
                "retained_biosphere_json": "[]",
                "project_terms_json": "[]",
                **_chemical_route_semantics(route, service),
            }
        )
    return rows


def _project_recipe(
    context: _RecipeContext,
    construction: dict[str, object],
    *,
    service_id: str,
    scale: float,
    services: dict[str, dict[str, object]],
    exchanges: dict[tuple[tuple[str, str], str], float],
) -> None:
    """Compile one explicit project recipe at its requested output scale."""

    for component in cast(
        list[dict[str, object]],
        construction["components"],
    ):
        amount = scale * float(cast(float | int | str, component["amount"]))
        if "service_id" in component:
            _flatten_service(
                context,
                str(component["service_id"]),
                scale=amount,
                services=services,
                exchanges=exchanges,
            )
        else:
            _native_exchange(
                context,
                component,
                service_id=service_id,
                scale=scale,
                exchanges=exchanges,
            )
    for term in cast(
        list[dict[str, object]],
        construction.get("biosphere_terms", []),
    ):
        flow = context.biosphere.flow(cast(dict[str, object], term["selector"]))
        flow_key = cast(tuple[str, str], flow.key)
        _add_amount(
            exchanges,
            flow_key,
            "biosphere",
            scale * float(cast(float | int | str, term["amount"])),
        )


def _flatten_service(
    context: _RecipeContext,
    service_id: str,
    *,
    scale: float,
    services: dict[str, dict[str, object]],
    exchanges: dict[tuple[tuple[str, str], str], float],
) -> None:
    """Add one provider service to a compiled Brightway exchange mapping.

    Args:
        context: Prepared activity resolvers and trace records for one background
            scenario and anchor year.
        service_id: Provider service to compile.
        scale: Demand for the service in its declared functional unit.
        services: Provider-catalog entries indexed by service identifier.
        exchanges: Mutable mapping from Brightway input and exchange type to amount.

    Raises:
        KeyError: If ``service_id`` is absent from the provider catalog.
        ValueError: If the selected service cannot be resolved to the exchanges declared
            by its construction.

    Notes:
        The service construction selects the applicable native activity, normalized
        operation or project recipe. Nested project services call
        this function recursively; each construction removes the inputs it replaces before
        adding its retained exchanges, which prevents duplicate burdens.
        Nested native operations link to member-local frontier activities,
        preserving their production basis and native exchange metadata.
    """
    service = services[service_id]
    _service_trace(context, service)
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    if kind == "native_activity":
        _native_exchange(
            context,
            {
                **construction,
                "component_id": "native_activity",
                "amount": float(cast(float | int | str, construction.get("amount", 1.0))),
            },
            service_id=service_id,
            scale=scale,
            exchanges=exchanges,
        )
    elif kind in {
        "chemical_water_collection",
        "ptt_renewable_steam",
        "pa_finish_container",
        "tps_native_treatment",
        "sbr_open_burning",
        "treatment_origin_operation",
        "pdo_manufacture",
        "phb_incineration",
        "phb_ash_market",
        "phb_ash_treatment",
        "pbs_incineration",
        "pbs_ash_market",
        "pbs_ash_treatment",
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
        "input_normalized_operation",
        "operation_only",
        "plastic_residue_haul",
        "tps_residue_landfill",
        "tps_leachate_collection",
        "tps_wastewater_treatment",
        "tps_sludge_collection",
        "tps_sludge_incineration",
        "tps_sludge_landfarming",
        "tps_ash_market",
        "tps_ash_treatment",
        "water_collection",
        "hydraulic_water_treatment",
        "hco_nickel_manufacture",
        "pa_castor_hco",
        "pa_ester_ingredient",
        "pa_ethoxylation",
        "pa_tridecyl_stearate",
        "pa_spinning_support",
        "pa_finish_concentrate",
        "pa_finish_operation",
        "pa_finish_nitrogen",
        "pa_finish_residue_incineration",
        "pa_finish_residue_ash",
        "pa_condensate_collection",
        "sbr_incineration",
        "sbr_ash_market",
        "sbr_ash_treatment",
        "sbr_dump",
        "sbr_compound_landfill",
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
        _add_amount(
            exchanges,
            (context.provider_database, provider_service_code(service_id)),
            "technosphere",
            scale,
        )
    else:
        if kind == "sbr_compounding":
            construction = _sbr_compounding_recipe(context, construction)
        elif kind == "polyethylene_recovery_operation":
            construction = _polyethylene_recovery_recipe(context, construction)
        elif kind == "pellet_recovery_operation":
            construction = _pellet_recovery_recipe(context, construction)
        elif kind == "pet_recovery_operation":
            construction = _pet_recovery_recipe(context, construction)
        elif kind == "weee_recovery_operation":
            construction = _weee_recovery_recipe(context, construction)
        elif kind == "phb_recovery_operation":
            construction = _phb_recovery_recipe(context, construction)
        elif kind == "tps_recovery_operation":
            construction = _tps_recovery_recipe(context, construction, services)
        elif kind == "tps_residue_treatment":
            construction = _tps_residue_treatment_recipe(construction, services)
        elif kind == "tps_sack_delivery":
            construction = _tps_sack_delivery_recipe(context, construction)
        elif kind == "pet_food_grade_finish":
            construction = _pet_food_grade_finish_recipe(
                context, construction, services, service_id=service_id
            )
        _project_recipe(
            context,
            construction,
            service_id=service_id,
            scale=scale,
            services=services,
            exchanges=exchanges,
        )
