"""Dated provider-service frontier database and preparation ledgers."""

import json
from dataclasses import dataclass
from typing import Any, cast

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.shared.runtime.brightway import (
    write_brightway_database,
)

from ..chemical_organic_wastewater import chemical_organic_wastewater_datasets
from .catalogue import (
    PreparedProviderAuthority,
    _service_source_closures,
    provider_frontier_database_name,
    provider_service_code,
)
from .coating_compilation import epoxy_coating_datasets, road_coating_datasets
from .leachate_load import _CarbonOrigin
from .native_resolution import _NativeResolverCache
from .pa_phosphate_water import pa_phosphate_water_datasets
from .pdo_manufacture import _pdo_reference
from .polyolefins import polyolefin_datasets
from .recipe_compilation import (
    _RecipeContext,
    _route_contexts,
    _route_semantic_rows,
    _service_dataset,
)
from .treatment_compilation import attach_landfill_carbon_owners, attach_tps_carbon_owners


@dataclass(frozen=True, slots=True)
class ProviderFrontier:
    """One dated provider-service database and its preparation ledgers."""

    database_name: str
    semantic_rows: tuple[dict[str, object], ...]
    technology_rows: tuple[dict[str, object], ...]


def _attach_applicability(
    rows: list[dict[str, object]],
    contexts: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Attach the matching route contexts to each provider table row."""
    result = []
    for row in rows:
        service_id = str(row["provider_service_id"])
        row_origin = str(row.get("ultimate_carbon_origin", ""))
        applicability = [
            {
                "route_id": route["route_id"],
                "route_component_id": route["route_component_id"],
                "provider_demand_type": route["provider_demand_type"],
                "virgin_feedstock_route": route["virgin_feedstock_route"],
                "ultimate_carbon_origin": route.get("ultimate_carbon_origin", ""),
                "material": route["material"],
                "constituent": route["constituent"],
                "product": route["product"],
                "good": route["good"],
                "lca_fate": route.get("lca_fate", ""),
                "infiltration_class": route.get("infiltration_class", ""),
                "h12_applicability": route["region"],
                "valid_from_year": route["valid_from_year"],
                "valid_to_year": route["valid_to_year"],
            }
            for route in contexts.get(service_id, ())
            if not row_origin or str(route.get("ultimate_carbon_origin", "")) in (row_origin, "all")
        ]
        result.append(
            {
                **row,
                "applicability_json": json.dumps(
                    applicability,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
    return result


def prepare_provider_frontier(
    bd: Any,
    *,
    background_scenario: str,
    year: int,
    ordinary_database: str,
    technology_database: str,
    provider: PreparedProviderAuthority,
    material_recipe: EcoinventMaterialRecipe,
    resolver_cache: _NativeResolverCache,
) -> ProviderFrontier:
    """Create one dated provider-service frontier from declared recipes and native activities."""

    catalogue = provider.catalogue
    database_name = provider_frontier_database_name(
        background_scenario, year, interface_database=provider.interface_database
    )
    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], catalogue["services"])
    }
    ordinary_resolver = resolver_cache.activities(bd, ordinary_database)
    technology_resolver = resolver_cache.activities(bd, technology_database)
    context = _RecipeContext(
        background_scenario=background_scenario,
        year=year,
        provider_database=database_name,
        source=resolver_cache.activities(bd, resolver_cache.source_database),
        ordinary=ordinary_resolver,
        technology=technology_resolver,
        biosphere=resolver_cache.flows(bd),
        catalogue=catalogue,
        material_recipe=material_recipe,
        method_authorities={
            str(row["method_authority_id"]): row for row in provider.method_authorities
        },
        complete_source_ids=_service_source_closures(catalogue),
        semantic_rows=[],
        technology_rows=[],
        traced_services=set(),
    )
    frontier_services = tuple(
        service
        for service in services.values()
        if bool(service.get("foreground_service", False))
        or bool(service.get("frontier_service", False))
    )
    # Resolve each polymer once for its operation and all allocated monomer components.
    data = polyolefin_datasets(context, services, database_name)
    # Build each chemical load with its complete receiving chain, retaining all ten linked records
    # and their actual compositions.
    for service in frontier_services:
        construction = cast(dict[str, object], service["construction"])
        if construction["kind"] == "pdo_manufacture":
            reference = _pdo_reference(cast(_CarbonOrigin, construction["origin"]))
            data.update(
                chemical_organic_wastewater_datasets(
                    context,
                    service_id=str(
                        cast(dict[str, object], construction["water_collection"])["service_id"]
                    ),
                    load=reference.load,
                    solution_volume_m3=reference.solution_volume_m3,
                    water_kg=reference.water_kg,
                    complete_carbon_oxidation=True,
                    phosphorus_equilibrium=True,
                )
            )
        elif service["provider_service_id"] == "pa_tea_monooleyl_phosphate":
            receiving = cast(dict[str, dict[str, float]], construction["receiving_loads"])
            data.update(pa_phosphate_water_datasets(context, **receiving["water"]))
    for service in frontier_services:
        service_id = str(service["provider_service_id"])
        key = (database_name, provider_service_code(service_id))
        if cast(dict[str, object], service["construction"])["kind"] == "epoxy_coating":
            compiled = epoxy_coating_datasets(context, service, database_name)
        elif cast(dict[str, object], service["construction"])["kind"] == "road_coating":
            compiled = road_coating_datasets(context, service, database_name)
        else:
            compiled = {
                key: data[key]
                if key in data
                else _service_dataset(context, service, services, database_name)
            }
        for compiled_key, dataset in compiled.items():
            data[compiled_key] = context.isolate_early_provider_upstream(dataset)
    data.update(context.hybrid_datasets)
    attach_tps_carbon_owners(data, services, database_name)
    attach_landfill_carbon_owners(data, services, database_name)
    write_brightway_database(
        bd,
        database_name,
        data,
        {
            "pytoce_provider_identity": provider.identity,
            "background_scenario": background_scenario,
            "anchor_year": year,
        },
    )
    # Shared preparation preserves catalog row order. Overlay only calculated
    # fields so absent optional metadata keeps its original meaning.
    routes = [
        {
            **raw,
            **{field: resolved[field] for field in ("share", "infiltration_share") if field in raw},
        }
        for raw, resolved in zip(
            cast(list[dict[str, object]], catalogue["routes"]),
            provider.routes.to_dict(orient="records"),
            strict=True,
        )
    ]
    route_contexts = _route_contexts(routes, services)
    semantic_rows = [
        *context.semantic_rows,
        *_route_semantic_rows(
            catalogue,
            services,
            routes=routes,
            background_scenario=background_scenario,
            year=year,
        ),
    ]
    return ProviderFrontier(
        database_name=database_name,
        semantic_rows=tuple(_attach_applicability(semantic_rows, route_contexts)),
        technology_rows=tuple(_attach_applicability(context.technology_rows, route_contexts)),
    )
