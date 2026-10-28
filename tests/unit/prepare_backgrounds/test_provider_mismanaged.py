"""Mismanaged treatment preserves infiltration, constituent and regional fate shares."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from infiltration_fixtures import infiltration_source_data, resolved_infiltration_provider
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _route_semantic_class,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    mismanaged_fate_authorities,
    resolve_open_dump_routes,
)
from pytoce.run_mfa.model.materials import load_material_composition


def test_mismanaged_eol_provider_authority_is_complete_and_additive(tmp_path: Path) -> None:
    """Close each constituent treatment portfolio and every regional fate split."""
    provider = resolved_infiltration_provider(tmp_path)
    catalogue = provider.catalogue
    source = next(
        row
        for row in provider.source_authorities
        if row["source_id"] == "ecoinvent_3_12_row_open_dump_portfolio"
    )
    derivation = json.loads(cast(str, source["derivation_json"]))
    assert derivation["conditional_denominator"] == 17.0
    expected = {
        "hyperarid infiltration class (-250mm)": 0.0588235294117647,
        "dry infiltration class (100mm)": 0.117647058823529,
        "moist infiltration class (300mm)": 0.176470588235294,
        "wet infiltration class (500mm)": 0.235294117647059,
        "very wet infiltration class (1000mm)": 0.411764705882353,
    }
    assert derivation["class_shares"] == expected
    dumping = provider.routes.query("provider_demand_type == 'mismanaged_open_dump_treatment'")
    assert dumping.groupby("infiltration_class")["infiltration_share"].first().to_dict() == expected
    assert dumping["infiltration_source_id"].eq("ecoinvent_3_12_row_open_dump_portfolio").all()
    assert (dumping["infiltration_share"] * dumping["material_component_share"]).equals(
        dumping["share"]
    )
    services = cast(list[dict[str, Any]], catalogue["services"])
    routes = cast(list[dict[str, object]], provider.routes.to_dict(orient="records"))

    mismanaged_services = [
        service
        for service in services
        if str(service["provider_service_id"]).startswith("mismanaged_")
    ]
    mismanaged_routes = [
        route for route in routes if str(route["provider_demand_type"]).startswith("mismanaged_")
    ]
    service_by_id = {
        str(service["provider_service_id"]): service for service in mismanaged_services
    }
    base_service_ids = {
        service_id for service_id in service_by_id if not service_id.endswith("__non_fossil")
    }
    assert {
        service_id.removesuffix("__non_fossil")
        for service_id in service_by_id
        if service_id.endswith("__non_fossil")
    } == base_service_ids
    assert all(
        service_by_id[service_id]["construction"]["kind"] == "native_activity"
        and service_by_id[service_id]["construction"]["amount"] == -1.0
        and service_by_id[service_id]["construction"]["direct_treatment_carbon_owner"]
        == "native_activity"
        for service_id in base_service_ids
    )
    assert all(
        service_by_id[f"{service_id}__non_fossil"]["construction"]["kind"]
        == "treatment_origin_operation"
        for service_id in base_service_ids
    )
    route_shares: dict[tuple[str, str, str, str, str], float] = {}
    for route in mismanaged_routes:
        key = (
            str(route["provider_demand_type"]),
            str(route["material"]),
            str(route["constituent"]),
            str(route["virgin_feedstock_route"]),
            str(route["ultimate_carbon_origin"]),
        )
        route_shares[key] = route_shares.get(key, 0.0) + float(cast(float, route["share"]))
    np.testing.assert_allclose(list(route_shares.values()), 1.0, rtol=0.0, atol=3e-15)
    assert set(route_shares) == {
        (demand, item.material, item.constituent, formulation, origin)
        for item in load_material_composition().constituents
        if (item.material, item.constituent)
        not in {("Fibres", "polyamide_6_resin"), ("Fibres", "polyamide_66_resin")}
        for demand in ("mismanaged_open_burning_treatment", "mismanaged_open_dump_treatment")
        for formulation, origin in (
            # Both TPS fates retain the historical formulation; each service
            # contains mixed principal and auxiliary carbon origins.
            # Other current treatment routes select the native carbon class.
            tuple((formulation, "all") for formulation in ("fossil", "biogenic", "atmospheric_dac"))
            if item.mechanical_recovery == "tps_compound"
            else tuple(("all", origin) for origin in ("fossil", "biogenic", "atmospheric_dac"))
        )
    }

    for route in mismanaged_routes:
        if route["constituent"] not in {"polypropylene", "pbat"}:
            continue
        service = next(
            item for item in services if item["provider_service_id"] == route["provider_service_id"]
        )
        semantic = _route_semantic_class(catalogue, route, service)
        if route["constituent"] == "polypropylene":
            assert semantic == (
                "unchanged_same_product"
                if route["ultimate_carbon_origin"] == "fossil"
                else "operation_only_extraction"
            )
            assert route["analogue_status"] == "direct_material_match"
        elif route["provider_demand_type"] == "mismanaged_open_burning_treatment":
            assert service["construction"]["kind"] == "polyester_open_burning"
            assert semantic == "service_normalization"
            assert route["analogue_status"] == "actual_material_with_transferred_burning_function"
        else:
            assert semantic == route["analogue_status"] == "explicit_cross_material_analogue"

    fate = mismanaged_fate_authorities()
    np.testing.assert_allclose(
        fate["open_burning_share"] + fate["open_dump_share"],
        1.0,
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize("material", ("HDPE", "Elastomers (tyres)"))
@pytest.mark.parametrize("year", (2020, 2050))
def test_frontier_reports_resolved_class_shares(tmp_path: Path, year: int, material: str) -> None:
    """Keep the saved conditional shares and native source in both anchor periods."""
    names = BrightwayNames(tmp_path, "class_frontier", "biosphere", "native")
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    raw_routes = [
        row
        for row in cast(list[dict[str, Any]], catalogue["routes"])
        if row["provider_demand_type"] == "mismanaged_open_dump_treatment"
        and row["material"] == material
        and row["ultimate_carbon_origin"] == "fossil"
    ]
    service_ids = {row["provider_service_id"] for row in raw_routes}
    with selected_brightway_project(names) as bd:
        services = [
            row
            for row in cast(list[dict[str, Any]], catalogue["services"])
            if row["provider_service_id"] in service_ids
        ]
        rules = services[0]["construction"]["direct_treatment_carbon_mappings"]
        flows = {
            (names.biosphere_database, str(index)): {
                **rule["selector"],
                "type": "emission",
            }
            for index, rule in enumerate(rules)
        }
        bd.Database(names.biosphere_database).write(flows, searchable=False)
        native = infiltration_source_data(names.source_database)
        for service in services:
            key = (names.source_database, service["provider_service_id"])
            selector = service["construction"]["selector"]
            native[key] = {
                "name": selector["name"],
                "reference product": selector["reference_product"],
                "location": selector["exact_location"],
                "unit": selector["unit"],
                "type": "process",
                "exchanges": [
                    {"input": key, "type": "production", "amount": -1.0},
                    *({"input": flow, "type": "biosphere", "amount": 0.01} for flow in flows),
                ],
            }
        bd.Database(names.source_database).write(native, searchable=False)
        native_before = bd.Database(names.source_database).load()
        routes, _source = resolve_open_dump_routes(names.source_database, provider.routes)
        resolved = routes.loc[
            routes["route_component_id"].isin(row["route_component_id"] for row in raw_routes)
        ]
        selected = replace(
            provider,
            catalogue={
                **catalogue,
                "routes": raw_routes,
                "services": [
                    row
                    for row in cast(list[dict[str, Any]], catalogue["services"])
                    if row["provider_service_id"] in service_ids
                ],
            },
            routes=resolved,
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=names.source_database,
            technology_database=names.source_database,
            material_recipe=physical_material_recipe(),
            provider=selected,
            resolver_cache=_NativeResolverCache(names.source_database, names.biosphere_database),
        )
        assert bd.Database(names.source_database).load() == native_before
    rows = [row for row in frontier.semantic_rows if row["ledger_entry_type"] == "route_mapping"]
    assert len(rows) == (5 if material == "HDPE" else 10)
    assert {
        row["route_component_id"]: json.loads(cast(str, row["scaling_algebra"]))["route_share"]
        for row in rows
    } == resolved.set_index("route_component_id")["share"].to_dict()
    assert all(
        "ecoinvent_3_12_row_open_dump_portfolio"
        in json.loads(cast(str, row["complete_source_authority_ids_json"]))
        for row in rows
    )
    assert all(row["sbr_share_in_material"] == (0.0 if material == "HDPE" else 1.0) for row in rows)
    assert all(row["whole_tyre_constituents_excluded"] is (material != "HDPE") for row in rows)
