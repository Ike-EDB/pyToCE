"""Focused provider recipe compilation contracts."""

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    frontier_database as provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    native_resolution,
    recipe_compilation,
)


def _selector(name: str, product: str, unit: str) -> dict[str, object]:
    return {
        "name": name,
        "reference_product": product,
        "unit": unit,
        "exact_location": "RER",
    }


def _process(
    database: str,
    code: str,
    name: str,
    product: str,
    *,
    exchanges: list[dict[str, object]] | None = None,
    production_amount: float = 1.0,
    comment: str | None = None,
    location: str = "RER",
) -> dict[str, object]:
    process = {
        "name": name,
        "reference product": product,
        "unit": "kilogram",
        "location": location,
        "type": "process",
        "exchanges": [
            {
                "input": (database, code),
                "amount": production_amount,
                "type": "production",
            },
            *(exchanges or []),
        ],
    }
    if comment is not None:
        process["comment"] = comment
    return process


def _catalogue() -> dict[str, object]:
    return {
        "schema_version": "input_normalized_operation_test_v1",
        "semantic_contract": {
            "explicit_cross_material_route_ids": [],
            "service_normalization_route_ids": [
                "mechanical_recycling_operation",
                "conversion_operation",
            ],
        },
        "source_authorities": [
            {
                "source_id": "bounded_native_recycling",
                "role": "bounded_recycling_operation_method",
                "direct_source_authority_ids": [],
            },
            {
                "source_id": "ecoinvent_3_12_cutoff",
                "role": "native_activity_authority",
                "direct_source_authority_ids": [],
            },
            {
                "source_id": "bounded_native_conversion",
                "role": "bounded_conversion_operation_method",
                "direct_source_authority_ids": [],
            },
            {
                "source_id": "bounded_native_materials",
                "role": "bounded source-normalized material method",
                "direct_source_authority_ids": [],
            },
            {
                "source_id": "premise_2_4_6",
                "role": "bounded matching-member technology authority",
                "direct_source_authority_ids": [],
            },
        ],
        "method_authority_registry": {
            "prefix_rules": [],
            "exact_source_bindings": {
                "bounded_native_recycling": ["bounded_native_recycling"],
                "bounded_native_conversion": ["bounded_native_conversion"],
                "bounded_native_materials": ["bounded_native_materials"],
            },
            "project_recipe_defaults": {
                "technology_boundary": "bounded native recycling operation",
                "allocation_boundary": "bounded native cut-off allocation",
                "geography_year_boundary": "bounded native location and year",
                "limitation": "bounded operation-only test authority",
            },
        },
        "services": [
            {
                "provider_service_id": "mechanical_recycling_operation",
                "name": "mechanical recycling operation",
                "reference_product": "mechanical recycling operation",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "mechanical_recycling_operation",
                "foreground_service": True,
                "citation": "bounded_native_recycling",
                "construction": {
                    "kind": "input_normalized_operation",
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "native recycling",
                            "recycled polymer",
                            "kilogram",
                        ),
                    },
                    "denominator_selector": _selector(
                        "waste polymer supply",
                        "waste polymer",
                        "kilogram",
                    ),
                    "omit_technosphere": [
                        _selector(
                            "plastic reject treatment",
                            "plastic reject",
                            "kilogram",
                        )
                    ],
                },
            },
            {
                "provider_service_id": "conversion_operation",
                "name": "saleable conversion operation",
                "reference_product": "conversion operation",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "processing_conversion",
                "foreground_service": True,
                "citation": "bounded_native_conversion",
                "construction": {
                    "kind": "conversion_operation",
                    "ultimate_carbon_class": "fossil",
                    "waste_replacements": [],
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "native conversion",
                            "conversion operation",
                            "kilogram",
                        ),
                    },
                    "operation": "conversion_injection",
                },
            },
            {
                "provider_service_id": "scaled_operation_portfolio",
                "name": "scaled operation portfolio",
                "reference_product": "scaled operation portfolio",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "processing_conversion",
                "foreground_service": True,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {
                            "service_id": "operation_with_declared_omissions",
                            "amount": 0.25,
                        }
                    ],
                },
            },
            *[
                {
                    "provider_service_id": service_id,
                    "name": service_id,
                    "reference_product": product,
                    "unit": "kilogram",
                    "location": "RER",
                    "stage_owner": "virgin_resin_production",
                    "foreground_service": False,
                    "frontier_service": service_id
                    in {"alternative_mdi", "alternative_polyol", "alternative_amylose"},
                    "citation": "bounded_native_materials",
                    "construction": {
                        "kind": "native_activity",
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(name, product, "kilogram"),
                    },
                }
                for service_id, name, product in (
                    ("alternative_mdi", "alternative MDI", "alternative MDI"),
                    (
                        "alternative_polyol",
                        "alternative polyol",
                        "alternative polyol",
                    ),
                    ("alternative_epoxy", "alternative epoxy", "alternative epoxy"),
                    (
                        "alternative_polyester",
                        "alternative polyester",
                        "alternative polyester",
                    ),
                    (
                        "alternative_amylose",
                        "alternative amylose",
                        "alternative amylose",
                    ),
                )
            ],
            {
                "provider_service_id": "complete_adhesive",
                "name": "complete adhesive",
                "reference_product": "complete adhesive",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "operation_only",
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "native adhesive",
                            "native adhesive",
                            "kilogram",
                        ),
                    },
                    "replace_technosphere": [
                        {
                            "replacement_id": "mdi",
                            "selector": _selector(
                                "native MDI",
                                "native MDI",
                                "kilogram",
                            ),
                        },
                        {
                            "replacement_id": "polyol",
                            "selector": _selector(
                                "native polyol",
                                "native polyol",
                                "kilogram",
                            ),
                        },
                    ],
                    "replacements": [
                        {"replacement_id": "mdi", "service_id": "alternative_mdi"},
                        {
                            "replacement_id": "polyol",
                            "service_id": "alternative_polyol",
                        },
                    ],
                    "omit_biosphere": [
                        {
                            "selector": {
                                "name": "Process emission",
                                "unit": "kilogram",
                                "categories": ["air"],
                            },
                            "reason": "the source operation does not authorize this direct flow",
                        }
                    ],
                },
            },
            {
                "provider_service_id": "operation_with_declared_omissions",
                "frontier_service": True,
                "name": "operation with declared boundary omissions",
                "reference_product": "operation with declared boundary omissions",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "processing_conversion",
                "foreground_service": False,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "operation_only",
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "native fibre operation",
                            "finished fibre",
                            "kilogram",
                        ),
                    },
                    "omit_technosphere": [
                        {
                            "omission_id": "principal_polymer",
                            "selector": _selector(
                                "principal polymer supply",
                                "principal polymer",
                                "kilogram",
                            ),
                            "reason": "principal polymer is owned by the resin stage",
                        },
                        {
                            "omission_id": "conversion_scrap",
                            "selector": _selector(
                                "conversion polymer scrap",
                                "waste polymer",
                                "kilogram",
                            ),
                            "reason": "conversion loss is owned by the process model",
                        },
                    ],
                    "omission_balance_groups": [
                        {
                            "group_id": "net_polymer_boundary",
                            "omission_ids": [
                                "principal_polymer",
                                "conversion_scrap",
                            ],
                            "reason": (
                                "The omitted polymer input and internal scrap must "
                                "equal the native reference production."
                            ),
                        }
                    ],
                },
            },
            {
                "provider_service_id": "coating_blend",
                "name": "dry coating blend",
                "reference_product": "dry coating blend",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {"service_id": "alternative_epoxy", "amount": 0.5},
                        {"service_id": "alternative_polyester", "amount": 0.5},
                    ],
                },
            },
            {
                "provider_service_id": "selected_glucose",
                "name": "glucose with selected-origin starch",
                "reference_product": "glucose",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "operation_only",
                    "base": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": _selector(
                            "native glucose production",
                            "glucose",
                            "kilogram",
                        ),
                    },
                    "replace_technosphere": [
                        {
                            "replacement_id": "starch_feed",
                            "selector": _selector(
                                "market for native starch",
                                "native starch",
                                "kilogram",
                            ),
                        }
                    ],
                    "replacements": [
                        {
                            "replacement_id": "starch_feed",
                            "service_id": "alternative_amylose",
                            "multiplier": 1.0050251256281406,
                        }
                    ],
                },
            },
            {
                "provider_service_id": "source_dependent_portfolio",
                "name": "source-dependent portfolio",
                "reference_product": "source-dependent portfolio",
                "unit": "kilogram",
                "location": "RER",
                "stage_owner": "virgin_resin_production",
                "foreground_service": True,
                "citation": "bounded_native_materials",
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {"service_id": "complete_adhesive", "amount": 0.2},
                        {"service_id": "coating_blend", "amount": 0.3},
                        {"service_id": "selected_glucose", "amount": 0.5},
                    ],
                },
            },
        ],
        "routes": [
            {
                "route_id": "mechanical_recycling_operation",
                "route_component_id": "mechanical_recycling_operation",
                "provider_demand_type": "mechanical_recycling_operation",
                "region": "all",
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "polymer",
                "good": "all",
                "mapping_basis": "kg_mechanical_recycling_input",
                "valid_from_year": 2019,
                "valid_to_year": 2100,
                "component_role": "mechanical_recycling_operation",
                "component_material": "HDPE",
                "virgin_feedstock_route": "all",
                "ultimate_carbon_origin": "all",
                "share": 1.0,
                "throughput_multiplier": 1.0,
                "provider_service_id": "mechanical_recycling_operation",
            },
            {
                "route_id": "conversion_operation",
                "route_component_id": "conversion_operation",
                "provider_demand_type": "processing_conversion",
                "region": "all",
                "material": "HDPE",
                "constituent": "all",
                "product": "polymer",
                "good": "all",
                "mapping_basis": "kg_processed_plastic",
                "valid_from_year": 2019,
                "valid_to_year": 2100,
                "component_role": "processing_conversion",
                "component_material": "HDPE",
                "virgin_feedstock_route": "all",
                "ultimate_carbon_origin": "all",
                "share": 1.0,
                "throughput_multiplier": 1.0,
                "provider_service_id": "conversion_operation",
            },
        ],
    }


def _write_bounded_frontier(
    bd: Any,
    catalogue: dict[str, object],
) -> tuple[
    str,
    recipe_compilation._RecipeContext,
    tuple[dict[str, object], ...],
]:
    """Run the recipe owners directly against the bounded native databases."""

    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], catalogue["services"])
    }
    context = recipe_compilation._RecipeContext(
        background_scenario="remind_ssp2_npi",
        year=2040,
        provider_database=provider_catalogue.provider_frontier_database_name(
            "remind_ssp2_npi",
            2040,
            interface_database=provider_catalogue.PROVIDER_INTERFACE_DATABASE,
        ),
        source=native_resolution._ActivityResolver(bd, "ordinary"),
        ordinary=native_resolution._ActivityResolver(bd, "ordinary"),
        technology=native_resolution._ActivityResolver(bd, "technology"),
        biosphere=native_resolution._FlowResolver(bd, "biosphere"),
        material_recipe=physical_material_recipe(),
        catalogue=catalogue,
        method_authorities={
            str(row["method_authority_id"]): row
            for row in provider_catalogue._method_authority_rows(catalogue)
        },
        complete_source_ids=provider_catalogue._service_source_closures(catalogue),
        semantic_rows=[],
        technology_rows=[],
        traced_services=set(),
    )
    database_name = provider_catalogue.provider_frontier_database_name(
        context.background_scenario,
        context.year,
        interface_database=provider_catalogue.PROVIDER_INTERFACE_DATABASE,
    )
    bd.Database(database_name).write(
        {
            (
                database_name,
                provider_catalogue.provider_service_code(service_id),
            ): recipe_compilation._service_dataset(
                context,
                service,
                services,
                database_name,
            )
            for service_id, service in services.items()
            if service.get("foreground_service", False) or service.get("frontier_service", False)
        },
        searchable=False,
    )
    route_contexts = recipe_compilation._route_contexts(catalogue["routes"], services)
    semantic_rows = provider_frontier._attach_applicability(
        [
            *context.semantic_rows,
            *recipe_compilation._route_semantic_rows(
                catalogue,
                services,
                routes=catalogue["routes"],
                background_scenario=context.background_scenario,
                year=context.year,
            ),
        ],
        route_contexts,
    )
    return database_name, context, tuple(semantic_rows)


def _prepare_selected_native_frontier(
    bd: Any,
    catalogue: dict[str, object],
    selected: tuple[str, ...],
    *,
    year: int = 2040,
) -> provider_frontier.ProviderFrontier:
    """Compile the selected real recipe graphs against the fixture native store."""
    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], catalogue["services"])
    }
    required: set[str] = set()
    pending = list(selected)
    while pending:
        service_id = pending.pop()
        if service_id in required:
            continue
        required.add(service_id)
        pending.extend(provider_catalogue._service_dependencies(services[service_id]))
    bounded = {
        **catalogue,
        "services": [service for key, service in services.items() if key in required],
        "routes": [],
    }
    provider = provider_catalogue.PreparedProviderAuthority(
        identity="bounded_native_recipe_graph",
        interface_database=provider_catalogue.PROVIDER_INTERFACE_DATABASE,
        catalogue=bounded,
        routes=pd.DataFrame(),
        method_authorities=provider_catalogue._method_authority_rows(bounded),
    )
    return provider_frontier.prepare_provider_frontier(
        bd,
        background_scenario="remind_ssp2_npi",
        year=year,
        ordinary_database="native",
        technology_database="native",
        material_recipe=physical_material_recipe(),
        provider=provider,
        resolver_cache=native_resolution._NativeResolverCache("native", "biosphere"),
    )


def _assert_native_inventory(
    bd: Any, activity: Any, expected: dict[tuple[tuple[str, str], str], float]
) -> None:
    """Check all native supplies and elementary totals against independent equations.

    The longest declared fixture chain is PA6 through caprolactam, oxime,
    cyclohexanone, cyclohexane, benzene and methanol to a native supply; published
    matrix paths are shorter. Native demands are normalized by each activity's
    signed production amount, including nonunit references. Each total has
    same-sign contributions, and the duplicate MTO feed sum 2+0.5 is exact.
    Eight rounded factors have bound 8u/(1-8u), with u=eps32/2. Eight float32
    epsilons allow that error plus binary64 arithmetic in these fixtures only.
    Exact coefficient/metadata assertions remain with their construction owners.
    Divide signed native waste demand by its negative production when comparing
    activity supply.

    For an expected zero native supply, allow float64 epsilon times the solved
    supply vector's 1-norm. An inactive origin can have a tiny nonzero value
    from sparse-solve arithmetic. The tolerance applies at this fixture's scale;
    nonzero supplies use the relative check.
    """
    calculation = LCA({activity: 1.0})
    calculation.lci()
    native_expected, native_actual = {}, {}
    for node in bd.Database("native"):
        output = sum(float(exchange["amount"]) for exchange in node.production())
        native_expected[node.key] = expected.get((node.key, "technosphere"), 0.0) / output
        index = calculation.dicts.activity.get(node.id)
        native_actual[node.key] = 0.0 if index is None else float(calculation.supply_array[index])
    zero_tolerance = np.finfo(np.float64).eps * np.linalg.norm(calculation.supply_array, ord=1)
    for key, amount in native_expected.items():
        if amount == 0.0:
            assert abs(native_actual[key]) <= zero_tolerance
        else:
            assert native_actual[key] == pytest.approx(
                amount, rel=8 * np.finfo(np.float32).eps, abs=0
            )
    bio_expected, bio_actual = {}, {}
    for flow in bd.Database("biosphere"):
        bio_expected[flow.key] = expected.get((flow.key, "biosphere"), 0.0)
        index = calculation.dicts.biosphere.get(flow.id)
        bio_actual[flow.key] = (
            0.0 if index is None else float(calculation.inventory[index, :].sum())
        )
    assert bio_actual == pytest.approx(bio_expected, rel=8 * np.finfo(np.float32).eps, abs=0)


def _gas(code: str, amount: float, *, dated: bool) -> dict[str, object]:
    """Supply synthetic gas rows with original lognormal or dated deterministic metadata."""
    carbon = {"co2": 12 / 44, "ch4": 12 / 16, "co": 12 / 28}[code.split("_")[0]]
    return {
        "input": ("biosphere", code),
        "flow": code,
        "amount": amount,
        "type": "biosphere",
        "unit": "kilogram",
        "uncertainty type": 0 if dated else 2,
        "loc": amount if dated else math.log(amount) + 0.17,
        **({} if dated else {"scale": 0.3, "scale without pedigree": 0.2}),
        "pedigree": {"reliability": 2},
        "allocation": "invented native allocation",
        **(
            {}
            if dated
            else {
                "properties": {
                    "carbon allocation": {"amount": carbon, "unit": "kg"},
                    "carbon content": {"amount": carbon, "unit": "dimensionless"},
                    "carbon content, fossil": {"amount": carbon, "unit": "dimensionless"},
                    "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                }
            }
        ),
    }


def _landfill_sources(
    catalogue: dict[str, Any], *, dated: bool
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, tuple[float, float]]]:
    """Construct selected synthetic landfill chains with kilogram and cubic meter
    services and fossil auxiliaries."""
    nodes = {
        ("native", "utility"): _process(
            "native",
            "utility",
            "fixture utility",
            "utility",
            exchanges=[_gas("co2_high_f", 0.7, dated=dated)],
        ),
        ("native", "flocculant"): _process(
            "native",
            "flocculant",
            "market for polyacrylamide",
            "polyacrylamide",
            location="GLO",
            exchanges=[_gas("co2_high_f", 0.8, dated=dated)],
        ),
    }
    edges = {
        "landfill": [("leachate_market", -0.2)],
        "leachate_market": [("wastewater_treatment", -0.4)],
        "wastewater_treatment": [("sludge70_market", -0.3), ("sludge97_market", -0.6)],
        "sludge70_market": [("sludge_incineration", -1.0)],
        "sludge97_market": [("sludge_landfarming", -1.0)],
        "sludge_incineration": [],
        "sludge_landfarming": [],
    }
    fractions = {}
    chains = catalogue["semantic_contract"]["landfill_carbon_origin"]["chains"]
    for index, (parent, chain) in enumerate(chains.items(), 1):
        toc = index * 0.001 * (1.4 if dated else 1.0)
        flocculant = 0.002 + index * 0.0001
        # Independent source equations, expressed using the published transfer totals.
        influent = toc / 0.116237319347025875
        waste_sludge = influent * 0.658860928290208689
        auxiliary = flocculant * 36.03 / 71.08
        q = waste_sludge / (waste_sludge + auxiliary)
        gas = (waste_sludge + auxiliary) * 0.20525339975098886 * 0.603
        biological_co2 = influent * 0.224901752362765436 * 44 / 12
        gas_co2 = gas * (1 - 0.65 * 0.0075) * 44 / 12
        co2 = biological_co2 + gas_co2
        ch4 = gas * 0.65 * 0.0075 * 16 / 12
        fractions[parent] = q, (biological_co2 + q * gas_co2) / co2
        gas_rows = {
            "landfill": [("co2_high_f", 0.2), ("ch4_high_f", 0.005), ("co_high_f", 0.001)],
            "wastewater_treatment": [
                ("co2_high_f", co2 * 0.3),
                ("co2_urban_f", co2 * 0.7),
                ("ch4_high_f", ch4 * 0.3),
                ("ch4_urban_f", ch4 * 0.7),
            ],
            "sludge_incineration": [
                ("co2_urban_f", 0.05),
                ("ch4_urban_f", 0.0006),
                ("co_urban_f", 0.0007),
            ],
            "sludge_landfarming": [("co2_high_f", 0.04)],
        }
        for role, selector in chain["owners"].items():
            code = f"{parent}_{role}"
            exchanges = [
                {
                    "input": ("native", f"{parent}_{child}"),
                    "amount": amount,
                    "type": "technosphere",
                    "uncertainty type": 0,
                    "loc": amount,
                }
                for child, amount in edges[role]
            ]
            exchanges.append(
                {"input": ("native", "utility"), "amount": 0.1, "type": "technosphere"}
            )
            exchanges.extend(
                _gas(flow, amount, dated=dated) for flow, amount in gas_rows.get(role, [])
            )
            if role == "wastewater_treatment":
                exchanges.extend(
                    [
                        {
                            "input": ("native", "flocculant"),
                            "amount": flocculant,
                            "type": "technosphere",
                        },
                        {"input": ("biosphere", "toc"), "amount": toc, "type": "biosphere"},
                    ]
                )
            nodes[("native", code)] = {
                **_process(
                    "native",
                    code,
                    selector["name"],
                    selector["reference_product"],
                    location=selector["exact_location"],
                    production_amount=-1.0,
                    exchanges=exchanges,
                ),
                "unit": selector["unit"],
            }
    return nodes, fractions
