"""Provider catalog, route, and source-authority contracts."""

import json
from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _route_semantic_class,
)
from pytoce.run_mfa.model.materials import load_material_composition


def test_constituent_routes_have_complete_executable_services() -> None:
    """Supply every constituent through its complete provider graph."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    production = [
        route
        for route in catalogue["routes"]
        if route["provider_demand_type"]
        in {
            "virgin_resin_production",
            "polymerization_operation",
            "virgin_monomer_supply",
            "recovered_monomer_supply",
        }
    ]
    expected = set()
    for item in load_material_composition().constituents:
        prefix = (item.material, item.constituent)
        if item.chemical_polymer is None:
            expected.update(
                (*prefix, "virgin_resin_production", "polymer", origin, "all")
                for origin in ("fossil", "biogenic", "atmospheric_dac")
            )
        else:
            expected.add((*prefix, "polymerization_operation", "polymer", "all", "all"))
            for monomer in ("ethylene", "propylene"):
                for origin in ("fossil", "biogenic", "atmospheric_dac"):
                    expected.add((*prefix, "virgin_monomer_supply", monomer, origin, "all"))
                    expected.add((*prefix, "recovered_monomer_supply", monomer, "all", origin))
    observed = {}
    for route in production:
        key = tuple(
            route[column]
            for column in (
                "material",
                "constituent",
                "provider_demand_type",
                "product",
                "virgin_feedstock_route",
                "ultimate_carbon_origin",
            )
        )
        observed[key] = observed.get(key, 0.0) + route["share"]
        assert services[route["provider_service_id"]]["foreground_service"]
        assert route["region"] == route["good"] == "all"
        if route["material"] in {"BioPlastics", "Fibres"}:
            family = "bioplastics" if route["material"] == "BioPlastics" else "fibre"
            assert route["method_authority_id"] == f"fixed_{family}_constituent_provider_mapping"
            assert route["selection_reason"] and route["limitation"]
            if route["constituent"] in {"other_bio_based", "other_biodegradable"}:
                assert (
                    _route_semantic_class(catalogue, route, services[route["provider_service_id"]])
                    == "explicit_cross_material_analogue"
                )
    assert set(observed) == expected
    assert all(share == pytest.approx(1.0) for share in observed.values())

    reachable: set[str] = set()
    pending = [route["provider_service_id"] for route in catalogue["routes"]]
    while pending:
        service_id = pending.pop()
        if service_id in reachable:
            continue
        reachable.add(service_id)
        pending.extend(provider_catalogue._service_dependencies(services[service_id]))
    # Every foreground/frontier root must be reached from a public route.
    frontier_roots = {
        service_id
        for service_id, service in services.items()
        if bool(service.get("foreground_service", False))
        or bool(service.get("frontier_service", False))
    }
    assert frontier_roots <= reachable

    treatment_stages: dict[str, set[str]] = {}
    for route in catalogue["routes"]:
        if route["provider_demand_type"] in {
            "incineration_treatment",
            "landfill_treatment",
            "mismanaged_open_burning_treatment",
            "mismanaged_open_dump_treatment",
        }:
            treatment_stages.setdefault(route["provider_service_id"], set()).add(
                route["provider_demand_type"]
            )
    assert all(len(stages) == 1 for stages in treatment_stages.values())

    published = {
        key
        for key, service in services.items()
        if service.get("foreground_service") or service.get("frontier_service")
    }
    for key, service in services.items():
        if service["construction"]["kind"] == "operation_only":
            assert key in published
            assert {
                item["service_id"] for item in service["construction"].get("replacements", [])
            } <= published

    semantic_contract = catalogue["semantic_contract"]
    assert set(semantic_contract["classes"]) == {
        "unchanged_same_product",
        "premise_transformed_same_product",
        "operation_only_extraction",
        "source_backed_portfolio",
        "service_normalization",
        "explicit_cross_material_analogue",
    }
    route_ids = {route["route_id"] for route in catalogue["routes"]}
    overrides = set(semantic_contract["explicit_cross_material_route_ids"]) | set(
        semantic_contract["service_normalization_route_ids"]
    )
    assert overrides <= route_ids


def test_provider_sources_close_recursively() -> None:
    """Bind every project route and component to its complete source closure."""

    catalogue = provider_catalogue.load_provider_catalogue()
    source_ids = {str(row["source_id"]) for row in catalogue["source_authorities"]}
    abs_source = next(
        row for row in catalogue["source_authorities"] if row["source_id"] == "midden_2022_abs_san"
    )
    assert abs_source["retained_pdf_sha256"] == (
        "2ce5588211660749b8a788351d00308c94e1cb806e1a4c02187389de62ec8045"
    )
    method_rows = provider_catalogue._method_authority_rows(catalogue)
    assert all(
        set(row_ids) <= source_ids
        for row_ids in (json.loads(str(row["source_authority_ids_json"])) for row in method_rows)
    )

    closures = provider_catalogue._service_source_closures(catalogue)
    services = {str(service["provider_service_id"]): service for service in catalogue["services"]}
    for formulation, origin in (("native", "biogenic"), ("synthetic", "fossil")):
        recovery = f"mechanical_recycling_tps_{formulation}"
        assert set(closures[f"tps_{origin}"]) <= set(closures[recovery])
        assert f"tps_{origin}" not in provider_catalogue._service_dependencies(services[recovery])
    for service_id, service in services.items():
        closure = set(closures[service_id])
        assert closure
        assert closure <= source_ids
        direct_selector_sources = set(
            provider_catalogue._construction_source_authority_ids(service["construction"])
        )
        assert direct_selector_sources <= set(
            provider_catalogue._service_direct_source_ids(
                catalogue,
                service,
            )
        )
        assert direct_selector_sources <= closure
        for dependency in provider_catalogue._service_dependencies(service):
            assert set(closures[dependency]) <= closure

    retained_composition_sources = {
        "pur_biogenic": {
            "oecd_global_plastics_outlook_annex",
            "transience_product_service_database_v1",
            "wurzer_2016_polyurethane_applications",
            "plasticseurope_ecoprofiles",
            "kemona_piotrowska_2020_pur_emea",
            "liang_2021_us_polyurethane_mfa",
        },
        "abs_asa_san_biogenic": {
            "oecd_global_plastics_outlook_annex",
            "transience_product_service_database_v1",
            "plasticseurope_ecoprofiles",
            "levi_cullen_2018_global_chemical_flows",
        },
        "tyre_sbr_biogenic": {
            "oecd_global_plastics_outlook_annex",
            "unep_ryberg_plastics_value_chain",
            "transience_product_service_database_v1",
            "etrma_2022_natural_rubber_supply_chain",
        },
    }
    for service_id, expected_sources in retained_composition_sources.items():
        assert expected_sources <= set(closures[service_id])


def test_abs_and_san_use_one_continuous_mass_boundary_for_every_carbon_origin() -> None:
    """Close product, purge, oxidation, and utilities across carbon origins."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    exact_sources = set(
        catalogue["method_authority_registry"]["exact_source_bindings"][
            "abs_san_continuous_mass_selected_origin"
        ]
    )
    assert {
        "midden_2022_abs_san",
        "abs_continuous_mass_patents",
        "versalis_abs_continuous_mass",
        "epa_thermal_oxidizer_98_percent",
        "epa_recuperative_incinerator_efficiency_range",
        "epa_hazardous_waste_fuel_nitrogen",
        "zhang_2026_acrylonitrile_oxidation",
    } <= exact_sources

    abs_operation = services["abs_continuous_mass_operation"]["construction"]["components"]
    san_operation = services["san_continuous_mass_operation"]["construction"]["components"]
    assert (
        services["abs_continuous_mass_operation"]["representative_method"]["utility_source"]
        == "MIDDEN 2022 section 3.7 and Table 16"
    )
    assert (
        services["san_continuous_mass_operation"]["representative_method"]["utility_source"]
        == "MIDDEN 2022 section 3.6 and Table 15"
    )
    for service_id in ("abs_continuous_mass_operation", "san_continuous_mass_operation"):
        method = services[service_id]["representative_method"]
        assert "The volatile purge is assumed to follow" in method["purge_composition_role"]
        assert "supplying the RER foreground operation" in method["inbound_transport"]
        assert "thermally initiated natural-grade recipe" in method["optional_project_additives"]
    assert {row["component_id"]: row["amount"] for row in abs_operation} == {
        "electricity": pytest.approx(0.277777777777778),
        "steam_heat": pytest.approx(1.5),
    }
    assert {row["component_id"]: row["amount"] for row in san_operation} == {
        "electricity": pytest.approx(0.245098039215686),
        "steam_heat": pytest.approx(1.470588235294118),
    }

    for origin, carbon_dioxide in (
        ("biogenic", "Carbon dioxide, non-fossil"),
        ("atmospheric_dac", "Carbon dioxide, non-fossil"),
        ("fossil", "Carbon dioxide, fossil"),
    ):
        abs = services[f"abs_{origin}"]
        san = services[f"san_{origin}"]
        abs_components = abs["construction"]["components"]
        san_components = san["construction"]["components"]
        abs_component_amounts = {
            row.get("service_id", row.get("component_id")): row["amount"] for row in abs_components
        }
        san_component_amounts = {
            row.get("service_id", row.get("component_id")): row["amount"] for row in san_components
        }
        styrene_service = f"styrene_delivered_{origin}"
        acrylonitrile_service = f"acrylonitrile_delivered_{origin}"
        pbr_service = "polybutadiene" if origin == "fossil" else f"polybutadiene_delivered_{origin}"
        assert "Table 16" in services[f"abs_{origin}"]["representative_method"]["source"]
        assert abs_component_amounts == {
            styrene_service: pytest.approx(0.512755102040816),
            acrylonitrile_service: pytest.approx(0.30765306122449),
            pbr_service: pytest.approx(0.2),
            "abs_continuous_mass_operation": pytest.approx(1.0),
        }
        assert san_component_amounts == {
            styrene_service: pytest.approx(0.789473684210526),
            acrylonitrile_service: pytest.approx(0.263157894736842),
            "san_continuous_mass_operation": pytest.approx(1.0),
        }
        abs_direct = {
            term["selector"]["name"]: term["amount"]
            for term in abs["construction"]["biosphere_terms"]
        }
        san_direct = {
            term["selector"]["name"]: term["amount"]
            for term in san["construction"]["biosphere_terms"]
        }
        assert abs_direct == {
            carbon_dioxide: pytest.approx(0.0609151220106445),
            "Nitrogen oxides": pytest.approx(0.000650228968792402),
            "Styrene": pytest.approx(0.000255102040816327),
            "Acrylonitrile": pytest.approx(0.000153061224489796),
        }
        assert san_direct == {
            carbon_dioxide: pytest.approx(0.162849855607452),
            "Nitrogen oxides": pytest.approx(0.0011179375252922),
            "Styrene": pytest.approx(0.000789473684210526),
            "Acrylonitrile": pytest.approx(0.000263157894736842),
        }
        assert (
            abs_component_amounts[styrene_service]
            + abs_component_amounts[acrylonitrile_service]
            + abs_component_amounts[pbr_service]
        ) == pytest.approx(1 / 0.98)
        assert (
            san_component_amounts[styrene_service] + san_component_amounts[acrylonitrile_service]
        ) == pytest.approx(1 / 0.95)


def test_core_material_routes_bind_their_constituent_treatment_and_conversion() -> None:
    """Treat and convert each constituent through its declared providers."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    composition = load_material_composition()
    constituents = {(item.material, item.constituent) for item in composition.constituents}
    actual_pa = {("Fibres", "polyamide_6_resin"), ("Fibres", "polyamide_66_resin")}
    # Coated PA recycling and disposal use actual receiving events. Its fixed
    # resin and finished-principal conversion services remain in this catalog.
    mechanical_constituents = {
        (item.material, item.constituent)
        for item in composition.constituents
        if item.mechanical_recovery is not None
    } - actual_pa
    materials = {material for material, _ in constituents}
    contracts = {
        "mechanical_recycling_operation": "kg_mechanical_recycling_input",
        "processing_conversion": "kg_processed_plastic",
    }
    for demand_type, mapping_basis in contracts.items():
        selected = [
            route for route in catalogue["routes"] if route["provider_demand_type"] == demand_type
        ]
        coordinates = (
            mechanical_constituents
            if demand_type == "mechanical_recycling_operation"
            else constituents
        )
        assert {(route["material"], route["constituent"]) for route in selected} == coordinates
        assert all(route["mapping_basis"] == mapping_basis for route in selected)
        assert all(route["product"] == "polymer" for route in selected)
        assert all(route["region"] == route["good"] == "all" for route in selected)
        assert all(
            (route["valid_from_year"], route["valid_to_year"]) == (2019, 2100) for route in selected
        )
        for material, constituent in coordinates:
            for production_route in ("fossil", "biogenic", "atmospheric_dac"):
                for origin in ("fossil", "biogenic", "atmospheric_dac"):
                    assert fsum(
                        route["share"]
                        for route in selected
                        if (route["material"], route["constituent"]) == (material, constituent)
                        and route["virgin_feedstock_route"] in {"all", production_route}
                        and route["ultimate_carbon_origin"] in {"all", origin}
                    ) == pytest.approx(1.0, rel=0, abs=3e-16)

    # LBNL 2019 output pools select complete forming functions, independently
    # of the GPPS/HIPS resin analog. Check actual routes for every origin.
    ps_forming = {
        "conversion_foaming": 9.5 / 20.6,
        "conversion_injection": 9.9 / 20.6,
        "conversion_film": 1.2 / 20.6,
    }
    for origin in ("fossil", "biogenic", "atmospheric_dac"):
        selected = [
            route
            for route in catalogue["routes"]
            if route["material"] == "PS"
            and route["provider_demand_type"] == "processing_conversion"
            and route["ultimate_carbon_origin"] in {"all", origin}
        ]
        assert {
            services[row["provider_service_id"]]["construction"]["operation"]: row["share"]
            for row in selected
        } == pytest.approx(ps_forming)
        assert len(selected) == len(ps_forming)

    recovery_routes = {
        (row["material"], row["constituent"]): row["provider_service_id"]
        for row in catalogue["routes"]
        if row["provider_demand_type"] == "mechanical_recycling_operation"
    }
    for owner, coordinates in {
        "mechanical_recycling_biopolyester": {
            ("BioPlastics", name) for name in ("pla", "pbat", "pbs", "other_bio_based")
        },
        "mechanical_recycling_pa": {
            ("BioPlastics", "bio_pa"),
            ("Others", "pa"),
        },
        "mechanical_recycling_engineering": {
            ("ABS_ASA_SAN", "abs_asa_san"),
            ("Others", "other_thermoplastics"),
            ("Others", "pc"),
        },
        "mechanical_recycling_pp": {("PP", "pp"), ("Fibres", "polypropylene")},
        "mechanical_recycling_hdpe": {("HDPE", "hdpe"), ("BioPlastics", "bio_pe")},
    }.items():
        assert {key for key, service in recovery_routes.items() if service == owner} == coordinates

    eligible = {
        (item.material, item.constituent)
        for item in load_material_composition().constituents
        if item.chemical_polymer is not None
    }
    for demand_type in ("chemical_recycling_preparation", "chemical_recycling_conversion"):
        chemical_routes = [
            route for route in catalogue["routes"] if route["provider_demand_type"] == demand_type
        ]
        assert {
            (route["material"], route["constituent"], route["ultimate_carbon_origin"])
            for route in chemical_routes
        } == {
            (material, constituent, origin)
            for material, constituent in eligible
            for origin in ("fossil", "biogenic", "atmospheric_dac")
        }
        assert all(
            route["mapping_basis"] == "kg_tracked_plastic_chemical_recycling_input"
            for route in chemical_routes
        )
        assert all(route["share"] == 1.0 for route in chemical_routes)
        assert all(route["provider_service_id"] in services for route in chemical_routes)
        assert all(
            route["resolution_status"] == "physical_constituent_feed_and_monomer_yields"
            for route in chemical_routes
        )

    cured_resins = {
        "general_coatings": "epoxy",
        "marine_coatings": "epoxy",
        "thermosets": "upr",
        "road_marking_coatings": "road",
    }
    for demand_type in {"incineration_treatment", "landfill_treatment"}:
        selected = [
            route for route in catalogue["routes"] if route["provider_demand_type"] == demand_type
        ]
        assert {route["material"] for route in selected} == materials
        compound = [
            route
            for route in selected
            if services[route["provider_service_id"]]["construction"]["kind"]
            in {"tps_residue_treatment", "tps_native_treatment"}
        ]
        assert {(route["constituent"], route["virgin_feedstock_route"]) for route in compound} == {
            (constituent, origin)
            for constituent in ("starch_blends", "other_biodegradable")
            for origin in ("fossil", "biogenic", "atmospheric_dac")
        }
        assert all(
            route["ultimate_carbon_origin"] == "all" and route["share"] == 1 for route in compound
        )
        assert {(route["material"], route["constituent"]) for route in selected} == (
            constituents - actual_pa
        )
        # TPS disposal selects its historical formulation. Other treatments
        # retain the principal carbon origin, including cured epoxy and UPR.
        selected = [route for route in selected if route not in compound]
        assert {route["ultimate_carbon_origin"] for route in selected} == {
            "fossil",
            "biogenic",
            "atmospheric_dac",
        }
        assert not any(route["ultimate_carbon_origin"] == "all" for route in selected)
        native_constituents = {(route["material"], route["constituent"]) for route in selected}
        for material, constituent in native_constituents:
            for origin in {"fossil", "biogenic", "atmospheric_dac"}:
                assert sum(
                    route["share"]
                    for route in selected
                    if (route["material"], route["constituent"]) == (material, constituent)
                    and route["ultimate_carbon_origin"] == origin
                ) == pytest.approx(1.0)
        for route in selected:
            construction = services[route["provider_service_id"]]["construction"]
            if route["constituent"] in cured_resins:
                # Resin, hardener, and mineral mixtures use their composition-specific receiving
                # functions.
                resin = cured_resins[route["constituent"]]
                treatment = demand_type.removesuffix("_treatment")
                assert construction["kind"] == f"{resin}_{treatment}"
                assert construction["principal_carbon_origin"] == route["ultimate_carbon_origin"]
                continue
            if route["constituent"] == "pha" and demand_type == "incineration_treatment":
                assert construction["kind"] == "phb_incineration"
                assert construction["principal_origin"] == route["ultimate_carbon_origin"]
                assert construction["source"] == {
                    "name": "treatment of waste plastic, mixture, municipal incineration",
                    "reference_product": "waste plastic, mixture",
                    "unit": "kilogram",
                    "exact_location": "RoW",
                }
                continue
            if route["constituent"] == "non_tyre_elastomers":
                # The retired non-tire compound carries filler and plasticizer;
                # its principal kg therefore purchases the whole filled waste.
                origin = route["ultimate_carbon_origin"]
                assert construction["principal_carbon_origin"] == origin
                if demand_type == "landfill_treatment":
                    assert construction["kind"] == "sbr_compound_landfill"
                    assert construction["treatment"] == {
                        "service_id": f"sbr_filled_{origin}_landfill"
                    }
                else:
                    assert construction["kind"] == "sbr_incineration"
                continue
            if route["ultimate_carbon_origin"] == "fossil":
                assert construction["kind"] == "native_activity"
                assert construction["amount"] == -1.0
                assert construction["direct_treatment_carbon_owner"] == "native_activity"
                assert services[route["provider_service_id"]]["ultimate_carbon_origins"] == [
                    "fossil"
                ]
                mappings = construction["direct_treatment_carbon_mappings"]
            else:
                assert construction["kind"] == "treatment_origin_operation"
                assert services[route["provider_service_id"]]["ultimate_carbon_origins"] == [
                    "biogenic",
                    "atmospheric_dac",
                ]
                mappings = construction["replace_biosphere"]
            assert construction["carbon_attribution_status"] == ("separable_treated_plastic_carbon")
            assert mappings
            assert all(
                mapping["carbon_attribution_status"] == "separable_treated_plastic_carbon"
                and mapping["mapping_status"] == "exact_non_fossil_counterpart"
                for mapping in mappings
            )

    for demand in ("incineration_treatment", "landfill_treatment"):
        for constituent, expected in (
            ("polypropylene", False),
            ("pbat", True),
        ):
            selected = [
                route
                for route in catalogue["routes"]
                if route["provider_demand_type"] == demand and route["constituent"] == constituent
            ]
            assert {route["ultimate_carbon_origin"] for route in selected} == {
                "fossil",
                "biogenic",
                "atmospheric_dac",
            }
            for route in selected:
                semantic = _route_semantic_class(
                    catalogue, route, services[route["provider_service_id"]]
                )
                if expected:
                    assert semantic == "explicit_cross_material_analogue"
                    assert route["analogue_status"] == "explicit_cross_material_analogue"
                else:
                    assert semantic == (
                        "unchanged_same_product"
                        if route["ultimate_carbon_origin"] == "fossil"
                        else "operation_only_extraction"
                    )
                    assert route["analogue_status"] == "direct_material_match"

    conversion_routes = [
        route
        for route in catalogue["routes"]
        if route["provider_demand_type"] == "processing_conversion"
    ]
    assert {route["throughput_multiplier"] for route in conversion_routes} == {1.0}
    for service_id in {
        "conversion_injection",
        "conversion_film",
        "conversion_pipe",
        "conversion_blow",
        "conversion_calendering",
        "conversion_foaming",
    }:
        construction = services[service_id]["construction"]
        assert construction["kind"] == "conversion_operation"
        assert construction["operation"] == service_id
        assert construction["base"]["source_role"] == "native_ecoinvent_provider"

    public_catalogue = str(catalogue)
    assert "sc01_" not in public_catalogue
    assert ".agent/" not in public_catalogue


def test_methanol_to_olefin_services_use_product_specific_native_operations() -> None:
    """Bind alternative methanol to the dedicated native olefin operations."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}

    propylene_selector = {
        "name": "propylene production, from methanol-to-propylene conversion",
        "reference_product": "propylene",
        "unit": "kilogram",
        "exact_location": "CN",
    }
    for service_id in ("propylene_biogenic", "propylene_atmospheric_dac"):
        service = services[service_id]
        assert service["citation"] == (
            "ecoinvent_3_12_methanol_to_propylene_feedstock_substitution"
        )
        assert service["construction"]["base"]["selector"] == propylene_selector
        assert "replace_biosphere" not in service["construction"]

    ethylene_selector = {
        "name": (
            "light olefins production, from methanol-to-olefins conversion, "
            "ethylene-to-propylene ratio of 1.51"
        ),
        "reference_product": "ethylene",
        "unit": "kilogram",
        "exact_location": "CN",
    }
    expected_origin_replacements = [
        (
            {
                "name": "Carbon dioxide, fossil",
                "unit": "kilogram",
                "categories": ["air"],
            },
            {
                "name": "Carbon dioxide, non-fossil",
                "unit": "kilogram",
                "categories": ["air"],
            },
        ),
        (
            {
                "name": "Carbon monoxide, fossil",
                "unit": "kilogram",
                "categories": ["air"],
            },
            {
                "name": "Carbon monoxide, non-fossil",
                "unit": "kilogram",
                "categories": ["air"],
            },
        ),
    ]
    for service_id in ("ethylene_biogenic", "ethylene_atmospheric_dac"):
        service = services[service_id]
        assert service["citation"] == (
            "ecoinvent_3_12_ethylene_rich_methanol_to_olefins_feedstock_substitution"
        )
        construction = service["construction"]
        assert construction["base"]["selector"] == ethylene_selector
        assert [
            (rule["selector"], rule["replacement_selector"])
            for rule in construction["replace_biosphere"]
        ] == expected_origin_replacements


def test_acrylonitrile_services_preserve_native_operations() -> None:
    """Change the propylene feed and its evidenced direct-carbon loss only."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    for origin in ("biogenic", "atmospheric_dac"):
        acrylonitrile = services[f"acrylonitrile_{origin}"]["construction"]
        assert acrylonitrile["replacements"] == [
            {
                "replacement_id": "propylene",
                "service_id": f"propylene_{origin}",
            }
        ]
        assert [
            (rule["selector"], rule["replacement_selector"])
            for rule in acrylonitrile["replace_biosphere"]
        ] == [
            (
                {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ["air", "urban air close to ground"],
                },
                {
                    "name": "Carbon dioxide, non-fossil",
                    "unit": "kilogram",
                    "categories": ["air", "urban air close to ground"],
                },
            )
        ]


def test_pvc_pet_ps_and_sbr_routes_bind_complete_precursor_boundaries() -> None:
    """Preserve final native operations and every named incorporated precursor."""

    catalogue = provider_catalogue.load_provider_catalogue()
    services = {service["provider_service_id"]: service for service in catalogue["services"]}
    pvc_citation = "ecoinvent_3_12_complete_vcm_pvc_operation_selected_origin_ethylene_backbone"
    vcm_citation = "medrano_garcia_2023_vcm_native_linked"
    assert catalogue["method_authority_registry"]["exact_source_bindings"][pvc_citation] == [
        "ecoinvent_3_12_cutoff",
        "ecvm_plasticseurope_2026_vcm_pvc",
        "uba_ikp_2000_pvc_polymerisation",
        "aguilar_vasquez_2023_pvc",
    ]
    assert catalogue["method_authority_registry"]["exact_source_bindings"][vcm_citation] == [
        "medrano_garcia_2023_vcm",
        "ecoinvent_3_12_cutoff",
        "premise_2_4_6",
    ]

    for origin in ("biogenic", "atmospheric_dac"):
        pvc = services[f"pvc_{origin}"]["construction"]
        assert services[f"pvc_{origin}"]["citation"] == pvc_citation
        assert pvc == {
            "kind": "project_recipe",
            "components": [
                {"service_id": f"pvc_emulsion_{origin}", "amount": 0.1},
                {"service_id": f"pvc_suspension_{origin}", "amount": 0.9},
            ],
        }
        vinyl_chloride = services[f"vinyl_chloride_{origin}"]
        assert vinyl_chloride["citation"] == vcm_citation
        assert vinyl_chloride["location"] == "GLO"
        assert vinyl_chloride["representative_method"]["allocation"].startswith(
            "VCM receives all reported net material, utility, waste and direct-emission burdens."
        )
        assert (
            "Recovered HCl and chlorine remain internal"
            in vinyl_chloride["representative_method"]["allocation"]
        )
        assert vinyl_chloride["representative_method"]["native_service_geography"].startswith(
            "European-priority"
        )
        assert (
            "VCM, EDC, and TCE"
            in vinyl_chloride["representative_method"]["direct_carbon_dioxide_custody"]
        )
        assert (
            "none of its emissions are relabelled"
            in vinyl_chloride["representative_method"]["direct_carbon_dioxide_custody"]
        )
        reaction_balance = vinyl_chloride["representative_method"]["reaction_balance"]
        assert reaction_balance == {
            "feed_kg": 1.159,
            "vcm_kg": 1.0,
            "direct_carbon_dioxide_kg": 0.0178,
            "listed_flow_mass_residual_kg": 0.1412,
            "source_reported_table_s9_water_output_m3": 0.00441,
            "hydrogen_implied_water_kg": 0.14940422852208332,
            "oxygen_implied_water_kg": 0.13743827126252586,
            "additional_water_exchange_required": False,
            "water_formation_evidence": (
                "The reaction forms water. The inventory uses the reported net exchanges "
                "with internal HCl and chlorine recovery."
            ),
            "residual_interpretation": (
                "Reported net exchanges, including the aggregate water output, are retained."
            ),
            "elemental_input_minus_output_residual_kg": {
                "carbon": -0.0013410394594294273,
                "hydrogen": 0.01671934081046461,
                "chlorine": 0.0037636968766000534,
                "oxygen": 0.12205800177236475,
            },
        }
        vcm = vinyl_chloride["construction"]
        assert vcm["kind"] == "project_recipe"
        components = {
            component.get("component_id", component.get("service_id")): component
            for component in vcm["components"]
        }
        assert components[f"ethylene_{origin}"]["amount"] == 0.453
        assert components["chlorine"]["amount"] == 0.571
        assert components["oxygen"]["amount"] == 0.135
        assert components["tap_water"]["amount"] == 0.384
        assert components["natural_gas_heat"]["amount"] == 2.38
        assert components["high_voltage_electricity"]["amount"] == 0.5189
        assert components["ambient_cooling_operation"]["amount"] == 4.11 + 2.15 * 1.83
        assert components["chlorine"]["selector"]["exact_location"] == "RER"
        assert components["oxygen"]["selector"]["exact_location"] == "RER"
        assert components["natural_gas_heat"]["selector"]["location_by_period"] == {
            "baseline": "Europe without Switzerland",
            "future": "EUR",
        }
        assert components["high_voltage_electricity"]["selector"]["location_by_period"] == {
            "baseline": "RER",
            "future": "EUR",
        }
        direct = {
            (term["selector"]["name"], tuple(term["selector"]["categories"])): term["amount"]
            for term in vcm["biosphere_terms"]
        }
        assert direct == {
            ("Water", ("water",)): 0.00441,
            ("Carbon dioxide, non-fossil", ("air",)): 0.0178,
        }

        for service_id in (f"pvc_emulsion_{origin}", f"pvc_suspension_{origin}"):
            assert services[service_id]["citation"] == pvc_citation
            assert services[service_id]["location"] == "CN"
            assert (
                services[service_id]["construction"]["base"]["selector"]["exact_location"] == "CN"
            )
        assert services[f"pvc_emulsion_{origin}"]["construction"]["replace_technosphere"][0][
            "selector"
        ]["preferred_locations"] == ["CN"]
        assert services[f"pvc_suspension_{origin}"]["construction"]["replace_technosphere"][0][
            "selector"
        ]["preferred_locations"] == ["CN"]
        assert services[f"pvc_{origin}"]["location"] == "CN"
        expected_origin = "biogenic" if origin == "biogenic" else "atmospheric-DAC"
        assert services[f"pvc_{origin}"]["reference_product"] == (
            f"PVC portfolio with {expected_origin}-carbon-origin VCM backbone"
        )
        method = services[f"pvc_{origin}"]["representative_method"]
        assert method["suspension_pvc_share"] == 0.9
        assert method["emulsion_pvc_share"] == 0.1
        assert "selected carbon origin applies only to the VCM backbone" in method["claim_boundary"]
        assert "backbone" in services[f"pvc_emulsion_{origin}"]["name"]
        assert "backbone" in services[f"pvc_suspension_{origin}"]["name"]

        pet = services[f"pet_{origin}"]["construction"]
        assert {rule["replacement_id"] for rule in pet["replace_technosphere"]} == {
            "ethylene_glycol",
            "diethylene_glycol",
            "purified_terephthalic_acid",
        }
        p_xylene = services[f"p_xylene_{origin}"]
        assert p_xylene["citation"] == "manalal_2026_mta_xylenes"
        p_xylene_components = {
            component.get("component_id", component.get("service_id")): component
            for component in p_xylene["construction"]["components"]
        }
        assert p_xylene_components[f"methanol_{origin}"]["amount"] == 2.85510182090549
        assert p_xylene_components["green_hydrogen"]["source_role"] == ("premise_only_provider")
        assert p_xylene_components["ambient_cooling_operation"]["amount"] == 4.523065282414024
        assert p_xylene_components["spent_zeolite_disposal"]["selector"] == {
            "name": "market for waste zeolite",
            "reference_product": "waste zeolite",
            "unit": "kilogram",
            "exact_location": "RoW",
        }
        assert p_xylene["construction"]["biosphere_terms"] == [
            {
                "selector": {
                    "name": "Carbon dioxide, non-fossil",
                    "unit": "kilogram",
                    "categories": ["air"],
                },
                "amount": 0.182708833924356,
            }
        ]

        for service_id in (f"emulsion_sbr_{origin}", f"solution_sbr_{origin}"):
            sbr = services[service_id]["construction"]
            assert {rule["replacement_id"] for rule in sbr["replace_technosphere"]} == {
                "styrene",
                "butadiene",
            }

        expected_origin_mappings = {
            f"ethyl_benzene_{origin}": {"Carbon dioxide, fossil"},
            f"styrene_{origin}": {
                "Carbon dioxide, fossil",
                "Carbon monoxide, fossil",
                "Methane, fossil",
            },
            f"polybutadiene_{origin}": {"Carbon dioxide, fossil"},
            f"emulsion_sbr_{origin}": {"Carbon dioxide, fossil"},
            f"solution_sbr_{origin}": {"Carbon dioxide, fossil"},
            f"gpps_{origin}": {"Carbon dioxide, fossil"},
            f"hips_{origin}": {"Carbon dioxide, fossil"},
        }
        for service_id, source_flows in expected_origin_mappings.items():
            rules = services[service_id]["construction"]["replace_biosphere"]
            assert {rule["selector"]["name"] for rule in rules} == source_flows
            assert {rule["replacement_selector"]["name"] for rule in rules} == {
                flow.replace("fossil", "non-fossil") for flow in source_flows
            }
            assert all(
                rule["selector"]["categories"] == ["air"]
                and rule["replacement_selector"]["categories"] == ["air"]
                for rule in rules
            )

        assert "with" in services[f"pvc_{origin}"]["name"]
        assert "with" in services[f"pet_{origin}"]["name"]
        assert "with" in services[f"ps_{origin}"]["name"]
        assert "with" in services[f"tyre_sbr_{origin}"]["name"]


def test_calendering_uses_pvc_waste_and_shares_the_existing_landfill_children() -> None:
    """Preserve actual PVC treatment functions, signed units and carbon compartments."""
    catalogue = provider_catalogue.load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    for suffix in ("", "__non_fossil"):
        construction = services[f"conversion_calendering{suffix}"]["construction"]
        assert construction["base"]["selector"] == {
            "name": "calendering, rigid sheets",
            "reference_product": "calendering, rigid sheets",
            "unit": "kilogram",
            "exact_location": "RER",
        }
        waste = construction["waste_replacements"]
        assert len(waste) == 1
        assert waste[0]["service_id"] == "pvc_waste_market_rer_non_fossil"
        assert waste[0]["selector"]["name"] == "market group for waste polyvinylchloride"
        assert waste[0]["selector"]["sign"] == "negative"

    landfill = services["pvc_waste_landfill_row_non_fossil"]["construction"]
    assert (
        landfill["treatment_replacements"]
        == services["landfill_pvc__non_fossil"]["construction"]["treatment_replacements"]
    )
    assert landfill["treatment_replacements"][0]["service_id"] == (
        "landfill_pvc__leachate_market__non_fossil"
    )
    # Internal native waste uses -1 kg; public landfill uses +1 kg. Both use the shared treatment
    # calculation.
    assert landfill["preserve_native_production"]
    for service in services.values():
        if not service["provider_service_id"].startswith("pvc_waste_"):
            continue
        construction = service["construction"]
        if construction["kind"] != "treatment_origin_operation":
            continue
        assert construction["preserve_native_production"]
        selector = construction["base"]["selector"]
        rules = construction["replace_biosphere"]
        gases = {rule["selector"]["name"] for rule in rules}
        expected = {"Carbon dioxide, fossil", "Methane, fossil"}
        if "open dump" not in selector["name"] and "unsanitary landfill" not in selector["name"]:
            expected.add("Carbon monoxide, fossil")
        assert gases == expected
        compartment = (
            "urban air close to ground"
            if "incineration" in selector["name"]
            else "non-urban air or from high stacks"
        )
        for rule in rules:
            assert rule["selector"]["categories"] == ["air", compartment]
            assert rule["replacement_selector"] == {
                **rule["selector"],
                "name": rule["selector"]["name"].replace(", fossil", ", non-fossil"),
            }
