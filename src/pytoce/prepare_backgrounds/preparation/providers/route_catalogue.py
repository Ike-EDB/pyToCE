"""Scientific route additions for the shared provider catalog."""

import re
from copy import deepcopy
from typing import cast

from pytoce.run_mfa.model.materials import MaterialComposition, load_material_composition

from .route_authorities import (
    _DIRECT_TREATMENT_CARBON_SPECIES,
    OPEN_DUMP_PORTFOLIO_SOURCE_ID,
    _chemical_recycling_authority,
    _csv_rows,
    _direct_carbon_mappings,
    _hybrid_provider_temporal_authority,
    _sbr_treatment_authority,
)

_TREATMENT_DEMAND_TYPES = {
    "incineration_treatment",
    "landfill_treatment",
    "mismanaged_open_burning_treatment",
    "mismanaged_open_dump_treatment",
}
_TREATMENT_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_CHEMICAL_ORIGINS = {
    "fossil": "fossil",
    "biogenic": "non_fossil",
    "atmospheric_dac": "non_fossil",
}


def _slug(value: str) -> str:
    """Convert a catalog label to lowercase ``snake_case``."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _mismanaged_services(selector_rows: tuple[dict[str, str], ...]) -> list[dict[str, object]]:
    """Build native ecoinvent services for open burning and open dumping."""
    return [
        {
            "provider_service_id": row["provider_service_id"],
            "name": row["activity_name"],
            "reference_product": row["reference_product"],
            "unit": row["unit"],
            "location": row["location"],
            "stage_owner": row["stage_owner"],
            "foreground_service": True,
            "citation": "ecoinvent_3_12_native_mismanaged_eol_treatment",
            "construction": {
                "kind": row["construction_kind"],
                "source_role": "native_ecoinvent_provider",
                "selector": {
                    "name": row["activity_name"],
                    "reference_product": row["reference_product"],
                    "unit": row["unit"],
                    "exact_location": row["location"],
                },
                "amount": float(row["construction_amount"]),
            },
        }
        for row in selector_rows
    ]


def _mismanaged_routes(
    mapping_rows: tuple[dict[str, str], ...],
    selector_rows: tuple[dict[str, str], ...],
) -> list[dict[str, object]]:
    """Build material-specific open-burning and open-dump routes.

    Each constituent is matched to its declared provider portfolio.
    Open burning uses the portfolio shares directly. Shared preparation reads
    the licensed class portfolio to determine open-dump shares. Family shares
    were already applied in MFA.
    """
    selectors = {
        (row["fate"], row["provider_service_family"], row["infiltration_class"]): row
        for row in selector_rows
    }
    infiltration_classes = tuple(
        row["infiltration_class"]
        for row in selector_rows
        if row["fate"] == "open_dump" and row["provider_service_family"] == "PE"
    )
    routes: list[dict[str, object]] = []
    for row in mapping_rows:
        fate = row["fate"]
        classes = ("",) if fate == "open_burning" else infiltration_classes
        for infiltration_class in classes:
            selector = selectors[(fate, row["component_id"], infiltration_class)]
            component_share = float(row["component_share"])
            stage = f"mismanaged_{fate}_treatment"
            component_parts = [
                stage,
                _slug(row["material"]),
                row["constituent"],
                _slug(row["component_id"]),
            ]
            if infiltration_class:
                component_parts.append(_slug(infiltration_class))
            routes.append(
                {
                    "route_id": f"{stage}__{_slug(row['material'])}__{row['constituent']}",
                    "route_component_id": "__".join(component_parts),
                    "provider_demand_type": stage,
                    "region": "all",
                    "material": row["material"],
                    "constituent": row["constituent"],
                    "product": "polymer",
                    "good": "all",
                    "mapping_basis": "kg_waste_input",
                    "valid_from_year": 2019,
                    "valid_to_year": 2100,
                    "component_role": stage,
                    "component_material": row["component_id"],
                    "virgin_feedstock_route": "all",
                    "ultimate_carbon_origin": "all",
                    "share": component_share if fate == "open_burning" else None,
                    "throughput_multiplier": 1.0,
                    "provider_service_id": selector["provider_service_id"],
                    "lca_fate": fate,
                    "infiltration_class": infiltration_class,
                    "material_component_share": component_share,
                    "infiltration_share": 1.0 if fate == "open_burning" else None,
                    **(
                        {"infiltration_source_id": OPEN_DUMP_PORTFOLIO_SOURCE_ID}
                        if fate == "open_dump"
                        else {}
                    ),
                    "provider_class": row["provider_class"],
                    "analogue_status": row["analogue_status"],
                    "selection_reason": selector["selection_reason"],
                }
            )
    return routes


def _with_mismanaged_eol(catalogue: dict[str, object]) -> dict[str, object]:
    """Add the mismanaged waste services and fate routes to the catalog."""
    selector_rows = _csv_rows("provider_selector_catalogue.csv")
    mapping_rows = _csv_rows("material_fate_provider_mapping.csv")
    fate_rows = _csv_rows("h12_mismanaged_eol_fate_shares.csv")
    result = dict(catalogue)
    result["schema_version"] = "pytoce_provider_catalogue_schema_v1"
    result["services"] = [
        *cast(list[dict[str, object]], catalogue["services"]),
        *_mismanaged_services(selector_rows),
    ]
    result["routes"] = [
        *cast(list[dict[str, object]], catalogue["routes"]),
        *_mismanaged_routes(mapping_rows, selector_rows),
    ]
    result["source_authorities"] = [
        *cast(list[dict[str, object]], catalogue["source_authorities"]),
        {
            "source_id": fate_rows[0]["source_id"],
            "doi": fate_rows[0]["source_doi"],
            "dataset_doi": fate_rows[0]["dataset_doi"],
            "role": "global uncollected-waste open-burning and non-burned split",
            "direct_source_authority_ids": [],
        },
        {
            "source_id": OPEN_DUMP_PORTFOLIO_SOURCE_ID,
            "role": "Conditional open-dump class portfolio from the licensed RoW PE market",
            "calculation": (
                "Absolute class input divided by the sum of the five open-dump inputs; "
                "fifteen significant digits. Multiply by the material component share."
            ),
            "geography_year_boundary": "Baseline RoW portfolio for all 12 world regions, 2019-2100",
            "direct_source_authority_ids": ["ecoinvent_3_12_cutoff"],
            "limitation": "The same source climate portfolio applies across 12 world regions.",
        },
    ]
    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    semantic["explicit_cross_material_route_ids"] = [
        *cast(list[str], semantic["explicit_cross_material_route_ids"]),
        *sorted(
            {
                str(route["route_id"])
                for route in cast(list[dict[str, object]], result["routes"])
                if route.get("analogue_status") == "explicit_cross_material_analogue"
                and str(route["provider_demand_type"]).startswith("mismanaged_")
            }
        ),
    ]
    semantic["mismanaged_eol_authority_files"] = [
        "h12_mismanaged_eol_fate_shares.csv",
        "material_fate_provider_mapping.csv",
        "provider_selector_catalogue.csv",
    ]
    result["semantic_contract"] = semantic
    return result


def _chemical_operation_service(
    stage_name: str,
    stage: dict[str, object],
    *,
    carbon_class: str,
    method_authority_id: str,
    resolution_status: str,
) -> dict[str, object]:
    """Build one chemical-recycling stage service for a carbon origin class."""
    construction = {
        **stage,
        "kind": f"chemical_recycling_{stage_name}_operation",
        "source_role": "native_ecoinvent_provider",
        "ultimate_carbon_class": carbon_class,
        "resolution_status": resolution_status,
    }
    if stage_name == "preparation":
        construction["reject_market"] = {
            "service_id": "plastic_waste_market_gb_non_fossil",
        }
    if stage_name == "conversion":
        construction["direct_treatment_carbon_mappings"] = _direct_carbon_mappings(
            ("air",),
            species=(("Carbon monoxide, fossil", "Carbon monoxide, non-fossil"),),
        )
    return {
        "provider_service_id": f"chemical_recycling_{stage_name}_{carbon_class}",
        "name": f"chemical recycling {stage_name} operation",
        "reference_product": f"chemical recycling {stage_name} service",
        "unit": "kilogram",
        "location": "GB",
        "stage_owner": stage["stage_owner"],
        "foreground_service": True,
        "citation": method_authority_id,
        "representative_method": {
            "ultimate_carbon_class": carbon_class,
            "resolution_status": resolution_status,
        },
        "construction": construction,
    }


def _with_chemical_input_operations(
    catalogue: dict[str, object],
    composition: MaterialComposition,
) -> dict[str, object]:
    """Add input-normalized preparation and hydrothermal operations for eligible constituents.

    The composition owns eligibility and selective treatment amounts.
    Recovered monomer supply and shared polymerization have separate owners.
    """
    authority = _chemical_recycling_authority()
    method_authority_id = str(authority["method_authority_id"])
    resolution_status = str(authority["resolution_status"])
    stages = cast(dict[str, dict[str, object]], authority["stages"])
    services = cast(list[dict[str, object]], catalogue["services"])
    chemical_services = [
        _chemical_operation_service(
            stage_name,
            stages[stage_name],
            carbon_class=carbon_class,
            method_authority_id=method_authority_id,
            resolution_status=resolution_status,
        )
        for stage_name in ("preparation", "conversion")
        for carbon_class in ("fossil", "non_fossil")
    ]

    chemical_routes: list[dict[str, object]] = []
    explicit_ids: list[str] = []
    for item in composition.constituents:
        if item.chemical_polymer is None:
            continue
        for origin, carbon_class in _CHEMICAL_ORIGINS.items():
            for stage_name, stage in stages.items():
                demand_type = str(stage["provider_demand_type"])
                route_id = f"{item.material.casefold()}_{item.constituent}_{demand_type}__{origin}"
                chemical_routes.append(
                    {
                        "route_id": route_id,
                        "route_component_id": route_id,
                        "provider_demand_type": demand_type,
                        "region": "all",
                        "material": item.material,
                        "constituent": item.constituent,
                        "product": "polymer",
                        "good": "all",
                        "mapping_basis": stage["mapping_basis"],
                        "valid_from_year": 2019,
                        "valid_to_year": 2100,
                        "component_role": stage["component_role"],
                        "component_material": item.material,
                        "virgin_feedstock_route": "all",
                        "ultimate_carbon_origin": origin,
                        "share": 1.0,
                        "throughput_multiplier": 1.0,
                        "provider_service_id": f"chemical_recycling_{stage_name}_{carbon_class}",
                        "method_authority_id": method_authority_id,
                        "analogue_status": "eligible_polyolefin_constituent_operation",
                        "resolution_status": resolution_status,
                    }
                )
                explicit_ids.append(route_id)

    result = dict(catalogue)
    result["schema_version"] = "pytoce_provider_catalogue_schema_v1"
    result["services"] = [*services, *chemical_services]
    result["routes"] = [
        *cast(list[dict[str, object]], catalogue["routes"]),
        *chemical_routes,
    ]
    result["source_authorities"] = [
        *cast(list[dict[str, object]], catalogue["source_authorities"]),
        *cast(list[dict[str, object]], authority["source_authorities"]),
    ]
    registry = dict(cast(dict[str, object], catalogue["method_authority_registry"]))
    bindings = dict(cast(dict[str, list[str]], registry["exact_source_bindings"]))
    bindings[method_authority_id] = cast(list[str], authority["source_authority_ids"])
    registry["exact_source_bindings"] = bindings
    exact_methods = dict(
        cast(
            dict[str, dict[str, object]],
            registry.get("exact_method_authorities", {}),
        )
    )
    exact_methods[method_authority_id] = cast(
        dict[str, object],
        authority["method_authority"],
    )
    registry["exact_method_authorities"] = exact_methods
    result["method_authority_registry"] = registry
    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    semantic["explicit_cross_material_route_ids"] = [
        *cast(list[str], semantic["explicit_cross_material_route_ids"]),
        *explicit_ids,
    ]
    semantic["chemical_recycling_authority_file"] = "chemical_recycling_authority.json"
    semantic["chemical_recycling_resolution_status"] = resolution_status
    result["semantic_contract"] = semantic
    return result


def _with_sbr_compounding_origins(catalogue: dict[str, object]) -> dict[str, object]:
    """Apply the one Others SBR formulation to its three principal carbon origins.

    The compound ingredients, factory duties and auxiliary carbon stay fixed.
    Only the principal's identity and its filled-scrap landfill chain differ;
    biogenic and DAC carbon need separate traces despite sharing EF flow names.
    """
    services = list(cast(list[dict[str, object]], catalogue["services"]))
    base = next(
        service
        for service in services
        if service["provider_service_id"] == "others_esbr_compounding_conversion"
    )
    for origin in ("biogenic", "atmospheric_dac"):
        service = deepcopy(base)
        service["provider_service_id"] = f"others_esbr_compounding_conversion__{origin}"
        construction = cast(dict[str, object], service["construction"])
        construction["principal_carbon_origin"] = origin
        scrap = cast(dict[str, object], construction["filled_finishing_scrap"])
        scrap["treatment"] = {"service_id": f"sbr_filled_{origin}_landfill"}
        services.append(service)
    return {**catalogue, "services": services}


def _with_conversion_portfolios(
    catalogue: dict[str, object], composition: MaterialComposition
) -> dict[str, object]:
    """Map each constituent's finished output to its declared conversion operations.

    MFA already applies family composition. These route fractions partition
    only the selected constituent's output, so no family weight is applied here.
    The composition owner also normalizes rounded portfolio weights.
    """
    routes: list[dict[str, object]] = []
    route_ids = []
    registry = dict(cast(dict[str, object], catalogue["method_authority_registry"]))
    bindings = dict(cast(dict[str, list[str]], registry["exact_source_bindings"]))
    methods = dict(cast(dict[str, dict[str, object]], registry["exact_method_authorities"]))
    origin_specific = {
        str(construction["operation"])
        for service in cast(list[dict[str, object]], catalogue["services"])
        for construction in (cast(dict[str, object], service["construction"]),)
        if "operation" in construction and construction.get("ultimate_carbon_class") == "non_fossil"
    }
    # Filled SBR residues carry principal and auxiliary carbon independently.
    # Their three declared principal origins remain distinct even though EF
    # characterizes biogenic and DAC gases through the same native flow class.
    actual_origin_services = {
        (str(c["operation"]), str(c["principal_carbon_origin"])): str(
            service["provider_service_id"]
        )
        for service in cast(list[dict[str, object]], catalogue["services"])
        for c in (cast(dict[str, object], service["construction"]),)
        if "operation" in c and "principal_carbon_origin" in c
    }
    actual_origin_operations = {operation for operation, _ in actual_origin_services}
    origin_specific |= actual_origin_operations
    for item in composition.constituents:
        portfolio = item.conversion
        method_id = f"fixed_conversion_portfolio__{portfolio.name}"
        bindings[method_id] = list(portfolio.source_authority_ids)
        methods[method_id] = {
            "technology_boundary": portfolio.interpretation,
            "allocation_boundary": (
                "Conditional shares partition finished constituent output; "
                "family shares are applied once in MFA."
            ),
            "geography_year_boundary": (
                "Stationary global processing selection across 12 world regions and years; "
                "actual process geography and dates remain with each provider."
            ),
            "limitation": (
                "Fixed processing shares apply across 12 world regions and years. "
                "Process yields and inventory quantities are defined separately."
            ),
        }
        route_id = f"{_slug(item.material)}__{item.constituent}__processing_conversion"
        route_ids.append(route_id)
        for component in portfolio.components:
            origins = _TREATMENT_ORIGINS if component.operation in origin_specific else ("all",)
            for origin in origins:
                routes.append(
                    {
                        "route_id": route_id,
                        "route_component_id": f"{route_id}__{component.operation}__{origin}",
                        "provider_demand_type": "processing_conversion",
                        "region": "all",
                        "material": item.material,
                        "constituent": item.constituent,
                        "product": "polymer",
                        "good": "all",
                        "mapping_basis": "kg_processed_plastic",
                        "valid_from_year": 2019,
                        "valid_to_year": 2100,
                        "component_role": "processing_conversion",
                        "component_material": item.constituent,
                        "virgin_feedstock_route": "all",
                        "ultimate_carbon_origin": origin,
                        "share": float(component.share),
                        "throughput_multiplier": 1.0,
                        "provider_service_id": (
                            actual_origin_services[component.operation, origin]
                            if component.operation in actual_origin_operations
                            else f"{component.operation}__non_fossil"
                            if origin in {"biogenic", "atmospheric_dac"}
                            else component.operation
                        ),
                        "method_authority_id": method_id,
                    }
                )
    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    semantic["service_normalization_route_ids"] = route_ids
    registry["exact_source_bindings"] = bindings
    registry["exact_method_authorities"] = methods
    return {
        **catalogue,
        "material_composition_identity": composition.identity,
        "routes": [*cast(list[dict[str, object]], catalogue["routes"]), *routes],
        "semantic_contract": semantic,
        "method_authority_registry": registry,
        "source_authorities": [
            *cast(list[dict[str, object]], catalogue["source_authorities"]),
            *composition.source_authorities,
        ],
    }


def _with_sbr_treatment_correction(catalogue: dict[str, object]) -> dict[str, object]:
    """Apply the retained native treatment proxies for the SBR polymer fraction.

    In ``Elastomers (tyres)``, only the SBR polymer fraction is
    assigned to the selected rubber, polystyrene, or polyethylene
    treatment activity. Nonplastic tire constituents are excluded. The shares derived
    from styrene and butadiene are conditional on the SBR constituent. Family
    composition is already applied by MFA and remains descriptive provenance.
    The finished ``compound22`` formulation in ``Others`` uses its declared
    treatment routes.
    """

    authority = _sbr_treatment_authority()
    method = cast(dict[str, object], authority["method_authority"])
    method_id = str(method["method_authority_id"])
    bases = cast(dict[str, dict[str, object]], authority["functional_bases"])
    proxy_status = "sbr_styrene_pe_hydrocarbon_treatment_proxy"
    treatment_demands = {
        "incineration_treatment",
        "landfill_treatment",
        "mismanaged_open_burning_treatment",
        "mismanaged_open_dump_treatment",
    }

    corrected_routes: list[dict[str, object]] = []
    for raw in cast(list[dict[str, object]], catalogue["routes"]):
        route = dict(raw)
        material = str(route["material"])
        demand = str(route["provider_demand_type"])
        is_tyre = material == "Elastomers (tyres)" and demand in treatment_demands
        if not is_tyre:
            corrected_routes.append(route)
            continue

        basis = bases[material]
        if demand == "incineration_treatment":
            proxy_role = "compatible_unspecified_waste_rubber"
            provider_class = "native compatible waste-rubber proxy"
            analogue_status = "compatible_waste_rubber_unspecified_proxy"
            equivalence = (
                "The native waste-rubber municipal-incineration inventory treats the declared "
                "dry SBR share without adding a whole tyre functional unit."
            )
        else:
            # Landfill and mismanaged-fate resources name the two SBR fractions; their
            # declared material identifies the styrene or butadiene treatment proxy.
            proxy_role = {
                "SBR styrene-derived fraction": "styrene_derived_fraction",
                "PS": "styrene_derived_fraction",
                "SBR butadiene-derived fraction": "butadiene_derived_fraction",
                "PE": "butadiene_derived_fraction",
            }[str(route["component_material"])]
            provider_class = str(route["provider_class"])
            analogue_status = proxy_status
            equivalence = (
                "The SBR-equivalent polymer share is treated additively by the declared native "
                "PS and PE inventory components."
            )

        limitation = str(method["limitation"])
        corrected_routes.append(
            {
                **route,
                "provider_class": provider_class,
                "analogue_status": analogue_status,
                "selection_reason": equivalence,
                "method_authority_id": method_id,
                "material_functional_unit": str(basis["functional_unit"]),
                "styrene_component_share": float(
                    cast(float | int | str, basis["styrene_derived_mass_share"])
                ),
                "butadiene_analogue_component_share": float(
                    cast(float | int | str, basis["butadiene_derived_mass_share"])
                ),
                "sbr_share_in_material": float(
                    cast(float | int | str, basis["sbr_share_in_material"])
                ),
                "sbr_proxy_component_role": proxy_role,
                "whole_tyre_constituents_excluded": True,
                "no_treatment_sensitivity_executed": True,
                "source_role": "native_ecoinvent_provider",
                "resolution_status": str(authority["resolution_status"]),
                "functional_equivalence_claim": equivalence,
                "limitation": limitation,
            }
        )

    result = dict(catalogue)
    result["schema_version"] = "pytoce_provider_catalogue_schema_v1"
    result["routes"] = corrected_routes
    result["source_authorities"] = [
        *cast(list[dict[str, object]], catalogue["source_authorities"]),
        *cast(list[dict[str, object]], authority["source_authorities"]),
    ]
    registry = dict(cast(dict[str, object], catalogue["method_authority_registry"]))
    bindings = dict(cast(dict[str, list[str]], registry["exact_source_bindings"]))
    bindings[method_id] = cast(list[str], authority["source_authority_ids"])
    registry["exact_source_bindings"] = bindings
    exact_methods = dict(
        cast(dict[str, dict[str, object]], registry.get("exact_method_authorities", {}))
    )
    exact_methods[method_id] = method
    registry["exact_method_authorities"] = exact_methods
    result["method_authority_registry"] = registry

    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    explicit_ids = list(cast(list[str], semantic["explicit_cross_material_route_ids"]))
    affected_route_ids = {
        str(route["route_id"])
        for route in corrected_routes
        if (
            route["material"] == "Elastomers (tyres)"
            and route["provider_demand_type"] == "incineration_treatment"
        )
        or (
            route["material"] in {"Elastomers (tyres)", "Others"}
            and route["provider_demand_type"]
            in {"mismanaged_open_burning_treatment", "mismanaged_open_dump_treatment"}
        )
    }
    semantic["explicit_cross_material_route_ids"] = list(
        dict.fromkeys([*explicit_ids, *sorted(affected_route_ids)])
    )
    semantic["sbr_treatment_authority_file"] = "sbr_treatment_authority.json"
    semantic["sbr_treatment_resolution_status"] = str(authority["resolution_status"])
    result["semantic_contract"] = semantic
    return result


def _with_hybrid_provider_temporal_authority(
    catalogue: dict[str, object],
) -> dict[str, object]:
    """Bind the early provider-technology/general-background separation."""

    authority = _hybrid_provider_temporal_authority()
    result = dict(catalogue)
    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    semantic["hybrid_provider_temporal_authority_file"] = "hybrid_provider_temporal_authority.json"
    semantic["hybrid_provider_temporal_authority"] = authority
    result["semantic_contract"] = semantic
    return result


def _treatment_compartment(provider_demand_type: str) -> tuple[str, ...]:
    """Select the direct-emission compartment for one treatment route."""
    if provider_demand_type == "incineration_treatment":
        return ("air", "urban air close to ground")
    return ("air", "non-urban air or from high stacks")


def _treatment_origin_service(
    service: dict[str, object],
    provider_demand_type: str,
    replacements: list[dict[str, object]],
) -> dict[str, object]:
    """Clone a treatment service with non-fossil direct carbon flows."""
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    return {
        **service,
        "provider_service_id": f"{service_id}__non_fossil",
        "name": f"{service['name']} with non-fossil direct waste carbon",
        "citation": "ecoinvent_3_12_direct_treatment_carbon_origin_mapping",
        "provider_demand_type": provider_demand_type,
        "ultimate_carbon_origins": ["biogenic", "atmospheric_dac"],
        "construction": {
            "kind": "treatment_origin_operation",
            "carbon_attribution_status": "separable_treated_plastic_carbon",
            "base": {
                "source_role": construction["source_role"],
                "selector": construction["selector"],
            },
            "replace_biosphere": replacements,
        },
    }


def _with_treatment_origins(catalogue: dict[str, object]) -> dict[str, object]:
    """Separate direct treatment emissions by the plastic carbon origin.

    The native ecoinvent treatment service represents fossil plastic carbon.
    A second service retains every native exchange except that direct CO2, CH4,
    and CO flows from treated plastic use their non-fossil biosphere identities.
    Open dumping maps the CO2 and CH4 flows present in the selected source. Routes are duplicated by
    fossil, biogenic, and direct air capture
    origin so each foreground treatment amount selects the matching service.
    TPS services already attribute their historical formulation's mixed origins;
    this includes complete native municipal and dump proxies on whole wet mass.
    PHB municipal incineration has three composition-specific origin services;
    its carbon allocation and connected ash chain are already compiled there.
    Polyester burning already has homogeneous ultimate-origin profiles, selected
    by the foreground's actual carbon allocation.
    """
    routes = cast(list[dict[str, object]], catalogue["routes"])
    services = cast(list[dict[str, object]], catalogue["services"])
    # Actual cured receivers own their auxiliaries and carbon origins. Select
    # them first to avoid creating unused paint services with remapped carbon
    # flows or unused leachate chains for these routes.
    cured_route_ids = {
        str(route["route_id"])
        for route in routes
        if str(route["provider_demand_type"]) in _TREATMENT_DEMAND_TYPES
        and route["constituent"]
        in {"general_coatings", "marine_coatings", "thermosets", "road_marking_coatings"}
    }
    compound_treatment_ids = {
        str(service["provider_service_id"])
        for service in services
        if cast(dict[str, object], service["construction"])["kind"]
        in {
            "tps_residue_treatment",
            "tps_open_burning",
            "tps_native_treatment",
            "polyester_open_burning",
            "sbr_open_burning",
            "sbr_compound_landfill",
            "sbr_incineration",
            "sbr_dump",
            "phb_incineration",
        }
    }
    treatment_routes = [
        route
        for route in routes
        if str(route["provider_demand_type"]) in _TREATMENT_DEMAND_TYPES
        and str(route["route_id"]) not in cured_route_ids
        and str(route["provider_service_id"]) not in compound_treatment_ids
    ]
    # Every packaged treatment service has one stage. The catalog
    # invariant is checked with the actual maintained routes in package tests.
    demand_by_service = {
        str(route["provider_service_id"]): str(route["provider_demand_type"])
        for route in treatment_routes
    }

    expanded_services: list[dict[str, object]] = []
    for service in services:
        service_id = str(service["provider_service_id"])
        service_demand = demand_by_service.get(service_id)
        if service_demand is None:
            expanded_services.append(service)
            continue
        construction = dict(cast(dict[str, object], service["construction"]))
        species = (
            _DIRECT_TREATMENT_CARBON_SPECIES[:2]
            if service_demand == "mismanaged_open_dump_treatment"
            else _DIRECT_TREATMENT_CARBON_SPECIES
        )
        categories = list(_treatment_compartment(service_demand))
        replacements: list[dict[str, object]] = [
            {
                "selector": {
                    "name": source_name,
                    "unit": "kilogram",
                    "categories": categories,
                },
                "replacement_selector": {
                    "name": replacement_name,
                    "unit": "kilogram",
                    "categories": categories,
                },
                "attribution_basis": (
                    "The selected ecoinvent treatment activity attributes this exact "
                    "direct waste-carbon flow to the treated plastic input."
                ),
                "carbon_attribution_status": "separable_treated_plastic_carbon",
                "mapping_status": "exact_non_fossil_counterpart",
                "reason": (
                    "Preserve the native direct-treatment amount and compartment while "
                    "representing non-fossil plastic carbon."
                ),
                "limitation": (
                    "Only the direct treated-plastic carbon-flow identity changes; all "
                    "other native exchanges and metadata remain unchanged."
                ),
            }
            for source_name, replacement_name in species
        ]
        construction["direct_treatment_carbon_owner"] = "native_activity"
        construction["carbon_attribution_status"] = "separable_treated_plastic_carbon"
        construction["direct_treatment_carbon_mappings"] = replacements
        expanded_services.append(
            {
                **service,
                "provider_demand_type": service_demand,
                "ultimate_carbon_origins": ["fossil"],
                "construction": construction,
            }
        )
        expanded_services.append(_treatment_origin_service(service, service_demand, replacements))

    expanded_routes: list[dict[str, object]] = []
    by_id = {str(service["provider_service_id"]): service for service in services}
    for route in routes:
        demand = str(route["provider_demand_type"])
        if str(route["route_id"]) in cured_route_ids:
            expanded_routes.extend(
                {
                    **route,
                    "route_id": f"{route['route_id']}__{origin}",
                    "virgin_feedstock_route": "all",
                    "ultimate_carbon_origin": origin,
                    **_cured_resin_treatment_fields(route, origin, by_id),
                }
                for origin in _TREATMENT_ORIGINS
            )
            continue
        if (
            demand not in _TREATMENT_DEMAND_TYPES
            or str(route["provider_service_id"]) in compound_treatment_ids
        ):
            expanded_routes.append(route)
            continue
        base_service_id = str(route["provider_service_id"])
        for origin in _TREATMENT_ORIGINS:
            expanded_routes.append(
                {
                    **route,
                    "route_id": f"{route['route_id']}__{origin}",
                    "route_component_id": (f"{route['route_component_id']}__{origin}"),
                    "virgin_feedstock_route": "all",
                    "ultimate_carbon_origin": origin,
                    "provider_service_id": (
                        base_service_id if origin == "fossil" else f"{base_service_id}__non_fossil"
                    ),
                    "treatment_carbon_mapping_status": (
                        "native_fossil_identity"
                        if origin == "fossil"
                        else "exact_non_fossil_counterpart"
                    ),
                }
            )

    result = dict(catalogue)
    result["schema_version"] = "pytoce_provider_catalogue_schema_v1"
    result["services"] = expanded_services
    result["routes"] = expanded_routes
    treatment_route_ids = cured_route_ids | {str(route["route_id"]) for route in treatment_routes}
    semantic = dict(cast(dict[str, object], catalogue["semantic_contract"]))
    for field in (
        "explicit_cross_material_route_ids",
        "service_normalization_route_ids",
    ):
        expanded_ids: list[str] = []
        for route_id in cast(list[str], semantic[field]):
            if route_id in treatment_route_ids:
                expanded_ids.extend(f"{route_id}__{origin}" for origin in _TREATMENT_ORIGINS)
            else:
                expanded_ids.append(route_id)
        semantic[field] = expanded_ids
    result["semantic_contract"] = semantic
    return result


def _cured_resin_treatment_fields(
    route: dict[str, object], origin: str, services: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Select one cured-matrix receiver on the existing principal-kg route.

    Fixed auxiliaries belong inside the treatment. The MFA amount is multiplied once by the five
    infiltration-class shares.
    """
    demand = str(route["provider_demand_type"])
    endings = {
        "incineration_treatment": "municipal_incineration",
        "landfill_treatment": "landfill",
        "mismanaged_open_burning_treatment": "open_burning",
    }
    ending = (
        f"open_dump_{_slug(str(route['infiltration_class']))}"
        if demand == "mismanaged_open_dump_treatment"
        else endings[demand]
    )
    material = {
        "general_coatings": "epoxy",
        "marine_coatings": "epoxy",
        "thermosets": "upr",
        "road_marking_coatings": "road",
    }[str(route["constituent"])]
    service_id = f"{material}_{origin}_{ending}"
    return {
        "route_component_id": f"{route['route_id']}__{service_id}",
        "component_material": f"cured_{material}",
        "provider_service_id": service_id,
        "method_authority_id": services[service_id]["citation"],
        "provider_class": "actual cured matrix treatment",
        "analogue_status": "adapted native treatment function",
        "treatment_carbon_mapping_status": f"actual_{material}_principal_origin",
        "selection_reason": (
            "Treat actual cured resin and its incorporated auxiliaries with the "
            "declared complete receiving function on one principal-kg basis."
        ),
        "limitation": (
            "The selected treatment functions use the nominal cured composition "
            "and its phase-transfer coefficients."
        ),
    }


def _landfill_gas_rules(role: str) -> list[dict[str, object]]:
    """Declare the wastewater-derived fraction of each native child gas."""
    categories = (
        ("air", "non-urban air or from high stacks"),
        ("air", "urban air close to ground"),
    )
    species = _DIRECT_TREATMENT_CARBON_SPECIES
    selections = {
        "wastewater_treatment": [(compartment, species[:2]) for compartment in categories],
        "sludge_incineration": [(categories[1], species)],
        "sludge_landfarming": [(categories[0], species[:1])],
    }
    return [
        {
            **rule,
            "wastewater_carbon_fraction": (
                "carbon_dioxide"
                if role == "wastewater_treatment"
                and cast(dict[str, object], rule["selector"])["name"] == "Carbon dioxide, fossil"
                else "sludge"
            ),
            "attribution_basis": "Doka 2021 wastewater and fossil flocculant carbon transfers.",
            "carbon_attribution_status": "source_partitioned_waste_and_flocculant_carbon",
            "reason": "Only wastewater-derived gas follows the incoming waste carbon origin.",
            "limitation": (
                "The source uses common digestion of wastewater and flocculant carbon. "
                "A dated methane change is partitioned uniformly within its actual gas row."
            ),
        }
        for compartment, selected_species in selections[role]
        for rule in _direct_carbon_mappings(compartment, species=selected_species)
    ]


def _with_landfill_carbon(catalogue: dict[str, object]) -> dict[str, object]:
    """Attach declared native leachate/sludge chains to their landfill services."""
    semantic = cast(dict[str, object], catalogue["semantic_contract"])
    authority = cast(dict[str, object], semantic["landfill_carbon_origin"])
    services = cast(list[dict[str, object]], catalogue["services"])
    by_id = {str(service["provider_service_id"]): service for service in services}
    children = {
        "landfill": ("leachate_market",),
        "leachate_market": ("wastewater_treatment",),
        "wastewater_treatment": ("sludge70_market", "sludge97_market"),
        "sludge70_market": ("sludge_incineration",),
        "sludge97_market": ("sludge_landfarming",),
        "sludge_incineration": (),
        "sludge_landfarming": (),
    }
    for parent_id, chain in cast(dict[str, dict[str, object]], authority["chains"]).items():
        owners = cast(dict[str, dict[str, object]], chain["owners"])
        for role, selector in owners.items():
            replacements = [
                {
                    "selector": {
                        **owners[child],
                        "location": owners[child]["exact_location"],
                        "exchange_type": "technosphere",
                        "sign": "negative",
                    },
                    "service_id": f"{parent_id}__{child}__non_fossil",
                }
                for child in children[role]
            ]
            if role == "landfill":
                parent_ids = [parent_id, f"{parent_id}__non_fossil"]
                if parent_id == "landfill_mixed_plastic":
                    parent_ids.append("plastic_waste_landfill_row_non_fossil")
                for service_id in parent_ids:
                    service = by_id[service_id]
                    service["citation"] = authority["method"]
                    cast(dict[str, object], service["construction"])["treatment_replacements"] = (
                        replacements
                    )
                continue
            source = {"source_role": "native_ecoinvent_provider", "selector": selector}
            construction: dict[str, object] = {
                "kind": "waste_market_origin_operation",
                **source,
                "treatment_replacements": replacements,
            }
            if not role.endswith("market"):
                construction = {
                    "kind": "treatment_origin_operation",
                    "base": source,
                    "preserve_native_production": True,
                    "replace_biosphere": _landfill_gas_rules(role),
                    "treatment_replacements": replacements,
                    "wastewater_source": {
                        "source_role": "native_ecoinvent_provider",
                        "selector": owners["wastewater_treatment"],
                    },
                }
            services.append(
                {
                    "provider_service_id": f"{parent_id}__{role}__non_fossil",
                    "name": f"{selector['name']} with wastewater carbon origin",
                    "reference_product": selector["reference_product"],
                    "unit": selector["unit"],
                    "location": selector["exact_location"],
                    "stage_owner": "embedded_landfill_wastewater_treatment",
                    "foreground_service": False,
                    "frontier_service": True,
                    "citation": authority["method"],
                    "provider_demand_type": "embedded_plastic_waste_treatment",
                    "ultimate_carbon_origins": ["biogenic", "atmospheric_dac"],
                    "construction": construction,
                }
            )
    return catalogue


def augment_provider_catalogue(catalogue: dict[str, object]) -> dict[str, object]:
    """Apply every scientific route augmentation in its canonical order."""

    composition = load_material_composition()
    catalogue = _with_sbr_compounding_origins(catalogue)
    catalogue = _with_treatment_origins(
        _with_hybrid_provider_temporal_authority(
            _with_sbr_treatment_correction(
                _with_chemical_input_operations(
                    _with_mismanaged_eol(_with_conversion_portfolios(catalogue, composition)),
                    composition,
                )
            )
        )
    )
    return _with_landfill_carbon(catalogue)
