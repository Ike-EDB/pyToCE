"""Synthetic inventories for chemical recycling and reject treatment compilation."""

from pathlib import Path
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _landfill_sources, _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    native_resolution,
    recipe_compilation,
)


@pytest.fixture
def chemical_compilation_case(
    tmp_path: Path,
) -> Any:
    """Supply chemical operations and the complete mixed-plastic landfill chain."""
    catalogue = provider_catalogue.load_provider_catalogue()
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="chemical_recycling_compilation",
        biosphere_database="biosphere",
        source_database="technology",
    )
    with selected_brightway_project(names) as bd:
        biosphere = {
            ("biosphere", "water"): {
                "name": "Water",
                "unit": "cubic meter",
                "categories": ("water",),
                "type": "emission",
            },
            ("biosphere", "toc"): {
                "name": "TOC, Total Organic Carbon",
                "unit": "kilogram",
                "categories": ("water", "surface water"),
                "type": "emission",
            },
        }
        species = (
            ("co2", "Carbon dioxide"),
            ("ch4", "Methane"),
            ("co", "Carbon monoxide"),
        )
        compartments = (
            ("generic", ("air",)),
            ("high", ("air", "non-urban air or from high stacks")),
            ("urban", ("air", "urban air close to ground")),
        )
        for compartment_id, categories in compartments:
            for species_id, name in species:
                for origin_id, suffix in (("f", "fossil"), ("nf", "non-fossil")):
                    biosphere[("biosphere", f"{species_id}_{compartment_id}_{origin_id}")] = {
                        "name": f"{name}, {suffix}",
                        "unit": "kilogram",
                        "categories": categories,
                        "type": "emission",
                    }
        bd.Database("biosphere").write(biosphere, searchable=False)
        nodes = {
            ("technology", "processed_output"): _process(
                "technology",
                "processed_output",
                "waste mixed polyolefins, processed, Recycled Content cut-off",
                "waste mixed polyolefins, processed",
                location="GLO",
            ),
            ("technology", "processed_market"): _process(
                "technology",
                "processed_market",
                "market for waste mixed polyolefins, processed",
                "waste mixed polyolefins, processed",
                location="GB",
            ),
            ("technology", "metal"): _process(
                "technology", "metal", "metal cut-off output", "metal scrap", location="GLO"
            ),
            ("technology", "utility"): _process(
                "technology", "utility", "utility supply", "utility", location="GB"
            ),
        }
        treatment_details = (
            ("open_burning", "treatment of waste plastic, mixture, open burning", "GLO", "high"),
            (
                "incineration",
                "treatment of waste plastic, mixture, municipal incineration",
                "GB",
                "urban",
            ),
        )
        for treatment_id, treatment_name, location, compartment_id in treatment_details:
            nodes[("technology", treatment_id)] = _process(
                "technology",
                treatment_id,
                treatment_name,
                "waste plastic, mixture",
                production_amount=-1.0,
                location=location,
                exchanges=[
                    {
                        "input": ("biosphere", f"{species_id}_{compartment_id}_f"),
                        "amount": amount,
                        "type": "biosphere",
                    }
                    for (species_id, _name), amount in zip(
                        species,
                        (0.5, 0.02, 0.01),
                        strict=True,
                    )
                ],
            )
        chains = cast(dict[str, Any], catalogue["semantic_contract"])["landfill_carbon_origin"][
            "chains"
        ]
        landfill, _ = _landfill_sources(
            {
                "semantic_contract": {
                    "landfill_carbon_origin": {
                        "chains": {"landfill_mixed_plastic": chains["landfill_mixed_plastic"]}
                    }
                }
            },
            dated=False,
        )
        keys = {
            key: (
                "technology",
                "landfill"
                if key[1] == "landfill_mixed_plastic_landfill"
                else f"landfill_{key[1]}"
                if key[1] in {"utility", "flocculant"}
                else key[1],
            )
            for key in landfill
        }
        nodes.update(
            {
                keys[key]: {
                    **value,
                    "exchanges": [
                        {**row, "input": keys.get(row["input"], row["input"])}
                        for row in value["exchanges"]
                    ],
                }
                for key, value in landfill.items()
            }
        )
        nodes[("technology", "reject_market")] = _process(
            "technology",
            "reject_market",
            "market for waste plastic, mixture",
            "waste plastic, mixture",
            production_amount=-1.0,
            location="GB",
            exchanges=[
                {
                    "input": ("technology", treatment_id),
                    "amount": amount,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": -3.0,
                    "scale": 0.1,
                }
                for treatment_id, amount in (
                    ("open_burning", -0.1),
                    ("landfill", -0.2),
                    ("incineration", -0.7),
                )
            ]
            + [
                {
                    "input": ("technology", "utility"),
                    "amount": 0.03,
                    "type": "technosphere",
                }
            ],
        )
        nodes[("technology", "sorting")] = _process(
            "technology",
            "sorting",
            "treatment of waste mixed polyolefins, unprocessed, sorting",
            "waste mixed polyolefins, unprocessed",
            production_amount=-1.0,
            location="GB",
            exchanges=[
                {
                    "input": ("technology", "processed_output"),
                    "amount": -0.8,
                    "type": "technosphere",
                },
                {
                    "input": ("technology", "reject_market"),
                    "amount": -0.15,
                    "type": "technosphere",
                    "uncertainty type": 3,
                    "loc": -2.0,
                    "scale": 0.2,
                },
                {
                    "input": ("technology", "metal"),
                    "amount": -0.05,
                    "type": "technosphere",
                },
                {
                    "input": ("technology", "utility"),
                    "amount": 0.2,
                    "type": "technosphere",
                },
                {
                    "input": ("biosphere", "water"),
                    "amount": 0.001,
                    "type": "biosphere",
                },
            ],
        )
        for index, (product, denominator) in enumerate(
            zip(
                ("naphtha", "distillate gas oil", "heavy gas oil", "heavy wax residue"),
                (1.0, 2.0, 4.0, 5.0),
                strict=True,
            )
        ):
            nodes[("technology", f"htt_{index}")] = _process(
                "technology",
                f"htt_{index}",
                "treatment of waste mixed polyolefins, processed, hydrothermal treatment",
                product,
                location="GB",
                exchanges=[
                    {
                        "input": ("technology", "processed_market"),
                        "amount": denominator,
                        "type": "technosphere",
                    },
                    {
                        "input": ("technology", "utility"),
                        "amount": 2.0 * denominator,
                        "type": "technosphere",
                        "uncertainty type": 3,
                        "loc": 0.5,
                        "scale": 0.1,
                    },
                    {
                        "input": ("biosphere", "co_generic_f"),
                        "amount": 0.1 * denominator,
                        "type": "biosphere",
                    },
                ],
            )
        bd.Database("technology").write(nodes, searchable=False)

        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], catalogue["services"])
        }
        context = recipe_compilation._RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2030,
            provider_database="provider_frontier",
            source=native_resolution._ActivityResolver(bd, "technology"),
            ordinary=native_resolution._ActivityResolver(bd, "technology"),
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
        yield context, services


def _chemical_reject_services(
    catalogue: dict[str, Any], selected: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Include the finite mixed-waste treatment chain used by chemical fixtures."""
    required = set(selected) | {
        "plastic_waste_market_gb_non_fossil",
        "plastic_waste_open_burning_glo_non_fossil",
        "plastic_waste_landfill_row_non_fossil",
        "plastic_waste_incineration_gb_non_fossil",
    }
    chain = catalogue["semantic_contract"]["landfill_carbon_origin"]["chains"][
        "landfill_mixed_plastic"
    ]
    required.update(
        f"landfill_mixed_plastic__{role}__non_fossil"
        for role in chain["owners"]
        if role != "landfill"
    )
    return {
        str(service["provider_service_id"]): service
        for service in catalogue["services"]
        if service["provider_service_id"] in required
    }


def _chemical_reject_gases(
    weights: tuple[float, float, float], *, year: int
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Derive principal and auxiliary gases per kg of the invented reject mix.

    Negative source references give landfill activity duties 1, .2, .08,
    .024, .048, .024 and .048. Only wastewater/sludge carbon fractions apply
    to principal diagnostics; the flocculant and utility remain fossil LCI.
    These equations use the synthetic source quantities, using the source equations.
    """
    influent = 0.001 / 0.116237319347025875
    sludge = influent * 0.658860928290208689
    auxiliary = 0.0021 * 36.03 / 71.08
    q = sludge / (sludge + auxiliary)
    gas = (sludge + auxiliary) * 0.20525339975098886 * 0.603
    biological_co2 = influent * 0.224901752362765436 * 44 / 12
    gas_co2 = gas * (1 - 0.65 * 0.0075) * 44 / 12
    gas_ch4 = gas * 0.65 * 0.0075 * 16 / 12
    primary_co2 = biological_co2 + q * gas_co2
    parent_factor, incineration_factor = (2.0, 3.0) if year == 2020 else (1.0, 1.0)
    principal = {
        ("co2", "high"): 0.2 * parent_factor + 0.08 * 0.3 * primary_co2 + 0.048 * 0.04 * q,
        ("co2", "urban"): 0.08 * 0.7 * primary_co2 + 0.024 * 0.05 * q,
        ("ch4", "high"): 0.005 * parent_factor + 0.08 * 0.3 * q * gas_ch4,
        ("ch4", "urban"): 0.08 * 0.7 * q * gas_ch4 + 0.024 * 0.0006 * q,
        ("co", "high"): 0.001 * parent_factor,
        ("co", "urban"): 0.024 * 0.0007 * q,
    }
    fossil_auxiliary = {
        ("co2", "high"): (
            0.08 * 0.3 * (1 - q) * gas_co2
            + 0.048 * 0.04 * (1 - q)
            + 0.1 * (1 + 0.2 + 0.08 + 0.024 + 0.048 + 0.024 + 0.048) * 0.7
            + 0.08 * 0.0021 * 0.8
        ),
        ("co2", "urban"): 0.08 * 0.7 * (1 - q) * gas_co2 + 0.024 * 0.05 * (1 - q),
        ("ch4", "high"): 0.08 * 0.3 * (1 - q) * gas_ch4,
        ("ch4", "urban"): 0.08 * 0.7 * (1 - q) * gas_ch4 + 0.024 * 0.0006 * (1 - q),
        ("co", "high"): 0.0,
        ("co", "urban"): 0.024 * 0.0007 * (1 - q),
    }
    principal = {key: weights[1] * value for key, value in principal.items()}
    for species, amount in (("co2", 0.5), ("ch4", 0.02), ("co", 0.01)):
        principal[(species, "high")] += weights[0] * amount
        principal[(species, "urban")] += weights[2] * incineration_factor * amount
    return principal, {key: weights[1] * value for key, value in fossil_auxiliary.items()}
