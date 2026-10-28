"""Test road coating, bead manufacture, mixing and container treatment with synthetic inputs."""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
    chemical_residue_biosphere,
    chemical_residue_native_inventory,
)
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)


@pytest.mark.parametrize("future", [False, True])
def test_road_marking_separates_bead_production_coating_and_binder_mixing(
    tmp_path: Path,
    future: bool,
) -> None:
    """Check bead and mixing demands and container incineration for each carbon origin."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    service_ids = {
        "others_road_marking_conversion",
        "others_road_marking_conversion__biogenic",
        "others_road_marking_conversion__atmospheric_dac",
        "others_recycled_glass_bead_manufacturing",
        "others_glass_bead_furnace_operation",
        "others_road_marking_mixing_operation",
    }
    services = {
        service["provider_service_id"]: service
        for service in catalogue["services"]
        if service["provider_service_id"] in service_ids
    }
    provider = PreparedProviderAuthority(
        identity="bounded_road_glass_authority",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    parent_id = "others_road_marking_conversion"
    bead_id = "others_recycled_glass_bead_manufacturing"
    furnace_id = "others_glass_bead_furnace_operation"
    mixing_id = "others_road_marking_mixing_operation"
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="road_glass_operations",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        biosphere = chemical_residue_biosphere()
        biosphere["biosphere", "road_mma_proxy"] = {
            "name": "NMVOC, non-methane volatile organic compounds",
            "unit": "kilogram",
            "categories": ("air", "urban air close to ground"),
            "type": "emission",
        }
        bd.Database("biosphere").write(biosphere, searchable=False)
        native = chemical_residue_native_inventory(catalogue, "native")
        leaves = {}
        labels = {}

        def leaf(selector: dict[str, Any], label: str) -> tuple[str, str]:
            """Supply a named invented material or utility, sharing exact descriptors."""
            descriptor = (
                selector["name"],
                selector["reference_product"],
                selector["unit"],
                (
                    selector["location_by_period"]["future" if future else "baseline"]
                    if "location_by_period" in selector
                    else selector.get("exact_location", "RER")
                ),
            )
            key = leaves.setdefault(descriptor, ("native", f"road_leaf_{len(leaves)}"))
            labels[label] = key
            if key not in native:
                native[key] = {
                    "name": descriptor[0],
                    "reference product": descriptor[1],
                    "unit": descriptor[2],
                    "location": descriptor[3],
                    "type": "process",
                    "exchanges": [
                        {
                            "input": key,
                            "type": "production",
                            "amount": -1.0 if label == "glass_dust_landfill" else 1.0,
                        },
                        {
                            "input": ("biosphere", "co2"),
                            "type": "biosphere",
                            "amount": 0.001 * len(leaves),
                        },
                    ],
                }
            return key

        for label, supply in services[parent_id]["construction"]["supplies"].items():
            if "selector" in supply:
                leaf(supply["selector"], label)
        for component in services[bead_id]["construction"]["components"]:
            if "selector" in component:
                leaf(component["selector"], component["component_id"])
        mixer_inputs = [
            {
                "input": leaf(omission["selector"], omission["omission_id"]),
                "type": "technosphere",
                "amount": 0.1,
            }
            for omission in services[mixing_id]["construction"]["omit_technosphere"]
        ]
        for label, name, product, unit, location in (
            (
                "gas",
                "market group for natural gas, high pressure",
                "natural gas, high pressure",
                "cubic meter",
                "Europe without Switzerland",
            ),
            (
                "furnace_electricity",
                "market group for electricity, low voltage",
                "electricity, low voltage",
                "kilowatt hour",
                "Europe without Switzerland",
            ),
            (
                "mixing_electricity",
                "market group for electricity, low voltage",
                "electricity, low voltage",
                "kilowatt hour",
                "RER",
            ),
            ("mixing_factory", "invented mixing factory", "mixing factory", "unit", "RER"),
            ("furnace_factory", "invented furnace", "furnace", "unit", "RER"),
            ("furnace_lubricant", "invented lubricant", "lubricant", "kilogram", "RER"),
        ):
            leaf(
                {
                    "name": name,
                    "reference_product": product,
                    "unit": unit,
                    "exact_location": location,
                },
                label,
            )
        # Complete the existing heat_europe furnace while building the disposable source, retaining
        # a unique descriptor.
        for code, service_id in (("mixing", mixing_id), ("heat_europe", furnace_id)):
            selector = services[service_id]["construction"]["base"]["selector"]
            native[("native", code)] = {
                "name": selector["name"],
                "reference product": selector["reference_product"],
                "unit": selector["unit"],
                "location": selector["exact_location"],
                "type": "process",
                "exchanges": [
                    {
                        "input": ("native", code),
                        "type": "production",
                        "amount": 1.0 if code == "mixing" else 2.0,
                    },
                ],
            }
        native[("native", "mixing")]["exchanges"].extend(
            [
                *mixer_inputs,
                {"input": labels["mixing_electricity"], "type": "technosphere", "amount": 0.02},
                {"input": labels["mixing_factory"], "type": "technosphere", "amount": 0.003},
                {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.004},
            ]
        )
        native[("native", "heat_europe")]["exchanges"].extend(
            [
                {"input": labels["gas"], "type": "technosphere", "amount": 0.08},
                {"input": labels["furnace_electricity"], "type": "technosphere", "amount": 0.004},
                {"input": labels["furnace_factory"], "type": "technosphere", "amount": 0.0001},
                {"input": labels["furnace_lubricant"], "type": "technosphere", "amount": 0.002},
                {
                    "input": ("biosphere", "co2"),
                    "type": "biosphere",
                    "amount": 0.08,
                    "uncertainty type": 2,
                    "loc": math.log(0.08),
                    "scale": 0.1,
                    "allocation": "native furnace combustion",
                },
            ]
        )
        if future:
            # Original composition remains separate from the future fixture's
            # missing exchange properties. Dated production/reference quantities double.
            bd.Database("original").write(
                chemical_residue_native_inventory(catalogue, "original"), searchable=False
            )
            for code in ("mc_hwi", "mc_fixed", "mc_ash"):
                for row in native["native", code]["exchanges"]:
                    row["amount"] *= 2
                    if "loc" in row:
                        row["loc"] = (
                            row["loc"] + math.log(2)
                            if row.get("uncertainty type") == 2
                            else row["loc"] * 2
                        )
                    for bound in ("minimum", "maximum"):
                        if bound in row:
                            row[bound] *= 2
        bd.Database("native").write(native, searchable=False)
        if future:
            apply_premise_chemical_receiving_metadata(bd, "native")
        original = {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for source in (("native", "original") if future else ("native",))
            for node in bd.Database(source)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("original" if future else "native", "biosphere"),
        )
        furnace = bd.get_node(
            database=frontier.database_name, code=provider_service_code(furnace_id)
        )
        bead = bd.get_node(database=frontier.database_name, code=provider_service_code(bead_id))
        for origin, selected_id in (
            ("fossil", parent_id),
            ("biogenic", parent_id + "__biogenic"),
            ("atmospheric_dac", parent_id + "__atmospheric_dac"),
        ):
            parent = bd.get_node(
                database=frontier.database_name, code=provider_service_code(selected_id)
            )
            container = bd.get_node(
                database=frontier.database_name,
                code=f"road_coating__{selected_id}__container_incineration",
            )
            reference = parent["road_reference_per_retained_principal_kg"]
            # Normalize the source premix to one retained principal kg after
            # 1% container loss and 0.00202 kg MMA emitted per application.
            premix, beads = 1 / 0.21875, 0.25 / 0.21875
            assert reference["mixing_kg"] == pytest.approx(premix)
            assert reference["container_residue"]["mass_kg"] == pytest.approx(0.0101 / 0.21875)
            parent_inputs = {row.input.key: row["amount"] for row in parent.technosphere()}
            assert parent_inputs[bead.key] == pytest.approx(beads)
            assert parent_inputs[container.key] == pytest.approx(-0.0101 / 0.21875)
            assert [(row.input.key, row["amount"]) for row in parent.biosphere()] == [
                (("biosphere", "road_mma_proxy"), pytest.approx(0.00202 / 0.21875))
            ]
            assert not any(
                row.input.key in {("native", "cr_road"), ("native", "cr_rail")}
                for row in container.technosphere()
            )
            principal = reference["container_residue"]["principal_carbon_kg"]
            assert principal[("fossil", "biogenic", "atmospheric_dac").index(origin)] > 0
            calculation = LCA({parent: 1.0})
            calculation.lci()
            # Brightway stores each matrix edge as float32. Bead manufacture
            # remains a separate node; its mixing demand joins the premix once.
            stored_beads = float(np.float32(beads))
            total_mixing = float(np.float32(premix)) + stored_beads
            gas_demand = stored_beads * float(np.float32(0.5812210139133216))
            furnace_scale = gas_demand / float(np.float32(0.08))
            expected = dict.fromkeys(leaves.values(), 0.0)
            for label, amount in {
                "rutile_titanium_dioxide": 0.075 / 0.21875,
                "ground_calcium_carbonate": 0.15 / 0.21875,
                "corundum": 0.05 / 0.21875,
                "fumed_silica_activated_silica_analogue": 0.002 / 0.21875,
                "organic_peroxide_hardener_analogue": 0.005 / 0.21875,
                "hardener_phlegmatizer_dotp_analogue": 0.005 / 0.21875,
            }.items():
                expected[labels[label]] = float(np.float32(amount))
            expected[labels["silica_sand"]] = float(
                np.float32(np.float32(0.1 / 0.21875) + np.float32(0.15 / 0.21875))
            )
            # Complete bead supply, dust receiving, cold coating and heating
            # remain separately justified quantities per finished wet bead kg.
            for label, amount in {
                "sorted_recycled_glass_cullet": 1.009515432592356,
                "uncoated_bead_production_electricity": 0.044071298213832875,
                "glass_dust_landfill": 0.009995202302895,
                "methyl_silicone_oil_pdms": 0.000119942427635,
                "sodium_silicate_coating_active": 0.000071965456581,
                "coating_dilution_water": 0.000287861826323,
            }.items():
                expected[labels[label]] = stored_beads * float(np.float32(amount))
            expected[labels["mixing_electricity"]] = total_mixing * float(np.float32(0.02))
            expected[labels["mixing_factory"]] = total_mixing * float(np.float32(0.003))
            expected[labels["gas"]] = gas_demand
            expected[labels["furnace_factory"]] = furnace_scale * float(np.float32(0.0001))
            expected[labels["furnace_lubricant"]] = furnace_scale * float(np.float32(0.002))
            assert {
                key: float(
                    cast(NDArray[np.float64], calculation.supply_array)[
                        calculation.dicts.activity[bd.get_node(key=key).id]
                    ]
                )
                for key in expected
            } == pytest.approx(expected, rel=1e-13, abs=0)
            # Sum bead/mixer and container-incineration CO2 once. Other elementary flows retain
            # their category units.
            marker = bd.get_node(database="biosphere", code="co2")
            hwi_lca = LCA({container: -float(np.float32(0.0101 / 0.21875))})
            hwi_lca.lci()
            hwi_marker = float(hwi_lca.inventory[hwi_lca.dicts.biosphere[marker.id], :].sum())
            expected_co2 = (
                total_mixing * float(np.float32(0.004))
                + furnace_scale * float(np.float32(0.08))
                + sum(
                    amount * float(np.float32(native[key]["exchanges"][1]["amount"]))
                    for key, amount in expected.items()
                )
                + hwi_marker
            )
            assert float(
                calculation.inventory[calculation.dicts.biosphere[marker.id], :].sum()
            ) == pytest.approx(expected_co2, rel=1e-13, abs=0)
        retained = [
            {key: value for key, value in exchange.items() if key != "output"}
            for exchange in original[("native", "heat_europe")]
            if exchange["type"] != "production"
            and exchange["input"] != labels["furnace_electricity"]
        ]
        assert sorted(
            [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in furnace.exchanges()
                if exchange["type"] != "production"
            ],
            key=repr,
        ) == sorted(retained, key=repr)
        assert original == {
            node.key: [dict(exchange) for exchange in node.exchanges()]
            for source in (("native", "original") if future else ("native",))
            for node in bd.Database(source)
        }
