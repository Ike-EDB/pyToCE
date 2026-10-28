"""Complete PET utility accounting through real compilation and Brightway LCI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pet_finishing_fixtures import pet_finishing_inventory

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


@pytest.mark.parametrize("year", [2025, 2030])
def test_pet_food_grade_finishing_retains_complete_utilities_once(
    tmp_path: Path, year: int
) -> None:
    """Check both real grade routes, all fuel suppliers, meters and native records."""
    catalogue = load_provider_catalogue()
    routes = {row["route_id"]: row["provider_service_id"] for row in catalogue["routes"]}
    assert {
        key: routes[key]
        for key in (
            "pet_mechanical_recycling",
            "bioplastics_mechanical_recycling__bio_pet",
            "fibres_mechanical_recycling__polyethylene_terephthalate",
        )
    } == {
        "pet_mechanical_recycling": "mechanical_recycling_pet",
        "bioplastics_mechanical_recycling__bio_pet": "mechanical_recycling_pet",
        "fibres_mechanical_recycling__polyethylene_terephthalate": "mechanical_recycling_pet_fibre",
    }
    operation_ids = (
        "recycling_pet_finishing_support",
        "pet_finishing_natural_gas_combustion",
        "pet_finishing_propane_combustion",
        "pet_finishing_lpg_combustion_support",
    )
    parent_id = "recycling_pet_food_grade_finishing"
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["provider_service_id"] in (*operation_ids, parent_id)
    }
    # This nested service is normally flattened into the public bottle-PET
    # provider. Publish it in this bounded authority to solve its complete LCI.
    services[parent_id] = {**services[parent_id], "foreground_service": True}
    provider = PreparedProviderAuthority(
        identity="invented_pet_finishing",
        interface_database="interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="pet_finishing",
        biosphere_database="biosphere",
        source_database="original",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": categories,
                }
                for code, name, categories in (
                    ("co2", "Carbon dioxide, fossil", ("air", "urban air close to ground")),
                    ("sulfur", "Sulfur dioxide", ("air", "urban air close to ground")),
                    ("pm", "Particulates, > 10 um", ("air",)),
                    ("nitrogen", "Nitrogen", ("air",)),
                )
            },
            searchable=False,
        )
        bd.Database("original").write(
            pet_finishing_inventory(catalogue, "original", original=True), searchable=False
        )
        bd.Database("native").write(
            pet_finishing_inventory(catalogue, "native", future=year >= 2030), searchable=False
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        parent = bd.get_node(database=frontier.database_name, code=provider_service_code(parent_id))
        # Independent values from APR meters and the admitted unit
        # conversions, combined with deliberately invented original fuel quality.
        nitrogen = (
            0.0031
            * (14.73 * 6894.757293168)
            * 0.0280134
            / (8.31446261815324 * ((60 - 32) / 1.8 + 273.15))
        )
        gas = 0.034 * (14.73 * 6894.757293168 / 101300) * (288.15 / ((60 - 32) / 1.8 + 273.15))
        propane = 0.00029 * (4.24 * 0.45359237 / 3.785411784)
        lpg = 0.000083 / (
            0.5 / (4.24 * 0.45359237 / 3.785411784) + 0.5 / (4.84 * 0.45359237 / 3.785411784)
        )
        support = lpg * 45 * 3 / (0.153 * 40)
        co2 = lpg * 0.8 * 44 / 12 * 0.995
        sulfur = lpg * 0.00005 * 2
        demands = {
            (frontier.database_name, provider_service_code(key)): amount
            for key, amount in zip(operation_ids, (1.0, gas, propane, support), strict=True)
        }
        demands.update(
            {
                ("native", "nitrogen_vent"): nitrogen,
                ("native", "electricity_0"): 0.48,
                ("native", "lpg"): lpg,
            }
        )
        assert {ex.input.key: ex["amount"] for ex in parent.technosphere()} == pytest.approx(
            demands, rel=2e-13, abs=0
        )
        assert {ex.input.key: ex["amount"] for ex in parent.biosphere()} == pytest.approx(
            {("biosphere", "co2"): co2, ("biosphere", "sulfur"): sulfur}, rel=2e-13, abs=0
        )

        calculation = LCA({parent: 1.0})
        calculation.lci()
        source_codes = ("finisher", "natural_gas", "propane", "propane")
        electricity_inputs = (
            {"electricity_iam"} if year >= 2030 else {f"electricity_{i}" for i in range(10)}
        )
        omitted = (
            {"flake", "reject"} | electricity_inputs,
            electricity_inputs,
            electricity_inputs,
            electricity_inputs | {f"propane_{i}" for i in range(17)} | {"co2", "sulfur"},
        )
        for index, service_id in enumerate(operation_ids):
            operation = bd.get_node(
                database=frontier.database_name, code=provider_service_code(service_id)
            )
            original = bd.get_node(database="native", code=source_codes[index])
            expected = []
            for exchange in original.exchanges():
                if exchange.input.key[1] in omitted[index]:
                    continue
                record = {key: value for key, value in dict(exchange).items() if key != "output"}
                if record["type"] == "production":
                    record["input"] = operation.key
                    if index in {1, 2}:
                        record = {
                            "input": operation.key,
                            "amount": 0.2
                            if index == 1
                            else sum((i + 1) * 0.001 for i in range(17)),
                            "type": "production",
                        }
                expected.append(record)
            actual = [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in operation.exchanges()
            ]
            assert sorted(actual, key=repr) == sorted(expected, key=repr)
        # All 17 native suppliers participate, whereas every direct furnace
        # electricity input was replaced by the one whole-process meter.
        f = np.float32
        prop_activity = float(f(propane)) / float(f(0.153))
        for i in range(17):
            node = bd.get_node(database="native", code=f"propane_{i}")
            assert calculation.supply_array[calculation.dicts.activity[node.id]] == pytest.approx(
                prop_activity * float(f((i + 1) * 0.001))
            )
        electricity = bd.get_node(database="native", code="electricity_0")
        assert calculation.supply_array[calculation.dicts.activity[electricity.id]] == float(
            f(0.48)
        )
        if year >= 2030:
            iam_electricity = bd.get_node(database="native", code="electricity_iam")
            assert calculation.supply_array[
                calculation.dicts.activity[iam_electricity.id]
            ] == float(f(0.48))
        # The native duplicate particulate rows are summed in float32 when
        # Brightway builds the sparse matrix; stored records above stay intact.
        particulate = float(f(0.004)) / 2 + (
            float(f(gas)) / float(f(0.2)) + prop_activity + float(f(support)) / 3
        ) * float(f(0.004) + f(-0.001))
        for code, expected in (("nitrogen", float(f(nitrogen))), ("pm", particulate)):
            flow = bd.get_node(database="biosphere", code=code)
            assert float(
                calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
            ) == pytest.approx(expected, rel=2e-13)
