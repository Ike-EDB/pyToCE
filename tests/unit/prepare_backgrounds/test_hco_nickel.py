"""Check the admitted supported-Ni preparation vector through its real compiler."""

from math import fsum
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from bw2calc import LCA
from hco_nickel_fixtures import hco_nickel_biosphere, hco_nickel_native_inventory
from numpy.typing import NDArray
from provider_recipe_fixtures import _prepare_selected_native_frontier

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    load_provider_catalogue,
    provider_service_code,
)


@pytest.mark.parametrize("year", [2025, 2040])
def test_reported_nickel_preparation_keeps_dated_complete_native_supplies(
    tmp_path: Path,
    year: int,
) -> None:
    """Check the complete reported surrogate."""
    catalogue = load_provider_catalogue()
    names = BrightwayNames(tmp_path / "brightway", "hco_nickel", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(hco_nickel_biosphere(), searchable=False)
        native = hco_nickel_native_inventory(catalogue, "native", year=year)
        bd.Database("native").write(native, searchable=False)
        before = {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("native")
        }
        frontier = _prepare_selected_native_frontier(
            bd, catalogue, ("hco_supported_nickel",), year=year
        )
        node = bd.get_node(
            database=frontier.database_name, code=provider_service_code("hco_supported_nickel")
        )
        reference = node["catalyst_manufacturing_reference"]
        assert reference["powder_kg"] == pytest.approx({"Ni": 0.13, "SiO2": 0.87})
        # ANL Table 5 uses one US short ton, 5.0 million Btu fuel, 0.07 million Btu electricity, and
        # 120 US gallons consumed water. Pt-derived acid/NOx coefficients supply environmental
        # proxies.
        ton_kg = 907.18474
        fuel_mj = 5.0 * 1055.05585262 / ton_kg
        water_kg = 120 * 3.785411784 / ton_kg
        gas_per_mj = 0.04 if year == 2025 else 0.035
        expected = {
            "nickel": 0.13,
            "silica": 0.87,
            "nitric_acid": 0.04,
            "hydrochloric_acid": 0.08,
            "sodium_hydroxide": 0.003,
            "water": water_kg,
            "electricity": 0.07 * 1055.05585262 / (3.6 * ton_kg),
            "heat": fuel_mj / (36.0 * gas_per_mj),
        }
        units = {label: "kilogram" for label in expected}
        units.update({"electricity": "kilowatt hour", "heat": "megajoule"})
        reported = cast(dict[str, dict[str, object]], reference["native_supplies"])
        assert {label: row["amount"] for label, row in reported.items()} == pytest.approx(expected)
        assert {label: row["unit"] for label, row in reported.items()} == units
        assert reference["fuel_input_mj"] == pytest.approx(fuel_mj)
        assert reference["NOx_kg"] == pytest.approx(17.237 / ton_kg)
        assert reference["evaporated_water_kg"] == pytest.approx(water_kg)
        rows = list(node.exchanges())
        assert {
            row["input"]: row["amount"] for row in rows if row["type"] == "technosphere"
        } == pytest.approx(
            {("native", "hco_" + label): amount for label, amount in expected.items()}
        )
        assert {
            row["input"][1]: row["amount"] for row in rows if row["type"] == "biosphere"
        } == pytest.approx({"nox": 17.237 / ton_kg, "water_air": water_kg / 1000})
        assert next(iter(node.production()))["amount"] == 1.0

        result = LCA({node: 1.0})
        result.lci()
        supply_array = cast(NDArray[np.float64] | None, result.supply_array)
        assert supply_array is not None
        for label, amount in expected.items():
            supplier = bd.get_node(database="native", code="hco_" + label)
            production = next(iter(supplier.production()))["amount"]
            assert supply_array[result.dicts.activity[supplier.id]] == pytest.approx(
                amount / production, rel=3e-7
            )
        electricity = bd.get_node(database="native", code="hco_electricity")
        assert electricity["location"] == ("RER" if year == 2025 else "EUR")
        gas = bd.get_node(database="native", code="hco_gas")
        assert supply_array[result.dicts.activity[gas.id]] == pytest.approx(
            fuel_mj / 36 / 3, rel=3e-7
        )
        factory = bd.get_node(database="native", code="hco_factory")
        assert supply_array[result.dicts.activity[factory.id]] == pytest.approx(
            expected["heat"] * 0.002 / 4, rel=3e-7
        )
        # Synthetic marker intensities share a numerical value in their respective native units. The
        # weighted sum reports LCI demand.
        marker_per_unit = 0.001 if year == 2025 else 0.0006
        expected_lci = {
            "nox": 17.237 / ton_kg,
            "water_air": water_kg / 1000,
            "co2": expected["heat"] * (0.08 if year == 2025 else 0.07),
            "marker": marker_per_unit
            * (fsum(expected.values()) + fuel_mj / 36 + expected["heat"] * 0.002),
        }
        for code, amount in expected_lci.items():
            flow = bd.get_node(database="biosphere", code=code)
            assert result.inventory[result.dicts.biosphere[flow.id], :].sum() == pytest.approx(
                amount, rel=3e-7
            )
        assert {
            n.key: [dict(row) for row in n.exchanges()] for n in bd.Database("native")
        } == before
