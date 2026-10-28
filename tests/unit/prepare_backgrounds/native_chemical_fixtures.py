"""Synthetic chemical supply and regional wastewater inventories for compilation tests."""

from copy import deepcopy
from math import log
from typing import Any

from tests.unit.prepare_backgrounds.chemical_organic_water_fixtures import (
    add_chemical_organic_water_inventory,
    chemical_organic_water_biosphere,
)
from tests.unit.prepare_backgrounds.chemical_steam_fixtures import (
    chemical_steam_biosphere,
    chemical_steam_native_inventory,
)


def native_chemical_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Create synthetic chemical and wastewater flows with the required names and units."""
    result = chemical_steam_biosphere()
    for code, name in (("mibk_hydrogen", "Hydrogen"), ("mibk_oxygen", "Oxygen")):
        result[("biosphere", code)] = {
            "name": name,
            "unit": "kilogram",
            "categories": ("air",),
            "type": "emission",
        }
    return chemical_organic_water_biosphere(result)


def _native_chemical_row(
    database: str, code: str, amount: float, *, biosphere: bool = False
) -> dict[str, Any]:
    """Give a fixture dose visible uncertainty and allocation metadata."""
    return {
        "input": ("biosphere" if biosphere else database, code),
        "type": "biosphere" if biosphere else "technosphere",
        "amount": amount,
        "uncertainty type": 2 if amount > 0 else 0,
        "loc": log(amount) + 0.01 if amount > 0 else amount,
        "scale": 0.2,
        "minimum": min(amount / 2, amount * 2),
        "maximum": max(amount / 2, amount * 2),
        "allocation": "invented MIBK native allocation",
    }


def _native_chemical_supplier(
    database: str,
    code: str,
    name: str,
    product: str,
    location: str,
    *,
    unit: str = "kilogram",
    production: float = 1.0,
) -> dict[str, Any]:
    """Provide a small synthetic supplier with a positive reference production amount."""
    return {
        "name": name,
        "reference product": product,
        "unit": unit,
        "location": location,
        "exchanges": [{"input": (database, code), "type": "production", "amount": production}],
    }


def _native_regional_water_sources(
    result: dict[tuple[str, str], dict[str, Any]], database: str
) -> None:
    """Compose four synthetic municipal inventories with distinct signed references.

    These municipal quantities are synthetic.
    Europe has no purchased heat; its explicit source recipe must add heat.
    The common fixture already supplies four regions. Replace those records
    directly so each selected native descriptor has one supplier.
    """
    plant = deepcopy(result[(database, "plant_row")])
    market = deepcopy(result[(database, "market_row")])
    locations = ("CA-QC", "Europe without Switzerland", "RoW", "CH")
    portfolio_rows = []
    for i, location in enumerate(locations):
        suffix = ("ca_qc", "europe", "row", "ch")[i]
        plant_code = "plant_row" if suffix == "row" else "mibk_plant_" + suffix
        market_code = {
            "europe": "steam_ww",
            "row": "market_row",
            "ca_qc": "mibk_water_ca_qc",
            "ch": "mibk_water_ch",
        }[suffix]
        donor_plant = deepcopy(plant)
        donor_plant["location"] = location
        donor_plant["exchanges"][0]["input"] = (database, plant_code)
        # Different source references expose accidental direct use of whole
        # dataset amounts. Keep every original uncertainty parameter coherent.
        for row in donor_plant["exchanges"]:
            factor = (4 + i) / 4
            row["amount"] *= factor
            if row.get("uncertainty type") == 2:
                row["loc"] += log(factor)
                row["minimum"] *= factor
                row["maximum"] *= factor
            elif "loc" in row:
                row["loc"] *= factor
        if suffix == "europe":
            donor_plant["exchanges"] = [
                row
                for row in donor_plant["exchanges"]
                if row["input"][1] not in {"heat_a", "heat_b"}
            ]
        donor_market = deepcopy(market)
        donor_market["location"] = location
        donor_market["exchanges"][0]["input"] = (database, market_code)
        donor_market["exchanges"][0]["amount"] = -(2 + i)
        donor_market["exchanges"][1]["input"] = (database, plant_code)
        donor_market["exchanges"][1]["amount"] = -(2 + i) * (i + 1) / 5
        donor_market["exchanges"][1]["loc"] = donor_market["exchanges"][1]["amount"]
        result[(database, plant_code)] = donor_plant
        result[(database, market_code)] = donor_market
        portfolio_rows.append(_native_chemical_row(database, market_code, -(i + 1) * 0.001))
    # The extended common fixture's three additional regions and European
    # heater are superseded by these deliberate nonunit native sources.
    # Keep their existing shared-graph callers on those same duties;
    # duplicate descriptors would make selection depend on database row order.
    replacements = {
        "plant_ca_qc": "mibk_plant_ca_qc",
        "plant_europe": "mibk_plant_europe",
        "plant_ch": "mibk_plant_ch",
        "market_ca_qc": "mibk_water_ca_qc",
        "market_europe": "steam_ww",
        "market_ch": "mibk_water_ch",
        "heat_europe": "mibk_heat",
    }
    for old in replacements:
        del result[(database, old)]
    for dataset in result.values():
        for row in dataset["exchanges"]:
            row_database, code = row["input"]
            if row_database == database and code in replacements:
                row["input"] = database, replacements[code]
    result[(database, "mibk_doa")] = {
        **_native_chemical_supplier(
            database, "mibk_doa", "dioctyl adipate production", "dioctyl adipate", "GLO"
        ),
        "exchanges": [
            {"input": (database, "mibk_doa"), "type": "production", "amount": 1.0},
            *portfolio_rows,
            _native_chemical_row(database, "steam_power", 0.3),
            _native_chemical_row(database, "mc_co2_fossil", 0.1, biosphere=True),
        ],
    }


def native_chemical_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build synthetic chemical inventories with distinct production amounts.

    Invented MIBK reaction and utility rows remain available for preservation checks.
    Steam, packaging inputs and Swiss wastewater sources provide the flows and
    supplies required by the package's chemical manufacturing calculations.
    """
    result = chemical_steam_native_inventory(catalogue, database)
    for code, name, product, location, unit in (
        (
            "mibk_h2",
            "market for hydrogen, gaseous, medium pressure, merchant",
            "hydrogen, gaseous, medium pressure, merchant",
            "RER",
            "kilogram",
        ),
        ("mibk_acetone", "market for acetone, liquid", "acetone, liquid", "RER", "kilogram"),
        (
            "mibk_heat",
            "heat production, natural gas, at industrial furnace >100kW",
            "heat, district or industrial, natural gas",
            "Europe without Switzerland",
            "megajoule",
        ),
        (
            "mibk_power",
            "market for electricity, medium voltage",
            "electricity, medium voltage",
            "CH",
            "kilowatt hour",
        ),
        (
            "mibk_cooling_power",
            "market group for electricity, high voltage",
            "electricity, high voltage",
            "RER",
            "kilowatt hour",
        ),
    ):
        result[(database, code)] = _native_chemical_supplier(
            database, code, name, product, location, unit=unit
        )
    result[(database, "mibk_heat")]["exchanges"].append(
        _native_chemical_row(database, "mc_co2_fossil", 0.03, biosphere=True)
    )
    _native_regional_water_sources(result, database)
    for region, basis in (("RER", 2.0), ("RoW", 3.0)):
        code = "mibk_" + region.lower()
        result[(database, code)] = {
            **_native_chemical_supplier(
                database,
                code,
                "methyl isobutyl ketone production, one-step process, from acetone and hydrogen",
                "methyl isobutyl ketone",
                region,
                production=basis,
            ),
            "native_note": "Original MIBK reaction, feed and utilities " + region,
            "comment": "Invented native source: catalyst omitted.",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": basis},
                _native_chemical_row(database, "mibk_acetone", 1.5 * basis),
                _native_chemical_row(database, "mibk_h2", 0.02 * basis),
                _native_chemical_row(database, "steam_power", 0.3 * basis),
                _native_chemical_row(database, "steam_" + region.lower() + "_market", 0.6 * basis),
                _native_chemical_row(database, "market_row", -0.0001 * basis),
                _native_chemical_row(database, "mc_co2_fossil", 0.07 * basis, biosphere=True),
            ],
        }
    add_chemical_organic_water_inventory(result, catalogue, database, native_chemical_biosphere())
    return result
