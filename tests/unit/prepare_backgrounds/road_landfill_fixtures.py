"""Synthetic paint-treatment inventories for filled road-marking material.

All operating amounts are synthetic. This fixture deliberately omits assay and
release rows for Ti/Si/Ca/Al/K/Mg/Na/S so tests can check releases calculated
from the elements introduced by road material.
"""

from math import log

from epoxy_landfill_fixtures import OWNERS as OWNERS
from epoxy_landfill_fixtures import biosphere_inventory as paint_biosphere
from epoxy_landfill_fixtures import native_inventory as paint_inventory
from material_recipe_fixtures import physical_material_recipe
from pa_dump_receiving_fixtures import DUMP_CLASSES, LAND, LAND_ROWS, dump_biosphere
from tps_gas_fixtures import flow_key

ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
RURAL = ("air", "non-urban air or from high stacks")
URBAN = ("air", "urban air close to ground")
SURFACE = ("water", "surface water")
LONG = ("water", "ground-, long-term")
RAW = ("water",)
SOIL = ("soil", "agricultural")
MINERALS = {
    "Silicon": ("Si", "silicon"),
    "Calcium II": ("Ca", "calcium"),
    "Aluminium III": ("Al", "aluminium"),
    "Titanium ion": ("Ti", "titanium"),
    "Potassium I": ("K", "potassium"),
    "Magnesium": ("Mg", "magnesium"),
    "Sodium I": ("Na", "sodium"),
}


def road_conversion():
    """Obtain the admitted road conversion from the parsed material recipe fixture."""
    return next(
        row
        for row in physical_material_recipe().conversion
        if row.operation == "others_road_marking_conversion"
    )


def receiving_biosphere() -> dict:
    """Provide treatment flow names, units and compartments with synthetic IDs."""
    data = paint_biosphere()
    for row in dump_biosphere("biosphere").values():
        data[flow_key(row["name"], row["categories"])] = row
    for categories in (
        RURAL,
        URBAN,
        SURFACE,
        LONG,
        RAW,
        SOIL,
        ("water", "ground-"),
        ("air", "low population density, long-term"),
        ("soil", "industrial"),
    ):
        for name in MINERALS:
            data[flow_key(name, categories)] = {
                "name": name,
                "unit": "kilogram",
                "categories": categories,
                "type": "emission",
            }
    # Elemental sulfur and aqueous sulfate represent different chemical masses.
    for name, categories in (
        ("Sulfur", RAW),
        ("Sulfur", SURFACE),
        ("Sulfur", SOIL),
        ("Sulfur dioxide", URBAN),
    ):
        data[flow_key(name, categories)] = {
            "name": name,
            "unit": "kilogram",
            "categories": categories,
            "type": "emission",
        }
    return data


def sanitary_inventory(database: str, *, future: bool) -> dict:
    """Supply synthetic paint inventories with the selected mineral rows omitted."""
    data = paint_inventory(database, future=future)
    biosphere = receiving_biosphere()
    absent = {*MINERALS, "Sulfur", "Sulfate", "Sulfur dioxide"}
    for code, _, _ in OWNERS.values():
        node = data[database, code]
        node["exchanges"] = [
            row
            for row in node["exchanges"]
            if row["type"] != "biosphere" or biosphere[row["input"]]["name"] not in absent
        ]
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        properties = production.get("properties", {})
        for property_name in (*(name for _, name in MINERALS.values()), "sulfur"):
            removed = properties.pop(f"{property_name} content", None)
            if removed is not None and "oxygen content" in properties:
                properties["oxygen content"]["amount"] += removed["amount"]
    return data


def dump_inventory(catalogue: dict, database: str, *, original: bool) -> dict:
    """Supply five original climates and distinct future land-use quantities."""
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    data = {}
    for index, (_, suffix, precipitation) in enumerate(DUMP_CLASSES):
        selector = services[f"road_fossil_open_dump_{suffix}"]["construction"]["source"]
        key = database, f"dump_{index}"
        production = -(index + (2 if original else 7))
        rows = [{"input": key, "type": "production", "amount": production}]
        for name, _, coefficient in LAND_ROWS:
            for fraction in (0.2, 0.8):
                amount = -production * coefficient * fraction * (1 if original else 1.7)
                rows.append(
                    {
                        "input": flow_key(name, LAND),
                        "type": "biosphere",
                        "amount": amount,
                        "uncertainty type": 2,
                        "loc": log(amount) + 0.03,
                        "scale": 0.12,
                    }
                )
        data[key] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "exchanges": rows,
            "comment": (
                f"(MAT) of {8 if original else 31} (MAP) of {precipitation if original else 1}"
            ),
        }
    return data
