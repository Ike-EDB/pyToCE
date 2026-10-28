"""Small complete native-shaped UPR receiving boundaries with invented quantities.

Reuse the existing municipal, plastic-leachate, dump and burning inventories.
The fixture adds selected UPR test distinctions: original Al/Co calibration,
omitted B/Si/Ca donor properties, and explicit mineral flow identities.
No licensed inventory is imported and no package receiving result is injected.
"""

from dataclasses import replace
from math import log
from typing import Any

from material_recipe_fixtures import physical_material_recipe
from pa_dump_receiving_fixtures import DUMP_CLASSES, LAND, LAND_ROWS, dump_biosphere
from pa_municipal_fixtures import pa_municipal_biosphere, pa_municipal_native_inventory
from pa_open_burning_fixtures import open_burning_biosphere, open_burning_native_inventory
from sbr_landfill_fixtures import biosphere_inventory as plastic_biosphere
from sbr_landfill_fixtures import native_inventory as plastic_inventory
from tps_gas_fixtures import flow_key

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    EcoinventMaterialRecipe,
    PrincipalExit,
)

ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
MINERALS = {
    "Boron": "B",
    "Silicon": "Si",
    "Calcium II": "Ca",
    "Aluminium III": "Al",
    "Cobalt II": "Co",
}
SURFACE = ("water", "surface water")
LONG = ("water", "ground-, long-term")
RURAL = ("air", "non-urban air or from high stacks")
URBAN = ("air", "urban air close to ground")
AGRICULTURAL = ("soil", "agricultural")


def upr_recipe() -> EcoinventMaterialRecipe:
    """Use the admitted ConversionRequirement and its material composition."""
    conversion = ConversionRequirement(
        "others_upr_hand_lay_up_conversion",
        43163838324174400 / 35680245530124351,
        67075941681801 / 205053863772800,
        (
            PrincipalExit("styrene_air", 8208310784 / 84118311615),
            PrincipalExit("post_release_solid", 103474753 / 922564095),
        ),
    )
    recipe = physical_material_recipe()
    return replace(
        recipe,
        conversion=tuple(
            conversion if row.operation == conversion.operation else row
            for row in recipe.conversion
        ),
    )


def receiving_biosphere(family: str) -> dict:
    """Supply exact receiving flow semantics and the finite UPR mineral domain."""
    if family == "municipal":
        data = pa_municipal_biosphere()
    elif family == "sanitary":
        data = plastic_biosphere()
    else:
        # Canonical keys give each elementary flow a unique identity across the two source fixtures.
        data = {
            flow_key(row["name"], row["categories"]): row
            for row in (
                *dump_biosphere("biosphere").values(),
                *open_burning_biosphere("biosphere").values(),
            )
        }
    existing = {(row["name"], tuple(row["categories"])) for row in data.values()}
    for categories in (
        SURFACE,
        LONG,
        ("water",),
        ("water", "ground-"),
        RURAL,
        URBAN,
        ("air", "low population density, long-term"),
        AGRICULTURAL,
        ("soil", "industrial"),
    ):
        for name in MINERALS:
            if (name, categories) not in existing:
                data[flow_key(name, categories)] = {
                    "name": name,
                    "unit": "kilogram",
                    "categories": categories,
                    "type": "emission",
                }
    return data


def municipal_inventory(catalogue: dict[str, Any], database: str, *, original: bool) -> dict:
    """Keep the full five-node source chain with original-only Al/Co references."""
    data = pa_municipal_native_inventory(catalogue, database, original=original)
    keys = {
        (row["name"], tuple(row["categories"])): key
        for key, row in receiving_biosphere("municipal").items()
    }
    for code in ("phb_furnace", "phb_bottom_terminal", "phb_residual_terminal"):
        node = data[database, code]
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        if original:
            properties = production["properties"]
            properties["aluminium content"] = {"amount": 0.003, "unit": "dimensionless"}
            properties["cobalt content"] = {"amount": 0.0005, "unit": "dimensionless"}
            if code == "phb_furnace":
                properties["oxygen content"]["amount"] -= 0.0035
        # Positive original air/water and short/later ash references are
        # deliberately unrelated to the actual UPR composition/transfer table.
        compartments = (URBAN, SURFACE) if code == "phb_furnace" else (SURFACE, LONG)
        for name in ("Aluminium III", "Cobalt II"):
            for categories, coefficient in zip(compartments, (1e-7, 3e-7), strict=True):
                amount = coefficient * -production["amount"] * (1 if original else 1.8)
                node["exchanges"].append(
                    {
                        "input": keys[name, categories],
                        "type": "biosphere",
                        "amount": amount,
                        "uncertainty type": 2,
                        "loc": log(amount) + 0.03,
                        "scale": 0.21,
                        "minimum": amount / 2,
                        "maximum": amount * 2,
                    }
                )
    return data


def sanitary_inventory(database: str, *, future: bool) -> dict:
    """Retain the complete MIXED-shaped fixture chain with selected mineral references omitted."""
    data = plastic_inventory(database, future=future)
    biosphere = receiving_biosphere("sanitary")
    keys = {(row["name"], tuple(row["categories"])): key for key, row in biosphere.items()}
    collection = data[database, "collection"]
    collection["exchanges"] = [
        row
        for row in collection["exchanges"]
        if row["type"] != "biosphere"
        or biosphere[row["input"]]["name"] not in ("Boron", "Silicon", "Calcium II")
    ]
    for row in collection["exchanges"]:
        if row["type"] == "biosphere" and biosphere[row["input"]]["name"] == "Cobalt II":
            row["input"] = keys["Cobalt II", SURFACE]
    # This fixture uses surface-water Al/Co bypass templates instead of the
    # scaffold's generic-water compartment. B/Si/Ca templates are omitted.
    collection["exchanges"].append(
        {
            "input": keys["Aluminium III", SURFACE],
            "type": "biosphere",
            "amount": 0.013,
            "uncertainty type": 2,
            "loc": log(0.013),
            "scale": 0.1,
        }
    )
    site = data[database, "site"]
    site_basis = -next(row["amount"] for row in site["exchanges"] if row["type"] == "production")
    # The fixture adds Al/Co air and later-water templates with synthetic positive quantities.
    for name in ("Aluminium III", "Cobalt II"):
        for categories, coefficient in ((RURAL, 1e-8), (LONG, 1e-4)):
            amount = site_basis * coefficient * (1.8 if future else 1)
            site["exchanges"].append(
                {
                    "input": keys[name, categories],
                    "type": "biosphere",
                    "amount": amount,
                    "uncertainty type": 2,
                    "loc": log(amount) + 0.03,
                    "scale": 0.13,
                }
            )
    for code in ("thermal_sludge", "thermal_landfarming", "thermal_bottom", "thermal_residual"):
        node = data[database, code]
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        properties = production["properties"]
        # These fixture thermal sources retain Al/Co calibration and omit B/Si/Ca.
        # This fixture omits unreported source properties.
        for element in ("boron", "silicon", "calcium"):
            properties.pop(element + " content", None)
        node["exchanges"] = [
            row
            for row in node["exchanges"]
            if row["type"] != "biosphere"
            or biosphere[row["input"]]["name"] not in ("Boron", "Silicon", "Calcium II")
        ]
        if code == "thermal_bottom":
            properties["dry mass"]["amount"] = 0.802198
            properties["water in wet mass"]["amount"] = 0.197802
        if future:
            # Future fixture records omit intensive source properties, so the
            # compiler uses original calibration while preserving dated operations.
            production["properties"] = {}
    data[database, "site"]["comment"] = (
        "(MAT) of 30 (MAP) of 500"
        if future
        else "(MAT) of 16.743579766537 (MAP) of 1250.84108527132"
    )
    return data


def unmanaged_inventory(catalogue: dict[str, Any], database: str, *, original: bool) -> dict:
    """Use six complete uncontrolled site functions with independent native land rows."""
    data = open_burning_native_inventory(
        database, original=original, biosphere_database="biosphere"
    )
    old_biosphere = open_burning_biosphere("biosphere")
    for row in data[database, "burning"]["exchanges"]:
        if row["type"] == "biosphere":
            flow = old_biosphere[row["input"]]
            row["input"] = flow_key(flow["name"], flow["categories"])
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    for index, (_, suffix, precipitation) in enumerate(DUMP_CLASSES):
        selector = services[f"upr_fossil_open_dump_{suffix}"]["construction"]["source"]
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
        rows.append(
            {"input": flow_key("Methane, fossil", RURAL), "type": "biosphere", "amount": 7.0}
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
