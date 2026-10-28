"""Synthetic Brightway inventories for the three TPS burning formulations."""

from typing import Any

from pa_open_burning_fixtures import (
    _AIR,
    _SOIL,
    open_burning_biosphere,
    open_burning_native_inventory,
)

ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
SERVICE_IDS = tuple(f"tps_open_burning_{origin}" for origin in ORIGINS)
_DIOXIN = "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin"
# The fixture declares PA, TPS salt, and organochlorine endpoints before calculation.
_TPS_FLOWS = (
    ("Hydrochloric acid", "kilogram", _AIR),
    ("Polychlorinated biphenyls", "kilogram", _AIR),
    (_DIOXIN, "kilogram", _AIR),
    (_DIOXIN, "kilogram", _SOIL),
    ("Chloride", "kilogram", _SOIL),
    ("Magnesium", "kilogram", _AIR),
    ("Magnesium", "kilogram", _SOIL),
    ("Zinc II", "kilogram", _AIR),
    ("Zinc II", "kilogram", _SOIL),
)


def tps_burning_biosphere(database: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Extend the complete existing burning fixture with exact TPS flow identities."""
    result = open_burning_biosphere(database)
    result.update(
        {
            (database, f"tps_{index}"): {
                "name": name,
                "unit": unit,
                "categories": categories,
                "type": "emission",
            }
            for index, (name, unit, categories) in enumerate(_TPS_FLOWS)
        }
    )
    return result


def tps_burning_native_inventory(
    database: str, *, original: bool, biosphere_database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep original and dated donors with distinct signed, nonunit waste outputs.

    The source amounts are invented controls. P and Mg donor emission rows are
    deliberately omitted, while the fixture biosphere retains their target
    descriptors. The complete TPS operation uses the declared actual material composition.
    """
    result = open_burning_native_inventory(
        database, original=original, biosphere_database=biosphere_database
    )
    biosphere = tps_burning_biosphere(biosphere_database)
    node = result[database, "burning"]
    rows = node["exchanges"]
    rows[0]["amount"] = -3.0 if original else -7.0
    node["exchanges"] = [
        row
        for row in rows
        if row["type"] == "production"
        or biosphere[row["input"]]["name"] not in {"Phosphorus", "Magnesium"}
    ]
    node["exchanges"].extend(
        {
            "input": key,
            "type": "biosphere",
            "amount": 0.003 if original else 0.009,
            "uncertainty type": 0,
            "loc": 0.003 if original else 0.009,
        }
        for key, flow in biosphere.items()
        if key[1].startswith("tps_") and flow["name"] != "Magnesium"
    )
    return result
