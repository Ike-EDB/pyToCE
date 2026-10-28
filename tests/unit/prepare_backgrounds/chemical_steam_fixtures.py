"""Synthetic supply inventories for steam used in chemical production."""

import json
from pathlib import Path
from typing import Any

from tests.unit.prepare_backgrounds.chemical_residue_fixtures import (
    chemical_residue_biosphere,
    chemical_residue_native_inventory,
)


def chemical_steam_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Combine treatment flow names and synthetic steam-production emissions."""
    packet = json.loads(
        (Path(__file__).parents[2] / "fixtures/chemical_steam/native_operations.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        **chemical_residue_biosphere(),
        **{("biosphere", code): record for code, record in packet["biosphere"].items()},
    }


def chemical_steam_native_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add European and rest-of-world steam and drum supplies to waste treatment.

    Quantities and codes are synthetic. Different reference production amounts
    and regional shares test normalization by the correct production amount.
    Original upstream supplies, utilities and waste treatment remain linked.
    """
    packet = json.loads(
        (Path(__file__).parents[2] / "fixtures/chemical_steam/native_operations.json").read_text(
            encoding="utf-8"
        )
    )
    result = chemical_residue_native_inventory(catalogue, database)
    for code, record in packet["activities"].items():
        record["exchanges"] = [
            {
                **row,
                "input": ("biosphere" if row["type"] == "biosphere" else database, row["input"]),
            }
            for row in record["exchanges"]
        ]
        result[(database, code)] = record
    return result
