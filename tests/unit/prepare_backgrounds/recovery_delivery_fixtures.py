"""Synthetic linked utility and freight supplies for material-recovery tests."""

from typing import Any


def recovery_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Declare fossil carbon dioxide with a synthetic ID and no licensed quantities."""
    return {
        ("biosphere", "co2"): {
            "name": "Carbon dioxide, fossil",
            "unit": "kilogram",
            "categories": ("air",),
            "type": "emission",
        }
    }


def recovery_native_inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Supply synthetic inventories for the package's material-recovery calculations.

    Tests add preparation, finishing and freight inputs for each material.
    These supplies include nonzero upstream emissions for the operation tested. Wastewater
    collection and treatment
    have negative reference amounts of 2 and 5 cubic meters, respectively.
    """
    data: dict[tuple[str, str], dict[str, Any]] = {}
    for code, name, product, unit, location, production, input_key, amount in (
        (
            "electricity",
            "market group for electricity, low voltage",
            "electricity, low voltage",
            "kilowatt hour",
            "RER",
            1.0,
            ("biosphere", "co2"),
            0.1,
        ),
        (
            "upstream_electricity",
            "electricity, medium voltage",
            "electricity, medium voltage",
            "kilowatt hour",
            "GLO",
            1.0,
            ("biosphere", "co2"),
            0.1,
        ),
        (
            "auxiliary",
            "invented process auxiliary",
            "invented process auxiliary",
            "kilogram",
            "GLO",
            1.0,
            ("biosphere", "co2"),
            0.1,
        ),
        (
            "capital",
            "invented process equipment",
            "invented process equipment",
            "unit",
            "GLO",
            1.0,
            ("biosphere", "co2"),
            0.1,
        ),
        (
            "freight",
            "invented freight service",
            "invented freight service",
            "ton kilometer",
            "GLO",
            1.0,
            ("biosphere", "co2"),
            0.1,
        ),
        (
            "tap_water",
            "market for tap water",
            "tap water",
            "kilogram",
            "Europe without Switzerland",
            1.0,
            (database, "upstream_electricity"),
            0.001,
        ),
        (
            "wastewater_treatment",
            "treatment of wastewater, average, wastewater treatment",
            "wastewater, average",
            "cubic meter",
            "Europe without Switzerland",
            -5.0,
            (database, "electricity"),
            2.0,
        ),
        (
            "wastewater_market",
            "market for wastewater, average",
            "wastewater, average",
            "cubic meter",
            "Europe without Switzerland",
            -2.0,
            (database, "wastewater_treatment"),
            -1.5,
        ),
    ):
        key = database, code
        data[key] = {
            "name": name,
            "reference product": product,
            "unit": unit,
            "location": location,
            "type": "process",
            "exchanges": [
                {"input": key, "type": "production", "amount": production},
                {
                    "input": input_key,
                    "type": "biosphere" if input_key[0] == "biosphere" else "technosphere",
                    "amount": amount,
                },
            ],
        }
    return data
