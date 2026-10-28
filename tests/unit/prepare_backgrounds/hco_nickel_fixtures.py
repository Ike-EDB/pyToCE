"""Invented complete native supplies for the reported catalyst preparation vector."""

from math import log
from typing import cast


def hco_nickel_biosphere() -> dict[tuple[str, str], dict[str, object]]:
    """Provide real flow semantics with invented codes and no LCIA factors."""
    return {
        ("biosphere", code): {
            "name": name,
            "unit": unit,
            "categories": ("air",),
            "type": "emission",
        }
        for code, name, unit in (
            ("co2", "Carbon dioxide, fossil", "kilogram"),
            ("nox", "Nitrogen oxides", "kilogram"),
            ("water_air", "Water", "cubic meter"),
            ("marker", "Invented source burden", "kilogram"),
        )
    }


def hco_nickel_native_inventory(
    catalogue: dict[str, object],
    database: str,
    *,
    year: int = 2025,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build dated material supplies and a gas-fired furnace with nonunit output.

    The furnace's gas requirement varies independently of its production basis.
    Its original gas heating-value property stays fixed. Full linked capital,
    fuel and environmental rows expose lost support or a doubled fuel purchase.
    Every non-production exchange retains uncertainty and allocation metadata.
    """
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    construction = cast(dict[str, object], services["hco_supported_nickel"]["construction"])
    supplies = cast(dict[str, dict[str, object]], construction["supplies"])
    original = year == 2025
    output = 2.0 if original else 5.0
    marker_per_unit = 0.001 if original else 0.0006
    records: dict[tuple[str, str], dict[str, object]] = {}

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, object]:
        """Give native rows source uncertainty independent of deterministic recipe demands."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "amount": amount,
            "type": "biosphere" if biosphere else "technosphere",
            "uncertainty type": 2,
            "loc": log(amount) + 0.02,
            "scale": 0.19,
            "allocation": "invented native allocation",
        }

    for label, selector in supplies.items():
        key = database, "hco_" + label
        rows: list[dict[str, object]] = [
            {"input": key, "amount": output, "type": "production"},
            exchange("marker", output * marker_per_unit, biosphere=True),
        ]
        if label == "heat":
            gas = exchange("hco_gas", (0.04 if original else 0.035) * output)
            gas["properties"] = {"heating value, net": {"amount": 36.0, "unit": "MJ"}}
            rows.extend(
                (
                    gas,
                    exchange("hco_factory", 0.002 * output),
                    exchange("co2", (0.08 if original else 0.07) * output, biosphere=True),
                )
            )
        periods = cast(dict[str, str], selector.get("location_by_period", {}))
        records[key] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": periods["baseline" if original else "future"]
            if periods
            else selector["exact_location"],
            "type": "process",
            "comment": "Invented complete supply, not licensed LCI.",
            "exchanges": rows,
        }
    for code, product, unit, reference in (
        ("hco_gas", "natural gas, high pressure", "cubic meter", 3.0),
        ("hco_factory", "fixture furnace capital", "unit", 4.0),
    ):
        key = database, code
        records[key] = {
            "name": "fixture " + product,
            "reference product": product,
            "unit": unit,
            "location": "RER",
            "type": "process",
            "exchanges": [
                {"input": key, "type": "production", "amount": reference},
                exchange("marker", reference * marker_per_unit, biosphere=True),
            ],
        }
    return records
