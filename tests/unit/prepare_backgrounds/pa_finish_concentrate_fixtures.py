"""Synthetic treatment, mixing and packaging inventories for polyamide finish."""

from math import log
from typing import Any

from chemical_residue_fixtures import chemical_residue_biosphere, chemical_residue_native_inventory


def concentrate_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Add an identifiable mixing and delivery emission to the treatment flows."""
    data = chemical_residue_biosphere()
    data["biosphere", "marker"] = {
        "name": "Fixture supply marker",
        "unit": "kilogram",
        "categories": ("air",),
        "type": "emission",
    }
    return data


def concentrate_native_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Combine synthetic treatment, mixing and freight inventories.

    Mixing and freight have distinct dated quantities and uncertainty records.
    Package calculations construct the fitted container, ingredients and
    treatment processes; these are not supplied as completed test products.
    """
    data = chemical_residue_native_inventory(catalogue, database)
    construction = next(
        row["construction"]
        for row in catalogue["services"]
        if row["provider_service_id"] == "pa6_finish_concentrate"
    )

    def row(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
        """Give synthetic exchanges identifiable distributions and allocation labels."""
        return {
            "input": ("biosphere" if bio else database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": log(abs(amount)) + 0.04,
            "scale": 0.18,
            "negative": amount < 0,
            "minimum": min(amount / 4, amount * 4),
            "maximum": max(amount / 4, amount * 4),
            "allocation": "invented concentrate operation",
        }

    def process(
        code: str,
        product: str,
        exchanges: list[dict[str, Any]],
        *,
        unit: str = "kilogram",
        production: float = 1.0,
        selector: dict[str, Any] | None = None,
    ) -> None:
        """Create a synthetic supplier with the specified name and reference amount."""
        description = selector or {
            "name": "invented " + code,
            "reference_product": product,
            "unit": unit,
            "exact_location": "RER",
        }
        data[database, code] = {
            "name": description["name"],
            "reference product": description["reference_product"],
            "unit": description["unit"],
            "location": description["exact_location"],
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": production},
                *exchanges,
            ],
        }

    for code, product, unit in (
        ("pac_power", "electricity, medium voltage", "kilowatt hour"),
        ("pac_factory", "chemical factory, organics", "unit"),
        ("pac_nitrogen", "nitrogen, liquid", "kilogram"),
        ("pac_paint_feed", "alkyd resin", "kilogram"),
        ("pac_paint_waste", "waste paint", "kilogram"),
        ("pac_lubricant", "lubricating oil", "kilogram"),
        ("pac_filled_road", "transport, freight, lorry", "ton kilometer"),
        ("pac_filled_rail", "transport, freight, rail", "ton kilometer"),
        ("pac_filled_ship", "transport, freight, ship", "ton kilometer"),
    ):
        process(
            code,
            product,
            [row("marker", 0.2 if original else 0.1, bio=True)],
            unit=unit,
            production=-2.0 if code == "pac_paint_waste" else 1.0,
        )
    basis = 4.0 if original else 7.0
    process(
        "pac_mixing",
        "",
        [
            row("pac_power", (0.13 if original else 0.21) * basis),
            row("pac_factory", (0.002 if original else 0.003) * basis),
            row("pac_nitrogen", (0.03 if original else 0.04) * basis),
            row("cr_nitrogen_air", (0.03 if original else 0.04) * basis, bio=True),
            row("pac_paint_feed", 1.3 * basis),
            row("pac_paint_waste", -0.08 * basis),
            row("mc_co2_fossil", 0.06 * basis, bio=True),
        ],
        production=basis,
        selector=construction["mixing_source"],
    )
    cargo_basis = 3.0 if original else 6.0
    dated = 1.0 if original else 0.7
    process(
        "pac_delivery",
        "",
        [
            row("pac_filled_road", 0.083 * cargo_basis * dated),
            row("pac_filled_rail", 0.012 * cargo_basis * dated),
            row("pac_filled_ship", 0.031 * cargo_basis * dated),
            row("pac_lubricant", cargo_basis),
        ],
        production=cargo_basis,
        selector=construction["delivery_source"],
    )
    return data
