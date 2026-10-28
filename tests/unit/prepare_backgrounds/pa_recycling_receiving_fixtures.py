"""Invented complete WEEE, PET cooling and cargo-delivery source graphs."""

from math import log
from typing import Any

from tests.unit.prepare_backgrounds.recovery_delivery_fixtures import (
    recovery_biosphere,
    recovery_native_inventory,
)

RECYCLING_SERVICES = (
    "recycling_weee_preparation",
    "recycling_pet_finishing",
    "recycling_pet_pellet_delivery",
)


def recycling_native_inventory(catalogue: dict[str, Any], database: str, *, future: bool) -> dict:
    """Extend native delivery sources with selected recycling duties.

    Two tap-water suppliers and two PM rows preserve native row ownership.
    Nonunit source productions differ across dates; original water calibration
    is deliberately different from the later operating row amounts. Omitted
    feed/residue/solvent rows have nonzero sentinels, so their retention cannot
    be hidden by a zero-valued fixture.
    """
    data = recovery_native_inventory(database)
    services = {item["provider_service_id"]: item for item in catalogue["services"]}

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict:
        """Supply signed lognormal amounts with a distribution location parameter."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": log(abs(amount)) + 0.02,
            "scale": 0.13,
            "negative": amount < 0,
        }

    def add(code: str, selector: dict, production: float, rows: list[dict]) -> None:
        """Write one complete fixture activity with a defined reference product."""
        data[database, code] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": production},
                *rows,
            ],
        }

    # Distinct geographic suppliers form a portfolio for the same product.
    add(
        "second_water",
        {
            "name": "market for tap water",
            "reference_product": "tap water",
            "unit": "kilogram",
            "exact_location": "CH",
        },
        1,
        [exchange("upstream_electricity", 0.003)],
    )
    for code, product in (
        ("wash_water_receiving", "wastewater from recycling of waste PP"),
        ("cooling_water_receiving", "wastewater from PET pelletising"),
    ):
        add(
            code,
            {
                "name": "market for " + product,
                "reference_product": product,
                "unit": "cubic meter",
                "exact_location": "RER",
            },
            -3,
            [exchange("wastewater_market", -2), exchange("electricity", 0.1)],
        )

    for index, service_id in enumerate(RECYCLING_SERVICES):
        construction = services[service_id]["construction"]
        production = (5.0 if future else 2.0) if index < 2 else 4.0
        omitted = []
        for number, rule in enumerate(construction["omit_technosphere"]):
            selector = rule["selector"]
            code = f"omitted_{index}_{number}"
            # Delivery's principal supplier is the same actual pelletizer;
            # this preserves the source graph instead of defining an alias.
            if index == 2:
                code = RECYCLING_SERVICES[1]
            else:
                add(
                    code,
                    {**selector, "exact_location": selector["location"]},
                    1,
                    [exchange("co2", 100, biosphere=True)],
                )
            if selector["reference_product"] == "waste plastic, WEEE, unsorted":
                # Match the invented EcoSpold basis: two wet input kg/ref kg,
                # with 80% dry matter. Future properties are deliberately absent.
                feed = exchange(code, 2 * production)
                if not future:
                    feed["properties"] = {"dry mass": {"amount": 0.8, "unit": "kg"}}
                omitted.append(feed)
            elif selector["reference_product"] == "polyethylene terephthalate, flakes, recycled":
                omitted.append(exchange(code, 1.2 * production))
            else:
                omitted.append(exchange(code, 0.7))
        if index == 0:
            water = 30.0 if future else 9.6
            rows = [
                exchange("electricity", production * (0.8 if future else 0.6)),
                exchange("tap_water", water * 0.75),
                exchange("second_water", water * 0.25),
                exchange("wash_water_receiving", -water / 1000),
                exchange("capital", production * 0.004),
                exchange("pm", production * 0.00024, biosphere=True),
                exchange("pm", production * 0.00016, biosphere=True),
                exchange("solvent", 100, biosphere=True),
                *omitted,
            ]
        elif index == 1:
            rows = [
                exchange("electricity", production * (1.8 if future else 1.5)),
                exchange("tap_water", production * 0.0004),
                exchange("cooling_water_receiving", -production * 0.0004 / 1000),
                exchange("capital", production * 0.002),
                exchange("pm", production * 0.002, biosphere=True),
                *omitted,
            ]
        else:
            rows = [exchange("freight", production * 0.03), *omitted]
        add(service_id, construction["base"]["selector"], production, rows)
    return data


def recycling_biosphere() -> dict:
    """Reuse native flow descriptors with invented keys and no copied factors."""
    biosphere = recovery_biosphere()
    for code, name, unit in (
        ("pm", "Particulate Matter, > 10 um", "kilogram"),
        ("solvent", "VOC, volatile organic compounds", "kilogram"),
        ("water", "Water", "cubic meter"),
    ):
        biosphere["biosphere", code] = {
            "name": name,
            "unit": unit,
            "categories": ("air",),
            "type": "emission",
        }
    return biosphere
