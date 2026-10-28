"""Synthetic PET finishing inventories with original and dated fuel quantities."""

from typing import Any


def pet_finishing_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool = False, future: bool = False
) -> dict[tuple[str, str], dict[str, Any]]:
    """Supply synthetic finishing inventories, including 17 propane suppliers.

    Dated propane input rows omit heating-value properties. Original propane
    quantities differ; liquefied petroleum gas has allocated carbon that differs
    from its unweighted self-input. The package must use original fuel quality and
    dated operating quantities. The future fixture supplies an EUR electricity
    market group; its RER market links the APR electricity requirement to that group.
    """
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    food = services["recycling_pet_food_grade_finishing"]["construction"]
    data: dict[tuple[str, str], dict[str, Any]] = {}

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, Any]:
        """Make a signed synthetic exchange whose uncertainty must remain intact."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 3,
            "loc": amount + 0.012,
            "scale": 0.004,
            "comment": "Invented unmodified native record.",
        }

    def add(
        code: str, selector: dict[str, Any], production: float, exchanges: list[dict[str, Any]]
    ) -> None:
        """Add a synthetic activity with an explicit reference production amount."""
        data[(database, code)] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": production},
                *exchanges,
            ],
        }

    for code, unit, production in (
        ("water", "kilogram", 1.0),
        ("wastewater", "cubic meter", -2.0),
        ("capital", "unit", 1.0),
        ("gas", "cubic meter", 1.0),
        ("nitrogen_supply", "kilogram", 1.0),
        ("flake", "kilogram", 1.0),
        ("reject", "kilogram", -1.0),
    ):
        add(
            code,
            {"name": code, "reference_product": code, "unit": unit, "exact_location": "RER"},
            production,
            [exchange("co2", 100.0 if code in {"flake", "reject"} else 0.7, biosphere=True)],
        )
    for code, name, product in (
        (
            "flake",
            "market for polyethylene terephthalate, flakes, recycled",
            "polyethylene terephthalate, flakes, recycled",
        ),
        ("reject", "market group for waste plastic, mixture", "waste plastic, mixture"),
        ("gas", "market group for natural gas, high pressure", "natural gas, high pressure"),
    ):
        data[(database, code)].update({"name": name, "reference product": product})
    data[(database, "gas")]["location"] = "EUR" if future else "Europe without Switzerland"

    for i in range(10):
        add(
            f"electricity_{i}",
            {
                "name": "market group for electricity, low voltage"
                if future or i % 2 == 0
                else "market for electricity, low voltage",
                "reference_product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "exact_location": "RER" if i == 0 else f"invented_{i}",
            },
            1.0,
            [exchange("co2", 0.5, biosphere=True)],
        )
    if future:
        add(
            "electricity_iam",
            {
                "name": "market group for electricity, low voltage",
                "reference_product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "exact_location": "EUR",
            },
            1.0,
            [exchange("co2", 0.5, biosphere=True)],
        )
        data[(database, "electricity_0")]["exchanges"][1:] = [exchange("electricity_iam", 1.0)]
    for i in range(17):
        add(
            f"propane_{i}",
            {
                "name": "market for propane",
                "reference_product": "propane",
                "unit": "kilogram",
                "exact_location": f"invented_{i}",
            },
            1.0,
            [exchange("co2", 0.7, biosphere=True)],
        )
    finish = services["recycling_pet_finishing_support"]["construction"]["base"]["selector"]
    add(
        "finisher",
        finish,
        2.0,
        [
            exchange("flake", 3.0),
            exchange("reject", -0.2),
            exchange("electricity_iam" if future else "electricity_0", 0.9),
            exchange("water", 0.03),
            exchange("wastewater", -0.04),
            exchange("capital", 0.002),
            exchange("pm", 0.004, biosphere=True),
        ],
    )
    for fuel, production, fuel_exchanges in (
        ("natural_gas", 2.0, [exchange("gas", 0.2)]),
        (
            "propane",
            3.0,
            [exchange(f"propane_{i}", (i + 1) * (0.002 if original else 0.001)) for i in range(17)],
        ),
    ):
        base = services[f"pet_finishing_{fuel}_combustion"]["construction"]["base"]["selector"]
        if original and fuel == "propane":
            for record in fuel_exchanges:
                record["properties"] = {"heating value, net": {"amount": 40.0}}
        add(
            fuel,
            base,
            production,
            [
                *fuel_exchanges,
                *(
                    [exchange("electricity_iam", 0.55 if fuel == "propane" else 0.01)]
                    if future
                    else [
                        exchange(f"electricity_{i}", 0.01 * (i + 1))
                        for i in range(10 if fuel == "propane" else 1)
                    ]
                ),
                exchange("capital", 0.002),
                exchange("co2", 0.08, biosphere=True),
                exchange("sulfur", 0.003, biosphere=True),
                exchange("pm", 0.004, biosphere=True),
                exchange("pm", -0.001, biosphere=True),
                exchange("pm", 0.0, biosphere=True),
            ],
        )
    nitrogen = next(
        item
        for item in food["components"]
        if item.get("component_id") == "nitrogen_supply_and_vent"
    )
    add(
        "nitrogen_vent",
        nitrogen["selector"],
        2.0,
        [exchange("nitrogen_supply", 2.0), exchange("nitrogen", 2.0, biosphere=True)],
    )
    lpg_exchanges = [exchange("co2", 0.7, biosphere=True)]
    if original:
        lpg_input = exchange("lpg", 0.1)
        lpg_input["properties"] = {
            "heating value, net": {"amount": 45.0},
            "carbon content, fossil": {"amount": 0.8},
        }
        lpg_exchanges.append(lpg_input)
    add("lpg", food["lpg"]["selector"], 1.0, lpg_exchanges)
    data[(database, "lpg")]["exchanges"][0]["properties"] = {
        "heating value, net": {"amount": 45.0},
        "carbon content, fossil": {"amount": 0.04},
    }
    return data
