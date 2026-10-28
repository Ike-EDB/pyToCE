"""Synthetic source inventories for the PA ethoxylation compiler."""

import math
from typing import cast

PA_ETHOXYLATION_SERVICES = ("pa_oleth10", "pa_peg40_hco")


def ethoxylation_biosphere() -> dict[tuple[str, str], dict[str, object]]:
    """Keep source flow semantics with invented identifiers and coefficients."""
    descriptions = (
        ("co2_fossil", "Carbon dioxide, fossil", "kilogram", ("air",)),
        ("co2_bio", "Carbon dioxide, non-fossil", "kilogram", ("air",)),
        ("eo_air", "Ethylene oxide", "kilogram", ("air",)),
        ("mist", "Octaethylene glycol monododecyl ether", "kilogram", ("air",)),
        ("alcohol_water", "Dodecanol", "kilogram", ("water",)),
        ("water_air", "Water", "cubic meter", ("air",)),
        ("water_surface", "Water", "cubic meter", ("water", "surface water")),
        ("marker", "Fixture supply marker", "kilogram", ("air",)),
    )
    return {
        ("biosphere", code): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
        for code, name, unit, categories in descriptions
    }


def ethoxylation_native_inventory(
    catalogue: dict[str, object], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, object]]:
    """Make nonunit original/future operations with different supplier shares.

    All exchange quantities are synthetic. The main operation includes
    four signed wastewater links, both environmental proxies, water emissions
    and a factory with a marker flow. The complete operating inventory remains
    separate from oleyl alcohol, hydrogenated castor oil and ethylene oxide inputs.
    """
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    construction = cast(dict[str, object], services[PA_ETHOXYLATION_SERVICES[0]]["construction"])
    data: dict[tuple[str, str], dict[str, object]] = {}
    output = 2.0 if original else 5.0

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, object]:
        """Attach observable signed lognormal metadata and a source allocation."""
        row: dict[str, object] = {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(abs(amount)) + 0.03,
            "scale": 0.21,
            "negative": amount < 0,
            "minimum": min(amount / 2, amount * 2),
            "maximum": max(amount / 2, amount * 2),
            "allocation": "invented source allocation",
        }
        if biosphere and code == "co2_fossil":
            row.update(name="Carbon dioxide, fossil", unit="kilogram", categories=("air",))
            if original:
                row.update(
                    flow=code,
                    properties={
                        "carbon content, fossil": {"amount": 0.27, "unit": "dimensionless"},
                        "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                        "carbon allocation": {"amount": 0.27, "unit": "kilogram"},
                        "dry mass": {"amount": 1.0, "unit": "kilogram"},
                        "wet mass": {"amount": 1.0, "unit": "kilogram"},
                    },
                )
        return row

    def process(
        code: str,
        product: str,
        rows: list[dict[str, object]],
        *,
        production: float = 1.0,
        unit: str = "kilogram",
        selector: dict[str, object] | None = None,
    ) -> None:
        """Create an ordinary Brightway process, including nonunit treatment outputs."""
        descriptor = selector or {
            "name": f"fixture {code}",
            "reference_product": product,
            "exact_location": "GLO",
            "unit": unit,
        }
        data[(database, code)] = {
            "name": descriptor["name"],
            "reference product": descriptor["reference_product"],
            "location": descriptor["exact_location"],
            "unit": descriptor["unit"],
            "type": "process",
            "exchanges": [
                {"input": (database, code), "amount": production, "type": "production"},
                *rows,
            ],
        }

    for code, product, unit in (
        ("donor_alcohol", "dodecanol", "kilogram"),
        ("eo_a", "ethylene oxide", "kilogram"),
        ("eo_b", "ethylene oxide", "kilogram"),
        ("electricity", "electricity, medium voltage", "kilowatt hour"),
        ("heat", "heat, from steam, in chemical industry", "megajoule"),
        ("factory", "chemical factory, organics", "unit"),
    ):
        process(code, product, [exchange("marker", 0.1, biosphere=True)], unit=unit)
    for index in range(4):
        process(
            f"water_{index}",
            "wastewater, average",
            [exchange("marker", 0.02, biosphere=True)],
            production=-2.0,
            unit="cubic meter",
        )
    eo_share = 0.6 if original else 0.25
    main = [
        exchange("donor_alcohol", 0.4 * output),
        exchange("eo_a", 0.68 * output * eo_share),
        exchange("eo_b", 0.68 * output * (1 - eo_share)),
        exchange("factory", 0.002 * output),
        exchange("electricity", (0.7 if original else 0.9) * output),
        exchange("heat", 2.5 * output),
        *(exchange(f"water_{i}", -(i + 1) * 0.0001 * output) for i in range(4)),
        exchange("co2_fossil", (0.04 if original else 0.07) * output, biosphere=True),
        exchange("eo_air", 0.002 * output, biosphere=True),
        exchange("mist", (0.006 if original else 0.009) * output, biosphere=True),
        exchange("alcohol_water", 0.001 * output, biosphere=True),
        exchange("water_air", 0.003 * output, biosphere=True),
        exchange("water_surface", 0.004 * output, biosphere=True),
    ]
    process(
        "ethoxylation",
        "non-ionic surfactant",
        main,
        production=output,
        selector=cast(dict[str, object], construction["source"]),
    )
    selector = cast(dict[str, object], construction["oleyl_alcohol_source"])
    process(
        "oleyl",
        str(selector["reference_product"]),
        [exchange("marker", 0.1, biosphere=True)],
        selector=selector,
    )
    return data
