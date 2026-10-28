"""Invented complete native suppliers for castor/HCO parent-owner checks."""

import math
from typing import cast


def castor_hco_biosphere() -> dict[tuple[str, str], dict[str, object]]:
    """Use native flow semantics and synthetic keys."""
    return {
        ("biosphere", key): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
        for key, name, unit, categories in (
            ("water_air", "Water", "cubic meter", ("air",)),
            ("hydrogen", "Hydrogen", "kilogram", ("air",)),
            ("oil_water", "Oils, non-fossil", "kilogram", ("water", "surface water")),
            ("water_surface", "Water", "cubic meter", ("water", "surface water")),
            ("hexane", "Hexane", "kilogram", ("air",)),
            ("correction", "Carbon dioxide, non-fossil, resource correction", "kilogram", ("air",)),
            ("marker", "Fixture supply burden", "kilogram", ("air",)),
        )
    }


def castor_hco_native_inventory(
    catalogue: dict[str, object], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, object]]:
    """Create nonunit native extraction, refinery, seed/freight and material sources.

    Values are invented, including material properties. Every ordinary supply
    has its own observable upstream burden. No fake package operation or waste
    receiver is created. The actual crude/refined-oil frontier has a complete
    native graph; HCO's connected receiving is checked separately.
    """
    services = {
        str(s["provider_service_id"]): s
        for s in cast(list[dict[str, object]], catalogue["services"])
    }
    castor = cast(dict[str, object], services["pa_castor_crude_oil"]["construction"])
    refinery = cast(dict[str, object], services["pa_castor_oil"]["construction"])
    hco = cast(dict[str, object], services["pa_hydrogenated_castor_oil"]["construction"])
    supplies = cast(dict[str, dict[str, object]], castor["supplies"])
    hco_supplies = cast(dict[str, dict[str, object]], hco["supplies"])
    data: dict[tuple[str, str], dict[str, object]] = {}
    output = 2.0 if original else 5.0

    def exchange(
        code: str,
        amount: float,
        *,
        biosphere: bool = False,
        properties: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Give source rows observable uncertainty/allocation metadata to preserve."""
        row: dict[str, object] = {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(abs(amount)) + 0.02,
            "scale": 0.19,
            "negative": amount < 0,
            "allocation": "invented source allocation",
        }
        if properties is not None:
            row["properties"] = properties
        return row

    def process(
        code: str,
        selector: dict[str, object],
        rows: list[dict[str, object]],
        *,
        reference: float = 1.0,
        properties: dict[str, dict[str, object]] | None = None,
    ) -> None:
        """Write an ordinary native-shaped process with its actual reference basis."""
        production: dict[str, object] = {
            "input": (database, code),
            "type": "production",
            "amount": reference,
        }
        if properties is not None:
            production["properties"] = properties
        locations = cast(dict[str, object], selector.get("location_by_period", {}))
        data[database, code] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": locations["baseline" if original else "future"]
            if locations
            else selector["exact_location"],
            "type": "process",
            "exchanges": [production, *rows],
        }

    def descriptor(product: str, unit: str = "kilogram") -> dict[str, object]:
        """Name a native-only fixture supplier that requires no catalog selection."""
        return {
            "name": "fixture " + product,
            "reference_product": product,
            "unit": unit,
            "exact_location": "GLO",
        }

    def property_value(value: float, unit: str = "kg") -> dict[str, object]:
        """Record an invented source property in its declared scientific unit."""
        return {"amount": value, "unit": unit}

    for code, product in (
        ("soybean", "soybean"),
        ("hexane_supply", "hexane"),
        ("mill", "oil mill"),
        ("gas", "natural gas, high pressure"),
        ("seed_india", "castor bean from IN"),
        ("seed_row", "castor bean from RoW"),
    ):
        unit = "cubic meter" if code == "gas" else "unit" if code == "mill" else "kilogram"
        process(code, descriptor(product, unit), [exchange("marker", 0.007, biosphere=True)])
    for code in ("road_a", "road_b", "rail", "ship"):
        process(
            code,
            descriptor("transport, " + code, "ton kilometer"),
            [exchange("marker", 0.003, biosphere=True)],
        )

    process(
        "drying_electricity",
        supplies["drying_electricity"],
        [exchange("marker", 0.02 if original else 0.01, biosphere=True)],
        reference=2,
    )
    process(
        "clay",
        hco_supplies["filter_aid"],
        [exchange("marker", 0.002, biosphere=True)],
        reference=3,
        properties={"dry mass": property_value(0.98)},
    )
    process(
        "hydrogen",
        hco_supplies["hydrogen"],
        [exchange("marker", 0.015, biosphere=True)],
        reference=2,
    )
    process(
        "citric_acid",
        hco_supplies["citric_acid"],
        [exchange("marker", 0.006 if original else 0.003, biosphere=True)],
        reference=3 if original else 5,
    )
    process(
        "steam",
        hco_supplies["steam"],
        [exchange("marker", 0.008 if original else 0.004, biosphere=True)],
        reference=4 if original else 6,
    )
    # HCO's integrated electrical meter and the castor dryer use the same
    # medium-voltage supply already defined above. Keep one native descriptor.
    seed_output = 3.0 if original else 6.0
    freight = (0.03, 0.05, 0.002, 0.1) if original else (0.02, 0.06, 0.004, 0.08)
    process(
        "seed",
        supplies["seed"],
        [
            exchange("seed_india", 0.7 * seed_output),
            exchange("seed_row", 0.3 * seed_output),
            *(
                exchange(code, amount * seed_output)
                for code, amount in zip(("road_a", "road_b", "rail", "ship"), freight, strict=True)
            ),
        ],
        reference=seed_output,
        properties={"dry mass": property_value(0.87)},
    )
    process(
        "drying_heat",
        supplies["drying_heat"],
        [
            exchange(
                "gas", 0.025 * output, properties={"heating value, net": property_value(40.0, "MJ")}
            ),
            exchange("mill", 1e-9 * output),
            exchange("marker", 0.006 * output, biosphere=True),
        ],
        reference=output,
    )
    process(
        "extraction",
        cast(dict[str, object], castor["source"]),
        [
            exchange("soybean", 2.4 * output, properties={"dry mass": property_value(0.89)}),
            exchange("drying_electricity", (0.12 if original else 0.08) * output),
            exchange("hexane_supply", 0.003 * output),
            exchange("mill", 1e-9 * output),
            exchange("hexane", 0.0002 * output, biosphere=True),
            exchange("water_air", 0.0001 * output, biosphere=True),
            exchange("water_surface", 0.0005 * output, biosphere=True),
            exchange("oil_water", 0.002 * output, biosphere=True),
            exchange("correction", -0.03 * output, biosphere=True),
        ],
        reference=output,
        properties={"allocation factor": property_value(0.4, "dimensionless")},
    )
    process(
        "crude_market",
        cast(
            dict[str, object],
            cast(list[dict[str, object]], refinery["replace_technosphere"])[0]["selector"],
        ),
        [exchange("extraction", 1.0)],
    )
    refinery_rows = []
    for code, product, unit, amount in (
        ("refinery_electricity", "electricity, medium voltage", "kilowatt hour", 0.012),
        ("refinery_heat", "heat, district or industrial, natural gas", "megajoule", 0.04),
        ("refinery_naoh", "sodium hydroxide, without water", "kilogram", 0.003),
        ("refinery_water", "tap water", "kilogram", 0.15),
        ("refinery_capital", "vegetable oil refinery", "unit", 1e-10),
    ):
        process(code, descriptor(product, unit), [exchange("marker", 0.007, biosphere=True)])
        refinery_rows.append(exchange(code, amount * (3 if original else 7)))
    process(
        "refinery_wwt",
        descriptor("wastewater from vegetable oil refinery", "cubic meter"),
        [exchange("marker", 0.007, biosphere=True)],
        reference=-1.0,
    )
    process(
        "refinery_wastewater_market",
        descriptor("collected wastewater from vegetable oil refinery", "cubic meter"),
        [exchange("refinery_wwt", -1.0)],
        reference=-1.0,
    )
    refinery_output = 3.0 if original else 7.0
    crude = exchange("crude_market", (1.05 if original else 1.08) * refinery_output)
    crude.update(
        activity="invented original crude-oil market",
        flow="invented crude-soybean-oil product",
        name="soybean oil, crude",
        classifications={
            "By-product classification": "allocatable product",
            "CPC": "21611: Soya bean oil, crude",
            "HS2017": "120110: Soya beans; seed, whether or not broken",
        },
    )
    if original:
        crude["properties"] = {
            "carbon content": property_value(0.768, "dimensionless"),
            "heating value, net": property_value(38.4, "MJ"),
            "dry mass": property_value(1.0),
        }
    process(
        "refinery",
        cast(dict[str, object], cast(dict[str, object], refinery["base"])["selector"]),
        [
            crude,
            *refinery_rows,
            exchange("refinery_wastewater_market", -0.0002 * refinery_output),
            exchange("water_surface", 0.00003 * refinery_output, biosphere=True),
        ],
        reference=refinery_output,
        properties={"allocation factor": property_value(0.99, "dimensionless")}
        if original
        else None,
    )
    production = cast(list[dict[str, object]], data[database, "refinery"]["exchanges"])[0]
    production.update(
        classifications=dict(cast(dict[str, str], crude["classifications"])),
        allocation="invented native refinery allocation",
    )
    return data
