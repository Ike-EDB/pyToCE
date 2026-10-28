"""Small municipal sources with invented loads and independently dated suppliers."""

import math
from typing import Any

LAND_NAMES = (
    "Occupation, construction site",
    "Occupation, industrial area",
    "Occupation, urban/industrial fallow (non-use)",
    "Transformation, from pasture, man made",
    "Transformation, to industrial area",
    "Transformation, to urban/industrial fallow (non-use)",
)


def water_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Use real flow semantics with invented codes and no licensed factors."""
    descriptors = [
        ("surface", "Water", "cubic meter", ("water", "surface water")),
        ("water_urban", "Water", "cubic meter", ("air", "urban air close to ground")),
        ("water_rural", "Water", "cubic meter", ("air", "non-urban air or from high stacks")),
        ("heat_urban", "Heat, waste", "megajoule", ("air", "urban air close to ground")),
        ("heat_rural", "Heat, waste", "megajoule", ("air", "non-urban air or from high stacks")),
        ("waste", "Non-hazardous waste disposed", "kilogram", ("inventory indicator", "waste")),
        ("co2", "Carbon dioxide, fossil", "kilogram", ("air",)),
        ("cod", "COD, Chemical Oxygen Demand", "kilogram", ("water", "surface water")),
        *[
            (
                f"land_{i}",
                name,
                "square meter-year" if name.startswith("Occupation,") else "square meter",
                ("natural resource", "land"),
            )
            for i, name in enumerate(LAND_NAMES)
        ],
    ]
    return {
        ("biosphere", code): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
        for code, name, unit, categories in descriptors
    }


def water_native_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build all four selected native plant/market pairs with nonunit outputs.

    Only originals carry climate; future sources change supplier identities and
    electricity shares. The upstream electricity emits fossil CO2 even when
    target pure water removes all direct sewage-derived carbon. Europe has
    no purchased native heat, so its declared full external heating supplier
    is separate. Shared native treatment fixtures retain the same RoW keys and amounts.
    """
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    data: dict[tuple[str, str], dict[str, Any]] = {}

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, Any]:
        """Give a source exchange signed deterministic metadata for preservation."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 0,
            "loc": amount,
            "comment": "Invented source quantity.",
        }

    def add(
        code: str, selector: dict[str, Any], production: float, rows: list[dict[str, Any]]
    ) -> None:
        """Store one complete source activity on the supplied reference-output basis."""
        data[(database, code)] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [
                {
                    "input": (database, code),
                    "type": "production",
                    "amount": production,
                    "properties": {"wet mass": {"amount": 1010}, "dry mass": {"amount": 10}},
                },
                *rows,
            ],
        }

    for code, product, unit in (
        ("sewer_rural", "sewer grid, rural", "kilometer"),
        ("sewer_urban", "sewer grid, urban", "kilometer"),
        ("limestone", "limestone, crushed, washed", "kilogram"),
        ("plant", "wastewater treatment plant", "unit"),
        ("screenings", "screenings", "kilogram"),
        ("sand", "sand", "kilogram"),
        ("iron", "iron sulfate", "kilogram"),
        ("pam", "polyacrylamide", "kilogram"),
        ("sludge", "sewage sludge, 70% water", "kilogram"),
        ("electricity_a", "electricity, medium voltage", "kilowatt hour"),
        ("electricity_b", "electricity, medium voltage", "kilowatt hour"),
        ("heat_a", "heat, district or industrial, natural gas", "megajoule"),
        ("heat_b", "heat, district or industrial, natural gas", "megajoule"),
    ):
        # Future fixture heat suppliers use the generic industrial-heat product
        # while retaining their MJ unit and amounts.
        if not original and code.startswith("heat_"):
            product = "heat, district or industrial"
        add(
            code,
            {
                "name": code + (" original" if original else " future"),
                "reference_product": product,
                "unit": unit,
                "exact_location": "CH" if original else "RoW",
            },
            -1.0 if code in {"screenings", "sludge"} else 1.0,
            [exchange("co2", 0.125 if code.startswith("electricity") else 0.0, biosphere=True)],
        )
    for suffix in ("row", "ch", "ca_qc", "europe"):
        plant = services["hydraulic_water_treatment_" + suffix]["construction"]
        market = services["hydraulic_water_collection_" + suffix]["construction"]
        power_a, power_b = (1.0, 3.0) if original else (3.0, 1.0)
        first = exchange("electricity_a", power_a)
        first.update(
            {
                "uncertainty type": 2,
                "loc": math.log(power_a) + 0.03,
                "scale": 0.2,
                "minimum": power_a / 2,
                "maximum": power_a * 2,
            }
        )
        rows = [
            first,
            exchange("electricity_b", power_b),
            exchange("limestone", 0.008),
            exchange("plant", 0.002),
            exchange("screenings", -0.004),
            exchange("sand", 0.006),
            exchange("iron", 0.3),
            exchange("pam", 0.2),
            exchange("sludge", -0.4),
            exchange("co2", 9.0, biosphere=True),
            exchange("cod", 7.0, biosphere=True),
            *[exchange(f"land_{i}", (i + 1) / 100, biosphere=True) for i in range(6)],
        ]
        if suffix == "europe":
            add(
                "heat_europe",
                plant["source_heat"],
                3.0,
                [exchange("co2", 0.06, biosphere=True)],
            )
        else:
            rows += [exchange("heat_a", 0.1), exchange("heat_b", 0.3)]
        add("plant_" + suffix, plant["selector"], -4.0, rows)
        if original:
            data[(database, "plant_" + suffix)]["comment"] = (
                "Mean annual temperature (MAT) of 10; mean annual precipitation (MAP) of 1000; "
                "actual evapotranspiration (EAT) of 500."
            )
        add(
            "market_" + suffix,
            market["selector"],
            -2.0,
            [
                exchange("plant_" + suffix, -0.5),
                exchange("sewer_rural", 0.02),
                exchange("sewer_urban", 0.04),
                exchange("cod", 12.0, biosphere=True),
            ],
        )
    return data
