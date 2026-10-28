"""Synthetic source inventories for three PA ester and amide recipes."""

import math
from typing import cast

PA_ESTER_SERVICES = (
    "pa_tridecyl_stearate",
    "pa_diglycerol_dioleate",
    "pa_hmda_dimeramide",
)


def ester_biosphere() -> dict[tuple[str, str], dict[str, object]]:
    """Use actual direct-flow semantics with invented identifiers and marker flows."""
    return {
        ("biosphere", code): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "natural resource" if categories[0] == "natural resource" else "emission",
        }
        for code, name, unit, categories in (
            ("co2", "Carbon dioxide, fossil", "kilogram", ("air",)),
            ("co2_bio", "Carbon dioxide, non-fossil", "kilogram", ("air",)),
            ("marker", "Fixture supply marker", "kilogram", ("air",)),
            ("ethylene_air", "Ethylene", "kilogram", ("air",)),
            ("water_air", "Water", "cubic meter", ("air",)),
            ("bod", "BOD5, Biological Oxygen Demand", "kilogram", ("water", "surface water")),
            ("cod", "COD, Chemical Oxygen Demand", "kilogram", ("water", "surface water")),
            ("doc", "DOC, Dissolved Organic Carbon", "kilogram", ("water", "surface water")),
            ("ethylene_water", "Ethylene", "kilogram", ("water", "surface water")),
            ("toc", "TOC, Total Organic Carbon", "kilogram", ("water", "surface water")),
            ("water_water", "Water", "cubic meter", ("water", "surface water")),
            (
                "cooling",
                "Water, cooling, unspecified natural origin",
                "cubic meter",
                ("natural resource", "in water"),
            ),
            ("river", "Water, river", "cubic meter", ("natural resource", "in water")),
            ("well", "Water, well, in ground", "cubic meter", ("natural resource", "in water")),
        )
    }


def ester_native_inventory(
    catalogue: dict[str, object], database: str, *, future: bool
) -> dict[tuple[str, str], dict[str, object]]:
    """Supply catalog descriptors with synthetic quantities and nonunit production.

    Every whole chemical supplier has an observable upstream marker. The two
    delivery markets contain four dated freight rows and two alternative
    chemical producers, which must be replaced by the actual selected supplier.
    The package compiles treatment and operating inventories in the
    connected PA test; this fixture supplies no replacement package records.
    """
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    data: dict[tuple[str, str], dict[str, object]] = {}
    descriptors: set[tuple[str, str, str, str]] = set()
    output = 5.0 if future else 2.0

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, object]:
        """Provide observable lognormal uncertainty and synthetic allocation metadata."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(abs(amount)) + 0.02,
            "scale": 0.17,
            "minimum": min(amount / 2, amount * 2),
            "maximum": max(amount / 2, amount * 2),
            "allocation": "invented native allocation",
        }

    def process(code: str, selector: dict[str, object], rows: list[dict[str, object]]) -> None:
        """Write one distinct native descriptor with its actual production reference."""
        descriptor = (
            str(selector["name"]),
            str(selector["reference_product"]),
            str(selector["unit"]),
            str(selector["exact_location"]),
        )
        if descriptor in descriptors:
            return
        descriptors.add(descriptor)
        data[(database, code)] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": output},
                *rows,
            ],
        }

    for sid in PA_ESTER_SERVICES:
        construction = cast(dict[str, object], services[sid]["construction"])
        for field, selector in cast(dict[str, dict[str, object]], construction["sources"]).items():
            process(
                field,
                selector,
                [exchange("marker", (0.08 if future else 0.05) * output, biosphere=True)],
            )
    for index, product in enumerate(
        (
            "transport, freight, lorry, unspecified",
            "transport, freight train",
            "transport, freight, sea, transoceanic ship",
            "transport, freight, inland waterways, barge",
        )
    ):
        process(
            f"freight_{index}",
            {
                "name": f"fixture freight {index}",
                "reference_product": product,
                "unit": "ton kilometer",
                "exact_location": "GLO",
            },
            [exchange("marker", 0.01 * output, biosphere=True)],
        )
    for sid in (PA_ESTER_SERVICES[0], PA_ESTER_SERVICES[2]):
        construction = cast(dict[str, object], services[sid]["construction"])
        selector = cast(dict[str, object], construction["freight_source"])
        product = str(selector["reference_product"])
        for index in range(2):
            process(
                f"excluded_{sid}_{index}",
                {
                    "name": f"fixture alternative {sid} producer {index}",
                    "reference_product": product,
                    "unit": "kilogram",
                    "exact_location": "GLO",
                },
                [exchange("marker", 1.0 * output, biosphere=True)],
            )
        process(
            f"market_{sid}",
            selector,
            [
                exchange(f"excluded_{sid}_{index}", (0.3 if index == 0 else 0.7) * output)
                for index in range(2)
            ]
            + [
                exchange(
                    f"freight_{index}",
                    (index + 1) * (0.04 if future else 0.03) * output,
                )
                for index in range(4)
            ],
        )
    for code, product, unit in (
        ("adipic", "adipic acid", "kilogram"),
        ("ethylene", "ethylene", "kilogram"),
        ("factory", "chemical factory, organics", "unit"),
        ("electricity", "electricity, medium voltage", "kilowatt hour"),
        ("heat", "heat, district or industrial, natural gas", "megajoule"),
        ("steam", "heat, from steam, in chemical industry", "megajoule"),
        ("wastewater", "wastewater, average", "cubic meter"),
        ("deionised", "water, deionised", "kilogram"),
    ):
        process(
            code,
            {
                "name": f"fixture {code}",
                "reference_product": product,
                "unit": unit,
                "exact_location": "GLO",
            },
            [exchange("marker", 0.02 * output, biosphere=True)],
        )
    # Invented amounts retain the donor's 23 main-feed records and complete
    # 12 technosphere/12 biosphere operating topology, including duplicates.
    # Future acid intentionally changes, exposing an unintended moving target
    # utilization; future operating records independently change by 40%.
    dated = 1.4 if future else 1.0
    construction = cast(dict[str, object], services[PA_ESTER_SERVICES[0]]["construction"])
    process(
        "doa",
        cast(dict[str, object], construction["source"]),
        [
            {
                **exchange("adipic", (0.7 if future else 0.4) * output),
                "name": "adipic acid",
                "CAS number": "124-04-9",
                "flow": "invented-adipic-flow",
                "activity": "invented-adipic-activity",
                "comment": "Invented adipic stoichiometry comment",
                "properties": {"carbon content, fossil": {"amount": 0.49, "unit": "dimensionless"}},
                "classifications": {"CPC": "invented adipic classification"},
                "pedigree": {"reliability": 2, "completeness": 5},
                "scale without pedigree": 0.03,
            }
        ]
        + [exchange("ethylene", (index + 1) * 0.002 * output) for index in range(22)]
        + [
            exchange(code, amount * dated * output)
            for code, amount in (
                ("factory", 3e-10),
                ("electricity", 0.3),
                ("heat", 1.8),
                ("steam", 0.04),
                ("steam", 0.17),
                ("wastewater", -1e-6),
                ("wastewater", -2e-6),
                ("wastewater", -3e-6),
                ("wastewater", -4e-6),
                ("deionised", 0.02),
                ("deionised", 0.03),
                ("deionised", 0.06),
            )
        ]
        + [
            exchange(code, amount * dated * output, biosphere=True)
            for code, amount in (
                ("co2", 0.25),
                ("ethylene_air", 0.003),
                ("water_air", 0.0015),
                ("bod", 0.014),
                ("cod", 0.014),
                ("doc", 0.0036),
                ("ethylene_water", 0.032),
                ("toc", 0.0036),
                ("water_water", 0.015),
                ("cooling", 0.0147),
                ("river", 0.001),
                ("well", 0.0008),
            )
        ],
    )
    # The original fixture CO2 exchange carries material properties. The future
    # fixture omits those properties while retaining the declared flow semantics.
    if not future:
        doa_rows = cast(list[dict[str, object]], data[(database, "doa")]["exchanges"])
        co2 = next(row for row in doa_rows if row["input"] == ("biosphere", "co2"))
        co2["properties"] = {
            "carbon content, fossil": {"amount": 0.27, "unit": "dimensionless"},
            "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
            "dry mass": {"amount": 1.0, "unit": "kg"},
        }
    return data
