"""Synthetic polyamide operations with wastewater treatment and separate condensate."""

import math
from typing import Any

from water_treatment_fixtures import water_biosphere, water_native_inventory


def pa_operation_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Reuse water and carbon-dioxide flow names and add nitrogen to unspecified air."""
    flows = water_biosphere()
    flows["biosphere", "pa_n2"] = {
        "name": "Nitrogen",
        "unit": "kilogram",
        "categories": ("air",),
        "type": "emission",
    }
    flows["biosphere", "pa_water_air"] = {
        "name": "Water",
        "unit": "cubic meter",
        "categories": ("air",),
        "type": "emission",
    }
    return flows


def pa_operation_native_inventory(
    catalogue: dict[str, Any],
    database: str,
    *,
    original: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add synthetic ester and amide operations with wastewater in four regions.

    Heat, factory and water inputs represent operating requirements. Reactants
    and carbon dioxide represent unrelated source chemistry to be excluded.
    Each collection market links to wastewater treatment, including chemical
    oxygen demand emissions. The same sources also supply the separate
    calculation for nominally pure condensate; all quantities are synthetic.
    """
    data = water_native_inventory(catalogue, database, original=original)
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    basis = 2.0 if original else 5.0

    def row(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
        """Attach identifiable uncertainty and allocation to each synthetic exchange."""
        return {
            "input": ("biosphere" if bio else database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(abs(amount)) + 0.02,
            "scale": 0.17,
            "negative": amount < 0,
            "allocation": "invented PA operation",
        }

    def native(
        code: str,
        selector: dict[str, Any],
        output: float,
        rows: list[dict[str, Any]],
    ) -> None:
        """Add a synthetic process with the specified name, product and location."""
        data[database, code] = {
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

    for code, product, unit in (
        ("pa_factory", "chemical factory, organics", "unit"),
        ("pa_steam", "heat, from steam, in chemical industry", "megajoule"),
        ("pa_reactant", "unrelated donor reactant", "kilogram"),
        ("pa_n2_a", "nitrogen, liquid", "kilogram"),
        ("pa_n2_b", "nitrogen, liquid", "kilogram"),
    ):
        native(
            code,
            {"name": code, "reference_product": product, "unit": unit, "exact_location": "GLO"},
            1.0,
            [row("co2", 0.007, bio=True)],
        )
    for sid, code, multiplier in (
        ("pa_ester_operation", "pa_ester", 1.0),
        ("pa_amide_operation", "pa_amide", 1.3),
    ):
        scaled = basis * multiplier
        rows = [
            row("pa_factory", 0.003 * scaled),
            row("electricity_a", 0.3 * scaled),
            row("heat_a", 1.7 * scaled),
            row("pa_steam", 0.8 * scaled),
            row("pa_reactant", 4 * scaled),
            row("co2", 2 * scaled, bio=True),
            row("pa_water_air", 0.002 * scaled, bio=True),
            row("surface", 0.003 * scaled, bio=True),
        ]
        for i, suffix in enumerate(("row", "ch", "ca_qc", "europe")):
            wastewater = row("market_" + suffix, -(i + 1) * 0.0001 * scaled)
            wastewater.update({"name": "wastewater, average", "unit": "cubic meter"})
            if original:
                wastewater.update(
                    {
                        "activity": "invented native wastewater producer " + suffix,
                        "flow": "invented wastewater product",
                        "properties": {"dry mass": {"amount": 10.0, "unit": "kg"}},
                    }
                )
            rows.append(wastewater)
        native(code, services[sid]["construction"]["source"], basis, rows)
    share = 0.3 if original else 0.8
    native(
        "pa_purge",
        services["pa_finish_nitrogen"]["construction"]["source"],
        basis,
        [
            row("pa_n2_a", 0.019 * basis * share),
            row("pa_n2_b", 0.019 * basis * (1 - share)),
            row("pa_n2", 0.019 * basis, bio=True),
            row("pa_reactant", basis),
            row("co2", basis, bio=True),
        ],
    )
    return data
