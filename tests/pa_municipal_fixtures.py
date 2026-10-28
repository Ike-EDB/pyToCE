"""Synthetic municipal incineration and ash treatment inventories for PHB and PA tests."""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from tests.unit.prepare_backgrounds.native_hwi_fixtures import (
    native_hwi_biosphere,
    native_hwi_inventory,
)

URBAN = ("air", "urban air close to ground")
SURFACE = ("water", "surface water")
LONG = ("water", "ground-, long-term")
ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/phb_incineration/native.json").read_text(encoding="utf-8")
)


def _biosphere():
    """Add elementary flows with required names, units and compartments using synthetic IDs."""
    data = native_hwi_biosphere()
    descriptors = [(name, "kilogram", URBAN) for name in FIXTURE["process_emissions"]]
    descriptors += [
        (name, unit, categories)
        for name, unit, categories in (
            ("Carbon dioxide, fossil", "kilogram", URBAN),
            ("Carbon dioxide, non-fossil", "kilogram", URBAN),
            ("Methane, non-fossil", "kilogram", URBAN),
            ("Water", "cubic meter", URBAN),
            ("Water", "cubic meter", SURFACE),
            ("Heat, waste", "megajoule", URBAN),
            ("Non-hazardous waste disposed", "kilogram", ("inventory indicator", "waste")),
        )
    ]
    existing = {(row["name"], row["unit"], tuple(row["categories"])) for row in data.values()}
    for name, unit, categories in descriptors:
        if (name, unit, categories) not in existing:
            data[("biosphere", f"phb_flow_{len(data)}")] = {
                "name": name,
                "unit": unit,
                "categories": categories,
                "type": "emission",
            }
            existing.add((name, unit, categories))
    return data


def _native(catalogue, database, *, original):
    """Build synthetic municipal incineration and ash treatment inventories."""
    data = native_hwi_inventory(catalogue, database, original=original)
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    descriptors = {
        (row["name"], row["unit"], tuple(row["categories"])): key
        for key, row in _biosphere().items()
    }

    def row(code, amount, *, bio=False):
        """Retain observable native uncertainty on positive source quantities."""
        return {
            "input": code if bio else (database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "uncertainty type": 2 if amount > 0 else 0,
            "loc": math.log(amount) + 0.03 if amount > 0 else amount,
            "scale": 0.21,
            "minimum": amount / 2,
            "maximum": amount * 2,
            "allocation": "invented source allocation",
        }

    def emission(name, amount, categories=SURFACE, unit="kilogram"):
        """Use a declared elementary flow already present in the fixture biosphere."""
        return row(descriptors[name, unit, categories], amount, bio=True)

    def add(code, selector, basis, inputs, *, waste=False, properties=None):
        """Scale inputs per kilogram by each activity's absolute reference production."""
        scaled = []
        for ex in inputs:
            ex = dict(ex)
            ex["amount"] *= basis
            ex["loc"] = (
                ex["loc"] + math.log(basis) if ex["uncertainty type"] == 2 else ex["loc"] * basis
            )
            ex["minimum"] *= basis
            ex["maximum"] *= basis
            scaled.append(ex)
        data[(database, code)] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [
                {
                    "type": "production",
                    "input": (database, code),
                    "amount": -basis if waste else basis,
                    "properties": properties or {},
                },
                *scaled,
            ],
        }

    for code, product, unit in (
        ("phb_facility", "municipal waste incineration facility", "unit"),
        ("phb_v_proxy", "chromium trioxide, flakes", "kilogram"),
        ("phb_water_a", "water, decarbonised", "kilogram"),
        ("phb_water_b", "water, decarbonised", "kilogram"),
        ("phb_haul", "transport, freight, lorry", "ton kilometer"),
    ):
        add(
            code,
            {
                "name": f"invented {code}",
                "reference_product": product,
                "unit": unit,
                "exact_location": "RoW",
            },
            1,
            [emission("Carbon dioxide, fossil", 0.02, URBAN)],
        )
    # The future fixture uses a distinct product spelling for the selected heat function.
    if not original:
        data[(database, "mc_ng")]["reference product"] = "heat, district or industrial"
    properties = {
        f"{name} content": {"amount": value} for name, value in FIXTURE["dry_elements"].items()
    }
    properties.update(
        {
            "dry mass": {"amount": FIXTURE["dry_mass"]},
            "wet mass": {"amount": 1},
            "water in wet mass": {"amount": FIXTURE["water"]},
        }
    )
    supplies = dict(FIXTURE["supplies"])
    if not original:
        supplies.update(phb_water_a=0.56, phb_water_b=0.14, mc_tio2=0.00003, phb_v_proxy=0.0008)
    source = [row(code, amount) for code, amount in supplies.items()]
    source += [
        emission(name, amount, URBAN) for name, amount in FIXTURE["process_emissions"].items()
    ]
    source += [emission("Carbon dioxide, fossil", 1.1, URBAN)]
    source += [row("phb_bottom_market", -0.2), row("phb_residual_market", -0.04)]
    add(
        "phb_furnace",
        services["phb_fossil_incineration"]["construction"]["source"],
        2 if original else 5,
        source,
        waste=True,
        properties=properties if original else {},
    )
    # The hydration child consumes its process water and releases its mineral carbon separately from
    # parent hydration.
    add(
        "phb_hydrated",
        {
            "name": "invented hydrated cement treatment",
            "reference_product": "waste cement, hydrated",
            "unit": "kilogram",
            "exact_location": "RoW",
        },
        3 if original else 7,
        [row("mc_tap_b", 0.01), emission("Carbon dioxide, fossil", 0.04, URBAN)],
        waste=True,
    )
    ash_properties = {
        f"{name} content": {"amount": value} for name, value in FIXTURE["ash_dry_elements"].items()
    }
    ash_properties["dry mass"] = {"amount": FIXTURE["ash_dry_mass"]}
    for fraction in ("bottom", "residual"):
        terminal = services[f"phb_fossil_{fraction}_ash_treatment"]["construction"]["source"]
        market = services[f"phb_fossil_{fraction}_ash_market"]["construction"]["source"]
        inputs = [
            row("mc_ash_support_a", 0.2 if original else 0.8),
            row("mc_ash_support_b", 0.8 if original else 0.2),
            row("mc_ash_facility", 0.0002),
        ]
        if fraction == "residual":
            inputs += [row("mc_cement_a", 0.4), row("mc_tap_a", 0.6), row("phb_hydrated", -1)]
        for compartment, carbon_share, n_share in ((SURFACE, 0.02, 0.1), (LONG, 0.4, 0.902)):
            inputs += [
                emission(name, FIXTURE["ash_dry_mass"] * 0.12 * carbon_share * factor, compartment)
                for name, factor in (
                    ("DOC, Dissolved Organic Carbon", 1),
                    ("TOC, Total Organic Carbon", 1),
                    ("COD, Chemical Oxygen Demand", 32 / 12),
                    ("BOD5, Biological Oxygen Demand", 32 / 12),
                )
            ]
            inputs += [
                emission(
                    "Nitrate", FIXTURE["ash_dry_mass"] * 0.0002 * n_share * 62 / 14, compartment
                ),
                emission(
                    "Titanium ion", FIXTURE["ash_dry_mass"] * 0.02 * carbon_share, compartment
                ),
                emission("Vanadium V", FIXTURE["ash_dry_mass"] * 0.01 * carbon_share, compartment),
                emission("Zinc II", 0.000001, compartment),
            ]
        inputs += [emission("Non-hazardous waste disposed", 1, ("inventory indicator", "waste"))]
        add(
            f"phb_{fraction}_terminal",
            terminal,
            3 if original else 7,
            inputs,
            waste=True,
            properties=ash_properties if original else {},
        )
        add(
            f"phb_{fraction}_market",
            market,
            4 if original else 6,
            [row(f"phb_{fraction}_terminal", -1), row("phb_haul", 0.077)],
            waste=True,
        )
    return data


def pa_municipal_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Add flows for PA treatment, including phosphorus and calcium."""
    data = _biosphere()
    descriptors: set[tuple[str, str, tuple[str, ...]]] = {
        (name, "kilogram", categories)
        for name in ("Sodium I", "Sulfate", "Nitrate", "Phosphate", "Calcium II")
        for categories in (SURFACE, LONG)
    }
    descriptors.update(
        (name, "kilogram", URBAN)
        for name in ("Cyanide", "Dinitrogen monoxide", "Sulfur dioxide", "Sodium I", "Phosphorus")
    )
    existing = {(row["name"], row["unit"], tuple(row["categories"])) for row in data.values()}
    for descriptor in sorted(descriptors - existing):
        name, unit, categories = descriptor
        data["biosphere", f"pa_municipal_flow_{len(data)}"] = {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
    return data


def pa_municipal_native_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict:
    """Add synthetic PA incineration and ash treatment records to the PHB fixture.

    All quantities are invented. Original Na replaces an equal donor oxygen
    fraction, preserving the dry source composition. New P/Ca output references
    deliberately have no donor property/row. Future records omit composition properties.
    """
    data = _native(catalogue, database, original=original)
    biosphere = {
        (row["name"], row["unit"], tuple(row["categories"])): key
        for key, row in pa_municipal_biosphere().items()
    }
    furnace = data[database, "phb_furnace"]
    production = furnace["exchanges"][0]
    basis = abs(production["amount"])
    if original:
        properties = production["properties"]
        properties["sodium content"] = {"amount": 0.01}
        properties["oxygen content"]["amount"] -= 0.01
    for name, coefficient, categories in (
        ("Cyanide", 0.000001, URBAN),
        ("Dinitrogen monoxide", 0.00003, URBAN),
        ("Sulfur dioxide", 0.000002, URBAN),
        ("Sodium I", 0.000004, URBAN),
        ("Nitrate", 0.000003, SURFACE),
        ("Sulfate", 0.000006, SURFACE),
        ("Sodium I", 0.000005, SURFACE),
    ):
        amount = basis * coefficient
        furnace["exchanges"].append(
            {
                "input": biosphere[name, "kilogram", categories],
                "type": "biosphere",
                "amount": amount,
                "loc": math.log(amount) + 0.03,
                "uncertainty type": 2,
                "scale": 0.21,
            }
        )
    for fraction in ("bottom", "residual"):
        # Retain three individual valid source supply rows, with total .077
        # tkm/kg. Different rail/barge/lorry products have complete small LCI.
        market = data[database, f"phb_{fraction}_market"]
        freight = market["exchanges"].pop()
        for code, product, coefficient in (
            ("phb_haul", "transport, freight, lorry", 0.02),
            ("pa_rail", "transport, freight train", 0.03),
            ("pa_barge", "transport, freight, inland waterways, barge", 0.027),
        ):
            if code != "phb_haul":
                node = deepcopy(data[database, "phb_haul"])
                node["name"] = f"invented {code}"
                node["reference product"] = product
                node["exchanges"][0]["input"] = (database, code)
                data[database, code] = node
            row = dict(freight)
            scale = coefficient / 0.077
            row.update(
                input=(database, code),
                amount=freight["amount"] * scale,
                loc=freight["loc"] + math.log(scale),
                minimum=freight["minimum"] * scale,
                maximum=freight["maximum"] * scale,
            )
            market["exchanges"].append(row)
        terminal = data[database, f"phb_{fraction}_terminal"]
        production = terminal["exchanges"][0]
        basis = abs(production["amount"])
        if original:
            production["properties"].update(
                {"sulfur content": {"amount": 0.002}, "sodium content": {"amount": 0.003}}
            )
        # Only the selected bottom-S source has the rounded saturated endpoint;
        # the residual's native S release remains a valid subunit fraction.
        for categories, sulfur, sodium in (
            (SURFACE, 0.15, 0.2),
            (LONG, 0.85003 if fraction == "bottom" else 0.65, 0.7),
        ):
            for name, coefficient in (
                ("Sulfate", 0.8 * 0.002 * sulfur * 96 / 32),
                ("Sodium I", 0.8 * 0.003 * sodium),
            ):
                amount = basis * coefficient
                terminal["exchanges"].append(
                    {
                        "input": biosphere[name, "kilogram", categories],
                        "type": "biosphere",
                        "amount": amount,
                        "uncertainty type": 2,
                        "loc": math.log(amount) + 0.03,
                        "scale": 0.21,
                    }
                )
    return data
