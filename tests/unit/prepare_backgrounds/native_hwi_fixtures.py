"""Synthetic hazardous-waste incineration, support, ash and wastewater inventories."""

import math
from typing import Any

from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    CHEMICAL_RESIDUE_ASH_SOURCE,
    CHEMICAL_RESIDUE_FIXED_SOURCE,
    CHEMICAL_RESIDUE_PAM_SOURCE,
    CHEMICAL_RESIDUE_SOURCE,
)
from tests.unit.prepare_backgrounds.water_treatment_fixtures import (
    water_biosphere,
    water_native_inventory,
)

HYDRAULIC_SERVICES = (
    "hydraulic_water_collection_row",
    "hydraulic_water_treatment_row",
)


def native_hwi_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Use source flow names and units with synthetic IDs and no impact factors."""
    data = water_biosphere()
    urban = ("air", "urban air close to ground")
    surface = ("water", "surface water")
    long = ("water", "ground-, long-term")
    descriptors = [
        ("mc_water_air", "Water", "cubic meter", ("air",)),
        ("mc_water_final", "Water", "cubic meter", ("water",)),
        (
            "mc_water_resource",
            "Water, well, in ground",
            "cubic meter",
            ("natural resource", "in water"),
        ),
        ("mc_oxygen", "Oxygen", "kilogram", ("natural resource", "in air")),
        ("mc_co2_fossil", "Carbon dioxide, fossil", "kilogram", urban),
        ("mc_co2_bio", "Carbon dioxide, non-fossil", "kilogram", urban),
        ("mc_co2_bio_air", "Carbon dioxide, non-fossil", "kilogram", ("air",)),
        ("mc_co_fossil", "Carbon monoxide, fossil", "kilogram", urban),
        ("mc_co_bio", "Carbon monoxide, non-fossil", "kilogram", urban),
        ("mc_hcl_air", "Hydrochloric acid", "kilogram", urban),
        ("mc_nh3_air", "Ammonia", "kilogram", urban),
        ("mc_nox_air", "Nitrogen oxides", "kilogram", urban),
        ("mc_n2o_air", "Dinitrogen monoxide", "kilogram", urban),
        ("mc_pm_urban", "Particulate Matter, < 2.5 um", "kilogram", urban),
        ("mc_pm", "Particulate Matter, < 2.5 um", "kilogram", ("air",)),
        ("mc_ti_air", "Titanium ion", "kilogram", ("air",)),
        ("mc_v_air", "Vanadium V", "kilogram", ("air",)),
        ("mc_na_air", "Sodium I", "kilogram", urban),
        ("mc_heat_water", "Heat, waste", "megajoule", surface),
        ("mc_hazardous", "Hazardous waste disposed", "kilogram", ("inventory indicator", "waste")),
        (
            "mc_placed_C",
            "Organic carbon, placed in landfill",
            "kilogram",
            ("inventory indicator", "waste"),
        ),
        (
            "mc_placed_waste",
            "Waste mass, total, placed in landfill",
            "kilogram",
            ("inventory indicator", "waste"),
        ),
    ]
    for suffix, categories in (("short", surface), ("long", long)):
        descriptors += [
            (f"mc_{code}_{suffix}", name, "kilogram", categories)
            for code, name in (
                ("doc", "DOC, Dissolved Organic Carbon"),
                ("toc", "TOC, Total Organic Carbon"),
                ("bod", "BOD5, Biological Oxygen Demand"),
                ("na", "Sodium I"),
                ("ca", "Calcium II"),
                ("cl", "Chloride"),
                ("ti", "Titanium ion"),
                ("v", "Vanadium V"),
                ("nitrate", "Nitrate"),
                ("zinc", "Zinc II"),
            )
        ]
    descriptors.append(("mc_cod_long", "COD, Chemical Oxygen Demand", "kilogram", long))
    for code, name, unit, categories in descriptors:
        data[("biosphere", code)] = {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
    return data


def native_hwi_inventory(
    catalogue: dict[str, Any],
    database: str,
    *,
    original: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Supply synthetic original and dated incineration and treatment inventories.

    Treatment quantities are per kilogram of input; other supplies use their
    stated units. Future reference amounts and supplier shares differ. Only
    original inventories provide waste composition for the calculation.
    Ash treatment includes 0.6 kg water per kg ash to test input retention.
    """
    data = water_native_inventory(catalogue, database, original=original)

    def row(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
        """Give a synthetic input identifiable uncertainty and an allocation label."""
        record = {
            "input": ("biosphere" if bio else database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "uncertainty type": 0,
            "loc": amount,
            "allocation": "invented source allocation",
        }
        if amount > 0:
            record.update(
                {
                    "uncertainty type": 2,
                    "loc": math.log(amount) + 0.03,
                    "scale": 0.21,
                    "minimum": amount / 2,
                    "maximum": amount * 2,
                }
            )
        return record

    def add(
        code: str,
        selector: dict[str, Any],
        basis: float,
        inputs: list[dict[str, Any]],
        *,
        waste: bool = False,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Scale per-unit inputs by the absolute reference production amount."""
        scaled = []
        for source in inputs:
            record = dict(source)
            record["amount"] *= basis
            record["loc"] = (
                source["loc"] + math.log(basis)
                if source["uncertainty type"] == 2
                else source["loc"] * basis
            )
            for bound in ("minimum", "maximum"):
                if bound in source:
                    record[bound] *= basis
            scaled.append(record)
        production = {
            "input": (database, code),
            "type": "production",
            "amount": -basis if waste else basis,
            "uncertainty type": 0,
            "loc": -basis if waste else basis,
            "properties": properties or {"wet mass": {"amount": 1}, "dry mass": {"amount": 1}},
        }
        data[(database, code)] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "type": "process",
            "exchanges": [production, *scaled],
        }

    def leaf(code: str, product: str, unit: str = "kilogram", *, waste: bool = False) -> None:
        """Add a synthetic upstream supply with identifiable emissions."""
        add(
            code,
            {
                "name": f"invented {code} {'original' if original else 'future'}",
                "reference_product": product,
                "unit": unit,
                "exact_location": "RER" if original else "EUR",
            },
            1,
            [row("co2", 0.02, bio=True)],
            waste=waste,
        )

    for code, product, unit in (
        ("mc_factory", "chemical factory, organics", "unit"),
        ("mc_low_power", "electricity, low voltage", "kilowatt hour"),
        ("mc_oil_fuel", "light fuel oil", "kilogram"),
        ("mc_oil_furnace", "oil furnace", "unit"),
        ("mc_ng", "heat, district or industrial, natural gas", "megajoule"),
        ("mc_hcl", "hydrochloric acid, without water, in 30% solution state", "kilogram"),
        ("mc_naoh", "sodium hydroxide, without water, in 50% solution state", "kilogram"),
        ("mc_cao", "quicklime, milled, packed", "kilogram"),
        ("mc_ammonia", "ammonia, anhydrous, liquid", "kilogram"),
        ("mc_tio2", "titanium dioxide", "kilogram"),
        ("mc_v_proxy", "chromium", "kilogram"),
        ("mc_inorganic", "chemical, inorganic", "kilogram"),
        ("mc_organic", "chemical, organic, basic precursors and materials", "kilogram"),
        ("mc_hwi_facility", "hazardous waste incineration facility", "unit"),
        ("mc_ash_facility", "residual material landfill", "unit"),
        ("mc_ash_support_a", "process-specific burdens, residual material landfill", "kilogram"),
        ("mc_ash_support_b", "process-specific burdens, residual material landfill", "kilogram"),
        ("mc_cement_a", "cement, unspecified", "kilogram"),
        ("mc_cement_b", "cement, unspecified", "kilogram"),
        ("mc_tap_a", "tap water", "kilogram"),
        ("mc_tap_b", "tap water", "kilogram"),
    ):
        leaf(code, product, unit)
    leaf("mc_hydrated_cement", "waste cement, hydrated", waste=True)
    add("mc_pam", CHEMICAL_RESIDUE_PAM_SOURCE, 1, [row("co2", 0.1, bio=True)])
    oil_selector = {
        "name": "invented complete oil boiler",
        "reference_product": "heat, central or small-scale, other than natural gas",
        "unit": "megajoule",
        "exact_location": "CH",
    }
    add(
        "mc_oil",
        oil_selector,
        2 if original else 4,
        [
            row("mc_low_power", 0.004),
            row("mc_oil_fuel", 0.03125),
            row("mc_oil_furnace", 0.0001),
            row("co2", 0.03125 * 0.86 * 44.009 / 12.011, bio=True),
        ],
    )
    data[(database, "mc_oil")]["parameters"] = [{"name": "scaling_factor", "amount": 1.25}]

    add(
        "mc_fixed",
        CHEMICAL_RESIDUE_FIXED_SOURCE,
        4 if original else 8,
        [
            row("mc_ammonia", 0.002),
            row("mc_ng", 0.3),
            row("mc_tio2", 0.0002 if original else 0.00003),
            row("mc_v_proxy", 0.00005 if original else 0.0004),
            row("mc_nh3_air", 0.00002, bio=True),
            row("mc_nox_air", 0.001, bio=True),
            row("mc_pm_urban", 0.0001, bio=True),
            row("mc_water_air", 0.0005, bio=True),
            row("mc_water_final", 0.0045, bio=True),
            row("mc_water_resource", 0.005, bio=True),
        ],
    )
    wash_rows = [
        row("mc_hwi_facility", 0.00002),
        row("mc_fixed", 1),
        row("mc_oil", 4),
        row("mc_hcl", 0.02),
        row("mc_naoh", 0.1),
        row("mc_cao", 0.03),
        row("mc_inorganic", 0.01),
        row("mc_ammonia", 0.0001),
        row("mc_ng", 0.02),
        row("mc_tio2", 0.0001),
        row("mc_v_proxy", 0.00002),
        row("mc_organic", 0.0001),
        row("mc_cement_a", 0.2),
        row("mc_ash_support_a", 0.1),
        row("mc_ash_facility", 0.0001),
        row("mc_hydrated_cement", -0.5),
        row("mc_co2_fossil", 1, bio=True),
        row("mc_co_fossil", 0.00001, bio=True),
        row("mc_hcl_air", 0.00003, bio=True),
        row("mc_n2o_air", 0.00001, bio=True),
        row("mc_water_air", 0.0008, bio=True),
        row("mc_hazardous", 1, bio=True),
        row("mc_zinc_short", 0.00001, bio=True),
    ]
    for suffix, doc in (("short", 0.00003), ("long", 0.002)):
        wash_rows += [
            row(f"mc_{name}_{suffix}", doc * multiplier, bio=True)
            for name, multiplier in (("doc", 1), ("toc", 1), ("bod", 5))
        ]
        wash_rows.append(row("cod" if suffix == "short" else "mc_cod_long", 5 * doc, bio=True))
    for ex in wash_rows:
        if original and ex["input"][1] in {"mc_co2_fossil", "mc_co_fossil"}:
            carbon_fraction = 12.011 / (44.009 if ex["input"][1] == "mc_co2_fossil" else 28.010)
            ex["properties"] = {
                "carbon content, fossil": {"amount": carbon_fraction},
                "carbon content, non-fossil": {"amount": 0},
            }
    add(
        "mc_hwi",
        CHEMICAL_RESIDUE_SOURCE,
        3 if original else 6,
        wash_rows,
        waste=True,
        properties={
            "wet mass": {"amount": 1},
            "dry mass": {"amount": 0.5},
            "water in wet mass": {"amount": 0.5},
            "carbon allocation": {"amount": 0.3},
        },
    )
    if original:
        data[(database, "mc_hwi")]["comment"] = (
            "Invented waste composition (wet, in ppm): H2O 500000; O 40000; H 60000; "
            "C 300000; S 20000; N 10000; P 5000; B 1000; Cl 50000; F 3000; "
            "Cd 10; Co 20; Cr 30; Cu 40; Hg 5; Ni 20; Pb 50; Zn 100; Si 1000; Na n.a.;"
        )
    ash_rows = [
        row("mc_cement_a", 0.25),
        row("mc_cement_b", 0.15),
        row("mc_ash_support_a", 0.25 if original else 0.8),
        row("mc_ash_support_b", 0.75 if original else 0.2),
        row("mc_ash_facility", 0.0003),
        row("mc_hydrated_cement", -1),
        row("mc_tap_a", 0.6),
        row("mc_hazardous", 1, bio=True),
        row("mc_placed_C", 0.05, bio=True),
        row("mc_placed_waste", 1, bio=True),
    ]
    for suffix, transfer in (("short", 0.1), ("long", 0.4)):
        ash_rows += [
            row(f"mc_{name}_{suffix}", 0.05 * transfer * multiplier, bio=True)
            for name, multiplier in (("doc", 1), ("toc", 1), ("bod", 3))
        ]
        ash_rows.append(
            row("cod" if suffix == "short" else "mc_cod_long", 0.05 * transfer * 3, bio=True)
        )
        ash_rows += [
            row(f"mc_{element}_{suffix}", amount, bio=True)
            for element, amount in (
                ("na", 0.02 if suffix == "short" else 0.07),
                ("ca", 0.08 if suffix == "short" else 0.1),
                ("v", 0.001 if suffix == "short" else 0.002),
                ("nitrate", 0.006 if suffix == "short" else 0.012),
                ("zinc", 0.004 if suffix == "short" else 0.005),
            )
        ]
    properties = {
        "wet mass": {"amount": 1},
        "dry mass": {"amount": 1},
        "carbon allocation": {"amount": 0.05},
        "carbon content": {"amount": 0.05},
        "sodium content": {"amount": 0.1},
        "calcium content": {"amount": 0.2},
        "vanadium content": {"amount": 0.02},
        "nitrogen content": {"amount": 0.03},
        "zinc content": {"amount": 0.01},
        "aluminium content": {"amount": 0.07},
        "potassium content": {"amount": 0.11},
        "magnesium content": {"amount": 0.035},
    }
    add(
        "mc_ash",
        CHEMICAL_RESIDUE_ASH_SOURCE,
        5 if original else 10,
        ash_rows,
        waste=True,
        properties=properties if original else None,
    )
    return data
