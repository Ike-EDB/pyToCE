"""Synthetic plastic-leachate inventories for package treatment calculations.

The gas, wastewater and sludge inputs reuse existing synthetic fixtures.
Plastic-leachate names and global (GLO) versus rest-of-world (RoW) locations
are listed. All exchange quantities and codes are synthetic;
ordinary tests use no licensed inventory or existing Brightway store.
"""

from tests.unit.prepare_backgrounds.tps_gas_fixtures import flow_key
from tests.unit.prepare_backgrounds.tps_site_fixtures import biosphere_inventory as site_biosphere
from tests.unit.prepare_backgrounds.tps_site_fixtures import native_inventory as site_native
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    biosphere_inventory as thermal_biosphere,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import native_inventory as thermal_native
from tests.unit.prepare_backgrounds.tps_water_fixtures import exchange

LONG = ("water", "ground-, long-term")
AIR = ("air", "non-urban air or from high stacks")
LAND = ("natural resource", "land")
SURFACE = ("water", "surface water")
LEACHATE = "leachate, SLF, waste plastic, mixture"
SLUDGE = "sewage sludge, 70% water, WWT-SLF, waste plastic, mixture"
FIELD = "sewage sludge, 97% water, WWT-SLF, waste plastic, mixture"
BOTTOM = "bottom ash, MSWI-WWT-SLF, waste plastic, mixture"
RESIDUAL = "residues, MSWI-WWT-SLF, waste plastic, mixture"
OWNERS = {
    "landfill": (
        "site",
        "treatment of waste plastic, mixture, sanitary landfill",
        "waste plastic, mixture",
        "RoW",
    ),
    "leachate_collection": ("collection", f"market for {LEACHATE}", LEACHATE, "RoW"),
    "wastewater_treatment": (
        "plant",
        f"treatment of {LEACHATE}, wastewater treatment",
        LEACHATE,
        "RoW",
    ),
    "sludge_incineration_market": ("incineration_market", f"market for {SLUDGE}", SLUDGE, "RoW"),
    "sludge_landfarming_market": ("landfarming_market", f"market for {FIELD}", FIELD, "RoW"),
    "sludge_incineration": (
        "thermal_sludge",
        f"treatment of {SLUDGE}, municipal incineration",
        SLUDGE,
        "GLO",
    ),
    "sludge_landfarming": (
        "thermal_landfarming",
        f"treatment of {FIELD}, landfarming",
        FIELD,
        "RoW",
    ),
    "bottom_ash_market": ("thermal_bottom_market", f"market for {BOTTOM}", BOTTOM, "GLO"),
    "residual_ash_market": ("thermal_residual_market", f"market for {RESIDUAL}", RESIDUAL, "GLO"),
    "bottom_ash_treatment": (
        "thermal_bottom",
        f"treatment of {BOTTOM}, slag compartment",
        BOTTOM,
        "GLO",
    ),
    "residual_ash_treatment": (
        "thermal_residual",
        f"treatment of {RESIDUAL}, residual material landfill",
        RESIDUAL,
        "GLO",
    ),
}
LATER_NAMES = (
    "TOC, Total Organic Carbon",
    "DOC, Dissolved Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
    "Ammonium",
    "Nitrogen, organic bound",
    "Nitrite",
    "Nitrate",
    "Sulfate",
    "Hydrogen sulfide",
    "Zinc II",
)
LAND_NAMES = (
    "Occupation, construction site",
    "Occupation, dump site",
    "Occupation, shrub land, sclerophyllous",
    "Occupation, traffic area, road network",
    "Transformation, from dump site, sanitary landfill",
    "Transformation, from pasture, man made",
    "Transformation, from shrub land, sclerophyllous",
    "Transformation, to dump site, sanitary landfill",
    "Transformation, to forest, unspecified",
    "Transformation, to shrub land, sclerophyllous",
    "Transformation, to traffic area, road network",
)


def biosphere_inventory() -> dict:
    """Combine synthetic flow sets with the source names needed for SBR treatment."""
    result = site_biosphere()
    for row in thermal_biosphere().values():
        result[flow_key(row["name"], row["categories"])] = row
    descriptors = [(name, "kilogram", LONG) for name in LATER_NAMES]
    descriptors += [("Zinc II", "kilogram", AIR)]
    descriptors += [
        (name, "square meter-year" if name.startswith("Occupation") else "square meter", LAND)
        for name in LAND_NAMES
    ]
    descriptors += [
        (name, "kilogram", compartment)
        for name in ("Beryllium II", "Strontium II", "Thallium I", "Titanium ion")
        for compartment in (SURFACE, LONG)
    ]
    for name, unit, categories in descriptors:
        result[flow_key(name, categories)] = {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
    return result


def native_inventory(database: str, *, future: bool) -> dict:
    """Supply synthetic treatment inventories with specified supplier locations.

    The plastic-sludge source has four additional trace-element properties and
    no calcium property. Replace the municipal .0088 calcium fraction by four
    .0022 fractions, keeping the dry mass fractions at one. Gas and landfarming
    source emissions contain fossil carbon only. Non-fossil flows are also
    available for emissions calculated from the polymer and tallow inputs.
    """
    result = site_native(database, future=future)
    thermal = thermal_native(database, future=future)
    flow_map = {
        key: flow_key(row["name"], row["categories"]) for key, row in thermal_biosphere().items()
    }
    for key, node in thermal.items():
        for ex in node["exchanges"]:
            if ex["type"] == "biosphere":
                ex["input"] = flow_map[ex["input"]]
            else:
                ex["input"] = (database, "thermal_" + ex["input"][1])
        result[database, "thermal_" + key[1]] = node
    for code in ("incineration", "landfarming"):
        del result[database, code]
    for node in result.values():
        for ex in node["exchanges"]:
            if ex["input"] == (database, "incineration"):
                ex["input"] = (database, "thermal_sludge")
            elif ex["input"] == (database, "landfarming"):
                ex["input"] = (database, "thermal_landfarming")
    for code, name, product, location in OWNERS.values():
        result[database, code].update(
            name=name, **{"reference product": product, "location": location}
        )
    # The wet source climate gives a calculated evaporation lower bound of 0.
    # Original climate metadata is retained when future suppliers change.
    result[database, "plant"]["comment"] = "(MAT) of 16 (MAP) of 1500 (EAT) of 500"
    biosphere = biosphere_inventory()
    for code in ("thermal_sludge", "thermal_landfarming", "thermal_bottom", "thermal_residual"):
        node = result[database, code]
        p = next(ex for ex in node["exchanges"] if ex["type"] == "production")
        properties = p["properties"]
        properties.pop("calcium content")
        for element in ("beryllium", "strontium", "titanium", "thallium"):
            properties[f"{element} content"] = {"amount": 0.0022, "unit": "dimensionless"}
        node["exchanges"] = [
            ex
            for ex in node["exchanges"]
            if ex["type"] != "biosphere"
            or not (
                biosphere[ex["input"]]["name"] in {"Calcium II", "Phosphate"}
                or biosphere[ex["input"]]["name"].endswith(", non-fossil")
            )
        ]
        if code in ("thermal_bottom", "thermal_residual"):
            for name in ("Beryllium II", "Strontium II", "Thallium I", "Titanium ion"):
                for compartment, q in ((SURFACE, 1e-7), (LONG, 2e-7)):
                    node["exchanges"].append(
                        exchange(flow_key(name, compartment), q * -p["amount"], "biosphere")
                    )
    # Plastic-leachate landfarming sources contain elements absent from SBR sludge.
    # Include synthetic rows with those source names to test their removal.
    for name in ("Beryllium II", "Strontium II", "Thallium I", "Titanium ion"):
        result[database, "thermal_landfarming"]["exchanges"].append(
            exchange(flow_key(name, SURFACE), 0.000001, "biosphere")
        )
    site = result[database, "site"]
    existing = {ex["input"] for ex in site["exchanges"] if ex["type"] == "biosphere"}
    for key in [
        *(flow_key(name, LONG) for name in LATER_NAMES),
        *(flow_key(name, LAND) for name in LAND_NAMES),
        flow_key("Zinc II", AIR),
    ]:
        if key not in existing:
            site["exchanges"].append(exchange(key, 0.012, "biosphere"))
    return result
