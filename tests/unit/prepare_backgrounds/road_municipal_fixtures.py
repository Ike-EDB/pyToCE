"""Synthetic paint incineration and ash inventories for road-material tests.

The municipal fixture supplies signed reference amounts, uncertainty records
and linked utilities. Paint-source distinctions are represented by the names,
original ash moisture and absence of the specified mineral release rows.
"""

from pa_municipal_fixtures import pa_municipal_biosphere, pa_municipal_native_inventory

ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
URBAN = ("air", "urban air close to ground")
SURFACE = ("water", "surface water")
LONG = ("water", "ground-, long-term")
MINERALS = {
    "Titanium ion": "titanium",
    "Silicon": "silicon",
    "Calcium II": "calcium",
    "Aluminium III": "aluminium",
    "Potassium I": "potassium",
    "Magnesium": "magnesium",
    "Sodium I": "sodium",
    "Sulfate": "sulfur",
}


def road_municipal_biosphere():
    """Declare source flow names and units with synthetic identifiers."""
    data = pa_municipal_biosphere()
    existing = {(row["name"], tuple(row["categories"])) for row in data.values()}
    for categories in (URBAN, SURFACE, LONG):
        for name in MINERALS:
            if (name, categories) not in existing:
                data["biosphere", f"road_flow_{len(data)}"] = {
                    "name": name,
                    "unit": "kilogram",
                    "categories": categories,
                    "type": "emission",
                }
    return data


def road_municipal_inventory(catalogue, database, *, original):
    """Link synthetic utilities and ash treatment with distinct reference amounts."""
    data = pa_municipal_native_inventory(catalogue, database, original=original)
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    biosphere = road_municipal_biosphere()
    nodes = {"phb_furnace": "road_fossil_municipal_incineration"}
    for fraction in ("bottom", "residual"):
        for suffix, kind in (("market", "market"), ("terminal", "treatment")):
            nodes[f"phb_{fraction}_{suffix}"] = f"road_fossil_municipal_{fraction}_ash_{kind}"
    for code, sid in nodes.items():
        node = data[database, code]
        selector = services[sid]["construction"]["source"]
        node.update(
            name=selector["name"],
            **{
                "reference product": selector["reference_product"],
                "location": selector["exact_location"],
                "unit": selector["unit"],
            },
        )
        if code == "phb_furnace":
            continue
        production = node["exchanges"][0]
        if original:
            properties = production["properties"]
            # Admitted paint municipal ash moisture differs from paint sludge.
            water = 0.173671 if "bottom" in code else 0.0
            properties["dry mass"] = {"amount": 1.0 - water}
            properties["water in wet mass"] = {"amount": water}
            properties["wet mass"] = {"amount": 1.0}
            for name in MINERALS.values():
                properties.pop(name + " content", None)
            if "residual" in code:
                properties.pop("vanadium content", None)
        absent = set(MINERALS)
        if "residual" in code:
            absent.add("Vanadium V")
        node["exchanges"] = [
            row
            for row in node["exchanges"]
            if row["type"] != "biosphere" or biosphere[row["input"]]["name"] not in absent
        ]
    return data
