"""Synthetic paint-landfill inventories.

Water, gas and sludge fixtures supply utilities with reference amounts other
than one. Paint-treatment names and absent ash elements are listed.
The deliberately incompatible paint composition must be replaced by the
composition calculated for cured epoxy when the package constructs its load.
"""

from sbr_landfill_fixtures import biosphere_inventory
from sbr_landfill_fixtures import native_inventory as plastic_inventory
from tps_gas_fixtures import flow_key
from tps_water_fixtures import exchange

OWNERS = {
    "landfill": ("site", "waste paint", "treatment of waste paint, sanitary landfill"),
    "leachate_collection": (
        "collection",
        "leachate, SLF, waste paint",
        "market for leachate, SLF, waste paint",
    ),
    "wastewater_treatment": (
        "plant",
        "leachate, SLF, waste paint",
        "treatment of leachate, SLF, waste paint, wastewater treatment",
    ),
    "sludge_incineration_market": (
        "incineration_market",
        "sewage sludge, 70% water, WWT-SLF, waste paint",
        "market for sewage sludge, 70% water, WWT-SLF, waste paint",
    ),
    "sludge_landfarming_market": (
        "landfarming_market",
        "sewage sludge, 97% water, WWT-SLF, waste paint",
        "market for sewage sludge, 97% water, WWT-SLF, waste paint",
    ),
    "sludge_incineration": (
        "thermal_sludge",
        "sewage sludge, 70% water, WWT-SLF, waste paint",
        "treatment of sewage sludge, 70% water, WWT-SLF, waste paint, municipal incineration",
    ),
    "sludge_landfarming": (
        "thermal_landfarming",
        "sewage sludge, 97% water, WWT-SLF, waste paint",
        "treatment of sewage sludge, 97% water, WWT-SLF, waste paint, landfarming",
    ),
    "bottom_ash_market": (
        "thermal_bottom_market",
        "bottom ash, MSWI-WWT-SLF, waste paint",
        "market for bottom ash, MSWI-WWT-SLF, waste paint",
    ),
    "residual_ash_market": (
        "thermal_residual_market",
        "residues, MSWI-WWT-SLF, waste paint",
        "market for residues, MSWI-WWT-SLF, waste paint",
    ),
    "bottom_ash_treatment": (
        "thermal_bottom",
        "bottom ash, MSWI-WWT-SLF, waste paint",
        "treatment of bottom ash, MSWI-WWT-SLF, waste paint, slag compartment",
    ),
    "residual_ash_treatment": (
        "thermal_residual",
        "residues, MSWI-WWT-SLF, waste paint",
        "treatment of residues, MSWI-WWT-SLF, waste paint, residual material landfill",
    ),
}


def native_inventory(database: str, *, future: bool) -> dict:
    """Represent paint-source distinctions with synthetic treatment quantities."""
    data = plastic_inventory(database, future=future)
    biosphere = biosphere_inventory()
    for code, product, name in OWNERS.values():
        node = data[database, code]
        node.update(name=name, location="RoW", **{"reference product": product})
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        production["name"] = product
    # Original climate and hydraulic coefficients determine the target load.
    # Deliberately different future values expose accidental redetermination.
    data[database, "site"]["comment"] = (
        "(MAT) of 30 (MAP) of 500" if future else "(MAT) of 16 (MAP) of 1500"
    )
    # This paint fixture retains heat rows only for rural air and long-term groundwater.
    # Omitting soil and short-term groundwater rows tests missing-flow matching.
    # Quantities are synthetic; original deterministic and dated lognormal
    # rows test uncertainty retention and normalization by reference amount.
    site = data[database, "site"]
    site["exchanges"] = [
        row
        for row in site["exchanges"]
        if row["type"] != "biosphere" or biosphere[row["input"]]["name"] != "Heat, waste"
    ]
    p = next(row["amount"] for row in site["exchanges"] if row["type"] == "production")
    for compartment, amount in (
        (("air", "non-urban air or from high stacks"), 197 if future else 71),
        (("water", "ground-, long-term"), 211 if future else 73),
    ):
        row = exchange(flow_key("Heat, waste", compartment), amount * -p, "biosphere")
        if not future:
            for field in ("scale", "minimum", "maximum", "negative"):
                row.pop(field, None)
            row.update(loc=row["amount"], **{"uncertainty type": 0})
        site["exchanges"].append(row)
    for code in ("thermal_bottom", "thermal_residual"):
        node = data[database, code]
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        properties = production["properties"]
        for name in ("sulfur", "titanium", "vanadium"):
            properties["oxygen content"]["amount"] += properties.pop(f"{name} content")["amount"]
        if code == "thermal_bottom":
            properties["dry mass"]["amount"] = 0.813953
            properties["water in wet mass"]["amount"] = 0.186047
        node["exchanges"] = [
            row
            for row in node["exchanges"]
            if row["type"] != "biosphere"
            or biosphere[row["input"]]["name"] not in {"Sulfate", "Titanium ion", "Vanadium V"}
        ]
    # This paint incineration fixture omits aqueous Na/Ca/Cl release rows.
    # The target load determines surface-water releases from fresh NaOH/CaO/HCl.
    node = data[database, "thermal_sludge"]
    node["exchanges"] = [
        row
        for row in node["exchanges"]
        if row["type"] != "biosphere"
        or biosphere[row["input"]]["name"] not in {"Sodium I", "Calcium II", "Chloride"}
    ]
    return data
