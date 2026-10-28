"""Invented native gas inventories for pyToCE's two TPS gas owners.

Semantic names represent the supported ecoinvent interface. Exchange amounts,
production magnitudes, identifiers, and distributions are synthetic.
The tests load the packaged TPS formulations.
"""

from copy import deepcopy
from math import log

RURAL = ("air", "non-urban air or from high stacks")
URBAN = ("air", "urban air close to ground")
RESOURCE = ("natural resource", "in air")
ORIGINS = ("biogenic", "fossil", "atmospheric_dac")
NMVOC = "NMVOC, non-methane volatile organic compounds"
PM = "Particulate Matter, < 2.5 um"
# Synthetic codes represent ecoinvent 3.12 gas-provider semantics. These sources supply combustion
# factors.
GAS_SOURCES = {
    "chp": {
        "name": "heat and power co-generation, biogas, gas engine",
        "reference_product": "heat, central or small-scale, other than natural gas",
        "unit": "megajoule",
        "exact_location": "RoW",
    },
    "furnace": {
        "name": "heat production, natural gas, at industrial furnace >100kW",
        "reference_product": "heat, district or industrial, natural gas",
        "unit": "megajoule",
        "exact_location": "Europe without Switzerland",
    },
}

# Synthetic emission intensities are kg/MJ of CHP heat; the future N2O test factor is zero.
ENGINE_FACTORS = {
    "Carbon monoxide, non-fossil": (0.0002, 0.00015),
    "Methane, non-fossil": (0.0001, 0.00012),
    NMVOC: (0.00003, 0.00002),
    "Nitrogen oxides": (0.0004, 0.0003),
    "Dinitrogen monoxide": (0.000005, 0.0),
    "Platinum": (2e-9, 3e-9),
}
# Vary positive native outputs, gas m3/MJ, and LHV MJ/m3 separately. Original exchanges supply LHV
# properties; dated records may omit them. The unused furnace-electricity row checks that selection
# retains only the specified demands.
CHP_PRODUCTION = (3.0, 7.0)
FURNACE_PRODUCTION = (2.0, 5.0)
CHP_FUEL = (0.08, 0.10)
FURNACE_FUEL = (0.04, 0.025)
CHP_LHV = 18.0
FURNACE_LHV = 36.0
FURNACE_ELECTRICITY = (0.002, 0.5)
FURNACE_N2O = (0.000003, 0.000007)  # invented kg/MJ heat
LOG_OFFSET = 0.07  # deliberate non-mean lognormal location offset
LOG_SPREAD = 0.23


def flow_key(name: str, categories: tuple[str, ...]) -> tuple[str, str]:
    """Return an invented flow code with the exact external semantic descriptor."""
    return "biosphere", name + "::" + "/".join(categories)


def biosphere_inventory() -> dict:
    """Declare gas species with synthetic identities in both air compartments."""
    gases = [
        f"{gas}, {origin}"
        for gas in ("Carbon dioxide", "Carbon monoxide", "Methane")
        for origin in ("fossil", "non-fossil")
    ]
    gases += [
        NMVOC,
        PM,
        "Nitrogen oxides",
        "Dinitrogen monoxide",
        "Platinum",
        "Sulfur dioxide",
        "Ammonia",
    ]
    descriptors = {(name, "kilogram", part) for name in gases for part in (RURAL, URBAN)}
    descriptors.update(("Water", "cubic meter", part) for part in (RURAL, URBAN))
    descriptors.update(("Heat, waste", "megajoule", part) for part in (RURAL, URBAN))
    descriptors.add(("Oxygen", "kilogram", RESOURCE))
    return {
        flow_key(name, part): {"name": name, "unit": unit, "categories": part, "type": "emission"}
        for name, unit, part in sorted(descriptors)
    }


def exchange(key: tuple[str, str], amount: float, kind: str = "technosphere") -> dict:
    """Give positive invented exchanges non-mean lognormal metadata."""
    row = {
        "input": key,
        "amount": amount,
        "type": kind,
        "uncertainty type": 0,
        "loc": amount,
        "comment": "Invented gas source control",
        "allocation": "cut-off control",
    }
    if amount > 0:
        row.update(
            {
                "uncertainty type": 2,
                "loc": log(amount) + LOG_OFFSET,
                "scale": LOG_SPREAD,
                "minimum": amount / 2,
                "maximum": 2 * amount,
            }
        )
    return row


def _activity(database, code, descriptor, production, rows):
    """Scale extensive per-unit rows to signed nonunit production once."""
    scaled = []
    for original in rows:
        row = deepcopy(original)
        factor = abs(production)
        row["amount"] *= factor
        if row["uncertainty type"] == 2:
            row["loc"] += log(factor)
            row["minimum"] *= factor
            row["maximum"] *= factor
        else:
            row["loc"] = row["amount"]
        scaled.append(row)
    return {
        "name": descriptor["name"],
        "reference product": descriptor["reference_product"],
        "unit": descriptor["unit"],
        "location": descriptor["exact_location"],
        "type": "process",
        "exchanges": [exchange((database, code), production, "production"), *scaled],
    }


def native_inventory(database: str, *, future: bool) -> dict:
    """Create bounded combustion-factor donors on invented native reference bases.

    Fuel and electricity suppliers have nonunit production and dated emissions.
    Complete landfill-site and wastewater-plant supplies live in their parent
    fixtures; the factor donors supply their specified combustion coefficients.
    """
    date = int(future)
    result = {}
    marker = flow_key("Carbon dioxide, fossil", URBAN)
    supplies = {
        "electricity": ("electricity, low voltage", "kilowatt hour"),
        "biogas": ("biogas", "cubic meter"),
        "natural_gas": ("natural gas, high pressure", "cubic meter"),
    }
    for code, (product, unit) in supplies.items():
        # 2/4-unit supplier references and .03/.07 kg fossil CO2 per product
        # unit distinguish actual demand from the number of supplier activities.
        rows = [exchange(marker, (0.03, 0.07)[date], "biosphere")]
        result[database, code] = _activity(
            database,
            code,
            {
                "name": "Invented " + code,
                "reference_product": product,
                "unit": unit,
                "exact_location": "EUR" if future else "RER",
            },
            (2.0, 4.0)[date],
            rows,
        )
    fuel = exchange((database, "biogas"), CHP_FUEL[date])
    if not future:
        fuel["properties"] = {"heating value, net": {"amount": CHP_LHV, "unit": "MJ/m3"}}
    rows = [fuel]
    for name, values in ENGINE_FACTORS.items():
        value = values[date]
        row = exchange(flow_key(name, RURAL), value, "biosphere")
        # Donor carbon metadata deliberately disagrees with all target pools.
        row["properties"] = {"carbon allocation": {"amount": 0.123, "unit": "kg"}}
        rows.append(row)
    # Native donor chemistry/heat are replaced; these large invented quantities
    # make retaining them alongside the actual TPS gas unmistakable.
    for name, value, part in (
        ("Carbon dioxide, non-fossil", 2.0, RURAL),
        ("Sulfur dioxide", 0.02, RURAL),
        ("Water", 0.01, RURAL),
        ("Oxygen", 1.0, RESOURCE),
        ("Heat, waste", 10.0, RURAL),
    ):
        rows.append(exchange(flow_key(name, part), value, "biosphere"))
    result[database, "chp"] = _activity(
        database, "chp", GAS_SOURCES["chp"], CHP_PRODUCTION[date], rows
    )
    fuel = exchange((database, "natural_gas"), FURNACE_FUEL[date])
    if not future:
        fuel["properties"] = {"heating value, net": {"amount": FURNACE_LHV, "unit": "MJ/m3"}}
    rows = [
        fuel,
        exchange((database, "electricity"), FURNACE_ELECTRICITY[date]),
        exchange(flow_key("Dinitrogen monoxide", RURAL), FURNACE_N2O[date], "biosphere"),
        exchange(flow_key("Carbon dioxide, fossil", RURAL), 3.0, "biosphere"),
        exchange(flow_key("Nitrogen oxides", RURAL), 0.002, "biosphere"),
        exchange(flow_key("Water", RURAL), 0.02, "biosphere"),
        exchange(flow_key("Oxygen", RESOURCE), 2.0, "biosphere"),
        exchange(flow_key("Heat, waste", RURAL), 20.0, "biosphere"),
    ]
    result[database, "furnace"] = _activity(
        database, "furnace", GAS_SOURCES["furnace"], FURNACE_PRODUCTION[date], rows
    )
    return result
