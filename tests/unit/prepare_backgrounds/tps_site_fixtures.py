"""Small invented TPS sites joining the accepted gas and water test inventories.

The tests preserve source definitions and model equations. Site quantities,
production magnitudes and identifiers are invented. Shared test inputs stay
unchanged. Elementary flows from the water inventory use identifiers derived
from their names and environmental compartments, following the gas inventory's
naming scheme.
"""

from copy import deepcopy

from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    RURAL,
    flow_key,
)
from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    biosphere_inventory as gas_biosphere,
)
from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    native_inventory as gas_native,
)
from tests.unit.prepare_backgrounds.tps_water_fixtures import (
    LAND,
    ORIGINS,
    WASTE,
    activity,
    exchange,
    water_catalogue,
)
from tests.unit.prepare_backgrounds.tps_water_fixtures import (
    biosphere_inventory as water_biosphere,
)
from tests.unit.prepare_backgrounds.tps_water_fixtures import (
    native_inventory as water_native,
)

GROUND = ("water", "ground-, long-term")
SOIL = ("soil", "industrial")
SITE_NAME = "treatment of municipal solid waste, sanitary landfill"
SITE_PRODUCT = "municipal solid waste"
SITE_PRODUCTION = (-4.0, -9.0)
# Invented original hydraulic coefficient, m3/kg wet waste. Future donor water
# differs deliberately; the actual input uses the original source reference.
ORIGINAL_LEACHATE = 0.006


def biosphere_inventory() -> dict:
    """Join native-shaped descriptors under the gas fixture's existing key scheme."""
    result = gas_biosphere()
    for descriptor in water_biosphere().values():
        result[flow_key(descriptor["name"], descriptor["categories"])] = descriptor
    for name, unit, categories in (
        ("Sulfate", "kilogram", GROUND),
        ("Hydrogen sulfide", "kilogram", GROUND),
        ("Heat, waste", "megajoule", SOIL),
        ("Organic carbon, placed in landfill", "kilogram", WASTE),
        ("Waste mass, total, placed in landfill", "kilogram", WASTE),
        ("Occupation, construction site", "square meter-year", LAND),
        ("Transformation, to dump site, sanitary landfill", "square meter", LAND),
        ("Chloride", "kilogram", GROUND),
        ("TOC, Total Organic Carbon", "kilogram", GROUND),
        ("Heat, waste", "megajoule", GROUND),
    ):
        result[flow_key(name, categories)] = {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
    return result


def native_inventory(database: str, *, future: bool) -> dict:
    """Add a signed site and complete support to the existing gas/water inventories."""
    result = gas_native(database, future=future)
    water = water_native(database, future=future)
    flow_map = {
        key: flow_key(row["name"], row["categories"]) for key, row in water_biosphere().items()
    }
    for node in water.values():
        for row in node["exchanges"]:
            if row["type"] == "biosphere":
                row["input"] = flow_map[row["input"]]
    result.update(water)
    for code, product, unit, production in (
        ("site_equipment", "invented complete landfill equipment", "unit", 9 if future else 4),
        ("site_admin_a", "electricity, low voltage", "kilowatt hour", 5 if future else 2),
        ("site_admin_b", "electricity, low voltage", "kilowatt hour", 7 if future else 3),
        ("site_heat", "heat, district or industrial, other than natural gas", "megajoule", 6),
        ("site_retirement", "waste steel", "kilogram", -3),
    ):
        result[database, code] = activity(
            database,
            code,
            f"invented {code}",
            product,
            unit,
            production,
            [exchange(flow_key("Carbon dioxide, fossil", RURAL), 0.013, "biosphere")],
        )
    # Every quantity here is per kg wet waste before activity() applies |p|.
    rows = [
        exchange((database, "site_equipment"), 0.003 if future else 0.002),
        exchange((database, "site_admin_a"), 0.04 if future else 0.02),
        exchange((database, "site_admin_b"), 0.02 if future else 0.03),
        exchange((database, "site_heat"), 0.07 if future else 0.04),
        exchange((database, "site_retirement"), -0.001),
        exchange((database, "collection"), -(0.009 if future else ORIGINAL_LEACHATE)),
        exchange(
            flow_key("Occupation, construction site", LAND), 0.03 if future else 0.02, "biosphere"
        ),
        exchange(
            flow_key("Transformation, to dump site, sanitary landfill", LAND), 0.001, "biosphere"
        ),
    ]
    # Large donor chemistry is deliberately incompatible with the actual TPS load.
    # Selected nonzero replacement rows preserve native identities and scaled
    # distributions; municipal chemistry assigned to separate gas and leachate
    # owners is omitted.
    for name, categories, amount in (
        ("Carbon dioxide, fossil", RURAL, 71.0),
        ("Methane, non-fossil", RURAL, 72.0),
        ("Heat, waste", RURAL, 73.0),
        ("Heat, waste", SOIL, 74.0),
        ("Sulfate", GROUND, 0.02),
        ("Hydrogen sulfide", GROUND, 0.01),
        ("Organic carbon, placed in landfill", WASTE, 75.0),
        ("Waste mass, total, placed in landfill", WASTE, 76.0),
        ("Non-hazardous waste disposed", WASTE, 77.0),
        ("Chloride", GROUND, 78.0),
        ("TOC, Total Organic Carbon", GROUND, 79.0),
        ("Heat, waste", GROUND, 80.0),
    ):
        rows.append(exchange(flow_key(name, categories), amount, "biosphere"))
    result[database, "site"] = activity(
        database,
        "site",
        SITE_NAME,
        SITE_PRODUCT,
        "kilogram",
        SITE_PRODUCTION[int(future)],
        rows,
    )
    return result


def site_catalogue() -> dict:
    """Extend water declarations and retain the packaged TPS recipes."""
    catalogue = water_catalogue()
    for origin in ORIGINS:
        template = next(
            s
            for s in catalogue["services"]
            if s["provider_service_id"] == f"test_tps_plant_{origin}"
        )
        site = deepcopy(template)
        site.update(
            provider_service_id=f"test_tps_site_{origin}",
            name=f"test_tps_site_{origin}",
            reference_product=SITE_PRODUCT,
            unit="kilogram",
            construction={
                "kind": "tps_residue_landfill",
                "source_role": "native_ecoinvent_provider",
                "formulation_service_id": f"tps_{origin}",
                "source": {
                    "name": SITE_NAME,
                    "reference_product": SITE_PRODUCT,
                    "unit": "kilogram",
                    "exact_location": "RoW",
                },
                "leachate": {"service_id": f"test_tps_collection_{origin}"},
            },
        )
        catalogue["services"].append(site)
    return catalogue


def remap_terminal_biosphere(inventory: dict) -> dict:
    """Join the existing water fixture's small external sludge terminal emissions."""
    mapping = {
        key: flow_key(row["name"], row["categories"]) for key, row in water_biosphere().items()
    }
    for node in inventory.values():
        for row in node["exchanges"]:
            if row["type"] == "biosphere":
                row["input"] = mapping[row["input"]]
    return inventory
