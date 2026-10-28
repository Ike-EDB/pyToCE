"""Compose complete invented Swiss organic-water sources for chemical manufacturing.

The municipal, gas and thermal fixtures retain their distinct nonunit production
references. Only external source records are assembled here; the package builds
every actual organic-water, sludge, ash and gas operation itself.
"""

from copy import deepcopy
from math import log
from typing import Any

from tests.unit.prepare_backgrounds.swiss_sludge_fixtures import (
    biosphere_inventory as swiss_biosphere,
)
from tests.unit.prepare_backgrounds.swiss_sludge_fixtures import native_inventory as swiss_inventory
from tests.unit.prepare_backgrounds.tps_gas_fixtures import biosphere_inventory as gas_biosphere
from tests.unit.prepare_backgrounds.tps_gas_fixtures import native_inventory as gas_inventory
from tests.unit.prepare_backgrounds.tps_water_fixtures import biosphere_inventory as water_biosphere
from tests.unit.prepare_backgrounds.tps_water_fixtures import native_inventory as water_inventory


def _flow_descriptor(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """Identify an elementary flow by its real name, unit and compartment."""
    return record["name"], record["unit"], tuple(record["categories"])


def chemical_organic_water_biosphere(
    existing: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add CH treatment identities while retaining existing chemical-flow keys."""
    result = dict(existing)
    descriptors = {_flow_descriptor(record) for record in existing.values()}
    for source in (water_biosphere(), gas_biosphere(), swiss_biosphere()):
        for key, record in source.items():
            descriptor = _flow_descriptor(record)
            if descriptor not in descriptors:
                result[key] = record
                descriptors.add(descriptor)
    return result


def _scaled_source_rows(rows: list[dict[str, Any]], factor: float) -> list[dict[str, Any]]:
    """Rescale extensive synthetic rows and preserve intensive source properties."""
    result = deepcopy(rows)
    for row in result:
        row["amount"] *= factor
        if row.get("uncertainty type") == 2:
            row["loc"] += log(factor)
            row["minimum"] *= factor
            row["maximum"] *= factor
        elif "loc" in row:
            row["loc"] = row["amount"]
    return result


def add_chemical_organic_water_inventory(
    inventory: dict[tuple[str, str], dict[str, Any]],
    catalogue: dict[str, Any],
    database: str,
    biosphere: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Connect native CH inputs to an existing chemical-source fixture.

    The CH collection and plant retain their -5/-7 m3 references, -4 m3 child
    demand and hydraulic suppliers. Their regional siblings remain unchanged.
    Sludge, gas and ash inventories come from the independent fixtures. Shared
    descriptors identify unique native records, including complete gas inventories.
    """
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    native_descriptors = {
        (record["name"], record["reference product"], record["unit"], record["location"]): key
        for key, record in inventory.items()
    }
    flow_descriptors = {_flow_descriptor(record): key for key, record in biosphere.items()}
    flow_aliases = {
        key: flow_descriptors[_flow_descriptor(record)]
        for source in (water_biosphere(), gas_biosphere(), swiss_biosphere())
        for key, record in source.items()
    }

    def merge(
        source: dict[tuple[str, str], dict[str, Any]],
        prefix: str,
        *,
        complete_gas: bool = False,
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Bind one complete external graph, keeping shared native suppliers unique."""
        aliases = {}
        for key, record in source.items():
            descriptor = (
                record["name"],
                record["reference product"],
                record["unit"],
                record["location"],
            )
            target = native_descriptors.setdefault(descriptor, (database, prefix + key[1]))
            aliases[key] = target
        for key, record in source.items():
            target = aliases[key]
            # Replace the gas supplier records with complete inventories while
            # preserving each source's production basis.
            replace_gas = complete_gas and key[1] in {"chp", "furnace"}
            if target in inventory and not replace_gas:
                continue
            copied = deepcopy(record)
            if target in inventory:
                old_output = next(
                    row["amount"]
                    for row in inventory[target]["exchanges"]
                    if row["type"] == "production"
                )
                new_output = next(
                    row["amount"] for row in copied["exchanges"] if row["type"] == "production"
                )
                copied["exchanges"] = _scaled_source_rows(
                    copied["exchanges"], old_output / new_output
                )
            for row in copied["exchanges"]:
                row["input"] = (
                    flow_aliases[row["input"]]
                    if row["type"] == "biosphere"
                    else aliases[row["input"]]
                )
            inventory[target] = copied
        return aliases

    thermal = swiss_inventory(database, future=False)
    # The Swiss fixture already contains the complete FAE/FLUWA and both ash
    # assays. Landfarming keeps its independently specified -8 kg wet reference.
    for code, service_id in (
        ("sludge", "pa_phosphate_sludge_incineration_ch"),
        ("landfarming", "pa_phosphate_sludge_landfarming_ch"),
        ("bottom_market", "pa_phosphate_bottom_ash_market_ch"),
        ("bottom", "pa_phosphate_bottom_ash_treatment_ch"),
        ("residual_market", "pa_phosphate_residual_ash_market_ch"),
        ("residual", "pa_phosphate_residual_ash_treatment_ch"),
    ):
        selector = services[service_id]["construction"]["source"]
        thermal[database, code].update(
            name=selector["name"],
            **{"reference product": selector["reference_product"]},
            unit=selector["unit"],
            location=selector["exact_location"],
        )
    thermal_aliases = merge(thermal, "organic_ch_")
    gas = gas_inventory(database, future=False)
    merge(gas, "organic_gas_", complete_gas=True)

    water = water_inventory(database, future=False)
    # Terminal links use the complete Swiss sludge functions.
    for code, thermal_code in (("incineration", "sludge"), ("landfarming", "landfarming")):
        target = thermal_aliases[database, thermal_code]
        water[database, code] = deepcopy(inventory[target])
        for row in water[database, code]["exchanges"]:
            if row["type"] == "production":
                row["input"] = (database, code)
    for role in ("incineration", "landfarming"):
        selector = services[f"pa_phosphate_sludge_{role}_market_ch"]["construction"]["source"]
        water[database, role + "_market"].update(
            name=selector["name"],
            **{"reference product": selector["reference_product"]},
            location=selector["exact_location"],
        )
    # Install the selected donor rows once.
    source_plant = water.pop((database, "plant"))
    source_collection = water.pop((database, "collection"))
    water_aliases = merge(water, "organic_water_")

    plant = inventory[database, "mibk_plant_ch"]
    plant["exchanges"] = [
        row
        for row in plant["exchanges"]
        if row["type"] != "technosphere"
        or inventory[row["input"]]["reference product"] != "sewage sludge, 70% water"
    ]
    extra_plant = [
        row
        for row in source_plant["exchanges"]
        if row["type"] == "technosphere"
        and row["input"][1].startswith(
            ("sand_", "paper_", "plastic_", "incineration_", "landfarming_")
        )
    ]
    # Original CH plant is -7 m3; the independent wastewater fixture is -3 m3.
    for row in _scaled_source_rows(extra_plant, 7 / 3):
        row["input"] = water_aliases[row["input"]]
        plant["exchanges"].append(row)
    collection = inventory[database, "mibk_water_ch"]
    collection["exchanges"] = [row for row in collection["exchanges"] if row["type"] != "biosphere"]
    raw_rows = [row for row in source_collection["exchanges"] if row["type"] == "biosphere"]
    # Original CH collection is -5 m3, versus -2 m3 in the raw-load fixture.
    for row in _scaled_source_rows(raw_rows, 5 / 2):
        row["input"] = flow_aliases[row["input"]]
        collection["exchanges"].append(row)
