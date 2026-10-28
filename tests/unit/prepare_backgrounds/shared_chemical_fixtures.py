"""Combine synthetic chemical and treatment inventories for shared preparation."""

from typing import Any

from chemical_residue_fixtures import triethanolamine_biosphere, triethanolamine_native_inventory
from native_chemical_fixtures import native_chemical_biosphere, native_chemical_inventory
from pa_dump_receiving_fixtures import dump_biosphere, dump_native_inventory
from pa_landfill_receiving_fixtures import _biosphere as landfill_biosphere_inventory
from pa_landfill_receiving_fixtures import _native as landfill_native_inventory
from pa_open_burning_fixtures import open_burning_biosphere, open_burning_native_inventory
from pa_recycling_receiving_fixtures import recycling_biosphere, recycling_native_inventory
from ptt_steam_fixtures import ptt_steam_native_inventory

from tests.pa_municipal_fixtures import pa_municipal_biosphere, pa_municipal_native_inventory


def shared_chemical_inputs(
    catalogue: dict[str, Any], database: str, biosphere: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Combine synthetic chemical and sanitary-landfill inputs for the XML importer.

    Individual tests use separate suppliers. Shared preparation combines steam,
    water, cooling electricity and biogas conversion, with one inventory for
    each distinct name, product, unit and location. Input quantities are
    synthetic; package calculations remain unchanged. Elementary-flow keys
    refer to the database created by the package's XML importer.
    """
    data = {
        **triethanolamine_native_inventory(catalogue, database),
        **native_chemical_inventory(catalogue, database),
        **ptt_steam_native_inventory(database),
    }
    aliases = {
        (database, "tea_steam"): (database, "steam_rer_market"),
        (database, "tea_water"): (database, "steam_water"),
        (database, "mibk_cooling_power"): (database, "tea_cooling_power"),
        # Biogas conversion supplies polytrimethylene terephthalate process heat
        # and Swiss gas treatment; replace the separate utility-only fixture.
        (database, "ptt_biogas"): (database, "organic_gas_chp"),
        # Triethanolamine and chemical fixtures share regional wastewater sources.
        # Only the chemical fixture's Swiss plant links to Swiss sludge treatment;
        # retain one inventory for each name, product, unit and location.
        (database, "plant_ca_qc"): (database, "mibk_plant_ca_qc"),
        (database, "plant_europe"): (database, "mibk_plant_europe"),
        (database, "plant_ch"): (database, "mibk_plant_ch"),
        (database, "market_ca_qc"): (database, "mibk_water_ca_qc"),
        (database, "market_europe"): (database, "steam_ww"),
        (database, "market_ch"): (database, "mibk_water_ch"),
        (database, "heat_europe"): (database, "mibk_heat"),
    }
    for key in aliases:
        del data[key]
    for node in data.values():
        for exchange in node["exchanges"]:
            key = exchange["input"]
            exchange["input"] = (
                (biosphere, key[1]) if exchange["type"] == "biosphere" else aliases.get(key, key)
            )
    flows = {
        (biosphere, key[1]): node
        for key, node in {**triethanolamine_biosphere(), **native_chemical_biosphere()}.items()
    }
    # Polyamide operating quantities are reconstructed from original treatment
    # and washing inventories. Reuse synthetic supplies with matching names,
    # products, units and locations; keep other inventory identifiers distinct.
    for prefix, additions, added_flows in (
        (
            "landfill_source_",
            landfill_native_inventory(catalogue, database, future=False),
            landfill_biosphere_inventory()[0],
        ),
        (
            "recycling_source_",
            recycling_native_inventory(catalogue, database, future=False),
            recycling_biosphere(),
        ),
        (
            "municipal_source_",
            pa_municipal_native_inventory(catalogue, database, original=True),
            pa_municipal_biosphere(),
        ),
        (
            "burning_source_",
            open_burning_native_inventory(database, original=True, biosphere_database="biosphere"),
            open_burning_biosphere("biosphere"),
        ),
        (
            "dump_source_",
            dump_native_inventory(catalogue, database, biosphere_database="biosphere"),
            dump_biosphere("biosphere"),
        ),
    ):
        existing = {
            (node["name"], node["reference product"], node["unit"], node["location"]): key
            for key, node in data.items()
        }
        added_aliases = {
            key: existing[descriptor]
            for key, node in additions.items()
            if (
                descriptor := (
                    node["name"],
                    node["reference product"],
                    node["unit"],
                    node["location"],
                )
            )
            in existing
        }
        added_keys = {key: added_aliases.get(key, (database, prefix + key[1])) for key in additions}
        existing_flows = {
            (node["name"], node["unit"], tuple(node["categories"])): key
            for key, node in flows.items()
        }
        flow_keys = {}
        for key, node in added_flows.items():
            descriptor = node["name"], node["unit"], tuple(node["categories"])
            # Preserve sanitary-landfill identifiers. Prefix other fixture codes
            # with their source names to avoid collisions.
            code = key[1] if prefix == "landfill_source_" else prefix + key[1]
            target = existing_flows.get(descriptor, (biosphere, code))
            flow_keys[key] = target
            if descriptor not in existing_flows:
                flows[target] = node
                existing_flows[descriptor] = target
        for key, node in additions.items():
            if key in added_aliases:
                continue
            for row in node["exchanges"]:
                row["input"] = (
                    flow_keys[row["input"]]
                    if row["type"] == "biosphere"
                    else added_keys[row["input"]]
                )
            data[added_keys[key]] = node
    return data, flows
