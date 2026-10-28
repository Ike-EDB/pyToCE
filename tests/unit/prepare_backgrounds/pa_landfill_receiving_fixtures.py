"""Join invented plastic-leachate sources with complete native HWI support.

All quantities are existing invented test controls. Missing original phosphorus
properties mirror the admitted source boundary.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from tests.unit.prepare_backgrounds.native_hwi_fixtures import (
    native_hwi_biosphere,
    native_hwi_inventory,
)
from tests.unit.prepare_backgrounds.sbr_landfill_fixtures import (
    OWNERS,
    biosphere_inventory,
    native_inventory,
)


def _biosphere() -> tuple[dict, dict, dict]:
    """Use one elementary identity per descriptor when joining independent fixtures."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    descriptors: dict[tuple[str, str, tuple[str, ...]], tuple[str, str]] = {}
    maps = []
    for source in (native_hwi_biosphere(), biosphere_inventory()):
        mapped = {}
        for key, row in source.items():
            descriptor = row["name"], row["unit"], tuple(row["categories"])
            if descriptor not in descriptors:
                target = ("biosphere", f"pa_landfill_flow_{len(merged)}")
                descriptors[descriptor] = target
                merged[target] = row
            mapped[key] = descriptors[descriptor]
        maps.append(mapped)
    # Add the flow descriptors required for PA phosphorus, sodium, water
    # and heat. Exchange quantities are supplied by the receiving calculation.
    extra = [
        ("Phosphorus", "kilogram", ("water", "surface water")),
        ("Phosphate", "kilogram", ("water", "ground-, long-term")),
        ("Sodium I", "kilogram", ("air", "non-urban air or from high stacks")),
        ("Phosphorus", "kilogram", ("air", "urban air close to ground")),
        ("Heat, waste", "megajoule", ("soil", "industrial")),
    ]
    extra.extend(
        (name, unit, categories)
        for name, unit in (("Water", "cubic meter"), ("Heat, waste", "megajoule"))
        for categories in (
            ("water", "surface water"),
            ("air", "urban air close to ground"),
            ("air", "non-urban air or from high stacks"),
        )
    )
    for name, unit, categories in extra:
        descriptor = name, unit, categories
        if descriptor not in descriptors:
            target = ("biosphere", f"pa_landfill_flow_{len(merged)}")
            descriptors[descriptor] = target
            merged[target] = {
                "name": name,
                "unit": unit,
                "categories": categories,
                "type": "emission",
            }
    return merged, maps[0], maps[1]


def _native(catalogue: dict[str, Any], database: str, *, future: bool) -> dict:
    """Join source graphs with distinct provider identities; remove exchange
    properties only from selected future OWNERS records."""
    biosphere, hwi_flows, landfill_flows = _biosphere()
    hwi = native_hwi_inventory(catalogue, database, original=not future)
    receiving = native_inventory(database, future=future)
    # Prefix every HWI key: both independent fixtures otherwise call their own
    # different hydraulic plant 'plant'. This preserves each process.
    result = {}
    for key, node in hwi.items():
        for row in node["exchanges"]:
            row["input"] = (
                hwi_flows[row["input"]]
                if row["type"] == "biosphere"
                else (database, "mc_support_" + row["input"][1])
            )
        result[database, "mc_support_" + key[1]] = node
    for node in receiving.values():
        for row in node["exchanges"]:
            if row["type"] == "biosphere":
                row["input"] = landfill_flows[row["input"]]
    result.update(receiving)
    # The independently specified site differs from its 16 C / 1500 mm/y
    # wastewater plant. Future site comments below omit original climate.
    result[database, "site"]["comment"] = "(MAT) of 18 (MAP) of 1200"
    # Actual PA bypass binds its elemental-P endpoint directly.
    collection = result[database, "collection"]
    collection["exchanges"] = [
        row
        for row in collection["exchanges"]
        if row["type"] != "biosphere" or biosphere[row["input"]]["name"] != "Phosphorus"
    ]
    # Two exact suppliers occur in both source fixtures. Use the complete HWI
    # PAM source and the complete receiving gas furnace, rebinding every input.
    aliases = {
        (database, "thermal_actual_pam"): (database, "mc_support_mc_pam"),
        (database, "mc_support_heat_europe"): (database, "furnace"),
    }
    for node in result.values():
        for row in node["exchanges"]:
            row["input"] = aliases.get(row["input"], row["input"])
    for key in aliases:
        del result[key]
    for code in ("thermal_sludge", "thermal_landfarming", "thermal_bottom", "thermal_residual"):
        production = next(
            row for row in result[database, code]["exchanges"] if row["type"] == "production"
        )
        production["properties"].pop("phosphorus content")
    if future:
        # The future fixture removes exchange properties from selected OWNERS
        # records and changes the industrial-heat product label. Other records
        # retain their declared properties; original calibration remains separate.
        for code, _, _, _ in OWNERS.values():
            record = result[database, code]
            record["comment"] = "Dated operating source without original climate metadata."
            for row in record["exchanges"]:
                row.pop("properties", None)
        for record in result.values():
            if record["reference product"] == "heat, district or industrial, natural gas":
                record["reference product"] = "heat, district or industrial"
    return result


@contextmanager
def landfill_receiving_boundary(
    root: Path, *, future: bool
) -> Iterator[tuple[Any, _RecipeContext]]:
    """Write disposable original and dated sources and expose their recipe context."""
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    names = BrightwayNames(root / "brightway", "pa_landfill_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(_biosphere()[0], searchable=False)
        bd.Database("original").write(
            _native(catalogue, "original", future=False), searchable=False
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(_native(catalogue, database, future=True), searchable=False)
        ordinary = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi" if future else None,
            year=2050 if future else 2025,
            provider_database="frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=ordinary,
            technology=ordinary,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context
