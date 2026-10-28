"""Reuse complete invented CH water graphs at original and dated source boundaries.

The native chemical fixture includes municipal collection, biological treatment,
Swiss FLUWA, both ash terminals and their complete native suppliers. No
package receiving record is replaced by a fixture-produced treatment result.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from math import log
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
from tests.unit.prepare_backgrounds.native_chemical_fixtures import (
    native_chemical_biosphere,
    native_chemical_inventory,
)


def _dated_water_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Double each native receiving basis and change a real reagent portfolio.

    Intensive properties retain the original source basis. Removing future
    properties and changing its climate comment requires admission to use the
    original database. The peroxide's two actual suppliers change shares and
    normalized dose; source calibration must retain its original magnitude.
    """
    records = native_chemical_inventory(catalogue, database)
    selectors = [
        service["construction"]["source"]
        for service in catalogue["services"]
        if service["provider_service_id"].startswith("pa_phosphate_")
        and service["provider_service_id"].endswith("_ch")
    ]
    descriptors = {
        (row["name"], row["reference_product"], row["unit"], row["exact_location"])
        for row in selectors
    }
    receiving_inputs: set[tuple[str, str]] = set()
    for record in records.values():
        descriptor = (
            record["name"],
            record["reference product"],
            record["unit"],
            record["location"],
        )
        if descriptor in descriptors:
            for row in record["exchanges"]:
                if row["type"] == "technosphere":
                    receiving_inputs.add(row["input"])
                row.pop("properties", None)
                row["amount"] *= 2
                if row.get("uncertainty type") == 2:
                    row["loc"] += log(2)
                elif "loc" in row:
                    row["loc"] *= 2
                for field in ("minimum", "maximum"):
                    if field in row:
                        row[field] *= 2
            record["comment"] = "Invented dated operation; original climate remains shared."
    for key in receiving_inputs:
        record = records[key]
        if record["reference product"] == "heat, district or industrial, natural gas":
            record["reference product"] = "heat, district or industrial"
    sludge = records[database, "organic_ch_sludge"]
    for row in sludge["exchanges"]:
        if row["input"][1] in {"organic_ch_peroxide_a", "organic_ch_peroxide_b"}:
            share = 0.7 if row["input"][1].endswith("_a") else 0.3
            amount = 4 * 0.0018 * share
            row.update(
                amount=amount, loc=log(amount) + 0.03, minimum=amount / 2, maximum=amount * 2
            )
        if row["input"][1] == "organic_ch_hydroxide":
            # Permitted stored native missing-parameter markers; the real
            # cut-off output amount, deterministic loc and false flag survive.
            row.update({"uncertainty type": 0, "loc": row["amount"], "negative": False})
            for field in ("scale", "shape", "minimum", "maximum"):
                row[field] = float("nan")
    return records


@contextmanager
def water_receiving_boundary(root: Path, *, future: bool) -> Iterator[tuple[Any, _RecipeContext]]:
    """Write complete original and dated native receiving sources."""
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    names = BrightwayNames(root / "brightway", "pa_water_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(native_chemical_biosphere(), searchable=False)
        bd.Database("original").write(
            native_chemical_inventory(catalogue, "original"), searchable=False
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(
                _dated_water_inventory(catalogue, database), searchable=False
            )
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
        # The reference owner always sees original calibration. Dated source
        # selection remains the actual ordinary resolver in the yielded context.
        yield bd, replace(context, semantic_rows=[], technology_rows=[], traced_services=set())
