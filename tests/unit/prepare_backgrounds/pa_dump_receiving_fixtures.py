"""Synthetic native dump datasets with illustrative land use exchanges."""

from collections.abc import Iterator
from contextlib import contextmanager
from math import log
from pathlib import Path
from typing import Any, cast

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

DUMP_CLASSES = (
    ("hyperarid infiltration class (-250mm)", "hyperarid_infiltration_class_250mm", 350),
    ("dry infiltration class (100mm)", "dry_infiltration_class_100mm", 550),
    ("moist infiltration class (300mm)", "moist_infiltration_class_300mm", 900),
    ("wet infiltration class (500mm)", "wet_infiltration_class_500mm", 1100),
    ("very wet infiltration class (1000mm)", "very_wet_infiltration_class_1000mm", 1900),
)
RURAL = ("air", "non-urban air or from high stacks")
LATER_AIR = ("air", "low population density, long-term")
GROUND = ("water", "ground-")
LATER_GROUND = ("water", "ground-, long-term")
SOIL = ("soil", "industrial")
LAND = ("natural resource", "land")
WASTE = ("inventory indicator", "waste")
LAND_ROWS = (
    ("Occupation, dump site", "square meter-year", 0.007),
    ("Transformation, from pasture, man made", "square meter", 0.0004),
    ("Transformation, to dump site", "square meter", 0.0004),
)


def dump_biosphere(database: str = "dump_biosphere") -> dict[tuple[str, str], dict[str, Any]]:
    """Declare flow semantics under synthetic keys."""
    names = {
        RURAL: (
            "Carbon dioxide, fossil",
            "Carbon dioxide, non-fossil",
            "Methane, fossil",
            "Methane, non-fossil",
            "Nitrogen oxides",
            "Sulfur dioxide",
            "Sodium I",
            "Sulfate",
            "Nitrate",
            "Phosphorus",
            "Particulate Matter, > 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
            "Particulate Matter, < 2.5 um",
        ),
        LATER_AIR: (
            "Sulfate",
            "Nitrate",
            "Phosphorus",
            "Sodium I",
            "Particulate Matter, > 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
            "Particulate Matter, < 2.5 um",
        ),
        SOIL: ("Carbon", "Nitrogen", "Sulfur", "Phosphorus", "Sodium I"),
        WASTE: (
            "Non-hazardous waste disposed",
            "Organic carbon, placed in landfill",
            "Waste mass, total, placed in landfill",
        ),
    }
    leachate = (
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
        "Phosphate",
        "Sodium I",
    )
    names[GROUND] = names[LATER_GROUND] = leachate
    descriptors = [(name, "kilogram", c) for c, values in names.items() for name in values]
    descriptors.extend(
        ("Water", "cubic meter", c) for c in (RURAL, LATER_AIR, GROUND, LATER_GROUND)
    )
    descriptors.extend(("Heat, waste", "megajoule", c) for c in (SOIL, LATER_GROUND))
    descriptors.extend((name, unit, LAND) for name, unit, _ in LAND_ROWS)
    return {
        (database, f"flow_{i}"): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "natural resource" if categories == LAND else "emission",
        }
        for i, (name, unit, categories) in enumerate(descriptors)
    }


def dump_native_inventory(
    catalogue: dict[str, Any], database: str, *, biosphere_database: str = "dump_biosphere"
) -> dict[tuple[str, str], dict[str, Any]]:
    """Expose the same five signed original native sites for complete parent admission."""
    services = {str(s["provider_service_id"]): s for s in cast(list[dict], catalogue["services"])}
    biosphere = dump_biosphere(biosphere_database)
    keys = {(row["name"], row["categories"]): key for key, row in biosphere.items()}
    records = {}
    for i, (_, suffix, precipitation) in enumerate(DUMP_CLASSES):
        selector = services[f"mismanaged_open_dump_pur_{suffix}"]["construction"]["selector"]
        key = database, f"site_{i}"
        production = -(i + 2.0)
        rows = [{"input": key, "type": "production", "amount": production}]
        for name, _, per_kg in LAND_ROWS:
            amount = per_kg * abs(production)
            rows.append(
                {
                    "input": keys[name, LAND],
                    "type": "biosphere",
                    "amount": amount,
                    "uncertainty type": 2,
                    "loc": log(amount) + 0.03,
                    "scale": 0.12,
                }
            )
        # The fixture supplies P/phosphate biosphere descriptors for reconstruction from actual PA
        # composition.
        rows.append({"input": keys["Methane, fossil", RURAL], "type": "biosphere", "amount": 7.0})
        records[key] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "comment": (
                "Test original mean annual temperature (MAT) of 8 C and "
                f"precipitation (MAP) of {precipitation} mm/year."
            ),
            "exchanges": rows,
        }
    return records


@contextmanager
def dump_source_boundary(root: Path) -> Iterator[_RecipeContext]:
    """Write five tiny native sites; leave dated and supplier databases absent."""
    authority = prepared_provider_authority()
    catalogue = authority.catalogue
    names = BrightwayNames(root / "brightway", "pa_dump", "dump_biosphere", "dump_original")
    biosphere = dump_biosphere()
    records = dump_native_inventory(catalogue, names.source_database)
    with selected_brightway_project(names) as bd:
        bd.Database(names.biosphere_database).write(biosphere, searchable=False)
        bd.Database(names.source_database).write(records, searchable=False)
        original = _ActivityResolver(bd, names.source_database)
        # The original-only unmanaged functions use their specified source records.
        future = _ActivityResolver(bd, "not_a_dump_source")
        yield _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050,
            provider_database="not_a_dump_frontier",
            source=original,
            ordinary=future,
            technology=future,
            biosphere=_FlowResolver(bd, names.biosphere_database),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in authority.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
