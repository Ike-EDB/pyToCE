"""Check PA6 and PA66 energy and finishing links before inventory persistence."""

from pathlib import Path
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


def _energy_inventory(database: str) -> dict[tuple[str, str], dict[str, object]]:
    """Create synthetic electricity, heat and steam supplies with the selected descriptors."""
    rows = {}
    for code, name, product, unit in (
        (
            "power",
            "market group for electricity, medium voltage",
            "electricity, medium voltage",
            "kilowatt hour",
        ),
        (
            "heat",
            "market group for heat, district or industrial, other than natural gas",
            "heat, district or industrial, other than natural gas",
            "megajoule",
        ),
        (
            "steam",
            "market for heat, from steam, in chemical industry",
            "heat, from steam, in chemical industry",
            "megajoule",
        ),
    ):
        rows[database, code] = {**_process(database, code, name, product), "unit": unit}
    return rows


@pytest.mark.parametrize("original", [True, False])
def test_pa_conversion_joins_complete_supplies_without_a_pet_operation(
    tmp_path: Path, original: bool
) -> None:
    """Compile PA6 and PA66 finishing and energy requirements per kg finished polymer.

    The returned exchanges are checked before writing them to Brightway.
    Separate tests cover finish concentrate manufacture and spinning support.
    Spinning water, emissions, and maintenance are calculated for each event.
    This test checks their links.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_conversion", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write({}, searchable=False)
        bd.Database("original").write(_energy_inventory("original"), searchable=False)
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(_energy_inventory(database), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
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
        # Existing BREF energy points are 8.25/25 MJ per kg finished PA6/PA66.
        # Li's carrier split separates kWh electricity from the two MJ duties.
        for grade, electricity, heat, steam in (
            ("pa6", 1.1878936494564127, 0.7794259236971958, 3.1941569382597184),
            ("pa66", 2.0521841102316034, 14.66754617414248, 2.944591029023747),
        ):
            sid = f"fibres_{grade}_conversion"
            result = _service_dataset(context, services[sid], services, "frontier")
            rows = cast(list[dict[str, Any]], result["exchanges"])
            assert result["unit"] == "kilogram"
            assert len(rows) == 6
            expected = {
                ("production", ("frontier", provider_service_code(sid))): 1.0,
                (
                    "technosphere",
                    ("frontier", provider_service_code(f"{grade}_finish_concentrate")),
                ): 1.0,
                (
                    "technosphere",
                    ("frontier", provider_service_code(f"{grade}_spinning_support")),
                ): 1.0,
                ("technosphere", (database, "power")): electricity,
                ("technosphere", (database, "heat")): heat,
                ("technosphere", (database, "steam")): steam,
            }
            assert {(row["type"], row["input"]): row["amount"] for row in rows} == (
                pytest.approx(expected)
            )
