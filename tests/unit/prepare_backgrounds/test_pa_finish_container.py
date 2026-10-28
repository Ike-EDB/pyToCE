"""Verify complete PA finish packaging at its real native source boundary."""

from pathlib import Path

import pytest
from chemical_steam_fixtures import chemical_steam_biosphere, chemical_steam_native_inventory
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
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.mark.parametrize("original", [True, False])
def test_container_supplies_whole_native_functions_and_actual_empty_delivery(
    tmp_path: Path,
    original: bool,
) -> None:
    """Supply published part masses once and retain native manufacturing support."""
    provider = prepared_provider_authority()
    names = BrightwayNames(tmp_path / "brightway", "container", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_steam_biosphere(), searchable=False)
        native = chemical_steam_native_inventory(provider.catalogue, "source")
        bd.Database("source").write(native, searchable=False)
        database = "source" if original else "future"
        if not original:
            future = chemical_steam_native_inventory(provider.catalogue, database)
            for key, node in future.items():
                if key[1].startswith("drum_"):
                    for row in node["exchanges"]:
                        if row["type"] != "production":
                            row["amount"] *= 2
            bd.Database(database).write(future, searchable=False)
        before = {
            node.key: [dict(row) for row in node.exchanges()]
            for native_database in {"source", database}
            for node in bd.Database(native_database)
        }
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="frontier",
            source=_ActivityResolver(bd, "source"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=provider.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(provider.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        services = {str(row["provider_service_id"]): row for row in provider.catalogue["services"]}
        record = _service_dataset(context, services["pa_finish_container"], services, "frontier")
        # RIPA's nominal 21 lb body, 6 oz closures and 1 oz finished gasket
        # define installed tare. Native conversion bases come from the real
        # EcoSpold-shaped fixture: 0.9 blow and 0.8 injection finished/native kg.
        parts = {
            "hdpe_body": 21 * 0.45359237,
            "hdpe_closures": 6 * 0.45359237 / 16,
            "technical_seals": 0.45359237 / 16,
        }
        tare = sum(parts.values())
        assert record["fitted_tare_kg"] == pytest.approx(tare)
        assert record["installed_parts_kg"] == pytest.approx(parts)
        amounts = {
            row["input"][1]: row["amount"]
            for row in record["exchanges"]
            if row["type"] == "technosphere"
        }
        assert {
            row["input"][0] for row in record["exchanges"] if row["type"] == "technosphere"
        } == {database}
        distance = 200 * 1.609344
        assert amounts == pytest.approx(
            {
                "drum_hdpe": 0.53 * parts["hdpe_body"] / 0.9 + parts["hdpe_closures"] / 0.8,
                "drum_hdpe_recycled": 0.47 * parts["hdpe_body"] / 0.9,
                "drum_blow": parts["hdpe_body"] / 0.9,
                "drum_injection": parts["hdpe_closures"] / 0.8,
                "drum_seals": parts["technical_seals"],
                "drum_road": 15.96 * distance / 250,
            }
        )
        assert len(record["exchanges"]) == 7
        assert record["source_balance_outputs"] == pytest.approx(
            {
                "container_capacity_l": 55 * 3.785411784,
                "empty_drum_physical_transport_tkm": tare / 1000 * distance,
                "empty_drum_native_transport_tkm": 15.96 * distance / 250,
            }
        )
        assert {row["selected_background_member"] for row in context.technology_rows} == {
            None if original else "remind_ssp2_npi"
        }
        assert {row["anchor_year"] for row in context.technology_rows} == {
            2025 if original else 2050
        }
        for key, exchanges in before.items():
            assert [dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()] == (
                exchanges
            )
